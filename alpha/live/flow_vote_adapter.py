"""Month-End Flow Vote decision adapter: frozen Close_T shares, filled in the next session's closing auction.

The live host runs the strategy's own `compute_signals` over observed prices through Close_T and reads row T, as the
backtest's `process_orders` does before its next-close fill. Shares come from the observed end-of-day account:
    NAV_T = cash + sum(signed shares_i * Close_T_i),   target_i = trunc(NAV_T * w_i / Close_T_i)
No later quote or NAV changes them. No research borrow fee is debited here; the broker's own financing applies.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module
from data.flow_vote_snapshot_contract import FLOW_VOTE_PROFILE_STR


FLOW_VOTE_STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_month_end_flow_vote"
FLOW_VOTE_CONTRACT_STR = "flow_vote_close_target_shares_v1"
FLOW_VOTE_EXECUTION_POLICY_STR = "next_close_moc"
FLOW_VOTE_ASSET_TUPLE = ("SPY", "TLT")
FLOW_VOTE_SHORT_ASSET_SET = frozenset({"TLT"})
FLOW_VOTE_STATE_VERSION_INT = 1
_WEIGHT_TOLERANCE_FLOAT = 1e-12
# The only books the frozen rule can name: flat, the final week (votes / 3 in SPY, the rest TLT) and the next-month pair.
_ALLOWED_BOOK_TUPLE = ((0.0, 0.0), (0.0, 1.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0), (1.0, 0.0), (0.5, -0.5))


def validate_flow_vote_release(release_obj: LiveRelease) -> None:
    required_field_dict = {
        "data_profile_str": FLOW_VOTE_PROFILE_STR,
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": "eod_snapshot_ready",
        "execution_policy_str": FLOW_VOTE_EXECUTION_POLICY_STR,
        "pod_budget_fraction_float": 1.0,
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj:
            raise ValueError(f"Flow Vote requires {field_str}={expected_value_obj!r}.")
    if release_obj.mode_str not in ("paper", "live"):
        # The virtual broker would fill at the next session's Norgate close, which exists only in a later snapshot.
        raise ValueError("Flow Vote runs in paper or live mode only.")
    if set(release_obj.params_dict) - {"capital_base_float"}:
        raise ValueError("Flow Vote uses the frozen default strategy parameters; overrides require separate parity qualification.")


def is_flow_vote_decision_bool(decision_plan_obj: DecisionPlan | None) -> bool:
    return decision_plan_obj is not None and decision_plan_obj.snapshot_metadata_dict.get("sizing_contract_str") == FLOW_VOTE_CONTRACT_STR


def whole_flow_vote_position_map_dict(position_map_dict: dict[str, float]) -> dict[str, float]:
    result_dict = {}
    for asset_str, amount_float in position_map_dict.items():
        amount_float = float(amount_float)
        if not math.isfinite(amount_float) or amount_float != math.trunc(amount_float):
            raise ValueError("Flow Vote requires finite whole-share account positions.")
        if amount_float == 0.0:
            continue
        if asset_str not in FLOW_VOTE_ASSET_TUPLE or (amount_float < 0.0 and asset_str not in FLOW_VOTE_SHORT_ASSET_SET):
            raise ValueError(f"Flow Vote found an unsupported account position: {asset_str}.")
        result_dict[str(asset_str)] = amount_float
    return result_dict


def require_flow_vote_position_match(expected_position_dict: dict[str, float], actual_position_dict: dict[str, float]) -> None:
    if whole_flow_vote_position_map_dict(expected_position_dict) != whole_flow_vote_position_map_dict(actual_position_dict):
        raise ValueError("Flow Vote account positions changed after the frozen decision; reconcile before proceeding.")


def _validate_target_weight_dict(target_weight_dict: dict[str, float]) -> None:
    if set(target_weight_dict) != set(FLOW_VOTE_ASSET_TUPLE) | {"Cash"}:
        raise ValueError("Flow Vote requires the complete target weight book.")
    if any(not math.isfinite(float(weight_float)) for weight_float in target_weight_dict.values()):
        raise ValueError("Flow Vote target weights must be finite.")
    book_tuple = (float(target_weight_dict["SPY"]), float(target_weight_dict["TLT"]))
    if not any(all(abs(a - b) <= _WEIGHT_TOLERANCE_FLOAT for a, b in zip(book_tuple, allowed_tuple)) for allowed_tuple in _ALLOWED_BOOK_TUPLE):
        raise ValueError("Flow Vote target weights differ from the frozen rule's books.")
    if abs(float(target_weight_dict["Cash"]) - (1.0 - book_tuple[0] - book_tuple[1])) > _WEIGHT_TOLERANCE_FLOAT:
        raise ValueError("Flow Vote cash weight must equal one minus the net book.")


def _standing_event_dict(order_schedule_df: pd.DataFrame, next_session_ts: pd.Timestamp) -> dict:
    """The schedule row in force at the next session's close: the latest fill at or before it."""
    eligible_df = order_schedule_df.loc[order_schedule_df.index <= next_session_ts]
    if eligible_df.empty:
        raise ValueError("Flow Vote has no scheduled book at or before the next session.")
    fill_ts = eligible_df.index[-1]
    row_obj = eligible_df.iloc[-1]
    if eligible_df.index.has_duplicates and (eligible_df.index == fill_ts).sum() > 1:
        raise ValueError("Flow Vote schedule names two books for one fill session.")
    return {"event_fill_date_str": fill_ts.date().isoformat(), "leg_str": str(row_obj["leg_str"]),
            "measure_date_str": pd.Timestamp(row_obj["measure_date"]).date().isoformat(),
            "SPY": float(row_obj["SPY"]), "TLT": float(row_obj["TLT"])}


def build_flow_vote_decision_from_prices(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState,
    pricing_data_df: pd.DataFrame,
    snapshot_metadata_dict: dict[str, object],
) -> DecisionPlan:
    from strategies.taa_beyond_6040 import strategy_taa_month_end_flow_vote as vote_module
    from strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow import SIGNAL_NAMESPACE_STR

    validate_flow_vote_release(release_obj)
    if release_obj.mode_str == "live" and not release_obj.enabled_bool:
        raise ValueError("Flow Vote disabled LIVE templates cannot build operational decisions.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("Flow Vote has no completed signal session.")
    signal_date_str = signal_date_ts.date().isoformat()
    calendar_obj = scheduler_utils.get_exchange_calendar_obj("XNYS")
    previous_session_ts = calendar_obj.previous_session(signal_date_ts)
    next_session_ts = pd.Timestamp(calendar_obj.next_session(signal_date_ts))
    next_session_ts = (next_session_ts.tz_localize(None) if next_session_ts.tzinfo is not None else next_session_ts).normalize()
    if (
        snapshot_metadata_dict.get("norgate_snapshot_date_str") != signal_date_str
        or snapshot_metadata_dict.get("norgate_data_profile_str") != FLOW_VOTE_PROFILE_STR
        or not snapshot_metadata_dict.get("norgate_manifest_hash_str")
    ):
        raise ValueError("Flow Vote requires the exact decision-session snapshot and manifest identity.")
    # *** CRITICAL*** Truncate before computing signals. Nothing after Close_T may enter the signal or the NAV mark.
    pricing_data_df = pricing_data_df.loc[:signal_date_ts].copy()
    if (
        pricing_data_df.index.has_duplicates
        or not pricing_data_df.index.is_monotonic_increasing
        or signal_date_ts not in pricing_data_df.index
        or previous_session_ts not in pricing_data_df.index
    ):
        raise ValueError("Flow Vote requires unique ordered prices for T and its prior exchange session.")
    if pod_state_obj is None or (
        pod_state_obj.pod_id_str != release_obj.pod_id_str
        or pod_state_obj.account_route_str != release_obj.account_route_str
        or pod_state_obj.user_id_str != release_obj.user_id_str
    ):
        raise ValueError("Flow Vote requires its own saved EOD account state.")
    state_timestamp_ts = scheduler_utils.to_market_timestamp_ts(pod_state_obj.updated_timestamp_ts, "XNYS")
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS")
    expected_source_str = "virtual_broker" if release_obj.mode_str == "incubation" else "broker"
    if (
        pod_state_obj.snapshot_stage_str != "eod"
        or pod_state_obj.snapshot_source_str != expected_source_str
        or state_timestamp_ts.date() != signal_date_ts.date()
        or state_timestamp_ts < close_timestamp_ts
        or state_timestamp_ts > scheduler_utils.to_market_timestamp_ts(as_of_ts, "XNYS")
    ):
        raise ValueError("Flow Vote requires a trusted EOD account snapshot from the signal session; bootstrap/default or stale NAV is insufficient.")
    position_map_dict = whole_flow_vote_position_map_dict(pod_state_obj.position_amount_map)
    prior_state_dict = dict(pod_state_obj.strategy_state_dict)

    strategy_obj = vote_module.MonthEndVoteFlowStrategy(vote_module.DEFAULT_CONFIG)
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    signal_row_ser = signal_df.loc[signal_date_ts]
    event_dict = _standing_event_dict(strategy_obj.order_schedule_df, next_session_ts)
    fill_tomorrow_bool = event_dict["event_fill_date_str"] == next_session_ts.date().isoformat()
    # The backtest's own trigger and the schedule must agree on whether tomorrow's close is a fill.
    if bool(signal_row_ser[(SIGNAL_NAMESPACE_STR, "rebalance_bool")]) != fill_tomorrow_bool:
        raise ValueError("Flow Vote signal row and schedule disagree about the next session's fill.")
    if fill_tomorrow_bool and (
        abs(float(signal_row_ser[(SIGNAL_NAMESPACE_STR, "SPY")]) - event_dict["SPY"]) > _WEIGHT_TOLERANCE_FLOAT
        or abs(float(signal_row_ser[(SIGNAL_NAMESPACE_STR, "TLT")]) - event_dict["TLT"]) > _WEIGHT_TOLERANCE_FLOAT
    ):
        raise ValueError("Flow Vote signal row and schedule disagree about the next session's book.")
    target_weight_dict = {"SPY": event_dict["SPY"], "TLT": event_dict["TLT"], "Cash": 1.0 - event_dict["SPY"] - event_dict["TLT"]}
    _validate_target_weight_dict(target_weight_dict)

    prior_receipt_dict = prior_state_dict.get("flow_vote_execution_receipt_dict", {})
    receipt_invalid_bool = False
    receipt_matches_bool = False
    # Only a pod that has never decided initializes. After a decision, an empty receipt means its first fill did not
    # complete: that event is caught up at the next close like any other.
    initialization_bool = not prior_receipt_dict and not prior_state_dict.get("last_signal_date_str")
    if prior_receipt_dict:
        try:
            receipt_share_dict = prior_receipt_dict["last_applied_target_share_map_dict"]
            receipt_matches_bool = (
                prior_receipt_dict["last_applied_event_fill_date_str"] == event_dict["event_fill_date_str"]
                and set(receipt_share_dict) == set(FLOW_VOTE_ASSET_TUPLE)
                and whole_flow_vote_position_map_dict(receipt_share_dict) == position_map_dict
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            receipt_invalid_bool = True
    close_price_dict = {asset_str: float(signal_row_ser[(asset_str, "Close")]) for asset_str in FLOW_VOTE_ASSET_TUPLE}
    if any(not math.isfinite(close_float) or close_float <= 0.0 for close_float in close_price_dict.values()):
        raise ValueError("Flow Vote Close_T prices must be finite and positive.")
    # NAV_Close_T = EOD cash + sum(signed shares_i * Close_T_i). Short proceeds are already in cash.
    close_nav_float = float(pod_state_obj.cash_float) + sum(
        amount_float * close_price_dict[asset_str] for asset_str, amount_float in position_map_dict.items()
    )
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("Flow Vote Close_T NAV must be finite and positive.")
    if initialization_bool and not position_map_dict and not fill_tomorrow_bool:
        # *** CRITICAL*** A new flat pod waits for the next scheduled fill, as the backtest starts flat and trades only on
        # fill sessions. Its first receipt records the book in force as applied at zero shares.
        rebalance_bool = False
    elif initialization_bool and position_map_dict:
        raise ValueError("Flow Vote found holdings without an execution receipt; a reviewed initialization is required.")
    else:
        rebalance_bool = not receipt_matches_bool
    if rebalance_bool:
        # *** CRITICAL*** q_i = trunc(NAV_Close_T * w_i / Close_T_i). int truncates the TLT short toward zero.
        target_share_dict = {
            asset_str: float(int(close_nav_float * target_weight_dict[asset_str] / close_price_dict[asset_str]))
            for asset_str in FLOW_VOTE_ASSET_TUPLE
        }
    else:
        target_share_dict = {asset_str: float(position_map_dict.get(asset_str, 0.0)) for asset_str in FLOW_VOTE_ASSET_TUPLE}
    no_order_bool = all(target_share_dict[asset_str] == position_map_dict.get(asset_str, 0.0) for asset_str in FLOW_VOTE_ASSET_TUPLE)
    candidate_receipt_dict = {
        "last_applied_event_fill_date_str": event_dict["event_fill_date_str"],
        "last_applied_target_weight_map_dict": dict(target_weight_dict),
        "last_applied_target_share_map_dict": dict(target_share_dict),
    }
    next_state_dict = {
        "flow_vote_state_version_int": FLOW_VOTE_STATE_VERSION_INT,
        "last_signal_date_str": signal_date_str,
        "flow_vote_execution_receipt_dict": prior_receipt_dict if isinstance(prior_receipt_dict, dict) else {},
    }
    warning_list = []
    if prior_state_dict and prior_state_dict.get("last_signal_date_str") != previous_session_ts.date().isoformat():
        warning_list.append("flow_vote_strategy_state_cache_gap")
    if receipt_invalid_bool:
        warning_list.append("flow_vote_execution_receipt_invalid")
    month_row_df = strategy_obj.month_table_df.loc[strategy_obj.month_table_df["measure_date"] == pd.Timestamp(event_dict["measure_date_str"])]
    month_row_dict = {} if month_row_df.empty else {
        "month_period_str": str(month_row_df["month_period"].iloc[-1]),
        "pressure_bps_float": float(month_row_df["pressure_bps_float"].iloc[-1]),
        "bucket_float": None if pd.isna(month_row_df["bucket_float"].iloc[-1]) else float(month_row_df["bucket_float"].iloc[-1]),
        "half_spread_float": float(month_row_df["half_spread_float"].iloc[-1]),
        "spy_vote_count_int": int(month_row_df["spy_vote_count_int"].iloc[-1]),
    }
    metadata_dict = dict(snapshot_metadata_dict)
    metadata_dict.update({
        "strategy_family_str": "month_end_flow_vote",
        "sizing_contract_str": FLOW_VOTE_CONTRACT_STR,
        "fixed_target_share_map_dict": target_share_dict,
        "sizing_close_price_map_dict": close_price_dict,
        "sizing_close_nav_float": close_nav_float,
        "sizing_account_timestamp_str": pod_state_obj.updated_timestamp_ts.isoformat(),
        "sizing_account_cash_float": float(pod_state_obj.cash_float),
        "base_strategy_state_dict": prior_state_dict,
        "flow_vote_standing_event_dict": event_dict,
        "flow_vote_month_row_dict": month_row_dict,
        "flow_vote_fill_next_session_bool": fill_tomorrow_bool,
        "flow_vote_catch_up_bool": bool(rebalance_bool and not fill_tomorrow_bool),
        "flow_vote_warning_bool": bool(warning_list),
        "flow_vote_warning_code_list": warning_list,
        # Proposed application only. The daily finalizer commits this receipt after refreshed actual positions equal the
        # complete frozen share book (state_store_v2.complete_daily_cycle).
        "flow_vote_candidate_execution_receipt_dict": candidate_receipt_dict,
        "rebalance_bool": bool(rebalance_bool),
        "no_order_bool": bool(no_order_bool),
        "initialization_bool": bool(initialization_bool),
        "borrow_accounting_str": "observed_account_cash_only_no_synthetic_research_fee",
    })
    return DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=scheduler_utils.build_signal_timestamp_ts(signal_date_ts, release_obj),
        submission_timestamp_ts=scheduler_utils.build_submission_timestamp_ts(signal_date_ts, release_obj),
        target_execution_timestamp_ts=scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj),
        execution_policy_str=release_obj.execution_policy_str,
        decision_base_position_map=position_map_dict,
        snapshot_metadata_dict=metadata_dict, strategy_state_dict=next_state_dict,
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict={asset_str: target_weight_dict[asset_str] for asset_str in FLOW_VOTE_ASSET_TUPLE},
        cash_reserve_weight_float=float(target_weight_dict["Cash"]),
        preserve_untouched_positions_bool=not rebalance_bool,
        rebalance_omitted_assets_to_zero_bool=rebalance_bool,
    )


def _load_flow_vote_decision_prices(release_obj: LiveRelease, as_of_ts: datetime) -> tuple[pd.DataFrame, dict]:
    from strategies.taa_beyond_6040 import strategy_taa_month_end_flow_vote as vote_module

    validate_flow_vote_release(release_obj)
    if not snapshot_module.is_snapshot_mode_enabled_bool():
        raise ValueError("Flow Vote operational decisions require the qualified local snapshot profile.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("Flow Vote has no completed signal session.")
    initial_manifest_obj = snapshot_module.load_valid_snapshot_manifest(
        FLOW_VOTE_PROFILE_STR, minimum_snapshot_date_str=signal_date_ts.date().isoformat(),
    )
    pricing_data_df = vote_module.get_month_end_flow_data(replace(vote_module.DEFAULT_CONFIG, end_date_str=signal_date_ts.date().isoformat()))
    metadata_dict = snapshot_module.build_data_source_metadata_dict(FLOW_VOTE_PROFILE_STR)
    if metadata_dict.get("norgate_manifest_hash_str") != initial_manifest_obj.manifest_hash_str:
        raise ValueError("Flow Vote snapshot changed while preparing decision metadata.")
    return pricing_data_df, metadata_dict


def build_flow_vote_decision_plan(release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None) -> DecisionPlan:
    pricing_data_df, metadata_dict = _load_flow_vote_decision_prices(release_obj, as_of_ts)
    return build_flow_vote_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df, metadata_dict)


def validated_flow_vote_target_share_dict(decision_plan_obj: DecisionPlan, release_obj: LiveRelease, actual_position_dict: dict[str, float]) -> dict[str, float]:
    validate_flow_vote_release(release_obj)
    if not is_flow_vote_decision_bool(decision_plan_obj):
        raise ValueError("Flow Vote decision lacks its frozen share contract.")
    if (decision_plan_obj.release_id_str, decision_plan_obj.pod_id_str, decision_plan_obj.account_route_str) != (
        release_obj.release_id_str, release_obj.pod_id_str, release_obj.account_route_str,
    ):
        raise ValueError("Flow Vote decision identity differs from its release.")
    require_flow_vote_position_match(decision_plan_obj.decision_base_position_map, actual_position_dict)
    metadata_dict = decision_plan_obj.snapshot_metadata_dict
    _validate_target_weight_dict({
        **{asset_str: decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0) for asset_str in FLOW_VOTE_ASSET_TUPLE},
        "Cash": decision_plan_obj.cash_reserve_weight_float,
    })
    target_share_dict = dict(metadata_dict["fixed_target_share_map_dict"])
    close_price_dict = dict(metadata_dict["sizing_close_price_map_dict"])
    close_nav_float = float(metadata_dict["sizing_close_nav_float"])
    if set(target_share_dict) != set(FLOW_VOTE_ASSET_TUPLE) or set(close_price_dict) != set(FLOW_VOTE_ASSET_TUPLE):
        raise ValueError("Flow Vote frozen share/price book is incomplete.")
    whole_flow_vote_position_map_dict(target_share_dict)
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("Flow Vote frozen NAV is invalid.")
    for asset_str in FLOW_VOTE_ASSET_TUPLE:
        close_float = float(close_price_dict[asset_str])
        if not math.isfinite(close_float) or close_float <= 0.0:
            raise ValueError("Flow Vote frozen sizing close is invalid.")
        expected_share_float = (
            float(int(close_nav_float * decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0) / close_float))
            if metadata_dict["rebalance_bool"] else float(actual_position_dict.get(asset_str, 0.0))
        )
        if target_share_dict[asset_str] != expected_share_float:
            raise ValueError("Flow Vote frozen shares disagree with their Close_T sizing evidence.")
    return {asset_str: float(amount_float) for asset_str, amount_float in target_share_dict.items()}
