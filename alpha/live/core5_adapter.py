"""CORE5 decision adapter: frozen Close_T shares and committed daily state.

The live host uses the research features/weights but the exchange calendar and
an observed EOD account snapshot. No research borrow fee is debited here.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import exchange_calendars as exchange_calendar_module

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module


CORE5_STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5"
CORE5_CONTRACT_STR = "core5_close_target_shares_v1"
CORE5_ASSET_TUPLE = ("SPY", "IEF", "GLD", "DBC", "UUP", "BIL")


def validate_core5_release(release_obj: LiveRelease, *, require_live_qualification_bool: bool = True) -> None:
    required_field_dict = {
        "data_profile_str": snapshot_module.CORE5_PROFILE_STR,
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": "eod_snapshot_ready",
        "execution_policy_str": "next_open_moo",
        "pod_budget_fraction_float": 1.0,
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj:
            raise ValueError(f"CORE5 requires {field_str}={expected_value_obj!r}.")
    if set(release_obj.params_dict) - {"capital_base_float", "core5_live_qualification_dict"}:
        raise ValueError("CORE5 uses the approved default strategy parameters; overrides require separate parity qualification.")
    if require_live_qualification_bool and release_obj.mode_str == "live" and release_obj.enabled_bool:
        # Operator evidence is a deployment prerequisite, not current borrow or
        # funding proof. runner also requires fresh core5_broker account-wide,
        # USD cash, margin previews and incremental shortability before submit.
        qualification_dict = release_obj.params_dict.get("core5_live_qualification_dict")
        if not isinstance(qualification_dict, dict):
            raise ValueError("CORE5 LIVE requires an account-bound qualification record.")
        for field_str in ("release_id_str", "account_route_str"):
            if qualification_dict.get(field_str) != getattr(release_obj, field_str):
                raise ValueError(f"CORE5 LIVE qualification {field_str} does not match this release.")
        for field_str in (
            "margin_account_confirmed_bool", "dbc_borrow_and_recall_policy_confirmed_bool",
            "forward_execution_qualified_bool", "operator_approved_bool",
        ):
            if qualification_dict.get(field_str) is not True:
                raise ValueError(f"CORE5 LIVE qualification requires {field_str}=true.")
        if not isinstance(qualification_dict.get("evidence_reference_str"), str) or not qualification_dict["evidence_reference_str"].strip():
            raise ValueError("CORE5 LIVE qualification requires an evidence reference.")
        try:
            approved_timestamp_ts = datetime.fromisoformat(qualification_dict["approved_at_str"])
            expiry_timestamp_ts = datetime.fromisoformat(qualification_dict["expires_at_str"])
        except (KeyError, TypeError, ValueError) as exception_obj:
            raise ValueError("CORE5 LIVE qualification requires valid approval/expiry timestamps.") from exception_obj
        if (
            approved_timestamp_ts.tzinfo is None or expiry_timestamp_ts.tzinfo is None
            or not approved_timestamp_ts <= datetime.now(UTC) < expiry_timestamp_ts
        ):
            raise ValueError("CORE5 LIVE qualification is future-dated, expired or lacks a timezone.")


def is_core5_decision_bool(decision_plan_obj: DecisionPlan | None) -> bool:
    return decision_plan_obj is not None and decision_plan_obj.snapshot_metadata_dict.get("sizing_contract_str") == CORE5_CONTRACT_STR


def _whole_position_map_dict(position_map_dict: dict[str, float]) -> dict[str, float]:
    result_dict = {}
    for asset_str, amount_float in position_map_dict.items():
        amount_float = float(amount_float)
        if not math.isfinite(amount_float) or amount_float != math.trunc(amount_float):
            raise ValueError("CORE5 requires finite whole-share account positions.")
        if amount_float == 0.0:
            continue
        if asset_str not in CORE5_ASSET_TUPLE or (amount_float < 0.0 and asset_str != "DBC"):
            raise ValueError(f"CORE5 found an unsupported account position: {asset_str}.")
        result_dict[str(asset_str)] = amount_float
    return result_dict


def require_core5_position_match(expected_position_dict: dict[str, float], actual_position_dict: dict[str, float]) -> None:
    if _whole_position_map_dict(expected_position_dict) != _whole_position_map_dict(actual_position_dict):
        raise ValueError("CORE5 account positions changed after the frozen decision; reconcile before proceeding.")


def _validate_target_weight_dict(target_weight_dict: dict[str, float]) -> None:
    if set(target_weight_dict) != set(CORE5_ASSET_TUPLE) | {"Cash"}:
        raise ValueError("CORE5 requires the complete target weight book.")
    if any(not math.isfinite(float(weight_float)) for weight_float in target_weight_dict.values()):
        raise ValueError("CORE5 target weights must be finite.")
    for asset_str in CORE5_ASSET_TUPLE[:-1]:
        weight_float = float(target_weight_dict[asset_str])
        if asset_str == "DBC" and -.10 - 1e-12 <= weight_float < 0.0:
            continue
        if not (abs(weight_float) <= 1e-12 or abs(weight_float - .20) <= 1e-12):
            raise ValueError("CORE5 target weights differ from the approved sleeves/short cap.")
    long_book_float = sum(max(0.0, float(target_weight_dict[asset_str])) for asset_str in CORE5_ASSET_TUPLE)
    if (
        not 0.0 <= float(target_weight_dict["BIL"]) <= 1.0 + 1e-12
        or abs(long_book_float - 1.0) > 1e-12
        or abs(float(target_weight_dict["Cash"]) - max(0.0, -float(target_weight_dict["DBC"]))) > 1e-12
    ):
        raise ValueError("CORE5 must preserve the 100% long/BIL book and restricted short proceeds.")


def _replay_core5_state_dict(strategy_obj, pricing_data_df: pd.DataFrame, signal_df: pd.DataFrame, cached_state_dict: dict | None = None) -> tuple[dict, dict]:
    """Replay signal memory only; never invent account cash, positions or fills."""
    from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module

    # *** CRITICAL *** Reconstruct the research calendar anchor from observations
    # through Close_T only. On the first ready Close_T, tomorrow's open is unknown;
    # the exchange calendar supplies its date, never a future price.
    exchange_calendar_obj = exchange_calendar_module.get_calendar("XNYS",
        start=pricing_data_df.index[0], end=pricing_data_df.index[-1] + pd.Timedelta(days=10))
    ready_bool_ser = signal_df.loc[:, [
        (core5_module.signal_namespace_str(asset_str), "long_state_ser")
        for asset_str in core5_module.RISK_ASSET_TUPLE]].notna().all(axis=1)
    ready_bool_ser &= signal_df[(core5_module.signal_namespace_str("DBC"), "annualized_volatility_ser")].notna()
    first_execution_ts = None
    for position_int in range(1, len(pricing_data_df.index)):
        if not ready_bool_ser.iloc[position_int - 1]:
            continue
        open_price_ser = pricing_data_df.iloc[position_int].loc[[(asset_str, "Open") for asset_str in CORE5_ASSET_TUPLE]]
        if all(math.isfinite(float(price_float)) and float(price_float) > 0 for price_float in open_price_ser):
            first_execution_ts = pricing_data_df.index[position_int]
            break
    next_session_ts = exchange_calendar_obj.next_session(pricing_data_df.index[-1])
    if first_execution_ts is None and ready_bool_ser.iloc[-1]:
        first_execution_ts = next_session_ts
    if first_execution_ts is None:
        raise ValueError("CORE5 has no actionable replay decision.")
    calendar_start_ts = max(first_execution_ts, pd.Timestamp(core5_module.DEFAULT_CONFIG.backtest_start_date_str))
    if calendar_start_ts > next_session_ts:
        raise ValueError("CORE5 replay precedes its approved backtest start.")
    first_decision_position_int = int(pricing_data_df.index.searchsorted(calendar_start_ts)) - 1
    decision_idx = pricing_data_df.index[first_decision_position_int:]
    expected_session_idx = exchange_calendar_obj.sessions_in_range(
        decision_idx[0], decision_idx[-1])
    if not decision_idx.equals(expected_session_idx):
        raise ValueError("CORE5 replay requires every exchange session in its decision history.")
    target_weight_dict = {}
    last_rebalance_date_str = None
    cached_replay_state_dict = None
    for decision_date_ts in decision_idx:
        # *** CRITICAL *** Features were truncated at Close_T before computation.
        # Replay the engine's event rule; between events retain the last weights,
        # including the DBC short weight fixed at its last rebalance volatility.
        close_row_ser = signal_df.loc[decision_date_ts]
        strategy_obj.previous_bar = decision_date_ts
        strategy_obj._validate_required_close_prices(close_row_ser)
        long_state_ser = strategy_obj._long_state_ser(close_row_ser)
        rebalance_bool = (
            not target_weight_dict
            or bool(close_row_ser[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.LONG_STATE_CHANGED_FIELD_STR)])
            or bool(close_row_ser[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.MONTH_END_REBALANCE_FIELD_STR)])
        )
        if rebalance_bool:
            target_weight_dict = strategy_obj._target_weight_ser(close_row_ser, long_state_ser).to_dict()
            _validate_target_weight_dict(target_weight_dict)
            last_rebalance_date_str = decision_date_ts.date().isoformat()
        if cached_state_dict and cached_state_dict.get("last_signal_date_str") == decision_date_ts.date().isoformat():
            cached_replay_state_dict = {
                "core5_state_version_int": 1,
                "initialized_bool": True,
                "last_signal_date_str": decision_date_ts.date().isoformat(),
                "last_long_state_map_dict": {asset_str: int(value_float) for asset_str, value_float in long_state_ser.items()},
                "last_target_weight_map_dict": dict(target_weight_dict),
                "last_rebalance_date_str": last_rebalance_date_str,
            }
    replay_state_dict = {
        "core5_state_version_int": 1,
        "initialized_bool": True,
        "last_signal_date_str": decision_idx[-1].date().isoformat(),
        "last_long_state_map_dict": {asset_str: int(value_float) for asset_str, value_float in long_state_ser.items()},
        "last_target_weight_map_dict": target_weight_dict,
        "last_rebalance_date_str": last_rebalance_date_str,
    }
    return replay_state_dict, {
        "core5_replay_start_date_str": decision_idx[0].date().isoformat(),
        "core5_replay_decision_count_int": len(decision_idx),
        "core5_data_revision_warning_bool": bool(cached_state_dict) and (
            cached_replay_state_dict is None or any(
                cached_state_dict.get(field_str) != value_obj
                for field_str, value_obj in cached_replay_state_dict.items()
            )
        ),
    }


def build_core5_decision_from_prices(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState,
    pricing_data_df: pd.DataFrame,
    snapshot_metadata_dict: dict[str, object],
) -> DecisionPlan:
    from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module

    validate_core5_release(release_obj)
    if release_obj.mode_str == "live" and not release_obj.enabled_bool:
        raise ValueError("CORE5 disabled LIVE templates cannot build operational decisions.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("CORE5 has no completed signal session.")
    signal_date_str = signal_date_ts.date().isoformat()
    previous_session_ts = scheduler_utils.get_exchange_calendar_obj("XNYS").previous_session(signal_date_ts)
    if (
        snapshot_metadata_dict.get("norgate_snapshot_date_str") != signal_date_str
        or snapshot_metadata_dict.get("norgate_data_profile_str") != snapshot_module.CORE5_PROFILE_STR
        or not snapshot_metadata_dict.get("norgate_manifest_hash_str")
    ):
        raise ValueError("CORE5 requires the exact decision-session snapshot and manifest identity.")
    # *** CRITICAL*** Truncate before computing features. Nothing after Close_T
    # may enter either the signal or the cash-plus-signed-position NAV mark.
    pricing_data_df = pricing_data_df.loc[:signal_date_ts].copy()
    if (
        pricing_data_df.index.has_duplicates
        or not pricing_data_df.index.is_monotonic_increasing
        or signal_date_ts not in pricing_data_df.index
        or previous_session_ts not in pricing_data_df.index
    ):
        raise ValueError("CORE5 requires unique ordered prices for T and its prior exchange session.")
    if pod_state_obj is None or (
        pod_state_obj.pod_id_str != release_obj.pod_id_str
        or pod_state_obj.account_route_str != release_obj.account_route_str
        or pod_state_obj.user_id_str != release_obj.user_id_str
    ):
        raise ValueError("CORE5 requires its own saved EOD account state.")
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
        raise ValueError("CORE5 requires a trusted EOD account snapshot from the signal session; bootstrap/default or stale NAV is insufficient.")
    position_map_dict = _whole_position_map_dict(pod_state_obj.position_amount_map)
    prior_state_dict = dict(pod_state_obj.strategy_state_dict)

    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    replay_state_dict, replay_metadata_dict = _replay_core5_state_dict(
        strategy_obj, pricing_data_df, signal_df, prior_state_dict,
    )
    close_row_ser = signal_df.loc[signal_date_ts]
    strategy_obj.previous_bar = signal_date_ts
    strategy_obj._validate_required_close_prices(close_row_ser)
    changed_bool = bool(close_row_ser[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.LONG_STATE_CHANGED_FIELD_STR)])
    month_end_bool = scheduler_utils.is_last_session_of_month_bool(signal_date_ts, "XNYS")
    target_weight_dict = dict(replay_state_dict["last_target_weight_map_dict"])
    prior_receipt_dict = prior_state_dict.get("core5_execution_receipt_dict", {})
    receipt_matches_bool = False
    receipt_invalid_bool = False
    if prior_receipt_dict:
        try:
            receipt_share_dict = prior_receipt_dict["last_applied_target_share_map_dict"]
            receipt_matches_bool = (
                prior_receipt_dict["last_applied_rebalance_date_str"] == replay_state_dict["last_rebalance_date_str"]
                and set(receipt_share_dict) == set(CORE5_ASSET_TUPLE)
                and _whole_position_map_dict(receipt_share_dict) == position_map_dict
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            receipt_invalid_bool = True
    # Signal memory always comes from prices. The separate execution receipt
    # only preserves share units between successfully applied strategy events.
    # Receipt weights are reporting only: revised adjusted history or floating
    # precision cannot create another rebalance for the same applied event.
    rebalance_bool = not receipt_matches_bool
    close_price_dict = {asset_str: float(close_row_ser[(asset_str, "Close")]) for asset_str in CORE5_ASSET_TUPLE}
    # NAV_Close_T = EOD cash + sum(signed shares_i * Close_T_i).
    # Short proceeds are already in cash; subtracting them again is incorrect.
    close_nav_float = float(pod_state_obj.cash_float) + sum(
        amount_float * close_price_dict[asset_str] for asset_str, amount_float in position_map_dict.items()
    )
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("CORE5 Close_T NAV must be finite and positive.")
    if rebalance_bool:
        # *** CRITICAL*** q_i = trunc(NAV_Close_T * w_i / Close_T_i).
        # int truncates shorts toward zero; next-open prices cannot resize q_i.
        target_share_dict = {
            asset_str: float(int(close_nav_float * target_weight_dict[asset_str] / close_price_dict[asset_str]))
            for asset_str in CORE5_ASSET_TUPLE
        }
    else:
        target_share_dict = {asset_str: float(position_map_dict.get(asset_str, 0.0)) for asset_str in CORE5_ASSET_TUPLE}
    _validate_target_weight_dict(target_weight_dict)
    no_order_bool = all(target_share_dict[asset_str] == position_map_dict.get(asset_str, 0.0) for asset_str in CORE5_ASSET_TUPLE)
    next_state_dict = {
        **replay_state_dict,
        "core5_execution_receipt_dict": prior_receipt_dict if isinstance(prior_receipt_dict, dict) else {},
    }
    candidate_receipt_dict = {
        "last_applied_rebalance_date_str": replay_state_dict["last_rebalance_date_str"],
        "last_applied_target_weight_map_dict": target_weight_dict,
        "last_applied_target_share_map_dict": target_share_dict,
    }
    warning_list = []
    if prior_state_dict and prior_state_dict.get("last_signal_date_str") != previous_session_ts.date().isoformat():
        warning_list.append("core5_strategy_state_cache_gap")
    if replay_metadata_dict["core5_data_revision_warning_bool"]:
        warning_list.append("core5_strategy_state_cache_revised")
    if receipt_invalid_bool:
        warning_list.append("core5_execution_receipt_invalid")
    metadata_dict = dict(snapshot_metadata_dict)
    metadata_dict.update({
        "strategy_family_str": "adaptive_macro_core5",
        "sizing_contract_str": CORE5_CONTRACT_STR,
        "fixed_target_share_map_dict": target_share_dict,
        "sizing_close_price_map_dict": close_price_dict,
        "sizing_close_nav_float": close_nav_float,
        "sizing_account_timestamp_str": pod_state_obj.updated_timestamp_ts.isoformat(),
        "sizing_account_cash_float": float(pod_state_obj.cash_float),
        "base_strategy_state_dict": prior_state_dict,
        **replay_metadata_dict,
        "core5_state_basis_str": "price_history_replay_through_close_t",
        "core5_warning_bool": bool(warning_list),
        "core5_warning_code_list": warning_list,
        "core5_catch_up_bool": rebalance_bool and replay_state_dict["last_rebalance_date_str"] != signal_date_str,
        # Proposed application only. The daily finalizer commits this receipt
        # after refreshed actual positions equal the complete frozen share book.
        "core5_candidate_execution_receipt_dict": candidate_receipt_dict,
        "rebalance_bool": bool(rebalance_bool),
        "no_order_bool": bool(no_order_bool),
        "initialization_bool": not bool(prior_state_dict),
        "long_state_changed_bool": changed_bool,
        "month_end_bool": month_end_bool,
        "research_borrow_rate_float": core5_module.DEFAULT_ANNUAL_DBC_BORROW_RATE_FLOAT,
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
        full_target_weight_map_dict={asset_str: target_weight_dict[asset_str] for asset_str in CORE5_ASSET_TUPLE},
        cash_reserve_weight_float=float(target_weight_dict["Cash"]),
        preserve_untouched_positions_bool=not rebalance_bool,
        rebalance_omitted_assets_to_zero_bool=rebalance_bool,
    )


def _load_core5_decision_prices(release_obj: LiveRelease, as_of_ts: datetime) -> tuple[pd.DataFrame, dict]:
    from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module

    validate_core5_release(release_obj)
    if not snapshot_module.is_snapshot_mode_enabled_bool():
        raise ValueError("CORE5 operational decisions require the qualified local snapshot profile.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("CORE5 has no completed signal session.")
    initial_manifest_obj = snapshot_module.load_valid_snapshot_manifest(
        snapshot_module.CORE5_PROFILE_STR, minimum_snapshot_date_str=signal_date_ts.date().isoformat(),
    )
    pricing_data_df = core5_module.get_adaptive_macro_core5_data(replace(core5_module.DEFAULT_CONFIG, end_date_str=signal_date_ts.date().isoformat()))
    metadata_dict = snapshot_module.build_data_source_metadata_dict(snapshot_module.CORE5_PROFILE_STR)
    if metadata_dict.get("norgate_manifest_hash_str") != initial_manifest_obj.manifest_hash_str:
        raise ValueError("CORE5 snapshot changed while preparing decision metadata.")
    return pricing_data_df, metadata_dict


def build_core5_decision_plan(release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None) -> DecisionPlan:
    pricing_data_df, metadata_dict = _load_core5_decision_prices(release_obj, as_of_ts)
    return build_core5_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df, metadata_dict)


def validated_core5_target_share_dict(decision_plan_obj: DecisionPlan, release_obj: LiveRelease, actual_position_dict: dict[str, float]) -> dict[str, float]:
    validate_core5_release(release_obj)
    if not is_core5_decision_bool(decision_plan_obj):
        raise ValueError("CORE5 decision lacks its frozen share contract.")
    if (decision_plan_obj.release_id_str, decision_plan_obj.pod_id_str, decision_plan_obj.account_route_str) != (
        release_obj.release_id_str, release_obj.pod_id_str, release_obj.account_route_str,
    ):
        raise ValueError("CORE5 decision identity differs from its release.")
    require_core5_position_match(decision_plan_obj.decision_base_position_map, actual_position_dict)
    metadata_dict = decision_plan_obj.snapshot_metadata_dict
    _validate_target_weight_dict({
        **{asset_str: decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0) for asset_str in CORE5_ASSET_TUPLE},
        "Cash": decision_plan_obj.cash_reserve_weight_float,
    })
    target_share_dict = dict(metadata_dict["fixed_target_share_map_dict"])
    close_price_dict = dict(metadata_dict["sizing_close_price_map_dict"])
    close_nav_float = float(metadata_dict["sizing_close_nav_float"])
    if set(target_share_dict) != set(CORE5_ASSET_TUPLE) or set(close_price_dict) != set(CORE5_ASSET_TUPLE):
        raise ValueError("CORE5 frozen share/price book is incomplete.")
    _whole_position_map_dict(target_share_dict)
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("CORE5 frozen NAV is invalid.")
    for asset_str in CORE5_ASSET_TUPLE:
        close_float = float(close_price_dict[asset_str])
        if not math.isfinite(close_float) or close_float <= 0.0:
            raise ValueError("CORE5 frozen sizing close is invalid.")
        expected_share_float = (
            float(int(close_nav_float * decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0) / close_float))
            if metadata_dict["rebalance_bool"] else float(actual_position_dict.get(asset_str, 0.0))
        )
        if target_share_dict[asset_str] != expected_share_float:
            raise ValueError("CORE5 frozen shares disagree with their Close_T sizing evidence.")
    return {asset_str: float(amount_float) for asset_str, amount_float in target_share_dict.items()}
