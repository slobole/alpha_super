"""CORE5 decision adapter: frozen Close_T shares and committed daily state.

The live host uses the research features/weights but the exchange calendar and
an observed EOD account snapshot. No research borrow fee is debited here.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module


CORE5_STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5"
CORE5_CONTRACT_STR = "core5_close_target_shares_v1"
CORE5_ASSET_TUPLE = ("SPY", "IEF", "GLD", "DBC", "UUP", "BIL")


def validate_core5_release(release_obj: LiveRelease) -> None:
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
    if set(release_obj.params_dict) - {"capital_base_float"}:
        raise ValueError("CORE5 uses the approved default strategy parameters; overrides require separate parity qualification.")
    if release_obj.mode_str == "live":
        raise ValueError("CORE5 physical LIVE activation requires forward-execution and account borrow/margin qualification; local incubation and paper probes are supported.")


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


def build_core5_decision_from_prices(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState,
    pricing_data_df: pd.DataFrame,
    snapshot_metadata_dict: dict[str, object],
) -> DecisionPlan:
    from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module

    validate_core5_release(release_obj)
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
    if prior_state_dict:
        if prior_state_dict.get("core5_state_version_int") != 1 or prior_state_dict.get("initialized_bool") is not True:
            raise ValueError("CORE5 saved strategy state is invalid or belongs to another strategy.")
        if prior_state_dict.get("last_signal_date_str") != previous_session_ts.date().isoformat():
            raise ValueError("CORE5 has a duplicate or missed decision session; reuse the saved plan or resolve the gap.")
        if not prior_state_dict.get("last_rebalance_date_str"):
            raise ValueError("CORE5 initialized state has no last rebalance date.")
        if pd.Timestamp(prior_state_dict["last_rebalance_date_str"]) > previous_session_ts:
            raise ValueError("CORE5 last rebalance cannot follow its committed signal date.")
        _validate_target_weight_dict(prior_state_dict.get("last_target_weight_map_dict", {}))
    elif position_map_dict:
        raise ValueError("CORE5 cannot initialize over unexplained existing positions.")

    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    close_row_ser = signal_df.loc[signal_date_ts]
    strategy_obj.previous_bar = signal_date_ts
    strategy_obj._validate_required_close_prices(close_row_ser)
    long_state_ser = strategy_obj._long_state_ser(close_row_ser)
    current_long_state_dict = {asset_str: int(value_float) for asset_str, value_float in long_state_ser.items()}
    changed_bool = False
    if prior_state_dict:
        previous_long_state_ser = strategy_obj._long_state_ser(signal_df.loc[previous_session_ts])
        if prior_state_dict.get("last_long_state_map_dict") != {
            asset_str: int(value_float) for asset_str, value_float in previous_long_state_ser.items()
        }:
            raise ValueError("CORE5 previous-session signal differs from committed state; historical revision requires review.")
        changed_bool = bool(long_state_ser.ne(previous_long_state_ser).any())
    month_end_bool = scheduler_utils.is_last_session_of_month_bool(signal_date_ts, "XNYS")
    rebalance_bool = not prior_state_dict or changed_bool or month_end_bool
    close_price_dict = {asset_str: float(close_row_ser[(asset_str, "Close")]) for asset_str in CORE5_ASSET_TUPLE}
    # NAV_Close_T = EOD cash + sum(signed shares_i * Close_T_i).
    # Short proceeds are already in cash; subtracting them again is incorrect.
    close_nav_float = float(pod_state_obj.cash_float) + sum(
        amount_float * close_price_dict[asset_str] for asset_str, amount_float in position_map_dict.items()
    )
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("CORE5 Close_T NAV must be finite and positive.")
    if rebalance_bool:
        target_weight_ser = strategy_obj._target_weight_ser(close_row_ser, long_state_ser)
        target_weight_dict = {asset_str: float(weight_float) for asset_str, weight_float in target_weight_ser.items()}
        # *** CRITICAL*** q_i = trunc(NAV_Close_T * w_i / Close_T_i).
        # int truncates shorts toward zero; next-open prices cannot resize q_i.
        target_share_dict = {
            asset_str: float(int(close_nav_float * target_weight_dict[asset_str] / close_price_dict[asset_str]))
            for asset_str in CORE5_ASSET_TUPLE
        }
    else:
        target_weight_dict = dict(prior_state_dict["last_target_weight_map_dict"])
        target_share_dict = {asset_str: float(position_map_dict.get(asset_str, 0.0)) for asset_str in CORE5_ASSET_TUPLE}
    _validate_target_weight_dict(target_weight_dict)
    no_order_bool = all(target_share_dict[asset_str] == position_map_dict.get(asset_str, 0.0) for asset_str in CORE5_ASSET_TUPLE)
    next_state_dict = {
        "core5_state_version_int": 1,
        "initialized_bool": True,
        "last_signal_date_str": signal_date_str,
        "last_long_state_map_dict": current_long_state_dict,
        "last_target_weight_map_dict": target_weight_dict,
        "last_rebalance_date_str": signal_date_str if rebalance_bool else prior_state_dict["last_rebalance_date_str"],
    }
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


def build_core5_decision_plan(release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None) -> DecisionPlan:
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
