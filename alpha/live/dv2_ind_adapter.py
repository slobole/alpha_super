"""DV2-IND on the existing incremental next-open MOO execution path.

Frozen research rules; quantities use the ordinary pre-submit reference price.
No strategy parameters, portfolio weights, or broker lifecycle are changed.
"""
from __future__ import annotations

from datetime import datetime
from dataclasses import replace
import math

import numpy as np
import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module
from data.dv2_ind_snapshot_contract import DV2_IND_PROFILE_STR, DV2_IND_ETF_TUPLE, DV2_IND_HISTORY_START_STR

DV2_IND_STRATEGY_IMPORT_STR = "strategies.dv2.strategy_mr_dv2_industry_etf:DVO2IndustryEtfStrategy"
DV2_IND_SIZING_CONTRACT_STR = "dv2_ind_close_nav_entries_v1"


def validate_dv2_ind_release(release_obj: LiveRelease) -> None:
    required_field_dict = {
        "strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR,
        "data_profile_str": DV2_IND_PROFILE_STR,
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": "eod_snapshot_ready",
        "execution_policy_str": "next_open_moo",
        "pod_budget_fraction_float": 1.0,
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj or isinstance(getattr(release_obj, field_str), bool):
            raise ValueError(f"DV2-IND requires {field_str}={expected_value_obj!r}.")
    if set(release_obj.params_dict) - {"capital_base_float"}:
        raise ValueError("DV2-IND uses frozen research parameters; overrides require qualification.")
    capital_obj = release_obj.params_dict.get("capital_base_float", 100000.0)
    if isinstance(capital_obj, bool) or not math.isfinite(float(capital_obj)) or float(capital_obj) <= 0:
        raise ValueError("DV2-IND capital_base_float must be finite and positive.")


def build_dv2_ind_decision_plan(release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None) -> DecisionPlan:
    from alpha.live import strategy_host
    from strategies.dv2 import strategy_mr_dv2_industry_etf as strategy_module

    validate_dv2_ind_release(release_obj)
    if not snapshot_module.is_snapshot_mode_enabled_bool():
        raise ValueError("DV2-IND decisions require the validated snapshot profile.")
    if pod_state_obj is None:
        raise ValueError("DV2-IND requires trusted same-session EOD account state.")
    if pod_state_obj is not None:
        if any(getattr(pod_state_obj, field_str) != getattr(release_obj, field_str)
                for field_str in ("pod_id_str", "user_id_str", "account_route_str")):
            raise ValueError("DV2-IND account state must match the release identity.")
        if not math.isfinite(float(pod_state_obj.total_value_float)) or pod_state_obj.total_value_float <= 0:
            raise ValueError("DV2-IND requires positive finite account equity.")
        if any(asset_str not in DV2_IND_ETF_TUPLE or not math.isfinite(float(amount_float))
                or float(amount_float) < 0 or float(amount_float) != math.trunc(float(amount_float))
                for asset_str, amount_float in pod_state_obj.position_amount_map.items()):
            raise ValueError("DV2-IND requires known long whole-share account holdings.")
        if sum(float(amount_float) > 0 for amount_float in pod_state_obj.position_amount_map.values()) > 10:
            raise ValueError("DV2-IND account exceeds ten held slots; reconcile first.")
    market_timestamp_ts = scheduler_utils.to_market_timestamp_ts(as_of_ts, "XNYS")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("DV2-IND requires a completed session.")
    state_timestamp_ts = scheduler_utils.to_market_timestamp_ts(pod_state_obj.updated_timestamp_ts, "XNYS")
    expected_source_str = "virtual_broker" if release_obj.mode_str == "incubation" else "broker"
    if (pod_state_obj.snapshot_stage_str != "eod" or pod_state_obj.snapshot_source_str != expected_source_str
            or state_timestamp_ts.date() != signal_date_ts.date()
            or state_timestamp_ts < scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS")
            or state_timestamp_ts > market_timestamp_ts or not math.isfinite(float(pod_state_obj.cash_float))):
        raise ValueError("DV2-IND requires trusted same-session EOD account state.")
    state_identity_str = pod_state_obj.strategy_state_dict.get("dv2_ind_strategy_import_str")
    if state_identity_str not in (None, DV2_IND_STRATEGY_IMPORT_STR) or (
            any(pod_state_obj.position_amount_map.values()) and state_identity_str != DV2_IND_STRATEGY_IMPORT_STR):
        raise ValueError("DV2-IND cannot adopt holdings from an unidentified strategy state.")
    target_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts.to_pydatetime(), release_obj)
    if scheduler_utils.is_execution_window_expired_bool("next_open_moo", target_timestamp_ts, market_timestamp_ts):
        raise ValueError("DV2-IND cannot build an expired daily auction cycle.")
    signal_date_str = signal_date_ts.date().isoformat()
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(DV2_IND_PROFILE_STR)
    if manifest_obj.snapshot_date_ts != signal_date_ts:
        raise ValueError("DV2-IND requires the exact latest completed session snapshot.")
    # *** CRITICAL *** Truncate before any rolling/shift computation. Close_T
    # indicators and history count cannot use T+1 bars or a revised later file.
    pricing_data_df = strategy_module.get_prices(list(DV2_IND_ETF_TUPLE), ["$SPX"],
        start_date=DV2_IND_HISTORY_START_STR, end_date=signal_date_str)
    if (pricing_data_df.empty or pricing_data_df.index.has_duplicates
            or not pricing_data_df.index.is_monotonic_increasing):
        raise ValueError("DV2-IND requires ordered unique pricing history.")
    pricing_data_df = pricing_data_df.loc[pricing_data_df.index <= signal_date_ts].copy()
    if pricing_data_df.empty or pricing_data_df.index[-1] != signal_date_ts:
        raise ValueError("DV2-IND pricing must end on the exact decision close.")
    expected_session_idx = scheduler_utils.get_exchange_calendar_obj("XNYS").sessions_in_range(pricing_data_df.index[0], signal_date_ts)
    if not pricing_data_df.index.equals(expected_session_idx) or len(pricing_data_df) < 252:
        raise ValueError("DV2-IND requires complete exchange-session history and 252-session warmup.")
    prior_session_ts = expected_session_idx[-2]
    for asset_str, amount_float in pod_state_obj.position_amount_map.items():
        if amount_float and (not np.isfinite(float(pricing_data_df.loc[prior_session_ts, (asset_str, "High")]))
                or float(pricing_data_df.loc[prior_session_ts, (asset_str, "High")]) <= 0):
            raise ValueError("DV2-IND held exit requires a valid prior-session high.")
    decision_nav_float = float(pod_state_obj.cash_float) + sum(
        float(amount_float) * float(pricing_data_df.loc[signal_date_ts, (asset_str, "Unadjusted Close")])
        for asset_str, amount_float in pod_state_obj.position_amount_map.items() if amount_float)
    if not math.isfinite(decision_nav_float) or decision_nav_float <= 0:
        raise ValueError("DV2-IND Close_T NAV must be finite and positive.")
    strategy_obj = strategy_module._new_strategy_obj(
        float(release_obj.params_dict.get("capital_base_float", 100000.0)),
        strategy_module.build_history_universe_df(pricing_data_df))
    strategy_obj.name = release_obj.pod_id_str
    strategy_host._seed_strategy_state(strategy_obj, pod_state_obj)
    # Entry dollars = (EOD cash + held shares * nominal Close_T) / 10.
    strategy_obj._total_value_history_list = [decision_nav_float]
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    close_row_ser = signal_df.loc[signal_date_ts]
    for asset_str in DV2_IND_ETF_TUPLE:
        required_field_list = ["Close", "High", "Low", "raw_price", "adv_63", "dv2", "sma_200", "p126d_return", "natr"]
        if not all((asset_str, field_str) in signal_df.columns for field_str in required_field_list):
            raise ValueError(f"DV2-IND missing signal fields for {asset_str}.")
        if strategy_obj.universe_df.loc[signal_date_ts, asset_str] == 1:
            feature_vec = close_row_ser.loc[[(asset_str, field_str) for field_str in required_field_list]].to_numpy(dtype=float)
            if not np.isfinite(feature_vec).all():
                raise ValueError(f"DV2-IND incomplete warmup or invalid features for {asset_str}.")
    strategy_obj.previous_bar = signal_date_ts
    strategy_obj.current_bar = pd.Timestamp(target_timestamp_ts.date())
    strategy_obj.iterate(signal_df, close_row_ser, pd.Series(dtype=float))
    final_manifest_obj = snapshot_module.load_valid_snapshot_manifest(DV2_IND_PROFILE_STR)
    if final_manifest_obj.manifest_hash_str != manifest_obj.manifest_hash_str:
        raise ValueError("DV2-IND snapshot identity changed during the decision.")
    decision_plan_obj = strategy_host._build_decision_plan_from_orders(
        release_obj, signal_date_ts.to_pydatetime(), strategy_obj,
        snapshot_metadata_dict={
            "strategy_family_str": "dv2_industry_etf",
            "sizing_contract_str": DV2_IND_SIZING_CONTRACT_STR,
            "norgate_manifest_hash_str": manifest_obj.manifest_hash_str,
            "research_sizing_basis_str": "previous_close_equity_per_ten_slots",
            "live_sizing_basis_str": "frozen_Close_T_NAV_and_pre_submit_reference_quote",
            "decision_nav_float": decision_nav_float,
            "decision_cash_float": float(pod_state_obj.cash_float),
            "strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR,
            "exact_share_parity_bool": False,
            "universe_basis_str": "fixed_19_ETFs_selected_in_research",
        })
    return replace(decision_plan_obj, strategy_state_dict={**decision_plan_obj.strategy_state_dict,
        "dv2_ind_strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR})


def validate_dv2_ind_execution_contract(release_obj, decision_plan_obj, broker_snapshot_obj, live_price_snapshot_obj=None):
    """Entry shares = floor(NAV_Close_T * 0.1 / reference_quote_submit)."""
    validate_dv2_ind_release(release_obj)
    if decision_plan_obj is None:
        raise ValueError("DV2-IND requires its saved decision.")
    for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str"):
        if getattr(decision_plan_obj, field_str) != getattr(release_obj, field_str):
            raise ValueError("DV2-IND decision identity mismatch.")
    if (broker_snapshot_obj.account_route_str != release_obj.account_route_str
            or live_price_snapshot_obj is not None and live_price_snapshot_obj.account_route_str != release_obj.account_route_str):
        raise ValueError("DV2-IND broker/quote account mismatch.")
    metadata_dict = decision_plan_obj.snapshot_metadata_dict
    if (metadata_dict.get("sizing_contract_str") != DV2_IND_SIZING_CONTRACT_STR
            or metadata_dict.get("strategy_import_str") != DV2_IND_STRATEGY_IMPORT_STR
            or decision_plan_obj.execution_policy_str != "next_open_moo"
            or decision_plan_obj.decision_book_type_str != "incremental_entry_exit_book"
            or not decision_plan_obj.preserve_untouched_positions_bool
            or decision_plan_obj.rebalance_omitted_assets_to_zero_bool
            or decision_plan_obj.target_share_map_dict):
        raise ValueError("DV2-IND requires its frozen incremental Close_T NAV contract.")
    if (set(decision_plan_obj.entry_target_weight_map_dict) | set(decision_plan_obj.exit_asset_set)) - set(DV2_IND_ETF_TUPLE):
        raise ValueError("DV2-IND intent contains an unknown ETF.")
    if any(not math.isfinite(float(weight_float)) or abs(float(weight_float)-0.1)>1e-12
            for weight_float in decision_plan_obj.entry_target_weight_map_dict.values()):
        raise ValueError("DV2-IND entries must target one tenth of Close_T NAV.")
    position_map_list = []
    for position_dict in (decision_plan_obj.decision_base_position_map, broker_snapshot_obj.position_amount_map):
        if any(asset_str not in DV2_IND_ETF_TUPLE or not math.isfinite(float(share_float))
                or float(share_float) < 0 or float(share_float) != math.trunc(float(share_float))
                for asset_str, share_float in position_dict.items()):
            raise ValueError("DV2-IND holdings must be known long whole shares.")
        position_map_list.append({asset_str: float(share_float) for asset_str, share_float in position_dict.items() if share_float})
    if position_map_list[0] != position_map_list[1] or broker_snapshot_obj.open_order_id_list:
        raise ValueError("DV2-IND account holdings/orders changed after the frozen decision.")
    held_asset_set = set(position_map_list[0])
    if (set(decision_plan_obj.entry_target_weight_map_dict) & held_asset_set
            or not set(decision_plan_obj.exit_asset_set).issubset(held_asset_set)
            or len(held_asset_set - set(decision_plan_obj.exit_asset_set)) + len(decision_plan_obj.entry_target_weight_map_dict) > 10):
        raise ValueError("DV2-IND intent violates held-entry, exit or ten-slot rules.")
    decision_nav_float = float(metadata_dict.get("decision_nav_float", float("nan")))
    if (not math.isfinite(decision_nav_float) or decision_nav_float <= 0
            or not math.isfinite(float(metadata_dict.get("decision_cash_float", float("nan"))))
            or not math.isfinite(float(broker_snapshot_obj.net_liq_float)) or broker_snapshot_obj.net_liq_float <= 0
            or not math.isfinite(float(broker_snapshot_obj.cash_float))):
        raise ValueError("DV2-IND requires finite cash and positive decision/broker NAV.")
    return decision_nav_float
