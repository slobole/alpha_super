"""Frozen E2 selections on the existing monthly broker-truth execution path.

Weights use completed month-end Close_T information. The existing execution
engine sizes whole shares from broker NAV and reference prices before the next
open; research Close_T share quantities are not an execution-parity claim.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from importlib import import_module
import math

import numpy as np
import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module


NDX_E2_STRATEGY_IMPORT_TUPLE = (
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap:SectorCapVxnScaledAtrNormalizedNdxStrategy",
    "strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap:SectorCapNatr20VxnScaledNdxStrategy",
)
NDX_E2_SIZING_CONTRACT_STR = "legacy_monthly_preopen_v1"
NDX_E2_DECISION_HISTORY_MONTHS_INT = 36


def validate_ndx_e2_release(release_obj: LiveRelease) -> None:
    """Keep E2 defaults reproducible by the existing deployment reference runner."""
    if release_obj.strategy_import_str not in NDX_E2_STRATEGY_IMPORT_TUPLE:
        raise ValueError("Unsupported NDX E2 strategy identity.")
    required_field_dict = {
        "data_profile_str": snapshot_module.NDX_E2_PROFILE_STR,
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": "month_end_snapshot_ready",
        "execution_policy_str": "next_month_first_open",
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj:
            raise ValueError(f"NDX E2 requires {field_str}={expected_value_obj!r}.")
    strategy_module = import_module(release_obj.strategy_import_str.split(":")[0])
    allowed_parameter_set = {
        "max_positions_int", "lookback_month_int", "index_trend_window_int",
        "stock_trend_window_int", "regime_symbol_str", "vxn_symbol_str",
        "target_vxn_pct_float", "min_exposure_scale_float", "max_exposure_scale_float",
        "sector_cap_int", "slippage_float", "commission_per_share_float", "commission_minimum_float",
    }
    for parameter_str, value_obj in release_obj.params_dict.items():
        if parameter_str == "capital_base_float":
            if isinstance(value_obj, bool) or not math.isfinite(float(value_obj)) or float(value_obj) <= 0:
                raise ValueError("NDX E2 capital_base_float must be finite and positive.")
        elif (parameter_str not in allowed_parameter_set or isinstance(value_obj, bool)
                or value_obj != getattr(strategy_module.DEFAULT_CONFIG, parameter_str)):
            raise ValueError("NDX E2 uses frozen strategy parameters; overrides require parity qualification.")
    budget_float = float(release_obj.pod_budget_fraction_float)
    if isinstance(release_obj.pod_budget_fraction_float, bool) or not math.isfinite(budget_float) or not 0 < budget_float <= 1:
        raise ValueError("NDX E2 requires a finite dedicated-account budget fraction in (0, 1].")


def build_ndx_e2_decision_plan(
    release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None,
) -> DecisionPlan:
    """Use the unchanged research ranking/classes, then the legacy monthly weights contract."""
    from alpha.live import strategy_host
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap import build_strategy

    validate_ndx_e2_release(release_obj)
    if not snapshot_module.is_snapshot_mode_enabled_bool():
        raise ValueError("NDX E2 live decisions require the frozen snapshot profile.")
    if pod_state_obj is not None and any(
            getattr(pod_state_obj, field_str) != getattr(release_obj, field_str)
            for field_str in ("pod_id_str", "user_id_str", "account_route_str")):
        raise ValueError("NDX E2 account state must match the release identity.")
    signal_date_ts = scheduler_utils.get_latest_completed_month_end_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("NDX E2 requires a completed month-end session.")
    target_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj)
    if scheduler_utils.is_execution_window_expired_bool(release_obj.execution_policy_str, target_timestamp_ts, as_of_ts):
        raise ValueError("NDX E2 cannot build a stale monthly execution cycle.")
    signal_date_str = signal_date_ts.date().isoformat()
    snapshot_metadata_dict = snapshot_module.build_data_source_metadata_dict(release_obj.data_profile_str)
    if (snapshot_metadata_dict.get("norgate_snapshot_date_str") != signal_date_str
            or snapshot_metadata_dict.get("norgate_data_profile_str") != release_obj.data_profile_str
            or not snapshot_metadata_dict.get("norgate_manifest_hash_str")):
        raise ValueError("NDX E2 requires the exact completed month-end snapshot and manifest identity.")
    manifest_obj = snapshot_module.load_valid_snapshot_manifest(
        release_obj.data_profile_str, snapshot_date_str=signal_date_str)
    if manifest_obj.manifest_hash_str != snapshot_metadata_dict["norgate_manifest_hash_str"]:
        raise ValueError("NDX E2 snapshot identity changed before the decision.")
    data_contract_dict = manifest_obj.manifest_dict["data_contract"]

    module_name_str, class_name_str = release_obj.strategy_import_str.split(":")
    strategy_module = import_module(module_name_str)
    # *** CRITICAL *** The decision reads only a trailing window of the frozen
    # snapshot. The longest lookback is ROC over 12 month-ends; SMA200, SMA100
    # and ATR20 are shorter. The snapshot keeps full 1999+ history for
    # references. A window that is too short raises below (NaN regime or
    # features); it cannot change selections silently.
    history_start_ts = signal_date_ts - pd.DateOffset(months=NDX_E2_DECISION_HISTORY_MONTHS_INT)
    config_obj = replace(strategy_module.DEFAULT_CONFIG, end_date_str=signal_date_str,
        history_start_date_str=history_start_ts.date().isoformat(),
        backtest_start_date_str=(history_start_ts + pd.Timedelta(days=1)).date().isoformat(),
        capital_base_float=float(release_obj.params_dict.get("capital_base_float", strategy_module.DEFAULT_CONFIG.capital_base_float)))
    natr_bool = release_obj.strategy_import_str == NDX_E2_STRATEGY_IMPORT_TUPLE[1]
    data_loader_fn = (strategy_module.get_natr20_vxn_scaled_ndx_data if natr_bool
        else strategy_module.get_vxn_scaled_atr_normalized_ndx_data)
    pricing_data_df, universe_df, _, vxn_signal_df = data_loader_fn(config_obj)
    # *** CRITICAL *** All signal inputs are limited to completed Close_T before
    # parent rolling/monthly features run. No partial next month or T+1 helper
    # row may alter ranking, sector selection or VXN scaling.
    pricing_data_df = pricing_data_df.loc[pricing_data_df.index <= signal_date_ts].copy()
    universe_df = universe_df.loc[universe_df.index <= signal_date_ts].copy()
    vxn_signal_df = vxn_signal_df.loc[vxn_signal_df.index <= signal_date_ts].copy()
    if (pricing_data_df.empty or pricing_data_df.index.has_duplicates
            or not pricing_data_df.index.is_monotonic_increasing
            or pricing_data_df.index[-1] != signal_date_ts
            or universe_df.empty or universe_df.index[-1] != signal_date_ts
            or vxn_signal_df.empty or vxn_signal_df.index.has_duplicates
            or not vxn_signal_df.index.is_monotonic_increasing):
        raise ValueError("NDX E2 requires ordered prices/universe through the exact decision close and causal VXN history.")
    sector_by_symbol_dict = snapshot_module.load_ndx_e2_sector_map(
        list(universe_df.columns.astype(str)), snapshot_date_str=signal_date_str)
    execution_date_ts = pd.Timestamp(target_timestamp_ts.date())
    schedule_df = pd.DataFrame({"decision_date_ts": [signal_date_ts]},
        index=pd.DatetimeIndex([execution_date_ts], name="execution_date_ts"))
    strategy_obj = build_strategy(getattr(strategy_module, class_name_str), release_obj.pod_id_str,
        config_obj, schedule_df, vxn_signal_df, universe_df, sector_by_symbol_dict)
    strategy_host._seed_strategy_state(strategy_obj, pod_state_obj)
    signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    close_row_ser = signal_df.loc[signal_date_ts]
    # *** CRITICAL *** The parent aligns monthly features onto daily prices;
    # an invalid T can therefore retain a price row but have NaN regime/features.
    # Reject that case instead of converting unavailable evidence into all cash.
    # A defined False remains valid: regime_T = (Close_SPY,T > SMA200_SPY,T).
    regime_close_float = float(close_row_ser[(config_obj.regime_symbol_str, "Close")])
    regime_sma_float = float(close_row_ser[(config_obj.regime_symbol_str,
        f"regime_sma_{config_obj.index_trend_window_int}_ser")])
    regime_pass_obj = close_row_ser[(config_obj.regime_symbol_str, "regime_pass_bool")]
    if (not math.isfinite(regime_close_float) or regime_close_float <= 0
            or not math.isfinite(regime_sma_float) or regime_sma_float <= 0
            or pd.isna(regime_pass_obj)
            or bool(regime_pass_obj) != (regime_close_float > regime_sma_float)):
        raise ValueError("NDX E2 requires a valid completed monthly feature date and finite regime evidence.")
    active_symbol_list = universe_df.columns[universe_df.loc[signal_date_ts].eq(1)].astype(str).tolist()
    active_feature_df = close_row_ser.unstack().reindex(active_symbol_list)
    numeric_field_list = [f"monthly_roc_{config_obj.lookback_month_int}_ser", "atr_20_ser", "risk_adj_score_ser"]
    valid_feature_ser = pd.Series(np.isfinite(active_feature_df[numeric_field_list].to_numpy(dtype=float)).all(axis=1),
        index=active_feature_df.index)
    valid_feature_ser &= active_feature_df["atr_20_ser"].gt(0) & active_feature_df["stock_trend_pass_bool"].notna()
    if not valid_feature_ser.any():
        raise ValueError("NDX E2 requires valid active-member monthly candidate features.")
    strategy_obj.previous_bar = signal_date_ts
    strategy_obj.current_bar = execution_date_ts
    target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser)
    # Match legacy NDX strategy-memory bookkeeping. Research-generated Close_T
    # share orders are intentionally not forwarded: only frozen weights feed the
    # existing pre-open broker-NAV/reference-price sizing path.
    strategy_obj.iterate(signal_df.loc[:signal_date_ts], close_row_ser, pd.Series(dtype=float))
    target_weight_dict = {str(asset_str): float(weight_float) for asset_str, weight_float in target_weight_ser.items()
        if abs(float(weight_float)) > 1e-12}
    # weight_i,T = 1/10 * clip(22/VXN_asof(T), .25, 1); selected names are the
    # parent's descending ranking with at most four names per frozen sector.
    # The unchanged execution layer later uses floor(weight_i,T * NAV_submit *
    # budget_fraction / reference_price_i,submit), not a Close_T share target.
    vxn_row_ser = vxn_signal_df.iloc[-1]
    snapshot_metadata_dict.update({
        "strategy_family_str": "ndx_e2_natr20_sector_cap" if natr_bool else "ndx_e2_atr_sector_cap",
        "strategy_import_str": release_obj.strategy_import_str,
        "ndx_e2_sizing_contract_str": NDX_E2_SIZING_CONTRACT_STR,
        "research_quantity_basis_str": "Close_T",
        "sector_cap_int": config_obj.sector_cap_int,
        "sector_by_symbol_dict": dict(data_contract_dict["sector_by_symbol_dict"]),
        "sector_map_hash_str": data_contract_dict["sector_map_hash_str"],
        "sector_snapshot_date_str": data_contract_dict["sector_snapshot_date_str"],
        "sector_classification_basis_str": data_contract_dict["sector_classification_basis_str"],
        "sector_observed_timestamp_str": data_contract_dict["sector_observed_timestamp_str"],
        "vxn_symbol_str": config_obj.vxn_symbol_str,
        "vxn_reference_date_str": vxn_signal_df.index[-1].date().isoformat(),
        "vxn_exposure_scale_float": float(vxn_row_ser["vxn_exposure_scale_float"]),
        "vxn_close_float": float(vxn_row_ser["vxn_close"]),
        "target_vxn_pct_float": config_obj.target_vxn_pct_float,
        "min_exposure_scale_float": config_obj.min_exposure_scale_float,
        "max_exposure_scale_float": config_obj.max_exposure_scale_float,
    })
    final_metadata_dict = snapshot_module.build_data_source_metadata_dict(release_obj.data_profile_str)
    if any(final_metadata_dict.get(field_str) != snapshot_metadata_dict.get(field_str)
            for field_str in ("norgate_manifest_hash_str", "norgate_snapshot_date_str", "norgate_data_profile_str")):
        raise ValueError("NDX E2 snapshot identity changed during the decision.")
    return strategy_host._build_full_target_weight_decision_plan(
        release_obj=release_obj, signal_date_ts=signal_date_ts.to_pydatetime(),
        decision_base_position_map_dict={str(asset_str): float(amount_float)
            for asset_str, amount_float in strategy_obj.get_positions().items() if abs(float(amount_float)) > 1e-9},
        full_target_weight_map_dict=target_weight_dict,
        cash_reserve_weight_float=max(0.0, 1.0 - sum(target_weight_dict.values())),
        strategy_state_dict=strategy_host._extract_strategy_state_dict(strategy_obj),
        snapshot_metadata_dict=snapshot_metadata_dict)
