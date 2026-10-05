"""MR capsule: exact-session signals and one Close_T account sizing basis.

Parking quantities are frozen at Close_T; stock entry dollars use the same NAV.
The execution layer converts those dollars with its observed reference quote.
This adapter does not submit orders or enable a deployment.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data.norgate_loader import build_data_source_metadata_dict


MR_CAPSULE_CONTRACT_STR = "mr_capsule_close_targets_v1"
MR_CAPSULE_STRATEGY_IMPORT_TUPLE = tuple(
    f"strategies.mr_capsule.strategy_mr_{pod_str}_{mode_str}"
    for pod_str in ("dv2_vix_gated", "hpi_vote_vix_gated")
    for mode_str in ("cash", "bil", "spmo")
)
MR_CAPSULE_PROFILE_TUPLE = (
    "norgate_eod_sp500_mr_capsule_pit",
    "norgate_eod_sp500_hpi_mr_capsule_pit",
)


def validate_mr_capsule_release(release_obj: LiveRelease) -> None:
    if release_obj.strategy_import_str not in MR_CAPSULE_STRATEGY_IMPORT_TUPLE:
        raise ValueError("Unsupported MR capsule strategy identity.")
    if release_obj.mode_str == "live":
        raise ValueError("MR capsule LIVE trading is locked pending explicit owner approval; use paper or incubation.")
    hpi_bool = "hpi_vote" in release_obj.strategy_import_str
    required_field_dict = {
        "data_profile_str": MR_CAPSULE_PROFILE_TUPLE[int(hpi_bool)],
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": "eod_snapshot_ready",
        "execution_policy_str": "next_open_moo",
        "pod_budget_fraction_float": 1.0,
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj:
            raise ValueError(f"MR capsule requires {field_str}={expected_value_obj!r}.")
    if set(release_obj.params_dict) - {"capital_base_float", "margin_account_confirmed_bool"}:
        raise ValueError("MR capsule uses its frozen strategy parameters; overrides require parity qualification.")
    if release_obj.enabled_bool and release_obj.mode_str != "incubation" and (
        release_obj.params_dict.get("margin_account_confirmed_bool") is not True
    ):
        raise ValueError("MR capsule same-auction funding requires an explicitly confirmed margin account.")


def _validate_vix_history(vix_close_ser: pd.Series, pricing_index: pd.DatetimeIndex, signal_date_ts: pd.Timestamp) -> None:
    # *** CRITICAL *** Only observations through Close_T can affect the expanding
    # threshold. An as-of fallback must never turn a stale VIX into today's signal.
    if (
        vix_close_ser.index.has_duplicates
        or not vix_close_ser.index.is_monotonic_increasing
        or len(vix_close_ser) < 500
        or vix_close_ser.index[0] != pd.Timestamp("1990-01-02")
        or vix_close_ser.index[-1] != signal_date_ts
        or not np.isfinite(vix_close_ser.to_numpy(dtype=float)).all()
        or not vix_close_ser.gt(0.0).all()
        or not pricing_index.isin(vix_close_ser.index).all()
    ):
        raise ValueError("MR capsule requires complete, ordered, finite VIX history from 1990-01-02 through the exact decision session.")


def _validate_account_state(release_obj: LiveRelease, pod_state_obj: PodState | None, signal_date_ts: pd.Timestamp, as_of_ts: datetime) -> dict[str, float]:
    if pod_state_obj is None or any(
        getattr(pod_state_obj, field_str) != getattr(release_obj, field_str)
        for field_str in ("pod_id_str", "user_id_str", "account_route_str")
    ):
        raise ValueError("MR capsule requires its own trusted EOD account state.")
    state_timestamp_ts = scheduler_utils.to_market_timestamp_ts(pod_state_obj.updated_timestamp_ts, "XNYS")
    expected_source_str = "virtual_broker" if release_obj.mode_str == "incubation" else "broker"
    if (
        pod_state_obj.snapshot_stage_str != "eod"
        or pod_state_obj.snapshot_source_str != expected_source_str
        or state_timestamp_ts.date() != signal_date_ts.date()
        or state_timestamp_ts < scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS")
        or state_timestamp_ts > scheduler_utils.to_market_timestamp_ts(as_of_ts, "XNYS")
        or not math.isfinite(float(pod_state_obj.cash_float))
        or not math.isfinite(float(pod_state_obj.total_value_float))
        or pod_state_obj.total_value_float <= 0.0
    ):
        raise ValueError("MR capsule requires a finite, exact-session EOD account snapshot; bootstrap or stale state is insufficient.")
    position_map_dict = {}
    for asset_str, amount_float in pod_state_obj.position_amount_map.items():
        amount_float = float(amount_float)
        if not math.isfinite(amount_float) or amount_float < 0.0 or amount_float != math.trunc(amount_float):
            raise ValueError("MR capsule requires nonnegative whole-share positions.")
        if amount_float:
            position_map_dict[str(asset_str)] = amount_float
    prior_identity_str = pod_state_obj.strategy_state_dict.get("mr_capsule_strategy_import_str")
    if prior_identity_str not in (None, release_obj.strategy_import_str) or (position_map_dict and prior_identity_str is None):
        raise ValueError("MR capsule cannot adopt positions or state from another strategy; reconcile initialization explicitly.")
    parking_mode_str = release_obj.strategy_import_str.rsplit("_", 1)[-1]
    if (parking_mode_str == "cash" and {"BIL", "SPMO"} & set(position_map_dict)) or (
        parking_mode_str == "bil" and "SPMO" in position_map_dict
    ):
        raise ValueError("MR capsule parking holdings do not match the frozen variant.")
    return position_map_dict


def build_mr_capsule_decision_plan(release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None) -> DecisionPlan:
    from alpha.live import strategy_host
    from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated
    from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

    validate_mr_capsule_release(release_obj)
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("MR capsule has no completed decision session.")
    position_map_dict = _validate_account_state(release_obj, pod_state_obj, signal_date_ts, as_of_ts)
    snapshot_metadata_dict = build_data_source_metadata_dict(release_obj.data_profile_str)
    if (
        snapshot_metadata_dict.get("norgate_snapshot_date_str") != signal_date_ts.date().isoformat()
        or snapshot_metadata_dict.get("norgate_data_profile_str") != release_obj.data_profile_str
        or not snapshot_metadata_dict.get("norgate_manifest_hash_str")
    ):
        raise ValueError("MR capsule requires the exact decision-session snapshot and manifest identity.")
    end_date_str = signal_date_ts.date().isoformat()
    hpi_bool = "hpi_vote" in release_obj.strategy_import_str
    # *** CRITICAL *** Same loaders and the same strategy builder as the Bench backtest
    # (run_dv2_capsule_pod / run_hpi_capsule_pod): one construction path for both.
    if hpi_bool:
        from strategies.hpi import stateful_long as hpi_module
        pricing_data_df, universe_df = hpi_vote_vix_gated.load_hpi_capsule_pricing_data(end_date_str)
        build_strategy_fn = hpi_vote_vix_gated.build_hpi_capsule_strategy
    else:
        pricing_data_df, universe_df = dv2_vix_gated.load_pricing_data(end_date_str)
        build_strategy_fn = dv2_vix_gated.build_dv2_capsule_strategy
    # *** CRITICAL *** Truncate the frame before indicators; the next session is
    # a calendar label only, never a source of tomorrow's prices or fills.
    pricing_data_df = pricing_data_df.loc[:signal_date_ts].copy()
    previous_session_ts = scheduler_utils.get_exchange_calendar_obj("XNYS").previous_session(signal_date_ts)
    if (
        pricing_data_df.index.has_duplicates or not pricing_data_df.index.is_monotonic_increasing
        or signal_date_ts not in pricing_data_df.index or previous_session_ts not in pricing_data_df.index
    ):
        raise ValueError("MR capsule requires ordered prices for T and the prior XNYS session.")
    vix_close_ser = load_vix_close_ser(end_date_str)
    _validate_vix_history(vix_close_ser, pricing_data_df.index, signal_date_ts)
    parking_mode_str = release_obj.strategy_import_str.rsplit("_", 1)[-1]
    required_symbol_set = set(position_map_dict)
    if parking_mode_str != "cash":
        required_symbol_set.add("BIL")
    if parking_mode_str == "spmo":
        required_symbol_set.add("SPMO")
    close_price_dict = {}
    for asset_str in required_symbol_set:
        close_float = float(pricing_data_df.loc[signal_date_ts].get((asset_str, "Close"), np.nan))
        if not math.isfinite(close_float) or close_float <= 0.0:
            raise ValueError(f"MR capsule has no valid decision close for {asset_str}; funding cannot be established.")
        close_price_dict[asset_str] = close_float
    if parking_mode_str == "spmo":
        if ("SPMO", "Volume") not in pricing_data_df.columns:
            raise ValueError("MR capsule SPMO requires observed trading-volume history.")
        # *** CRITICAL *** These are the same <= Close_T observations used by
        # the parking guard. A provider gap is not evidence of a no-trade day.
        spmo_volume_ser = pd.to_numeric(pricing_data_df[("SPMO", "Volume")].iloc[-21:], errors="coerce")
        if not np.isfinite(spmo_volume_ser.to_numpy(dtype=float)).all() or spmo_volume_ser.lt(0.0).any():
            raise ValueError("MR capsule SPMO requires finite nonnegative trading-volume history.")
    # NAV_Close_T = EOD_cash + sum(q_i * Close_i,T). Both stock entry dollars and
    # parking target shares use this same basis, independent of bootstrap capital.
    decision_nav_float = float(pod_state_obj.cash_float) + sum(
        amount_float * close_price_dict[asset_str] for asset_str, amount_float in position_map_dict.items()
    )
    if not math.isfinite(decision_nav_float) or decision_nav_float <= 0.0:
        raise ValueError("MR capsule Close_T NAV must be positive and finite.")
    strategy_obj = build_strategy_fn(
        strategy_name_str=release_obj.pod_id_str,
        parking_enabled_bool=parking_mode_str != "cash",
        spmo_parking_enabled_bool=parking_mode_str == "spmo",
        universe_df=universe_df,
        vix_close_ser=vix_close_ser,
        capital_base_float=float(pod_state_obj.total_value_float),
    )
    strategy_obj.require_current_gate_observation_bool = True
    strategy_host._seed_strategy_state(strategy_obj, pod_state_obj)
    strategy_obj._total_value_history_list = [decision_nav_float]
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    strategy_obj.previous_bar = signal_date_ts
    strategy_obj.current_bar = pd.Timestamp(scheduler_utils.next_business_day_timestamp_ts(signal_date_ts, "XNYS").date())
    close_row_ser = signal_df.loc[signal_date_ts]
    if hpi_bool:
        strategy_host._validate_hpi_live_feature_readiness(
            hpi_module=hpi_module, strategy_obj=strategy_obj, close_row_ser=close_row_ser,
            universe_df=universe_df, decision_date_ts=signal_date_ts,
        )
    # Inherited HPI contract: marker means assumed next-open tradability, not a
    # future price. Reconcile resolves actual fills under the capsule exit policy.
    tradable_open_marker_ser = pd.Series(1.0, index=sorted(position_map_dict), dtype=float)
    strategy_obj.iterate(signal_df, close_row_ser, tradable_open_marker_ser)
    snapshot_metadata_dict.update({
        "strategy_family_str": "mr_capsule_hpi_vote" if hpi_bool else "mr_capsule_dv2",
        "strategy_import_str": release_obj.strategy_import_str,
        "sizing_contract_str": MR_CAPSULE_CONTRACT_STR,
        "mr_capsule_parking_mode_str": parking_mode_str,
        "decision_nav_float": decision_nav_float,
        "decision_cash_float": float(pod_state_obj.cash_float),
        "vix_decision_date_str": end_date_str,
        "gate_open_bool": strategy_obj._gate_open_at_decision(),
        "account_snapshot_timestamp_str": pod_state_obj.updated_timestamp_ts.isoformat(),
        "parking_trade_ids_used_for_live_identity_bool": False,
    })
    decision_plan_obj = strategy_host._build_decision_plan_from_orders(
        release_obj, signal_date_ts, strategy_obj, snapshot_metadata_dict,
    )
    return replace(decision_plan_obj, strategy_state_dict={
        **decision_plan_obj.strategy_state_dict,
        "mr_capsule_strategy_import_str": release_obj.strategy_import_str,
    })
