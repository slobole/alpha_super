"""Offline capsule behavior qualification from saved, trusted local research inputs.

No Norgate, broker, deployment, full backtest, or active configuration operation
is performed. The latest-session account is an explicitly reconstructed fixture,
not broker truth; matching its intents does not prove current data freshness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
if str(REPO_PATH_OBJ) not in sys.path:
    sys.path.insert(0, str(REPO_PATH_OBJ))

from scripts.research.mr_capsule_build_20261004.qualify_wiring import (
    _norgate_mode, _sha256_str, _source_hash_dict, compare_frames, validate_output_path,
)


def compare_gate_switches(gate_open_ser: pd.Series, pricing_index: pd.DatetimeIndex,
                          trading_decision_index: pd.DatetimeIndex) -> dict:
    """Compare the literal old VIX-row adjacency with the actual new method."""
    from strategies.mr_capsule.capsule_pod import CapsulePodMixin
    from strategies.mr_capsule.vix_stress_gate import gate_state_at

    pricing_df = pd.DataFrame(index=pricing_index)
    probe_obj = SimpleNamespace(gate_open_ser=gate_open_ser, previous_bar=None)
    mismatch_list = []
    trading_decision_set = set(trading_decision_index)
    for decision_ts in pricing_index:
        probe_obj.previous_bar = decision_ts
        gate_open_bool = gate_state_at(gate_open_ser, decision_ts)
        # Literal pre-change logic: previous VIX row versus the as-of current row.
        gate_position_int = int(gate_open_ser.index.searchsorted(decision_ts, side="right")) - 1
        old_switch_bool = gate_position_int >= 1 and bool(gate_open_ser.iloc[gate_position_int - 1]) != gate_open_bool
        new_switch_bool = CapsulePodMixin._gate_switched_at_decision(probe_obj, gate_open_bool, pricing_df)
        if old_switch_bool != new_switch_bool:
            mismatch_list.append({"decision_date_str": str(decision_ts.date()), "old_switch_bool": bool(old_switch_bool),
                                  "new_switch_bool": bool(new_switch_bool), "used_by_baseline_bool": decision_ts in trading_decision_set})
    active_mismatch_count_int = sum(row_dict["used_by_baseline_bool"] for row_dict in mismatch_list)
    return {"passed_bool": active_mismatch_count_int == 0, "compared_history_session_count_int": len(pricing_index),
            "baseline_decision_count_int": len(trading_decision_set), "full_history_mismatch_count_int": len(mismatch_list),
            "baseline_mismatch_count_int": active_mismatch_count_int, "mismatch_list": mismatch_list,
            "boundary_note_str": "Every input session is checked; differences outside the recorded baseline decision calendar are reported separately and cannot affect its executed orders."}


def parking_price_coverage(pricing_df: pd.DataFrame, transaction_df: pd.DataFrame,
                           execution_index: pd.DatetimeIndex) -> dict:
    """Reconstruct actual held parking quantities before and after each open."""
    if execution_index.empty or execution_index.has_duplicates or not execution_index.is_monotonic_increasing:
        return {"passed_bool": False, "error_str": "Baseline execution dates must be nonempty, unique and ordered."}
    if not execution_index.isin(pricing_df.index).all():
        return {"passed_bool": False, "error_str": "Baseline execution dates are absent from the saved pricing frame."}
    coverage_dict = {}
    for symbol_str in ("BIL", "SPMO"):
        symbol_transaction_df = transaction_df.loc[transaction_df["asset"].eq(symbol_str)].copy()
        symbol_transaction_df["bar"] = pd.to_datetime(symbol_transaction_df["bar"])
        if not symbol_transaction_df["bar"].isin(execution_index).all():
            return {"passed_bool": False, "error_str": f"{symbol_str} transactions fall outside the baseline NAV calendar."}
        # *** CRITICAL *** This is transaction accounting, not signal padding:
        # absent transactions mean zero delta. q_after(T)=sum_{s<=T} fills(s),
        # q_before(T)=q_after(T)-fills(T), including a liquidation on T itself.
        delta_vec = symbol_transaction_df.groupby("bar")["amount"].sum().reindex(execution_index, fill_value=0.0).to_numpy(dtype=float)
        held_after_vec = np.cumsum(delta_vec)
        held_before_vec = held_after_vec - delta_vec
        open_vec = pricing_df[(symbol_str, "Open")].reindex(execution_index).to_numpy(dtype=float)
        close_vec = pricing_df[(symbol_str, "Close")].reindex(execution_index).to_numpy(dtype=float)
        bad_open_vec = ~np.isfinite(open_vec) | (open_vec <= 0.0)
        bad_close_vec = ~np.isfinite(close_vec) | (close_vec <= 0.0)
        exposed_before_vec = np.abs(held_before_vec) > 1e-9
        exposed_after_vec = np.abs(held_after_vec) > 1e-9
        failure_vec = (exposed_before_vec | exposed_after_vec) & (bad_open_vec | bad_close_vec)
        failure_list = [{"date_str": str(execution_index[row_int].date()), "held_before_float": float(held_before_vec[row_int]),
                         "held_after_float": float(held_after_vec[row_int]), "open_str": str(open_vec[row_int]), "close_str": str(close_vec[row_int])}
                        for row_int in np.flatnonzero(failure_vec)]
        coverage_dict[symbol_str] = {"held_before_session_count_int": int(exposed_before_vec.sum()),
                                    "held_after_session_count_int": int(exposed_after_vec.sum()), "failure_list": failure_list}
    return {"passed_bool": all(not row_dict["failure_list"] for row_dict in coverage_dict.values()), "symbol_coverage_dict": coverage_dict,
            "scope_note_str": "If no historically held ETF encountered a missing execution Open/Close, the new parking liquidation exemption cannot change those recorded runs."}


def compare_csv_frames(current_df: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    """Require exact economic values; separately report constant global order-id offsets."""
    if set(current_df.columns) != set(original_df.columns) or len(current_df) != len(original_df):
        return {"passed_bool": False, "error_str": "CSV row count or columns differ.",
                "current_shape_list": list(current_df.shape), "original_shape_list": list(original_df.shape),
                "current_only_column_list": sorted(set(current_df.columns) - set(original_df.columns)),
                "original_only_column_list": sorted(set(original_df.columns) - set(current_df.columns))}
    numeric_column_list = [column_str for column_str in current_df.columns
                           if pd.api.types.is_numeric_dtype(current_df[column_str]) and pd.api.types.is_numeric_dtype(original_df[column_str])]
    numeric_result_dict = compare_frames(current_df[numeric_column_list], original_df[numeric_column_list]) if numeric_column_list and len(current_df) else {"passed_bool": True, "column_error_list": []}
    text_error_list = []
    for column_str in set(current_df.columns) - set(numeric_column_list):
        current_ser, original_ser = current_df[column_str], original_df[column_str]
        if column_str == "bar":
            current_ser, original_ser = pd.to_datetime(current_ser), pd.to_datetime(original_ser)
        mismatch_ser = ~(current_ser.eq(original_ser) | (current_ser.isna() & original_ser.isna()))
        if mismatch_ser.any():
            text_error_list.append({"column_str": str(column_str), "mismatch_count_int": int(mismatch_ser.sum())})
    differing_column_list = sorted({row_dict.get("column_str", "<schema_or_index>") for row_dict in numeric_result_dict["column_error_list"] + text_error_list})
    economic_difference_list = [column_str for column_str in differing_column_list if column_str != "order_id"]
    order_id_result_dict = {"present_bool": "order_id" in current_df, "constant_offset_allowed_bool": False}
    if "order_id" in differing_column_list:
        current_id_ser = pd.to_numeric(current_df["order_id"], errors="coerce")
        original_id_ser = pd.to_numeric(original_df["order_id"], errors="coerce")
        # Validate integer Series before any float conversion; 2**53 + 1 must
        # not round down and falsely look like a constant 2**53 offset.
        valid_bool = all(np.isfinite(value_ser.to_numpy()).all() and value_ser.ge(0).all()
                         and value_ser.mod(1).eq(0).all() and value_ser.le(2**53).all()
                         for value_ser in (current_id_ser, original_id_ser))
        if valid_bool:
            # Python integer subtraction avoids disguising an offset through float rounding.
            offset_list = [int(current_value) - int(original_value) for current_value, original_value in zip(current_id_ser, original_id_ser)]
            constant_bool = len(set(offset_list)) == 1
            order_id_result_dict.update(constant_offset_allowed_bool=constant_bool,
                offset_int=offset_list[0] if constant_bool else None, distinct_offset_count_int=len(set(offset_list)),
                example_current_list=current_df["order_id"].head(3).tolist(), example_original_list=original_df["order_id"].head(3).tolist())
        order_id_result_dict["rule_str"] = "Only a constant whole order_id offset is allowed: the engine counter persists across sequential runs. trade_id and every economic field remain exact."
    strict_equal_bool = numeric_result_dict["passed_bool"] and not text_error_list
    return {"passed_bool": strict_equal_bool or (not economic_difference_list and order_id_result_dict["constant_offset_allowed_bool"]),
            "strict_equal_bool": strict_equal_bool, "economic_columns_equal_bool": not economic_difference_list,
            "differing_column_list": differing_column_list, "economic_differing_column_list": economic_difference_list,
            "order_id_comparison_dict": order_id_result_dict, "numeric_comparison_dict": numeric_result_dict,
            "text_error_list": text_error_list, "row_count_int": len(current_df),
            "column_order_equal_bool": current_df.columns.equals(original_df.columns)}


def _read_csv(path_obj: Path) -> pd.DataFrame:
    return pd.read_csv(path_obj, float_precision="round_trip")


def _canonical_research_intents(strategy_obj, decision_nav_float: float) -> dict:
    entry_weight_dict, share_target_dict, exit_symbol_list, entry_priority_list = {}, {}, [], []
    raw_order_list = []
    for order_obj in strategy_obj.get_orders():
        symbol_str, amount_float = str(order_obj.asset), float(order_obj.amount)
        if type(order_obj).__name__ != "MarketOrder":
            raise ValueError("Unexpected research order class.")
        raw_order_list.append({"symbol_str": symbol_str, "amount_float": amount_float, "unit_str": str(order_obj.unit), "target_bool": bool(order_obj.target)})
        if order_obj.target and amount_float == 0.0:
            exit_symbol_list.append(symbol_str)
        elif order_obj.target and order_obj.unit == "shares" and symbol_str in {"BIL", "SPMO"}:
            share_target_dict[symbol_str] = amount_float
        elif not order_obj.target and order_obj.unit == "value" and amount_float > 0.0:
            entry_weight_dict[symbol_str] = amount_float / decision_nav_float
            entry_priority_list.append(symbol_str)
        else:
            raise ValueError(f"Unexpected research intent: {raw_order_list[-1]}")
    return {"entry_weight_dict": entry_weight_dict, "share_target_dict": share_target_dict,
            "exit_symbol_list": sorted(exit_symbol_list), "entry_priority_list": entry_priority_list, "raw_order_list": raw_order_list}


def _research_decision(pod_str, mode_str, pricing_df, universe_df, vix_close_ser, position_dict, cash_float, nav_float, signal_date_ts, execution_date_ts, signal_cache_dict=None):
    from strategies.mr_capsule.dv2_vix_gated import DV2VixGatedStrategy, default_trade_id_int
    from strategies.mr_capsule.hpi_vote_vix_gated import HPIVoteVixGatedStrategy
    from strategies.hpi.stateful_long import TURNOVER_FIELD_STR, ENTRY_HORIZON_VOTE_STR

    stock_trade_dict = {symbol_str: trade_id_int for trade_id_int, symbol_str in enumerate(sorted(set(position_dict) - {"BIL", "SPMO"}), start=1)}
    if pod_str == "dv2":
        strategy_obj = DV2VixGatedStrategy(name="offline_research_fixture", benchmarks=["$SPX"], capital_base=nav_float)
        strategy_obj.trade_id = len(stock_trade_dict)
        strategy_obj.current_trade = defaultdict(default_trade_id_int, stock_trade_dict)
    else:
        strategy_obj = HPIVoteVixGatedStrategy(name="offline_research_fixture", benchmarks=["$SPXTR"], capital_base=nav_float,
            ranking_field_str=TURNOVER_FIELD_STR, entry_mode_str=ENTRY_HORIZON_VOTE_STR, backtest_start_date_str="2004-01-01")
        strategy_obj.trade_id_int = len(stock_trade_dict)
        strategy_obj.current_trade_map = defaultdict(default_trade_id_int, stock_trade_dict)
        strategy_obj.pending_exit_symbol_set = set()
    # Independent seeding: deliberately do not call live _seed_strategy_state.
    strategy_obj._position_amount_map = dict(position_dict)
    strategy_obj._total_value_history_list = [nav_float]
    strategy_obj.cash = cash_float
    strategy_obj.universe_df = universe_df
    strategy_obj.vix_close_ser = vix_close_ser
    strategy_obj.parking_enabled_bool = mode_str != "cash"
    strategy_obj.spmo_parking_enabled_bool = mode_str == "spmo"
    strategy_obj.require_current_gate_observation_bool = True
    # *** CRITICAL *** Cached stock indicators must use this same <= Close_T
    # input object. The fixed stock indicators do not depend on parking mode;
    # each new research object still rebuilds its own VIX gate and parking state.
    if pricing_df.index[-1] != signal_date_ts:
        raise ValueError("Reference indicator input must end at the exact decision session.")
    if signal_cache_dict is not None and "signal_df" in signal_cache_dict:
        if signal_cache_dict["pricing_object_id_int"] != id(pricing_df) or signal_cache_dict["pod_str"] != pod_str:
            raise ValueError("Reference indicator cache belongs to different inputs or stock rules.")
        strategy_obj._prepare_capsule_state()
        signal_df = signal_cache_dict["signal_df"]
    else:
        signal_df = strategy_obj.compute_signals(pricing_df.copy())
        if signal_cache_dict is not None:
            signal_cache_dict.update(signal_df=signal_df, pricing_object_id_int=id(pricing_df), pod_str=pod_str)
    strategy_obj.previous_bar, strategy_obj.current_bar = signal_date_ts, execution_date_ts
    strategy_obj.iterate(signal_df, signal_df.loc[signal_date_ts], pd.Series(1.0, index=sorted(position_dict), dtype=float))
    return strategy_obj


def _host_fixture(pod_str: str, mode_str: str, pricing_df: pd.DataFrame, universe_df: pd.DataFrame,
                  vix_close_ser: pd.Series, inputs_path_obj: Path, input_hash_dict: dict, signal_cache_dict: dict | None = None) -> dict:
    from alpha.live import mr_capsule_adapter, scheduler_utils, strategy_host
    from alpha.live.models import PodState
    from alpha.live.release_manifest import parse_release_manifest
    from data.norgate_loader import use_norgate_data_profile
    from data.norgate_snapshot_store import load_valid_snapshot_manifest
    from strategies.hpi import stateful_long as hpi_module
    from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated, vix_stress_gate

    signal_date_ts = pd.Timestamp(pricing_df.index[-1])
    pod_name_str = "dv2_vix_gated" if pod_str == "dv2" else "hpi_vote_vix_gated"
    release_obj = parse_release_manifest(str(REPO_PATH_OBJ / f"docs/live/release_templates/pod_mr_{pod_name_str}_{mode_str}_daily_moo.yaml.example"))
    release_obj = replace(release_obj, mode_str="incubation", account_route_str=f"SIM_OFFLINE_{pod_str}_{mode_str}")
    latest_member_ser = universe_df.loc[signal_date_ts]
    member_symbol_list = sorted(str(symbol_str) for symbol_str in latest_member_ser[latest_member_ser.eq(1)].index)
    close_row_ser = pricing_df.loc[signal_date_ts]
    stock_symbol_list = [symbol_str for symbol_str in member_symbol_list
                         if np.isfinite(float(close_row_ser.get((symbol_str, "Close"), np.nan)))
                         and 10.0 <= float(close_row_ser[(symbol_str, "Close")]) <= 1_000.0][:2]
    if len(stock_symbol_list) != 2:
        raise ValueError("Cannot form the frozen fixture recipe of two current stocks priced between $10 and $1,000.")
    position_dict = {symbol_str: float(int(10_000.0 / float(close_row_ser[(symbol_str, "Close")]))) for symbol_str in stock_symbol_list}
    if mode_str != "cash":
        position_dict["BIL"] = float(int(20_000.0 / float(close_row_ser[("BIL", "Close")])))
    if mode_str == "spmo":
        position_dict["SPMO"] = float(int(10_000.0 / float(close_row_ser[("SPMO", "Close")])))
    cash_float = 50_000.0
    nav_float = cash_float + sum(share_float * float(close_row_ser[(symbol_str, "Close")]) for symbol_str, share_float in position_dict.items())
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS")
    execution_date_ts = pd.Timestamp(scheduler_utils.next_business_day_timestamp_ts(signal_date_ts, "XNYS").date())
    state_dict = {"trade_id_int": 2, "current_trade_map": {symbol_str: trade_id_int for trade_id_int, symbol_str in enumerate(stock_symbol_list, start=1)},
                  "pending_exit_symbol_list": [], "mr_capsule_strategy_import_str": release_obj.strategy_import_str}
    state_obj = PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=position_dict, cash_float=cash_float, total_value_float=nav_float, strategy_state_dict=state_dict,
        updated_timestamp_ts=close_timestamp_ts + timedelta(minutes=15), snapshot_stage_str="eod", snapshot_source_str="virtual_broker")
    snapshot_root_path_obj = inputs_path_obj / "snapshots"
    manifest_path_obj = snapshot_root_path_obj / release_obj.data_profile_str / signal_date_ts.date().isoformat() / "manifest.json"
    if manifest_path_obj.exists():
        manifest_obj = load_valid_snapshot_manifest(release_obj.data_profile_str, snapshot_date_str=signal_date_ts.date().isoformat(), snapshot_root_str=str(snapshot_root_path_obj))
        manifest_hash_str = manifest_obj.manifest_hash_str
        metadata_origin_str = "Actual isolated exported manifest; data access uses saved direct frames."
    else:
        manifest_hash_str = input_hash_dict[f"direct_{pod_str}_pricing"]
        metadata_origin_str = "Reconstructed fixture identity using saved-input hash; no exported manifest was available."
    metadata_dict = {"norgate_data_source_mode_str": "snapshot", "norgate_snapshot_date_str": signal_date_ts.date().isoformat(),
        "norgate_data_profile_str": release_obj.data_profile_str, "norgate_manifest_hash_str": manifest_hash_str,
        "qualification_fixture_str": "reconstructed historical snapshot/account fixture; not broker truth", "metadata_origin_str": metadata_origin_str}

    def saved_hpi_inputs(**keyword_dict):
        if keyword_dict["end_date_str"] != signal_date_ts.date().isoformat():
            raise ValueError("Unexpected HPI fixture date request.")
        stock_frame_df = pricing_df.loc[:, ~pricing_df.columns.get_level_values(0).isin(["BIL", "SPMO"])].copy()
        return list(universe_df.columns), universe_df.copy(), stock_frame_df

    def saved_parking_prices(symbol_str, **keyword_dict):
        if symbol_str not in {"BIL", "SPMO"} or keyword_dict["end_date_str"] != signal_date_ts.date().isoformat():
            raise ValueError("Unexpected ETF fixture data request.")
        return pricing_df.xs(symbol_str, level=0, axis=1).copy()

    # Patch data-access boundaries only. Indicators, gate/parking rules, host
    # state seeding, calendar, feature-readiness checks and order mapping execute.
    host_start_float = time.monotonic()
    with ExitStack() as stack_obj:
        stack_obj.enter_context(_norgate_mode(snapshot_root_path_obj))
        stack_obj.enter_context(use_norgate_data_profile(release_obj.data_profile_str))
        stack_obj.enter_context(patch("data.norgate_loader._load_direct_norgate_module", side_effect=RuntimeError("Offline qualification forbids Norgate calls.")))
        stack_obj.enter_context(patch.object(mr_capsule_adapter, "build_data_source_metadata_dict", side_effect=lambda *_: dict(metadata_dict)))
        stack_obj.enter_context(patch.object(strategy_host, "build_data_source_metadata_dict", side_effect=lambda *_: dict(metadata_dict)))
        stack_obj.enter_context(patch.object(dv2_vix_gated, "load_pricing_data", side_effect=lambda *_: (pricing_df.copy(), universe_df.copy())))
        stack_obj.enter_context(patch.object(hpi_module, "load_exact_hpi_inputs", side_effect=saved_hpi_inputs))
        stack_obj.enter_context(patch.object(hpi_vote_vix_gated, "load_price_timeseries", side_effect=saved_parking_prices))
        stack_obj.enter_context(patch.object(vix_stress_gate, "load_vix_close_ser", side_effect=lambda *_: vix_close_ser.copy()))
        host_decision_obj = strategy_host.build_decision_plan_for_release(release_obj, close_timestamp_ts + timedelta(hours=2), state_obj)
    host_elapsed_float = time.monotonic() - host_start_float
    reference_cache_reused_bool = signal_cache_dict is not None and "signal_df" in signal_cache_dict
    reference_start_float = time.monotonic()
    research_obj = _research_decision(pod_str, mode_str, pricing_df, universe_df, vix_close_ser, position_dict, cash_float, nav_float, signal_date_ts, execution_date_ts, signal_cache_dict)
    reference_elapsed_float = time.monotonic() - reference_start_float
    research_intent_dict = _canonical_research_intents(research_obj, nav_float)
    host_intent_dict = {"entry_weight_dict": host_decision_obj.entry_target_weight_map_dict,
        "share_target_dict": host_decision_obj.target_share_map_dict, "exit_symbol_list": sorted(host_decision_obj.exit_asset_set),
        "entry_priority_list": host_decision_obj.entry_priority_list}
    mismatched_field_list = [field_str for field_str, value_obj in host_intent_dict.items() if value_obj != research_intent_dict[field_str]]
    expected_state_dict = {"mr_capsule_strategy_import_str": release_obj.strategy_import_str,
        "trade_id_int": int(research_obj.trade_id if pod_str == "dv2" else research_obj.trade_id_int),
        "current_trade_map": dict(research_obj.current_trade if pod_str == "dv2" else research_obj.current_trade_map)}
    if pod_str == "hpi":
        expected_state_dict["pending_exit_symbol_list"] = sorted(research_obj.pending_exit_symbol_set)
    if host_decision_obj.strategy_state_dict != expected_state_dict:
        mismatched_field_list.append("strategy_state_dict")
    if host_decision_obj.decision_base_position_map != position_dict or host_decision_obj.snapshot_metadata_dict["decision_nav_float"] != nav_float:
        mismatched_field_list.append("positions_or_nav")
    if pd.Timestamp(host_decision_obj.target_execution_timestamp_ts.date()) != execution_date_ts:
        mismatched_field_list.append("execution_session")
    return {"passed_bool": not mismatched_field_list, "mismatched_field_list": mismatched_field_list,
        "strategy_import_str": release_obj.strategy_import_str, "decision_date_str": str(signal_date_ts.date()), "execution_date_str": str(execution_date_ts.date()),
        "position_dict": position_dict, "cash_float": cash_float, "nav_float": nav_float,
        "fixture_state_dict": state_dict, "host_state_dict": host_decision_obj.strategy_state_dict, "research_state_dict": expected_state_dict,
        "host_intent_dict": host_intent_dict, "research_intent_dict": research_intent_dict,
        "fixture_only_bool": True, "broker_truth_bool": False, "current_freshness_proven_bool": False,
        "metadata_origin_str": metadata_origin_str, "manifest_or_fixture_hash_str": manifest_hash_str,
        "host_compute_signals_unmocked_bool": True, "host_elapsed_seconds_float": host_elapsed_float,
        "reference_elapsed_seconds_float": reference_elapsed_float, "reference_indicator_cache_reused_bool": reference_cache_reused_bool,
        "reference_cache_input_sha256_str": input_hash_dict[f"direct_{pod_str}_pricing"],
        "reference_cache_rule_str": "Stock indicators computed once per pod from the identical saved frame ending at Close_T. Fresh independent account/trade state and VIX/parking state are seeded for each mode; every host call computes all indicators normally.",
        "entry_intent_count_int": len(host_decision_obj.entry_target_weight_map_dict), "exit_intent_count_int": len(host_decision_obj.exit_asset_set),
        "parking_target_count_int": len(host_decision_obj.target_share_map_dict)}


def run_offline_qualification(inputs_path_obj: Path, baseline_path_obj: Path, original_path_obj: Path, output_path_obj: Path) -> dict:
    from strategies.mr_capsule.vix_stress_gate import stress_gate_open_ser

    required_input_list = [f"direct_{pod_str}_{kind_str}" for pod_str in ("dv2", "hpi") for kind_str in ("pricing", "universe")] + ["direct_vix"]
    for name_str in required_input_list:
        if not (inputs_path_obj / f"{name_str}.pkl.gz").is_file():
            raise FileNotFoundError(f"Saved input is not ready: {name_str}.pkl.gz")
    for pod_str in ("dv2", "hpi"):
        for mode_str in ("parked", "cash", "bil"):
            for kind_str in ("nav", "transactions"):
                if not (baseline_path_obj / f"{pod_str}_{mode_str}_{kind_str}.csv").is_file():
                    raise FileNotFoundError(f"Baseline is not ready: {pod_str}_{mode_str}_{kind_str}.csv")
    output_path_obj = validate_output_path(output_path_obj, REPO_PATH_OBJ / "results" / "research")
    output_path_obj.mkdir(parents=True, exist_ok=True)
    input_hash_dict = {name_str: _sha256_str(inputs_path_obj / f"{name_str}.pkl.gz") for name_str in required_input_list}
    source_hash_dict = _source_hash_dict()
    additional_source_list = ["scripts/research/mr_capsule_build_20261004/qualify_wiring_offline.py", "alpha/engine/strategy.py", "alpha/engine/backtest.py",
                              "alpha/engine/order.py", "alpha/live/release_manifest.py", "alpha/live/scheduler_utils.py"]
    for pod_name_str in ("dv2_vix_gated", "hpi_vote_vix_gated"):
        for mode_str in ("cash", "bil", "spmo"):
            additional_source_list.extend([f"strategies/mr_capsule/strategy_mr_{pod_name_str}_{mode_str}.py",
                f"docs/live/release_templates/pod_mr_{pod_name_str}_{mode_str}_daily_moo.yaml.example"])
    for path_str in additional_source_list:
        source_hash_dict[path_str] = _sha256_str(REPO_PATH_OBJ / path_str)
    report_dict = {"status_str": "running", "started_at_str": datetime.now(timezone.utc).isoformat(),
        "input_sha256_dict": input_hash_dict, "source_sha256_dict": source_hash_dict, "baseline_comparison_dict": {}, "baseline_artifact_hash_dict": {},
        "gate_comparison_dict": {}, "parking_price_coverage_dict": {}, "host_fixture_dict": {}, "skipped_original_list": [],
        "all_six_historical_reproductions_confirmed_bool": False, "native_snapshot_parity_confirmed_bool": False,
        "scope_str": "Historical data/behavior qualification only. Reconstructed accounts are not broker evidence or current freshness approval."}
    report_path_obj = output_path_obj / "offline_qualification.json"
    try:
        input_report_path_obj = inputs_path_obj / "qualification.json"
        if input_report_path_obj.exists():
            input_report_dict = json.loads(input_report_path_obj.read_text(encoding="utf-8"))
            for name_str, hash_str in input_hash_dict.items():
                if input_report_dict.get("artifact_dict", {}).get(name_str, {}).get("sha256_str") != hash_str:
                    raise ValueError(f"Saved input hash does not match qualification provenance: {name_str}.")
            report_dict["input_transport_qualification_status_str"] = input_report_dict.get("status_str", "unknown")
            report_dict["native_snapshot_parity_confirmed_bool"] = input_report_dict.get("status_str") == "pass"
            report_dict["input_hash_provenance_verified_bool"] = True
        else:
            raise ValueError("Saved input qualification.json is required to verify its recorded artifact hashes.")
        vix_close_ser = pd.read_pickle(inputs_path_obj / "direct_vix.pkl.gz")
        gate_open_ser = stress_gate_open_ser(vix_close_ser)
        for pod_str in ("dv2", "hpi"):
            print(f"Offline historical checks: {pod_str}", flush=True)
            pricing_df = pd.read_pickle(inputs_path_obj / f"direct_{pod_str}_pricing.pkl.gz")
            universe_df = pd.read_pickle(inputs_path_obj / f"direct_{pod_str}_universe.pkl.gz")
            decision_date_set = set()
            for mode_str in ("parked", "cash", "bil"):
                tag_str = f"{pod_str}_{mode_str}"
                nav_df = _read_csv(baseline_path_obj / f"{tag_str}_nav.csv")
                transaction_df = _read_csv(baseline_path_obj / f"{tag_str}_transactions.csv")
                execution_index = pd.DatetimeIndex(pd.to_datetime(nav_df.iloc[:, 0]))
                expected_execution_index = pricing_df.index[pricing_df.index >= pd.Timestamp("2004-01-01")]
                if not execution_index.equals(expected_execution_index):
                    raise ValueError(f"{tag_str} NAV dates must equal the complete saved pricing calendar from 2004-01-01 through its latest session.")
                prior_position_vec = pricing_df.index.searchsorted(execution_index, side="left") - 1
                if (prior_position_vec < 0).any():
                    raise ValueError("Baseline starts without a prior input session.")
                decision_date_set.update(pricing_df.index[prior_position_vec])
                report_dict["parking_price_coverage_dict"][tag_str] = parking_price_coverage(pricing_df, transaction_df, execution_index)
                for kind_str, frame_df in (("nav", nav_df), ("transactions", transaction_df)):
                    file_name_str = f"{tag_str}_{kind_str}.csv"
                    report_dict["baseline_artifact_hash_dict"][file_name_str] = _sha256_str(baseline_path_obj / file_name_str)
                    original_file_path_obj = original_path_obj / file_name_str
                    if not original_file_path_obj.is_file():
                        report_dict["skipped_original_list"].append(file_name_str)
                        report_dict["baseline_comparison_dict"][file_name_str] = {"status_str": "skipped", "reason_str": "Original artifact is absent."}
                    else:
                        report_dict["baseline_comparison_dict"][file_name_str] = {
                            **compare_csv_frames(frame_df, _read_csv(original_file_path_obj)), "original_sha256_str": _sha256_str(original_file_path_obj)}
            report_dict["gate_comparison_dict"][pod_str] = compare_gate_switches(gate_open_ser, pricing_df.index, pd.DatetimeIndex(sorted(decision_date_set)))
            signal_cache_dict = {}
            for mode_str in ("cash", "bil", "spmo"):
                print(f"Real indicators and independent latest-session intent fixture: {pod_str}/{mode_str}", flush=True)
                report_dict["host_fixture_dict"][f"{pod_str}_{mode_str}"] = _host_fixture(pod_str, mode_str, pricing_df, universe_df, vix_close_ser, inputs_path_obj, input_hash_dict, signal_cache_dict)
                report_path_obj.write_text(json.dumps(report_dict, indent=2, allow_nan=False), encoding="utf-8")
            del pricing_df, universe_df, signal_cache_dict
        report_dict["changed_source_path_list"] = [path_str for path_str, hash_str in source_hash_dict.items() if _sha256_str(REPO_PATH_OBJ / path_str) != hash_str]
        report_dict["changed_input_path_list"] = [name_str for name_str, hash_str in input_hash_dict.items() if _sha256_str(inputs_path_obj / f"{name_str}.pkl.gz") != hash_str]
        result_list = [result_dict for key_str in ("baseline_comparison_dict", "gate_comparison_dict", "parking_price_coverage_dict", "host_fixture_dict")
                       for result_dict in report_dict[key_str].values() if result_dict.get("status_str") != "skipped"]
        report_dict["passed_bool"] = not report_dict["changed_source_path_list"] and not report_dict["changed_input_path_list"] and all(result_dict["passed_bool"] for result_dict in result_list)
        report_dict["all_six_historical_reproductions_confirmed_bool"] = (
            len(report_dict["baseline_comparison_dict"]) == 12 and not report_dict["skipped_original_list"]
            and all(result_dict.get("passed_bool", False) for result_dict in report_dict["baseline_comparison_dict"].values())
        )
        report_dict["status_str"] = ("pass_with_skips" if report_dict["skipped_original_list"] else "pass") if report_dict["passed_bool"] else "fail"
    except Exception as exception_obj:
        report_dict.update(status_str="fail", passed_bool=False, error_dict={"type_str": type(exception_obj).__name__, "message_str": str(exception_obj)})
    finally:
        report_dict["completed_at_str"] = datetime.now(timezone.utc).isoformat()
        report_path_obj.write_text(json.dumps(report_dict, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{report_dict['status_str'].upper()}: {report_path_obj}", flush=True)
    return report_dict


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    for argument_str in ("inputs-dir", "baseline-dir", "original-dir", "output-dir"):
        parser_obj.add_argument(f"--{argument_str}", required=True, type=Path)
    args_obj = parser_obj.parse_args()
    report_dict = run_offline_qualification(args_obj.inputs_dir, args_obj.baseline_dir, args_obj.original_dir, args_obj.output_dir)
    return 0 if report_dict["passed_bool"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
