"""Audit cached native MR histories through the actual host, without provider/broker I/O.

The root audit runner owns native-data serialization and the historical run.
It saves pricing.parquet, auxiliary.pkl.gz (universe_df, vix_close_ser where
needed) and trace.pkl.gz (decision/execution dates, account state, orders and
post-iterate state). The baseline directory is supplied explicitly on the CLI.
This checker patches data access and historical snapshot identity only. It does
not qualify deployed snapshot transport, historical vintages or broker fills.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH_OBJ))

VARIANT_TEMPLATE_DICT = {
    "dv2": "pod_dv2_daily_moo.yaml.example",
    "dv2_ind": "pod_dv2_industry_etf_daily_moo.yaml.example",
    "hpi235": "pod_hpi_sp500_2_3_5_vote_daily_moo.yaml.example",
    **{f"{pod_str}_{mode_str}": f"pod_mr_{'dv2_vix_gated' if pod_str == 'dv2' else 'hpi_vote_vix_gated'}_{mode_str}_daily_moo.yaml.example"
       for pod_str in ("dv2", "hpi") for mode_str in ("cash", "bil", "spmo")},
}
SOURCE_PATH_TUPLE = (
    "alpha/live/strategy_host.py", "alpha/live/mr_capsule_adapter.py", "alpha/live/dv2_ind_adapter.py",
    "alpha/live/execution_engine.py", "alpha/live/daily_reconcile.py", "alpha/live/scheduler_utils.py",
    "alpha/live/models.py", "alpha/engine/strategy.py", "alpha/engine/backtest.py",
    "alpha/engine/backtester.py", "alpha/indicators.py", "data/norgate_loader.py",
    "data/norgate_snapshot_store.py", "strategies/dv2/strategy_mr_dv2.py",
    "strategies/dv2/strategy_mr_dv2_liquidity_floor.py", "strategies/dv2/strategy_mr_dv2_industry_etf.py",
    "strategies/hpi/stateful_long.py", "strategies/mr_capsule/dv2_vix_gated.py",
    "strategies/mr_capsule/hpi_vote_vix_gated.py", "strategies/mr_capsule/capsule_pod.py",
    "strategies/mr_capsule/vix_stress_gate.py", "strategies/mr_capsule/parking.py",
    "scripts/research/wired_signoff_20261015/mr_native_replay.py",
)


def source_hash_dict() -> dict[str, str]:
    return {path_str: hashlib.sha256((REPO_PATH_OBJ / path_str).read_bytes()).hexdigest()
            for path_str in SOURCE_PATH_TUPLE}


def install_decision_recorder(strategy_obj) -> dict:
    """Observe a historical strategy without changing its rules or orders."""
    from alpha.live import strategy_host

    trace_dict = {}
    original_iterate_fn = strategy_obj.iterate

    def recorded_iterate(signal_df, close_row_ser, open_price_ser):
        if signal_df is None or close_row_ser is None:
            return original_iterate_fn(signal_df, close_row_ser, open_price_ser)
        # *** CRITICAL *** Observe Close_T state before T+1 fills/dividends.
        # q_T, cash_T and NAV_T must be the account on the decision boundary.
        decision_ts = pd.Timestamp(strategy_obj.previous_bar)
        record_dict = {
            "position_amount_map": {str(asset_str): float(amount_float)
                for asset_str, amount_float in strategy_obj.get_positions().items() if amount_float != 0},
            "cash_float": float(strategy_obj.cash), "nav_float": float(strategy_obj.previous_total_value),
            "strategy_state_dict": strategy_host._extract_strategy_state_dict(strategy_obj),
            "missing_next_open_symbol_list": [str(asset_str) for asset_str, amount_float in strategy_obj.get_positions().items()
                if amount_float and not np.isfinite(float(open_price_ser.get(asset_str, np.nan)))],
        }
        original_iterate_fn(signal_df, close_row_ser, open_price_ser)
        record_dict["order_list"] = [strategy_host._build_order_shape_dict(order_obj) for order_obj in strategy_obj.get_orders()]
        record_dict["after_strategy_state_dict"] = strategy_host._extract_strategy_state_dict(strategy_obj)
        trace_dict[decision_ts] = record_dict

    strategy_obj.iterate = recorded_iterate
    return trace_dict


def select_replay_dates(trace_dict: dict, vix_close_ser: pd.Series | None, session_count_int: int) -> tuple[list, dict]:
    from strategies.mr_capsule.vix_stress_gate import stress_gate_open_ser

    eligible_date_list = sorted(date_ts for date_ts in trace_dict if date_ts >= pd.Timestamp("2019-01-01"))
    if len(eligible_date_list) < session_count_int:
        raise ValueError("Fewer than the required number of eligible historical sessions.")
    pool_dict = {"entries": [], "exits": [], "same_open_refill": [], "parking": [], "no_order": [],
                 "gate_opening": [], "gate_closing": [], "pending_exit": []}
    gate_ser = None if vix_close_ser is None else stress_gate_open_ser(vix_close_ser)
    for decision_ts in eligible_date_list:
        record_dict = trace_dict[decision_ts]
        order_list = record_dict["order_list"]
        entry_bool = any(not order_dict["target_bool"] and order_dict["unit_str"] == "value" for order_dict in order_list)
        exit_bool = any(order_dict["target_bool"] and order_dict["amount_float"] == 0 for order_dict in order_list)
        if entry_bool:
            pool_dict["entries"].append(decision_ts)
        if exit_bool:
            pool_dict["exits"].append(decision_ts)
        if entry_bool and exit_bool:
            pool_dict["same_open_refill"].append(decision_ts)
        if any(order_dict["asset_str"] in ("BIL", "SPMO") for order_dict in order_list):
            pool_dict["parking"].append(decision_ts)
        if not order_list:
            pool_dict["no_order"].append(decision_ts)
        if record_dict["strategy_state_dict"].get("pending_exit_symbol_list"):
            pool_dict["pending_exit"].append(decision_ts)
        if gate_ser is not None and decision_ts in gate_ser.index:
            gate_position_int = gate_ser.index.get_loc(decision_ts)
            if gate_position_int > 0 and bool(gate_ser.iloc[gate_position_int]) != bool(gate_ser.iloc[gate_position_int - 1]):
                pool_dict["gate_opening" if gate_ser.iloc[gate_position_int] else "gate_closing"].append(decision_ts)
    chosen_date_set = {eligible_date_list[-1]}
    for category_str in ("gate_opening", "gate_closing", "parking", "same_open_refill", "entries", "exits", "pending_exit", "no_order"):
        category_date_list = pool_dict[category_str]
        if category_date_list:
            for position_int in np.linspace(0, len(category_date_list)-1, min(4, len(category_date_list)), dtype=int):
                chosen_date_set.add(category_date_list[position_int])
    for position_int in np.linspace(0, len(eligible_date_list)-1, session_count_int, dtype=int):
        if len(chosen_date_set) >= session_count_int:
            break
        chosen_date_set.add(eligible_date_list[position_int])
    informative_date_list = [date_ts for date_ts in eligible_date_list if trace_dict[date_ts]["order_list"]]
    if len(informative_date_list) < 20:
        raise ValueError("Fewer than 20 native sessions with actual order intent are available.")
    for position_int in np.linspace(0, len(informative_date_list)-1, min(len(informative_date_list), 40), dtype=int):
        if sum(bool(trace_dict[date_ts]["order_list"]) for date_ts in chosen_date_set) >= 20:
            break
        chosen_date_set.add(informative_date_list[position_int])
    coverage_dict = {category_str: sum(date_ts in chosen_date_set for date_ts in category_date_list)
                     for category_str, category_date_list in pool_dict.items()}
    coverage_dict["with_order_intent"] = sum(bool(trace_dict[date_ts]["order_list"]) for date_ts in chosen_date_set)
    return sorted(chosen_date_set), coverage_dict


def compare_decision(decision_obj, record_dict: dict, variant_str: str, decision_ts: pd.Timestamp) -> dict:
    from alpha.live import scheduler_utils
    engine_order_list = record_dict["order_list"]
    expected_exit_set = {order_dict["asset_str"] for order_dict in engine_order_list
                         if order_dict["target_bool"] and order_dict["amount_float"] == 0}
    expected_entry_list = [(order_dict["asset_str"], float(order_dict["amount_float"])) for order_dict in engine_order_list
                           if not order_dict["target_bool"] and order_dict["unit_str"] == "value"]
    expected_target_dict = {order_dict["asset_str"]: float(order_dict["amount_float"]) for order_dict in engine_order_list
                            if order_dict["target_bool"] and order_dict["unit_str"] == "shares" and order_dict["amount_float"] > 0}
    nav_float = float(decision_obj.snapshot_metadata_dict.get("decision_nav_float", record_dict["nav_float"]))
    actual_entry_list = [(asset_str, float(decision_obj.entry_target_weight_map_dict[asset_str]) * nav_float)
                        for asset_str in decision_obj.entry_priority_list]
    expected_state_dict = record_dict.get("after_strategy_state_dict")
    allowed_extra_set = {"mr_capsule_strategy_import_str", "dv2_ind_strategy_import_str"}
    state_equal_bool = None if expected_state_dict is None else (
        all(decision_obj.strategy_state_dict.get(key_str) == value_obj for key_str, value_obj in expected_state_dict.items())
        and not (set(decision_obj.strategy_state_dict)-set(expected_state_dict)-allowed_extra_set))
    expected_signal_ts = scheduler_utils.get_session_close_timestamp_ts(decision_ts, "XNYS")
    execution_date_ts = pd.Timestamp(record_dict.get("execution_date", scheduler_utils.next_business_day_timestamp_ts(decision_ts, "XNYS").date()))
    expected_execution_ts = scheduler_utils.get_session_open_timestamp_ts(execution_date_ts, "XNYS")
    calendar_next_ts = scheduler_utils.next_business_day_timestamp_ts(decision_ts, "XNYS")
    expected_next_open_ts = scheduler_utils.get_session_open_timestamp_ts(pd.Timestamp(calendar_next_ts.date()), "XNYS")
    max_value_diff_float = max((abs(actual_value_float-expected_value_float)
        for (_, actual_value_float), (_, expected_value_float) in zip(actual_entry_list, expected_entry_list)), default=0.0)
    # DV2-IND explicitly values physical account shares with nominal Close_T;
    # historical adjusted-ledger account units are not physical broker evidence.
    # Its parity contract is entries/exits/order ranking and 1/10 weights only.
    sizing_equal_bool = (all(abs(float(weight_float)-0.1) <= 1e-12
                            for weight_float in decision_obj.entry_target_weight_map_dict.values()) if variant_str == "dv2_ind"
                         else max_value_diff_float <= 1e-8 * max(1.0, nav_float))
    result_dict = {
        "exit_equal_bool": decision_obj.exit_asset_set == expected_exit_set,
        "entry_order_equal_bool": [asset_str for asset_str, _ in actual_entry_list] == [asset_str for asset_str, _ in expected_entry_list],
        "parking_share_targets_equal_bool": decision_obj.target_share_map_dict == expected_target_dict,
        "strategy_state_equal_bool": state_equal_bool, "sizing_intent_equal_bool": sizing_equal_bool,
        "strategy_state_compared_key_list": [] if expected_state_dict is None else sorted(expected_state_dict),
        "strategy_state_unavailable_reason_str": "Older root trace lacks post-iterate state." if expected_state_dict is None else None,
        "signal_close_timestamp_equal_bool": decision_obj.signal_timestamp_ts == expected_signal_ts,
        "execution_timestamp_equal_bool": decision_obj.target_execution_timestamp_ts == expected_execution_ts == expected_next_open_ts,
        "entry_dollar_max_absolute_difference_float": max_value_diff_float,
        "research_nav_float": float(record_dict["nav_float"]), "host_decision_nav_float": nav_float,
        "expected_entry_list": expected_entry_list, "actual_entry_list": actual_entry_list,
        "expected_exit_list": sorted(expected_exit_set), "actual_exit_list": sorted(decision_obj.exit_asset_set),
        "expected_parking_target_dict": expected_target_dict, "actual_parking_target_dict": decision_obj.target_share_map_dict,
        "missing_next_open_symbol_list": record_dict.get("missing_next_open_symbol_list", []),
    }
    result_dict["passed_bool"] = all(result_dict[field_str] for field_str in
        ("exit_equal_bool", "entry_order_equal_bool", "parking_share_targets_equal_bool", "sizing_intent_equal_bool",
         "signal_close_timestamp_equal_bool", "execution_timestamp_equal_bool")) and state_equal_bool is True
    return result_dict


def build_cached_host_decision(variant_str, release_obj, decision_ts, record_dict, input_dict):
    from alpha.live import strategy_host, mr_capsule_adapter, dv2_ind_adapter, scheduler_utils
    from alpha.live.models import PodState
    from data import norgate_snapshot_store
    from strategies.dv2 import strategy_mr_dv2, strategy_mr_dv2_industry_etf
    from strategies.hpi import stateful_long
    from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated, vix_stress_gate

    pricing_df = input_dict["pricing_df"]
    universe_df = input_dict["universe_df"]
    # *** CRITICAL *** Cached native histories contain later rows. Parent
    # loaders return <= T; capsule/IND adapters additionally truncate themselves.
    prefix_df = pricing_df.loc[:decision_ts]
    prefix_universe_df = universe_df.loc[:decision_ts]
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(decision_ts, "XNYS")
    state_dict = dict(record_dict["strategy_state_dict"])
    if variant_str not in ("dv2", "hpi235", "dv2_ind"):
        state_dict["mr_capsule_strategy_import_str"] = release_obj.strategy_import_str
    if variant_str == "dv2_ind":
        state_dict["dv2_ind_strategy_import_str"] = release_obj.strategy_import_str
    account_obj = PodState(release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
        dict(record_dict["position_amount_map"]), float(record_dict["cash_float"]), float(record_dict["nav_float"]),
        state_dict, close_timestamp_ts + timedelta(minutes=30), snapshot_stage_str="eod", snapshot_source_str="broker")
    metadata_dict = {"norgate_snapshot_date_str": decision_ts.date().isoformat(),
        "norgate_data_profile_str": release_obj.data_profile_str, "norgate_manifest_hash_str": "native_cached_historical_fixture",
        "qualification_fixture_str": "current-vintage historical native inputs; snapshot identity stub; not broker truth"}
    with ExitStack() as stack_obj:
        stack_obj.enter_context(patch.object(strategy_host, "build_data_source_metadata_dict", side_effect=lambda *_: dict(metadata_dict)))
        stack_obj.enter_context(patch.object(mr_capsule_adapter, "build_data_source_metadata_dict", side_effect=lambda *_: dict(metadata_dict)))
        stack_obj.enter_context(patch.object(strategy_mr_dv2, "build_index_constituent_matrix", side_effect=lambda **_: (list(prefix_universe_df.columns), prefix_universe_df)))
        stack_obj.enter_context(patch.object(strategy_mr_dv2, "get_prices", side_effect=lambda *_a, **_k: prefix_df))
        stack_obj.enter_context(patch.object(stateful_long, "load_exact_hpi_inputs", side_effect=lambda **_: (list(prefix_universe_df.columns), prefix_universe_df, prefix_df)))
        stack_obj.enter_context(patch.object(dv2_vix_gated, "load_pricing_data", side_effect=lambda *_a, **_k: (pricing_df, universe_df)))
        stack_obj.enter_context(patch.object(hpi_vote_vix_gated, "load_hpi_capsule_pricing_data", side_effect=lambda *_a, **_k: (pricing_df, universe_df)))
        if "vix_close_ser" in input_dict:
            stack_obj.enter_context(patch.object(vix_stress_gate, "load_vix_close_ser", side_effect=lambda *_a, **_k: input_dict["vix_close_ser"].loc[:decision_ts]))
        if variant_str == "dv2_ind":
            stack_obj.enter_context(patch.object(norgate_snapshot_store, "is_snapshot_mode_enabled_bool", return_value=True))
            stack_obj.enter_context(patch.object(norgate_snapshot_store, "load_valid_snapshot_manifest",
                return_value=SimpleNamespace(snapshot_date_ts=decision_ts, manifest_hash_str="native_cached_historical_fixture")))
            stack_obj.enter_context(patch.object(strategy_mr_dv2_industry_etf, "get_prices", side_effect=lambda *_a, **_k: pricing_df))
        return strategy_host.build_decision_plan_for_release(release_obj, close_timestamp_ts + timedelta(hours=2), account_obj)


def main() -> int:
    from alpha.live.release_manifest import parse_release_manifest

    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--variant", choices=tuple(VARIANT_TEMPLATE_DICT), required=True)
    parser_obj.add_argument("--baseline-dir", type=Path, required=True)
    parser_obj.add_argument("--sessions", type=int, default=24)
    parser_obj.add_argument("--output-dir", type=Path, required=True)
    args_obj = parser_obj.parse_args()
    if args_obj.sessions < 20:
        raise ValueError("At least 20 real historical host sessions are required.")
    output_path_obj = args_obj.output_dir.resolve()
    review_root_path_obj = (REPO_PATH_OBJ / "results/review/wired_signoff_20261015").resolve()
    if not output_path_obj.is_relative_to(review_root_path_obj) or output_path_obj == review_root_path_obj:
        raise ValueError("Replay evidence must stay below the isolated sign-off results root.")
    output_path_obj.mkdir(parents=True, exist_ok=False)
    (output_path_obj / "executed_replay_checker.py").write_bytes(Path(__file__).read_bytes())
    input_path_obj = args_obj.baseline_dir / "pricing.parquet"
    auxiliary_path_obj = args_obj.baseline_dir / "auxiliary.pkl.gz"
    trace_path_obj = args_obj.baseline_dir / "trace.pkl.gz"
    input_dict = {"pricing_df": pd.read_parquet(input_path_obj), **pd.read_pickle(auxiliary_path_obj)}
    raw_trace_obj = pd.read_pickle(trace_path_obj)
    if isinstance(raw_trace_obj, list):
        trace_dict = {}
        for raw_record_dict in raw_trace_obj:
            record_dict = {"position_amount_map": raw_record_dict["positions"], "cash_float": raw_record_dict["cash"],
                "nav_float": raw_record_dict["nav"], "strategy_state_dict": raw_record_dict["state"],
                "order_list": raw_record_dict["orders"], "execution_date": raw_record_dict["execution_date"]}
            if "after_state" in raw_record_dict:
                record_dict["after_strategy_state_dict"] = raw_record_dict["after_state"]
            trace_dict[pd.Timestamp(raw_record_dict["decision_date"])] = record_dict
    else:
        trace_dict = {pd.Timestamp(date_obj): record_dict for date_obj, record_dict in raw_trace_obj.items()}
    initial_source_dict = source_hash_dict()
    release_obj = parse_release_manifest(str(REPO_PATH_OBJ / "docs/live/release_templates" / VARIANT_TEMPLATE_DICT[args_obj.variant]))
    release_obj = replace(release_obj, pod_budget_fraction_float=1.0)
    date_list, coverage_dict = select_replay_dates(trace_dict, input_dict.get("vix_close_ser"), args_obj.sessions)
    result_dict = {"generated_at_utc_str": datetime.now(timezone.utc).isoformat(), "variant_str": args_obj.variant,
        "strategy_import_str": release_obj.strategy_import_str, "status_str": "running", "source_hash_dict": initial_source_dict,
        "input_sha256_str": hashlib.sha256(input_path_obj.read_bytes()).hexdigest(),
        "auxiliary_sha256_str": hashlib.sha256(auxiliary_path_obj.read_bytes()).hexdigest(),
        "trace_sha256_str": hashlib.sha256(trace_path_obj.read_bytes()).hexdigest(),
        "coverage_dict": coverage_dict, "row_list": [], "new_strategy_variants_int": 0,
        "limitations_list": ["Current-vintage native reconstruction, not contemporaneous historical snapshots.",
            "Each host call is seeded from historical-engine state; this is open-loop decision parity, not closed-loop broker execution.",
            "Provider access and historical snapshot identity are patched; transport/native snapshot validation requires separate evidence.",
            "DV2-IND compares one-tenth entry weight, ranking and exits; adjusted-ledger holdings do not prove nominal physical-account sizing.",
            "No broker, deployment, order, account-state mutation or parameter search."]}
    report_path_obj = output_path_obj / "replay.json"
    for decision_ts in date_list:
        started_float = time.monotonic()
        try:
            decision_obj = build_cached_host_decision(args_obj.variant, release_obj, decision_ts, trace_dict[decision_ts], input_dict)
            row_dict = compare_decision(decision_obj, trace_dict[decision_ts], args_obj.variant, decision_ts)
            # Positive control: the exact same orders with a wrong target date
            # must fail the harness, or its timing PASS is meaningless.
            wrong_date_obj = replace(decision_obj, target_execution_timestamp_ts=decision_obj.target_execution_timestamp_ts+timedelta(days=1))
            row_dict["wrong_execution_date_positive_control_caught_bool"] = not compare_decision(
                wrong_date_obj, trace_dict[decision_ts], args_obj.variant, decision_ts)["passed_bool"]
            row_dict["passed_bool"] = row_dict["passed_bool"] and row_dict["wrong_execution_date_positive_control_caught_bool"]
            row_dict["gate_open_bool"] = decision_obj.snapshot_metadata_dict.get("gate_open_bool")
            row_dict["signal_timestamp_str"] = decision_obj.signal_timestamp_ts.isoformat()
            row_dict["target_execution_timestamp_str"] = decision_obj.target_execution_timestamp_ts.isoformat()
        except Exception as exception_obj:
            row_dict = {"passed_bool": False, "error_str": f"{type(exception_obj).__name__}: {exception_obj}"}
        row_dict.update({"decision_date_str": decision_ts.date().isoformat(), "elapsed_seconds_float": time.monotonic()-started_float})
        result_dict["row_list"].append(row_dict)
        report_path_obj.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
        print(f"{args_obj.variant} {decision_ts.date()} {'PASS' if row_dict['passed_bool'] else 'FAIL'} {row_dict.get('error_str', '')}", flush=True)
    result_dict["source_unchanged_bool"] = initial_source_dict == source_hash_dict()
    result_dict["session_count_int"] = len(result_dict["row_list"])
    result_dict["passed_session_count_int"] = sum(row_dict["passed_bool"] for row_dict in result_dict["row_list"])
    result_dict["status_str"] = "pass" if result_dict["source_unchanged_bool"] and all(row_dict["passed_bool"] for row_dict in result_dict["row_list"]) else "fail"
    report_path_obj.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
    return 0 if result_dict["status_str"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
