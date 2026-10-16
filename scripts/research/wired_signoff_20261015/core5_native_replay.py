"""Offline native-engine to real-host replay. Never loads Norgate or a broker.

The caller supplies a saved current-vintage price frame. Engine account states
include its dividend/borrow/cost model; these are not historical broker states.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))

import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.live import core5_adapter, scheduler_utils, strategy_host
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, LivePriceSnapshot, LiveRelease, PodState
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


def replay_dict(pricing_df: pd.DataFrame, sessions_int: int, budget_float: float) -> dict:
    config_obj = replace(core5_module.DEFAULT_CONFIG, capital_base_float=budget_float)
    calendar_idx = core5_module.build_execution_calendar_idx(pricing_df, config_obj)
    oracle_obj = core5_module._build_strategy_obj(config_obj, calendar_idx)
    original_iterate_fn = oracle_obj.iterate
    captured_list = []

    def capture_iterate(signal_df, close_row_ser, open_price_ser):
        # *** CRITICAL *** Capture prior holdings/cash before T+1 dividends,
        # fills and borrow. NAV_T = cash_T + sum(q_i,T * Close_i,T).
        position_dict = oracle_obj.get_positions().to_dict()
        cash_float = float(oracle_obj.cash)
        nav_float = cash_float + sum(float(amount_float) * float(close_row_ser[(asset_str, "Close")])
            for asset_str, amount_float in position_dict.items())
        previous_event_str = (oracle_obj.rebalance_target_weight_row_dict_list[-1]["decision_date_ts"].date().isoformat()
            if oracle_obj.rebalance_target_weight_row_dict_list else None)
        original_iterate_fn(signal_df, close_row_ser, open_price_ser)
        target_dict = {asset_str: float(position_dict.get(asset_str, 0)) for asset_str in core5_adapter.CORE5_ASSET_TUPLE}
        leg_dict = defaultdict(list)
        for order_obj in oracle_obj.get_orders():
            amount_float = order_obj.amount_in_shares(float(close_row_ser[(order_obj.asset, "Close")]),
                float(oracle_obj.previous_total_value), target_dict.get(order_obj.asset, 0))
            target_dict[order_obj.asset] += amount_float
            if amount_float:
                leg_dict[order_obj.asset].append(float(amount_float))
        captured_list.append({"decision_date_str": str(oracle_obj.previous_bar.date()),
            "execution_date_str": str(oracle_obj.current_bar.date()), "cash_float": cash_float,
            "nav_float": nav_float, "position_dict": position_dict, "previous_event_str": previous_event_str,
            "event_str": oracle_obj.rebalance_target_weight_row_dict_list[-1]["decision_date_ts"].date().isoformat(),
            "target_dict": target_dict, "leg_dict": dict(leg_dict),
            "weight_dict": oracle_obj.last_target_weight_ser.to_dict()})

    oracle_obj.iterate = capture_iterate
    run_daily(oracle_obj, pricing_df, calendar=calendar_idx, show_progress=False,
        show_signal_progress_bool=False, audit_override_bool=False)
    selected_int_set = set(range(max(0, len(captured_list) - sessions_int), len(captured_list)))
    # Add native event coverage if the tail does not contain a DBC reversal.
    reversal_int_list = [index_int for index_int, row_dict in enumerate(captured_list)
        if len(row_dict["leg_dict"].get("DBC", [])) == 2]
    if reversal_int_list:
        selected_int_set.update(range(max(0, reversal_int_list[-1] - 1),
            min(len(captured_list), reversal_int_list[-1] + 2)))
    release_obj = LiveRelease(release_id_str="signoff.core5.native.v1", user_id_str="offline_audit",
        pod_id_str="core5_native_audit", account_route_str="SIM_NATIVE_ONLY",
        strategy_import_str=core5_adapter.CORE5_STRATEGY_IMPORT_STR, mode_str="incubation",
        session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str="norgate_eod_core5",
        params_dict={}, risk_profile_str="audit_only", enabled_bool=False, source_path_str="in_memory",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False)
    evidence_list = []
    for index_int in sorted(selected_int_set):
        row_dict = captured_list[index_int]
        signal_date_ts = pd.Timestamp(row_dict["decision_date_str"])
        as_of_ts = (signal_date_ts.tz_localize("America/New_York") + pd.Timedelta(hours=18)).to_pydatetime()
        receipt_dict = {} if row_dict["previous_event_str"] is None else {
            "last_applied_rebalance_date_str": row_dict["previous_event_str"],
            "last_applied_target_share_map_dict": {asset_str: row_dict["position_dict"].get(asset_str, 0)
                for asset_str in core5_adapter.CORE5_ASSET_TUPLE}}
        state_obj = PodState(release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
            row_dict["position_dict"], row_dict["cash_float"], row_dict["nav_float"],
            {"core5_execution_receipt_dict": receipt_dict}, as_of_ts,
            snapshot_stage_str="eod", snapshot_source_str="virtual_broker")
        metadata_dict = {"norgate_snapshot_date_str": row_dict["decision_date_str"],
            "norgate_data_profile_str": "norgate_eod_core5", "norgate_manifest_hash_str": "saved_vintage_reconstruction"}
        with patch.object(core5_adapter, "_load_core5_decision_prices", return_value=(pricing_df, metadata_dict)):
            decision_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, state_obj)
        assert decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == row_dict["target_dict"], row_dict["decision_date_str"]
        assert decision_obj.strategy_state_dict["last_rebalance_date_str"] == row_dict["event_str"]
        assert decision_obj.strategy_state_dict["last_target_weight_map_dict"] == row_dict["weight_dict"]
        assert decision_obj.signal_timestamp_ts == scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS")
        assert decision_obj.target_execution_timestamp_ts == scheduler_utils.get_session_open_timestamp_ts(pd.Timestamp(row_dict["execution_date_str"]), "XNYS")
        # Move NAV and quotes independently: a correlated change can cancel in
        # an incorrect weight*NAV/quote implementation and give false assurance.
        for nav_multiplier_float, quote_multiplier_float in ((1., .8), (1., 1.2), (.8, 1.), (1.2, 1.)):
            broker_obj = BrokerSnapshot(release_obj.account_route_str, decision_obj.submission_timestamp_ts,
                row_dict["cash_float"], row_dict["nav_float"] * nav_multiplier_float, row_dict["position_dict"],
                net_liq_float=row_dict["nav_float"] * nav_multiplier_float)
            quote_obj = LivePriceSnapshot(release_obj.account_route_str, decision_obj.submission_timestamp_ts,
                "offline_perturbed_quote", {asset_str: float(pricing_df.loc[signal_date_ts, (asset_str, "Close")]) * quote_multiplier_float
                    for asset_str in core5_adapter.CORE5_ASSET_TUPLE})
            plan_obj = build_vplan(release_obj, decision_obj, broker_obj, quote_obj)
            actual_leg_dict = defaultdict(list)
            for request_obj in build_broker_order_request_list_from_vplan(plan_obj):
                assert request_obj.unit_str == "shares" and not request_obj.target_bool
                assert request_obj.broker_order_type_str == "MOO"
                actual_leg_dict[request_obj.asset_str].append(request_obj.amount_float)
            assert dict(actual_leg_dict) == row_dict["leg_dict"], row_dict["decision_date_str"]
        evidence_list.append({"decision_date_str": row_dict["decision_date_str"],
            "execution_date_str": row_dict["execution_date_str"], "target_share_dict": row_dict["target_dict"],
            "order_leg_dict": row_dict["leg_dict"], "event_date_str": row_dict["event_str"],
            "month_end_bool": decision_obj.snapshot_metadata_dict["month_end_bool"],
            "rebalance_bool": decision_obj.snapshot_metadata_dict["rebalance_bool"]})
        print(f"HOST_REPLAY {row_dict['decision_date_str']} legs={sum(map(len, row_dict['leg_dict'].values()))}", flush=True)
    assert len(evidence_list) >= 20
    assert any(row_dict["month_end_bool"] for row_dict in evidence_list)
    assert any(row_dict["order_leg_dict"] for row_dict in evidence_list)
    return {"status_str": "passed", "capital_base_float": budget_float,
        "canonical_engine_session_count_int": len(captured_list), "native_host_session_count_int": len(evidence_list),
        "native_host_rebalance_count_int": sum(row_dict["rebalance_bool"] for row_dict in evidence_list),
        "native_host_month_end_count_int": sum(row_dict["month_end_bool"] for row_dict in evidence_list),
        "native_host_dbc_reversal_count_int": sum(len(row_dict["order_leg_dict"].get("DBC", [])) == 2 for row_dict in evidence_list),
        "independent_nav_quote_invariance_case_count_int": len(evidence_list) * 4,
        "oracle_borrow_fee_float": oracle_obj.borrow_fee_total_float, "evidence_list": evidence_list,
        "limitations_list": ["Current-vintage saved prices; not saved historical decision-time snapshots.",
            "Host loader alone is substituted; actual strategy_host dispatcher, signal replay, sizing, VPlan and broker-request builder run.",
            "Per-session account state and applied receipt are seeded from the canonical native engine; no independent broker lifecycle proof.",
            "Research costs/dividends/borrow fund the seeded accounts; live cash is observed instead."]}


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--prices", type=Path, required=True)
    parser_obj.add_argument("--sessions", type=int, default=42)
    parser_obj.add_argument("--budget", type=float, default=1_000_000.0)
    parser_obj.add_argument("--output", type=Path, required=True)
    args_obj = parser_obj.parse_args()
    args_obj.output.parent.mkdir(parents=True, exist_ok=True)
    pricing_df = pd.read_parquet(args_obj.prices) if args_obj.prices.suffix == ".parquet" else pd.read_pickle(args_obj.prices)
    source_list = ["alpha/live/core5_adapter.py", "alpha/live/strategy_host.py", "alpha/live/execution_engine.py",
        "strategies/taa_beyond_6040/strategy_taa_adaptive_macro_core5.py", "alpha/engine/strategy.py",
        "alpha/engine/order.py", "alpha/engine/backtest.py", "alpha/engine/backtester.py",
        "alpha/live/scheduler_utils.py", "alpha/live/models.py", "data/norgate_snapshot_store.py",
        "scripts/research/wired_signoff_20261015/core5_native_replay.py"]
    report_dict = {"generated_at_utc_str": datetime.now(UTC).isoformat(), "broker_verified_bool": False,
        "forward_test_bool": False, "new_strategy_variants_int": 0,
        "prices_path_str": str(args_obj.prices.resolve()), "prices_sha256_str": hashlib.sha256(args_obj.prices.read_bytes()).hexdigest(),
        "input_first_date_str": str(pricing_df.index[0].date()), "input_last_date_str": str(pricing_df.index[-1].date()),
        "source_sha256_dict": {source_str: hashlib.sha256((REPO_PATH / source_str).read_bytes()).hexdigest() for source_str in source_list}}
    try:
        report_dict.update(replay_dict(pricing_df, args_obj.sessions, args_obj.budget))
        assert hashlib.sha256(args_obj.prices.read_bytes()).hexdigest() == report_dict["prices_sha256_str"]
        for source_str, source_hash_str in report_dict["source_sha256_dict"].items():
            assert hashlib.sha256((REPO_PATH / source_str).read_bytes()).hexdigest() == source_hash_str, source_str
    except Exception as error_obj:
        report_dict.update(status_str="failed", error_str=f"{type(error_obj).__name__}: {error_obj}")
        raise
    finally:
        args_obj.output.write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
