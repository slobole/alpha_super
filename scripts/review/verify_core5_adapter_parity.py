"""Compare local CORE5 decisions and order legs to the historical engine.

Reads a saved price Parquet. No broker, network, release or scheduler actions.
This is current-vintage reconstruction, not saved decision-time replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from alpha.live.core5_adapter import CORE5_ASSET_TUPLE, CORE5_STRATEGY_IMPORT_STR, build_core5_decision_from_prices
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, LivePriceSnapshot, LiveRelease, PodState
from alpha.live import scheduler_utils
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


def compare_adapter_to_engine(pricing_data_df: pd.DataFrame, start_date_str: str) -> dict:
    release_obj = LiveRelease(
        release_id_str="core5.local_oracle.v1", user_id_str="local_test", pod_id_str="core5_oracle",
        account_route_str="SIM_CORE5_ORACLE", strategy_import_str=CORE5_STRATEGY_IMPORT_STR,
        mode_str="incubation", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str="norgate_eod_core5", params_dict={},
        risk_profile_str="local_qualification", enabled_bool=False, source_path_str="in_memory_only",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
    )
    calendar_idx = core5_module.build_execution_calendar_idx(pricing_data_df, backtest_start_date_str=start_date_str)
    oracle_obj = core5_module.AdaptiveMacroCore5Strategy()
    original_iterate_fn = oracle_obj.iterate
    original_process_fn = oracle_obj.process_orders
    committed_state_dict = {}
    pending_decision_obj = None
    evidence_row_list = []

    def compare_iterate(signal_df, close_row_ser, open_price_ser):
        nonlocal pending_decision_obj
        # *** CRITICAL*** Entry precedes T+1 dividends, opening fills and borrow.
        # The oracle's cash and positions describe Close_T. Adapter computes its
        # own features from the full raw prefix ending at T, never T+1.
        decision_date_ts = oracle_obj.previous_bar
        as_of_ts = (pd.Timestamp(decision_date_ts).tz_localize("America/New_York") + pd.Timedelta(hours=18)).to_pydatetime()
        position_map_dict = oracle_obj.get_positions().to_dict()
        nav_float = float(oracle_obj.previous_total_value)
        state_obj = PodState(release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
            position_map_dict, float(oracle_obj.cash), nav_float, dict(committed_state_dict), as_of_ts,
            snapshot_stage_str="eod", snapshot_source_str="virtual_broker")
        decision_obj = build_core5_decision_from_prices(release_obj, as_of_ts, state_obj, pricing_data_df, {
            "norgate_snapshot_date_str": str(decision_date_ts.date()), "norgate_data_profile_str": "norgate_eod_core5",
            "norgate_manifest_hash_str": "local_prefix_reconstruction_not_historical_manifest",
        })
        assert decision_obj.signal_timestamp_ts == scheduler_utils.get_session_close_timestamp_ts(decision_date_ts, "XNYS")
        assert decision_obj.target_execution_timestamp_ts == scheduler_utils.get_session_open_timestamp_ts(oracle_obj.current_bar, "XNYS")
        assert decision_obj.submission_timestamp_ts < decision_obj.target_execution_timestamp_ts
        assert abs(decision_obj.snapshot_metadata_dict["sizing_close_nav_float"] - nav_float) < 1e-7
        prior_rebalance_int = len(oracle_obj.rebalance_target_weight_row_dict_list)
        original_iterate_fn(signal_df, close_row_ser, open_price_ser)
        assert decision_obj.snapshot_metadata_dict["rebalance_bool"] == (len(oracle_obj.rebalance_target_weight_row_dict_list) > prior_rebalance_int)
        assert decision_obj.strategy_state_dict["last_target_weight_map_dict"] == oracle_obj.last_target_weight_ser.to_dict()
        expected_position_dict = dict(position_map_dict)
        expected_leg_dict = defaultdict(list)
        for order_obj in oracle_obj.get_orders():
            assert isinstance(order_obj, MarketOrder)
            amount_float = order_obj.amount_in_shares(float(close_row_ser[(order_obj.asset, "Close")]), nav_float, expected_position_dict.get(order_obj.asset, 0.0))
            expected_position_dict[order_obj.asset] = expected_position_dict.get(order_obj.asset, 0.0) + amount_float
            if amount_float:
                expected_leg_dict[order_obj.asset].append(float(amount_float))
        expected_target_dict = {asset_str: float(expected_position_dict.get(asset_str, 0.0)) for asset_str in CORE5_ASSET_TUPLE}
        assert decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == expected_target_dict
        assert decision_obj.snapshot_metadata_dict["no_order_bool"] == (not expected_leg_dict)
        for quote_multiplier_float in (0.8, 1.2):
            account_obj = BrokerSnapshot(release_obj.account_route_str, decision_obj.submission_timestamp_ts,
                cash_float=oracle_obj.cash, total_value_float=nav_float * quote_multiplier_float, position_amount_map=position_map_dict)
            quote_obj = LivePriceSnapshot(release_obj.account_route_str, decision_obj.submission_timestamp_ts, "perturbed_test_quote",
                {asset_str: float(close_row_ser[(asset_str, "Close")]) * quote_multiplier_float for asset_str in CORE5_ASSET_TUPLE})
            vplan_obj = build_vplan(release_obj, decision_obj, account_obj, quote_obj)
            actual_leg_dict = defaultdict(list)
            for request_obj in build_broker_order_request_list_from_vplan(vplan_obj):
                assert request_obj.unit_str == "shares" and request_obj.broker_order_type_str == "MOO"
                assert request_obj.target_bool is False and request_obj.order_class_str == "MarketOrder"
                actual_leg_dict[request_obj.asset_str].append(request_obj.amount_float)
            assert actual_leg_dict == expected_leg_dict
        evidence_row_list.append({
            "decision_date_str": str(decision_date_ts.date()), "execution_date_str": str(oracle_obj.current_bar.date()),
            "signal_timestamp_str": decision_obj.signal_timestamp_ts.isoformat(),
            "submission_timestamp_str": decision_obj.submission_timestamp_ts.isoformat(),
            "target_execution_timestamp_str": decision_obj.target_execution_timestamp_ts.isoformat(),
            "rebalance_bool": decision_obj.snapshot_metadata_dict["rebalance_bool"],
            "month_end_bool": decision_obj.snapshot_metadata_dict["month_end_bool"],
            "no_order_bool": not bool(expected_leg_dict), "dbc_two_legs_bool": len(expected_leg_dict.get("DBC", [])) == 2,
            "target_share_map_dict": expected_target_dict, "order_leg_map_dict": dict(expected_leg_dict),
        })
        pending_decision_obj = decision_obj

    def compare_process(prices_df):
        nonlocal committed_state_dict
        original_process_fn(prices_df)
        expected_target_dict = pending_decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
        actual_position_dict = oracle_obj.get_positions().to_dict()
        assert {asset_str: float(actual_position_dict.get(asset_str, 0.0)) for asset_str in CORE5_ASSET_TUPLE} == expected_target_dict
        committed_state_dict = dict(pending_decision_obj.strategy_state_dict)

    oracle_obj.iterate = compare_iterate
    oracle_obj.process_orders = compare_process
    run_daily(oracle_obj, pricing_data_df, calendar=calendar_idx, show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    return {
        "decision_count_int": len(evidence_row_list),
        "rebalance_count_int": sum(row_dict["rebalance_bool"] for row_dict in evidence_row_list),
        "no_order_count_int": sum(row_dict["no_order_bool"] for row_dict in evidence_row_list),
        "month_end_count_int": sum(row_dict["month_end_bool"] for row_dict in evidence_row_list),
        "dbc_two_leg_count_int": sum(row_dict["dbc_two_legs_bool"] for row_dict in evidence_row_list),
        "oracle_borrow_fee_float": float(oracle_obj.borrow_fee_total_float),
        "decision_row_list": evidence_row_list,
    }


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--prices", type=Path, required=True)
    parser_obj.add_argument("--start-date", required=True)
    parser_obj.add_argument("--output-dir", type=Path, required=True)
    args_obj = parser_obj.parse_args()
    args_obj.output_dir.mkdir(parents=True, exist_ok=False)
    report_dict = {"status_str": "running", "generated_at_utc_str": datetime.now(UTC).isoformat(),
        "source_str": "saved_current_vintage_prices_not_decision_time_replay",
        "new_strategy_variants_int": 0, "broker_verified_bool": False, "forward_test_bool": False,
        "start_date_str": args_obj.start_date,
        "prices_sha256_str": hashlib.sha256(args_obj.prices.read_bytes()).hexdigest(),
        "limitations_list": ["Account cash seeded from the historical engine, including its research borrow/dividend/fee model.",
            "Proves decisions, shares and ordered legs per asset; does not qualify real account cash, fills, borrow or margin."]}
    source_path_list = ["alpha/live/core5_adapter.py", "alpha/live/execution_engine.py", "alpha/live/scheduler_utils.py", "alpha/live/models.py",
        "alpha/engine/strategy.py", "alpha/engine/order.py",
        "alpha/engine/backtester.py", "alpha/engine/backtest.py", "alpha/engine/execution_timing.py",
        "strategies/taa_beyond_6040/strategy_taa_adaptive_macro_core5.py", "scripts/review/verify_core5_adapter_parity.py"]
    report_dict["source_sha256_dict"] = {source_str: hashlib.sha256((REPO_ROOT_PATH / source_str).read_bytes()).hexdigest() for source_str in source_path_list}
    try:
        report_dict.update(compare_adapter_to_engine(pd.read_parquet(args_obj.prices), args_obj.start_date))
        for source_str, source_hash_str in report_dict["source_sha256_dict"].items():
            assert hashlib.sha256((REPO_ROOT_PATH / source_str).read_bytes()).hexdigest() == source_hash_str, source_str
        assert hashlib.sha256(args_obj.prices.read_bytes()).hexdigest() == report_dict["prices_sha256_str"]
        report_dict["status_str"] = "passed"
        print(json.dumps({key_str: value_obj for key_str, value_obj in report_dict.items() if key_str.endswith("_int")}))
    except Exception as error_obj:
        report_dict.update(status_str="failed", error_str=f"{type(error_obj).__name__}: {error_obj}")
        raise
    finally:
        (args_obj.output_dir / "qualification.json").write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
