"""Broker-free counterexample: parent VPlan proceeds after position mismatch.

Uses a synthetic DecisionPlan to isolate the actual runner pre-VPlan guard.
No host/signal/native-data conclusion follows from this execution-layer probe.
"""
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH_OBJ))

from alpha.live import runner, scheduler_utils
from alpha.live.models import DecisionPlan
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import parse_release_manifest
from alpha.live.state_store_v2 import LiveStateStore


def main():
    output_path_obj = REPO_PATH_OBJ / "results/review/wired_signoff_20261015/mr_outer_guard_probe_v3"
    output_path_obj.mkdir(parents=True, exist_ok=False)
    market_timezone_obj = ZoneInfo("America/New_York")
    signal_ts = datetime(2024, 1, 2, 16, tzinfo=market_timezone_obj)
    target_ts = datetime(2024, 1, 3, 9, 30, tzinfo=market_timezone_obj)
    as_of_ts = datetime(2024, 1, 3, 9, 24, tzinfo=market_timezone_obj)
    result_list = []
    for variant_str, template_str in (("dv2", "pod_dv2_daily_moo.yaml.example"),
        ("hpi235", "pod_hpi_sp500_2_3_5_vote_daily_moo.yaml.example")):
        release_obj = parse_release_manifest(str(REPO_PATH_OBJ / "docs/live/release_templates" / template_str))
        release_obj = replace(release_obj, enabled_bool=True, auto_submit_enabled_bool=False,
            account_route_str="DU1234567", pod_budget_fraction_float=1.0)
        state_store_obj = LiveStateStore(str(output_path_obj / f"{variant_str}.sqlite3"))
        state_store_obj.upsert_release(release_obj)
        decision_obj = DecisionPlan(release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str,
            release_obj.account_route_str, signal_ts, scheduler_utils.build_submission_timestamp_ts(signal_ts, release_obj),
            target_ts, "next_open_moo", {}, {"norgate_data_profile_str": release_obj.data_profile_str,
            "norgate_snapshot_date_str": "2024-01-02", "norgate_manifest_hash_str": "isolated_probe"}, {},
            entry_target_weight_map_dict={"NEW": 0.1}, target_weight_map={"NEW": 0.1}, entry_priority_list=["NEW"])
        state_store_obj.insert_decision_plan(decision_obj)
        broker_obj = StubBrokerAdapter()
        fresh_position_dict = {f"HELD{symbol_int}": 100.0 for symbol_int in range(10)}
        broker_obj.seed_account_snapshot(release_obj.account_route_str, cash_float=900000.0, total_value_float=1000000.0,
            net_liq_float=1000000.0, position_amount_map=fresh_position_dict, snapshot_timestamp_ts=as_of_ts,
            session_mode_str="paper")
        broker_obj.seed_live_price_snapshot(release_obj.account_route_str, {"NEW": 100.0}, snapshot_timestamp_ts=as_of_ts)
        detail_dict = runner.build_vplans(state_store_obj, broker_obj, as_of_ts, "paper",
            log_path_str=str(output_path_obj / f"{variant_str}.jsonl"), trace_enabled_bool=False)
        vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
        result_list.append({"variant_str": variant_str, "runner_result_dict": detail_dict,
            "fresh_holding_count_int": len(fresh_position_dict), "decision_base_holding_count_int": 0,
            "created_status_str": None if vplan_obj is None else vplan_obj.status_str,
            "additional_entry_share_float": None if vplan_obj is None else vplan_obj.target_share_map.get("NEW"),
            "outcome_str": "WARNING_ONLY_VPLAN_CREATED" if detail_dict["created_vplan_count_int"] else "BLOCKED"})
    report_dict = {"scope_str": "Actual runner.build_vplans, synthetic intent/account, StubBrokerAdapter only; stops before submit.",
        "production_changed_bool": False, "broker_called_bool": False,
        "result_list": result_list,
        "conclusion_str": "Parent pre-VPlan reconciliation is warning-only: a fresh ten-name account can receive an eleventh entry from a decision computed as flat. Execution impact depends on stale/out-of-band holdings; this is not evidence that a normal account has drifted."}
    (output_path_obj / "probe.json").write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report_dict, indent=2, default=str))


if __name__ == "__main__":
    main()
