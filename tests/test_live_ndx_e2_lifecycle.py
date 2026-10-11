"""Actual E2 host, scheduler, SQL and monthly runner with a synthetic broker."""
from dataclasses import replace
from datetime import timedelta

import pytest
import yaml

from alpha.live import logging_utils, runner, scheduler_service, scheduler_utils
from alpha.live.daily_reconcile import is_daily_reconcile_release_bool
from alpha.live.dispatch_state import is_guarded_release_bool
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.ndx_e2_adapter import NDX_E2_STRATEGY_IMPORT_TUPLE
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import parse_release_manifest
from alpha.live.state_store_v2 import LiveStateStore
from test_live_ndx_e2_adapter import AS_OF_TS, SIGNAL_DATE_TS, e2_case  # noqa: F401


@pytest.fixture
def lifecycle_case(e2_case, tmp_path, monkeypatch):
    source_release_obj, source_state_obj, *_ = e2_case
    component_int = NDX_E2_STRATEGY_IMPORT_TUPLE.index(source_release_obj.strategy_import_str)
    account_route_str = f"DU1200{component_int + 1}"
    releases_path_obj = tmp_path / "releases" / "owner"
    releases_path_obj.mkdir(parents=True)
    manifest_path_obj = releases_path_obj / "e2.yaml"
    manifest_dict = {
        "identity": {"release_id": "e2.v1", "user_id": "owner", "pod_id": "e2"},
        "deployment": {"mode": "paper", "enabled_bool": True},
        "broker": {"account_route": account_route_str},
        "strategy": {"strategy_import_str": source_release_obj.strategy_import_str,
            "data_profile_str": source_release_obj.data_profile_str, "params": {}},
        "market": {"session_calendar_id_str": "XNYS"},
        "schedule": {"signal_clock_str": "month_end_snapshot_ready", "execution_policy_str": "next_month_first_open"},
        "execution": {"pod_budget_fraction_float": 1.0, "auto_submit_enabled_bool": True},
        "bootstrap": {"initial_cash_float": 100000.0},
        "risk": {"risk_profile_str": "monthly"},
    }
    manifest_path_obj.write_text(yaml.safe_dump(manifest_dict), encoding="utf-8")
    release_obj = parse_release_manifest(str(manifest_path_obj))
    store_obj = LiveStateStore(str(tmp_path / "e2.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(replace(source_state_obj, account_route_str=account_route_str,
        position_amount_map={"OLD": 10.0}, cash_float=99000.0,
        snapshot_stage_str="eod", snapshot_source_str="broker"))
    monkeypatch.setattr(scheduler_utils, "load_latest_norgate_heartbeat_session_label_ts", lambda _: SIGNAL_DATE_TS)
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(tmp_path / "critical.jsonl"))
    monkeypatch.setattr(IBKRSocketClient, "connect", lambda *_: pytest.fail("Real broker forbidden"))
    monkeypatch.setattr(IBKRSocketClient, "daily_connection", lambda *_: pytest.fail("Real broker forbidden"))
    monkeypatch.setattr(runner, "_reconcile_daily_cycles", lambda *args, **kwargs: pytest.fail("E2 entered daily reconciliation"))
    broker_obj = StubBrokerAdapter()
    runner_kwarg_dict = {"releases_root_path_str": str(tmp_path / "releases"), "env_mode_str": "paper",
        "log_path_str": str(tmp_path / "events.jsonl"), "trace_enabled_bool": False}
    return store_obj, release_obj, broker_obj, runner_kwarg_dict


def _next_phase_str(store_obj, as_of_ts, runner_kwarg_dict):
    return scheduler_service.get_scheduler_decision(store_obj, as_of_ts,
        runner_kwarg_dict["releases_root_path_str"], "paper").next_phase_str


def test_e2_monthly_auto_execution_and_omitted_exit_survive_restarts(lifecycle_case):
    store_obj, release_obj, broker_obj, runner_kwarg_dict = lifecycle_case
    assert not is_daily_reconcile_release_bool(release_obj)
    assert not is_guarded_release_bool(release_obj)
    assert _next_phase_str(store_obj, AS_OF_TS, runner_kwarg_dict) == "build_decision_plan"
    result_dict = runner.build_decision_plans(store_obj, AS_OF_TS,
        auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)
    assert result_dict["created_decision_plan_count_int"] == 1
    decision_obj = store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    weight_dict = dict(decision_obj.full_target_weight_map_dict)
    assert weight_dict and set(weight_dict.values()) == {.05}
    assert decision_obj.decision_base_position_map == {"OLD": 10.0}
    assert decision_obj.target_share_map_dict == {}
    assert "OLD" not in weight_dict and decision_obj.rebalance_omitted_assets_to_zero_bool
    assert runner.build_decision_plans(store_obj, AS_OF_TS,
        auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)["created_decision_plan_count_int"] == 0

    store_obj = LiveStateStore(store_obj.db_path_str)
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int) == decision_obj
    submit_ts = decision_obj.submission_timestamp_ts
    broker_obj.seed_account_snapshot(release_obj.account_route_str, cash_float=119000.0,
        total_value_float=120000.0, position_amount_map={"OLD": 10.0},
        snapshot_timestamp_ts=submit_ts, session_mode_str="paper")
    broker_obj.seed_live_price_snapshot(release_obj.account_route_str,
        {**{asset_str: 125.0 for asset_str in weight_dict}, "OLD": 100.0}, snapshot_timestamp_ts=submit_ts)
    assert _next_phase_str(store_obj, submit_ts, runner_kwarg_dict) == "build_vplan"
    assert runner.build_vplans(store_obj, broker_obj, submit_ts, **runner_kwarg_dict)["created_vplan_count_int"] == 1
    plan_obj = store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    # Close_T weights stay 5%; floor(120000 * .05 / 125) = 48 shares at pre-open.
    assert {asset_str: plan_obj.target_share_map[asset_str] for asset_str in weight_dict} == {
        asset_str: 48.0 for asset_str in weight_dict}
    assert plan_obj.target_share_map["OLD"] == 0.0 and plan_obj.order_delta_map["OLD"] == -10.0
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).full_target_weight_map_dict == weight_dict
    request_list = build_broker_order_request_list_from_vplan(plan_obj)
    assert all(request_obj.account_route_str == release_obj.account_route_str
        and request_obj.broker_order_type_str == "MOO" and request_obj.submission_deadline_timestamp_str is None
        for request_obj in request_list)
    assert [request_obj.amount_float for request_obj in request_list if request_obj.asset_str == "OLD"] == [-10.0]

    store_obj = LiveStateStore(store_obj.db_path_str)
    assert build_broker_order_request_list_from_vplan(store_obj.get_vplan_by_id(plan_obj.vplan_id_int)) == request_list
    assert runner.build_vplans(store_obj, broker_obj, submit_ts, **runner_kwarg_dict)["created_vplan_count_int"] == 0
    assert _next_phase_str(store_obj, submit_ts, runner_kwarg_dict) == "submit_vplan"
    assert runner.submit_ready_vplans(store_obj, broker_obj, submit_ts, manual_only_bool=False,
        **runner_kwarg_dict)["submitted_vplan_count_int"] == 1
    assert broker_obj.submitted_order_request_list == request_list

    store_obj = LiveStateStore(store_obj.db_path_str)
    assert runner.submit_ready_vplans(store_obj, broker_obj, submit_ts, manual_only_bool=False,
        **runner_kwarg_dict)["submitted_vplan_count_int"] == 0
    reconcile_ts = decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    assert _next_phase_str(store_obj, reconcile_ts, runner_kwarg_dict) == "post_execution_reconcile"
    assert runner.post_execution_reconcile(store_obj, broker_obj, reconcile_ts,
        **runner_kwarg_dict)["completed_vplan_count_int"] == 1
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "completed"
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    assert state_obj.position_amount_map.get("OLD", 0.0) == 0.0
    assert {asset_str: state_obj.position_amount_map[asset_str] for asset_str in weight_dict} == {
        asset_str: 48.0 for asset_str in weight_dict}
    assert state_obj.cash_float == 120000.0 - len(weight_dict) * 6000.0
    assert state_obj.snapshot_stage_str == "post_execution"
    assert len(store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int)) == len(request_list)
    store_obj = LiveStateStore(store_obj.db_path_str)
    assert runner.post_execution_reconcile(store_obj, broker_obj, reconcile_ts,
        **runner_kwarg_dict)["completed_vplan_count_int"] == 0
    assert broker_obj.submitted_order_request_list == request_list
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM decision_plan").fetchone()[0] == 1
        assert connection_obj.execute("SELECT COUNT(*) FROM vplan").fetchone()[0] == 1
        assert connection_obj.execute("SELECT COUNT(*) FROM daily_completion_request").fetchone()[0] == 0


def test_e2_invalid_month_end_input_cannot_persist_a_decision_or_orders(lifecycle_case, e2_case):
    store_obj, release_obj, broker_obj, runner_kwarg_dict = lifecycle_case
    before_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    before_history_list = store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)
    e2_case[4]["norgate_snapshot_date_str"] = "2013-06-27"
    with pytest.raises(ValueError, match="exact completed month-end"):
        runner.build_decision_plans(store_obj, AS_OF_TS,
            auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)
    store_obj = LiveStateStore(store_obj.db_path_str)
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str) is None
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str) is None
    assert store_obj.get_pod_state(release_obj.pod_id_str) == before_state_obj
    assert store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str) == before_history_list
    assert runner.submit_ready_vplans(store_obj, broker_obj, AS_OF_TS, manual_only_bool=False,
        **runner_kwarg_dict)["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list
