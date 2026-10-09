"""Daily rows must not change production monthly reconciliation or payloads."""
import ast
from dataclasses import replace
from datetime import timedelta
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from alpha.live import runner
from alpha.live.models import DecisionPlan
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.state_store_v2 import LiveStateStore
from test_live_daily_reconcile import CLOSE_TS, OPEN_TS, daily_case
from test_live_daily_scheduler_fairness import MONTHLY_STRATEGY_LIST
from test_live_mr_capsule_target_shares import _capsule_inputs
from test_live_reference_compare import _build_release, _insert_vplan


@pytest.fixture(scope="module")
def baseline_runner_dict():
    root_path_obj = Path(__file__).resolve().parents[1]
    source_str = subprocess.check_output(["git", "-c", f"safe.directory={root_path_obj.as_posix()}",
        "-c", "core.fsmonitor=false", "show", "54b417f:alpha/live/runner.py"],
        cwd=root_path_obj, text=True, encoding="utf-8")
    name_set = {"post_execution_reconcile", "_pod_has_unresolved_execution_bool",
        "is_vplan_execution_exception_parked", "_decision_plan_trace_payload_dict", "show_decision_plan_summary"}
    namespace_dict = dict(vars(runner))
    for function_node_obj in ast.parse(source_str).body:
        if isinstance(function_node_obj, ast.FunctionDef) and function_node_obj.name in name_set:
            function_source_str = ast.get_source_segment(source_str, function_node_obj)
            exec(compile(function_source_str, "54b417f:alpha/live/runner.py", "exec"), namespace_dict)
    namespace_dict["source_str"] = source_str
    return namespace_dict


def _monthly_case(tmp_path, strategy_str):
    store_obj = LiveStateStore(str(tmp_path / "monthly.sqlite3"))
    release_obj = replace(_build_release(strategy_str, pod_id_str="monthly"),
        signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open")
    store_obj.upsert_release(release_obj)
    plan_obj = _insert_vplan(store_obj, release_obj, OPEN_TS - timedelta(days=3), OPEN_TS, 5000.0)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(plan_obj.decision_plan_id_int, "submitted")
    broker_obj = StubBrokerAdapter()
    broker_obj.seed_account_snapshot(release_obj.account_route_str, cash_float=4000.0,
        total_value_float=5000.0, position_amount_map={"AAPL": 10.0}, snapshot_timestamp_ts=OPEN_TS)
    broker_obj.seed_live_price_snapshot(release_obj.account_route_str, {"AAPL": 100.0}, OPEN_TS)
    return store_obj, release_obj, plan_obj, broker_obj


def _poison_daily_rows(store_obj, corruption_str):
    release_obj, decision_obj, _, _ = _capsule_inputs()
    release_obj = replace(release_obj, enabled_bool=False)
    store_obj.upsert_release(release_obj)
    plan_obj = _insert_vplan(store_obj, release_obj, OPEN_TS - timedelta(days=3), OPEN_TS, 5000.0)
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
    with store_obj._connect() as connection_obj:
        if corruption_str == "decision":
            connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str='invalid JSON' "
                "WHERE decision_plan_id_int=?", (plan_obj.decision_plan_id_int,))
        elif corruption_str == "vplan":
            connection_obj.execute("UPDATE vplan SET target_share_json_str='invalid JSON' WHERE vplan_id_int=?",
                (plan_obj.vplan_id_int,))
        else:
            connection_obj.execute("UPDATE live_release SET params_json_str='invalid JSON' WHERE release_id_str=?",
                (release_obj.release_id_str,))
    return release_obj, plan_obj


@pytest.mark.parametrize("strategy_str", MONTHLY_STRATEGY_LIST)
@pytest.mark.parametrize("corruption_str", ["decision", "vplan", "release"])
def test_monthly_reconcile_never_hydrates_daily_rows(tmp_path, monkeypatch, baseline_runner_dict,
        strategy_str, corruption_str):
    baseline_store_obj, baseline_release_obj, baseline_plan_obj, baseline_broker_obj = _monthly_case(
        tmp_path / "baseline", strategy_str)
    baseline_result_dict = baseline_runner_dict["post_execution_reconcile"](baseline_store_obj,
        baseline_broker_obj, CLOSE_TS, "paper", pod_id_str=baseline_release_obj.pod_id_str,
        trace_enabled_bool=False, log_path_str=str(tmp_path / "baseline.log"))
    store_obj, release_obj, plan_obj, broker_obj = _monthly_case(tmp_path / "candidate", strategy_str)
    daily_release_obj, daily_plan_obj = _poison_daily_rows(store_obj, corruption_str)
    original_decision_fn = store_obj.get_decision_plan_by_id
    original_release_fn = store_obj.get_release_by_id
    original_vplan_fn = store_obj._row_to_vplan

    def decision_fn(decision_id_int):
        assert decision_id_int != daily_plan_obj.decision_plan_id_int
        return original_decision_fn(decision_id_int)

    def release_fn(release_id_str):
        assert release_id_str != daily_release_obj.release_id_str
        return original_release_fn(release_id_str)

    def vplan_fn(row_obj):
        assert row_obj["release_id_str"] != daily_release_obj.release_id_str
        return original_vplan_fn(row_obj)

    def unexpected_daily_fn(*_argument_tuple, **_argument_dict):
        raise AssertionError("Monthly reconcile entered the daily pipeline")

    monkeypatch.setattr(store_obj, "get_decision_plan_by_id", decision_fn)
    monkeypatch.setattr(store_obj, "get_release_by_id", release_fn)
    monkeypatch.setattr(store_obj, "_row_to_vplan", vplan_fn)
    monkeypatch.setattr(runner, "_reconcile_daily_cycles", unexpected_daily_fn)
    result_dict = runner.post_execution_reconcile(store_obj, broker_obj, CLOSE_TS, "paper",
        pod_id_str=release_obj.pod_id_str, trace_enabled_bool=False, log_path_str=str(tmp_path / "candidate.log"))
    assert result_dict == baseline_result_dict == {"completed_vplan_count_int": 1}
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == baseline_store_obj.get_vplan_by_id(
        baseline_plan_obj.vplan_id_int).status_str == "completed"
    assert store_obj.get_pod_state(release_obj.pod_id_str) == baseline_store_obj.get_pod_state(baseline_release_obj.pod_id_str)


@pytest.mark.parametrize("strategy_str", MONTHLY_STRATEGY_LIST)
@pytest.mark.parametrize("corruption_str", ["decision", "vplan", "release"])
def test_bad_daily_cycle_cannot_abort_monthly_work_in_a_mixed_pass(tmp_path, strategy_str, corruption_str):
    store_obj, release_obj, plan_obj, broker_obj = _monthly_case(tmp_path, strategy_str)
    _poison_daily_rows(store_obj, corruption_str)
    log_path_obj = tmp_path / "mixed.log"
    result_dict = runner.post_execution_reconcile(store_obj, broker_obj, CLOSE_TS, "paper",
        trace_enabled_bool=False, log_path_str=str(log_path_obj))
    assert result_dict == {"completed_vplan_count_int": 1}
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "completed"
    assert "daily_cycle_" in log_path_obj.read_text(encoding="utf-8")


def test_scoped_daily_query_excludes_other_pods_modes_and_monthly_metadata(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    for mode_str, pod_str, strategy_str in [("paper", "other_daily", release_obj.strategy_import_str),
            ("live", "other_live", release_obj.strategy_import_str),
            ("paper", "monthly", MONTHLY_STRATEGY_LIST[0])]:
        other_release_obj = replace(release_obj, release_id_str=pod_str, pod_id_str=pod_str,
            account_route_str=f"DU_{pod_str}", mode_str=mode_str, strategy_import_str=strategy_str)
        store_obj.upsert_release(other_release_obj)
        other_decision_obj = store_obj.insert_decision_plan(DecisionPlan(other_release_obj.release_id_str,
            other_release_obj.user_id_str, pod_str, other_release_obj.account_route_str,
            decision_obj.signal_timestamp_ts, decision_obj.submission_timestamp_ts,
            decision_obj.target_execution_timestamp_ts, decision_obj.execution_policy_str, {}, {}, {}))
        with store_obj._connect() as connection_obj:
            connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str='invalid JSON' "
                "WHERE decision_plan_id_int=?", (other_decision_obj.decision_plan_id_int,))
    assert [plan_obj.decision_plan_id_int for plan_obj in store_obj.get_pending_daily_decision_plan_list(
        pod_id_str=release_obj.pod_id_str, env_mode_str="paper")] == [decision_obj.decision_plan_id_int]
    assert store_obj.get_pending_daily_decision_plan_list(pod_id_str="monthly", env_mode_str="paper") == []


def test_bad_daily_decision_does_not_starve_another_daily_cycle(daily_case, tmp_path):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    bad_release_obj = replace(release_obj, release_id_str="bad", pod_id_str="bad", account_route_str="DU_BAD")
    store_obj.upsert_release(bad_release_obj)
    bad_decision_obj = store_obj.insert_decision_plan(replace(decision_obj,
        decision_plan_id_int=None, release_id_str="bad", pod_id_str="bad", account_route_str="DU_BAD",
        target_execution_timestamp_ts=OPEN_TS - timedelta(days=1)))
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str='invalid JSON' "
            "WHERE decision_plan_id_int=?", (bad_decision_obj.decision_plan_id_int,))
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.as_of_ts = CLOSE_TS
    result_int = runner._reconcile_daily_cycles(store_obj,
        SimpleNamespace(get_adapter=lambda _release_obj: broker_obj), CLOSE_TS, "paper", None,
        str(tmp_path / "daily.log"), False, str(tmp_path / "traces"))
    assert result_int == 1
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    assert "daily_cycle_load_failed" in (tmp_path / "daily.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("strategy_str", MONTHLY_STRATEGY_LIST)
def test_monthly_eod_guard_never_reads_decision_metadata(tmp_path, strategy_str, baseline_runner_dict):
    store_obj, release_obj, plan_obj, _ = _monthly_case(tmp_path, strategy_str)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str='invalid JSON'")
    for status_str in ("ready", "submitted", "submitting", "completed"):
        store_obj.mark_vplan_status(plan_obj.vplan_id_int, status_str)
        assert runner._pod_has_unresolved_execution_bool(store_obj, release_obj.pod_id_str,
            release_obj=release_obj) == baseline_runner_dict["_pod_has_unresolved_execution_bool"](
                store_obj, release_obj.pod_id_str)


def test_legacy_park_helper_is_exact_production_source(baseline_runner_dict):
    source_str = baseline_runner_dict["source_str"]
    baseline_node_obj = next(node_obj for node_obj in ast.parse(source_str).body
        if isinstance(node_obj, ast.FunctionDef) and node_obj.name == "is_vplan_execution_exception_parked")
    assert inspect.getsource(runner.is_vplan_execution_exception_parked).strip() == ast.get_source_segment(
        source_str, baseline_node_obj).strip()


@pytest.mark.parametrize("strategy_str", MONTHLY_STRATEGY_LIST)
def test_monthly_decision_summary_and_trace_match_production(tmp_path, monkeypatch, strategy_str, baseline_runner_dict):
    store_obj, release_obj, plan_obj, _ = _monthly_case(tmp_path, strategy_str)
    decision_obj = store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int)
    assert runner._decision_plan_trace_payload_dict(decision_obj, release_obj) == baseline_runner_dict[
        "_decision_plan_trace_payload_dict"](decision_obj)
    monkeypatch.setattr(runner, "_load_release_list_and_sync", lambda *_arg_tuple, **_arg_dict: [release_obj])
    monkeypatch.setitem(baseline_runner_dict, "_load_release_list_and_sync", runner._load_release_list_and_sync)
    expected_dict = baseline_runner_dict["show_decision_plan_summary"](store_obj, CLOSE_TS, str(tmp_path),
        decision_plan_id_int=decision_obj.decision_plan_id_int)
    actual_dict = runner.show_decision_plan_summary(store_obj, CLOSE_TS, str(tmp_path),
        decision_plan_id_int=decision_obj.decision_plan_id_int)
    assert json.dumps(actual_dict, sort_keys=True) == json.dumps(expected_dict, sort_keys=True)
