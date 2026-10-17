"""Scheduled-task list and its PASS / FAIL check (alpha/live/scheduled_tasks.py) plus the System health row."""

from datetime import UTC, datetime, timedelta
import json

from alpha.live import scheduled_tasks as st

NOW = datetime(2026, 12, 10, 12, 0, tzinfo=UTC)


def _task(wrapper, *, last=NOW - timedelta(minutes=3), result=0, state="Ready", next_run=NOW + timedelta(minutes=2)):
    return {"State": state, "Args": f'powershell.exe -NoProfile -File "C:\\repo\\scripts\\{wrapper}"',
            "LastRun": last.isoformat() if last else "", "Result": result, "NextRun": next_run.isoformat() if next_run else ""}


def _states(query_dict):
    rows = st.check_task_list(now_ts=NOW, query_fn=lambda name: query_dict.get(name))
    return {row["name_str"]: row for row in rows}


def test_every_expected_task_has_a_wrapper_in_the_repo():
    for spec in st.TASK_TUPLE:
        assert (st.REPO_ROOT_PATH / "scripts" / spec.wrapper_str).is_file(), spec.wrapper_str
    assert (st.REPO_ROOT_PATH / "scripts" / "setup_live_tasks.ps1").is_file()


def test_healthy_tasks_pass_and_missing_ones_fail():
    rows = _states({spec.name_str: _task(spec.wrapper_str) for spec in st.TASK_TUPLE if spec.name_str != "AlphaBacktestRuler"})
    assert rows["AlphaLiveOpsWatchdog"]["state_str"] == "PASS"
    assert rows["AlphaBacktestRuler"]["state_str"] == "FAIL" and "Not registered" in rows["AlphaBacktestRuler"]["detail_str"]


def test_disabled_wrong_action_failed_and_late_runs_fail():
    spec = st.TASK_TUPLE[0]
    assert _states({spec.name_str: _task(spec.wrapper_str, state="Disabled")})[spec.name_str]["state_str"] == "FAIL"
    assert _states({spec.name_str: _task("other.ps1")})[spec.name_str]["state_str"] == "FAIL"
    assert "code 1" in _states({spec.name_str: _task(spec.wrapper_str, result=1)})[spec.name_str]["detail_str"]
    late = _states({spec.name_str: _task(spec.wrapper_str, last=NOW - timedelta(hours=2))})[spec.name_str]
    assert late["state_str"] == "FAIL" and "last ran" in late["detail_str"]


def test_never_run_and_running_tasks_pass():
    spec = st.TASK_TUPLE[2]
    assert _states({spec.name_str: _task(spec.wrapper_str, last=None, result=267011)})[spec.name_str]["state_str"] == "PASS"
    assert _states({spec.name_str: _task(spec.wrapper_str, result=267009)})[spec.name_str]["state_str"] == "PASS"


def test_query_error_is_a_fail_not_a_crash():
    def broken(name):
        raise RuntimeError("access denied")
    rows = st.check_task_list(now_ts=NOW, query_fn=broken)
    assert all(row["state_str"] == "FAIL" for row in rows)


def test_receipt_is_written_and_read_by_system_health(tmp_path):
    log_path = tmp_path / "live_events.jsonl"
    log_path.write_text("", encoding="utf-8")
    receipt = st.run_check(event_log_path_str=str(log_path), performance_db_path_str=None, now_ts=NOW,
                           query_fn=lambda name: None)
    saved = json.loads((tmp_path / st.RECEIPT_FILE_NAME_STR).read_text(encoding="utf-8"))
    assert saved["verdict_str"] == "FAIL" and receipt["fail_count_int"] >= len(st.TASK_TUPLE)

    from alpha.live.dashboard_v4.system_data import _scheduled_tasks_dict
    row = _scheduled_tasks_dict(tmp_path / st.RECEIPT_FILE_NAME_STR, NOW + timedelta(minutes=1))
    assert row["state_str"] == "error" and row["now_str"].startswith("FAIL")
    saved.update(fail_count_int=0, verdict_str="PASS",
                 task_row_list=[dict(r, state_str="PASS") for r in saved["task_row_list"]],
                 environment_row_list=[dict(r, state_str="PASS") for r in saved["environment_row_list"]])
    (tmp_path / st.RECEIPT_FILE_NAME_STR).write_text(json.dumps(saved), encoding="utf-8")
    assert _scheduled_tasks_dict(tmp_path / st.RECEIPT_FILE_NAME_STR, NOW + timedelta(minutes=1))["state_str"] == "ok"
    assert _scheduled_tasks_dict(tmp_path / st.RECEIPT_FILE_NAME_STR, NOW + timedelta(days=5))["state_str"] == "warning"
