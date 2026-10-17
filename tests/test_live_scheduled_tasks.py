"""Scheduled-task list and its PASS / FAIL check (alpha/live/scheduled_tasks.py) plus the System health row."""

from datetime import UTC, datetime, timedelta
import json
from xml.sax.saxutils import escape

import pytest

from alpha.live import scheduled_tasks as st

NOW = datetime(2026, 12, 10, 12, 0, tzinfo=UTC)


def _contract_xml(wrapper_str):
    spec_obj = next((spec_obj for spec_obj in st.TASK_TUPLE if spec_obj.wrapper_str == wrapper_str), st.TASK_TUPLE[0])
    argument_str = f'-NoProfile -ExecutionPolicy Bypass -File "{st.REPO_ROOT_PATH / "scripts" / wrapper_str}"'
    if spec_obj.name_str == "AlphaLiveOpsWatchdog":
        argument_str += " -Mode live"
        trigger_str = '<TimeTrigger><StartBoundary>2026-12-01T06:00:00</StartBoundary><Repetition><Interval>PT5M</Interval></Repetition></TimeTrigger>'
        limit_str = "PT10M"
    elif spec_obj.name_str == "AlphaIbkrPerformanceSync":
        trigger_str = '<CalendarTrigger><StartBoundary>2026-12-01T06:15:00</StartBoundary><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>'
        limit_str = "PT10M"
    else:
        ruler_bool = spec_obj.name_str == "AlphaBacktestRuler"
        time_str = '10:00' if ruler_bool else '06:30'
        day_list = ['Saturday'] if ruler_bool else ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_str = ''.join(f'<{day_str}/>' for day_str in day_list)
        trigger_str = f'<CalendarTrigger><StartBoundary>2026-12-01T{time_str}:00</StartBoundary><ScheduleByWeek><WeeksInterval>1</WeeksInterval><DaysOfWeek>{day_str}</DaysOfWeek></ScheduleByWeek></CalendarTrigger>'
        limit_str = "PT8H" if ruler_bool else "PT2H30M"
    catchup_str = 'false' if spec_obj.name_str.startswith('AlphaBacktest') else 'true'
    return (f'<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><Actions><Exec><Command>powershell.exe</Command>'
            f'<Arguments>{escape(argument_str)}</Arguments><WorkingDirectory>{escape(str(st.REPO_ROOT_PATH))}</WorkingDirectory></Exec></Actions>'
            f'<Triggers>{trigger_str}</Triggers><Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><ExecutionTimeLimit>{limit_str}</ExecutionTimeLimit>'
            f'<StartWhenAvailable>{catchup_str}</StartWhenAvailable></Settings><Principals><Principal><LogonType>S4U</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals></Task>')


def _task(wrapper, *, last=NOW - timedelta(minutes=3), result=0, state="Ready", next_run=NOW + timedelta(minutes=2)):
    return {"State": state, "Args": f'powershell.exe -NoProfile -File "C:\\repo\\scripts\\{wrapper}"',
            "Xml": _contract_xml(wrapper),
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


def test_a_turned_off_task_is_off_not_missing(tmp_path):
    spec = st.TASK_TUPLE[2]
    row = _states({spec.name_str: _task(spec.wrapper_str, state="Disabled")})[spec.name_str]
    assert row["state_str"] == "OFF" and "-Enable" in row["detail_str"]
    log_path = tmp_path / "live_events.jsonl"
    log_path.write_text("", encoding="utf-8")
    query = {s.name_str: _task(s.wrapper_str, state="Disabled" if s.name_str == spec.name_str else "Ready") for s in st.TASK_TUPLE}
    receipt = st.run_check(event_log_path_str=str(log_path), performance_db_path_str=None, now_ts=NOW, query_fn=query.get)
    assert all(r["state_str"] != "FAIL" for r in receipt["task_row_list"])
    saved = json.loads((tmp_path / st.RECEIPT_FILE_NAME_STR).read_text(encoding="utf-8"))
    saved["environment_row_list"] = [dict(r, state_str="PASS") for r in saved["environment_row_list"]]
    (tmp_path / st.RECEIPT_FILE_NAME_STR).write_text(json.dumps(saved), encoding="utf-8")
    from alpha.live.dashboard_v4.system_data import _scheduled_tasks_dict
    row = _scheduled_tasks_dict(tmp_path / st.RECEIPT_FILE_NAME_STR, NOW + timedelta(minutes=1))
    assert row["state_str"] == "warning" and row["now_str"].startswith("OFF")


@pytest.mark.parametrize('old_str,new_str', [
    ('-Mode live', '-Mode paper'), ('IgnoreNew', 'Parallel'), ('PT5M', 'PT24H'),
    ('-Mode live', '-Mode "live'),
    ('PT10M', 'PT1H'), ('S4U', 'InteractiveToken'),
    ('<Command>powershell.exe', '<Command>C:\\retired\\powershell.exe'),
    ('</WorkingDirectory>', '\\retired</WorkingDirectory>'),
    ('</Actions>', '<Exec><Command>other.exe</Command></Exec></Actions>'),
])
def test_wrong_watchdog_contract_fails(old_str, new_str):
    spec_obj = st.TASK_TUPLE[0]
    task_dict = _task(spec_obj.wrapper_str)
    task_dict['Xml'] = task_dict['Xml'].replace(old_str, new_str)
    assert _states({spec_obj.name_str: task_dict})[spec_obj.name_str]['state_str'] == 'FAIL'


@pytest.mark.parametrize('task_index_int,old_str,new_str', [
    (1, '06:15', '09:30'), (2, '<Friday/>', '<Saturday/>'),
    (2, '<WeeksInterval>1', '<WeeksInterval>2'), (3, '<Saturday/>', '<Monday/>'),
    (2, '<StartWhenAvailable>false', '<StartWhenAvailable>true'),
])
def test_wrong_calendar_or_catchup_fails(task_index_int, old_str, new_str):
    spec_obj = st.TASK_TUPLE[task_index_int]
    task_dict = _task(spec_obj.wrapper_str)
    task_dict['Xml'] = task_dict['Xml'].replace(old_str, new_str)
    assert _states({spec_obj.name_str: task_dict})[spec_obj.name_str]['state_str'] == 'FAIL'


def test_retired_release_and_missing_xml_fail():
    spec_obj = st.TASK_TUPLE[0]
    task_dict = _task(spec_obj.wrapper_str)
    task_dict['Xml'] = task_dict['Xml'].replace(escape(str(st.REPO_ROOT_PATH)), 'C:\\retired')
    assert _states({spec_obj.name_str: task_dict})[spec_obj.name_str]['state_str'] == 'FAIL'
    task_dict.pop('Xml')
    assert _states({spec_obj.name_str: task_dict})[spec_obj.name_str]['state_str'] == 'FAIL'


def test_watchdog_next_run_tomorrow_is_not_every_five_minutes():
    spec_obj = st.TASK_TUPLE[0]
    assert _states({spec_obj.name_str: _task(spec_obj.wrapper_str, next_run=NOW + timedelta(days=1))})[spec_obj.name_str]['state_str'] == 'FAIL'


@pytest.mark.parametrize('task_index_int,expected_str', [(0, 'PASS'), (1, 'FAIL'), (2, 'FAIL')])
def test_utc_boundary_allowed_only_for_interval_watchdog(task_index_int, expected_str):
    spec_obj = st.TASK_TUPLE[task_index_int]
    task_dict = _task(spec_obj.wrapper_str)
    task_dict['Xml'] = task_dict['Xml'].replace('</StartBoundary>', 'Z</StartBoundary>')
    assert _states({spec_obj.name_str: task_dict})[spec_obj.name_str]['state_str'] == expected_str
