"""One list of the Live Ops scheduled tasks and a doctor-style check of them.

    python -m alpha.live.scheduled_tasks list      # the expected tasks (JSON, read by setup_live_tasks.ps1)
    python -m alpha.live.scheduled_tasks check     # PASS / FAIL lines, exit 1 on any FAIL

`scripts/setup_live_tasks.ps1` registers every task in TASK_TUPLE and then runs `check`.
The check only reads (Windows Task Scheduler, files, the Flex store in mode=ro). It writes one
receipt, `scheduled_tasks_check.json`, next to the watchdog report; Dashboard V4 System health
shows it. It never registers, starts or stops a task.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
RECEIPT_FILE_NAME_STR = "scheduled_tasks_check.json"
SCHEMA_VERSION_STR = "scheduled_tasks_check.v1"
NEVER_RUN_RESULT_SET = frozenset({267011})          # SCHED_S_TASK_HAS_NOT_RUN
RUNNING_RESULT_SET = frozenset({267009})            # SCHED_S_TASK_RUNNING
EXPECTED_TIME_ZONE_STR = "Eastern Standard Time"


@dataclass(frozen=True)
class TaskSpec:
    name_str: str
    wrapper_str: str            # scripts/<wrapper>.ps1 that the task must run
    schedule_str: str           # human text, also shown on System health
    max_gap_hours_float: float  # a healthy task has run within this many hours


TASK_TUPLE = (
    TaskSpec("AlphaLiveOpsWatchdog", "run_live_ops_watchdog.ps1", "Every 5 min", 0.5),
    TaskSpec("AlphaIbkrPerformanceSync", "run_ibkr_performance_sync.ps1", "Daily 06:15 ET", 26.0),
    TaskSpec("AlphaBacktestCheck", "run_backtest_check.ps1", "Mon-Fri 06:30 ET", 74.0),
    TaskSpec("AlphaBacktestRuler", "run_backtest_ruler.ps1", "Saturday 10:00 ET", 8 * 24.0),
)


def _query_task_dict(name_str: str) -> dict | None:
    """Read one task from Windows Task Scheduler; None when it is not registered."""
    command_str = (f"$t = Get-ScheduledTask -TaskName '{name_str}' -ErrorAction SilentlyContinue; "
                   "if ($null -eq $t) { 'null' } else { $i = $t | Get-ScheduledTaskInfo; "
                   "[pscustomobject]@{State=[string]$t.State; Args=(($t.Actions | ForEach-Object { $_.Execute + ' ' + $_.Arguments }) -join ' | '); "
                   "LastRun=$(if ($i.LastRunTime) { $i.LastRunTime.ToUniversalTime().ToString('o') } else { '' }); "
                   "Result=[int64]$i.LastTaskResult; "
                   "NextRun=$(if ($i.NextRunTime) { $i.NextRunTime.ToUniversalTime().ToString('o') } else { '' })} | ConvertTo-Json -Compress }")
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command_str],
                               capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip()[:200] or "Task Scheduler query failed")
    text_str = completed.stdout.strip()
    return None if text_str in {"", "null"} else json.loads(text_str)


def _parse_ts(value_str: str):
    try:
        timestamp_ts = datetime.fromisoformat(str(value_str).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Task Scheduler reports 1999-11-30 for "never run".
    return timestamp_ts if timestamp_ts.year > 2000 else None


def check_task_list(spec_tuple=TASK_TUPLE, *, now_ts: datetime, query_fn=_query_task_dict) -> list[dict]:
    result_list = []
    for spec in spec_tuple:
        try:
            task_dict = query_fn(spec.name_str)
        except Exception as error_obj:
            result_list.append({"name_str": spec.name_str, "state_str": "FAIL", "detail_str": f"Cannot read Task Scheduler: {error_obj}"[:200]})
            continue
        if task_dict is None:
            result_list.append({"name_str": spec.name_str, "state_str": "FAIL", "detail_str": "Not registered - run scripts\\setup_live_tasks.ps1"})
            continue
        problem_list, note_list = [], []
        if task_dict.get("State") == "Disabled":
            problem_list.append("disabled")
        if spec.wrapper_str not in str(task_dict.get("Args", "")):
            problem_list.append(f"does not run {spec.wrapper_str}")
        if not _parse_ts(task_dict.get("NextRun", "")):
            problem_list.append("no next run")
        last_ts, result_int = _parse_ts(task_dict.get("LastRun", "")), int(task_dict.get("Result") or 0)
        if last_ts is None or result_int in NEVER_RUN_RESULT_SET:
            note_list.append("not run yet")
        else:
            if result_int not in RUNNING_RESULT_SET and result_int != 0:
                problem_list.append(f"last run failed (code {result_int})")
            if now_ts - last_ts > timedelta(hours=spec.max_gap_hours_float):
                problem_list.append(f"last ran {last_ts.astimezone().strftime('%a %m-%d %H:%M')}")
            else:
                note_list.append(f"last ran {last_ts.astimezone().strftime('%a %m-%d %H:%M')} OK")
        result_list.append({"name_str": spec.name_str, "state_str": "FAIL" if problem_list else "PASS",
                            "detail_str": spec.schedule_str + " - " + ", ".join(problem_list or note_list)})
    return result_list


def check_environment_list(*, event_log_dir_path: Path, performance_db_path_str: str | None) -> list[dict]:
    result_list = []

    def add(name_str, ok_bool, detail_str, *, warn_bool=False):
        result_list.append({"name_str": name_str, "state_str": "PASS" if ok_bool else ("WARN" if warn_bool else "FAIL"), "detail_str": detail_str})

    try:
        completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "(Get-TimeZone).Id"],
                                   capture_output=True, text=True, timeout=30)
        zone_str = completed.stdout.strip()
        add("Time zone", zone_str == EXPECTED_TIME_ZONE_STR, zone_str or "unknown")
    except Exception as error_obj:
        add("Time zone", False, f"unreadable: {type(error_obj).__name__}")
    uv_str = shutil.which("uv") or next((str(path_obj) for path_obj in (Path.home() / ".local/bin/uv.exe", Path.home() / ".cargo/bin/uv.exe")
                                         if path_obj.is_file()), "")
    add("uv", bool(uv_str), uv_str or "not found on PATH or in the user install folders")
    for spec in TASK_TUPLE:
        wrapper_path = REPO_ROOT_PATH / "scripts" / spec.wrapper_str
        add(f"Wrapper {spec.wrapper_str}", wrapper_path.is_file(), str(wrapper_path))
    from scripts.norgate_config_env import default_config_env_path_obj

    config_path = default_config_env_path_obj()
    add("config.env", config_path.is_file(), str(config_path))
    add("Live ops folder", event_log_dir_path.is_dir(), str(event_log_dir_path))
    if performance_db_path_str and Path(performance_db_path_str).is_file():
        try:
            with closing(sqlite3.connect(Path(performance_db_path_str).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as connection_obj:
                count_int = connection_obj.execute("SELECT COUNT(*) FROM flex_import").fetchone()[0]
            add("Flex store", count_int > 0, f"{count_int} saved imports - read-only")
        except sqlite3.Error as error_obj:
            add("Flex store", False, f"unreadable: {error_obj}"[:200])
    else:
        add("Flex store", False, str(performance_db_path_str or "path unknown"))
    report_path = event_log_dir_path / "backtest_check_latest.json"
    if report_path.is_file():
        try:
            payload_dict = json.loads(report_path.read_text(encoding="utf-8"))
            failed_list = [pod_dict.get("pod_id_str") for pod_dict in payload_dict.get("pod_list") or []
                           if pod_dict.get("reference_status_str") == "failed"]
            add("Backtest check report", not failed_list,
                f"{payload_dict.get('as_of_date_str')} - {len(payload_dict.get('pod_list') or [])} pods"
                + (f" - failed: {', '.join(map(str, failed_list))}" if failed_list else ""))
        except (OSError, ValueError, AttributeError) as error_obj:
            add("Backtest check report", False, f"unreadable: {type(error_obj).__name__}")
    else:
        add("Backtest check report", False, "not written yet (first run Mon-Fri 06:30 ET)", warn_bool=True)
    return result_list


def run_check(*, event_log_path_str: str, performance_db_path_str: str | None, now_ts: datetime | None = None,
              query_fn=_query_task_dict) -> dict:
    now_ts = now_ts or datetime.now(UTC)
    event_log_dir_path = Path(event_log_path_str).resolve().parent
    task_list = check_task_list(now_ts=now_ts, query_fn=query_fn)
    environment_list = check_environment_list(event_log_dir_path=event_log_dir_path, performance_db_path_str=performance_db_path_str)
    row_list = task_list + environment_list
    fail_int = sum(1 for row_dict in row_list if row_dict["state_str"] == "FAIL")
    receipt_dict = {"schema_version_str": SCHEMA_VERSION_STR, "checked_at_utc_str": now_ts.isoformat(),
                    "verdict_str": "FAIL" if fail_int else "PASS", "fail_count_int": fail_int,
                    "task_row_list": task_list, "environment_row_list": environment_list,
                    "expected_task_list": [asdict(spec) for spec in TASK_TUPLE]}
    receipt_path = event_log_dir_path / RECEIPT_FILE_NAME_STR
    if event_log_dir_path.is_dir():
        temporary_path = receipt_path.with_name(receipt_path.name + f".{os.getpid()}.tmp")
        temporary_path.write_text(json.dumps(receipt_dict, indent=1), encoding="utf-8")
        os.replace(temporary_path, receipt_path)
    return receipt_dict


def main(argv=None) -> int:
    from alpha.live.dashboard import DEFAULT_EVENT_LOG_PATH_STR

    parser = argparse.ArgumentParser(description="Live Ops scheduled tasks: list them, or check them (PASS / FAIL).")
    parser.add_argument("command", choices=("list", "check"))
    parser.add_argument("--event-log-path", default=DEFAULT_EVENT_LOG_PATH_STR, help="The receipt goes next to this file.")
    parser.add_argument("--performance-db", default=None)
    parser.add_argument("--skip-env-file", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps([asdict(spec) for spec in TASK_TUPLE]))
        return 0
    if not args.skip_env_file:
        from scripts.norgate_config_env import load_config_env_file
        load_config_env_file()
    if args.performance_db is None:
        from alpha.live.ibkr_performance import resolve_performance_db_path_str
        args.performance_db = resolve_performance_db_path_str()
    receipt_dict = run_check(event_log_path_str=args.event_log_path, performance_db_path_str=args.performance_db)
    for row_dict in receipt_dict["task_row_list"] + receipt_dict["environment_row_list"]:
        print(f"[{row_dict['state_str']}] {row_dict['name_str']}: {row_dict['detail_str']}")
    print(f"[{receipt_dict['verdict_str']}] Scheduled tasks: {receipt_dict['fail_count_int']} problem(s).")
    return 1 if receipt_dict["fail_count_int"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
