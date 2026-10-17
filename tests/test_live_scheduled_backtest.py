"""Absolute scheduled diagnostic windows; no real backtests or Scheduler mutations."""

from datetime import datetime, timezone
from unittest.mock import Mock
import subprocess
import os
import sys
import time

import pytest

from alpha.live import scheduled_backtest as sb


@pytest.mark.parametrize('date_str,ruler_bool,expected_hour_int', [
    ('2026-10-16T06:29:59-04:00', False, None),
    ('2026-10-16T06:30:00-04:00', False, 9),
    ('2026-10-16T08:30:00-04:00', False, 9),
    ('2026-10-16T09:00:00-04:00', False, None),
    ('2026-10-17T08:30:00-04:00', False, None),
    ('2026-10-19T10:00:00-04:00', True, None),
    ('2026-10-17T09:59:59-04:00', True, None),
    ('2026-10-17T10:00:00-04:00', True, 18),
    ('2026-10-17T18:00:00-04:00', True, None),
    ('2026-12-18T13:30:00+00:00', False, 9),
])
def test_new_york_windows_including_dst(date_str, ruler_bool, expected_hour_int):
    result_ts = sb.deadline_ts(datetime.fromisoformat(date_str), rulers_only_bool=ruler_bool)
    assert (result_ts.hour if result_ts else None) == expected_hour_int


def test_late_start_launches_nothing():
    launch_fn = Mock()
    assert sb.run_scheduled_check(now_fn=lambda: datetime(2026, 10, 16, 15, tzinfo=timezone.utc), launch_fn=launch_fn) == 0
    launch_fn.assert_not_called()


def test_absolute_deadline_kills_only_launched_process_tree(monkeypatch):
    clock_list = iter(datetime.fromisoformat(value_str) for value_str in (
        '2026-10-16T08:30:00-04:00', '2026-10-16T08:59:59-04:00', '2026-10-16T09:00:00-04:00'))
    process_obj = Mock(pid=1234)
    process_obj.poll.return_value = None
    process_obj.wait.side_effect = subprocess.TimeoutExpired('backtest', 1)
    stop_fn = Mock()
    monkeypatch.setattr(sb, 'stop_process_tree', stop_fn)
    assert sb.run_scheduled_check(now_fn=lambda: next(clock_list), launch_fn=lambda command_list: process_obj) == 1
    process_obj.wait.assert_called_once_with(timeout=1.0)
    stop_fn.assert_called_once_with(process_obj)


def test_normal_exit_propagates_and_does_not_kill_finished_process(monkeypatch):
    process_obj = Mock(returncode=7)
    process_obj.poll.return_value = 7
    kill_fn = Mock()
    monkeypatch.setattr(sb.subprocess, 'run', kill_fn)
    assert sb.run_scheduled_check(now_fn=lambda: datetime.fromisoformat('2026-10-16T08:30:00-04:00'), launch_fn=lambda command_list: process_obj) == 7
    kill_fn.assert_not_called()


def test_windows_termination_targets_child_tree(monkeypatch):
    process_obj = Mock(pid=1234)
    process_obj.poll.return_value = None
    run_fn = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(sb.os, 'name', 'nt')
    monkeypatch.setattr(sb.subprocess, 'run', run_fn)
    sb.stop_process_tree(process_obj)
    assert run_fn.call_args.args[0] == ['taskkill.exe', '/PID', '1234', '/T', '/F']
    process_obj.wait.assert_called_once_with(timeout=10)


def test_already_exited_venv_launcher_does_not_turn_cleanup_into_failure(monkeypatch):
    process_obj = Mock(pid=1234)
    process_obj.poll.side_effect = [None, 1]
    monkeypatch.setattr(sb.os, 'name', 'nt')
    monkeypatch.setattr(sb.subprocess, 'run', lambda *arg_tuple, **kwarg_dict: subprocess.CompletedProcess([], 255))
    sb.stop_process_tree(process_obj)
    process_obj.wait.assert_called_once_with(timeout=10)


def test_failed_windows_termination_is_not_silenced(monkeypatch):
    process_obj = Mock(pid=1234)
    process_obj.poll.return_value = None
    monkeypatch.setattr(sb.os, 'name', 'nt')
    monkeypatch.setattr(sb.subprocess, 'run', lambda *arg_tuple, **kwarg_dict: subprocess.CompletedProcess([], 5))
    with pytest.raises(subprocess.CalledProcessError):
        sb.stop_process_tree(process_obj)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process-tree termination')
def test_deadline_reaps_real_harmless_child_and_grandchild(tmp_path):
    marker_path = tmp_path / 'grandchild_survived'
    grandchild_str = f'import time; from pathlib import Path; time.sleep(2); Path({str(marker_path)!r}).write_text("survived")'
    child_str = f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {grandchild_str!r}]); print("ready", flush=True); time.sleep(30)'
    process_list = []
    def launch_fn(command_list):
        process_obj = subprocess.Popen([sys.executable, '-c', child_str], stdout=subprocess.PIPE, text=True)
        process_list.append(process_obj)
        assert process_obj.stdout.readline().strip() == 'ready'
        return process_obj
    clock_list = iter(datetime.fromisoformat(value_str) for value_str in ('2026-10-16T08:59:59-04:00', '2026-10-16T09:00:00-04:00'))
    try:
        assert sb.run_scheduled_check(now_fn=lambda: next(clock_list), launch_fn=launch_fn) == 1
        assert process_list[0].poll() is not None
        time.sleep(2.5)
        assert not marker_path.exists()
    finally:
        for process_obj in process_list:
            sb.stop_process_tree(process_obj)
            process_obj.stdout.close()
