"""Scheduler liveness comes from bounded per-Pod LIVE event evidence."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alpha.live.dashboard_v4 import scheduler_status
from alpha.live.logging_utils import build_structured_event_record_dict


BASE_TS = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
POD_STR = "pod_example"


def _record_dict(name_str="scheduler_sleeping", *, event_ts=BASE_TS, trace_bool=False, **field_dict):
    payload_dict = {
        "env_mode_str": "live", "related_pod_id_list": [POD_STR],
        "next_phase_str": "idle_probe", "reason_code_str": "no_due_work",
        "sleep_seconds_float": 3600, "as_of_timestamp_str": (event_ts - timedelta(minutes=5)).isoformat(),
        **field_dict,
    }
    if trace_bool:
        payload_dict = {
            "mode_str": payload_dict["env_mode_str"], "pod_id_str": POD_STR,
            "reason_code_str": payload_dict["reason_code_str"],
            "payload_dict": {"scheduler_decision_dict": payload_dict.copy(), **payload_dict},
        }
    return build_structured_event_record_dict(name_str, payload_dict, timestamp_obj=event_ts)


def _write_records(path_obj, *record_dict_list):
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text("".join(json.dumps(record_dict) + "\n" for record_dict in record_dict_list), encoding="utf-8")


def _read_dict(path_obj, seconds_int=0, **argument_dict):
    return scheduler_status.load_scheduler_status_dict(str(path_obj), POD_STR,
        as_of_ts=BASE_TS + timedelta(seconds=seconds_int), **argument_dict)


def _trace_path_obj(tmp_path, phase_str="idle_probe", backup_int=0):
    name_str = "trace_events.jsonl" + (f".{backup_int}" if backup_int else "")
    return tmp_path / "pods" / POD_STR / f"live_{POD_STR}_scheduler_{phase_str}" / name_str


@pytest.mark.parametrize("seconds_int,state_str,alive_bool", [
    (0, "sleeping", True), (3599, "sleeping", True), (3600, "sleeping", True),
    (3660, "sleeping", True), (3661, "late", None), (3900, "late", None), (3901, "stopped", False),
])
def test_promised_sleep_controls_liveness_not_fixed_heartbeat_age(tmp_path, seconds_int, state_str, alive_bool):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict())
    result_dict = _read_dict(path_obj, seconds_int)
    assert result_dict["state_str"] == state_str
    assert result_dict["alive_bool"] is alive_bool
    assert result_dict["last_seen_timestamp_str"] == BASE_TS.isoformat()
    assert result_dict["promised_wake_timestamp_str"] == (BASE_TS + timedelta(hours=1)).isoformat()
    if state_str == "stopped":
        assert "Check the service and logs" in result_dict["detail_str"]
        assert "process" not in result_dict["detail_str"]


@pytest.mark.parametrize("backup_int", [1, 10])
def test_global_log_rotation_is_read_with_missing_or_truncated_current(tmp_path, backup_int):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj.with_name(path_obj.name + f".{backup_int}"), _record_dict())
    assert _read_dict(path_obj)["state_str"] == "sleeping"
    path_obj.write_bytes(b"")
    assert _read_dict(path_obj)["state_str"] == "sleeping"


def test_truncation_cannot_reuse_a_cached_green_status(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict())
    assert _read_dict(path_obj)["alive_bool"] is True
    path_obj.write_bytes(b"")
    assert _read_dict(path_obj)["state_str"] == "unknown"


def test_unscoped_long_sync_cannot_certify_live_scheduler_or_remove_overdue_warning(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    # The actual sync writer has Pod/release lists but no mode identity. A
    # long in-flight sync can be alive; these records cannot prove that here.
    sync_dict = build_structured_event_record_dict("norgate_snapshot_sync_started", {
        "pod_id_list": [POD_STR], "release_id_list": ["release_one"], "status_str": "syncing"},
        timestamp_obj=BASE_TS + timedelta(seconds=3900))
    _write_records(path_obj, _record_dict(), sync_dict)
    result_dict = _read_dict(path_obj, 3901)
    assert result_dict["state_str"] == "stopped"
    assert "Check the service and logs" in result_dict["detail_str"]
    assert result_dict["last_seen_timestamp_str"] == BASE_TS.isoformat()


@pytest.mark.parametrize("conflict_str", ["pod", "mode", "related"])
def test_new_selected_pod_identity_conflict_cannot_leave_old_sleep_green(tmp_path, conflict_str):
    path_obj = tmp_path / "events.jsonl"
    conflict_dict = _record_dict(event_ts=BASE_TS + timedelta(seconds=10), trace_bool=True)
    if conflict_str == "pod":
        conflict_dict["payload_dict"]["pod_id_str"] = "other_pod"
    elif conflict_str == "mode":
        conflict_dict["payload_dict"]["mode_str"] = "paper"
    else:
        conflict_dict["payload_dict"]["payload_dict"]["related_pod_id_list"] = ["other_pod"]
    _write_records(path_obj, _record_dict(), conflict_dict)
    result_dict = _read_dict(path_obj, 20)
    assert result_dict["state_str"] == "unknown"
    assert result_dict["alive_bool"] is None


def test_partial_append_is_ignored_without_losing_last_complete_record(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict())
    with path_obj.open("ab") as source_file_obj:
        source_file_obj.write(b'{"event_name_str":"scheduler_sleeping"')
    assert _read_dict(path_obj)["state_str"] == "sleeping"


@pytest.mark.parametrize("content_bytes", [b"not json\n", b"[]\n", b"\xff\n", b'{"partial":'])
def test_malformed_or_only_partial_log_fails_closed(tmp_path, content_bytes):
    path_obj = tmp_path / "events.jsonl"
    path_obj.write_bytes(content_bytes)
    assert _read_dict(path_obj)["state_str"] == "unknown"


@pytest.mark.parametrize("field_dict", [
    {"env_mode_str": "paper"}, {"env_mode_str": "sim"}, {"env_mode_str": None},
    {"related_pod_id_list": ["another_pod"]}, {"related_pod_id_list": []},
    {"pod_id_str": "another_pod"}, {"mode_str": "paper"},
])
def test_wrong_mode_or_pod_never_becomes_liveness(tmp_path, field_dict):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(**field_dict))
    assert _read_dict(path_obj)["state_str"] == "unknown"


@pytest.mark.parametrize("name_str", ["scheduler_started", "scheduler_woke", "scheduler_error_retry"])
def test_unscoped_global_events_are_not_attributed_to_a_pod(tmp_path, name_str):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(name_str, related_pod_id_list=[]))
    assert _read_dict(path_obj)["state_str"] == "unknown"


@pytest.mark.parametrize("duration_obj", [-1, float("inf"), float("nan"), "3600", None, True, 1e100])
def test_invalid_sleep_duration_fails_closed(tmp_path, duration_obj):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(sleep_seconds_float=duration_obj))
    assert _read_dict(path_obj)["state_str"] == "unknown"


@pytest.mark.parametrize("timestamp_str", ["not a time", "2026-09-21T12:00:00", "2026-09-21T12:00:01+00:00"])
def test_malformed_naive_and_future_event_timestamps_fail_closed(tmp_path, timestamp_str):
    path_obj = tmp_path / "events.jsonl"
    record_dict = _record_dict()
    record_dict["event_timestamp_str"] = timestamp_str
    _write_records(path_obj, record_dict)
    assert _read_dict(path_obj)["state_str"] == "unknown"


@pytest.mark.parametrize("name_str", ["scheduler_due_now", "scheduler.decision", "scheduler.tick_result"])
def test_run_once_activity_alone_does_not_prove_a_continuous_service(tmp_path, name_str):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(name_str))
    result_dict = _read_dict(path_obj)
    assert result_dict["state_str"] == "unknown"
    assert result_dict["alive_bool"] is None
    assert "continuous service is not verified" in result_dict["detail_str"]


def test_active_work_after_current_sleep_is_bounded_and_never_called_stopped(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj,
        _record_dict(event_ts=BASE_TS - timedelta(seconds=30), sleep_seconds_float=30),
        _record_dict("scheduler_due_now", next_phase_str="post_execution_reconcile"))
    assert _read_dict(path_obj, 120)["state_str"] == "running"
    assert _read_dict(path_obj, 121)["state_str"] == "unknown"
    assert _read_dict(path_obj, 9999)["state_str"] == "unknown"


def test_late_run_once_does_not_resurrect_old_sleep_evidence(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj,
        _record_dict(event_ts=BASE_TS - timedelta(hours=2), sleep_seconds_float=30),
        _record_dict("scheduler_due_now"))
    assert _read_dict(path_obj)["state_str"] == "unknown"


@pytest.mark.parametrize("reason_str", ["execution_exception_parked", "manual_review_required"])
def test_manual_hold_is_taken_from_real_scheduler_phase(tmp_path, reason_str):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(next_phase_str="manual_review_pending", reason_code_str=reason_str))
    result_dict = _read_dict(path_obj, 1800)
    assert result_dict["state_str"] == "holding"
    assert "will not retry by itself" in result_dict["detail_str"]


def test_missing_ack_does_not_invent_a_hold(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(next_phase_str="post_execution_reconcile",
        reason_code_str="waiting_for_post_execution_reconcile", sleep_seconds_float=30))
    result_dict = _read_dict(path_obj)
    assert result_dict["state_str"] == "sleeping"
    assert "holds" not in result_dict["detail_str"]


def test_data_wait_comes_from_actual_scheduler_flag(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict(norgate_snapshot_sync_active_wait_bool=True))
    assert "waits for data" in _read_dict(path_obj)["detail_str"]


@pytest.mark.parametrize("backup_int", [0, 1, 5])
def test_real_double_wrapped_trace_error_and_retry_are_scoped_redacted(tmp_path, backup_int):
    path_obj = tmp_path / "events.jsonl"
    _write_records(_trace_path_obj(tmp_path, backup_int=backup_int),
        _record_dict("scheduler.error_retry", trace_bool=True, error_retry_seconds_int=30,
            error_str="Authorization: Bearer secretkey Account DU123456 host=private password=hunter2"))
    result_dict = _read_dict(path_obj)
    assert result_dict["state_str"] == "error"
    assert result_dict["alive_bool"] is True
    assert result_dict["promised_wake_timestamp_str"] == (BASE_TS + timedelta(seconds=30)).isoformat()
    assert all(secret_str not in str(result_dict) for secret_str in ("secretkey", "DU123456", "hunter2", "private"))
    assert _read_dict(path_obj, 91)["state_str"] == "late"
    assert _read_dict(path_obj, 331)["state_str"] == "stopped"


def test_newest_real_event_across_phase_traces_and_main_log_wins(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict())
    _write_records(_trace_path_obj(tmp_path, "manual_review_pending"),
        _record_dict("scheduler.sleeping", event_ts=BASE_TS + timedelta(seconds=1), trace_bool=True,
            next_phase_str="manual_review_pending", reason_code_str="manual_review_required"))
    assert _read_dict(path_obj, 2)["state_str"] == "holding"
    _write_records(path_obj, _record_dict(event_ts=BASE_TS + timedelta(seconds=3)))
    assert _read_dict(path_obj, 4)["state_str"] == "sleeping"


def test_error_wins_when_distinct_sources_have_identical_write_times(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict())
    _write_records(_trace_path_obj(tmp_path),
        _record_dict("scheduler.error_retry", trace_bool=True, error_retry_seconds_int=30))
    assert _read_dict(path_obj)["state_str"] == "error"


def test_conflicting_timestamp_copies_fail_closed(tmp_path):
    path_obj = tmp_path / "events.jsonl"
    record_dict = _record_dict()
    record_dict["ts_utc"] = (BASE_TS + timedelta(seconds=1)).isoformat()
    _write_records(path_obj, record_dict)
    assert _read_dict(path_obj)["state_str"] == "unknown"


def test_unreadable_log_returns_no_exception_or_raw_path(tmp_path, monkeypatch):
    path_obj = tmp_path / "events.jsonl"
    def denied_fn(*argument_list, **keyword_dict):
        raise PermissionError("password=do-not-show C:/private-account")
    monkeypatch.setattr(Path, "open", denied_fn)
    result_dict = _read_dict(path_obj)
    assert result_dict["state_str"] == "unknown"
    assert "password" not in str(result_dict) and "private-account" not in str(result_dict)


@pytest.mark.parametrize("pod_str", ["../other", "..", ".", "x/y", "x\\y", "C:\\private", "", "x" * 201])
def test_pod_path_traversal_is_rejected_before_file_access(tmp_path, monkeypatch, pod_str):
    def forbidden_fn(*argument_list, **keyword_dict):
        raise AssertionError("Invalid Pod must not open files")
    monkeypatch.setattr(Path, "open", forbidden_fn)
    result_dict = scheduler_status.load_scheduler_status_dict(str(tmp_path / "events.jsonl"), pod_str, as_of_ts=BASE_TS)
    assert result_dict["state_str"] == "unknown"


def test_trace_root_cannot_escape_configured_event_log_directory(tmp_path, monkeypatch):
    def forbidden_fn(*argument_list, **keyword_dict):
        raise AssertionError("Escaped source must not open files")
    monkeypatch.setattr(Path, "open", forbidden_fn)
    assert _read_dict(tmp_path / "events.jsonl", trace_root_path_str=str(tmp_path.parent / "other"))["state_str"] == "unknown"


def test_missing_custom_source_does_not_fallback_to_real_default(tmp_path, monkeypatch):
    original_open_fn = Path.open
    def contained_open_fn(path_obj, *argument_list, **keyword_dict):
        path_obj.relative_to(tmp_path)
        return original_open_fn(path_obj, *argument_list, **keyword_dict)
    monkeypatch.setattr(Path, "open", contained_open_fn)
    assert _read_dict(tmp_path / "events.jsonl")["state_str"] == "unknown"


def test_tail_bound_does_not_read_older_evidence_or_directory_walk(tmp_path, monkeypatch):
    path_obj = tmp_path / "events.jsonl"
    _write_records(path_obj, _record_dict())
    with path_obj.open("ab") as source_file_obj:
        source_file_obj.write((b'{"event_name_str":"unrelated"}\n') * 3000)
    def forbidden_fn(*argument_list, **keyword_dict):
        raise AssertionError("Reader must not enumerate directories")
    monkeypatch.setattr(Path, "iterdir", forbidden_fn)
    monkeypatch.setattr(Path, "glob", forbidden_fn)
    monkeypatch.setattr(Path, "rglob", forbidden_fn)
    assert _read_dict(path_obj)["state_str"] == "unknown"
    with path_obj.open("ab") as source_file_obj:
        source_file_obj.write((json.dumps(_record_dict()) + "\n").encode())
    assert _read_dict(path_obj)["state_str"] == "sleeping"


def test_symlink_escape_returns_unknown(tmp_path):
    outside_path_obj = tmp_path.parent / f"{tmp_path.name}-outside.jsonl"
    _write_records(outside_path_obj, _record_dict())
    path_obj = tmp_path / "events.jsonl"
    try:
        path_obj.symlink_to(outside_path_obj)
    except OSError:
        pytest.skip("Symlinks are unavailable on this host")
    assert _read_dict(path_obj)["state_str"] == "unknown"
