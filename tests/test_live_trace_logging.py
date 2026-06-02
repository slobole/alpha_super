from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import alpha.live.logging_utils as logging_utils


def _read_jsonl_list(path_obj: Path) -> list[dict[str, object]]:
    return [
        json.loads(line_str)
        for line_str in path_obj.read_text(encoding="utf-8").splitlines()
        if line_str.strip()
    ]


def _make_trace_run_folder_path_obj(
    trace_root_path_obj: Path,
    *,
    pod_id_str: str,
    run_id_str: str,
    now_ts: datetime,
    age_days_int: int,
) -> Path:
    run_folder_path_obj = trace_root_path_obj / pod_id_str / run_id_str
    trace_file_path_obj = run_folder_path_obj / "trace_events.jsonl"
    trace_file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    trace_file_path_obj.write_text("{}\n", encoding="utf-8")
    mtime_float = (now_ts - timedelta(days=age_days_int)).timestamp()
    os.utime(trace_file_path_obj, (mtime_float, mtime_float))
    os.utime(run_folder_path_obj, (mtime_float, mtime_float))
    os.utime(run_folder_path_obj.parent, (mtime_float, mtime_float))
    return run_folder_path_obj


def test_pod_trace_path_is_sanitized_and_event_shape_is_standard(tmp_path: Path):
    as_of_ts = datetime(2026, 6, 1, 13, 23, 30, tzinfo=UTC)
    trace_context_dict = logging_utils.build_pod_trace_context_dict(
        mode_str="live",
        pod_id_str="pod/live:bad",
        account_route_str="U1",
        release_id_str="release_001",
        as_of_timestamp_str=as_of_ts.isoformat(),
    )

    trace_path_str = logging_utils.log_pod_trace_event(
        "doctor.final_verdict",
        trace_context_dict=trace_context_dict,
        status_str="PASS",
        reason_code_str="pass",
        payload_dict={"answer_str": "ready"},
        trace_enabled_bool=True,
        trace_log_root_path_str=str(tmp_path),
    )

    trace_path_obj = Path(str(trace_path_str))
    assert trace_path_obj.exists()
    assert "pod_live_bad" in str(trace_path_obj)
    record_dict = _read_jsonl_list(trace_path_obj)[0]
    assert record_dict["event_name_str"] == "doctor.final_verdict"
    assert record_dict["mode_str"] == "live"
    assert record_dict["pod_id_str"] == "pod/live:bad"
    assert record_dict["account_route_str"] == "U1"
    assert record_dict["release_id_str"] == "release_001"
    assert record_dict["status_str"] == "PASS"
    assert record_dict["reason_code_str"] == "pass"
    assert record_dict["payload_dict"]["payload_dict"]["answer_str"] == "ready"


def test_trace_secret_redaction_is_recursive(tmp_path: Path):
    trace_context_dict = logging_utils.build_pod_trace_context_dict(
        mode_str="paper",
        pod_id_str="pod_01",
        as_of_timestamp_str="2026-06-01T13:23:30+00:00",
    )

    trace_path_str = logging_utils.log_pod_trace_event(
        "config.loaded",
        trace_context_dict=trace_context_dict,
        status_str="PASS",
        reason_code_str="config_loaded",
        payload_dict={
            "NORGATE_API_TOKEN": "secret-token",
            "nested_dict": {"password_str": "secret-password"},
            "broker_message_str": "Authorization failed bearer abc123 and token=xyz",
            "order_request_key_str": "visible-correlation-key",
        },
        trace_enabled_bool=True,
        trace_log_root_path_str=str(tmp_path),
    )

    record_dict = _read_jsonl_list(Path(str(trace_path_str)))[0]
    payload_dict = record_dict["payload_dict"]["payload_dict"]
    assert payload_dict["NORGATE_API_TOKEN"] == logging_utils.TRACE_REDACTED_VALUE_STR
    assert payload_dict["nested_dict"]["password_str"] == logging_utils.TRACE_REDACTED_VALUE_STR
    assert "abc123" not in payload_dict["broker_message_str"]
    assert "token=xyz" not in payload_dict["broker_message_str"]
    assert payload_dict["order_request_key_str"] == "visible-correlation-key"


def test_trace_logging_failure_does_not_raise(monkeypatch, tmp_path: Path):
    def fail_write(*args, **kwargs):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(logging_utils, "_write_line_to_rolling_log", fail_write)
    trace_context_dict = logging_utils.build_pod_trace_context_dict(
        mode_str="live",
        pod_id_str="pod_01",
        as_of_timestamp_str="2026-06-01T13:23:30+00:00",
    )

    trace_path_str = logging_utils.log_pod_trace_event(
        "vplan.submit_result",
        trace_context_dict=trace_context_dict,
        status_str="BLOCK",
        reason_code_str="missing_broker_response_ack",
        payload_dict={},
        trace_enabled_bool=True,
        trace_log_root_path_str=str(tmp_path),
    )

    assert trace_path_str is None


def test_trace_vplan_order_intent_payload_failure_is_contained(monkeypatch):
    from alpha.live.models import VPlan
    import alpha.live.runner as runner_module

    def fail_build_requests(_vplan_obj):
        raise RuntimeError("request builder failed")

    monkeypatch.setattr(
        runner_module,
        "build_broker_order_request_list_from_vplan",
        fail_build_requests,
    )
    vplan_obj = VPlan(
        release_id_str="release_001",
        user_id_str="user_001",
        pod_id_str="pod_01",
        account_route_str="DU1",
        decision_plan_id_int=1,
        signal_timestamp_ts=datetime(2026, 5, 29, 20, 0, tzinfo=UTC),
        submission_timestamp_ts=datetime(2026, 6, 1, 13, 23, 30, tzinfo=UTC),
        target_execution_timestamp_ts=datetime(2026, 6, 1, 13, 30, tzinfo=UTC),
        execution_policy_str="next_open_moo",
        broker_snapshot_timestamp_ts=datetime(2026, 6, 1, 13, 20, tzinfo=UTC),
        live_reference_snapshot_timestamp_ts=datetime(2026, 6, 1, 13, 20, tzinfo=UTC),
        live_price_source_str="test",
        net_liq_float=100000.0,
        available_funds_float=100000.0,
        excess_liquidity_float=100000.0,
        pod_budget_fraction_float=0.5,
        pod_budget_float=50000.0,
        current_broker_position_map={},
        live_reference_price_map={"AAPL": 100.0},
        target_share_map={"AAPL": 10.0},
        order_delta_map={"AAPL": 10.0},
        vplan_row_list=[],
    )

    payload_dict = runner_module._vplan_order_intent_trace_payload_dict(vplan_obj)

    assert payload_dict["broker_order_request_count_int"] == 0
    assert payload_dict["broker_order_request_error_str"] == "request builder failed"


def test_trace_retention_deletes_old_run_folders_and_keeps_recent_and_active(tmp_path: Path):
    now_ts = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    trace_root_path_obj = tmp_path / "pods"
    old_run_folder_path_obj = _make_trace_run_folder_path_obj(
        trace_root_path_obj,
        pod_id_str="pod_01",
        run_id_str="old_run",
        now_ts=now_ts,
        age_days_int=91,
    )
    recent_run_folder_path_obj = _make_trace_run_folder_path_obj(
        trace_root_path_obj,
        pod_id_str="pod_01",
        run_id_str="recent_run",
        now_ts=now_ts,
        age_days_int=30,
    )
    active_run_folder_path_obj = _make_trace_run_folder_path_obj(
        trace_root_path_obj,
        pod_id_str="pod_01",
        run_id_str="active_old_run",
        now_ts=now_ts,
        age_days_int=120,
    )

    cleanup_detail_dict = logging_utils.cleanup_pod_trace_retention_dict(
        trace_log_root_path_str=str(trace_root_path_obj),
        active_trace_log_path_str=str(active_run_folder_path_obj / "trace_events.jsonl"),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )

    assert cleanup_detail_dict["deleted_run_folder_count_int"] == 1
    assert cleanup_detail_dict["skipped_active_run_folder_count_int"] == 1
    assert not old_run_folder_path_obj.exists()
    assert recent_run_folder_path_obj.exists()
    assert active_run_folder_path_obj.exists()


def test_trace_retention_skips_symlink_run_folder(tmp_path: Path):
    now_ts = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    trace_root_path_obj = tmp_path / "pods"
    pod_folder_path_obj = trace_root_path_obj / "pod_01"
    outside_folder_path_obj = tmp_path / "outside"
    pod_folder_path_obj.mkdir(parents=True)
    outside_folder_path_obj.mkdir()
    symlink_run_folder_path_obj = pod_folder_path_obj / "symlink_run"
    try:
        symlink_run_folder_path_obj.symlink_to(
            outside_folder_path_obj,
            target_is_directory=True,
        )
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable in this Windows environment")

    cleanup_detail_dict = logging_utils.cleanup_pod_trace_retention_dict(
        trace_log_root_path_str=str(trace_root_path_obj),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )

    assert cleanup_detail_dict["skipped_symlink_count_int"] >= 1
    assert symlink_run_folder_path_obj.exists()
    assert outside_folder_path_obj.exists()


def test_trace_retention_rejects_non_default_root_without_override(tmp_path: Path):
    now_ts = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    trace_root_path_obj = tmp_path / "not_default_pods"
    old_run_folder_path_obj = _make_trace_run_folder_path_obj(
        trace_root_path_obj,
        pod_id_str="pod_01",
        run_id_str="old_run",
        now_ts=now_ts,
        age_days_int=91,
    )

    cleanup_detail_dict = logging_utils.cleanup_pod_trace_retention_dict(
        trace_log_root_path_str=str(trace_root_path_obj),
        now_ts=now_ts,
    )

    assert cleanup_detail_dict["cleanup_skip_reason_str"] == "non_default_trace_root"
    assert cleanup_detail_dict["skipped_unsafe_path_count_int"] == 1
    assert old_run_folder_path_obj.exists()


def test_trace_retention_skips_resolved_run_folder_outside_root(monkeypatch, tmp_path: Path):
    now_ts = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    trace_root_path_obj = tmp_path / "pods"
    outside_folder_path_obj = tmp_path / "outside"
    outside_folder_path_obj.mkdir()
    run_folder_path_obj = _make_trace_run_folder_path_obj(
        trace_root_path_obj,
        pod_id_str="pod_01",
        run_id_str="old_run",
        now_ts=now_ts,
        age_days_int=91,
    )
    original_safe_resolved_path_fn = logging_utils._safe_resolved_path_obj

    def fake_safe_resolved_path_obj(path_obj: Path):
        if path_obj == run_folder_path_obj:
            return outside_folder_path_obj.resolve()
        return original_safe_resolved_path_fn(path_obj)

    monkeypatch.setattr(
        logging_utils,
        "_safe_resolved_path_obj",
        fake_safe_resolved_path_obj,
    )

    cleanup_detail_dict = logging_utils.cleanup_pod_trace_retention_dict(
        trace_log_root_path_str=str(trace_root_path_obj),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )

    assert cleanup_detail_dict["skipped_unsafe_path_count_int"] == 1
    assert run_folder_path_obj.exists()
    assert outside_folder_path_obj.exists()


def test_trace_retention_skips_old_folder_without_trace_marker(tmp_path: Path):
    now_ts = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    trace_root_path_obj = tmp_path / "pods"
    non_trace_run_folder_path_obj = trace_root_path_obj / "pod_01" / "not_a_trace_run"
    non_trace_run_folder_path_obj.mkdir(parents=True)
    random_file_path_obj = non_trace_run_folder_path_obj / "random.txt"
    random_file_path_obj.write_text("not trace data", encoding="utf-8")
    mtime_float = (now_ts - timedelta(days=120)).timestamp()
    os.utime(random_file_path_obj, (mtime_float, mtime_float))
    os.utime(non_trace_run_folder_path_obj, (mtime_float, mtime_float))

    cleanup_detail_dict = logging_utils.cleanup_pod_trace_retention_dict(
        trace_log_root_path_str=str(trace_root_path_obj),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )

    assert cleanup_detail_dict["skipped_non_trace_folder_count_int"] == 1
    assert non_trace_run_folder_path_obj.exists()


def test_trace_retention_failure_does_not_block_trace_write(monkeypatch, tmp_path: Path):
    logging_utils._POD_TRACE_RETENTION_CLEANUP_DATE_BY_ROOT_DICT.clear()

    def fail_cleanup(**kwargs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(logging_utils, "cleanup_pod_trace_retention_dict", fail_cleanup)
    trace_context_dict = logging_utils.build_pod_trace_context_dict(
        mode_str="live",
        pod_id_str="pod_01",
        as_of_timestamp_str="2026-06-01T13:23:30+00:00",
    )

    trace_path_str = logging_utils.log_pod_trace_event(
        "vplan.submit_result",
        trace_context_dict=trace_context_dict,
        status_str="PASS",
        reason_code_str="submitted",
        payload_dict={},
        trace_enabled_bool=True,
        trace_log_root_path_str=str(tmp_path),
        allow_non_default_trace_root_bool=True,
    )

    assert trace_path_str is not None
    assert Path(trace_path_str).exists()


def test_trace_retention_cleanup_is_throttled(monkeypatch, tmp_path: Path):
    logging_utils._POD_TRACE_RETENTION_CLEANUP_DATE_BY_ROOT_DICT.clear()
    cleanup_call_list = []

    def fake_cleanup(**kwargs):
        cleanup_call_list.append(kwargs)
        return {"cleanup_attempted_bool": True}

    monkeypatch.setattr(logging_utils, "cleanup_pod_trace_retention_dict", fake_cleanup)
    trace_context_dict = logging_utils.build_pod_trace_context_dict(
        mode_str="paper",
        pod_id_str="pod_01",
        as_of_timestamp_str="2026-06-01T13:23:30+00:00",
    )

    for _ in range(2):
        logging_utils.log_pod_trace_event(
            "scheduler.decision",
            trace_context_dict=trace_context_dict,
            status_str="WAIT",
            reason_code_str="not_due",
            payload_dict={},
            trace_enabled_bool=True,
            trace_log_root_path_str=str(tmp_path),
            allow_non_default_trace_root_bool=True,
        )

    assert len(cleanup_call_list) == 1


def test_trace_retention_cleanup_throttle_is_per_root_and_day(monkeypatch, tmp_path: Path):
    logging_utils._POD_TRACE_RETENTION_CLEANUP_DATE_BY_ROOT_DICT.clear()
    cleanup_call_list = []
    now_ts = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)

    def fake_cleanup(**kwargs):
        cleanup_call_list.append(kwargs)
        return {"cleanup_attempted_bool": True, "error_count_int": 0}

    monkeypatch.setattr(logging_utils, "cleanup_pod_trace_retention_dict", fake_cleanup)
    root_a_path_obj = tmp_path / "root_a"
    root_b_path_obj = tmp_path / "root_b"

    logging_utils.cleanup_pod_trace_retention_if_due_dict(
        trace_log_root_path_str=str(root_a_path_obj),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )
    logging_utils.cleanup_pod_trace_retention_if_due_dict(
        trace_log_root_path_str=str(root_a_path_obj),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )
    logging_utils.cleanup_pod_trace_retention_if_due_dict(
        trace_log_root_path_str=str(root_b_path_obj),
        now_ts=now_ts,
        allow_non_default_trace_root_bool=True,
    )
    logging_utils.cleanup_pod_trace_retention_if_due_dict(
        trace_log_root_path_str=str(root_a_path_obj),
        now_ts=now_ts + timedelta(days=1),
        allow_non_default_trace_root_bool=True,
    )

    assert len(cleanup_call_list) == 3
    assert cleanup_call_list[0]["trace_log_root_path_str"] == str(root_a_path_obj)
    assert cleanup_call_list[1]["trace_log_root_path_str"] == str(root_b_path_obj)
    assert cleanup_call_list[2]["trace_log_root_path_str"] == str(root_a_path_obj)
