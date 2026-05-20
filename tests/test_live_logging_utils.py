from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alpha.live.logging_utils import (
    build_structured_event_record_dict,
    log_event,
    log_trace_event,
    render_operator_message_str,
    resolve_pod_run_trace_log_path_str,
    resolve_operator_log_path_str,
)


def test_render_operator_message_waiting_uses_utc_without_milliseconds():
    message_str = render_operator_message_str(
        level_str="INFO",
        phase_action_str="cycle.wait",
        timestamp_obj=datetime(2024, 2, 1, 14, 23, 30, 987654, tzinfo=UTC),
        field_map_dict={
            "pod": "pod_test_01",
            "account": "DU1",
            "reason": "waiting for the submission window to open",
            "next": "2024-02-01 14:23:30 UTC",
        },
    )

    assert (
        message_str
        == "[2024-02-01 14:23:30 UTC] INFO cycle.wait pod=pod_test_01 account=DU1 "
        "reason=waiting for the submission window to open next=2024-02-01 14:23:30 UTC"
    )


def test_render_operator_message_build_vplan_ok_is_compact():
    message_str = render_operator_message_str(
        level_str="INFO",
        phase_action_str="build_vplan.ok",
        timestamp_obj=datetime(2024, 2, 1, 14, 23, 30, tzinfo=UTC),
        field_map_dict={
            "pod": "pod_test_01",
            "account": "DU1",
            "plan_id": 7,
            "vplan": 3,
            "asset_count": 2,
            "budget": "5000.00",
        },
    )

    assert (
        message_str
        == "[2024-02-01 14:23:30 UTC] INFO build_vplan.ok pod=pod_test_01 account=DU1 "
        "plan_id=7 vplan=3 asset_count=2 budget=5000.00"
    )


def test_render_operator_message_reconcile_ok_is_compact():
    message_str = render_operator_message_str(
        level_str="INFO",
        phase_action_str="reconcile.ok",
        timestamp_obj=datetime(2024, 2, 1, 14, 35, 0, tzinfo=UTC),
        field_map_dict={
            "pod": "pod_test_01",
            "account": "DU1",
            "vplan": 3,
            "fills": 2,
            "residuals": 0,
        },
    )

    assert (
        message_str
        == "[2024-02-01 14:35:00 UTC] INFO reconcile.ok pod=pod_test_01 account=DU1 "
        "vplan=3 fills=2 residuals=0"
    )


def test_log_event_keeps_json_audit_shape_unchanged(tmp_path: Path):
    audit_log_path_obj = tmp_path / "audit_events.jsonl"

    log_event(
        "build_vplan_created",
        {"pod_id_str": "pod_test_01", "reason_code_str": "vplan_ready"},
        log_path_str=str(audit_log_path_obj),
    )

    log_record_dict = json.loads(audit_log_path_obj.read_text(encoding="utf-8").strip())

    assert log_record_dict["event_name_str"] == "build_vplan_created"
    assert log_record_dict["pod_id_str"] == "pod_test_01"
    assert log_record_dict["reason_code_str"] == "vplan_ready"
    assert "event_timestamp_str" in log_record_dict
    assert log_record_dict["ts_utc"] == log_record_dict["event_timestamp_str"]
    assert log_record_dict["level_str"] == "INFO"
    assert log_record_dict["payload_dict"]["pod_id_str"] == "pod_test_01"


def test_resolve_operator_log_path_for_custom_audit_file(tmp_path: Path):
    audit_log_path_obj = tmp_path / "custom_events.jsonl"

    resolved_operator_log_path_str = resolve_operator_log_path_str(str(audit_log_path_obj))

    assert resolved_operator_log_path_str.endswith("custom_events_operator.log")


def test_structured_event_record_adds_common_correlation_fields():
    event_record_dict = build_structured_event_record_dict(
        "submit_vplan_completed",
        {
            "severity_str": "critical",
            "mode_str": "paper",
            "pod_id_str": "pod_test_01",
            "account_route_str": "DU1",
            "run_id_str": "run_001",
            "decision_plan_id_int": 6,
            "vplan_id_int": 5,
            "asset_str": "AAPL",
        },
    )

    assert event_record_dict["level_str"] == "CRITICAL"
    assert event_record_dict["mode_str"] == "paper"
    assert event_record_dict["pod_id_str"] == "pod_test_01"
    assert event_record_dict["account_id_str"] == "DU1"
    assert event_record_dict["run_id_str"] == "run_001"
    assert event_record_dict["decision_plan_id_int"] == 6
    assert event_record_dict["vplan_id_int"] == 5
    assert event_record_dict["asset_str"] == "AAPL"
    assert event_record_dict["payload_dict"]["account_route_str"] == "DU1"


def test_log_event_rotates_jsonl_without_breaking_active_file(tmp_path: Path):
    audit_log_path_obj = tmp_path / "audit_events.jsonl"

    log_event(
        "event_one",
        {"pod_id_str": "pod_test_01", "run_id_str": "run_001"},
        log_path_str=str(audit_log_path_obj),
        max_bytes_int=1,
        backup_count_int=1,
    )
    log_event(
        "event_two",
        {"pod_id_str": "pod_test_01", "run_id_str": "run_001"},
        log_path_str=str(audit_log_path_obj),
        max_bytes_int=1,
        backup_count_int=1,
    )

    rotated_log_path_obj = audit_log_path_obj.with_name("audit_events.jsonl.1")
    assert audit_log_path_obj.exists()
    assert rotated_log_path_obj.exists()
    assert json.loads(audit_log_path_obj.read_text(encoding="utf-8").strip())["event_name_str"] == "event_two"
    assert json.loads(rotated_log_path_obj.read_text(encoding="utf-8").strip())["event_name_str"] == "event_one"


def test_log_trace_event_is_opt_in_and_uses_pod_run_layout(tmp_path: Path):
    trace_root_path_obj = tmp_path / "pods"
    disabled_path_str = log_trace_event(
        "engine.bar.completed",
        {"pod_id_str": "pod/test", "run_id_str": "run:001"},
        trace_enabled_bool=False,
        trace_log_root_path_str=str(trace_root_path_obj),
    )

    assert disabled_path_str is None
    assert not trace_root_path_obj.exists()

    enabled_path_str = log_trace_event(
        "engine.bar.completed",
        {"pod_id_str": "pod/test", "run_id_str": "run:001"},
        trace_enabled_bool=True,
        trace_log_root_path_str=str(trace_root_path_obj),
    )

    expected_trace_path_str = resolve_pod_run_trace_log_path_str(
        pod_id_str="pod/test",
        run_id_str="run:001",
        trace_log_root_path_str=str(trace_root_path_obj),
    )
    assert enabled_path_str == expected_trace_path_str
    trace_record_dict = json.loads(Path(enabled_path_str).read_text(encoding="utf-8").strip())
    assert trace_record_dict["event_name_str"] == "engine.bar.completed"
