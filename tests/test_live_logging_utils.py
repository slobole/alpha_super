from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alpha.live.logging_utils import (
    log_event,
    render_operator_message_str,
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


def test_resolve_operator_log_path_for_custom_audit_file(tmp_path: Path):
    audit_log_path_obj = tmp_path / "custom_events.jsonl"

    resolved_operator_log_path_str = resolve_operator_log_path_str(str(audit_log_path_obj))

    assert resolved_operator_log_path_str.endswith("custom_events_operator.log")
