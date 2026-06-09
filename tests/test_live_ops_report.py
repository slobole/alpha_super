from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from alpha.live import scheduler_utils
import alpha.live.dashboard as dashboard_module
import alpha.live.runner as runner_module
from alpha.live.ops_report import apply_consumer_staleness_dict
from alpha.live.ops_report import build_ops_report_dict
from alpha.live.ops_report import normalize_severity_str
import scripts.live_ops_heartbeat as heartbeat_module


def _summary_dict(
    *,
    as_of_ts: datetime,
    severity_str: str = "green",
    row_update_dict: dict[str, object] | None = None,
) -> dict[str, object]:
    row_dict = {
        "pod_id_str": "pod_taa_live_01",
        "mode_str": "live",
        "account_route_str": "U1",
        "strategy_import_str": "strategies.taa_df.strategy_taa_df",
        "db_status_str": "ok",
        "health_str": severity_str,
        "next_action_str": "status",
        "required_action_dict": {
            "label_str": "No action",
            "severity_str": severity_str,
            "reason_str": "POD is idle or completed.",
            "inspect_command_name_str": "status",
        },
        "debug_summary_dict": {
            "severity_str": severity_str,
            "verdict_label_str": "healthy",
            "primary_reason_str": "POD is healthy.",
        },
    }
    if row_update_dict is not None:
        row_dict.update(row_update_dict)
    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "pod_row_dict_list": [row_dict],
    }


def test_ops_report_green_normalization_is_exact_only() -> None:
    assert normalize_severity_str("green") == "green"
    assert normalize_severity_str("submission window passed") == "gray"
    assert normalize_severity_str("success") == "gray"
    assert normalize_severity_str("complete") == "gray"
    assert normalize_severity_str("healthy") == "gray"


def test_ops_report_marks_stale_source_red() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    report_dict = build_ops_report_dict(
        _summary_dict(as_of_ts=generated_at_ts - timedelta(minutes=20)),
        mode_str="live",
        generated_at_ts=generated_at_ts,
        stale_after_seconds_int=60,
        vps_id_str="vps_01",
    )

    assert report_dict["overall_severity_str"] == "red"
    assert report_dict["source_stale_bool"] is True
    assert report_dict["pod_report_dict_list"][0]["severity_str"] == "red"
    assert report_dict["pod_report_dict_list"][0]["reason_code_str"] == "inspector_source_stale"


def test_ops_report_fresh_red_pod_makes_overall_red() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    report_dict = build_ops_report_dict(
        _summary_dict(as_of_ts=generated_at_ts, severity_str="red"),
        mode_str="live",
        generated_at_ts=generated_at_ts,
    )

    assert report_dict["overall_severity_str"] == "red"
    assert report_dict["overall_reason_str"] == "1 POD(s) need operator action."
    assert report_dict["pod_report_dict_list"][0]["severity_str"] == "red"


def test_ops_report_expired_submission_window_is_red_even_if_upstream_is_yellow() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    report_dict = build_ops_report_dict(
        _summary_dict(
            as_of_ts=generated_at_ts,
            severity_str="yellow",
            row_update_dict={
                "next_action_str": "expire_stale",
                "reason_code_str": "submission_window_expired",
            },
        ),
        mode_str="live",
        generated_at_ts=generated_at_ts,
    )

    pod_report_dict = report_dict["pod_report_dict_list"][0]
    assert report_dict["overall_severity_str"] == "red"
    assert pod_report_dict["severity_str"] == "red"
    assert pod_report_dict["reason_code_str"] == "missed_execution_window"
    assert pod_report_dict["next_operator_action_dict"]["label_str"] == "Review missed window"


def test_ops_report_taa_planned_past_target_without_vplan_is_red() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    target_ts = generated_at_ts - timedelta(minutes=5)
    report_dict = build_ops_report_dict(
        _summary_dict(
            as_of_ts=generated_at_ts,
            severity_str="green",
            row_update_dict={
                "latest_decision_plan_status_str": "planned",
                "latest_decision_plan_target_execution_timestamp_str": target_ts.isoformat(),
                "latest_vplan_status_str": None,
            },
        ),
        mode_str="live",
        generated_at_ts=generated_at_ts,
    )

    pod_report_dict = report_dict["pod_report_dict_list"][0]
    assert report_dict["overall_severity_str"] == "red"
    assert pod_report_dict["severity_str"] == "red"
    assert pod_report_dict["reason_code_str"] == "missed_execution_window"


def test_ops_report_ready_vplan_past_target_is_red() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    target_ts = generated_at_ts - timedelta(minutes=5)
    report_dict = build_ops_report_dict(
        _summary_dict(
            as_of_ts=generated_at_ts,
            severity_str="green",
            row_update_dict={
                "latest_decision_plan_status_str": "vplan_ready",
                "latest_vplan_status_str": "ready",
                "latest_vplan_target_execution_timestamp_str": target_ts.isoformat(),
            },
        ),
        mode_str="live",
        generated_at_ts=generated_at_ts,
    )

    pod_report_dict = report_dict["pod_report_dict_list"][0]
    assert report_dict["overall_severity_str"] == "red"
    assert pod_report_dict["severity_str"] == "red"
    assert pod_report_dict["reason_code_str"] == "missed_execution_window"


def test_ops_report_next_open_market_uses_engine_expiry_grace() -> None:
    target_ts = datetime(2026, 6, 9, 13, 30, tzinfo=UTC)
    inside_grace_ts = target_ts + timedelta(seconds=30)
    after_grace_ts = target_ts + timedelta(
        seconds=scheduler_utils.DEFAULT_OPEN_MARKET_EXPIRY_GRACE_SECONDS_INT + 1
    )
    summary_dict = _summary_dict(
        as_of_ts=target_ts,
        severity_str="green",
        row_update_dict={
            "execution_policy_str": "next_open_market",
            "latest_decision_execution_policy_str": "next_open_market",
            "latest_decision_plan_status_str": "planned",
            "latest_decision_plan_target_execution_timestamp_str": target_ts.isoformat(),
            "latest_vplan_status_str": None,
        },
    )

    inside_report_dict = build_ops_report_dict(
        summary_dict,
        mode_str="live",
        generated_at_ts=inside_grace_ts,
    )
    after_report_dict = build_ops_report_dict(
        summary_dict,
        mode_str="live",
        generated_at_ts=after_grace_ts,
    )

    assert inside_report_dict["overall_severity_str"] == "green"
    assert inside_report_dict["pod_report_dict_list"][0]["reason_code_str"] != "missed_execution_window"
    assert after_report_dict["overall_severity_str"] == "red"
    assert after_report_dict["pod_report_dict_list"][0]["reason_code_str"] == "missed_execution_window"


def test_ops_report_past_target_with_submitted_or_completed_vplan_is_not_missed() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    target_ts = generated_at_ts - timedelta(minutes=5)
    for vplan_status_str in ("submitted", "completed"):
        report_dict = build_ops_report_dict(
            _summary_dict(
                as_of_ts=generated_at_ts,
                severity_str="green",
                row_update_dict={
                    "latest_decision_plan_status_str": "submitted",
                    "latest_decision_plan_target_execution_timestamp_str": target_ts.isoformat(),
                    "latest_vplan_status_str": vplan_status_str,
                    "latest_vplan_target_execution_timestamp_str": target_ts.isoformat(),
                },
            ),
            mode_str="live",
            generated_at_ts=generated_at_ts,
        )

        assert report_dict["overall_severity_str"] == "green"
        assert report_dict["pod_report_dict_list"][0]["reason_code_str"] != "missed_execution_window"


def test_ops_report_no_enabled_pods_is_gray_not_green() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    report_dict = build_ops_report_dict(
        {
            "as_of_timestamp_str": generated_at_ts.isoformat(),
            "pod_row_dict_list": [],
        },
        mode_str="live",
        generated_at_ts=generated_at_ts,
    )

    assert report_dict["overall_severity_str"] == "gray"
    assert report_dict["overall_reason_str"] == "No enabled PODs were found for this report scope."


def test_consumer_staleness_downgrades_old_green_report() -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
    report_dict = build_ops_report_dict(
        _summary_dict(as_of_ts=generated_at_ts),
        mode_str="live",
        generated_at_ts=generated_at_ts,
        stale_after_seconds_int=60,
    )

    consumed_report_dict = apply_consumer_staleness_dict(
        report_dict,
        consumed_at_ts=generated_at_ts + timedelta(minutes=5),
    )

    assert consumed_report_dict["consumer_stale_bool"] is True
    assert consumed_report_dict["overall_severity_str"] == "red"
    assert "stale green is not allowed" in consumed_report_dict["overall_reason_str"]


def test_runner_ops_report_json_does_not_open_state_store(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    generated_at_ts = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)

    class FakeDashboardApp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_build_dashboard_summary_dict(app_obj, as_of_ts=None):
        assert isinstance(app_obj, FakeDashboardApp)
        assert as_of_ts == generated_at_ts
        return _summary_dict(as_of_ts=generated_at_ts)

    def fail_state_store(*args, **kwargs):
        raise AssertionError("ops_report must not open LiveStateStore job path")

    monkeypatch.setattr(dashboard_module, "DashboardApp", FakeDashboardApp)
    monkeypatch.setattr(
        dashboard_module,
        "build_dashboard_summary_dict",
        fake_build_dashboard_summary_dict,
    )
    monkeypatch.setattr(runner_module, "LiveStateStore", fail_state_store)

    return_code_int = runner_module.main(
        [
            "ops_report",
            "--json",
            "--mode",
            "live",
            "--releases-root",
            str(tmp_path / "releases"),
            "--as-of-ts",
            generated_at_ts.isoformat(),
            "--vps-id",
            "vps_01",
        ]
    )

    assert return_code_int == 0
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["schema_version_str"] == "live_ops_inspector.v1"
    assert output_dict["vps_id_str"] == "vps_01"
    assert output_dict["overall_severity_str"] == "green"
    assert output_dict["pod_report_dict_list"][0]["pod_id_str"] == "pod_taa_live_01"


def test_live_ops_heartbeat_missing_url_is_disabled(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ALPHA_INSPECTOR_HEARTBEAT_URL", raising=False)

    return_code_int = heartbeat_module.main(["--json", "--vps-id", "vps_01"])

    assert return_code_int == 0
    output_dict = json.loads(capsys.readouterr().out)
    assert output_dict["status_str"] == "disabled"
    assert output_dict["reason_code_str"] == "heartbeat_url_missing"
    assert output_dict["payload_dict"]["vps_id_str"] == "vps_01"
