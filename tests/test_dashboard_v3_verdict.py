from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from alpha.live.dashboard_v3.verdict import resolve_top_bar_verdict
from alpha.live.ops_report import build_ops_report_dict


def _row_dict(
    pod_id_str: str,
    mode_str: str,
    severity_str: str,
    *,
    row_update_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_dict = {
        "pod_id_str": pod_id_str,
        "mode_str": mode_str,
        "health_str": severity_str,
        "required_action_dict": {
            "label_str": "No action",
            "severity_str": severity_str,
            "reason_str": "stub reason",
        },
        "debug_summary_dict": {
            "severity_str": severity_str,
            "primary_reason_str": "stub primary reason",
        },
    }
    if row_update_dict is not None:
        row_dict.update(row_update_dict)
    return row_dict


def test_mode_top_bar_ignores_other_mode_inspector_red() -> None:
    generated_at_ts = datetime(2026, 5, 21, 16, 0, tzinfo=UTC)
    summary_dict = {
        "as_of_timestamp_str": generated_at_ts.isoformat(),
        "pod_row_dict_list": [
            _row_dict("live_pod", "live", "green"),
            _row_dict(
                "incubation_pod",
                "incubation",
                "yellow",
                row_update_dict={
                    "next_action_str": "expire_stale",
                    "reason_code_str": "submission_window_expired",
                },
            ),
        ],
    }
    summary_dict["inspector_report_dict"] = build_ops_report_dict(
        summary_dict,
        generated_at_ts=generated_at_ts,
        stale_after_seconds_int=999999999,
        vps_id_str="vps_01",
    )

    live_verdict_obj = resolve_top_bar_verdict(summary_dict, mode_str="live")
    global_verdict_obj = resolve_top_bar_verdict(summary_dict)

    assert live_verdict_obj.severity_str == "green"
    assert live_verdict_obj.title_str == "All clear"
    assert global_verdict_obj.severity_str == "red"
    assert global_verdict_obj.title_str == "Inspector flag"


def test_mode_top_bar_ignores_other_mode_red_pod_without_inspector_report() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("live_pod", "live", "green"),
            _row_dict("paper_pod", "paper", "red"),
        ],
    }

    live_verdict_obj = resolve_top_bar_verdict(summary_dict, mode_str="live")
    global_verdict_obj = resolve_top_bar_verdict(summary_dict)

    assert live_verdict_obj.severity_str == "green"
    assert global_verdict_obj.severity_str == "red"
