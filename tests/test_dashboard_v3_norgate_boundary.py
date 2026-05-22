from __future__ import annotations

from typing import Any

from alpha.live.dashboard_v3.app import create_app


def _lifecycle_step_dict_list() -> list[dict[str, Any]]:
    return [
        {
            "step_key_str": "decision",
            "label_str": "Decision",
            "status_str": "planned",
            "severity_str": "green",
        },
        {
            "step_key_str": "vplan",
            "label_str": "VPlan",
            "status_str": "waiting",
            "severity_str": "yellow",
        },
    ]


def _pod_row_dict() -> dict[str, Any]:
    return {
        "pod_id_str": "pod_gate",
        "mode_str": "paper",
        "account_route_str": "DU_gate",
        "strategy_import_str": "strategies.demo:Demo",
        "health_str": "yellow",
        "next_action_str": "build_vplan",
        "required_action_dict": {
            "label_str": "Build VPlan",
            "severity_str": "yellow",
            "reason_str": "ready_to_build_vplan",
        },
        "debug_summary_dict": {
            "severity_str": "yellow",
            "verdict_label_str": "Build VPlan",
            "primary_reason_str": "ready_to_build_vplan",
        },
        "lifecycle_step_dict_list": _lifecycle_step_dict_list(),
        "latest_decision_plan_id_int": 7,
        "latest_decision_norgate_profile_str": "norgate_eod_sp500_pit",
        "latest_decision_norgate_snapshot_date_str": "2026-05-21",
    }


def _data_freshness_dict() -> dict[str, Any]:
    return {
        "norgate_current_cycle_gate_dict": {
            "gate_enabled_bool": True,
            "gate_required_bool": False,
            "current_cycle_continuation_allowed_bool": True,
            "blocked_stage_str": "none",
            "severity_str": "green",
            "status_label_str": "Not required for current cycle",
            "detail_str": (
                "Valid current DecisionPlan exists; VPlan / submit / reconcile "
                "continue without Norgate sync."
            ),
        },
        "item_dict_list": [
            {
                "label_str": "Norgate",
                "value_str": "2026-05-21",
                "severity_str": "green",
                "detail_str": (
                    "Not required for current cycle, Valid current DecisionPlan exists; "
                    "VPlan / submit / reconcile continue without Norgate sync."
                ),
            }
        ],
    }


class _Provider:
    def __init__(self) -> None:
        self.row_dict = _pod_row_dict()

    def get_summary_dict(self) -> dict[str, Any]:
        return {
            "pod_row_dict_list": [self.row_dict],
            "alert_dict_list": [],
            "alert_summary_dict": {},
            "mode_list": ["paper"],
        }

    def get_pod_detail_dict(self, pod_id_str: str) -> dict[str, Any]:
        if pod_id_str != "pod_gate":
            raise KeyError(pod_id_str)
        return {
            "pod_row_dict": self.row_dict,
            "required_action_dict": self.row_dict["required_action_dict"],
            "lifecycle_step_dict_list": self.row_dict["lifecycle_step_dict_list"],
            "data_freshness_dict": _data_freshness_dict(),
            "eod_snapshot_dict": {},
            "rehearsal_status_dict": {},
            "debug_story_dict": {},
            "pod_pnl_dict": {"equity_point_dict_list": [], "point_count_int": 0},
            "latest_decision_plan_dict": {
                "decision_plan_id_int": 7,
                "decision_book_type_str": "incremental_entry_exit_book",
                "signal_timestamp_str": "2026-05-21T20:00:00+00:00",
                "target_execution_timestamp_str": "2026-05-22T13:30:00+00:00",
                "snapshot_metadata_dict": {
                    "norgate_data_profile_str": "norgate_eod_sp500_pit",
                    "norgate_snapshot_date_str": "2026-05-21",
                },
            },
            "latest_vplan_dict": None,
            "latest_execution_report_dict": None,
            "event_dict_list": [],
            "latest_diff_dict": {},
        }

    def get_pod_event_dict_list(
        self,
        pod_id_str: str,
        limit_int: int = 80,
    ) -> list[dict[str, Any]]:
        return []


def test_pod_detail_shows_norgate_current_cycle_gate(tmp_path) -> None:
    app_obj = create_app(
        _Provider(),
        journal_path_str=str(tmp_path / "journal.jsonl"),
        expected_pnl_path_str=str(tmp_path / "expected.json"),
        notification_state_path_str=str(tmp_path / "notifications.json"),
    )
    client_obj = app_obj.test_client()

    response_obj = client_obj.get("/fragments/pod-detail/pod_gate")

    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "DecisionPlan gate" in response_text_str
    assert "Norgate gate" in response_text_str
    assert "Not required for current cycle" in response_text_str
    assert "VPlan / submit / reconcile continue without Norgate sync" in response_text_str
