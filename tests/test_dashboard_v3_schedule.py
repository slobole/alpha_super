"""Unit tests for ``alpha.live.dashboard_v3.schedule``."""

from __future__ import annotations

from datetime import datetime, timezone

from alpha.live.dashboard_v3.schedule import build_schedule_entry_list


REFERENCE_NOW_DT = datetime(2026, 5, 21, 14, 30, 0, tzinfo=timezone.utc)


def _row_dict(
    pod_id_str: str,
    next_action_str: str = "submit_vplan",
    target_timestamp_str: str | None = "2026-05-21T15:00:00+00:00",
) -> dict:
    return {
        "pod_id_str": pod_id_str,
        "mode_str": "live",
        "next_action_str": next_action_str,
        "latest_vplan_target_execution_timestamp_str": target_timestamp_str,
    }


def test_schedule_sorts_by_target_time_ascending() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("pod_b", target_timestamp_str="2026-05-21T16:00:00+00:00"),
            _row_dict("pod_a", target_timestamp_str="2026-05-21T15:00:00+00:00"),
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(summary_dict, now_dt=REFERENCE_NOW_DT)
    pod_id_str_list = [entry_obj.pod_id_str for entry_obj in schedule_entry_obj_list]
    assert pod_id_str_list == ["pod_a", "pod_b"]


def test_schedule_drops_wait_actions() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("pod_a", next_action_str="wait"),
            _row_dict("pod_b", next_action_str="submit_vplan"),
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(summary_dict, now_dt=REFERENCE_NOW_DT)
    pod_id_str_list = [entry_obj.pod_id_str for entry_obj in schedule_entry_obj_list]
    assert pod_id_str_list == ["pod_b"]


def test_schedule_relative_time_uses_minutes_for_near_future() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("pod_a", target_timestamp_str="2026-05-21T14:38:00+00:00"),
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(summary_dict, now_dt=REFERENCE_NOW_DT)
    assert schedule_entry_obj_list[0].relative_str == "in 8m"


def test_schedule_relative_time_uses_hours_for_far_future() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("pod_a", target_timestamp_str="2026-05-21T17:38:00+00:00"),
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(summary_dict, now_dt=REFERENCE_NOW_DT)
    assert schedule_entry_obj_list[0].relative_str == "in 3h 8m"


def test_schedule_marks_past_events_as_ago() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("pod_a", target_timestamp_str="2026-05-21T14:20:00+00:00"),
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(summary_dict, now_dt=REFERENCE_NOW_DT)
    assert schedule_entry_obj_list[0].relative_str.endswith("ago")


def test_schedule_respects_limit_int() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict(f"pod_{idx:02d}", target_timestamp_str=f"2026-05-21T1{idx}:00:00+00:00")
            for idx in range(10)
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(summary_dict, now_dt=REFERENCE_NOW_DT, limit_int=3)
    assert len(schedule_entry_obj_list) == 3
