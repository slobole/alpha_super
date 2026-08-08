"""Unit tests for ``alpha.live.dashboard_v3.schedule``."""

from __future__ import annotations

from datetime import datetime, timezone

from alpha.live.dashboard_v3.schedule import (
    build_next_trading_window,
    build_schedule_entry_list,
)


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


def test_schedule_can_be_scoped_to_live_mode() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _row_dict("pod_live"),
            {**_row_dict("pod_incubation"), "mode_str": "incubation"},
        ]
    }
    schedule_entry_obj_list = build_schedule_entry_list(
        summary_dict,
        now_dt=REFERENCE_NOW_DT,
        mode_str="live",
    )
    assert [entry_obj.pod_id_str for entry_obj in schedule_entry_obj_list] == [
        "pod_live"
    ]


def test_next_trading_window_surfaces_month_end_while_scheduler_waits() -> None:
    now_dt = datetime(2026, 8, 8, 10, 0, 0, tzinfo=timezone.utc)
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "not_month_end_session",
            },
            {
                "pod_id_str": "pod_ndx_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "not_month_end_session",
            },
            {
                "pod_id_str": "pod_incubation",
                "mode_str": "incubation",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "build_decision_plan",
                "reason_code_str": "ready_to_build_decision_plan",
            },
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=now_dt,
    )

    assert window_obj.has_data_bool is True
    assert window_obj.action_required_bool is False
    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.submission_timestamp_str == "2026-09-01T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"
    assert window_obj.trading_session_count_int == 16
    assert window_obj.norgate_label_str == "Required for 2026-08-31"
    assert window_obj.pod_id_str_list == ["pod_ndx_live", "pod_taa_live"]


def test_next_trading_window_prefers_current_required_action() -> None:
    now_dt = datetime(2026, 8, 31, 20, 30, 0, tzinfo=timezone.utc)
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "build_decision_plan",
                "reason_code_str": "ready_to_build_decision_plan",
                "health_str": "yellow",
                "required_action_dict": {
                    "severity_str": "yellow",
                    "detail_str": "DecisionPlan is due for the month-end signal.",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=now_dt,
    )

    assert window_obj.action_required_bool is True
    assert window_obj.action_str == "build_decision_plan"
    assert window_obj.status_label_str == "Operator action required"
    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.submission_timestamp_str == "2026-09-01T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"


def test_active_window_does_not_mix_previous_vplan_with_current_decision() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "next_action_str": "build_vplan",
                "latest_decision_signal_timestamp_str": "2026-08-31T16:00:00-04:00",
                "latest_decision_plan_submission_timestamp_str": "2026-09-01T09:23:30-04:00",
                "latest_decision_plan_target_execution_timestamp_str": "2026-09-01T09:30:00-04:00",
                "latest_vplan_submission_timestamp_str": "2026-08-03T09:23:30-04:00",
                "latest_vplan_target_execution_timestamp_str": "2026-08-03T09:30:00-04:00",
                "latest_vplan_cycle_role_str": "previous",
                "latest_vplan_is_for_latest_decision_bool": False,
                "required_action_dict": {
                    "label_str": "Build VPlan",
                    "severity_str": "yellow",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 31, 20, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.submission_timestamp_str == "2026-09-01T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"


def test_build_vplan_without_vplan_uses_decision_timestamps() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "next_action_str": "build_vplan",
                "latest_decision_signal_timestamp_str": "2026-08-31T16:00:00-04:00",
                "latest_decision_plan_submission_timestamp_str": "2026-09-01T09:23:30-04:00",
                "latest_decision_plan_target_execution_timestamp_str": "2026-09-01T09:30:00-04:00",
                "latest_vplan_cycle_role_str": "none",
                "latest_vplan_is_for_latest_decision_bool": None,
                "required_action_dict": {
                    "label_str": "Build VPlan",
                    "severity_str": "yellow",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 31, 20, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.submission_timestamp_str == "2026-09-01T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"


def test_next_trading_window_keeps_current_cycle_while_waiting_for_norgate() -> None:
    now_dt = datetime(2026, 8, 31, 20, 20, 0, tzinfo=timezone.utc)
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "norgate_snapshot_sync_waiting",
                "required_action_dict": {
                    "label_str": "Wait Norgate data",
                    "severity_str": "yellow",
                    "detail_str": "Waiting inside the normal Norgate publication window.",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=now_dt,
    )

    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"
    assert window_obj.status_label_str == "Wait Norgate data"
    assert window_obj.severity_str == "yellow"
    assert window_obj.action_required_bool is False


def test_next_trading_window_uses_just_closed_month_end_before_ready_buffer() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "snapshot_not_ready",
                "required_action_dict": {
                    "label_str": "Wait Norgate data",
                    "severity_str": "yellow",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 31, 20, 1, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"


def test_just_closed_month_end_does_not_reuse_stale_no_action_status() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "not_month_end_session",
                "required_action_dict": {
                    "label_str": "No action",
                    "severity_str": "green",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 31, 20, 1, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.status_label_str == "Awaiting scheduler refresh"
    assert window_obj.severity_str == "gray"
    assert window_obj.action_required_bool is False


def test_next_trading_window_respects_new_year_market_holiday() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "not_month_end_session",
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 12, 15, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-12-31T16:00:00-05:00"
    assert window_obj.submission_timestamp_str == "2027-01-04T09:23:30-05:00"
    assert window_obj.target_timestamp_str == "2027-01-04T09:30:00-05:00"


def test_stale_midmonth_norgate_wait_does_not_reopen_expired_cycle() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "snapshot_not_ready_for_session",
                "required_action_dict": {
                    "label_str": "Review Norgate data",
                    "severity_str": "red",
                },
            }
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 8, 10, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-08-31T16:00:00-04:00"
    assert window_obj.target_timestamp_str == "2026-09-01T09:30:00-04:00"


def test_future_window_session_count_excludes_closed_current_session() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live",
                "mode_str": "live",
                "signal_clock_str": "month_end_snapshot_ready",
                "session_calendar_id_str": "XNYS",
                "execution_policy_str": "next_month_first_open",
                "next_action_str": "wait",
                "reason_code_str": "not_month_end_session",
            }
        ]
    }

    before_close_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 3, 19, 0, 0, tzinfo=timezone.utc),
    )
    after_close_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 3, 21, 0, 0, tzinfo=timezone.utc),
    )

    assert before_close_obj.trading_session_count_int == 21
    assert after_close_obj.trading_session_count_int == 20


def test_next_trading_window_surfaces_worst_pod_for_shared_cycle() -> None:
    now_dt = datetime(2026, 8, 31, 20, 20, 0, tzinfo=timezone.utc)
    common_row_dict = {
        "mode_str": "live",
        "signal_clock_str": "month_end_snapshot_ready",
        "session_calendar_id_str": "XNYS",
        "execution_policy_str": "next_month_first_open",
        "next_action_str": "wait",
        "reason_code_str": "norgate_snapshot_sync_waiting",
    }
    summary_dict = {
        "pod_row_dict_list": [
            {
                **common_row_dict,
                "pod_id_str": "pod_green",
                "required_action_dict": {
                    "label_str": "No action",
                    "severity_str": "green",
                },
            },
            {
                **common_row_dict,
                "pod_id_str": "pod_red",
                "required_action_dict": {
                    "label_str": "Review Norgate data",
                    "severity_str": "red",
                    "detail_str": "Norgate is beyond the publication grace period.",
                },
            },
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=now_dt,
    )

    assert window_obj.severity_str == "red"
    assert window_obj.status_label_str == "Review Norgate data"
    assert window_obj.action_required_bool is True
    assert "beyond the publication grace period" in window_obj.detail_str
    assert window_obj.pod_id_str_list == ["pod_green", "pod_red"]


def test_required_action_without_target_beats_waiting_target() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            {
                **_row_dict(
                    "pod_waiting",
                    next_action_str="wait",
                    target_timestamp_str="2026-09-01T09:30:00-04:00",
                ),
                "required_action_dict": {
                    "label_str": "Waiting for ACKs",
                    "severity_str": "yellow",
                },
            },
            {
                **_row_dict(
                    "pod_action",
                    next_action_str="build_decision_plan",
                    target_timestamp_str=None,
                ),
                "required_action_dict": {
                    "label_str": "Build DecisionPlan",
                    "severity_str": "yellow",
                },
            },
        ]
    }

    window_obj = build_next_trading_window(
        summary_dict,
        mode_str="live",
        now_dt=datetime(2026, 8, 31, 20, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.status_label_str == "Build DecisionPlan"
    assert window_obj.action_str == "build_decision_plan"
    assert window_obj.action_required_bool is True


def test_future_target_without_stage_evidence_is_not_on_schedule() -> None:
    window_obj = build_next_trading_window(
        {
            "pod_row_dict_list": [
                _row_dict(
                    "pod_partial",
                    next_action_str="wait",
                    target_timestamp_str="2026-09-01T09:30:00-04:00",
                )
            ]
        },
        mode_str="live",
        now_dt=datetime(2026, 8, 31, 20, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.severity_str == "gray"
    assert window_obj.status_label_str == "Cannot verify"
    assert "missing signal or submission evidence" in window_obj.detail_str


def test_next_trading_window_is_unknown_without_live_pods() -> None:
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [{"pod_id_str": "paper", "mode_str": "paper"}]},
        mode_str="live",
        now_dt=REFERENCE_NOW_DT,
    )
    assert window_obj.has_data_bool is False
    assert window_obj.status_label_str == "Cannot verify"


def _daily_pod_row_dict(pod_id_str: str = "pod_dv2_live") -> dict:
    return {
        "pod_id_str": pod_id_str,
        "mode_str": "live",
        "signal_clock_str": "eod_snapshot_ready",
        "session_calendar_id_str": "XNYS",
        "execution_policy_str": "next_open_moo",
        "next_action_str": "wait",
        "reason_code_str": "awaiting_eod_cycle",
    }


def test_daily_window_midsession_targets_tomorrows_open() -> None:
    # Wednesday 2026-08-05 14:00 UTC = 10:00 ET, mid-session. The coming
    # signal is today's close; execution is Thursday's open MOO.
    now_dt = datetime(2026, 8, 5, 14, 0, 0, tzinfo=timezone.utc)
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [_daily_pod_row_dict()]},
        mode_str="live",
        now_dt=now_dt,
    )
    assert window_obj.has_data_bool is True
    assert window_obj.action_required_bool is False
    assert window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
    assert window_obj.submission_timestamp_str == "2026-08-06T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-08-06T09:30:00-04:00"
    assert window_obj.norgate_label_str == "Required for 2026-08-05"
    assert window_obj.pod_id_str_list == ["pod_dv2_live"]


def test_daily_window_after_close_keeps_open_cycle() -> None:
    # Wednesday 21:00 UTC = 17:00 ET, after the close: the cycle whose
    # signal was today's close is still pending until Thursday's open.
    now_dt = datetime(2026, 8, 5, 21, 0, 0, tzinfo=timezone.utc)
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [_daily_pod_row_dict()]},
        mode_str="live",
        now_dt=now_dt,
    )
    assert window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
    assert window_obj.target_timestamp_str == "2026-08-06T09:30:00-04:00"
    assert window_obj.trading_session_count_int == 0
    assert window_obj.status_label_str == "Awaiting scheduler refresh"
    assert "scheduler" in window_obj.detail_str.lower()


def test_daily_build_decision_plan_derives_current_cycle_timestamps() -> None:
    daily_row_dict = {
        **_daily_pod_row_dict(),
        "next_action_str": "build_decision_plan",
        "reason_code_str": "ready_to_build_decision_plan",
        "required_action_dict": {
            "label_str": "Build DecisionPlan",
            "severity_str": "yellow",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 5, 21, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.action_str == "build_decision_plan"
    assert window_obj.action_required_bool is True
    assert window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
    assert window_obj.submission_timestamp_str == "2026-08-06T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-08-06T09:30:00-04:00"


def test_daily_malformed_persisted_target_fails_loud() -> None:
    daily_row_dict = {
        **_daily_pod_row_dict(),
        "next_action_str": "build_decision_plan",
        "reason_code_str": "ready_to_build_decision_plan",
        "latest_decision_plan_target_execution_timestamp_str": "not-a-timestamp",
        "required_action_dict": {
            "label_str": "Build DecisionPlan",
            "severity_str": "yellow",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 10, 13, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.has_data_bool is False
    assert window_obj.severity_str == "red"
    assert window_obj.status_label_str == "Cannot verify"
    assert window_obj.pod_id_str_list == ["pod_dv2_live"]
    assert "Persisted target timestamp is invalid" in window_obj.detail_str


def test_daily_persisted_cycle_expires_at_exact_moo_open() -> None:
    daily_row_dict = {
        **_daily_pod_row_dict(),
        "latest_decision_signal_timestamp_str": "2026-08-07T16:00:00-04:00",
        "latest_decision_plan_submission_timestamp_str": "2026-08-10T09:23:30-04:00",
        "latest_decision_plan_target_execution_timestamp_str": "2026-08-10T09:30:00-04:00",
        "required_action_dict": {
            "label_str": "Waiting for MOO execution",
            "severity_str": "yellow",
        },
    }

    before_open_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 10, 13, 29, 59, tzinfo=timezone.utc),
    )
    at_open_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 10, 13, 30, 0, tzinfo=timezone.utc),
    )

    assert before_open_window_obj.signal_timestamp_str == "2026-08-07T16:00:00-04:00"
    assert before_open_window_obj.target_timestamp_str == "2026-08-10T09:30:00-04:00"
    assert before_open_window_obj.status_label_str == "Waiting for MOO execution"
    assert at_open_window_obj.signal_timestamp_str == "2026-08-10T16:00:00-04:00"
    assert at_open_window_obj.submission_timestamp_str == "2026-08-11T09:23:30-04:00"
    assert at_open_window_obj.target_timestamp_str == "2026-08-11T09:30:00-04:00"
    assert at_open_window_obj.status_label_str == "Awaiting scheduler refresh"
    assert at_open_window_obj.action_required_bool is False


def test_daily_missing_decision_becomes_missed_at_submission_deadline() -> None:
    pending_daily_row_dict = {
        **_daily_pod_row_dict(),
        "db_status_str": "ok",
        "latest_pod_state_timestamp_str": "2026-08-05T15:00:00-04:00",
        "reason_code_str": "carry_forward_snapshot_ready",
        "required_action_dict": {
            "label_str": "No action",
            "severity_str": "green",
        },
    }
    expired_daily_row_dict = {
        **_daily_pod_row_dict(),
        "db_status_str": "ok",
        "latest_pod_state_timestamp_str": "2026-08-05T15:00:00-04:00",
        "reason_code_str": "snapshot_window_expired",
        "required_action_dict": {
            "label_str": "No action",
            "severity_str": "green",
        },
    }

    before_submission_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [pending_daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 6, 13, 23, 29, tzinfo=timezone.utc),
    )
    at_submission_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [expired_daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 6, 13, 23, 30, tzinfo=timezone.utc),
    )
    at_open_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [expired_daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 6, 13, 30, 0, tzinfo=timezone.utc),
    )

    assert before_submission_window_obj.status_label_str == "Awaiting scheduler refresh"
    for missed_window_obj in (at_submission_window_obj, at_open_window_obj):
        assert missed_window_obj.severity_str == "red"
        assert missed_window_obj.status_label_str == "Missed DecisionPlan cycle"
        assert missed_window_obj.action_required_bool is True
        assert missed_window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
        assert missed_window_obj.submission_timestamp_str == "2026-08-06T09:23:30-04:00"
        assert missed_window_obj.target_timestamp_str == "2026-08-06T09:30:00-04:00"


def test_expired_daily_build_action_is_not_attached_to_next_cycle() -> None:
    daily_row_dict = {
        **_daily_pod_row_dict(),
        "db_status_str": "ok",
        "latest_pod_state_timestamp_str": "2026-08-05T15:00:00-04:00",
        "next_action_str": "build_decision_plan",
        "reason_code_str": "ready_to_build_decision_plan",
        "latest_decision_plan_target_execution_timestamp_str": "2026-08-06T09:30:00-04:00",
        "required_action_dict": {
            "label_str": "Build DecisionPlan",
            "severity_str": "yellow",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 6, 13, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.status_label_str == "Missed DecisionPlan cycle"
    assert window_obj.action_str == "missed_decision_cycle"
    assert window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
    assert window_obj.target_timestamp_str == "2026-08-06T09:30:00-04:00"


def test_new_no_state_pod_is_not_marked_as_missed_cycle() -> None:
    new_daily_row_dict = {
        **_daily_pod_row_dict(),
        "db_status_str": "empty",
        "reason_code_str": "snapshot_window_expired",
        "required_action_dict": {
            "label_str": "No state yet",
            "severity_str": "gray",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [new_daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 6, 14, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.severity_str == "gray"
    assert window_obj.status_label_str == "No state yet"
    assert window_obj.action_required_bool is False
    assert window_obj.status_label_str != "Missed DecisionPlan cycle"


def test_next_open_market_action_expires_after_canonical_grace() -> None:
    action_row_dict = {
        **_row_dict(
            "pod_market_open",
            next_action_str="submit_vplan",
            target_timestamp_str="2026-08-10T09:30:00-04:00",
        ),
        "execution_policy_str": "next_open_market",
        "required_action_dict": {
            "label_str": "Submit VPlan",
            "severity_str": "yellow",
        },
    }

    at_grace_boundary_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [action_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 10, 13, 31, 0, tzinfo=timezone.utc),
    )
    after_grace_window_obj = build_next_trading_window(
        {"pod_row_dict_list": [action_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 10, 13, 31, 1, tzinfo=timezone.utc),
    )

    assert at_grace_boundary_window_obj.status_label_str == "Submit VPlan"
    assert at_grace_boundary_window_obj.action_required_bool is True
    assert after_grace_window_obj.status_label_str == "Cannot verify"
    assert after_grace_window_obj.action_required_bool is False


def test_post_execution_action_stays_active_after_target() -> None:
    reconcile_row_dict = {
        **_row_dict(
            "pod_reconcile",
            next_action_str="post_execution_reconcile",
            target_timestamp_str="2026-08-10T09:30:00-04:00",
        ),
        "execution_policy_str": "next_open_moo",
        "required_action_dict": {
            "label_str": "Waiting reconcile",
            "severity_str": "yellow",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [reconcile_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 10, 14, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.status_label_str == "Waiting reconcile"
    assert window_obj.action_str == "post_execution_reconcile"
    assert window_obj.target_timestamp_str == "2026-08-10T09:30:00-04:00"


def test_just_closed_daily_cycle_does_not_reuse_stale_no_action_status() -> None:
    daily_row_dict = {
        **_daily_pod_row_dict(),
        "required_action_dict": {
            "label_str": "No action",
            "severity_str": "green",
            "detail_str": "POD is idle or completed.",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 5, 20, 1, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
    assert window_obj.status_label_str == "Awaiting scheduler refresh"
    assert window_obj.severity_str == "gray"
    assert window_obj.action_required_bool is False


def test_mixed_book_fails_loudly_when_one_pod_window_cannot_be_resolved() -> None:
    unresolved_daily_row_dict = {
        **_daily_pod_row_dict("pod_missing_calendar"),
        "session_calendar_id_str": "",
    }

    window_obj = build_next_trading_window(
        {
            "pod_row_dict_list": [
                _daily_pod_row_dict("pod_valid"),
                unresolved_daily_row_dict,
            ]
        },
        mode_str="live",
        now_dt=datetime(2026, 8, 5, 14, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.has_data_bool is False
    assert window_obj.status_label_str == "Cannot verify"
    assert window_obj.pod_id_str_list == ["pod_missing_calendar"]
    assert "pod_missing_calendar" in window_obj.detail_str


def test_unresolved_calendar_does_not_hide_required_action() -> None:
    unresolved_action_row_dict = {
        **_daily_pod_row_dict("pod_missing_calendar"),
        "session_calendar_id_str": "",
        "next_action_str": "build_decision_plan",
        "reason_code_str": "ready_to_build_decision_plan",
        "required_action_dict": {
            "label_str": "Build DecisionPlan",
            "severity_str": "yellow",
        },
    }

    window_obj = build_next_trading_window(
        {
            "pod_row_dict_list": [
                _daily_pod_row_dict("pod_valid"),
                unresolved_action_row_dict,
            ]
        },
        mode_str="live",
        now_dt=datetime(2026, 8, 5, 14, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.status_label_str == "Build DecisionPlan"
    assert window_obj.severity_str == "yellow"
    assert window_obj.action_required_bool is True
    assert "pod_missing_calendar" in window_obj.detail_str


def test_unresolved_red_pod_sets_card_to_worst_severity() -> None:
    valid_action_row_dict = {
        **_daily_pod_row_dict("pod_valid"),
        "next_action_str": "build_decision_plan",
        "reason_code_str": "ready_to_build_decision_plan",
        "required_action_dict": {
            "label_str": "Build DecisionPlan",
            "severity_str": "yellow",
        },
    }
    unresolved_red_row_dict = {
        **_daily_pod_row_dict("pod_missing_calendar"),
        "session_calendar_id_str": "",
        "required_action_dict": {
            "label_str": "Review Norgate data",
            "severity_str": "red",
            "detail_str": "Norgate evidence is missing.",
        },
    }

    window_obj = build_next_trading_window(
        {
            "pod_row_dict_list": [
                valid_action_row_dict,
                unresolved_red_row_dict,
            ]
        },
        mode_str="live",
        now_dt=datetime(2026, 8, 5, 21, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.severity_str == "red"
    assert window_obj.status_label_str == "Review Norgate data"
    assert window_obj.action_required_bool is True
    assert "Norgate evidence is missing" in window_obj.detail_str
    assert "pod_missing_calendar" in window_obj.detail_str


def test_daily_shared_window_uses_worst_pod_severity() -> None:
    green_daily_row_dict = {
        **_daily_pod_row_dict("pod_green"),
        "required_action_dict": {
            "label_str": "No action",
            "severity_str": "green",
        },
    }
    red_daily_row_dict = {
        **_daily_pod_row_dict("pod_red"),
        "required_action_dict": {
            "label_str": "Review Norgate data",
            "severity_str": "red",
            "detail_str": "Norgate is beyond its publication window.",
        },
    }

    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [green_daily_row_dict, red_daily_row_dict]},
        mode_str="live",
        now_dt=datetime(2026, 8, 5, 14, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.severity_str == "red"
    assert window_obj.status_label_str == "Review Norgate data"
    assert window_obj.pod_id_str_list == ["pod_green", "pod_red"]


def test_daily_window_friday_evening_targets_monday_open() -> None:
    # Friday 2026-08-07 21:00 UTC = 17:00 ET. Signal = Friday's close,
    # execution skips the weekend to Monday's open.
    now_dt = datetime(2026, 8, 7, 21, 0, 0, tzinfo=timezone.utc)
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [_daily_pod_row_dict()]},
        mode_str="live",
        now_dt=now_dt,
    )
    assert window_obj.signal_timestamp_str == "2026-08-07T16:00:00-04:00"
    assert window_obj.target_timestamp_str == "2026-08-10T09:30:00-04:00"


def test_daily_window_respects_black_friday_early_close() -> None:
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [_daily_pod_row_dict()]},
        mode_str="live",
        now_dt=datetime(2026, 11, 27, 18, 30, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-11-27T13:00:00-05:00"
    assert window_obj.submission_timestamp_str == "2026-11-30T09:23:30-05:00"
    assert window_obj.target_timestamp_str == "2026-11-30T09:30:00-05:00"


def test_daily_window_skips_labor_day_market_holiday() -> None:
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [_daily_pod_row_dict()]},
        mode_str="live",
        now_dt=datetime(2026, 9, 4, 21, 0, 0, tzinfo=timezone.utc),
    )

    assert window_obj.signal_timestamp_str == "2026-09-04T16:00:00-04:00"
    assert window_obj.submission_timestamp_str == "2026-09-08T09:23:30-04:00"
    assert window_obj.target_timestamp_str == "2026-09-08T09:30:00-04:00"


def test_mixed_book_prefers_nearest_signal() -> None:
    # A daily pod's tonight-close signal beats the monthly pod's month-end.
    now_dt = datetime(2026, 8, 5, 14, 0, 0, tzinfo=timezone.utc)
    monthly_row_dict = {
        "pod_id_str": "pod_taa_live",
        "mode_str": "live",
        "signal_clock_str": "month_end_snapshot_ready",
        "session_calendar_id_str": "XNYS",
        "execution_policy_str": "next_month_first_open",
        "next_action_str": "wait",
        "reason_code_str": "not_month_end_session",
    }
    window_obj = build_next_trading_window(
        {"pod_row_dict_list": [monthly_row_dict, _daily_pod_row_dict()]},
        mode_str="live",
        now_dt=now_dt,
    )
    assert window_obj.signal_timestamp_str == "2026-08-05T16:00:00-04:00"
    assert window_obj.pod_id_str_list == ["pod_dv2_live"]
