from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from alpha.live.models import LiveRelease
from alpha.live.scheduler_utils import (
    build_submission_timestamp_ts,
    build_target_execution_timestamp_ts,
    is_release_due_for_build,
    select_due_release_list,
)


def make_release(
    signal_clock_str: str,
    execution_policy_str: str,
    session_calendar_id_str: str = "XNYS",
    data_profile_str: str = "norgate_eod_sp500_pit",
) -> LiveRelease:
    return LiveRelease(
        release_id_str=f"{session_calendar_id_str}.{signal_clock_str}.{execution_policy_str}",
        user_id_str="user_001",
        pod_id_str=f"pod_{session_calendar_id_str}_{signal_clock_str}",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str=session_calendar_id_str,
        signal_clock_str=signal_clock_str,
        execution_policy_str=execution_policy_str,
        data_profile_str=data_profile_str,
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )


def test_scheduler_selects_daily_and_monthly_releases_from_norgate_snapshot(monkeypatch):
    daily_release_obj = make_release("eod_snapshot_ready", "next_open_moo")
    monthly_release_obj = make_release("month_end_snapshot_ready", "next_month_first_open")

    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    as_of_timestamp_ts = datetime(2024, 1, 31, 22, 5, tzinfo=UTC)

    assert is_release_due_for_build(daily_release_obj, as_of_timestamp_ts) is True
    assert is_release_due_for_build(monthly_release_obj, as_of_timestamp_ts) is True

    due_release_list = select_due_release_list(
        [daily_release_obj, monthly_release_obj],
        as_of_timestamp_ts,
    )
    due_release_id_set = {release_obj.release_id_str for release_obj in due_release_list}

    assert daily_release_obj.release_id_str in due_release_id_set
    assert monthly_release_obj.release_id_str in due_release_id_set


def test_scheduler_uses_real_next_session_after_long_weekend():
    release_obj = make_release("eod_snapshot_ready", "next_open_moo")
    signal_date_ts = datetime(2024, 3, 28, 16, 0)

    submission_timestamp_ts = build_submission_timestamp_ts(signal_date_ts, release_obj)
    target_execution_timestamp_ts = build_target_execution_timestamp_ts(signal_date_ts, release_obj)

    assert submission_timestamp_ts.date().isoformat() == "2024-04-01"
    assert submission_timestamp_ts.hour == 9
    assert submission_timestamp_ts.minute == 20
    assert target_execution_timestamp_ts.date().isoformat() == "2024-04-01"
    assert target_execution_timestamp_ts.hour == 9
    assert target_execution_timestamp_ts.minute == 30


def test_scheduler_does_not_build_on_weekend_even_with_stale_snapshot(monkeypatch):
    release_obj = make_release("eod_snapshot_ready", "next_open_moo")
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-02-02"),
    )

    assert is_release_due_for_build(
        release_obj,
        datetime(2024, 2, 3, 14, 0, tzinfo=UTC),
    ) is False


def test_scheduler_uses_real_early_close_for_pre_close_and_moc():
    release_obj = make_release("pre_close_15m", "same_day_moc")
    signal_date_ts = datetime(2024, 11, 29, 12, 0)

    signal_timestamp_ts = build_target_execution_timestamp_ts(signal_date_ts, release_obj)
    submission_timestamp_ts = build_submission_timestamp_ts(signal_date_ts, release_obj)

    assert signal_timestamp_ts.hour == 13
    assert signal_timestamp_ts.minute == 0
    assert submission_timestamp_ts.hour == 12
    assert submission_timestamp_ts.minute == 50


def test_scheduler_respects_exchange_calendar_timezones_for_xtse_and_xasx(monkeypatch):
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    xtse_release_obj = make_release("eod_snapshot_ready", "next_open_moo", session_calendar_id_str="XTSE")
    xasx_release_obj = make_release("eod_snapshot_ready", "next_open_moo", session_calendar_id_str="XASX")

    assert is_release_due_for_build(
        xtse_release_obj,
        datetime(2024, 1, 31, 22, 10, tzinfo=UTC),
    ) is True
    assert is_release_due_for_build(
        xasx_release_obj,
        datetime(2024, 1, 31, 6, 10, tzinfo=UTC),
    ) is True

    xasx_execution_timestamp_ts = build_target_execution_timestamp_ts(
        datetime(2024, 1, 31, 16, 0),
        xasx_release_obj,
    )

    assert xasx_execution_timestamp_ts.tzinfo is not None
    assert xasx_execution_timestamp_ts.hour == 10
    assert xasx_execution_timestamp_ts.minute == 0
