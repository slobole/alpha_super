from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live import norgate_snapshot_sync as norgate_sync_module
from alpha.live import scheduler_utils
from alpha.live.models import LiveRelease


YEAR_INT = 2026
SESSION_CALENDAR_ID_STR = "XNYS"
PROFILE_STR = "norgate_eod_sp500_pit"
MARKET_TZ = ZoneInfo("America/New_York")


def _make_release(
    signal_clock_str: str,
    execution_policy_str: str,
    release_id_str: str,
) -> LiveRelease:
    return LiveRelease(
        release_id_str=release_id_str,
        user_id_str="user_001",
        pod_id_str=f"pod_{release_id_str}",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str=SESSION_CALENDAR_ID_STR,
        signal_clock_str=signal_clock_str,
        execution_policy_str=execution_policy_str,
        data_profile_str=PROFILE_STR,
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )


def _calendar_session_list(year_int: int) -> list[pd.Timestamp]:
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(SESSION_CALENDAR_ID_STR)
    session_index = calendar_obj.sessions_in_range(
        pd.Timestamp(f"{year_int}-01-01"),
        pd.Timestamp(f"{year_int}-12-31"),
    )
    return [pd.Timestamp(session_label_ts).normalize() for session_label_ts in session_index]


def _calendar_month_end_session_list(year_int: int) -> list[pd.Timestamp]:
    month_end_by_period_dict: dict[pd.Period, pd.Timestamp] = {}
    for session_label_ts in _calendar_session_list(year_int):
        month_end_by_period_dict[session_label_ts.to_period("M")] = session_label_ts
    return list(month_end_by_period_dict.values())


def _market_clock_timestamp_list(year_int: int) -> list[datetime]:
    market_timestamp_index = pd.date_range(
        start=pd.Timestamp(f"{year_int}-01-01 00:00", tz=MARKET_TZ),
        end=pd.Timestamp(f"{year_int + 1}-01-01 00:00", tz=MARKET_TZ),
        freq="30min",
        inclusive="left",
    )
    return [market_timestamp_ts.to_pydatetime() for market_timestamp_ts in market_timestamp_index]


def _stale_local_detail_dict(snapshot_date_str: str) -> dict[str, object]:
    return {
        "all_profiles_ready_bool": True,
        "snapshot_date_by_profile_dict": {PROFILE_STR: snapshot_date_str},
        "manifest_hash_by_profile_dict": {PROFILE_STR: "hash"},
        "error_by_profile_dict": {},
    }


def test_daily_eod_clock_builds_exactly_one_cycle_per_exchange_session():
    expected_session_list = _calendar_session_list(YEAR_INT)
    built_session_list: list[pd.Timestamp] = []
    last_built_session_label_ts: pd.Timestamp | None = None

    for as_of_ts in _market_clock_timestamp_list(YEAR_INT):
        required_session_label_ts = scheduler_utils.get_latest_completed_session_label_ts(
            as_of_ts,
            SESSION_CALENDAR_ID_STR,
        )
        if required_session_label_ts is None or required_session_label_ts.year != YEAR_INT:
            continue
        if (
            last_built_session_label_ts is None
            or required_session_label_ts > last_built_session_label_ts
        ):
            built_session_list.append(required_session_label_ts)
            last_built_session_label_ts = required_session_label_ts

    assert built_session_list == expected_session_list
    assert len(built_session_list) == 251
    assert len(set(built_session_list)) == len(built_session_list)
    assert pd.Timestamp("2026-01-01") not in built_session_list
    assert pd.Timestamp("2026-07-03") not in built_session_list


def test_daily_eod_clock_respects_real_early_close_plus_snapshot_buffer():
    before_buffer_session_label_ts = scheduler_utils.get_latest_completed_session_label_ts(
        datetime(2026, 11, 27, 13, 9, tzinfo=MARKET_TZ),
        SESSION_CALENDAR_ID_STR,
    )
    after_buffer_session_label_ts = scheduler_utils.get_latest_completed_session_label_ts(
        datetime(2026, 11, 27, 13, 10, tzinfo=MARKET_TZ),
        SESSION_CALENDAR_ID_STR,
    )

    assert scheduler_utils.get_session_close_timestamp_ts(
        pd.Timestamp("2026-11-27"),
        SESSION_CALENDAR_ID_STR,
    ) == datetime(2026, 11, 27, 13, 0, tzinfo=MARKET_TZ)
    assert before_buffer_session_label_ts == pd.Timestamp("2026-11-25")
    assert after_buffer_session_label_ts == pd.Timestamp("2026-11-27")


def test_monthly_clock_builds_exactly_one_cycle_per_exchange_month_end():
    monthly_release_obj = _make_release(
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
        release_id_str="monthly",
    )
    expected_month_end_session_list = _calendar_month_end_session_list(YEAR_INT)
    built_month_end_session_list: list[pd.Timestamp] = []
    last_built_month_end_session_label_ts: pd.Timestamp | None = None

    for as_of_ts in _market_clock_timestamp_list(YEAR_INT):
        required_session_label_ts = scheduler_utils.get_latest_completed_month_end_session_label_ts(
            as_of_ts,
            SESSION_CALENDAR_ID_STR,
        )
        if required_session_label_ts is None or required_session_label_ts.year != YEAR_INT:
            continue
        if (
            last_built_month_end_session_label_ts is None
            or required_session_label_ts > last_built_month_end_session_label_ts
        ):
            built_month_end_session_list.append(required_session_label_ts)
            last_built_month_end_session_label_ts = required_session_label_ts

            submission_timestamp_ts = scheduler_utils.build_submission_timestamp_ts(
                required_session_label_ts.to_pydatetime(),
                monthly_release_obj,
            )
            target_execution_timestamp_ts = scheduler_utils.build_target_execution_timestamp_ts(
                required_session_label_ts.to_pydatetime(),
                monthly_release_obj,
            )
            first_next_month_session_label_ts = (
                scheduler_utils.get_first_next_month_session_label_ts(
                    required_session_label_ts,
                    SESSION_CALENDAR_ID_STR,
                )
            )

            assert target_execution_timestamp_ts == scheduler_utils.get_session_open_timestamp_ts(
                first_next_month_session_label_ts,
                SESSION_CALENDAR_ID_STR,
            )
            assert submission_timestamp_ts < target_execution_timestamp_ts
            assert submission_timestamp_ts.date() == target_execution_timestamp_ts.date()

    assert built_month_end_session_list == expected_month_end_session_list
    assert [session_label_ts.date().isoformat() for session_label_ts in built_month_end_session_list] == [
        "2026-01-30",
        "2026-02-27",
        "2026-03-31",
        "2026-04-30",
        "2026-05-29",
        "2026-06-30",
        "2026-07-31",
        "2026-08-31",
        "2026-09-30",
        "2026-10-30",
        "2026-11-30",
        "2026-12-31",
    ]


def test_stale_norgate_snapshot_waits_inside_window_then_turns_red_after_deadline():
    daily_release_obj = _make_release(
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        release_id_str="daily",
    )
    required_session_label_ts = pd.Timestamp("2026-06-30")
    session_close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(
        required_session_label_ts,
        SESSION_CALENDAR_ID_STR,
    )
    inside_window_ts = session_close_timestamp_ts + timedelta(minutes=20)
    after_deadline_ts = session_close_timestamp_ts + timedelta(minutes=181)

    inside_window_detail_dict = norgate_sync_module._with_cycle_freshness_detail_dict(
        _stale_local_detail_dict("2026-06-29"),
        [daily_release_obj],
        inside_window_ts,
    )
    after_deadline_detail_dict = norgate_sync_module._with_cycle_freshness_detail_dict(
        _stale_local_detail_dict("2026-06-29"),
        [daily_release_obj],
        after_deadline_ts,
    )

    assert inside_window_detail_dict["required_snapshot_date_by_release_dict"] == {
        "daily": "2026-06-30",
    }
    assert inside_window_detail_dict["minimum_required_snapshot_date_by_profile_dict"] == {
        PROFILE_STR: "2026-06-30",
    }
    assert inside_window_detail_dict["stale_profile_list"] == [PROFILE_STR]
    assert inside_window_detail_dict["snapshot_fresh_for_cycle_bool"] is False
    assert inside_window_detail_dict["snapshot_stale_past_alert_deadline_bool"] is False
    assert inside_window_detail_dict["stale_alert_deadline_by_release_dict"] == {
        "daily": (session_close_timestamp_ts + timedelta(minutes=180)).isoformat(),
    }

    assert after_deadline_detail_dict["stale_profile_list"] == [PROFILE_STR]
    assert after_deadline_detail_dict["snapshot_fresh_for_cycle_bool"] is False
    assert after_deadline_detail_dict["snapshot_stale_past_alert_deadline_bool"] is True
