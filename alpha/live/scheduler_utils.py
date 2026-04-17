"""
SUMMARY: compute timing for signal/submission/execution


compute signal/submission/execution times for a release

build timing=f(signal clock,market calendar,Norgate readiness)
submit timing=f(execution policy,market calendar)

calendar helpers
    real market sessions (exchange calendar)
Norgate helpers
    real data readiness (Norgate heartbeat)
timestamp builders
    exact signal / submit / execution times
gate evaluators
    decide whether something is due now


"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from alpha.live.models import LiveRelease


DEFAULT_SUBMISSION_BUFFER_MINUTES_INT = 10
DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT = (6 * 60) + 30
PRE_CLOSE_SIGNAL_BUFFER_MINUTES_INT = 15
SUPPORTED_SESSION_CALENDAR_ID_TUPLE: tuple[str, ...] = ("XNYS", "XTSE", "XASX")
SUPPORTED_SIGNAL_CLOCK_TUPLE: tuple[str, ...] = (
    "eod_snapshot_ready",
    "month_end_snapshot_ready",
    "pre_close_15m",
)
SIGNAL_CLOCK_ALIAS_MAP: dict[str, str] = {
    "eod_close_plus_10m": "eod_snapshot_ready",
    "month_end_eod": "month_end_snapshot_ready",
    "pre_close_1545_et": "pre_close_15m",
}
DATA_PROFILE_HEARTBEAT_SYMBOL_MAP: dict[str, str] = {
    "norgate_eod_sp500_pit": "$SPX",
    "norgate_eod_etf_plus_vix_helper": "$SPX",
    "norgate_eod_ndx_pit": "$SPX",
}


def normalize_signal_clock_str(signal_clock_str: str) -> str:
    return SIGNAL_CLOCK_ALIAS_MAP.get(signal_clock_str, signal_clock_str)


@lru_cache(maxsize=None)
def get_exchange_calendar_obj(session_calendar_id_str: str):
    if session_calendar_id_str not in SUPPORTED_SESSION_CALENDAR_ID_TUPLE:
        raise ValueError(
            f"Unsupported session_calendar_id_str '{session_calendar_id_str}'. "
            f"Expected one of {SUPPORTED_SESSION_CALENDAR_ID_TUPLE}."
        )
    return xcals.get_calendar(session_calendar_id_str)


def get_market_timezone_obj(session_calendar_id_str: str) -> ZoneInfo:
    calendar_obj = get_exchange_calendar_obj(session_calendar_id_str)
    return ZoneInfo(str(calendar_obj.tz))


def to_market_timestamp_ts(
    raw_timestamp_ts: datetime,
    session_calendar_id_str: str,
) -> datetime:
    market_timezone_obj = get_market_timezone_obj(session_calendar_id_str)
    if raw_timestamp_ts.tzinfo is None:
        return raw_timestamp_ts.replace(tzinfo=market_timezone_obj)
    return raw_timestamp_ts.astimezone(market_timezone_obj)


def _session_label_from_date(
    date_obj,
    session_calendar_id_str: str,
) -> pd.Timestamp | None:
    calendar_obj = get_exchange_calendar_obj(session_calendar_id_str)
    session_label_ts = pd.Timestamp(date_obj)
    if not calendar_obj.is_session(session_label_ts):
        return None
    return session_label_ts


def session_label_from_timestamp_ts(
    raw_timestamp_ts: datetime,
    session_calendar_id_str: str,
) -> pd.Timestamp | None:
    market_timestamp_ts = to_market_timestamp_ts(raw_timestamp_ts, session_calendar_id_str)
    return _session_label_from_date(market_timestamp_ts.date(), session_calendar_id_str)


def _coerce_session_label_ts(
    raw_signal_timestamp_ts: datetime,
    session_calendar_id_str: str,
) -> pd.Timestamp:
    if isinstance(raw_signal_timestamp_ts, pd.Timestamp):
        if raw_signal_timestamp_ts.tzinfo is None:
            date_obj = raw_signal_timestamp_ts.date()
        else:
            date_obj = raw_signal_timestamp_ts.tz_convert(
                get_market_timezone_obj(session_calendar_id_str)
            ).date()
    else:
        if raw_signal_timestamp_ts.tzinfo is None:
            date_obj = raw_signal_timestamp_ts.date()
        else:
            date_obj = to_market_timestamp_ts(raw_signal_timestamp_ts, session_calendar_id_str).date()

    session_label_ts = _session_label_from_date(date_obj, session_calendar_id_str)
    if session_label_ts is None:
        raise ValueError(
            f"Date '{date_obj}' does not represent a trading session for {session_calendar_id_str}."
        )
    return session_label_ts


def _to_market_datetime_ts(
    utc_or_local_timestamp_ts,
    session_calendar_id_str: str,
) -> datetime:
    if isinstance(utc_or_local_timestamp_ts, pd.Timestamp):
        return utc_or_local_timestamp_ts.tz_convert(
            get_market_timezone_obj(session_calendar_id_str)
        ).to_pydatetime()
    return utc_or_local_timestamp_ts.astimezone(get_market_timezone_obj(session_calendar_id_str))


def get_session_open_timestamp_ts(
    session_label_ts: pd.Timestamp,
    session_calendar_id_str: str,
) -> datetime:
    calendar_obj = get_exchange_calendar_obj(session_calendar_id_str)
    return _to_market_datetime_ts(
        calendar_obj.session_open(session_label_ts),
        session_calendar_id_str,
    )


def get_session_close_timestamp_ts(
    session_label_ts: pd.Timestamp,
    session_calendar_id_str: str,
) -> datetime:
    calendar_obj = get_exchange_calendar_obj(session_calendar_id_str)
    return _to_market_datetime_ts(
        calendar_obj.session_close(session_label_ts),
        session_calendar_id_str,
    )


def get_next_session_label_ts(
    session_label_ts: pd.Timestamp,
    session_calendar_id_str: str,
) -> pd.Timestamp:
    calendar_obj = get_exchange_calendar_obj(session_calendar_id_str)
    return calendar_obj.next_session(session_label_ts)


def get_first_next_month_session_label_ts(
    session_label_ts: pd.Timestamp,
    session_calendar_id_str: str,
) -> pd.Timestamp:
    calendar_obj = get_exchange_calendar_obj(session_calendar_id_str)
    next_session_label_ts = calendar_obj.next_session(session_label_ts)
    while (
        next_session_label_ts.year == session_label_ts.year
        and next_session_label_ts.month == session_label_ts.month
    ):
        next_session_label_ts = calendar_obj.next_session(next_session_label_ts)
    return next_session_label_ts


def is_last_session_of_month_bool(
    session_label_ts: pd.Timestamp,
    session_calendar_id_str: str,
) -> bool:
    next_session_label_ts = get_next_session_label_ts(session_label_ts, session_calendar_id_str)
    return (
        next_session_label_ts.year != session_label_ts.year
        or next_session_label_ts.month != session_label_ts.month
    )


def load_latest_norgate_heartbeat_session_label_ts(
    data_profile_str: str,
) -> pd.Timestamp | None:
    if data_profile_str not in DATA_PROFILE_HEARTBEAT_SYMBOL_MAP:
        return None

    try:
        import norgatedata
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError(
            "Norgate snapshot readiness requires the norgatedata package and local updater."
        ) from exc

    heartbeat_symbol_str = DATA_PROFILE_HEARTBEAT_SYMBOL_MAP[data_profile_str]
    heartbeat_df = norgatedata.price_timeseries(
        heartbeat_symbol_str,
        timeseriesformat="pandas-dataframe",
    )
    if heartbeat_df is None or len(heartbeat_df.index) == 0:
        return None
    return pd.Timestamp(heartbeat_df.index[-1]).normalize()


def build_signal_timestamp_ts(
    signal_date_ts: datetime,
    release_obj: LiveRelease,
) -> datetime:
    normalized_signal_clock_str = normalize_signal_clock_str(release_obj.signal_clock_str)
    session_label_ts = _coerce_session_label_ts(signal_date_ts, release_obj.session_calendar_id_str)
    session_close_timestamp_ts = get_session_close_timestamp_ts(
        session_label_ts,
        release_obj.session_calendar_id_str,
    )
    if normalized_signal_clock_str in ("eod_snapshot_ready", "month_end_snapshot_ready"):
        return session_close_timestamp_ts
    if normalized_signal_clock_str == "pre_close_15m":
        return session_close_timestamp_ts - timedelta(minutes=PRE_CLOSE_SIGNAL_BUFFER_MINUTES_INT)
    raise ValueError(f"Unsupported signal_clock_str '{release_obj.signal_clock_str}'.")


def build_submission_timestamp_ts(
    signal_date_ts: datetime,
    release_obj: LiveRelease,
) -> datetime:
    session_label_ts = _coerce_session_label_ts(signal_date_ts, release_obj.session_calendar_id_str)

    if release_obj.execution_policy_str == "next_open_moo":
        next_session_label_ts = get_next_session_label_ts(session_label_ts, release_obj.session_calendar_id_str)
        next_session_open_timestamp_ts = get_session_open_timestamp_ts(
            next_session_label_ts,
            release_obj.session_calendar_id_str,
        )
        return next_session_open_timestamp_ts - timedelta(
            seconds=DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT
        )

    if release_obj.execution_policy_str == "same_day_moc":
        session_close_timestamp_ts = get_session_close_timestamp_ts(
            session_label_ts,
            release_obj.session_calendar_id_str,
        )
        return session_close_timestamp_ts - timedelta(minutes=DEFAULT_SUBMISSION_BUFFER_MINUTES_INT)

    if release_obj.execution_policy_str == "next_month_first_open":
        first_next_month_session_label_ts = get_first_next_month_session_label_ts(
            session_label_ts,
            release_obj.session_calendar_id_str,
        )
        first_next_month_open_timestamp_ts = get_session_open_timestamp_ts(
            first_next_month_session_label_ts,
            release_obj.session_calendar_id_str,
        )
        return first_next_month_open_timestamp_ts - timedelta(
            seconds=DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT
        )

    raise ValueError(f"Unsupported execution_policy_str '{release_obj.execution_policy_str}'.")


def build_target_execution_timestamp_ts(
    signal_date_ts: datetime,
    release_obj: LiveRelease,
) -> datetime:
    session_label_ts = _coerce_session_label_ts(signal_date_ts, release_obj.session_calendar_id_str)

    if release_obj.execution_policy_str == "next_open_moo":
        next_session_label_ts = get_next_session_label_ts(session_label_ts, release_obj.session_calendar_id_str)
        return get_session_open_timestamp_ts(next_session_label_ts, release_obj.session_calendar_id_str)

    if release_obj.execution_policy_str == "same_day_moc":
        return get_session_close_timestamp_ts(session_label_ts, release_obj.session_calendar_id_str)

    if release_obj.execution_policy_str == "next_month_first_open":
        first_next_month_session_label_ts = get_first_next_month_session_label_ts(
            session_label_ts,
            release_obj.session_calendar_id_str,
        )
        return get_session_open_timestamp_ts(
            first_next_month_session_label_ts,
            release_obj.session_calendar_id_str,
        )

    raise ValueError(f"Unsupported execution_policy_str '{release_obj.execution_policy_str}'.")


def next_business_day_timestamp_ts(
    signal_date_ts: datetime,
    session_calendar_id_str: str = "XNYS",
) -> datetime:
    session_label_ts = _coerce_session_label_ts(signal_date_ts, session_calendar_id_str)
    next_session_label_ts = get_next_session_label_ts(session_label_ts, session_calendar_id_str)
    return get_session_open_timestamp_ts(next_session_label_ts, session_calendar_id_str)


def first_business_day_of_next_month_timestamp_ts(
    signal_date_ts: datetime,
    session_calendar_id_str: str = "XNYS",
) -> datetime:
    session_label_ts = _coerce_session_label_ts(signal_date_ts, session_calendar_id_str)
    first_next_month_session_label_ts = get_first_next_month_session_label_ts(
        session_label_ts,
        session_calendar_id_str,
    )
    return get_session_open_timestamp_ts(first_next_month_session_label_ts, session_calendar_id_str)


def _build_due_from_snapshot_ready_bool(
    release_obj: LiveRelease,
    as_of_ts: datetime,
) -> bool:
    return evaluate_build_gate_dict(release_obj, as_of_ts)["due_bool"]


def evaluate_build_gate_dict(
    release_obj: LiveRelease,
    as_of_ts: datetime,
) -> dict[str, object]:
    latest_heartbeat_session_label_ts = load_latest_norgate_heartbeat_session_label_ts(
        release_obj.data_profile_str
    )
    if latest_heartbeat_session_label_ts is None:
        return {
            "due_bool": False,
            "reason_code_str": "snapshot_not_ready",
            "latest_heartbeat_session_date_str": None,
        }

    latest_heartbeat_session_date_str = latest_heartbeat_session_label_ts.date().isoformat()
    if normalize_signal_clock_str(release_obj.signal_clock_str) == "month_end_snapshot_ready":
        if not is_last_session_of_month_bool(
            latest_heartbeat_session_label_ts,
            release_obj.session_calendar_id_str,
        ):
            return {
                "due_bool": False,
                "reason_code_str": "not_month_end_session",
                "latest_heartbeat_session_date_str": latest_heartbeat_session_date_str,
            }

    market_timestamp_ts = to_market_timestamp_ts(as_of_ts, release_obj.session_calendar_id_str)
    market_date_obj = market_timestamp_ts.date()
    signal_date_obj = latest_heartbeat_session_label_ts.date()
    if market_date_obj == signal_date_obj:
        return {
            "due_bool": True,
            "reason_code_str": "snapshot_ready",
            "latest_heartbeat_session_date_str": latest_heartbeat_session_date_str,
        }

    next_session_label_ts = get_next_session_label_ts(
        latest_heartbeat_session_label_ts,
        release_obj.session_calendar_id_str,
    )
    if market_date_obj != next_session_label_ts.date():
        return {
            "due_bool": False,
            "reason_code_str": "snapshot_not_ready_for_session",
            "latest_heartbeat_session_date_str": latest_heartbeat_session_date_str,
        }

    submission_timestamp_ts = build_submission_timestamp_ts(
        latest_heartbeat_session_label_ts.to_pydatetime(),
        release_obj,
    )
    if market_timestamp_ts < submission_timestamp_ts:
        return {
            "due_bool": True,
            "reason_code_str": "carry_forward_snapshot_ready",
            "latest_heartbeat_session_date_str": latest_heartbeat_session_date_str,
        }
    return {
        "due_bool": False,
        "reason_code_str": "snapshot_window_expired",
        "latest_heartbeat_session_date_str": latest_heartbeat_session_date_str,
    }

def evaluate_execution_window_dict(
    order_plan_submission_timestamp_ts: datetime,
    as_of_ts: datetime,
) -> dict[str, object]:
    if as_of_ts >= order_plan_submission_timestamp_ts:
        return {"due_bool": True, "reason_code_str": "submission_window_open"}
    return {"due_bool": False, "reason_code_str": "waiting_for_submission_window"}


def get_signal_clock_reason_code_str(
    release_obj: LiveRelease,
    as_of_ts: datetime,
) -> str:
    normalized_signal_clock_str = normalize_signal_clock_str(release_obj.signal_clock_str)

    if normalized_signal_clock_str in ("eod_snapshot_ready", "month_end_snapshot_ready"):
        return str(evaluate_build_gate_dict(release_obj, as_of_ts)["reason_code_str"])

    if normalized_signal_clock_str == "pre_close_15m":
        session_label_ts = session_label_from_timestamp_ts(as_of_ts, release_obj.session_calendar_id_str)
        if session_label_ts is None:
            return "no_session"
        signal_timestamp_ts = build_signal_timestamp_ts(session_label_ts.to_pydatetime(), release_obj)
        session_close_timestamp_ts = get_session_close_timestamp_ts(
            session_label_ts,
            release_obj.session_calendar_id_str,
        )
        market_timestamp_ts = to_market_timestamp_ts(as_of_ts, release_obj.session_calendar_id_str)
        if signal_timestamp_ts <= market_timestamp_ts < session_close_timestamp_ts:
            return "signal_window_open"
        if market_timestamp_ts < signal_timestamp_ts:
            return "before_signal_window"
        return "signal_window_closed"

    return "unsupported_signal_clock"


def is_release_due_for_build(release_obj: LiveRelease, as_of_ts: datetime) -> bool:
    normalized_signal_clock_str = normalize_signal_clock_str(release_obj.signal_clock_str)

    if normalized_signal_clock_str in ("eod_snapshot_ready", "month_end_snapshot_ready"):
        return bool(evaluate_build_gate_dict(release_obj, as_of_ts)["due_bool"])

    if normalized_signal_clock_str == "pre_close_15m":
        return get_signal_clock_reason_code_str(release_obj, as_of_ts) == "signal_window_open"

    return False


def select_due_release_list(
    release_list: list[LiveRelease],
    as_of_ts: datetime,
) -> list[LiveRelease]:
    due_release_list: list[LiveRelease] = []
    for release_obj in release_list:
        if not release_obj.enabled_bool:
            continue
        if is_release_due_for_build(release_obj, as_of_ts):
            due_release_list.append(release_obj)
    return due_release_list


def utc_now_ts() -> datetime:
    return datetime.now(UTC)
