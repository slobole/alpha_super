"""Jinja filters and tests used by Dashboard V3 templates.

Concentrating formatting here keeps templates terse and lets us unit-test
the tricky bits (severity normalisation, weight handling) without touching
HTML.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from jinja2 import Undefined

from alpha.live.dashboard_v3.data import _effective_severity_str, _normalize_severity_str


def _coerce_value_obj(value_obj: Any) -> Any:
    """Treat Jinja ``Undefined`` (missing dict keys via attribute access) as
    ``None`` so filters never raise ``UndefinedError`` on optional fields."""
    if isinstance(value_obj, Undefined):
        return None
    return value_obj


# All wall-clock display in the dashboard is in US market time (New York).
# Stored timestamps are UTC (see logging_utils.build_structured_event_record_dict);
# a naive timestamp is therefore assumed to be UTC before converting to ET.
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
MARKET_TIMEZONE_SUFFIX_STR = "ET"
CLOCK_DATE_FORMAT_STR = "%m-%d %H:%M"
CLOCK_TIME_FORMAT_STR = "%H:%M:%S"

_CLOCK_PARSE_FORMAT_STR_TUPLE = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def _parse_timestamp_to_market_dt(value_obj: Any) -> datetime | None:
    """Parse a timestamp string and return it as a New-York-aware ``datetime``.

    Naive inputs are treated as UTC (matching how events are persisted), then
    converted to market time. Returns ``None`` when the value cannot be parsed
    so callers can fall back to the raw text.
    """
    value_obj = _coerce_value_obj(value_obj)
    if not value_obj:
        return None
    text_str = str(value_obj).replace("Z", "+0000")
    parsed_dt_obj: datetime | None = None
    for parse_format_str in _CLOCK_PARSE_FORMAT_STR_TUPLE:
        try:
            parsed_dt_obj = datetime.strptime(text_str, parse_format_str)
            break
        except ValueError:
            continue
    if parsed_dt_obj is None:
        # ISO 8601 fallback via fromisoformat (handles many tz styles in 3.11+).
        try:
            parsed_dt_obj = datetime.fromisoformat(str(value_obj).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed_dt_obj.tzinfo is None:
        parsed_dt_obj = parsed_dt_obj.replace(tzinfo=timezone.utc)
    return parsed_dt_obj.astimezone(MARKET_TIMEZONE_OBJ)


def filter_money_str(value_obj: Any) -> str:
    value_obj = _coerce_value_obj(value_obj)
    if value_obj is None:
        return "—"
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "—"
    sign_str = "-" if value_float < 0 else ""
    absolute_value_float = abs(value_float)
    return f"{sign_str}${absolute_value_float:,.0f}"


def filter_number_str(value_obj: Any) -> str:
    value_obj = _coerce_value_obj(value_obj)
    if value_obj is None:
        return "—"
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "—"
    if abs(value_float) >= 1000:
        return f"{value_float:,.2f}"
    return f"{value_float:.4f}".rstrip("0").rstrip(".") or "0"


def filter_percent_str(value_obj: Any) -> str:
    value_obj = _coerce_value_obj(value_obj)
    if value_obj is None:
        return "—"
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "—"
    return f"{value_float * 100:.2f}%"


def filter_weight_str(value_obj: Any) -> str:
    value_obj = _coerce_value_obj(value_obj)
    if value_obj is None or value_obj == "":
        return "—"
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return str(value_obj)
    return f"{value_float * 100:.2f}%"


def filter_bps_str(value_obj: Any) -> str:
    value_obj = _coerce_value_obj(value_obj)
    if value_obj is None:
        return "—"
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return "—"
    return f"{value_float:+.1f} bps"


def filter_clock_str(value_obj: Any) -> str:
    """Date + minute in market time, e.g. ``05-21 16:00 ET``. No milliseconds."""
    if not _coerce_value_obj(value_obj):
        return ""
    market_dt_obj = _parse_timestamp_to_market_dt(value_obj)
    if market_dt_obj is None:
        return str(value_obj)
    return f"{market_dt_obj.strftime(CLOCK_DATE_FORMAT_STR)} {MARKET_TIMEZONE_SUFFIX_STR}"


def filter_clock_sec_str(value_obj: Any) -> str:
    """Second-precision market time, e.g. ``16:00:02 ET``. For intraday rows."""
    if not _coerce_value_obj(value_obj):
        return ""
    market_dt_obj = _parse_timestamp_to_market_dt(value_obj)
    if market_dt_obj is None:
        return str(value_obj)
    return f"{market_dt_obj.strftime(CLOCK_TIME_FORMAT_STR)} {MARKET_TIMEZONE_SUFFIX_STR}"


# Known machine reason codes → plain English. Anything not listed degrades
# gracefully to a de-underscored, capitalised form via filter_humanize_reason_str.
REASON_LABEL_DICT: dict[str, str] = {
    "calendar_month_end_label_resolved_to_last_tradable_session": (
        "Month-end label resolved to last tradable session"
    ),
}


def filter_humanize_reason_str(value_obj: Any) -> str:
    value_obj = _coerce_value_obj(value_obj)
    if value_obj is None or str(value_obj) == "":
        return ""
    code_str = str(value_obj)
    if code_str in REASON_LABEL_DICT:
        return REASON_LABEL_DICT[code_str]
    return code_str.replace("_", " ").strip().capitalize()


def filter_effective_severity_str(row_dict_obj: Any) -> str:
    if not isinstance(row_dict_obj, dict):
        return "gray"
    return _effective_severity_str(row_dict_obj)


def filter_step_severity_str(step_dict_obj: Any) -> str:
    if not isinstance(step_dict_obj, dict):
        return "gray"
    return _normalize_severity_str(
        step_dict_obj.get("severity_str") or step_dict_obj.get("status_str") or "gray"
    )


def filter_normalize_severity_str(value_obj: Any) -> str:
    return _normalize_severity_str(str(value_obj or "gray"))


FILTER_MAP_DICT: dict[str, Any] = {
    "money": filter_money_str,
    "number": filter_number_str,
    "percent": filter_percent_str,
    "weight": filter_weight_str,
    "bps": filter_bps_str,
    "clock": filter_clock_str,
    "clock_sec": filter_clock_sec_str,
    "humanize_reason": filter_humanize_reason_str,
    "effective_severity": filter_effective_severity_str,
    "step_severity": filter_step_severity_str,
    "normalize_severity": filter_normalize_severity_str,
}


__all__ = ["FILTER_MAP_DICT"]
