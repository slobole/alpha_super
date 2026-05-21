"""Jinja filters and tests used by Dashboard V3 templates.

Concentrating formatting here keeps templates terse and lets us unit-test
the tricky bits (severity normalisation, weight handling) without touching
HTML.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from jinja2 import Undefined

from alpha.live.dashboard_v3.data import _effective_severity_str, _normalize_severity_str


def _coerce_value_obj(value_obj: Any) -> Any:
    """Treat Jinja ``Undefined`` (missing dict keys via attribute access) as
    ``None`` so filters never raise ``UndefinedError`` on optional fields."""
    if isinstance(value_obj, Undefined):
        return None
    return value_obj


CLOCK_FORMAT_STR = "%H:%M:%S"
CLOCK_DATE_FORMAT_STR = "%m-%d %H:%M"


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
    value_obj = _coerce_value_obj(value_obj)
    if not value_obj:
        return ""
    text_str = str(value_obj)
    for parse_format_str in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            parsed_dt_obj = datetime.strptime(text_str.replace("Z", "+0000"), parse_format_str)
        except ValueError:
            continue
        return parsed_dt_obj.strftime(CLOCK_DATE_FORMAT_STR)
    # ISO 8601 fallback via fromisoformat (handles many tz styles in 3.11+).
    try:
        parsed_dt_obj = datetime.fromisoformat(text_str.replace("Z", "+00:00"))
    except ValueError:
        return text_str
    return parsed_dt_obj.strftime(CLOCK_DATE_FORMAT_STR)


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
    "effective_severity": filter_effective_severity_str,
    "step_severity": filter_step_severity_str,
    "normalize_severity": filter_normalize_severity_str,
}


__all__ = ["FILTER_MAP_DICT"]
