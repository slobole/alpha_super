"""Cross-pod schedule aggregator for the Dashboard V3 schedule strip.

Surfaces the next ~4 scheduled events across the whole multi-pod book
sorted by target execution time. Operators live by the clock — knowing
"submit_vplan for dv2_caspersky in 1h 38m" is more useful than scanning
individual pod cards.

The data already lives on each ``pod_row_dict`` as
``next_action_str`` + ``latest_*_target_execution_timestamp_str``; this
module only sorts and filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_SCHEDULE_LIMIT_INT = 6


@dataclass
class ScheduleEntry:
    pod_id_str: str
    mode_str: str
    action_str: str
    target_timestamp_str: str | None
    relative_str: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pod_id_str": self.pod_id_str,
            "mode_str": self.mode_str,
            "action_str": self.action_str,
            "target_timestamp_str": self.target_timestamp_str,
            "relative_str": self.relative_str,
        }


def build_schedule_entry_list(
    summary_dict: dict[str, Any],
    now_dt: datetime | None = None,
    limit_int: int = DEFAULT_SCHEDULE_LIMIT_INT,
) -> list[ScheduleEntry]:
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    pod_row_dict_list = summary_dict.get("pod_row_dict_list") or []
    candidate_list: list[ScheduleEntry] = []
    for row_dict in pod_row_dict_list:
        action_str = row_dict.get("next_action_str")
        if not action_str or action_str == "wait":
            continue
        target_timestamp_str = (
            row_dict.get("latest_vplan_target_execution_timestamp_str")
            or row_dict.get("latest_decision_plan_target_execution_timestamp_str")
        )
        candidate_list.append(
            ScheduleEntry(
                pod_id_str=str(row_dict.get("pod_id_str") or "?"),
                mode_str=str(row_dict.get("mode_str") or "?"),
                action_str=str(action_str),
                target_timestamp_str=str(target_timestamp_str) if target_timestamp_str else None,
                relative_str=_format_relative_time_str(target_timestamp_str, now_dt),
            )
        )
    candidate_list.sort(key=lambda entry_obj: (
        entry_obj.target_timestamp_str or "9999",
        entry_obj.pod_id_str,
    ))
    return candidate_list[:limit_int]


def _format_relative_time_str(target_timestamp_str: Any, now_dt: datetime) -> str:
    if not target_timestamp_str:
        return "—"
    try:
        target_dt = _parse_iso_datetime(str(target_timestamp_str))
    except ValueError:
        return str(target_timestamp_str)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    delta_seconds_int = int((target_dt - now_dt).total_seconds())
    if delta_seconds_int <= 0:
        absolute_seconds_int = -delta_seconds_int
        return f"{_format_duration_str(absolute_seconds_int)} ago"
    return f"in {_format_duration_str(delta_seconds_int)}"


def _parse_iso_datetime(text_str: str) -> datetime:
    return datetime.fromisoformat(text_str.replace("Z", "+00:00"))


def _format_duration_str(seconds_int: int) -> str:
    if seconds_int < 60:
        return f"{seconds_int}s"
    if seconds_int < 3600:
        minutes_int = seconds_int // 60
        seconds_remaining_int = seconds_int % 60
        if seconds_remaining_int == 0:
            return f"{minutes_int}m"
        return f"{minutes_int}m {seconds_remaining_int}s"
    hours_int = seconds_int // 3600
    minutes_remaining_int = (seconds_int % 3600) // 60
    if minutes_remaining_int == 0:
        return f"{hours_int}h"
    return f"{hours_int}h {minutes_remaining_int}m"
