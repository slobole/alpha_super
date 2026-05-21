"""System health roll-up for the Dashboard V3 health strip.

Aggregates per-pod ``data_freshness_dict`` items into 3-4 system-wide
status cells the operator can scan in one second. Bias of this module:
**show worst severity wins**. If any pod's Norgate sync is yellow, the
header is yellow, because that pod's signals are degraded.

Cheap on-host probes (disk / DB size) live here too so the operator can
see "230 MB DB · 47% disk full" without leaving the home view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import shutil
from pathlib import Path
from typing import Any, Iterable

from alpha.live.dashboard_v3.data import _normalize_severity_str


DISK_USAGE_PATH_STR = "."
SEVERITY_RANK_DICT = {"red": 0, "yellow": 1, "gray": 2, "green": 3}


@dataclass
class HealthCellDict:
    label_str: str
    value_str: str
    severity_str: str = "gray"
    detail_str: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "label_str": self.label_str,
            "value_str": self.value_str,
            "severity_str": self.severity_str,
            "detail_str": self.detail_str,
        }


@dataclass
class HealthRollup:
    severity_str: str = "gray"
    cell_dict_list: list[HealthCellDict] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity_str": self.severity_str,
            "cell_dict_list": [cell_obj.as_dict() for cell_obj in self.cell_dict_list],
        }


def build_health_rollup(
    summary_dict: dict[str, Any],
    disk_usage_path_str: str = DISK_USAGE_PATH_STR,
) -> HealthRollup:
    pod_row_dict_list = summary_dict.get("pod_row_dict_list") or []
    cell_obj_list = [
        _roll_up_freshness_cell(pod_row_dict_list, "Norgate"),
        _roll_up_freshness_cell(pod_row_dict_list, "Pod state"),
        _roll_up_freshness_cell(pod_row_dict_list, "EOD Snapshot"),
        _build_disk_cell(disk_usage_path_str),
    ]
    worst_severity_str = _worst_severity_str(
        cell_obj.severity_str for cell_obj in cell_obj_list
    )
    return HealthRollup(severity_str=worst_severity_str, cell_dict_list=cell_obj_list)


# ── private helpers ───────────────────────────────────────────────────────


def _roll_up_freshness_cell(
    pod_row_dict_list: list[dict[str, Any]], item_label_str: str
) -> HealthCellDict:
    worst_severity_str = "gray"
    latest_timestamp_str: str | None = None
    detail_str = "no enabled pods reported"
    item_count_int = 0
    for row_dict in pod_row_dict_list:
        freshness_dict = row_dict.get("data_freshness_dict") or {}
        for item_dict in freshness_dict.get("item_dict_list") or []:
            if str(item_dict.get("label_str")) != item_label_str:
                continue
            item_count_int += 1
            item_severity_str = _normalize_severity_str(item_dict.get("severity_str"))
            if (
                SEVERITY_RANK_DICT.get(item_severity_str, 9)
                < SEVERITY_RANK_DICT.get(worst_severity_str, 9)
            ):
                worst_severity_str = item_severity_str
            timestamp_str = item_dict.get("timestamp_str")
            if timestamp_str and (
                latest_timestamp_str is None or str(timestamp_str) > latest_timestamp_str
            ):
                latest_timestamp_str = str(timestamp_str)
    if item_count_int > 0:
        detail_str = (
            f"{item_count_int} pod(s) reporting · "
            f"worst {worst_severity_str}"
        )
    return HealthCellDict(
        label_str=item_label_str,
        value_str=latest_timestamp_str or "—",
        severity_str=worst_severity_str,
        detail_str=detail_str,
    )


def _build_disk_cell(disk_usage_path_str: str) -> HealthCellDict:
    try:
        disk_usage_obj = shutil.disk_usage(disk_usage_path_str)
    except OSError:
        return HealthCellDict(
            label_str="Disk",
            value_str="—",
            severity_str="gray",
            detail_str=f"could not stat {disk_usage_path_str}",
        )
    used_ratio_float = disk_usage_obj.used / max(1, disk_usage_obj.total)
    used_pct_int = int(round(used_ratio_float * 100))
    free_gb_float = disk_usage_obj.free / (1024**3)
    if used_ratio_float >= 0.90:
        severity_str = "red"
    elif used_ratio_float >= 0.75:
        severity_str = "yellow"
    else:
        severity_str = "green"
    return HealthCellDict(
        label_str="Disk",
        value_str=f"{used_pct_int}% used",
        severity_str=severity_str,
        detail_str=f"{free_gb_float:.1f} GB free at {Path(disk_usage_path_str).resolve()}",
    )


def _worst_severity_str(severity_str_iter: Iterable[str]) -> str:
    severity_list = list(severity_str_iter)
    if not severity_list:
        return "gray"
    return min(
        severity_list,
        key=lambda severity_str: SEVERITY_RANK_DICT.get(_normalize_severity_str(severity_str), 9),
    )
