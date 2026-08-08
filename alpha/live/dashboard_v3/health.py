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
    mode_str: str | None = None,
) -> HealthRollup:
    pod_row_dict_list = [
        row_dict
        for row_dict in summary_dict.get("pod_row_dict_list") or []
        if mode_str is None or str(row_dict.get("mode_str") or "") == mode_str
    ]
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
    item_record_list: list[tuple[str, str, str]] = []
    for row_dict in pod_row_dict_list:
        freshness_dict = row_dict.get("data_freshness_dict") or {}
        matching_item_dict = next(
            (
                item_dict
                for item_dict in freshness_dict.get("item_dict_list") or []
                if str(item_dict.get("label_str")) == item_label_str
            ),
            None,
        )
        pod_id_str = str(row_dict.get("pod_id_str") or "?")
        if matching_item_dict is None:
            item_record_list.append(("gray", "—", pod_id_str))
            continue
        item_severity_str = _normalize_severity_str(
            matching_item_dict.get("severity_str")
        )
        timestamp_obj = (
            matching_item_dict.get("value_str")
            or matching_item_dict.get("timestamp_str")
        )
        item_record_list.append(
            (item_severity_str, str(timestamp_obj or "—"), pod_id_str)
        )
    if not item_record_list:
        return HealthCellDict(
            label_str=item_label_str,
            value_str="—",
            severity_str="gray",
            detail_str="no enabled pods reported",
        )

    worst_severity_str = _worst_severity_str(
        severity_str for severity_str, _, _ in item_record_list
    )
    worst_record_list = [
        record_tuple
        for record_tuple in item_record_list
        if record_tuple[0] == worst_severity_str
    ]
    _, displayed_value_str, displayed_pod_id_str = min(
        worst_record_list,
        key=lambda record_tuple: record_tuple[1],
    )
    return HealthCellDict(
        label_str=item_label_str,
        value_str=displayed_value_str,
        severity_str=worst_severity_str,
        detail_str=(
            f"{len(item_record_list)} pod(s) reporting · worst "
            f"{worst_severity_str}: {displayed_pod_id_str}"
        ),
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
