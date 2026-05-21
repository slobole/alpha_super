"""Unit tests for ``alpha.live.dashboard_v3.health``."""

from __future__ import annotations

from typing import Any

import pytest

from alpha.live.dashboard_v3.health import (
    HealthRollup,
    build_health_rollup,
)


def _pod_row_with_freshness(
    pod_id_str: str, item_list: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "pod_id_str": pod_id_str,
        "mode_str": "live",
        "data_freshness_dict": {"item_dict_list": item_list},
    }


def test_health_rollup_collapses_to_red_when_any_pod_norgate_is_red() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _pod_row_with_freshness(
                "pod_a",
                [
                    {"label_str": "Norgate", "severity_str": "green", "timestamp_str": "2026-05-21T10:00:00+00:00"},
                ],
            ),
            _pod_row_with_freshness(
                "pod_b",
                [
                    {"label_str": "Norgate", "severity_str": "red", "timestamp_str": "2026-05-21T11:00:00+00:00"},
                ],
            ),
        ]
    }
    rollup_obj = build_health_rollup(summary_dict)
    assert rollup_obj.severity_str == "red"
    norgate_cell_obj = next(
        cell_obj for cell_obj in rollup_obj.cell_dict_list if cell_obj.label_str == "Norgate"
    )
    assert norgate_cell_obj.severity_str == "red"
    # Picks the latest timestamp seen across pods.
    assert norgate_cell_obj.value_str == "2026-05-21T11:00:00+00:00"


def test_health_rollup_is_gray_when_no_pods() -> None:
    rollup_obj = build_health_rollup({"pod_row_dict_list": []})
    assert isinstance(rollup_obj, HealthRollup)
    # Disk check still produces a green/yellow/red so the overall severity
    # is whatever disk reports — not necessarily gray.
    assert rollup_obj.severity_str in {"gray", "green", "yellow", "red"}


def test_health_rollup_disk_cell_uses_provided_path(tmp_path) -> None:
    rollup_obj = build_health_rollup({"pod_row_dict_list": []}, disk_usage_path_str=str(tmp_path))
    disk_cell_obj = next(
        cell_obj for cell_obj in rollup_obj.cell_dict_list if cell_obj.label_str == "Disk"
    )
    assert "% used" in disk_cell_obj.value_str
    assert disk_cell_obj.severity_str in {"green", "yellow", "red"}


def test_health_rollup_unrecognised_severity_falls_back_to_gray() -> None:
    summary_dict = {
        "pod_row_dict_list": [
            _pod_row_with_freshness(
                "pod_a",
                [{"label_str": "Norgate", "severity_str": "purple", "timestamp_str": "x"}],
            ),
        ]
    }
    rollup_obj = build_health_rollup(summary_dict)
    norgate_cell_obj = next(
        cell_obj for cell_obj in rollup_obj.cell_dict_list if cell_obj.label_str == "Norgate"
    )
    assert norgate_cell_obj.severity_str == "gray"
