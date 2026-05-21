"""Tests for the live-vs-backtest tracking module."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha.live.dashboard_v3.expected_pnl import (
    build_tracking_comparison,
    load_expected_pnl_map,
)


@pytest.fixture(name="expected_pnl_path_str")
def fixture_expected_pnl_path_str(tmp_path: Path) -> str:
    yaml_path_obj = tmp_path / "expected_pnl.yaml"
    yaml_path_obj.write_text(
        """\
pod_a:
  daily_mean_return_float: 0.001
  daily_volatility_float: 0.005

pod_b:
  daily_mean_return_float: 0.0
  daily_volatility_float: 0.01
  band_sigma_float: 1.0
  sample_count_int: 250
""",
        encoding="utf-8",
    )
    return str(yaml_path_obj)


def test_load_returns_empty_map_when_file_missing(tmp_path) -> None:
    entry_map_dict = load_expected_pnl_map(str(tmp_path / "missing.yaml"))
    assert entry_map_dict == {}


def test_load_parses_two_pods(expected_pnl_path_str) -> None:
    entry_map_dict = load_expected_pnl_map(expected_pnl_path_str)
    assert set(entry_map_dict.keys()) == {"pod_a", "pod_b"}
    pod_a_obj = entry_map_dict["pod_a"]
    assert pod_a_obj.daily_mean_return_float == 0.001
    assert pod_a_obj.band_sigma_float == 2.0  # default
    pod_b_obj = entry_map_dict["pod_b"]
    assert pod_b_obj.sample_count_int == 250
    assert pod_b_obj.band_sigma_float == 1.0


def test_load_skips_entries_missing_required_fields(tmp_path) -> None:
    yaml_path_obj = tmp_path / "expected_pnl.yaml"
    yaml_path_obj.write_text(
        "good_pod:\n  daily_mean_return_float: 0.0\n  daily_volatility_float: 0.01\n"
        "bad_pod:\n  daily_volatility_float: 0.01\n",  # missing mean
        encoding="utf-8",
    )
    entry_map_dict = load_expected_pnl_map(str(yaml_path_obj))
    assert "good_pod" in entry_map_dict
    assert "bad_pod" not in entry_map_dict


def test_comparison_marks_outside_band(expected_pnl_path_str) -> None:
    entry_map_dict = load_expected_pnl_map(expected_pnl_path_str)
    comparison_obj = build_tracking_comparison("pod_a", actual_daily_return_float=0.02, expected_pnl_map_dict=entry_map_dict)
    assert comparison_obj.has_data_bool is True
    assert comparison_obj.is_outside_band_bool is True
    assert comparison_obj.severity_str == "yellow"
    assert "outside band" in comparison_obj.summary_str


def test_comparison_within_band_is_green(expected_pnl_path_str) -> None:
    entry_map_dict = load_expected_pnl_map(expected_pnl_path_str)
    comparison_obj = build_tracking_comparison("pod_a", actual_daily_return_float=0.0011, expected_pnl_map_dict=entry_map_dict)
    assert comparison_obj.is_outside_band_bool is False
    assert comparison_obj.severity_str == "green"
    assert "within band" in comparison_obj.summary_str


def test_comparison_unknown_pod_returns_no_data(expected_pnl_path_str) -> None:
    entry_map_dict = load_expected_pnl_map(expected_pnl_path_str)
    comparison_obj = build_tracking_comparison(
        "no_such_pod", actual_daily_return_float=0.01, expected_pnl_map_dict=entry_map_dict
    )
    assert comparison_obj.has_data_bool is False
    assert comparison_obj.severity_str == "gray"


def test_comparison_missing_actual_returns_no_data(expected_pnl_path_str) -> None:
    entry_map_dict = load_expected_pnl_map(expected_pnl_path_str)
    comparison_obj = build_tracking_comparison(
        "pod_a", actual_daily_return_float=None, expected_pnl_map_dict=entry_map_dict
    )
    assert comparison_obj.has_data_bool is False
