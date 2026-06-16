"""Unit tests for ``alpha.live.dashboard_v3.charts``."""

from __future__ import annotations

from alpha.live.dashboard_v3.charts import (
    CHART_VIEW_HEIGHT_INT,
    CHART_VIEW_WIDTH_INT,
    PNL_BAR_BLOCK_HEIGHT_INT,
    build_equity_chart_dict,
)


def _point(date_str: str, equity_float: float, pnl_float: float = 0.0) -> dict:
    return {
        "market_date_str": date_str,
        "equity_float": equity_float,
        "daily_pnl_float": pnl_float,
    }


def test_empty_point_list_has_no_curve() -> None:
    chart_obj = build_equity_chart_dict([])
    assert chart_obj.has_curve_bool is False
    assert chart_obj.point_count_int == 0
    assert chart_obj.path_d_str == ""


def test_single_point_has_no_curve_but_keeps_value() -> None:
    chart_obj = build_equity_chart_dict([_point("2026-05-01", 10000.0)])
    assert chart_obj.has_curve_bool is False
    assert chart_obj.point_count_int == 1
    assert chart_obj.latest_equity_float == 10000.0
    assert chart_obj.latest_market_date_str == "2026-05-01"


def test_two_points_produces_valid_path_str() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ])
    assert chart_obj.has_curve_bool is True
    assert chart_obj.point_count_int == 2
    # Path starts with M and ends with an L command at the right edge.
    assert chart_obj.path_d_str.startswith("M ")
    assert " L " in chart_obj.path_d_str
    # First x is 0 and last x is the full chart width.
    parts_list = chart_obj.path_d_str.split()
    first_x_float = float(parts_list[1])
    last_x_float = float(parts_list[-2])
    assert abs(first_x_float - 0.0) < 1e-6
    assert abs(last_x_float - CHART_VIEW_WIDTH_INT) < 1e-6
    assert chart_obj.curve_area_d_str.startswith("M ")
    assert chart_obj.curve_area_d_str.endswith("Z")


def test_chart_exposes_static_grid_lines_for_svg_rendering() -> None:
    chart_dict = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ]).as_dict()

    grid_line_dict_list = chart_dict["grid_line_dict_list"]
    assert len(grid_line_dict_list) == 3
    for grid_line_dict in grid_line_dict_list:
        assert 0 < grid_line_dict["y_float"] < CHART_VIEW_HEIGHT_INT
    assert chart_dict["pnl_zero_y_float"] == PNL_BAR_BLOCK_HEIGHT_INT / 2


def test_chart_records_min_max_range_labels() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 11000.0),
        _point("2026-05-03", 9500.0),
    ])
    assert chart_obj.range_min_float == 9500.0
    assert chart_obj.range_max_float == 11000.0
    assert "9,500" in chart_obj.range_min_label_str
    assert "11,000" in chart_obj.range_max_label_str


def test_drawdown_polygon_emitted_when_curve_falls_below_peak() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 11000.0),
        _point("2026-05-03", 9500.0),
    ])
    assert chart_obj.drawdown_d_str != ""
    assert chart_obj.drawdown_d_str.endswith("Z")


def test_drawdown_polygon_omitted_when_curve_is_monotonic_up() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 11000.0),
        _point("2026-05-03", 12000.0),
    ])
    assert chart_obj.drawdown_d_str == ""


def test_window_30d_truncates_long_history() -> None:
    long_point_list = [
        _point(f"2026-05-{day_int:02d}", 10000.0 + day_int)
        for day_int in range(1, 32)
    ]
    chart_obj = build_equity_chart_dict(long_point_list, window_str="30d")
    assert chart_obj.point_count_int == 30


def test_window_all_preserves_full_history() -> None:
    long_point_list = [
        _point(f"2026-05-{day_int:02d}", 10000.0 + day_int)
        for day_int in range(1, 32)
    ]
    chart_obj = build_equity_chart_dict(long_point_list, window_str="all")
    assert chart_obj.point_count_int == 31


def test_pnl_bars_built_proportional_to_max_abs() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0, pnl_float=0.0),
        _point("2026-05-02", 10100.0, pnl_float=100.0),
        _point("2026-05-03", 9900.0, pnl_float=-200.0),
    ])
    assert len(chart_obj.pnl_bar_dict_list) == 3
    largest_bar_dict = max(chart_obj.pnl_bar_dict_list, key=lambda d: d["height_float"])
    # The largest bar should correspond to the -200 PnL day.
    assert largest_bar_dict["is_positive_bool"] is False
    # All bars fit within the half-height of the bar block.
    for bar_dict in chart_obj.pnl_bar_dict_list:
        assert bar_dict["y_float"] >= 0
        assert bar_dict["y_float"] + bar_dict["height_float"] <= PNL_BAR_BLOCK_HEIGHT_INT


def test_unknown_window_falls_back_to_all() -> None:
    chart_obj = build_equity_chart_dict(
        [_point("2026-05-01", 10000.0), _point("2026-05-02", 11000.0)],
        window_str="bogus",
    )
    assert chart_obj.point_count_int == 2
