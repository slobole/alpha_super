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
    # First/last x values sit inside the reserved plot area, leaving room for
    # visible Y-axis labels.
    parts_list = chart_obj.path_d_str.split()
    first_x_float = float(parts_list[1])
    last_x_float = float(parts_list[-2])
    chart_dict = chart_obj.as_dict()
    assert abs(first_x_float - chart_dict["plot_left_int"]) < 1e-6
    assert abs(last_x_float - chart_dict["plot_right_int"]) < 1e-6
    assert chart_obj.curve_area_d_str.startswith("M ")
    assert chart_obj.curve_area_d_str.endswith("Z")


def test_chart_exposes_axis_ticks_for_svg_rendering() -> None:
    chart_dict = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ]).as_dict()

    y_axis_tick_dict_list = chart_dict["y_axis_tick_dict_list"]
    x_axis_tick_dict_list = chart_dict["x_axis_tick_dict_list"]
    assert len(y_axis_tick_dict_list) == 3
    assert [tick_dict["label_str"] for tick_dict in y_axis_tick_dict_list] == [
        "+5.00%", "+2.50%", "+0.00%",
    ]
    for tick_dict in y_axis_tick_dict_list:
        assert 0 < tick_dict["y_float"] < CHART_VIEW_HEIGHT_INT
    assert [tick_dict["label_str"] for tick_dict in x_axis_tick_dict_list] == [
        "May 1", "May 2",
    ]
    assert [tick_dict["text_anchor_str"] for tick_dict in x_axis_tick_dict_list] == [
        "start", "end",
    ]
    assert chart_dict["pnl_zero_y_float"] == PNL_BAR_BLOCK_HEIGHT_INT / 2
    assert chart_dict["latest_since_start_pnl_label_str"] == "+$500"
    assert chart_dict["latest_since_start_return_label_str"] == "+5.00%"


def test_chart_records_min_max_range_labels() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 11000.0),
        _point("2026-05-03", 9500.0),
    ])
    assert abs(chart_obj.range_min_float - (-0.05)) < 1e-12
    assert abs(chart_obj.range_max_float - 0.10) < 1e-12
    assert chart_obj.range_min_label_str == "-5.00%"
    assert chart_obj.range_max_label_str == "+10.00%"


def test_cumulative_pnl_axis_includes_zero_when_curve_falls_below_start() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 11000.0),
        _point("2026-05-03", 9500.0),
    ])
    chart_dict = chart_obj.as_dict()
    assert chart_dict["range_min_label_str"] == "-5.00%"
    assert chart_dict["range_max_label_str"] == "+10.00%"
    assert chart_dict["zero_y_float"] > chart_dict["plot_top_int"]
    assert chart_dict["zero_y_float"] < chart_dict["plot_bottom_int"]


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
    assert len(chart_obj.pnl_bar_dict_list) == 2
    largest_bar_dict = max(chart_obj.pnl_bar_dict_list, key=lambda d: d["height_float"])
    # The largest bar should correspond to the down-return day.
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


def test_pct_mode_is_the_default_and_labels_axis_in_percent() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ])
    assert chart_obj.value_mode_str == "pct"
    assert chart_obj.range_max_label_str == "+5.00%"
    assert all("%" in tick["label_str"] for tick in chart_obj.as_dict()["y_axis_tick_dict_list"])


def test_dollar_mode_switches_axis_and_labels_to_dollars() -> None:
    point_list = [_point("2026-05-01", 10000.0), _point("2026-05-02", 10500.0)]
    pct_chart_obj = build_equity_chart_dict(point_list, value_mode_str="pct")
    dollar_chart_obj = build_equity_chart_dict(point_list, value_mode_str="dollar")
    assert dollar_chart_obj.value_mode_str == "dollar"
    # The y-axis top now reads in dollars; the geometry differs from % mode.
    assert dollar_chart_obj.range_max_label_str == "+$500"
    assert dollar_chart_obj.range_min_label_str == "+$0"
    assert all("$" in tick["label_str"] for tick in dollar_chart_obj.as_dict()["y_axis_tick_dict_list"])
    # The since-start scalar labels are mode-independent (both always computed).
    assert dollar_chart_obj.latest_since_start_pnl_label_str == "+$500"
    assert pct_chart_obj.latest_since_start_return_label_str == "+5.00%"


def test_unknown_value_mode_falls_back_to_pct() -> None:
    chart_obj = build_equity_chart_dict(
        [_point("2026-05-01", 10000.0), _point("2026-05-02", 11000.0)],
        value_mode_str="bogus",
    )
    assert chart_obj.value_mode_str == "pct"


def test_max_drawdown_footnote_measures_from_running_peak() -> None:
    # Up to 11000 (peak), down to 9900, back up. *** the running peak must only
    # look backward, so the trough is -10% from the 11000 peak, not from a later
    # higher value.
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 11000.0),
        _point("2026-05-03", 9900.0),
        _point("2026-05-04", 12000.0),
    ])
    assert chart_obj.max_drawdown_label_str == "-10.00%"   # 9900/11000 - 1


def test_monotonic_up_curve_reports_no_drawdown() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
        _point("2026-05-03", 11000.0),
    ])
    assert chart_obj.max_drawdown_label_str == "0.00%"


def test_annualized_vol_footnote_is_computed() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10200.0),
        _point("2026-05-03", 9900.0),
        _point("2026-05-04", 10100.0),
    ])
    assert chart_obj.annualized_vol_label_str.endswith("%")


def test_vol_footnote_degrades_gracefully_for_two_points() -> None:
    # A single return is not enough for a sample stdev → vol stays "—".
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ])
    assert chart_obj.annualized_vol_label_str == "—"
