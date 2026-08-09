"""Unit tests for ``alpha.live.dashboard_v3.charts``."""

from __future__ import annotations

from alpha.live.dashboard_v3.charts import (
    CHART_VIEW_HEIGHT_INT,
    CHART_VIEW_WIDTH_INT,
    DAILY_PANEL_VIEW_HEIGHT_INT,
    build_book_risk_dict,
    build_equity_chart_dict,
    build_monthly_return_dict_list,
)


def _point(
    date_str: str,
    equity_float: float,
    pnl_float: float = 0.0,
    *,
    daily_pct_float: float | None = None,
    flow_float: float = 0.0,
) -> dict:
    point_dict = {
        "market_date_str": date_str,
        "equity_float": equity_float,
        "daily_pnl_float": pnl_float,
        "flow_float": flow_float,
    }
    if daily_pct_float is not None:
        point_dict["daily_pnl_pct_float"] = daily_pct_float
    return point_dict


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


def test_window_1w_keeps_five_return_intervals_plus_baseline() -> None:
    long_point_list = [
        _point(f"2026-05-{day_int:02d}", 10000.0 + day_int)
        for day_int in range(1, 32)
    ]
    chart_obj = build_equity_chart_dict(long_point_list, window_str="1w")
    assert chart_obj.point_count_int == 6


def test_mtd_and_ytd_keep_the_previous_period_eod_as_baseline() -> None:
    point_list = [
        _point("2025-12-31", 9000.0),
        _point("2026-01-02", 9100.0),
        _point("2026-04-30", 10000.0),
        _point("2026-05-01", 10100.0),
        _point("2026-05-04", 10200.0),
    ]

    mtd_chart_obj = build_equity_chart_dict(point_list, window_str="mtd")
    ytd_chart_obj = build_equity_chart_dict(point_list, window_str="ytd")

    assert mtd_chart_obj.earliest_market_date_str == "2026-04-30"
    assert ytd_chart_obj.earliest_market_date_str == "2025-12-31"
    assert mtd_chart_obj.window_is_partial_bool is False
    assert ytd_chart_obj.window_is_partial_bool is False


def test_mtd_marks_history_without_a_prior_eod_baseline_as_partial() -> None:
    chart_obj = build_equity_chart_dict(
        [
            _point("2026-05-15", 10000.0),
            _point("2026-05-18", 10100.0),
        ],
        window_str="mtd",
    )

    assert chart_obj.window_is_partial_bool is True
    assert chart_obj.window_note_str == "Partial period · prior EOD baseline unavailable."


def test_ytd_marks_history_without_a_prior_eod_baseline_as_partial() -> None:
    chart_obj = build_equity_chart_dict(
        [
            _point("2026-05-15", 10000.0),
            _point("2026-05-18", 10100.0),
        ],
        window_str="ytd",
    )

    assert chart_obj.window_str == "ytd"
    assert chart_obj.window_is_partial_bool is True
    assert chart_obj.window_note_str == "Partial period · prior EOD baseline unavailable."


def test_mtd_does_not_treat_an_unusable_prior_eod_as_a_baseline() -> None:
    chart_obj = build_equity_chart_dict(
        [
            {"market_date_str": "2026-04-30", "equity_float": None},
            _point("2026-05-01", 10000.0),
            _point("2026-05-04", 10100.0),
        ],
        window_str="mtd",
    )

    assert chart_obj.earliest_market_date_str == "2026-05-01"
    assert chart_obj.window_is_partial_bool is True


def test_malformed_period_date_falls_back_to_all() -> None:
    chart_obj = build_equity_chart_dict(
        [
            _point("2026-05-15", 10000.0),
            _point("not-a-date", 10100.0),
        ],
        window_str="mtd",
    )

    assert chart_obj.window_str == "all"
    assert chart_obj.window_note_str == "Requested period unavailable · invalid EOD date."


def test_window_all_preserves_full_history() -> None:
    long_point_list = [
        _point(f"2026-05-{day_int:02d}", 10000.0 + day_int)
        for day_int in range(1, 32)
    ]
    chart_obj = build_equity_chart_dict(long_point_list, window_str="all")
    assert chart_obj.point_count_int == 31


def test_daily_bars_built_proportional_to_max_abs() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0, pnl_float=0.0),
        _point("2026-05-02", 10100.0, pnl_float=100.0),
        _point("2026-05-03", 9900.0, pnl_float=-200.0),
    ])
    # Default pct mode → daily return bars (first point has no prior, so 2 bars).
    assert len(chart_obj.pnl_bar_dict_list) == 2
    largest_bar_dict = max(chart_obj.pnl_bar_dict_list, key=lambda d: d["height_float"])
    # The largest bar should correspond to the down day.
    assert largest_bar_dict["is_positive_bool"] is False
    # All bars fit inside the panel viewBox.
    for bar_dict in chart_obj.pnl_bar_dict_list:
        assert bar_dict["y_float"] >= 0
        assert bar_dict["y_float"] + bar_dict["height_float"] <= DAILY_PANEL_VIEW_HEIGHT_INT
    # IBKR-style panel exposes a centered zero line and a symmetric ±max axis.
    assert chart_obj.daily_zero_y_float > 0
    assert len(chart_obj.daily_y_axis_tick_dict_list) == 3
    assert chart_obj.daily_y_axis_tick_dict_list[1]["label_str"] in ("+0.00%", "+$0")


def test_daily_bars_follow_the_dollar_toggle() -> None:
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0, pnl_float=0.0),
        _point("2026-05-02", 10100.0, pnl_float=100.0),
        _point("2026-05-03", 9900.0, pnl_float=-200.0),
    ], value_mode_str="dollar")
    # In $ mode the bars use the daily dollar PnL and label it in dollars.
    label_str_list = [bar_dict["pnl_label_str"] for bar_dict in chart_obj.pnl_bar_dict_list]
    assert any("$" in label_str for label_str in label_str_list)
    assert "-$200" in label_str_list


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
    assert chart_obj.vol_observation_count_int == 3


def test_book_risk_vol_uses_only_latest_twenty_returns() -> None:
    equity_float = 10000.0
    point_list = [_point("2026-01-01", equity_float)]
    for day_int in range(2, 27):
        return_float = 0.01 if day_int % 2 == 0 else -0.005
        equity_float *= 1.0 + return_float
        point_list.append(
            _point(f"2026-01-{day_int:02d}", equity_float)
        )

    full_risk_obj = build_book_risk_dict(point_list)
    trailing_risk_obj = build_book_risk_dict(point_list[-21:])
    full_chart_obj = build_equity_chart_dict(point_list)
    trailing_chart_obj = build_equity_chart_dict(point_list[-21:])

    assert full_risk_obj.vol_observation_count_int == 20
    assert full_risk_obj.annualized_vol_label_str == trailing_risk_obj.annualized_vol_label_str
    assert full_chart_obj.vol_observation_count_int == 20
    assert full_chart_obj.annualized_vol_label_str == trailing_chart_obj.annualized_vol_label_str


def test_vol_footnote_degrades_gracefully_for_two_points() -> None:
    # A single return is not enough for a sample stdev → vol stays "—".
    chart_obj = build_equity_chart_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ])
    risk_obj = build_book_risk_dict([
        _point("2026-05-01", 10000.0),
        _point("2026-05-02", 10500.0),
    ])
    assert chart_obj.annualized_vol_label_str == "—"
    assert chart_obj.vol_observation_count_int == 1
    assert risk_obj.annualized_vol_label_str == "—"
    assert risk_obj.vol_observation_count_int == 1


def test_flow_adjusted_return_drives_every_chart_readout() -> None:
    point_list = [
        _point("2026-04-30", 10000.0),
        _point(
            "2026-05-31",
            20100.0,
            pnl_float=100.0,
            daily_pct_float=0.01,
            flow_float=10000.0,
        ),
        _point(
            "2026-06-01",
            19900.0,
            pnl_float=-200.0,
            daily_pct_float=(19900.0 / 20100.0) - 1.0,
        ),
    ]

    chart_dict = build_equity_chart_dict(point_list).as_dict()
    point_by_date_dict = {
        point_dict["market_date_str"]: point_dict
        for point_dict in chart_dict["point_dict_list"]
    }
    assert point_by_date_dict["2026-05-31"]["daily_pct_label_str"] == "+1.00%"
    assert [
        bar_dict["pnl_label_str"]
        for bar_dict in chart_dict["pnl_bar_dict_list"]
        if bar_dict["market_date_str"] == "2026-05-31"
    ] == ["+1.00%"]

    monthly_return_by_label_dict = {
        month_dict["month_label_str"]: month_dict
        for month_dict in build_monthly_return_dict_list(point_list)
    }
    assert monthly_return_by_label_dict["May 26"]["return_label_str"] == "+1.00%"

    risk_dict = build_book_risk_dict(point_list).as_dict()
    assert risk_dict["max_drawdown_label_str"] == "-1.00%"
    assert risk_dict["current_drawdown_label_str"] == "-1.00%"
    assert risk_dict["peak_equity_label_str"] == "$20,100"


def test_full_withdrawal_and_redeposit_does_not_invent_return() -> None:
    point_list = [
        _point("2026-05-01", 10000.0),
        _point(
            "2026-05-02",
            0.0,
            flow_float=-10000.0,
            daily_pct_float=0.0,
        ),
        _point(
            "2026-05-03",
            10000.0,
            flow_float=10000.0,
        ),
    ]

    pct_chart_dict = build_equity_chart_dict(point_list).as_dict()
    assert pct_chart_dict["return_unavailable_bool"] is True
    assert pct_chart_dict["has_curve_bool"] is False

    dollar_chart_dict = build_equity_chart_dict(
        point_list,
        value_mode_str="dollar",
    ).as_dict()
    assert dollar_chart_dict["has_curve_bool"] is True
    assert dollar_chart_dict["latest_since_start_pnl_label_str"] == "+$0"
    assert dollar_chart_dict["latest_since_start_return_label_str"] == "—"

    monthly_return_dict_list = build_monthly_return_dict_list(point_list)
    assert monthly_return_dict_list[-1]["is_available_bool"] is False
    assert monthly_return_dict_list[-1]["return_label_str"] == "—"
    assert build_book_risk_dict(point_list).has_data_bool is False


def test_latest_zero_equity_is_preserved_in_risk_state() -> None:
    risk_dict = build_book_risk_dict(
        [
            _point("2026-05-01", 100.0),
            _point("2026-05-02", 0.0, pnl_float=-100.0, daily_pct_float=-1.0),
        ]
    ).as_dict()
    assert risk_dict["has_data_bool"] is True
    assert risk_dict["current_equity_label_str"] == "$0"
    assert risk_dict["current_drawdown_label_str"] == "-100.00%"


# ── build_pod_holdings_pie_dict — presentation math only, no I/O ──────────

def test_holdings_pie_weights_targets_and_drift():
    from alpha.live.dashboard_v3.charts import build_pod_holdings_pie_dict

    pie_dict = build_pod_holdings_pie_dict(
        [
            {"asset_str": "SPY", "market_value_float": 4000.0},
            {"asset_str": "TLT", "market_value_float": -1000.0},
            {"asset_str": "GHOST", "market_value_float": None},
        ],
        cash_float=1000.0,
        equity_float=5000.0,
        target_weight_map_dict={"SPY": 0.6, "TLT": 0.2},
    )
    assert pie_dict["has_data_bool"]
    by_label_dict = {s["label_str"]: s for s in pie_dict["slice_dict_list"]}
    # *** CRITICAL*** Short legs weight by absolute value: |4000|+|-1000|+1000 = 6000.
    assert by_label_dict["SPY"]["weight_pct_label_str"] == "66.7%"
    assert by_label_dict["TLT"]["weight_pct_label_str"] == "16.7%"
    assert by_label_dict["Cash"]["weight_pct_label_str"] == "16.7%"
    assert by_label_dict["SPY"]["drift_pp_label_str"] == "+6.7pp"
    assert by_label_dict["Cash"]["drift_pp_label_str"] == "—"
    # Unpriced positions are surfaced, never silently dropped.
    assert pie_dict["unpriced_count_int"] == 1
    assert abs(sum(s["weight_float"] for s in pie_dict["slice_dict_list"]) - 1.0) < 1e-6


def test_holdings_pie_empty_book_reports_no_data():
    from alpha.live.dashboard_v3.charts import build_pod_holdings_pie_dict

    pie_dict = build_pod_holdings_pie_dict([], cash_float=0.0, equity_float=None)
    assert not pie_dict["has_data_bool"]
