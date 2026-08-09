"""SVG equity-chart math for Dashboard V3.

Pure functions. No DOM, no Flask. Produces a render-ready dict that templates
plug into a fixed-viewBox ``<svg>``. The two consumers are:

1. The combined-book curve at the top of ``/live`` and ``/paper``.
2. The per-pod curve inside each pod's EOD stage card.

The chart never tries to be pretty — it tries to be honest. Drawdown shading
shows the area between the running peak and the current curve so the
operator can see "we're still under water from the May 12 peak" at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
import statistics
from typing import Any


TRADING_DAYS_PER_YEAR_INT = 252
ROLLING_VOL_SESSION_COUNT_INT = 20


CHART_VIEW_WIDTH_INT = 600
CHART_VIEW_HEIGHT_INT = 170
CHART_PLOT_LEFT_INT = 58
CHART_PLOT_RIGHT_INT = 8
CHART_PLOT_TOP_INT = 10
CHART_PLOT_BOTTOM_INT = 24
# Daily-P&L panel (IBKR-style green/red bars around a centered zero line). Its
# own viewBox with margins for a real value axis and date labels.
DAILY_PANEL_VIEW_HEIGHT_INT = 96
DAILY_PANEL_PLOT_TOP_INT = 10
DAILY_PANEL_PLOT_BOTTOM_INT = 20
# Cap the daily bar width (viewBox units) so a handful of EOD points render as
# distinct bars instead of merging into one fat slab.
MAX_PNL_BAR_WIDTH_FLOAT = 14.0
SUPPORTED_WINDOW_STR_LIST = ["1w", "mtd", "ytd", "all"]
# The headline curve can be read two equivalent ways from the SAME equity
# series: cumulative return percent (default, comparable across pods) or
# cumulative dollar P&L since the first visible sample.
SUPPORTED_VALUE_MODE_STR_LIST = ["pct", "dollar"]

_MONTH_ABBREVIATION_STR_LIST = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@dataclass
class EquityChartDict:
    point_count_int: int = 0
    has_curve_bool: bool = False
    return_unavailable_bool: bool = False
    return_unavailable_reason_str: str = ""
    path_d_str: str = ""
    curve_area_d_str: str = ""
    drawdown_d_str: str = ""
    zero_y_float: float = 0.0
    y_axis_tick_dict_list: list[dict[str, Any]] = field(default_factory=list)
    x_axis_tick_dict_list: list[dict[str, Any]] = field(default_factory=list)
    point_dict_list: list[dict[str, Any]] = field(default_factory=list)
    pnl_bar_dict_list: list[dict[str, Any]] = field(default_factory=list)
    daily_y_axis_tick_dict_list: list[dict[str, Any]] = field(default_factory=list)
    daily_zero_y_float: float = 0.0
    # Two understated risk readouts shown as a small footnote under the curve.
    max_drawdown_label_str: str = "—"
    annualized_vol_label_str: str = "—"
    vol_observation_count_int: int = 0
    range_min_float: float = 0.0
    range_max_float: float = 0.0
    range_min_label_str: str = "—"
    range_max_label_str: str = "—"
    latest_equity_float: float | None = None
    latest_market_date_str: str | None = None
    earliest_market_date_str: str | None = None
    latest_since_start_pnl_label_str: str = "—"
    latest_since_start_return_label_str: str = "—"
    latest_daily_pnl_label_str: str = "—"
    latest_daily_pct_label_str: str = "—"
    latest_daily_is_positive_bool: bool = True
    window_str: str = "all"
    window_is_partial_bool: bool = False
    window_note_str: str = ""
    value_mode_str: str = "pct"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_count_int": self.point_count_int,
            "has_curve_bool": self.has_curve_bool,
            "return_unavailable_bool": self.return_unavailable_bool,
            "return_unavailable_reason_str": self.return_unavailable_reason_str,
            "path_d_str": self.path_d_str,
            "curve_area_d_str": self.curve_area_d_str,
            "drawdown_d_str": self.drawdown_d_str,
            "zero_y_float": self.zero_y_float,
            "y_axis_tick_dict_list": self.y_axis_tick_dict_list,
            "x_axis_tick_dict_list": self.x_axis_tick_dict_list,
            "point_dict_list": self.point_dict_list,
            "pnl_bar_dict_list": self.pnl_bar_dict_list,
            "daily_y_axis_tick_dict_list": self.daily_y_axis_tick_dict_list,
            "daily_zero_y_float": self.daily_zero_y_float,
            "max_drawdown_label_str": self.max_drawdown_label_str,
            "annualized_vol_label_str": self.annualized_vol_label_str,
            "vol_observation_count_int": self.vol_observation_count_int,
            "vol_window_session_count_int": ROLLING_VOL_SESSION_COUNT_INT,
            "range_min_float": self.range_min_float,
            "range_max_float": self.range_max_float,
            "range_min_label_str": self.range_min_label_str,
            "range_max_label_str": self.range_max_label_str,
            "latest_equity_float": self.latest_equity_float,
            "latest_market_date_str": self.latest_market_date_str,
            "earliest_market_date_str": self.earliest_market_date_str,
            "latest_since_start_pnl_label_str": self.latest_since_start_pnl_label_str,
            "latest_since_start_return_label_str": self.latest_since_start_return_label_str,
            "latest_daily_pnl_label_str": self.latest_daily_pnl_label_str,
            "latest_daily_pct_label_str": self.latest_daily_pct_label_str,
            "latest_daily_is_positive_bool": self.latest_daily_is_positive_bool,
            "window_str": self.window_str,
            "window_is_partial_bool": self.window_is_partial_bool,
            "window_note_str": self.window_note_str,
            "value_mode_str": self.value_mode_str,
            "width_int": CHART_VIEW_WIDTH_INT,
            "height_int": CHART_VIEW_HEIGHT_INT,
            "plot_left_int": CHART_PLOT_LEFT_INT,
            "plot_right_int": CHART_VIEW_WIDTH_INT - CHART_PLOT_RIGHT_INT,
            "plot_top_int": CHART_PLOT_TOP_INT,
            "plot_bottom_int": CHART_VIEW_HEIGHT_INT - CHART_PLOT_BOTTOM_INT,
            "daily_panel_height_int": DAILY_PANEL_VIEW_HEIGHT_INT,
            "daily_panel_plot_left_int": CHART_PLOT_LEFT_INT,
            "daily_panel_plot_right_int": CHART_VIEW_WIDTH_INT - CHART_PLOT_RIGHT_INT,
            "daily_panel_plot_top_int": DAILY_PANEL_PLOT_TOP_INT,
            "daily_panel_plot_bottom_int": DAILY_PANEL_VIEW_HEIGHT_INT - DAILY_PANEL_PLOT_BOTTOM_INT,
        }


def build_equity_chart_dict(
    equity_point_dict_list: list[dict[str, Any]] | None,
    *,
    window_str: str = "all",
    value_mode_str: str = "pct",
) -> EquityChartDict:
    if value_mode_str not in SUPPORTED_VALUE_MODE_STR_LIST:
        value_mode_str = "pct"
    if not equity_point_dict_list:
        return EquityChartDict(window_str=window_str, value_mode_str=value_mode_str)

    valid_equity_point_dict_list = [
        point_dict
        for point_dict in equity_point_dict_list
        if _float_or_none(point_dict.get("equity_float")) is not None
    ]
    if not valid_equity_point_dict_list:
        return EquityChartDict(window_str=window_str, value_mode_str=value_mode_str)
    window_str, window_is_partial_bool, window_note_str = _resolve_chart_window_context(
        valid_equity_point_dict_list,
        window_str,
    )
    clean_point_list = _truncate_for_window(
        valid_equity_point_dict_list,
        window_str,
    )
    equity_pairs_list = [
        (
            str(point_dict.get("market_date_str") or ""),
            _float_or_none(point_dict.get("equity_float")),
            _float_or_none(point_dict.get("daily_pnl_float")),
            float(point_dict.get("flow_float") or 0.0),
        )
        for point_dict in clean_point_list
    ]
    daily_return_pct_list = _flow_adjusted_daily_return_pct_list(clean_point_list)
    point_count_int = len(equity_pairs_list)
    if point_count_int == 0:
        return EquityChartDict(window_str=window_str, value_mode_str=value_mode_str)
    if point_count_int == 1:
        only_date_str, only_equity_float, _, _ = equity_pairs_list[0]
        return EquityChartDict(
            point_count_int=1,
            has_curve_bool=False,
            range_min_float=only_equity_float or 0.0,
            range_max_float=only_equity_float or 0.0,
            range_min_label_str=_format_money_str(only_equity_float),
            range_max_label_str=_format_money_str(only_equity_float),
            latest_equity_float=only_equity_float,
            latest_market_date_str=only_date_str,
            earliest_market_date_str=only_date_str,
            window_str=window_str,
            window_is_partial_bool=window_is_partial_bool,
            window_note_str=window_note_str,
            value_mode_str=value_mode_str,
        )

    return_unavailable_bool = (
        float(equity_pairs_list[0][1] or 0.0) == 0.0
        or any(
            daily_return_pct_float is None
            for daily_return_pct_float in daily_return_pct_list[1:]
        )
    )
    if return_unavailable_bool and value_mode_str == "pct":
        return EquityChartDict(
            point_count_int=point_count_int,
            has_curve_bool=False,
            return_unavailable_bool=True,
            return_unavailable_reason_str=(
                "Return unavailable because the EOD series contains a zero "
                "capital base or a non-consecutive/ambiguous interval."
            ),
            latest_equity_float=equity_pairs_list[-1][1],
            latest_market_date_str=equity_pairs_list[-1][0],
            earliest_market_date_str=equity_pairs_list[0][0],
            window_str=window_str,
            window_is_partial_bool=window_is_partial_bool,
            window_note_str=window_note_str,
            value_mode_str=value_mode_str,
        )

    equity_value_list = [pair[1] for pair in equity_pairs_list]
    first_equity_float = float(equity_value_list[0] or 0.0)
    # Flow-adjusted cumulatives: declared deposits/withdrawals are stripped
    # so the headline percent matches the time-weighted stat strip — two
    # different "cumulative %" on one screen is how trust dies. Flows on the
    # first visible sample are baseline capital, not P&L. With no declared
    # flows both series equal the old equity-ratio math exactly.
    cumulative_pnl_value_list: list[float] = []
    cumulative_return_pct_list: list[float] = []
    running_flow_float = 0.0
    growth_float = 1.0
    twr_valid_bool = not return_unavailable_bool
    previous_equity_float: float | None = None
    for index_int, (_, equity_obj, _, flow_float) in enumerate(equity_pairs_list):
        equity_float = float(equity_obj or 0.0)
        if previous_equity_float is None:
            flow_float = 0.0
        else:
            running_flow_float += flow_float
            daily_return_pct_float = daily_return_pct_list[index_int]
            if twr_valid_bool and daily_return_pct_float is not None:
                growth_float *= 1.0 + daily_return_pct_float
        cumulative_pnl_value_list.append(
            equity_float - first_equity_float - running_flow_float
        )
        cumulative_return_pct_list.append(growth_float - 1.0 if first_equity_float else 0.0)
        previous_equity_float = equity_float
    # The %/$ toggle only decides which already-computed cumulative series drives
    # the y-geometry and the labels. Both are derived from the same equity series
    # with no future information, so switching modes can never alter the curve's
    # shape or introduce lookahead — it is a pure display choice.
    is_dollar_mode_bool = value_mode_str == "dollar"
    active_value_list = (
        cumulative_pnl_value_list if is_dollar_mode_bool else cumulative_return_pct_list
    )
    format_value_fn = (
        _format_signed_money_str if is_dollar_mode_bool else _format_signed_pct_str
    )

    range_min_float = min(0.0, min(active_value_list))
    range_max_float = max(0.0, max(active_value_list))
    if abs(range_max_float - range_min_float) < 1e-9:
        range_min_float -= 0.0001
        range_max_float += 0.0001
    value_range_float = max(1e-9, float(range_max_float) - float(range_min_float))

    plot_width_float = CHART_VIEW_WIDTH_INT - CHART_PLOT_LEFT_INT - CHART_PLOT_RIGHT_INT
    plot_height_float = CHART_VIEW_HEIGHT_INT - CHART_PLOT_TOP_INT - CHART_PLOT_BOTTOM_INT
    horizontal_step_float = plot_width_float / max(1, point_count_int - 1)

    point_xy_list: list[tuple[float, float]] = []
    point_dict_list: list[dict[str, Any]] = []
    for index_int, (date_str, equity_float, pnl_float, _) in enumerate(equity_pairs_list):
        cumulative_pnl_float = cumulative_pnl_value_list[index_int]
        cumulative_return_pct_float = cumulative_return_pct_list[index_int]
        active_value_float = active_value_list[index_int]
        x_float = CHART_PLOT_LEFT_INT + index_int * horizontal_step_float
        y_float = (
            CHART_VIEW_HEIGHT_INT
            - CHART_PLOT_BOTTOM_INT
            - ((active_value_float - range_min_float) / value_range_float) * plot_height_float
        )
        point_xy_list.append((x_float, y_float))
        daily_pct_float = daily_return_pct_list[index_int]
        point_dict_list.append({
            "x_float": round(x_float, 2),
            "y_float": round(y_float, 2),
            "market_date_str": date_str,
            "equity_label_str": _format_money_str(equity_float),
            "cumulative_pnl_label_str": _format_signed_money_str(cumulative_pnl_float),
            "cumulative_return_label_str": _format_signed_pct_str(cumulative_return_pct_float),
            "daily_pnl_label_str": _format_signed_money_str(pnl_float) if pnl_float is not None else "—",
            "daily_pct_label_str": _format_signed_pct_str(daily_pct_float),
            "daily_pct_float": daily_pct_float,
        })

    path_d_str = _build_polyline_path_str(point_xy_list)
    zero_y_float = _y_for_chart_value_float(0.0, range_min_float, value_range_float)
    curve_area_d_str = _build_curve_area_path_str(point_xy_list, zero_y_float)
    daily_panel_dict = _build_daily_panel_dict(
        equity_pairs_list,
        daily_return_pct_list,
        point_count_int,
        is_dollar_mode_bool,
    )
    y_axis_tick_dict_list = _build_y_axis_tick_dict_list(
        range_min_float, range_max_float, format_value_fn
    )
    x_axis_tick_dict_list = _build_x_axis_tick_dict_list(equity_pairs_list, point_xy_list)
    risk_footnote_dict = (
        {
            "max_drawdown_label_str": "—",
            "annualized_vol_label_str": "—",
            "vol_observation_count_int": 0,
        }
        if return_unavailable_bool
        else _build_risk_footnote_dict(
            cumulative_return_pct_list,
            daily_return_pct_list,
        )
    )

    latest_pnl_float = equity_pairs_list[-1][2]
    latest_since_start_pnl_float = cumulative_pnl_value_list[-1]
    latest_since_start_return_pct_float = cumulative_return_pct_list[-1]
    latest_daily_pct_float = point_dict_list[-1]["daily_pct_float"]

    return EquityChartDict(
        point_count_int=point_count_int,
        has_curve_bool=True,
        path_d_str=path_d_str,
        curve_area_d_str=curve_area_d_str,
        drawdown_d_str="",
        zero_y_float=zero_y_float,
        y_axis_tick_dict_list=y_axis_tick_dict_list,
        x_axis_tick_dict_list=x_axis_tick_dict_list,
        point_dict_list=point_dict_list,
        pnl_bar_dict_list=daily_panel_dict["bar_dict_list"],
        daily_y_axis_tick_dict_list=daily_panel_dict["y_axis_tick_dict_list"],
        daily_zero_y_float=daily_panel_dict["zero_y_float"],
        max_drawdown_label_str=risk_footnote_dict["max_drawdown_label_str"],
        annualized_vol_label_str=risk_footnote_dict["annualized_vol_label_str"],
        vol_observation_count_int=risk_footnote_dict["vol_observation_count_int"],
        range_min_float=float(range_min_float),
        range_max_float=float(range_max_float),
        range_min_label_str=format_value_fn(range_min_float),
        range_max_label_str=format_value_fn(range_max_float),
        latest_equity_float=equity_value_list[-1],
        latest_market_date_str=equity_pairs_list[-1][0],
        earliest_market_date_str=equity_pairs_list[0][0],
        latest_since_start_pnl_label_str=_format_signed_money_str(latest_since_start_pnl_float),
        latest_since_start_return_label_str=(
            _format_signed_pct_str(latest_since_start_return_pct_float)
            if twr_valid_bool
            else "—"
        ),
        latest_daily_pnl_label_str=_format_signed_money_str(latest_pnl_float),
        latest_daily_pct_label_str=_format_signed_pct_str(latest_daily_pct_float),
        latest_daily_is_positive_bool=(latest_daily_pct_float or 0.0) >= 0,
        window_str=window_str,
        window_is_partial_bool=window_is_partial_bool,
        window_note_str=window_note_str,
        value_mode_str=value_mode_str,
    )


# ── private helpers ───────────────────────────────────────────────────────


def _resolve_chart_window_context(
    equity_point_dict_list: list[dict[str, Any]],
    window_str: str,
) -> tuple[str, bool, str]:
    if window_str == "1w":
        partial_bool = len(equity_point_dict_list) < 6
        return (
            window_str,
            partial_bool,
            "Partial window · fewer than five return intervals."
            if partial_bool
            else "",
        )
    if window_str not in ("mtd", "ytd"):
        return window_str, False, ""
    try:
        latest_date_obj = date.fromisoformat(
            str(equity_point_dict_list[-1].get("market_date_str") or "")
        )
    except ValueError:
        return "all", False, "Requested period unavailable · invalid EOD date."
    first_period_index_int = next(
        (
            index_int
            for index_int, point_dict in enumerate(equity_point_dict_list)
            if _point_is_in_chart_period_bool(point_dict, latest_date_obj, window_str)
        ),
        0,
    )
    partial_bool = first_period_index_int == 0
    return (
        window_str,
        partial_bool,
        "Partial period · prior EOD baseline unavailable."
        if partial_bool
        else "",
    )


def _point_is_in_chart_period_bool(
    point_dict: dict[str, Any],
    latest_date_obj: date,
    window_str: str,
) -> bool:
    try:
        point_date_obj = date.fromisoformat(
            str(point_dict.get("market_date_str") or "")
        )
    except ValueError:
        return False
    return point_date_obj.year == latest_date_obj.year and (
        window_str == "ytd" or point_date_obj.month == latest_date_obj.month
    )


def _truncate_for_window(
    equity_point_dict_list: list[dict[str, Any]], window_str: str
) -> list[dict[str, Any]]:
    if window_str == "1w":
        # One baseline plus five observed sessions gives five return intervals.
        return equity_point_dict_list[-6:]
    if window_str not in ("mtd", "ytd") or not equity_point_dict_list:
        return equity_point_dict_list
    try:
        latest_date_obj = date.fromisoformat(
            str(equity_point_dict_list[-1].get("market_date_str") or "")
        )
    except ValueError:
        return equity_point_dict_list
    first_period_index_int = 0
    for index_int, point_dict in enumerate(equity_point_dict_list):
        if _point_is_in_chart_period_bool(point_dict, latest_date_obj, window_str):
            first_period_index_int = index_int
            break
    # Keep the previous EOD as the period baseline when it exists.
    return equity_point_dict_list[max(0, first_period_index_int - 1):]


def _build_polyline_path_str(point_xy_list: list[tuple[float, float]]) -> str:
    if not point_xy_list:
        return ""
    parts_list = []
    for index_int, (x_float, y_float) in enumerate(point_xy_list):
        prefix_str = "M" if index_int == 0 else "L"
        parts_list.append(f"{prefix_str} {x_float:.2f} {y_float:.2f}")
    return " ".join(parts_list)


def _build_curve_area_path_str(
    point_xy_list: list[tuple[float, float]],
    zero_y_float: float,
) -> str:
    if len(point_xy_list) < 2:
        return ""
    parts_list = []
    for index_int, (x_float, y_float) in enumerate(point_xy_list):
        prefix_str = "M" if index_int == 0 else "L"
        parts_list.append(f"{prefix_str} {x_float:.2f} {y_float:.2f}")
    last_x_float = point_xy_list[-1][0]
    first_x_float = point_xy_list[0][0]
    parts_list.append(f"L {last_x_float:.2f} {zero_y_float:.2f}")
    parts_list.append(f"L {first_x_float:.2f} {zero_y_float:.2f}")
    parts_list.append("Z")
    return " ".join(parts_list)


def _y_for_chart_value_float(
    value_float: float,
    range_min_float: float,
    value_range_float: float,
) -> float:
    plot_height_float = CHART_VIEW_HEIGHT_INT - CHART_PLOT_TOP_INT - CHART_PLOT_BOTTOM_INT
    return round(
        CHART_VIEW_HEIGHT_INT
        - CHART_PLOT_BOTTOM_INT
        - ((value_float - range_min_float) / value_range_float) * plot_height_float,
        2,
    )


def _build_y_axis_tick_dict_list(
    range_min_float: float,
    range_max_float: float,
    format_value_fn,
) -> list[dict[str, Any]]:
    value_range_float = max(1e-9, range_max_float - range_min_float)
    tick_value_list = [
        range_max_float,
        range_min_float + value_range_float / 2.0,
        range_min_float,
    ]
    return [
        {
            "value_float": value_float,
            "y_float": _y_for_chart_value_float(value_float, range_min_float, value_range_float),
            "label_str": format_value_fn(value_float),
        }
        for value_float in tick_value_list
    ]


def _build_risk_footnote_dict(
    cumulative_return_pct_list: list[float],
    daily_return_pct_list: list[float | None],
) -> dict[str, Any]:
    """Two understated risk readouts for the footnote under the curve.

    Max drawdown is the deepest dip below the running peak; *** CRITICAL*** the
    peak only looks at sessions up to and including each point, never ahead of it.
    Volatility is the sample standard deviation of daily returns annualized by
    ``sqrt(252)`` — the same convention as ``build_book_risk_dict``.
    """
    performance_index_list = [
        1.0 + cumulative_return_pct_float
        for cumulative_return_pct_float in cumulative_return_pct_list
    ]
    max_drawdown_float = 0.0
    running_peak_float = performance_index_list[0]
    for performance_index_float in performance_index_list:
        if performance_index_float > running_peak_float:
            running_peak_float = performance_index_float
        if running_peak_float:
            drawdown_float = (performance_index_float / running_peak_float) - 1.0
            if drawdown_float < max_drawdown_float:
                max_drawdown_float = drawdown_float
    max_drawdown_label_str = (
        f"{max_drawdown_float * 100:.2f}%" if max_drawdown_float < -1e-9 else "0.00%"
    )

    valid_daily_return_pct_list = [
        daily_return_pct_float
        for daily_return_pct_float in daily_return_pct_list[1:]
        if daily_return_pct_float is not None
    ]
    # *** CRITICAL *** backward-only risk window: only completed returns at or
    # before the latest displayed EOD can enter the rolling statistic.
    rolling_daily_return_pct_list = valid_daily_return_pct_list[
        -ROLLING_VOL_SESSION_COUNT_INT:
    ]
    annualized_vol_label_str = "—"
    if len(rolling_daily_return_pct_list) >= 2:
        daily_vol_float = statistics.stdev(rolling_daily_return_pct_list)
        annualized_vol_label_str = (
            f"{daily_vol_float * math.sqrt(TRADING_DAYS_PER_YEAR_INT) * 100:.1f}%"
        )

    return {
        "max_drawdown_label_str": max_drawdown_label_str,
        "annualized_vol_label_str": annualized_vol_label_str,
        "vol_observation_count_int": len(rolling_daily_return_pct_list),
    }


def _build_x_axis_tick_dict_list(
    equity_pairs_list: list[tuple[str, float | None, float | None]],
    point_xy_list: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    if not equity_pairs_list or not point_xy_list:
        return []
    candidate_index_list = sorted({0, len(equity_pairs_list) // 2, len(equity_pairs_list) - 1})
    tick_dict_list: list[dict[str, Any]] = []
    for position_int, index_int in enumerate(candidate_index_list):
        if position_int == 0:
            text_anchor_str = "start"
        elif position_int == len(candidate_index_list) - 1:
            text_anchor_str = "end"
        else:
            text_anchor_str = "middle"
        tick_dict_list.append(
            {
                "x_float": round(point_xy_list[index_int][0], 2),
                "date_str": equity_pairs_list[index_int][0],
                "label_str": _format_short_date_label_str(equity_pairs_list[index_int][0]),
                "text_anchor_str": text_anchor_str,
            }
        )
    return tick_dict_list


def _build_daily_panel_dict(
    equity_pairs_list: list[tuple[str, float | None, float | None, float]],
    daily_return_pct_list: list[float | None],
    point_count_int: int,
    is_dollar_mode_bool: bool,
) -> dict[str, Any]:
    """IBKR-style daily-P&L panel: green/red bars around a centered zero line.

    Each day's value follows the active %/$ toggle — the daily dollar P&L in
    ``$`` mode, or the day-over-day equity return in ``%`` mode. Bars are scaled
    against the largest absolute day so the axis is symmetric (+max at the top,
    -max at the bottom). The panel reuses the equity panel's left/right margins so
    its bars line up under the curve and share the same date ticks.
    """
    empty_dict: dict[str, Any] = {
        "bar_dict_list": [],
        "y_axis_tick_dict_list": [],
        "zero_y_float": round(
            DAILY_PANEL_PLOT_TOP_INT
            + (DAILY_PANEL_VIEW_HEIGHT_INT - DAILY_PANEL_PLOT_TOP_INT - DAILY_PANEL_PLOT_BOTTOM_INT) / 2.0,
            2,
        ),
    }
    format_value_fn = _format_signed_money_str if is_dollar_mode_bool else _format_signed_pct_str
    daily_value_list: list[float | None] = []
    for index_int, (_date_str, _equity_float, pnl_float, _flow_float) in enumerate(equity_pairs_list):
        if is_dollar_mode_bool:
            daily_value_list.append(_float_or_none(pnl_float))
        else:
            daily_value_list.append(daily_return_pct_list[index_int])
    valid_value_list = [value for value in daily_value_list if value is not None]
    if not valid_value_list:
        return empty_dict
    max_abs_float = max(abs(value) for value in valid_value_list)
    if max_abs_float <= 0:
        return empty_dict

    plot_height_float = (
        DAILY_PANEL_VIEW_HEIGHT_INT - DAILY_PANEL_PLOT_TOP_INT - DAILY_PANEL_PLOT_BOTTOM_INT
    )
    zero_y_float = DAILY_PANEL_PLOT_TOP_INT + plot_height_float / 2.0
    half_height_float = plot_height_float / 2.0 - 2.0
    plot_width_float = CHART_VIEW_WIDTH_INT - CHART_PLOT_LEFT_INT - CHART_PLOT_RIGHT_INT
    horizontal_step_float = plot_width_float / max(1, point_count_int - 1)
    bar_width_float = max(0.8, min(horizontal_step_float * 0.7, MAX_PNL_BAR_WIDTH_FLOAT))

    bar_dict_list: list[dict[str, Any]] = []
    for index_int, daily_value_float in enumerate(daily_value_list):
        if daily_value_float is None:
            continue
        bar_height_float = abs(float(daily_value_float)) / max_abs_float * half_height_float
        is_positive_bool = float(daily_value_float) >= 0
        center_x_float = CHART_PLOT_LEFT_INT + index_int * horizontal_step_float
        bar_x_float = min(
            max(float(CHART_PLOT_LEFT_INT), center_x_float - bar_width_float / 2.0),
            CHART_VIEW_WIDTH_INT - CHART_PLOT_RIGHT_INT - bar_width_float,
        )
        bar_dict_list.append({
            "x_float": round(bar_x_float, 2),
            "y_float": round(zero_y_float - bar_height_float if is_positive_bool else zero_y_float, 2),
            "width_float": round(bar_width_float, 2),
            "height_float": round(max(0.5, bar_height_float), 2),
            "is_positive_bool": is_positive_bool,
            "market_date_str": equity_pairs_list[index_int][0],
            "pnl_label_str": format_value_fn(daily_value_float),
        })

    y_axis_tick_dict_list = [
        {"y_float": round(zero_y_float - half_height_float, 2), "label_str": format_value_fn(max_abs_float)},
        {"y_float": round(zero_y_float, 2), "label_str": format_value_fn(0.0)},
        {"y_float": round(zero_y_float + half_height_float, 2), "label_str": format_value_fn(-max_abs_float)},
    ]
    return {
        "bar_dict_list": bar_dict_list,
        "y_axis_tick_dict_list": y_axis_tick_dict_list,
        "zero_y_float": round(zero_y_float, 2),
    }


def _float_or_none(value_obj: Any) -> float | None:
    if value_obj is None:
        return None
    try:
        return float(value_obj)
    except (TypeError, ValueError):
        return None


def _flow_adjusted_daily_return_pct_list(
    equity_point_dict_list: list[dict[str, Any]],
) -> list[float | None]:
    """One aligned daily-return series for every dashboard consumer.

    ``daily_pnl_pct_float`` from the EOD accounting layer is authoritative.
    The fallback preserves compatibility with older fixtures and artifacts,
    while still stripping any declared interval flow.
    """
    daily_return_pct_list: list[float | None] = []
    previous_equity_float: float | None = None
    for point_dict in equity_point_dict_list:
        equity_float = _float_or_none(point_dict.get("equity_float"))
        if equity_float is None or previous_equity_float in (None, 0.0):
            daily_return_pct_list.append(None)
        else:
            if "daily_pnl_pct_float" in point_dict:
                # An explicit None means accounting declared this interval
                # unavailable; never replace it with a prettier fallback.
                daily_return_pct_list.append(
                    _float_or_none(point_dict.get("daily_pnl_pct_float"))
                )
            else:
                flow_float = float(point_dict.get("flow_float") or 0.0)
                # *** CRITICAL*** This is the backward-compatible EOD return:
                # r_t = (E_t - F_interval) / E_(t-1) - 1. Never use E_t/E_(t-1)
                # when a declared external flow exists in the interval.
                daily_return_pct_list.append(
                    ((equity_float - flow_float) / previous_equity_float) - 1.0
                )
        previous_equity_float = equity_float
    return daily_return_pct_list


def _format_money_str(value_obj: Any) -> str:
    value_float = _float_or_none(value_obj)
    if value_float is None:
        return "—"
    sign_str = "-" if value_float < 0 else ""
    return f"{sign_str}${abs(value_float):,.0f}"


def _format_signed_money_str(value_obj: Any) -> str:
    value_float = _float_or_none(value_obj)
    if value_float is None:
        return "—"
    sign_str = "+" if value_float >= 0 else "-"
    return f"{sign_str}${abs(value_float):,.0f}"


def _format_signed_pct_str(pct_float: float | None) -> str:
    if pct_float is None:
        return "—"
    return f"{pct_float * 100:+.2f}%"


def _format_month_label_str(year_month_str: str) -> str:
    try:
        year_str, month_str = year_month_str.split("-")[:2]
        return f"{_MONTH_ABBREVIATION_STR_LIST[int(month_str)]} {year_str[2:]}"
    except (ValueError, IndexError):
        return year_month_str


def _format_short_date_label_str(date_str: str) -> str:
    try:
        _year_str, month_str, day_str = date_str.split("-")[:3]
        return f"{_MONTH_ABBREVIATION_STR_LIST[int(month_str)]} {int(day_str)}"
    except (ValueError, IndexError):
        return date_str


def build_monthly_return_dict_list(
    equity_point_dict_list: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Per-calendar-month return % from the EOD equity series.

    Each month compounds the same flow-adjusted daily returns used by the
    headline and daily panel. Read-only reporting — no forecasting.
    """
    clean_point_list = [
        point_dict
        for point_dict in (equity_point_dict_list or [])
        if _float_or_none(point_dict.get("equity_float")) is not None
        and len(str(point_dict.get("market_date_str") or "")) >= 7
    ]
    if not clean_point_list:
        return []

    daily_return_pct_list = _flow_adjusted_daily_return_pct_list(clean_point_list)
    monthly_growth_dict: dict[str, float] = {}
    monthly_available_dict: dict[str, bool] = {}
    month_key_order_list: list[str] = []
    return_history_available_bool = (
        float(clean_point_list[0].get("equity_float") or 0.0) != 0.0
    )
    for point_index_int, (point_dict, daily_return_pct_float) in enumerate(
        zip(clean_point_list, daily_return_pct_list, strict=True)
    ):
        date_str = str(point_dict.get("market_date_str") or "")
        month_key_str = date_str[:7]
        if month_key_str not in monthly_growth_dict:
            month_key_order_list.append(month_key_str)
            monthly_growth_dict[month_key_str] = 1.0
            monthly_available_dict[month_key_str] = return_history_available_bool
        if point_index_int > 0 and daily_return_pct_float is None:
            return_history_available_bool = False
        monthly_available_dict[month_key_str] = (
            monthly_available_dict[month_key_str]
            and return_history_available_bool
        )
        if return_history_available_bool and daily_return_pct_float is not None:
            monthly_growth_dict[month_key_str] *= 1.0 + daily_return_pct_float

    monthly_return_dict_list: list[dict[str, Any]] = []
    for month_key_str in month_key_order_list:
        is_available_bool = monthly_available_dict[month_key_str]
        return_pct_float = (
            monthly_growth_dict[month_key_str] - 1.0
            if is_available_bool
            else None
        )
        monthly_return_dict_list.append(
            {
                "month_label_str": _format_month_label_str(month_key_str),
                "return_pct_float": return_pct_float,
                "return_label_str": _format_signed_pct_str(return_pct_float),
                "is_positive_bool": (
                    return_pct_float is not None and return_pct_float >= 0
                ),
                "is_available_bool": is_available_bool,
            }
        )
    return monthly_return_dict_list


# ── allocation pie ─────────────────────────────────────────────────────────
#
# A point-in-time composition of one mode's book across the strategies (pods)
# currently running. Each slice is a pod sized by its net-liquidation equity
# (positions + its own cash), so slices sum to the mode's total book and every
# running strategy stays visible at its true size. Cash is *not* a slice — it is
# reported separately as a share of the book so the operator can still see how
# much of the book is idle. Pure SVG, no JS — matches build_equity_chart_dict.

PIE_VIEW_SIZE_INT = 100
PIE_CENTER_FLOAT = 50.0
PIE_RADIUS_FLOAT = 46.0

# Distinct, light-theme-friendly slice colors, cycled if there are more pods.
ALLOCATION_PALETTE_STR_LIST = [
    "#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626",
    "#0891b2", "#db2777", "#65a30d", "#475569", "#ea580c",
]

# Each pod is split into two slices that share its base color: the invested
# positions slice is solid, the cash slice is the same color drawn lighter so
# the two read as one pod at a glance.
POSITIONS_SLICE_OPACITY_STR = "1"
CASH_SLICE_OPACITY_STR = "0.4"


@dataclass
class AllocationPieDict:
    has_data_bool: bool = False
    pod_count_int: int = 0
    slice_dict_list: list[dict[str, Any]] = field(default_factory=list)
    total_equity_float: float = 0.0
    total_equity_label_str: str = "—"
    total_cash_float: float = 0.0
    total_cash_label_str: str = "—"
    cash_pct_label_str: str = "—"
    excluded_pod_count_int: int = 0
    clamped_pod_count_int: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_data_bool": self.has_data_bool,
            "pod_count_int": self.pod_count_int,
            "slice_dict_list": self.slice_dict_list,
            "total_equity_float": self.total_equity_float,
            "total_equity_label_str": self.total_equity_label_str,
            "total_cash_float": self.total_cash_float,
            "total_cash_label_str": self.total_cash_label_str,
            "cash_pct_label_str": self.cash_pct_label_str,
            "excluded_pod_count_int": self.excluded_pod_count_int,
            "clamped_pod_count_int": self.clamped_pod_count_int,
            "view_size_int": PIE_VIEW_SIZE_INT,
        }


def build_allocation_pie_dict(
    pod_alloc_dict_list: list[dict[str, Any]] | None,
) -> AllocationPieDict:
    """Build a strategy-allocation pie for one mode.

    Each input dict carries ``label_str``, optional ``sublabel_str``,
    ``equity_float`` (net liquidation = positions + cash) and ``cash_float``.

    Every pod contributes up to two slices: its invested *positions* value
    (``equity − cash``) and its *cash*. The two always sum to the pod's equity,
    so the pod keeps the same share of the book — we just split that share into
    what is at work in the market versus what is sitting in cash. Pods without a
    positive equity cannot occupy a slice and are excluded (but counted), since
    a pie can only render non-negative shares.
    """
    cleaned_dict_list: list[dict[str, Any]] = []
    excluded_pod_count_int = 0
    clamped_pod_count_int = 0
    total_cash_float = 0.0
    for pod_alloc_dict in pod_alloc_dict_list or []:
        equity_float = _float_or_none(pod_alloc_dict.get("equity_float"))
        cash_float = _float_or_none(pod_alloc_dict.get("cash_float")) or 0.0
        if equity_float is None or equity_float <= 0:
            excluded_pod_count_int += 1
            continue
        total_cash_float += cash_float  # true cash, may be negative on margin
        # *** CRITICAL*** A pie slice cannot be negative, so the cash drawn
        # inside the pie is clamped to [0, equity]; this keeps each pod's two
        # slices non-negative and summing exactly to its equity. The footer cash
        # readout below still uses the true (possibly negative) cash, so leverage
        # is never silently erased — clamped pods are counted and flagged.
        cash_in_pie_float = min(max(cash_float, 0.0), equity_float)
        if cash_in_pie_float != cash_float:
            clamped_pod_count_int += 1
        cleaned_dict_list.append({
            "label_str": str(pod_alloc_dict.get("label_str") or "—"),
            "sublabel_str": str(pod_alloc_dict.get("sublabel_str") or ""),
            "equity_float": equity_float,
            "cash_in_pie_float": cash_in_pie_float,
            "positions_float": equity_float - cash_in_pie_float,
        })

    if not cleaned_dict_list:
        return AllocationPieDict(excluded_pod_count_int=excluded_pod_count_int)

    # Largest pod first so colors + legend order are stable and readable.
    cleaned_dict_list.sort(key=lambda pod_dict: pod_dict["equity_float"], reverse=True)
    total_equity_float = sum(pod_dict["equity_float"] for pod_dict in cleaned_dict_list)

    # Expand each pod into its positions slice then its cash slice, sharing the
    # pod's base color. Zero-value parts are dropped so we never emit an
    # invisible zero-width arc (e.g. a fully-invested or all-cash pod).
    part_dict_list: list[dict[str, Any]] = []
    for pod_index_int, pod_dict in enumerate(cleaned_dict_list):
        color_str = ALLOCATION_PALETTE_STR_LIST[pod_index_int % len(ALLOCATION_PALETTE_STR_LIST)]
        if pod_dict["positions_float"] > 0:
            part_dict_list.append({
                "label_str": pod_dict["label_str"],
                "sublabel_str": pod_dict["sublabel_str"],
                "kind_str": "positions",
                "color_str": color_str,
                "fill_opacity_str": POSITIONS_SLICE_OPACITY_STR,
                "value_float": pod_dict["positions_float"],
            })
        if pod_dict["cash_in_pie_float"] > 0:
            part_dict_list.append({
                "label_str": pod_dict["label_str"],
                "sublabel_str": pod_dict["sublabel_str"],
                "kind_str": "cash",
                "color_str": color_str,
                "fill_opacity_str": CASH_SLICE_OPACITY_STR,
                "value_float": pod_dict["cash_in_pie_float"],
            })

    is_single_slice_bool = len(part_dict_list) == 1
    slice_dict_list: list[dict[str, Any]] = []
    cumulative_deg_float = 0.0
    for index_int, part_dict in enumerate(part_dict_list):
        fraction_float = (
            part_dict["value_float"] / total_equity_float if total_equity_float else 0.0
        )
        if is_single_slice_bool:
            path_d_str = ""
            is_full_circle_bool = True
        else:
            start_deg_float = cumulative_deg_float
            # Snap the final slice to a full 360° to absorb float drift.
            end_deg_float = (
                360.0
                if index_int == len(part_dict_list) - 1
                else cumulative_deg_float + fraction_float * 360.0
            )
            path_d_str = _pie_slice_path_str(start_deg_float, end_deg_float)
            is_full_circle_bool = False
            cumulative_deg_float = end_deg_float
        slice_dict_list.append({
            "label_str": part_dict["label_str"],
            "sublabel_str": part_dict["sublabel_str"],
            "kind_str": part_dict["kind_str"],
            "color_str": part_dict["color_str"],
            "fill_opacity_str": part_dict["fill_opacity_str"],
            "equity_float": part_dict["value_float"],
            "equity_label_str": _format_money_str(part_dict["value_float"]),
            "pct_float": fraction_float,
            "pct_label_str": f"{fraction_float * 100:.1f}%",
            "path_d_str": path_d_str,
            "is_full_circle_bool": is_full_circle_bool,
        })

    cash_pct_float = total_cash_float / total_equity_float if total_equity_float else 0.0
    return AllocationPieDict(
        has_data_bool=True,
        pod_count_int=len(cleaned_dict_list),
        slice_dict_list=slice_dict_list,
        total_equity_float=total_equity_float,
        total_equity_label_str=_format_money_str(total_equity_float),
        total_cash_float=total_cash_float,
        total_cash_label_str=_format_money_str(total_cash_float),
        cash_pct_label_str=f"{cash_pct_float * 100:.1f}%",
        excluded_pod_count_int=excluded_pod_count_int,
        clamped_pod_count_int=clamped_pod_count_int,
    )


def _pie_point_xy(angle_deg_float: float) -> tuple[float, float]:
    # 0° at the top, increasing clockwise (SVG y grows downward).
    angle_rad_float = math.radians(angle_deg_float - 90.0)
    return (
        PIE_CENTER_FLOAT + PIE_RADIUS_FLOAT * math.cos(angle_rad_float),
        PIE_CENTER_FLOAT + PIE_RADIUS_FLOAT * math.sin(angle_rad_float),
    )


def _pie_slice_path_str(start_deg_float: float, end_deg_float: float) -> str:
    start_x_float, start_y_float = _pie_point_xy(start_deg_float)
    end_x_float, end_y_float = _pie_point_xy(end_deg_float)
    large_arc_int = 1 if (end_deg_float - start_deg_float) > 180.0 else 0
    return (
        f"M {PIE_CENTER_FLOAT} {PIE_CENTER_FLOAT} "
        f"L {start_x_float:.3f} {start_y_float:.3f} "
        f"A {PIE_RADIUS_FLOAT} {PIE_RADIUS_FLOAT} 0 {large_arc_int} 1 "
        f"{end_x_float:.3f} {end_y_float:.3f} Z"
    )


# ── book risk strip ────────────────────────────────────────────────────────
#
# Realized risk reported straight from the combined-book EOD equity series — no
# forecasting, no strategy logic. Drawdown is measured from the running peak;
# volatility is the sample standard deviation of daily returns, annualized by
# sqrt(252) to match the house convention used elsewhere in the engine.


@dataclass
class BookRiskDict:
    has_data_bool: bool = False
    point_count_int: int = 0
    current_equity_label_str: str = "—"
    peak_equity_label_str: str = "—"
    peak_market_date_str: str | None = None
    current_drawdown_pct_float: float = 0.0
    current_drawdown_label_str: str = "—"
    is_underwater_bool: bool = False
    max_drawdown_label_str: str = "—"
    days_underwater_int: int = 0
    daily_vol_label_str: str = "—"
    annualized_vol_label_str: str = "—"
    vol_observation_count_int: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_data_bool": self.has_data_bool,
            "point_count_int": self.point_count_int,
            "current_equity_label_str": self.current_equity_label_str,
            "peak_equity_label_str": self.peak_equity_label_str,
            "peak_market_date_str": self.peak_market_date_str,
            "current_drawdown_pct_float": self.current_drawdown_pct_float,
            "current_drawdown_label_str": self.current_drawdown_label_str,
            "is_underwater_bool": self.is_underwater_bool,
            "max_drawdown_label_str": self.max_drawdown_label_str,
            "days_underwater_int": self.days_underwater_int,
            "daily_vol_label_str": self.daily_vol_label_str,
            "annualized_vol_label_str": self.annualized_vol_label_str,
            "vol_observation_count_int": self.vol_observation_count_int,
            "vol_window_session_count_int": ROLLING_VOL_SESSION_COUNT_INT,
        }


def build_book_risk_dict(
    equity_point_dict_list: list[dict[str, Any]] | None,
) -> BookRiskDict:
    clean_point_list = [
        point_dict
        for point_dict in (equity_point_dict_list or [])
        if _float_or_none(point_dict.get("equity_float")) is not None
    ]
    if not clean_point_list:
        return BookRiskDict()

    date_str_list = [
        str(point_dict.get("market_date_str") or "")
        for point_dict in clean_point_list
    ]
    equity_value_list = [
        float(point_dict["equity_float"])
        for point_dict in clean_point_list
    ]
    daily_return_pct_list = _flow_adjusted_daily_return_pct_list(clean_point_list)
    if (
        equity_value_list[0] == 0.0
        or any(
            daily_return_pct_float is None
            for daily_return_pct_float in daily_return_pct_list[1:]
        )
    ):
        return BookRiskDict()
    performance_index_list: list[float] = []
    growth_float = 1.0
    for daily_return_pct_float in daily_return_pct_list:
        if daily_return_pct_float is not None:
            growth_float *= 1.0 + daily_return_pct_float
        performance_index_list.append(growth_float)
    current_equity_float = equity_value_list[-1]

    # Flow-adjusted drawdown vs the running performance peak. *** CRITICAL***
    # the peak only sees sessions through each point, never future returns.
    running_peak_float = performance_index_list[0]
    overall_peak_float = running_peak_float
    overall_peak_index_int = 0
    overall_peak_date_str = date_str_list[0]
    max_drawdown_pct_float = 0.0
    last_high_index_int = 0
    for index_int, performance_index_float in enumerate(performance_index_list):
        if performance_index_float >= running_peak_float:
            running_peak_float = performance_index_float
            last_high_index_int = index_int
        drawdown_pct_float = (
            (running_peak_float - performance_index_float) / running_peak_float
        )
        if drawdown_pct_float > max_drawdown_pct_float:
            max_drawdown_pct_float = drawdown_pct_float
        if performance_index_float > overall_peak_float:
            overall_peak_float = performance_index_float
            overall_peak_index_int = index_int
            overall_peak_date_str = date_str_list[index_int]

    current_drawdown_pct_float = (
        (overall_peak_float - performance_index_list[-1]) / overall_peak_float
        if overall_peak_float > 0
        else 0.0
    )
    days_underwater_int = (len(clean_point_list) - 1) - last_high_index_int

    # Realized daily returns → sample stdev → annualized volatility.
    daily_return_list = [
        daily_return_pct_float
        for daily_return_pct_float in daily_return_pct_list[1:]
        if daily_return_pct_float is not None
    ]
    # *** CRITICAL *** backward-only rolling risk: use at most the latest 20
    # completed return intervals, never future observations.
    rolling_daily_return_list = daily_return_list[-ROLLING_VOL_SESSION_COUNT_INT:]
    daily_vol_label_str = "—"
    annualized_vol_label_str = "—"
    if len(rolling_daily_return_list) >= 2:
        daily_vol_float = statistics.stdev(rolling_daily_return_list)
        daily_vol_label_str = f"{daily_vol_float * 100:.2f}%"
        annualized_vol_label_str = (
            f"{daily_vol_float * math.sqrt(TRADING_DAYS_PER_YEAR_INT) * 100:.1f}%"
        )

    return BookRiskDict(
        has_data_bool=True,
        point_count_int=len(clean_point_list),
        current_equity_label_str=_format_money_str(current_equity_float),
        peak_equity_label_str=_format_money_str(
            equity_value_list[overall_peak_index_int]
        ),
        peak_market_date_str=overall_peak_date_str,
        current_drawdown_pct_float=current_drawdown_pct_float,
        current_drawdown_label_str=(
            "flat" if current_drawdown_pct_float <= 0 else f"-{current_drawdown_pct_float * 100:.2f}%"
        ),
        is_underwater_bool=current_drawdown_pct_float > 0,
        max_drawdown_label_str=f"-{max_drawdown_pct_float * 100:.2f}%" if max_drawdown_pct_float > 0 else "0.00%",
        days_underwater_int=days_underwater_int,
        daily_vol_label_str=daily_vol_label_str,
        annualized_vol_label_str=annualized_vol_label_str,
        vol_observation_count_int=len(rolling_daily_return_list),
    )


# ── cross-pod exposure ─────────────────────────────────────────────────────
#
# Nets each mode's positions by ticker ACROSS all pods, so overlapping (or
# offsetting) bets become visible. Each pod contributes valued positions
# (asset, signed shares, market value); we sum signed market value per ticker,
# then derive gross/net/long/short and leverage vs the book's equity. Read-only
# reporting — no quant logic. All values USD. Pure function, like the builders
# above; the per-pod valuation lives in
# alpha.live.dashboard.build_position_exposure_dict_list.


def _format_signed_share_str(share_float: float) -> str:
    sign_str = "+" if share_float >= 0 else "-"
    abs_share_float = abs(share_float)
    if abs_share_float >= 1000:
        return f"{sign_str}{abs_share_float:,.0f}"
    body_str = f"{abs_share_float:.2f}".rstrip("0").rstrip(".") or "0"
    return f"{sign_str}{body_str}"


def build_cross_pod_exposure_dict(
    pod_exposure_input_dict_list: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Net positions by ticker across a mode's pods.

    Each input dict: ``{pod_id_str, equity_float, position_exposure_dict_list}``
    where each position carries ``asset_str``, ``share_float`` and
    ``market_value_float`` (None when unpriced). Unpriced positions are counted
    but cannot enter the $-netting (surfaced, never silently dropped).
    """
    total_equity_float = 0.0
    unpriced_count_int = 0
    asset_accumulator_dict: dict[str, dict[str, Any]] = {}
    contributing_pod_id_set: set[str] = set()

    for pod_input_dict in pod_exposure_input_dict_list or []:
        pod_id_str = str(pod_input_dict.get("pod_id_str") or "?")
        equity_float = _float_or_none(pod_input_dict.get("equity_float"))
        if equity_float is not None and equity_float > 0:
            total_equity_float += equity_float
        for position_dict in pod_input_dict.get("position_exposure_dict_list") or []:
            if not position_dict.get("is_priced_bool"):
                unpriced_count_int += 1
                continue
            market_value_float = _float_or_none(position_dict.get("market_value_float"))
            share_float = _float_or_none(position_dict.get("share_float"))
            if market_value_float is None or share_float is None:
                unpriced_count_int += 1
                continue
            asset_str = str(position_dict.get("asset_str") or "?")
            accumulator_dict = asset_accumulator_dict.setdefault(
                asset_str,
                {"net_value_float": 0.0, "net_share_float": 0.0, "holder_dict_list": []},
            )
            accumulator_dict["net_value_float"] += market_value_float
            accumulator_dict["net_share_float"] += share_float
            accumulator_dict["holder_dict_list"].append(
                {
                    "pod_id_str": pod_id_str,
                    "share_float": share_float,
                    "share_label_str": _format_signed_share_str(share_float),
                    "value_label_str": _format_money_str(market_value_float),
                    "is_long_bool": share_float >= 0,
                }
            )
            contributing_pod_id_set.add(pod_id_str)

    if not asset_accumulator_dict:
        return {
            "has_data_bool": False,
            "unpriced_count_int": unpriced_count_int,
            "asset_row_dict_list": [],
        }

    asset_row_dict_list: list[dict[str, Any]] = []
    gross_value_float = 0.0
    net_value_float = 0.0
    long_value_float = 0.0
    short_value_float = 0.0
    for asset_str, accumulator_dict in asset_accumulator_dict.items():
        asset_net_value_float = accumulator_dict["net_value_float"]
        gross_value_float += abs(asset_net_value_float)
        net_value_float += asset_net_value_float
        if asset_net_value_float >= 0:
            long_value_float += asset_net_value_float
        else:
            short_value_float += asset_net_value_float
        has_long_holder_bool = any(h["share_float"] > 0 for h in accumulator_dict["holder_dict_list"])
        has_short_holder_bool = any(h["share_float"] < 0 for h in accumulator_dict["holder_dict_list"])
        concentration_pct_float = (
            abs(asset_net_value_float) / total_equity_float if total_equity_float > 0 else 0.0
        )
        asset_row_dict_list.append(
            {
                "asset_str": asset_str,
                "net_share_float": accumulator_dict["net_share_float"],
                "net_share_label_str": _format_signed_share_str(accumulator_dict["net_share_float"]),
                "net_value_float": asset_net_value_float,
                "net_value_label_str": _format_money_str(asset_net_value_float),
                "concentration_pct_float": concentration_pct_float,
                "concentration_label_str": f"{concentration_pct_float * 100:.1f}%",
                "pod_count_int": len(accumulator_dict["holder_dict_list"]),
                "is_long_bool": asset_net_value_float >= 0,
                "is_offset_bool": has_long_holder_bool and has_short_holder_bool,
                "holder_dict_list": accumulator_dict["holder_dict_list"],
            }
        )

    asset_row_dict_list.sort(key=lambda row_dict: abs(row_dict["net_value_float"]), reverse=True)
    leverage_label_str = (
        f"{gross_value_float / total_equity_float:.2f}x" if total_equity_float > 0 else "—"
    )

    return {
        "has_data_bool": True,
        "pod_count_int": len(contributing_pod_id_set),
        "asset_count_int": len(asset_row_dict_list),
        "total_equity_float": total_equity_float,
        "total_equity_label_str": _format_money_str(total_equity_float),
        "gross_value_label_str": _format_money_str(gross_value_float),
        "net_value_label_str": _format_money_str(net_value_float),
        "long_value_label_str": _format_money_str(long_value_float),
        "short_value_label_str": _format_money_str(short_value_float),
        "leverage_label_str": leverage_label_str,
        "unpriced_count_int": unpriced_count_int,
        "asset_row_dict_list": asset_row_dict_list,
    }


__all__ = [
    "CHART_VIEW_HEIGHT_INT",
    "CHART_VIEW_WIDTH_INT",
    "DAILY_PANEL_VIEW_HEIGHT_INT",
    "AllocationPieDict",
    "BookRiskDict",
    "EquityChartDict",
    "SUPPORTED_VALUE_MODE_STR_LIST",
    "SUPPORTED_WINDOW_STR_LIST",
    "build_allocation_pie_dict",
    "build_book_risk_dict",
    "build_cross_pod_exposure_dict",
    "build_equity_chart_dict",
    "build_monthly_return_dict_list",
]


# Muted stack tones mirroring the desk overlay cycle in alpha/engine/theme.py:
# the sleeves of one pod are parts of a whole, not competing series.
_HOLDINGS_COLOR_STR_LIST = [
    "#5a6943", "#8db388", "#5b8f70", "#436965",
    "#88a2b3", "#5b608f", "#524369", "#af88b3",
    "#8f5b7a", "#694347", "#b39988", "#8f895b",
]
_HOLDINGS_CASH_COLOR_STR = "#cbd5e1"


def build_pod_holdings_pie_dict(
    position_exposure_dict_list: list[dict[str, Any]] | None,
    cash_float: Any,
    equity_float: Any,
    target_weight_map_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One pod's current holdings as donut slices with target drift.

    Read-only presentation math over data the dashboard already carries:
    ``position_exposure_dict_list`` (asset, market value) from the pod row,
    ``cash_float``/``equity_float`` from the same snapshot, and the latest
    decision plan's target weights when available.

    *** CRITICAL*** Weights use absolute market value over the sum of
    absolute exposures plus clamped cash, so a short leg still occupies a
    visible share instead of shrinking the pie. Drift is
    ``current_weight - target_weight`` in percentage points and is only
    reported for priced assets when a target exists — an unpriced position
    shows as unpriced, never as a fake 0% drift.
    """
    equity_value_float = _float_or_none(equity_float)
    cash_value_float = _float_or_none(cash_float) or 0.0
    priced_dict_list = [
        item_dict
        for item_dict in (position_exposure_dict_list or [])
        if item_dict.get("market_value_float") is not None
    ]
    unpriced_count_int = len(position_exposure_dict_list or []) - len(priced_dict_list)
    cash_in_pie_float = max(cash_value_float, 0.0)
    denominator_float = (
        sum(abs(float(item_dict["market_value_float"])) for item_dict in priced_dict_list)
        + cash_in_pie_float
    )
    if denominator_float <= 0.0:
        return {"has_data_bool": False, "unpriced_count_int": unpriced_count_int}

    slice_dict_list: list[dict[str, Any]] = []
    start_deg_float = 0.0
    entry_list = [
        (
            str(item_dict["asset_str"]),
            abs(float(item_dict["market_value_float"])),
            float(item_dict["market_value_float"]),
        )
        for item_dict in priced_dict_list
    ]
    if cash_in_pie_float > 0.0:
        entry_list.append(("Cash", cash_in_pie_float, cash_in_pie_float))
    for index_int, (label_str, weight_value_float, signed_value_float) in enumerate(entry_list):
        weight_float = weight_value_float / denominator_float
        sweep_deg_float = weight_float * 360.0
        end_deg_float = start_deg_float + sweep_deg_float
        is_cash_bool = label_str == "Cash"
        color_str = (
            _HOLDINGS_CASH_COLOR_STR
            if is_cash_bool
            else _HOLDINGS_COLOR_STR_LIST[index_int % len(_HOLDINGS_COLOR_STR_LIST)]
        )
        target_float = None
        drift_pp_float = None
        if not is_cash_bool and target_weight_map_dict:
            target_obj = target_weight_map_dict.get(label_str)
            target_float = _float_or_none(target_obj)
            if target_float is not None:
                drift_pp_float = (weight_float - target_float) * 100.0
        slice_dict_list.append(
            {
                "label_str": label_str,
                "is_cash_bool": is_cash_bool,
                "market_value_float": signed_value_float,
                "weight_float": weight_float,
                "weight_pct_label_str": f"{weight_float * 100.0:.1f}%",
                "target_pct_label_str": (
                    f"{target_float * 100.0:.1f}%" if target_float is not None else "—"
                ),
                "drift_pp_float": drift_pp_float,
                "drift_pp_label_str": (
                    f"{drift_pp_float:+.1f}pp" if drift_pp_float is not None else "—"
                ),
                "color_str": color_str,
                "is_full_circle_bool": weight_float >= 0.9999,
                "path_d_str": (
                    "" if weight_float >= 0.9999
                    else _pie_slice_path_str(start_deg_float, end_deg_float)
                ),
            }
        )
        start_deg_float = end_deg_float
    return {
        "has_data_bool": True,
        "slice_dict_list": slice_dict_list,
        "unpriced_count_int": unpriced_count_int,
        "equity_label_str": (
            f"${equity_value_float:,.0f}" if equity_value_float is not None else "—"
        ),
        "view_size_int": PIE_VIEW_SIZE_INT,
    }
