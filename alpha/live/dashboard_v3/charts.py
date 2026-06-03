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
import math
import statistics
from typing import Any


TRADING_DAYS_PER_YEAR_INT = 252


CHART_VIEW_WIDTH_INT = 600
CHART_VIEW_HEIGHT_INT = 120
CHART_VERTICAL_PADDING_INT = 6
PNL_BAR_BLOCK_HEIGHT_INT = 32
# Cap the daily-PnL bar width (viewBox units) so a handful of EOD points render
# as distinct bars instead of merging into one fat slab.
MAX_PNL_BAR_WIDTH_FLOAT = 14.0
SUPPORTED_WINDOW_STR_LIST = ["30d", "90d", "all"]

_MONTH_ABBREVIATION_STR_LIST = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@dataclass
class EquityChartDict:
    point_count_int: int = 0
    has_curve_bool: bool = False
    path_d_str: str = ""
    drawdown_d_str: str = ""
    point_dict_list: list[dict[str, Any]] = field(default_factory=list)
    pnl_bar_dict_list: list[dict[str, Any]] = field(default_factory=list)
    range_min_float: float = 0.0
    range_max_float: float = 0.0
    range_min_label_str: str = "—"
    range_max_label_str: str = "—"
    latest_equity_float: float | None = None
    latest_market_date_str: str | None = None
    earliest_market_date_str: str | None = None
    latest_daily_pnl_label_str: str = "—"
    latest_daily_pct_label_str: str = "—"
    latest_daily_is_positive_bool: bool = True
    window_str: str = "all"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_count_int": self.point_count_int,
            "has_curve_bool": self.has_curve_bool,
            "path_d_str": self.path_d_str,
            "drawdown_d_str": self.drawdown_d_str,
            "point_dict_list": self.point_dict_list,
            "pnl_bar_dict_list": self.pnl_bar_dict_list,
            "range_min_float": self.range_min_float,
            "range_max_float": self.range_max_float,
            "range_min_label_str": self.range_min_label_str,
            "range_max_label_str": self.range_max_label_str,
            "latest_equity_float": self.latest_equity_float,
            "latest_market_date_str": self.latest_market_date_str,
            "earliest_market_date_str": self.earliest_market_date_str,
            "latest_daily_pnl_label_str": self.latest_daily_pnl_label_str,
            "latest_daily_pct_label_str": self.latest_daily_pct_label_str,
            "latest_daily_is_positive_bool": self.latest_daily_is_positive_bool,
            "window_str": self.window_str,
            "width_int": CHART_VIEW_WIDTH_INT,
            "height_int": CHART_VIEW_HEIGHT_INT,
            "bar_height_int": PNL_BAR_BLOCK_HEIGHT_INT,
        }


def build_equity_chart_dict(
    equity_point_dict_list: list[dict[str, Any]] | None,
    *,
    window_str: str = "all",
) -> EquityChartDict:
    if not equity_point_dict_list:
        return EquityChartDict(window_str=window_str)

    clean_point_list = _truncate_for_window(equity_point_dict_list, window_str)
    equity_pairs_list = [
        (
            str(point_dict.get("market_date_str") or ""),
            _float_or_none(point_dict.get("equity_float")),
            _float_or_none(point_dict.get("daily_pnl_float")),
        )
        for point_dict in clean_point_list
    ]
    equity_pairs_list = [pair for pair in equity_pairs_list if pair[1] is not None]
    point_count_int = len(equity_pairs_list)
    if point_count_int == 0:
        return EquityChartDict(window_str=window_str)
    if point_count_int == 1:
        only_date_str, only_equity_float, _ = equity_pairs_list[0]
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
        )

    equity_value_list = [pair[1] for pair in equity_pairs_list]
    range_min_float = min(equity_value_list)  # type: ignore[type-var]
    range_max_float = max(equity_value_list)  # type: ignore[type-var]
    value_range_float = max(1e-9, float(range_max_float) - float(range_min_float))

    horizontal_step_float = CHART_VIEW_WIDTH_INT / max(1, point_count_int - 1)
    inner_height_float = CHART_VIEW_HEIGHT_INT - 2 * CHART_VERTICAL_PADDING_INT

    point_xy_list: list[tuple[float, float]] = []
    point_dict_list: list[dict[str, Any]] = []
    for index_int, (date_str, equity_float, pnl_float) in enumerate(equity_pairs_list):
        x_float = index_int * horizontal_step_float
        y_float = (
            CHART_VIEW_HEIGHT_INT
            - CHART_VERTICAL_PADDING_INT
            - ((float(equity_float) - float(range_min_float)) / value_range_float) * inner_height_float  # type: ignore[arg-type]
        )
        point_xy_list.append((x_float, y_float))
        # Daily return % from the equity ratio vs the prior session (honest
        # daily return; None for the first point, which has no prior).
        previous_equity_float = equity_pairs_list[index_int - 1][1] if index_int > 0 else None
        daily_pct_float = (
            (float(equity_float) / float(previous_equity_float)) - 1.0
            if previous_equity_float not in (None, 0)
            else None
        )
        point_dict_list.append({
            "x_float": round(x_float, 2),
            "y_float": round(y_float, 2),
            "market_date_str": date_str,
            "equity_label_str": _format_money_str(equity_float),
            "daily_pnl_label_str": _format_signed_money_str(pnl_float) if pnl_float is not None else "—",
            "daily_pct_label_str": _format_signed_pct_str(daily_pct_float),
            "daily_pct_float": daily_pct_float,
        })

    path_d_str = _build_polyline_path_str(point_xy_list)
    drawdown_d_str = _build_drawdown_polygon_path_str(equity_pairs_list, point_xy_list)
    pnl_bar_dict_list = _build_pnl_bar_dict_list(equity_pairs_list, horizontal_step_float)

    latest_pnl_float = equity_pairs_list[-1][2]
    latest_daily_pct_float = point_dict_list[-1]["daily_pct_float"]

    return EquityChartDict(
        point_count_int=point_count_int,
        has_curve_bool=True,
        path_d_str=path_d_str,
        drawdown_d_str=drawdown_d_str,
        point_dict_list=point_dict_list,
        pnl_bar_dict_list=pnl_bar_dict_list,
        range_min_float=float(range_min_float),
        range_max_float=float(range_max_float),
        range_min_label_str=_format_money_str(range_min_float),
        range_max_label_str=_format_money_str(range_max_float),
        latest_equity_float=equity_value_list[-1],
        latest_market_date_str=equity_pairs_list[-1][0],
        earliest_market_date_str=equity_pairs_list[0][0],
        latest_daily_pnl_label_str=_format_signed_money_str(latest_pnl_float),
        latest_daily_pct_label_str=_format_signed_pct_str(latest_daily_pct_float),
        latest_daily_is_positive_bool=(latest_daily_pct_float or 0.0) >= 0,
        window_str=window_str,
    )


# ── private helpers ───────────────────────────────────────────────────────


def _truncate_for_window(
    equity_point_dict_list: list[dict[str, Any]], window_str: str
) -> list[dict[str, Any]]:
    if window_str == "30d":
        return equity_point_dict_list[-30:]
    if window_str == "90d":
        return equity_point_dict_list[-90:]
    return equity_point_dict_list


def _build_polyline_path_str(point_xy_list: list[tuple[float, float]]) -> str:
    if not point_xy_list:
        return ""
    parts_list = []
    for index_int, (x_float, y_float) in enumerate(point_xy_list):
        prefix_str = "M" if index_int == 0 else "L"
        parts_list.append(f"{prefix_str} {x_float:.2f} {y_float:.2f}")
    return " ".join(parts_list)


def _build_drawdown_polygon_path_str(
    equity_pairs_list: list[tuple[str, float | None, float | None]],
    point_xy_list: list[tuple[float, float]],
) -> str:
    """Polygon between the running-peak line and the equity curve.

    Renders as semi-transparent fill underneath the curve to flag periods of
    drawdown. Returns "" if the curve never has a meaningful dip.
    """
    if len(equity_pairs_list) < 2:
        return ""
    running_peak_float = float(equity_pairs_list[0][1] or 0.0)
    underwater_xy_list: list[tuple[float, float]] = []
    for index_int, (_date_str, equity_float, _pnl_float) in enumerate(equity_pairs_list):
        if equity_float is None:
            continue
        if float(equity_float) > running_peak_float:
            running_peak_float = float(equity_float)
        underwater_xy_list.append((index_int, running_peak_float))
    if not any(equity_float is not None and float(equity_float) < peak_float
               for (_, equity_float, _), (_, peak_float) in zip(equity_pairs_list, underwater_xy_list)):
        return ""

    # Top edge follows the running peak.
    min_equity_value_float = min(
        float(equity_float) for (_, equity_float, _) in equity_pairs_list if equity_float is not None
    )
    range_max_float = max(peak_float for (_, peak_float) in underwater_xy_list)
    value_range_float = max(1e-9, range_max_float - min_equity_value_float)
    inner_height_float = CHART_VIEW_HEIGHT_INT - 2 * CHART_VERTICAL_PADDING_INT
    horizontal_step_float = CHART_VIEW_WIDTH_INT / max(1, len(equity_pairs_list) - 1)

    def y_for_float(value_float: float) -> float:
        return (
            CHART_VIEW_HEIGHT_INT
            - CHART_VERTICAL_PADDING_INT
            - ((value_float - min_equity_value_float) / value_range_float) * inner_height_float
        )

    parts_list: list[str] = []
    for index_int, (_, peak_float) in enumerate(underwater_xy_list):
        x_float = index_int * horizontal_step_float
        prefix_str = "M" if index_int == 0 else "L"
        parts_list.append(f"{prefix_str} {x_float:.2f} {y_for_float(peak_float):.2f}")
    # Bottom edge follows the curve in reverse.
    for index_int in range(len(point_xy_list) - 1, -1, -1):
        x_float, y_float = point_xy_list[index_int]
        parts_list.append(f"L {x_float:.2f} {y_float:.2f}")
    parts_list.append("Z")
    return " ".join(parts_list)


def _build_pnl_bar_dict_list(
    equity_pairs_list: list[tuple[str, float | None, float | None]],
    horizontal_step_float: float,
) -> list[dict[str, Any]]:
    pnl_value_list = [pair[2] for pair in equity_pairs_list if pair[2] is not None]
    if not pnl_value_list:
        return []
    max_abs_float = max(abs(value_float) for value_float in pnl_value_list)
    if max_abs_float <= 0:
        return []
    bar_dict_list: list[dict[str, Any]] = []
    half_height_float = PNL_BAR_BLOCK_HEIGHT_INT / 2
    # Capped width + centered under each point so a few EOD bars read as
    # distinct bars (with gaps) rather than one merged slab.
    bar_width_float = max(1.0, min(horizontal_step_float * 0.6, MAX_PNL_BAR_WIDTH_FLOAT))
    for index_int, (date_str, _equity_float, pnl_float) in enumerate(equity_pairs_list):
        if pnl_float is None:
            continue
        bar_height_float = abs(float(pnl_float)) / max_abs_float * half_height_float
        is_positive_bool = float(pnl_float) >= 0
        center_x_float = index_int * horizontal_step_float
        bar_x_float = min(
            max(0.0, center_x_float - bar_width_float / 2.0),
            CHART_VIEW_WIDTH_INT - bar_width_float,
        )
        bar_dict_list.append({
            "x_float": round(bar_x_float, 2),
            "y_float": round(half_height_float - bar_height_float if is_positive_bool else half_height_float, 2),
            "width_float": round(bar_width_float, 2),
            "height_float": round(max(0.5, bar_height_float), 2),
            "is_positive_bool": is_positive_bool,
            "market_date_str": date_str,
            "pnl_label_str": _format_signed_money_str(pnl_float),
        })
    return bar_dict_list


def _float_or_none(value_obj: Any) -> float | None:
    if value_obj is None:
        return None
    try:
        return float(value_obj)
    except (TypeError, ValueError):
        return None


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


def build_monthly_return_dict_list(
    equity_point_dict_list: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Per-calendar-month return % from the EOD equity series.

    Month-end equity = the last point in each ``YYYY-MM``; a month's return is
    ``month_end / previous_month_end - 1``. The first month uses the series'
    first equity as its base (return since the book started, within that month).
    Read-only reporting — no forecasting.
    """
    equity_pair_list = [
        (str(point_dict.get("market_date_str") or ""), _float_or_none(point_dict.get("equity_float")))
        for point_dict in (equity_point_dict_list or [])
    ]
    equity_pair_list = [
        pair for pair in equity_pair_list
        if pair[1] is not None and pair[1] > 0 and len(pair[0]) >= 7
    ]
    if not equity_pair_list:
        return []

    month_end_equity_dict: dict[str, float] = {}
    month_key_order_list: list[str] = []
    for date_str, equity_float in equity_pair_list:
        month_key_str = date_str[:7]
        if month_key_str not in month_end_equity_dict:
            month_key_order_list.append(month_key_str)
        month_end_equity_dict[month_key_str] = equity_float  # type: ignore[assignment]

    first_equity_float = equity_pair_list[0][1]
    monthly_return_dict_list: list[dict[str, Any]] = []
    previous_month_end_float: float | None = None
    for month_key_str in month_key_order_list:
        month_end_float = month_end_equity_dict[month_key_str]
        base_float = previous_month_end_float if previous_month_end_float is not None else first_equity_float
        return_pct_float = (month_end_float / base_float - 1.0) if base_float else 0.0
        monthly_return_dict_list.append(
            {
                "month_label_str": _format_month_label_str(month_key_str),
                "return_pct_float": return_pct_float,
                "return_label_str": _format_signed_pct_str(return_pct_float),
                "is_positive_bool": return_pct_float >= 0,
            }
        )
        previous_month_end_float = month_end_float
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
            "view_size_int": PIE_VIEW_SIZE_INT,
        }


def build_allocation_pie_dict(
    pod_alloc_dict_list: list[dict[str, Any]] | None,
) -> AllocationPieDict:
    """Build a strategy-allocation pie for one mode.

    Each input dict carries ``label_str``, optional ``sublabel_str``,
    ``equity_float`` (net liquidation) and ``cash_float``. Pods without a
    positive equity cannot occupy a slice and are excluded (but counted), since
    a pie can only render non-negative shares.
    """
    cleaned_dict_list: list[dict[str, Any]] = []
    excluded_pod_count_int = 0
    total_cash_float = 0.0
    for pod_alloc_dict in pod_alloc_dict_list or []:
        equity_float = _float_or_none(pod_alloc_dict.get("equity_float"))
        cash_float = _float_or_none(pod_alloc_dict.get("cash_float")) or 0.0
        if equity_float is None or equity_float <= 0:
            excluded_pod_count_int += 1
            continue
        cleaned_dict_list.append({
            "label_str": str(pod_alloc_dict.get("label_str") or "—"),
            "sublabel_str": str(pod_alloc_dict.get("sublabel_str") or ""),
            "equity_float": equity_float,
            "cash_float": cash_float,
        })
        total_cash_float += cash_float

    if not cleaned_dict_list:
        return AllocationPieDict(excluded_pod_count_int=excluded_pod_count_int)

    # Largest slice first so colors + legend order are stable and readable.
    cleaned_dict_list.sort(key=lambda pod_dict: pod_dict["equity_float"], reverse=True)
    total_equity_float = sum(pod_dict["equity_float"] for pod_dict in cleaned_dict_list)
    is_single_slice_bool = len(cleaned_dict_list) == 1

    slice_dict_list: list[dict[str, Any]] = []
    cumulative_deg_float = 0.0
    for index_int, pod_dict in enumerate(cleaned_dict_list):
        fraction_float = pod_dict["equity_float"] / total_equity_float
        color_str = ALLOCATION_PALETTE_STR_LIST[index_int % len(ALLOCATION_PALETTE_STR_LIST)]
        if is_single_slice_bool:
            path_d_str = ""
            is_full_circle_bool = True
        else:
            start_deg_float = cumulative_deg_float
            # Snap the final slice to a full 360° to absorb float drift.
            end_deg_float = (
                360.0
                if index_int == len(cleaned_dict_list) - 1
                else cumulative_deg_float + fraction_float * 360.0
            )
            path_d_str = _pie_slice_path_str(start_deg_float, end_deg_float)
            is_full_circle_bool = False
            cumulative_deg_float = end_deg_float
        slice_dict_list.append({
            "label_str": pod_dict["label_str"],
            "sublabel_str": pod_dict["sublabel_str"],
            "color_str": color_str,
            "equity_float": pod_dict["equity_float"],
            "equity_label_str": _format_money_str(pod_dict["equity_float"]),
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
        }


def build_book_risk_dict(
    equity_point_dict_list: list[dict[str, Any]] | None,
) -> BookRiskDict:
    equity_pair_list = [
        (str(point_dict.get("market_date_str") or ""), _float_or_none(point_dict.get("equity_float")))
        for point_dict in (equity_point_dict_list or [])
    ]
    equity_pair_list = [pair for pair in equity_pair_list if pair[1] is not None and pair[1] > 0]
    if not equity_pair_list:
        return BookRiskDict()

    current_date_str, current_equity_float = equity_pair_list[-1]

    # Drawdown vs the running peak. *** CRITICAL*** running peak must only look
    # at sessions up to and including each point — never ahead of it.
    running_peak_float = equity_pair_list[0][1]
    running_peak_date_str = equity_pair_list[0][0]
    overall_peak_float = running_peak_float
    overall_peak_date_str = running_peak_date_str
    max_drawdown_pct_float = 0.0
    last_high_index_int = 0
    for index_int, (date_str, equity_float) in enumerate(equity_pair_list):
        if equity_float >= running_peak_float:
            running_peak_float = equity_float
            running_peak_date_str = date_str
            last_high_index_int = index_int
        drawdown_pct_float = (running_peak_float - equity_float) / running_peak_float
        if drawdown_pct_float > max_drawdown_pct_float:
            max_drawdown_pct_float = drawdown_pct_float
        if equity_float > overall_peak_float:
            overall_peak_float = equity_float
            overall_peak_date_str = date_str

    current_drawdown_pct_float = (
        (overall_peak_float - current_equity_float) / overall_peak_float
        if overall_peak_float > 0
        else 0.0
    )
    days_underwater_int = (len(equity_pair_list) - 1) - last_high_index_int

    # Realized daily returns → sample stdev → annualized volatility.
    daily_return_list = [
        (equity_pair_list[index_int][1] / equity_pair_list[index_int - 1][1]) - 1.0
        for index_int in range(1, len(equity_pair_list))
        if equity_pair_list[index_int - 1][1]
    ]
    daily_vol_label_str = "—"
    annualized_vol_label_str = "—"
    if len(daily_return_list) >= 2:
        daily_vol_float = statistics.stdev(daily_return_list)
        daily_vol_label_str = f"{daily_vol_float * 100:.2f}%"
        annualized_vol_label_str = (
            f"{daily_vol_float * math.sqrt(TRADING_DAYS_PER_YEAR_INT) * 100:.1f}%"
        )

    return BookRiskDict(
        has_data_bool=True,
        point_count_int=len(equity_pair_list),
        current_equity_label_str=_format_money_str(current_equity_float),
        peak_equity_label_str=_format_money_str(overall_peak_float),
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
    )


__all__ = [
    "CHART_VIEW_HEIGHT_INT",
    "CHART_VIEW_WIDTH_INT",
    "AllocationPieDict",
    "BookRiskDict",
    "EquityChartDict",
    "PNL_BAR_BLOCK_HEIGHT_INT",
    "SUPPORTED_WINDOW_STR_LIST",
    "build_allocation_pie_dict",
    "build_book_risk_dict",
    "build_equity_chart_dict",
    "build_monthly_return_dict_list",
]
