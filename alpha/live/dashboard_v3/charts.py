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
from typing import Any


CHART_VIEW_WIDTH_INT = 600
CHART_VIEW_HEIGHT_INT = 120
CHART_VERTICAL_PADDING_INT = 6
PNL_BAR_BLOCK_HEIGHT_INT = 32
SUPPORTED_WINDOW_STR_LIST = ["30d", "90d", "all"]


@dataclass
class EquityChartDict:
    point_count_int: int = 0
    has_curve_bool: bool = False
    path_d_str: str = ""
    drawdown_d_str: str = ""
    pnl_bar_dict_list: list[dict[str, Any]] = field(default_factory=list)
    range_min_float: float = 0.0
    range_max_float: float = 0.0
    range_min_label_str: str = "—"
    range_max_label_str: str = "—"
    latest_equity_float: float | None = None
    latest_market_date_str: str | None = None
    earliest_market_date_str: str | None = None
    window_str: str = "all"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_count_int": self.point_count_int,
            "has_curve_bool": self.has_curve_bool,
            "path_d_str": self.path_d_str,
            "drawdown_d_str": self.drawdown_d_str,
            "pnl_bar_dict_list": self.pnl_bar_dict_list,
            "range_min_float": self.range_min_float,
            "range_max_float": self.range_max_float,
            "range_min_label_str": self.range_min_label_str,
            "range_max_label_str": self.range_max_label_str,
            "latest_equity_float": self.latest_equity_float,
            "latest_market_date_str": self.latest_market_date_str,
            "earliest_market_date_str": self.earliest_market_date_str,
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
    for index_int, (_date_str, equity_float, _pnl_float) in enumerate(equity_pairs_list):
        x_float = index_int * horizontal_step_float
        y_float = (
            CHART_VIEW_HEIGHT_INT
            - CHART_VERTICAL_PADDING_INT
            - ((float(equity_float) - float(range_min_float)) / value_range_float) * inner_height_float  # type: ignore[arg-type]
        )
        point_xy_list.append((x_float, y_float))

    path_d_str = _build_polyline_path_str(point_xy_list)
    drawdown_d_str = _build_drawdown_polygon_path_str(equity_pairs_list, point_xy_list)
    pnl_bar_dict_list = _build_pnl_bar_dict_list(equity_pairs_list, horizontal_step_float)

    return EquityChartDict(
        point_count_int=point_count_int,
        has_curve_bool=True,
        path_d_str=path_d_str,
        drawdown_d_str=drawdown_d_str,
        pnl_bar_dict_list=pnl_bar_dict_list,
        range_min_float=float(range_min_float),
        range_max_float=float(range_max_float),
        range_min_label_str=_format_money_str(range_min_float),
        range_max_label_str=_format_money_str(range_max_float),
        latest_equity_float=equity_value_list[-1],
        latest_market_date_str=equity_pairs_list[-1][0],
        earliest_market_date_str=equity_pairs_list[0][0],
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
    for index_int, (_date_str, _equity_float, pnl_float) in enumerate(equity_pairs_list):
        if pnl_float is None:
            continue
        bar_height_float = abs(float(pnl_float)) / max_abs_float * half_height_float
        is_positive_bool = float(pnl_float) >= 0
        bar_x_float = index_int * horizontal_step_float
        bar_dict_list.append({
            "x_float": round(bar_x_float, 2),
            "y_float": round(half_height_float - bar_height_float if is_positive_bool else half_height_float, 2),
            "width_float": round(max(1.0, horizontal_step_float * 0.6), 2),
            "height_float": round(max(0.5, bar_height_float), 2),
            "is_positive_bool": is_positive_bool,
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


__all__ = [
    "CHART_VIEW_HEIGHT_INT",
    "CHART_VIEW_WIDTH_INT",
    "EquityChartDict",
    "PNL_BAR_BLOCK_HEIGHT_INT",
    "SUPPORTED_WINDOW_STR_LIST",
    "build_equity_chart_dict",
]
