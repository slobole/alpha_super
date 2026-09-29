"""HTML building blocks for the fund menu report (tables, heat cells, chart payloads).

Every number shown in the report comes from the saved study outputs; this module
only formats. Text is escaped; chart payloads are JSON embedded in the page.
"""

from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd


def esc(value_obj) -> str:
    return html.escape(str(value_obj), quote=True)


def is_missing(value_obj) -> bool:
    return value_obj is None or (isinstance(value_obj, float) and (math.isnan(value_obj) or math.isinf(value_obj)))


def pct(value_obj, digit_int: int = 1, signed_bool: bool = False) -> str:
    if is_missing(value_obj):
        return "–"
    value_float = float(value_obj) * 100.0
    text_str = f"{value_float:+.{digit_int}f}%" if signed_bool else f"{value_float:.{digit_int}f}%"
    return text_str.replace("-", "−")


def num(value_obj, digit_int: int = 2) -> str:
    if is_missing(value_obj):
        return "–"
    return f"{float(value_obj):.{digit_int}f}".replace("-", "−")


def money(value_obj) -> str:
    if is_missing(value_obj):
        return "–"
    value_float = float(value_obj)
    if value_float >= 1_000_000:
        return f"${value_float / 1_000_000:.1f}M"
    return f"${value_float / 1_000:.0f}K"


def heat_class(value_obj, step_tuple: tuple[float, float, float] = (0.03, 0.10, 0.20)) -> str:
    """Diverging bin for a return: red below zero, blue above, neutral near zero."""
    if is_missing(value_obj):
        return "h-na"
    value_float = float(value_obj)
    small_float, mid_float, big_float = step_tuple
    if value_float <= -big_float:
        return "h-n3"
    if value_float <= -mid_float:
        return "h-n2"
    if value_float <= -small_float:
        return "h-n1"
    if value_float < small_float:
        return "h-0"
    if value_float < mid_float:
        return "h-p1"
    if value_float < big_float:
        return "h-p2"
    return "h-p3"


def table(header_list: list[str], row_list: list[list[str]], row_class_list: list[str] | None = None,
          table_class_str: str = "", caption_str: str = "", left_column_count_int: int = 1) -> str:
    """Rows are pre-rendered cell HTML strings (already escaped where needed)."""
    head_html = "".join(
        f'<th class="{"l" if i < left_column_count_int else ""}">{h}</th>' for i, h in enumerate(header_list)
    )
    body_list = []
    for row_index_int, cell_list in enumerate(row_list):
        row_class_str = row_class_list[row_index_int] if row_class_list else ""
        cells_html = "".join(
            cell if cell.startswith("<td") else f'<td class="{"l" if i < left_column_count_int else ""}">{cell}</td>'
            for i, cell in enumerate(cell_list)
        )
        body_list.append(f'<tr class="{row_class_str}">{cells_html}</tr>')
    caption_html = f'<caption class="note" style="caption-side:bottom;text-align:left;padding:6px 10px">{caption_str}</caption>' if caption_str else ""
    return (
        f'<div class="table-wrap"><table class="{table_class_str}"><thead><tr>{head_html}</tr></thead>'
        f'<tbody>{"".join(body_list)}</tbody>{caption_html}</table></div>'
    )


def heat_cell(value_obj, step_tuple=(0.03, 0.10, 0.20), digit_int: int = 1) -> str:
    return f'<td class="h {heat_class(value_obj, step_tuple)}">{pct(value_obj, digit_int, signed_bool=True)}</td>'


def weekly_nav_payload(nav_df: pd.DataFrame, column_list: list[str]) -> dict:
    """Friday-close NAV (last session of each week) for light chart payloads."""
    weekly_df = nav_df[column_list].resample("W-FRI").last().dropna(how="all")
    return {
        "dates": [ts.date().isoformat() for ts in weekly_df.index],
        "series": {column_str: [None if is_missing(v) else round(float(v), 5) for v in weekly_df[column_str]] for column_str in column_list},
    }


def weekly_drawdown_payload(nav_df: pd.DataFrame, column_list: list[str]) -> dict:
    """Daily drawdown, reduced to each week's deepest point so troughs keep their depth."""
    drawdown_df = nav_df[column_list] / nav_df[column_list].cummax() - 1.0
    weekly_df = drawdown_df.resample("W-FRI").min().dropna(how="all")
    return {
        "dates": [ts.date().isoformat() for ts in weekly_df.index],
        "series": {column_str: [None if is_missing(v) else round(float(v), 5) for v in weekly_df[column_str]] for column_str in column_list},
    }


def json_payload(payload_obj) -> str:
    def default(value_obj):
        if isinstance(value_obj, (np.floating,)):
            return None if not np.isfinite(value_obj) else float(value_obj)
        if isinstance(value_obj, (np.integer,)):
            return int(value_obj)
        return str(value_obj)

    text_str = json.dumps(payload_obj, default=default, separators=(",", ":"))
    return text_str.replace("</", "<\\/")
