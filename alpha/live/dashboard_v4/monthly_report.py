"""Monthly client report: the client's programs, from the canonical accounting and daily TWR.

Read-only. Other accounts (such as VAULT, often cash in shekels) are named but
never converted into the report's USD numbers. Holdings need saved same-close
positions and independently evidenced cash, or are left out.
"""

import calendar
from contextlib import closing
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache
import math
from pathlib import Path
import sqlite3
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import CAPITAL_FIELD_TUPLE, ClientReportingError
from alpha.live.client_reporting import content_hash_str
from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list
from alpha.live.dashboard_v3.client_operations import build_client_operations_dict
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict
from alpha.live.dashboard_v4.pod_holdings import load_close_holdings_dict
from alpha.live.dashboard_v4.positions_history import report_scope_tuple
from alpha.live.dashboard_v4.strategy_scope import strategy_workspace_dict
from alpha.live.scheduler_utils import get_exchange_calendar_obj


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
MONTH_OPTION_LIMIT_INT = 24
MONTH_COLUMN_LIMIT_INT = 9
FLOW_ROW_LIMIT_INT = 12
HOLDING_ROW_LIMIT_INT = 12
NAME_XML_BYTE_LIMIT_INT = 32_000_000
BRAND_NAME_STR = "Loren Slobodinsky"
BRAND_DESC_STR = "Systematic investing"
DEFAULT_PROGRAM_STR = "Portfolio"
# One hue family per program: the program colour, then its Pods' shades in account
# order (a fourth Pod reuses the last shade). Adjacent shades pass the dataviz
# check (CVD and normal-vision ΔE ≥ 15); the legend names every Pod.
PROGRAM_FAMILY_TUPLE = (
    ("#1D4F91", "#5A86C8", "#A3BDE6"),
    ("#13875C", "#5FB892", "#B2DFC9"),
    ("#5B4BA8", "#8E81D0", "#C4BCEC"),
)
PROGRAM_FAMILY_INDEX_DICT = {"Steady": 0, "Growth": 1}
# Fixed class order. The coloured classes pass the dataviz check on white as an
# adjacent cycle (CVD ΔE ≥ 8, normal vision ≥ 15); hedges and cash are neutral
# greys, always named in the legend.
ASSET_CLASS_TUPLE = (
    ("Nasdaq 100", "#2F6BB3"), ("Stocks", "#1F9A78"), ("Gold", "#B8841A"), ("US dollar", "#D0577F"),
    ("Bonds", "#7455A8"), ("Commodities", "#B5562F"), ("Hedges", "#7D8796"), ("Cash", "#C9C2B2"),
)
SYMBOL_CLASS_DICT = {
    **dict.fromkeys(("QQQ", "QQQM", "QLD", "TQQQ"), "Nasdaq 100"),
    **dict.fromkeys(("SPY", "VOO", "IVV", "VTI", "IWM", "DIA", "RSP", "MDY", "SSO", "UPRO", "XLB", "XLC",
                     "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "KIE", "SMH", "SOXX"), "Stocks"),
    **dict.fromkeys(("GLD", "IAU", "GLDM", "SGOL"), "Gold"),
    "UUP": "US dollar",
    **dict.fromkeys(("IEF", "TLT", "IEI", "SHY", "TLH", "GOVT", "VGIT", "VGLT", "EDV", "ZROZ", "TIP", "AGG",
                     "BND", "LQD", "HYG"), "Bonds"),
    **dict.fromkeys(("DBC", "PDBC", "GSG", "COMT", "DBA", "DBB", "USO", "SLV"), "Commodities"),
    **dict.fromkeys(("BTAL", "VIXM", "VIXY", "VXX", "SH", "PSQ", "SQQQ", "QID"), "Hedges"),
    **dict.fromkeys(("BIL", "SGOV", "SHV"), "Cash"),
}
SYMBOL_NAME_DICT = {
    "QQQ": "Nasdaq 100", "QQQM": "Nasdaq 100", "QLD": "Nasdaq 100 2×", "TQQQ": "Nasdaq 100 3×",
    "SPY": "S&P 500", "VOO": "S&P 500", "IVV": "S&P 500", "SSO": "S&P 500 2×", "UPRO": "S&P 500 3×",
    "VTI": "US stocks", "IWM": "US small caps", "DIA": "Dow Jones 30", "RSP": "S&P 500 equal weight",
    "MDY": "US mid caps", "XLB": "Materials", "XLC": "Communication", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Consumer staples", "XLRE": "Real estate",
    "XLU": "Utilities", "XLV": "Health care", "XLY": "Consumer discretionary", "KIE": "Insurance",
    "SMH": "Semiconductors", "SOXX": "Semiconductors",
    "GLD": "Gold", "IAU": "Gold", "GLDM": "Gold", "SGOL": "Gold", "UUP": "US dollar",
    "IEF": "Treasuries 7–10y", "TLT": "Treasuries 20y+", "IEI": "Treasuries 3–7y", "SHY": "Treasuries 1–3y",
    "TLH": "Treasuries 10–20y", "GOVT": "Treasuries", "VGIT": "Treasuries 3–10y", "VGLT": "Treasuries 10y+",
    "EDV": "Treasuries 20y+", "ZROZ": "Treasuries 25y+", "TIP": "Inflation-linked Treasuries",
    "AGG": "US bonds", "BND": "US bonds", "LQD": "Corporate bonds", "HYG": "High-yield bonds",
    "DBC": "Commodities", "PDBC": "Commodities", "GSG": "Commodities", "COMT": "Commodities",
    "DBA": "Agriculture", "DBB": "Base metals", "USO": "Oil", "SLV": "Silver",
    "BTAL": "Anti-beta hedge", "VIXM": "Volatility hedge", "VIXY": "Volatility hedge", "VXX": "Volatility hedge",
    "SH": "S&P 500 short", "PSQ": "Nasdaq 100 short", "SQQQ": "Nasdaq 100 3× short", "QID": "Nasdaq 100 2× short",
    "BIL": "T-bills", "SGOV": "T-bills", "SHV": "T-bills",
}
LEVERAGE_SUFFIX_DICT = {"QLD": " 2×", "TQQQ": " 3×", "SSO": " 2×", "UPRO": " 3×", "QID": " −2×", "SQQQ": " −3×"}


def _empty_dict():
    return {"available_bool": False, "error_str": "", "client_name_str": "", "month_str": "",
        "month_label_str": "", "period_label_str": "", "month_name_str": "", "back_url_str": "",
        "month_option_list": [], "status_note_str": "", "draft_bool": False, "draft_reason_str": "",
        "art_url_str": "", "brand_name_str": BRAND_NAME_STR, "brand_desc_str": BRAND_DESC_STR,
        "money_row_list": [], "split_dict": {"aria_str": "", "segment_list": [], "group_list": []},
        "single_program_bool": True, "program_list": [], "month_table_dict": {"label_list": [], "row_list": []},
        "flow_dict": {"row_list": [], "pod_name_list": [], "pod_total_list": [], "total_str": "", "reason_str": ""},
        "activity_list": [], "pod_choice_list": []}


def _rounded_decimal(value_obj, exponent_str):
    return Decimal(str(value_obj)).quantize(Decimal(exponent_str), rounding=ROUND_HALF_UP)


def _field_decimal(attribute_dict, field_str):
    """One saved Flex amount; a missing or non-finite value is unusable, never zero."""
    value_decimal = Decimal(str(attribute_dict[field_str]))
    if not value_decimal.is_finite():
        raise ValueError(f"Non-finite {field_str}")
    return value_decimal


def _money_str(value_obj, *, signed_bool=False, cents_bool=False):
    """Statement money: whole dollars (cents for small fees), real minus sign, no "−$0"."""
    if value_obj is None:
        return "—"
    amount_decimal = _rounded_decimal(value_obj, ".01" if cents_bool else "1")
    if amount_decimal == 0:
        return "$0.00" if cents_bool else "$0"
    sign_str = "−" if amount_decimal < 0 else "+" if signed_bool else ""
    return f"{sign_str}${abs(amount_decimal):,.2f}" if cents_bool else f"{sign_str}${abs(amount_decimal):,.0f}"


def _money_tone_str(value_obj):
    return "neg" if value_obj is not None and _rounded_decimal(value_obj, "1") < 0 else ""


def _percent_str(value_float, *, signed_bool=True, suffix_str="%"):
    """One decimal of a return or share; the only % shown anywhere in the report."""
    if value_float is None:
        return "—"
    amount_decimal = _rounded_decimal(value_float * 100, ".1")
    if amount_decimal == 0:
        return "0.0" + suffix_str
    sign_str = "−" if amount_decimal < 0 else "+" if signed_bool else ""
    return f"{sign_str}{abs(amount_decimal)}{suffix_str}"


def _percent_tone_str(value_float):
    return "neg" if value_float is not None and _rounded_decimal(value_float * 100, ".1") < 0 else ""


def _long_day_str(date_str):
    date_obj = date.fromisoformat(date_str)
    return f"{calendar.month_name[date_obj.month]} {date_obj.day}"


def _short_day_str(date_str):
    date_obj = date.fromisoformat(date_str)
    return f"{calendar.month_abbr[date_obj.month]} {date_obj.day}"


def _month_bounds_tuple(month_str):
    year_int, month_int = int(month_str[:4]), int(month_str[5:7])
    return f"{month_str}-01", f"{month_str}-{calendar.monthrange(year_int, month_int)[1]:02d}"


def _last_session_str(first_str, last_str):
    session_list = get_exchange_calendar_obj("XNYS").sessions_in_range(first_str, last_str)
    return session_list[-1].date().isoformat() if len(session_list) else ""


def _next_month_obj(month_obj):
    return date(month_obj.year + month_obj.month // 12, month_obj.month % 12 + 1, 1)


def _month_option_list(start_str, latest_str, end_str=None):
    """Months from the first measured day to the latest finalized close, newest first.

    A month is final once a finalized statement reaches its last calendar day:
    IBKR can still report weekend or holiday activity after the last session.
    When every Pod has closed (end_str), the months stop there and the last one
    is final once statements reach the last Pod session.
    """
    option_list, cursor_obj = [], date.fromisoformat(start_str[:7] + "-01")
    while cursor_obj.isoformat()[:7] <= (min(latest_str, end_str) if end_str else latest_str)[:7]:
        month_str = cursor_obj.isoformat()[:7]
        first_str, last_str = _month_bounds_tuple(month_str)
        session_str = _last_session_str(first_str, min(last_str, end_str or last_str)) or last_str
        complete_bool = latest_str >= (session_str if end_str and end_str <= last_str else last_str)
        option_list.append({"month_str": month_str, "complete_bool": complete_bool,
            "sessions_done_bool": latest_str >= session_str,
            "label_str": cursor_obj.strftime("%B %Y") + ("" if complete_bool else " · to date")})
        cursor_obj = _next_month_obj(cursor_obj)
    return option_list[::-1][:MONTH_OPTION_LIMIT_INT]


@lru_cache(maxsize=4)
def _holder_name_cached_dict(db_path_str, mtime_ns_int, size_int, query_name_str):
    db_path_obj = Path(db_path_str)
    with closing(sqlite3.connect(f"{db_path_obj.as_uri()}?mode=ro", uri=True, timeout=1.0)) as connection_obj:
        xml_row_obj = connection_obj.execute(
            "SELECT CASE WHEN length(CAST(raw_xml_str AS BLOB))<=? THEN raw_xml_str END FROM flex_import "
            "WHERE query_name_str=? ORDER BY import_id_int DESC LIMIT 1", (NAME_XML_BYTE_LIMIT_INT, query_name_str)).fetchone()
    if xml_row_obj is None or not isinstance(xml_row_obj[0], str):
        return {}
    if "<!DOCTYPE" in xml_row_obj[0].upper() or "<!ENTITY" in xml_row_obj[0].upper():
        return {}
    name_by_account_dict = {}
    for statement_obj in ElementTree.fromstring(xml_row_obj[0]).iter("FlexStatement"):
        account_str = str(statement_obj.attrib.get("accountId") or "").strip()
        information_obj = statement_obj.find("AccountInformation")
        name_str = " ".join(str(information_obj.attrib.get("name") or "").split())[:80] if information_obj is not None else ""
        if account_str and name_str:
            name_by_account_dict[account_str] = name_str
    return name_by_account_dict


def load_holder_name_str(db_path_str, account_route_set, *, query_name_str):
    """IBKR account holder of these accounts from the client's latest saved Flex report; "" unless one name."""
    if not db_path_str:
        return ""
    try:
        db_path_obj = Path(db_path_str).expanduser().resolve()
        stat_obj = db_path_obj.stat()
        name_by_account_dict = _holder_name_cached_dict(str(db_path_obj), stat_obj.st_mtime_ns, stat_obj.st_size, query_name_str)
    except Exception:
        # Display only: an unreadable source leaves the configured client name.
        return ""
    name_set = {name_by_account_dict[account_str] for account_str in account_route_set if account_str in name_by_account_dict}
    return next(iter(name_set)) if len(name_set) == 1 else ""


def _return_series_tuple(report_dict):
    """The Performance page's own return path and daily returns for this report."""
    path_list, daily_list = report_dict["return_path_list"], report_dict["twr_daily_list"]
    if not report_dict["client_twr_configured_bool"] and report_dict["twr_float"] is not None and len(report_dict["strategy_list"]) == 1:
        path_list = report_dict["strategy_list"][0]["performance_dict"]["return_path_list"]
        daily_list = report_dict["strategy_list"][0]["daily_list"]
    return path_list, daily_list


def _linked_return_float(daily_list, from_str, to_str):
    selected_list = [row_dict for row_dict in daily_list if from_str <= row_dict["market_date_str"] <= to_str]
    if not selected_list or any(row_dict.get("return_float") is None for row_dict in selected_list):
        return None
    growth_decimal = Decimal(1)
    for row_dict in selected_list:
        growth_decimal *= 1 + Decimal(str(row_dict["return_float"]))
    return float(growth_decimal - 1)


def _window_pnl_float(daily_list, from_str, to_str):
    selected_list = [row_dict for row_dict in daily_list if from_str <= row_dict["market_date_str"] <= to_str]
    if not selected_list or any(row_dict.get("pnl_float") is None for row_dict in selected_list):
        return None
    return float(sum((Decimal(str(row_dict["pnl_float"])) for row_dict in selected_list), Decimal(0)))


def _monthly_return_dict(daily_list):
    """Calendar-month linking of the same daily returns the Performance grid links."""
    if not daily_list or any(row_dict.get("return_float") is None for row_dict in daily_list):
        return {}
    growth_dict = {}
    for row_dict in daily_list:
        month_str = row_dict["market_date_str"][:7]
        growth_dict[month_str] = growth_dict.get(month_str, Decimal(1)) * (1 + Decimal(str(row_dict["return_float"])))
    return {month_str: float(growth_decimal - 1) for month_str, growth_decimal in growth_dict.items()}


def _pod_label_tuple(number_int, account_route_str, alias_str):
    """Client-facing Pod name: its number and the IBKR account masked as everywhere in Live Ops, then its IBKR alias."""
    masked_str = account_route_str[:1] + "···" + account_route_str[-3:] if len(account_route_str) >= 4 else "···"
    return f"Pod {number_int} · {masked_str}", alias_str


def _pod_list(report_dict, active_pod_set, name_fn):
    """One entry per Pod (its account periods linked), in account order."""
    pod_by_id_dict = {}
    for strategy_dict in report_dict["strategy_list"]:
        pod_by_id_dict.setdefault(strategy_dict["pod_id_str"], []).append(strategy_dict)
    result_list = []
    for pod_id_str, entry_list in pod_by_id_dict.items():
        last_dict = max(entry_list, key=lambda strategy_dict: strategy_dict["to_date_str"])
        twr_float, growth_dict = None, {}
        if all(strategy_dict["twr_float"] is not None for strategy_dict in entry_list):
            twr_float = math.prod(1 + strategy_dict["twr_float"] for strategy_dict in entry_list) - 1
        for strategy_dict in entry_list:
            for month_dict in strategy_dict["performance_dict"]["monthly_return_list"]:
                growth_dict[month_dict["month_str"]] = growth_dict.get(month_dict["month_str"], 1.) * (1 + month_dict["return_float"])
        active_bool = pod_id_str in active_pod_set
        head_str, alias_str = name_fn(pod_id_str, last_dict["account_route_str"])
        result_list.append({"pod_id_str": pod_id_str, "name_str": " · ".join(part_str for part_str in (head_str, alias_str) if part_str),
            "head_str": head_str, "alias_str": alias_str,
            "account_route_str": last_dict["account_route_str"], "active_bool": active_bool,
            "close_str": last_dict["to_date_str"], "closing_float": last_dict["closing_nav_float"] if active_bool else None,
            "from_str": min(strategy_dict["from_date_str"] for strategy_dict in entry_list),
            "twr_float": twr_float, "month_return_dict": {month_str: growth_float - 1 for month_str, growth_float in growth_dict.items()}})
    return result_list


def _program_workspace_dict(workspace_dict, pod_id_set):
    """The same saved strategy accounts, limited to one program's Pods."""
    result_dict = strategy_workspace_dict(workspace_dict)
    client_dict = result_dict["client_dict"]
    client_dict["accounts"] = [account_dict for account_dict in client_dict["accounts"] if account_dict.get("pod_id") in pod_id_set]
    if client_dict["accounts"]:
        client_dict["mandate_start_date"] = min(account_dict["effective_from"] for account_dict in client_dict["accounts"])
    if "valuation_account_list" in result_dict:
        result_dict["valuation_account_list"] = [account_dict for account_dict in result_dict["valuation_account_list"]
            if account_dict.get("pod_id") in pod_id_set]
    return result_dict


def _donut_segment_list(item_list, *, center_float, radius_float, gap_float=2.5):
    """Arc paths for (fraction, colour) items, clockwise from 12 o'clock with a small gap."""
    total_float = sum(fraction_float for fraction_float, _ in item_list)
    if total_float <= 0:
        return []
    center_str, radius_str = f"{center_float:g}", f"{radius_float:g}"
    if len(item_list) == 1:
        return [{"color_str": item_list[0][1], "d_str": f"M{center_str} {center_float - radius_float:g} A{radius_str} {radius_str} 0 1 1 "
            f"{center_str} {center_float + radius_float:g} A{radius_str} {radius_str} 0 1 1 {center_str} {center_float - radius_float:g}"}]
    segment_list, angle_float, gap_angle_float = [], 0., gap_float / radius_float
    for fraction_float, color_str in item_list:
        sweep_float = fraction_float / total_float * 2 * math.pi
        start_float, end_float = angle_float + gap_angle_float / 2, angle_float + sweep_float - gap_angle_float / 2
        angle_float += sweep_float
        if end_float <= start_float:
            continue
        point_list = [(center_float + radius_float * math.sin(value_float), center_float - radius_float * math.cos(value_float))
            for value_float in (start_float, end_float)]
        segment_list.append({"color_str": color_str, "d_str": f"M{point_list[0][0]:.2f} {point_list[0][1]:.2f} A{radius_str} {radius_str} 0 "
            f"{1 if end_float - start_float > math.pi else 0} 1 {point_list[1][0]:.2f} {point_list[1][1]:.2f}"})
    return segment_list


def _chart_dict(path_list, *, single_bool, wide_bool=False):
    """Return since start: one line, a few % gridlines, month labels, the end value.

    One program: 340 wide beside the holdings, or the card's full 644 without them.
    """
    width_int, height_int, left_int, right_int = ((644, 236, 36, 594) if wide_bool else (340, 236, 36, 290)) if single_bool else (300, 132, 34, 252)
    top_float, bottom_float = 10., height_int - 22.
    result_dict = {"available_bool": False, "reason_str": "Return since start is not available yet.", "wide_bool": single_bool and wide_bool,
        "width_int": width_int, "height_int": height_int, "plot_left": left_int, "plot_right": right_int,
        "tick_list": [], "month_list": [], "path_str": "", "end_x": 0, "end_y": 0, "end_label_str": ""}
    point_list = [(date.fromisoformat(point_dict["market_date_str"][:10]).toordinal()
        + (0 if point_dict["market_date_str"].endswith(" SOD") else .75), point_dict["cumulative_return_float"])
        for point_dict in path_list]
    if len(point_list) < 2:
        return result_dict
    value_list = [value_float for _, value_float in point_list]
    low_float, high_float = min(0., *value_list), max(0., *value_list)
    span_float = max(high_float - low_float, 1e-9)
    step_float = next((value_float for value_float in (.01, .02, .025, .05, .1, .2, .25, .5, 1., 2., 5.)
        if span_float / value_float <= 3), 10.)
    # Gridlines cover the line's high and low; a dip or peak under half a step stays at 0%.
    tick_value_list = [index_int * step_float for index_int in range(
        math.floor(low_float / step_float + 1e-9) if low_float < -step_float / 2 else 0,
        (math.ceil(high_float / step_float - 1e-9) if high_float > step_float / 2 else 0) + 1)]
    if len(tick_value_list) < 2:
        tick_value_list.append(step_float if high_float >= -low_float else -step_float)
    domain_low_float, domain_high_float = min(low_float, *tick_value_list), max(high_float, *tick_value_list)
    pad_float = (domain_high_float - domain_low_float) * .06
    domain_low_float, domain_high_float = domain_low_float - pad_float, domain_high_float + pad_float

    def y_float(value_float):
        return bottom_float - (value_float - domain_low_float) / (domain_high_float - domain_low_float) * (bottom_float - top_float)

    start_float, end_float = point_list[0][0], point_list[-1][0]

    def x_float(ordinal_float):
        return left_int + (ordinal_float - start_float) / max(end_float - start_float, .75) * (right_int - left_int)

    label_list, first_obj, last_obj = [], date.fromordinal(int(start_float)), date.fromordinal(int(end_float))
    cursor_obj, multi_year_bool = first_obj.replace(day=1), first_obj.year != last_obj.year
    while cursor_obj <= last_obj:
        label_list.append([left_int if cursor_obj <= first_obj else x_float(cursor_obj.toordinal()),
            str(cursor_obj.year) if multi_year_bool and cursor_obj.month == 1 else cursor_obj.strftime("%b")])
        cursor_obj = _next_month_obj(cursor_obj)
    # A start late in its month gives way to the next month's label.
    if len(label_list) > 1 and label_list[1][0] - label_list[0][0] < 26:
        label_list.pop(0)
    kept_list = []
    for x_value_float, label_str in label_list:
        if x_value_float <= right_int - 14 and (not kept_list or x_value_float - kept_list[-1][0] >= 26):
            kept_list.append((x_value_float, label_str))
    coordinate_list = [(x_float(ordinal_float), y_float(value_float)) for ordinal_float, value_float in point_list]
    result_dict.update(available_bool=True, reason_str="",
        tick_list=[{"y": round(y_float(value_float), 1), "zero_bool": abs(value_float) < 1e-12,
            "label_str": ("+" if value_float > 1e-12 else "−" if value_float < -1e-12 else "") + f"{abs(value_float) * 100:g}%"}
            for value_float in tick_value_list],
        month_list=[{"x": round(x_value_float + 1, 1), "label_str": label_str} for x_value_float, label_str in kept_list],
        path_str="M" + " L".join(f"{x_value_float:.1f} {y_value_float:.1f}" for x_value_float, y_value_float in coordinate_list),
        end_x=round(coordinate_list[-1][0], 1), end_y=round(coordinate_list[-1][1], 1),
        end_label_str=_percent_str(value_list[-1]))
    return result_dict


def _class_str(symbol_str):
    # A symbol outside the fund list is a single stock.
    return SYMBOL_CLASS_DICT.get(symbol_str, "Stocks")


def _pod_holdings_dict(pod_dict, *, client_dict, snapshot_obj, provider_obj, as_of_ts, database_path_str,
                       cash_by_pod_dict=None):
    """Valued holdings at the Pod's close from one saved IBKR source; never values from another day."""
    close_str = pod_dict["close_str"]
    result_dict = {"available_bool": False, "reason_str": f"Holdings on {_short_day_str(close_str)} were not saved.",
        "position_list": [], "cash_float": None, "total_float": None}
    nav_row_list = [row_obj for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str == pod_dict["account_route_str"] and row_obj.market_date_str == close_str]
    if len(nav_row_list) != 1:
        return result_dict
    position_list, cash_float = None, None
    try:
        # 1. IBKR Flex closing positions of this exact close (the Pod page's source).
        if client_dict.get("is_demo") and hasattr(provider_obj, "get_close_holdings_dict"):
            close_dict = provider_obj.get_close_holdings_dict(nav_row_list[0], query_name_str=client_dict["query_name"], as_of_ts=as_of_ts)
        elif database_path_str:
            close_dict = load_close_holdings_dict(database_path_str, nav_row_list[0], query_name_str=client_dict["query_name"], as_of_ts=as_of_ts)
        else:
            close_dict = {"available_bool": False}
        if close_dict.get("available_bool"):
            position_list = [(item_dict["symbol_str"], item_dict["value_float"]) for item_dict in close_dict["position_list"]
                if item_dict["shares_float"] != 0]
            # *** CRITICAL *** exact-close accounting: cash must be independently
            # observed on this date. NAV minus positions includes accruals, not just cash.
            cash_float = (cash_by_pod_dict or {}).get(pod_dict["pod_id_str"])
            if client_dict.get("is_demo") and close_dict.get("cash_date_str") == close_str:
                cash_float = close_dict.get("cash_float")
            if cash_float is None:
                result_dict["reason_str"] = f"Holdings on {_short_day_str(close_str)} lack verified same-day cash."
        # 2. The saved IBKR portfolio, only when its end-of-day capture is this close.
        elif not client_dict.get("is_demo") and client_dict.get("operations_source") == "local" and hasattr(provider_obj, "get_target_for_pod"):
            target_obj = provider_obj.get_target_for_pod(pod_dict["pod_id_str"])
            if target_obj is not None and (target_obj.release_obj.pod_id_str, target_obj.release_obj.account_route_str) == (
                    pod_dict["pod_id_str"], pod_dict["account_route_str"]):
                broker_dict = load_broker_holdings_dict(target_obj, as_of_ts=as_of_ts)
                observed_ts = datetime.fromisoformat(broker_dict["observed_timestamp_str"]).astimezone(MARKET_TIMEZONE_OBJ) if broker_dict["available_bool"] else None
                if observed_ts is not None and observed_ts.date().isoformat() == close_str and observed_ts.time() >= time(16):
                    position_list = [(item_dict["symbol_str"], item_dict["value_float"]) for item_dict in broker_dict["position_list"]]
                    cash_float = broker_dict["cash_float"]
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        position_list = None
    if position_list is None or cash_float is None or not all(math.isfinite(value_float) for _, value_float in position_list):
        return result_dict
    total_float = sum(value_float for _, value_float in position_list) + cash_float
    if not math.isfinite(total_float) or total_float <= 0:
        return result_dict
    total_decimal = sum((Decimal(str(value_float)) for _, value_float in position_list), Decimal(str(cash_float)))
    if abs(total_decimal - nav_row_list[0].closing_nav_decimal) > Decimal(".01"):
        result_dict["reason_str"] = f"Holdings on {_short_day_str(close_str)} do not reconcile to account NAV."
        return result_dict
    result_dict.update(available_bool=True, reason_str="", position_list=sorted(position_list, key=lambda pair_tuple: -pair_tuple[1]),
        cash_float=cash_float, total_float=float(nav_row_list[0].closing_nav_decimal))
    return result_dict


def _program_holding_dict(program_name_str, pod_list, holdings_by_pod_dict):
    """One donut by asset class for the program's Pods at the same close."""
    result_dict = {"available_bool": False, "reason_str": "", "aria_str": "", "segment_list": [], "legend_list": []}
    active_list = [pod_dict for pod_dict in pod_list if pod_dict["active_bool"]]
    missing_list = [pod_dict for pod_dict in active_list if not holdings_by_pod_dict[pod_dict["pod_id_str"]]["available_bool"]]
    if not active_list or missing_list:
        result_dict["reason_str"] = (holdings_by_pod_dict[missing_list[0]["pod_id_str"]]["reason_str"] if missing_list
            else "No holdings at this close.")
        return result_dict
    value_by_class_dict, symbol_by_class_dict, total_float = {}, {}, 0.
    for pod_dict in active_list:
        holdings_dict = holdings_by_pod_dict[pod_dict["pod_id_str"]]
        total_float += holdings_dict["total_float"]
        for symbol_str, value_float in holdings_dict["position_list"]:
            class_str = _class_str(symbol_str)
            value_by_class_dict[class_str] = value_by_class_dict.get(class_str, 0.) + value_float
            symbol_by_class_dict.setdefault(class_str, []).append(symbol_str)
        value_by_class_dict["Cash"] = value_by_class_dict.get("Cash", 0.) + holdings_dict["cash_float"]
    # A short or negative cash is never hidden by netting it inside a class.
    if (any(value_float < 0 for pod_dict in active_list for _, value_float in holdings_by_pod_dict[pod_dict["pod_id_str"]]["position_list"])
            or any(holdings_by_pod_dict[pod_dict["pod_id_str"]]["cash_float"] < 0 for pod_dict in active_list)):
        result_dict["reason_str"] = "Holdings include borrowing or short positions. Details lists them."
        return result_dict
    item_list = []
    for class_str, color_str in ASSET_CLASS_TUPLE:
        weight_float = value_by_class_dict.get(class_str, 0.) / total_float
        if _rounded_decimal(weight_float * 100, ".1") == 0:
            continue
        symbol_list = list(dict.fromkeys(symbol_by_class_dict.get(class_str, [])))
        stock_count_int = sum(symbol_str not in SYMBOL_CLASS_DICT for symbol_str in symbol_list)
        # Weights are by value; a leveraged fund carries its multiple next to the ticker.
        detail_str = (f"{len(symbol_list)} stocks" if class_str == "Stocks" and stock_count_int > 3 else
            ", ".join(symbol_str + LEVERAGE_SUFFIX_DICT.get(symbol_str, "") for symbol_str in symbol_list)
            if len(symbol_list) <= 3 else f"{len(symbol_list)} funds")
        item_list.append({"weight_float": weight_float, "color_str": color_str, "name_str": class_str,
            "pct_str": _percent_str(weight_float, signed_bool=False), "detail_str": detail_str})
    result_dict.update(available_bool=True,
        aria_str=f"{program_name_str} holdings: " + ", ".join(f"{item_dict['name_str']} {item_dict['pct_str']}" for item_dict in item_list),
        segment_list=_donut_segment_list([(item_dict["weight_float"], item_dict["color_str"]) for item_dict in item_list
            if item_dict["weight_float"] > 0], center_float=52, radius_float=40),
        legend_list=[{key_str: item_dict[key_str] for key_str in ("color_str", "name_str", "pct_str", "detail_str")} for item_dict in item_list])
    return result_dict


def _pod_holding_table_dict(pod_dict, color_str, holdings_dict):
    """Page 2: every holding of one Pod as a share of the Pod's value, then cash."""
    result_dict = {"color_str": color_str, "name_str": pod_dict["name_str"], "row_list": [], "name_bool": False,
        "reason_str": holdings_dict["reason_str"]}
    if not holdings_dict["available_bool"]:
        return result_dict
    total_float, position_list = holdings_dict["total_float"], holdings_dict["position_list"]
    shown_list = position_list if len(position_list) <= HOLDING_ROW_LIMIT_INT else position_list[:HOLDING_ROW_LIMIT_INT - 1]
    row_list = [{"symbol_str": symbol_str, "name_str": SYMBOL_NAME_DICT.get(symbol_str, ""),
        "weight_str": _percent_str(value_float / total_float, signed_bool=False), "muted_bool": False}
        for symbol_str, value_float in shown_list]
    if len(shown_list) < len(position_list):
        rest_list = position_list[len(shown_list):]
        row_list.append({"symbol_str": f"{len(rest_list)} more", "name_str": "", "muted_bool": True,
            "weight_str": _percent_str(sum(value_float for _, value_float in rest_list) / total_float, signed_bool=False)})
    row_list.append({"symbol_str": "Cash", "name_str": "", "muted_bool": True,
        "weight_str": _percent_str(holdings_dict["cash_float"] / total_float, signed_bool=False)})
    result_dict.update(row_list=row_list, name_bool=any(row_dict["name_str"] for row_dict in row_list), reason_str="")
    return result_dict


def _flow_dict(account_list, snapshot_obj, pod_list, *, from_str, to_str, put_in_float, display_from_str=None):
    """Page 2: each Pod's value at its start, then every deposit, withdrawal and move.

    Rows sum to the money put in (closing value minus gain since start); any gap
    withholds the table instead of showing numbers that do not add up.
    """
    result_dict = {"row_list": [], "pod_name_list": [{"head_str": pod_dict.get("head_str", pod_dict["name_str"]),
        "alias_str": pod_dict.get("alias_str", "")} for pod_dict in pod_list],
        "pod_total_list": [], "total_str": "", "reason_str": "Money in and out could not be matched to the account values."}
    if put_in_float is None:
        result_dict["reason_str"] = "Money in and out appears once the account history since start is complete."
        return result_dict
    pod_index_dict = {pod_dict["pod_id_str"]: index_int for index_int, pod_dict in enumerate(pod_list)}
    row_by_key_dict = {(row_obj.account_route_str, row_obj.market_date_str): row_obj for row_obj in snapshot_obj.row_tuple
        if from_str <= row_obj.market_date_str <= to_str}
    event_dict = {}

    def add_event(date_str, kind_int, pod_id_str, amount_decimal):
        cell_list = event_dict.setdefault((date_str, kind_int), [Decimal(0)] * len(pod_list))
        cell_list[pod_index_dict[pod_id_str]] += amount_decimal

    try:
        for account_dict in account_list:
            first_str = max(from_str, account_dict["effective_from"])
            last_str = min(to_str, account_dict.get("effective_to") or to_str)
            date_list = sorted(date_str for route_str, date_str in row_by_key_dict
                if route_str == account_dict["account_route"] and first_str <= date_str <= last_str)
            if first_str > last_str:
                continue
            if not date_list:
                return result_dict
            add_event(date_list[0], 0, account_dict["pod_id"], row_by_key_dict[(account_dict["account_route"], date_list[0])].opening_nav_decimal)
            for date_str in date_list:
                attribute_dict = row_by_key_dict[(account_dict["account_route"], date_str)].attribute_dict
                capital_decimal = sum((_field_decimal(attribute_dict, field_str) for field_str in CAPITAL_FIELD_TUPLE), Decimal(0))
                linking_decimal = _field_decimal(attribute_dict, "linkingAdjustments")
                if capital_decimal:
                    add_event(date_str, 1, account_dict["pod_id"], capital_decimal)
                if linking_decimal:
                    add_event(date_str, 2, account_dict["pod_id"], linking_decimal)
            if account_dict.get("effective_to") and account_dict["effective_to"] < to_str:
                add_event(date_list[-1], 3, account_dict["pod_id"], -row_by_key_dict[(account_dict["account_route"], date_list[-1])].closing_nav_decimal)
    except (KeyError, InvalidOperation, ValueError, TypeError):
        return result_dict
    event_list = sorted(event_dict.items())
    total_list = [sum((cell_list[index_int] for _, cell_list in event_list), Decimal(0)) for index_int in range(len(pod_list))]
    if abs(sum(total_list, Decimal(0)) - Decimal(str(put_in_float))) > Decimal(".05"):
        return result_dict
    row_list = []
    for (date_str, kind_int), cell_list in event_list:
        total_decimal = sum(cell_list, Decimal(0))
        kind_str = ("Opening capital", "Net inflow" if total_decimal > 0 else "Net outflow" if total_decimal < 0 else "Offsetting movements",
            "Broker adjustment", "Pod closed, money out")[kind_int]
        if display_from_str is None or date_str >= display_from_str:
            row_list.append({"date_str": f"{_short_day_str(date_str)}, {date_str[:4]}", "kind_str": kind_str, "cell_list": cell_list})
    if len(row_list) > FLOW_ROW_LIMIT_INT:
        early_list, row_list = row_list[:len(row_list) - FLOW_ROW_LIMIT_INT + 1], row_list[len(row_list) - FLOW_ROW_LIMIT_INT + 1:]
        row_list.insert(0, {"date_str": "Before " + row_list[0]["date_str"], "kind_str": "Earlier, net",
            "cell_list": [sum((row_dict["cell_list"][index_int] for row_dict in early_list), Decimal(0)) for index_int in range(len(pod_list))]})
    for row_dict in row_list:
        total_decimal = sum(row_dict["cell_list"], Decimal(0))
        row_dict.update(total_str=_money_str(total_decimal, signed_bool=row_dict["kind_str"] != "Opening capital", cents_bool=True),
            total_tone_str=_money_tone_str(total_decimal),
            cell_list=[{"value_str": _money_str(cell_decimal, signed_bool=row_dict["kind_str"] != "Opening capital", cents_bool=True) if cell_decimal else "",
                "tone_str": _money_tone_str(cell_decimal)} for cell_decimal in row_dict["cell_list"]])
    result_dict.update(row_list=row_list, reason_str="", pod_total_list=[_money_str(total_decimal) for total_decimal in total_list],
        total_str=_money_str(sum(total_list, Decimal(0))))
    return result_dict


def _activity_list(account_list, snapshot_obj, *, from_str, to_str):
    """Page 2: the month's costs and income from the same saved IBKR rows."""
    row_list = [row_obj for account_dict in account_list for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str == account_dict["account_route"]
        and max(from_str, account_dict["effective_from"]) <= row_obj.market_date_str <= min(to_str, account_dict.get("effective_to") or to_str)]

    def total_decimal(field_tuple):
        try:
            return sum((_field_decimal(row_obj.attribute_dict, field_str) for row_obj in row_list for field_str in field_tuple), Decimal(0))
        except (KeyError, InvalidOperation, ValueError):
            return None

    result_list = []
    commission_decimal, dividend_decimal = total_decimal(("commissions",)), total_decimal(("dividends",))
    tax_decimal, interest_decimal = total_decimal(("withholdingTax",)), total_decimal(("interest",))
    capital_decimal = total_decimal(CAPITAL_FIELD_TUPLE)
    if commission_decimal is not None:
        result_list.append({"label_str": "Commissions", "value_str": _money_str(-commission_decimal, cents_bool=True)})
    if dividend_decimal is not None:
        result_list.append({"label_str": "Dividends paid", "value_str": _money_str(dividend_decimal, cents_bool=True)})
    if tax_decimal:
        result_list.append({"label_str": "Tax withheld", "value_str": _money_str(-tax_decimal, cents_bool=True)})
    if interest_decimal is not None:
        result_list.append({"label_str": "Interest posted", "value_str": _money_str(interest_decimal, cents_bool=True)})
    if capital_decimal is not None:
        result_list.append({"label_str": "Added or withdrawn", "value_str": _money_str(capital_decimal, signed_bool=True)})
    return result_list if row_list else []


def _month_column_list(start_str, month_str):
    month_list, cursor_obj = [], date.fromisoformat(month_str + "-01")
    while len(month_list) < MONTH_COLUMN_LIMIT_INT and cursor_obj.isoformat()[:7] >= start_str[:7]:
        month_list.insert(0, cursor_obj.isoformat()[:7])
        cursor_obj = date(cursor_obj.year - (cursor_obj.month == 1), (cursor_obj.month - 2) % 12 + 1, 1)
    return month_list


def _month_row_dict(class_str, name_str, return_dict, since_float, month_list):
    return {"class_str": class_str, "name_str": name_str, "since_str": _percent_str(since_float, suffix_str=""),
        "since_tone_str": _percent_tone_str(since_float),
        "cell_list": [{"value_str": _percent_str(return_dict[month_str], suffix_str="") if month_str in return_dict else "",
            "tone_str": _percent_tone_str(return_dict.get(month_str))} for month_str in month_list]}


def build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, *, as_of_ts, month_str=None,
                              database_path_str=None, program_by_pod_dict=None, alias_by_pod_dict=None,
                              month_url_fn=lambda month_str: "?month=" + month_str, art_url_fn=lambda month_str: ""):
    """The report for one month; None when that month is outside the saved history.

    Totals and returns are the Performance page's for the same dates: the month is
    [max(first day, first measured day), month end or latest close], "since start"
    begins on the first measured day. A month in progress, or any day the saved
    accounts cannot prove, makes the report a marked draft.
    """
    if as_of_ts.tzinfo is None:
        raise ValueError("A timezone-aware clock is required.")
    result_dict = _empty_dict()
    # The configured name is often a technical id (the releases' user id); only a name with a space is shown.
    configured_name_str = " ".join(str(workspace_dict.get("client_dict", {}).get("display_name") or "").split())
    result_dict["client_name_str"] = configured_name_str if " " in configured_name_str else ""
    if workspace_dict.get("financial_error_str") or snapshot_obj.unavailable_reason_str:
        result_dict["error_str"] = "Saved financial data could not be verified."
        return result_dict
    strategy_view_dict = strategy_workspace_dict(workspace_dict)
    account_list = strategy_view_dict["client_dict"]["accounts"]
    if not account_list:
        # The report covers programs; other accounts alone (such as VAULT) are not a report.
        result_dict["error_str"] = ("No program has started yet." if workspace_dict["client_dict"]["accounts"]
            else "No account has saved performance yet.")
        return result_dict
    # "Since start" is the first Pod day. Once every Pod has closed, the report ends with the last one.
    start_str = min(account_dict["effective_from"] for account_dict in account_list)
    end_str = (max(account_dict["effective_to"] for account_dict in account_list)
        if all(account_dict.get("effective_to") for account_dict in account_list) else None)
    try:
        client_dict, _, valuation_list = report_scope_tuple(workspace_dict, snapshot_obj, as_of_ts=as_of_ts)
        horizon_account_list = client_dict["accounts"] if valuation_list is None else valuation_list
        today_str = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
        # *** CRITICAL *** D+1 source horizon, not a last-complete-day fallback.
        # Any owned account (including VAULT) can establish the requested date;
        # the canonical report must still prove every active account on every day.
        latest_str = max((row_obj.market_date_str for row_obj in snapshot_obj.row_tuple
            if row_obj.market_date_str < today_str and any(
                account_dict["account_route"] == row_obj.account_route_str
                and account_dict.get("effective_from", start_str) <= row_obj.market_date_str
                <= (account_dict.get("effective_to") or "9999-12-31")
                for account_dict in horizon_account_list)), default="")
    except (ClientReportingError, ValueError, TypeError, KeyError, AttributeError):
        latest_str = ""
    if latest_str < start_str:
        result_dict["error_str"] = "No finalized account value is available yet."
        return result_dict
    option_list = _month_option_list(start_str, latest_str, end_str)
    if month_str is None:
        option_dict = next((item_dict for item_dict in option_list if item_dict["complete_bool"]), option_list[0])
    else:
        option_dict = next((item_dict for item_dict in option_list if item_dict["month_str"] == month_str), None)
        if option_dict is None:
            return None
    month_str, complete_bool = option_dict["month_str"], option_dict["complete_bool"]
    month_first_str, month_last_str = _month_bounds_tuple(month_str)
    from_str, to_str = max(month_first_str, start_str), min(month_last_str, latest_str, end_str or month_last_str)
    month_obj = date.fromisoformat(month_first_str)
    holder_name_str = "" if workspace_dict["client_dict"].get("is_demo") else load_holder_name_str(
        database_path_str, {account_dict["account_route"] for account_dict in workspace_dict["client_dict"]["accounts"]},
        query_name_str=workspace_dict["client_dict"]["query_name"])
    result_dict.update(month_str=month_str, month_label_str=month_obj.strftime("%B %Y"),
        client_name_str=holder_name_str or result_dict["client_name_str"],
        month_option_list=[{"url_str": month_url_fn(item_dict["month_str"]), "label_str": item_dict["label_str"],
            "selected_bool": item_dict["month_str"] == month_str} for item_dict in option_list],
        art_url_str=art_url_fn(month_str))
    try:
        month_page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=as_of_ts,
            from_date_str=from_str, to_date_str=to_str, level_str="pods")
        since_page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=as_of_ts,
            from_date_str=start_str, to_date_str=to_str, level_str="pods")
    except ValueError:
        result_dict["error_str"] = "Saved financial data could not be verified for this month."
        return result_dict
    month_report_dict, since_report_dict = month_page_dict["report_dict"], since_page_dict["report_dict"]
    if not (since_report_dict or month_report_dict):
        result_dict["error_str"] = month_page_dict["error_str"] or "Saved financial data could not be verified for this month."
        return result_dict
    pod_source_dict = since_report_dict or month_report_dict
    # Partial results stay visible as "—"; the page is then a marked draft.
    month_report_dict = month_report_dict or {}
    since_report_dict = since_report_dict or {}
    summary_month_dict, summary_since_dict = month_report_dict, since_report_dict
    # Other accounts are named only: their USD value would mostly be the shekel rate, not the programs.
    other_name_list = list(dict.fromkeys(account_dict["display_name"] for account_dict in workspace_dict["client_dict"]["accounts"]
        if account_dict.get("role") == "reserve" and account_dict["effective_from"] <= to_str
        and (account_dict.get("effective_to") or "9999-12-31") >= from_str))
    close_str = summary_month_dict.get("closing_date_str") or summary_since_dict.get("closing_date_str") or to_str
    month_suffix_str = "" if complete_bool else f" · to {_short_day_str(close_str)}"
    result_dict.update(available_bool=True, period_label_str=result_dict["month_label_str"] + month_suffix_str,
        month_name_str=calendar.month_name[month_obj.month] + month_suffix_str)

    # *** CRITICAL *** headline NAV, P&L and TWR cover the programs only, the
    # same scope as the Performance page. Never add other accounts here: IBKR
    # translates their non-USD cash, so their USD change is mostly FX.
    closing_float = summary_since_dict.get("closing_nav_float", summary_month_dict.get("closing_nav_float"))
    since_pnl_float = summary_since_dict.get("pnl_float")
    put_in_float = closing_float - since_pnl_float if closing_float is not None and since_pnl_float is not None else None
    put_in_decimal = (_rounded_decimal(closing_float, "1") - _rounded_decimal(since_pnl_float, "1")
        if put_in_float is not None else None)
    money_row_list = [
        {"class_str": "lead", "label_str": "Value on " + _long_day_str(close_str),
            "value_str": _money_str(closing_float), "tone_str": "", "pct_str": "", "pct_tone_str": ""},
        {"class_str": "", "label_str": "Investment gain/loss · " + calendar.month_name[month_obj.month],
            "value_str": _money_str(summary_month_dict.get("pnl_float"), signed_bool=True), "tone_str": _money_tone_str(summary_month_dict.get("pnl_float")),
            "pct_str": _percent_str(summary_month_dict.get("twr_float")), "pct_tone_str": _percent_tone_str(summary_month_dict.get("twr_float"))},
        {"class_str": "", "label_str": "Gain since start", "value_str": _money_str(since_pnl_float, signed_bool=True),
            "tone_str": _money_tone_str(since_pnl_float), "pct_str": _percent_str(summary_since_dict.get("twr_float")),
            "pct_tone_str": _percent_tone_str(summary_since_dict.get("twr_float"))},
        {"class_str": "", "label_str": "Opening capital + net movements", "value_str": _money_str(put_in_decimal), "tone_str": "",
            "pct_str": "", "pct_tone_str": ""},
    ]
    result_dict["money_row_list"] = money_row_list
    result_dict["summary_scope_str"] = "Your portfolio · USD"
    result_dict["bridge_row_list"] = [{"label_str": label_str, "value_str": _money_str(summary_month_dict.get(field_str), signed_bool=signed_bool, cents_bool=True)}
        for label_str, field_str, signed_bool in (("Opening value", "opening_nav_float", False),
            ("Net capital movements", "capital_movement_float", True), ("Investment gain/loss", "pnl_float", True),
            ("Closing value", "closing_nav_float", False))]
    for label_str, field_str in (("Accounts entering/leaving", "scope_movement_float"), ("Broker adjustments", "linking_adjustment_float")):
        if summary_month_dict.get(field_str):
            result_dict["bridge_row_list"].insert(-1, {"label_str": label_str, "value_str": _money_str(summary_month_dict[field_str], signed_bool=True, cents_bool=True)})

    # Programs retain their historical identity even after the last Pod closes.
    # Pod numbers follow the first measured day (then account order, as on Performance), so a Pod keeps
    # its number when others are added or retired.
    pod_number_dict = {pod_id_str: index_int + 1 for index_int, pod_id_str in enumerate(dict.fromkeys(account_dict["pod_id"]
        for _, account_dict in sorted(enumerate(account_list), key=lambda pair_tuple: (pair_tuple[1]["effective_from"], pair_tuple[0]))))}
    pod_list = _pod_list(pod_source_dict, {account_dict["pod_id"] for account_dict in account_list
        if account_dict["effective_from"] <= close_str <= (account_dict.get("effective_to") or "9999-12-31")},
        lambda pod_id_str, route_str: _pod_label_tuple(pod_number_dict[pod_id_str], route_str, (alias_by_pod_dict or {}).get(pod_id_str, "")))
    program_by_pod_dict = program_by_pod_dict or {}
    # Case never makes a second program: the first spelling seen names it.
    display_by_key_dict = {}
    for pod_dict in pod_list:
        raw_str = program_by_pod_dict.get(pod_dict["pod_id_str"]) or DEFAULT_PROGRAM_STR
        display_by_key_dict.setdefault(raw_str.casefold(), raw_str)

    def program_name_str(pod_dict):
        return display_by_key_dict[(program_by_pod_dict.get(pod_dict["pod_id_str"]) or DEFAULT_PROGRAM_STR).casefold()]

    program_name_list = list(dict.fromkeys(program_name_str(pod_dict) for pod_dict in pod_list))
    family_index_dict, free_list = {}, [index_int for index_int in range(len(PROGRAM_FAMILY_TUPLE))
        if index_int not in {PROGRAM_FAMILY_INDEX_DICT.get(name_str) for name_str in program_name_list}]
    for program_index_int, name_str in enumerate(program_name_list):
        family_index_dict[name_str] = PROGRAM_FAMILY_INDEX_DICT.get(name_str)
        if family_index_dict[name_str] is None:
            family_index_dict[name_str] = free_list.pop(0) if free_list else program_index_int % len(PROGRAM_FAMILY_TUPLE)
    pod_color_dict = {}
    for name_str in program_name_list:
        family_tuple = PROGRAM_FAMILY_TUPLE[family_index_dict[name_str]]
        active_member_list = [pod_dict for pod_dict in pod_list if program_name_str(pod_dict) == name_str and pod_dict["active_bool"]]
        for index_int, pod_dict in enumerate(active_member_list):
            pod_color_dict[pod_dict["pod_id_str"]] = family_tuple[min(index_int, len(family_tuple) - 1)]
    single_bool = len(program_name_list) == 1
    try:
        operations_dict = build_client_operations_dict(workspace_dict["client_dict"], workspace_dict.get("summary_dict", {}),
            as_of_ts=as_of_ts, local_account_list=workspace_dict.get("operations_account_list"))
        cash_list = load_portfolio_cash_list(workspace_dict["client_dict"], pod_source_dict, provider_obj, operations_dict, as_of_ts=as_of_ts)
    except (ValueError, TypeError, KeyError, AttributeError):
        cash_list = []
    cash_by_pod_dict = {row_dict["pod_id_str"]: row_dict["cash_float"] for row_dict in cash_list
        if row_dict["market_date_str"] == close_str and isinstance(row_dict.get("cash_float"), (int, float))
        and math.isfinite(row_dict["cash_float"])}
    holdings_by_pod_dict = {pod_dict["pod_id_str"]: _pod_holdings_dict(pod_dict, client_dict=workspace_dict["client_dict"],
        snapshot_obj=snapshot_obj, provider_obj=provider_obj, as_of_ts=as_of_ts, database_path_str=database_path_str,
        cash_by_pod_dict=cash_by_pod_dict)
        if pod_dict["active_bool"] else {"available_bool": False, "reason_str": ""} for pod_dict in pod_list}
    active_total_float = closing_float or 0.
    month_list = _month_column_list(start_str, month_str)
    program_list, group_list, split_item_list, month_row_list, program_status_list = [], [], [], [], []
    program_evidence_dict = {}
    for name_str in program_name_list:
        member_list = [pod_dict for pod_dict in pod_list if program_name_str(pod_dict) == name_str]
        closed_bool = not any(pod_dict["active_bool"] for pod_dict in member_list)
        program_to_str = min(to_str, max(pod_dict["close_str"] for pod_dict in member_list)) if closed_bool else to_str
        color_str = PROGRAM_FAMILY_TUPLE[family_index_dict[name_str]][0]
        if single_bool and not closed_bool:
            program_report_dict = since_report_dict
        else:
            program_page_dict = build_performance_page_dict(_program_workspace_dict(workspace_dict, {pod_dict["pod_id_str"] for pod_dict in member_list}),
                snapshot_obj, as_of_ts=as_of_ts, from_date_str=min(pod_dict["from_str"] for pod_dict in member_list), to_date_str=program_to_str,
                level_str="pods", transfer_scope_client_dict=workspace_dict["client_dict"])
            program_report_dict = program_page_dict["report_dict"] or {}
        program_status_list.append(program_report_dict.get("status_str"))
        path_list, daily_list = _return_series_tuple(program_report_dict) if program_report_dict else ([], [])
        program_evidence_dict[name_str] = {"report_hash_str": program_report_dict.get("report_hash_str"),
            "transfer_scope_dict": program_report_dict.get("transfer_scope_dict")}
        program_start_str = min(pod_dict["from_str"] for pod_dict in member_list)
        program_value_float = 0. if closed_bool else program_report_dict.get("closing_nav_float")
        month_return_float = _linked_return_float(daily_list, from_str, to_str)
        month_pnl_float = _window_pnl_float(daily_list, from_str, to_str)
        since_float = program_report_dict.get("twr_float")
        stat_list = [{"label_str": calendar.month_name[month_obj.month], "value_str": _percent_str(month_return_float),
            "tone_str": _percent_tone_str(month_return_float)}]
        if program_start_str[:4] < month_str[:4]:
            year_float = _linked_return_float(daily_list, month_str[:4] + "-01-01", to_str)
            stat_list.append({"label_str": month_str[:4], "value_str": _percent_str(year_float), "tone_str": _percent_tone_str(year_float)})
        stat_list.append({"label_str": "Since start", "value_str": _percent_str(since_float), "tone_str": _percent_tone_str(since_float)})
        holding_dict = _program_holding_dict(name_str, member_list, holdings_by_pod_dict)
        table_list = [_pod_holding_table_dict(pod_dict, pod_color_dict[pod_dict["pod_id_str"]], holdings_by_pod_dict[pod_dict["pod_id_str"]])
            for pod_dict in member_list if pod_dict["active_bool"]]
        program_list.append({"name_str": name_str, "color_str": color_str,
            "since_str": "from " + program_start_str + (" · closed " + program_to_str if closed_bool else ""),
            "closed_bool": closed_bool,
            "value_str": _money_str(program_value_float),
            "change_str": _money_str(month_pnl_float, signed_bool=True) + " this month", "change_tone_str": _money_tone_str(month_pnl_float),
            "stat_list": stat_list, "chart_dict": _chart_dict(path_list, single_bool=single_bool, wide_bool=not holding_dict["available_bool"]),
            # Print only evidenced holdings; missing-data reasons also print below.
            "holding_dict": holding_dict, "pod_holding_list": [table_dict for table_dict in table_list if table_dict["row_list"]]})
        share_list = [(pod_dict, pod_dict["closing_float"] / active_total_float) for pod_dict in member_list
            if pod_dict["active_bool"] and pod_dict["closing_float"] and active_total_float > 0]
        if share_list:
            group_list.append({"name_str": name_str, "pct_str": _percent_str(sum(share_float for _, share_float in share_list), signed_bool=False),
                "pod_list": [{"name_str": pod_dict["name_str"], "head_str": pod_dict["head_str"], "alias_str": pod_dict["alias_str"],
                    "pct_str": _percent_str(share_float, signed_bool=False), "color_str": pod_color_dict[pod_dict["pod_id_str"]]}
                    for pod_dict, share_float in share_list]})
            split_item_list += [(share_float, pod_color_dict[pod_dict["pod_id_str"]], name_str, pod_dict["name_str"]) for pod_dict, share_float in share_list]
        month_row_list.append(_month_row_dict("prog", name_str, _monthly_return_dict(daily_list), since_float, month_list))
        # Without the since-start report the Pods come from the month alone: no "since start" figure.
        month_row_list += [_month_row_dict("pod", pod_dict["name_str"], pod_dict["month_return_dict"],
                pod_dict["twr_float"] if since_report_dict else None, month_list)
            for pod_dict in member_list if any(month_str in pod_dict["month_return_dict"] for month_str in month_list)]
    # This is the strategy-capital ledger, separate from the all-account summary.
    strategy_put_in_float = (since_report_dict["closing_nav_float"] - since_report_dict["pnl_float"]
        if since_report_dict.get("closing_nav_float") is not None and since_report_dict.get("pnl_float") is not None else None)
    flow_dict = (_flow_dict(account_list, snapshot_obj, pod_list, from_str=start_str, to_str=to_str,
        put_in_float=strategy_put_in_float, display_from_str=from_str) if account_list else result_dict["flow_dict"])
    result_dict.update(single_program_bool=single_bool, program_list=program_list,
        # The Programs panel lists every Pod in the report with its current program ("" = Portfolio).
        pod_choice_list=[{"pod_id_str": pod_dict["pod_id_str"], "name_str": pod_dict["name_str"],
            "program_str": program_by_pod_dict.get(pod_dict["pod_id_str"], "")} for pod_dict in pod_list],
        split_dict={"group_list": group_list,
            "segment_list": _donut_segment_list([(share_float, color_str) for share_float, color_str, _, _ in split_item_list], center_float=56, radius_float=44),
            "aria_str": ". ".join(f"{group_dict['name_str']} {group_dict['pct_str']}: " + ", ".join(
                f"{pod_dict['name_str']} {pod_dict['pct_str']}" for pod_dict in group_dict["pod_list"]) for group_dict in group_list)},
        month_table_dict={"row_list": month_row_list, "label_list": [
            date.fromisoformat(value_str + "-01").strftime("%b") + (" ’" + value_str[2:4] if len({item_str[:4] for item_str in month_list}) > 1
                and (index_int == 0 or value_str.endswith("-01")) else "") for index_int, value_str in enumerate(month_list)]},
        flow_dict=flow_dict, activity_list=_activity_list(account_list, snapshot_obj, from_str=from_str, to_str=to_str))

    # Draft: the badge prints a short reason; the screen bar adds the detail.
    detail_str = ""
    if not complete_bool:
        draft_reason_str = ("Final month-end statement not saved yet" if option_dict["sessions_done_bool"]
            else f"Month in progress, through {_short_day_str(close_str)}")
    elif summary_month_dict.get("status_str") != "ready":
        draft_reason_str, detail_str = "Month data incomplete", next(iter(summary_month_dict.get("issue_list") or []), "")
    elif summary_since_dict.get("twr_float") is None:
        draft_reason_str, detail_str = "Return since start unavailable", summary_since_dict.get("twr_reason_str") or next(
            iter(summary_since_dict.get("issue_list") or []), "")
    elif summary_since_dict.get("status_str") != "ready":
        draft_reason_str, detail_str = "Data since start incomplete", next(iter(summary_since_dict.get("issue_list") or []), "")
    elif any(status_str != "ready" for status_str in program_status_list):
        draft_reason_str = "Program data incomplete"
    elif flow_dict["reason_str"]:
        draft_reason_str = "Capital movements do not reconcile"
    else:
        draft_reason_str = ""
    # The screen and issued copy both retain draft and missing-data explanations.
    note_list = [f"Draft · {draft_reason_str}"] if draft_reason_str else []
    note_list += [detail_str] if detail_str else []
    note_list += list(dict.fromkeys(program_dict["holding_dict"]["reason_str"] for program_dict in program_list
        if not program_dict["holding_dict"]["available_bool"] and program_dict["holding_dict"]["reason_str"]))
    if flow_dict["reason_str"]:
        note_list.append(flow_dict["reason_str"])
    result_dict.update(draft_bool=bool(draft_reason_str), draft_reason_str=draft_reason_str, status_note_str=" · ".join(note_list))
    result_dict["note_list"] = note_list
    result_dict["is_demo_bool"] = workspace_dict["client_dict"].get("is_demo") is True
    result_dict["method_note_str"] = ("Amounts in USD. Portfolio/program TWR is calculated from daily NAV with end-of-day capital flows; "
        "it is not official consolidated IBKR or exact intraday TWR. Pod rows use IBKR account TWR. "
        "Dollar gains include recorded costs, income and FX; they are not TWR multiplied by net contributions. "
        "Opening capital is the value at the start of measurement, not lifetime deposits. "
        "Initial/final months may be partial; each program's measurement dates are shown. "
        "Posted income/costs are already included in investment gains, together with accrual changes.")
    result_dict["scope_note_str"] = ("All values cover your programs. Your other IBKR accounts are not included: "
        + ", ".join(other_name_list) + "." if other_name_list else "All values cover the measured portfolio and its capital movements.")
    result_dict["fee_note_str"] = str(workspace_dict["client_dict"].get("fee_basis") or "Recorded broker costs are included.")
    result_dict["generated_at_str"] = as_of_ts.isoformat()
    result_dict["evidence_dict"] = {"month_report_hash_str": summary_month_dict.get("report_hash_str"),
        "since_report_hash_str": summary_since_dict.get("report_hash_str"), "program_by_pod_dict": program_by_pod_dict,
        "source_checksum_list": sorted({source_dict["checksum_str"] for source_dict in summary_since_dict.get("source_list", [])}),
        "method_version_str": summary_since_dict.get("method_version_str"), "renderer_version_str": "monthly_report_v2",
        "program_evidence_dict": program_evidence_dict}
    # Bind save/print to all displayed values, warnings and provenance, not just NAV.
    result_dict["report_id_str"] = content_hash_str({key_str: value_obj for key_str, value_obj in result_dict.items()
        if key_str != "generated_at_str"})[:16]
    return result_dict
