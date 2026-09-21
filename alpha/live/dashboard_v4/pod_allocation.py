"""Pure display geometry for a verified, single-close Pod holdings snapshot."""

from datetime import date
from decimal import Decimal
import math

from alpha.live.dashboard_v4.finance import _money_str, _ring_path_str, _share_str


def _finite_bool(value_obj):
    try:
        return type(value_obj) in {int, float} and math.isfinite(value_obj)
    except OverflowError:
        return False


def build_pod_allocation_dict(source_dict, *, color_str):
    """Display close quantities and values already verified by the source adapter.

    The adapter owns account/date provenance. This defensive accounting gate is
    abs(sum(position values) + cash - NAV) <= $0.01, using decimal input text.
    The caller may instead supply an explicitly labelled market-estimated total
    in nav_float; the official account NAV is never modified by that estimate.
    Every displayed weight is signed value / the supplied total; neither residual cash nor
    target weights nor entry tags are inferred. No reads or valuation occur.
    """
    result_dict = {"available_bool": False, "donut_available_bool": False,
        "close_date_str": "", "reason_str": "Closing position values unavailable",
        "cash_percent_str": "—", "slice_list": [], "label_list": [],
        "row_list": [], "cash_row_dict": {}, "holdings_changed_bool": False, "color_str": color_str}
    if not isinstance(source_dict, dict) or source_dict.get("available_bool") is not True:
        if isinstance(source_dict, dict) and isinstance(source_dict.get("reason_str"), str) and source_dict["reason_str"]:
            result_dict["reason_str"] = source_dict["reason_str"]
        return result_dict
    close_date_str = source_dict.get("close_date_str")
    try:
        if not isinstance(close_date_str, str) or date.fromisoformat(close_date_str).isoformat() != close_date_str:
            return result_dict
    except ValueError:
        return result_dict
    nav_float, cash_float = source_dict.get("nav_float"), source_dict.get("cash_float")
    position_list = source_dict.get("position_list")
    if not _finite_bool(nav_float) or nav_float <= 0 or not _finite_bool(cash_float) or not isinstance(position_list, list):
        return result_dict
    symbol_set = set()
    for position_dict in position_list:
        if not isinstance(position_dict, dict):
            return result_dict
        symbol_str = position_dict.get("symbol_str")
        shares_float, value_float = position_dict.get("shares_float"), position_dict.get("value_float")
        if (not isinstance(symbol_str, str) or not symbol_str.strip() or symbol_str != symbol_str.strip()
                or symbol_str in symbol_set or not _finite_bool(shares_float) or not _finite_bool(value_float)):
            return result_dict
        if ((shares_float == 0 and value_float != 0) or (shares_float > 0 and value_float < 0)
                or (shares_float < 0 and value_float > 0)):
            return result_dict
        # A row may repeat the adapter's close date, but cannot contradict it.
        if position_dict.get("close_date_str", close_date_str) != close_date_str:
            return result_dict
        symbol_set.add(symbol_str)
    nav_decimal = Decimal(str(nav_float))
    total_decimal = sum((Decimal(str(position_dict["value_float"])) for position_dict in position_list), Decimal(str(cash_float)))
    if abs(total_decimal - nav_decimal) > Decimal("0.01"):
        result_dict["reason_str"] = "Closing values do not match NAV"
        return result_dict
    weight_list = [float(Decimal(str(position_dict["value_float"])) / nav_decimal) for position_dict in position_list]
    cash_weight_float = float(Decimal(str(cash_float)) / nav_decimal)
    if not all(math.isfinite(weight_float) for weight_float in weight_list + [cash_weight_float]):
        return result_dict
    largest_weight_float = max([0.0] + weight_list)

    def row_dict(symbol_str, shares_float, value_float, weight_float, key_str):
        return {"key_str": key_str, "symbol_str": symbol_str,
            "shares_str": "" if shares_float is None else f"{shares_float:,.6f}".rstrip("0").rstrip("."),
            "value_str": _money_str(value_float).replace("$", ""), "weight_str": _share_str(weight_float),
            "shares_float": shares_float, "value_float": value_float, "weight_float": weight_float,
            # Bars compare holdings visually; the number remains value / NAV.
            "bar_width_float": (min(100.0, 100 * weight_float / largest_weight_float) if largest_weight_float else 0.0)
                if key_str != "cash" and 0 <= weight_float <= 1 else None,
            "new_bool": False, "target_percent_float": None}

    result_dict.update(available_bool=True, close_date_str=close_date_str, reason_str="",
        cash_percent_str=_share_str(cash_weight_float), holdings_changed_bool=source_dict.get("holdings_changed_bool") is True,
        cash_row_dict=row_dict("Cash", None, cash_float, cash_weight_float, "cash"))
    result_dict["row_list"] = sorted([row_dict(position_dict["symbol_str"], position_dict["shares_float"],
        position_dict["value_float"], weight_float, "position:" + position_dict["symbol_str"])
        for position_dict, weight_float in zip(position_list, weight_list)], key=lambda position_dict: position_dict["symbol_str"])
    if cash_float < 0 or any(position_dict["shares_float"] < 0 or position_dict["value_float"] < 0 for position_dict in position_list):
        result_dict["reason_str"] = "Short positions or negative cash: table only"
        return result_dict
    if cash_weight_float > 1 or any(weight_float > 1 for weight_float in weight_list):
        result_dict["reason_str"] = "Weight exceeds 100%: table only"
        return result_dict
    if total_decimal <= 0:
        result_dict["reason_str"] = "No positive closing values to chart"
        return result_dict
    result_dict["donut_available_bool"] = True
    slice_row_list = sorted(result_dict["row_list"], key=lambda position_dict: (-position_dict["value_float"], position_dict["symbol_str"]))
    slice_row_list.append(result_dict["cash_row_dict"])
    start_float = 0.0
    for position_dict in slice_row_list:
        if position_dict["value_float"] <= 0:
            continue
        # Geometry closes the ring across only the accepted <= one-cent rounding
        # difference. Displayed percentages retain the exact NAV denominator.
        fraction_float = float(Decimal(str(position_dict["value_float"])) / total_decimal)
        result_dict["slice_list"].append({"key_str": position_dict["key_str"], "label_str": position_dict["symbol_str"],
            "weight_str": position_dict["weight_str"], "value_str": position_dict["value_str"],
            "color_str": "#dfe3e9" if position_dict["key_str"] == "cash" else color_str,
            "path_str": _ring_path_str(start_float, fraction_float, 52, 82)})
        if position_dict["weight_float"] >= .15:
            angle_float = 2 * math.pi * (start_float + fraction_float / 2) - math.pi / 2
            result_dict["label_list"].append({"key_str": position_dict["key_str"], "label_str": position_dict["symbol_str"],
                "x_float": 116 + 67 * math.cos(angle_float), "y_float": 125 + 67 * math.sin(angle_float)})
        start_float += fraction_float
    return result_dict
