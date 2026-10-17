"""Treemap data for the Positions page: one block per Pod, one tile per holding.

Display only. Tile size is the dollar value (or the exposure) and colour is the
return; both come from figures the page already verified. The browser only lays
the tiles out (static/positions.js); every number and label is computed here.
"""

import math


# Daily-reset leveraged and inverse ETFs: index exposure per dollar held. Any
# other symbol is 1x. Exposure is a display aid; it never feeds sizing.
EXPOSURE_MULTIPLIER_DICT = {
    "TQQQ": 3.0, "QLD": 2.0, "PSQ": -1.0, "QID": -2.0, "SQQQ": -3.0,
    "UPRO": 3.0, "SPXL": 3.0, "SSO": 2.0, "SH": -1.0, "SDS": -2.0, "SPXU": -3.0, "SPXS": -3.0,
    "TMF": 3.0, "UBT": 2.0, "TBT": -2.0, "TMV": -3.0, "UGL": 2.0, "AGQ": 2.0,
}
TILE_LIMIT_INT = 400


def exposure_multiplier_float(symbol_str):
    return EXPOSURE_MULTIPLIER_DICT.get(symbol_str, 1.0)


def compact_money_str(value_float, *, signed_bool=False):
    """Whole dollars for tile and block labels; exact cents stay in tables."""
    if value_float is None or not math.isfinite(value_float):
        return "—"
    rounded_float = round(value_float)
    prefix_str = "−" if rounded_float < 0 else "+" if signed_bool and rounded_float > 0 else ""
    return f"{prefix_str}${abs(rounded_float):,.0f}"


def signed_percent_str(value_float):
    if value_float is None or not math.isfinite(value_float):
        return "—"
    if round(value_float, 3) == 0:
        return "0.0%"
    return ("+" if value_float > 0 else "−") + f"{abs(value_float):.1%}"


def _finite_bool(value_obj):
    return isinstance(value_obj, (int, float)) and not isinstance(value_obj, bool) and math.isfinite(value_obj)


def finite_tree_bool(value_obj):
    """True when no float anywhere is NaN/inf: JSON for the browser must parse."""
    if isinstance(value_obj, float):
        return math.isfinite(value_obj)
    if isinstance(value_obj, dict):
        return all(finite_tree_bool(item_obj) for item_obj in value_obj.values())
    if isinstance(value_obj, (list, tuple)):
        return all(finite_tree_bool(item_obj) for item_obj in value_obj)
    return True


def multiplier_note_str(multiplier_float):
    if multiplier_float == 1.0:
        return ""
    return f"{multiplier_float:+.0f}x".replace("+", "") + (" inverse" if multiplier_float < 0 else "")


def map_item_dict(*, key_str, symbol_str, pod_name_str, value_float, pnl_float, return_float,
                  tip_list, status_str="open", status_note_str=""):
    """One tile. value_float sizes the tile (absolute for shorts)."""
    multiplier_float = exposure_multiplier_float(symbol_str)
    tip_row_list = [list(row_tuple) for row_tuple in tip_list]
    if multiplier_float != 1.0:
        tip_row_list.append(["Exposure", compact_money_str(abs(value_float) * abs(multiplier_float))
            + " (" + multiplier_note_str(multiplier_float) + ")"])
    return {"key_str": key_str, "symbol_str": symbol_str, "pod_name_str": pod_name_str,
        "value_float": abs(value_float), "multiplier_float": multiplier_float,
        "pnl_float": pnl_float, "return_float": return_float,
        "pnl_str": compact_money_str(pnl_float, signed_bool=True) if pnl_float is not None else "",
        "return_str": signed_percent_str(return_float) if return_float is not None else "",
        "status_str": status_str, "status_note_str": status_note_str, "tip_list": tip_row_list}


def build_now_map_dict(map_source_list, pod_identity_list, *, pod_str, book_value_float, asof_str):
    """Current holdings from one verified IBKR valuation of every owned Pod.

    map_source_list is empty unless the page's money is complete: every Pod
    valued, quantities matched and one ET date. No partial book is drawn.
    """
    unavailable_dict = {"available_bool": False, "mode_str": "now", "pod_list": [],
        "reason_str": "The map needs current IBKR values for every Pod on one date."}
    if not map_source_list or not _finite_bool(book_value_float) or book_value_float <= 0:
        return unavailable_dict
    identity_by_pod_dict = {identity_dict["pod_id_str"]: identity_dict for identity_dict in pod_identity_list}
    pod_list, tile_count_int = [], 0
    for source_dict in map_source_list:
        pod_id_str = source_dict["pod_id_str"]
        if pod_str != "all" and pod_id_str != pod_str:
            continue
        identity_dict = identity_by_pod_dict[pod_id_str]
        cash_float = source_dict["cash_float"]
        invested_float = sum(row_dict["value_float"] for row_dict in source_dict["row_list"])
        nav_float = invested_float + cash_float
        pnl_known_bool = all(row_dict["pnl_float"] is not None for row_dict in source_dict["row_list"])
        pod_pnl_float = sum(row_dict["pnl_float"] for row_dict in source_dict["row_list"]) if pnl_known_bool else None
        pod_cost_float = sum(row_dict["cost_float"] for row_dict in source_dict["row_list"]) if pnl_known_bool else None
        item_list = []
        for row_dict in source_dict["row_list"]:
            if row_dict["shares_float"] == 0 or row_dict["value_float"] == 0:
                continue
            cost_float, pnl_float = row_dict["cost_float"], row_dict["pnl_float"]
            return_float = pnl_float / cost_float if pnl_float is not None and cost_float else None
            tip_list = [("Pod", identity_dict["name_str"]), ("Shares", row_dict["shares_str"]),
                ("Value", row_dict["value_str"]),
                ("Weight in Pod", f"{row_dict['value_float'] / nav_float:.1%}" if nav_float > 0 else "—"),
                ("Weight in book", f"{row_dict['value_float'] / book_value_float:.1%}")]
            if pnl_float is not None:
                tip_list.append(("Open P&L", row_dict["pnl_str"] + (" (" + signed_percent_str(return_float) + ")" if return_float is not None else "")))
            if row_dict["shares_float"] < 0:
                tip_list.append(("Side", "Short"))
            item_dict = map_item_dict(key_str=row_dict["symbol_str"] + "|" + pod_id_str,
                symbol_str=row_dict["symbol_str"], pod_name_str=identity_dict["name_str"],
                value_float=row_dict["value_float"], pnl_float=pnl_float, return_float=return_float,
                tip_list=tip_list, status_str="short" if row_dict["shares_float"] < 0 else "open")
            item_dict["mover_detail_str"] = identity_dict["name_str"] + (" · " + signed_percent_str(return_float) if return_float is not None else "")
            item_list.append(item_dict)
        tile_count_int += len(item_list)
        pod_list.append({"pod_id_str": pod_id_str, "name_str": identity_dict["name_str"], "color_str": identity_dict["color_str"],
            "nav_float": nav_float, "nav_str": compact_money_str(nav_float), "cash_float": max(cash_float, 0.0),
            "cash_str": compact_money_str(cash_float),
            "pnl_float": pod_pnl_float, "cost_float": pod_cost_float,
            "pnl_str": compact_money_str(pod_pnl_float, signed_bool=True) if pnl_known_bool else "",
            "return_str": signed_percent_str(pod_pnl_float / pod_cost_float) if pnl_known_bool and pod_cost_float else "",
            "item_list": item_list})
    if not pod_list or tile_count_int > TILE_LIMIT_INT:
        return unavailable_dict
    if not any(pod_dict["item_list"] or pod_dict["cash_float"] > 0 for pod_dict in pod_list):
        return {**unavailable_dict, "reason_str": "No current holdings or cash in this view."}
    map_dict = {"available_bool": True, "mode_str": "now", "reason_str": "", "book_value_float": book_value_float,
        "asof_str": asof_str, "pod_list": pod_list,
        "size_legend_str": "Size = value", "exposure_legend_str": "Size = gross exposure (value × leverage; TQQQ = 3x) · cash not shown",
        "colour_legend_str": "Colour = open P&L since entry"}
    if not finite_tree_bool(map_dict):
        return {**unavailable_dict, "reason_str": "Map data could not be verified."}
    return map_dict


def bar_list(item_list):
    """Signed bars around a centre line; widths are percent of the whole track."""
    largest_float = max([abs(item_dict["value_float"]) for item_dict in item_list] + [0.0])
    for item_dict in item_list:
        width_float = 0.0 if largest_float == 0 else abs(item_dict["value_float"]) / largest_float * 50
        item_dict.update(width_percent_float=width_float, left_percent_float=50.0 if item_dict["value_float"] >= 0 else 50.0 - width_float,
            tone_str="pos" if item_dict["value_float"] > 0 else "neg" if item_dict["value_float"] < 0 else "")
    return item_list


def side_dict(*, big_value_float, big_detail_str, pod_bar_list, item_list):
    """The map's side column: total, P&L by Pod, three best and three worst."""
    ranked_list = sorted((item_dict for item_dict in item_list if item_dict["pnl_float"] is not None),
        key=lambda item_dict: (-item_dict["pnl_float"], item_dict["symbol_str"], item_dict["key_str"]))
    best_list = [item_dict for item_dict in ranked_list if item_dict["pnl_float"] > 0][:3]
    worst_list = [item_dict for item_dict in reversed(ranked_list) if item_dict["pnl_float"] < 0][:3]

    def mover_dict(item_dict):
        return {"key_str": item_dict["key_str"], "symbol_str": item_dict["symbol_str"], "pnl_str": item_dict["pnl_str"],
            "tone_str": "pos" if item_dict["pnl_float"] > 0 else "neg", "detail_str": item_dict["mover_detail_str"]}
    known_bool = big_value_float is not None and math.isfinite(big_value_float)
    return {"big_str": compact_money_str(big_value_float, signed_bool=True) if known_bool else "—",
        "big_tone_str": "pos" if known_bool and big_value_float > 0 else "neg" if known_bool and big_value_float < 0 else "",
        "big_detail_str": big_detail_str, "pod_bar_list": bar_list(pod_bar_list),
        "best_list": [mover_dict(item_dict) for item_dict in best_list],
        "worst_list": [mover_dict(item_dict) for item_dict in worst_list]}


def now_side_dict(map_dict):
    """Open P&L since entry for the Pods on the map; empty when P&L is not verified."""
    if not map_dict["available_bool"]:
        return {}
    pod_list = map_dict["pod_list"]
    item_list = [item_dict for pod_dict in pod_list for item_dict in pod_dict["item_list"]]
    pnl_known_bool = all(pod_dict["pnl_float"] is not None for pod_dict in pod_list)
    total_float = sum(pod_dict["pnl_float"] for pod_dict in pod_list) if pnl_known_bool else None
    cost_float = sum(pod_dict["cost_float"] for pod_dict in pod_list) if pnl_known_bool else None
    detail_str = "Open P&L since entry" + (" · " + signed_percent_str(total_float / cost_float) + " on cost" if cost_float else "")
    pod_bar_list = [{"name_str": pod_dict["name_str"], "color_str": pod_dict["color_str"], "value_float": pod_dict["pnl_float"],
        "value_str": pod_dict["pnl_str"]} for pod_dict in pod_list if pod_dict["pnl_float"] is not None]
    return side_dict(big_value_float=total_float, big_detail_str=detail_str if pnl_known_bool else "Open P&L not verified for every Pod",
        pod_bar_list=pod_bar_list, item_list=item_list)
