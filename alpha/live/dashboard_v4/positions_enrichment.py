"""Saved money and today's execution evidence for the read-only Positions book."""

import math
from zoneinfo import ZoneInfo

from alpha.live.dashboard_v4.finance import _money_str, _share_str
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict
from alpha.live.dashboard_v4.positions_activity import load_position_activity_dict
from alpha.live.dashboard_v4.positions_data import observed_timestamp_ts


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _quantity_str(quantity_float):
    if not math.isfinite(quantity_float):
        return "—"
    return f"{quantity_float:,.6f}".rstrip("0").rstrip(".")


def _pnl_dict(pnl_float, cost_float):
    return {"pl_str": ("+" if pnl_float > 0 else "") + _money_str(pnl_float),
        "pl_percent_str": f"{pnl_float / cost_float:+.1%}" if cost_float > 0 else "—",
        "pl_tone_str": "pos" if pnl_float > 0 else "neg" if pnl_float < 0 else ""}


def _money_dict(holder_list, denominator_float):
    result_dict = {"value_str": "—", "weight_str": "—", "weight_percent_float": None,
        "pl_str": "—", "pl_percent_str": "—", "pl_tone_str": "", "pl_detail_str": ""}
    if not holder_list or any(holder_dict.get("value_float") is None for holder_dict in holder_list):
        return result_dict
    if len({holder_dict["value_date_str"] for holder_dict in holder_list}) != 1:
        return result_dict
    value_float = sum(holder_dict["value_float"] for holder_dict in holder_list)
    if not math.isfinite(value_float):
        return result_dict
    result_dict.update(value_float=value_float, value_str=_money_str(value_float))
    if denominator_float is not None and denominator_float > 0:
        weight_float = value_float / denominator_float
        result_dict.update(weight_str=_share_str(weight_float), weight_percent_float=weight_float * 100)
    if all(holder_dict.get("unrealized_pnl_float") is not None for holder_dict in holder_list):
        pnl_float = sum(holder_dict["unrealized_pnl_float"] for holder_dict in holder_list)
        # Cost is gross for offsetting long/short legs; no net-cost division.
        cost_float = sum(abs(holder_dict["share_float"] * holder_dict["average_cost_float"]) for holder_dict in holder_list)
        if math.isfinite(pnl_float) and math.isfinite(cost_float):
            result_dict.update(pnl_float=pnl_float, cost_float=cost_float, **_pnl_dict(pnl_float, cost_float))
    return result_dict


def _today_dict(holder_list, available_bool):
    result_dict = {"today_str": "", "today_detail_str": "", "today_pending_bool": False, "changed_bool": False}
    if not available_bool:
        return {**result_dict, "today_str": "Unknown"}
    activity_list = [holder_dict["activity_dict"] for holder_dict in holder_list if holder_dict.get("activity_dict")]
    if not activity_list:
        return result_dict
    changed_list = [activity_dict for activity_dict in activity_list if activity_dict["changed_bool"]]
    delta_float = sum(activity_dict["filled_delta_float"] for activity_dict in changed_list)
    if changed_list:
        result_dict.update(changed_bool=True, today_str=("+" if delta_float > 0 else "") + _quantity_str(delta_float))
        if delta_float == 0:
            result_dict["today_str"] = "Traded"
        # New/Closed describe the merged position, not just one holder's leg.
        if len(activity_list) == len(holder_list) and all(activity_dict.get("before_float") is not None
                and activity_dict.get("after_float") is not None for activity_dict in activity_list):
            if all(activity_dict["before_float"] == 0 for activity_dict in activity_list) and any(
                    activity_dict["after_float"] != 0 for activity_dict in activity_list):
                result_dict["today_str"] = "New · " + result_dict["today_str"]
            elif all(activity_dict["after_float"] == 0 for activity_dict in activity_list) and any(
                    activity_dict["before_float"] != 0 for activity_dict in activity_list):
                result_dict["today_str"] = "Closed · " + result_dict["today_str"]
    pending_list = sorted({activity_dict["status_str"] for activity_dict in activity_list
        if activity_dict["status_str"] not in {"Filled", "Cancelled"}})
    if pending_list:
        result_dict.update(today_pending_bool=True, today_detail_str=" · ".join(pending_list))
    elif not changed_list and any(activity_dict["status_str"] == "Cancelled" for activity_dict in activity_list):
        result_dict["today_str"] = "Cancelled"
    return result_dict


def enrich_positions_dict(result_dict, source_by_pod_dict, *, pod_str, as_of_ts):
    """Use matched saved quantities; never multiply new shares by an old mark.

    Weights use the entire book's marked holdings plus cash. All portfolio
    totals require every owned pod on one ET observation date. Filters only
    scope table rows; tiles and By pod keep the portfolio scope.
    """
    symbol_dict, valuation_dict, activity_dict, value_time_list = {}, {}, {}, []
    for pod_id_str, source_dict in source_by_pod_dict.items():
        broker_dict = load_broker_holdings_dict(source_dict["target_obj"], as_of_ts=as_of_ts)
        if broker_dict["available_bool"] and not math.isfinite(sum(abs(item_dict["value_float"])
                for item_dict in broker_dict["position_list"]) + abs(broker_dict["cash_float"])):
            broker_dict = {"available_bool": False}
        activity_dict[pod_id_str] = (load_position_activity_dict(source_dict["target_obj"], as_of_ts=as_of_ts)
            if result_dict["source_fresh_bool"] else {"available_bool": False, "symbol_dict": {}})
        position_dict = source_dict["position_map_dict"]
        mark_dict, date_str = {}, ""
        if broker_dict["available_bool"]:
            mark_dict = {item_dict["symbol_str"]: item_dict for item_dict in broker_dict["position_list"]}
            value_ts = observed_timestamp_ts(broker_dict["observed_timestamp_str"], as_of_ts)
            date_str = value_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
            value_time_list.append(value_ts)
            matched_bool = position_dict == {symbol_str: item_dict["shares_float"] for symbol_str, item_dict in mark_dict.items()}
            valuation_dict[pod_id_str] = {**broker_dict, "matched_bool": matched_bool, "date_str": date_str}
        today_dict = activity_dict[pod_id_str].get("symbol_dict", {}) if activity_dict[pod_id_str]["available_bool"] else {}
        for symbol_str in position_dict.keys() | today_dict.keys():
            shares_float = position_dict.get(symbol_str, 0.)
            holder_dict = {**source_dict["identity_dict"], "share_float": shares_float, "share_str": _quantity_str(shares_float),
                "position_asof_str": source_dict["position_asof_str"],
                "position_timestamp_str": source_dict["position_timestamp_str"],
                "source_str": source_dict["source_str"], "timestamp_basis_str": source_dict["timestamp_basis_str"]}
            mark_row_dict = mark_dict.get(symbol_str)
            if mark_row_dict and mark_row_dict["shares_float"] == shares_float:
                holder_dict.update(mark_row_dict, value_date_str=date_str)
            elif shares_float == 0 and broker_dict["available_bool"] and symbol_str not in mark_dict:
                holder_dict.update(value_float=0., value_date_str=date_str, average_cost_float=0., unrealized_pnl_float=0.)
            if symbol_str in today_dict:
                holder_dict["activity_dict"] = dict(today_dict[symbol_str])
                if holder_dict["activity_dict"].get("after_float") != shares_float:
                    holder_dict["activity_dict"].update(before_float=None, after_float=None, new_bool=False, closed_bool=False)
            symbol_dict.setdefault(symbol_str, []).append(holder_dict)

    all_pod_list = result_dict["pod_row_list"]
    money_complete_bool = bool(all_pod_list) and len(valuation_dict) == len(all_pod_list) and all(
        item_dict["matched_bool"] for item_dict in valuation_dict.values()) and len({
        item_dict["date_str"] for item_dict in valuation_dict.values()}) == 1
    denominator_float = (sum(sum(row_dict["value_float"] for row_dict in item_dict["position_list"])
        + item_dict["cash_float"] for item_dict in valuation_dict.values()) if money_complete_bool else None)
    if denominator_float is not None and not math.isfinite(sum(sum(abs(row_dict["value_float"])
            for row_dict in item_dict["position_list"]) + abs(item_dict["cash_float"]) for item_dict in valuation_dict.values())):
        money_complete_bool, denominator_float = False, None
    selected_pod_list = [row_dict["pod_id_str"] for row_dict in all_pod_list if pod_str == "all" or row_dict["pod_id_str"] == pod_str]
    activity_complete_bool = bool(selected_pod_list) and all(activity_dict.get(pod_id_str, {}).get("available_bool")
        for pod_id_str in selected_pod_list)
    result_dict.update(values_available_bool=bool(valuation_dict), pnl_available_bool=False,
        changed_available_bool=activity_complete_bool, money_complete_bool=money_complete_bool,
        values_asof_str="", values_note_str="")
    full_row_list, selected_row_list = [], []
    for symbol_str, holder_list in sorted(symbol_dict.items()):
        full_row_list.append({"symbol_str": symbol_str, "pod_list": holder_list, **_money_dict(holder_list, denominator_float)})
        selected_holder_list = [holder_dict for holder_dict in holder_list if pod_str == "all" or holder_dict["pod_id_str"] == pod_str]
        if not selected_holder_list:
            continue
        selected_row_list.append({"symbol_str": symbol_str, "name_str": "", "pod_list": selected_holder_list,
            "share_str": _quantity_str(sum(holder_dict["share_float"] for holder_dict in selected_holder_list)),
            "offset_bool": any(holder_dict["share_float"] > 0 for holder_dict in selected_holder_list)
                and any(holder_dict["share_float"] < 0 for holder_dict in selected_holder_list),
            **_money_dict(selected_holder_list, denominator_float), **_today_dict(selected_holder_list, activity_complete_bool)})
    selected_row_list.sort(key=lambda row_dict: (row_dict.get("value_float") is None, -row_dict.get("value_float", 0), row_dict["symbol_str"]))
    result_dict.update(all_count_int=len(selected_row_list), changed_count_int=sum(row_dict["changed_bool"] for row_dict in selected_row_list))
    if not valuation_dict:
        return selected_row_list

    first_str = min(value_time_list).astimezone(MARKET_TIMEZONE_OBJ).strftime("%Y-%m-%d %H:%M:%S")
    last_str = max(value_time_list).astimezone(MARKET_TIMEZONE_OBJ).strftime("%Y-%m-%d %H:%M:%S")
    result_dict["values_asof_str"] = "IBKR values · " + first_str + (" → " + last_str if last_str != first_str else "") + " ET"
    if not money_complete_bool:
        result_dict["values_note_str"] = ("Holdings changed; values pending." if any(not item_dict["matched_bool"]
            for item_dict in valuation_dict.values()) else "Values incomplete or from different dates.")
    result_dict.update(financial_asof_str=result_dict["values_asof_str"], financial_basis_str="Saved IBKR holdings + cash",
        financial_error_str="", financial_delayed_bool=False)
    invested_float = sum(sum(row_dict["value_float"] for row_dict in item_dict["position_list"])
        for item_dict in valuation_dict.values()) if money_complete_bool else None
    cash_float = sum(item_dict["cash_float"] for item_dict in valuation_dict.values()) if money_complete_bool else None
    result_dict["tile_list"][0].update(value_str=_money_str(invested_float), detail_str="Values incomplete", bar_percent_float=None)
    if denominator_float is not None and denominator_float > 0:
        result_dict["tile_list"][0].update(detail_str=_share_str(invested_float / denominator_float) + " · cash "
            + _share_str(cash_float / denominator_float), bar_percent_float=100 * invested_float / denominator_float)
    result_dict["total_dict"].update(invested_str=_money_str(invested_float), cash_str=_money_str(cash_float),
        weight_str="100.0%" if denominator_float is not None and denominator_float > 0 else "—")
    open_row_list = [row_dict for row_dict in full_row_list if any(holder_dict["share_float"] != 0 for holder_dict in row_dict["pod_list"])]
    result_dict["pnl_available_bool"] = any(row_dict.get("pnl_float") is not None for row_dict in open_row_list)
    pnl_complete_bool = money_complete_bool and all(row_dict.get("pnl_float") is not None for row_dict in open_row_list)
    if pnl_complete_bool:
        pnl_float = sum(row_dict["pnl_float"] for row_dict in open_row_list)
        cost_float = sum(row_dict["cost_float"] for row_dict in open_row_list)
        pnl_dict = _pnl_dict(pnl_float, cost_float)
        result_dict["pnl_available_bool"] = True
        result_dict["tile_list"][1].update(value_str=pnl_dict["pl_str"], detail_str=pnl_dict["pl_percent_str"], tone_str=pnl_dict["pl_tone_str"])
        result_dict["total_dict"].update(pnl_str=pnl_dict["pl_str"], pnl_tone_str=pnl_dict["pl_tone_str"])
        if open_row_list:
            ranked_list = sorted(open_row_list, key=lambda row_dict: (-row_dict["pnl_float"], row_dict["symbol_str"]))
            for tile_int, row_dict in ((2, ranked_list[0]), (3, ranked_list[-1])):
                result_dict["tile_list"][tile_int].update(value_str=row_dict["symbol_str"] + " " + row_dict["pl_str"],
                    detail_str=row_dict["pl_percent_str"], tone_str=row_dict["pl_tone_str"])
    elif result_dict["pnl_available_bool"]:
        for tile_dict in result_dict["tile_list"][1:]:
            tile_dict.update(detail_str="P&L incomplete")
    for pod_row_dict in all_pod_list:
        pod_row_dict.update(invested_str="—", cash_str="—", weight_str="—")
        value_dict = valuation_dict.get(pod_row_dict["pod_id_str"])
        if not value_dict or not value_dict["matched_bool"]:
            continue
        pod_invested_float = sum(row_dict["value_float"] for row_dict in value_dict["position_list"])
        pod_row_dict.update(invested_str=_money_str(pod_invested_float), cash_str=_money_str(value_dict["cash_float"]))
        if denominator_float is not None and denominator_float > 0:
            pod_row_dict["weight_str"] = _share_str((pod_invested_float + value_dict["cash_float"]) / denominator_float)
        if all(row_dict.get("unrealized_pnl_float") is not None for row_dict in value_dict["position_list"]):
            pnl_float = sum(row_dict["unrealized_pnl_float"] for row_dict in value_dict["position_list"])
            pod_row_dict.update(pnl_str=_pnl_dict(pnl_float, 0)["pl_str"], pnl_tone_str=_pnl_dict(pnl_float, 0)["pl_tone_str"])
    return selected_row_list
