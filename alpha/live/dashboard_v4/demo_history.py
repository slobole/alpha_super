"""Synthetic daily MTM sections for the local month-view preview only.

Each demo account-day gets invented, clearly labelled demo XML that the real
parser (positions_history.parse_day_mtm_dict) reads. Per-symbol MTM is split
from that day's demo Change in NAV so it reconciles exactly. Not a broker
export and not production-format acceptance evidence.
"""

from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
import hashlib
import random
import xml.etree.ElementTree as ElementTree

from alpha.live.dashboard_v4.positions_history import ROW_STR, SECTION_STR, parse_day_mtm_dict
from alpha.live.scheduler_utils import get_exchange_calendar_obj


# Pool and style per demo account (see dashboard_v3/demo.py account order).
DEMO_STYLE_DICT = {
    "DEMO_1_0": ("daily", ("AMD", "CRM", "DIS", "ADBE", "CSCO", "LOW", "MRK", "NKE", "QCOM", "TGT", "UNH", "HD", "ABT", "GIS", "PEP", "KO"), 4),
    "DEMO_1_1": ("daily", ("AAPL", "MSFT", "NVDA", "ORCL", "INTC", "IBM", "TXN", "AMAT", "MU", "AVGO", "QCOM", "ADI"), 3),
    "DEMO_1_2": ("monthly", ("AMZN", "AVGO", "NVDA", "META", "NFLX", "COST", "PLTR", "AMD", "LRCX", "AXON", "TSLA", "CSCO", "MU", "ISRG"), 8),
    "DEMO_1_3": ("monthly", ("TQQQ", "GLD", "DBC", "UUP", "IEF", "BTAL", "TLT"), 4),
}
CENT = Decimal("0.01")
CONID_DICT = {symbol_str: 1000 + index_int for index_int, symbol_str in enumerate(sorted({symbol_str
    for _, pool_tuple, _ in DEMO_STYLE_DICT.values() for symbol_str in pool_tuple}))}


def _seed_int(text_str):
    return int(hashlib.sha256(text_str.encode()).hexdigest()[:12], 16)


def _schedule_dict(account_str, session_list):
    """{date: {symbol: weight}} target holdings at each session's close."""
    style_str, pool_tuple, slot_int = DEMO_STYLE_DICT[account_str]
    random_obj = random.Random(_seed_int(account_str))
    schedule_dict = {}
    if style_str == "monthly":
        held_list = list(pool_tuple[:slot_int])
        month_str = ""
        for session_str in session_list:
            if session_str[:7] != month_str and month_str:
                for _ in range(1 if slot_int <= 4 else 3):
                    index_int = random_obj.randrange(1 if slot_int <= 4 else 0, len(held_list))
                    held_list[index_int] = random_obj.choice([symbol_str for symbol_str in pool_tuple if symbol_str not in held_list])
            month_str = session_str[:7]
            weight_dict = ({held_list[0]: .36, **{symbol_str: .54 / (len(held_list) - 1) for symbol_str in held_list[1:]}}
                if slot_int <= 4 else {symbol_str: .95 / len(held_list) for symbol_str in held_list})
            schedule_dict[session_str] = weight_dict
        return schedule_dict
    slot_list = [None] * slot_int
    left_list = [0] * slot_int
    for session_str in session_list:
        for slot_index_int in range(slot_int):
            if left_list[slot_index_int] <= 0:
                if slot_list[slot_index_int] is not None and random_obj.random() < .35:
                    slot_list[slot_index_int], left_list[slot_index_int] = None, random_obj.randint(1, 2)
                else:
                    taken_set = {symbol_str for symbol_str in slot_list if symbol_str}
                    slot_list[slot_index_int] = random_obj.choice([symbol_str for symbol_str in pool_tuple if symbol_str not in taken_set])
                    left_list[slot_index_int] = random_obj.randint(2, 6)
            left_list[slot_index_int] -= 1
        schedule_dict[session_str] = {symbol_str: .9 / slot_int for symbol_str in slot_list if symbol_str}
    return schedule_dict


def attach_demo_history(provider_obj, snapshot_obj):
    """Provider hook used only when the client registry is a demo."""
    row_by_account_dict = {}
    for row_obj in snapshot_obj.row_tuple:
        row_by_account_dict.setdefault(row_obj.account_route_str, {})[row_obj.market_date_str] = row_obj
    calendar_obj = get_exchange_calendar_obj("XNYS")
    day_cache_dict = {}

    def _account_days_dict(account_str):
        if account_str in day_cache_dict:
            return day_cache_dict[account_str]
        row_dict = row_by_account_dict.get(account_str, {})
        if account_str not in DEMO_STYLE_DICT or not row_dict:
            day_cache_dict[account_str] = {}
            return {}
        date_list = sorted(row_dict)
        session_set = {session_obj.date().isoformat() for session_obj in calendar_obj.sessions_in_range(date_list[0], date_list[-1])}
        session_list = [date_str for date_str in date_list if date_str in session_set]
        schedule_dict = _schedule_dict(account_str, session_list)
        random_obj = random.Random(_seed_int(account_str + ":prices"))
        price_dict, quantity_dict, day_dict = {}, {}, {}
        for date_str in date_list:
            nav_row_obj = row_dict[date_str]
            if date_str not in session_set:
                day_dict[date_str] = []
                continue
            target_dict = schedule_dict[date_str]
            nav_decimal = nav_row_obj.opening_nav_decimal
            # Rows: everything held at the previous close or bought today.
            symbol_list = sorted(set(quantity_dict) | set(target_dict))
            for symbol_str in symbol_list:
                price_dict.setdefault(symbol_str, Decimal(random_obj.randint(2000, 40000)) / 100)
            # A continuing holding keeps its shares; only new names are sized.
            close_quantity_dict = {symbol_str: quantity_dict.get(symbol_str) or max(1, int(Decimal(str(weight_float)) * nav_decimal / price_dict[symbol_str]))
                for symbol_str, weight_float in target_dict.items()}
            value_dict = {symbol_str: max(quantity_dict.get(symbol_str, 0), close_quantity_dict.get(symbol_str, 0)) * price_dict[symbol_str]
                for symbol_str in symbol_list}
            total_value_decimal = sum(value_dict.values(), Decimal(0))
            mtm_decimal = Decimal(nav_row_obj.attribute_dict["mtm"])
            gauss_dict = {symbol_str: Decimal(str(round(random_obj.gauss(0, 1), 4))) for symbol_str in symbol_list}
            weighted_mean_decimal = (sum((value_dict[symbol_str] * gauss_dict[symbol_str] for symbol_str in symbol_list), Decimal(0))
                / total_value_decimal) if total_value_decimal else Decimal(0)
            share_dict = {}
            for symbol_str in symbol_list:
                base_decimal = mtm_decimal * value_dict[symbol_str] / total_value_decimal if total_value_decimal else Decimal(0)
                noise_decimal = Decimal(".012") * value_dict[symbol_str] * (gauss_dict[symbol_str] - weighted_mean_decimal)
                share_dict[symbol_str] = (base_decimal + noise_decimal).quantize(CENT, rounding=ROUND_HALF_UP)
            if symbol_list:
                share_dict[symbol_list[-1]] += mtm_decimal - sum(share_dict.values(), Decimal(0))
            row_list = []
            for symbol_str in symbol_list:
                previous_quantity_int = quantity_dict.get(symbol_str, 0)
                close_quantity_int = close_quantity_dict.get(symbol_str, 0)
                previous_price_decimal = price_dict[symbol_str]
                pnl_decimal = share_dict[symbol_str]
                # Price moves so the MTM split is exact; new buys earn transaction MTM.
                basis_int = previous_quantity_int or close_quantity_int
                close_price_decimal = max(Decimal("0.01"), (previous_price_decimal + pnl_decimal / basis_int).quantize(Decimal("0.0001")))
                prior_decimal = pnl_decimal if previous_quantity_int else Decimal(0)
                transaction_decimal = Decimal(0) if previous_quantity_int else pnl_decimal
                row_list.append({"symbol": symbol_str, "conid": str(CONID_DICT[symbol_str]),
                    "prevCloseQuantity": str(previous_quantity_int), "prevClosePrice": str(previous_price_decimal),
                    "closeQuantity": str(close_quantity_int), "closePrice": str(close_price_decimal),
                    "transactionMtm": str(transaction_decimal), "priorOpenMtm": str(prior_decimal)})
                price_dict[symbol_str] = close_price_decimal
            quantity_dict = {symbol_str: quantity_int for symbol_str, quantity_int in close_quantity_dict.items() if quantity_int}
            day_dict[date_str] = row_list
        day_cache_dict[account_str] = day_dict
        return day_dict

    def get_day_mtm_dict(nav_row_obj, *, query_name_str, as_of_ts):
        row_list = _account_days_dict(nav_row_obj.account_route_str).get(nav_row_obj.market_date_str)
        if row_list is None:
            return {"available_bool": False, "reason_str": "Demo MTM unavailable", "missing_section_bool": False, "row_list": []}
        compact_str = nav_row_obj.market_date_str.replace("-", "")
        root_obj = ElementTree.Element("FlexQueryResponse", queryName=query_name_str)
        statements_obj = ElementTree.SubElement(root_obj, "FlexStatements", count="1")
        statement_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId=nav_row_obj.account_route_str)
        ElementTree.SubElement(statement_obj, "AccountInformation", accountId=nav_row_obj.account_route_str, currency="USD")
        ElementTree.SubElement(statement_obj, "ChangeInNAV", **nav_row_obj.attribute_dict)
        section_obj = ElementTree.SubElement(statement_obj, SECTION_STR)
        for attribute_dict in row_list:
            ElementTree.SubElement(section_obj, ROW_STR, accountId=nav_row_obj.account_route_str, currency="USD",
                assetCategory="STK", reportDate=compact_str, **attribute_dict)
        raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
        synthetic_nav_obj = replace(nav_row_obj, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())
        return parse_day_mtm_dict(raw_xml_str, synthetic_nav_obj, query_name_str=query_name_str, as_of_ts=as_of_ts)

    provider_obj.get_day_mtm_dict = get_day_mtm_dict
