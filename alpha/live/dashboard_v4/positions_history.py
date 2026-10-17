"""Month view of LIVE positions from saved daily IBKR Flex reports.

Read-only and display-only. Month totals (P&L, TWR) are the Performance page's
own client report for the same dates and scope. Per-symbol rows come from
IBKR's "Mark-to-Market Performance Summary in Base" section: for each
instrument and day, the previous and current close quantity and price and the
day's MTM P&L. Each saved import is parsed once and cached by import id and
checksum; the newest import of the query that covers a day is that day's
source, exactly as for the NAV rows.

Every day must prove itself before a month is shown:
- each row's prior-open MTM equals its own quantity times the price change;
- quantities carry over from one session to the next;
- the rows add up to that day's Change in NAV mark-to-market (and commissions);
- gross position value stays within four times that day's NAV.
Any failure withholds the whole month with its reason. XML names used
(MTMPerformanceSummaryInBase > MTMPerformanceSummaryUnderlying with accountId,
assetCategory, currency, symbol, conid, prevCloseQuantity, prevClosePrice,
closeQuantity, closePrice, transactionMtm, priorOpenMtm, optional commissions
and reportDate) still need acceptance against the first real export.
"""

from calendar import monthrange
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import re
import sqlite3
from threading import Lock
import xml.etree.ElementTree as ElementTree
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v4.finance import _money_str
from alpha.live.dashboard_v4.pod_holdings import _scope_valid_bool
from alpha.live.dashboard_v4.positions_map import (
    compact_money_str, finite_tree_bool, map_item_dict, side_dict, signed_percent_str,
)
from alpha.live.scheduler_utils import get_exchange_calendar_obj


SECTION_STR = "MTMPerformanceSummaryInBase"
ROW_STR = "MTMPerformanceSummaryUnderlying"
XML_BYTE_LIMIT_INT = 32_000_000
ROW_LIMIT_INT = 50_000
IMPORT_LIMIT_INT = 5_000
IMPORT_CACHE_LIMIT_INT = 8
MONTH_OPTION_LIMIT_INT = 12
DAY_LIMIT_INT = 64 * 31
# Rows must add up to the day's Change in NAV within five cents.
RECONCILE_TOLERANCE_DECIMAL = Decimal("0.05")
# Prior-open MTM vs quantity x price change: cents plus price rounding per share.
ROW_TOLERANCE_DECIMAL = Decimal("0.02")
ROW_TOLERANCE_PER_SHARE_DECIMAL = Decimal("0.0001")
GROSS_NAV_LIMIT_DECIMAL = Decimal(4)
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
UNVERIFIED_STR = "Per-symbol P&L could not be verified."
MISSING_SECTION_STR = "The saved IBKR report has no per-symbol P&L section."
MONTH_RE = re.compile(r"(20[0-9]{2})-(0[1-9]|1[0-2])")
_cache_dict = OrderedDict()
_cache_lock = Lock()


def _unavailable_dict(reason_str=UNVERIFIED_STR, *, missing_section_bool=False):
    return {"available_bool": False, "reason_str": reason_str, "missing_section_bool": missing_section_bool, "row_list": []}


def _decimal_obj(attribute_dict, key_str, default_obj=None):
    if key_str not in attribute_dict:
        if default_obj is None:
            raise KeyError(key_str)
        return default_obj
    value_obj = Decimal(attribute_dict[key_str])
    if not value_obj.is_finite() or not math.isfinite(float(value_obj)):
        raise ValueError("Nonfinite MTM field")
    return value_obj


def _iso_date_str(value_str):
    """Flex dates are YYYYMMDD (or ISO); anything else is not a date."""
    if not isinstance(value_str, str) or not re.fullmatch(r"[0-9]{8}|[0-9]{4}-[0-9]{2}-[0-9]{2}", value_str):
        raise ValueError("Invalid Flex date")
    compact_str = value_str.replace("-", "")
    return date(int(compact_str[:4]), int(compact_str[4:6]), int(compact_str[6:])).isoformat()


def _row_dict(attribute_dict, account_str):
    """One validated instrument row, or ValueError with the reason."""
    contract_str, symbol_str = attribute_dict.get("conid", ""), attribute_dict.get("symbol", "")
    if (attribute_dict.get("accountId") != account_str or attribute_dict.get("assetCategory") != "STK"
            or attribute_dict.get("model") or attribute_dict.get("currency") != "USD"
            or ("multiplier" in attribute_dict and _decimal_obj(attribute_dict, "multiplier") != 1)
            or ("fxRateToBase" in attribute_dict and _decimal_obj(attribute_dict, "fxRateToBase") != 1)
            or not contract_str.isascii() or not contract_str.isdigit() or int(contract_str) <= 0
            or not symbol_str or symbol_str != symbol_str.strip() or len(symbol_str) > 80
            or any(ord(character_str) < 32 for character_str in symbol_str)):
        raise ValueError(UNVERIFIED_STR)
    previous_quantity_decimal = _decimal_obj(attribute_dict, "prevCloseQuantity")
    previous_price_decimal = _decimal_obj(attribute_dict, "prevClosePrice")
    close_quantity_decimal = _decimal_obj(attribute_dict, "closeQuantity")
    close_price_decimal = _decimal_obj(attribute_dict, "closePrice")
    transaction_decimal = _decimal_obj(attribute_dict, "transactionMtm")
    prior_decimal = _decimal_obj(attribute_dict, "priorOpenMtm")
    commission_decimal = _decimal_obj(attribute_dict, "commissions", Decimal(0))
    if previous_price_decimal < 0 or close_price_decimal < 0:
        raise ValueError(UNVERIFIED_STR)
    # *** CRITICAL *** quantity/price drive value, size and return: the row's
    # own MTM must equal its overnight quantity times the price change.
    if (abs(previous_quantity_decimal * (close_price_decimal - previous_price_decimal) - prior_decimal)
            > ROW_TOLERANCE_DECIMAL + abs(previous_quantity_decimal) * ROW_TOLERANCE_PER_SHARE_DECIMAL):
        raise ValueError(f"The IBKR row for {symbol_str} does not match its own quantity and prices.")
    return {"symbol_str": symbol_str, "conid_int": int(contract_str), "commission_seen_bool": "commissions" in attribute_dict,
        "previous_quantity_decimal": previous_quantity_decimal, "close_quantity_decimal": close_quantity_decimal,
        "close_price_decimal": close_price_decimal, "transaction_decimal": transaction_decimal,
        "prior_decimal": prior_decimal, "commission_decimal": commission_decimal,
        "previous_quantity_float": float(previous_quantity_decimal), "previous_price_float": float(previous_price_decimal),
        "close_quantity_float": float(close_quantity_decimal), "close_price_float": float(close_price_decimal),
        "transaction_mtm_float": float(transaction_decimal), "prior_open_mtm_float": float(prior_decimal),
        "commission_float": float(commission_decimal)}


def parse_import_dict(raw_xml_str, *, query_name_str):
    """Parse one saved import once: {(account, date): block}.

    A block holds that day's Change in NAV attributes (None when the statement
    has none), its rows (None when the section is absent) and an error reason.
    Rows without a conid are IBKR subtotals and are skipped; the NAV
    reconciliation proves nothing real was skipped.
    """
    root_obj = ElementTree.fromstring(raw_xml_str)
    if root_obj.tag != "FlexQueryResponse" or root_obj.get("queryName") != query_name_str:
        raise ValueError("Saved Flex response/query identity mismatch.")
    block_dict = {}
    for statement_obj in root_obj.findall(".//FlexStatement"):
        account_str = statement_obj.get("accountId") or ""
        if not account_str:
            continue
        info_obj = statement_obj.find("AccountInformation")
        error_str = "" if (info_obj is not None and info_obj.get("accountId") == account_str
            and info_obj.get("currency") == "USD") else "Broker account identity/base currency is not confirmed."
        nav_by_date_dict = {}
        for nav_obj in statement_obj.findall("ChangeInNAV"):
            try:
                from_str, to_str = _iso_date_str(nav_obj.get("fromDate")), _iso_date_str(nav_obj.get("toDate"))
            except ValueError:
                from_str, to_str = "", "?"
            if from_str != to_str or from_str in nav_by_date_dict:
                error_str = error_str or UNVERIFIED_STR
                continue
            nav_by_date_dict[from_str] = dict(nav_obj.attrib)
        day_set = set(nav_by_date_dict)
        if not day_set:
            # A statement without Change in NAV (an MTM-only query) names its own day.
            try:
                from_str, to_str = _iso_date_str(statement_obj.get("fromDate")), _iso_date_str(statement_obj.get("toDate"))
                if from_str == to_str:
                    day_set = {from_str}
            except ValueError:
                pass
        section_list = statement_obj.findall(SECTION_STR)
        local_dict = {day_str: {"nav_attribute_dict": nav_by_date_dict.get(day_str), "row_list": [] if section_list else None,
            "error_str": error_str} for day_str in day_set}
        if len(section_list) > 1 or (section_list and len(section_list[0]) > ROW_LIMIT_INT):
            for day_dict in local_dict.values():
                day_dict["error_str"] = day_dict["error_str"] or UNVERIFIED_STR
            section_list = []
        multi_day_bool = len(day_set) != 1
        seen_dict = {}
        for row_obj in (section_list[0] if section_list else []):
            attribute_dict = row_obj.attrib
            if row_obj.tag != ROW_STR:
                for day_dict in local_dict.values():
                    day_dict["error_str"] = day_dict["error_str"] or UNVERIFIED_STR
                break
            if not attribute_dict.get("conid", ""):
                continue
            report_date_str = attribute_dict.get("reportDate")
            try:
                if multi_day_bool:
                    # Several days in one statement: every row must name its day.
                    day_str = _iso_date_str(report_date_str)
                    if day_set and day_str not in day_set:
                        raise ValueError(UNVERIFIED_STR)
                    local_dict.setdefault(day_str, {"nav_attribute_dict": None, "row_list": [], "error_str": error_str})
                else:
                    day_str = next(iter(day_set))
                    if report_date_str is not None and _iso_date_str(report_date_str) != day_str:
                        raise ValueError(UNVERIFIED_STR)
            except ValueError:
                for day_dict in local_dict.values():
                    day_dict["error_str"] = day_dict["error_str"] or UNVERIFIED_STR
                break
            day_dict = local_dict[day_str]
            try:
                row_dict = _row_dict(attribute_dict, account_str)
            except (ValueError, KeyError, InvalidOperation, OverflowError) as error_obj:
                day_dict["error_str"] = day_dict["error_str"] or (str(error_obj) if isinstance(error_obj, ValueError)
                    and str(error_obj).startswith("The IBKR row") else UNVERIFIED_STR)
                continue
            seen_set = seen_dict.setdefault(day_str, set())
            if row_dict["symbol_str"] in seen_set or row_dict["conid_int"] in seen_set:
                day_dict["error_str"] = day_dict["error_str"] or UNVERIFIED_STR
                continue
            seen_set.update({row_dict["symbol_str"], row_dict["conid_int"]})
            day_dict["row_list"].append(row_dict)
        for day_str, day_dict in local_dict.items():
            key_tuple = (account_str, day_str)
            if key_tuple in block_dict:
                # Two statements for one account-day: ambiguous, never merged.
                block_dict[key_tuple]["error_str"] = "Two saved statements cover this day."
                continue
            block_dict[key_tuple] = day_dict
    return block_dict


def day_mtm_dict(block_dict, nav_row_obj):
    """One account-day: the import's block checked against the day's NAV row."""
    if block_dict is None:
        return _unavailable_dict("No saved IBKR report covers this day.")
    if block_dict["error_str"]:
        return _unavailable_dict(block_dict["error_str"])
    if block_dict["nav_attribute_dict"] is not None and block_dict["nav_attribute_dict"] != nav_row_obj.attribute_dict:
        return _unavailable_dict()
    if block_dict["row_list"] is None:
        return _unavailable_dict(MISSING_SECTION_STR, missing_section_bool=True)
    row_list = block_dict["row_list"]
    try:
        mtm_decimal = _decimal_obj(nav_row_obj.attribute_dict, "mtm")
        if abs(sum((row_dict["prior_decimal"] + row_dict["transaction_decimal"] for row_dict in row_list), Decimal(0))
                - mtm_decimal) > RECONCILE_TOLERANCE_DECIMAL:
            return _unavailable_dict("Per-symbol P&L does not add up to the day's IBKR mark-to-market.")
        if (any(row_dict["commission_seen_bool"] for row_dict in row_list) and "commissions" in nav_row_obj.attribute_dict
                and abs(sum((row_dict["commission_decimal"] for row_dict in row_list), Decimal(0))
                    - _decimal_obj(nav_row_obj.attribute_dict, "commissions")) > RECONCILE_TOLERANCE_DECIMAL):
            return _unavailable_dict("Per-symbol commissions do not add up to the day's IBKR commissions.")
        gross_decimal = sum((abs(row_dict["close_quantity_decimal"] * row_dict["close_price_decimal"]) for row_dict in row_list), Decimal(0))
        if gross_decimal > GROSS_NAV_LIMIT_DECIMAL * abs(nav_row_obj.closing_nav_decimal) + 1:
            return _unavailable_dict("Position values exceed four times the day's NAV.")
    except (ValueError, KeyError, InvalidOperation, OverflowError):
        return _unavailable_dict()
    return {"available_bool": True, "reason_str": "", "missing_section_bool": False,
        "market_date_str": nav_row_obj.market_date_str,
        "row_list": [{key_str: value_obj for key_str, value_obj in row_dict.items() if not key_str.endswith("_decimal")}
            for row_dict in row_list]}


def _raw_xml_ok_bool(raw_xml_str, checksum_str):
    if not isinstance(raw_xml_str, str):
        return False
    raw_bytes = raw_xml_str.encode("utf-8")
    return (len(raw_bytes) <= XML_BYTE_LIMIT_INT and "<!DOCTYPE" not in raw_xml_str.upper() and "<!ENTITY" not in raw_xml_str.upper()
        and hashlib.sha256(raw_bytes).hexdigest() == checksum_str)


def parse_day_mtm_dict(raw_xml_str, nav_row_obj, *, query_name_str, as_of_ts):
    """One account-day from one raw import (demo and tests); no cache."""
    if not _scope_valid_bool(nav_row_obj, query_name_str, as_of_ts) or not _raw_xml_ok_bool(raw_xml_str, nav_row_obj.source_checksum_str):
        return _unavailable_dict()
    try:
        block_dict = parse_import_dict(raw_xml_str, query_name_str=query_name_str)
    except (ElementTree.ParseError, ValueError, TypeError, KeyError, InvalidOperation, OverflowError):
        return _unavailable_dict()
    return day_mtm_dict(block_dict.get((nav_row_obj.account_route_str, nav_row_obj.market_date_str)), nav_row_obj)


def load_day_mtm_map_dict(database_path_str, nav_row_list, *, query_name_str, as_of_ts, mtm_query_name_str=None):
    """Each NAV row's day from the newest saved import of the MTM query covering it.

    With one query (MTM section inside the NAV query) that import must be the
    NAV row's own import. Imports are parsed outside the read transaction, once,
    and cached by (query, import id, checksum).
    """
    mtm_query_str = mtm_query_name_str or query_name_str
    result_dict = {(row_obj.account_route_str, row_obj.market_date_str): _unavailable_dict() for row_obj in nav_row_list}
    if not nav_row_list or len(nav_row_list) > DAY_LIMIT_INT:
        return result_dict
    if not database_path_str or not Path(database_path_str).is_file():
        return {key_tuple: _unavailable_dict("No saved IBKR performance database.") for key_tuple in result_dict}
    row_list = [row_obj for row_obj in nav_row_list if _scope_valid_bool(row_obj, query_name_str, as_of_ts)]
    chosen_dict, raw_dict = {}, {}
    try:
        database_path_obj = Path(database_path_str).resolve()
        with closing(sqlite3.connect(database_path_obj.as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("PRAGMA query_only=ON")
            progress_list = [0]

            def stop_query_bool():
                progress_list[0] += 1
                return progress_list[0] > 4000

            connection_obj.set_progress_handler(stop_query_bool, 1_000)
            connection_obj.execute("BEGIN")
            import_list = connection_obj.execute(
                "SELECT import_id_int, imported_timestamp_str, request_from_date_str, request_to_date_str, checksum_str "
                "FROM flex_import WHERE query_name_str=? ORDER BY import_id_int LIMIT ?",
                (mtm_query_str, IMPORT_LIMIT_INT + 1)).fetchall()
            if len(import_list) > IMPORT_LIMIT_INT:
                return result_dict
            range_list = []
            for import_obj in import_list:
                imported_ts = datetime.fromisoformat(import_obj["imported_timestamp_str"])
                from_obj = date.fromisoformat(import_obj["request_from_date_str"])
                to_obj = date.fromisoformat(import_obj["request_to_date_str"])
                if imported_ts.tzinfo is None or imported_ts > as_of_ts or from_obj > to_obj:
                    # One unverifiable import makes every day's newest source unprovable.
                    return result_dict
                range_list.append((import_obj["import_id_int"], from_obj, to_obj, import_obj["checksum_str"]))
            for row_obj in row_list:
                market_date_obj = date.fromisoformat(row_obj.market_date_str)
                covering_list = [range_tuple for range_tuple in range_list if range_tuple[1] <= market_date_obj <= range_tuple[2]]
                if not covering_list:
                    result_dict[(row_obj.account_route_str, row_obj.market_date_str)] = _unavailable_dict("No saved IBKR report covers this day.")
                    continue
                import_id_int, _, _, checksum_str = max(covering_list)
                if mtm_query_str == query_name_str and (import_id_int != row_obj.source_import_id_int or checksum_str != row_obj.source_checksum_str):
                    continue
                chosen_dict[(row_obj.account_route_str, row_obj.market_date_str)] = (row_obj, import_id_int, checksum_str)
            with _cache_lock:
                needed_set = {(import_id_int, checksum_str) for _, import_id_int, checksum_str in chosen_dict.values()
                    if (mtm_query_str, import_id_int, checksum_str) not in _cache_dict}
            for import_id_int, checksum_str in sorted(needed_set):
                fetched_list = connection_obj.execute(
                    "SELECT CASE WHEN length(CAST(raw_xml_str AS BLOB))<=? THEN raw_xml_str END AS raw_xml_str "
                    "FROM flex_import WHERE import_id_int=? AND query_name_str=? AND checksum_str=? LIMIT 2",
                    (XML_BYTE_LIMIT_INT, import_id_int, mtm_query_str, checksum_str)).fetchall()
                if len(fetched_list) == 1:
                    raw_dict[(import_id_int, checksum_str)] = fetched_list[0]["raw_xml_str"]
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, OverflowError):
        return result_dict
    # Parse outside the read transaction: the importer is never blocked by it.
    for (import_id_int, checksum_str), raw_xml_str in raw_dict.items():
        try:
            if not _raw_xml_ok_bool(raw_xml_str, checksum_str):
                raise ValueError("Saved import failed its checksum.")
            parsed_dict = parse_import_dict(raw_xml_str, query_name_str=mtm_query_str)
        except (ElementTree.ParseError, ValueError, TypeError, KeyError, InvalidOperation, OverflowError):
            parsed_dict = None
        with _cache_lock:
            _cache_dict[(mtm_query_str, import_id_int, checksum_str)] = parsed_dict
            _cache_dict.move_to_end((mtm_query_str, import_id_int, checksum_str))
            while len(_cache_dict) > IMPORT_CACHE_LIMIT_INT:
                _cache_dict.popitem(last=False)
    for key_tuple, (row_obj, import_id_int, checksum_str) in chosen_dict.items():
        with _cache_lock:
            parsed_dict = _cache_dict.get((mtm_query_str, import_id_int, checksum_str), None)
        if parsed_dict is None:
            continue
        result_dict[key_tuple] = day_mtm_dict(parsed_dict.get(key_tuple), row_obj)
    return result_dict


def _month_bounds_tuple(month_str):
    match_obj = MONTH_RE.fullmatch(month_str or "")
    if match_obj is None:
        raise ValueError("Invalid month")
    year_int, month_int = int(match_obj.group(1)), int(match_obj.group(2))
    return f"{month_str}-01", f"{month_str}-{monthrange(year_int, month_int)[1]:02d}"


def _last_session_str(first_str, last_str):
    session_list = get_exchange_calendar_obj("XNYS").sessions_in_range(first_str, last_str)
    return session_list[-1].date().isoformat() if len(session_list) else last_str


def report_scope_tuple(workspace_dict, snapshot_obj, *, as_of_ts):
    """The Performance page's report scope (performance.build_performance_page_dict).

    A complete saved scope uses the client's dated accounts, so Pods that start
    or retire mid-month keep every month reportable.
    """
    today_str = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
    client_dict = deepcopy(workspace_dict["client_dict"])
    scope_complete_bool = workspace_dict.get("financial_scope_complete_bool", False)
    valuation_list = None
    if not scope_complete_bool or client_dict["mandate_start_date"] > today_str:
        valuation_list = workspace_dict.get("valuation_account_list", [])
        client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], today_str)
        saved_date_list = [row_obj.market_date_str for row_obj in snapshot_obj.row_tuple if row_obj.market_date_str < today_str]
        if saved_date_list:
            client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], min(saved_date_list))
    return client_dict, scope_complete_bool, valuation_list


def month_option_list(workspace_dict, snapshot_obj, owned_list, *, as_of_ts):
    """Months with finalized NAV for the owned accounts, newest first."""
    try:
        client_dict, _, valuation_list = report_scope_tuple(workspace_dict, snapshot_obj, as_of_ts=as_of_ts)
        latest_str = financial_dates_dict(client_dict, snapshot_obj, as_of_ts=as_of_ts,
            valuation_account_list=valuation_list)["latest_complete_str"]
    except (ClientReportingError, ValueError, TypeError, KeyError, AttributeError):
        return []
    if not latest_str:
        return []
    route_set = {account_dict["account_route"] for account_dict in owned_list}
    month_set = {row_obj.market_date_str[:7] for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str in route_set and client_dict["mandate_start_date"] <= row_obj.market_date_str <= latest_str}
    option_list = []
    for month_str in sorted(month_set, reverse=True)[:MONTH_OPTION_LIMIT_INT]:
        first_str, last_str = _month_bounds_tuple(month_str)
        month_obj = date.fromisoformat(first_str)
        to_date_bool = latest_str < _last_session_str(first_str, last_str)
        option_list.append({"month_str": month_str, "label_str": month_obj.strftime("%b") + (" " + month_obj.strftime("%Y")
            if month_obj.year != as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).year else "") + (" (to date)" if to_date_bool else ""),
            "name_str": month_obj.strftime("%B %Y"), "to_date_bool": to_date_bool})
    return option_list


def month_name_str(month_str):
    try:
        return date.fromisoformat(_month_bounds_tuple(month_str)[0]).strftime("%B %Y")
    except ValueError:
        return "This month"


def _day_label_str(date_str):
    date_obj = date.fromisoformat(date_str)
    return date_obj.strftime("%b ") + str(date_obj.day)


def _continuity_error_str(day_map_dict, account_route_str, session_list):
    """Quantities must carry over from one session's close to the next session's open."""
    previous_dict, previous_str = None, ""
    for date_str in session_list:
        row_list = day_map_dict[(account_route_str, date_str)]["row_list"]
        if previous_dict is not None:
            opening_dict = {row_dict["conid_int"]: row_dict["previous_quantity_float"] for row_dict in row_list}
            for conid_int in set(previous_dict) | set(opening_dict):
                if abs(previous_dict.get(conid_int, 0.0) - opening_dict.get(conid_int, 0.0)) > 1e-9:
                    return (f"Quantities do not carry over from {_day_label_str(previous_str)} to {_day_label_str(date_str)} "
                        "in the saved IBKR reports (a corporate action or a missing row).")
        previous_dict = {row_dict["conid_int"]: row_dict["close_quantity_float"] for row_dict in row_list if row_dict["close_quantity_float"]}
        previous_str = date_str
    return ""


def _symbol_aggregate_dict(day_map_dict, account_route_str, date_list, session_set):
    """Fold one Pod's daily MTM rows into one record per symbol."""
    aggregate_dict, symbol_pnl_by_date_dict = {}, {}
    for date_str in date_list:
        day_total_float = 0.0
        for row_dict in day_map_dict[(account_route_str, date_str)]["row_list"]:
            pnl_float = row_dict["prior_open_mtm_float"] + row_dict["transaction_mtm_float"] + row_dict["commission_float"]
            day_total_float += pnl_float
            if not any(row_dict[field_str] for field_str in ("previous_quantity_float", "close_quantity_float",
                    "transaction_mtm_float", "commission_float")):
                continue
            item_dict = aggregate_dict.setdefault(row_dict["symbol_str"], {"symbol_str": row_dict["symbol_str"],
                "first_str": date_str, "last_str": date_str, "session_int": 0, "exposure_sum_float": 0.0,
                "pnl_float": 0.0, "close_by_date_dict": {}, "short_bool": False})
            item_dict["last_str"] = date_str
            item_dict["pnl_float"] += pnl_float
            item_dict["close_by_date_dict"][date_str] = row_dict["close_quantity_float"]
            item_dict["short_bool"] = item_dict["short_bool"] or row_dict["previous_quantity_float"] < 0 or row_dict["close_quantity_float"] < 0
            if date_str in session_set:
                item_dict["session_int"] += 1
                # Capital in the name that day: the larger of the overnight and closing position.
                item_dict["exposure_sum_float"] += max(abs(row_dict["previous_quantity_float"] * row_dict["previous_price_float"]),
                    abs(row_dict["close_quantity_float"] * row_dict["close_price_float"]))
        symbol_pnl_by_date_dict[date_str] = day_total_float
    return aggregate_dict, symbol_pnl_by_date_dict


def _failure_reason_str(day_map_dict, owned_list, session_set):
    failed_list = [(key_tuple, day_dict) for key_tuple, day_dict in sorted(day_map_dict.items(), key=lambda pair_tuple: pair_tuple[0][1])
        if not day_dict["available_bool"]]
    if not failed_list:
        return ""
    covered_list = sorted(date_str for (_, date_str), day_dict in day_map_dict.items()
        if day_dict["available_bool"] and date_str in session_set)
    first_failed_str = failed_list[0][0][1]
    if all(day_dict["missing_section_bool"] for _, day_dict in failed_list):
        if not covered_list:
            return ("The saved IBKR reports for this month have no per-symbol P&L, so it cannot be shown yet "
                "(see the Mark-to-Market section in the Flex setup guide).")
        if covered_list[0] > first_failed_str:
            return (f"Per-symbol P&L starts {_day_label_str(covered_list[0])} in the saved IBKR reports, so this month is "
                "incomplete. The next full month will show here.")
        return f"Per-symbol P&L is missing from {_day_label_str(first_failed_str)} in the saved IBKR reports, so this month is incomplete."
    reason_set = {day_dict["reason_str"] for _, day_dict in failed_list}
    if len(reason_set) == 1 and not covered_list:
        return next(iter(reason_set))
    (route_str, date_str), day_dict = failed_list[0]
    pod_name_str = next((account_dict["display_name"] for account_dict in owned_list if account_dict["account_route"] == route_str), route_str)
    return f"{_day_label_str(date_str)} · {pod_name_str}: {day_dict['reason_str']}"


def build_month_page_dict(workspace_dict, snapshot_obj, owned_list, identity_list, provider_obj, *,
                          month_str, pod_str, as_of_ts, database_path_str):
    """Everything the Positions page needs for one month, or why it cannot show it.

    Tiles and By pod keep the portfolio scope; map and table follow the Pod filter.
    """
    first_str, last_str = _month_bounds_tuple(month_str)
    month_obj = date.fromisoformat(first_str)
    result_dict = {"mode_str": "month", "month_str": month_str, "available_bool": False, "reason_str": "",
        "name_str": month_obj.strftime("%B %Y"), "short_str": month_obj.strftime("%b"), "to_date_bool": False,
        "row_list": [], "pod_row_list": [], "total_dict": {}, "tile_list": [], "map_dict": {"available_bool": False, "pod_list": []},
        "side_dict": {}, "source_str": "", "verdict_str": month_obj.strftime("%B %Y") + " unavailable", "verdict_detail_str": ""}

    def unavailable_month_dict(reason_str):
        result_dict.update(reason_str=reason_str, verdict_detail_str=reason_str)
        return result_dict
    try:
        client_dict, scope_complete_bool, valuation_list = report_scope_tuple(workspace_dict, snapshot_obj, as_of_ts=as_of_ts)
        latest_str = financial_dates_dict(client_dict, snapshot_obj, as_of_ts=as_of_ts, valuation_account_list=valuation_list)["latest_complete_str"]
        from_str = max(first_str, client_dict["mandate_start_date"])
        to_str = min(last_str, latest_str or "")
        if not latest_str or from_str > to_str:
            return unavailable_month_dict("No finalized close in this month yet.")
        report_dict = build_client_report_dict(client_dict, snapshot_obj, from_date_str=from_str, to_date_str=to_str,
            as_of_ts=as_of_ts, scope_complete_bool=scope_complete_bool, valuation_account_list=valuation_list)
        if not report_dict["flows_complete_bool"] or report_dict["pnl_float"] is None:
            return unavailable_month_dict("Month P&L could not be verified" + (": " + report_dict["issue_list"][0] if report_dict["issue_list"] else "."))
    except (ClientReportingError, ValueError, TypeError, KeyError, AttributeError):
        return unavailable_month_dict("Month P&L could not be verified.")
    result_dict.update(to_date_bool=to_str < _last_session_str(first_str, last_str), from_date_str=from_str, to_date_str=to_str)
    session_list = [session_obj.date().isoformat() for session_obj in get_exchange_calendar_obj("XNYS").sessions_in_range(from_str, to_str)]
    session_set = set(session_list)
    identity_by_pod_dict = {identity_dict["pod_id_str"]: identity_dict for identity_dict in identity_list}
    strategy_by_pod_dict = {strategy_dict["pod_id_str"]: strategy_dict for strategy_dict in report_dict["strategy_list"]}
    nav_by_key_dict = {(row_obj.account_route_str, row_obj.market_date_str): row_obj for row_obj in snapshot_obj.row_tuple}
    pod_plan_list, nav_row_list = [], []
    for account_dict in owned_list:
        strategy_dict = strategy_by_pod_dict.get(account_dict["pod_id"])
        if strategy_dict is None or strategy_dict["pnl_float"] is None:
            continue
        date_list = [day_dict["market_date_str"] for day_dict in strategy_dict["daily_list"]]
        if any((account_dict["account_route"], date_str) not in nav_by_key_dict for date_str in date_list):
            return unavailable_month_dict("Month NAV is incomplete.")
        nav_row_list.extend(nav_by_key_dict[(account_dict["account_route"], date_str)] for date_str in date_list)
        pod_plan_list.append((account_dict, strategy_dict, date_list))
    if not pod_plan_list:
        return unavailable_month_dict("No owned Pod has finalized NAV in this month.")
    client_query_str = workspace_dict["client_dict"]["query_name"]
    demo_getter_fn = getattr(provider_obj, "get_day_mtm_dict", None) if workspace_dict["client_dict"].get("is_demo") is True else None
    if callable(demo_getter_fn):
        day_map_dict = {(row_obj.account_route_str, row_obj.market_date_str): demo_getter_fn(row_obj,
            query_name_str=client_query_str, as_of_ts=as_of_ts) for row_obj in nav_row_list}
    else:
        day_map_dict = load_day_mtm_map_dict(database_path_str, nav_row_list, query_name_str=client_query_str, as_of_ts=as_of_ts)
    # Weekend/holiday NAV rows exist only for cash activity. With zero MTM, a
    # missing per-symbol section there is an empty day, not missing evidence.
    for key_tuple, day_dict in list(day_map_dict.items()):
        if day_dict["missing_section_bool"] and key_tuple[1] not in session_set:
            try:
                zero_bool = Decimal(nav_by_key_dict[key_tuple].attribute_dict.get("mtm", "NaN")) == 0
            except (InvalidOperation, KeyError):
                zero_bool = False
            if zero_bool:
                day_map_dict[key_tuple] = {"available_bool": True, "reason_str": "", "missing_section_bool": False, "row_list": []}
    reason_str = _failure_reason_str(day_map_dict, owned_list, session_set)
    if reason_str:
        return unavailable_month_dict(reason_str)
    for account_dict, _, date_list in pod_plan_list:
        continuity_str = _continuity_error_str(day_map_dict, account_dict["account_route"], [date_str for date_str in date_list if date_str in session_set])
        if continuity_str:
            return unavailable_month_dict(account_dict["display_name"] + ": " + continuity_str)

    pod_result_list = []
    for account_dict, strategy_dict, date_list in pod_plan_list:
        aggregate_dict, symbol_pnl_by_date_dict = _symbol_aggregate_dict(day_map_dict, account_dict["account_route"], date_list, session_set)
        pod_session_list = [date_str for date_str in date_list if date_str in session_set]
        session_count_int = len(pod_session_list)
        nav_list = [float(nav_by_key_dict[(account_dict["account_route"], date_str)].closing_nav_decimal) for date_str in pod_session_list]
        average_nav_float = sum(nav_list) / len(nav_list) if nav_list else None
        pod_pnl_float = strategy_dict["pnl_float"]
        symbol_pnl_float = sum(symbol_pnl_by_date_dict.values())
        identity_dict = identity_by_pod_dict[account_dict["pod_id"]]
        # Status from the Pod's last session: weekend NAV rows carry no positions.
        last_session_str = pod_session_list[-1] if pod_session_list else date_list[-1]
        item_list = []
        for item_dict in aggregate_dict.values():
            last_close_float = item_dict["close_by_date_dict"].get(last_session_str, 0.0)
            status_str = ("open" if result_dict["to_date_bool"] else "kept") if last_close_float != 0 else "sold"
            held_value_float = item_dict["exposure_sum_float"] / item_dict["session_int"] if item_dict["session_int"] else 0.0
            size_float = item_dict["exposure_sum_float"] / session_count_int if session_count_int else 0.0
            return_float = item_dict["pnl_float"] / held_value_float if held_value_float > 0 else None
            item_list.append({**item_dict, "pod_id_str": account_dict["pod_id"], "status_str": status_str,
                "held_value_float": held_value_float, "size_float": size_float, "return_float": return_float,
                "session_count_int": session_count_int})
        pod_result_list.append({"account_dict": account_dict, "identity_dict": identity_dict, "item_list": item_list,
            "pnl_float": pod_pnl_float, "twr_float": strategy_dict["twr_float"], "symbol_pnl_float": symbol_pnl_float,
            "other_float": pod_pnl_float - symbol_pnl_float, "average_nav_float": average_nav_float,
            "session_count_int": session_count_int})

    # Book average: every month session, summed NAV of the Pods active that day.
    book_nav_list = [sum(float(nav_by_key_dict[(account_dict["account_route"], date_str)].closing_nav_decimal)
        for account_dict, _, date_list in pod_plan_list if date_str in date_list) for date_str in session_list]
    book_average_float = sum(book_nav_list) / len(book_nav_list) if book_nav_list else 0.0
    month_short_str = result_dict["short_str"]

    def held_str(item_dict):
        return _day_label_str(item_dict["first_str"]) + " – " + _day_label_str(item_dict["last_str"])

    def status_text_str(item_dict):
        return {"open": "Open now", "kept": "Held to month end"}.get(item_dict["status_str"], "Sold " + _day_label_str(item_dict["last_str"]))

    all_item_list = [item_dict for pod_dict in pod_result_list for item_dict in pod_dict["item_list"]]
    sold_count_int = sum(item_dict["status_str"] == "sold" for item_dict in all_item_list)
    # Table rows and map tiles follow the Pod filter.
    selected_pod_list = [pod_dict for pod_dict in pod_result_list if pod_str == "all" or pod_dict["account_dict"]["pod_id"] == pod_str]
    map_pod_list, map_item_all_list = [], []
    for pod_dict in selected_pod_list:
        identity_dict, nav_float = pod_dict["identity_dict"], pod_dict["average_nav_float"] or 0.0
        map_item_list = []
        for item_dict in sorted(pod_dict["item_list"], key=lambda item_dict: (-item_dict["size_float"], item_dict["symbol_str"])):
            if item_dict["size_float"] <= 0:
                continue
            tip_list = [("Pod", identity_dict["name_str"]),
                ("Held", held_str(item_dict) + f" · {item_dict['session_int']} of {item_dict['session_count_int']} sessions"),
                ("Status", status_text_str(item_dict)),
                ("Avg value while held", _money_str(item_dict["held_value_float"])),
                ("Avg weight in Pod", f"{item_dict['size_float'] / nav_float:.1%}" if nav_float > 0 else "—"),
                (f"P&L in {month_short_str}", _money_str(item_dict["pnl_float"], signed_bool=True)
                    + (" (" + signed_percent_str(item_dict["return_float"]) + ")" if item_dict["return_float"] is not None else ""))]
            if item_dict["short_bool"]:
                tip_list.append(("Side", "Short"))
            map_item_dict_obj = map_item_dict(key_str=item_dict["symbol_str"] + "|" + item_dict["pod_id_str"],
                symbol_str=item_dict["symbol_str"], pod_name_str=identity_dict["name_str"],
                value_float=item_dict["size_float"], pnl_float=item_dict["pnl_float"], return_float=item_dict["return_float"],
                tip_list=tip_list, status_str=item_dict["status_str"],
                status_note_str="sold " + date.fromisoformat(item_dict["last_str"]).strftime("%m-%d") if item_dict["status_str"] == "sold" else "")
            map_item_dict_obj["mover_detail_str"] = (identity_dict["name_str"] + " · " + signed_percent_str(item_dict["return_float"])
                + " · " + held_str(item_dict) + (" (sold)" if item_dict["status_str"] == "sold" else ""))
            map_item_list.append(map_item_dict_obj)
        map_item_all_list.extend(map_item_list)
        invested_float = sum(item_dict["size_float"] for item_dict in pod_dict["item_list"])
        map_pod_list.append({"pod_id_str": identity_dict["pod_id_str"], "name_str": identity_dict["name_str"],
            "color_str": identity_dict["color_str"], "nav_float": nav_float, "nav_str": compact_money_str(nav_float) + " avg",
            "cash_float": max(nav_float - invested_float, 0.0), "cash_str": compact_money_str(max(nav_float - invested_float, 0.0)),
            "pnl_float": pod_dict["pnl_float"], "pnl_str": compact_money_str(pod_dict["pnl_float"], signed_bool=True),
            "return_str": signed_percent_str(pod_dict["twr_float"]) if pod_dict["twr_float"] is not None else "",
            "item_list": map_item_list})
    period_label_str = result_dict["name_str"] + (" (to date)" if result_dict["to_date_bool"] else "")
    held_any_bool = any(pod_dict["item_list"] for pod_dict in map_pod_list)
    result_dict["map_dict"] = {"available_bool": held_any_bool, "mode_str": "month",
        "reason_str": "" if held_any_bool else "Nothing was held in this view during the month.",
        "book_value_float": book_average_float, "pod_list": map_pod_list,
        "asof_str": f"IBKR daily reports · {len(session_list)} sessions",
        "size_legend_str": f"Size = average weight in {month_short_str} (time held counts)",
        "exposure_legend_str": f"Size = average gross exposure in {month_short_str} (value × leverage; TQQQ = 3x) · cash not shown",
        "colour_legend_str": f"Colour = return while held in {month_short_str}", "hatch_legend_str": "sold during the month"}
    if not finite_tree_bool(result_dict["map_dict"]):
        result_dict["map_dict"] = {"available_bool": False, "mode_str": "month", "pod_list": [], "reason_str": "Map data could not be verified."}
    selected_pnl_float = sum(pod_dict["pnl_float"] for pod_dict in selected_pod_list)
    if pod_str == "all":
        twr_float = report_dict["twr_float"]
    else:
        twr_float = selected_pod_list[0]["twr_float"] if selected_pod_list else None
    pod_bar_list = [{"name_str": pod_dict["identity_dict"]["name_str"], "color_str": pod_dict["identity_dict"]["color_str"],
        "value_float": pod_dict["symbol_pnl_float"], "value_str": compact_money_str(pod_dict["symbol_pnl_float"], signed_bool=True)}
        for pod_dict in selected_pod_list]
    other_float = sum(pod_dict["other_float"] for pod_dict in selected_pod_list)
    pod_bar_list.append({"name_str": "Other", "color_str": "", "other_bool": True, "value_float": other_float,
        "value_str": compact_money_str(other_float, signed_bool=True),
        "title_str": "Interest, dividends and fees: P&L that belongs to no single position. Commissions on rows stay with their position."})
    result_dict["side_dict"] = side_dict(big_value_float=selected_pnl_float,
        big_detail_str="P&L in " + period_label_str + (" · " + signed_percent_str(twr_float) + " TWR" if twr_float is not None else ""),
        pod_bar_list=pod_bar_list, item_list=map_item_all_list)
    # Table rows (Pod-filtered), largest average weight first. Weight is of the book.
    for pod_dict in selected_pod_list:
        identity_dict = pod_dict["identity_dict"]
        for item_dict in pod_dict["item_list"]:
            weight_float = item_dict["size_float"] / book_average_float if book_average_float > 0 else None
            result_dict["row_list"].append({"key_str": item_dict["symbol_str"] + "|" + item_dict["pod_id_str"],
                "symbol_str": item_dict["symbol_str"], "pod_dict": dict(identity_dict),
                "held_str": held_str(item_dict), "session_str": f"{item_dict['session_int']} of {item_dict['session_count_int']} sessions",
                "value_str": _money_str(item_dict["held_value_float"]), "weight_str": f"{weight_float:.1%}" if weight_float is not None else "—",
                "weight_percent_float": weight_float * 100 if weight_float is not None else None,
                "pnl_str": _money_str(item_dict["pnl_float"], signed_bool=True), "pnl_tone_str": "pos" if item_dict["pnl_float"] > 0 else "neg" if item_dict["pnl_float"] < 0 else "",
                "return_str": signed_percent_str(item_dict["return_float"]) + " while held" if item_dict["return_float"] is not None else "",
                "status_str": status_text_str(item_dict), "status_tag_bool": item_dict["status_str"] in {"open", "sold"},
                "short_bool": item_dict["short_bool"],
                "sort_dict": {"symbol": item_dict["symbol_str"], "pod": identity_dict["name_str"], "held": item_dict["first_str"],
                    "value": round(item_dict["held_value_float"], 2), "weight": round(item_dict["size_float"], 2),
                    "pnl": round(item_dict["pnl_float"], 2), "status": status_text_str(item_dict)}})
    result_dict["row_list"].sort(key=lambda row_dict: (-row_dict["sort_dict"]["weight"], row_dict["symbol_str"], row_dict["pod_dict"]["name_str"]))
    # Tiles and By pod: portfolio scope.
    book_pnl_float = report_dict["pnl_float"]
    ranked_list = sorted(all_item_list, key=lambda item_dict: (-item_dict["pnl_float"], item_dict["symbol_str"]))
    name_by_pod_dict = {pod_dict["account_dict"]["pod_id"]: pod_dict["identity_dict"]["name_str"] for pod_dict in pod_result_list}
    tile_label_str = month_short_str + (" (to date)" if result_dict["to_date_bool"] else " " + month_obj.strftime("%Y"))
    result_dict["tile_list"] = [
        {"label_str": "P&L · " + tile_label_str, "value_str": _money_str(book_pnl_float, signed_bool=True),
            "tone_str": "pos" if book_pnl_float > 0 else "neg" if book_pnl_float < 0 else "",
            "detail_str": (signed_percent_str(report_dict["twr_float"]) + " TWR · " if report_dict["twr_float"] is not None else "")
                + "positions " + compact_money_str(sum(pod_dict["symbol_pnl_float"] for pod_dict in pod_result_list), signed_bool=True)
                + ", other " + compact_money_str(sum(pod_dict["other_float"] for pod_dict in pod_result_list), signed_bool=True)},
        {"label_str": "Held · " + tile_label_str, "value_str": f"{len(all_item_list)} positions", "tone_str": "",
            "detail_str": f"{sold_count_int} sold during the month · {len(session_list)} sessions"}]
    for label_str, item_dict in (("Best", ranked_list[0] if ranked_list else None), ("Worst", ranked_list[-1] if ranked_list else None)):
        if item_dict is None:
            result_dict["tile_list"].append({"label_str": label_str + " · " + tile_label_str, "value_str": "—", "tone_str": "", "detail_str": "No positions"})
            continue
        result_dict["tile_list"].append({"label_str": label_str + " · " + tile_label_str,
            "value_str": item_dict["symbol_str"] + " " + _money_str(item_dict["pnl_float"], signed_bool=True),
            "tone_str": "pos" if item_dict["pnl_float"] > 0 else "neg" if item_dict["pnl_float"] < 0 else "",
            "detail_str": (signed_percent_str(item_dict["return_float"]) + " while held · " if item_dict["return_float"] is not None else "")
                + name_by_pod_dict[item_dict["pod_id_str"]]})
    for pod_dict in pod_result_list:
        identity_dict = pod_dict["identity_dict"]
        result_dict["pod_row_list"].append({**identity_dict, "held_str": str(len(pod_dict["item_list"])),
            "sold_str": str(sum(item_dict["status_str"] == "sold" for item_dict in pod_dict["item_list"])),
            "nav_str": _money_str(pod_dict["average_nav_float"]), "pnl_str": _money_str(pod_dict["pnl_float"], signed_bool=True),
            "pnl_tone_str": "pos" if pod_dict["pnl_float"] > 0 else "neg" if pod_dict["pnl_float"] < 0 else "",
            "twr_str": signed_percent_str(pod_dict["twr_float"]) if pod_dict["twr_float"] is not None else "—"})
    result_dict["total_dict"] = {"name_str": "Portfolio", "held_str": str(len(all_item_list)), "sold_str": str(sold_count_int),
        "nav_str": _money_str(book_average_float), "pnl_str": _money_str(book_pnl_float, signed_bool=True),
        "pnl_tone_str": "pos" if book_pnl_float > 0 else "neg" if book_pnl_float < 0 else "",
        "twr_str": signed_percent_str(report_dict["twr_float"]) if report_dict["twr_float"] is not None else "—"}
    result_dict.update(available_bool=True, reason_str="",
        verdict_str=result_dict["name_str"] + (" so far" if result_dict["to_date_bool"] else "") + ": "
            + ("up " if book_pnl_float >= 0 else "down ") + _money_str(abs(book_pnl_float))
            + (" (" + signed_percent_str(report_dict["twr_float"]) + ")" if report_dict["twr_float"] is not None else "") + ".",
        verdict_detail_str=f"{len(all_item_list)} positions held, {sold_count_int} sold during the month.",
        source_str="Daily IBKR reports · " + _day_label_str(from_str) + " – " + _day_label_str(to_str)
            + (" (to date)" if result_dict["to_date_bool"] else ""))
    return result_dict
