"""Optional closing holdings from the exact saved Flex source of account NAV.

IBKR field meanings: https://www.ibkrguides.com/reportingreference/reportguide/open%20positionsfq.htm
Only USD stock/ETF SUMMARY rows with unit multipliers are supported. The XML
attribute contract has synthetic coverage; a real expanded export is still needed
for production acceptance. This reader performs no import, migration or broker IO.
"""

from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import sqlite3
from threading import Lock
import xml.etree.ElementTree as ElementTree
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import BrokerNavRow, ClientReportingError, parse_broker_nav_import


XML_BYTE_LIMIT_INT = 8_000_000
POSITION_LIMIT_INT = 2_000
CACHE_LIMIT_INT = 16
NEWER_IMPORT_LIMIT_INT = 2_000
_cache_dict = OrderedDict()
_cache_lock = Lock()


def _unavailable_dict(reason_str):
    return {"available_bool": False, "reason_str": reason_str, "close_date_str": "",
        "position_list": [], "source_str": ""}


def _date_obj(value_str):
    result_obj = date.fromisoformat(value_str)
    if result_obj.isoformat() != value_str:
        raise ValueError("Noncanonical date")
    return result_obj


def _scope_valid_bool(nav_row_obj, query_name_str, as_of_ts):
    try:
        return (isinstance(nav_row_obj, BrokerNavRow) and bool(nav_row_obj.account_route_str)
            and type(nav_row_obj.source_import_id_int) is int and nav_row_obj.source_import_id_int > 0
            and isinstance(query_name_str, str) and bool(query_name_str)
            and as_of_ts.tzinfo is not None
            # *** CRITICAL *** retrospective D+1 display: never expose today's
            # Activity Flex holdings as a finalized close or execution input.
            and _date_obj(nav_row_obj.market_date_str) < as_of_ts.astimezone(ZoneInfo("America/New_York")).date())
    except (TypeError, ValueError, AttributeError):
        return False


def _decimal_obj(attribute_dict, key_str):
    value_obj = Decimal(attribute_dict[key_str])
    if not value_obj.is_finite() or not math.isfinite(float(value_obj)):
        raise ValueError("Nonfinite position field")
    return value_obj


def parse_close_holdings_dict(raw_xml_str, nav_row_obj, *, query_name_str, as_of_ts):
    """Validate one immutable source; never infer value from current quantities.

    Position value is the broker's saved value. For the supported unit-multiplier
    instruments, verify abs(shares * closing mark - value) <= USD 0.01. NAV weights
    and cash reconciliation belong to the display builder; percentOfNAV is ignored
    because IBKR defines that field against the asset class, not account NAV.
    """
    unavailable_dict = _unavailable_dict("Closing position data could not be verified.")
    if not _scope_valid_bool(nav_row_obj, query_name_str, as_of_ts) or not isinstance(raw_xml_str, str):
        return unavailable_dict
    try:
        raw_bytes = raw_xml_str.encode("utf-8")
        if (len(raw_bytes) > XML_BYTE_LIMIT_INT or "<!DOCTYPE" in raw_xml_str.upper() or "<!ENTITY" in raw_xml_str.upper()
                or hashlib.sha256(raw_bytes).hexdigest() != nav_row_obj.source_checksum_str):
            return unavailable_dict
        cache_key_tuple = (query_name_str, nav_row_obj.source_import_id_int, nav_row_obj.source_checksum_str,
            nav_row_obj.account_route_str, nav_row_obj.market_date_str, nav_row_obj.opening_nav_decimal,
            nav_row_obj.closing_nav_decimal, nav_row_obj.twr_decimal, tuple(sorted(nav_row_obj.attribute_dict.items())))
        with _cache_lock:
            cached_dict = _cache_dict.get(cache_key_tuple)
            if cached_dict is not None:
                _cache_dict.move_to_end(cache_key_tuple)
                return deepcopy(cached_dict)
        nav_list = parse_broker_nav_import(raw_xml_str, allowed_account_set={nav_row_obj.account_route_str},
            query_name_str=query_name_str, source_import_id_int=nav_row_obj.source_import_id_int,
            source_checksum_str=nav_row_obj.source_checksum_str)
        selected_list = [row_obj for row_obj in nav_list if row_obj.market_date_str == nav_row_obj.market_date_str]
        if len(selected_list) != 1 or selected_list[0] != nav_row_obj:
            return unavailable_dict
        root_obj = ElementTree.fromstring(raw_xml_str)
        compact_date_str = nav_row_obj.market_date_str.replace("-", "")
        statement_list = [statement_obj for statement_obj in root_obj.findall(".//FlexStatement")
            if statement_obj.get("accountId") == nav_row_obj.account_route_str
            and any(nav_obj.get("fromDate") == compact_date_str and nav_obj.get("toDate") == compact_date_str
                for nav_obj in statement_obj.findall("ChangeInNAV"))]
        if len(statement_list) != 1:
            return unavailable_dict
        if any(statement_obj is not statement_list[0]
                and statement_obj.get("accountId") == nav_row_obj.account_route_str
                and any(position_obj.get("reportDate") == compact_date_str
                    for position_obj in statement_obj.findall(".//OpenPosition"))
                for statement_obj in root_obj.findall(".//FlexStatement")):
            return unavailable_dict
        section_list = statement_list[0].findall("OpenPositions")
        if not section_list:
            return _unavailable_dict("The saved report has no closing positions.")
        if len(section_list) != 1 or len(section_list[0]) > POSITION_LIMIT_INT:
            return unavailable_dict
        position_list, symbol_set, contract_set = [], set(), set()
        for position_obj in section_list[0]:
            attribute_dict = position_obj.attrib
            symbol_str = attribute_dict.get("symbol", "")
            contract_str = attribute_dict.get("conid", "")
            if (position_obj.tag != "OpenPosition" or attribute_dict.get("accountId") != nav_row_obj.account_route_str
                    or attribute_dict.get("reportDate") != compact_date_str or attribute_dict.get("model")
                    or attribute_dict.get("currency") != "USD" or attribute_dict.get("assetCategory") != "STK"
                    or attribute_dict.get("levelOfDetail") != "SUMMARY" or _decimal_obj(attribute_dict, "multiplier") != 1
                    or ("fxRateToBase" in attribute_dict and _decimal_obj(attribute_dict, "fxRateToBase") != 1)
                    or not symbol_str or symbol_str != symbol_str.strip() or len(symbol_str) > 80
                    or any(ord(character_str) < 32 for character_str in symbol_str)
                    or not contract_str.isascii() or not contract_str.isdigit() or int(contract_str) <= 0
                    or symbol_str in symbol_set or int(contract_str) in contract_set):
                return unavailable_dict
            share_decimal = _decimal_obj(attribute_dict, "position")
            mark_decimal = _decimal_obj(attribute_dict, "markPrice")
            value_decimal = _decimal_obj(attribute_dict, "positionValue")
            if mark_decimal < 0 or abs(share_decimal * mark_decimal - value_decimal) > Decimal("0.01"):
                return unavailable_dict
            symbol_set.add(symbol_str)
            contract_set.add(int(contract_str))
            if share_decimal != 0:
                position_list.append({"symbol_str": symbol_str, "shares_float": float(share_decimal), "value_float": float(value_decimal)})
        result_dict = {"available_bool": True, "reason_str": "", "close_date_str": nav_row_obj.market_date_str,
            "position_list": position_list, "source_str": "IBKR Flex closing positions"}
        with _cache_lock:
            _cache_dict[cache_key_tuple] = deepcopy(result_dict)
            _cache_dict.move_to_end(cache_key_tuple)
            while len(_cache_dict) > CACHE_LIMIT_INT:
                _cache_dict.popitem(last=False)
        return result_dict
    except (ClientReportingError, ElementTree.ParseError, ValueError, TypeError, KeyError, InvalidOperation, OverflowError):
        return unavailable_dict


def load_close_holdings_dict(database_path_str, nav_row_obj, *, query_name_str, as_of_ts):
    """Read the NAV source by identity, with no older-import/date fallback."""
    unavailable_dict = _unavailable_dict("Closing position data could not be verified.")
    if not _scope_valid_bool(nav_row_obj, query_name_str, as_of_ts):
        return unavailable_dict
    try:
        database_path_obj = Path(database_path_str).resolve()
        if not database_path_obj.is_file():
            return _unavailable_dict("No saved closing position report.")
        with closing(sqlite3.connect(database_path_obj.as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("PRAGMA query_only=ON")
            progress_list = [0]

            def stop_query_bool():
                progress_list[0] += 1
                return progress_list[0] > 200

            connection_obj.set_progress_handler(stop_query_bool, 1_000)
            connection_obj.execute("BEGIN")
            import_list = connection_obj.execute(
                "SELECT imported_timestamp_str, request_from_date_str, request_to_date_str, checksum_str, "
                "CASE WHEN length(CAST(raw_xml_str AS BLOB))<=? THEN raw_xml_str END AS raw_xml_str "
                "FROM flex_import WHERE import_id_int=? AND query_name_str=? LIMIT 2",
                (XML_BYTE_LIMIT_INT, nav_row_obj.source_import_id_int, query_name_str)).fetchall()
            if len(import_list) != 1:
                return unavailable_dict
            import_obj = import_list[0]
            imported_ts = datetime.fromisoformat(import_obj["imported_timestamp_str"])
            if (import_obj["checksum_str"] != nav_row_obj.source_checksum_str or imported_ts.tzinfo is None or imported_ts > as_of_ts
                    or not _date_obj(import_obj["request_from_date_str"]) <= _date_obj(nav_row_obj.market_date_str)
                        <= _date_obj(import_obj["request_to_date_str"])):
                return unavailable_dict
            newer_list = connection_obj.execute(
                "SELECT request_from_date_str, request_to_date_str FROM flex_import "
                "WHERE query_name_str=? AND import_id_int>? LIMIT ?",
                (query_name_str, nav_row_obj.source_import_id_int, NEWER_IMPORT_LIMIT_INT + 1)).fetchall()
            if len(newer_list) > NEWER_IMPORT_LIMIT_INT:
                return unavailable_dict
            for newer_obj in newer_list:
                from_date_obj, to_date_obj = _date_obj(newer_obj[0]), _date_obj(newer_obj[1])
                if from_date_obj > to_date_obj or from_date_obj <= _date_obj(nav_row_obj.market_date_str) <= to_date_obj:
                    return unavailable_dict
            return parse_close_holdings_dict(import_obj["raw_xml_str"], nav_row_obj,
                query_name_str=query_name_str, as_of_ts=as_of_ts)
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, OverflowError):
        return unavailable_dict
