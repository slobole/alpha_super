"""Read one self-contained saved IBKR portfolio observation; never query IBKR."""

from contextlib import closing
from datetime import timedelta
from decimal import Decimal
import json
import math
from pathlib import Path
import sqlite3

from alpha.live.dashboard_v4.positions_data import (
    IDENTITY_FIELD_TUPLE, POSITION_LIMIT_INT, observed_timestamp_ts,
)


VALUATION_BYTE_LIMIT_INT = 1048576
OWNER_FIELD_TUPLE = (*IDENTITY_FIELD_TUPLE, "mode_str")


def _finite_bool(value_obj):
    try:
        return type(value_obj) in {int, float} and math.isfinite(value_obj)
    except OverflowError:
        return False


def _unique_object_dict(pair_list):
    result_dict = dict(pair_list)
    if len(result_dict) != len(pair_list):
        raise ValueError("Duplicate portfolio field")
    return result_dict


def load_broker_holdings_dict(target_obj, *, as_of_ts):
    """Read the payload's own owner, timestamp, cash and complete value rows.

    Ordinary broker cache writes can be newer than this optional observation.
    None of their quantities, money or timestamps enter this value snapshot.
    A missing/failed capture never falls back to older portfolio values.
    """
    result_dict = {"available_bool": False, "reason_str": "IBKR position values not saved yet",
        "position_list": [], "cash_float": None, "broker_nav_float": None,
        "observed_timestamp_str": "", "source_str": "IBKR portfolio", "currency_str": "USD"}
    try:
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live" or release_obj.enabled_bool is not True or as_of_ts.tzinfo is None:
            return result_dict
        identity_dict = {field_str: getattr(release_obj, field_str) for field_str in OWNER_FIELD_TUPLE}
        if any(not isinstance(value_str, str) or not value_str.strip() for value_str in identity_dict.values()):
            return result_dict
        db_path_obj = Path(target_obj.db_path_str).resolve()
        with closing(sqlite3.connect(db_path_obj.as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("PRAGMA query_only=ON")
            progress_list = [0]

            def stop_large_read_bool():
                progress_list[0] += 1
                return progress_list[0] > 1000

            connection_obj.set_progress_handler(stop_large_read_bool, 1000)
            connection_obj.execute("BEGIN")
            column_set = {row_obj["name"] for row_obj in connection_obj.execute("PRAGMA table_info(broker_snapshot_cache)")}
            if "portfolio_valuation_json_str" not in column_set:
                return result_dict
            release_list = connection_obj.execute(
                "SELECT release_id_str,user_id_str,pod_id_str,account_route_str,mode_str FROM live_release "
                "WHERE release_id_str=? LIMIT 2", (release_obj.release_id_str,)).fetchall()
            if len(release_list) != 1 or any(release_list[0][field_str] != identity_dict[field_str] for field_str in OWNER_FIELD_TUPLE):
                raise ValueError("Saved owner differs")
            cache_list = connection_obj.execute(
                "SELECT CASE WHEN length(CAST(portfolio_valuation_json_str AS BLOB))<=? "
                "THEN portfolio_valuation_json_str END AS portfolio_valuation_json_str "
                "FROM broker_snapshot_cache WHERE account_route_str=? LIMIT 2",
                (VALUATION_BYTE_LIMIT_INT, release_obj.account_route_str)).fetchall()
            if not cache_list:
                return result_dict
            if len(cache_list) != 1:
                raise ValueError("Duplicate account observation")
            payload_str = cache_list[0]["portfolio_valuation_json_str"]
            if payload_str is None:
                return result_dict
            if not isinstance(payload_str, str) or len(payload_str.encode("utf-8")) > VALUATION_BYTE_LIMIT_INT:
                raise ValueError("Invalid portfolio record")
        payload_dict = json.loads(payload_str, object_pairs_hook=_unique_object_dict)
        if (not isinstance(payload_dict, dict) or type(payload_dict.get("schema_version_int")) is not int
                or payload_dict["schema_version_int"] != 1 or payload_dict.get("owner_dict") != identity_dict
                or payload_dict.get("account_route_str") != release_obj.account_route_str
                or payload_dict.get("source_str") != "IBKR portfolio"):
            raise ValueError("Unverified portfolio identity")
        timestamp_ts = observed_timestamp_ts(payload_dict.get("observed_timestamp_str"), as_of_ts)
        if timestamp_ts.utcoffset() != timedelta(0):
            raise ValueError("Portfolio time must be UTC")
        if payload_dict.get("available_bool") is False:
            result_dict["reason_str"] = "IBKR position values unavailable at the last capture"
            return result_dict
        if payload_dict.get("available_bool") is not True or payload_dict.get("currency_str") != "USD":
            raise ValueError("Unsupported portfolio currency")
        cash_float, nav_float = payload_dict.get("cash_float"), payload_dict.get("broker_nav_float")
        if not _finite_bool(cash_float) or not _finite_bool(nav_float):
            raise ValueError("Invalid portfolio cash or NAV")
        position_list = payload_dict.get("position_list")
        if not isinstance(position_list, list) or len(position_list) > POSITION_LIMIT_INT:
            raise ValueError("Incomplete portfolio rows")
        symbol_set, conid_set = set(), set()
        verified_position_list = []
        for position_dict in position_list:
            if not isinstance(position_dict, dict):
                raise ValueError("Invalid portfolio row")
            symbol_str, conid_int = position_dict.get("symbol_str"), position_dict.get("conid_int")
            if (not isinstance(symbol_str, str) or not symbol_str.strip() or symbol_str.strip() != symbol_str
                    or len(symbol_str) > 100 or symbol_str in symbol_set or type(conid_int) is not int
                    or conid_int <= 0 or conid_int in conid_set or position_dict.get("currency_str") != "USD"):
                raise ValueError("Ambiguous portfolio contract")
            shares_float, price_float, value_float = (position_dict.get(field_str)
                for field_str in ("shares_float", "market_price_float", "value_float"))
            if (not all(_finite_bool(value_obj) for value_obj in (shares_float, price_float, value_float))
                    or price_float < 0 or (shares_float != 0 and price_float == 0)
                    or (shares_float == 0 and value_float != 0)
                    or (shares_float > 0 and value_float < 0) or (shares_float < 0 and value_float > 0)):
                raise ValueError("Invalid portfolio value")
            value_decimal = Decimal(str(value_float))
            calculated_decimal = Decimal(str(shares_float)) * Decimal(str(price_float))
            if abs(calculated_decimal - value_decimal) > max(Decimal(".02"), abs(value_decimal) * Decimal(".000001")):
                raise ValueError("Portfolio value does not match quantity and mark")
            symbol_set.add(symbol_str)
            conid_set.add(conid_int)
            verified_position_dict = {field_str: position_dict[field_str] for field_str in (
                "symbol_str", "conid_int", "currency_str", "shares_float", "market_price_float", "value_float")}
            average_cost_obj, unrealized_pnl_obj = (position_dict.get(field_str)
                for field_str in ("average_cost_float", "unrealized_pnl_float"))
            if (all(_finite_bool(number_obj) and abs(number_obj) < 1e15
                    for number_obj in (average_cost_obj, unrealized_pnl_obj)) and average_cost_obj >= 0):
                cost_decimal = Decimal(str(shares_float)) * Decimal(str(average_cost_obj))
                # P&L = value - signed shares * average cost. Cost/P&L are an
                # optional pair; older records and bad P&L retain valid marks.
                # Five cents or one ppm of value/cost allows broker rounding.
                tolerance_decimal = max(Decimal(".05"), max(abs(value_decimal), abs(cost_decimal)) * Decimal(".000001"))
                if abs(Decimal(str(unrealized_pnl_obj)) - (value_decimal - cost_decimal)) <= tolerance_decimal:
                    verified_position_dict.update(average_cost_float=float(average_cost_obj),
                        unrealized_pnl_float=float(unrealized_pnl_obj))
            if shares_float != 0:
                verified_position_list.append(verified_position_dict)
        result_dict.update(available_bool=True, reason_str="", cash_float=cash_float,
            broker_nav_float=nav_float, observed_timestamp_str=timestamp_ts.isoformat(),
            position_list=verified_position_list)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, ArithmeticError, RecursionError, OSError, sqlite3.Error):
        result_dict["reason_str"] = "Saved IBKR position values could not be verified"
    return result_dict
