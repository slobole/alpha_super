"""The client's other accounts: Flex accounts that no LIVE strategy owns.

The releases decide what a strategy is. Every other account the saved Flex
reports belongs to the client too, so it is found here, never configured.
It enters the client's totals only: no entry creates a Pod, changes account
routing or authorizes broker actions.
"""

from contextlib import closing
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
from threading import Lock
from xml.etree import ElementTree

import pandas as pd

from alpha.live.client_reporting import NAV_METADATA_FIELD_SET, ClientReportingError
from alpha.live.dashboard_v4.account_alias import parse_alias_by_account_dict
from alpha.live.scheduler_utils import get_exchange_calendar_obj

# Fields that say which row it is, not what money did. A nonzero value in any
# other Change in NAV field (NAV, deposits, fees, interest...) is activity.
_IDENTITY_FIELD_SET = NAV_METADATA_FIELD_SET - {"startingValue", "endingValue"}
_statement_cache_dict: dict[str, tuple[dict[str, str], dict[str, str], dict[str, str]]] = {}
_statement_cache_lock = Lock()


def other_account_name_str(account_route_str, alias_str=""):
    """The IBKR alias, else the last four characters: reports carry no full account ID."""
    return alias_str or f"Other account …{account_route_str[-4:]}"


def _active_bool(nav_obj):
    # Missing required NAV is unknown, not proof of an unused account. Keep
    # the account in scope so the existing broker parser rejects the row.
    try:
        if any(not Decimal(nav_obj.get(field_str, "")).is_finite()
                for field_str in ("startingValue", "endingValue", "twr")):
            return True
    except InvalidOperation:
        return True
    for field_str, value_str in nav_obj.attrib.items():
        if field_str in _IDENTITY_FIELD_SET:
            continue
        try:
            if Decimal(value_str) != 0:
                return True
        except InvalidOperation:
            # Unreadable money is not proof of an empty account; reporting rejects it later.
            return True
    return False


def _statement_tuple(checksum_str, raw_xml_str):
    """Per account in one saved report: currency, alias and first active day (cached)."""
    with _statement_cache_lock:
        cached_tuple = _statement_cache_dict.get(checksum_str)
    if cached_tuple is not None:
        return cached_tuple
    try:
        root_obj = ElementTree.fromstring(raw_xml_str)
        alias_by_account_dict = parse_alias_by_account_dict(raw_xml_str)
    except ElementTree.ParseError as exception_obj:
        raise ClientReportingError("Saved Flex XML is malformed.") from exception_obj
    currency_by_account_dict, first_active_dict = {}, {}
    for statement_obj in root_obj.findall(".//FlexStatement"):
        account_str = str(statement_obj.get("accountId") or "").strip()
        if not account_str:
            continue
        info_obj = statement_obj.find("AccountInformation")
        currency_by_account_dict[account_str] = "" if info_obj is None else str(info_obj.get("currency") or "")
        for nav_obj in statement_obj.findall("ChangeInNAV"):
            if not _active_bool(nav_obj):
                continue
            try:
                date_str = datetime.strptime(str(nav_obj.get("toDate")), "%Y%m%d").date().isoformat()
            except ValueError as exception_obj:
                raise ClientReportingError("Invalid broker NAV date.") from exception_obj
            first_active_dict[account_str] = min(first_active_dict.get(account_str, date_str), date_str)
    result_tuple = (currency_by_account_dict, alias_by_account_dict, first_active_dict)
    with _statement_cache_lock:
        _statement_cache_dict[checksum_str] = result_tuple
    return result_tuple


def _baseline_session_str(start_str):
    """The last XNYS session before start_str; any calendar date is accepted."""
    end_ts = pd.Timestamp(start_str) - pd.Timedelta(days=1)
    session_list = get_exchange_calendar_obj("XNYS").sessions_in_range(end_ts - pd.Timedelta(days=14), end_ts)
    return session_list[-1].date().isoformat() if len(session_list) else end_ts.date().isoformat()


def discover_other_account_list(db_path_str, *, query_name_str, strategy_start_by_route_dict, excluded_route_set=frozenset()):
    """Every saved Flex account outside the strategies, from its first active day.

    *** CRITICAL *** retrospective membership, read from every saved import,
    including ones a later correction replaced: once an account showed money
    activity it stays a member, and a corrected-away day is missing (client
    totals withheld), never zero. Activity is any nonzero money field, so a
    deposit fully spent on fees still counts.

    strategy_start_by_route_dict maps each strategy account to its first Pod
    day (or None). A Pod's baseline session (the last XNYS session before that
    day) and later belong to an ordinary Pod, as before. An account already
    observed as Other retains even a one-day baseline history. Import-time
    route snapshots preserve that distinction through corrections and later
    bindings. Legacy imports without snapshots use the saved bindings; their
    earlier role changes cannot be reconstructed. Accounts in excluded_route_set
    (another owner's) are never counted. Read-only.
    """
    db_path_obj = Path(db_path_str).expanduser()
    if not db_path_obj.is_file():
        return []
    try:
        with closing(sqlite3.connect(db_path_obj.resolve().as_uri() + "?mode=ro", uri=True)) as connection_obj:
            connection_obj.execute("BEGIN")
            column_set = {row_tuple[1] for row_tuple in connection_obj.execute("PRAGMA table_info(flex_import)")}
            role_column_str = "strategy_account_routes_json_str" if "strategy_account_routes_json_str" in column_set else "NULL"
            import_row_list = connection_obj.execute(
                f"SELECT checksum_str, raw_xml_str, {role_column_str} FROM flex_import WHERE query_name_str = ? ORDER BY import_id_int",
                (query_name_str,),
            ).fetchall()
            legacy_route_set = {row_tuple[0] for row_tuple in connection_obj.execute("SELECT account_route_str FROM pod_binding")}
    except sqlite3.Error:
        return []
    alias_by_account_dict, currency_by_account_dict, first_active_dict, first_other_dict = {}, {}, {}, {}
    for checksum_str, raw_xml_str, strategy_routes_json_str in import_row_list:
        import_currency_dict, import_alias_dict, import_active_dict = _statement_tuple(checksum_str, raw_xml_str)
        import_strategy_set = legacy_route_set
        if strategy_routes_json_str is not None:
            try:
                strategy_route_list = json.loads(strategy_routes_json_str)
            except (ValueError, TypeError) as exception_obj:
                raise ClientReportingError("Saved Flex account-role evidence is invalid.") from exception_obj
            if not isinstance(strategy_route_list, list) or any(not isinstance(route_str, str) or not route_str for route_str in strategy_route_list):
                raise ClientReportingError("Saved Flex account-role evidence is invalid.")
            import_strategy_set = set(strategy_route_list)
        currency_by_account_dict.update(import_currency_dict)
        alias_by_account_dict.update(import_alias_dict)
        for account_str, date_str in import_active_dict.items():
            first_active_dict[account_str] = min(first_active_dict.get(account_str, date_str), date_str)
            if account_str not in import_strategy_set:
                first_other_dict[account_str] = min(first_other_dict.get(account_str, date_str), date_str)
    result_list = []
    for account_str, first_str in sorted(first_active_dict.items()):
        if account_str in excluded_route_set:
            continue
        effective_to_str = None
        if account_str in strategy_start_by_route_dict:
            start_str = strategy_start_by_route_dict[account_str]
            if not start_str or first_str >= start_str or (
                    first_other_dict.get(account_str, start_str) >= start_str and first_str >= _baseline_session_str(start_str)):
                continue
            effective_to_str = (date.fromisoformat(start_str) - timedelta(days=1)).isoformat()
        result_list.append({
            "role": "reserve", "account_route": account_str,
            "display_name": other_account_name_str(account_str, alias_by_account_dict.get(account_str, "")),
            "effective_from": first_str, "effective_to": effective_to_str,
        })
    non_usd_list = [account_dict["display_name"] for account_dict in result_list
        if currency_by_account_dict.get(account_dict["account_route"]) != "USD"]
    if non_usd_list:
        raise ClientReportingError(", ".join(non_usd_list) + ": only USD accounts can be added to the client's totals.")
    return result_list
