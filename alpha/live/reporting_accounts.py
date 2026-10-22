"""The client's other accounts: Flex accounts that no LIVE strategy owns.

The releases decide what a strategy is. Every other account the saved Flex
reports belongs to the client too, so it is found here, never configured.
It enters the client's totals only: no entry creates a Pod, changes account
routing or authorizes broker actions.
"""

from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Lock
from xml.etree import ElementTree

from alpha.live.client_reporting import ClientReportingError, load_broker_reporting_snapshot
from alpha.live.dashboard_v4.account_alias import parse_alias_by_account_dict

_statement_cache_dict: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
_statement_cache_lock = Lock()


def other_account_name_str(account_route_str, alias_str=""):
    """The IBKR alias, else the last four characters: reports carry no full account ID."""
    return alias_str or f"Other account …{account_route_str[-4:]}"


def _statement_tuple(checksum_str, raw_xml_str):
    """Currency and alias per account in one saved report, cached by checksum."""
    with _statement_cache_lock:
        cached_tuple = _statement_cache_dict.get(checksum_str)
    if cached_tuple is not None:
        return cached_tuple
    try:
        root_obj = ElementTree.fromstring(raw_xml_str)
        alias_by_account_dict = parse_alias_by_account_dict(raw_xml_str)
    except ElementTree.ParseError as exception_obj:
        raise ClientReportingError("Saved Flex XML is malformed.") from exception_obj
    currency_by_account_dict = {}
    for statement_obj in root_obj.findall(".//FlexStatement"):
        account_str = str(statement_obj.get("accountId") or "").strip()
        info_obj = statement_obj.find("AccountInformation")
        if account_str:
            currency_by_account_dict[account_str] = "" if info_obj is None else str(info_obj.get("currency") or "")
    result_tuple = (currency_by_account_dict, alias_by_account_dict)
    with _statement_cache_lock:
        _statement_cache_dict[checksum_str] = result_tuple
    return result_tuple


def discover_other_account_list(db_path_str, *, query_name_str, strategy_account_route_set):
    """Every saved Flex account outside the strategies, from its first money day.

    effective_from is the first saved day with a nonzero starting or ending NAV:
    before it the account held nothing. An account that never held money is
    left out. Once counted, a missing day withholds client totals for that day;
    missing NAV is never zero. Read-only: no database is created or migrated.
    """
    db_path_obj = Path(db_path_str).expanduser()
    if not db_path_obj.is_file():
        return []
    try:
        with closing(sqlite3.connect(db_path_obj.resolve().as_uri() + "?mode=ro", uri=True)) as connection_obj:
            import_row_list = connection_obj.execute(
                "SELECT checksum_str, raw_xml_str FROM flex_import WHERE query_name_str = ? ORDER BY import_id_int",
                (query_name_str,),
            ).fetchall()
    except sqlite3.Error:
        return []
    alias_by_account_dict, currency_by_account_dict = {}, {}
    for checksum_str, raw_xml_str in import_row_list:
        import_currency_dict, import_alias_dict = _statement_tuple(checksum_str, raw_xml_str)
        for account_str, currency_str in import_currency_dict.items():
            if account_str not in strategy_account_route_set:
                currency_by_account_dict[account_str] = currency_str
        alias_by_account_dict.update(import_alias_dict)
    if not currency_by_account_dict:
        return []
    non_usd_list = sorted(account_str for account_str, currency_str in currency_by_account_dict.items() if currency_str != "USD")
    if non_usd_list:
        raise ClientReportingError(
            ", ".join(other_account_name_str(account_str, alias_by_account_dict.get(account_str, ""))
                for account_str in non_usd_list)
            + ": only USD accounts can be added to the client's totals.")
    snapshot_obj = load_broker_reporting_snapshot(
        str(db_path_obj), allowed_account_set=set(currency_by_account_dict), query_name_str=query_name_str)
    first_date_dict = {}
    for row_obj in snapshot_obj.row_tuple:
        if row_obj.opening_nav_decimal or row_obj.closing_nav_decimal:
            first_date_dict.setdefault(row_obj.account_route_str, row_obj.market_date_str)
    return [{
        "role": "reserve", "account_route": account_str,
        "display_name": other_account_name_str(account_str, alias_by_account_dict.get(account_str, "")),
        "effective_from": first_date_dict[account_str], "effective_to": None,
    } for account_str in sorted(first_date_dict)]
