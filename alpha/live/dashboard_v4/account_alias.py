"""IBKR account aliases for display, read from the latest saved Flex report.

The Flex sync already stores each report's raw XML in the performance
database. This module only reads it (SQLite mode=ro): Account Information
carries the alias the owner set in IBKR, and pod_binding maps each LIVE
account to its pod. An account without an alias, a missing database or an
unreadable report gives no alias, and the page shows the masked account only.
"""

from contextlib import closing
from functools import lru_cache
from pathlib import Path
import sqlite3
from xml.etree import ElementTree

ALIAS_ATTRIBUTE_TUPLE = ("acctAlias", "accountAlias")
ALIAS_MAX_LENGTH_INT = 40


def _clean_alias_str(raw_obj) -> str:
    alias_str = " ".join(str(raw_obj or "").split())
    return alias_str[:ALIAS_MAX_LENGTH_INT]


def parse_alias_by_account_dict(xml_text_str: str) -> dict[str, str]:
    """Alias per account route from one Flex XML report; accounts without one are left out."""
    root_obj = ElementTree.fromstring(xml_text_str)
    alias_by_account_dict = {}
    for statement_obj in root_obj.iter("FlexStatement"):
        account_str = str(statement_obj.attrib.get("accountId") or "").strip()
        if not account_str:
            continue
        # Account Information is the documented home of the alias; a Change in
        # NAV row carries it too when the query includes the field there.
        for element_obj in (statement_obj.find("AccountInformation"), statement_obj.find("ChangeInNAV")):
            if element_obj is None:
                continue
            alias_str = next((_clean_alias_str(element_obj.attrib[name_str]) for name_str in ALIAS_ATTRIBUTE_TUPLE
                if _clean_alias_str(element_obj.attrib.get(name_str))), "")
            if alias_str:
                alias_by_account_dict[account_str] = alias_str
                break
    return alias_by_account_dict


@lru_cache(maxsize=4)
def _alias_by_pod_cached_dict(db_path_str: str, mtime_ns_int: int, size_int: int) -> dict[str, str]:
    db_path_obj = Path(db_path_str)
    with closing(sqlite3.connect(f"{db_path_obj.as_uri()}?mode=ro", uri=True, timeout=1.0)) as connection_obj:
        xml_row_obj = connection_obj.execute(
            "SELECT raw_xml_str FROM flex_import ORDER BY import_id_int DESC LIMIT 1").fetchone()
        binding_row_list = connection_obj.execute(
            "SELECT account_route_str, pod_id_str FROM pod_binding").fetchall()
    if xml_row_obj is None:
        return {}
    alias_by_account_dict = parse_alias_by_account_dict(xml_row_obj[0])
    return {pod_id_str: alias_by_account_dict[account_str]
        for account_str, pod_id_str in binding_row_list if account_str in alias_by_account_dict}


def load_alias_by_pod_dict(db_path_str: str | None) -> dict[str, str]:
    """Alias per pod id; empty when the database or the alias is not there yet."""
    if not db_path_str:
        return {}
    try:
        db_path_obj = Path(db_path_str).expanduser().resolve()
        stat_obj = db_path_obj.stat()
        return dict(_alias_by_pod_cached_dict(str(db_path_obj), stat_obj.st_mtime_ns, stat_obj.st_size))
    except Exception:
        # Display only: any unreadable source must leave the page as it was, with the masked account.
        return {}


def account_label_str(alias_str: str, masked_account_str: str) -> str:
    """'alias · masked account' when the account has an alias, else the masked account."""
    return f"{alias_str} · {masked_account_str}" if alias_str else masked_account_str
