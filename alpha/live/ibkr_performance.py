"""Read-only IBKR Flex performance history for the LIVE shadow page.

IBKR remains the authority for each account's daily time-weighted return.
This module only validates that daily account data, stores it locally, and
builds a transparent multi-account composite for operator comparison.

The indicative account composite for market date D is:

    adjusted_base_i,D = ending_nav_i,D / (1 + account_return_i,D)
    fund_return_D = sum(ending_nav_i,D) / sum(adjusted_base_i,D) - 1

This is an indicative account composite, not Fund TWR and not an official IBKR
consolidated result. If a flow-sensitive day is detected, the entire combined
history is withheld. Offsetting intraday flows can still evade that check, so
this line must remain a Shadow diagnostic.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import xml.etree.ElementTree as ElementTree
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.live import scheduler_utils
CAPITAL_DB_ENV_STR = "ALPHA_IBKR_PERFORMANCE_DB_PATH_STR"
DEFAULT_PERFORMANCE_DB_PATH_STR = r"C:\alpha\live_ops\ibkr_performance.sqlite3"
SUPPORTED_CURRENCY_STR = "USD"
SUPPORTED_CALENDAR_ID_STR = "XNYS"
SUPPORTED_WINDOW_STR_SET = {"1w", "mtd", "ytd", "all"}


class PerformanceContractError(ValueError):
    """The Flex report or account binding is unsafe to interpret."""


@dataclass(frozen=True)
class PodPerformanceBinding:
    account_route_str: str
    pod_id_str: str
    return_start_date_str: str | None
    return_end_date_str: str | None
    enabled_bool: bool
    session_calendar_id_str: str = SUPPORTED_CALENDAR_ID_STR


@dataclass(frozen=True)
class FlexDailyPerformance:
    account_route_str: str
    pod_id_str: str
    market_date_str: str
    starting_nav_float: float
    ending_nav_float: float
    twr_pct_float: float
    twr_float: float


@dataclass(frozen=True)
class FundDailyPerformance:
    market_date_str: str
    fund_return_float: float
    starting_nav_float: float
    ending_nav_float: float
    adjusted_base_float: float
    active_pod_count_int: int


def resolve_performance_db_path_str() -> str:
    return os.getenv(CAPITAL_DB_ENV_STR, "").strip() or DEFAULT_PERFORMANCE_DB_PATH_STR


def parse_flex_change_in_nav_xml(
    xml_text_str: str,
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
    *,
    request_from_date_str: str | None = None,
    request_to_date_str: str | None = None,
    expected_query_name_str: str | None = None,
) -> list[FlexDailyPerformance]:
    """Parse the exact Account Information + Change in NAV Flex contract."""

    if not binding_by_account_route_dict:
        raise PerformanceContractError("No LIVE account-to-Pod bindings are configured.")
    _validate_binding_dict(binding_by_account_route_dict)
    try:
        root_obj = ElementTree.fromstring(xml_text_str)
    except ElementTree.ParseError as exception_obj:
        raise PerformanceContractError(
            f"IBKR Flex XML is malformed: {exception_obj}."
        ) from exception_obj
    if root_obj.tag != "FlexQueryResponse":
        raise PerformanceContractError(
            f"Expected FlexQueryResponse root, got {root_obj.tag!r}."
        )
    if expected_query_name_str is not None:
        actual_query_name_str = str(root_obj.attrib.get("queryName") or "").strip()
        if actual_query_name_str != expected_query_name_str:
            raise PerformanceContractError(
                f"Expected Flex query {expected_query_name_str!r}, got "
                f"{actual_query_name_str or 'missing'!r}."
            )

    statement_obj_list = root_obj.findall(".//FlexStatement")
    flex_statements_obj = root_obj.find("FlexStatements")
    if flex_statements_obj is not None and "count" in flex_statements_obj.attrib:
        try:
            declared_statement_count_int = int(flex_statements_obj.attrib["count"])
        except ValueError as exception_obj:
            raise PerformanceContractError("FlexStatements count is invalid.") from exception_obj
        if declared_statement_count_int != len(statement_obj_list):
            raise PerformanceContractError(
                "FlexStatements count does not match the returned statements."
            )

    required_account_route_set = _required_account_route_set(
        binding_by_account_route_dict,
        request_from_date_str=request_from_date_str,
        request_to_date_str=request_to_date_str,
    )
    row_by_key_dict: dict[tuple[str, str], FlexDailyPerformance] = {}
    reported_account_route_set: set[str] = set()
    for statement_obj in statement_obj_list:
        statement_account_route_str = str(statement_obj.attrib.get("accountId") or "").strip()
        if not statement_account_route_str:
            raise PerformanceContractError("FlexStatement is missing accountId.")
        if statement_account_route_str not in binding_by_account_route_dict:
            raise PerformanceContractError(
                f"Flex contains unknown LIVE account {statement_account_route_str}."
            )
        account_info_obj = statement_obj.find("AccountInformation")
        if account_info_obj is None:
            raise PerformanceContractError(
                f"Account Information is missing for {statement_account_route_str}."
            )
        info_account_route_str = str(account_info_obj.attrib.get("accountId") or "").strip()
        info_currency_str = str(account_info_obj.attrib.get("currency") or "").strip().upper()
        if info_account_route_str != statement_account_route_str:
            raise PerformanceContractError(
                f"Account Information ID mismatch for {statement_account_route_str}."
            )
        if info_currency_str != SUPPORTED_CURRENCY_STR:
            raise PerformanceContractError(
                f"Account {statement_account_route_str} currency is {info_currency_str or 'missing'}; "
                "v1 supports USD only."
            )

        change_in_nav_obj_list = statement_obj.findall("ChangeInNAV")
        if not change_in_nav_obj_list and statement_account_route_str in required_account_route_set:
            raise PerformanceContractError(
                f"Change in NAV is missing for {statement_account_route_str}."
            )
        for change_in_nav_obj in change_in_nav_obj_list:
            row_obj = _parse_change_in_nav_row(
                change_in_nav_obj,
                binding_by_account_route_dict[statement_account_route_str],
                statement_account_route_str,
            )
            if request_from_date_str and row_obj.market_date_str < request_from_date_str:
                raise PerformanceContractError(
                    f"Flex row {row_obj.market_date_str} precedes requested range {request_from_date_str}."
                )
            if request_to_date_str and row_obj.market_date_str > request_to_date_str:
                raise PerformanceContractError(
                    f"Flex row {row_obj.market_date_str} exceeds requested range {request_to_date_str}."
                )
            row_key_tuple = (row_obj.account_route_str, row_obj.market_date_str)
            if row_key_tuple in row_by_key_dict:
                raise PerformanceContractError(
                    f"Duplicate Change in NAV row for {row_obj.account_route_str} "
                    f"on {row_obj.market_date_str}."
                )
            if _is_session_date_bool(row_obj.market_date_str):
                row_by_key_dict[row_key_tuple] = row_obj
            else:
                _validate_ignorable_non_session_row(row_obj)
            reported_account_route_set.add(statement_account_route_str)

    missing_statement_account_set = required_account_route_set - reported_account_route_set
    if missing_statement_account_set:
        raise PerformanceContractError(
            "Flex is missing configured account(s): "
            + ", ".join(sorted(missing_statement_account_set))
            + "."
        )
    row_obj_list = sorted(
        row_by_key_dict.values(),
        key=lambda row_obj: (row_obj.market_date_str, row_obj.account_route_str),
    )
    if not row_obj_list:
        raise PerformanceContractError("Flex contains no XNYS session rows.")
    _validate_active_account_coverage(row_obj_list, binding_by_account_route_dict)
    if request_from_date_str and request_to_date_str:
        _validate_requested_session_coverage(
            row_obj_list,
            binding_by_account_route_dict,
            request_from_date_str,
            request_to_date_str,
        )
    if not any(
        _binding_is_active_bool(
            binding_by_account_route_dict[row_obj.account_route_str],
            row_obj.market_date_str,
        )
        for row_obj in row_obj_list
    ):
        raise PerformanceContractError(
            "Flex contains no rows after a trusted Pod EOD baseline."
        )
    return row_obj_list


def build_fund_daily_performance_list(
    row_obj_list: Iterable[FlexDailyPerformance],
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
) -> list[FundDailyPerformance]:
    """Build the explicitly labeled, indicative daily account composite."""

    row_by_date_and_account_dict = {
        (row_obj.market_date_str, row_obj.account_route_str): row_obj
        for row_obj in row_obj_list
    }
    market_date_str_list = sorted(
        {market_date_str for market_date_str, _ in row_by_date_and_account_dict}
    )
    fund_row_obj_list: list[FundDailyPerformance] = []
    for market_date_str in market_date_str_list:
        active_binding_obj_list = _active_binding_list(
            binding_by_account_route_dict.values(), market_date_str
        )
        if not active_binding_obj_list:
            continue
        missing_account_route_list = [
            binding_obj.account_route_str
            for binding_obj in active_binding_obj_list
            if (market_date_str, binding_obj.account_route_str)
            not in row_by_date_and_account_dict
        ]
        if missing_account_route_list:
            raise PerformanceContractError(
                f"Cannot build fund return for {market_date_str}; missing account(s): "
                + ", ".join(sorted(missing_account_route_list))
                + "."
            )
        ending_nav_float = 0.0
        starting_nav_float = 0.0
        adjusted_base_float = 0.0
        for binding_obj in active_binding_obj_list:
            row_obj = row_by_date_and_account_dict[
                (market_date_str, binding_obj.account_route_str)
            ]
            account_growth_float = 1.0 + row_obj.twr_float
            if account_growth_float <= 0.0:
                raise PerformanceContractError(
                    f"Account {row_obj.account_route_str} return is <= -100% on "
                    f"{market_date_str}; adjusted base is undefined."
                )
            account_adjusted_base_float = row_obj.ending_nav_float / account_growth_float
            if account_adjusted_base_float <= 0.0 or not math.isfinite(
                account_adjusted_base_float
            ):
                raise PerformanceContractError(
                    f"Account {row_obj.account_route_str} has an invalid adjusted base "
                    f"on {market_date_str}."
                )
            if not math.isclose(
                account_adjusted_base_float,
                row_obj.starting_nav_float,
                rel_tol=0.0,
                abs_tol=0.01,
            ):
                raise PerformanceContractError(
                    "The combined history is unavailable because at least one "
                    f"flow-sensitive date was detected ({market_date_str}, account "
                    f"{row_obj.account_route_str}). Official Pod TWR remains available."
                )
            ending_nav_float += row_obj.ending_nav_float
            starting_nav_float += row_obj.starting_nav_float
            adjusted_base_float += account_adjusted_base_float
        fund_return_float = ending_nav_float / adjusted_base_float - 1.0
        fund_row_obj_list.append(
            FundDailyPerformance(
                market_date_str=market_date_str,
                fund_return_float=fund_return_float,
                starting_nav_float=starting_nav_float,
                ending_nav_float=ending_nav_float,
                adjusted_base_float=adjusted_base_float,
                active_pod_count_int=len(active_binding_obj_list),
            )
        )
    return fund_row_obj_list


class PerformanceStore:
    def __init__(self, db_path_str: str) -> None:
        self.db_path_obj = Path(db_path_str).expanduser()

    def initialize(self) -> None:
        self.db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection_obj, connection_obj:
            connection_obj.executescript(
                """
                CREATE TABLE IF NOT EXISTS flex_import (
                    import_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    imported_timestamp_str TEXT NOT NULL,
                    request_from_date_str TEXT NOT NULL,
                    request_to_date_str TEXT NOT NULL,
                    query_name_str TEXT NOT NULL,
                    checksum_str TEXT NOT NULL UNIQUE,
                    raw_xml_str TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pod_binding (
                    account_route_str TEXT PRIMARY KEY,
                    pod_id_str TEXT NOT NULL,
                    return_start_date_str TEXT,
                    return_end_date_str TEXT,
                    enabled_bool_int INTEGER NOT NULL,
                    session_calendar_id_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_performance (
                    account_route_str TEXT NOT NULL,
                    pod_id_str TEXT NOT NULL,
                    market_date_str TEXT NOT NULL,
                    starting_nav_float REAL NOT NULL,
                    ending_nav_float REAL NOT NULL,
                    twr_pct_float REAL NOT NULL,
                    twr_float REAL NOT NULL,
                    source_import_id_int INTEGER NOT NULL,
                    PRIMARY KEY (account_route_str, market_date_str),
                    FOREIGN KEY (source_import_id_int)
                        REFERENCES flex_import(import_id_int)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_performance_date
                    ON daily_performance(market_date_str);
                CREATE TABLE IF NOT EXISTS sync_attempt (
                    attempt_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempted_timestamp_str TEXT NOT NULL,
                    status_str TEXT NOT NULL CHECK (status_str IN ('success', 'failed')),
                    request_from_date_str TEXT NOT NULL,
                    request_to_date_str TEXT NOT NULL,
                    detail_str TEXT
                );
                """
            )

    def replace_range(
        self,
        *,
        xml_text_str: str,
        query_name_str: str,
        request_from_date_str: str,
        request_to_date_str: str,
        binding_obj_list: list[PodPerformanceBinding],
        imported_timestamp_str: str | None = None,
    ) -> bool:
        binding_by_account_route_dict = {
            binding_obj.account_route_str: binding_obj
            for binding_obj in binding_obj_list
        }
        self.initialize()
        with closing(self._connect()) as parity_connection_obj, parity_connection_obj:
            self._validate_existing_binding_parity(
                parity_connection_obj, binding_by_account_route_dict
            )
        row_obj_list = parse_flex_change_in_nav_xml(
            xml_text_str,
            binding_by_account_route_dict,
            request_from_date_str=request_from_date_str,
            request_to_date_str=request_to_date_str,
            expected_query_name_str=query_name_str,
        )
        checksum_str = hashlib.sha256(xml_text_str.encode("utf-8")).hexdigest()
        imported_timestamp_str = imported_timestamp_str or datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection_obj, connection_obj:
            connection_obj.execute("BEGIN IMMEDIATE")
            duplicate_row_obj = connection_obj.execute(
                "SELECT import_id_int FROM flex_import WHERE checksum_str = ?",
                (checksum_str,),
            ).fetchone()
            self._validate_existing_binding_parity(
                connection_obj, binding_by_account_route_dict
            )
            if duplicate_row_obj is not None:
                self._upsert_binding_rows(
                    connection_obj, binding_obj_list, imported_timestamp_str
                )
                connection_obj.commit()
                return False
            cursor_obj = connection_obj.execute(
                """
                INSERT INTO flex_import (
                    imported_timestamp_str, request_from_date_str,
                    request_to_date_str, query_name_str, checksum_str, raw_xml_str
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    imported_timestamp_str,
                    request_from_date_str,
                    request_to_date_str,
                    query_name_str,
                    checksum_str,
                    xml_text_str,
                ),
            )
            import_id_int = int(cursor_obj.lastrowid)
            self._upsert_binding_rows(
                connection_obj, binding_obj_list, imported_timestamp_str
            )
            connection_obj.execute(
                """
                DELETE FROM daily_performance
                WHERE market_date_str BETWEEN ? AND ?
                """,
                (request_from_date_str, request_to_date_str),
            )
            for row_obj in row_obj_list:
                if not _binding_is_active_bool(
                    binding_by_account_route_dict[row_obj.account_route_str],
                    row_obj.market_date_str,
                ):
                    continue
                connection_obj.execute(
                    """
                    INSERT INTO daily_performance (
                        account_route_str, pod_id_str, market_date_str,
                        starting_nav_float, ending_nav_float, twr_pct_float,
                        twr_float, source_import_id_int
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_obj.account_route_str,
                        row_obj.pod_id_str,
                        row_obj.market_date_str,
                        row_obj.starting_nav_float,
                        row_obj.ending_nav_float,
                        row_obj.twr_pct_float,
                        row_obj.twr_float,
                        import_id_int,
                    ),
                )
            connection_obj.commit()
        return True

    def record_sync_attempt(
        self,
        *,
        status_str: str,
        request_from_date_str: str,
        request_to_date_str: str,
        detail_str: str | None,
        attempted_timestamp_str: str | None = None,
    ) -> None:
        if status_str not in {"success", "failed"}:
            raise ValueError(f"Unsupported sync status {status_str!r}.")
        attempted_timestamp_str = attempted_timestamp_str or datetime.now(UTC).isoformat()
        self.initialize()
        with closing(self._connect()) as connection_obj, connection_obj:
            connection_obj.execute(
                """
                INSERT INTO sync_attempt (
                    attempted_timestamp_str, status_str,
                    request_from_date_str, request_to_date_str, detail_str
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    attempted_timestamp_str,
                    status_str,
                    request_from_date_str,
                    request_to_date_str,
                    detail_str,
                ),
            )
            connection_obj.commit()

    def contains_checksum_bool(self, checksum_str: str) -> bool:
        if not self.db_path_obj.exists():
            return False
        with closing(self._connect()) as connection_obj, connection_obj:
            return connection_obj.execute(
                "SELECT 1 FROM flex_import WHERE checksum_str = ? LIMIT 1",
                (checksum_str,),
            ).fetchone() is not None

    def has_rows_in_range_bool(
        self, request_from_date_str: str, request_to_date_str: str
    ) -> bool:
        if not self.db_path_obj.exists():
            return False
        with closing(self._connect()) as connection_obj, connection_obj:
            return connection_obj.execute(
                """
                SELECT 1 FROM daily_performance
                WHERE market_date_str BETWEEN ? AND ? LIMIT 1
                """,
                (request_from_date_str, request_to_date_str),
            ).fetchone() is not None

    def validate_binding_obj_list(
        self, binding_obj_list: list[PodPerformanceBinding]
    ) -> None:
        binding_by_account_route_dict = {
            binding_obj.account_route_str: binding_obj
            for binding_obj in binding_obj_list
        }
        _validate_binding_dict(binding_by_account_route_dict)
        if not self.db_path_obj.exists():
            return
        with closing(self._connect()) as connection_obj, connection_obj:
            self._validate_existing_binding_parity(
                connection_obj, binding_by_account_route_dict
            )
            stored_account_route_set = {
                str(row_obj["account_route_str"])
                for row_obj in connection_obj.execute(
                    "SELECT account_route_str FROM pod_binding"
                ).fetchall()
            }
            new_account_route_set = (
                set(binding_by_account_route_dict) - stored_account_route_set
            )
            if stored_account_route_set and new_account_route_set:
                raise PerformanceContractError(
                    "LIVE manifest account(s) are not stored in the Shadow: "
                    + ", ".join(sorted(new_account_route_set))
                    + "."
                )

    def load_binding_dict(self) -> dict[str, PodPerformanceBinding]:
        if not self.db_path_obj.exists():
            return {}
        with closing(self._connect()) as connection_obj, connection_obj:
            row_obj_list = connection_obj.execute(
                """
                SELECT account_route_str, pod_id_str, return_start_date_str,
                       return_end_date_str, enabled_bool_int,
                       session_calendar_id_str
                FROM pod_binding
                ORDER BY account_route_str
                """
            ).fetchall()
        return {
            str(row_obj["account_route_str"]): PodPerformanceBinding(
                account_route_str=str(row_obj["account_route_str"]),
                pod_id_str=str(row_obj["pod_id_str"]),
                return_start_date_str=row_obj["return_start_date_str"],
                return_end_date_str=row_obj["return_end_date_str"],
                enabled_bool=bool(row_obj["enabled_bool_int"]),
                session_calendar_id_str=str(row_obj["session_calendar_id_str"]),
            )
            for row_obj in row_obj_list
        }

    def load_daily_row_list(self) -> list[FlexDailyPerformance]:
        if not self.db_path_obj.exists():
            return []
        with closing(self._connect()) as connection_obj, connection_obj:
            row_obj_list = connection_obj.execute(
                """
                SELECT account_route_str, pod_id_str, market_date_str,
                       starting_nav_float, ending_nav_float, twr_pct_float,
                       twr_float
                FROM daily_performance
                ORDER BY market_date_str, account_route_str
                """
            ).fetchall()
        return [
            FlexDailyPerformance(
                account_route_str=str(row_obj["account_route_str"]),
                pod_id_str=str(row_obj["pod_id_str"]),
                market_date_str=str(row_obj["market_date_str"]),
                starting_nav_float=float(row_obj["starting_nav_float"]),
                ending_nav_float=float(row_obj["ending_nav_float"]),
                twr_pct_float=float(row_obj["twr_pct_float"]),
                twr_float=float(row_obj["twr_float"]),
            )
            for row_obj in row_obj_list
        ]

    def load_latest_import_dict(self) -> dict[str, Any] | None:
        if not self.db_path_obj.exists():
            return None
        with closing(self._connect()) as connection_obj, connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT imported_timestamp_str, request_from_date_str,
                       request_to_date_str, query_name_str, checksum_str
                FROM flex_import
                ORDER BY import_id_int DESC
                LIMIT 1
                """
            ).fetchone()
        return None if row_obj is None else dict(row_obj)

    def load_snapshot_tuple(
        self,
    ) -> tuple[
        dict[str, PodPerformanceBinding],
        list[FlexDailyPerformance],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        if not self.db_path_obj.exists():
            return {}, [], None, None
        with closing(self._connect()) as connection_obj, connection_obj:
            connection_obj.execute("BEGIN")
            binding_row_obj_list = connection_obj.execute(
                """
                SELECT account_route_str, pod_id_str, return_start_date_str,
                       return_end_date_str, enabled_bool_int,
                       session_calendar_id_str
                FROM pod_binding ORDER BY account_route_str
                """
            ).fetchall()
            daily_row_obj_list = connection_obj.execute(
                """
                SELECT account_route_str, pod_id_str, market_date_str,
                       starting_nav_float, ending_nav_float, twr_pct_float,
                       twr_float
                FROM daily_performance ORDER BY market_date_str, account_route_str
                """
            ).fetchall()
            latest_import_row_obj = connection_obj.execute(
                """
                SELECT imported_timestamp_str, request_from_date_str,
                       request_to_date_str, query_name_str, checksum_str
                FROM flex_import ORDER BY import_id_int DESC LIMIT 1
                """
            ).fetchone()
            latest_attempt_row_obj = connection_obj.execute(
                """
                SELECT attempted_timestamp_str, status_str,
                       request_from_date_str, request_to_date_str, detail_str
                FROM sync_attempt ORDER BY attempt_id_int DESC LIMIT 1
                """
            ).fetchone()
            connection_obj.commit()
        binding_by_account_route_dict = {
            str(row_obj["account_route_str"]): PodPerformanceBinding(
                account_route_str=str(row_obj["account_route_str"]),
                pod_id_str=str(row_obj["pod_id_str"]),
                return_start_date_str=row_obj["return_start_date_str"],
                return_end_date_str=row_obj["return_end_date_str"],
                enabled_bool=bool(row_obj["enabled_bool_int"]),
                session_calendar_id_str=str(row_obj["session_calendar_id_str"]),
            )
            for row_obj in binding_row_obj_list
        }
        daily_row_list = [
            FlexDailyPerformance(
                account_route_str=str(row_obj["account_route_str"]),
                pod_id_str=str(row_obj["pod_id_str"]),
                market_date_str=str(row_obj["market_date_str"]),
                starting_nav_float=float(row_obj["starting_nav_float"]),
                ending_nav_float=float(row_obj["ending_nav_float"]),
                twr_pct_float=float(row_obj["twr_pct_float"]),
                twr_float=float(row_obj["twr_float"]),
            )
            for row_obj in daily_row_obj_list
        ]
        return (
            binding_by_account_route_dict,
            daily_row_list,
            None if latest_import_row_obj is None else dict(latest_import_row_obj),
            None if latest_attempt_row_obj is None else dict(latest_attempt_row_obj),
        )

    def _upsert_binding_rows(
        self,
        connection_obj: sqlite3.Connection,
        binding_obj_list: list[PodPerformanceBinding],
        updated_timestamp_str: str,
    ) -> None:
        for binding_obj in binding_obj_list:
            connection_obj.execute(
                """
                INSERT INTO pod_binding (
                    account_route_str, pod_id_str, return_start_date_str,
                    return_end_date_str, enabled_bool_int,
                    session_calendar_id_str, updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_route_str) DO UPDATE SET
                    pod_id_str = excluded.pod_id_str,
                    return_start_date_str = excluded.return_start_date_str,
                    return_end_date_str = excluded.return_end_date_str,
                    enabled_bool_int = excluded.enabled_bool_int,
                    session_calendar_id_str = excluded.session_calendar_id_str,
                    updated_timestamp_str = excluded.updated_timestamp_str
                """,
                (
                    binding_obj.account_route_str,
                    binding_obj.pod_id_str,
                    binding_obj.return_start_date_str,
                    binding_obj.return_end_date_str,
                    int(binding_obj.enabled_bool),
                    binding_obj.session_calendar_id_str,
                    updated_timestamp_str,
                ),
            )

    def _validate_existing_binding_parity(
        self,
        connection_obj: sqlite3.Connection,
        binding_by_account_route_dict: dict[str, PodPerformanceBinding],
    ) -> None:
        existing_row_obj_list = connection_obj.execute(
            """
            SELECT account_route_str, pod_id_str, return_start_date_str,
                   return_end_date_str, session_calendar_id_str
            FROM pod_binding
            """
        ).fetchall()
        for existing_row_obj in existing_row_obj_list:
            account_route_str = str(existing_row_obj["account_route_str"])
            binding_obj = binding_by_account_route_dict.get(account_route_str)
            if binding_obj is None:
                raise PerformanceContractError(
                    f"Stored account {account_route_str} is absent from LIVE manifests."
                )
            existing_pod_id_str = str(existing_row_obj["pod_id_str"])
            if existing_pod_id_str != binding_obj.pod_id_str:
                raise PerformanceContractError(
                    f"Account {account_route_str} was stored for {existing_pod_id_str} "
                    f"but is now mapped to {binding_obj.pod_id_str}."
                )
            existing_start_date_str = existing_row_obj["return_start_date_str"]
            if (
                existing_start_date_str is not None
                and binding_obj.return_start_date_str != existing_start_date_str
            ):
                raise PerformanceContractError(
                    f"Account {account_route_str} trusted return start changed from "
                    f"{existing_start_date_str} to "
                    f"{binding_obj.return_start_date_str or 'missing'}."
                )
            existing_end_date_str = existing_row_obj["return_end_date_str"]
            if (
                existing_end_date_str is not None
                and binding_obj.return_end_date_str != existing_end_date_str
            ):
                raise PerformanceContractError(
                    f"Account {account_route_str} trusted return end changed from "
                    f"{existing_end_date_str} to "
                    f"{binding_obj.return_end_date_str or 'missing'}."
                )
            existing_calendar_id_str = str(
                existing_row_obj["session_calendar_id_str"]
            )
            if existing_calendar_id_str != binding_obj.session_calendar_id_str:
                raise PerformanceContractError(
                    f"Account {account_route_str} calendar changed from "
                    f"{existing_calendar_id_str} to {binding_obj.session_calendar_id_str}."
                )

    def _connect(self) -> sqlite3.Connection:
        connection_obj = sqlite3.connect(str(self.db_path_obj), timeout=30.0)
        connection_obj.row_factory = sqlite3.Row
        connection_obj.execute("PRAGMA foreign_keys = ON")
        connection_obj.execute("PRAGMA busy_timeout = 30000")
        return connection_obj


def build_performance_page_dict(
    db_path_str: str,
    *,
    window_str: str = "all",
    as_of_ts: datetime | None = None,
    current_binding_obj_list: list[PodPerformanceBinding] | None = None,
) -> dict[str, Any]:
    # Lazy import avoids making the standalone sync/storage module depend on
    # dashboard_v3.__init__, which intentionally exports the Flask app.
    from alpha.live.dashboard_v3.charts import build_equity_chart_dict

    window_str = window_str if window_str in SUPPORTED_WINDOW_STR_SET else "all"
    db_path_obj = Path(db_path_str).expanduser()
    if not db_path_obj.exists():
        return {
            "status_str": "not_initialized",
            "status_label_str": "Shadow not initialized",
            "detail_str": f"No performance database at {db_path_obj}.",
            "window_str": window_str,
            "pod_performance_dict_list": [],
            "fund_chart_dict": None,
        }
    try:
        store_obj = PerformanceStore(str(db_path_obj))
        if current_binding_obj_list is not None:
            store_obj.validate_binding_obj_list(current_binding_obj_list)
        (
            binding_by_account_route_dict,
            row_obj_list,
            latest_import_dict,
            latest_attempt_dict,
        ) = store_obj.load_snapshot_tuple()
        if not binding_by_account_route_dict or not row_obj_list:
            raise PerformanceContractError("The performance database contains no finalized rows.")
        _validate_active_account_coverage(row_obj_list, binding_by_account_route_dict)
        first_required_date_str = min(
            binding_obj.return_start_date_str
            for binding_obj in binding_by_account_route_dict.values()
            if binding_obj.return_start_date_str is not None
        )
        latest_stored_date_str = max(
            row_obj.market_date_str for row_obj in row_obj_list
        )
        _validate_requested_session_coverage(
            row_obj_list,
            binding_by_account_route_dict,
            first_required_date_str,
            latest_stored_date_str,
        )
        pod_performance_dict_list = _build_pod_performance_dict_list(
            row_obj_list, binding_by_account_route_dict
        )
        try:
            fund_row_obj_list = build_fund_daily_performance_list(
                row_obj_list, binding_by_account_route_dict
            )
        except PerformanceContractError as exception_obj:
            latest_active_account_route_set = {
                binding_obj.account_route_str
                for binding_obj in _active_binding_list(
                    binding_by_account_route_dict.values(), latest_stored_date_str
                )
            }
            latest_covered_account_route_set = {
                row_obj.account_route_str
                for row_obj in row_obj_list
                if row_obj.market_date_str == latest_stored_date_str
            }
            return {
                "status_str": "error",
                "status_label_str": "Cannot verify indicative composite",
                "detail_str": str(exception_obj),
                "window_str": window_str,
                "latest_sync_timestamp_str": (
                    None
                    if latest_import_dict is None
                    else latest_import_dict["imported_timestamp_str"]
                ),
                "coverage_through_date_str": latest_stored_date_str,
                "covered_account_count_int": len(
                    latest_active_account_route_set & latest_covered_account_route_set
                ),
                "expected_account_count_int": len(latest_active_account_route_set),
                "query_name_str": (
                    None
                    if latest_import_dict is None
                    else latest_import_dict["query_name_str"]
                ),
                "checksum_short_str": (
                    None
                    if latest_import_dict is None
                    else str(latest_import_dict["checksum_str"])[:12]
                ),
                "fund_chart_dict": None,
                "pod_performance_dict_list": pod_performance_dict_list,
            }
        if not fund_row_obj_list:
            raise PerformanceContractError("No strategy-period fund returns are available.")
        selected_fund_row_obj_list = _select_window_fund_row_list(
            fund_row_obj_list, window_str
        )
        chart_point_dict_list = _build_chart_point_dict_list(selected_fund_row_obj_list)
        fund_chart_dict = build_equity_chart_dict(
            chart_point_dict_list,
            window_str="all",
            value_mode_str="pct",
        ).as_dict()
        fund_chart_dict["selected_window_str"] = window_str
        fund_chart_dict["display_from_date_str"] = (
            None
            if not selected_fund_row_obj_list
            else selected_fund_row_obj_list[0].market_date_str
        )
        fund_chart_dict["display_to_date_str"] = (
            None
            if not selected_fund_row_obj_list
            else selected_fund_row_obj_list[-1].market_date_str
        )
        cumulative_fund_return_float = _compound_return_float(
            [row_obj.fund_return_float for row_obj in fund_row_obj_list]
        )
        latest_fund_row_obj = fund_row_obj_list[-1]
        latest_market_date_str = latest_fund_row_obj.market_date_str
        latest_active_account_route_set = {
            binding_obj.account_route_str
            for binding_obj in _active_binding_list(
                binding_by_account_route_dict.values(), latest_market_date_str
            )
        }
        latest_covered_account_route_set = {
            row_obj.account_route_str
            for row_obj in row_obj_list
            if row_obj.market_date_str == latest_market_date_str
        }
        status_str, status_label_str, detail_str = _shadow_freshness_tuple(
            latest_market_date_str,
            as_of_ts or datetime.now(UTC),
        )
        if (
            latest_attempt_dict is not None
            and latest_attempt_dict["status_str"] == "failed"
        ):
            status_str = "error"
            status_label_str = "Last shadow sync failed"
            detail_str = str(
                latest_attempt_dict.get("detail_str")
                or "The latest IBKR Flex sync failed."
            )
        return {
            "status_str": status_str,
            "status_label_str": status_label_str,
            "detail_str": detail_str,
            "window_str": window_str,
            "latest_sync_timestamp_str": (
                latest_attempt_dict["attempted_timestamp_str"]
                if latest_attempt_dict is not None
                and latest_attempt_dict["status_str"] == "success"
                else (
                    None
                    if latest_import_dict is None
                    else latest_import_dict["imported_timestamp_str"]
                )
            ),
            "coverage_through_date_str": latest_market_date_str,
            "covered_account_count_int": len(
                latest_active_account_route_set & latest_covered_account_route_set
            ),
            "expected_account_count_int": len(latest_active_account_route_set),
            "query_name_str": (
                None if latest_import_dict is None else latest_import_dict["query_name_str"]
            ),
            "checksum_short_str": (
                None
                if latest_import_dict is None
                else str(latest_import_dict["checksum_str"])[:12]
            ),
            "latest_fund_return_float": latest_fund_row_obj.fund_return_float,
            "mtd_fund_return_float": _period_return_float(fund_row_obj_list, "mtd"),
            "ytd_fund_return_float": _period_return_float(fund_row_obj_list, "ytd"),
            "cumulative_fund_return_float": cumulative_fund_return_float,
            "latest_ending_nav_float": latest_fund_row_obj.ending_nav_float,
            "fund_chart_dict": fund_chart_dict,
            "pod_performance_dict_list": pod_performance_dict_list,
            "recent_fund_row_dict_list": [
                {
                    "market_date_str": row_obj.market_date_str,
                    "fund_return_float": row_obj.fund_return_float,
                    "ending_nav_float": row_obj.ending_nav_float,
                    "starting_nav_float": row_obj.starting_nav_float,
                    "adjusted_base_float": row_obj.adjusted_base_float,
                    "derived_base_change_float": (
                        row_obj.adjusted_base_float
                        - row_obj.starting_nav_float
                    ),
                    "active_pod_count_int": row_obj.active_pod_count_int,
                }
                for row_obj in fund_row_obj_list[-10:][::-1]
            ],
        }
    except (OSError, sqlite3.Error, PerformanceContractError) as exception_obj:
        return {
            "status_str": "error",
            "status_label_str": "Cannot verify shadow performance",
            "detail_str": str(exception_obj),
            "window_str": window_str,
            "pod_performance_dict_list": [],
            "fund_chart_dict": None,
        }


def status_json_str(db_path_str: str) -> str:
    page_dict = build_performance_page_dict(db_path_str)
    status_dict = {
        key_str: value_obj
        for key_str, value_obj in page_dict.items()
        if key_str
        in {
            "status_str",
            "status_label_str",
            "detail_str",
            "latest_sync_timestamp_str",
            "coverage_through_date_str",
            "covered_account_count_int",
            "expected_account_count_int",
            "query_name_str",
            "checksum_short_str",
        }
    }
    return json.dumps(status_dict, sort_keys=True)


def _parse_change_in_nav_row(
    change_in_nav_obj: ElementTree.Element,
    binding_obj: PodPerformanceBinding,
    statement_account_route_str: str,
) -> FlexDailyPerformance:
    account_route_str = str(change_in_nav_obj.attrib.get("accountId") or "").strip()
    currency_str = str(change_in_nav_obj.attrib.get("currency") or "").strip().upper()
    if account_route_str != statement_account_route_str:
        raise PerformanceContractError(
            f"Change in NAV account mismatch for {statement_account_route_str}."
        )
    if currency_str != SUPPORTED_CURRENCY_STR:
        raise PerformanceContractError(
            f"Change in NAV currency for {account_route_str} is "
            f"{currency_str or 'missing'}; expected USD."
        )
    from_date_str = _parse_flex_date_str(change_in_nav_obj.attrib.get("fromDate"))
    to_date_str = _parse_flex_date_str(change_in_nav_obj.attrib.get("toDate"))
    if from_date_str != to_date_str:
        raise PerformanceContractError(
            f"Change in NAV for {account_route_str} is not broken out by day: "
            f"{from_date_str} to {to_date_str}."
        )
    starting_nav_float = _required_finite_float(
        change_in_nav_obj.attrib, "startingValue", account_route_str, to_date_str
    )
    ending_nav_float = _required_finite_float(
        change_in_nav_obj.attrib, "endingValue", account_route_str, to_date_str
    )
    twr_pct_float = _required_finite_float(
        change_in_nav_obj.attrib, "twr", account_route_str, to_date_str
    )
    if starting_nav_float < 0.0 or ending_nav_float < 0.0:
        raise PerformanceContractError(
            f"Negative NAV is unsupported for {account_route_str} on {to_date_str}."
        )
    twr_float = twr_pct_float / 100.0
    if twr_float <= -1.0:
        raise PerformanceContractError(
            f"TWR is <= -100% for {account_route_str} on {to_date_str}."
        )
    return FlexDailyPerformance(
        account_route_str=account_route_str,
        pod_id_str=binding_obj.pod_id_str,
        market_date_str=to_date_str,
        starting_nav_float=starting_nav_float,
        ending_nav_float=ending_nav_float,
        twr_pct_float=twr_pct_float,
        twr_float=twr_float,
    )


def _parse_flex_date_str(raw_date_obj: object) -> str:
    raw_date_str = str(raw_date_obj or "").strip()
    try:
        parsed_date_obj = datetime.strptime(raw_date_str, "%Y%m%d").date()
    except ValueError as exception_obj:
        raise PerformanceContractError(
            f"Invalid Flex date {raw_date_str!r}; expected yyyyMMdd."
        ) from exception_obj
    return parsed_date_obj.isoformat()


def _required_finite_float(
    attribute_dict: dict[str, str],
    field_name_str: str,
    account_route_str: str,
    market_date_str: str,
) -> float:
    raw_value_obj = attribute_dict.get(field_name_str)
    try:
        value_float = float(raw_value_obj)
    except (TypeError, ValueError) as exception_obj:
        raise PerformanceContractError(
            f"Missing or invalid {field_name_str} for {account_route_str} "
            f"on {market_date_str}."
        ) from exception_obj
    if not math.isfinite(value_float):
        raise PerformanceContractError(
            f"Non-finite {field_name_str} for {account_route_str} on {market_date_str}."
        )
    return value_float


def _validate_binding_dict(
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
) -> None:
    pod_id_set: set[str] = set()
    for account_route_str, binding_obj in binding_by_account_route_dict.items():
        if account_route_str != binding_obj.account_route_str:
            raise PerformanceContractError(
                f"Binding key {account_route_str} does not match its account route."
            )
        if binding_obj.session_calendar_id_str != SUPPORTED_CALENDAR_ID_STR:
            raise PerformanceContractError(
                f"Pod {binding_obj.pod_id_str} uses {binding_obj.session_calendar_id_str}; "
                "v1 performance supports XNYS only."
            )
        if binding_obj.pod_id_str in pod_id_set:
            raise PerformanceContractError(
                f"Pod {binding_obj.pod_id_str} is bound to more than one LIVE account."
            )
        pod_id_set.add(binding_obj.pod_id_str)


def _required_account_route_set(
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
    *,
    request_from_date_str: str | None,
    request_to_date_str: str | None,
) -> set[str]:
    if request_from_date_str is None or request_to_date_str is None:
        return set(binding_by_account_route_dict)
    return {
        binding_obj.account_route_str
        for binding_obj in binding_by_account_route_dict.values()
        if binding_obj.return_start_date_str is not None
        and binding_obj.return_start_date_str <= request_to_date_str
        and (
            binding_obj.return_end_date_str is None
            or binding_obj.return_end_date_str >= request_from_date_str
        )
    }


def _validate_ignorable_non_session_row(row_obj: FlexDailyPerformance) -> None:
    unchanged_nav_bool = math.isclose(
        row_obj.starting_nav_float,
        row_obj.ending_nav_float,
        rel_tol=0.0,
        abs_tol=0.01,
    )
    zero_return_bool = math.isclose(
        row_obj.twr_float, 0.0, rel_tol=0.0, abs_tol=1e-12
    )
    if not unchanged_nav_bool or not zero_return_bool:
        raise PerformanceContractError(
            f"IBKR reported non-zero performance on non-session date "
            f"{row_obj.market_date_str} for {row_obj.account_route_str}."
        )


def _validate_active_account_coverage(
    row_obj_list: Iterable[FlexDailyPerformance],
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
) -> None:
    row_key_set = {
        (row_obj.market_date_str, row_obj.account_route_str)
        for row_obj in row_obj_list
    }
    market_date_str_set = {market_date_str for market_date_str, _ in row_key_set}
    for market_date_str in sorted(market_date_str_set):
        expected_account_route_set = {
            binding_obj.account_route_str
            for binding_obj in _active_binding_list(
                binding_by_account_route_dict.values(), market_date_str
            )
        }
        actual_account_route_set = {
            account_route_str
            for row_market_date_str, account_route_str in row_key_set
            if row_market_date_str == market_date_str
            and account_route_str in expected_account_route_set
        }
        if actual_account_route_set != expected_account_route_set:
            missing_account_route_set = (
                expected_account_route_set - actual_account_route_set
            )
            raise PerformanceContractError(
                f"Flex coverage is incomplete on {market_date_str}; missing account(s): "
                + ", ".join(sorted(missing_account_route_set))
                + "."
            )


def _validate_requested_session_coverage(
    row_obj_list: Iterable[FlexDailyPerformance],
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
    request_from_date_str: str,
    request_to_date_str: str,
) -> None:
    if request_from_date_str > request_to_date_str:
        raise PerformanceContractError("Requested Flex date range is reversed.")
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(
        SUPPORTED_CALENDAR_ID_STR
    )
    requested_session_label_list = calendar_obj.sessions_in_range(
        pd.Timestamp(request_from_date_str), pd.Timestamp(request_to_date_str)
    )
    row_key_set = {
        (row_obj.market_date_str, row_obj.account_route_str)
        for row_obj in row_obj_list
    }
    for session_label_ts in requested_session_label_list:
        market_date_str = str(session_label_ts.date())
        for binding_obj in _active_binding_list(
            binding_by_account_route_dict.values(), market_date_str
        ):
            row_key_tuple = (market_date_str, binding_obj.account_route_str)
            if row_key_tuple not in row_key_set:
                raise PerformanceContractError(
                    f"Flex omitted {binding_obj.account_route_str} on requested "
                    f"XNYS session {market_date_str}."
                )


def _active_binding_list(
    binding_obj_iterable: Iterable[PodPerformanceBinding],
    market_date_str: str,
) -> list[PodPerformanceBinding]:
    return sorted(
        [
            binding_obj
            for binding_obj in binding_obj_iterable
            if _binding_is_active_bool(binding_obj, market_date_str)
        ],
        key=lambda binding_obj: binding_obj.pod_id_str,
    )


def _binding_is_active_bool(
    binding_obj: PodPerformanceBinding,
    market_date_str: str,
) -> bool:
    if binding_obj.return_start_date_str is None:
        return False
    if market_date_str < binding_obj.return_start_date_str:
        return False
    return (
        binding_obj.return_end_date_str is None
        or market_date_str <= binding_obj.return_end_date_str
    )


def _is_session_date_bool(market_date_str: str) -> bool:
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(
        SUPPORTED_CALENDAR_ID_STR
    )
    return bool(calendar_obj.is_session(pd.Timestamp(market_date_str)))


def _select_window_fund_row_list(
    fund_row_obj_list: list[FundDailyPerformance], window_str: str
) -> list[FundDailyPerformance]:
    if not fund_row_obj_list or window_str == "all":
        return fund_row_obj_list
    if window_str == "1w":
        return fund_row_obj_list[-5:]
    latest_date_obj = date.fromisoformat(fund_row_obj_list[-1].market_date_str)
    if window_str == "mtd":
        return [
            row_obj
            for row_obj in fund_row_obj_list
            if date.fromisoformat(row_obj.market_date_str).replace(day=1)
            == latest_date_obj.replace(day=1)
        ]
    return [
        row_obj
        for row_obj in fund_row_obj_list
        if date.fromisoformat(row_obj.market_date_str).year == latest_date_obj.year
    ]


def _build_chart_point_dict_list(
    fund_row_obj_list: list[FundDailyPerformance],
) -> list[dict[str, Any]]:
    if not fund_row_obj_list:
        return []
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(
        SUPPORTED_CALENDAR_ID_STR
    )
    first_market_date_ts = pd.Timestamp(fund_row_obj_list[0].market_date_str)
    baseline_date_str = str(calendar_obj.previous_session(first_market_date_ts).date())
    chart_point_dict_list: list[dict[str, Any]] = [
        {
            "market_date_str": baseline_date_str,
            "equity_float": 100.0,
            "daily_pnl_float": None,
            "daily_pnl_pct_float": None,
        }
    ]
    index_value_float = 100.0
    for fund_row_obj in fund_row_obj_list:
        previous_index_value_float = index_value_float
        index_value_float *= 1.0 + fund_row_obj.fund_return_float
        chart_point_dict_list.append(
            {
                "market_date_str": fund_row_obj.market_date_str,
                "equity_float": index_value_float,
                "daily_pnl_float": index_value_float - previous_index_value_float,
                "daily_pnl_pct_float": fund_row_obj.fund_return_float,
            }
        )
    return chart_point_dict_list


def _build_pod_performance_dict_list(
    row_obj_list: list[FlexDailyPerformance],
    binding_by_account_route_dict: dict[str, PodPerformanceBinding],
) -> list[dict[str, Any]]:
    result_dict_list: list[dict[str, Any]] = []
    for binding_obj in sorted(
        binding_by_account_route_dict.values(),
        key=lambda item_obj: item_obj.pod_id_str,
    ):
        pod_row_obj_list = [
            row_obj
            for row_obj in row_obj_list
            if row_obj.account_route_str == binding_obj.account_route_str
            and _binding_is_active_bool(binding_obj, row_obj.market_date_str)
        ]
        if not pod_row_obj_list:
            result_dict_list.append(
                {
                    "pod_id_str": binding_obj.pod_id_str,
                    "account_route_str": binding_obj.account_route_str,
                    "enabled_bool": binding_obj.enabled_bool,
                    "status_str": "pending_baseline",
                }
            )
            continue
        pod_row_obj_list.sort(key=lambda row_obj: row_obj.market_date_str)
        result_dict_list.append(
            {
                "pod_id_str": binding_obj.pod_id_str,
                "account_route_str": binding_obj.account_route_str,
                "enabled_bool": binding_obj.enabled_bool,
                "status_str": "available",
                "latest_market_date_str": pod_row_obj_list[-1].market_date_str,
                "latest_return_float": pod_row_obj_list[-1].twr_float,
                "mtd_return_float": _compound_return_float(
                    [
                        row_obj.twr_float
                        for row_obj in pod_row_obj_list
                        if row_obj.market_date_str[:7]
                        == pod_row_obj_list[-1].market_date_str[:7]
                    ]
                ),
                "cumulative_return_float": _compound_return_float(
                    [row_obj.twr_float for row_obj in pod_row_obj_list]
                ),
                "latest_ending_nav_float": pod_row_obj_list[-1].ending_nav_float,
            }
        )
    return result_dict_list


def _period_return_float(
    fund_row_obj_list: list[FundDailyPerformance], period_str: str
) -> float | None:
    selected_row_obj_list = _select_window_fund_row_list(
        fund_row_obj_list, period_str
    )
    if not selected_row_obj_list:
        return None
    return _compound_return_float(
        [row_obj.fund_return_float for row_obj in selected_row_obj_list]
    )


def _compound_return_float(return_float_list: list[float]) -> float:
    growth_float = 1.0
    for return_float in return_float_list:
        growth_float *= 1.0 + return_float
    return growth_float - 1.0


def _shadow_freshness_tuple(
    coverage_through_date_str: str,
    as_of_ts: datetime,
) -> tuple[str, str, str]:
    if as_of_ts.tzinfo is None:
        raise ValueError("as_of_ts must be timezone-aware.")
    new_york_now_ts = as_of_ts.astimezone(ZoneInfo("America/New_York"))
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(
        SUPPORTED_CALENDAR_ID_STR
    )
    local_date_ts = pd.Timestamp(new_york_now_ts.date())
    if calendar_obj.is_session(local_date_ts):
        required_session_date_str = str(
            calendar_obj.previous_session(local_date_ts).date()
        )
    else:
        required_session_date_str = str(
            calendar_obj.date_to_session(local_date_ts, direction="previous").date()
        )
    if coverage_through_date_str >= required_session_date_str:
        return (
            "available",
            "Shadow data available",
            "IBKR account TWR is authoritative per Pod; the combined line is an indicative Shadow diagnostic, not Fund TWR.",
        )
    if (new_york_now_ts.hour, new_york_now_ts.minute) < (8, 0):
        return (
            "pending",
            "Performance sync pending",
            f"Finalized through {coverage_through_date_str}; waiting for IBKR Flex coverage of {required_session_date_str}.",
        )
    return (
        "stale",
        "Shadow sync overdue",
        f"IBKR Flex is finalized through {coverage_through_date_str}, but {required_session_date_str} is required after 08:00 ET.",
    )
