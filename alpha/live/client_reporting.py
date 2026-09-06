"""Client financial reporting, independent of scheduler/release membership.

Only saved broker facts enter this projection. It never imports the runner,
opens a broker connection, updates a ledger or promotes the Shadow composite.
One configured account period is one independently valued strategy sleeve.
Dates include the whole broker reporting day: beginning-of-day through EOD.
"""

from __future__ import annotations

from contextlib import closing
from calendar import monthrange
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock
from typing import Any
import xml.etree.ElementTree as ElementTree

from alpha.live.scheduler_utils import get_exchange_calendar_obj
from alpha.live.client_benchmark import account_benchmark_dict, validate_benchmark_config


METHOD_VERSION_STR = "client_nav_bridge_v3"
CLIENT_TWR_METHOD_STR = "daily_nav_eod_v1"
BRIDGE_TOLERANCE_DECIMAL = Decimal("0.01")
CAPITAL_FIELD_TUPLE = (
    "depositsWithdrawals", "internalCashTransfers", "assetTransfers",
    "debitCardActivity", "billPay",
)
BOUNDARY_FIELD_STR = "linkingAdjustments"
NAV_METADATA_FIELD_SET = {
    "accountId", "acctAlias", "accountAlias", "model", "currency", "fromDate",
    "toDate", "startingValue", "endingValue", "twr",
}
_NAV_CACHE_LIMIT_INT = 128
_NAV_CACHE_XML_LIMIT_INT = 256_000
_nav_cache_dict = OrderedDict()
_nav_cache_lock = Lock()


class ClientReportingError(ValueError):
    """Saved evidence or reporting configuration cannot support this result."""


@dataclass(frozen=True)
class BrokerNavRow:
    account_route_str: str
    market_date_str: str
    opening_nav_decimal: Decimal
    closing_nav_decimal: Decimal
    twr_decimal: Decimal
    attribute_dict: dict[str, str]
    source_import_id_int: int
    source_checksum_str: str


@dataclass(frozen=True)
class BrokerReportingSnapshot:
    row_tuple: tuple[BrokerNavRow, ...] = ()
    import_tuple: tuple[dict[str, Any], ...] = ()
    latest_attempt_dict: dict[str, Any] | None = None
    unavailable_reason_str: str | None = None


def _date_str(raw_value_obj: object) -> str:
    value_str = str(raw_value_obj or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value_str):
        raise ClientReportingError("Reporting dates must use YYYY-MM-DD.")
    try:
        return date.fromisoformat(value_str).isoformat()
    except ValueError as exception_obj:
        raise ClientReportingError("Invalid reporting date.") from exception_obj


def _decimal_value(attribute_dict: dict, field_str: str) -> Decimal:
    try:
        value_decimal = Decimal(str(attribute_dict[field_str]))
    except (KeyError, InvalidOperation, ValueError) as exception_obj:
        raise ClientReportingError(f"Missing or invalid {field_str}.") from exception_obj
    if not value_decimal.is_finite():
        raise ClientReportingError(f"Non-finite {field_str}.")
    return value_decimal


def content_hash_str(payload_obj: object) -> str:
    canonical_str = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def account_performance_dict(row_list, *, complete_bool, from_date_str, to_date_str, session_date_set):
    """Official account return index and selected-period risk, never NAV returns.

    I0=1; It=I(t-1)*(1+rt); peak_t=max(1,I1..It); DD_t=It/peak_t-1.
    Every dated broker return, including reported non-session activity, enters
    once. No annualization, missing-return fill, sample tuning or fund composite.
    """
    result_dict = {"return_path_list": [], "monthly_return_list": [], "max_drawdown_float": None,
                   "current_drawdown_float": None, "exchange_session_count_int": sum(row_obj.market_date_str in session_date_set for row_obj in row_list),
                   "non_session_count_int": sum(row_obj.market_date_str not in session_date_set for row_obj in row_list)}
    if not complete_bool or not row_list:
        return result_dict
    index_decimal = peak_decimal = Decimal(1)
    max_drawdown_decimal = Decimal(0)
    month_growth_dict = {}
    result_dict["return_path_list"].append({"market_date_str": from_date_str + " SOD", "cumulative_return_float": 0.0, "drawdown_float": 0.0})
    # *** CRITICAL*** Retrospective, selected-period official account TWR only.
    # The running peak includes the opening 1.0 baseline, so a first-day loss
    # is a drawdown. This output never enters a signal or execution decision.
    for row_obj in row_list:
        index_decimal *= 1 + row_obj.twr_decimal
        peak_decimal = max(peak_decimal, index_decimal)
        drawdown_decimal = index_decimal / peak_decimal - 1
        max_drawdown_decimal = min(max_drawdown_decimal, drawdown_decimal)
        result_dict["return_path_list"].append({"market_date_str": row_obj.market_date_str, "cumulative_return_float": float(index_decimal - 1), "drawdown_float": float(drawdown_decimal)})
        month_str = row_obj.market_date_str[:7]
        month_growth_dict[month_str] = month_growth_dict.get(month_str, Decimal(1)) * (1 + row_obj.twr_decimal)
    result_dict["max_drawdown_float"] = float(max_drawdown_decimal)
    result_dict["current_drawdown_float"] = float(drawdown_decimal)
    for month_str, growth_decimal in month_growth_dict.items():
        year_int, month_int = (int(part_str) for part_str in month_str.split("-"))
        last_day_str = f"{month_str}-{monthrange(year_int, month_int)[1]:02d}"
        result_dict["monthly_return_list"].append({"month_str": month_str, "return_float": float(growth_decimal - 1), "partial_bool": from_date_str > month_str + "-01" or to_date_str < last_day_str})
    return result_dict


def validate_client_registry_dict(registry_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate explicit, effective-dated financial ownership; never infer it.

    Operational enabled flags are intentionally absent. The registry contains
    retired account periods too. Overlapping account ownership is unsupported.
    A reviewed NAV bridge profile is optional; without it dollar P&L is unknown.
    """
    if not isinstance(registry_dict, dict) or type(registry_dict.get("schema_version")) is not int or registry_dict["schema_version"] != 1:
        raise ClientReportingError("Client registry schema_version must be 1.")
    client_list = registry_dict.get("clients")
    if not isinstance(client_list, list) or not client_list:
        raise ClientReportingError("Configure at least one client.")
    client_id_set: set[str] = set()
    ownership_dict: dict[str, list[tuple[str, str, str]]] = {}
    for client_dict in client_list:
        if not isinstance(client_dict, dict):
            raise ClientReportingError("Every client must be a JSON object.")
        for field_str in ("client_id", "display_name", "fee_basis", "query_name"):
            if not isinstance(client_dict.get(field_str), str) or not client_dict[field_str].strip():
                raise ClientReportingError(f"Client {field_str} is required.")
        client_id_str = client_dict["client_id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", client_id_str) or client_id_str in client_id_set:
            raise ClientReportingError("Client IDs must be unique URL-safe identifiers.")
        client_id_set.add(client_id_str)
        if client_dict.get("base_currency") != "USD":
            raise ClientReportingError("Client reporting currently supports USD only; no implicit FX conversion.")
        mandate_start_str = _date_str(client_dict.get("mandate_start_date"))
        try:
            validate_benchmark_config(client_dict.get("benchmark"))
        except ValueError as exception_obj:
            raise ClientReportingError(str(exception_obj)) from exception_obj
        operations_source_str = client_dict.get("operations_source", "unconfigured")
        if not isinstance(operations_source_str, str) or operations_source_str not in {"unconfigured", "local", "snapshot"}:
            raise ClientReportingError("Operations source must be local, snapshot or unconfigured.")
        if operations_source_str == "snapshot" and (not isinstance(client_dict.get("operations_snapshot_path"), str) or not client_dict["operations_snapshot_path"].strip()):
            raise ClientReportingError("Snapshot operations require a server-configured operations_snapshot_path.")
        account_list = client_dict.get("accounts")
        if not isinstance(account_list, list) or not account_list:
            raise ClientReportingError("Configure at least one account period per client.")
        pod_period_dict: dict[str, list[tuple[str, str, str]]] = {}
        for account_dict in account_list:
            if not isinstance(account_dict, dict):
                raise ClientReportingError("Every account period must be a JSON object.")
            for field_str in ("account_route", "pod_id", "display_name"):
                if not isinstance(account_dict.get(field_str), str) or not account_dict[field_str].strip():
                    raise ClientReportingError(f"Account {field_str} is required.")
            if "reference_summary_path" in account_dict and (not isinstance(account_dict["reference_summary_path"], str) or not account_dict["reference_summary_path"].strip()):
                raise ClientReportingError("reference_summary_path must be an explicit server-configured saved JSON path.")
            from_str = _date_str(account_dict.get("effective_from"))
            to_str = _date_str(account_dict["effective_to"]) if account_dict.get("effective_to") is not None else "9999-12-31"
            if from_str < mandate_start_str or to_str < from_str:
                raise ClientReportingError("Account period is reversed or precedes the client mandate.")
            period_tuple = (from_str, to_str, client_id_str)
            ownership_dict.setdefault(account_dict["account_route"], []).append(period_tuple)
            pod_period_dict.setdefault(account_dict["pod_id"], []).append(period_tuple)
        _reject_overlap(pod_period_dict, "strategy")
        bridge_dict = client_dict.get("nav_bridge")
        if bridge_dict is not None:
            _validate_bridge_profile(bridge_dict)
        twr_dict = client_dict.get("client_twr")
        if twr_dict is not None:
            if not isinstance(twr_dict, dict) or twr_dict.get("method") != CLIENT_TWR_METHOD_STR:
                raise ClientReportingError("Client TWR requires the explicit daily_nav_eod_v1 method.")
            if bridge_dict is None:
                raise ClientReportingError("Client TWR requires a reviewed NAV bridge.")
            for field_str in ("reviewed_by", "evidence_ref"):
                if not isinstance(twr_dict.get(field_str), str) or not twr_dict[field_str].strip():
                    raise ClientReportingError(f"Client TWR requires {field_str}.")
    _reject_overlap(ownership_dict, "account")
    # Return a detached copy, preventing a caller's edits from rewriting a report.
    return json.loads(json.dumps(registry_dict, allow_nan=False))


def _reject_overlap(period_dict: dict[str, list[tuple[str, str, str]]], label_str: str) -> None:
    for period_list in period_dict.values():
        ordered_list = sorted(period_list)
        for previous_tuple, current_tuple in zip(ordered_list, ordered_list[1:]):
            if current_tuple[0] <= previous_tuple[1]:
                raise ClientReportingError(f"Overlapping {label_str} ownership is unsupported.")


def _validate_bridge_profile(bridge_dict: dict) -> None:
    if not isinstance(bridge_dict, dict):
        raise ClientReportingError("NAV bridge must be a JSON object.")
    for field_str in ("profile_id", "evidence_ref", "reviewed_by"):
        if not isinstance(bridge_dict.get(field_str), str) or not bridge_dict[field_str].strip():
            raise ClientReportingError(f"Reviewed NAV bridge requires {field_str}.")
    if bridge_dict.get("mode") != "MTM" or bridge_dict.get("nonoverlap_confirmed") is not True:
        raise ClientReportingError("NAV bridge needs a reviewed non-overlapping MTM field contract.")
    economic_list = bridge_dict.get("economic_fields")
    informational_list = bridge_dict.get("informational_fields", [])
    if not isinstance(economic_list, list) or not economic_list or not isinstance(informational_list, list):
        raise ClientReportingError("NAV bridge needs independent economic_fields.")
    field_list = economic_list + informational_list
    if any(not isinstance(field_str, str) or not field_str for field_str in field_list):
        raise ClientReportingError("NAV bridge field names must be nonempty strings.")
    reserved_set = set(CAPITAL_FIELD_TUPLE) | {BOUNDARY_FIELD_STR} | NAV_METADATA_FIELD_SET
    if len(set(field_list)) != len(field_list) or set(field_list) & reserved_set:
        raise ClientReportingError("NAV bridge fields overlap or include capital/NAV totals.")
    if {"realized", "changeInUnrealized"} & set(field_list):
        raise ClientReportingError("Do not mix realized/unrealized fields with an MTM bridge.")


def load_client_registry_dict(config_path_str: str) -> dict[str, Any]:
    try:
        registry_dict = json.loads(Path(config_path_str).read_text(encoding="utf-8-sig"))
        if not isinstance(registry_dict, dict):
            raise ClientReportingError("Client registry must be a JSON object.")
        return validate_client_registry_dict(registry_dict)
    except (OSError, json.JSONDecodeError) as exception_obj:
        raise ClientReportingError("Client reporting configuration is missing or unreadable.") from exception_obj


def parse_broker_nav_import(
    raw_xml_str: str, *, allowed_account_set: set[str], query_name_str: str,
    source_import_id_int: int, source_checksum_str: str,
) -> list[BrokerNavRow]:
    """Retain all source dates, including non-session cash/interest rows.

    This is account-level EOD reporting, not a signal or execution price join.
    Unrelated accounts are excluded before reading their financial fields.
    """
    if "<!DOCTYPE" in raw_xml_str.upper() or "<!ENTITY" in raw_xml_str.upper():
        raise ClientReportingError("DTD/entity declarations are not accepted in saved Flex XML.")
    if hashlib.sha256(raw_xml_str.encode("utf-8")).hexdigest() != source_checksum_str:
        raise ClientReportingError("Saved broker XML checksum mismatch.")
    # Recheck current bytes and scope on every request. SQL revisions, omitted
    # rows, sync attempts and report finality are deliberately NOT cached.
    cache_key_tuple = (tuple(sorted(allowed_account_set)), query_name_str, source_import_id_int, source_checksum_str)
    with _nav_cache_lock:
        cached_list = _nav_cache_dict.get(cache_key_tuple)
        if cached_list is not None:
            _nav_cache_dict.move_to_end(cache_key_tuple)
    if cached_list is not None:
        return deepcopy(cached_list)
    try:
        root_obj = ElementTree.fromstring(raw_xml_str)
    except ElementTree.ParseError as exception_obj:
        raise ClientReportingError("Saved Flex XML is malformed.") from exception_obj
    if root_obj.tag != "FlexQueryResponse" or root_obj.get("queryName") != query_name_str:
        raise ClientReportingError("Saved Flex response/query identity mismatch.")
    row_list: list[BrokerNavRow] = []
    seen_key_set: set[tuple[str, str]] = set()
    for statement_obj in root_obj.findall(".//FlexStatement"):
        account_str = str(statement_obj.get("accountId") or "")
        if account_str not in allowed_account_set:
            continue
        info_obj = statement_obj.find("AccountInformation")
        if info_obj is None or info_obj.get("accountId") != account_str or info_obj.get("currency") != "USD":
            raise ClientReportingError("Broker account identity/base currency is not confirmed.")
        for nav_obj in statement_obj.findall("ChangeInNAV"):
            attribute_dict = dict(nav_obj.attrib)
            if nav_obj.get("accountId") != account_str or nav_obj.get("currency") != "USD" or nav_obj.get("model"):
                raise ClientReportingError("Expected whole-account USD NAV, not a model/sub-sleeve.")
            try:
                from_str = datetime.strptime(str(nav_obj.get("fromDate")), "%Y%m%d").date().isoformat()
                to_str = datetime.strptime(str(nav_obj.get("toDate")), "%Y%m%d").date().isoformat()
            except ValueError as exception_obj:
                raise ClientReportingError("Invalid broker NAV date.") from exception_obj
            if from_str != to_str:
                raise ClientReportingError("Broker NAV must be broken out by day.")
            row_key_tuple = (account_str, to_str)
            if row_key_tuple in seen_key_set:
                raise ClientReportingError("Duplicate account/day in broker source.")
            seen_key_set.add(row_key_tuple)
            opening_decimal = _decimal_value(attribute_dict, "startingValue")
            closing_decimal = _decimal_value(attribute_dict, "endingValue")
            twr_decimal = _decimal_value(attribute_dict, "twr") / Decimal(100)
            if min(opening_decimal, closing_decimal) < 0 or twr_decimal <= -1:
                raise ClientReportingError("Negative NAV or TWR <= -100% is unsupported.")
            row_list.append(BrokerNavRow(
                account_str, to_str, opening_decimal, closing_decimal, twr_decimal,
                attribute_dict, source_import_id_int, source_checksum_str,
            ))
    # Bound retained XML-derived objects; oversized sources remain valid but
    # are decoded normally. Detach mutable attribute dictionaries at both ends.
    if len(raw_xml_str) <= _NAV_CACHE_XML_LIMIT_INT:
        cached_list = deepcopy(row_list)
        with _nav_cache_lock:
            _nav_cache_dict[cache_key_tuple] = cached_list
            _nav_cache_dict.move_to_end(cache_key_tuple)
            while len(_nav_cache_dict) > _NAV_CACHE_LIMIT_INT:
                _nav_cache_dict.popitem(last=False)
    return row_list


def load_broker_reporting_snapshot(
    db_path_str: str, *, allowed_account_set: set[str], query_name_str: str,
) -> BrokerReportingSnapshot:
    """Read one SQLite snapshot, honoring range replacement and source revisions.

    No migrations/initialization, no immutable=1 (the writer may use WAL).
    A newer import removes older facts in its declared range even when the new
    report omits that account/day. Omission must never resurrect stale history.
    """
    database_path_obj = Path(db_path_str).resolve()
    if not database_path_obj.is_file():
        return BrokerReportingSnapshot(unavailable_reason_str="No saved broker performance database.")
    try:
        with closing(sqlite3.connect(database_path_obj.as_uri() + "?mode=ro", uri=True)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("BEGIN")
            import_list = [dict(row_obj) for row_obj in connection_obj.execute(
                "SELECT * FROM flex_import WHERE query_name_str = ? ORDER BY import_id_int",
                (query_name_str,),
            ).fetchall()]
            attempt_obj = connection_obj.execute(
                "SELECT * FROM sync_attempt ORDER BY attempt_id_int DESC LIMIT 1"
            ).fetchone()
            latest_attempt_dict = dict(attempt_obj) if attempt_obj is not None else None
    except sqlite3.Error:
        return BrokerReportingSnapshot(unavailable_reason_str="Saved performance database is unavailable or has no reporting schema.")
    row_by_key_dict: dict[tuple[str, str], BrokerNavRow] = {}
    source_list: list[dict[str, Any]] = []
    for import_dict in import_list:
        from_str = _date_str(import_dict["request_from_date_str"])
        to_str = _date_str(import_dict["request_to_date_str"])
        if from_str > to_str:
            raise ClientReportingError("Saved broker import has a reversed date range.")
        imported_row_list = parse_broker_nav_import(
            import_dict["raw_xml_str"], allowed_account_set=allowed_account_set,
            query_name_str=query_name_str, source_import_id_int=import_dict["import_id_int"],
            source_checksum_str=import_dict["checksum_str"],
        )
        if any(not from_str <= row_obj.market_date_str <= to_str for row_obj in imported_row_list):
            raise ClientReportingError("Broker row lies outside its saved request range.")
        row_by_key_dict = {
            key_tuple: row_obj for key_tuple, row_obj in row_by_key_dict.items()
            if not from_str <= key_tuple[1] <= to_str
        }
        row_by_key_dict.update({(row_obj.account_route_str, row_obj.market_date_str): row_obj for row_obj in imported_row_list})
        source_list.append({key_str: value_obj for key_str, value_obj in import_dict.items() if key_str != "raw_xml_str"})
    return BrokerReportingSnapshot(
        tuple(sorted(row_by_key_dict.values(), key=lambda row_obj: (row_obj.market_date_str, row_obj.account_route_str))),
        tuple(source_list), latest_attempt_dict,
        None if import_list else "No saved imports match the configured broker query.",
    )


def _daily_bridge_dict(row_obj: BrokerNavRow, bridge_dict: dict | None) -> dict[str, Any]:
    if bridge_dict is None:
        return {"complete_bool": False, "reason_str": "A reviewed complete NAV-flow contract is not configured."}
    try:
        capital_decimal = sum((_decimal_value(row_obj.attribute_dict, field_str) for field_str in CAPITAL_FIELD_TUPLE), Decimal(0))
        boundary_decimal = _decimal_value(row_obj.attribute_dict, BOUNDARY_FIELD_STR)
        economic_decimal = sum((_decimal_value(row_obj.attribute_dict, field_str) for field_str in bridge_dict["economic_fields"]), Decimal(0))
        covered_set = NAV_METADATA_FIELD_SET | set(CAPITAL_FIELD_TUPLE) | {BOUNDARY_FIELD_STR} | set(bridge_dict["economic_fields"]) | set(bridge_dict.get("informational_fields", []))
        for field_str in row_obj.attribute_dict.keys() - covered_set:
            if _decimal_value(row_obj.attribute_dict, field_str) != 0:
                raise ClientReportingError(f"Unclassified nonzero NAV component: {field_str}.")
        # Formula is independent: source economics must reconcile, not a P&L
        # residual used to 'prove' its own equality. Absolute USD 0.01 tolerance.
        residual_decimal = row_obj.closing_nav_decimal - row_obj.opening_nav_decimal - capital_decimal - boundary_decimal - economic_decimal
        if abs(residual_decimal) > BRIDGE_TOLERANCE_DECIMAL:
            raise ClientReportingError("Independent NAV bridge does not reconcile within USD 0.01.")
    except ClientReportingError as exception_obj:
        return {"complete_bool": False, "reason_str": str(exception_obj)}
    return {
        "complete_bool": True, "capital_decimal": capital_decimal,
        "boundary_decimal": boundary_decimal,
        "pnl_decimal": row_obj.closing_nav_decimal - row_obj.opening_nav_decimal - capital_decimal - boundary_decimal,
        "residual_decimal": residual_decimal,
    }


def _account_active_bool(account_dict: dict, market_date_str: str) -> bool:
    return account_dict["effective_from"] <= market_date_str <= (account_dict.get("effective_to") or "9999-12-31")


def _money_float(value_decimal: Decimal | None) -> float | None:
    return None if value_decimal is None else float(value_decimal)


def _client_daily_twr_dict(day_list, *, complete_bool, from_date_str):
    """Daily EOD convention, not official broker or exact intraday TWR.

    r_D = independently bridged P&L_D / active-account SOD NAV_D.
    TWR = product(1+r_D)-1. Scope entries are SOD; flows are EOD.
    See docs/live/CLIENT_TWR.md. Any failed day withholds the whole path.
    """
    unavailable_dict = {"twr_float": None, "return_path_list": [], "twr_daily_list": []}
    if not complete_bool or not day_list:
        return {**unavailable_dict, "twr_reason_str": "Complete finalized NAV and capital data are required."}
    growth_decimal = Decimal(1)
    path_list = [{"market_date_str": from_date_str + " SOD", "cumulative_return_float": 0.0}]
    daily_list = []
    # *** CRITICAL *** Retrospective reporting only. The denominator includes
    # entrants at SOD, excludes exited accounts, and never uses future NAV.
    # Internal cash in transit must not silently reduce invested capital.
    for date_str, opening_decimal, pnl_decimal, boundary_decimal, internal_decimal in day_list:
        reason_str = None
        if boundary_decimal != 0:
            reason_str = "Broker linking adjustment timing is unresolved."
        elif internal_decimal != 0:
            reason_str = "Internal transfer or cash-in-transit evidence is unresolved."
        elif opening_decimal <= 0:
            reason_str = "Positive opening client capital is required on every reporting day."
        if reason_str:
            return {**unavailable_dict, "twr_reason_str": f"{date_str}: {reason_str}"}
        return_decimal = pnl_decimal / opening_decimal
        if return_decimal <= -1:
            return {**unavailable_dict, "twr_reason_str": f"{date_str}: Total-loss or negative return base is unsupported."}
        growth_decimal *= 1 + return_decimal
        daily_list.append({"market_date_str": date_str, "opening_nav_float": float(opening_decimal),
                           "pnl_float": float(pnl_decimal), "return_float": float(return_decimal)})
        path_list.append({"market_date_str": date_str, "cumulative_return_float": float(growth_decimal - 1)})
    return {"twr_float": float(growth_decimal - 1), "return_path_list": path_list,
            "twr_daily_list": daily_list, "twr_reason_str": None}


def build_client_report_dict(
    client_dict: dict[str, Any], snapshot_obj: BrokerReportingSnapshot, *,
    from_date_str: str, to_date_str: str, as_of_ts: datetime, benchmark_snapshot_obj=None,
) -> dict[str, Any]:
    """One selected-period projection for screens and immutable report exports.

    Dollar bridge:
        closing NAV = opening NAV + source capital + source linking adjustments
                      + mandate entry/exit capital + investment P&L.
    Entry capital is the entrant's SOD NAV; exit capital removes its last EOD
    NAV on the first day outside the scope. Neither is investment profit.

    Account TWR = product(1 + official daily account TWR) - 1.
    No account-TWR average or adjusted-base composite becomes client TWR.
    The date alignment below is an EOD valuation join, never a signal input.
    """
    client_dict = validate_client_registry_dict({"schema_version": 1, "clients": [client_dict]})["clients"][0]
    from_date_str, to_date_str = _date_str(from_date_str), _date_str(to_date_str)
    if from_date_str > to_date_str or as_of_ts.tzinfo is None:
        raise ClientReportingError("Select an ordered date range and timezone-aware report timestamp.")
    from zoneinfo import ZoneInfo

    market_today_str = as_of_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    if to_date_str > market_today_str:
        raise ClientReportingError("Financial reporting cannot include future market dates.")
    if from_date_str < client_dict["mandate_start_date"]:
        raise ClientReportingError("Selected start precedes the configured client mandate.")
    calendar_obj = get_exchange_calendar_obj("XNYS")
    try:
        session_list = [session_obj.date().isoformat() for session_obj in calendar_obj.sessions_in_range(from_date_str, to_date_str)]
    except ValueError as exception_obj:
        raise ClientReportingError("Reporting range is outside the supported exchange calendar.") from exception_obj
    account_list = client_dict["accounts"]
    owned_account_set = {account_dict["account_route"] for account_dict in account_list}
    row_by_key_dict: dict[tuple[str, str], BrokerNavRow] = {}
    for row_obj in snapshot_obj.row_tuple:
        if row_obj.account_route_str not in owned_account_set:
            continue
        row_key_tuple = (row_obj.account_route_str, row_obj.market_date_str)
        if row_key_tuple in row_by_key_dict:
            raise ClientReportingError("Duplicate account/day in reporting snapshot.")
        row_by_key_dict[row_key_tuple] = row_obj
    # *** CRITICAL *** EOD reporting only: retain observed non-session activity.
    # Never forward-fill missing NAV or assume a missing flow field is zero.
    extra_date_set = {
        row_obj.market_date_str for row_obj in row_by_key_dict.values()
        if from_date_str <= row_obj.market_date_str <= to_date_str
        and any(account_dict["account_route"] == row_obj.account_route_str and _account_active_bool(account_dict, row_obj.market_date_str) for account_dict in account_list)
    }
    membership_date_set: set[str] = set()
    for account_dict in account_list:
        boundary_list = [account_dict["effective_from"]]
        if account_dict.get("effective_to") is not None:
            end_date_obj = date.fromisoformat(account_dict["effective_to"])
            boundary_list.append(end_date_obj.isoformat())
            if end_date_obj < date.max:
                boundary_list.append((end_date_obj + timedelta(days=1)).isoformat())
        membership_date_set.update(boundary_str for boundary_str in boundary_list if from_date_str <= boundary_str <= to_date_str)
    # Effective membership is a calendar-date contract, including weekends.
    # A newly scoped account must not disappear because no exchange session ran.
    report_date_list = sorted(set(session_list) | extra_date_set | membership_date_set)
    issue_list: list[str] = []
    if snapshot_obj.unavailable_reason_str:
        issue_list.append(snapshot_obj.unavailable_reason_str)
    if not report_date_list:
        issue_list.append("No broker reporting days in the selected interval.")
    strategy_result_list: list[dict[str, Any]] = []
    contributing_row_list: list[BrokerNavRow] = []
    bridge_by_key_dict: dict[tuple[str, str], dict] = {}
    for account_dict in account_list:
        # Independent account returns do not inherit another account's weekend
        # activity requirement. The client NAV bridge retains the strict union.
        own_extra_set = {row_obj.market_date_str for row_obj in row_by_key_dict.values() if row_obj.account_route_str == account_dict["account_route"] and from_date_str <= row_obj.market_date_str <= to_date_str}
        own_boundary_set = {date_str for date_str in (account_dict["effective_from"], account_dict.get("effective_to")) if date_str and from_date_str <= date_str <= to_date_str}
        expected_date_list = [date_str for date_str in sorted(set(session_list) | own_extra_set | own_boundary_set) if _account_active_bool(account_dict, date_str)]
        if not expected_date_list:
            continue
        selected_row_list = [
            row_by_key_dict[(account_dict["account_route"], date_str)]
            for date_str in expected_date_list if (account_dict["account_route"], date_str) in row_by_key_dict
        ]
        missing_date_list = [date_str for date_str in expected_date_list if (account_dict["account_route"], date_str) not in row_by_key_dict]
        account_issue_list: list[str] = []
        if missing_date_list:
            account_issue_list.append("Missing broker days: " + ", ".join(missing_date_list[:8]) + (" …" if len(missing_date_list) > 8 else ""))
        if any(row_obj.market_date_str >= market_today_str for row_obj in selected_row_list):
            account_issue_list.append("Current-day Activity Flex values are not finalized; reporting uses D+1 confirmation.")
        for previous_obj, current_obj in zip(selected_row_list, selected_row_list[1:]):
            if abs(current_obj.opening_nav_decimal - previous_obj.closing_nav_decimal) > BRIDGE_TOLERANCE_DECIMAL:
                account_issue_list.append(f"NAV continuity break before {current_obj.market_date_str}; an unobserved movement is not profit.")
        account_complete_bool = bool(selected_row_list) and not account_issue_list
        growth_decimal = Decimal(1)
        account_pnl_decimal = Decimal(0)
        flows_complete_bool = account_complete_bool
        for row_obj in selected_row_list:
            growth_decimal *= 1 + row_obj.twr_decimal
            bridge_dict = _daily_bridge_dict(row_obj, client_dict.get("nav_bridge"))
            bridge_by_key_dict[(row_obj.account_route_str, row_obj.market_date_str)] = bridge_dict
            if bridge_dict["complete_bool"]:
                account_pnl_decimal += bridge_dict["pnl_decimal"]
            else:
                flows_complete_bool = False
                account_issue_list.append(f"{row_obj.market_date_str}: {bridge_dict['reason_str']}")
        contributing_row_list.extend(selected_row_list)
        opening_row_obj = row_by_key_dict.get((account_dict["account_route"], expected_date_list[0]))
        closing_row_obj = row_by_key_dict.get((account_dict["account_route"], expected_date_list[-1]))
        strategy_result_list.append({
            "account_route_str": account_dict["account_route"], "pod_id_str": account_dict["pod_id"],
            "display_name_str": account_dict["display_name"],
            "from_date_str": expected_date_list[0], "to_date_str": expected_date_list[-1],
            "opening_nav_float": _money_float(opening_row_obj.opening_nav_decimal) if opening_row_obj else None,
            "closing_nav_float": _money_float(closing_row_obj.closing_nav_decimal) if closing_row_obj else None,
            "twr_float": float(growth_decimal - 1) if account_complete_bool else None,
            "twr_method_str": "Geometrically linked official IBKR daily account TWR",
            "pnl_float": _money_float(account_pnl_decimal) if flows_complete_bool else None,
            "coverage_complete_bool": account_complete_bool, "flows_complete_bool": flows_complete_bool,
            "expected_day_count_int": len(expected_date_list), "observed_day_count_int": len(selected_row_list),
            "issue_list": list(dict.fromkeys(account_issue_list)),
            "performance_dict": account_performance_dict(selected_row_list, complete_bool=account_complete_bool,
                from_date_str=max(from_date_str, account_dict["effective_from"]), to_date_str=min(to_date_str, account_dict.get("effective_to") or to_date_str), session_date_set=set(session_list)),
        })
        issue_list.extend(f"{account_dict['display_name']}: {issue_str}" for issue_str in dict.fromkeys(account_issue_list))
    all_coverage_bool = bool(strategy_result_list) and all(row_dict["coverage_complete_bool"] for row_dict in strategy_result_list)
    if any((account_dict["account_route"], date_str) not in row_by_key_dict for date_str in report_date_list for account_dict in account_list if _account_active_bool(account_dict, date_str)):
        all_coverage_bool = False
        issue_list.append("Client NAV aggregation needs all active accounts on each reporting date; independent account returns may still be available.")
    # Activity Flex is finalized D+1, not an intraday/current-day NAV source.
    # Match the existing sync contract without calling any sync/runner code.
    if any(row_obj.market_date_str >= market_today_str for row_obj in contributing_row_list):
        all_coverage_bool = False
        issue_list.append("Current-day Activity Flex values are not finalized; reporting uses D+1 confirmation.")
    all_flows_bool = all_coverage_bool and all(row_dict["flows_complete_bool"] for row_dict in strategy_result_list)
    daily_book_list: list[dict[str, Any]] = []
    twr_day_list = []
    previous_active_set: set[str] = set()
    previous_row_dict: dict[str, BrokerNavRow] = {}
    scope_total_decimal = Decimal(0)
    capital_total_decimal = Decimal(0)
    linking_total_decimal = Decimal(0)
    pnl_total_decimal = Decimal(0)
    opening_book_decimal: Decimal | None = None
    closing_book_decimal: Decimal | None = None
    for date_index_int, market_date_str in enumerate(report_date_list):
        active_set = {account_dict["account_route"] for account_dict in account_list if _account_active_bool(account_dict, market_date_str)}
        current_row_dict = {account_str: row_by_key_dict[(account_str, market_date_str)] for account_str in active_set if (account_str, market_date_str) in row_by_key_dict}
        day_complete_bool = len(current_row_dict) == len(active_set)
        nav_decimal = sum((row_obj.closing_nav_decimal for row_obj in current_row_dict.values()), Decimal(0)) if day_complete_bool else None
        if date_index_int == 0:
            opening_book_decimal = sum((row_obj.opening_nav_decimal for row_obj in current_row_dict.values()), Decimal(0)) if day_complete_bool else None
        elif all_coverage_bool:
            scope_total_decimal += sum((current_row_dict[account_str].opening_nav_decimal for account_str in active_set - previous_active_set), Decimal(0))
            scope_total_decimal -= sum((previous_row_dict[account_str].closing_nav_decimal for account_str in previous_active_set - active_set), Decimal(0))
        if date_index_int > 0 and day_complete_bool:
            for account_str in active_set & previous_active_set & previous_row_dict.keys():
                if abs(current_row_dict[account_str].opening_nav_decimal - previous_row_dict[account_str].closing_nav_decimal) > BRIDGE_TOLERANCE_DECIMAL:
                    issue_list.append(f"Account NAV continuity break at {market_date_str}, including across strategy-period boundaries.")
                    all_coverage_bool = False
                    all_flows_bool = False
        day_pnl_decimal = Decimal(0)
        if all_flows_bool:
            day_boundary_decimal = day_internal_decimal = Decimal(0)
            for account_str, row_obj in current_row_dict.items():
                bridge_dict = bridge_by_key_dict[(account_str, market_date_str)]
                capital_total_decimal += bridge_dict["capital_decimal"]
                linking_total_decimal += bridge_dict["boundary_decimal"]
                day_pnl_decimal += bridge_dict["pnl_decimal"]
                # Absolute linking activity must not cancel across accounts.
                day_boundary_decimal += abs(bridge_dict["boundary_decimal"])
                day_internal_decimal += _decimal_value(row_obj.attribute_dict, "internalCashTransfers")
            pnl_total_decimal += day_pnl_decimal
            twr_day_list.append((market_date_str,
                sum((row_obj.opening_nav_decimal for row_obj in current_row_dict.values()), Decimal(0)),
                day_pnl_decimal, day_boundary_decimal, day_internal_decimal))
        daily_book_list.append({
            "market_date_str": market_date_str, "nav_float": _money_float(nav_decimal),
            "pnl_float": _money_float(day_pnl_decimal) if all_flows_bool else None,
            "cumulative_pnl_float": _money_float(pnl_total_decimal) if all_flows_bool else None,
            "account_count_int": len(active_set), "coverage_complete_bool": day_complete_bool,
        })
        previous_active_set, previous_row_dict = active_set, current_row_dict
        closing_book_decimal = nav_decimal
    if all_flows_bool and opening_book_decimal is not None and closing_book_decimal is not None:
        aggregate_residual_decimal = closing_book_decimal - opening_book_decimal - capital_total_decimal - linking_total_decimal - scope_total_decimal - pnl_total_decimal
        if abs(aggregate_residual_decimal) > BRIDGE_TOLERANCE_DECIMAL:
            all_flows_bool = False
            issue_list.append("Client NAV bridge does not reconcile within USD 0.01; totals are withheld.")
    if not all_flows_bool:
        # Never show a partial cumulative profit path after a coverage failure.
        for daily_dict in daily_book_list:
            daily_dict["pnl_float"] = None
            daily_dict["cumulative_pnl_float"] = None
    client_twr_float: float | None = None
    client_twr_method_str = "Unavailable: daily account NAV/TWR cannot prove exact consolidated TWR with intraday flows."
    if len(strategy_result_list) == 1 and all_coverage_bool:
        strategy_dict = strategy_result_list[0]
        if report_date_list and strategy_dict["from_date_str"] == report_date_list[0] and strategy_dict["to_date_str"] == report_date_list[-1]:
            client_twr_float = strategy_dict["twr_float"]
            client_twr_method_str = "Official IBKR account TWR (single account covers the whole selected period)"
    twr_result_dict = {"return_path_list": [], "twr_daily_list": [], "twr_reason_str": None}
    client_twr_configured_bool = client_dict.get("client_twr") is not None
    if client_twr_configured_bool:
        twr_result_dict = _client_daily_twr_dict(twr_day_list, complete_bool=all_flows_bool, from_date_str=from_date_str)
        client_twr_float = twr_result_dict.pop("twr_float")
        client_twr_method_str = "Calculated client daily TWR; end-of-day flow convention, not official consolidated IBKR TWR."
        if twr_result_dict["twr_reason_str"]:
            issue_list.append("Client TWR: " + twr_result_dict["twr_reason_str"])
    used_import_set = {row_obj.source_import_id_int for row_obj in contributing_row_list}
    source_field_tuple = (
        "import_id_int", "checksum_str", "imported_timestamp_str", "query_name_str",
        "request_from_date_str", "request_to_date_str",
    )
    # Include range-replacement tombstones as evidence too. A correction which
    # removes all rows must remain visible and change the frozen report hash.
    source_list = [
        {field_str: source_dict[field_str] for field_str in source_field_tuple if field_str in source_dict}
        for source_dict in snapshot_obj.import_tuple
        if source_dict["import_id_int"] in used_import_set or (
            source_dict.get("query_name_str") == client_dict["query_name"]
            and source_dict.get("request_from_date_str", "9999-12-31") <= to_date_str
            and source_dict.get("request_to_date_str", "0001-01-01") >= from_date_str
        )
    ]
    for strategy_dict in strategy_result_list:
        strategy_dict["benchmark_dict"] = account_benchmark_dict(strategy_dict, benchmark_snapshot_obj, as_of_date_str=market_today_str)
    result_dict = {
        "method_version_str": METHOD_VERSION_STR,
        "is_demo_bool": client_dict.get("is_demo") is True,
        "client_id_str": client_dict["client_id"], "client_name_str": client_dict["display_name"],
        "base_currency_str": client_dict["base_currency"], "fee_basis_str": client_dict["fee_basis"],
        "requested_from_date_str": from_date_str, "requested_to_date_str": to_date_str,
        "opening_date_str": report_date_list[0] if report_date_list else None,
        "closing_date_str": report_date_list[-1] if report_date_list else None,
        "valuation_basis_str": "Broker daily starting NAV through ending NAV; not first fill or intraday live NetLiq",
        "generated_at_str": as_of_ts.isoformat(), "scope_hash_str": content_hash_str(client_dict),
        "opening_nav_float": _money_float(opening_book_decimal), "closing_nav_float": _money_float(closing_book_decimal),
        "capital_movement_float": _money_float(capital_total_decimal) if all_flows_bool else None,
        "linking_adjustment_float": _money_float(linking_total_decimal) if all_flows_bool else None,
        "scope_movement_float": _money_float(scope_total_decimal) if all_coverage_bool else None,
        "pnl_float": _money_float(pnl_total_decimal) if all_flows_bool else None,
        "twr_float": client_twr_float, "twr_method_str": client_twr_method_str,
        "client_twr_configured_bool": client_twr_configured_bool,
        "twr_method_id_str": CLIENT_TWR_METHOD_STR if client_twr_configured_bool else "official_account_only",
        **twr_result_dict,
        "coverage_complete_bool": all_coverage_bool, "flows_complete_bool": all_flows_bool,
        "status_str": "ready" if all_flows_bool and (not client_twr_configured_bool or client_twr_float is not None) else "draft",
        "strategy_list": strategy_result_list, "daily_book_list": daily_book_list,
        "source_list": source_list,
        "issue_list": list(dict.fromkeys(issue_list)),
        "limitations_list": [
            "Client TWR is not estimated by averaging accounts or using the legacy adjusted-base composite.",
            "Net capital movements do not establish gross intraday-flow completeness.",
            "NAV includes unrealized gains/losses; investment P&L is not cash withdrawn.",
            "Fee coverage is exactly the stated fee basis; unreported external fees are not invented.",
        ],
    }
    # Source/config/result hash is stable across view refreshes. The frozen
    # export keeps generated_at separately; corrections produce a new hash.
    result_dict["report_hash_str"] = content_hash_str({key_str: value_obj for key_str, value_obj in result_dict.items() if key_str != "generated_at_str"})
    return result_dict
