"""Deterministic, visibly synthetic local preview. No broker/files/env access."""

from datetime import UTC, date, datetime, timedelta
from copy import deepcopy
from decimal import Decimal
import hashlib
import math
import xml.etree.ElementTree as ElementTree

from alpha.live.client_reporting import BrokerReportingSnapshot, CAPITAL_FIELD_TUPLE, parse_broker_nav_import
from alpha.live.scheduler_utils import get_exchange_calendar_obj
from alpha.live.client_benchmark import BenchmarkSnapshot


DEMO_START_STR = "2026-06-01"
DEMO_END_STR = "2026-09-04"


def build_demo_benchmark_snapshot():
    """A labeled synthetic TR path, not downloaded SPY market performance."""
    close_list = [("2026-05-29", 100.0)]
    price_float = 100.0
    for index_int, session_obj in enumerate(get_exchange_calendar_obj("XNYS").sessions_in_range(DEMO_START_STR, DEMO_END_STR)):
        price_float *= 1 + .0008 + .002 * math.sin(index_int * .41)
        close_list.append((session_obj.date().isoformat(), round(price_float, 8)))
    checksum_str = hashlib.sha256(repr(close_list).encode()).hexdigest()
    return BenchmarkSnapshot("SPY", tuple(close_list), checksum_str, checksum_str, DEMO_END_STR, "DEMO_ONLY", True)


class DemoOperationsProvider:
    """No live provider fallback: even old routes cannot reach real accounts."""

    results_root_path_str = ".codex_tmp/nonexistent-demo-artifacts"

    def __init__(self, registry_dict=None):
        self.row_list = []
        if registry_dict is None:
            registry_dict, _ = build_demo_fixture_tuple()
        for client_dict in registry_dict["clients"]:
            for index_int, account_dict in enumerate(client_dict["accounts"]):
                severity_str = "yellow" if client_dict["client_id"] == "demo-client" and index_int == 3 else "green"
                self.row_list.append({
                    "pod_id_str": account_dict["pod_id"], "account_route_str": account_dict["account_route"],
                    "mode_str": "live", "release_id_str": "DEMO-release-1", "db_status_str": "ok",
                    "next_action_str": "wait", "reason_code_str": "not_month_end_session",
                    "signal_clock_str": "month_end_snapshot_ready", "session_calendar_id_str": "XNYS",
                    "execution_policy_str": "next_month_first_open", "health_str": severity_str,
                    "required_action_dict": {"severity_str": severity_str, "label_str": "Cash buffer review" if severity_str == "yellow" else "No action", "detail_str": "Demonstration: inspect the saved cash-buffer warning." if severity_str == "yellow" else "Last cycle reconciled; next signal is at month end."},
                    "debug_summary_dict": {"severity_str": severity_str},
                    "latest_reconciliation_status_str": "passed", "latest_reconciliation_timestamp_str": "2026-09-01T13:40:00+00:00",
                    "latest_decision_plan_id_int": 2, "latest_decision_plan_status_str": "completed",
                    "latest_decision_signal_timestamp_str": "2026-08-31T20:00:00+00:00",
                    "latest_vplan_id_int": 2, "latest_vplan_status_str": "completed",
                    "latest_vplan_is_for_latest_decision_bool": True, "latest_vplan_cycle_role_str": "current_cycle",
                    "latest_pod_state_timestamp_str": "2026-09-04T20:10:00+00:00",
                    "latest_broker_snapshot_timestamp_str": "2026-09-04T20:10:00+00:00",
                    "latest_live_reference_snapshot_timestamp_str": "2026-09-01T13:23:00+00:00",
                    "latest_live_reference_source_str": "DEMO saved pre-submit reference",
                    "latest_decision_norgate_profile_str": "DEMO-profile", "latest_decision_norgate_snapshot_date_str": "2026-08-31",
                    "data_freshness_dict": {"item_dict_list": [{"label_str": label_str, "severity_str": "green", "value_str": "2026-09-04", "detail_str": "Synthetic last-required evidence present"} for label_str in ("Norgate", "Pod state", "EOD Snapshot")]},
                    "lifecycle_step_dict_list": [{"label_str": label_str, "severity_str": "green", "status_str": "complete"} for label_str in ("DB", "Decision", "VPlan", "ACK", "Fill", "Reconcile", "EOD")],
                    "position_exposure_dict_list": [{"asset_str": "TQQQ" if index_int == 0 else "DEMO_EQUITY", "share_float": 20 + 5 * index_int, "price_float": 70 if index_int == 0 else 150}],
                })

    def get_summary_dict(self):
        return {
            "as_of_timestamp_str": datetime.now(UTC).isoformat(),
            "pod_row_dict_list": deepcopy(self.row_list), "mode_list": ["live"],
            "combined_book_dict": {"environment_dict_list": []},
            "alert_dict_list": [], "alert_summary_dict": {},
        }

    def get_target_list(self):
        return []

    def get_target_for_pod(self, pod_id_str):
        return None

    def get_pod_detail_dict(self, pod_id_str):
        row_dict = next((row_dict for row_dict in self.row_list if row_dict["pod_id_str"] == pod_id_str), None)
        if row_dict is None:
            raise KeyError(pod_id_str)
        return {"pod_row_dict": deepcopy(row_dict), "lifecycle_step_dict_list": deepcopy(row_dict["lifecycle_step_dict_list"]), "event_dict_list": self.get_pod_event_dict_list(pod_id_str)}

    def get_pod_event_dict_list(self, pod_id_str, limit_int=80):
        row_dict = next((row_dict for row_dict in self.row_list if row_dict["pod_id_str"] == pod_id_str), None)
        if row_dict is None:
            return []
        return [{"pod_id_str": pod_id_str, "account_route_str": row_dict["account_route_str"], "mode_str": "live", "event_timestamp_str": "2026-09-01T13:40:00+00:00", "event_name_str": "post_execution_reconcile_completed", "level_str": "info", "message_str": "Synthetic cycle: positions matched; no remaining orders."}][:limit_int]

    def get_pod_trace_event_dict_list(self, pod_id_str, limit_int=80):
        return []


def build_demo_fixture_tuple():
    calendar_obj = get_exchange_calendar_obj("XNYS")
    session_date_set = {session_obj.date().isoformat() for session_obj in calendar_obj.sessions_in_range(DEMO_START_STR, DEMO_END_STR)}
    client_list, snapshot_dict = [], {}
    for client_index_int, (client_id_str, display_name_str, strategy_name_list) in enumerate([
        ("demo-owner", "DEMO - Operator's book", ["Tactical allocation", "Nasdaq momentum"]),
        ("demo-client", "DEMO - Client portfolio", ["Tactical allocation", "Nasdaq momentum", "Defensive rotation", "Equity mean reversion"]),
    ]):
        client_dict = {
            "client_id": client_id_str, "display_name": display_name_str, "base_currency": "USD",
            "mandate_start_date": DEMO_START_STR, "is_demo": True, "query_name": "DEMO_NAV",
            "operations_source": "local",
            "client_twr": {"method": "daily_nav_eod_v1", "reviewed_by": "Synthetic fixture",
                           "evidence_ref": "Deterministic demo: complete EOD flows, no internal transfers"},
            "fee_basis": "Synthetic demonstration only. Not actual broker performance or an investor statement.",
            "accounts": [], "nav_bridge": {
                "profile_id": "DEMO_ONLY", "evidence_ref": "Deterministic in-memory demonstration",
                "reviewed_by": "Synthetic fixture", "mode": "MTM", "nonoverlap_confirmed": True,
                "economic_fields": ["mtm"], "informational_fields": [],
            },
        }
        root_obj = ElementTree.Element("FlexQueryResponse", queryName="DEMO_NAV")
        statements_obj = ElementTree.SubElement(root_obj, "FlexStatements")
        account_set = set()
        for strategy_index_int, strategy_name_str in enumerate(strategy_name_list):
            account_str = f"DEMO_{client_index_int}_{strategy_index_int}"
            account_set.add(account_str)
            client_dict["accounts"].append({
                "account_route": account_str, "pod_id": f"demo_{client_index_int}_{strategy_index_int}",
                "display_name": strategy_name_str, "effective_from": DEMO_START_STR, "effective_to": None,
            })
            statement_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId=account_str)
            ElementTree.SubElement(statement_obj, "AccountInformation", accountId=account_str, currency="USD")
            nav_decimal = Decimal(20000 + 10000 * client_index_int - 5000 * strategy_index_int)
            market_day_obj, day_index_int = date.fromisoformat(DEMO_START_STR), 0
            while market_day_obj.isoformat() <= DEMO_END_STR:
                date_str = market_day_obj.isoformat()
                opening_decimal = nav_decimal
                return_decimal = Decimal(str(round(.00065 + .004 * math.sin(day_index_int * .51 + strategy_index_int) - .0015 * math.cos(day_index_int * .17), 6))) if date_str in session_date_set else Decimal(0)
                pnl_decimal = (opening_decimal * return_decimal).quantize(Decimal(".01"))
                capital_decimal = Decimal(2000) if date_str == "2026-07-15" and strategy_index_int == 0 else Decimal(0)
                nav_decimal += pnl_decimal + capital_decimal
                attribute_dict = {
                    "accountId": account_str, "currency": "USD", "fromDate": date_str.replace("-", ""), "toDate": date_str.replace("-", ""),
                    "startingValue": str(opening_decimal), "endingValue": str(nav_decimal), "twr": str(pnl_decimal / opening_decimal * 100),
                    "mtm": str(pnl_decimal), "linkingAdjustments": "0", **{field_str: "0" for field_str in CAPITAL_FIELD_TUPLE},
                }
                attribute_dict["depositsWithdrawals"] = str(capital_decimal)
                ElementTree.SubElement(statement_obj, "ChangeInNAV", **attribute_dict)
                market_day_obj += timedelta(days=1)
                day_index_int += 1
        xml_str = ElementTree.tostring(root_obj, encoding="unicode")
        checksum_str = hashlib.sha256(xml_str.encode()).hexdigest()
        row_list = parse_broker_nav_import(xml_str, allowed_account_set=account_set, query_name_str="DEMO_NAV", source_import_id_int=1, source_checksum_str=checksum_str)
        snapshot_dict[client_id_str] = BrokerReportingSnapshot(tuple(row_list), ({
            "import_id_int": 1, "checksum_str": checksum_str, "query_name_str": "DEMO_NAV",
            "request_from_date_str": DEMO_START_STR, "request_to_date_str": DEMO_END_STR,
            "imported_timestamp_str": "2026-09-05T06:00:00+00:00",
        },))
        client_list.append(client_dict)
    return {"schema_version": 1, "clients": client_list}, snapshot_dict
