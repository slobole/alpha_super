"""Read-only client projections of saved operational evidence.

Current operational ownership is separate from a historical report selection.
Only an exact current (Pod, broker account, LIVE) match can contribute. Global
Inspector, combined-book totals and shared log payloads never cross this boundary.
"""

from copy import deepcopy
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from alpha.live.dashboard_v3.operator_tools import redact_diagnostic_value
from alpha.live.dashboard_v3.schedule import TradingWindow, build_trading_window_list
from alpha.live.ops_report import parse_timestamp_ts


SEVERITY_RANK_DICT = {"red": 0, "yellow": 1, "gray": 2, "green": 3}
SOURCE_MAX_AGE_SECONDS_INT = 120
EVENT_LABEL_DICT = {
    "build_decision_plan_created": "Decision plan created",
    "build_vplan_created": "Execution plan created",
    "submit_vplan_completed": "Order submission recorded",
    "post_execution_reconcile_completed": "Reconciliation recorded",
    "eod_snapshot_completed": "End-of-day snapshot recorded",
    "decision_plan_expired": "Decision plan expired",
    "build_vplan_blocked": "Execution plan blocked",
    "submit_vplan_missing_broker_ack": "Broker acknowledgement missing",
    "exit_residual_detected": "Residual position detected",
    "execution_exception_parked": "Execution exception parked",
}
ROW_FIELD_TUPLE = (
    "pod_id_str", "account_route_str", "mode_str", "release_id_str", "db_status_str",
    "required_action_dict", "debug_summary_dict", "data_freshness_dict",
    "eod_snapshot_dict", "lifecycle_step_dict_list", "next_action_str", "reason_code_str",
    "signal_clock_str", "execution_policy_str", "session_calendar_id_str",
    "latest_reconciliation_status_str", "latest_reconciliation_timestamp_str",
    "latest_decision_plan_id_int", "latest_decision_plan_status_str",
    "latest_decision_signal_timestamp_str", "latest_decision_plan_submission_timestamp_str",
    "latest_decision_plan_target_execution_timestamp_str", "latest_vplan_id_int",
    "latest_vplan_status_str", "latest_vplan_submission_timestamp_str",
    "latest_vplan_target_execution_timestamp_str", "latest_pod_state_timestamp_str",
    "latest_broker_snapshot_timestamp_str", "position_exposure_dict_list",
    "position_unpriced_count_int", "latest_live_reference_snapshot_timestamp_str",
    "latest_live_reference_source_str", "data_profile_str",
    "latest_decision_norgate_snapshot_date_str", "latest_decision_norgate_profile_str",
    "latest_vplan_is_for_latest_decision_bool", "latest_vplan_cycle_role_str",
    "missed_target_execution_timestamp_str", "health_str",
)


def _severity_str(value_obj):
    return value_obj if isinstance(value_obj, str) and value_obj in SEVERITY_RANK_DICT else "gray"


def _worst_str(severity_list):
    return min(severity_list or ["gray"], key=lambda value_str: SEVERITY_RANK_DICT[value_str])


def active_account_list(client_dict, as_of_ts):
    market_date_str = as_of_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    return [account_dict for account_dict in client_dict["accounts"]
            if account_dict["effective_from"] <= market_date_str <= (account_dict.get("effective_to") or "9999-12-31")]


def load_operations_summary_dict(client_dict, provider_obj):
    """Only explicit server configuration selects a source; no inferred VPS."""
    source_str = client_dict.get("operations_source", "unconfigured")
    if source_str == "local":
        return provider_obj.get_summary_dict()
    if source_str == "snapshot":
        snapshot_path_obj = Path(client_dict["operations_snapshot_path"])
        with snapshot_path_obj.open("rb") as snapshot_file_obj:
            snapshot_bytes = snapshot_file_obj.read(20_000_001)
        if len(snapshot_bytes) > 20_000_000:
            raise ValueError("Operations snapshot exceeds the 20 MB limit.")
        envelope_dict = json.loads(snapshot_bytes.decode("utf-8"))
        if not isinstance(envelope_dict, dict) or envelope_dict.get("client_id_str") != client_dict["client_id"] or type(envelope_dict.get("schema_version_int")) is not int or envelope_dict["schema_version_int"] != 1 or not isinstance(envelope_dict.get("summary_dict"), dict):
            raise ValueError("Operations snapshot identity/schema mismatch.")
        return envelope_dict["summary_dict"]
    return {}


def build_client_operations_dict(client_dict, summary_dict, *, as_of_ts):
    """Pure projection. A fresh page is not a fresh broker/process probe."""
    if not isinstance(summary_dict, dict) or not isinstance(summary_dict.get("pod_row_dict_list", []), list):
        raise ValueError("Invalid saved operations summary.")
    source_timestamp_str = summary_dict.get("as_of_timestamp_str")
    source_ts = parse_timestamp_ts(source_timestamp_str or "")
    source_fresh_bool = source_ts is not None and 0 <= (as_of_ts - source_ts).total_seconds() <= SOURCE_MAX_AGE_SECONDS_INT
    account_list = active_account_list(client_dict, as_of_ts)
    result_list = []
    for account_dict in account_list:
        matching_list = [row_dict for row_dict in summary_dict.get("pod_row_dict_list", [])
                         if isinstance(row_dict, dict) and row_dict.get("pod_id_str") == account_dict["pod_id"]
                         and row_dict.get("account_route_str") == account_dict["account_route"]
                         and row_dict.get("mode_str") == "live"]
        # A duplicate Pod ID on this source makes detail/action lookup ambiguous,
        # even if exactly one of those rows happens to match the account route.
        pod_match_count_int = sum(isinstance(row_dict, dict) and row_dict.get("pod_id_str") == account_dict["pod_id"]
                                  for row_dict in summary_dict.get("pod_row_dict_list", []))
        verified_bool = len(matching_list) == 1 and pod_match_count_int == 1
        evidence_dict = {key_str: deepcopy(matching_list[0].get(key_str)) for key_str in ROW_FIELD_TUPLE} if verified_bool else {}
        # Ownership must bound evidence, not just the row label. Do not expose
        # holdings left over from an earlier mandate using the same identifiers.
        if verified_bool:
            state_ts = parse_timestamp_ts(evidence_dict.get("latest_pod_state_timestamp_str") or "")
            if state_ts is None or state_ts > as_of_ts or state_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat() < account_dict["effective_from"]:
                verified_bool, evidence_dict = False, {}
        reason_list, severity_list = [], []
        if not verified_bool:
            reason_list.append("Expected strategy/account is missing or ambiguous on this source. Enabled state cannot be inferred.")
            severity_list.append("gray")
        else:
            for key_str in ("required_action_dict", "debug_summary_dict", "data_freshness_dict"):
                if not isinstance(evidence_dict.get(key_str), dict):
                    evidence_dict[key_str] = {}
            for key_str in ("position_exposure_dict_list", "lifecycle_step_dict_list"):
                raw_list = evidence_dict.get(key_str)
                evidence_dict[key_str] = [item_dict for item_dict in raw_list if isinstance(item_dict, dict)] if isinstance(raw_list, list) else []
            raw_freshness_list = evidence_dict["data_freshness_dict"].get("item_dict_list")
            evidence_dict["data_freshness_dict"]["item_dict_list"] = [item_dict for item_dict in raw_freshness_list if isinstance(item_dict, dict)] if isinstance(raw_freshness_list, list) else []
            required_dict = evidence_dict.get("required_action_dict") or {}
            debug_dict = evidence_dict.get("debug_summary_dict") or {}
            severity_list.extend([_severity_str(required_dict.get("severity_str")), _severity_str(debug_dict.get("severity_str"))])
            if evidence_dict.get("db_status_str") != "ok":
                severity_list.append("gray")
                reason_list.append("Strategy database evidence is unavailable.")
            for label_str in ("Norgate", "Pod state", "EOD Snapshot"):
                freshness_list = evidence_dict["data_freshness_dict"].get("item_dict_list")
                item_list = [item_dict for item_dict in freshness_list if isinstance(item_dict, dict) and item_dict.get("label_str") == label_str] if isinstance(freshness_list, list) else []
                item_dict = item_list[0] if len(item_list) == 1 else {}
                item_severity_str = _severity_str(item_dict.get("severity_str"))
                severity_list.append(item_severity_str)
                if item_severity_str != "green":
                    reason_list.append(f"{label_str}: {item_dict.get('detail_str') or 'evidence requires review'}")
            if required_dict.get("severity_str") != "green":
                reason_list.append(required_dict.get("detail_str") or required_dict.get("reason_str") or "Operational action requires review.")
        if not source_fresh_bool:
            severity_list.append("gray")
            reason_list.append("Saved operational assessment is missing, future-dated or older than 120 seconds. Refresh is not a live process check.")
        status_str = _worst_str(severity_list)
        result_list.append({
            "pod_id_str": account_dict["pod_id"], "account_route_str": account_dict["account_route"],
            "display_name_str": account_dict["display_name"], "effective_from_str": account_dict["effective_from"],
            "severity_str": status_str, "matched_bool": verified_bool,
            "status_label_str": {"red": "Action required", "yellow": "Review / waiting", "gray": "Cannot verify", "green": "No action required"}[status_str],
            "issue_list": redact_diagnostic_value(reason_list), "evidence_dict": redact_diagnostic_value(evidence_dict),
        })
    individual_window_list, resolved_evidence_list, failed_window_list = [], [], []
    # Verify each Pod independently: one malformed calendar must neither hide
    # another Pod's next cycle nor leave the affected Pod falsely green.
    for strategy_dict in result_list:
        if not strategy_dict["matched_bool"]:
            continue
        try:
            pod_window_list = [window_obj.as_dict() for window_obj in build_trading_window_list(
                {"pod_row_dict_list": [strategy_dict["evidence_dict"]]}, mode_str="live", now_dt=as_of_ts)]
        except (KeyError, TypeError, ValueError, AttributeError):
            pod_window_list = []
        if not pod_window_list:
            pod_window_list = [TradingWindow(
                detail_str="Saved calendar evidence could not be resolved. Review this strategy's calendar and cycle details.",
                pod_id_str_list=[strategy_dict["pod_id_str"]],
            ).as_dict()]
            failed_window_list.extend(pod_window_list)
        else:
            resolved_evidence_list.append(strategy_dict["evidence_dict"])
        individual_window_list.extend(pod_window_list)
        for window_dict in pod_window_list:
            # Gray is also used for a valid idle/future window. Only missing
            # calendar proof is unknown; a normal idle window remains healthy.
            calendar_verified_bool = window_dict["has_data_bool"] and window_dict["status_label_str"] != "Cannot verify"
            calendar_severity_str = window_dict["severity_str"] if calendar_verified_bool or window_dict["severity_str"] in {"red", "yellow"} else "gray"
            if calendar_severity_str == "green" or (calendar_severity_str == "gray" and calendar_verified_bool):
                continue
            status_str = _worst_str([strategy_dict["severity_str"], calendar_severity_str])
            strategy_dict["severity_str"] = status_str
            strategy_dict["status_label_str"] = {"red": "Action required", "yellow": "Review / waiting", "gray": "Cannot verify"}[status_str]
            strategy_dict["issue_list"].insert(0, window_dict["status_label_str"] + ": " + window_dict["detail_str"])
    try:
        window_list = [window_obj.as_dict() for window_obj in build_trading_window_list(
            {"pod_row_dict_list": resolved_evidence_list}, mode_str="live", now_dt=as_of_ts)] + failed_window_list
    except (KeyError, TypeError, ValueError, AttributeError):
        window_list = individual_window_list  # Grouping is presentation only.
    severity_str = _worst_str([row_dict["severity_str"] for row_dict in result_list])
    return {
        "client_id_str": client_dict["client_id"], "mode_str": "live",
        "source_str": client_dict.get("operations_source", "unconfigured"),
        "source_timestamp_str": source_timestamp_str, "view_timestamp_str": as_of_ts.isoformat(),
        "source_fresh_bool": source_fresh_bool, "severity_str": severity_str,
        "title_str": {"red": "Action required", "yellow": "Review needed", "gray": "Cannot verify operations", "green": "No action required"}[severity_str],
        "strategy_list": result_list, "trading_window_list": window_list,
        "limitations_str": "Saved assessment only. This view neither checks running processes nor contacts the broker. Financial dates do not change current operational scope.",
    }


def safe_client_event_list(event_list, strategy_dict, *, from_date_str, to_date_str, as_of_ts=None):
    """Omit shared/ambiguous payloads. Return scalar event facts, never raw blobs."""
    as_of_ts = as_of_ts or datetime.now(UTC)
    result_list = []
    for event_dict in event_list:
        if not isinstance(event_dict, dict) or event_dict.get("pod_id_str") != strategy_dict["pod_id_str"]:
            continue
        account_alias_list = [event_dict[key_str] for key_str in ("account_id_str", "account_route_str", "account_str") if event_dict.get(key_str) not in (None, "")]
        mode_alias_list = [event_dict[key_str] for key_str in ("mode_str", "env_mode_str", "session_mode_str") if event_dict.get(key_str) not in (None, "")]
        if not account_alias_list or any(value_obj != strategy_dict["account_route_str"] for value_obj in account_alias_list):
            continue
        if not mode_alias_list or any(value_obj != "live" for value_obj in mode_alias_list):
            continue
        if event_dict.get("pod_str") not in (None, "", strategy_dict["pod_id_str"]) or any(event_dict.get(key_str) for key_str in ("related_pod_id_list", "pod_id_list", "pod_id_str_list")):
            continue
        # *** CRITICAL *** ownership-sensitive: occurrence time, not a payload's
        # scheduled target/as-of time, determines the inclusive ET mandate day.
        timestamp_str = event_dict.get("event_timestamp_str") or event_dict.get("ts_utc") or event_dict.get("timestamp_str") or event_dict.get("created_timestamp_str")
        try:
            event_ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if event_dict.get("event_timestamp_str") and event_dict.get("ts_utc"):
                canonical_ts = datetime.fromisoformat(event_dict["ts_utc"].replace("Z", "+00:00"))
                if canonical_ts.tzinfo is None or canonical_ts != event_ts:
                    continue
        except (ValueError, TypeError, AttributeError):
            continue
        if event_ts.tzinfo is None or event_ts > as_of_ts:
            continue
        event_ts = event_ts.astimezone(UTC)
        date_str = event_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        if not max(from_date_str, strategy_dict["effective_from_str"]) <= date_str <= min(to_date_str, strategy_dict.get("effective_to_str") or "9999-12-31"):
            continue
        projected_dict = {key_str: event_dict[key_str][:2000] for key_str in ("event_type_str", "event_name_str", "level_str", "message_str", "reason_str", "reason_code_str", "status_str") if isinstance(event_dict.get(key_str), str)}
        projected_dict.update(timestamp_str=event_ts.isoformat(), display_name_str=strategy_dict["display_name_str"])
        projected_dict["market_timestamp_str"] = event_ts.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
        event_name_str = projected_dict.get("event_name_str") or projected_dict.get("event_type_str") or "Saved event"
        projected_dict["label_str"] = EVENT_LABEL_DICT.get(event_name_str, event_name_str)
        projected_dict["material_bool"] = event_name_str in EVENT_LABEL_DICT or str(projected_dict.get("level_str", "")).lower() in {"warning", "warn", "error", "critical"}
        result_list.append(redact_diagnostic_value(projected_dict))
    return sorted(result_list, key=lambda row_dict: row_dict["timestamp_str"], reverse=True)


def load_client_activity_dict(client_dict, provider_obj, *, from_date_str, to_date_str, as_of_ts):
    """Historical ownership, independent of today's enabled/healthy Pods.

    Local logs are latest-500 per Pod, not a full audit archive. Snapshot events
    are optional saved exports. Neither source can prove complete log coverage.
    """
    account_list = [account_dict for account_dict in client_dict["accounts"]
                    if account_dict["effective_from"] <= to_date_str and (account_dict.get("effective_to") or "9999-12-31") >= from_date_str]
    source_str = client_dict.get("operations_source", "unconfigured")
    result_dict = {
        "source_str": source_str, "coverage_str": "unavailable", "issue_list": [],
        "read_at_str": as_of_ts.isoformat(), "export_timestamp_str": None,
        "latest_matched_event_timestamp_str": None, "strategy_period_count_int": len(account_list),
        "event_list": [], "material_event_list": [], "display_truncated_bool": False,
        "coverage_detail_str": "Partial evidence only: local logs retain at most the latest 500 events per Pod; snapshot exports may omit history. Shared, unowned or ambiguous events are omitted. Empty results do not prove no activity.",
    }
    if not account_list or source_str == "unconfigured":
        return result_dict
    raw_event_dict = {}
    try:
        if source_str == "snapshot":
            summary_dict = load_operations_summary_dict(client_dict, provider_obj)
            event_list = summary_dict.get("event_dict_list")
            if not isinstance(event_list, list) or len(event_list) > 50_000:
                raise ValueError("Missing or invalid event export.")
            export_timestamp_str = summary_dict.get("as_of_timestamp_str")
            if not isinstance(export_timestamp_str, str):
                raise ValueError("Missing export time.")
            export_ts = datetime.fromisoformat(export_timestamp_str.replace("Z", "+00:00"))
            if export_ts.tzinfo is None or export_ts > as_of_ts:
                raise ValueError("Missing or invalid export time.")
            result_dict["export_timestamp_str"] = export_ts.isoformat()
            # An export cannot contain an event occurring after its own cutoff.
            as_of_ts = min(as_of_ts, export_ts)
            for event_dict in event_list:
                if isinstance(event_dict, dict) and isinstance(event_dict.get("pod_id_str"), str):
                    raw_event_dict.setdefault(event_dict["pod_id_str"], []).append(event_dict)
            result_dict["coverage_str"] = "partial"
        elif source_str == "local":
            for pod_id_str in dict.fromkeys(account_dict["pod_id"] for account_dict in account_list):
                try:
                    event_list = provider_obj.get_pod_event_dict_list(pod_id_str, limit_int=500)
                    if not isinstance(event_list, list) or len(event_list) > 500:
                        raise ValueError("Invalid saved event list.")
                    raw_event_dict[pod_id_str] = event_list
                    result_dict["coverage_str"] = "partial"
                except (OSError, ValueError, KeyError, TypeError):
                    result_dict["issue_list"].append("A strategy's saved log could not be read. Activity evidence is incomplete.")
        for account_dict in account_list:
            strategy_dict = {
                "pod_id_str": account_dict["pod_id"], "account_route_str": account_dict["account_route"],
                "display_name_str": account_dict["display_name"], "effective_from_str": account_dict["effective_from"],
                "effective_to_str": account_dict.get("effective_to"),
            }
            result_dict["event_list"].extend(safe_client_event_list(raw_event_dict.get(account_dict["pod_id"], []), strategy_dict,
                from_date_str=from_date_str, to_date_str=to_date_str, as_of_ts=as_of_ts))
    except (OSError, ValueError, KeyError, TypeError):
        result_dict["issue_list"].append("Saved activity source could not be read or validated. No alternative source was used.")
    result_dict["event_list"].sort(key=lambda event_dict: event_dict["timestamp_str"], reverse=True)
    if result_dict["event_list"]:
        result_dict["latest_matched_event_timestamp_str"] = result_dict["event_list"][0]["timestamp_str"]
    result_dict["material_event_list"] = [event_dict for event_dict in result_dict["event_list"] if event_dict["material_bool"]][:3]
    result_dict["display_truncated_bool"] = len(result_dict["event_list"]) > 1000
    result_dict["event_list"] = result_dict["event_list"][:1000]
    return result_dict


def build_reference_exposure_list(operations_dict):
    """No mixed-date NAV denominator, no invented leveraged-ETF look-through.

    Value = saved shares * saved reference price. These are execution references,
    not current market marks; hence no portfolio weights or aggregate risk claim.
    """
    result_list = []
    for strategy_dict in operations_dict["strategy_list"]:
        evidence_dict = strategy_dict["evidence_dict"]
        for position_dict in evidence_dict.get("position_exposure_dict_list") or []:
            if not isinstance(position_dict, dict):
                continue
            share_obj, price_obj = position_dict.get("share_float"), position_dict.get("price_float")
            valid_bool = all(type(value_obj) in {int, float} and math.isfinite(value_obj) for value_obj in (share_obj, price_obj)) and price_obj > 0
            value_float = share_obj * price_obj if valid_bool else None
            if value_float is not None and not math.isfinite(value_float):
                value_float = None
            result_list.append({
                "display_name_str": strategy_dict["display_name_str"], "asset_str": position_dict.get("asset_str"),
                "share_float": share_obj if type(share_obj) in {int, float} and math.isfinite(share_obj) else None,
                "reference_value_float": value_float,
                "reference_timestamp_str": evidence_dict.get("latest_live_reference_snapshot_timestamp_str"),
                "position_timestamp_str": evidence_dict.get("latest_pod_state_timestamp_str"),
                "reference_source_str": evidence_dict.get("latest_live_reference_source_str") or "Unknown",
            })
    return result_list
