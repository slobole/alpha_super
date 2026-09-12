"""Read saved broker EOD cash for the financial chart; no capture or writes."""

from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from alpha.live import scheduler_utils
from alpha.live.ops_report import parse_timestamp_ts
from alpha.live.runner import DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT
from alpha.live.dashboard_v3.client_presentation import portfolio_account_list


def _is_selected_eod_bool(timestamp_str, date_str, calendar_str, as_of_ts):
    if not isinstance(timestamp_str, str):
        return False
    try:
        if datetime.fromisoformat(timestamp_str).tzinfo is None:
            return False
    except ValueError:
        return False
    timestamp_ts = parse_timestamp_ts(timestamp_str or "")
    if timestamp_ts is None or timestamp_ts > as_of_ts or timestamp_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat() != date_str:
        return False
    session_ts = scheduler_utils.session_label_from_timestamp_ts(timestamp_ts, calendar_str)
    if session_ts is None:
        return False
    close_ts = scheduler_utils.get_session_close_timestamp_ts(session_ts, calendar_str)
    # *** CRITICAL *** retrospective exact-date display. An intraday or other
    # day's cash cannot be paired with selected EOD NAV, even if values agree.
    return timestamp_ts >= close_ts + timedelta(minutes=DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT)


def load_portfolio_cash_list(client_dict, report_dict, provider_obj, operations_dict, *, as_of_ts):
    date_str = report_dict.get("closing_date_str")
    source_str = client_dict.get("operations_source")
    if not date_str or source_str not in {"local", "snapshot"}:
        return []
    result_list = []
    for account_dict in portfolio_account_list(report_dict):
        pod_str, route_str = account_dict.get("pod_id"), account_dict["account_route"]
        if not pod_str:
            continue
        try:
            strategy_list = [item_dict for item_dict in operations_dict.get("strategy_list", [])
                if item_dict["pod_id_str"] == pod_str and item_dict["account_route_str"] == route_str and item_dict["matched_bool"]]
            evidence_dict = strategy_list[0]["evidence_dict"] if len(strategy_list) == 1 else {}
            eod_dict = evidence_dict.get("eod_snapshot_dict") or {}
            if ((source_str == "snapshot" or client_dict.get("is_demo"))
                    and eod_dict.get("source_str") == "broker" and eod_dict.get("latest_market_date_str") == date_str
                    and _is_selected_eod_bool(eod_dict.get("latest_timestamp_str"), date_str, evidence_dict.get("session_calendar_id_str"), as_of_ts)):
                result_list.append({"pod_id_str": pod_str, "account_route_str": route_str, "market_date_str": date_str,
                    "cash_float": eod_dict.get("cash_float"), "equity_float": eod_dict.get("equity_float")})
                continue
            # Local history is authoritative, including duplicate checks.
            # Snapshot clients stay on their explicitly configured source.
            if source_str != "local":
                continue
            target_obj = provider_obj.get_target_for_pod(pod_str)
            if target_obj is None:
                continue
            release_obj = target_obj.release_obj
            if (release_obj.mode_str, release_obj.pod_id_str, release_obj.account_route_str) != ("live", pod_str, route_str):
                continue
            with closing(sqlite3.connect(Path(target_obj.db_path_str).resolve().as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
                connection_obj.row_factory = sqlite3.Row
                row_list = [dict(row_obj) for row_obj in connection_obj.execute(
                    "SELECT updated_timestamp_str, cash_float, total_value_float FROM pod_state_history "
                    "WHERE pod_id_str=? AND account_route_str=? AND user_id_str=? "
                    "AND snapshot_stage_str='eod' AND snapshot_source_str='broker'",
                    (pod_str, route_str, release_obj.user_id_str))]
            row_list = [row_dict for row_dict in row_list if _is_selected_eod_bool(
                row_dict["updated_timestamp_str"], date_str, release_obj.session_calendar_id_str, as_of_ts)]
            if not row_list:
                continue
            latest_ts = max(parse_timestamp_ts(row_dict["updated_timestamp_str"]) for row_dict in row_list)
            latest_list = [row_dict for row_dict in row_list if parse_timestamp_ts(row_dict["updated_timestamp_str"]) == latest_ts]
            if len(latest_list) != 1:
                continue
            result_list.append({"pod_id_str": pod_str, "account_route_str": route_str, "market_date_str": date_str,
                "cash_float": latest_list[0]["cash_float"], "equity_float": latest_list[0]["total_value_float"]})
        except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error):
            # Optional display evidence never hides NAV or opens another source.
            continue
    return result_list
