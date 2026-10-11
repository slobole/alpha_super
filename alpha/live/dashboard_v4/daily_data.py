"""Bounded, read-only facts for one daily (CORE5, MR capsule) cycle.

Daily pods close each cycle after the session from actual holdings. This reader
returns the saved facts the daily status projection needs, in one SQLite read
snapshot, and never keeps free text: no error_str, send_error_str, alert reasons
or client IDs. Optional tables written by newer daily code are checked first,
so an older state DB still reads. Nothing here calls a state writer.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha.live import scheduler_utils
from alpha.live.daily_reconcile import DAILY_STRATEGY_IMPORT_TUPLE, DAILY_TERMINAL_STATUS_SET
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.ops_report import parse_timestamp_ts


# Bounded reads: a daily pod has one decision per session.
RECENT_DECISION_LIMIT_INT = 10
OPEN_PREVIOUS_LIMIT_INT = 20
EXCEPTION_LIMIT_INT = 200
ROW_LIMIT_INT = 50
# Watchdog cutoff: a signal-session decision is due two minutes before the next open.
DECISION_CUTOFF_MINUTES_INT = 2
CLOSED_STATUS_SET = frozenset(DAILY_TERMINAL_STATUS_SET)
NOT_OPEN_STATUS_TUPLE = (*sorted(DAILY_TERMINAL_STATUS_SET), "superseded")
COMPLETION_KEY_LIKE_STR = "%:daily-completion:%"
CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
ASSET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}")
EXCEPTION_FIELD_TUPLE = ("asset_str", "side_str", "quantity_float", "reason_str", "expected_share_float",
    "actual_share_float", "target_weight_float", "intent_str")
UNKNOWN_DICT = {"status_str": "unknown"}


class _DailyDataError(ValueError):
    """Static operator-safe reason; never carries raw DB values."""


def is_daily_release_bool(release_obj) -> bool:
    return getattr(release_obj, "strategy_import_str", None) in DAILY_STRATEGY_IMPORT_TUPLE


def _code_str(value_obj):
    return value_obj if isinstance(value_obj, str) and CODE_RE.fullmatch(value_obj) else None


def _asset_str(value_obj):
    return value_obj if isinstance(value_obj, str) and ASSET_RE.fullmatch(value_obj) else None


def _float_or_none(value_obj):
    if isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)):
        return None
    return float(value_obj) if math.isfinite(value_obj) else None


def _timestamp_str(value_obj):
    timestamp_ts = parse_timestamp_ts(value_obj) if isinstance(value_obj, str) else None
    return timestamp_ts.isoformat() if timestamp_ts is not None else None


def _observed_ts(value_obj, as_of_ts):
    timestamp_ts = parse_timestamp_ts(value_obj) if isinstance(value_obj, str) else None
    if timestamp_ts is None or timestamp_ts > as_of_ts:
        raise _DailyDataError("A saved timestamp is missing or in the future.")
    return timestamp_ts


def _session_date_str(timestamp_str):
    timestamp_ts = parse_timestamp_ts(timestamp_str)
    if timestamp_ts is None:
        raise _DailyDataError("A saved cycle time is missing.")
    return timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()


def _session_close_ts(target_timestamp_str, calendar_id_str):
    """The target session's own exchange close, early closes included."""
    label_ts = scheduler_utils.session_label_from_timestamp_ts(parse_timestamp_ts(target_timestamp_str), calendar_id_str)
    if label_ts is None:
        raise _DailyDataError("A saved target is not an exchange session.")
    return scheduler_utils.get_session_close_timestamp_ts(label_ts, calendar_id_str).astimezone(UTC)


def _next_open_ts(as_of_ts, calendar_id_str):
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
    market_ts = scheduler_utils.to_market_timestamp_ts(as_of_ts, calendar_id_str)
    label_ts = calendar_obj.date_to_session(market_ts.date(), direction="next")
    open_ts = scheduler_utils.get_session_open_timestamp_ts(label_ts, calendar_id_str)
    if open_ts <= as_of_ts:
        open_ts = scheduler_utils.get_session_open_timestamp_ts(calendar_obj.next_session(label_ts), calendar_id_str)
    return open_ts.astimezone(UTC)


def _metadata_dict(row_obj):
    metadata_dict = json.loads(row_obj["snapshot_metadata_json_str"] or "{}")
    if not isinstance(metadata_dict, dict):
        raise _DailyDataError("Saved decision metadata is invalid.")
    return metadata_dict


def _exception_row_dict(row_obj):
    row_obj = row_obj if isinstance(row_obj, dict) else {}
    side_str = row_obj.get("side_str")
    return {
        "asset_str": _asset_str(row_obj.get("asset_str")),
        "side_str": side_str if side_str in {"BUY", "SELL"} else None,
        "quantity_float": _float_or_none(row_obj.get("quantity_float")),
        "reason_str": _code_str(row_obj.get("reason_str")),
        "expected_share_float": _float_or_none(row_obj.get("expected_share_float")),
        "actual_share_float": _float_or_none(row_obj.get("actual_share_float")),
        "target_weight_float": _float_or_none(row_obj.get("target_weight_float")),
        "intent_str": _code_str(row_obj.get("intent_str")),
    }


def _execution_result_dict(metadata_dict):
    result_obj = metadata_dict.get("daily_execution_result_dict")
    if not isinstance(result_obj, dict):
        return {}
    exception_list = result_obj.get("exception_list")
    exception_list = exception_list if isinstance(exception_list, list) else []
    return {"status_str": _code_str(result_obj.get("status_str")),
        "completed_timestamp_str": _timestamp_str(result_obj.get("completed_timestamp_str")),
        "exception_count_int": len(exception_list),
        "exception_list": [_exception_row_dict(row_obj) for row_obj in exception_list[:EXCEPTION_LIMIT_INT]]}


def _opening_dispatch_dict(metadata_dict):
    payload_obj = metadata_dict.get("opening_dispatch_result_dict")
    if not isinstance(payload_obj, dict):
        return {}
    count_dict = {}
    for source_str, destination_str in (("attempted_request_key_list", "attempted_count_int"),
            ("never_dispatched_request_key_list", "never_dispatched_count_int")):
        # Only counts: request keys identify broker orders and stay in the DB.
        key_list = payload_obj.get(source_str)
        count_dict[destination_str] = len(key_list) if isinstance(key_list, list) else 0
    return {"reason_code_str": _code_str(payload_obj.get("reason_code_str")),
        "severity_str": _code_str(payload_obj.get("severity_str")),
        "observed_timestamp_str": _timestamp_str(payload_obj.get("observed_timestamp_str")),
        "dispatch_deadline_timestamp_str": _timestamp_str(payload_obj.get("dispatch_deadline_timestamp_str")),
        **count_dict}


def _decision_dict(row_obj):
    return {"decision_plan_id_int": row_obj["decision_plan_id_int"], "release_id_str": row_obj["release_id_str"],
        "status_str": _code_str(row_obj["status_str"]),
        "signal_timestamp_str": _timestamp_str(row_obj["signal_timestamp_str"]),
        "submission_timestamp_str": _timestamp_str(row_obj["submission_timestamp_str"]),
        "target_execution_timestamp_str": _timestamp_str(row_obj["target_execution_timestamp_str"]),
        "signal_session_date_str": _session_date_str(row_obj["signal_timestamp_str"]),
        "target_session_date_str": _session_date_str(row_obj["target_execution_timestamp_str"]),
        "updated_timestamp_str": _timestamp_str(row_obj["updated_timestamp_str"])}


def _count_int(connection_obj, sql_str, params_tuple):
    return int(connection_obj.execute(sql_str, params_tuple).fetchone()[0])


def read_daily_cycle_dict(connection_obj, release_obj, *, decision_plan_id_int, as_of_ts):
    """Read one daily cycle inside the caller's SQLite read snapshot.

    ``decision_plan_id_int=None`` selects the newest decision of this pod and
    account. Returns ``{"status_str": "unknown"}`` for a non-daily release and
    a safe reason for unverifiable evidence; partial facts are never kept.
    """
    if not is_daily_release_bool(release_obj):
        return dict(UNKNOWN_DICT)
    try:
        return _read_daily_cycle_dict(connection_obj, release_obj, decision_plan_id_int, as_of_ts)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, sqlite3.Error):
        return {"status_str": "unknown", "reason_str": "Saved daily cycle could not be verified"}


def _read_daily_cycle_dict(connection_obj, release_obj, decision_plan_id_int, as_of_ts):
    if decision_plan_id_int is not None and (type(decision_plan_id_int) is not int or decision_plan_id_int <= 0):
        raise _DailyDataError("Invalid cycle selector.")
    as_of_ts = as_of_ts.replace(tzinfo=UTC) if as_of_ts.tzinfo is None else as_of_ts.astimezone(UTC)
    calendar_id_str = release_obj.session_calendar_id_str
    connection_obj.row_factory = sqlite3.Row
    table_set = {row_obj[0] for row_obj in connection_obj.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    saved_release_obj = connection_obj.execute(
        "SELECT * FROM live_release WHERE release_id_str=? AND user_id_str=? AND pod_id_str=? AND account_route_str=?",
        (release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str, release_obj.account_route_str),
    ).fetchone()
    if (saved_release_obj is None or saved_release_obj["mode_str"] != release_obj.mode_str
            or saved_release_obj["strategy_import_str"] not in DAILY_STRATEGY_IMPORT_TUPLE):
        raise _DailyDataError("Release identity mismatch.")
    # The pending-daily predicate of state_store_v2: daily releases of this pod,
    # here also limited to this owner, account and mode.
    strategy_placeholder_str = ",".join("?" for _strategy_str in DAILY_STRATEGY_IMPORT_TUPLE)
    scope_sql_str = (f"FROM decision_plan d JOIN live_release l ON l.release_id_str=d.release_id_str "
        f"WHERE d.pod_id_str=? AND d.user_id_str=? AND d.account_route_str=? AND l.pod_id_str=d.pod_id_str "
        f"AND l.account_route_str=d.account_route_str AND l.mode_str=? AND l.strategy_import_str IN ({strategy_placeholder_str})")
    scope_tuple = (release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
        release_obj.mode_str, *DAILY_STRATEGY_IMPORT_TUPLE)
    recent_list = connection_obj.execute(f"SELECT d.* {scope_sql_str} ORDER BY d.decision_plan_id_int DESC LIMIT ?",
        (*scope_tuple, RECENT_DECISION_LIMIT_INT)).fetchall()
    latest_id_int = recent_list[0]["decision_plan_id_int"] if recent_list else None
    if decision_plan_id_int is None:
        selected_obj = recent_list[0] if recent_list else None
    else:
        selected_obj = connection_obj.execute(f"SELECT d.* {scope_sql_str} AND d.decision_plan_id_int=?",
            (*scope_tuple, decision_plan_id_int)).fetchone()
        if selected_obj is None:
            raise _DailyDataError("Saved cycle not found.")

    # The watchdog's owed decision: the latest completed session's decision is
    # due at the next session's open minus two minutes.
    signal_label_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, calendar_id_str,
        snapshot_ready_buffer_minutes_int=0)
    signal_session_dict = {}
    if signal_label_ts is not None:
        signal_date_str = signal_label_ts.date().isoformat()
        next_label_ts = scheduler_utils.get_next_session_label_ts(signal_label_ts, calendar_id_str)
        cutoff_ts = (scheduler_utils.get_session_open_timestamp_ts(next_label_ts, calendar_id_str)
            - timedelta(minutes=DECISION_CUTOFF_MINUTES_INT)).astimezone(UTC)
        match_obj = next((row_obj for row_obj in recent_list
            if _session_date_str(row_obj["signal_timestamp_str"]) == signal_date_str), None)
        signal_session_dict = {"session_date_str": signal_date_str, "decision_cutoff_timestamp_str": cutoff_ts.isoformat(),
            "decision_plan_id_int": None if match_obj is None else match_obj["decision_plan_id_int"],
            "decision_status_str": None if match_obj is None else _code_str(match_obj["status_str"]),
            "capsule_halt_bool": False}

    result_dict = {"status_str": "ok", "reason_str": "", "as_of_timestamp_str": as_of_ts.isoformat(),
        "current_bool": selected_obj is not None and selected_obj["decision_plan_id_int"] == latest_id_int,
        "decision_dict": {}, "vplan_dict": {}, "opening_dispatch_dict": {}, "execution_result_dict": {},
        "completion_request_list": [], "pod_alert_list": [],
        "execution_alert_list": [], "post_close_report_dict": {}, "open_previous_list": [],
        "open_previous_truncated_bool": False, "last_closed_dict": {}, "signal_session_dict": signal_session_dict,
        "session_close_timestamp_str": None, "next_open_timestamp_str": _next_open_ts(as_of_ts, calendar_id_str).isoformat()}
    if selected_obj is None:
        # A daily pod before its first saved decision: only the owed decision applies.
        result_dict["current_bool"] = True
        if signal_session_dict:
            signal_session_dict["capsule_halt_bool"] = _capsule_halt_bool(connection_obj, table_set, release_obj,
                signal_session_dict["session_date_str"])
        return result_dict

    selected_id_int = selected_obj["decision_plan_id_int"]
    for field_str in ("created_timestamp_str", "updated_timestamp_str"):
        _observed_ts(selected_obj[field_str], as_of_ts)
    decision_dict = _decision_dict(selected_obj)
    metadata_dict = _metadata_dict(selected_obj)
    result_dict.update(decision_dict=decision_dict, opening_dispatch_dict=_opening_dispatch_dict(metadata_dict),
        execution_result_dict=_execution_result_dict(metadata_dict),
        session_close_timestamp_str=_session_close_ts(decision_dict["target_execution_timestamp_str"], calendar_id_str).isoformat())

    vplan_obj = connection_obj.execute(
        "SELECT * FROM vplan WHERE decision_plan_id_int=? ORDER BY vplan_id_int DESC LIMIT 1", (selected_id_int,),
    ).fetchone()
    if vplan_obj is not None:
        if any(vplan_obj[field_str] != selected_obj[field_str] for field_str in ("pod_id_str", "account_route_str", "user_id_str")):
            raise _DailyDataError("Plan identity mismatch.")
        for field_str in ("created_timestamp_str", "updated_timestamp_str"):
            _observed_ts(vplan_obj[field_str], as_of_ts)
        vplan_id_int = vplan_obj["vplan_id_int"]
        # Opening orders only: same-day completion orders are counted separately.
        result_dict["vplan_dict"] = {"vplan_id_int": vplan_id_int, "status_str": _code_str(vplan_obj["status_str"]),
            "submission_key_str": vplan_obj["submission_key_str"],
            "submission_timestamp_str": _timestamp_str(vplan_obj["submission_timestamp_str"]),
            "target_execution_timestamp_str": _timestamp_str(vplan_obj["target_execution_timestamp_str"]),
            "missing_ack_count_int": int(vplan_obj["missing_ack_count_int"] or 0),
            "submit_ack_status_str": _code_str(vplan_obj["submit_ack_status_str"]),
            "sent_order_count_int": _count_int(connection_obj, "SELECT COUNT(*) FROM vplan_broker_order "
                "WHERE vplan_id_int=? AND COALESCE(order_request_key_str,'') NOT LIKE ?", (vplan_id_int, COMPLETION_KEY_LIKE_STR)),
            "ack_count_int": _count_int(connection_obj, "SELECT COUNT(*) FROM vplan_broker_ack WHERE vplan_id_int=? "
                "AND broker_response_ack_bool=1 AND ack_status_str='broker_acked' "
                "AND COALESCE(order_request_key_str,'') NOT LIKE ?", (vplan_id_int, COMPLETION_KEY_LIKE_STR)),
            "fill_count_int": _count_int(connection_obj, "SELECT COUNT(*) FROM vplan_fill WHERE vplan_id_int=?", (vplan_id_int,))}

    if "daily_completion_request" in table_set:
        result_dict["completion_request_list"] = [{"asset_str": _asset_str(row_obj["asset_str"]),
                "vplan_id_int": row_obj["vplan_id_int"], "claimed_timestamp_str": _timestamp_str(row_obj["claimed_timestamp_str"]),
                "send_failed_bool": bool(row_obj["send_failed_int"])}
            for row_obj in connection_obj.execute("""SELECT asset_str,vplan_id_int,claimed_timestamp_str,
                send_error_str IS NOT NULL AS send_failed_int FROM daily_completion_request
                WHERE decision_plan_id_int=? AND pod_id_str=? AND account_route_str=? ORDER BY asset_str LIMIT ?""",
                (selected_id_int, release_obj.pod_id_str, release_obj.account_route_str, ROW_LIMIT_INT))]

    halt_date_list = sorted({decision_dict["signal_session_date_str"], signal_session_dict.get("session_date_str") or ""} - {""})
    if "daily_pod_alert" in table_set:
        key_list = [f"capsule_holding_halt:{date_str}" for date_str in halt_date_list]
        key_placeholder_str = ",".join("?" for _key_str in key_list)
        for row_obj in connection_obj.execute(f"""SELECT alert_kind_str,alert_key_str,decision_plan_id_int,payload_json_str,
                created_timestamp_str,delivered_timestamp_str FROM daily_pod_alert
                WHERE pod_id_str=? AND account_route_str=? AND mode_str=?
                  AND (decision_plan_id_int=? OR alert_key_str IN ({key_placeholder_str}))
                ORDER BY created_timestamp_str DESC, alert_key_str LIMIT ?""",
                (release_obj.pod_id_str, release_obj.account_route_str, release_obj.mode_str, selected_id_int,
                    *key_list, ROW_LIMIT_INT)):
            result_dict["pod_alert_list"].append(_pod_alert_dict(row_obj))
        halt_session_set = {alert_dict["session_date_str"] for alert_dict in result_dict["pod_alert_list"]
            if alert_dict["alert_kind_str"] == "capsule_holding_halt"}
        if signal_session_dict:
            signal_session_dict["capsule_halt_bool"] = signal_session_dict["session_date_str"] in halt_session_set

    if "mr_capsule_execution_alert" in table_set:
        column_set = {row_obj[1] for row_obj in connection_obj.execute("PRAGMA table_info(mr_capsule_execution_alert)")}
        vplan_id_obj = (result_dict["vplan_dict"] or {}).get("vplan_id_int")
        filter_str, filter_tuple = "vplan_id_int=?", (vplan_id_obj,)
        if "decision_plan_id_int" in column_set:
            filter_str, filter_tuple = "(decision_plan_id_int=? OR vplan_id_int=?)", (selected_id_int, vplan_id_obj)
        decision_select_str = "decision_plan_id_int" if "decision_plan_id_int" in column_set else "NULL AS decision_plan_id_int"
        result_dict["execution_alert_list"] = [{"alert_kind_str": _code_str(row_obj["alert_kind_str"]),
                "vplan_id_int": row_obj["vplan_id_int"], "decision_plan_id_int": row_obj["decision_plan_id_int"],
                "created_timestamp_str": _timestamp_str(row_obj["created_timestamp_str"]),
                "delivered_bool": row_obj["delivered_timestamp_str"] is not None}
            for row_obj in connection_obj.execute(f"""SELECT alert_kind_str,vplan_id_int,{decision_select_str},
                created_timestamp_str,delivered_timestamp_str FROM mr_capsule_execution_alert
                WHERE pod_id_str=? AND account_route_str=? AND mode_str=? AND {filter_str}
                ORDER BY created_timestamp_str DESC, alert_kind_str LIMIT ?""",
                (release_obj.pod_id_str, release_obj.account_route_str, release_obj.mode_str, *filter_tuple, ROW_LIMIT_INT))]

    if "daily_post_close_report" in table_set:
        report_obj = connection_obj.execute(
            "SELECT status_str,attempted_timestamp_str FROM daily_post_close_report WHERE decision_plan_id_int=?",
            (selected_id_int,)).fetchone()
        if report_obj is not None:
            result_dict["post_close_report_dict"] = {"status_str": _code_str(report_obj["status_str"]),
                "attempted_timestamp_str": _timestamp_str(report_obj["attempted_timestamp_str"])}

    status_placeholder_str = ",".join("?" for _status_str in NOT_OPEN_STATUS_TUPLE)
    previous_list = connection_obj.execute(f"""SELECT d.* {scope_sql_str} AND d.decision_plan_id_int<>?
        AND d.status_str NOT IN ({status_placeholder_str})
        AND julianday(d.target_execution_timestamp_str)<=julianday(?)
        ORDER BY d.target_execution_timestamp_str, d.decision_plan_id_int LIMIT ?""",
        (*scope_tuple, selected_id_int, *NOT_OPEN_STATUS_TUPLE, selected_obj["target_execution_timestamp_str"],
            OPEN_PREVIOUS_LIMIT_INT + 1)).fetchall()
    result_dict["open_previous_truncated_bool"] = len(previous_list) > OPEN_PREVIOUS_LIMIT_INT
    for row_obj in previous_list[:OPEN_PREVIOUS_LIMIT_INT]:
        target_str = _timestamp_str(row_obj["target_execution_timestamp_str"])
        result_dict["open_previous_list"].append({"decision_plan_id_int": row_obj["decision_plan_id_int"],
            "status_str": _code_str(row_obj["status_str"]), "target_execution_timestamp_str": target_str,
            "target_session_date_str": _session_date_str(target_str),
            "session_close_timestamp_str": _session_close_ts(target_str, calendar_id_str).isoformat()})

    closed_placeholder_str = ",".join("?" for _status_str in sorted(CLOSED_STATUS_SET))
    closed_obj = connection_obj.execute(f"""SELECT d.* {scope_sql_str} AND d.decision_plan_id_int<>?
        AND d.status_str IN ({closed_placeholder_str})
        AND julianday(d.target_execution_timestamp_str)<=julianday(?)
        ORDER BY julianday(d.target_execution_timestamp_str) DESC, d.decision_plan_id_int DESC LIMIT 1""",
        (*scope_tuple, selected_id_int, *sorted(CLOSED_STATUS_SET), selected_obj["target_execution_timestamp_str"])).fetchone()
    if closed_obj is not None:
        closed_result_dict = _execution_result_dict(_metadata_dict(closed_obj))
        result_dict["last_closed_dict"] = {"decision_plan_id_int": closed_obj["decision_plan_id_int"],
            "status_str": _code_str(closed_obj["status_str"]),
            "target_session_date_str": _session_date_str(closed_obj["target_execution_timestamp_str"]),
            "completed_timestamp_str": closed_result_dict.get("completed_timestamp_str"),
            "exception_count_int": closed_result_dict.get("exception_count_int", 0)}
    return result_dict


def _pod_alert_dict(row_obj):
    payload_dict = json.loads(row_obj["payload_json_str"] or "{}")
    payload_dict = payload_dict if isinstance(payload_dict, dict) else {}
    kind_str = _code_str(row_obj["alert_kind_str"])
    alert_dict = {"alert_kind_str": kind_str, "decision_plan_id_int": row_obj["decision_plan_id_int"],
        "created_timestamp_str": _timestamp_str(row_obj["created_timestamp_str"]),
        "delivered_bool": row_obj["delivered_timestamp_str"] is not None,
        "session_date_str": None, "session_close_timestamp_str": None,
        "order_details_available_bool": None, "owned_order_row_list": []}
    if kind_str == "capsule_holding_halt":
        # Keep the session only; the reason and holdings stay in the alert outbox.
        alert_dict["session_date_str"] = str(row_obj["alert_key_str"]).removeprefix("capsule_holding_halt:")[:10]
    elif kind_str == "cycle_overdue":
        row_list = payload_dict.get("owned_order_row_list")
        alert_dict.update(session_close_timestamp_str=_timestamp_str(payload_dict.get("session_close_timestamp_str")),
            order_details_available_bool=payload_dict.get("order_details_available_bool") is True,
            # Owning client IDs and the saved error text are withheld.
            owned_order_row_list=[{"asset_str": _asset_str(item_dict.get("asset_str")),
                "side_str": item_dict.get("side_str") if item_dict.get("side_str") in {"BUY", "SELL"} else None,
                "quantity_float": _float_or_none(item_dict.get("quantity_float"))}
                for item_dict in (row_list if isinstance(row_list, list) else [])[:ROW_LIMIT_INT] if isinstance(item_dict, dict)])
    return alert_dict


def _capsule_halt_bool(connection_obj, table_set, release_obj, session_date_str):
    if "daily_pod_alert" not in table_set:
        return False
    return connection_obj.execute("""SELECT 1 FROM daily_pod_alert WHERE pod_id_str=? AND account_route_str=?
        AND mode_str=? AND alert_key_str=? LIMIT 1""", (release_obj.pod_id_str, release_obj.account_route_str,
            release_obj.mode_str, f"capsule_holding_halt:{session_date_str}")).fetchone() is not None


def load_daily_cycle_dict(target_obj, *, as_of_ts: datetime, decision_plan_id_int=None):
    """Open the pod DB read-only (URI mode=ro) and read in one BEGIN snapshot."""
    release_obj = getattr(target_obj, "release_obj", None)
    if release_obj is None or not is_daily_release_bool(release_obj):
        return dict(UNKNOWN_DICT)
    try:
        db_path_obj = Path(target_obj.db_path_str).resolve()
        # mode=ro cannot create or migrate a missing DB. Closing ends the read
        # snapshot without a write.
        with closing(sqlite3.connect(f"{db_path_obj.as_uri()}?mode=ro", uri=True, timeout=1.0)) as connection_obj:
            connection_obj.execute("BEGIN")
            return read_daily_cycle_dict(connection_obj, release_obj,
                decision_plan_id_int=decision_plan_id_int, as_of_ts=as_of_ts)
    except (OSError, TypeError, ValueError, sqlite3.Error):
        return {"status_str": "unknown", "reason_str": "Saved daily cycle could not be verified"}
