"""Read-only timing projection for a current Pod, separate from saved cycles."""

from datetime import timedelta

from alpha.live import scheduler_utils
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v3.schedule import build_trading_window_list
from alpha.live.ops_report import parse_timestamp_ts
from alpha.live.scheduler_service import DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT


def _time_str(timestamp_ts, now_ts):
    market_ts = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ)
    now_market_ts = now_ts.astimezone(MARKET_TIMEZONE_OBJ)
    format_str = "%H:%M:%S" if market_ts.date() == now_market_ts.date() else (
        "%m-%d %H:%M:%S" if market_ts.year == now_market_ts.year else "%Y-%m-%d %H:%M:%S")
    return market_ts.strftime(format_str)


def build_next_operation_dict(row_dict, cycle_dict, *, now_ts, action_required_bool=False):
    """Retain current obligations; forecast only a verified waiting/completed Pod.

    Calendar times are expected operations, never evidence of a running service.
    A Decision's close is only its earliest boundary: data readiness is unknown.
    """
    next_str = cycle_dict["next_str"]
    next_time_str = cycle_dict["next_time_str"]
    next_timestamp_str = cycle_dict["next_timestamp_str"]
    if next_str == "—":
        next_str = "Time unknown" if cycle_dict["pill_str"] == "Unknown" else "Not scheduled"
    elif not next_time_str:
        next_time_str = "Time unknown"
    result_dict = {"next_str": next_str, "next_time_str": next_time_str,
        "next_timestamp_str": next_timestamp_str, "next_detail_str": "", "next_forecast_bool": False}
    if (action_required_bool or cycle_dict["stale_bool"] or cycle_dict["cycle_role_str"] != "current"
            or cycle_dict["pill_str"] not in {"Waiting", "On track"}
            or row_dict.get("next_action_str") != "wait"
            or row_dict.get("latest_decision_plan_status_str") != "completed"
            or row_dict.get("latest_vplan_status_str") != "completed"
            or any(step_dict["state_str"] not in {"Done", "None"} for step_dict in cycle_dict["step_dict_list"][:-1])):
        return result_dict

    eod_step_dict = cycle_dict["step_dict_list"][-1]
    if eod_step_dict["state_str"] not in {"Done", "None", "Planned"}:
        return result_dict
    candidate_list = []
    eod_ts = parse_timestamp_ts(eod_step_dict["planned_timestamp_str"])
    if eod_step_dict["state_str"] == "Planned" and eod_ts is not None:
        candidate_list.append((eod_ts, {**result_dict, "next_detail_str": "Scheduled"}))
    try:
        calendar_id_str = str(row_dict.get("session_calendar_id_str") or "")
        calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
        market_ts = scheduler_utils.to_market_timestamp_ts(now_ts, calendar_id_str)
        if eod_step_dict["state_str"] in {"Done", "None"}:
            session_ts = calendar_obj.date_to_session(market_ts.date(), direction="next")
            eod_ts = scheduler_utils.get_session_close_timestamp_ts(session_ts, calendar_id_str) + timedelta(
                minutes=DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT)
            if eod_ts <= now_ts and eod_step_dict["state_str"] == "Done":
                session_ts = scheduler_utils.get_next_session_label_ts(session_ts, calendar_id_str)
                eod_ts = scheduler_utils.get_session_close_timestamp_ts(session_ts, calendar_id_str) + timedelta(
                    minutes=DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT)
            if eod_ts > now_ts:
                candidate_list.append((eod_ts, {"next_str": "EOD", "next_time_str": _time_str(eod_ts, now_ts),
                    "next_timestamp_str": eod_ts.isoformat(), "next_detail_str": "Scheduled",
                    "next_forecast_bool": True}))

        # Reuse the same read-only calendar forecast as V3. Do not call the
        # scheduler's next_due/get_scheduler_decision: those synchronize state.
        window_list = build_trading_window_list({"pod_row_dict_list": [row_dict]}, mode_str="live", now_dt=now_ts)
        window_obj = window_list[0] if len(window_list) == 1 else None
        if window_obj and window_obj.action_required_bool:
            return {"next_str": "Review saved evidence", "next_time_str": "", "next_timestamp_str": "",
                "next_detail_str": "you · now", "next_forecast_bool": False,
                "schedule_action_dict": {"severity_str": window_obj.severity_str,
                    "label_str": window_obj.status_label_str, "reason_str": window_obj.detail_str,
                    "timestamp_str": window_obj.submission_timestamp_str}}
        signal_ts = parse_timestamp_ts(window_obj.signal_timestamp_str) if window_obj and window_obj.has_data_bool else None
        if signal_ts is not None and not window_obj.action_required_bool and window_obj.action_str == "wait":
            candidate_list.append((signal_ts, {"next_str": "Decide", "next_time_str": "after " + _time_str(signal_ts, now_ts),
                # No exact deadline or countdown: vendor readiness is not known.
                "next_timestamp_str": "", "next_detail_str": "Scheduled · when data is ready",
                "next_forecast_bool": True}))
    except (ValueError, TypeError, KeyError, OverflowError):
        # Keep a verified saved EOD time if calendar metadata is unavailable.
        pass
    if candidate_list:
        return min(candidate_list, key=lambda item_tuple: item_tuple[0])[1]
    if next_str == "Not scheduled":
        result_dict.update(next_str="Time unknown", next_detail_str="Schedule unavailable")
    return result_dict
