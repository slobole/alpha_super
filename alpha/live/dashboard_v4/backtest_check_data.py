"""Read-only V4 view of the nightly backtest check report (alpha/live/backtest_check.py).

The job writes `backtest_check_latest.json` next to the watchdog report; this module only
reads it (shared read, size cap, schema check, LIVE identity check) and turns it into
small display dicts. Nothing here runs a backtest, opens a pod database or writes.
Every view is optional: with no report the pages render exactly as before.
"""

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from alpha.live.dashboard_v4.console_data import open_shared_read_obj

SCHEMA_VERSION_STR = "backtest_check.v1"
REPORT_FILE_NAME_STR = "backtest_check_latest.json"
REPORT_BYTE_LIMIT_INT = 4 * 1024 * 1024
# The job runs on session days at about 08:00 ET; a report is current until the next session's 10:00 ET.
STALE_GRACE_HOUR_INT = 10
POD_LIMIT_INT = 64
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
BACKTEST_COLOR_STR = "#8891a0"
_CACHE_DICT = {}


# ---------------------------------------------------------------------------------------------- loading
def _report_path(provider_obj):
    for source_obj in (provider_obj, getattr(provider_obj, "app_obj", lambda: None)()):
        path_str = getattr(source_obj, "event_log_path_str", None) if source_obj is not None else None
        if isinstance(path_str, str) and path_str:
            return Path(path_str).parent / REPORT_FILE_NAME_STR
    return None


def _live_identity_set(provider_obj):
    source_obj = provider_obj if callable(getattr(provider_obj, "get_target_list", None)) else provider_obj.app_obj()
    return {(target_obj.release_obj.pod_id_str, target_obj.release_obj.account_route_str, target_obj.release_obj.release_id_str)
            for target_obj in source_obj.get_target_list()
            if target_obj.release_obj.mode_str == "live" and target_obj.release_obj.enabled_bool is True}


def _timestamp_ts(value_obj):
    try:
        timestamp_ts = datetime.fromisoformat(str(value_obj).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return timestamp_ts if timestamp_ts.tzinfo is not None else None


def _expected_date_str(as_of_ts):
    """The latest session date whose STALE_GRACE_HOUR_INT ET has passed: a current report is at least this new."""
    import exchange_calendars as xcals

    market_ts = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ)
    session_list = [ts.date() for ts in xcals.get_calendar("XNYS").sessions_in_range(
        (market_ts.date() - timedelta(days=10)).isoformat(), market_ts.date().isoformat())]
    passed_list = [day_obj for day_obj in session_list
                   if day_obj < market_ts.date() or market_ts.hour >= STALE_GRACE_HOUR_INT]
    return passed_list[-1].isoformat() if passed_list else market_ts.date().isoformat()


def _stale_bool(payload_dict, generated_ts, as_of_ts):
    report_date_str = payload_dict.get("as_of_date_str")
    if not isinstance(report_date_str, str):
        report_date_str = generated_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
    try:
        return report_date_str < _expected_date_str(as_of_ts)
    except Exception:  # calendar unavailable: fall back to age
        return (as_of_ts - generated_ts).total_seconds() > 4 * 86400


def _valid_report_bool(payload_dict):
    if not isinstance(payload_dict, dict) or payload_dict.get("schema_version_str") != SCHEMA_VERSION_STR:
        return False
    pod_list = payload_dict.get("pod_list")
    if not isinstance(pod_list, list) or len(pod_list) > POD_LIMIT_INT or any(not isinstance(pod_dict, dict) for pod_dict in pod_list):
        return False
    identity_list = []
    for pod_dict in pod_list:
        if any(not isinstance(pod_dict.get(field_str), str) or not pod_dict[field_str]
               for field_str in ("pod_id_str", "account_route_str", "release_id_str")):
            return False
        identity_list.append(pod_dict["pod_id_str"])
        if not isinstance(pod_dict.get("decision_dict"), dict) or not isinstance(pod_dict.get("curve_dict", {}), dict):
            return False
        decision_dict, curve_dict = pod_dict["decision_dict"], pod_dict.get("curve_dict", {})
        if (not isinstance(decision_dict.get("status_str"), str) or decision_dict["status_str"] not in {"same", "differs", "not_checked"}
                or not isinstance(pod_dict.get("reference_status_str"), str)
                or pod_dict["reference_status_str"] not in {"ok", "incomplete", "skipped", "failed", "not_run"}):
            return False
        for row_list in (decision_dict.get("row_list", []), pod_dict.get("record_list", [])):
            if not isinstance(row_list, list) or any(not isinstance(row_dict, dict) for row_dict in row_list):
                return False
        for record_dict in pod_dict.get("record_list", []):
            if (not isinstance(record_dict.get("status_str"), str)
                    or record_dict["status_str"] not in {"same", "differs", "not_checked"}):
                return False
            date.fromisoformat(record_dict["session_str"])
        if any(row_dict.get("same_bool") is not None and type(row_dict.get("same_bool")) is not bool
               for row_dict in decision_dict.get("row_list", [])):
            return False
        point_list = curve_dict.get("point_list", [])
        if not isinstance(point_list, list):
            return False
        date_list = []
        for point_dict in point_list:
            if (not isinstance(point_dict, dict) or not _finite(point_dict.get("live_float"))
                    or not _finite(point_dict.get("backtest_float")) or point_dict["live_float"] <= -1
                    or point_dict["backtest_float"] <= -1):
                return False
            date_list.append(date.fromisoformat(point_dict["date_str"]).isoformat())
        if date_list != sorted(set(date_list)):
            return False
        rare_dict = pod_dict.get("rare_loss_dict")
        if rare_dict is not None and (not isinstance(rare_dict, dict) or type(rare_dict.get("window_int")) is not int
                                      or rare_dict["window_int"] not in {1, 3, 5}):
            return False
    gate_dict = payload_dict.get("cost_gate_dict")
    if gate_dict is not None:
        if (not isinstance(gate_dict, dict) or not isinstance(gate_dict.get("status_str"), str)
                or gate_dict["status_str"] not in {"collecting", "on_track", "above", "passed", "failed", "time_cap"}
                or type(gate_dict.get("fill_count_int")) is not int or gate_dict["fill_count_int"] < 0):
            return False
        for field_str in ("cost_bps_list", "pod_row_list", "pod_identity_list"):
            if not isinstance(gate_dict.get(field_str, []), list):
                return False
        if any(not _finite(cost_float) for cost_float in gate_dict.get("cost_bps_list", [])):
            return False
        for row_dict in gate_dict.get("pod_row_list", []):
            if not isinstance(row_dict, dict) or not isinstance(row_dict.get("pod_id_str"), str) or not row_dict["pod_id_str"]:
                return False
        for identity_dict in gate_dict.get("pod_identity_list", []):
            if not isinstance(identity_dict, dict) or any(not isinstance(identity_dict.get(field_str), str) or not identity_dict[field_str]
                for field_str in ("pod_id_str", "account_route_str", "release_id_str")):
                return False
    return len(identity_list) == len(set(identity_list)) and _timestamp_ts(payload_dict.get("generated_at_utc_str")) is not None


def _dated_evidence_dict(evidence_dict, generated_ts, as_of_ts):
    checked_ts = _timestamp_ts(evidence_dict.get("checked_at_utc_str", generated_ts.isoformat()))
    if checked_ts is None or checked_ts > as_of_ts:
        raise ValueError("Invalid evidence timestamp.")
    checked_date_str = evidence_dict.get("checked_as_of_date_str")
    if checked_date_str is None:
        checked_date_str = checked_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
    if date.fromisoformat(checked_date_str) > as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date():
        raise ValueError("Future evidence date.")
    return {**evidence_dict, "generated_ts": checked_ts,
            "stale_bool": _stale_bool({"as_of_date_str": checked_date_str}, checked_ts, as_of_ts)}


def load_report_dict(provider_obj, *, as_of_ts):
    """The validated report scoped to the current LIVE pods, or None. Never raises."""
    try:
        if callable(getattr(provider_obj, "get_backtest_check_report_dict", None)):
            payload_dict = provider_obj.get_backtest_check_report_dict(as_of_ts=as_of_ts)  # demo
            identity_set = None
        else:
            path_obj = _report_path(provider_obj)
            if path_obj is None or not path_obj.is_file():
                return None
            stat_obj = path_obj.stat()
            key_tuple = (str(path_obj), stat_obj.st_mtime_ns, stat_obj.st_size)
            payload_dict = _CACHE_DICT.get(key_tuple)
            if payload_dict is None:
                if stat_obj.st_size > REPORT_BYTE_LIMIT_INT:
                    return None
                # The job replaces the file atomically; a shared read never blocks that rename.
                with open_shared_read_obj(path_obj) as file_obj:
                    content_bytes = file_obj.read(REPORT_BYTE_LIMIT_INT + 1)
                if len(content_bytes) > REPORT_BYTE_LIMIT_INT:
                    return None
                payload_dict = json.loads(content_bytes)
                _CACHE_DICT.clear()
                _CACHE_DICT[key_tuple] = payload_dict
            identity_set = _live_identity_set(provider_obj)
        if not _valid_report_bool(payload_dict):
            return None
        generated_ts = _timestamp_ts(payload_dict["generated_at_utc_str"])
        if generated_ts > as_of_ts.astimezone(timezone.utc):
            return None
        pod_list = [pod_dict for pod_dict in payload_dict["pod_list"]
                    if identity_set is None or (pod_dict.get("pod_id_str"), pod_dict.get("account_route_str"),
                                                pod_dict.get("release_id_str")) in identity_set]
        pod_list = [_dated_evidence_dict(pod_dict, generated_ts, as_of_ts) for pod_dict in pod_list]
        gate_dict = payload_dict.get("cost_gate_dict")
        if gate_dict is not None:
            if not isinstance(gate_dict, dict):
                return None
            gate_dict = _dated_evidence_dict(gate_dict, generated_ts, as_of_ts)
            if identity_set is not None:
                gate_pod_set = {row_dict["pod_id_str"] for row_dict in gate_dict.get("pod_row_list", [])}
                gate_identity_list = gate_dict.get("pod_identity_list") or [pod_dict for pod_dict in payload_dict["pod_list"]
                                                                          if pod_dict["pod_id_str"] in gate_pod_set]
                if (not gate_pod_set or {pod_dict["pod_id_str"] for pod_dict in gate_identity_list} != gate_pod_set
                        or any((pod_dict["pod_id_str"], pod_dict["account_route_str"], pod_dict["release_id_str"])
                               not in identity_set for pod_dict in gate_identity_list)):
                    gate_dict = None
        result_dict = {**payload_dict, "pod_list": pod_list, "cost_gate_dict": gate_dict, "generated_ts": generated_ts,
                       "stale_bool": _stale_bool(payload_dict, generated_ts, as_of_ts)}
        # *** CRITICAL*** build every view once here: a malformed or hand-edited report is dropped
        # whole instead of failing a page later.
        for pod_dict in pod_list:
            pod_view_dict(result_dict, pod_dict.get("pod_id_str"))
        performance_view_dict(result_dict, {})
        system_row_dict(result_dict, as_of_ts=as_of_ts)
        return result_dict
    except Exception:  # report only: any unreadable report is simply not shown
        return None


# ---------------------------------------------------------------------------------------------- formatting
def _finite(value_obj):
    return isinstance(value_obj, (int, float)) and not isinstance(value_obj, bool) and math.isfinite(value_obj)


def _pct_str(value_obj, digits_int=2):
    if not _finite(value_obj):
        return "—"
    return ("+" if value_obj >= 0 else "−") + f"{abs(value_obj) * 100:.{digits_int}f}%"


def _bps_str(value_obj):
    return f"{value_obj:.1f}" if _finite(value_obj) else "—"


def _day_str(date_str):
    try:
        return date.fromisoformat(str(date_str)[:10]).strftime("%a %m-%d")
    except ValueError:
        return "—"


def _time_str(timestamp_ts):
    return timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%a %m-%d %H:%M") if timestamp_ts else "—"


def _text(value_obj, limit_int=200):
    return str(value_obj)[:limit_int] if isinstance(value_obj, str) else ""


# ---------------------------------------------------------------------------------------------- Pod page
def _list(value_obj):
    return value_obj if isinstance(value_obj, list) else []


def _record_view_dict(record_list):
    mark_list = [item_dict for item_dict in _list(record_list) if isinstance(item_dict, dict)][-60:]
    checked_list = [item_dict for item_dict in mark_list if item_dict.get("status_str") in {"same", "differs"}]
    same_int = sum(1 for item_dict in checked_list if item_dict["status_str"] == "same")
    return {"mark_list": [{"class_str": {"same": "", "differs": "bad"}.get(item_dict.get("status_str"), "none"),
                           "title_str": _day_str(item_dict.get("session_str")) + " · " + {"same": "same", "differs": "differs"}.get(item_dict.get("status_str"), "not checked")}
                          for item_dict in mark_list],
            "same_int": same_int, "checked_int": len(checked_list), "missing_int": len(mark_list) - len(checked_list),
            "start_str": _day_str(mark_list[0].get("session_str")) if mark_list else "—"}


def rare_view_dict(rare_dict):
    if not isinstance(rare_dict, dict) or not _finite(rare_dict.get("live_float")):
        return None
    window_int = rare_dict.get("window_int") if type(rare_dict.get("window_int")) is int else 0
    if window_int not in {1, 3, 5}:
        return None
    days_str = "1 day" if window_int == 1 else f"{window_int} days"
    cause_str = rare_dict.get("cause_str")
    return {"state_str": "fail" if cause_str == "ours" else "late",
            "title_str": f"Rare loss · {_pct_str(rare_dict['live_float'])} in {days_str}.",
            "detail_str": (f"Worse than 99 of 100 {days_str} windows in the backtest (1-in-100 line {_pct_str(rare_dict.get('line_float'), 1)})."
                           + (" Beyond the backtest's worst." if rare_dict.get("beyond_worst_bool") else "")),
            "cause_str": cause_str,
            "sub_str": f"Backtest, same days {_pct_str(rare_dict.get('backtest_float'))} · "
                       + {"strategy": "the strategy did this, not a fault",
                          "ours": "the backtest did not lose this: look on our side (Plan vs actual, fills)",
                          }.get(cause_str, "the backtest for these days is not available")}


def pod_view_dict(report_dict, pod_id_str):
    """Everything the Pod page shows for one pod, or None when the report has nothing for it."""
    if not report_dict:
        return None
    pod_dict = next((item_dict for item_dict in report_dict["pod_list"] if item_dict.get("pod_id_str") == pod_id_str), None)
    if pod_dict is None:
        return None
    decision_dict = pod_dict.get("decision_dict") if isinstance(pod_dict.get("decision_dict"), dict) else {}
    status_str = decision_dict.get("status_str") if decision_dict.get("status_str") in {"same", "differs"} else "not_checked"
    row_list = [row_dict for row_dict in _list(decision_dict.get("row_list")) if isinstance(row_dict, dict)][:200]
    differ_int = sum(1 for row_dict in row_list if row_dict.get("same_bool") is False)
    stale_bool = report_dict["stale_bool"] or pod_dict.get("stale_bool", False)
    if status_str == "differs":
        title_str, state_str = f"{differ_int} of {len(row_list)} targets differ from the backtest", "fail"
    elif status_str == "same":
        title_str, state_str = "Same as the backtest", "done"
    else:
        title_str, state_str = "Not checked", "unk"
    if stale_bool:
        status_str, state_str, title_str = "not_checked", "unk", "Historical check · not current"
    curve_dict = pod_dict.get("curve_dict") if isinstance(pod_dict.get("curve_dict"), dict) else {}
    point_list = [point_dict for point_dict in _list(curve_dict.get("point_list"))
                  if isinstance(point_dict, dict) and isinstance(point_dict.get("date_str"), str)
                  and _finite(point_dict.get("live_float")) and _finite(point_dict.get("backtest_float"))]
    session_str = _text(decision_dict.get("session_str") or pod_dict.get("session_str"), 10)
    return {
        "status_str": status_str, "state_str": state_str, "title_str": title_str, "differ_int": differ_int,
        "detail_str": " · ".join(part_str for part_str in (
            f"session {_day_str(session_str)}" if session_str else "",
            f"checked {_time_str(pod_dict.get('generated_ts', report_dict['generated_ts']))}",
            "report is not from the latest session" if stale_bool else "") if part_str),
        "reason_str": ("Evidence is not from the latest session." if stale_bool else
                       _text(decision_dict.get("reason_str") or pod_dict.get("reference_reason_str"))),
        "coverage_reason_str": _text(pod_dict.get("coverage_reason_str") or curve_dict.get("coverage_reason_str")),
        "ruler_reason_str": _text(pod_dict.get("ruler_reason_str")),
        "basis_str": _text(decision_dict.get("basis_str")),
        "data_str": " · ".join(part_str for part_str in (
            "Norgate snapshot " + _text(decision_dict.get("data_snapshot_str"), 20) if decision_dict.get("data_snapshot_str") else "",
            _text(decision_dict.get("data_hash_str"), 12)) if part_str),
        "table_dict": {"column_list": ["Symbol", "Live decision", "Backtest", "Same?"],
                       "row_list": [{"cell_list": [_text(row_dict.get("asset_str"), 20), _text(row_dict.get("live_str"), 30),
                                                   _text(row_dict.get("backtest_str"), 30)],
                                     "state_str": "" if row_dict.get("same_bool") else "unk" if stale_bool or row_dict.get("same_bool") is None else "fail",
                                     "match_str": "unk" if stale_bool or row_dict.get("same_bool") is None else "done" if row_dict.get("same_bool") else "fail"} for row_dict in row_list],
                       "note_str": "", "empty_str": _text(decision_dict.get("reason_str")) or "Nothing to compare for this session."},
        "record_dict": _record_view_dict(pod_dict.get("record_list")),
        "point_list": point_list, "gap_str": _pct_str(curve_dict.get("gap_float")),
        "gap_month_str": _pct_str(curve_dict.get("gap_month_float")),
        "gap_float": curve_dict.get("gap_float") if _finite(curve_dict.get("gap_float")) else None,
        "live_str": _pct_str(point_list[-1]["live_float"]) if point_list else "—",
        "backtest_str": _pct_str(point_list[-1]["backtest_float"]) if point_list else "—",
        "start_str": _day_str(curve_dict.get("start_str")) if curve_dict.get("start_str") else "—",
        "end_str": _day_str(point_list[-1]["date_str"]) if point_list else "—",
        "curve_period_str": (_day_str(point_list[0]["date_str"]) + " through " + _day_str(point_list[-1]["date_str"])) if point_list else "—",
        "rare_dict": None if stale_bool else rare_view_dict(pod_dict.get("rare_loss_dict")),
    }


def add_backtest_series(chart_dict, return_path_list, view_dict):
    """Second line on the Pod Value chart: the backtest's return on the same days, from the chart's own start."""
    from alpha.live.dashboard_v4.charts import _drawing_dict

    if not chart_dict.get("available_bool") or not view_dict or len(view_dict["point_list"]) < 2 or not return_path_list:
        return chart_dict
    backtest_by_date_dict = {point_dict["date_str"]: point_dict["backtest_float"] for point_dict in view_dict["point_list"]}
    live_segment, test_segment, base_float = [], [], None
    first_live_str = next((row_dict["market_date_str"] for row_dict in return_path_list
                           if not str(row_dict.get("market_date_str", "")).endswith(" SOD")), None)
    if first_live_str is None:
        return chart_dict
    if any(str(row_dict.get("market_date_str", "")) not in backtest_by_date_dict for row_dict in return_path_list
           if not str(row_dict.get("market_date_str", "")).endswith(" SOD")):
        return chart_dict
    earlier_list = [date_str for date_str in backtest_by_date_dict if date_str < first_live_str]
    base_float = backtest_by_date_dict[max(earlier_list)] if earlier_list else 0.0
    if not _finite(base_float) or base_float <= -1:
        return chart_dict
    test_segment_list, current_list = [], []
    for row_dict in return_path_list:
        date_str, value_obj = str(row_dict.get("market_date_str", "")), row_dict.get("cumulative_return_float")
        if not _finite(value_obj):
            continue
        live_segment.append({"market_date_str": date_str, "value_float": value_obj})
        if date_str.endswith(" SOD"):
            current_list.append({"market_date_str": date_str, "value_float": 0.0})
        elif date_str in backtest_by_date_dict:
            current_list.append({"market_date_str": date_str, "value_float": (1 + backtest_by_date_dict[date_str]) / (1 + base_float) - 1})
        elif current_list:
            test_segment_list.append(current_list)
            current_list = []
    if current_list:
        test_segment_list.append(current_list)
    if not any(len(segment_list) > 1 for segment_list in test_segment_list):
        return chart_dict
    drawing_dict = _drawing_dict(
        [{"segment_list": [live_segment], "name_str": "Live"},
         {"segment_list": test_segment_list, "name_str": "Backtest, same days", "color_str": BACKTEST_COLOR_STR,
          "area_bool": False, "dash_bool": True}],
        [row_dict["market_date_str"] for row_dict in return_path_list], unit_str="pct")
    if not drawing_dict.get("available_bool"):
        return chart_dict
    return {**chart_dict, "drawing_dict": drawing_dict, "backtest_bool": True}


# ---------------------------------------------------------------------------------------------- Overview, Performance, System
def gate_view_dict(report_dict):
    gate_dict = (report_dict or {}).get("cost_gate_dict")
    if not isinstance(gate_dict, dict):
        return None
    count_int = gate_dict.get("fill_count_int") if type(gate_dict.get("fill_count_int")) is int else 0
    target_int = gate_dict.get("fill_target_int") if type(gate_dict.get("fill_target_int")) is int and gate_dict.get("fill_target_int") > 0 else 200
    status_str = _text(gate_dict.get("status_str"), 20)
    average_obj = gate_dict.get("average_bps_float")
    label_dict = {"collecting": ("next", "Collecting fills"), "on_track": ("done", "On track"), "above": ("late", "Cost above the gate"),
                   "passed": ("now", "Cost component passed"), "failed": ("fail", "Cost component failed"), "time_cap": ("late", "Time cap reached")}
    state_str, label_str = label_dict.get(status_str, ("unk", "Unknown"))
    stale_bool = bool((report_dict or {}).get("stale_bool") or gate_dict.get("stale_bool"))
    if stale_bool:
        state_str, label_str = "unk", "Historical cost component · not current"
    cost_list = [value_obj for value_obj in _list(gate_dict.get("cost_bps_list")) if _finite(value_obj)][:2000]
    return {"status_str": status_str, "state_str": state_str, "label_str": label_str, "stale_bool": stale_bool, "count_int": count_int,
            "target_int": target_int, "gate_str": _bps_str(gate_dict.get("gate_bps_float")),
            "average_str": _bps_str(average_obj) if count_int >= 30 else "—",
            "buy_str": _bps_str(gate_dict.get("buy_bps_float")), "sell_str": _bps_str(gate_dict.get("sell_bps_float")),
            "range_str": (_bps_str(gate_dict.get("range_low_bps_float")) + "–" + _bps_str(gate_dict.get("range_high_bps_float")))
                         if _finite(gate_dict.get("range_low_bps_float")) else "—",
            "parking_int": gate_dict.get("parking_fill_count_int") or 0, "missing_open_int": gate_dict.get("missing_open_count_int") or 0,
            "time_cap_str": _day_str(gate_dict.get("time_cap_date_str")) if gate_dict.get("time_cap_date_str") else "—",
            "progress_float": max(0.0, min(100.0, count_int / target_int * 100)),
            "cost_bar_float": max(0.0, min(100.0, (average_obj / 8.0 * 100) if _finite(average_obj) else 0.0)),
            "halt_str": "Cost component only. Unexplained halts require a separate review; this is not full gate approval.",
            "pod_row_list": [{"pod_id_str": _text(row_dict.get("pod_id_str"), 120), "fill_count_int": row_dict.get("fill_count_int") or 0,
                              "buy_str": _bps_str(row_dict.get("buy_bps_float")), "sell_str": _bps_str(row_dict.get("sell_bps_float")),
                              "all_str": _bps_str(row_dict.get("all_bps_float")), "worst_str": _bps_str(row_dict.get("worst_bps_float"))}
                             for row_dict in _list(gate_dict.get("pod_row_list")) if isinstance(row_dict, dict)][:16],
            "cost_count_int": len(cost_list), "chart_dict": _gate_chart_dict(cost_list, gate_dict.get("gate_bps_float"))}


def _gate_chart_dict(cost_list, gate_obj):
    """Per-fill cost dots, running average and the gate line, on a 0..200 fills axis (SVG coordinates)."""
    width_float, height_float, low_float, high_float = 900.0, 180.0, -6.0, 12.0
    target_float = max(200.0, float(len(cost_list)))

    def x_float(index_int):
        return 40 + index_int / target_float * (width_float - 50)

    def y_float(value_float):
        return 10 + (high_float - max(low_float, min(high_float, value_float))) / (high_float - low_float) * (height_float - 30)
    running_list, total_float = [], 0.0
    for index_int, value_float in enumerate(cost_list, start=1):
        total_float += value_float
        running_list.append(f"{x_float(index_int):.1f},{y_float(total_float / index_int):.1f}")
    return {"width_float": width_float, "height_float": height_float,
            "dot_list": [{"x_float": round(x_float(index_int), 1), "y_float": round(y_float(value_float), 1)}
                         for index_int, value_float in enumerate(cost_list, start=1)],
            "early_str": " ".join(running_list[:30]), "line_str": " ".join(running_list[29:]),
            "gate_y_float": round(y_float(gate_obj if _finite(gate_obj) else 4.0), 1),
            "zero_y_float": round(y_float(0.0), 1),
            "tick_list": [{"y_float": round(y_float(value_int), 1), "label_str": str(value_int).replace("-", "−")} for value_int in (-5, 0, 5, 10)],
            "x_tick_list": [{"x_float": round(x_float(value_int), 1), "label_str": str(value_int)} for value_int in (0, 50, 100, 150, 200)],
            "todo_x_float": round(x_float(len(cost_list)), 1), "end_x_float": round(x_float(target_float), 1)}


def attention_list(report_dict, name_by_pod_dict):
    """Overview rows: only red and amber, never a normal day."""
    if not report_dict or report_dict["stale_bool"]:
        return []
    result_list = []
    for pod_dict in report_dict["pod_list"]:
        pod_id_str = pod_dict.get("pod_id_str")
        if pod_id_str not in name_by_pod_dict:
            continue
        view_dict = pod_view_dict(report_dict, pod_id_str)
        if view_dict["status_str"] == "differs":
            result_list.append({"state_str": "fail", "pod_id_str": pod_id_str, "pod_name_str": name_by_pod_dict[pod_id_str],
                                "title_str": "Decision differs from the backtest.",
                                "detail_str": f"{view_dict['differ_int']} of {len(view_dict['table_dict']['row_list'])} targets · {view_dict['detail_str']}",
                                "age_str": "—", "kind_str": "backtest_check", "next_command_list": [], "tab_str": "backtest"})
        rare_dict = view_dict["rare_dict"]
        if rare_dict and rare_dict["cause_str"] == "ours":
            result_list.append({"state_str": "fail", "pod_id_str": pod_id_str, "pod_name_str": name_by_pod_dict[pod_id_str],
                                "title_str": rare_dict["title_str"], "detail_str": rare_dict["sub_str"], "age_str": "—",
                                "kind_str": "rare_loss", "next_command_list": [], "tab_str": "backtest"})
    gate_dict = gate_view_dict(report_dict)
    if gate_dict and not gate_dict["stale_bool"] and gate_dict["status_str"] in {"above", "failed", "time_cap"}:
        pod_id_str = next((row_dict["pod_id_str"] for row_dict in gate_dict["pod_row_list"] if row_dict["pod_id_str"] in name_by_pod_dict), None)
        if pod_id_str:
            result_list.append({"state_str": {"above": "late", "time_cap": "late", "failed": "fail"}[gate_dict["status_str"]],
                                "pod_id_str": pod_id_str, "pod_name_str": "MR capsule",
                                "title_str": {"above": "MR cost above the gate.", "failed": "MR cost gate failed.",
                                              "time_cap": "MR cost gate reached its time cap."}[gate_dict["status_str"]],
                                "detail_str": f"{gate_dict['average_str']} bps per side · {gate_dict['count_int']} of {gate_dict['target_int']} fills.",
                                "age_str": "—", "kind_str": "cost_gate", "next_command_list": [], "level_str": "backtest"})
    return result_list


def performance_view_dict(report_dict, name_by_pod_dict):
    pod_row_list, card_list = [], []
    same_int = checked_int = 0
    for pod_dict in report_dict["pod_list"]:
        pod_id_str = pod_dict.get("pod_id_str")
        view_dict = pod_view_dict(report_dict, pod_id_str)
        record_dict = view_dict["record_dict"]
        same_int += record_dict["same_int"]
        checked_int += record_dict["checked_int"]
        rare_dict = view_dict["rare_dict"]
        now_state_str, now_str = view_dict["state_str"], view_dict["title_str"]
        if rare_dict:
            now_state_str, now_str = rare_dict["state_str"], rare_dict["title_str"].rstrip(".")
        if view_dict["status_str"] == "not_checked" and view_dict["reason_str"]:
            now_str = "Not checked · " + view_dict["reason_str"]
        elif not rare_dict and view_dict["coverage_reason_str"]:
            now_state_str, now_str = "late", view_dict["coverage_reason_str"]
        name_str = name_by_pod_dict.get(pod_id_str, pod_id_str)
        pod_row_list.append({"pod_id_str": pod_id_str, "name_str": name_str, "decision_str": f"{record_dict['same_int']} of {record_dict['checked_int']}",
                             "decision_bad_bool": record_dict["same_int"] < record_dict["checked_int"],
                             "live_str": view_dict["live_str"], "backtest_str": view_dict["backtest_str"],
                             "gap_str": view_dict["gap_str"], "gap_month_str": view_dict["gap_month_str"],
                             "gap_warn_bool": view_dict["gap_float"] is not None and view_dict["gap_float"] < -0.008,
                             "rare_bool": bool(rare_dict), "now_state_str": now_state_str, "now_str": now_str,
                             "highlight_bool": view_dict["status_str"] == "differs"})
        card_list.append({"pod_id_str": pod_id_str, "name_str": name_str, "gap_str": view_dict["gap_str"],
                           "curve_period_str": view_dict["curve_period_str"],
                          "gap_warn_bool": pod_row_list[-1]["gap_warn_bool"], "live_str": view_dict["live_str"],
                          "backtest_str": view_dict["backtest_str"], "spark_dict": _spark_dict(view_dict["point_list"])})
    differ_int = sum(1 for row_dict in pod_row_list if row_dict["highlight_bool"])
    return {"pod_row_list": pod_row_list, "card_list": card_list, "same_int": same_int, "checked_int": checked_int,
            "differ_int": differ_int, "rare_int": sum(1 for row_dict in pod_row_list if row_dict["rare_bool"]),
            "generated_str": _time_str(report_dict["generated_ts"]), "stale_bool": report_dict["stale_bool"],
            "gate_dict": gate_view_dict(report_dict)}


def _spark_dict(point_list):
    if len(point_list) < 2:
        return None
    value_list = [point_dict["live_float"] for point_dict in point_list] + [point_dict["backtest_float"] for point_dict in point_list]
    low_float, high_float = min(value_list), max(value_list)
    span_float = (high_float - low_float) or 1.0

    def point_str(field_str):
        return " ".join(f"{4 + index_int / (len(point_list) - 1) * 212:.1f},{60 - (point_dict[field_str] - low_float) / span_float * 56:.1f}"
                        for index_int, point_dict in enumerate(point_list))
    return {"live_str": point_str("live_float"), "backtest_str": point_str("backtest_float")}


def system_row_dict(report_dict, *, as_of_ts):
    """Collector entry for System health · Runs on a schedule (only when the report file exists)."""
    if report_dict is None:
        return {"state_str": "unknown", "now_str": "Report unreadable", "last_timestamp_str": "",
                "expected_str": "Session days ~08:00 ET", "detail_str": "", "checked_bool": True}
    pod_list = report_dict["pod_list"]
    differ_int = sum(1 for pod_dict in pod_list if (pod_dict.get("decision_dict") or {}).get("status_str") == "differs")
    failed_int = sum(1 for pod_dict in pod_list if pod_dict.get("reference_status_str") == "failed")
    ok_int = sum(1 for pod_dict in pod_list if pod_dict.get("reference_status_str") == "ok")
    now_str = f"{ok_int} of {len(pod_list)} pods checked" + (f" · {differ_int} differ{'s' if differ_int == 1 else ''}" if differ_int else "") + (f" · {failed_int} failed" if failed_int else "")
    stale_pod_int = sum(1 for pod_dict in pod_list if pod_dict.get("stale_bool"))
    incomplete_int = sum(1 for pod_dict in pod_list if pod_dict.get("coverage_reason_str") or pod_dict.get("reference_status_str") == "incomplete")
    gate_stale_bool = bool((report_dict.get("cost_gate_dict") or {}).get("stale_bool"))
    if stale_pod_int:
        now_str += f" · {stale_pod_int} stale"
    if incomplete_int:
        now_str += f" · {incomplete_int} incomplete"
    if gate_stale_bool:
        now_str += " · cost evidence stale"
    state_str = "warning" if report_dict["stale_bool"] or failed_int or stale_pod_int or incomplete_int or gate_stale_bool else "ok"
    if report_dict["stale_bool"]:
        now_str = "Not from the latest session · " + now_str
    return {"state_str": state_str, "now_str": now_str[:200], "last_timestamp_str": report_dict["generated_ts"].isoformat(),
            "expected_str": "Session days ~08:00 ET", "detail_str": "Report only: it never blocks orders.", "checked_bool": True}


def report_file_exists_bool(provider_obj):
    if callable(getattr(provider_obj, "get_backtest_check_report_dict", None)):
        return True
    try:
        path_obj = _report_path(provider_obj)
        return path_obj is not None and path_obj.is_file()
    except (OSError, AttributeError, TypeError):
        return False
