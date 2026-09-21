"""Pure, allowlisted System health presentation of saved LIVE evidence."""

from datetime import date, datetime, timezone

from alpha.live.dashboard_v3.client_operations import SOURCE_MAX_AGE_SECONDS_INT
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v4.scheduler_status import LATE_AFTER_SECONDS_INT, STOPPED_AFTER_SECONDS_INT
from alpha.live.scheduler_service import DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT


STATE_MAP_DICT = {"green": "done", "yellow": "late", "red": "fail", "gray": "unk",
    "ok": "done", "warning": "late", "error": "fail", "unknown": "unk"}
STATE_RANK_DICT = {"fail": 0, "late": 1, "unk": 2, "now": 3, "skip": 4, "done": 5}
PHASE_LABEL_DICT = {"build_decision_plan": "Decision", "build_vplan": "Plan",
    "submit_vplan": "Submit", "post_execution_reconcile": "Position check",
    "eod_snapshot": "EOD", "expire_stale": "Expired plans", "idle_probe": "Saved state"}
POLICY_LABEL_DICT = {"next_open_moo": "Next open", "next_open_market": "Next open",
    "next_month_first_open": "First open of month", "same_day_moc": "At the close"}


def _timestamp_ts(value_obj):
    if not isinstance(value_obj, str):
        return None
    try:
        timestamp_ts = datetime.fromisoformat(value_obj.replace("Z", "+00:00"))
        return timestamp_ts.astimezone(timezone.utc) if timestamp_ts.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _fresh_bool(value_obj, as_of_ts):
    timestamp_ts = _timestamp_ts(value_obj)
    return bool(timestamp_ts and 0 <= (as_of_ts - timestamp_ts).total_seconds() < SOURCE_MAX_AGE_SECONDS_INT)


def _time_str(value_obj, as_of_ts):
    timestamp_ts = _timestamp_ts(value_obj)
    if timestamp_ts is None or timestamp_ts > as_of_ts:
        return "—"
    local_ts = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ)
    return local_ts.strftime("%H:%M:%S" if local_ts.date() == as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date() else "%m-%d %H:%M:%S")


def _wake_str(value_obj):
    timestamp_ts = _timestamp_ts(value_obj)
    return timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%m-%d %H:%M:%S") if timestamp_ts else "—"


def _date_str(value_obj, as_of_ts):
    if not isinstance(value_obj, str):
        return "—"
    try:
        day_obj = date.fromisoformat(value_obj)
        return day_obj.isoformat() if len(value_obj) == 10 and day_obj <= as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date() else "—"
    except ValueError:
        return "—"


def _worst_str(state_list):
    return min(state_list or ["unk"], key=lambda state_str: STATE_RANK_DICT[state_str])


def _norgate_state_str(norgate_dict, have_str, need_str):
    state_str = STATE_MAP_DICT.get(norgate_dict.get("severity_str"), "unk")
    if state_str in {"fail", "late"}:
        return state_str
    # A persisted green label cannot override its own current data facts.
    if (have_str == "—" or norgate_dict.get("snapshot_fresh_for_cycle_bool") is False
        or (need_str != "—" and have_str < need_str)):
        return "unk"
    return state_str


def _row_dict(key_str, label_str, state_str="unk", now_str="Unknown", last_str="—", expected_str="Not verified"):
    return {"key_str": key_str, "label_str": label_str, "state_str": state_str,
        "now_str": now_str, "last_str": last_str, "expected_str": expected_str}


def _scheduler_dict(status_dict, *, fresh_bool, as_of_ts):
    state_str = status_dict.get("state_str", "unknown")
    last_str = _time_str(status_dict.get("last_seen_timestamp_str"), as_of_ts)
    if not fresh_bool or not _fresh_bool(status_dict.get("checked_timestamp_str"), as_of_ts) or last_str == "—":
        return {"state_str": "unk", "label_str": "Unknown", "last_str": "—", "wake_str": "—", "alive_bool": False}
    wake_ts = _timestamp_ts(status_dict.get("promised_wake_timestamp_str"))
    if wake_ts and state_str in {"sleeping", "holding", "error"}:
        overdue_float = (as_of_ts - wake_ts).total_seconds()
        if overdue_float > STOPPED_AFTER_SECONDS_INT:
            state_str = "stopped"
        elif overdue_float > LATE_AFTER_SECONDS_INT and state_str != "error":
            state_str = "late"
    known_bool = status_dict.get("alive_bool") is True and state_str in {"sleeping", "running", "holding", "error"}
    if wake_ts and (as_of_ts - wake_ts).total_seconds() > LATE_AFTER_SECONDS_INT:
        known_bool = False
    if state_str == "sleeping" and wake_ts is None:
        known_bool = False
    if state_str in {"error", "late", "stopped"}:
        tone_str = "late" if state_str == "late" else "fail"
        label_str = {"error": "Error · check log", "late": "Wake overdue", "stopped": "Not responding · check service"}[state_str]
    elif known_bool:
        tone_str = "skip" if state_str == "holding" else "done"
        if state_str == "holding":
            label_str = "Holding · waits for your review"
        elif status_dict.get("waiting_for_data_bool") is True:
            label_str = "Waiting for data"
        else:
            label_str = "Sleeping" if state_str == "sleeping" else "Running"
            phase_str = PHASE_LABEL_DICT.get(status_dict.get("next_phase_str"))
            if state_str == "running" and phase_str:
                label_str += " · " + phase_str
    else:
        tone_str, label_str = "unk", "Unknown · recent activity only"
    return {"state_str": tone_str, "label_str": label_str, "last_str": last_str,
        "wake_str": _wake_str(status_dict.get("promised_wake_timestamp_str")),
        "alive_bool": known_bool and state_str not in {"late", "stopped"}}


def _aux_row_dict(source_dict, key_str, label_str, *, as_of_ts):
    row_dict = source_dict.get(key_str + "_dict") or {}
    if (source_dict.get("scope_verified_bool") is not True
        or not _fresh_bool(source_dict.get("checked_timestamp_str"), as_of_ts) or not isinstance(row_dict, dict)):
        return _row_dict(key_str, label_str)
    state_str = row_dict.get("state_str", "unk")
    state_str = STATE_MAP_DICT.get(state_str, state_str)
    state_str = state_str if state_str in STATE_RANK_DICT else "unk"
    last_obj = row_dict.get("last_timestamp_str")
    last_str = _time_str(last_obj, as_of_ts)
    if last_obj and last_str == "—":
        return _row_dict(key_str, label_str)
    # The collector supplies fixed safe messages, never raw log/error text.
    return _row_dict(key_str, label_str, state_str,
        str(row_dict.get("now_str") or "Unknown")[:200], last_str,
        str(row_dict.get("expected_str") or "Not verified")[:200])


def build_system_page_dict(overview_dict, workspace_dict, source_dict, *, as_of_ts: datetime):
    if as_of_ts.tzinfo is None or as_of_ts.utcoffset() is None:
        raise ValueError("System health requires an aware clock.")
    summary_dict = workspace_dict.get("summary_dict") or {}
    scope_verified_bool = (source_dict.get("scope_verified_bool") is True
        and _fresh_bool(source_dict.get("checked_timestamp_str"), as_of_ts))
    fresh_bool = (overview_dict.get("source_fresh_bool") is True
        and _fresh_bool(summary_dict.get("as_of_timestamp_str"), as_of_ts)
        and not workspace_dict.get("operations_error_str") and scope_verified_bool)
    account_list = workspace_dict.get("operations_account_list") or []
    account_map_dict = {}
    for account_dict in account_list:
        if isinstance(account_dict, dict):
            account_map_dict.setdefault(account_dict.get("pod_id"), []).append(account_dict)
    raw_list = summary_dict.get("pod_row_dict_list") or []
    pod_list, saved_list, scheduler_list, eod_state_list = [], [], [], []
    for pod_dict in overview_dict.get("pod_list") or []:
        pod_id_str = pod_dict["pod_id_str"]
        account_match_list = account_map_dict.get(pod_id_str, [])
        match_list = [row_dict for row_dict in raw_list if isinstance(row_dict, dict) and row_dict.get("pod_id_str") == pod_id_str]
        release_match_list = [release_dict for release_dict in source_dict.get("release_list") or []
            if isinstance(release_dict, dict) and release_dict.get("pod_id_str") == pod_id_str
            and release_dict.get("mode_str") == "live" and release_dict.get("enabled_bool") is True]
        saved_dict = match_list[0] if (fresh_bool and len(account_match_list) == len(match_list) == len(release_match_list) == 1
            and match_list[0].get("mode_str") == "live"
            and bool(release_match_list[0].get("release_id_str"))
            and match_list[0].get("release_id_str") == release_match_list[0]["release_id_str"]
            and isinstance(account_match_list[0].get("account_route"), str) and bool(account_match_list[0]["account_route"])
            and match_list[0].get("account_route_str") == account_match_list[0].get("account_route")
            and _fresh_bool(match_list[0].get("as_of_timestamp_str", summary_dict.get("as_of_timestamp_str")), as_of_ts)) else {}
        saved_list.append(saved_dict)
        scheduler_dict = _scheduler_dict(pod_dict.get("scheduler_dict") or {}, fresh_bool=bool(saved_dict), as_of_ts=as_of_ts)
        scheduler_list.append(scheduler_dict)
        norgate_dict = saved_dict.get("norgate_snapshot_status_dict") or {}
        eod_dict = saved_dict.get("eod_snapshot_dict") or {}
        data_str = _date_str(norgate_dict.get("snapshot_date_str"), as_of_ts)
        required_date_str = _date_str((norgate_dict.get("required_snapshot_date_by_release_dict") or {}).get(saved_dict.get("release_id_str")), as_of_ts)
        data_state_str = _norgate_state_str(norgate_dict, data_str, required_date_str)
        eod_str = _time_str(eod_dict.get("latest_timestamp_str"), as_of_ts)
        eod_state_str = STATE_MAP_DICT.get(eod_dict.get("severity_str"), "unk")
        eod_status_str = eod_dict.get("status_str")
        if saved_dict and any(step_dict.get("name_str") == "EOD" and step_dict.get("state_str") == "now" for step_dict in pod_dict.get("step_list") or []):
            eod_state_str, eod_label_str = "now", "Waiting for capture"
        else:
            eod_label_str = {"due_missing": "Due snapshot missing", "blocked_by_execution": "Waiting for execution"}.get(eod_status_str, "")
            if eod_str == "—" and eod_state_str not in {"fail", "late"}:
                eod_state_str = "unk"
            if not eod_label_str and eod_state_str in {"fail", "late"}:
                eod_label_str = "Needs action" if eod_state_str == "fail" else "Needs review"
        eod_state_list.append(eod_state_str)
        if eod_label_str:
            eod_str = eod_label_str + (" · " + eod_str if eod_str != "—" else "")
        pod_list.append({"pod_id_str": pod_id_str, "name_str": pod_dict["name_str"],
            "scheduler_state_str": scheduler_dict["state_str"], "scheduler_str": scheduler_dict["label_str"],
            "last_str": scheduler_dict["last_str"], "wake_str": scheduler_dict["wake_str"],
            "data_str": data_str, "data_state_str": data_state_str,
            "broker_str": _time_str(saved_dict.get("latest_broker_snapshot_timestamp_str"), as_of_ts),
            "eod_str": eod_str, "eod_state_str": eod_state_str})
    total_int = len(pod_list)
    alive_int = sum(item_dict["alive_bool"] for item_dict in scheduler_list)
    scheduler_row_dict = _row_dict("schedulers", "Schedulers · LIVE", _worst_str([item_dict["state_str"] for item_dict in scheduler_list]),
        f"{alive_int} of {total_int} alive" if total_int else "No enabled LIVE Pods verified",
        "—", "One per Pod · wakes when promised")
    seen_list = [item_dict["last_str"] for item_dict in scheduler_list if item_dict["last_str"] != "—"]
    if seen_list:
        scheduler_row_dict["last_str"] = "See Pods below"
    broker_time_list = [row_dict.get("latest_broker_snapshot_timestamp_str") for row_dict in saved_list
        if _time_str(row_dict.get("latest_broker_snapshot_timestamp_str"), as_of_ts) != "—"]
    gateway_row_dict = _row_dict("gateway", "Broker gateway", now_str=f"Connection unverified · {len(broker_time_list)} of {total_int} saved reads",
        last_str=_time_str(max(broker_time_list, key=_timestamp_ts), as_of_ts) if broker_time_list else "—",
        expected_str="Saved reads do not prove a current connection")
    continuous_list = [scheduler_row_dict, gateway_row_dict,
        _aux_row_dict(source_dict, "alerts", "Alerts · Discord", as_of_ts=as_of_ts),
        _row_dict("dashboard", "Dashboard", "done", "Responding", _time_str(as_of_ts.isoformat(), as_of_ts), "Refresh every 15 s")]

    norgate_state_list, sync_time_list, have_list, need_list, fred_state_list, fred_date_list = [], [], [], [], [], []
    for saved_dict in saved_list:
        norgate_dict = saved_dict.get("norgate_snapshot_status_dict") or {}
        have_str = _date_str(norgate_dict.get("snapshot_date_str"), as_of_ts)
        need_str = _date_str((norgate_dict.get("required_snapshot_date_by_release_dict") or {}).get(saved_dict.get("release_id_str")), as_of_ts)
        have_list.append(have_str)
        need_list.append(need_str)
        norgate_state_list.append(_norgate_state_str(norgate_dict, have_str, need_str))
        if _time_str(norgate_dict.get("last_sync_utc_str"), as_of_ts) != "—":
            sync_time_list.append(norgate_dict["last_sync_utc_str"])
        fred_list = [item_dict for item_dict in (saved_dict.get("data_freshness_dict") or {}).get("item_dict_list") or []
            if item_dict.get("label_str") == "DTB3/FRED"]
        fred_dict = fred_list[0] if len(fred_list) == 1 else {}
        fred_date_str = _date_str(saved_dict.get("dtb3_latest_observation_date_str"), as_of_ts)
        fred_date_list.append(fred_date_str)
        fred_state_str = STATE_MAP_DICT.get(fred_dict.get("severity_str"), "unk")
        # DTB3 values come from saved DecisionPlan metadata. A successful old
        # download cannot establish the health of today's FRED feed.
        fred_state_list.append(fred_state_str if fred_state_str in {"fail", "late"} else "unk")
    have_str = have_list[0] if have_list and len(set(have_list)) == 1 else "See Pods below" if have_list else "—"
    need_str = need_list[0] if need_list and len(set(need_list)) == 1 else "Varies by Pod" if need_list else "—"
    sync_last_str = _time_str(max(sync_time_list, key=_timestamp_ts), as_of_ts) if sync_time_list else "—"
    sync_data_str = "Dates vary by Pod" if have_str == "See Pods below" else "Saved data " + have_str if have_str != "—" else "Unknown"
    sync_row_dict = _row_dict("norgate", "Data sync · Norgate", _worst_str(norgate_state_list),
        sync_data_str, sync_last_str, "As required by each Pod")
    if sync_row_dict["state_str"] in {"fail", "late"}:
        sync_row_dict["now_str"] = ("Needs action" if sync_row_dict["state_str"] == "fail" else "Needs review") + " · " + (
            sync_data_str if have_str != "—" else "Data date unknown")
    scheduled_list = [_aux_row_dict(source_dict, "watchdog", "Watchdog", as_of_ts=as_of_ts),
        _aux_row_dict(source_dict, "deadman", "Dead-man ping", as_of_ts=as_of_ts), sync_row_dict,
        _aux_row_dict(source_dict, "flex", "Flex import", as_of_ts=as_of_ts)]
    market_state_str = _worst_str(norgate_state_list)
    if market_state_str not in {"fail", "late"} and (not need_list or "—" in need_list):
        market_state_str = "unk"
    market_row_dict = _row_dict("market_data", "Market data", market_state_str,
        "Dates vary by Pod" if have_str == "See Pods below" else "Have " + have_str, sync_last_str, "Needed " + need_str)
    fred_str = ("Last decision used " + min(fred_date_list) + " · current feed unverified"
        if fred_date_list and "—" not in fred_date_list else "Current feed unverified")
    fred_state_str = _worst_str(fred_state_list)
    if fred_state_str in {"fail", "late"}:
        fred_str = ("Saved failure" if fred_state_str == "fail" else "Saved warning") + " · " + fred_str
    fred_row_dict = _row_dict("fred", "Rates · FRED", fred_state_str, fred_str,
        "—", "Current feed status is not saved")
    event_row_dict = _aux_row_dict(source_dict, "event_log", "Event log", as_of_ts=as_of_ts)
    event_row_dict["expected_str"] = f"Idle scheduler may sleep {DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT // 60} min · {LATE_AFTER_SECONDS_INT} s grace"
    database_row_dict = _aux_row_dict(source_dict, "database", "Database", as_of_ts=as_of_ts)
    disk_list = [cell_dict for cell_dict in overview_dict.get("health_list") or [] if cell_dict.get("label_str") == "Disk"]
    disk_dict = disk_list[0] if len(disk_list) == 1 else {}
    disk_state_str = STATE_MAP_DICT.get(disk_dict.get("severity_str"), "unk") if fresh_bool else "unk"
    disk_value_str = disk_dict.get("value_str") or "Unknown"
    if not isinstance(disk_value_str, str) or not disk_value_str.removesuffix("% used").isdigit() or not disk_value_str.endswith("% used"):
        disk_state_str, disk_value_str = "unk", "Unknown"
    disk_row_dict = _row_dict("disk", "Disk", disk_state_str, disk_value_str if fresh_bool else "Unknown",
        _time_str(as_of_ts.isoformat(), as_of_ts) if fresh_bool else "—", "Warn at 75% · fail at 90%")
    group_list = [{"label_str": "Runs all the time", "row_list": continuous_list},
        {"label_str": "Runs on a schedule", "row_list": scheduled_list},
        {"label_str": "Data and space", "row_list": [market_row_dict, fred_row_dict, event_row_dict, database_row_dict, disk_row_dict]}]
    existing_state_str = (overview_dict.get("system_dict") or {}).get("state_str", "unk")
    state_str = _worst_str([row_dict["state_str"] for group_dict in group_list for row_dict in group_dict["row_list"]]
        + eod_state_list + ([existing_state_str] if existing_state_str in STATE_RANK_DICT else ["unk"]))
    verdict_str = {"fail": "System needs action.", "late": "System needs review.",
        "unk": "Some system checks are unverified."}.get(state_str, "Saved system checks are current.")
    detail_str = "Review failed or overdue checks." if state_str in {"fail", "late"} else "Saved evidence only. A current broker connection is not checked."
    release_list = []
    if scope_verified_bool:
        for release_dict in source_dict.get("release_list") or []:
            if release_dict.get("mode_str") != "live":
                continue
            release_list.append({key_str: release_dict.get(key_str) for key_str in
                ("pod_id_str", "name_str", "mode_str", "enabled_bool", "release_id_str", "account_str")})
            release_list[-1]["trades_str"] = POLICY_LABEL_DICT.get(release_dict.get("execution_policy_str"), "Unknown")
    return {"verdict_str": verdict_str, "detail_str": detail_str, "state_str": state_str,
        "group_list": group_list, "pod_list": pod_list, "release_list": release_list,
        "as_of_timestamp_str": as_of_ts.isoformat()}
