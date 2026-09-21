"""Bounded Activity candidates from the existing, read-only LIVE cycle reader."""

from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import quote, urlencode

from alpha.live.dashboard_v3.client_operations import SOURCE_MAX_AGE_SECONDS_INT
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v4.pod import STEP_TAB_DICT, build_pod_page_dict


POD_LIMIT_INT = 32
POD_READ_LIMIT_INT = 12
TOTAL_READ_LIMIT_INT = 60
CYCLE_ID_FIELD_TUPLE = ("cycle_key_str", "release_id_str", "decision_plan_id_int", "vplan_id_int")


def _timestamp_ts(value_obj):
    try:
        timestamp_ts = value_obj if isinstance(value_obj, datetime) else datetime.fromisoformat(str(value_obj).replace("Z", "+00:00"))
        return timestamp_ts.astimezone(UTC) if timestamp_ts.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _row_dict(pod_dict, cycle_dict, *, stage_str, timestamp_ts, title_str, detail_str, tab_str):
    identity_str = "|".join((pod_dict["pod_id_str"], cycle_dict["release_id_str"], cycle_dict["cycle_key_str"], stage_str))
    if stage_str == "eod":
        identity_str = "|".join((pod_dict["pod_id_str"], "eod", timestamp_ts.isoformat()))
    market_ts = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ) if timestamp_ts else None
    return {"id_str": sha256(identity_str.encode()).hexdigest()[:24],
        "timestamp_str": timestamp_ts.isoformat() if timestamp_ts else "",
        "time_str": market_ts.strftime("%H:%M:%S") if market_ts else "—",
        "day_str": market_ts.date().isoformat() if market_ts else "",
        "day_label_str": market_ts.strftime("%a %Y-%m-%d") if market_ts else "",
        "pod_id_str": pod_dict["pod_id_str"], "pod_name_str": pod_dict["name_str"],
        "type_str": "cycles", "state_str": "done", "title_str": title_str,
        "detail_str": detail_str, "code_str": "", "stage_str": stage_str,
        "evidence_list": [{"label_str": "Cycle", "value_str": cycle_dict["cycle_key_str"]}],
        "evidence_url_str": "/pods/" + quote(pod_dict["pod_id_str"], safe="") + "?" +
            urlencode({"cycle": cycle_dict["cycle_key_str"], "tab": tab_str}) + "#evidence",
        "child_list": [], **{field_str: cycle_dict[field_str] for field_str in CYCLE_ID_FIELD_TUPLE}}


def _project_list(overview_dict, pod_dict, source_dict, cycle_dict, *, as_of_ts, from_ts):
    selected_dict = source_dict.get("selected_cycle_dict") or {}
    if source_dict.get("status_str") != "ok" or any(selected_dict.get(field_str) != cycle_dict.get(field_str) for field_str in CYCLE_ID_FIELD_TUPLE):
        raise ValueError("Cycle selection unavailable")
    release_dict = source_dict.get("selected_release_dict") or {}
    saved_row_dict = source_dict.get("pod_row_dict") or {}
    if release_dict.get("mode_str") != "live" or saved_row_dict.get("mode_str") != "live" or release_dict.get("pod_id_str") != pod_dict["pod_id_str"]:
        raise ValueError("Cycle scope unavailable")
    if (release_dict.get("release_id_str") != cycle_dict.get("release_id_str")
        or saved_row_dict.get("latest_decision_plan_id_int") != cycle_dict.get("decision_plan_id_int")
        or saved_row_dict.get("latest_vplan_id_int") != cycle_dict.get("vplan_id_int")
        or source_dict.get("decision_dict", {}).get("decision_plan_id_int") != cycle_dict.get("decision_plan_id_int")
        or source_dict.get("vplan_dict", {}).get("vplan_id_int") != cycle_dict.get("vplan_id_int")):
        raise ValueError("Selected cycle identity unavailable")
    for source_key_str in ("pod_row_dict", "decision_dict", "vplan_dict"):
        if source_key_str == "vplan_dict" and cycle_dict.get("vplan_id_int") is None:
            continue
        if any(not release_dict.get(field_str) or source_dict.get(source_key_str, {}).get(field_str) != release_dict[field_str]
               for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str")):
            raise ValueError("Cycle identity unavailable")
    source_ts = _timestamp_ts(saved_row_dict.get("as_of_timestamp_str"))
    if source_ts is None or not 0 <= (as_of_ts - source_ts).total_seconds() <= SOURCE_MAX_AGE_SECONDS_INT or saved_row_dict.get("source_stale_bool"):
        raise ValueError("Saved cycle read is out of date")
    # A freshly verified historical DB read is independent of current shell
    # liveness. Only this local presentation copy gets its own read freshness;
    # the Overview header and saved source timestamps/flags remain unchanged.
    page_dict = build_pod_page_dict({**overview_dict, "source_fresh_bool": True}, source_dict, {},
        pod_id_str=pod_dict["pod_id_str"], as_of_ts=as_of_ts)
    step_list = page_dict["step_list"]
    row_list = []
    eod_ts = _timestamp_ts((source_dict.get("eod_dict") or {}).get("latest_timestamp_str"))
    if step_list[6]["state_str"] == "Done" and eod_ts is not None and from_ts <= eod_ts <= as_of_ts:
        row_list.append(_row_dict(pod_dict, cycle_dict, stage_str="eod", timestamp_ts=eod_ts,
            title_str="EOD snapshot saved.", detail_str="Saved broker close.", tab_str="events"))
    # EOD is independent. Assessment-time Late/Unknown is never turned into a
    # historical event, and an idle summary is never a completed trading cycle.
    if any(step_dict["state_str"] not in {"Done", "None"} for step_dict in step_list[:6]) or step_list[5]["state_str"] != "Done":
        return row_list
    reconcile_ts = _timestamp_ts((source_dict.get("reconciliation_dict") or {}).get("created_timestamp_str"))
    policy_str = cycle_dict.get("execution_policy_str")
    if reconcile_ts is None or not from_ts <= reconcile_ts <= as_of_ts or policy_str not in {"same_day_moc", "next_open_moo", "next_open_market", "next_month_first_open"}:
        return row_list
    timestamp_dict = {"Data": None,
        "Decide": _timestamp_ts(source_dict["decision_dict"].get("created_timestamp_str")),
        "Plan": _timestamp_ts(source_dict["vplan_dict"].get("created_timestamp_str")),
        "Submit": None, "Fill": _timestamp_ts((source_dict.get("cycle_evidence_dict") or {}).get("actual_fill_timestamp_str")),
        "Reconcile": reconcile_ts}
    if any(timestamp_dict[label_str] is None or timestamp_dict[label_str] > as_of_ts for label_str in ("Decide", "Plan")):
        return row_list
    if step_list[4]["state_str"] == "Done" and (timestamp_dict["Fill"] is None or timestamp_dict["Fill"] > as_of_ts):
        return row_list
    ack_time_list = [_timestamp_ts(ack_dict.get("response_timestamp_str")) for ack_dict in source_dict.get("ack_list") or []]
    submission_ts = _timestamp_ts(source_dict["vplan_dict"].get("submission_timestamp_str"))
    # A legacy ACK may carry the planned boundary as a fallback timestamp. The
    # Pod view deliberately withholds that value as an actual observation.
    if submission_ts is not None and ack_time_list and all(ack_ts is not None and submission_ts < ack_ts <= as_of_ts for ack_ts in ack_time_list):
        timestamp_dict["Submit"] = max(ack_time_list)
    row_dict = _row_dict(pod_dict, cycle_dict, stage_str="cycle", timestamp_ts=reconcile_ts,
        title_str=("Close" if policy_str == "same_day_moc" else "Open") + " cycle completed.",
        detail_str=step_list[4]["fact_str"] + " · Positions checked", tab_str="plan")
    row_dict["fold_candidate_bool"] = True  # Caller must retain/block on exact-cycle historical problems.
    title_dict = {"Data": "Data ready.", "Decide": "Decision saved.", "Plan": "Plan saved.",
        "Submit": "Orders acknowledged.", "Fill": "Orders filled.", "Reconcile": "Positions checked."}
    for step_dict in step_list[:6]:
        label_str = step_dict["label_str"]
        child_dict = _row_dict(pod_dict, cycle_dict, stage_str=label_str.lower(),
            timestamp_ts=timestamp_dict[label_str] if step_dict["state_str"] == "Done" else None,
            title_str=title_dict[label_str] if step_dict["state_str"] == "Done" else "No orders required.",
            detail_str=step_dict["fact_str"], tab_str=STEP_TAB_DICT[label_str])
        title_key_str = "".join(character_str for character_str in child_dict["title_str"].casefold() if character_str.isalnum())
        detail_key_str = "".join(character_str for character_str in child_dict["detail_str"].casefold() if character_str.isalnum())
        if detail_key_str == title_key_str or (label_str == "Submit" and detail_key_str == "allordersacknowledged"):
            child_dict["detail_str"] = ""
        child_dict["cross_day_str"] = child_dict["day_str"][5:] if child_dict["day_str"] and child_dict["day_str"] != row_dict["day_str"] else ""
        if label_str == "Data":
            child_dict["evidence_list"].append({"label_str": "Data date", "value_str": step_dict["fact_str"]})
        row_dict["child_list"].append(child_dict)
    row_list.append(row_dict)
    return row_list


def build_activity_cycles_dict(provider_obj, overview_dict, *, as_of_ts, from_ts):
    """Return fold candidates only; never suppress or reinterpret raw log rows."""
    result_dict = {"row_list": [], "warning_list": [], "folded_key_list": []}
    as_of_ts, from_ts = _timestamp_ts(as_of_ts), _timestamp_ts(from_ts)
    if as_of_ts is None or from_ts is None or from_ts > as_of_ts:
        result_dict["warning_list"].append("Saved cycles unavailable: invalid activity period.")
        return result_dict
    pod_list = overview_dict.get("pod_list") or []
    candidate_list, read_count_dict, seen_pod_set = [], {}, set()
    total_int = 0
    truncated_bool = len(pod_list) > POD_LIMIT_INT
    from_date_str, to_date_str = (timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat() for timestamp_ts in (from_ts, as_of_ts))
    for pod_dict in pod_list[:POD_LIMIT_INT]:
        pod_id_str = pod_dict.get("pod_id_str")
        if not isinstance(pod_id_str, str) or not pod_id_str or pod_id_str in seen_pod_set or pod_dict.get("mode_str", "live") != "live" or pod_dict.get("enabled_bool", True) is not True:
            continue
        seen_pod_set.add(pod_id_str)
        read_count_dict[pod_id_str] = 1
        total_int += 1
        try:
            source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=as_of_ts)
            if source_dict.get("status_str") == "empty":
                continue
            if source_dict.get("status_str") != "ok":
                raise ValueError("Saved cycle read unavailable")
            truncated_bool |= bool(source_dict.get("history_truncated_bool"))
            truncated_bool |= len(source_dict.get("cycle_list") or []) > 62
            selected_dict = source_dict.get("selected_cycle_dict") or {}
            for cycle_dict in (source_dict.get("cycle_list") or [])[:62]:
                target_ts = _timestamp_ts(cycle_dict.get("target_execution_timestamp_str"))
                if target_ts is None:
                    raise ValueError("Saved cycle time unavailable")
                if from_date_str <= target_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat() <= to_date_str and target_ts <= as_of_ts:
                    candidate_list.append((target_ts, pod_dict, cycle_dict,
                        source_dict if all(selected_dict.get(field_str) == cycle_dict.get(field_str) for field_str in CYCLE_ID_FIELD_TUPLE) else None))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            result_dict["warning_list"].append("Saved cycles unavailable for " + pod_dict.get("name_str", pod_id_str) + ".")
    # First acquire each Pod's bounded index, then spend remaining reads on the
    # newest eligible cycles across Pods, rather than exhausting the first Pod.
    seen_cycle_set = set()
    for _, pod_dict, cycle_dict, source_dict in sorted(candidate_list, key=lambda item_tuple: item_tuple[0], reverse=True):
        pod_id_str = pod_dict["pod_id_str"]
        cycle_key_tuple = (pod_id_str, cycle_dict.get("release_id_str"), cycle_dict.get("cycle_key_str"))
        if cycle_key_tuple in seen_cycle_set:
            continue
        seen_cycle_set.add(cycle_key_tuple)
        if source_dict is None:
            if read_count_dict[pod_id_str] >= POD_READ_LIMIT_INT or total_int >= TOTAL_READ_LIMIT_INT:
                truncated_bool = True
                continue
            read_count_dict[pod_id_str] += 1
            total_int += 1
        try:
            if source_dict is None:
                source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=as_of_ts,
                    decision_plan_id_int=cycle_dict["decision_plan_id_int"], vplan_id_int=cycle_dict.get("vplan_id_int"))
            result_dict["row_list"].extend(_project_list(overview_dict, pod_dict, source_dict, cycle_dict, as_of_ts=as_of_ts, from_ts=from_ts))
        except (AttributeError, KeyError, TypeError, ValueError, OSError, StopIteration):
            result_dict["warning_list"].append("Saved cycles unavailable for " + pod_dict["name_str"] + ".")
    if truncated_bool:
        result_dict["warning_list"].append("Some saved cycles are not shown (read limit).")
    result_dict["warning_list"] = list(dict.fromkeys(result_dict["warning_list"]))
    # The same account close can be linked to several cycles in its session.
    # Its stable ID denotes the one saved EOD observation, not each lookup.
    result_dict["row_list"] = list({row_dict["id_str"]: row_dict for row_dict in result_dict["row_list"]}.values())
    result_dict["row_list"].sort(key=lambda row_dict: row_dict["timestamp_str"], reverse=True)
    return result_dict
