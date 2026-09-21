"""Small read-only System health projections; no probes, task runs or writers."""

from contextlib import closing
from datetime import date, datetime, timezone
from itertools import islice
import json
from pathlib import Path
import re
import sqlite3
import time
from zoneinfo import ZoneInfo

import yaml

from alpha.live.dashboard_v3.operator_tools import strategy_display_name_str
from alpha.live.dashboard_v4.scheduler_status import LATE_AFTER_SECONDS_INT
from alpha.live.ops_report import DEFAULT_STALE_AFTER_SECONDS_INT
from alpha.live.release_manifest import parse_release_manifest
from alpha.live.scheduler_service import DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT


POD_LIMIT_INT = 64
MANIFEST_LIMIT_INT = 128
JSON_BYTE_LIMIT_INT = 512 * 1024
SOURCE_FRESH_SECONDS_INT = 120
IDENTITY_PATTERN_OBJ = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
IDENTITY_FIELD_TUPLE = ("pod_id_str", "account_route_str", "release_id_str", "user_id_str", "mode_str")


def _row_dict(now_str="Unknown", *, state_str="unknown", last_timestamp_str="", expected_str="Saved evidence", detail_str=""):
    return {"state_str": state_str, "now_str": now_str, "last_timestamp_str": last_timestamp_str,
        "expected_str": expected_str, "detail_str": detail_str}


def _timestamp_ts(value_obj, as_of_ts):
    if not isinstance(value_obj, str) or len(value_obj) > 64:
        return None
    try:
        timestamp_ts = datetime.fromisoformat(value_obj.replace("Z", "+00:00"))
        if timestamp_ts.tzinfo is None or timestamp_ts.utcoffset() is None:
            return None
        timestamp_ts = timestamp_ts.astimezone(timezone.utc)
        return timestamp_ts if timestamp_ts <= as_of_ts else None
    except (ValueError, OverflowError):
        return None


def _scope_tuple(provider_obj, workspace_dict):
    if workspace_dict.get("operations_error_str"):
        raise ValueError("Operations unavailable")
    account_list = workspace_dict.get("operations_account_list")
    if not isinstance(account_list, list) or len(account_list) > POD_LIMIT_INT:
        raise ValueError("Invalid scope")
    pair_dict = {}
    for account_dict in account_list:
        pair_tuple = (account_dict["pod_id"], account_dict["account_route"])
        if any(not isinstance(value_str, str) or not IDENTITY_PATTERN_OBJ.fullmatch(value_str) for value_str in pair_tuple):
            raise ValueError("Invalid identity")
        if pair_tuple in pair_dict:
            raise ValueError("Duplicate identity")
        pair_dict[pair_tuple] = account_dict
    if len({pair_tuple[0] for pair_tuple in pair_dict}) != len(pair_dict) or len({pair_tuple[1] for pair_tuple in pair_dict}) != len(pair_dict):
        raise ValueError("Ambiguous identity")
    source_obj = provider_obj
    if not callable(getattr(source_obj, "get_target_list", None)):
        source_obj = provider_obj.app_obj()
    raw_list = source_obj.get_target_list()
    if not isinstance(raw_list, (list, tuple)) or len(raw_list) > MANIFEST_LIMIT_INT:
        raise ValueError("Unbounded scope")
    live_list = [target_obj for target_obj in raw_list if target_obj.release_obj.mode_str == "live"]
    target_list = [target_obj for target_obj in live_list if target_obj.release_obj.enabled_bool is True]
    if len(target_list) > POD_LIMIT_INT:
        raise ValueError("Unbounded scope")
    release_list = [target_obj.release_obj for target_obj in live_list]
    # Read configuration only here, before touching any runtime source. Targets
    # normally omit disabled releases, including an entirely disabled deployment.
    root_str = getattr(provider_obj, "releases_root_path_str", None) or getattr(source_obj, "releases_root_path_str", None)
    if root_str:
        manifest_list = list(islice(Path(root_str).rglob("*.yaml"), MANIFEST_LIMIT_INT + 1))
        if len(manifest_list) > MANIFEST_LIMIT_INT or any(path_obj.stat().st_size > JSON_BYTE_LIMIT_INT for path_obj in manifest_list):
            raise ValueError("Release metadata too large")
        release_list = [parse_release_manifest(str(path_obj)) for path_obj in manifest_list]
        release_list = [release_obj for release_obj in release_list if release_obj.mode_str == "live"]
    owner_set = {release_obj.user_id_str for release_obj in release_list if release_obj.enabled_bool is True}
    if not owner_set:
        owner_set = {release_obj.user_id_str for release_obj in release_list}
    if len(owner_set) != 1:
        raise ValueError("Owner unavailable")
    owner_str = next(iter(owner_set))
    target_pair_list = [(target_obj.release_obj.pod_id_str, target_obj.release_obj.account_route_str) for target_obj in target_list]
    if len(set(target_pair_list)) != len(target_pair_list) or set(target_pair_list) != set(pair_dict):
        raise ValueError("Current operations mismatch")
    for target_obj in target_list:
        release_obj = target_obj.release_obj
        if any(not isinstance(getattr(release_obj, field_str), str) or not IDENTITY_PATTERN_OBJ.fullmatch(getattr(release_obj, field_str))
                for field_str in IDENTITY_FIELD_TUPLE) or release_obj.user_id_str != owner_str:
            raise ValueError("Invalid release")
    if root_str:
        enabled_pair_list = [(release_obj.pod_id_str, release_obj.account_route_str) for release_obj in release_list
            if release_obj.mode_str == "live" and release_obj.enabled_bool is True]
        if sorted(enabled_pair_list) != sorted(target_pair_list) or any(release_obj.user_id_str != owner_str
                for release_obj in release_list if release_obj.mode_str == "live" and release_obj.enabled_bool is True):
            raise ValueError("Release metadata changed")
        for target_obj in target_list:
            matching_list = [release_obj for release_obj in release_list if release_obj.release_id_str == target_obj.release_obj.release_id_str]
            if len(matching_list) != 1 or any(getattr(matching_list[0], field_str) != getattr(target_obj.release_obj, field_str)
                    for field_str in IDENTITY_FIELD_TUPLE):
                raise ValueError("Release metadata changed")
    summary_list = (workspace_dict.get("summary_dict") or {}).get("pod_row_dict_list", [])
    if not isinstance(summary_list, list) or any(not isinstance(row_dict, dict) for row_dict in summary_list):
        raise ValueError("Invalid saved scope")
    for target_obj in target_list:
        matching_list = [row_dict for row_dict in summary_list if row_dict.get("pod_id_str") == target_obj.release_obj.pod_id_str]
        if len(matching_list) > 1 or matching_list and any(matching_list[0].get(field_str) != getattr(target_obj.release_obj, field_str)
                for field_str in IDENTITY_FIELD_TUPLE):
            raise ValueError("Saved scope changed")
    safe_list = []
    seen_set = set()
    for release_obj in release_list:
        if release_obj.mode_str != "live" or release_obj.user_id_str != owner_str:
            continue
        if any(not isinstance(getattr(release_obj, field_str), str) or not IDENTITY_PATTERN_OBJ.fullmatch(getattr(release_obj, field_str))
                for field_str in IDENTITY_FIELD_TUPLE) or type(release_obj.enabled_bool) is not bool or release_obj.release_id_str in seen_set:
            raise ValueError("Invalid release metadata")
        seen_set.add(release_obj.release_id_str)
        name_str = strategy_display_name_str({"pod_id_str": release_obj.pod_id_str,
            "strategy_import_str": getattr(release_obj, "strategy_import_str", "")})
        policy_str = getattr(release_obj, "execution_policy_str", "")
        if policy_str not in {"next_open_moo", "next_open_market", "same_day_moc", "next_month_first_open"}:
            policy_str = "Unknown"
        safe_list.append({"pod_id_str": release_obj.pod_id_str, "name_str": name_str, "mode_str": "live",
            "enabled_bool": release_obj.enabled_bool, "release_id_str": release_obj.release_id_str,
            "execution_policy_str": policy_str, "account_str": "•••" + (release_obj.account_route_str[-3:] if len(release_obj.account_route_str) > 3 else "")})
    path_str = getattr(provider_obj, "event_log_path_str", None) or getattr(source_obj, "event_log_path_str", None)
    return target_list, safe_list, Path(path_str) if isinstance(path_str, str) and path_str else None


def _event_log_dict(path_obj, as_of_ts):
    result_dict = _row_dict("Saved log unavailable", expected_str="File write within 61 min",
        detail_str="File write time; scheduler health is checked separately.")
    if path_obj is None:
        return result_dict
    try:
        with path_obj.open("rb") as file_obj:
            file_obj.read(1)
            stat_obj = path_obj.stat()
        timestamp_ts = datetime.fromtimestamp(stat_obj.st_mtime, timezone.utc)
        if timestamp_ts > as_of_ts:
            return result_dict
        old_bool = (as_of_ts - timestamp_ts).total_seconds() > DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT + LATE_AFTER_SECONDS_INT
        result_dict.update(state_str="warning" if old_bool else "ok",
            now_str=f"Readable · {stat_obj.st_size / 1048576:.1f} MB" + (" · no recent write" if old_bool else ""),
            last_timestamp_str=timestamp_ts.isoformat())
    except (OSError, ValueError, OverflowError):
        pass
    return result_dict


def _database_dict(target_list, workspace_dict, as_of_ts):
    result_dict = _row_dict("Saved database check unavailable", expected_str="Saved LIVE state readable")
    if not target_list:
        return result_dict
    summary_dict = workspace_dict.get("summary_dict") or {}
    source_ts = _timestamp_ts(summary_dict.get("as_of_timestamp_str"), as_of_ts)
    if source_ts is None or (as_of_ts - source_ts).total_seconds() > SOURCE_FRESH_SECONDS_INT:
        result_dict["now_str"] = "Saved database check is out of date"
        return result_dict
    row_list = summary_dict.get("pod_row_dict_list")
    if not isinstance(row_list, list) or any(not isinstance(row_dict, dict) for row_dict in row_list):
        return result_dict
    path_set, size_int = set(), 0
    try:
        for target_obj in target_list:
            release_obj = target_obj.release_obj
            matched_list = [row_dict for row_dict in row_list if row_dict.get("pod_id_str") == release_obj.pod_id_str]
            if len(matched_list) != 1 or any(matched_list[0].get(field_str) != getattr(release_obj, field_str) for field_str in IDENTITY_FIELD_TUPLE):
                return result_dict
            if "as_of_timestamp_str" in matched_list[0]:
                row_ts = _timestamp_ts(matched_list[0]["as_of_timestamp_str"], as_of_ts)
                if row_ts is None or (as_of_ts - row_ts).total_seconds() > SOURCE_FRESH_SECONDS_INT:
                    result_dict["now_str"] = "Saved database check is out of date"
                    return result_dict
            status_str = matched_list[0].get("db_status_str")
            if status_str != "ok":
                return _row_dict("Saved state unavailable", state_str="error" if status_str in {"error", "missing"} else "unknown",
                    last_timestamp_str=source_ts.isoformat(), expected_str=result_dict["expected_str"])
        # A matching summary provides readability evidence; stat cannot prove SQL
        # integrity and must not turn an unreadable or old summary green.
        for target_obj in target_list:
            path_obj = Path(target_obj.db_path_str).resolve()
            if path_obj not in path_set:
                if not path_obj.is_file():
                    return result_dict
                size_int += path_obj.stat().st_size
                path_set.add(path_obj)
    except (OSError, ValueError, TypeError):
        return result_dict
    return _row_dict(f"Readable · {size_int / 1048576:.1f} MB", state_str="ok",
        last_timestamp_str=source_ts.isoformat(), expected_str=result_dict["expected_str"])


def _watchdog_dict(path_obj, target_list, as_of_ts):
    expected_str = "Run schedule not saved"
    result_dict = _row_dict("No verified saved report", expected_str=expected_str)
    if path_obj is None or not target_list:
        return result_dict
    try:
        with (path_obj.parent / "ops_report_latest.json").open("rb") as file_obj:
            content_bytes = file_obj.read(JSON_BYTE_LIMIT_INT + 1)
        if len(content_bytes) > JSON_BYTE_LIMIT_INT:
            return result_dict
        report_dict = json.loads(content_bytes)
        if not isinstance(report_dict, dict) or report_dict.get("schema_version_str") != "live_ops_inspector.v1" or report_dict.get("mode_str") not in {"live", "all"}:
            return result_dict
        report_list = report_dict.get("pod_report_dict_list")
        if not isinstance(report_list, list) or len(report_list) > MANIFEST_LIMIT_INT or any(not isinstance(row_dict, dict) for row_dict in report_list):
            return result_dict
        live_list = [row_dict for row_dict in report_list if row_dict.get("mode_str") == "live"]
        expected_set = {(target_obj.release_obj.pod_id_str, target_obj.release_obj.account_route_str) for target_obj in target_list}
        observed_list = [(row_dict.get("pod_id_str"), row_dict.get("account_route_str")) for row_dict in live_list]
        if len(observed_list) != len(expected_set) or set(observed_list) != expected_set:
            return result_dict
        target_dict = {target_obj.release_obj.pod_id_str: target_obj.release_obj for target_obj in target_list}
        if any(row_dict.get(field_str, getattr(target_dict[row_dict["pod_id_str"]], field_str)) != getattr(target_dict[row_dict["pod_id_str"]], field_str)
                for row_dict in live_list for field_str in ("user_id_str", "release_id_str")):
            return result_dict
        timestamp_ts = _timestamp_ts(report_dict.get("generated_at_utc_str"), as_of_ts)
        if timestamp_ts is None:
            return result_dict
        stale_bool = (as_of_ts - timestamp_ts).total_seconds() > DEFAULT_STALE_AFTER_SECONDS_INT
        return _row_dict("Report over 15 min old · run not verified" if stale_bool else "Report saved · run not verified", state_str="warning" if stale_bool else "unknown",
            last_timestamp_str=timestamp_ts.isoformat(), expected_str=expected_str,
            detail_str="Report generation only; alert and dead-man delivery are not recorded here.")
    except (OSError, ValueError, TypeError, RecursionError):
        return result_dict


def _flex_dict(path_str, target_list, as_of_ts):
    result_dict = _row_dict("No verified saved report", expected_str="Run schedule not saved", detail_str="Scheduled-run receipt unavailable.")
    if not isinstance(path_str, str) or not path_str or not target_list:
        return result_dict
    try:
        path_obj = Path(path_str).resolve()
        if not path_obj.is_file():
            return result_dict
        timestamp_list, date_list = [], []
        with closing(sqlite3.connect(path_obj.as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.execute("PRAGMA query_only=ON")
            deadline_float = time.monotonic() + .3
            connection_obj.set_progress_handler(lambda: int(time.monotonic() > deadline_float), 1000)
            connection_obj.execute("BEGIN")
            for target_obj in target_list:
                release_obj = target_obj.release_obj
                binding_list = connection_obj.execute("SELECT pod_id_str,enabled_bool_int FROM pod_binding WHERE account_route_str=? LIMIT 2",
                    (release_obj.account_route_str,)).fetchall()
                if binding_list != [(release_obj.pod_id_str, 1)]:
                    return result_dict
                report_list = connection_obj.execute("SELECT d.pod_id_str,d.market_date_str,i.imported_timestamp_str "
                    "FROM daily_performance d JOIN flex_import i ON d.source_import_id_int=i.import_id_int "
                    "WHERE d.account_route_str=? ORDER BY d.market_date_str DESC LIMIT 1", (release_obj.account_route_str,)).fetchall()
                if len(report_list) != 1 or report_list[0][0] != release_obj.pod_id_str:
                    return result_dict
                date_str = report_list[0][1]
                if not isinstance(date_str, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_str):
                    return result_dict
                market_date_obj = date.fromisoformat(date_str)
                timestamp_ts = _timestamp_ts(report_list[0][2], as_of_ts)
                if timestamp_ts is None or market_date_obj >= as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date() or timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).date() < market_date_obj:
                    return result_dict
                timestamp_list.append(timestamp_ts)
                date_list.append(date_str)
        # An unchanged import may be a successful repeated sync. These facts
        # prove saved account coverage, not the last task run or delivery.
        result_dict.update(now_str=f"Report close {min(date_list)} · run not verified", last_timestamp_str=min(timestamp_list).isoformat(),
            detail_str=f"Close {min(date_list)} · Scheduled-run receipt unavailable.")
    except (OSError, ValueError, TypeError, sqlite3.Error):
        pass
    return result_dict


def load_system_source_dict(provider_obj, workspace_dict, *, as_of_ts, performance_db_path_str=None):
    if as_of_ts.tzinfo is None or as_of_ts.utcoffset() is None:
        raise ValueError("System health requires a timezone-aware clock")
    as_of_ts = as_of_ts.astimezone(timezone.utc)
    result_dict = {"checked_timestamp_str": as_of_ts.isoformat(), "scope_verified_bool": False, "release_list": [],
        "event_log_dict": _row_dict(), "database_dict": _row_dict(), "watchdog_dict": _row_dict(), "flex_dict": _row_dict(),
        "alerts_dict": _row_dict("No saved delivery receipt", expected_str="Saved delivery receipt"),
        "deadman_dict": _row_dict("No saved ping receipt", expected_str="Saved ping receipt")}
    try:
        target_list, release_list, path_obj = _scope_tuple(provider_obj, workspace_dict)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, yaml.YAMLError):
        return result_dict
    result_dict["scope_verified_bool"] = True
    result_dict["release_list"] = release_list
    if not target_list:
        return result_dict
    result_dict.update(release_list=release_list, event_log_dict=_event_log_dict(path_obj, as_of_ts),
        database_dict=_database_dict(target_list, workspace_dict, as_of_ts),
        watchdog_dict=_watchdog_dict(path_obj, target_list, as_of_ts),
        flex_dict=_flex_dict(performance_db_path_str, target_list, as_of_ts))
    return result_dict
