"""Bounded reads of saved broker quantities; no prices, model fallback or writes."""

from contextlib import closing
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3


POSITION_LIMIT_INT = 2000
RELEASE_LIMIT_INT = 200
MAP_BYTE_LIMIT_INT = 262144
IDENTITY_FIELD_TUPLE = ("release_id_str", "user_id_str", "pod_id_str", "account_route_str")


def observed_timestamp_ts(timestamp_str, as_of_ts):
    if not isinstance(timestamp_str, str):
        raise ValueError("Missing observation time")
    timestamp_ts = datetime.fromisoformat(timestamp_str)
    if timestamp_ts.tzinfo is None or as_of_ts.tzinfo is None or timestamp_ts > as_of_ts:
        raise ValueError("Invalid observation time")
    return timestamp_ts


def validated_position_map_dict(position_dict):
    if not isinstance(position_dict, dict) or len(position_dict) > POSITION_LIMIT_INT:
        raise ValueError("Invalid position map")
    if any(not isinstance(symbol_str, str) or not symbol_str.strip() or symbol_str != symbol_str.strip()
           or len(symbol_str) > 100 or type(share_float) not in {int, float} or not math.isfinite(share_float)
           for symbol_str, share_float in position_dict.items()):
        raise ValueError("Invalid saved quantity")
    return dict(position_dict)


def _position_map_dict(json_str):
    if not isinstance(json_str, str) or len(json_str.encode("utf-8")) > MAP_BYTE_LIMIT_INT:
        raise ValueError("Invalid position source")

    def unique_map_dict(pair_list):
        result_dict = dict(pair_list)
        if len(result_dict) != len(pair_list):
            raise ValueError("Duplicate position symbol")
        return result_dict

    return validated_position_map_dict(json.loads(json_str, object_pairs_hook=unique_map_dict))


def load_positions_dict(target_obj, *, as_of_ts):
    """Choose the newest scoped broker observation in one read-only transaction.

    Cache times are broker sample times. Reconciliation times are recording
    times, disclosed separately. A failed reconciliation still contains actual
    broker quantities; its model map and target quantities are never substituted.
    The caller validates current configured ownership. Saved release flags are
    historical; account-only caches additionally require unambiguous ownership.
    """
    result_dict = {"available_bool": False, "reason_str": "Saved broker positions unavailable",
        "position_map_dict": {}, "position_timestamp_str": None, "source_str": "",
        "timestamp_basis_str": "", "release_id_str": "", "user_id_str": "",
        "pod_id_str": "", "account_route_str": "", "mode_str": "live"}
    try:
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live" or release_obj.enabled_bool is not True or as_of_ts.tzinfo is None:
            return result_dict
        identity_dict = {field_str: getattr(release_obj, field_str) for field_str in IDENTITY_FIELD_TUPLE}
        if any(not isinstance(value_str, str) or not value_str.strip() for value_str in identity_dict.values()):
            return result_dict
        owner_field_tuple = IDENTITY_FIELD_TUPLE[1:]
        candidate_list = []
        db_path_obj = Path(target_obj.db_path_str).resolve()
        with closing(sqlite3.connect(db_path_obj.as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            progress_count_int = 0

            def stop_large_read_bool():
                nonlocal progress_count_int
                progress_count_int += 1
                return progress_count_int > 1000

            connection_obj.set_progress_handler(stop_large_read_bool, 1000)
            connection_obj.execute("BEGIN")
            release_list = connection_obj.execute(
                "SELECT release_id_str,user_id_str,pod_id_str,account_route_str,mode_str FROM live_release "
                "WHERE pod_id_str=? LIMIT ?",
                (release_obj.pod_id_str, RELEASE_LIMIT_INT + 1)).fetchall()
            if not release_list or len(release_list) > RELEASE_LIMIT_INT:
                raise ValueError("Missing or excessive ownership history")
            if any(row_obj["mode_str"] != "live" or any(row_obj[field_str] != identity_dict[field_str]
                   for field_str in owner_field_tuple) for row_obj in release_list):
                raise ValueError("Conflicting saved ownership")
            release_id_set = {row_obj["release_id_str"] for row_obj in release_list}
            if len(release_id_set) != len(release_list) or release_obj.release_id_str not in release_id_set:
                raise ValueError("Release identity mismatch")
            # A former Pod's release may remain enabled in this history table.
            # It cannot invalidate this Pod's scoped reconciliation, but the
            # cache has only an account key and cannot prove which Pod owned it.
            cache_owner_conflict_bool = connection_obj.execute(
                "SELECT 1 FROM live_release WHERE account_route_str=? "
                "AND (pod_id_str<>? OR user_id_str<>? OR mode_str<>'live') LIMIT 1",
                (release_obj.account_route_str, release_obj.pod_id_str, release_obj.user_id_str)).fetchone() is not None
            table_set = {row_obj[0] for row_obj in connection_obj.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('broker_snapshot_cache','vplan_reconciliation_snapshot')")}
            if "broker_snapshot_cache" in table_set and not cache_owner_conflict_bool:
                cache_list = connection_obj.execute(
                    "SELECT snapshot_timestamp_str,CASE WHEN length(CAST(position_json_str AS BLOB))<=? THEN position_json_str END AS position_json_str "
                    "FROM broker_snapshot_cache WHERE account_route_str=? LIMIT 2",
                    (MAP_BYTE_LIMIT_INT, release_obj.account_route_str)).fetchall()
                if len(cache_list) > 1:
                    raise ValueError("Duplicate broker snapshot")
                for row_obj in cache_list:
                    candidate_list.append({"timestamp_ts": observed_timestamp_ts(row_obj["snapshot_timestamp_str"], as_of_ts),
                        "position_map_dict": _position_map_dict(row_obj["position_json_str"]),
                        "source_str": "broker_snapshot", "timestamp_basis_str": "observed"})
            if "vplan_reconciliation_snapshot" in table_set:
                reconcile_list = connection_obj.execute(
                    "SELECT decision_plan_id_int,vplan_id_int,created_timestamp_str,"
                    "CASE WHEN length(CAST(broker_position_json_str AS BLOB))<=? THEN broker_position_json_str END AS broker_position_json_str "
                    "FROM vplan_reconciliation_snapshot WHERE pod_id_str=? AND stage_str='post_execution' "
                    "ORDER BY vplan_reconciliation_snapshot_id_int DESC LIMIT 2", (MAP_BYTE_LIMIT_INT, release_obj.pod_id_str)).fetchall()
                for row_obj in reconcile_list:
                    timestamp_ts = observed_timestamp_ts(row_obj["created_timestamp_str"], as_of_ts)
                    plan_obj = connection_obj.execute("SELECT release_id_str,user_id_str,pod_id_str,account_route_str,"
                        "decision_plan_id_int,target_execution_timestamp_str FROM vplan WHERE vplan_id_int=?", (row_obj["vplan_id_int"],)).fetchone()
                    decision_obj = connection_obj.execute("SELECT release_id_str,user_id_str,pod_id_str,account_route_str FROM decision_plan WHERE decision_plan_id_int=?",
                        (row_obj["decision_plan_id_int"],)).fetchone()
                    if (plan_obj is None or decision_obj is None or plan_obj["decision_plan_id_int"] != row_obj["decision_plan_id_int"]
                            or plan_obj["release_id_str"] not in release_id_set
                            or any(plan_obj[field_str] != decision_obj[field_str] for field_str in IDENTITY_FIELD_TUPLE)
                            or any(plan_obj[field_str] != identity_dict[field_str] for field_str in owner_field_tuple)):
                        raise ValueError("Reconciliation ownership mismatch")
                    if timestamp_ts < observed_timestamp_ts(plan_obj["target_execution_timestamp_str"], as_of_ts):
                        raise ValueError("Reconciliation precedes execution")
                    candidate_list.append({"timestamp_ts": timestamp_ts,
                        "position_map_dict": _position_map_dict(row_obj["broker_position_json_str"]),
                        "source_str": "broker_reconciliation", "timestamp_basis_str": "recorded"})
            if not candidate_list:
                return result_dict
            latest_ts = max(candidate_dict["timestamp_ts"] for candidate_dict in candidate_list)
            latest_list = [candidate_dict for candidate_dict in candidate_list if candidate_dict["timestamp_ts"] == latest_ts]
            if any(candidate_dict["position_map_dict"] != latest_list[0]["position_map_dict"] for candidate_dict in latest_list):
                raise ValueError("Conflicting simultaneous broker positions")
            selected_dict = next((candidate_dict for candidate_dict in latest_list
                if candidate_dict["source_str"] == "broker_snapshot"), latest_list[0])
        result_dict.update(identity_dict, available_bool=True, reason_str="",
            position_map_dict=selected_dict["position_map_dict"], position_timestamp_str=latest_ts.isoformat(),
            source_str=selected_dict["source_str"], timestamp_basis_str=selected_dict["timestamp_basis_str"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, sqlite3.Error):
        pass
    return result_dict
