"""One saved broker EOD observation for display, independent of Flex reporting."""

from contextlib import closing
from datetime import datetime, time, timedelta
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from alpha.live.dashboard_v3.client_cash import _is_selected_eod_bool
from alpha.live.dashboard_v4.positions_data import (
    IDENTITY_FIELD_TUPLE, MAP_BYTE_LIMIT_INT, RELEASE_LIMIT_INT,
    _position_map_dict, observed_timestamp_ts,
)


HISTORY_LIMIT_INT = 256
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def load_eod_holdings_dict(target_obj, *, close_date_str=None, as_of_ts):
    """Return exact-date or latest prior-day quantities, cash and broker NAV.

    This operator view reads saved broker facts, never model/target quantities.
    An explicit date cannot borrow another day; omitted dates select the latest
    completed broker EOD. Both use the canonical exchange close plus 10 minutes.
    Malformed latest evidence cannot fall back to an older healthy observation.
    """
    result_dict = {"available_bool": False, "reason_str": "Saved broker close unavailable",
        "close_date_str": "", "position_map_dict": {}, "cash_float": None, "broker_nav_float": None,
        "observed_timestamp_str": "", "source_str": "", "release_id_str": "", "user_id_str": "",
        "pod_id_str": "", "account_route_str": "", "mode_str": "live"}
    try:
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live" or release_obj.enabled_bool is not True or as_of_ts.tzinfo is None:
            return result_dict
        identity_dict = {field_str: getattr(release_obj, field_str) for field_str in IDENTITY_FIELD_TUPLE}
        if any(not isinstance(value_str, str) or not value_str.strip() for value_str in identity_dict.values()):
            return result_dict
        today_obj = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date()
        date_clause_str, date_value_tuple = "", ()
        if close_date_str is not None:
            selected_date_obj = datetime.strptime(close_date_str, "%Y-%m-%d").date()
            if selected_date_obj.isoformat() != close_date_str or selected_date_obj >= today_obj:
                return result_dict
            day_start_ts = datetime.combine(selected_date_obj, time(), MARKET_TIMEZONE_OBJ)
            day_end_ts = day_start_ts + timedelta(days=1)
            # SQL narrows the exact ET day. Unparseable timestamps still enter
            # validation, so corrupt evidence cannot silently revive older rows.
            date_clause_str = " AND (julianday(updated_timestamp_str) IS NULL OR (julianday(updated_timestamp_str)>=julianday(?) AND julianday(updated_timestamp_str)<julianday(?)))"
            date_value_tuple = (day_start_ts.isoformat(), day_end_ts.isoformat())
        db_path_obj = Path(target_obj.db_path_str).resolve()
        with closing(sqlite3.connect(db_path_obj.as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("PRAGMA query_only=ON")
            progress_list = [0]

            def stop_large_read_bool():
                progress_list[0] += 1
                return progress_list[0] > 1000

            connection_obj.set_progress_handler(stop_large_read_bool, 1000)
            connection_obj.execute("BEGIN")
            release_list = connection_obj.execute(
                "SELECT release_id_str,user_id_str,pod_id_str,account_route_str,mode_str FROM live_release "
                "WHERE pod_id_str=? LIMIT ?", (release_obj.pod_id_str, RELEASE_LIMIT_INT + 1)).fetchall()
            if not release_list or len(release_list) > RELEASE_LIMIT_INT:
                return result_dict
            if any(row_obj["mode_str"] != "live" or any(row_obj[field_str] != identity_dict[field_str]
                    for field_str in IDENTITY_FIELD_TUPLE[1:]) for row_obj in release_list):
                return result_dict
            release_id_set = {row_obj["release_id_str"] for row_obj in release_list}
            if len(release_id_set) != len(release_list) or release_obj.release_id_str not in release_id_set:
                return result_dict
            # Sort by observation time, not insertion order: delayed recording
            # must not replace a newer broker sample. NULL dates fail first.
            history_list = connection_obj.execute(
                "SELECT pod_state_history_id_int,updated_timestamp_str,user_id_str,account_route_str,"
                "julianday(updated_timestamp_str) AS ordering_time_float "
                "FROM pod_state_history WHERE pod_id_str=? AND snapshot_stage_str='eod' AND snapshot_source_str='broker'"
                + date_clause_str + " ORDER BY julianday(updated_timestamp_str) IS NULL DESC, "
                "julianday(updated_timestamp_str) DESC,pod_state_history_id_int DESC LIMIT ?",
                (release_obj.pod_id_str, *date_value_tuple, HISTORY_LIMIT_INT)).fetchall()
            selected_list = []
            selected_order_float = None
            for row_obj in history_list:
                timestamp_ts = observed_timestamp_ts(row_obj["updated_timestamp_str"], as_of_ts)
                date_str = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
                if row_obj["user_id_str"] != release_obj.user_id_str or row_obj["account_route_str"] != release_obj.account_route_str:
                    return result_dict
                # *** CRITICAL *** quantities and cash stay in one actual EOD
                # row. Today's observation is not a prior-day closing portfolio.
                if date_str >= today_obj.isoformat():
                    continue
                if close_date_str is not None and date_str != close_date_str:
                    return result_dict
                if not _is_selected_eod_bool(row_obj["updated_timestamp_str"], date_str, release_obj.session_calendar_id_str, as_of_ts):
                    continue
                if selected_list and row_obj["ordering_time_float"] < selected_order_float:
                    break
                # SQLite's date order rounds submilliseconds. Inspect the full
                # equal SQL-time group before proving one latest observation.
                if not selected_list or timestamp_ts > selected_list[0][0]:
                    selected_list = []
                    selected_order_float = row_obj["ordering_time_float"]
                if not selected_list or timestamp_ts == selected_list[0][0]:
                    selected_list.append((timestamp_ts, date_str, row_obj["pod_state_history_id_int"]))
            if (len(history_list) == HISTORY_LIMIT_INT and selected_order_float is not None
                    and history_list[-1]["ordering_time_float"] == selected_order_float):
                return result_dict
            if len(selected_list) != 1:
                return result_dict
            timestamp_ts, date_str, history_id_int = selected_list[0]
            selected_obj = connection_obj.execute(
                "SELECT CASE WHEN length(CAST(position_json_str AS BLOB))<=? THEN position_json_str END AS position_json_str, "
                "cash_float,total_value_float FROM pod_state_history WHERE pod_state_history_id_int=?",
                (MAP_BYTE_LIMIT_INT, history_id_int)).fetchone()
            position_map_dict = _position_map_dict(selected_obj["position_json_str"])
            cash_float, nav_float = selected_obj["cash_float"], selected_obj["total_value_float"]
            if any(type(value_float) not in {int, float} or not math.isfinite(value_float) for value_float in (cash_float, nav_float)):
                return result_dict
        result_dict.update(identity_dict, available_bool=True, reason_str="", close_date_str=date_str,
            position_map_dict=position_map_dict, cash_float=cash_float, broker_nav_float=nav_float,
            observed_timestamp_str=timestamp_ts.isoformat(), source_str="Saved broker EOD")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, sqlite3.Error):
        pass
    return result_dict
