"""ET-day activity proves actual executions; tests write isolated fixture DBs only."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import sqlite3

import pytest

from alpha.live.dashboard_v4 import positions_activity
from alpha.live.dashboard_v4.positions_activity import load_position_activity_dict
from alpha.live.state_store_v2 import LiveStateStore
from test_dashboard_v4_evidence import (
    FILL_TS, NOW_TS, SUBMIT_TS, _sparse_fixture_tuple, build_fixture_tuple, update_db,
)
from test_dashboard_v4_pod_data import _reconcile


def _read(target_obj, as_of_ts=NOW_TS):
    return load_position_activity_dict(target_obj, as_of_ts=as_of_ts)


def _copy_cycle(target_obj, *, minutes_int=10, target_days_int=0):
    """Copy a production-shaped fixture to an independently keyed second cycle."""
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        for table_str, id_str in (("decision_plan", "decision_plan_id_int"), ("vplan", "vplan_id_int"),
                ("vplan_row", "vplan_row_id_int"), ("vplan_broker_order", "broker_order_record_id_int"), ("vplan_fill", "fill_record_id_int")):
            row_list = connection_obj.execute("SELECT * FROM " + table_str).fetchall()
            for row_obj in row_list:
                row_dict = dict(row_obj)
                row_dict.pop(id_str)
                if "decision_plan_id_int" in row_dict:
                    row_dict["decision_plan_id_int"] = 2
                if "vplan_id_int" in row_dict:
                    row_dict["vplan_id_int"] = 2
                for field_str, value_obj in list(row_dict.items()):
                    if field_str.endswith("timestamp_str") and value_obj:
                        row_dict[field_str] = (datetime.fromisoformat(value_obj) + timedelta(minutes=minutes_int)).isoformat()
                if "target_execution_timestamp_str" in row_dict:
                    row_dict["target_execution_timestamp_str"] = (FILL_TS + timedelta(days=target_days_int)).isoformat()
                for field_str in ("submission_key_str", "order_request_key_str"):
                    if row_dict.get(field_str):
                        row_dict[field_str] = row_dict[field_str].replace("vplan:1", "vplan:2")
                if table_str == "vplan_fill":
                    row_dict["raw_payload_json_str"] = json.dumps({"exec_id_str": "second-" + row_dict["broker_order_id_str"]})
                connection_obj.execute("INSERT INTO " + table_str + " (" + ",".join(row_dict) + ") VALUES ("
                    + ",".join("?" for _ in row_dict) + ")", tuple(row_dict.values()))


def test_signed_same_symbol_legs_are_gross_activity_without_writes(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    path_obj = tmp_path / "pod.sqlite3"
    before_bytes, before_mtime_int = path_obj.read_bytes(), path_obj.stat().st_mtime_ns
    result_dict = _read(target_obj)
    assert result_dict["available_bool"]
    assert result_dict["market_date_str"] == "2026-09-18"
    assert result_dict["assessed_timestamp_str"] == NOW_TS.isoformat()
    item_dict = result_dict["symbol_dict"]["SPY"]
    assert (item_dict["bought_float"], item_dict["sold_float"], item_dict["filled_delta_float"]) == (10, 8, 2)
    assert item_dict["changed_bool"] and item_dict["status_str"] == "Filled"
    assert item_dict["before_float"] is None and not item_dict["new_bool"]
    assert item_dict["observed_timestamp_str"] == FILL_TS.isoformat()
    assert path_obj.read_bytes() == before_bytes and path_obj.stat().st_mtime_ns == before_mtime_int


def test_round_trip_is_changed_even_with_zero_net_shares(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path, amount_list=[10, -10])
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["filled_delta_float"] == 0 and item_dict["changed_bool"]
    assert item_dict["bought_float"] == item_dict["sold_float"] == 10


def test_all_cycles_including_old_target_plans_filled_today_are_counted(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _copy_cycle(target_obj, target_days_int=-1)
    result_dict = _read(target_obj)
    assert result_dict["available_bool"]
    item_dict = result_dict["symbol_dict"]["SPY"]
    assert (item_dict["bought_float"], item_dict["sold_float"], item_dict["filled_delta_float"]) == (20, 16, 4)
    assert item_dict["cycle_key_str"] == "vplan:2"


def test_latest_activity_link_uses_observed_time_not_highest_plan_id(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _copy_cycle(target_obj, minutes_int=-10)
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["cycle_key_str"] == "vplan:1"
    assert item_dict["observed_timestamp_str"] == FILL_TS.isoformat()


def test_prior_day_fills_do_not_enter_today_delta_of_selected_cycle(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    yesterday_ts = SUBMIT_TS - timedelta(days=1)
    update_db(target_obj, "UPDATE vplan_broker_order SET submitted_timestamp_str=?", (yesterday_ts.isoformat(),))
    update_db(target_obj, "UPDATE vplan_fill SET fill_timestamp_str=? WHERE fill_amount_float<0", ((FILL_TS - timedelta(days=1)).isoformat(),))
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert (item_dict["bought_float"], item_dict["sold_float"]) == (10, 0)
    assert item_dict["before_float"] is None and not item_dict["new_bool"]


@pytest.mark.parametrize("amount_float,status_str", [(10, "Buy pending"), (-10, "Sell pending")])
def test_ready_plan_is_pending_not_changed(tmp_path, amount_float, status_str):
    target_obj, _ = build_fixture_tuple(tmp_path, amount_list=[amount_float])
    for table_str in ("vplan_fill", "vplan_broker_order"):
        update_db(target_obj, "DELETE FROM " + table_str)
    update_db(target_obj, "UPDATE vplan SET status_str='ready'")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["status_str"] == status_str
    assert not item_dict["changed_bool"] and not item_dict["new_bool"] and not item_dict["closed_bool"]
    assert item_dict["filled_delta_float"] == 0


@pytest.mark.parametrize("fraction_float,status_str", [(0, "Buy pending"), (.3, "Partial")])
def test_submitted_order_reports_only_verified_filled_quantity(tmp_path, fraction_float, status_str):
    target_obj, _ = build_fixture_tuple(tmp_path, amount_list=[10], filled_fraction_float=fraction_float)
    update_db(target_obj, "UPDATE vplan SET status_str='submitted'")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["status_str"] == status_str and item_dict["filled_delta_float"] == 10 * fraction_float
    assert item_dict["changed_bool"] is bool(fraction_float)


def test_cancelled_order_retains_actual_partial_execution(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path, amount_list=[10], filled_fraction_float=.3)
    update_db(target_obj, "UPDATE vplan_broker_order SET status_str='Cancelled'")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["status_str"] == "Cancelled" and item_dict["changed_bool"]
    assert item_dict["filled_delta_float"] == 3


def test_same_day_matching_broker_endpoints_prove_new_and_closed(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert (item_dict["before_float"], item_dict["after_float"]) == (0, 2)
    assert item_dict["new_bool"] and not item_dict["closed_bool"]
    update_db(target_obj, "UPDATE vplan SET current_broker_position_json_str='{\"SPY\":-2}'")
    update_db(target_obj, "UPDATE vplan_reconciliation_snapshot SET broker_position_json_str='{}'")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert (item_dict["before_float"], item_dict["after_float"]) == (-2, 0)
    assert item_dict["closed_bool"] and item_dict["changed_bool"]


def test_daily_endpoints_require_continuity_across_all_executed_cycles(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _copy_cycle(target_obj, minutes_int=20)
    _reconcile(target_obj)
    _reconcile(target_obj, vplan_id_int=2, decision_id_int=2, timestamp_ts=FILL_TS + timedelta(minutes=26))
    update_db(target_obj, "UPDATE vplan SET current_broker_position_json_str='{\"SPY\":2}' WHERE vplan_id_int=2")
    update_db(target_obj, "UPDATE vplan_reconciliation_snapshot SET broker_position_json_str='{\"SPY\":4}' WHERE vplan_id_int=2")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert (item_dict["before_float"], item_dict["after_float"]) == (0, 4)
    assert item_dict["new_bool"] and item_dict["filled_delta_float"] == 4
    update_db(target_obj, "UPDATE vplan SET current_broker_position_json_str='{\"SPY\":1}' WHERE vplan_id_int=2")
    update_db(target_obj, "UPDATE vplan_reconciliation_snapshot SET broker_position_json_str='{\"SPY\":3}' WHERE vplan_id_int=2")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["changed_bool"] and item_dict["filled_delta_float"] == 4
    assert item_dict["before_float"] is None and not item_dict["new_bool"]


@pytest.mark.parametrize("sql_str", [
    "UPDATE vplan_reconciliation_snapshot SET status_str='failed'",
    "UPDATE vplan_reconciliation_snapshot SET broker_position_json_str='{\"SPY\":99}'",
    "UPDATE vplan SET broker_snapshot_timestamp_str='2026-09-17T13:00:00+00:00'",
])
def test_missing_endpoint_proof_does_not_hide_valid_fills_or_invent_tags(tmp_path, sql_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    update_db(target_obj, sql_str)
    result_dict = _read(target_obj)
    assert result_dict["available_bool"]
    item_dict = result_dict["symbol_dict"]["SPY"]
    assert item_dict["changed_bool"] and item_dict["before_float"] is None
    assert not item_dict["new_bool"] and not item_dict["closed_bool"]


def test_sparse_terminal_legacy_orders_require_existing_ack_contract(tmp_path):
    target_obj, _ = _sparse_fixture_tuple(tmp_path)
    assert _read(target_obj)["available_bool"]
    update_db(target_obj, "DELETE FROM vplan_broker_ack")
    assert not _read(target_obj)["available_bool"]


def test_mixed_sparse_cycle_requires_full_verified_ack_set(tmp_path):
    target_obj, _ = _sparse_fixture_tuple(tmp_path, mixed_bool=True)
    assert _read(target_obj)["available_bool"]
    update_db(target_obj, "UPDATE vplan_broker_ack SET broker_response_ack_bool=0 WHERE broker_order_id_str='order-1'")
    assert not _read(target_obj)["available_bool"]


@pytest.mark.parametrize("sql_str", [
    "UPDATE vplan_fill SET account_route_str='FOREIGN'",
    "UPDATE vplan_fill SET decision_plan_id_int=99",
    "UPDATE vplan_fill SET broker_order_id_str='orphan'",
    "UPDATE vplan_fill SET vplan_id_int=999",
    "UPDATE vplan_fill SET fill_amount_float=-fill_amount_float",
    "UPDATE vplan_fill SET fill_amount_float=999",
    "UPDATE vplan_fill SET fill_price_float=0",
    "UPDATE vplan_fill SET raw_payload_json_str='{}'",
    "UPDATE vplan_fill SET raw_payload_json_str='{\"exec_id_str\":\"same\"}'",
    "UPDATE vplan_fill SET raw_payload_json_str='{\"exec_id_str\":\"x\",\"exec_id_str\":\"y\"}'",
    "UPDATE vplan_broker_order SET order_request_key_str='invalid'",
    "UPDATE vplan SET order_delta_json_str='{\"SPY\":99}'",
    "UPDATE decision_plan SET account_route_str='FOREIGN'",
    "DELETE FROM vplan_fill",
])
def test_corrupt_evidence_never_becomes_no_activity_or_partial_success(tmp_path, sql_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, sql_str)
    result_dict = _read(target_obj)
    assert not result_dict["available_bool"] and result_dict["symbol_dict"] == {}


@pytest.mark.parametrize("timestamp_str", ["garbage", "2026-09-18T13:30:00", "2026-09-18T15:00:00+00:00"])
def test_invalid_naive_and_future_fill_timestamps_fail_closed(tmp_path, timestamp_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan_fill SET fill_timestamp_str=?", (timestamp_str,))
    assert not _read(target_obj)["available_bool"]


def test_et_midnight_differs_from_utc_date(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan SET target_execution_timestamp_str='2026-09-18T04:00:00+00:00'")
    update_db(target_obj, "UPDATE vplan_broker_order SET submitted_timestamp_str='2026-09-18T03:00:00+00:00'")
    update_db(target_obj, "UPDATE vplan_fill SET fill_timestamp_str='2026-09-18T03:59:59.999999+00:00' WHERE fill_amount_float<0")
    update_db(target_obj, "UPDATE vplan_fill SET fill_timestamp_str='2026-09-18T04:00:00+00:00' WHERE fill_amount_float>0")
    item_dict = _read(target_obj)["symbol_dict"]["SPY"]
    assert item_dict["bought_float"] == 10 and item_dict["sold_float"] == 0


def test_duplicate_execution_across_cycles_is_rejected(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _copy_cycle(target_obj)
    update_db(target_obj, "UPDATE vplan_fill SET raw_payload_json_str='{\"exec_id_str\":\"exec-0\"}' WHERE vplan_id_int=2 AND fill_amount_float>0")
    assert not _read(target_obj)["available_bool"]


def test_valid_empty_activity_is_distinct_from_missing_schema(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "DELETE FROM vplan_fill")
    update_db(target_obj, "UPDATE vplan SET target_execution_timestamp_str='2026-09-17T13:30:00+00:00'")
    result_dict = _read(target_obj)
    assert result_dict["available_bool"] and result_dict["symbol_dict"] == {}
    update_db(target_obj, "DROP TABLE vplan_broker_ack")
    assert not _read(target_obj)["available_bool"]


def test_empty_day_with_incomplete_evidence_schema_is_unknown(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "DELETE FROM vplan_fill")
    update_db(target_obj, "UPDATE vplan SET target_execution_timestamp_str='2026-09-17T13:30:00+00:00'")
    update_db(target_obj, "ALTER TABLE vplan_broker_order RENAME COLUMN filled_amount_float TO unavailable_amount_float")
    assert not _read(target_obj)["available_bool"]


def test_old_release_with_same_owner_is_supported_but_reassignment_is_not(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    new_release_obj = replace(target_obj.release_obj, release_id_str="new-release")
    LiveStateStore(target_obj.db_path_str).upsert_release(new_release_obj)
    target_obj = replace(target_obj, release_obj=new_release_obj)
    assert _read(target_obj)["available_bool"]
    update_db(target_obj, "UPDATE live_release SET account_route_str='FOREIGN' WHERE release_id_str='release'")
    assert not _read(target_obj)["available_bool"]


@pytest.mark.parametrize("field_str,value_obj", [("mode_str", "paper"), ("enabled_bool", False), ("account_route_str", "FOREIGN")])
def test_ineligible_target_is_unknown(tmp_path, field_str, value_obj):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, **{field_str: value_obj}))
    assert not _read(target_obj)["available_bool"]


def test_missing_database_is_not_created_and_naive_assessment_is_unknown(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    assert not _read(target_obj, NOW_TS.replace(tzinfo=None))["available_bool"]
    missing_path_obj = tmp_path / "missing.sqlite3"
    assert not _read(replace(target_obj, db_path_str=str(missing_path_obj)))["available_bool"]
    assert not missing_path_obj.exists()


def test_bounded_rows_cycles_and_json_fail_closed(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    monkeypatch.setattr(positions_activity, "ROW_LIMIT_INT", 1)
    assert not _read(target_obj)["available_bool"]
    monkeypatch.undo()
    monkeypatch.setattr(positions_activity, "CYCLE_LIMIT_INT", 0)
    assert not _read(target_obj)["available_bool"]
    monkeypatch.undo()
    update_db(target_obj, "UPDATE vplan_fill SET raw_payload_json_str=?", ('{"exec_id_str":"' + "x" * 17000 + '"}',))
    assert not _read(target_obj)["available_bool"]


def test_oversized_optional_timestamp_cannot_be_treated_as_missing(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, "UPDATE vplan_broker_order SET last_status_timestamp_str=?", ("x" * 600,))
    assert not _read(target_obj)["available_bool"]
