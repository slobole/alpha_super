"""Recovery UI reads real temporary DBs and never submits or writes their state."""
from contextlib import closing
from datetime import timedelta
import sqlite3

import pytest

from alpha.live import auto_complete
from alpha.live.dashboard_v4.auto_complete_view import (
    _fill_text_str, apply_recovery_overview, apply_recovery_pod,
)
from alpha.live.dashboard_v4.missing_trades import load_missing_trades_dict, manual_auto_completion_block_reason_str
from alpha.live.dashboard_v4.pod import build_evidence_tables_dict
from alpha.live.ops_report import parse_timestamp_ts
from alpha.live.models import BrokerOrderEvent, BrokerSnapshot
from alpha.live.state_store_v2 import LiveStateStore
from test_dashboard_v4_evidence import FILL_TS, SUBMIT_TS, build_fixture_tuple, update_db


def _fixture(tmp_path, *, amount_float=100.0, message_str="Order would cross related resting order"):
    target_obj, _ = build_fixture_tuple(tmp_path, amount_list=[amount_float], filled_fraction_float=0)
    update_db(target_obj, "UPDATE vplan SET status_str='submitted'")
    update_db(target_obj, "UPDATE decision_plan SET status_str='submitted'")
    update_db(target_obj, "UPDATE vplan_broker_order SET status_str='Cancelled',last_status_timestamp_str=?", (SUBMIT_TS.isoformat(),))
    store_obj = LiveStateStore(target_obj.db_path_str)
    vplan_obj = store_obj.get_vplan_by_id(1)
    store_obj.insert_vplan_broker_order_event_list([BrokerOrderEvent(
        broker_order_id_str="order-0", decision_plan_id_int=1, vplan_id_int=1, account_route_str="U111",
        asset_str="SPY", order_request_key_str=None, status_str="Cancelled", filled_amount_float=0,
        remaining_amount_float=abs(amount_float), event_timestamp_ts=SUBMIT_TS,
        message_str=message_str, raw_payload_dict={"error_code_int": 202})])
    snapshot_obj = BrokerSnapshot("U111", SUBMIT_TS, 10000, 10000, {})
    store_obj.upsert_broker_snapshot_cache(snapshot_obj, release_obj=target_obj.release_obj)
    return target_obj, vplan_obj, snapshot_obj


def _attempt(target_obj, *, status_str="working", order_id_str=None, remaining_float=100.0, quantity_float=100.0):
    with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
        auto_complete._ensure_schema(connection_obj)
        attempted_str = (FILL_TS + timedelta(minutes=5)).isoformat()
        ended_str = None if status_str == "working" else (FILL_TS + timedelta(minutes=5, seconds=3)).isoformat()
        connection_obj.execute("""INSERT INTO auto_completion_attempt
            (vplan_id_int,asset_str,decision_plan_id_int,pod_id_str,account_route_str,original_broker_order_id_str,
             original_message_str,attempted_timestamp_str,updated_timestamp_str,ended_timestamp_str,last_checked_timestamp_str,
             order_request_key_str,status_str,reason_str,quantity_float,remaining_quantity_float,broker_order_id_str)
             VALUES (1,'SPY',1,'pod','U111','order-0','Order would cross related resting order',?,?,?,?,
                     'test:auto-complete:SPY',?,'IBKR cancelled it again',?,?,?)""",
            (attempted_str, ended_str or attempted_str, ended_str, ended_str,
             status_str, quantity_float, remaining_float, order_id_str))


def _missing(target_obj, as_of_ts):
    return load_missing_trades_dict(target_obj, as_of_ts=as_of_ts, shared_account_bool=False, completion_event_list=[])


def test_preopen_cross_is_planned_and_read_only(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    path_obj = tmp_path / "pod.sqlite3"
    before_bytes, before_ns_int = path_obj.read_bytes(), path_obj.stat().st_mtime_ns
    result_dict = _missing(target_obj, SUBMIT_TS + timedelta(seconds=3))
    row_dict = result_dict["row_list"][0]
    assert result_dict["saved_intent_bool"] and row_dict["state_str"] == "planned"
    assert row_dict["quantity_int"] == 100 and "09:35" in row_dict["text_str"]
    assert not row_dict["can_complete_bool"]
    assert path_obj.read_bytes() == before_bytes and path_obj.stat().st_mtime_ns == before_ns_int


def test_interrupted_submit_still_shows_saved_cross_recovery(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    update_db(target_obj, "UPDATE vplan SET status_str='submitting'")
    result_dict = _missing(target_obj, SUBMIT_TS)
    assert result_dict["status_str"] == "ok"
    assert result_dict["row_list"][0]["state_str"] == "planned"
    assert manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=4))
    update_db(target_obj, "DELETE FROM vplan_broker_order_event")
    update_db(target_obj, "UPDATE vplan_broker_order SET status_str='PendingSubmit'")
    assert _missing(target_obj, SUBMIT_TS)["reason_str"] == "Plan not sent"


def test_manual_blackout_has_calendar_boundaries_and_is_symbol_scoped(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    assert not manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=3, seconds=59))
    assert manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=4))
    assert not manual_auto_completion_block_reason_str(target_obj, "QQQ", FILL_TS + timedelta(minutes=5))
    assert not manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=6))
    row_dict = _missing(target_obj, FILL_TS + timedelta(minutes=6))["row_list"][0]
    assert row_dict["state_str"] == "not_completed" and row_dict["can_complete_bool"]
    assert "did not run by 09:36" in row_dict["text_str"]


def test_cancel_without_saved_snapshot_uses_intent_not_a_fabricated_observation(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    update_db(target_obj, "DELETE FROM broker_snapshot_cache")
    result_dict = _missing(target_obj, SUBMIT_TS)
    assert result_dict["positions_timestamp_ts"] is None and result_dict["saved_intent_bool"]
    assert result_dict["row_list"][0]["quantity_int"] == 100


def test_unacknowledged_mark_does_not_invent_sent_order_and_stays_blocked(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    _attempt(target_obj, quantity_float=0, remaining_float=None)
    row_dict = _missing(target_obj, FILL_TS + timedelta(minutes=5, seconds=5))["row_list"][0]
    assert row_dict["state_str"] == "sent" and "Attempt started" in row_dict["text_str"]
    assert "Sent " not in row_dict["text_str"] and "order None" not in row_dict["text_str"]
    assert manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=32))
    assert "Awaiting broker check" in _missing(target_obj, FILL_TS + timedelta(minutes=32))["row_list"][0]["text_str"]


def test_active_short_cover_remains_blocked_after_positions_cross_target(tmp_path):
    target_obj, _, _ = _fixture(tmp_path, amount_float=50)
    update_db(target_obj, 'UPDATE vplan SET target_share_json_str=\'{"SPY": -50}\'')
    store_obj = LiveStateStore(target_obj.db_path_str)
    store_obj.upsert_broker_snapshot_cache(BrokerSnapshot("U111", FILL_TS + timedelta(minutes=7),
        10000, 10000, {"SPY": -40}), release_obj=target_obj.release_obj)
    _attempt(target_obj, quantity_float=50, remaining_float=10)
    row_dict = _missing(target_obj, FILL_TS + timedelta(minutes=7))["row_list"][0]
    assert row_dict["state_str"] == "sent" and row_dict["manual_blocked_bool"]
    assert manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=32))


def test_acknowledged_attempt_has_its_own_order_and_partial_end_has_residual(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    _attempt(target_obj, order_id_str="recovery-1")
    update_db(target_obj, """INSERT INTO vplan_broker_ack
        (decision_plan_id_int,vplan_id_int,account_route_str,order_request_key_str,asset_str,broker_order_type_str,
         local_submit_ack_bool,broker_response_ack_bool,ack_status_str,ack_source_str,broker_order_id_str,response_timestamp_str,raw_payload_json_str)
         VALUES (1,1,'U111','test:auto-complete:SPY','SPY','MKT',1,1,'broker_acked','test','recovery-1',?,'{}')""",
        ((FILL_TS + timedelta(minutes=5, seconds=1)).isoformat(),))
    assert "order recovery-1" in _missing(target_obj, FILL_TS + timedelta(minutes=5, seconds=2))["row_list"][0]["text_str"]
    update_db(target_obj, "UPDATE auto_completion_attempt SET status_str='not_completed',remaining_quantity_float=60,ended_timestamp_str=?",
              ((FILL_TS + timedelta(minutes=31)).isoformat(),))
    row_dict = _missing(target_obj, FILL_TS + timedelta(minutes=31))["row_list"][0]
    assert row_dict["quantity_int"] == 60 and row_dict["can_complete_bool"]
    assert not manual_auto_completion_block_reason_str(target_obj, "SPY", FILL_TS + timedelta(minutes=31))


def test_other_cancel_and_short_are_not_automatic(tmp_path):
    target_obj, _, _ = _fixture(tmp_path, message_str="Not enough buying power")
    row_dict = _missing(target_obj, SUBMIT_TS)["row_list"][0]
    assert row_dict["state_str"] == "not_automatic" and row_dict["can_complete_bool"]
    update_db(target_obj, "UPDATE vplan SET target_share_json_str='{" + '"SPY": -100' + "}',order_delta_json_str='{" + '"SPY": -100' + "}'")
    row_dict = _missing(target_obj, SUBMIT_TS)["row_list"][0]
    assert row_dict["short_bool"] and not row_dict["can_complete_bool"]


@pytest.mark.parametrize("side_str,price_float,expected_str", [
    ("BUY", 101, "+100.0 bps above"), ("BUY", 99, "-100.0 bps below"),
    ("SELL", 99, "+100.0 bps below"), ("SELL", 101, "-100.0 bps above"),
])
def test_display_cost_is_adverse_positive(side_str, price_float, expected_str):
    assert expected_str in _fill_text_str([{"fill_amount_float": 20, "fill_price_float": price_float,
                                           "official_open_price_float": 100}], side_str)


def test_missing_open_is_not_zero_cost():
    assert "open price not available" in _fill_text_str([{"fill_amount_float": 20,
        "fill_price_float": 100, "official_open_price_float": None}], "BUY")


def test_completed_row_keeps_fill_price_unavailable_honest_and_expires_next_day(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    _attempt(target_obj, status_str="completed", order_id_str="recovery-1", remaining_float=0)
    row_dict = _missing(target_obj, FILL_TS + timedelta(minutes=6))["row_list"][0]
    assert row_dict["state_str"] == "completed" and "Fill price not available" in row_dict["text_str"]
    assert not row_dict["can_complete_bool"]


def test_overview_replaces_only_execution_attention(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    missing_dict = _missing(target_obj, SUBMIT_TS)
    pod_dict = {"pod_id_str": "pod", "name_str": "TAA", "state_str": "fail"}
    original_dict = {"pod_id_str": "pod", "kind_str": "cycle", "step_str": "Fill", "state_str": "fail"}
    overview_dict = {"attention_list": [original_dict]}
    apply_recovery_overview(overview_dict, pod_dict, missing_dict)
    assert overview_dict["attention_list"] == [] and pod_dict["pill_str"] == "Auto-complete 09:35"
    for kind_str, step_str in [("database", ""), ("cycle", "Data"), ("scheduler", "")]:
        attention_dict = {**original_dict, "kind_str": kind_str, "step_str": step_str}
        overview_dict = {"attention_list": [attention_dict]}
        apply_recovery_overview(overview_dict, pod_dict, missing_dict)
        assert overview_dict["attention_list"] == [attention_dict]


def test_pod_projection_keeps_unrelated_issue(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    missing_dict = _missing(target_obj, SUBMIT_TS)
    page_dict = {"pod_id_str": "pod", "header_dict": {}, "attention_dict": {"kind_str": "database", "title_str": "DB unavailable"}}
    apply_recovery_pod(page_dict, missing_dict)
    assert page_dict["attention_dict"]["title_str"] == "DB unavailable"


def test_completed_overview_next_is_scheduled_reconciliation(tmp_path):
    target_obj, _, _ = _fixture(tmp_path)
    _attempt(target_obj, status_str="completed", order_id_str="recovery-1", remaining_float=0)
    missing_dict = _missing(target_obj, FILL_TS + timedelta(minutes=6))
    pod_dict = {"pod_id_str": "pod", "name_str": "TAA", "state_str": "fail"}
    apply_recovery_overview({"attention_list": []}, pod_dict, missing_dict)
    assert pod_dict["next_str"] == "Wait for scheduled reconciliation"


def test_orders_table_appends_recovery_separately():
    source_dict = {"order_list": [{"asset_str": "SPY", "broker_order_id_str": "9", "amount_float": 60,
        "filled_amount_float": 0, "status_str": "Submitted", "order_request_key_str": "test:auto-complete:SPY",
        "submitted_timestamp_str": (FILL_TS + timedelta(minutes=5)).isoformat()}]}
    result_dict = build_evidence_tables_dict(source_dict, as_of_ts=FILL_TS + timedelta(minutes=6), fresh_bool=True)
    order_row = result_dict["orders"]["row_list"][0]["cell_list"]
    assert order_row[1] == "BUY 60 · MKT" and order_row[3] == "auto-complete" and order_row[-1] == "9"


@pytest.mark.parametrize("case_str,text_str", [
    ("planned", "Auto-complete 09:35"), ("sent", "order DEMO-AUTO-1"),
    ("unconfirmed", "Attempt started"), ("completed", "+9.1 bps above the open"),
    ("partial", "Partly filled; order ended"), ("not_completed", "IBKR cancelled it again"),
    ("not_automatic", "Not enough buying power"),
])
def test_synthetic_recovery_pages_render_real_readers(case_str, text_str):
    from alpha.live.dashboard_v4.auto_complete_demo import create_auto_completion_demo_app
    app_obj = create_auto_completion_demo_app(case_str)
    try:
        response_obj = app_obj.test_client().get("/pods/demo_1_1?tab=orders")
        assert response_obj.status_code == 200
        html_str = response_obj.get_data(as_text=True)
        assert text_str in html_str
        if case_str in {"planned", "sent", "unconfirmed", "completed"}:
            assert 'data-kind="complete" data-asset="MSFT"' not in html_str
        else:
            assert 'data-kind="complete" data-asset="MSFT"' in html_str
        if case_str in {"sent", "completed", "partial", "not_completed"}:
            assert "BUY 17 · MKT" in html_str and "auto-complete" in html_str
            provider_obj = app_obj.config["auto_complete_demo_provider_obj"]
            as_of_ts = parse_timestamp_ts(next(row_dict for row_dict in provider_obj.row_list
                if row_dict["pod_id_str"] == "demo_1_1")["as_of_timestamp_str"])
            source_dict = provider_obj.get_pod_cycles_dict("demo_1_1", as_of_ts=as_of_ts)
            evidence_dict = source_dict["cycle_evidence_dict"]
            assert evidence_dict["state_str"] == "partial", evidence_dict
            table_dict = build_evidence_tables_dict(source_dict, as_of_ts=as_of_ts, fresh_bool=True)
            original_map_dict = {row_dict["cell_list"][0]: row_dict["cell_list"] for row_dict in table_dict["orders"]["row_list"]
                if row_dict["cell_list"][3] != "auto-complete"}
            assert original_map_dict["AAPL"][5:7] == ["31", "170.00"]
            assert original_map_dict["NVDA"][5:7] == ["44", "130.00"]
            assert original_map_dict["MSFT"][3] == "broker_error"
            assert original_map_dict["AAPL"][3] != "broker_error" and original_map_dict["NVDA"][3] != "broker_error"
        if case_str != "not_automatic":
            activity_str = app_obj.test_client().get("/activity").get_data(as_text=True)
            assert "Automatic completion planned" in activity_str
    finally:
        app_obj.config["auto_complete_demo_provider_obj"].close()


@pytest.mark.parametrize("sql_str", [
    "UPDATE auto_completion_attempt SET request_json_str=NULL",
    "UPDATE auto_completion_attempt SET account_route_str='other'",
    "UPDATE auto_completion_attempt SET original_broker_order_id_str='missing'",
    "UPDATE vplan_broker_order SET amount_float=18 WHERE broker_order_id_str='DEMO-AUTO-1'",
])
def test_recovery_evidence_rejects_mismatched_canonical_request(sql_str):
    from alpha.live.dashboard_v4.auto_complete_demo import create_auto_completion_demo_app
    app_obj = create_auto_completion_demo_app("partial")
    provider_obj = app_obj.config["auto_complete_demo_provider_obj"]
    try:
        target_obj = provider_obj.get_target_for_pod("demo_1_1")
        with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
            connection_obj.execute(sql_str)
        as_of_ts = parse_timestamp_ts(next(row_dict for row_dict in provider_obj.row_list
            if row_dict["pod_id_str"] == "demo_1_1")["as_of_timestamp_str"])
        evidence_dict = provider_obj.get_pod_cycles_dict("demo_1_1", as_of_ts=as_of_ts)["cycle_evidence_dict"]
        assert evidence_dict["state_str"] == "unknown" and evidence_dict["order_list"] == []
    finally:
        provider_obj.close()
