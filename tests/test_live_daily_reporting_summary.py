"""Sparse order reporting must not change daily settlement or broker requests."""
from dataclasses import asdict, replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from alpha.live import runner
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderAck, BrokerOrderFill, BrokerOrderRecord, BrokerOrderRequest, SessionOpenPrice
from alpha.live.state_store_v2 import LiveStateStore
from test_live_daily_reconcile import CLOSE_TS, OPEN_TS, daily_case


def _report_fixture(case_tuple):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = case_tuple
    request_list = build_broker_order_request_list_from_vplan(plan_obj)
    record_list, fill_list, ack_list = [], [], []
    for request_int, request_obj in enumerate(request_list, start=1):
        order_id_str = str(1000 + request_int)
        record_list.append(BrokerOrderRecord(
            order_id_str, decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
            release_obj.account_route_str, request_obj.asset_str, request_obj.order_request_key_str,
            request_obj.broker_order_type_str, "shares", request_obj.amount_float, 0.0, "Submitted",
            plan_obj.submission_timestamp_ts, remaining_amount_float=abs(request_obj.amount_float),
            submission_key_str=plan_obj.submission_key_str))
        fill_list.append(BrokerOrderFill(
            order_id_str, decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
            release_obj.account_route_str, request_obj.asset_str, request_obj.amount_float,
            100.0 + request_int, OPEN_TS, raw_payload_dict={"exec_id_str": f"exec:{order_id_str}"}))
        ack_list.append(BrokerOrderAck(
            decision_obj.decision_plan_id_int, plan_obj.vplan_id_int, release_obj.account_route_str,
            request_obj.order_request_key_str, request_obj.asset_str, request_obj.broker_order_type_str,
            True, True, "broker_acked", "open_order", order_id_str, int(order_id_str),
            plan_obj.submission_timestamp_ts))
    store_obj.upsert_vplan_broker_order_record_list(record_list)
    store_obj.upsert_vplan_broker_ack_list(ack_list)
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.get_session_open_price_list = lambda **_kwargs_dict: [
        SessionOpenPrice("2026-10-05", release_obj.account_route_str, request_obj.asset_str,
            100.0, "fixture_open", CLOSE_TS) for request_obj in request_list]
    return request_list, record_list, fill_list


@pytest.mark.parametrize("sparse_bool", [False, True])
def test_sparse_report_repairs_summary_without_changing_daily_settlement(daily_case, tmp_path, sparse_bool):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    request_list, record_list, fill_list = _report_fixture(daily_case)
    request_payload_list = [asdict(request_obj) for request_obj in request_list]
    broker_obj.position_dict.update(plan_obj.target_share_map)
    position_dict = dict(broker_obj.position_dict)
    report_list = [replace(record_obj, status_str="Filled",
        amount_float=0.0 if sparse_bool else record_obj.amount_float,
        filled_amount_float=0.0 if sparse_bool else abs(record_obj.amount_float),
        remaining_amount_float=0.0, avg_fill_price_float=None if sparse_bool else fill_obj.fill_price_float)
        for record_obj, fill_obj in zip(record_list, fill_list)]
    broker_obj.get_recent_order_state_snapshot = lambda **_kwargs_dict: (report_list, [], fill_list)

    assert runner._reconcile_daily_cycles(store_obj, SimpleNamespace(get_adapter=lambda _release_obj: broker_obj),
        CLOSE_TS, "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces")) == 1

    for order_dict, request_obj, fill_obj in zip(
            store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int), request_list, fill_list):
        assert order_dict["amount_float"] == request_obj.amount_float
        assert order_dict["filled_amount_float"] == abs(request_obj.amount_float)
        assert order_dict["remaining_amount_float"] == 0.0
        assert order_dict["avg_fill_price_float"] == fill_obj.fill_price_float
    assert [asdict(request_obj) for request_obj in build_broker_order_request_list_from_vplan(plan_obj)] == request_payload_list
    assert broker_obj.sent_request_list == []
    assert broker_obj.cancel_ref_list == []
    assert broker_obj.refresh_count_int == 2
    state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    assert state_obj.position_amount_map == position_dict
    assert state_obj.cash_float == 1000.0 and state_obj.total_value_float == 100000.0
    assert state_obj.strategy_state_dict == decision_obj.strategy_state_dict
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "completed"

    # A restart/repeated reporting read must not reopen settlement or send again.
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)
    assert runner._reconcile_daily_cycles(restarted_store_obj,
        SimpleNamespace(get_adapter=lambda _release_obj: broker_obj), CLOSE_TS + timedelta(minutes=1),
        "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces")) == 0
    assert broker_obj.sent_request_list == []


def test_later_empty_sparse_report_cannot_erase_verified_summary(daily_case, tmp_path):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    _, record_list, fill_list = _report_fixture(daily_case)
    full_list = [replace(record_obj, status_str="Filled", filled_amount_float=abs(record_obj.amount_float),
        remaining_amount_float=0.0, avg_fill_price_float=fill_obj.fill_price_float,
        last_status_timestamp_ts=OPEN_TS) for record_obj, fill_obj in zip(record_list, fill_list)]
    store_obj.upsert_vplan_broker_order_record_list(full_list)
    before_list = store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int, include_evidence_bool=True)
    sparse_list = [replace(record_obj, amount_float=0.0, filled_amount_float=0.0,
        avg_fill_price_float=None, last_status_timestamp_ts=plan_obj.submission_timestamp_ts)
        for record_obj in full_list]
    runner._persist_daily_execution_report(store_obj, broker_obj, release_obj, decision_obj, plan_obj,
        sparse_list, [], [], CLOSE_TS, str(tmp_path / "report.log"))
    assert store_obj.get_broker_order_row_dict_list_for_vplan(
        plan_obj.vplan_id_int, include_evidence_bool=True) == before_list
    assert broker_obj.sent_request_list == []


def test_fill_proof_cannot_hide_a_real_broker_position_gap(daily_case, tmp_path):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    _, record_list, fill_list = _report_fixture(daily_case)
    sparse_list = [replace(record_obj, status_str="Filled", amount_float=0.0,
        remaining_amount_float=0.0) for record_obj in record_list]
    broker_obj.get_recent_order_state_snapshot = lambda **_kwargs_dict: (sparse_list, [], fill_list)
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.position_dict["AAPL"] = 0.0
    assert runner._reconcile_daily_cycles(store_obj, SimpleNamespace(get_adapter=lambda _release_obj: broker_obj),
        CLOSE_TS, "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces")) == 1
    saved_decision_obj = store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int)
    assert saved_decision_obj.status_str == "completed_with_exceptions"
    assert saved_decision_obj.snapshot_metadata_dict["daily_execution_result_dict"]["exception_list"][0]["asset_str"] == "AAPL"
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map["AAPL"] == 0.0
    assert broker_obj.sent_request_list == []


QUANTITY_KEY_TUPLE = ("amount_float", "filled_amount_float", "remaining_amount_float", "avg_fill_price_float", "status_str")


def _unproved_fixture(case_tuple, failure_str):
    """A sparse Filled refresh whose proof is missing (never contradicted)."""
    store_obj, _release_obj, _decision_obj, plan_obj, broker_obj = case_tuple
    request_list, record_list, fill_list = _report_fixture(case_tuple)
    if failure_str == "no_ack":
        with store_obj._connect() as connection_obj:
            connection_obj.execute("DELETE FROM vplan_broker_ack")
    elif failure_str == "no_execution_id":
        fill_list[0] = replace(fill_list[0], raw_payload_dict={})
    elif failure_str == "duplicate_execution":
        fill_list.append(fill_list[0])
    else:
        with store_obj._connect() as connection_obj:
            if failure_str == "invalid_ack_flag":
                connection_obj.execute("UPDATE vplan_broker_ack SET broker_response_ack_bool=2")
            else:
                response_ts = (plan_obj.submission_timestamp_ts - timedelta(seconds=1)
                    if failure_str == "early_ack" else CLOSE_TS + timedelta(minutes=1))
                connection_obj.execute("UPDATE vplan_broker_ack SET response_timestamp_str=?", (response_ts.isoformat(),))
    sparse_list = [replace(record_obj, status_str="Filled", amount_float=0.0,
        remaining_amount_float=0.0) for record_obj in record_list]
    broker_obj.position_dict.update(plan_obj.target_share_map)
    broker_obj.get_recent_order_state_snapshot = lambda **_kwargs_dict: (sparse_list, [], fill_list)
    return request_list, fill_list


@pytest.mark.parametrize("failure_str", ["no_ack", "no_execution_id", "early_ack", "future_ack", "invalid_ack_flag"])
def test_unproved_report_keeps_the_sent_summary_and_still_writes_the_fills(daily_case, tmp_path, failure_str):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    request_list, fill_list = _unproved_fixture(daily_case, failure_str)
    before_list = store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int)

    assert runner._reconcile_daily_cycles(store_obj, SimpleNamespace(get_adapter=lambda _release_obj: broker_obj),
        CLOSE_TS, "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces")) == 1

    after_list = store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int, include_evidence_bool=True)
    for index_int, (before_dict, after_dict) in enumerate(zip(before_list, after_list, strict=True)):
        payload_dict = after_dict["raw_payload_dict"]
        if failure_str == "no_execution_id" and index_int > 0:
            # Only the first order lost its proof; the others are repaired.
            assert after_dict["filled_amount_float"] == abs(after_dict["amount_float"]) > 0
            assert payload_dict["order_summary_source_str"] == "canonical_request_and_executions"
            continue
        assert {key_str: after_dict[key_str] for key_str in QUANTITY_KEY_TUPLE} == {
            key_str: before_dict[key_str] for key_str in QUANTITY_KEY_TUPLE}
        assert payload_dict["order_summary_proof_gap_str"] == (
            "incomplete_executions" if failure_str == "no_execution_id" else "no_broker_ack")
    assert len(store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int)) == len(fill_list)
    with store_obj._connect() as connection_obj:
        report_status_str, error_str = connection_obj.execute(
            "SELECT status_str,error_str FROM daily_post_close_report WHERE decision_plan_id_int=?",
            (decision_obj.decision_plan_id_int,)).fetchone()
    assert report_status_str == "reported" and error_str is None
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == broker_obj.position_dict
    assert [asdict(request_obj) for request_obj in build_broker_order_request_list_from_vplan(plan_obj)] == [
        asdict(request_obj) for request_obj in request_list]
    assert broker_obj.sent_request_list == [] and broker_obj.cancel_ref_list == []
    assert broker_obj.refresh_count_int == 2


def test_completion_sell_is_repaired_from_its_claimed_request(daily_case, tmp_path):
    # The in-session completion sell is not an opening leg; its claimed request
    # is the canonical proof at the post-close report.
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    submission_key_str = plan_obj.submission_key_str or f"vplan:{plan_obj.decision_plan_id_int}"
    request_obj = BrokerOrderRequest(release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str, submission_key_str=submission_key_str,
        order_request_key_str=f"{submission_key_str}:daily-completion:MSFT", asset_str="MSFT",
        broker_order_type_str="MKT", order_class_str="MarketOrder", unit_str="shares", amount_float=-5.0,
        target_bool=False, trade_id_int=None, sizing_reference_price_float=100.0, portfolio_value_float=100000.0,
        decision_plan_id_int=decision_obj.decision_plan_id_int, vplan_id_int=plan_obj.vplan_id_int,
        execution_deadline_timestamp_str=CLOSE_TS.isoformat())
    sent_ts = OPEN_TS + timedelta(minutes=5)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("""INSERT INTO daily_completion_request (decision_plan_id_int,asset_str,vplan_id_int,
            pod_id_str,account_route_str,order_request_key_str,request_json_str,claimed_timestamp_str) VALUES (?,?,?,?,?,?,?,?)""",
            (decision_obj.decision_plan_id_int, "MSFT", plan_obj.vplan_id_int, release_obj.pod_id_str,
                release_obj.account_route_str, request_obj.order_request_key_str,
                json.dumps(asdict(request_obj), sort_keys=True), sent_ts.isoformat()))
    sent_obj = BrokerOrderRecord("2001", decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        release_obj.account_route_str, "MSFT", request_obj.order_request_key_str, "MKT", "shares", -5.0, 0.0,
        "Submitted", sent_ts, remaining_amount_float=5.0, submission_key_str=submission_key_str)
    store_obj.upsert_vplan_broker_order_record_list([sent_obj])
    store_obj.upsert_vplan_broker_ack_list([BrokerOrderAck(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        release_obj.account_route_str, request_obj.order_request_key_str, "MSFT", "MKT", True, True, "broker_acked",
        "open_order", "2001", 2001, sent_ts)])
    fill_obj = BrokerOrderFill("2001", decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        release_obj.account_route_str, "MSFT", -5.0, 99.5, sent_ts + timedelta(seconds=2),
        raw_payload_dict={"exec_id_str": "exec:2001"})
    sparse_obj = replace(sent_obj, status_str="Filled", amount_float=0.0, remaining_amount_float=0.0)

    runner._persist_daily_execution_report(store_obj, broker_obj, release_obj, decision_obj, plan_obj,
        [sparse_obj], [], [fill_obj], CLOSE_TS, str(tmp_path / "report.log"))

    row_dict, = [row_dict for row_dict in store_obj.get_broker_order_row_dict_list_for_vplan(
        plan_obj.vplan_id_int, include_evidence_bool=True) if row_dict["broker_order_id_str"] == "2001"]
    assert (row_dict["amount_float"], row_dict["filled_amount_float"], row_dict["remaining_amount_float"],
        row_dict["avg_fill_price_float"], row_dict["status_str"]) == (-5.0, 5.0, 0.0, 99.5, "Filled")
    assert row_dict["raw_payload_dict"]["order_summary_source_str"] == "canonical_request_and_executions"
    assert len(store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int)) == 1
    assert broker_obj.sent_request_list == []


def test_contradicting_report_fails_and_cannot_block_holdings_settlement(daily_case, tmp_path):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    request_list, _fill_list = _unproved_fixture(daily_case, "duplicate_execution")
    before_list = store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int, include_evidence_bool=True)

    assert runner._reconcile_daily_cycles(store_obj, SimpleNamespace(get_adapter=lambda _release_obj: broker_obj),
        CLOSE_TS, "paper", None, str(tmp_path / "report.log"), False, str(tmp_path / "traces")) == 1

    assert store_obj.get_broker_order_row_dict_list_for_vplan(
        plan_obj.vplan_id_int, include_evidence_bool=True) == before_list
    assert store_obj.get_fill_row_dict_list_for_vplan(plan_obj.vplan_id_int) == []
    with store_obj._connect() as connection_obj:
        report_status_str, error_str = connection_obj.execute(
            "SELECT status_str,error_str FROM daily_post_close_report WHERE decision_plan_id_int=?",
            (decision_obj.decision_plan_id_int,)).fetchone()
    assert report_status_str == "failed" and error_str
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == broker_obj.position_dict
    assert [asdict(request_obj) for request_obj in build_broker_order_request_list_from_vplan(plan_obj)] == [
        asdict(request_obj) for request_obj in request_list]
    assert broker_obj.sent_request_list == [] and broker_obj.cancel_ref_list == []
    assert broker_obj.refresh_count_int == 2
