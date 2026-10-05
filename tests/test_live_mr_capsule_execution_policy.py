"""Recovery scenarios exercise real SQLite persistence and an offline broker."""
from dataclasses import replace
from datetime import timedelta
import json
import sqlite3

import pytest

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderRecord, BrokerOrderFill, SubmitBatchResult
from alpha.live.mr_capsule_recovery import reconcile_capsule_cycle, claim_capsule_request
from alpha.live.state_store_v2 import LiveStateStore
from test_live_mr_capsule_recovery import capsule_case, RECONCILE_TIMESTAMP_TS


def seed_execution(case_tuple, fill_by_asset_dict=None, *, unresolved_asset_str=None):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = case_tuple
    store_obj.mark_vplan_status(vplan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(vplan_obj.decision_plan_id_int, "submitted")
    fill_by_asset_dict = fill_by_asset_dict or {"AAPL": 80.0, "BIL": -40.0}
    position_dict = dict(vplan_obj.current_broker_position_map)
    for request_obj in build_broker_order_request_list_from_vplan(vplan_obj):
        fill_float = fill_by_asset_dict[request_obj.asset_str]
        complete_bool = fill_float == request_obj.amount_float
        record_obj = BrokerOrderRecord(
            broker_order_id_str=f"original:{request_obj.asset_str}", decision_plan_id_int=None, vplan_id_int=None,
            account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
            order_request_key_str=request_obj.order_request_key_str, broker_order_type_str="MOO", unit_str="shares",
            amount_float=request_obj.amount_float, filled_amount_float=abs(fill_float),
            status_str="Filled" if complete_bool else "Cancelled", submitted_timestamp_ts=vplan_obj.submission_timestamp_ts,
            submission_key_str=vplan_obj.submission_key_str,
            raw_payload_dict={"snapshot_source_str": "completed_order", "open_order_observed_bool": False})
        if request_obj.asset_str == unresolved_asset_str:
            record_obj = replace(record_obj, status_str="Submitted", raw_payload_dict={"snapshot_source_str": "open_order"})
        fill_list = [] if not fill_float else [BrokerOrderFill(
            broker_order_id_str=record_obj.broker_order_id_str, decision_plan_id_int=None, vplan_id_int=None,
            account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
            fill_amount_float=fill_float, fill_price_float=request_obj.sizing_reference_price_float,
            fill_timestamp_ts=vplan_obj.target_execution_timestamp_ts)]
        broker_obj.seed_broker_order_state(record_obj, broker_order_fill_list=fill_list)
        position_dict[request_obj.asset_str] = position_dict.get(request_obj.asset_str, 0.0) + fill_float
    broker_obj.seed_account_snapshot(account_route_str=release_obj.account_route_str, cash_float=-500.0,
        total_value_float=100000.0, position_amount_map=position_dict, snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS,
        session_mode_str="paper")
    return case_tuple


def reconcile_case(case_tuple, as_of_ts=RECONCILE_TIMESTAMP_TS):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = case_tuple
    return reconcile_capsule_cycle(store_obj, broker_obj, release_obj,
        store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str),
        store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int), as_of_ts)


def alert_rows(store_obj):
    with store_obj._connect() as connection_obj:
        return [dict(row_obj) for row_obj in connection_obj.execute("SELECT * FROM mr_capsule_execution_alert")]


@pytest.mark.parametrize("buy_fill_float", [0.0, 30.0])
def test_final_missed_buy_completes_from_actual_holdings_without_retry(capsule_case, buy_fill_float):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = seed_execution(
        capsule_case, {"AAPL": buy_fill_float, "BIL": -100.0})
    result_tuple = reconcile_case(capsule_case)
    assert result_tuple[0].passed_bool and result_tuple[1] == "accepted_residual"
    assert broker_obj.submitted_order_request_list == []
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map["AAPL"] == buy_fill_float
    residual_dict = result_tuple[2][0]
    assert (residual_dict["asset_str"], residual_dict["required_action_str"], residual_dict["residual_amount_float"]) == ("AAPL", "NONE", 80 - buy_fill_float)
    assert len(alert_rows(store_obj)) == 1


def test_bil_remainder_sold_once_and_fill_marked_late(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, _, tmp_path = seed_execution(capsule_case)
    result_tuple = reconcile_case(capsule_case)
    assert result_tuple[0].passed_bool and result_tuple[1] == "completed"
    request_obj, = broker_obj.submitted_order_request_list
    assert (request_obj.asset_str, request_obj.amount_float, request_obj.broker_order_type_str) == ("BIL", -60.0, "MKT")
    assert request_obj.execution_deadline_timestamp_str == "2024-02-01T16:00:00-05:00"
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map["BIL"] == 880.0
    late_fill_list = [row_dict for row_dict in store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
                     if row_dict["open_price_source_str"] == "late_execution"]
    assert len(late_fill_list) == 1 and late_fill_list[0]["official_open_price_float"] is None
    assert {row_dict["alert_kind_str"] for row_dict in alert_rows(store_obj)} == {"late_execution"}
    restarted_tuple = (LiveStateStore(str(tmp_path / "recovery.sqlite3")), *capsule_case[1:])
    reconcile_case(restarted_tuple)
    assert len(broker_obj.submitted_order_request_list) == 1
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).target_share_map == vplan_obj.target_share_map


@pytest.mark.parametrize("failure_str", ["timeout_after_accept", "timeout_before_send", "terminal_rejection"])
def test_single_attempt_survives_uncertain_send_and_rejection(capsule_case, monkeypatch, failure_str):
    store_obj, broker_obj, release_obj, vplan_obj, _, tmp_path = seed_execution(capsule_case)
    submit_fn = broker_obj.submit_order_request_list
    call_list = []
    def failed_submit(**argument_dict):
        call_list.append(argument_dict)
        if failure_str == "timeout_after_accept":
            submit_fn(**argument_dict)
            raise TimeoutError("ACK lost after broker accepted")
        if failure_str == "timeout_before_send":
            raise TimeoutError("No proof of whether sent")
        request_obj, = argument_dict["broker_order_request_list"]
        record_obj = BrokerOrderRecord(broker_order_id_str="late-rejected", decision_plan_id_int=None, vplan_id_int=None,
            account_route_str=release_obj.account_route_str, asset_str="BIL", order_request_key_str=request_obj.order_request_key_str,
            broker_order_type_str="MKT", unit_str="shares", amount_float=-60.0, filled_amount_float=0.0,
            status_str="Inactive", submitted_timestamp_ts=RECONCILE_TIMESTAMP_TS, submission_key_str=vplan_obj.submission_key_str,
            raw_payload_dict={"snapshot_source_str": "completed_order"})
        broker_obj.seed_broker_order_state(record_obj)
        return SubmitBatchResult(broker_order_record_list=[record_obj])
    monkeypatch.setattr(broker_obj, "submit_order_request_list", failed_submit)
    if failure_str.startswith("timeout_"):
        with pytest.raises(TimeoutError):
            reconcile_case(capsule_case)
        error_dict = store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).snapshot_metadata_dict[
            "mr_capsule_completion_send_error_dict"]
        assert error_dict["order_request_key_str"].endswith(":late:BIL")
        assert error_dict["error_str"]
        assert "unresolved_execution" in {row_dict["alert_kind_str"] for row_dict in alert_rows(store_obj)}
    else:
        result_tuple = reconcile_case(capsule_case)
    restarted_tuple = (LiveStateStore(str(tmp_path / "recovery.sqlite3")), *capsule_case[1:])
    next_result_tuple = reconcile_case(restarted_tuple)
    assert len(call_list) == 1
    assert next_result_tuple[0].passed_bool == (failure_str != "timeout_before_send")
    if failure_str == "timeout_before_send":
        assert next_result_tuple[1] == "awaiting_evidence"
        assert next_result_tuple[2][0]["required_action_str"] == "VERIFY"
    if failure_str == "terminal_rejection":
        assert result_tuple[2][0]["required_action_str"] == "SELL"
        assert abs(result_tuple[2][0]["residual_amount_float"]) == 60.0


def test_verified_sale_completes_while_unrelated_buy_remains_pending(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = seed_execution(
        capsule_case, {"AAPL": 30.0, "BIL": -40.0}, unresolved_asset_str="AAPL")
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(
        snapshot_obj, open_order_id_list=["original:AAPL"])
    result_tuple = reconcile_case(capsule_case)
    request_obj, = broker_obj.submitted_order_request_list
    assert (request_obj.asset_str, request_obj.amount_float) == ("BIL", -60.0)
    assert not result_tuple[0].passed_bool and result_tuple[1] == "awaiting_evidence"
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitted"
    assert broker_obj.get_account_snapshot(release_obj.account_route_str).position_amount_map["BIL"] == 880.0
    assert any(row_dict["asset_str"] == "AAPL" and row_dict["required_action_str"] == "VERIFY"
               for row_dict in result_tuple[2])


def test_unexplained_other_holding_blocks_sale_even_with_known_pending_buy(capsule_case):
    _, broker_obj, release_obj, _, _, _ = seed_execution(
        capsule_case, {"AAPL": 30.0, "BIL": -40.0}, unresolved_asset_str="AAPL")
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj,
        open_order_id_list=["original:AAPL"], position_amount_map={**snapshot_obj.position_amount_map, "MSFT": 9.0})
    result_tuple = reconcile_case(capsule_case)
    assert not result_tuple[0].passed_bool
    assert broker_obj.submitted_order_request_list == []
    assert any(row_dict["asset_str"] == "BIL" and row_dict["required_action_str"] == "VERIFY"
               for row_dict in result_tuple[2])


@pytest.mark.parametrize("block_str", ["original_open", "other_client_open", "observed_next_day", "after_close"])
def test_no_recovery_when_original_uncertain_or_outside_original_session(capsule_case, block_str):
    store_obj, broker_obj, release_obj, _, _, _ = seed_execution(capsule_case,
        unresolved_asset_str="BIL" if block_str == "original_open" else None)
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    as_of_ts = RECONCILE_TIMESTAMP_TS
    if block_str == "other_client_open":
        broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, open_order_id_list=["manual-pending"])
    elif block_str == "observed_next_day":
        broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj,
            snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS + timedelta(days=1))
    elif block_str == "after_close":
        as_of_ts = RECONCILE_TIMESTAMP_TS.replace(hour=16, minute=5)
        broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    result_tuple = reconcile_case(capsule_case, as_of_ts)
    assert broker_obj.submitted_order_request_list == []
    assert result_tuple[0].passed_bool == (block_str in {"observed_next_day", "after_close"})


def seed_manual_sale(case_tuple):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = case_tuple
    request_obj = next(request_obj for request_obj in build_broker_order_request_list_from_vplan(vplan_obj) if request_obj.asset_str == "BIL")
    record_obj = BrokerOrderRecord(broker_order_id_str="manual-order", decision_plan_id_int=None, vplan_id_int=None,
        account_route_str=release_obj.account_route_str, asset_str="BIL", order_request_key_str=None,
        broker_order_type_str="MKT", unit_str="shares", amount_float=-60.0, filled_amount_float=60.0,
        status_str="Filled", submitted_timestamp_ts=RECONCILE_TIMESTAMP_TS,
        raw_payload_dict={"snapshot_source_str": "completed_order"})
    broker_obj.seed_broker_order_state(record_obj, broker_order_fill_list=[BrokerOrderFill(
        broker_order_id_str="manual-order", decision_plan_id_int=None, vplan_id_int=None,
        account_route_str=release_obj.account_route_str, asset_str="BIL", fill_amount_float=-60.0,
        fill_price_float=100.02, fill_timestamp_ts=RECONCILE_TIMESTAMP_TS)])
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj,
        position_amount_map={**snapshot_obj.position_amount_map, "BIL": 880.0})
    return replace(request_obj, order_request_key_str=f"{vplan_obj.submission_key_str}:manual:manual-order", amount_float=-60.0, broker_order_type_str="MKT")


@pytest.mark.parametrize("crash_after_claim_bool", [False, True])
def test_manual_repair_adopted_after_restart_without_sql_edits(capsule_case, crash_after_claim_bool):
    store_obj, broker_obj, release_obj, vplan_obj, _, tmp_path = seed_execution(capsule_case)
    manual_request_obj = seed_manual_sale(capsule_case)
    if crash_after_claim_bool:
        assert claim_capsule_request(store_obj, vplan_obj, manual_request_obj, "manual", RECONCILE_TIMESTAMP_TS)
    restarted_tuple = (LiveStateStore(str(tmp_path / "recovery.sqlite3")), *capsule_case[1:])
    result_tuple = reconcile_case(restarted_tuple)
    assert result_tuple[0].passed_bool
    assert broker_obj.submitted_order_request_list == []
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "completed"
    order_list = store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    assert len(order_list) == 3
    assert next(row_dict for row_dict in order_list if row_dict["broker_order_id_str"] == "manual-order")["order_request_key_str"] == manual_request_obj.order_request_key_str
    assert {row_dict["open_price_source_str"] for row_dict in store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
            if row_dict["fill_price_float"] == 100.02} == {"late_execution"}


def test_unknown_original_cannot_be_cured_by_matching_manual_holdings(capsule_case):
    seed_execution(capsule_case, unresolved_asset_str="BIL")
    seed_manual_sale(capsule_case)
    result_tuple = reconcile_case(capsule_case)
    assert result_tuple[1] == "awaiting_evidence" and not result_tuple[0].passed_bool
    assert capsule_case[1].submitted_order_request_list == []


def test_unrelated_holdings_mismatch_has_symbol_quantity_and_action(capsule_case):
    store_obj, broker_obj, release_obj, _, _, _ = seed_execution(capsule_case, {"AAPL": 80.0, "BIL": -100.0})
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj,
        position_amount_map={**snapshot_obj.position_amount_map, "MSFT": 7.0})
    result_tuple = reconcile_case(capsule_case)
    assert not result_tuple[0].passed_bool
    residual_dict, = result_tuple[2]
    assert (residual_dict["asset_str"], residual_dict["residual_amount_float"], residual_dict["required_action_str"]) == ("MSFT", -2.0, "VERIFY")
    assert len(alert_rows(store_obj)) == 1


@pytest.mark.parametrize("bad_snapshot_str", ["wrong_account", "stale", "nonfinite"])
def test_invalid_snapshot_never_overwrites_pod_holdings(capsule_case, bad_snapshot_str):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = seed_execution(capsule_case)
    before_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    update_dict = {"account_route_str": "WRONG"} if bad_snapshot_str == "wrong_account" else (
        {"snapshot_timestamp_ts": vplan_obj.submission_timestamp_ts} if bad_snapshot_str == "stale" else {"cash_float": float("nan")})
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, **update_dict)
    assert not reconcile_case(capsule_case)[0].passed_bool
    assert len(alert_rows(store_obj)) == 1
    assert store_obj.get_pod_state(release_obj.pod_id_str) == before_state_obj
    assert broker_obj.submitted_order_request_list == []


def test_settlement_and_alert_rollback_as_one_transaction(capsule_case):
    store_obj, _, release_obj, vplan_obj, _, _ = seed_execution(capsule_case, {"AAPL": 0.0, "BIL": -100.0})
    before_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("CREATE TRIGGER fail_complete BEFORE UPDATE OF status_str ON decision_plan WHEN NEW.status_str='completed' BEGIN SELECT RAISE(ABORT,'completion crash'); END")
    with pytest.raises(sqlite3.IntegrityError, match="completion crash"):
        reconcile_case(capsule_case)
    assert store_obj.get_pod_state(release_obj.pod_id_str) == before_state_obj
    assert alert_rows(store_obj) == []
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "submitted"
    assert "mr_capsule_execution_result_dict" not in store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).snapshot_metadata_dict
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM vplan_reconciliation_snapshot WHERE stage_str='post_execution'").fetchone()[0] == 0
        connection_obj.execute("DROP TRIGGER fail_complete")
    assert reconcile_case(capsule_case)[0].passed_bool


def test_broker_holdings_rechecked_before_recovery_sale(capsule_case, monkeypatch):
    _, broker_obj, release_obj, _, _, _ = seed_execution(capsule_case)
    snapshot_fn = broker_obj.get_capsule_account_snapshot
    call_count_list = [0]
    def changed_holdings(account_route_str):
        call_count_list[0] += 1
        snapshot_obj = snapshot_fn(account_route_str)
        if call_count_list[0] >= 2:
            return replace(snapshot_obj, position_amount_map={**snapshot_obj.position_amount_map, "BIL": 880.0})
        return snapshot_obj
    monkeypatch.setattr(broker_obj, "get_capsule_account_snapshot", changed_holdings)
    result_tuple = reconcile_case(capsule_case)
    assert not result_tuple[0].passed_bool
    assert broker_obj.submitted_order_request_list == []


def test_optional_nonfinite_margin_does_not_block_resolved_execution(capsule_case):
    store_obj, broker_obj, release_obj, _, _, _ = seed_execution(capsule_case, {"AAPL": 0.0, "BIL": -100.0})
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj,
        available_funds_float=float("nan"), excess_liquidity_float=float("inf"))
    assert reconcile_case(capsule_case)[0].passed_bool
    alert_dict, = alert_rows(store_obj)
    payload_dict = json.loads(alert_dict["payload_json_str"])
    assert payload_dict["available_funds_float"] is None
    assert payload_dict["excess_liquidity_float"] is None


def test_original_open_reference_preserved_but_late_fill_is_separate(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = seed_execution(capsule_case)
    for asset_str, price_float in (("BIL", 99.9), ("AAPL", 125.1)):
        broker_obj.seed_session_open_price(account_route_str=release_obj.account_route_str,
            session_date_str="2024-02-01", asset_str=asset_str, official_open_price_float=price_float,
            open_price_source_str="test_official_open", snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS)
    assert reconcile_case(capsule_case)[0].passed_bool
    fill_list = store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int, include_order_identity_bool=True)
    original_list = [row_dict for row_dict in fill_list if row_dict["broker_order_id_str"].startswith("original:")]
    assert {row_dict["open_price_source_str"] for row_dict in original_list} == {"test_official_open"}
    assert {row_dict["official_open_price_float"] for row_dict in original_list} == {99.9, 125.1}
    late_dict, = [row_dict for row_dict in fill_list if not row_dict["broker_order_id_str"].startswith("original:")]
    assert late_dict["open_price_source_str"] == "late_execution" and late_dict["official_open_price_float"] is None


def test_missing_open_reference_does_not_prevent_sale_completion(capsule_case, monkeypatch):
    store_obj, broker_obj, release_obj, _, _, _ = seed_execution(capsule_case)
    def unavailable_reference(**argument_dict):
        raise TimeoutError("Open reference unavailable")
    monkeypatch.setattr(broker_obj, "get_session_open_price_list", unavailable_reference)
    assert reconcile_case(capsule_case)[0].passed_bool
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).snapshot_metadata_dict["mr_capsule_open_reference_error_str"] == "Open reference unavailable"


@pytest.mark.parametrize("invalid_float", [float("nan"), float("inf")])
def test_invalid_untraded_holding_alerts_the_actual_symbol(capsule_case, invalid_float):
    store_obj, broker_obj, release_obj, _, _, _ = seed_execution(capsule_case)
    before_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj,
        position_amount_map={**snapshot_obj.position_amount_map, "KEEP": invalid_float})
    result_tuple = reconcile_case(capsule_case)
    assert not result_tuple[0].passed_bool
    residual_dict = next(row_dict for row_dict in result_tuple[2] if row_dict["asset_str"] == "KEEP")
    assert residual_dict["required_action_str"] == "VERIFY" and residual_dict["residual_amount_float"] is None
    assert store_obj.get_pod_state(release_obj.pod_id_str) == before_state_obj
    assert len(alert_rows(store_obj)) == 1


def test_existing_store_getters_do_not_parse_optional_evidence(capsule_case):
    store_obj, _, _, vplan_obj, _, _ = seed_execution(capsule_case, {"AAPL": 80.0, "BIL": -100.0})
    assert reconcile_case(capsule_case)[0].passed_bool
    fill_list = store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    order_list = store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    assert all("account_route_str" not in row_dict and "raw_payload_dict" not in row_dict for row_dict in fill_list + order_list)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE vplan_fill SET raw_payload_json_str='bad optional payload'")
        connection_obj.execute("UPDATE vplan_broker_order SET raw_payload_json_str='bad optional payload'")
    assert store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int) == fill_list
    assert store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int) == order_list


def test_current_tick_open_cannot_relabel_original_session(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = seed_execution(capsule_case)
    broker_obj.seed_session_open_price(account_route_str=release_obj.account_route_str,
        session_date_str="2024-02-01", asset_str="BIL", official_open_price_float=120.0,
        open_price_source_str="ibkr.tick_open", snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS + timedelta(days=1))
    assert reconcile_case(capsule_case)[0].passed_bool
    assert "historical" in store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).snapshot_metadata_dict["mr_capsule_open_reference_error_str"]
    assert all(row_dict["official_open_price_float"] != 120.0 for row_dict in store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int))
