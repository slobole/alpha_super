"""SQL-backed capsule execution/recovery checks using only the in-memory broker stub."""
from dataclasses import replace
from datetime import datetime
import json
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from alpha.live import logging_utils, runner as runner_module
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord, DecisionPlan, LiveRelease, PodState
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.runner import build_decision_plans, build_vplans, expire_stale_decision_plans, post_execution_reconcile, submit_ready_vplans
from alpha.live.state_store_v2 import LiveStateStore
from test_live_runner import _install_partial_submit_ack_stub, _install_pending_submit_truth_stub


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
SUBMIT_TIMESTAMP_TS = datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ)
RECONCILE_TIMESTAMP_TS = datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ)


@pytest.fixture(params=["dv2_vix_gated", "hpi_vote_vix_gated"])
def capsule_case(request, tmp_path, monkeypatch):
    strategy_import_str = f"strategies.mr_capsule.strategy_mr_{request.param}_bil"
    profile_str = "norgate_eod_sp500_hpi_mr_capsule_pit" if "hpi" in request.param else "norgate_eod_sp500_mr_capsule_pit"
    release_obj = LiveRelease(
        release_id_str="capsule.recovery.v1", user_id_str="user_test", pod_id_str="capsule_test",
        account_route_str="DU_TEST", strategy_import_str=strategy_import_str, mode_str="paper",
        session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready", execution_policy_str="next_open_moo",
        data_profile_str=profile_str, params_dict={"margin_account_confirmed_bool": True}, risk_profile_str="standard",
        enabled_bool=True, source_path_str=str(tmp_path / "synthetic-release.yaml"), pod_budget_fraction_float=1.0,
    )
    state_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    state_store_obj.upsert_release(release_obj)
    base_position_dict = {"BIL": 980.0, "MSFT": 5.0}
    signal_timestamp_ts = datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ)
    strategy_state_dict = {"mr_capsule_strategy_import_str": strategy_import_str, "trade_id_int": 7}
    state_store_obj.upsert_pod_state(PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=base_position_dict, cash_float=1000.0, total_value_float=100000.0,
        strategy_state_dict=strategy_state_dict, updated_timestamp_ts=signal_timestamp_ts,
        snapshot_stage_str="eod", snapshot_source_str="broker",
    ))
    decision_plan_obj = state_store_obj.insert_decision_plan(DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str, pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str, signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=SUBMIT_TIMESTAMP_TS,
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo", decision_base_position_map=base_position_dict,
        snapshot_metadata_dict={
            "strategy_import_str": strategy_import_str, "sizing_contract_str": "mr_capsule_close_targets_v1",
            "mr_capsule_parking_mode_str": "bil", "decision_nav_float": 100000.0, "decision_cash_float": 1000.0,
            "norgate_data_profile_str": profile_str, "norgate_snapshot_date_str": "2024-01-31",
            "norgate_manifest_hash_str": "synthetic_fixture_hash",
        },
        strategy_state_dict=strategy_state_dict, entry_target_weight_map_dict={"AAPL": 0.1},
        entry_priority_list=["AAPL"], target_share_map_dict={"BIL": 880.0},
    ))
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_TEST", cash_float=1000.0, total_value_float=100500.0,
        position_amount_map=base_position_dict, snapshot_timestamp_ts=SUBMIT_TIMESTAMP_TS, session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU_TEST", asset_reference_price_map={"AAPL": 125.0, "BIL": 100.0},
        snapshot_timestamp_ts=SUBMIT_TIMESTAMP_TS,
    )
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(tmp_path / "critical.jsonl"))
    runner_kwarg_dict = {"log_path_str": str(tmp_path / "events.jsonl"), "trace_enabled_bool": False}
    build_result_dict = build_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", **runner_kwarg_dict,
    )
    assert build_result_dict["created_vplan_count_int"] == 1
    vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    assert vplan_obj.target_share_map == {"AAPL": 80.0, "BIL": 880.0}
    assert vplan_obj.order_delta_map == {"AAPL": 80.0, "BIL": -100.0}
    assert vplan_obj.current_broker_position_map["MSFT"] == 5.0
    # Close-T entry dollars = 100000 * 10% = $10000 despite broker NAV now being $100500.
    assert next(row_obj for row_obj in vplan_obj.vplan_row_list if row_obj.asset_str == "AAPL").estimated_target_notional_float == 10000.0
    assert state_store_obj.get_decision_plan_by_id(decision_plan_obj.decision_plan_id_int).target_share_map_dict == {"BIL": 880.0}
    return state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path


@pytest.mark.parametrize("outcome_str,bil_filled_float,status_str", [
    ("filled", 100.0, "Filled"), ("partial", 40.0, "Cancelled"), ("rejected", 0.0, "Inactive"),
])
def test_capsule_cycle_preserves_stocks_and_never_replays_submitted_batch(
    capsule_case, outcome_str, bil_filled_float, status_str, monkeypatch,
):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    _install_pending_submit_truth_stub(broker_adapter_obj, {
        "AAPL": {"status_str": "Filled", "filled_amount_float": 80.0, "fill_price_float": 125.0},
        "BIL": {"status_str": status_str, "filled_amount_float": bil_filled_float, "fill_price_float": 100.0},
    })
    submit_batch_fn = broker_adapter_obj.submit_order_request_list
    submitted_batch_list = []

    def capture_submit(**submit_kwarg_dict):
        submitted_batch_list.append(list(submit_kwarg_dict["broker_order_request_list"]))
        return submit_batch_fn(**submit_kwarg_dict)

    monkeypatch.setattr(broker_adapter_obj, "submit_order_request_list", capture_submit)
    submit_result_dict = submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert submit_result_dict["submitted_vplan_count_int"] == 1
    assert {order_obj.asset_str: order_obj.amount_float for order_obj in submitted_batch_list[0]} == {"AAPL": 80.0, "BIL": -100.0}
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    retry_result_dict = submit_ready_vplans(
        restarted_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert retry_result_dict["submitted_vplan_count_int"] == 0
    assert len(submitted_batch_list) == 1
    actual_position_dict = {"AAPL": 80.0, "BIL": 980.0 - bil_filled_float, "MSFT": 5.0}
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_TEST", cash_float=1000.0 + bil_filled_float * 100.0 - 10000.0,
        total_value_float=100500.0, position_amount_map=actual_position_dict,
        snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS, session_mode_str="paper",
    )
    reconcile_result_dict = post_execution_reconcile(
        restarted_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict,
    )
    latest_vplan_obj = restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    latest_decision_obj = restarted_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    successful_bool = outcome_str == "filled"
    assert reconcile_result_dict["completed_vplan_count_int"] == int(successful_bool)
    assert latest_vplan_obj.status_str == ("completed" if successful_bool else "submitted")
    assert latest_decision_obj.status_str == ("completed" if successful_bool else "submitted")
    state_obj = restarted_store_obj.get_pod_state(release_obj.pod_id_str)
    if successful_bool:
        assert state_obj.position_amount_map == actual_position_dict
        assert state_obj.snapshot_stage_str == "post_execution"
    else:
        assert state_obj.position_amount_map == actual_position_dict  # broker truth is saved even when the cycle stays unresolved
        assert latest_vplan_obj.target_share_map["BIL"] - actual_position_dict["BIL"] == -(100.0 - bil_filled_float)
    assert len(submitted_batch_list) == 1


def test_missing_bil_ack_stays_unresolved_after_restart(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    _install_partial_submit_ack_stub(broker_adapter_obj, {"AAPL"})
    submit_result_dict = submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert submit_result_dict["submitted_vplan_count_int"] == 1
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    latest_vplan_obj = restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    assert latest_vplan_obj.submit_ack_status_str == "missing_critical"
    assert latest_vplan_obj.missing_ack_count_int == 1
    missing_ack_list = [row_dict["asset_str"] for row_dict in restarted_store_obj.get_broker_ack_row_dict_list_for_vplan(vplan_obj.vplan_id_int) if not row_dict["broker_response_ack_bool"]]
    assert missing_ack_list == ["BIL"]
    retry_result_dict = submit_ready_vplans(
        restarted_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert retry_result_dict["submitted_vplan_count_int"] == 0
    reconcile_result_dict = post_execution_reconcile(
        restarted_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict,
    )
    assert reconcile_result_dict["completed_vplan_count_int"] == 0
    assert restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "submitted"
    assert restarted_store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == {"BIL": 980.0, "MSFT": 5.0}


def test_matching_positions_do_not_replace_missing_order_and_fill_evidence(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    _install_partial_submit_ack_stub(broker_adapter_obj, {"AAPL"})
    submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    # An external transfer/manual trade can make holdings match without proving these orders filled.
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_TEST", cash_float=1000.0, total_value_float=100500.0,
        position_amount_map={"AAPL": 80.0, "BIL": 880.0, "MSFT": 5.0},
        snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS, session_mode_str="paper",
    )
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    reconcile_result_dict = post_execution_reconcile(
        restarted_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict,
    )
    assert reconcile_result_dict["completed_vplan_count_int"] == 0
    assert restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "submitted"


def test_missing_ack_recovers_from_matching_broker_order_and_fill_evidence(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    _install_partial_submit_ack_stub(broker_adapter_obj, {"AAPL"})
    submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    for order_dict in state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int):
        asset_str = order_dict["asset_str"]
        amount_float = float(order_dict["amount_float"])
        fill_price_float = {"AAPL": 125.0, "BIL": 100.0}[asset_str]
        broker_adapter_obj.seed_broker_order_state(BrokerOrderRecord(
            broker_order_id_str=order_dict["broker_order_id_str"], decision_plan_id_int=None, vplan_id_int=None,
            account_route_str="DU_TEST", asset_str=asset_str, order_request_key_str=order_dict["order_request_key_str"],
            broker_order_type_str="MOO", unit_str="shares", amount_float=amount_float,
            filled_amount_float=abs(amount_float), remaining_amount_float=0.0, avg_fill_price_float=fill_price_float,
            status_str="Filled", submitted_timestamp_ts=SUBMIT_TIMESTAMP_TS, last_status_timestamp_ts=RECONCILE_TIMESTAMP_TS,
            submission_key_str=vplan_obj.submission_key_str, raw_payload_dict={},
        ), broker_order_fill_list=[BrokerOrderFill(
            broker_order_id_str=order_dict["broker_order_id_str"], decision_plan_id_int=None, vplan_id_int=None,
            account_route_str="DU_TEST", asset_str=asset_str, fill_amount_float=amount_float,
            fill_price_float=fill_price_float, fill_timestamp_ts=RECONCILE_TIMESTAMP_TS, raw_payload_dict={},
        )])
    actual_position_dict = {"AAPL": 80.0, "BIL": 880.0, "MSFT": 5.0}
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_TEST", cash_float=1000.0, total_value_float=100500.0,
        position_amount_map=actual_position_dict, snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS, session_mode_str="paper",
    )
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    reconcile_result_dict = post_execution_reconcile(
        restarted_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict,
    )
    assert reconcile_result_dict["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert restarted_store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == actual_position_dict
    assert len(restarted_store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)) == 2


@pytest.mark.parametrize("cash_float", [1_010.0, 990.0, 0.0, -1_000.0, 50_000.0])
def test_cash_change_after_vplan_submits_frozen_orders_once(capsule_case, cash_float, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    expected_request_list = runner_module.build_broker_order_request_list_from_vplan(vplan_obj)
    snapshot_obj = broker_adapter_obj.get_account_snapshot(release_obj.account_route_str)
    changed_nav_float = snapshot_obj.net_liq_float + cash_float - snapshot_obj.cash_float
    broker_adapter_obj._snapshot_map[release_obj.account_route_str] = replace(
        snapshot_obj, cash_float=cash_float, net_liq_float=changed_nav_float, total_value_float=changed_nav_float,
    )

    def unexpected_rebuild(*argument_tuple, **keyword_dict):
        raise AssertionError("A cash change must not refresh quotes or resize the saved VPlan.")

    monkeypatch.setattr(broker_adapter_obj, "get_live_price_snapshot", unexpected_rebuild)
    monkeypatch.setattr(runner_module, "build_vplan", unexpected_rebuild)
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    submit_result_dict = submit_ready_vplans(
        restarted_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert submit_result_dict["submitted_vplan_count_int"] == 1
    assert submit_result_dict["blocked_action_count_int"] == 0
    assert sorted(broker_adapter_obj.submitted_order_request_list, key=lambda request_obj: request_obj.order_request_key_str) == sorted(expected_request_list, key=lambda request_obj: request_obj.order_request_key_str)
    assert restarted_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).order_delta_map == vplan_obj.order_delta_map
    assert restarted_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).snapshot_metadata_dict["decision_cash_float"] == 1_000.0
    retry_result_dict = submit_ready_vplans(
        restarted_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert retry_result_dict["submitted_vplan_count_int"] == 0
    assert sorted(broker_adapter_obj.submitted_order_request_list, key=lambda request_obj: request_obj.order_request_key_str) == sorted(expected_request_list, key=lambda request_obj: request_obj.order_request_key_str)


@pytest.mark.parametrize("drift_str", [
    "positions", "open_orders", "cash_nan", "cash_inf", "cash_negative_inf",
    "nav_nan", "nav_inf", "nav_zero", "nav_negative",
])
def test_capsule_account_drift_after_vplan_blocks_submission(capsule_case, drift_str):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    snapshot_obj = broker_adapter_obj.get_account_snapshot(release_obj.account_route_str)
    changed_field_dict = {
        "positions": {"cash_float": 1010.0, "position_amount_map": {"BIL": 970.0, "MSFT": 5.0}},
        "open_orders": {"cash_float": 1010.0, "open_order_id_list": ["external_order"]},
        "cash_nan": {"cash_float": float("nan")},
        "cash_inf": {"cash_float": float("inf")},
        "cash_negative_inf": {"cash_float": -float("inf")},
        "nav_nan": {"net_liq_float": float("nan")},
        "nav_inf": {"net_liq_float": float("inf")},
        "nav_zero": {"net_liq_float": 0.0},
        "nav_negative": {"net_liq_float": -1.0},
    }[drift_str]
    broker_adapter_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, **changed_field_dict)
    submit_result_dict = submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    assert submit_result_dict["submitted_vplan_count_int"] == 0
    assert submit_result_dict["blocked_action_count_int"] == 1
    assert broker_adapter_obj.submitted_order_request_list == []
    assert state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "blocked"
    assert state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "blocked"


def test_next_day_eod_does_not_bypass_an_unresolved_capsule_batch(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    _install_partial_submit_ack_stub(broker_adapter_obj, {"AAPL"})
    submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    next_close_ts = datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ)
    old_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
    state_store_obj.upsert_pod_state(replace(old_state_obj, updated_timestamp_ts=next_close_ts, snapshot_stage_str="eod"))
    prior_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    builder_call_list = []

    def next_decision(**_kwarg_dict):
        builder_call_list.append(True)
        return replace(prior_plan_obj, decision_plan_id_int=None, status_str="planned", signal_timestamp_ts=next_close_ts,
                       submission_timestamp_ts=datetime(2024, 2, 2, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
                       target_execution_timestamp_ts=datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ))

    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: [release_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release", next_decision)
    result_dict = build_decision_plans(
        LiveStateStore(str(tmp_path / "recovery.sqlite3")), datetime(2024, 2, 1, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        str(tmp_path), auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict,
    )
    assert result_dict["created_decision_plan_count_int"] == 0
    assert result_dict["skipped_decision_plan_count_int"] == 1
    assert builder_call_list == []
    assert state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).decision_plan_id_int == prior_plan_obj.decision_plan_id_int


def test_invalid_capsule_state_does_not_starve_an_unrelated_due_pod(capsule_case, monkeypatch):
    state_store_obj, _, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    state_store_obj.mark_vplan_status(vplan_obj.vplan_id_int, "completed")
    state_store_obj.mark_decision_plan_status(vplan_obj.decision_plan_id_int, "completed")
    other_release_obj = replace(
        release_obj, release_id_str="unrelated.v1", pod_id_str="unrelated_pod", account_route_str="DU_OTHER",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy", data_profile_str="norgate_eod_sp500_pit", params_dict={},
    )
    state_store_obj.upsert_release(other_release_obj)
    prior_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    builder_pod_list = []

    def build_or_fail(release_obj, **_kwarg_dict):
        builder_pod_list.append(release_obj.pod_id_str)
        if release_obj.pod_id_str == "capsule_test":
            raise ValueError("MR capsule requires a finite, exact-session EOD account snapshot.")
        return replace(
            prior_plan_obj, release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str, decision_plan_id_int=None, status_str="planned",
            decision_base_position_map={}, target_share_map_dict={}, snapshot_metadata_dict={},
            signal_timestamp_ts=datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 2, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        )

    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: [release_obj, other_release_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release", build_or_fail)
    result_dict = build_decision_plans(
        state_store_obj, datetime(2024, 2, 1, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ), str(tmp_path),
        auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict,
    )
    assert builder_pod_list == ["capsule_test", "unrelated_pod"]
    assert result_dict["created_decision_plan_count_int"] == 1
    assert result_dict["skipped_decision_plan_count_int"] == 1
    assert state_store_obj.get_latest_decision_plan_for_pod("unrelated_pod") is not None


def test_capsule_completion_rolls_back_both_statuses_and_recovers_after_restart(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    _install_pending_submit_truth_stub(broker_adapter_obj, {
        "AAPL": {"status_str": "Filled", "filled_amount_float": 80.0, "fill_price_float": 125.0},
        "BIL": {"status_str": "Filled", "filled_amount_float": 100.0, "fill_price_float": 100.0},
    })
    submit_ready_vplans(
        state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict,
    )
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_TEST", cash_float=1000.0, total_value_float=100500.0,
        position_amount_map={"AAPL": 80.0, "BIL": 880.0, "MSFT": 5.0},
        snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS, session_mode_str="paper",
    )
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TRIGGER fail_second_completion_write
            BEFORE UPDATE OF status_str ON decision_plan WHEN NEW.status_str = 'completed'
            BEGIN SELECT RAISE(ABORT, 'simulated completion crash'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="simulated completion crash"):
        post_execution_reconcile(state_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict)
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    assert restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "submitted"
    assert restarted_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "submitted"
    with restarted_store_obj._connect() as connection_obj:
        connection_obj.execute("DROP TRIGGER fail_second_completion_write")
    result_dict = post_execution_reconcile(restarted_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert restarted_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert len(restarted_store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)) == 2


@pytest.mark.parametrize("terminal_path_str", ["expiry", "pre_submit_block"])
def test_provably_unsubmitted_cycle_can_advance_next_day_without_committing_proposed_state(
    capsule_case, terminal_path_str, monkeypatch,
):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    original_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
    proposed_state_dict = {**original_state_obj.strategy_state_dict, "trade_id_int": 999}
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET strategy_state_json_str = ? WHERE decision_plan_id_int = ?",
                               (json.dumps(proposed_state_dict), vplan_obj.decision_plan_id_int))
    next_close_ts = datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ)
    next_decision_ts = datetime(2024, 2, 1, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ)
    if terminal_path_str == "expiry":
        result_dict = expire_stale_decision_plans(state_store_obj, next_decision_ts, str(tmp_path), **runner_kwarg_dict)
        assert result_dict["expired_decision_plan_count_int"] == 1
    else:
        snapshot_obj = broker_adapter_obj.get_account_snapshot(release_obj.account_route_str)
        broker_adapter_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, open_order_id_list=["external_order"])
        result_dict = submit_ready_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
                                         vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
        assert result_dict["submitted_vplan_count_int"] == 0
    abandoned_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    assert abandoned_plan_obj.snapshot_metadata_dict["mr_capsule_unsubmitted_cycle_abandoned_bool"] is True
    assert state_store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == original_state_obj.strategy_state_dict
    assert not state_store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    state_store_obj.upsert_pod_state(replace(original_state_obj, updated_timestamp_ts=next_close_ts, snapshot_stage_str="eod"))
    builder_state_list = []

    def next_decision(pod_state_obj, **_kwarg_dict):
        builder_state_list.append(pod_state_obj.strategy_state_dict)
        return replace(
            abandoned_plan_obj, decision_plan_id_int=None, status_str="planned", signal_timestamp_ts=next_close_ts,
            snapshot_metadata_dict={key_str: value_obj for key_str, value_obj in abandoned_plan_obj.snapshot_metadata_dict.items()
                                    if key_str != "mr_capsule_unsubmitted_cycle_abandoned_bool"},
            strategy_state_dict={**pod_state_obj.strategy_state_dict, "trade_id_int": 8},
            submission_timestamp_ts=datetime(2024, 2, 2, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        )

    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: [release_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release", next_decision)
    result_dict = build_decision_plans(state_store_obj, next_decision_ts, str(tmp_path),
                                     auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)
    assert result_dict["created_decision_plan_count_int"] == 1
    assert builder_state_list == [original_state_obj.strategy_state_dict]


def test_claimed_without_ack_cannot_be_abandoned_or_expired_by_manual_submit(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    assert state_store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    assert not state_store_obj.abandon_unsubmitted_mr_capsule_cycle(vplan_obj.decision_plan_id_int, "expired")
    after_window_ts = datetime(2024, 2, 1, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ)
    expire_result_dict = expire_stale_decision_plans(state_store_obj, after_window_ts, str(tmp_path), **runner_kwarg_dict)
    assert expire_result_dict["expired_decision_plan_count_int"] == 0
    submit_result_dict = submit_ready_vplans(state_store_obj, broker_adapter_obj, after_window_ts, "paper", False,
                                           vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert submit_result_dict["submitted_vplan_count_int"] == 0
    assert state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "submitting"
    latest_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    assert latest_plan_obj.status_str == "vplan_ready"
    assert not latest_plan_obj.snapshot_metadata_dict.get("mr_capsule_unsubmitted_cycle_abandoned_bool")
    assert broker_adapter_obj.submitted_order_request_list == []


def test_capsule_vplan_validation_error_does_not_starve_another_due_plan(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    original_plan_obj = state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    bad_release_obj = replace(release_obj, release_id_str="bad_capsule.v1", pod_id_str="bad_capsule", account_route_str="DU_BAD")
    healthy_release_obj = replace(
        release_obj, release_id_str="healthy_parent.v1", pod_id_str="healthy_parent", account_route_str="DU_HEALTHY",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy", data_profile_str="norgate_eod_sp500_pit", params_dict={},
    )
    for added_release_obj in (bad_release_obj, healthy_release_obj):
        state_store_obj.upsert_release(added_release_obj)
        bad_bool = added_release_obj.pod_id_str == "bad_capsule"
        state_store_obj.insert_decision_plan(replace(
            original_plan_obj, decision_plan_id_int=None, status_str="planned", release_id_str=added_release_obj.release_id_str,
            pod_id_str=added_release_obj.pod_id_str, account_route_str=added_release_obj.account_route_str,
            decision_base_position_map=original_plan_obj.decision_base_position_map if bad_bool else {},
            target_share_map_dict=original_plan_obj.target_share_map_dict if bad_bool else {},
            snapshot_metadata_dict={**original_plan_obj.snapshot_metadata_dict, "norgate_data_profile_str": added_release_obj.data_profile_str},
        ))
        broker_adapter_obj.seed_account_snapshot(
            account_route_str=added_release_obj.account_route_str, cash_float=float("inf") if bad_bool else 100000.0,
            total_value_float=100500.0 if bad_bool else 100000.0,
            position_amount_map=original_plan_obj.decision_base_position_map if bad_bool else {},
            snapshot_timestamp_ts=SUBMIT_TIMESTAMP_TS, session_mode_str="paper",
        )
        broker_adapter_obj.seed_live_price_snapshot(
            account_route_str=added_release_obj.account_route_str,
            asset_reference_price_map={"AAPL": 125.0, "BIL": 100.0}, snapshot_timestamp_ts=SUBMIT_TIMESTAMP_TS,
        )
    result_dict = build_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", **runner_kwarg_dict)
    assert result_dict["created_vplan_count_int"] == 1
    assert result_dict["blocked_action_count_int"] == 1
    bad_plan_obj = state_store_obj.get_latest_decision_plan_for_pod("bad_capsule")
    assert bad_plan_obj.status_str == "blocked"
    assert bad_plan_obj.snapshot_metadata_dict["mr_capsule_unsubmitted_cycle_abandoned_bool"] is True
    assert state_store_obj.get_latest_vplan_for_pod("bad_capsule") is None
    assert state_store_obj.get_latest_vplan_for_pod("healthy_parent").status_str == "ready"


def test_competing_vplan_builder_preserves_winning_ready_plan(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("DELETE FROM vplan_row WHERE vplan_id_int = ?", (vplan_obj.vplan_id_int,))
        connection_obj.execute("DELETE FROM vplan WHERE vplan_id_int = ?", (vplan_obj.vplan_id_int,))
        connection_obj.execute("UPDATE decision_plan SET status_str = 'planned' WHERE decision_plan_id_int = ?",
                               (vplan_obj.decision_plan_id_int,))
    competing_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    original_insert_func = state_store_obj.insert_vplan
    winning_vplan_list = []

    def insert_after_competing_worker(candidate_vplan_obj):
        winning_vplan_list.append(competing_store_obj.insert_vplan(candidate_vplan_obj))
        return original_insert_func(candidate_vplan_obj)

    monkeypatch.setattr(state_store_obj, "insert_vplan", insert_after_competing_worker)
    result_dict = build_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", **runner_kwarg_dict)
    assert result_dict["created_vplan_count_int"] == 0
    assert result_dict["blocked_action_count_int"] == 1
    assert len(winning_vplan_list) == 1
    winning_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    assert winning_plan_obj.status_str == "vplan_ready"
    assert not winning_plan_obj.snapshot_metadata_dict.get("mr_capsule_unsubmitted_cycle_abandoned_bool")
    assert state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "ready"
    assert state_store_obj.claim_vplan_for_submission(winning_vplan_list[0].vplan_id_int)


def test_capsule_no_order_cycle_completes_through_normal_reconciliation(capsule_case):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    no_order_release_obj = replace(release_obj, release_id_str="no_order.v1", pod_id_str="no_order", account_route_str="DU_NO_ORDER")
    state_store_obj.upsert_release(no_order_release_obj)
    prior_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
    state_store_obj.upsert_pod_state(replace(prior_state_obj, pod_id_str="no_order", account_route_str="DU_NO_ORDER"))
    original_plan_obj = state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    state_store_obj.insert_decision_plan(replace(
        original_plan_obj, decision_plan_id_int=None, status_str="planned", release_id_str="no_order.v1",
        pod_id_str="no_order", account_route_str="DU_NO_ORDER", entry_target_weight_map_dict={}, target_weight_map={},
        target_share_map_dict={}, entry_priority_list=[],
    ))
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_NO_ORDER", cash_float=1000.0, total_value_float=100500.0,
        position_amount_map=prior_state_obj.position_amount_map, snapshot_timestamp_ts=SUBMIT_TIMESTAMP_TS, session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(account_route_str="DU_NO_ORDER", asset_reference_price_map={}, snapshot_timestamp_ts=SUBMIT_TIMESTAMP_TS)
    build_result_dict = build_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", **runner_kwarg_dict)
    assert build_result_dict["created_vplan_count_int"] == 1
    no_order_vplan_obj = state_store_obj.get_latest_vplan_for_pod("no_order")
    assert no_order_vplan_obj.vplan_row_list == []
    submit_result_dict = submit_ready_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "paper", False,
                                           vplan_id_int=no_order_vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert submit_result_dict["submitted_vplan_count_int"] == 1
    assert broker_adapter_obj.submitted_order_request_list == []
    # Reconcile needs a fresh broker observation even when the frozen plan has no orders.
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU_NO_ORDER", cash_float=1000.0, total_value_float=100500.0,
        position_amount_map=prior_state_obj.position_amount_map, snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS, session_mode_str="paper")
    result_dict = post_execution_reconcile(state_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
    assert state_store_obj.get_latest_decision_plan_for_pod("no_order").status_str == "completed"
    assert state_store_obj.get_pod_state("no_order").position_amount_map == prior_state_obj.position_amount_map


@pytest.mark.parametrize("terminal_str", ["abandoned", "completed"])
def test_finished_capsule_cycle_is_never_rebuilt_for_the_same_signal_session(capsule_case, monkeypatch, terminal_str):
    """A rerun on the same evening must not decide on Close_T again: T's signal would be traded twice."""
    state_store_obj, _, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = capsule_case
    if terminal_str == "abandoned":
        assert state_store_obj.abandon_unsubmitted_mr_capsule_cycle(vplan_obj.decision_plan_id_int, "blocked")
    else:
        with state_store_obj._connect() as connection_obj:
            connection_obj.execute("UPDATE vplan SET status_str = 'completed' WHERE vplan_id_int = ?", (vplan_obj.vplan_id_int,))
            connection_obj.execute(
                "UPDATE decision_plan SET status_str = 'completed' WHERE decision_plan_id_int = ?", (vplan_obj.decision_plan_id_int,),
            )
    builder_call_list = []
    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: [release_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release",
                        lambda **_kwarg_dict: builder_call_list.append(True))
    # The fixture's decision used Close 2024-01-31; this rerun on that same evening has the same signal session.
    result_dict = build_decision_plans(
        state_store_obj, datetime(2024, 1, 31, 18, 30, tzinfo=MARKET_TIMEZONE_OBJ), str(tmp_path),
        auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict,
    )
    assert builder_call_list == []
    assert result_dict["created_decision_plan_count_int"] == 0 and result_dict["skipped_decision_plan_count_int"] == 1
    assert result_dict["reason_count_map_dict"].get("mr_capsule_signal_cycle_already_completed") == 1
