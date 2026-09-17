from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live import runner
from alpha.live.execution_engine import build_vplan
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.incubation import IncubationBrokerAdapter
from alpha.live.models import BrokerSnapshot, DecisionPlan, LivePriceSnapshot, LiveRelease, PodState, SessionOpenPrice, VPlanRow
from alpha.live.state_store_v2 import LiveStateStore


@pytest.fixture(autouse=True)
def no_real_broker(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("No-order regression tests must never connect to a broker")
    monkeypatch.setattr(IBKRSocketClient, "connect", reject_connection)


def _no_order_case(tmp_path, case_str, execution_policy_str="next_open_moo"):
    position_dict = {} if case_str == "cash" else {"SPY": 100.0}
    book_type_str = "full_target_weight_book" if case_str == "unchanged_target" else "incremental_entry_exit_book"
    execution_ts = datetime(2026, 9, 14, 20 if execution_policy_str == "same_day_moc" else 13,
                            0 if execution_policy_str == "same_day_moc" else 30, tzinfo=UTC)
    submission_ts = execution_ts - timedelta(minutes=7)
    release_obj = LiveRelease(
        release_id_str="no_orders.v1", user_id_str="test", pod_id_str="no_orders",
        account_route_str="SIM_NO_ORDERS", strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="incubation", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str=execution_policy_str, data_profile_str="norgate_eod_sp500_pit",
        params_dict={"capital_base_float": 100_000.0}, risk_profile_str="test", enabled_bool=True,
        source_path_str="test.yaml", pod_budget_fraction_float=1.0,
    )
    store_obj = LiveStateStore(str(tmp_path / "no_orders.sqlite3"))
    store_obj.upsert_release(release_obj)
    initial_state_obj = PodState(
        release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
        position_dict, 90_000.0, 90_000.0 + 100.0 * sum(position_dict.values()),
        {"cycle": "before"}, submission_ts,
    )
    store_obj.upsert_pod_state(initial_state_obj)
    decision_obj = store_obj.insert_decision_plan(DecisionPlan(
        release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str,
        release_obj.account_route_str, datetime(2026, 9, 11, 20, tzinfo=UTC), submission_ts,
        execution_ts, execution_policy_str, position_dict, {}, {"cycle": "after"},
        decision_book_type_str=book_type_str,
        full_target_weight_map_dict={"SPY": 0.1} if case_str == "unchanged_target" else {},
    ))
    snapshot_obj = BrokerSnapshot(
        account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=submission_ts,
        position_amount_map=position_dict, cash_float=90_000.0,
        total_value_float=initial_state_obj.total_value_float, net_liq_float=initial_state_obj.total_value_float,
    )
    quote_obj = LivePriceSnapshot(release_obj.account_route_str, submission_ts, "fixture", {"SPY": 100.0})
    vplan_obj = build_vplan(release_obj, decision_obj, snapshot_obj, quote_obj)
    if case_str == "round_trip":
        vplan_obj = replace(vplan_obj, target_share_map={"SPY": 100.0}, order_delta_map={"SPY": 0.0},
                            vplan_row_list=[
                                VPlanRow("SPY", 100.0, 90.0, -10.0, 100.0, 9_000.0, "MOO"),
                                VPlanRow("SPY", 90.0, 100.0, 10.0, 100.0, 10_000.0, "MOO"),
                            ])
    vplan_obj = store_obj.insert_vplan(vplan_obj)
    price_call_list = []

    def make_broker(current_store_obj, as_of_ts):
        def open_lookup(route_str, asset_list, open_ts, calendar_str):
            price_call_list.append(("Open", tuple(asset_list)))
            return [SessionOpenPrice("2026-09-14", route_str, asset_str, 100.0, "ibkr.tick_open",
                                     execution_ts + timedelta(minutes=1)) for asset_str in asset_list]

        def official_lookup(asset_list, date_str, field_str):
            price_call_list.append((field_str, tuple(asset_list)))
            return {asset_str: 100.0 for asset_str in asset_list}

        return IncubationBrokerAdapter(
            state_store_obj=current_store_obj, as_of_ts=as_of_ts,
            official_price_lookup_func=official_lookup, ibkr_tick_open_lookup_func=open_lookup,
        )

    result_dict = runner.submit_ready_vplans(
        store_obj, make_broker(store_obj, submission_ts), submission_ts, "incubation", False,
        vplan_id_int=vplan_obj.vplan_id_int, log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False,
    )
    assert result_dict["submitted_vplan_count_int"] == 1
    return SimpleNamespace(store_obj=store_obj, release_obj=release_obj, decision_obj=decision_obj,
                           vplan_obj=vplan_obj, initial_state_obj=initial_state_obj, make_broker=make_broker,
                           after_execution_ts=execution_ts + timedelta(minutes=10), price_call_list=price_call_list,
                           log_path_str=str(tmp_path / "ops.log"))


@pytest.mark.parametrize("case_str", ["cash", "hold", "unchanged_target"])
@pytest.mark.parametrize("execution_policy_str", ["next_open_moo", "same_day_moc"])
def test_no_orders_complete_after_restart_without_financial_writes(tmp_path, case_str, execution_policy_str):
    case_obj = _no_order_case(tmp_path, case_str, execution_policy_str)
    assert all(row_obj.order_delta_share_float == 0.0 for row_obj in case_obj.vplan_obj.vplan_row_list)
    # Restart before settlement, then again before reconciliation.
    for attempt_int in range(2):
        store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
        broker_obj = case_obj.make_broker(store_obj, case_obj.after_execution_ts)
        snapshot_obj = broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
        assert snapshot_obj.cash_float == case_obj.initial_state_obj.cash_float
        assert snapshot_obj.position_amount_map == case_obj.initial_state_obj.position_amount_map
        assert store_obj.get_pod_state(case_obj.release_obj.pod_id_str) == case_obj.initial_state_obj
        assert not store_obj.has_committed_incubation_settlement(case_obj.vplan_obj.vplan_id_int)
    result_dict = runner.post_execution_reconcile(
        store_obj, broker_obj, case_obj.after_execution_ts, "incubation",
        log_path_str=case_obj.log_path_str, trace_enabled_bool=False,
    )
    assert result_dict["completed_vplan_count_int"] == 1
    assert store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "completed"
    assert store_obj.get_latest_decision_plan_for_pod(case_obj.release_obj.pod_id_str).status_str == "completed"
    state_obj = store_obj.get_pod_state(case_obj.release_obj.pod_id_str)
    assert state_obj.cash_float == case_obj.initial_state_obj.cash_float
    assert state_obj.position_amount_map == case_obj.initial_state_obj.position_amount_map
    assert state_obj.strategy_state_dict == case_obj.decision_obj.strategy_state_dict
    with closing(sqlite3.connect(store_obj.db_path_str)) as connection_obj:
        for table_str in ("vplan_fill", "cash_ledger_entry", "vplan_broker_order", "vplan_broker_ack"):
            assert connection_obj.execute(f"SELECT count(*) FROM {table_str}").fetchone()[0] == 0
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)
    assert runner.post_execution_reconcile(
        restarted_store_obj, case_obj.make_broker(restarted_store_obj, case_obj.after_execution_ts),
        case_obj.after_execution_ts, "incubation", log_path_str=case_obj.log_path_str, trace_enabled_bool=False,
    )["completed_vplan_count_int"] == 0
    assert restarted_store_obj.get_pod_state(case_obj.release_obj.pod_id_str) == state_obj
    eod_ts = datetime(2026, 9, 14, 20, 20, tzinfo=UTC)
    assert runner.eod_snapshot(
        restarted_store_obj, case_obj.make_broker(restarted_store_obj, eod_ts), eod_ts, "incubation",
        log_path_str=case_obj.log_path_str, trace_enabled_bool=False,
    )["eod_snapshot_count_int"] == 1


@pytest.mark.parametrize("case_str", ["cash", "unchanged_target"])
@pytest.mark.parametrize("quantity_float", [10.0, float("nan"), float("inf"), float("-inf")])
def test_empty_rows_with_outstanding_or_invalid_order_intent_stay_blocked(tmp_path, case_str, quantity_float):
    case_obj = _no_order_case(tmp_path, case_str)
    with closing(sqlite3.connect(case_obj.store_obj.db_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE vplan SET order_delta_json_str = ?", (json.dumps({"SPY": quantity_float}),))
    broker_obj = case_obj.make_broker(case_obj.store_obj, case_obj.after_execution_ts)
    with pytest.raises((ValueError, RuntimeError)):
        broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitted"
    assert case_obj.store_obj.get_pod_state(case_obj.release_obj.pod_id_str) == case_obj.initial_state_obj


def test_nonzero_opposite_legs_with_zero_net_delta_still_settle(tmp_path):
    case_obj = _no_order_case(tmp_path, "round_trip")
    broker_obj = case_obj.make_broker(case_obj.store_obj, case_obj.after_execution_ts)
    snapshot_obj = broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    assert snapshot_obj.position_amount_map == {"SPY": 100.0}
    assert snapshot_obj.cash_float == case_obj.initial_state_obj.cash_float - 2.0
    fill_list = case_obj.store_obj.get_fill_row_dict_list_for_vplan(case_obj.vplan_obj.vplan_id_int)
    assert sorted(fill_dict["fill_amount_float"] for fill_dict in fill_list) == [-10.0, 10.0]
    assert case_obj.store_obj.has_committed_incubation_settlement(case_obj.vplan_obj.vplan_id_int)


def test_real_orders_with_missing_prices_do_not_complete(tmp_path):
    case_obj = _no_order_case(tmp_path, "round_trip")
    broker_obj = case_obj.make_broker(case_obj.store_obj, case_obj.after_execution_ts)
    broker_obj.ibkr_tick_open_lookup_func = lambda *args: []
    with pytest.raises(RuntimeError, match="Missing IBKR tick-open price"):
        runner.post_execution_reconcile(
            case_obj.store_obj, broker_obj, case_obj.after_execution_ts, "incubation",
            log_path_str=case_obj.log_path_str, trace_enabled_bool=False,
        )
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitted"
    assert not case_obj.store_obj.get_fill_row_dict_list_for_vplan(case_obj.vplan_obj.vplan_id_int)
    assert case_obj.store_obj.get_pod_state(case_obj.release_obj.pod_id_str) == case_obj.initial_state_obj


def test_empty_plan_still_requires_matching_holdings(tmp_path):
    case_obj = _no_order_case(tmp_path, "unchanged_target")
    case_obj.store_obj.upsert_pod_state(replace(case_obj.initial_state_obj, position_amount_map={"SPY": 90.0}))
    result_dict = runner.post_execution_reconcile(
        case_obj.store_obj, case_obj.make_broker(case_obj.store_obj, case_obj.after_execution_ts),
        case_obj.after_execution_ts, "incubation", log_path_str=case_obj.log_path_str, trace_enabled_bool=False,
    )
    assert result_dict["completed_vplan_count_int"] == 0
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitted"
