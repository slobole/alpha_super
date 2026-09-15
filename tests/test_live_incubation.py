from __future__ import annotations

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Barrier
from types import SimpleNamespace

import pytest

from alpha.live import runner
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.incubation import IncubationBrokerAdapter
from alpha.live.models import LivePriceSnapshot, SessionOpenPrice
from alpha.live.state_store_v2 import LiveStateStore
from alpha.live.release_manifest import validate_release_manifest
from data.norgate_snapshot_store import get_active_data_profile_str
from test_live_core5_adapter import (
    _build, _controlled_signals, _state, _time, price_df, release_obj,
)


@pytest.fixture(autouse=True)
def no_real_broker(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("Incubation tests must never connect to a real broker")
    monkeypatch.setattr(IBKRSocketClient, "connect", reject_connection)


@pytest.fixture
def incubation_case_factory(release_obj, price_df, monkeypatch, tmp_path):
    def create_case(direction_str="initialization"):
        sim_release_obj = replace(
            release_obj, mode_str="incubation", account_route_str="SIM_CORE5",
            params_dict={"capital_base_float": 100_000.0},
        )
        store_obj = LiveStateStore(str(tmp_path / "incubation.sqlite3"))
        store_obj.upsert_release(sim_release_obj)
        change_dict = {}
        if direction_str != "initialization":
            prior_long_float = float(direction_str == "long_to_short")
            for date_str, long_float in (("2026-09-09", prior_long_float),
                                         ("2026-09-10", prior_long_float),
                                         ("2026-09-11", 1.0 - prior_long_float)):
                change_dict[(date_str, "DBC", "long_state_ser")] = long_float
                change_dict[(date_str, "DBC", "short_state_ser")] = 1.0 - long_float
        _controlled_signals(monkeypatch, price_df, change_dict)
        if direction_str != "initialization":
            prior_obj = _build(sim_release_obj, price_df, "2026-09-10",
                               replace(_state(sim_release_obj, "2026-09-10"), snapshot_source_str="virtual_broker"))
            store_obj.upsert_pod_state(replace(
                _state(sim_release_obj, "2026-09-11",
                       prior_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], prior_obj.strategy_state_dict),
                snapshot_source_str="virtual_broker",
            ))
        price_call_list = []

        def make_broker(current_store_obj, as_of_ts):
            def official_lookup(asset_list, date_str, field_str):
                assert get_active_data_profile_str() == "norgate_eod_core5"
                price_call_list.append(("official", date_str, field_str))
                return {asset_str: float(price_df.loc[date_str, (asset_str, field_str)]) for asset_str in asset_list}

            def live_lookup(route_str, asset_list, policy_str):
                return LivePriceSnapshot(route_str, as_of_ts, "injected_quote",
                                         {asset_str: 200.0 for asset_str in asset_list})

            def open_lookup(route_str, asset_list, open_ts, calendar_str):
                date_str = open_ts.astimezone(_time("2026-09-11").tzinfo).date().isoformat()
                price_call_list.append(("open", date_str, "Open"))
                return [SessionOpenPrice(date_str, route_str, asset_str,
                        float(price_df.loc[date_str, (asset_str, "Open")]), "ibkr.tick_open", as_of_ts)
                        for asset_str in asset_list]

            return IncubationBrokerAdapter(
                state_store_obj=current_store_obj, as_of_ts=as_of_ts,
                official_price_lookup_func=official_lookup, ibkr_live_price_lookup_func=live_lookup,
                ibkr_tick_open_lookup_func=open_lookup,
            )

        eod_ts = _time("2026-09-11")
        runner.eod_snapshot(store_obj, make_broker(store_obj, eod_ts), eod_ts, "incubation",
                            log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
        state_obj = store_obj.get_pod_state(sim_release_obj.pod_id_str)
        assert state_obj.snapshot_source_str == "virtual_broker"
        decision_obj = store_obj.insert_decision_plan(_build(sim_release_obj, price_df, "2026-09-11", state_obj))
        preopen_broker_obj = make_broker(store_obj, decision_obj.submission_timestamp_ts)
        build_dict = runner.build_vplans(store_obj, preopen_broker_obj, decision_obj.submission_timestamp_ts,
                    "incubation", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
        assert build_dict["created_vplan_count_int"] == 1
        vplan_obj = store_obj.get_latest_vplan_for_pod(sim_release_obj.pod_id_str)
        submit_dict = runner.submit_ready_vplans(
            store_obj, preopen_broker_obj, decision_obj.submission_timestamp_ts, "incubation", False,
            vplan_id_int=vplan_obj.vplan_id_int, log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False,
        )
        assert submit_dict["submitted_vplan_count_int"] == 1
        assert not any(call_tuple[0] == "open" for call_tuple in price_call_list)
        assert not store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
        return SimpleNamespace(
            store_obj=store_obj, release_obj=sim_release_obj, decision_obj=decision_obj,
            vplan_obj=vplan_obj, initial_state_obj=state_obj, make_broker=make_broker,
            after_open_ts=decision_obj.target_execution_timestamp_ts + timedelta(minutes=10),
            log_path_str=str(tmp_path / "ops.log"),
        )
    return create_case


def _table_snapshot_dict(store_obj):
    table_list = ["session_open_price", "vplan_broker_order", "vplan_broker_order_event",
                  "vplan_fill", "cash_ledger_entry", "pod_state", "pod_state_history"]
    with closing(sqlite3.connect(store_obj.db_path_str)) as connection_obj:
        return {table_str: connection_obj.execute("SELECT * FROM " + table_str + " ORDER BY rowid").fetchall()
                for table_str in table_list}


def _reconcile(case_obj, store_obj, broker_obj):
    return runner.post_execution_reconcile(
        store_obj, broker_obj, case_obj.after_open_ts, "incubation",
        log_path_str=case_obj.log_path_str, trace_enabled_bool=False,
    )


@pytest.mark.parametrize("direction_str", ["initialization", "long_to_short", "short_to_long"])
def test_actual_incubation_core5_lifecycle_and_restart(incubation_case_factory, price_df, direction_str):
    case_obj = incubation_case_factory(direction_str)
    store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    broker_obj = case_obj.make_broker(store_obj, case_obj.after_open_ts)
    snapshot_obj = broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    expected_position_dict = {asset_str: value_float for asset_str, value_float in
                             case_obj.decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"].items()
                             if value_float}
    assert snapshot_obj.position_amount_map == expected_position_dict
    expected_cash_float = case_obj.initial_state_obj.cash_float
    for row_obj in case_obj.vplan_obj.vplan_row_list:
        quantity_float = row_obj.order_delta_share_float
        if quantity_float:
            expected_cash_float -= quantity_float * price_df.loc["2026-09-14", (row_obj.asset_str, "Open")]
            expected_cash_float -= max(1.0, .005 * abs(quantity_float))
    assert snapshot_obj.cash_float == pytest.approx(expected_cash_float)
    assert store_obj.get_pod_state(case_obj.release_obj.pod_id_str).strategy_state_dict == case_obj.initial_state_obj.strategy_state_dict
    fill_list = store_obj.get_fill_row_dict_list_for_vplan(case_obj.vplan_obj.vplan_id_int, include_order_identity_bool=True)
    if direction_str != "initialization":
        dbc_list = [row_dict for row_dict in fill_list if row_dict["asset_str"] == "DBC"]
        assert len(dbc_list) == 2 and len({row_dict["broker_order_id_str"] for row_dict in dbc_list}) == 2
    # Restart after settlement, before reconciliation: persisted evidence must suffice.
    again_obj = LiveStateStore(store_obj.db_path_str)
    again_broker_obj = case_obj.make_broker(again_obj, case_obj.after_open_ts)
    assert _reconcile(case_obj, again_obj, again_broker_obj)["completed_vplan_count_int"] == 1
    assert again_obj.get_pod_state(case_obj.release_obj.pod_id_str).strategy_state_dict == case_obj.decision_obj.strategy_state_dict
    assert again_obj.has_committed_incubation_settlement(case_obj.vplan_obj.vplan_id_int)
    before_dict = _table_snapshot_dict(again_obj)
    assert _reconcile(case_obj, again_obj, again_broker_obj)["completed_vplan_count_int"] == 0
    again_broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    assert _table_snapshot_dict(again_obj) == before_dict


@pytest.mark.parametrize("method_str", [
    "upsert_session_open_price_list", "upsert_vplan_broker_order_record_list",
    "insert_vplan_broker_order_event_list", "upsert_vplan_fill_list",
    "insert_cash_ledger_entry_list", "upsert_pod_state",
])
def test_settlement_rolls_back_every_write_and_recovers(incubation_case_factory, monkeypatch, method_str):
    case_obj = incubation_case_factory()
    store_obj = case_obj.store_obj
    before_dict = _table_snapshot_dict(store_obj)
    original_method_func = getattr(store_obj, method_str)
    def interrupted_write(*args, **kwargs):
        original_method_func(*args, **kwargs)
        raise RuntimeError("injected interrupted write")
    with monkeypatch.context() as patch_obj:
        patch_obj.setattr(store_obj, method_str, interrupted_write)
        broker_obj = case_obj.make_broker(store_obj, case_obj.after_open_ts)
        with pytest.raises(RuntimeError, match="injected interrupted write"):
            broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
        assert not broker_obj._settled_state_by_vplan_id_dict
    assert _table_snapshot_dict(store_obj) == before_dict
    again_obj = LiveStateStore(store_obj.db_path_str)
    again_broker_obj = case_obj.make_broker(again_obj, case_obj.after_open_ts)
    assert _reconcile(case_obj, again_obj, again_broker_obj)["completed_vplan_count_int"] == 1


def test_concurrent_same_plan_settles_once(incubation_case_factory, monkeypatch):
    case_obj = incubation_case_factory()
    first_store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    second_store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    barrier_obj = Barrier(2)
    for store_obj in (first_store_obj, second_store_obj):
        original_func = store_obj.persist_incubation_settlement
        def concurrent_commit(*, original_func=original_func, **kwargs):
            barrier_obj.wait(timeout=10)
            return original_func(**kwargs)
        monkeypatch.setattr(store_obj, "persist_incubation_settlement", concurrent_commit)
    broker_list = [case_obj.make_broker(store_obj, case_obj.after_open_ts)
                   for store_obj in (first_store_obj, second_store_obj)]
    with ThreadPoolExecutor(max_workers=2) as executor_obj:
        snapshot_list = list(executor_obj.map(
            lambda broker_obj: broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str), broker_list))
    assert snapshot_list[0].cash_float == snapshot_list[1].cash_float
    assert snapshot_list[0].position_amount_map == snapshot_list[1].position_amount_map
    history_list = first_store_obj.get_pod_state_history_row_dict_list(case_obj.release_obj.pod_id_str)
    assert sum(row_dict["snapshot_stage_str"] == "post_execution" for row_dict in history_list) == 1
    fill_list = first_store_obj.get_fill_row_dict_list_for_vplan(case_obj.vplan_obj.vplan_id_int)
    assert len(first_store_obj.get_cash_ledger_row_dict_list_for_vplan(case_obj.vplan_obj.vplan_id_int)) == 2 * len(fill_list)


@pytest.mark.parametrize("field_str", ["cash_float", "position_amount_map", "strategy_state_dict"])
def test_settlement_rejects_changed_financial_baseline(incubation_case_factory, monkeypatch, field_str):
    case_obj = incubation_case_factory()
    store_obj = case_obj.store_obj
    mutation_dict = {"cash_float": 100_001.0, "position_amount_map": {"SPY": 1.0},
                     "strategy_state_dict": {"newer": True}}
    changed_state_obj = replace(case_obj.initial_state_obj, **{field_str: mutation_dict[field_str]})
    original_func = store_obj.persist_incubation_settlement
    def racing_update(**kwargs):
        store_obj.upsert_pod_state(changed_state_obj)
        return original_func(**kwargs)
    monkeypatch.setattr(store_obj, "persist_incubation_settlement", racing_update)
    with pytest.raises(RuntimeError, match="changed before settlement"):
        case_obj.make_broker(store_obj, case_obj.after_open_ts).get_account_snapshot(case_obj.release_obj.account_route_str)
    assert not store_obj.get_fill_row_dict_list_for_vplan(case_obj.vplan_obj.vplan_id_int)
    actual_state_obj = store_obj.get_pod_state(case_obj.release_obj.pod_id_str)
    assert getattr(actual_state_obj, field_str) == mutation_dict[field_str]


def test_legacy_fills_do_not_silently_prove_complete_settlement(incubation_case_factory):
    case_obj = incubation_case_factory()
    broker_obj = case_obj.make_broker(case_obj.store_obj, case_obj.after_open_ts)
    broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    with closing(sqlite3.connect(case_obj.store_obj.db_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE vplan_fill SET raw_payload_json_str = '{}'")
    with pytest.raises(RuntimeError, match="unverified legacy settlement"):
        case_obj.make_broker(case_obj.store_obj, case_obj.after_open_ts).get_account_snapshot(case_obj.release_obj.account_route_str)


def test_hard_process_exit_rolls_back_settlement(incubation_case_factory):
    case_obj = incubation_case_factory()
    before_dict = _table_snapshot_dict(case_obj.store_obj)
    script_str = '''
import os, sys
from datetime import timedelta
from alpha.live.state_store_v2 import LiveStateStore
from alpha.live.incubation import IncubationBrokerAdapter
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import SessionOpenPrice
def reject_connection(*args, **kwargs):
    raise AssertionError("Unexpected real broker access")
IBKRSocketClient.connect = reject_connection
store_obj = LiveStateStore(sys.argv[1])
vplan_obj = store_obj.get_submitted_vplan_list()[0]
as_of_ts = vplan_obj.target_execution_timestamp_ts + timedelta(minutes=10)
original_func = store_obj.upsert_vplan_fill_list
def interrupted_write(*args, **kwargs):
    original_func(*args, **kwargs)
    os._exit(17)
store_obj.upsert_vplan_fill_list = interrupted_write
broker_obj = IncubationBrokerAdapter(state_store_obj=store_obj, as_of_ts=as_of_ts,
    official_price_lookup_func=lambda asset_list, date_str, field_str: {asset_str:200.0 for asset_str in asset_list},
    ibkr_tick_open_lookup_func=lambda route_str, asset_list, open_ts, calendar_str: [
        SessionOpenPrice("2026-09-14", route_str, asset_str, 200.0, "ibkr.tick_open", as_of_ts) for asset_str in asset_list])
broker_obj.get_account_snapshot(vplan_obj.account_route_str)
'''
    result_obj = subprocess.run([sys.executable, "-B", "-c", script_str, case_obj.store_obj.db_path_str],
                                capture_output=True, text=True, timeout=30)
    assert result_obj.returncode == 17, result_obj.stderr
    assert _table_snapshot_dict(case_obj.store_obj) == before_dict
    broker_obj = case_obj.make_broker(case_obj.store_obj, case_obj.after_open_ts)
    assert _reconcile(case_obj, case_obj.store_obj, broker_obj)["completed_vplan_count_int"] == 1


def test_shared_moc_settlement_keeps_official_close_prices(incubation_case_factory, price_df):
    # This tests settlement of an already-prepared generic MOC plan, not CORE5 MOC signals.
    case_obj = incubation_case_factory()
    moc_release_obj = replace(case_obj.release_obj,
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        data_profile_str="norgate_eod_sp500_pit", execution_policy_str="same_day_moc")
    validate_release_manifest(moc_release_obj)
    store_obj = case_obj.store_obj
    store_obj.upsert_release(moc_release_obj)
    close_ts = _time("2026-09-14", 16)
    with closing(sqlite3.connect(store_obj.db_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE vplan SET execution_policy_str = ?, target_execution_timestamp_str = ?",
                               ("same_day_moc", close_ts.isoformat()))
        connection_obj.execute("UPDATE vplan_row SET broker_order_type_str = 'MOC'")
    def official_lookup(asset_list, date_str, field_str):
        assert date_str == "2026-09-14" and field_str == "Close"
        assert get_active_data_profile_str() == "norgate_eod_sp500_pit"
        return {asset_str: float(price_df.loc[date_str, (asset_str, field_str)]) for asset_str in asset_list}
    broker_obj = IncubationBrokerAdapter(state_store_obj=store_obj, as_of_ts=close_ts+timedelta(minutes=10),
                                         official_price_lookup_func=official_lookup)
    snapshot_obj = broker_obj.get_account_snapshot(moc_release_obj.account_route_str)
    expected_cash_float = case_obj.initial_state_obj.cash_float
    for row_obj in case_obj.vplan_obj.vplan_row_list:
        quantity_float = row_obj.order_delta_share_float
        if quantity_float:
            expected_cash_float -= quantity_float * price_df.loc["2026-09-14", (row_obj.asset_str, "Close")]
            expected_cash_float -= max(1.0, .005 * abs(quantity_float))
    assert snapshot_obj.cash_float == pytest.approx(expected_cash_float)
    assert store_obj.has_committed_incubation_settlement(case_obj.vplan_obj.vplan_id_int)


def _price_record(**kwargs):
    return replace(SessionOpenPrice("2026-09-14", "SIM_CORE5", "DBC", 25.0,
                                   "ibkr.tick_open", _time("2026-09-14", 10)), **kwargs)


def _price_only_broker(tmp_path, record_list):
    return IncubationBrokerAdapter(
        state_store_obj=LiveStateStore(str(tmp_path / "prices.sqlite3")),
        as_of_ts=_time("2026-09-14", 9) + timedelta(minutes=31),
        ibkr_tick_open_lookup_func=lambda *args: record_list,
    )


def _read_open(broker_obj, asset_list=None):
    return broker_obj.get_session_open_price_list(
        "SIM_CORE5", asset_list or ["DBC"], _time("2026-09-14", 9)+timedelta(minutes=30), "XNYS")


@pytest.mark.parametrize("mutation_dict", [
    {"account_route_str": "SIM_OTHER"}, {"session_date_str": "2026-09-15"},
    {"asset_str": "SPY"}, {"open_price_source_str": "wrong.source"},
    {"official_open_price_float": -1.0}, {"official_open_price_float": float("nan")},
    {"snapshot_timestamp_ts": _time("2026-09-14", 9)},
    {"snapshot_timestamp_ts": _time("2026-09-15", 10)},
    {"snapshot_timestamp_ts": datetime(2026, 9, 14, 14)},
    {"snapshot_timestamp_ts": datetime.now(UTC)+timedelta(days=2)},
])
def test_open_record_rejects_invalid_identity_source_and_capture(tmp_path, mutation_dict):
    broker_obj = _price_only_broker(tmp_path, [_price_record(**mutation_dict)])
    with pytest.raises(RuntimeError, match="Incubation opening-price"):
        _read_open(broker_obj)


def test_duplicate_open_record_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="duplicated"):
        _read_open(_price_only_broker(tmp_path, [_price_record(), _price_record()]))


def test_fresh_price_can_arrive_after_adapter_asof(tmp_path):
    assert _read_open(_price_only_broker(tmp_path, [_price_record()]))[0].official_open_price_float == 25.0


def test_capture_cannot_be_future_within_the_correct_session(tmp_path, monkeypatch):
    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return (_time("2026-09-14", 10) + timedelta(minutes=5)).astimezone(tz or UTC)
    monkeypatch.setattr("alpha.live.incubation.datetime", ClockDateTime)
    record_obj = _price_record(snapshot_timestamp_ts=_time("2026-09-14", 10)+timedelta(minutes=6))
    with pytest.raises(RuntimeError, match="capture"):
        _read_open(_price_only_broker(tmp_path, [record_obj]))
    assert _read_open(_price_only_broker(tmp_path, [_price_record()]))[0].official_open_price_float == 25.0


def test_missing_open_price_leaves_all_settlement_tables_unchanged(incubation_case_factory):
    case_obj = incubation_case_factory()
    before_dict = _table_snapshot_dict(case_obj.store_obj)
    broker_obj = case_obj.make_broker(case_obj.store_obj, case_obj.after_open_ts)
    broker_obj.ibkr_tick_open_lookup_func = lambda *args: []
    with pytest.raises(RuntimeError, match="Missing IBKR tick-open"):
        broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    assert _table_snapshot_dict(case_obj.store_obj) == before_dict


def test_valid_cached_old_open_and_missing_placeholder(tmp_path):
    broker_obj = _price_only_broker(tmp_path, [_price_record()])
    broker_obj.state_store_obj.upsert_session_open_price_list([_price_record(official_open_price_float=None)])
    assert _read_open(broker_obj)[0].official_open_price_float == 25.0
    broker_obj.state_store_obj.upsert_session_open_price_list([_price_record()])
    broker_obj.as_of_ts = _time("2026-09-15", 10)
    broker_obj.ibkr_tick_open_lookup_func = None
    assert _read_open(broker_obj)[0].official_open_price_float == 25.0
    before_dict = _table_snapshot_dict(broker_obj.state_store_obj)
    with pytest.raises(RuntimeError, match="Uncached IBKR tick-open"):
        _read_open(broker_obj, ["DBC", "SPY"])
    assert _table_snapshot_dict(broker_obj.state_store_obj) == before_dict


def test_capture_date_uses_exchange_timezone(tmp_path):
    # 00:30 UTC Tuesday is still Monday after the XNYS open.
    record_obj = _price_record(snapshot_timestamp_ts=datetime(2026, 9, 15, 0, 30, tzinfo=UTC))
    assert _read_open(_price_only_broker(tmp_path, [record_obj]))[0] == record_obj
