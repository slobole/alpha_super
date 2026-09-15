from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.live import core5_adapter as adapter_module, runner as runner_module, scheduler_utils
from alpha.live import scheduler_service as scheduler_module, dashboard as dashboard_module
from alpha.live.execution_engine import build_vplan, build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerSnapshot, LivePriceSnapshot, LiveRelease, PodState
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import validate_release_manifest
from alpha.live.state_store_v2 import LiveStateStore
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


def _time(date_str, hour_int=18):
    return (pd.Timestamp(date_str).tz_localize("America/New_York") + pd.Timedelta(hours=hour_int)).to_pydatetime()


@pytest.fixture
def release_obj():
    return LiveRelease(
        release_id_str="core5.test.v1", user_id_str="test", pod_id_str="core5_test",
        account_route_str="DU_CORE5", strategy_import_str=adapter_module.CORE5_STRATEGY_IMPORT_STR,
        mode_str="paper", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str="norgate_eod_core5", params_dict={},
        risk_profile_str="core5", enabled_bool=True, source_path_str="test.yaml",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
    )


@pytest.fixture
def price_df():
    date_idx = scheduler_utils.get_exchange_calendar_obj("XNYS").sessions_in_range("2024-01-02", "2026-09-15")
    bar_vec = np.arange(len(date_idx), dtype=float)
    price_dict = {}
    for asset_int, asset_str in enumerate(adapter_module.CORE5_ASSET_TUPLE):
        close_vec = 97.3 + 8 * asset_int + bar_vec * 0.012 + 8 * np.sin(bar_vec / (17 + asset_int))
        for field_str, multiplier_float in (("Open", 1.003), ("High", 1.006), ("Low", .995), ("Close", 1.0)):
            price_dict[(asset_str, field_str)] = close_vec * multiplier_float
        price_dict[(asset_str, "Dividend")] = np.zeros(len(date_idx))
        if asset_str != "BIL":
            price_dict[(f"ADAPTIVE_TR_{asset_str}", "Close")] = close_vec * 1.3
    result_df = pd.DataFrame(price_dict, index=date_idx)
    result_df.attrs["adjustment_by_symbol_dict"] = {asset_str: "CAPITALSPECIAL" for asset_str in adapter_module.CORE5_ASSET_TUPLE}
    result_df.attrs["signal_adjustment_by_symbol_dict"] = {f"ADAPTIVE_TR_{asset_str}": "TOTALRETURN" for asset_str in core5_module.RISK_ASSET_TUPLE}
    return result_df


def _state(release_obj, date_str, position_dict=None, strategy_state_dict=None, cash_float=100_000.0):
    return PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=position_dict or {}, cash_float=cash_float, total_value_float=999_999.0,
        strategy_state_dict=strategy_state_dict or {}, updated_timestamp_ts=_time(date_str, 17),
        snapshot_stage_str="eod", snapshot_source_str="broker",
    )


def _build(release_obj, price_df, date_str, state_obj):
    return adapter_module.build_core5_decision_from_prices(release_obj, _time(date_str), state_obj, price_df, {
        "norgate_data_source_mode_str": "snapshot", "norgate_data_profile_str": "norgate_eod_core5",
        "norgate_snapshot_date_str": date_str, "norgate_manifest_hash_str": "fixture_hash",
    })


def _controlled_signals(monkeypatch, price_df, change_dict=None):
    # These fixtures isolate the state-machine branches. Full feature parity is
    # tested separately against the actual research engine below.
    signal_df = price_df.copy()
    for asset_str in core5_module.RISK_ASSET_TUPLE:
        namespace_str = f"ADAPTIVE_TR_{asset_str}"
        signal_df[(namespace_str, "long_state_ser")] = 1.0
        signal_df[(namespace_str, "short_state_ser")] = 0.0
        signal_df[(namespace_str, "annualized_volatility_ser")] = .231
    for (date_str, asset_str, field_str), value_float in (change_dict or {}).items():
        signal_df.loc[pd.Timestamp(date_str), (f"ADAPTIVE_TR_{asset_str}", field_str)] = value_float
    monkeypatch.setattr(core5_module.AdaptiveMacroCore5Strategy, "compute_signals", lambda self, pricing_data_df: signal_df.loc[pricing_data_df.index].copy())
    return signal_df


def _vplan(release_obj, decision_obj, quote_float=150.0, nav_float=200_000.0):
    snapshot_obj = BrokerSnapshot(release_obj.account_route_str, decision_obj.submission_timestamp_ts,
        cash_float=100_000.0, total_value_float=nav_float, net_liq_float=nav_float,
        position_amount_map=decision_obj.decision_base_position_map)
    quote_obj = LivePriceSnapshot(release_obj.account_route_str, decision_obj.submission_timestamp_ts, "test_quote",
        {asset_str: quote_float for asset_str in adapter_module.CORE5_ASSET_TUPLE})
    return build_vplan(release_obj, decision_obj, snapshot_obj, quote_obj)


def test_initialization_uses_close_nav_and_ignores_next_open_quote(release_obj, price_df, monkeypatch):
    _controlled_signals(monkeypatch, price_df, {
        (date_str, "DBC", field_str): value_float
        for date_str in ("2026-09-10", "2026-09-11")
        for field_str, value_float in (("long_state_ser", 0.0), ("short_state_ser", 1.0))
    })
    decision_obj = _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11"))
    metadata_dict = decision_obj.snapshot_metadata_dict
    assert metadata_dict["initialization_bool"] and not metadata_dict["month_end_bool"]
    assert metadata_dict["sizing_close_nav_float"] == 100_000.0
    dbc_expected_int = int(-10_000 / price_df.loc["2026-09-11", ("DBC", "Close")])
    assert metadata_dict["fixed_target_share_map_dict"]["DBC"] == dbc_expected_int
    assert dbc_expected_int != np.floor(-10_000 / price_df.loc["2026-09-11", ("DBC", "Close")])
    first_vplan_obj = _vplan(release_obj, decision_obj)
    second_vplan_obj = _vplan(release_obj, decision_obj, quote_float=40, nav_float=1_000_000)
    assert first_vplan_obj.target_share_map == second_vplan_obj.target_share_map
    assert first_vplan_obj.pod_budget_float == 100_000.0
    assert "Cash" not in first_vplan_obj.target_share_map
    assert decision_obj.target_execution_timestamp_ts == _time("2026-09-14", 9) + timedelta(minutes=30)


def test_unchanged_day_holds_drifted_positions_and_volatility(release_obj, price_df, monkeypatch):
    signal_df = _controlled_signals(monkeypatch, price_df, {
        (date_str, "DBC", field_str): value_float
        for date_str in ("2026-09-10", "2026-09-11", "2026-09-14")
        for field_str, value_float in (("long_state_ser", 0.0), ("short_state_ser", 1.0))
    })
    first_obj = _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11"))
    position_dict = first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    changed_price_df = price_df.copy()
    changed_price_df.loc["2026-09-14", ("DBC", "Close")] *= 1.5
    signal_df.loc["2026-09-14", ("DBC", "Close")] = changed_price_df.loc["2026-09-14", ("DBC", "Close")]
    signal_df.loc["2026-09-14", ("ADAPTIVE_TR_DBC", "annualized_volatility_ser")] = 1.0
    second_obj = _build(release_obj, changed_price_df, "2026-09-14", _state(release_obj, "2026-09-14", position_dict, first_obj.strategy_state_dict, 25_000))
    assert not second_obj.snapshot_metadata_dict["rebalance_bool"]
    assert second_obj.snapshot_metadata_dict["no_order_bool"]
    assert second_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == position_dict
    assert second_obj.snapshot_metadata_dict["sizing_close_price_map_dict"]["DBC"] == changed_price_df.loc["2026-09-14", ("DBC", "Close")]
    assert second_obj.strategy_state_dict["last_target_weight_map_dict"] == first_obj.strategy_state_dict["last_target_weight_map_dict"]


def test_dbc_equality_does_not_create_a_new_short_trigger(release_obj, price_df, monkeypatch):
    _controlled_signals(monkeypatch, price_df, {
        ("2026-09-11", "DBC", "long_state_ser"): 0.0,
        ("2026-09-14", "DBC", "long_state_ser"): 0.0,
        ("2026-09-14", "DBC", "short_state_ser"): 1.0,
    })
    first_obj = _build(release_obj, price_df, "2026-09-10", _state(release_obj, "2026-09-10"))
    second_obj = _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11", first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], first_obj.strategy_state_dict))
    assert second_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]["DBC"] == 0
    third_obj = _build(release_obj, price_df, "2026-09-14", _state(release_obj, "2026-09-14", second_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], second_obj.strategy_state_dict))
    assert not third_obj.snapshot_metadata_dict["rebalance_bool"]
    assert third_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]["DBC"] == 0


@pytest.mark.parametrize("signal_date_str,previous_date_str,month_end_bool", [("2026-08-31", "2026-08-28", True), ("2026-09-11", "2026-09-10", False)])
def test_month_end_uses_exchange_calendar_not_terminal_price_row(release_obj, price_df, monkeypatch, signal_date_str, previous_date_str, month_end_bool):
    _controlled_signals(monkeypatch, price_df)
    first_obj = _build(release_obj, price_df, previous_date_str, _state(release_obj, previous_date_str))
    second_obj = _build(release_obj, price_df.loc[:signal_date_str], signal_date_str, _state(release_obj, signal_date_str, first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], first_obj.strategy_state_dict))
    assert second_obj.snapshot_metadata_dict["rebalance_bool"] == month_end_bool
    assert second_obj.snapshot_metadata_dict["month_end_bool"] == month_end_bool


@pytest.mark.parametrize("mutation_str", ["stale", "bootstrap", "wrong_account", "future", "missing_state", "fractional", "unknown_asset"])
def test_invalid_account_state_blocks_decision(release_obj, price_df, mutation_str):
    state_obj = _state(release_obj, "2026-09-11")
    if mutation_str == "stale": state_obj = replace(state_obj, updated_timestamp_ts=_time("2026-09-10", 17))
    elif mutation_str == "bootstrap": state_obj = replace(state_obj, snapshot_stage_str="unknown")
    elif mutation_str == "wrong_account": state_obj = replace(state_obj, account_route_str="DU_OTHER")
    elif mutation_str == "future": state_obj = replace(state_obj, updated_timestamp_ts=_time("2026-09-14", 17))
    elif mutation_str == "missing_state": state_obj = replace(state_obj, position_amount_map={"SPY": 2})
    elif mutation_str == "fractional": state_obj = replace(state_obj, position_amount_map={"DBC": -.5})
    else: state_obj = replace(state_obj, position_amount_map={"MSFT": 2})
    with pytest.raises(ValueError): _build(release_obj, price_df, "2026-09-11", state_obj)


def test_skipped_session_and_revised_previous_state_block(release_obj, price_df, monkeypatch):
    _controlled_signals(monkeypatch, price_df)
    first_obj = _build(release_obj, price_df, "2026-09-10", _state(release_obj, "2026-09-10"))
    with pytest.raises(ValueError, match="missed"):
        _build(release_obj, price_df, "2026-09-14", _state(release_obj, "2026-09-14", strategy_state_dict=first_obj.strategy_state_dict))
    first_obj.strategy_state_dict["last_long_state_map_dict"]["DBC"] = 0
    with pytest.raises(ValueError, match="historical revision"):
        _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11", strategy_state_dict=first_obj.strategy_state_dict))


@pytest.mark.parametrize("opening_short_bool", [True, False])
def test_sign_flip_preserves_two_legs_after_sql_restart(release_obj, price_df, monkeypatch, tmp_path, opening_short_bool):
    prior_long_float, next_long_float = (1.0, 0.0) if opening_short_bool else (0.0, 1.0)
    _controlled_signals(monkeypatch, price_df, {
        (date_str, "DBC", field_str): value_float
        for date_str, long_float in (("2026-09-09", prior_long_float), ("2026-09-10", prior_long_float), ("2026-09-11", next_long_float))
        for field_str, value_float in (("long_state_ser", long_float), ("short_state_ser", 1.0-long_float))
    })
    first_obj = _build(release_obj, price_df, "2026-09-10", _state(release_obj, "2026-09-10"))
    prior_position_dict = first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
    state_obj = _state(release_obj, "2026-09-11", prior_position_dict, first_obj.strategy_state_dict)
    decision_obj = _build(release_obj, price_df, "2026-09-11", state_obj)
    store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    store_obj.upsert_release(release_obj)
    decision_obj = store_obj.insert_decision_plan(decision_obj)
    vplan_obj = store_obj.insert_vplan(_vplan(release_obj, decision_obj))
    restarted_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    vplan_obj = restarted_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int)
    request_list = [request_obj for request_obj in build_broker_order_request_list_from_vplan(vplan_obj) if request_obj.asset_str == "DBC"]
    assert len(request_list) == 2
    assert request_list[0].amount_float == -prior_position_dict["DBC"]
    assert request_list[1].amount_float == vplan_obj.target_share_map["DBC"]
    assert request_list[0].order_request_key_str != request_list[1].order_request_key_str
    assert sum(request_obj.amount_float for request_obj in request_list) == vplan_obj.order_delta_map["DBC"]


def test_frozen_plan_rejects_changed_positions_or_tampered_quantities(release_obj, price_df):
    decision_obj = _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11"))
    with pytest.raises(ValueError, match="positions changed"):
        adapter_module.validated_core5_target_share_dict(decision_obj, release_obj, {"SPY": 1})
    decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]["DBC"] -= 1
    with pytest.raises(ValueError, match="disagree"):
        adapter_module.validated_core5_target_share_dict(decision_obj, release_obj, {})


@pytest.mark.parametrize("field_str,value_obj", [("mode_str", "live"), ("pod_budget_fraction_float", .5), ("execution_policy_str", "next_open_market"), ("data_profile_str", "norgate_eod_etf_plus_vix_helper"), ("signal_clock_str", "month_end_eod_snapshot_ready"), ("params_dict", {"commodity_short_cap_float": .5})])
def test_release_requires_frozen_core5_operational_contract(release_obj, field_str, value_obj):
    validate_release_manifest(release_obj)
    with pytest.raises(ValueError): validate_release_manifest(replace(release_obj, **{field_str: value_obj}))


def _store_and_broker(tmp_path, release_obj, state_obj):
    store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(state_obj)
    broker_obj = StubBrokerAdapter()
    broker_obj.seed_account_snapshot(release_obj.account_route_str, state_obj.cash_float, 100_000,
        state_obj.position_amount_map, snapshot_timestamp_ts=_time("2026-09-14", 9), session_mode_str="paper")
    broker_obj.seed_live_price_snapshot(release_obj.account_route_str,
        {asset_str: 200.0 for asset_str in adapter_module.CORE5_ASSET_TUPLE}, snapshot_timestamp_ts=_time("2026-09-14", 9))
    return store_obj, broker_obj


def test_runner_full_cycle_commits_state_once_after_reconcile(release_obj, price_df, tmp_path):
    state_obj = _state(release_obj, "2026-09-11")
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, state_obj)
    decision_obj = store_obj.insert_decision_plan(_build(release_obj, price_df, "2026-09-11", state_obj))
    detail_dict = runner_module.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["created_vplan_count_int"] == 1
    vplan_obj = store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    detail_dict = runner_module.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["submitted_vplan_count_int"] == 1
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == {}
    assert all(request_obj.broker_order_type_str == "MOO" and request_obj.unit_str == "shares" for request_obj in broker_obj.submitted_order_request_list)
    restarted_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    detail_dict = runner_module.post_execution_reconcile(restarted_store_obj, broker_obj,
        decision_obj.target_execution_timestamp_ts + timedelta(minutes=10), "paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == decision_obj.strategy_state_dict
    assert restarted_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"
    before_history_int = len(restarted_store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str))
    restarted_store_obj.complete_core5_cycle(decision_obj.decision_plan_id_int, vplan_obj.vplan_id_int)
    assert len(restarted_store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)) == before_history_int


@pytest.mark.parametrize("when_str", ["vplan", "submit"])
def test_runner_blocks_account_change_before_sizing_or_submit(release_obj, price_df, tmp_path, when_str):
    state_obj = _state(release_obj, "2026-09-11")
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, state_obj)
    decision_obj = store_obj.insert_decision_plan(_build(release_obj, price_df, "2026-09-11", state_obj))
    if when_str == "submit":
        vplan_obj = store_obj.insert_vplan(_vplan(release_obj, decision_obj))
    broker_obj.seed_account_snapshot(release_obj.account_route_str, 99_000, 100_000, {"SPY": 1}, snapshot_timestamp_ts=decision_obj.submission_timestamp_ts)
    if when_str == "vplan":
        detail_dict = runner_module.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
        assert detail_dict["created_vplan_count_int"] == 0
    else:
        detail_dict = runner_module.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", False,
            vplan_id_int=vplan_obj.vplan_id_int, log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
        assert detail_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "blocked"


def _no_order_cycle(release_obj, price_df, monkeypatch, tmp_path):
    _controlled_signals(monkeypatch, price_df)
    prior_obj = _build(release_obj, price_df, "2026-09-10", _state(release_obj, "2026-09-10"))
    state_obj = _state(release_obj, "2026-09-11", prior_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], prior_obj.strategy_state_dict)
    store_obj, _ = _store_and_broker(tmp_path, release_obj, state_obj)
    decision_obj = store_obj.insert_decision_plan(_build(release_obj, price_df, "2026-09-11", state_obj))
    assert decision_obj.snapshot_metadata_dict["no_order_bool"]
    return store_obj, decision_obj, state_obj


def test_no_order_commit_is_atomic_and_preserves_eod_timestamp(release_obj, price_df, monkeypatch, tmp_path):
    store_obj, decision_obj, state_obj = _no_order_cycle(release_obj, price_df, monkeypatch, tmp_path)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("CREATE TRIGGER test_commit_failure BEFORE UPDATE OF status_str ON decision_plan BEGIN SELECT RAISE(ABORT, 'injected crash'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected crash"):
        store_obj.complete_core5_cycle(decision_obj.decision_plan_id_int)
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == state_obj.strategy_state_dict
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "planned"
    with store_obj._connect() as connection_obj:
        connection_obj.execute("DROP TRIGGER test_commit_failure")
    store_obj.complete_core5_cycle(decision_obj.decision_plan_id_int)
    saved_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    assert saved_state_obj.strategy_state_dict == decision_obj.strategy_state_dict
    assert saved_state_obj.updated_timestamp_ts == state_obj.updated_timestamp_ts
    assert saved_state_obj.snapshot_stage_str == "eod"
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str) is None


def test_scheduler_recovers_no_order_after_open_without_broker(release_obj, price_df, monkeypatch, tmp_path):
    store_obj, decision_obj, _ = _no_order_cycle(release_obj, price_df, monkeypatch, tmp_path)
    monkeypatch.setattr(scheduler_module, "_load_release_list_and_sync", lambda *args, **kwargs: [release_obj])
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict", lambda *args, **kwargs: {"due_bool": True})
    schedule_obj = scheduler_module.get_scheduler_decision(store_obj, _time("2026-09-14", 12), "unused", "paper")
    assert schedule_obj.reason_code_str == "core5_complete_no_order_cycle"
    detail_dict = runner_module.expire_stale_decision_plans(store_obj, _time("2026-09-14", 12), "unused", "paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["expired_decision_plan_count_int"] == 0
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str) is None
    schedule_obj = scheduler_module.get_scheduler_decision(store_obj, _time("2026-09-14", 12), "unused", "paper")
    assert schedule_obj.next_phase_str != "build_decision_plan"


def test_manual_vplan_build_recovers_no_order_without_broker(release_obj, price_df, monkeypatch, tmp_path):
    store_obj, decision_obj, _ = _no_order_cycle(release_obj, price_df, monkeypatch, tmp_path)
    broker_obj = StubBrokerAdapter()
    def fail_broker_call(*args, **kwargs):
        pytest.fail("No-order completion must not request a broker snapshot")
    monkeypatch.setattr(broker_obj, "get_account_snapshot", fail_broker_call)
    detail_dict = runner_module.build_vplans(store_obj, broker_obj, _time("2026-09-14", 12), "paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["created_vplan_count_int"] == 0
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str) is None


@pytest.mark.parametrize("has_prior_eod_bool", [False, True])
def test_scheduler_captures_eod_before_core5_decision(release_obj, monkeypatch, tmp_path, has_prior_eod_bool):
    store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    store_obj.upsert_release(release_obj)
    if has_prior_eod_bool: store_obj.upsert_pod_state(_state(release_obj, "2026-09-10"))
    monkeypatch.setattr(scheduler_module, "_load_release_list_and_sync", lambda *args, **kwargs: [release_obj])
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict", lambda *args, **kwargs: {"due_bool": True})
    schedule_obj = scheduler_module.get_scheduler_decision(store_obj, _time("2026-09-11"), "unused", "paper")
    assert schedule_obj.next_phase_str == "eod_snapshot"
    store_obj.upsert_pod_state(_state(release_obj, "2026-09-11"))
    schedule_obj = scheduler_module.get_scheduler_decision(store_obj, _time("2026-09-11"), "unused", "paper")
    assert schedule_obj.next_phase_str == "build_decision_plan"


@pytest.mark.parametrize("mutation_str", ["preclose", "open_orders"])
def test_eod_rejects_stale_broker_source_and_open_orders(release_obj, monkeypatch, tmp_path, mutation_str):
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, _state(release_obj, "2026-09-10"))
    source_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(source_obj,
        snapshot_timestamp_ts=_time("2026-09-11", 15 if mutation_str == "preclose" else 17),
        open_order_id_list=["working_order"] if mutation_str == "open_orders" else [])
    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *args, **kwargs: [release_obj])
    detail_dict = runner_module.eod_snapshot(store_obj, broker_obj, _time("2026-09-11"), env_mode_str="paper", releases_root_path_str="unused", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["eod_snapshot_count_int"] == 0
    assert detail_dict["reason_count_map_dict"]["core5_eod_source_untrusted"] == 1
    assert store_obj.get_pod_state(release_obj.pod_id_str).updated_timestamp_ts == _time("2026-09-10", 17)


def test_malformed_saved_targets_fail_cleanly_and_first_ready_day_can_initialize(release_obj, price_df, monkeypatch):
    signal_df = _controlled_signals(monkeypatch, price_df)
    first_obj = _build(release_obj, price_df, "2026-09-10", _state(release_obj, "2026-09-10"))
    first_obj.strategy_state_dict.pop("last_rebalance_date_str")
    with pytest.raises(ValueError, match="rebalance date"):
        _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11", strategy_state_dict=first_obj.strategy_state_dict))
    signal_df.loc["2026-09-10", ("ADAPTIVE_TR_UUP", "long_state_ser")] = np.nan
    first_ready_obj = _build(release_obj, price_df, "2026-09-11", _state(release_obj, "2026-09-11"))
    assert first_ready_obj.snapshot_metadata_dict["initialization_bool"]


def test_multiday_engine_oracle_with_dividends_borrow_and_gapped_opens(price_df):
    from scripts.review.verify_core5_adapter_parity import compare_adapter_to_engine

    oracle_price_df = price_df.iloc[:260].copy()
    for asset_str in adapter_module.CORE5_ASSET_TUPLE:
        oracle_price_df.loc[oracle_price_df.index[::21], (asset_str, "Dividend")] = .23
    oracle_price_df[("$SPX", "Close")] = oracle_price_df[("SPY", "Close")] * 50
    report_dict = compare_adapter_to_engine(oracle_price_df, "2024-01-02")
    assert report_dict["decision_count_int"] > 100
    assert report_dict["rebalance_count_int"] > 10
    assert report_dict["no_order_count_int"] > 20
    assert report_dict["month_end_count_int"] >= 5
    assert report_dict["dbc_two_leg_count_int"] >= 2
    assert report_dict["oracle_borrow_fee_float"] > 0


@pytest.mark.parametrize("mutation_str", ["wrong_execution_date", "target_instead_of_delta"])
def test_oracle_detects_wrong_order_timing_or_target_semantics(price_df, monkeypatch, mutation_str):
    from scripts.review import verify_core5_adapter_parity as oracle_module
    oracle_price_df = price_df.iloc[:135].copy()
    oracle_price_df[("$SPX", "Close")] = oracle_price_df[("SPY", "Close")] * 50
    if mutation_str == "wrong_execution_date":
        original_builder_fn = oracle_module.build_core5_decision_from_prices
        def changed_builder(*args, **kwargs):
            decision_obj = original_builder_fn(*args, **kwargs)
            return replace(decision_obj, target_execution_timestamp_ts=decision_obj.target_execution_timestamp_ts + timedelta(days=1))
        monkeypatch.setattr(oracle_module, "build_core5_decision_from_prices", changed_builder)
    else:
        original_request_fn = oracle_module.build_broker_order_request_list_from_vplan
        monkeypatch.setattr(oracle_module, "build_broker_order_request_list_from_vplan",
            lambda *args, **kwargs: [replace(request_obj, target_bool=True) for request_obj in original_request_fn(*args, **kwargs)])
    with pytest.raises(AssertionError):
        oracle_module.compare_adapter_to_engine(oracle_price_df, "2024-01-02")


def _submitted_flip(release_obj, price_df, monkeypatch, tmp_path):
    _controlled_signals(monkeypatch, price_df, {
        ("2026-09-11", "DBC", "long_state_ser"): 0.0,
        ("2026-09-11", "DBC", "short_state_ser"): 1.0,
    })
    prior_obj = _build(release_obj, price_df, "2026-09-10", _state(release_obj, "2026-09-10"))
    state_obj = _state(release_obj, "2026-09-11", prior_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], prior_obj.strategy_state_dict)
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, state_obj)
    decision_obj = store_obj.insert_decision_plan(_build(release_obj, price_df, "2026-09-11", state_obj))
    vplan_obj = store_obj.insert_vplan(_vplan(release_obj, decision_obj))
    detail_dict = runner_module.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["submitted_vplan_count_int"] == 1
    record_map_dict = broker_obj._broker_order_record_map[release_obj.account_route_str]
    dbc_record_list = [next(record_obj for record_obj in record_map_dict.values() if record_obj.order_request_key_str == request_obj.order_request_key_str)
        for request_obj in build_broker_order_request_list_from_vplan(vplan_obj) if request_obj.asset_str == "DBC"]
    assert len(dbc_record_list) == 2
    return store_obj, broker_obj, decision_obj, vplan_obj, state_obj, dbc_record_list


@pytest.mark.parametrize("mutation_str", ["missing_fill", "wrong_sign", "partial_fill", "open_orders", "rejected_first_leg"])
def test_incomplete_flip_does_not_commit_even_if_positions_match(release_obj, price_df, monkeypatch, tmp_path, mutation_str):
    store_obj, broker_obj, decision_obj, vplan_obj, state_obj, dbc_record_list = _submitted_flip(release_obj, price_df, monkeypatch, tmp_path)
    route_str = release_obj.account_route_str
    first_record_obj = dbc_record_list[0]
    if mutation_str in {"missing_fill", "rejected_first_leg"}:
        broker_obj._fill_map[route_str] = [fill_obj for fill_obj in broker_obj._fill_map[route_str] if fill_obj.broker_order_id_str != first_record_obj.broker_order_id_str]
    elif mutation_str in {"wrong_sign", "partial_fill"}:
        broker_obj._fill_map[route_str] = [replace(fill_obj, fill_amount_float=fill_obj.fill_amount_float * (-1 if mutation_str == "wrong_sign" else .5))
            if fill_obj.broker_order_id_str == first_record_obj.broker_order_id_str else fill_obj for fill_obj in broker_obj._fill_map[route_str]]
    elif mutation_str == "open_orders":
        broker_obj._snapshot_map[route_str] = replace(broker_obj._snapshot_map[route_str], open_order_id_list=["working"])
    if mutation_str == "rejected_first_leg":
        broker_obj._broker_order_record_map[route_str][first_record_obj.broker_order_id_str] = replace(first_record_obj,
            status_str="Rejected", filled_amount_float=0, remaining_amount_float=abs(first_record_obj.amount_float))
        broker_obj._broker_order_event_map[route_str] = [event_obj for event_obj in broker_obj._broker_order_event_map[route_str] if event_obj.broker_order_id_str != first_record_obj.broker_order_id_str]
        position_dict = dict(broker_obj._snapshot_map[route_str].position_amount_map)
        position_dict["DBC"] = state_obj.position_amount_map["DBC"] + dbc_record_list[1].amount_float
        broker_obj._snapshot_map[route_str] = replace(broker_obj._snapshot_map[route_str], position_amount_map=position_dict)
    store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    detail_dict = runner_module.post_execution_reconcile(store_obj, broker_obj, decision_obj.target_execution_timestamp_ts + timedelta(minutes=10),
        "paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["completed_vplan_count_int"] == 0
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == state_obj.strategy_state_dict
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str != "completed"
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str != "completed"
    if mutation_str == "rejected_first_leg":
        assert runner_module.is_vplan_execution_exception_parked(store_obj, store_obj.get_vplan_by_id(vplan_obj.vplan_id_int))


def test_sparse_flip_refresh_preserves_identity_and_rejects_ambiguity(release_obj, price_df, monkeypatch, tmp_path):
    store_obj, _, _, vplan_obj, _, dbc_record_list = _submitted_flip(release_obj, price_df, monkeypatch, tmp_path)
    refreshed_obj = replace(dbc_record_list[0], order_request_key_str=None, submission_key_str=None, raw_payload_dict={})
    store_obj.upsert_vplan_broker_order_record_list([refreshed_obj])
    row_list = store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    assert {row_dict["order_request_key_str"] for row_dict in row_list if row_dict["asset_str"] == "DBC"} == {record_obj.order_request_key_str for record_obj in dbc_record_list}
    with pytest.raises(ValueError, match="Ambiguous"):
        store_obj.upsert_vplan_broker_order_record_list([replace(refreshed_obj, broker_order_id_str="unknown")])
    with pytest.raises(ValueError, match="conflicts"):
        store_obj.upsert_vplan_broker_order_record_list([replace(refreshed_obj, order_request_key_str=dbc_record_list[1].order_request_key_str)])
    assert store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int) == row_list


@pytest.mark.parametrize("current_float,target_float", [(10, -5), (-10, 5)])
def test_dashboard_reports_total_flip_delta(current_float, target_float):
    report_dict = dashboard_module._build_execution_report_from_vplan_dict({
        "current_broker_position_json_str": json.dumps({"DBC": current_float}), "target_share_json_str": json.dumps({"DBC": target_float}),
        "order_delta_json_str": json.dumps({"DBC": target_float-current_float}), "live_reference_price_json_str": json.dumps({"DBC": 100}),
        "vplan_row_dict_list": [
            {"asset_str": "DBC", "current_share_float": current_float, "target_share_float": 0, "order_delta_share_float": -current_float},
            {"asset_str": "DBC", "current_share_float": 0, "target_share_float": target_float, "order_delta_share_float": target_float}],
    }, broker_position_map_dict={"DBC": target_float})
    row_dict = report_dict["execution_row_dict_list"][0]
    assert row_dict["current_share_float"] == current_float
    assert row_dict["target_share_float"] == target_float
    assert row_dict["planned_order_delta_share_float"] == target_float-current_float


@pytest.mark.parametrize("future_bool", [False, True])
def test_eod_uses_post_read_clock_and_preserves_source_time(release_obj, monkeypatch, tmp_path, future_bool):
    as_of_ts = _time("2026-09-11")
    class CapturedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return (as_of_ts + timedelta(seconds=2)).astimezone(tz)
    monkeypatch.setattr(runner_module, "datetime", CapturedClock)
    store_obj, broker_obj = _store_and_broker(tmp_path, release_obj, _state(release_obj, "2026-09-10"))
    source_timestamp_ts = as_of_ts + timedelta(seconds=3 if future_bool else 1)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(broker_obj._snapshot_map[release_obj.account_route_str], snapshot_timestamp_ts=source_timestamp_ts)
    detail_dict = runner_module.eod_snapshot(store_obj, broker_obj, as_of_ts, env_mode_str="paper", log_path_str=str(tmp_path / "ops.log"), trace_enabled_bool=False)
    assert detail_dict["eod_snapshot_count_int"] == int(not future_bool)
    if not future_bool:
        assert store_obj.get_pod_state(release_obj.pod_id_str).updated_timestamp_ts == source_timestamp_ts


@pytest.mark.parametrize("mutation_str", ["valid", "direct", "stale", "newer", "changed_manifest"])
def test_host_requires_exact_snapshot_session_and_stable_manifest(release_obj, price_df, monkeypatch, mutation_str):
    from alpha.live import strategy_host
    snapshot_module = adapter_module.snapshot_module
    monkeypatch.setattr(snapshot_module, "is_snapshot_mode_enabled_bool", lambda: mutation_str != "direct")
    def load_manifest(profile_str, minimum_snapshot_date_str):
        assert profile_str == "norgate_eod_core5" and minimum_snapshot_date_str == "2026-09-11"
        return SimpleNamespace(manifest_hash_str="fixture_hash")
    monkeypatch.setattr(snapshot_module, "load_valid_snapshot_manifest", load_manifest)
    monkeypatch.setattr(core5_module, "get_adaptive_macro_core5_data", lambda config_obj: price_df)
    metadata_dict = {"norgate_data_profile_str": "norgate_eod_core5", "norgate_manifest_hash_str": "different" if mutation_str == "changed_manifest" else "fixture_hash",
        "norgate_snapshot_date_str": {"stale": "2026-09-10", "newer": "2026-09-14"}.get(mutation_str, "2026-09-11")}
    monkeypatch.setattr(snapshot_module, "build_data_source_metadata_dict", lambda *args: metadata_dict)
    if mutation_str == "valid":
        decision_obj = strategy_host.build_decision_plan_for_release(release_obj, _time("2026-09-11"), _state(release_obj, "2026-09-11"))
        assert decision_obj.snapshot_metadata_dict["initialization_bool"]
    else:
        with pytest.raises(ValueError):
            strategy_host.build_decision_plan_for_release(release_obj, _time("2026-09-11"), _state(release_obj, "2026-09-11"))
