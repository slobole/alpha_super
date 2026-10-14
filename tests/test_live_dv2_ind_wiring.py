"""Offline DV2-IND producer/consumer, causal decisions and daily lifecycle checks."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.live import runner, scheduler_utils, strategy_host, logging_utils, mr_capsule_notifications
from alpha.live.dv2_ind_adapter import DV2_IND_STRATEGY_IMPORT_STR, DV2_IND_SIZING_CONTRACT_STR
from alpha.live.dispatch_state import is_guarded_release_bool
from alpha.live.daily_reconcile import DAILY_STRATEGY_IMPORT_TUPLE, is_daily_reconcile_release_bool
from alpha.live.execution_engine import build_vplan, build_broker_order_request_list_from_vplan
from alpha.live.models import PodState, BrokerSnapshot, LivePriceSnapshot
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import parse_release_manifest, validate_release_manifest, validate_enabled_deployment_for_mode
from alpha.live.state_store_v2 import LiveStateStore
from data import norgate_loader, norgate_snapshot_store as snapshot_module
from data.dv2_ind_snapshot_contract import DV2_IND_PROFILE_STR, DV2_IND_ETF_TUPLE
from scripts import export_norgate_snapshot as export_module
from strategies.dv2 import strategy_mr_dv2_industry_etf as strategy_module
from daily_broker_fakes import install_daily_broker_stub

TEMPLATE_PATH = Path("docs/live/release_templates/pod_dv2_industry_etf_daily_moo.yaml.example")


class IndustryProvider:
    StockPriceAdjustmentType = SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN")
    PaddingType = SimpleNamespace(ALLMARKETDAYS="ALLMARKETDAYS", NONE="NONE")

    def __init__(self):
        calendar_obj = scheduler_utils.get_exchange_calendar_obj("XNYS")
        self.date_idx = calendar_obj.sessions_in_range("2024-01-02", "2026-10-13")[-420:]
        self.date_idx = pd.DatetimeIndex(list(self.date_idx), name="Date")
        self.frame_dict = {}
        self.stale_symbol_str = None
        for symbol_index_int, symbol_str in enumerate((*DV2_IND_ETF_TUPLE, "$SPXTR")):
            bar_vec = np.arange(len(self.date_idx), dtype=float)
            close_vec = (100 + symbol_index_int * 3 + bar_vec * .15).astype(np.float32)
            close_vec[-1] += 1 if symbol_str == "XBI" else -4
            price_df = pd.DataFrame({
                "Open": close_vec, "High": (close_vec + 1.0).astype(np.float32),
                "Low": (close_vec - 0.25).astype(np.float32), "Close": close_vec,
                "Volume": np.full(len(bar_vec), 1_000_000, dtype=np.int64),
                "Turnover": np.full(len(bar_vec), 100_000_000.0),
            }, index=self.date_idx)
            if symbol_str != "XBI":
                price_df.loc[self.date_idx[-1], "High"] = np.float32(close_vec[-1]+8)
            if symbol_str != "$SPXTR":
                price_df["Unadjusted Close"] = close_vec
                price_df["Dividend"] = np.zeros(len(bar_vec), dtype=np.float32)
            if symbol_str == "XPH":
                price_df["Turnover"] = 40_000_000.0
            if symbol_str == "IBB":
                price_df = price_df.iloc[-200:]
            self.frame_dict[symbol_str] = price_df

    def price_timeseries(self, symbol_str, **option_dict):
        if option_dict["padding_setting"] == "NONE" and symbol_str == self.stale_symbol_str:
            return self.frame_dict[symbol_str].iloc[:0].copy()
        return self.frame_dict[symbol_str].loc[option_dict["start_date"]:option_dict["end_date"]].copy()


@pytest.fixture
def industry_case(tmp_path, monkeypatch):
    provider_obj = IndustryProvider()
    monkeypatch.setattr(export_module, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: provider_obj)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    decision_date_str = provider_obj.date_idx[-1].date().isoformat()
    direct_price_df = strategy_module.get_prices(list(DV2_IND_ETF_TUPLE), ["$SPX"],
        start_date="2009-01-01", end_date=decision_date_str)
    snapshot_path_obj = export_module.export_profile_snapshot(snapshot_root_str=str(tmp_path),
        profile_str=DV2_IND_PROFILE_STR, snapshot_date_str=decision_date_str)
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setattr(norgate_loader, "_load_direct_norgate_module", lambda: pytest.fail("snapshot consumer called direct Norgate"))
    release_obj = replace(parse_release_manifest(str(TEMPLATE_PATH)), account_route_str="DU_TEST", enabled_bool=True)
    signal_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(provider_obj.date_idx[-1], "XNYS")
    position_dict = {"XBI": 10.0, "XPH": 5.0}
    state_obj = PodState(release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
        cash_float=95000.0, total_value_float=100000.0, position_amount_map=position_dict,
        strategy_state_dict={"trade_id_int": 7, "current_trade_map": {"XBI": 6, "XPH": 7},
            "dv2_ind_strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR},
        updated_timestamp_ts=signal_timestamp_ts+timedelta(minutes=10), snapshot_stage_str="eod", snapshot_source_str="broker")
    return SimpleNamespace(provider_obj=provider_obj, direct_price_df=direct_price_df,
        snapshot_path_obj=snapshot_path_obj, release_obj=release_obj, state_obj=state_obj,
        as_of_ts=signal_timestamp_ts+timedelta(minutes=20), tmp_path_obj=tmp_path)


def _decision(case_obj, state_obj=None):
    return strategy_host.build_decision_plan_for_release(case_obj.release_obj, case_obj.as_of_ts,
        case_obj.state_obj if state_obj is None else state_obj)


def test_snapshot_exact_native_prices_features_and_research_decision(industry_case):
    case_obj = industry_case
    assert DV2_IND_ETF_TUPLE == strategy_module.INDUSTRY_ETF_SYMBOL_TUPLE
    with snapshot_module.use_norgate_data_profile(DV2_IND_PROFILE_STR):
        snapshot_price_df = strategy_module.get_prices(list(DV2_IND_ETF_TUPLE), ["$SPX"],
            start_date="2009-01-01", end_date=case_obj.provider_obj.date_idx[-1].date().isoformat())
    pd.testing.assert_frame_equal(case_obj.direct_price_df, snapshot_price_df, check_exact=True)
    research_strategy_obj = strategy_module._new_strategy_obj(100000, strategy_module.build_history_universe_df(case_obj.direct_price_df))
    snapshot_strategy_obj = strategy_module._new_strategy_obj(100000, strategy_module.build_history_universe_df(snapshot_price_df))
    pd.testing.assert_frame_equal(research_strategy_obj.compute_signals(case_obj.direct_price_df),
        snapshot_strategy_obj.compute_signals(snapshot_price_df), check_exact=True)
    decision_obj = _decision(case_obj)
    strategy_host._seed_strategy_state(research_strategy_obj, case_obj.state_obj)
    research_strategy_obj._total_value_history_list = [decision_obj.snapshot_metadata_dict["decision_nav_float"]]
    research_strategy_obj.previous_bar = case_obj.provider_obj.date_idx[-1]
    research_strategy_obj.current_bar = pd.Timestamp(decision_obj.target_execution_timestamp_ts.date())
    signal_df = research_strategy_obj.compute_signals(case_obj.direct_price_df)
    research_strategy_obj.iterate(signal_df, signal_df.iloc[-1], pd.Series(dtype=float))
    expected_plan_obj = strategy_host._build_decision_plan_from_orders(case_obj.release_obj,
        research_strategy_obj.previous_bar.to_pydatetime(), research_strategy_obj)
    assert decision_obj.exit_asset_set == expected_plan_obj.exit_asset_set == {"XBI"}
    assert decision_obj.entry_priority_list == expected_plan_obj.entry_priority_list
    assert decision_obj.entry_target_weight_map_dict == expected_plan_obj.entry_target_weight_map_dict
    assert decision_obj.entry_priority_list and "IBB" not in decision_obj.entry_priority_list
    assert "XPH" not in decision_obj.entry_priority_list
    assert all(abs(weight_float-.1)<1e-12 for weight_float in decision_obj.entry_target_weight_map_dict.values())
    assert decision_obj.preserve_untouched_positions_bool
    assert decision_obj.snapshot_metadata_dict["exact_share_parity_bool"] is False


def test_future_row_cannot_change_held_exit_or_entry_intent(industry_case, monkeypatch):
    case_obj = industry_case
    baseline_obj = _decision(case_obj)
    future_price_df = case_obj.direct_price_df.copy()
    future_price_df.loc[case_obj.provider_obj.date_idx[-1]+pd.Timedelta(days=1)] = 1e9
    monkeypatch.setattr(strategy_module, "get_prices", lambda *argument_tuple, **argument_dict: future_price_df)
    altered_obj = _decision(case_obj)
    assert altered_obj.exit_asset_set == baseline_obj.exit_asset_set
    assert altered_obj.entry_priority_list == baseline_obj.entry_priority_list
    assert altered_obj.strategy_state_dict == baseline_obj.strategy_state_dict


@pytest.mark.parametrize("change_dict", [
    {"snapshot_stage_str": "bootstrap"}, {"snapshot_source_str": "unknown"},
    {"account_route_str": "DU_OTHER"}, {"cash_float": float("nan")},
    {"position_amount_map": {"UNKNOWN": 1}}, {"position_amount_map": {"XBI": -1}},
    {"position_amount_map": {"XBI": .5}}, {"strategy_state_dict": {}},
])
def test_untrusted_account_state_blocks(industry_case, change_dict):
    with pytest.raises(ValueError, match="DV2-IND"):
        _decision(industry_case, replace(industry_case.state_obj, **change_dict))


def _mutate_prices(case_obj, mutate_fn):
    price_path_obj = case_obj.snapshot_path_obj / "prices.parquet"
    altered_price_df = mutate_fn(pd.read_parquet(price_path_obj))
    altered_price_df.to_parquet(price_path_obj, index=False)
    manifest_path_obj = case_obj.snapshot_path_obj / "manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text())
    manifest_dict["files"]["prices.parquet"]["sha256"] = hashlib.sha256(price_path_obj.read_bytes()).hexdigest()
    manifest_path_obj.write_text(json.dumps(manifest_dict), encoding="utf-8")


@pytest.mark.parametrize("mutation_str", ["prefix", "gap", "turnover", "raw", "adjustment", "dtype"])
def test_corrupt_snapshot_fails_closed_even_with_updated_hash(industry_case, mutation_str):
    case_obj = industry_case
    def mutate_fn(price_df):
        symbol_mask_ser = price_df["symbol_str"].eq("SMH")
        row_index_int = price_df.index[symbol_mask_ser][0 if mutation_str=="prefix" else 50]
        if mutation_str in {"prefix", "gap"}:
            return price_df.drop(index=row_index_int)
        if mutation_str in {"turnover", "raw"}:
            return price_df.drop(columns="Turnover" if mutation_str=="turnover" else "Unadjusted Close")
        if mutation_str == "adjustment":
            price_df.loc[symbol_mask_ser, "adjustment_str"] = "TOTALRETURN"
        if mutation_str == "dtype":
            price_df["Close"] = price_df["Close"].astype(float)
            price_df.loc[row_index_int, "Close"] = float(price_df.loc[row_index_int,"Close"])+1e-8
        return price_df
    _mutate_prices(case_obj, mutate_fn)
    with pytest.raises((ValueError, snapshot_module.NorgateSnapshotValidationError), match="DV2-IND|fields"):
        snapshot_module.load_valid_snapshot_manifest(DV2_IND_PROFILE_STR)


def test_cached_snapshot_corruption_is_rechecked(industry_case):
    snapshot_module.load_valid_snapshot_manifest(DV2_IND_PROFILE_STR)
    price_path_obj = industry_case.snapshot_path_obj / "prices.parquet"
    price_path_obj.write_bytes(price_path_obj.read_bytes()+b"corrupt")
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="SHA256"):
        snapshot_module.load_valid_snapshot_manifest(DV2_IND_PROFILE_STR)


def test_disabled_release_and_frozen_parameters():
    release_obj = parse_release_manifest(str(TEMPLATE_PATH))
    assert not release_obj.enabled_bool and not release_obj.auto_submit_enabled_bool
    assert is_guarded_release_bool(release_obj) and is_daily_reconcile_release_bool(release_obj)
    from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
    assert DAILY_STRATEGY_IMPORT_TUPLE[-1] == CORE5_STRATEGY_IMPORT_STR
    with pytest.raises(ValueError, match="real IBKR account"):
        validate_enabled_deployment_for_mode([replace(release_obj, enabled_bool=True)], "paper")
    for change_dict in ({"execution_policy_str": "next_open_market"}, {"pod_budget_fraction_float": .5},
            {"params_dict": {"max_positions_int": 11}}, {"data_profile_str": "norgate_eod_sp500_pit"}):
        with pytest.raises(ValueError, match="DV2-IND"):
            validate_release_manifest(replace(release_obj, **change_dict))


@pytest.fixture
def daily_case(industry_case, monkeypatch):
    case_obj = industry_case
    decision_obj = _decision(case_obj)
    store_obj = LiveStateStore(str(case_obj.tmp_path_obj / "dv2.sqlite3"))
    store_obj.upsert_release(case_obj.release_obj)
    store_obj.upsert_pod_state(case_obj.state_obj)
    decision_obj = store_obj.insert_decision_plan(decision_obj)
    broker_obj = StubBrokerAdapter()
    install_daily_broker_stub(broker_obj)
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, case_obj.state_obj.cash_float,
        200000.0, case_obj.state_obj.position_amount_map, decision_obj.submission_timestamp_ts, session_mode_str="paper")
    broker_obj.seed_live_price_snapshot(case_obj.release_obj.account_route_str,
        {asset_str: 125.0 for asset_str in DV2_IND_ETF_TUPLE}, decision_obj.submission_timestamp_ts)
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(case_obj.tmp_path_obj / "critical.jsonl"))
    option_dict = {"log_path_str": str(case_obj.tmp_path_obj / "events.jsonl"), "trace_enabled_bool": False}
    assert runner.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", **option_dict)["created_vplan_count_int"] == 1
    vplan_obj = store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str)
    expected_shares_float = float(int(decision_obj.snapshot_metadata_dict["decision_nav_float"]*.1/125))
    assert all(vplan_obj.target_share_map[asset_str] == expected_shares_float for asset_str in decision_obj.entry_priority_list)
    case_obj.store_obj, case_obj.broker_obj, case_obj.decision_obj, case_obj.vplan_obj, case_obj.option_dict = store_obj, broker_obj, decision_obj, vplan_obj, option_dict
    return case_obj


def test_daily_submit_restart_actual_close_and_no_duplicate(daily_case):
    case_obj = daily_case
    request_list = build_broker_order_request_list_from_vplan(case_obj.vplan_obj)
    assert request_list and all(request_obj.broker_order_type_str=="MOO" for request_obj in request_list)
    result_dict = runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj,
        case_obj.decision_obj.submission_timestamp_ts, "paper", False, vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)
    assert result_dict["submitted_vplan_count_int"] == 1
    restarted_store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    assert runner.submit_ready_vplans(restarted_store_obj, case_obj.broker_obj,
        case_obj.decision_obj.submission_timestamp_ts, "paper", False, vplan_id_int=case_obj.vplan_obj.vplan_id_int,
        **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    execution_timestamp_ts = case_obj.vplan_obj.target_execution_timestamp_ts
    assert runner.post_execution_reconcile(restarted_store_obj, case_obj.broker_obj,
        execution_timestamp_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 0
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(execution_timestamp_ts.date(), "XNYS") + timedelta(minutes=1)
    snapshot_obj = case_obj.broker_obj.get_account_snapshot(case_obj.release_obj.account_route_str)
    case_obj.broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, snapshot_obj.cash_float,
        snapshot_obj.net_liq_float, snapshot_obj.position_amount_map, close_timestamp_ts, session_mode_str="paper")
    assert runner.post_execution_reconcile(restarted_store_obj, case_obj.broker_obj,
        close_timestamp_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_pod_state(case_obj.release_obj.pod_id_str).position_amount_map == snapshot_obj.position_amount_map
    assert runner.post_execution_reconcile(restarted_store_obj, case_obj.broker_obj,
        close_timestamp_ts, "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 0


def test_changed_holdings_block_saved_vplan_and_full_target_book(industry_case):
    case_obj = industry_case
    decision_obj = _decision(case_obj)
    broker_snapshot_obj = BrokerSnapshot(case_obj.release_obj.account_route_str, case_obj.as_of_ts, 95000, 100000,
        net_liq_float=100000, position_amount_map={"XBI": 11, "XPH": 5})
    price_snapshot_obj = LivePriceSnapshot(case_obj.release_obj.account_route_str, case_obj.as_of_ts,
        "synthetic", {asset_str: 125 for asset_str in DV2_IND_ETF_TUPLE})
    with pytest.raises(ValueError, match="holdings/orders changed"):
        build_vplan(case_obj.release_obj, decision_obj, broker_snapshot_obj, price_snapshot_obj)
    with pytest.raises(ValueError, match="incremental"):
        build_vplan(case_obj.release_obj, replace(decision_obj, decision_book_type_str="full_target_weight_book"),
            replace(broker_snapshot_obj, position_amount_map=case_obj.state_obj.position_amount_map), price_snapshot_obj)


def test_notification_scope_includes_dv2(industry_case):
    case_obj = industry_case
    scope_list = mr_capsule_notifications._configured_scope_list({"pod_row_dict_list": [{
        "strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR, "pod_id_str": case_obj.release_obj.pod_id_str,
        "account_route_str": case_obj.release_obj.account_route_str, "mode_str": "paper",
        "db_path_str": str(case_obj.tmp_path_obj/"dv2.sqlite3"), "source_path_str": "test.yaml",
    }]}, "paper")
    assert len(scope_list) == 1


@pytest.mark.parametrize("drift_str", ["holdings", "orders", "account", "disabled"])
def test_saved_vplan_pre_submit_recheck_blocks_before_claim(daily_case, monkeypatch, drift_str):
    case_obj = daily_case
    if drift_str == "disabled":
        case_obj.store_obj.upsert_release(replace(case_obj.release_obj, enabled_bool=False))
    else:
        original_fn = case_obj.broker_obj.get_capsule_account_snapshot
        def changed_snapshot_fn(account_route_str):
            snapshot_obj = original_fn(account_route_str)
            if drift_str == "holdings":
                return replace(snapshot_obj, position_amount_map={**snapshot_obj.position_amount_map, "XBI": 11.0})
            if drift_str == "orders":
                return replace(snapshot_obj, open_order_id_list=["foreign-order"])
            return replace(snapshot_obj, account_route_str="DU_OTHER")
        monkeypatch.setattr(case_obj.broker_obj, "get_capsule_account_snapshot", changed_snapshot_fn)
    assert runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj,
        case_obj.decision_obj.submission_timestamp_ts, "paper", False,
        vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert not case_obj.broker_obj.submitted_order_request_list
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str != "submitting"


def test_uncertain_second_send_restart_never_resends_and_alerts_once(daily_case, monkeypatch):
    from test_live_ibkr_socket_client import _ExpirySubmitIB, _expiry_client_obj
    from alpha.live.daily_broker import DailyExecutionSnapshot
    case_obj = daily_case
    clock_list = [case_obj.decision_obj.submission_timestamp_ts]
    class FrozenClock:
        @classmethod
        def now(cls, timezone_obj):
            return clock_list[0].astimezone(timezone_obj)
    monkeypatch.setattr("alpha.live.dispatch_state.datetime", FrozenClock)
    socket_broker_obj = _ExpirySubmitIB(clock_list)
    socket_obj = _expiry_client_obj(monkeypatch, socket_broker_obj)
    original_place_fn = socket_broker_obj.placeOrder
    def fail_second_fn(contract_obj, order_obj):
        if socket_broker_obj.placed_order_list:
            raise TimeoutError("second send has uncertain broker outcome")
        return original_place_fn(contract_obj, order_obj)
    monkeypatch.setattr(socket_broker_obj, "placeOrder", fail_second_fn)
    monkeypatch.setattr(case_obj.broker_obj, "submit_order_request_list", socket_obj.submit_order_request_list)
    assert runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj, clock_list[0], "paper", False,
        vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert len(socket_broker_obj.placed_order_list) == 1
    saved_plan_obj = case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int)
    assert len(saved_plan_obj.snapshot_metadata_dict["opening_dispatch_result_dict"]["attempted_request_key_list"]) == 2
    restarted_store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    assert runner.submit_ready_vplans(restarted_store_obj, case_obj.broker_obj, clock_list[0], "paper", False,
        vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)["submitted_vplan_count_int"] == 0
    assert len(socket_broker_obj.placed_order_list) == 1
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(case_obj.vplan_obj.target_execution_timestamp_ts.date(), "XNYS")+timedelta(minutes=1)
    # Actual holdings at close prove the exit succeeded; rejected buys remain missing.
    snapshot_obj = BrokerSnapshot(case_obj.release_obj.account_route_str, close_timestamp_ts,
        96000.0, 100000.0, net_liq_float=100000.0, position_amount_map={"XPH": 5.0})
    monkeypatch.setattr(case_obj.broker_obj, "get_daily_execution_snapshot", lambda account_route_str:
        DailyExecutionSnapshot(snapshot_obj, [], close_timestamp_ts, close_timestamp_ts))
    assert runner.post_execution_reconcile(restarted_store_obj, case_obj.broker_obj, close_timestamp_ts,
        "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_decision_plan_by_id(saved_plan_obj.decision_plan_id_int).status_str == "completed_with_exceptions"
    with restarted_store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 1
    assert runner.post_execution_reconcile(restarted_store_obj, case_obj.broker_obj, close_timestamp_ts,
        "paper", **case_obj.option_dict)["completed_vplan_count_int"] == 0
    summary_dict = {"pod_row_dict_list": [{"db_path_str": restarted_store_obj.db_path_str,
        "pod_id_str": case_obj.release_obj.pod_id_str, "account_route_str": case_obj.release_obj.account_route_str,
        "mode_str": "paper", "strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR}]}
    sent_payload_list = []
    def poster_fn(url_str, payload_dict):
        sent_payload_list.append(payload_dict)
        return True
    delivered_list = mr_capsule_notifications.deliver_execution_alerts(summary_dict, webhook_url_str="offline-test",
        webhook_poster_fn=poster_fn, now_ts=close_timestamp_ts)
    assert len(delivered_list) == len(sent_payload_list) == 1 and delivered_list[0].delivered_bool
    assert mr_capsule_notifications.deliver_execution_alerts(summary_dict, webhook_url_str="offline-test",
        webhook_poster_fn=lambda *_arg_tuple: pytest.fail("duplicate alert"), now_ts=close_timestamp_ts) == []


def test_eod_capture_builds_dv2_decision_from_blank_account(industry_case, monkeypatch):
    case_obj = industry_case
    store_obj = LiveStateStore(str(case_obj.tmp_path_obj/"from-eod.sqlite3"))
    store_obj.upsert_release(case_obj.release_obj)
    broker_obj = StubBrokerAdapter()
    install_daily_broker_stub(broker_obj)
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, 100000, 100000, {},
        case_obj.as_of_ts, session_mode_str="paper")
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(case_obj.tmp_path_obj/"critical.jsonl"))
    option_dict = {"log_path_str": str(case_obj.tmp_path_obj/"eod.log"), "trace_enabled_bool": False}
    assert runner.eod_snapshot(store_obj, broker_obj, case_obj.as_of_ts, "paper", **option_dict)["eod_snapshot_count_int"] == 1
    state_obj = store_obj.get_pod_state(case_obj.release_obj.pod_id_str)
    assert state_obj.snapshot_stage_str == "eod" and state_obj.snapshot_source_str == "broker"
    releases_root_path_obj = case_obj.tmp_path_obj / "your_user"
    releases_root_path_obj.mkdir()
    (releases_root_path_obj / "dv2.yaml").write_text(
        TEMPLATE_PATH.read_text().replace("enabled_bool: false", "enabled_bool: true", 1)
        .replace("DU_YOUR_PAPER_ACCOUNT", "DU_TEST"), encoding="utf-8")
    result_dict = runner.build_decision_plans(store_obj, case_obj.as_of_ts,
        str(releases_root_path_obj), "paper", auto_sync_norgate_snapshots_bool=False, **option_dict)
    assert result_dict["created_decision_plan_count_int"] == 1, result_dict
    decision_obj = store_obj.get_latest_decision_plan_for_pod(case_obj.release_obj.pod_id_str)
    assert decision_obj.entry_priority_list and decision_obj.strategy_state_dict["dv2_ind_strategy_import_str"] == DV2_IND_STRATEGY_IMPORT_STR


def test_no_intent_cycle_needs_no_quotes_and_no_sends(industry_case, monkeypatch):
    case_obj = industry_case
    no_order_state_obj = replace(case_obj.state_obj, position_amount_map={"XPH": 5.0})
    monkeypatch.setattr(strategy_module.DVO2IndustryEtfStrategy, "get_opportunities", lambda self, close: [])
    decision_obj = _decision(case_obj, no_order_state_obj)
    assert not decision_obj.entry_priority_list and not decision_obj.exit_asset_set
    store_obj = LiveStateStore(str(case_obj.tmp_path_obj/"no-order.sqlite3"))
    store_obj.upsert_release(case_obj.release_obj)
    store_obj.upsert_pod_state(no_order_state_obj)
    decision_obj = store_obj.insert_decision_plan(decision_obj)
    broker_obj = StubBrokerAdapter()
    install_daily_broker_stub(broker_obj)
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, 95000, 100000,
        no_order_state_obj.position_amount_map, decision_obj.submission_timestamp_ts, session_mode_str="paper")
    monkeypatch.setattr(broker_obj, "get_live_price_snapshot", lambda *_arg_tuple, **_arg_dict: pytest.fail("empty intent requested quotes"))
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(case_obj.tmp_path_obj/"critical.jsonl"))
    option_dict = {"log_path_str": str(case_obj.tmp_path_obj/"empty.log"), "trace_enabled_bool": False}
    assert runner.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", **option_dict)["created_vplan_count_int"] == 1
    vplan_obj = store_obj.get_latest_vplan_for_pod(case_obj.release_obj.pod_id_str)
    assert not vplan_obj.order_delta_map
    runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts, "paper", False,
        vplan_id_int=vplan_obj.vplan_id_int, **option_dict)
    assert not broker_obj.submitted_order_request_list
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(vplan_obj.target_execution_timestamp_ts.date(), "XNYS") + timedelta(minutes=1)
    broker_obj.seed_account_snapshot(case_obj.release_obj.account_route_str, 95000, 100000,
        no_order_state_obj.position_amount_map, close_timestamp_ts, session_mode_str="paper")
    assert runner.post_execution_reconcile(store_obj, broker_obj, close_timestamp_ts,
        "paper", **option_dict)["completed_vplan_count_int"] == 1
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"


def test_mid_build_manifest_mutation_and_short_history_block(industry_case, monkeypatch):
    case_obj = industry_case
    original_load_fn = snapshot_module.load_valid_snapshot_manifest
    original_price_fn = strategy_module.get_prices
    changed_bool = False
    def manifest_fn(*argument_tuple, **argument_dict):
        manifest_obj = original_load_fn(*argument_tuple, **argument_dict)
        return replace(manifest_obj, manifest_hash_str="changed") if changed_bool else manifest_obj
    def price_fn(*argument_tuple, **argument_dict):
        nonlocal changed_bool
        price_df = original_price_fn(*argument_tuple, **argument_dict)
        changed_bool = True
        return price_df
    monkeypatch.setattr(snapshot_module, "load_valid_snapshot_manifest", manifest_fn)
    monkeypatch.setattr(strategy_module, "get_prices", price_fn)
    with pytest.raises(ValueError, match="identity changed"):
        _decision(case_obj)
    changed_bool = False
    monkeypatch.setattr(strategy_module, "get_prices", lambda *_arg_tuple, **_arg_dict: case_obj.direct_price_df.iloc[-200:])
    with pytest.raises(ValueError, match="252-session warmup"):
        _decision(case_obj)


def test_stale_market_snapshot_blocks_exact_session(industry_case):
    case_obj = industry_case
    next_session_ts = scheduler_utils.get_next_session_label_ts(case_obj.provider_obj.date_idx[-1], "XNYS")
    next_close_ts = scheduler_utils.get_session_close_timestamp_ts(next_session_ts, "XNYS")
    state_obj = replace(case_obj.state_obj, updated_timestamp_ts=next_close_ts+timedelta(minutes=10))
    with pytest.raises(ValueError, match="exact latest completed session"):
        strategy_host.build_decision_plan_for_release(case_obj.release_obj, next_close_ts+timedelta(minutes=20), state_obj)


@pytest.mark.parametrize("reverse_bool", [False, True])
@pytest.mark.parametrize("account_route_str", ["DU_TEST", " du_test "])
def test_dedicated_account_rejects_monthly_collision_in_either_order(reverse_bool, account_route_str):
    from alpha.live.release_manifest import validate_release_list
    daily_release_obj = replace(parse_release_manifest(str(TEMPLATE_PATH)), enabled_bool=True, account_route_str="DU_TEST")
    monthly_release_obj = replace(parse_release_manifest(
        "docs/live/release_templates/pod_ndx_atr_normalized_monthly_open.yaml.example"),
        enabled_bool=True, account_route_str=account_route_str)
    release_list = [daily_release_obj, monthly_release_obj]
    with pytest.raises(ValueError, match="DV2-IND requires one dedicated account"):
        validate_release_list(list(reversed(release_list)) if reverse_bool else release_list)
    validate_release_list([daily_release_obj, replace(monthly_release_obj, account_route_str="DU_OTHER")])
    validate_release_list([daily_release_obj, replace(monthly_release_obj, enabled_bool=False)])


def test_dv2_overdue_alert_delivers_once(daily_case):
    from alpha.live import daily_notifications
    case_obj = daily_case
    alert_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(
        case_obj.vplan_obj.target_execution_timestamp_ts.date(), "XNYS") + timedelta(hours=1)
    assert daily_notifications.enqueue_daily_cycle_overdue(case_obj.store_obj, case_obj.release_obj,
        case_obj.decision_obj, alert_timestamp_ts, error_str="synthetic unreachable broker")
    assert not daily_notifications.enqueue_daily_cycle_overdue(case_obj.store_obj, case_obj.release_obj,
        case_obj.decision_obj, alert_timestamp_ts, error_str="synthetic unreachable broker")
    summary_dict = {"pod_row_dict_list": [{"db_path_str": case_obj.store_obj.db_path_str,
        "pod_id_str": case_obj.release_obj.pod_id_str, "account_route_str": case_obj.release_obj.account_route_str,
        "mode_str": "paper", "strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR}]}
    delivery_list = daily_notifications.deliver_daily_alerts(summary_dict, webhook_url_str="offline-test",
        webhook_poster_fn=lambda *_arg_tuple: True, now_ts=alert_timestamp_ts)
    assert len(delivery_list) == 1 and delivery_list[0].delivered_bool
    assert daily_notifications.deliver_daily_alerts(summary_dict, webhook_url_str="offline-test",
        webhook_poster_fn=lambda *_arg_tuple: pytest.fail("duplicate overdue alert"), now_ts=alert_timestamp_ts) == []


def test_watchdog_routes_both_dv2_outboxes_and_excludes_monthly_pods(daily_case, monkeypatch):
    from alpha.live import daily_notifications
    from scripts import live_ops_watchdog as watchdog_module
    case_obj = daily_case
    alert_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(
        case_obj.vplan_obj.target_execution_timestamp_ts.date(), "XNYS") + timedelta(hours=1)
    # Synthetic persisted events isolate the watchdog routing boundary.
    with case_obj.store_obj._connect() as connection_obj:
        mr_capsule_notifications.enqueue_daily_exception_alert(connection_obj,
            decision_plan_id_int=case_obj.decision_obj.decision_plan_id_int,
            vplan_id_int=case_obj.vplan_obj.vplan_id_int, pod_id_str=case_obj.release_obj.pod_id_str,
            account_route_str=case_obj.release_obj.account_route_str, mode_str="paper",
            exception_list=[{"asset_str": "SMH", "quantity_float": 5.0, "side_str": "BUY",
                "reason_str": "Synthetic missed auction buy", "expected_share_float": 5.0,
                "actual_share_float": 0.0}], created_timestamp_ts=alert_timestamp_ts)
    assert daily_notifications.enqueue_daily_cycle_overdue(case_obj.store_obj, case_obj.release_obj,
        case_obj.decision_obj, alert_timestamp_ts, error_str="Synthetic pending cycle")
    pod_row_list = [{"db_path_str": case_obj.store_obj.db_path_str,
        "pod_id_str": case_obj.release_obj.pod_id_str, "account_route_str": case_obj.release_obj.account_route_str,
        "mode_str": "paper", "strategy_import_str": DV2_IND_STRATEGY_IMPORT_STR}]
    for pod_name_str, strategy_import_str in (
            ("monthly_ndx", "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy"),
            ("monthly_taa", "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash")):
        pod_row_list.append({"db_path_str": str(case_obj.tmp_path_obj/(pod_name_str+".sqlite3")),
            "pod_id_str": pod_name_str, "account_route_str": "DU_"+pod_name_str.upper(),
            "mode_str": "paper", "strategy_import_str": strategy_import_str})
    summary_dict = {"pod_row_dict_list": pod_row_list}
    sent_payload_list = []
    def poster_fn(url_str, payload_dict):
        sent_payload_list.append(payload_dict)
        return True
    monkeypatch.setattr(watchdog_module.notifications_module, "post_discord_webhook_bool", poster_fn)
    scoped_pod_list = []
    original_scope_fn = mr_capsule_notifications._configured_scope_list
    def record_scope_fn(scoped_summary_dict, mode_str):
        scoped_pod_list.extend(row_dict["pod_id_str"] for row_dict in scoped_summary_dict["pod_row_dict_list"])
        return original_scope_fn(scoped_summary_dict, mode_str)
    monkeypatch.setattr(mr_capsule_notifications, "_configured_scope_list", record_scope_fn)
    assert watchdog_module._run_capsule_notifications_tuple(summary_dict, "live", "offline-test") == ({}, [])
    assert sent_payload_list == []
    result_dict, invalid_row_list = watchdog_module._run_capsule_notifications_tuple(summary_dict, "paper", "offline-test")
    assert result_dict.get("capsule_notification_attempt_count_int", 0) == len(sent_payload_list) == 2
    assert result_dict["capsule_notification_pending_count_int"] == 0
    assert invalid_row_list == [] and set(scoped_pod_list) == {case_obj.release_obj.pod_id_str}
    assert all(case_obj.release_obj.pod_id_str in payload_dict["content"] for payload_dict in sent_payload_list)
    result_dict, invalid_row_list = watchdog_module._run_capsule_notifications_tuple(summary_dict, "paper", "offline-test")
    assert result_dict["capsule_notification_attempt_count_int"] == 0
    assert result_dict["capsule_notification_pending_count_int"] == 0 and invalid_row_list == []
    assert len(sent_payload_list) == 2


def test_watchdog_preserves_existing_daily_membership_and_excludes_ndx_taa(tmp_path, monkeypatch):
    from scripts import live_ops_watchdog as watchdog_module
    from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
    from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
    delivery_call_list = []
    def execution_fn(summary_dict, **option_dict):
        delivery_call_list.append(("execution", summary_dict["pod_row_dict_list"][0]["strategy_import_str"]))
        return []
    def daily_fn(summary_dict, **option_dict):
        delivery_call_list.append(("daily", summary_dict["pod_row_dict_list"][0]["strategy_import_str"]))
        return []
    monkeypatch.setattr(watchdog_module.capsule_notifications_module, "deliver_execution_alerts", execution_fn)
    monkeypatch.setattr(watchdog_module.daily_notifications_module, "deliver_daily_alerts", daily_fn)
    monkeypatch.setattr(watchdog_module.capsule_notifications_module, "pending_execution_alert_count_int", lambda *_arg_tuple, **_arg_dict: 0)
    monkeypatch.setattr(watchdog_module.daily_notifications_module, "pending_daily_alert_count_int", lambda *_arg_tuple, **_arg_dict: 0)
    daily_strategy_list = [CORE5_STRATEGY_IMPORT_STR, *MR_CAPSULE_STRATEGY_IMPORT_TUPLE, DV2_IND_STRATEGY_IMPORT_STR]
    strategy_list = [*daily_strategy_list,
        "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
        "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"]
    summary_dict = {"pod_row_dict_list": [{"pod_id_str": f"pod_{index_int}",
        "account_route_str": f"DU_{index_int}", "mode_str": "paper",
        "db_path_str": str(tmp_path/f"pod_{index_int}.sqlite3"), "strategy_import_str": strategy_str}
        for index_int, strategy_str in enumerate(strategy_list)]}
    result_dict, invalid_row_list = watchdog_module._run_capsule_notifications_tuple(summary_dict, "paper", "offline-test")
    assert delivery_call_list == [(channel_str, strategy_str) for strategy_str in daily_strategy_list
        for channel_str in ("execution", "daily")]
    assert result_dict["capsule_notification_attempt_count_int"] == result_dict["capsule_notification_pending_count_int"] == 0
    assert invalid_row_list == []
