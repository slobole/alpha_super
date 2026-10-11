"""E2 uses frozen month-end selections and unchanged monthly broker sizing."""
from dataclasses import asdict, replace
from datetime import datetime
from importlib import import_module
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from alpha.live import ndx_e2_adapter as adapter_module, scheduler_utils, strategy_host
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, LivePriceSnapshot, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap import build_strategy


SIGNAL_DATE_TS = pd.Timestamp("2013-06-28")
AS_OF_TS = datetime(2013, 6, 28, 16, 10, tzinfo=ZoneInfo("America/New_York"))


@pytest.fixture(params=adapter_module.NDX_E2_STRATEGY_IMPORT_TUPLE)
def e2_case(request, monkeypatch):
    release_obj = LiveRelease(release_id_str="e2.v1", user_id_str="owner", pod_id_str="e2",
        account_route_str="DU_E2", strategy_import_str=request.param, mode_str="paper",
        session_calendar_id_str="XNYS", signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open", data_profile_str=snapshot_module.NDX_E2_PROFILE_STR,
        params_dict={}, risk_profile_str="monthly", enabled_bool=False, source_path_str="test",
        pod_budget_fraction_float=1.0)
    session_idx = scheduler_utils.get_exchange_calendar_obj("XNYS").sessions_in_range("2011-01-03", SIGNAL_DATE_TS)
    session_idx = pd.DatetimeIndex(session_idx).tz_localize(None)
    elapsed_vec = np.arange(len(session_idx), dtype=float)
    symbol_list = [f"A{index_int:02d}" for index_int in range(12)]
    field_dict = {}
    for index_int, symbol_str in enumerate([*symbol_list, "SPY"]):
        close_vec = (10 + 20 * index_int) * np.exp((.0002 + .00006 * index_int) * elapsed_vec)
        for field_str, factor_float in (("Open", 1), ("Close", 1), ("High", 1.01), ("Low", .99), ("Unadjusted Close", 1)):
            field_dict[(symbol_str, field_str)] = close_vec * factor_float
    pricing_df = pd.DataFrame(field_dict, index=session_idx)
    universe_df = pd.DataFrame(1, index=session_idx, columns=symbol_list)
    sector_dict = {symbol_str: "TECH" if index_int < 6 else f"SECTOR{index_int // 3}"
        for index_int, symbol_str in enumerate(symbol_list)}
    vxn_df = pd.DataFrame({"vxn_close": [44.0], "vxn_exposure_scale_float": [.5]}, index=[SIGNAL_DATE_TS])
    contract_dict = {"sector_by_symbol_dict": dict(sector_dict),
        "sector_map_hash_str": snapshot_module.ndx_e2_sector_map_hash_str(sector_dict),
        "sector_snapshot_date_str": "2013-06-28", "sector_classification_basis_str": "current_at_export",
        "sector_observed_timestamp_str": "2013-06-28T20:10:00+00:00"}
    metadata_dict = {"norgate_data_source_mode_str": "snapshot", "norgate_data_profile_str": release_obj.data_profile_str,
        "norgate_snapshot_date_str": "2013-06-28", "norgate_manifest_hash_str": "frozen-manifest"}
    manifest_obj = SimpleNamespace(manifest_hash_str="frozen-manifest", manifest_dict={"data_contract": contract_dict})
    input_dict = {"pricing_df": pricing_df, "universe_df": universe_df, "vxn_df": vxn_df}
    strategy_module = import_module(request.param.split(":")[0])
    loader_name_str = ("get_natr20_vxn_scaled_ndx_data" if request.param == adapter_module.NDX_E2_STRATEGY_IMPORT_TUPLE[1]
        else "get_vxn_scaled_atr_normalized_ndx_data")
    def load_frames(config_obj):
        assert config_obj.end_date_str == "2013-06-28"
        return input_dict["pricing_df"], input_dict["universe_df"], None, input_dict["vxn_df"]
    def frozen_sectors(symbol_list, *, snapshot_date_str=None):
        assert snapshot_date_str == "2013-06-28"
        return {symbol_str: sector_dict[symbol_str] for symbol_str in symbol_list}
    monkeypatch.setattr(strategy_module, loader_name_str, load_frames)
    monkeypatch.setattr(snapshot_module, "is_snapshot_mode_enabled_bool", lambda: True)
    monkeypatch.setattr(snapshot_module, "build_data_source_metadata_dict", lambda *_: dict(metadata_dict))
    monkeypatch.setattr(strategy_host, "build_data_source_metadata_dict", lambda *_: dict(metadata_dict))
    monkeypatch.setattr(snapshot_module, "load_valid_snapshot_manifest", lambda *_, **__: manifest_obj)
    monkeypatch.setattr(snapshot_module, "load_ndx_e2_sector_map", frozen_sectors)
    state_obj = PodState(pod_id_str="e2", user_id_str="owner", account_route_str="DU_E2",
        position_amount_map={}, cash_float=100000.0, total_value_float=100000.0,
        strategy_state_dict={}, updated_timestamp_ts=AS_OF_TS)
    return release_obj, state_obj, input_dict, sector_dict, metadata_dict, manifest_obj


def _build(e2_case, *, as_of_ts=AS_OF_TS):
    release_obj, state_obj, *_ = e2_case
    return strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, state_obj)


def test_actual_parent_features_and_sector_selection_match_backtest_class(e2_case):
    release_obj, _, input_dict, sector_dict, _, _ = e2_case
    module_str, class_str = release_obj.strategy_import_str.split(":")
    strategy_module = import_module(module_str)
    schedule_df = pd.DataFrame({"decision_date_ts": [SIGNAL_DATE_TS]}, index=pd.to_datetime(["2013-07-01"]))
    reference_obj = build_strategy(getattr(strategy_module, class_str), "reference", strategy_module.DEFAULT_CONFIG,
        schedule_df, input_dict["vxn_df"], input_dict["universe_df"], sector_dict)
    reference_obj.previous_bar = SIGNAL_DATE_TS
    reference_obj.current_bar = pd.Timestamp("2013-07-01")
    signal_df = reference_obj.compute_signals(input_dict["pricing_df"].copy())
    expected_ser = reference_obj.get_target_weight_ser(signal_df.loc[SIGNAL_DATE_TS])
    decision_obj = _build(e2_case)
    assert decision_obj.full_target_weight_map_dict == expected_ser.to_dict()
    assert expected_ser.size >= 8
    assert expected_ser.tolist() == pytest.approx([.05] * expected_ser.size)
    assert pd.Series([sector_dict[symbol_str] for symbol_str in expected_ser.index]).value_counts().max() <= 4
    assert decision_obj.target_share_map_dict == {}
    assert decision_obj.decision_book_type_str == "full_target_weight_book"
    assert decision_obj.rebalance_omitted_assets_to_zero_bool and not decision_obj.preserve_untouched_positions_bool
    assert decision_obj.signal_timestamp_ts.date().isoformat() == "2013-06-28"
    assert decision_obj.target_execution_timestamp_ts.date().isoformat() == "2013-07-01"
    assert decision_obj.target_execution_timestamp_ts.hour == 9 and decision_obj.target_execution_timestamp_ts.minute == 30
    assert decision_obj.snapshot_metadata_dict["sector_map_hash_str"] == snapshot_module.ndx_e2_sector_map_hash_str(sector_dict)
    assert decision_obj.snapshot_metadata_dict["sector_by_symbol_dict"] == sector_dict
    assert decision_obj.snapshot_metadata_dict["norgate_manifest_hash_str"] == "frozen-manifest"
    assert decision_obj.snapshot_metadata_dict["vxn_reference_date_str"] == "2013-06-28"
    assert decision_obj.snapshot_metadata_dict["ndx_e2_sizing_contract_str"] == "legacy_monthly_preopen_v1"


def test_future_price_universe_and_vxn_rows_cannot_change_frozen_decision(e2_case):
    before_obj = _build(e2_case)
    _, _, input_dict, *_ = e2_case
    future_ts = pd.Timestamp("2013-07-01")
    input_dict["pricing_df"].loc[future_ts] = input_dict["pricing_df"].iloc[-1] * 19
    input_dict["universe_df"].loc[future_ts] = 0
    input_dict["vxn_df"].loc[future_ts] = [11.0, 1.0]
    assert asdict(_build(e2_case)) == asdict(before_obj)


def test_preopen_build_uses_prior_completed_month_end(e2_case):
    preopen_ts = datetime(2013, 7, 1, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    assert asdict(_build(e2_case, as_of_ts=preopen_ts)) == asdict(_build(e2_case))


@pytest.mark.parametrize("as_of_ts", [datetime(2013, 7, 1, 9, 30, tzinfo=ZoneInfo("America/New_York")),
    datetime(2013, 7, 15, 16, 10, tzinfo=ZoneInfo("America/New_York"))])
def test_partial_next_month_cannot_become_a_late_monthly_trade(e2_case, as_of_ts):
    with pytest.raises(ValueError, match="stale monthly"):
        _build(e2_case, as_of_ts=as_of_ts)


@pytest.mark.parametrize("field_str,value_obj", [("norgate_snapshot_date_str", "2013-06-27"),
    ("norgate_snapshot_date_str", "2013-07-01"), ("norgate_data_profile_str", "legacy"),
    ("norgate_manifest_hash_str", "")])
def test_snapshot_must_match_exact_decision_session(e2_case, field_str, value_obj):
    e2_case[4][field_str] = value_obj
    with pytest.raises(ValueError, match="exact completed month-end"):
        _build(e2_case)


def test_direct_mode_does_not_silently_use_current_classifications(e2_case, monkeypatch):
    monkeypatch.setattr(snapshot_module, "is_snapshot_mode_enabled_bool", lambda: False)
    with pytest.raises(ValueError, match="frozen snapshot"):
        _build(e2_case)


def test_missing_month_end_price_cannot_use_a_stale_signal(e2_case):
    e2_case[2]["pricing_df"] = e2_case[2]["pricing_df"].iloc[:-1]
    with pytest.raises(ValueError, match="exact decision close"):
        _build(e2_case)


@pytest.mark.parametrize("offset_int", [-1, -10, -199])
def test_missing_spy_close_in_trailing_regime_window_cannot_liquidate(e2_case, offset_int):
    pricing_df = e2_case[2]["pricing_df"]
    pricing_df.loc[pricing_df.index[offset_int], ("SPY", "Close")] = np.nan
    with pytest.raises(ValueError, match="valid completed monthly feature date"):
        _build(e2_case)


@pytest.mark.parametrize("history_count_int", [100, 220])
def test_insufficient_regime_or_monthly_roc_history_cannot_liquidate(e2_case, history_count_int):
    e2_case[2]["pricing_df"] = e2_case[2]["pricing_df"].iloc[-history_count_int:]
    with pytest.raises(ValueError, match="valid completed monthly feature date"):
        _build(e2_case)


def test_all_invalid_active_candidate_features_cannot_liquidate(e2_case):
    pricing_df = e2_case[2]["pricing_df"]
    for symbol_str in e2_case[3]:
        for field_str in ("Open", "High", "Low", "Close", "Unadjusted Close"):
            pricing_df[(symbol_str, field_str)] = 100.0
    with pytest.raises(ValueError, match="valid active-member monthly candidate features"):
        _build(e2_case)


def test_valid_risk_off_regime_remains_an_all_cash_decision(e2_case):
    pricing_df = e2_case[2]["pricing_df"]
    pricing_df[("SPY", "Close")] = np.linspace(200.0, 100.0, len(pricing_df))
    decision_obj = _build(e2_case)
    assert decision_obj.full_target_weight_map_dict == {}
    assert decision_obj.cash_reserve_weight_float == 1.0


def test_snapshot_rollover_during_data_loading_cannot_commit_mixed_provenance(e2_case, monkeypatch):
    original_fn = snapshot_module.load_ndx_e2_sector_map
    def changed_sectors(symbol_list, *, snapshot_date_str=None):
        result_dict = original_fn(symbol_list, snapshot_date_str=snapshot_date_str)
        e2_case[4]["norgate_manifest_hash_str"] = "new-manifest"
        return result_dict
    monkeypatch.setattr(snapshot_module, "load_ndx_e2_sector_map", changed_sectors)
    with pytest.raises(ValueError, match="identity changed during"):
        _build(e2_case)


@pytest.mark.parametrize("params_dict", [{"sector_cap_int": 5}, {"target_vxn_pct_float": 23},
    {"lookback_month_int": 6}, {"unexpected": 1}, {"capital_base_float": float("nan")}])
def test_release_cannot_silently_diverge_from_reference_defaults(e2_case, params_dict):
    with pytest.raises(ValueError, match="NDX E2"):
        adapter_module.validate_ndx_e2_release(replace(e2_case[0], params_dict=params_dict))


@pytest.mark.parametrize("budget_float", [0.0, -0.5, 1.1, float("nan"), float("inf"), True])
def test_release_rejects_invalid_account_budget(e2_case, budget_float):
    with pytest.raises(ValueError, match="budget fraction"):
        adapter_module.validate_ndx_e2_release(replace(e2_case[0], pod_budget_fraction_float=budget_float))


def test_explicit_frozen_defaults_and_dedicated_account_budget_are_supported(e2_case):
    release_obj = replace(e2_case[0], params_dict={"sector_cap_int": 4, "target_vxn_pct_float": 22.0,
        "capital_base_float": 150000.0}, pod_budget_fraction_float=.5)
    adapter_module.validate_ndx_e2_release(release_obj)
    decision_obj = replace(strategy_host.build_decision_plan_for_release(release_obj, AS_OF_TS, e2_case[1]),
        decision_plan_id_int=17)
    snapshot_obj = BrokerSnapshot(account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=AS_OF_TS,
        cash_float=100000.0, total_value_float=100000.0, net_liq_float=100000.0, position_amount_map={})
    quote_obj = LivePriceSnapshot(account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=AS_OF_TS,
        price_source_str="synthetic_preopen", asset_reference_price_map={
            symbol_str: 100.0 for symbol_str in decision_obj.full_target_weight_map_dict})
    plan_obj = build_vplan(release_obj, decision_obj, snapshot_obj, quote_obj)
    assert set(plan_obj.target_share_map.values()) == {25.0}
    assert set(decision_obj.full_target_weight_map_dict.values()) == {.05}


@pytest.mark.parametrize("field_str", ["pod_id_str", "user_id_str", "account_route_str"])
def test_account_state_identity_cannot_cross_dedicated_accounts(e2_case, field_str):
    with pytest.raises(ValueError, match="state must match"):
        strategy_host.build_decision_plan_for_release(e2_case[0], AS_OF_TS,
            replace(e2_case[1], **{field_str: "other"}))


def test_preopen_nav_and_quote_resize_shares_without_changing_frozen_weights(e2_case):
    release_obj, *_ = e2_case
    decision_obj = replace(_build(e2_case), decision_plan_id_int=17)
    weight_dict = dict(decision_obj.full_target_weight_map_dict)
    asset_str = next(iter(weight_dict))
    base_quote_dict = {symbol_str: 100.0 for symbol_str in weight_dict}
    base_quote_dict["OLD"] = 100.0
    quantity_list = []
    for nav_float, reference_price_float in ((100000.0, 100.0), (120000.0, 100.0), (100000.0, 125.0)):
        snapshot_obj = BrokerSnapshot(account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=AS_OF_TS,
            cash_float=5000.0, total_value_float=nav_float, net_liq_float=nav_float, position_amount_map={"OLD": 10.0})
        quote_obj = LivePriceSnapshot(account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=AS_OF_TS,
            price_source_str="synthetic_preopen", asset_reference_price_map={**base_quote_dict, asset_str: reference_price_float})
        plan_obj = build_vplan(release_obj, decision_obj, snapshot_obj, quote_obj)
        quantity_list.append(plan_obj.target_share_map[asset_str])
        assert plan_obj.order_delta_map["OLD"] == -10.0
        assert all(request_obj.broker_order_type_str == "MOO" for request_obj in build_broker_order_request_list_from_vplan(plan_obj))
        assert decision_obj.full_target_weight_map_dict == weight_dict
    assert quantity_list == [50.0, 60.0, 40.0]
