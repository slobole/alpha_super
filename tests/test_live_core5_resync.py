"""Normal daily CORE5 replay, hold drift and missed-event catch-up."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.live import core5_adapter as adapter_module, scheduler_utils
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module
import test_live_core5_adapter as fixture_module


@pytest.fixture
def release_obj():
    return fixture_module.release_obj.__wrapped__()


@pytest.fixture
def pricing_df():
    return fixture_module.price_df.__wrapped__()


def _metadata_dict(date_str="2026-09-11"):
    return {"norgate_snapshot_date_str": date_str, "norgate_data_profile_str": "norgate_eod_core5",
        "norgate_manifest_hash_str": "fixture_hash"}


def _build(release_obj, pricing_df, state_obj, date_str="2026-09-11"):
    return adapter_module.build_core5_decision_from_prices(
        release_obj, fixture_module._time(date_str), state_obj, pricing_df, _metadata_dict(date_str))


def test_missed_session_replays_memory_and_sizes_only_from_actual_account(release_obj, pricing_df):
    stale_state_dict = {
        "core5_state_version_int": 1, "initialized_bool": True, "last_signal_date_str": "2026-08-03",
        "last_rebalance_date_str": "2026-07-31",
        "last_long_state_map_dict": {asset_str: 0 for asset_str in core5_module.RISK_ASSET_TUPLE},
        "last_target_weight_map_dict": {**{asset_str: 0.0 for asset_str in core5_module.RISK_ASSET_TUPLE}, "BIL": 1.0, "Cash": 0.0},
    }
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"SPY": 71, "DBC": -3}, stale_state_dict, 12_345.0)
    original_state_obj = deepcopy(state_obj)
    decision_obj = _build(release_obj, pricing_df, state_obj)
    metadata_dict = decision_obj.snapshot_metadata_dict
    assert metadata_dict["core5_state_basis_str"] == "price_history_replay_through_close_t"
    assert metadata_dict["rebalance_bool"] is True
    assert metadata_dict["base_strategy_state_dict"] == stale_state_dict
    assert decision_obj.decision_base_position_map == {"SPY": 71.0, "DBC": -3.0}
    assert decision_obj.strategy_state_dict["last_signal_date_str"] == "2026-09-11"
    assert metadata_dict["core5_candidate_execution_receipt_dict"]["last_applied_target_share_map_dict"] == metadata_dict["fixed_target_share_map_dict"]
    assert decision_obj.target_execution_timestamp_ts == scheduler_utils.get_session_open_timestamp_ts(
        pd.Timestamp("2026-09-14"), "XNYS")
    actual_nav_float = 12_345.0 + 71 * pricing_df.loc["2026-09-11", ("SPY", "Close")] - 3 * pricing_df.loc["2026-09-11", ("DBC", "Close")]
    assert metadata_dict["sizing_close_nav_float"] == pytest.approx(actual_nav_float)
    assert metadata_dict["fixed_target_share_map_dict"] == {
        asset_str: float(int(actual_nav_float * decision_obj.full_target_weight_map_dict.get(asset_str, 0.0)
                            / pricing_df.loc["2026-09-11", (asset_str, "Close")]))
        for asset_str in adapter_module.CORE5_ASSET_TUPLE}
    assert adapter_module.validated_core5_target_share_dict(decision_obj, release_obj,
        state_obj.position_amount_map) == metadata_dict["fixed_target_share_map_dict"]
    assert state_obj == original_state_obj


def test_reconstructed_state_matches_actual_backtest_through_close_t(release_obj, pricing_df):
    # Actual engine execution on Monday consumes Friday Close_T. The replay
    # builder sees Friday only and reconstructs memory, never engine holdings.
    oracle_price_df = pricing_df.loc[:"2026-09-14"].copy()
    oracle_price_df[("$SPX", "Close")] = oracle_price_df[("SPY", "Close")] * 50
    oracle_obj = core5_module.AdaptiveMacroCore5Strategy()
    run_daily(oracle_obj, oracle_price_df,
        calendar=core5_module.build_execution_calendar_idx(oracle_price_df), show_progress=False,
        show_signal_progress_bool=False, audit_override_bool=False)
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"BIL": 100}, {"stale_memory": True})
    decision_obj = _build(release_obj, pricing_df, state_obj)
    replay_state_dict = decision_obj.strategy_state_dict
    last_rebalance_dict = oracle_obj.rebalance_target_weight_row_dict_list[-1]
    assert replay_state_dict["last_target_weight_map_dict"] == oracle_obj.last_target_weight_ser.to_dict()
    assert replay_state_dict["last_rebalance_date_str"] == last_rebalance_dict["decision_date_ts"].date().isoformat()
    assert replay_state_dict["last_signal_date_str"] == oracle_obj.daily_target_weight_row_dict_list[-1]["decision_date_ts"].date().isoformat()
    signal_df = oracle_obj.compute_signals(pricing_df.loc[:"2026-09-11"].copy())
    assert replay_state_dict["last_long_state_map_dict"] == {
        asset_str: int(value_float) for asset_str, value_float in oracle_obj._long_state_ser(signal_df.loc["2026-09-11"]).items()}
    assert decision_obj.snapshot_metadata_dict["core5_replay_decision_count_int"] == len(oracle_obj.daily_target_weight_row_dict_list)


def test_replay_uses_explicit_historical_calendar_outside_default_window(pricing_df, monkeypatch):
    historical_df = pricing_df.copy()
    historical_df.index = adapter_module.exchange_calendar_module.get_calendar("XNYS",
        start="2006-01-03", end="2009-12-31").sessions[:len(historical_df)]
    default_calendar_obj = adapter_module.exchange_calendar_module.get_calendar("XNYS",
        start="2007-09-01", end="2030-12-31")
    monkeypatch.setattr(scheduler_utils, "get_exchange_calendar_obj", lambda _calendar_str: default_calendar_obj)
    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    signal_df = strategy_obj.compute_signals(historical_df)
    replay_state_dict, metadata_dict = adapter_module._replay_core5_state_dict(strategy_obj, historical_df, signal_df)
    assert metadata_dict["core5_replay_start_date_str"] == "2007-08-31"
    assert replay_state_dict["last_signal_date_str"] == historical_df.index[-1].date().isoformat()


def test_first_actionable_close_replays_without_requiring_tomorrows_open(pricing_df):
    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    first_execution_ts = core5_module.build_execution_calendar_idx(pricing_df)[0]
    first_decision_int = pricing_df.index.get_loc(first_execution_ts) - 1
    prefix_df = pricing_df.iloc[:first_decision_int + 1].copy()
    signal_df = strategy_obj.compute_signals(prefix_df)
    replay_state_dict, metadata_dict = adapter_module._replay_core5_state_dict(strategy_obj, prefix_df, signal_df)
    assert metadata_dict["core5_replay_decision_count_int"] == 1
    assert replay_state_dict["last_rebalance_date_str"] == prefix_df.index[-1].date().isoformat()


def test_catch_up_retains_last_rebalance_dbc_weight_instead_of_refreshing_volatility(release_obj, pricing_df, monkeypatch):
    signal_df = fixture_module._controlled_signals(monkeypatch, pricing_df)
    signal_df[(core5_module.signal_namespace_str("DBC"), "long_state_ser")] = 0.0
    signal_df[(core5_module.signal_namespace_str("DBC"), "short_state_ser")] = 1.0
    signal_df[(core5_module.signal_namespace_str("DBC"), "annualized_volatility_ser")] = .5
    signal_df.loc["2026-09-11", (core5_module.signal_namespace_str("DBC"), "annualized_volatility_ser")] = 2.0
    signal_df[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.LONG_STATE_CHANGED_FIELD_STR)] = False
    signal_df[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.MONTH_END_REBALANCE_FIELD_STR)] = core5_module._month_end_rebalance_ser(signal_df.index)
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"DBC": 100}, {"blocked_cycle": True})
    decision_obj = _build(release_obj, pricing_df, state_obj)
    assert decision_obj.full_target_weight_map_dict["DBC"] == pytest.approx(-.05)
    assert decision_obj.full_target_weight_map_dict["DBC"] != pytest.approx(-.0125)
    assert decision_obj.strategy_state_dict["last_rebalance_date_str"] == "2026-08-31"
    assert decision_obj.snapshot_metadata_dict["core5_catch_up_bool"] is True
    assert decision_obj.snapshot_metadata_dict["rebalance_bool"] is True
    assert decision_obj.preserve_untouched_positions_bool is False


def test_future_prices_cannot_change_replayed_memory_or_catch_up_shares(release_obj, pricing_df):
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"IEF": 30}, {"last_signal_date_str": "2025-01-02"})
    expected_obj = _build(release_obj, pricing_df.loc[:"2026-09-11"].copy(), state_obj)
    changed_df = pricing_df.copy()
    changed_df.loc[changed_df.index > pd.Timestamp("2026-09-11"), :] *= 123.0
    actual_obj = _build(release_obj, changed_df, state_obj)
    assert actual_obj == expected_obj


@pytest.mark.parametrize("holding_case_str", ["preserved", "partial", "liquidated", "receipt_missing", "receipt_invalid"])
def test_receipt_preserves_share_drift_only_after_actual_application(release_obj, pricing_df, monkeypatch, holding_case_str):
    fixture_module._controlled_signals(monkeypatch, pricing_df)
    first_obj = _build(release_obj, pricing_df, fixture_module._state(release_obj, "2026-09-10"), "2026-09-10")
    position_dict = dict(first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"])
    committed_state_dict = fixture_module._committed_state_dict(first_obj)
    if holding_case_str == "partial":
        position_dict["SPY"] -= 2
    elif holding_case_str == "liquidated":
        position_dict = {}
    elif holding_case_str == "receipt_missing":
        committed_state_dict.pop("core5_execution_receipt_dict")
    elif holding_case_str == "receipt_invalid":
        committed_state_dict["core5_execution_receipt_dict"] = "invalid"
    # A large cash change does not force daily weight rebalancing when the
    # successfully applied share units and strategy event remain unchanged.
    state_obj = fixture_module._state(release_obj, "2026-09-11", position_dict, committed_state_dict, 500_000)
    decision_obj = _build(release_obj, pricing_df, state_obj)
    assert decision_obj.snapshot_metadata_dict["rebalance_bool"] == (holding_case_str != "preserved")
    if holding_case_str == "preserved":
        assert decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] == position_dict
    else:
        assert decision_obj.snapshot_metadata_dict["core5_catch_up_bool"]
        assert decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]["SPY"] > position_dict.get("SPY", 0)
    # Creating a decision never turns proposed targets into a success receipt.
    assert decision_obj.strategy_state_dict["core5_execution_receipt_dict"] == (
        committed_state_dict.get("core5_execution_receipt_dict", {}) if holding_case_str != "receipt_invalid" else {})


def test_missed_event_is_caught_even_when_long_state_returns_to_prior_value(release_obj, pricing_df, monkeypatch):
    fixture_module._controlled_signals(monkeypatch, pricing_df, {
        ("2026-09-11", "SPY", "long_state_ser"): 0.0,
    })
    first_obj = _build(release_obj, pricing_df, fixture_module._state(release_obj, "2026-09-10"), "2026-09-10")
    state_obj = fixture_module._state(release_obj, "2026-09-15",
        first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], fixture_module._committed_state_dict(first_obj))
    decision_obj = _build(release_obj, pricing_df, state_obj, "2026-09-15")
    assert decision_obj.strategy_state_dict["last_long_state_map_dict"] == first_obj.strategy_state_dict["last_long_state_map_dict"]
    assert decision_obj.strategy_state_dict["last_rebalance_date_str"] == "2026-09-14"
    assert decision_obj.snapshot_metadata_dict["rebalance_bool"]
    assert decision_obj.snapshot_metadata_dict["core5_catch_up_bool"]


def test_data_revision_warning_does_not_make_cached_signals_authoritative(release_obj, pricing_df, monkeypatch):
    signal_df = fixture_module._controlled_signals(monkeypatch, pricing_df)
    first_obj = _build(release_obj, pricing_df, fixture_module._state(release_obj, "2026-09-10"), "2026-09-10")
    signal_df.loc["2026-09-10", (core5_module.signal_namespace_str("SPY"), "long_state_ser")] = 0.0
    signal_df.loc[["2026-09-10", "2026-09-11"],
        (core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.LONG_STATE_CHANGED_FIELD_STR)] = True
    state_obj = fixture_module._state(release_obj, "2026-09-11",
        first_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], fixture_module._committed_state_dict(first_obj))
    decision_obj = _build(release_obj, pricing_df, state_obj)
    assert decision_obj.snapshot_metadata_dict["core5_data_revision_warning_bool"]
    assert "core5_strategy_state_cache_revised" in decision_obj.snapshot_metadata_dict["core5_warning_code_list"]
    assert decision_obj.strategy_state_dict["last_rebalance_date_str"] == "2026-09-11"
    assert decision_obj.snapshot_metadata_dict["rebalance_bool"]


@pytest.mark.parametrize("mutation_str", ["stale_eod", "wrong_source", "wrong_account", "future_account", "missing_session", "wrong_snapshot", "missing_hash"])
def test_daily_replay_preserves_existing_account_and_data_contracts(release_obj, pricing_df, mutation_str):
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"BIL": 100}, {"missed_sessions": True})
    metadata_dict = _metadata_dict()
    if mutation_str == "stale_eod":
        state_obj = replace(state_obj, updated_timestamp_ts=fixture_module._time("2026-09-10"))
    elif mutation_str == "wrong_source":
        state_obj = replace(state_obj, snapshot_source_str="bootstrap")
    elif mutation_str == "wrong_account":
        state_obj = replace(state_obj, account_route_str="DU_OTHER")
    elif mutation_str == "future_account":
        state_obj = replace(state_obj, updated_timestamp_ts=fixture_module._time("2026-09-14"))
    elif mutation_str == "missing_session":
        pricing_df = pricing_df.drop(pd.Timestamp("2026-08-12"))
    elif mutation_str == "wrong_snapshot":
        metadata_dict["norgate_snapshot_date_str"] = "2026-09-10"
    else:
        metadata_dict.pop("norgate_manifest_hash_str")
    with pytest.raises(ValueError):
        adapter_module.build_core5_decision_from_prices(release_obj,
            fixture_module._time("2026-09-11"), state_obj, pricing_df, metadata_dict)


@pytest.mark.parametrize("changed_manifest_bool", [False, True])
def test_operational_replay_uses_exact_snapshot_and_rechecks_identity(release_obj, pricing_df, monkeypatch, changed_manifest_bool):
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"BIL": 10}, {"missed_sessions": True})
    monkeypatch.setattr(adapter_module.snapshot_module, "is_snapshot_mode_enabled_bool", lambda: True)
    def manifest_fn(profile_str, minimum_snapshot_date_str):
        assert profile_str == "norgate_eod_core5" and minimum_snapshot_date_str == "2026-09-11"
        return SimpleNamespace(manifest_hash_str="fixture_hash")
    monkeypatch.setattr(adapter_module.snapshot_module, "load_valid_snapshot_manifest", manifest_fn)
    def price_fn(config_obj):
        assert config_obj.end_date_str == "2026-09-11"
        return pricing_df
    monkeypatch.setattr(core5_module, "get_adaptive_macro_core5_data", price_fn)
    metadata_dict = _metadata_dict()
    if changed_manifest_bool:
        metadata_dict["norgate_manifest_hash_str"] = "changed"
    monkeypatch.setattr(adapter_module.snapshot_module, "build_data_source_metadata_dict", lambda _profile_str: metadata_dict)
    if changed_manifest_bool:
        with pytest.raises(ValueError, match="snapshot changed"):
            adapter_module.build_core5_decision_plan(release_obj, fixture_module._time("2026-09-11"), state_obj)
    else:
        assert adapter_module.build_core5_decision_plan(release_obj, fixture_module._time("2026-09-11"), state_obj) == _build(release_obj, pricing_df, state_obj)
