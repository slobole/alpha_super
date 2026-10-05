"""Real-indicator capsule host parity on deterministic synthetic histories.

Only data access and the absolute HPI cross-section size are replaced. Indicator,
opportunity, readiness, gate, parking, iterate, and host mapping code all run.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import exchange_calendars
import numpy as np
import pandas as pd
import pytest

from alpha.live import mr_capsule_adapter as adapter_mod, strategy_host
from alpha.live.models import PodState
from alpha.live.release_manifest import parse_release_manifest
from strategies.hpi import stateful_long as hpi_mod
from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated, vix_stress_gate


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
STOCK_SYMBOL_LIST = [f"S{symbol_int:02d}" for symbol_int in range(13)]
IDENTITY_TUPLE = tuple((pod_str, mode_str) for pod_str in ("dv2_vix_gated", "hpi_vote_vix_gated")
                      for mode_str in ("cash", "bil", "spmo"))


def _market_inputs(scenario_str):
    signal_date_ts = pd.Timestamp("2024-01-12" if scenario_str == "weekly" else "2024-01-10")
    full_session_idx = exchange_calendars.get_calendar("XNYS", start="1990-01-02", end="2024-01-22").sessions.tz_localize(None)
    signal_position_int = full_session_idx.get_loc(signal_date_ts)
    pricing_idx = full_session_idx[signal_position_int - 1450:signal_position_int + 4]
    bar_vec = np.arange(len(pricing_idx), dtype=float)
    decision_position_int = pricing_idx.get_loc(signal_date_ts)
    column_dict = {}
    for symbol_int, symbol_str in enumerate([*STOCK_SYMBOL_LIST, "BIL", "SPMO", "$SPX", "$SPXTR"]):
        if symbol_str == "BIL":
            close_vec = 91.0 + 0.01 * np.sin(bar_vec / 4.0)
        elif symbol_str == "SPMO":
            close_vec = 100.0 * np.exp(0.0002 * bar_vec + 0.03 * np.sin(bar_vec / 3.0))
        else:
            close_vec = (20.0 + symbol_int) * np.exp(0.0015 * bar_vec + 0.012 * np.sin(bar_vec / 3.0 + symbol_int / 5.0))
            if symbol_str in STOCK_SYMBOL_LIST:
                # Causal final dip: positive long trend, low IBS, and returns in
                # the bottom HPI tail. This creates real entries in both pods.
                close_vec[decision_position_int - 4:decision_position_int + 1] *= np.array([.985, .975, .955, .94, .925])
        high_vec, low_vec = close_vec * 1.008, close_vec * .993
        if symbol_str in STOCK_SYMBOL_LIST:
            high_vec[decision_position_int - 1:decision_position_int + 1] = close_vec[decision_position_int - 1:decision_position_int + 1] * 1.05
            low_vec[decision_position_int - 1:decision_position_int + 1] = close_vec[decision_position_int - 1:decision_position_int + 1] * .999
        volume_vec = np.full(len(pricing_idx), 1_000_000.0 + symbol_int * 1000.0)
        for field_str, value_vec in {
            "Open": close_vec * 1.001, "High": high_vec, "Low": low_vec, "Close": close_vec,
            "Volume": volume_vec, "Turnover": close_vec * volume_vec,
            "Unadjusted Close": close_vec, "Dividend": np.zeros(len(pricing_idx)),
        }.items():
            column_dict[(symbol_str, field_str)] = value_vec
    pricing_df = pd.DataFrame(column_dict, index=pricing_idx)
    universe_df = pd.DataFrame(1, index=pricing_idx, columns=STOCK_SYMBOL_LIST)
    universe_df.loc[signal_date_ts:, "S12"] = 0  # A former member must not enter today.
    vix_close_ser = pd.Series(10.0, index=full_session_idx[:signal_position_int + 1])
    if scenario_str == "closing":
        vix_close_ser.iloc[-16] = 50.0  # memory expires on this decision close
    elif scenario_str != "weekly":
        vix_close_ser.iloc[-1] = 50.0
    return signal_date_ts, pricing_df, universe_df, vix_close_ser


def _case_state(release_obj, signal_date_ts, pricing_df, scenario_str):
    held_stock_list = STOCK_SYMBOL_LIST[:10] if scenario_str in {"full_slots", "pending_exit"} else ["S00"]
    position_map_dict = {symbol_str: 10.0 for symbol_str in held_stock_list}
    mode_str = release_obj.strategy_import_str.rsplit("_", 1)[-1]
    if mode_str != "cash":
        position_map_dict["BIL"] = 100.0
    if mode_str == "spmo":
        position_map_dict["SPMO"] = 50.0
    held_value_float = sum(share_float * float(pricing_df.loc[signal_date_ts, (symbol_str, "Close")])
                           for symbol_str, share_float in position_map_dict.items())
    strategy_state_dict = {
        "mr_capsule_strategy_import_str": release_obj.strategy_import_str,
        "trade_id_int": 37,
        "current_trade_map": {symbol_str: 20 + symbol_int for symbol_int, symbol_str in enumerate(held_stock_list)},
        "pending_exit_symbol_list": ["S00"] if scenario_str == "pending_exit" else [],
    }
    return PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=position_map_dict, cash_float=100_000.0 - held_value_float,
        total_value_float=105_000.0,  # Account mark deliberately differs from reconstructed Close_T NAV.
        strategy_state_dict=strategy_state_dict,
        updated_timestamp_ts=datetime.combine(signal_date_ts.date(), datetime.min.time(), MARKET_TIMEZONE_OBJ).replace(hour=17),
        snapshot_stage_str="eod", snapshot_source_str="broker",
    )


def _install_inputs(monkeypatch, release_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser):
    metadata_dict = {"norgate_snapshot_date_str": signal_date_ts.date().isoformat(),
                     "norgate_data_profile_str": release_obj.data_profile_str, "norgate_manifest_hash_str": "synthetic-parity"}
    monkeypatch.setattr(adapter_mod, "build_data_source_metadata_dict", lambda *_: dict(metadata_dict))
    monkeypatch.setattr(strategy_host, "build_data_source_metadata_dict", lambda *_: dict(metadata_dict))
    monkeypatch.setattr(dv2_vix_gated, "load_pricing_data", lambda *_: (pricing_df, universe_df))
    monkeypatch.setattr(hpi_mod, "load_exact_hpi_inputs", lambda **_: (STOCK_SYMBOL_LIST, universe_df, pricing_df))
    # The live adapter and the backtest share hpi_vote_vix_gated.load_hpi_capsule_pricing_data, which calls the
    # loader bound in that module; patch that binding too so no test reaches real Norgate data.
    monkeypatch.setattr(hpi_vote_vix_gated, "load_exact_hpi_inputs", lambda **_: (STOCK_SYMBOL_LIST, universe_df, pricing_df))
    monkeypatch.setattr(hpi_vote_vix_gated, "append_parking_prices", lambda frame_df, *_: frame_df)
    monkeypatch.setattr(vix_stress_gate, "load_vix_close_ser", lambda *_: vix_close_ser)
    # Keep the actual feature-readiness validator and its 80% rule active.
    monkeypatch.setattr(strategy_host, "HPI_MINIMUM_READY_MEMBER_COUNT_INT", 12)


def _research_decision(release_obj, state_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser, *, full_signal_bool=False):
    hpi_bool = "hpi_vote" in release_obj.strategy_import_str
    if hpi_bool:
        strategy_obj = hpi_vote_vix_gated.HPIVoteVixGatedStrategy(
            name="research_parity", benchmarks=["$SPXTR"], capital_base=100_000.0,
            ranking_field_str=hpi_mod.TURNOVER_FIELD_STR, entry_mode_str=hpi_mod.ENTRY_HORIZON_VOTE_STR,
        )
        strategy_obj.trade_id_int = state_obj.strategy_state_dict["trade_id_int"]
        strategy_obj.current_trade_map = defaultdict(lambda: -1, state_obj.strategy_state_dict["current_trade_map"])
        strategy_obj.pending_exit_symbol_set = set(state_obj.strategy_state_dict["pending_exit_symbol_list"])
    else:
        strategy_obj = dv2_vix_gated.DV2VixGatedStrategy(name="research_parity", benchmarks=["$SPX"], capital_base=100_000.0)
        strategy_obj.trade_id = state_obj.strategy_state_dict["trade_id_int"]
        strategy_obj.current_trade = defaultdict(lambda: -1, state_obj.strategy_state_dict["current_trade_map"])
    # Seed independently of the live helper, with the exact Close_T cash/holdings.
    strategy_obj._position_amount_map = dict(state_obj.position_amount_map)
    strategy_obj.cash = state_obj.cash_float
    strategy_obj._total_value_history_list = [100_000.0]
    strategy_obj.universe_df = universe_df
    strategy_obj.vix_close_ser = vix_close_ser
    mode_str = release_obj.strategy_import_str.rsplit("_", 1)[-1]
    strategy_obj.parking_enabled_bool = mode_str != "cash"
    strategy_obj.spmo_parking_enabled_bool = mode_str == "spmo"
    compute_input_df = pricing_df if full_signal_bool else pricing_df.loc[:signal_date_ts]
    signal_df = strategy_obj.compute_signals(compute_input_df)
    strategy_obj.previous_bar = signal_date_ts
    strategy_obj.current_bar = pricing_df.index[pricing_df.index.get_loc(signal_date_ts) + 1]
    # The research engine receives the actual next-open row; all opens here are
    # present, so HPI's live tradability marker must preserve the same decision.
    open_price_ser = pricing_df.loc[strategy_obj.current_bar].xs("Open", level=1)
    strategy_obj.iterate(signal_df.loc[:signal_date_ts], signal_df.loc[signal_date_ts], open_price_ser)
    return strategy_obj, signal_df


def _assert_intent_parity(decision_obj, research_obj, state_obj):
    expected_entry_dict, expected_target_dict, expected_exit_set = {}, {}, set()
    for order_obj in research_obj.get_orders():
        assert type(order_obj).__name__ == "MarketOrder"
        if order_obj.target and order_obj.amount == 0:
            expected_exit_set.add(order_obj.asset)
        elif order_obj.target:
            assert order_obj.unit == "shares"
            expected_target_dict[order_obj.asset] = float(order_obj.amount)
        else:
            assert order_obj.unit == "value"
            expected_entry_dict[order_obj.asset] = float(order_obj.amount)
    actual_entry_dict = {symbol_str: weight_float * decision_obj.snapshot_metadata_dict["decision_nav_float"]
                         for symbol_str, weight_float in decision_obj.entry_target_weight_map_dict.items()}
    assert actual_entry_dict == pytest.approx(expected_entry_dict)
    assert decision_obj.entry_priority_list == list(expected_entry_dict)
    assert decision_obj.target_share_map_dict == expected_target_dict
    assert decision_obj.exit_asset_set == expected_exit_set
    assert decision_obj.decision_base_position_map == state_obj.position_amount_map
    assert decision_obj.snapshot_metadata_dict["decision_nav_float"] == pytest.approx(100_000.0)
    assert "S12" not in actual_entry_dict
    expected_trade_id_int = research_obj.trade_id_int if hasattr(research_obj, "trade_id_int") else research_obj.trade_id
    expected_trade_map_dict = research_obj.current_trade_map if hasattr(research_obj, "current_trade_map") else research_obj.current_trade
    assert decision_obj.strategy_state_dict["trade_id_int"] == expected_trade_id_int
    assert decision_obj.strategy_state_dict["current_trade_map"] == dict(expected_trade_map_dict)
    if hasattr(research_obj, "pending_exit_symbol_set"):
        assert set(decision_obj.strategy_state_dict["pending_exit_symbol_list"]) == research_obj.pending_exit_symbol_set


@pytest.mark.parametrize("pod_str,mode_str", IDENTITY_TUPLE)
@pytest.mark.parametrize("scenario_str", ["opening", "closing", "weekly", "full_slots"])
def test_actual_capsule_signals_and_orders_match_fresh_host(monkeypatch, pod_str, mode_str, scenario_str):
    release_obj = parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_{mode_str}_daily_moo.yaml.example")
    signal_date_ts, pricing_df, universe_df, vix_close_ser = _market_inputs(scenario_str)
    state_obj = _case_state(release_obj, signal_date_ts, pricing_df, scenario_str)
    _install_inputs(monkeypatch, release_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    as_of_ts = state_obj.updated_timestamp_ts.replace(hour=18)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, state_obj)
    research_obj, _ = _research_decision(release_obj, state_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    _assert_intent_parity(decision_obj, research_obj, state_obj)
    assert decision_obj.snapshot_metadata_dict["gate_open_bool"] is (scenario_str in {"opening", "full_slots"})
    if scenario_str == "opening":
        assert len(decision_obj.entry_target_weight_map_dict) == 9  # Actual indicators fill all free stock slots.
        if mode_str == "spmo":
            assert "SPMO" in decision_obj.exit_asset_set
    else:
        assert not decision_obj.entry_target_weight_map_dict
    if scenario_str in {"closing", "weekly"} and mode_str != "cash":
        assert decision_obj.target_share_map_dict
        if mode_str == "spmo":
            assert decision_obj.target_share_map_dict["SPMO"] > 0


@pytest.mark.parametrize("mode_str", ["cash", "bil", "spmo"])
def test_actual_hpi_pending_exit_replays_and_frees_one_full_slot(monkeypatch, mode_str):
    release_obj = parse_release_manifest(f"docs/live/release_templates/pod_mr_hpi_vote_vix_gated_{mode_str}_daily_moo.yaml.example")
    signal_date_ts, pricing_df, universe_df, vix_close_ser = _market_inputs("pending_exit")
    state_obj = _case_state(release_obj, signal_date_ts, pricing_df, "pending_exit")
    _install_inputs(monkeypatch, release_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    decision_obj = strategy_host.build_decision_plan_for_release(release_obj, state_obj.updated_timestamp_ts.replace(hour=18), state_obj)
    research_obj, signal_df = _research_decision(release_obj, state_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    _assert_intent_parity(decision_obj, research_obj, state_obj)
    assert signal_df.loc[signal_date_ts, ("S00", "ibs_value_ser")] < .10
    assert signal_df.loc[signal_date_ts, ("S00", "rsi2_value_ser")] < 90.0
    assert "S00" in decision_obj.exit_asset_set  # State, not today's exit signal, causes the exit.
    assert len(decision_obj.entry_target_weight_map_dict) == 1


@pytest.mark.parametrize("pod_str,mode_str", IDENTITY_TUPLE)
def test_future_price_perturbation_preserves_host_and_research_prefix(monkeypatch, pod_str, mode_str):
    release_obj = parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_{mode_str}_daily_moo.yaml.example")
    signal_date_ts, pricing_df, universe_df, vix_close_ser = _market_inputs("opening")
    state_obj = _case_state(release_obj, signal_date_ts, pricing_df, "opening")
    _install_inputs(monkeypatch, release_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    as_of_ts = state_obj.updated_timestamp_ts.replace(hour=18)
    original_plan_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, state_obj)
    _, prefix_signal_df = _research_decision(release_obj, state_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    future_pricing_df = pricing_df.copy()
    future_pricing_df.loc[future_pricing_df.index > signal_date_ts] *= 100.0
    _install_inputs(monkeypatch, release_obj, signal_date_ts, future_pricing_df, universe_df, vix_close_ser)
    perturbed_plan_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, state_obj)
    full_research_obj, full_signal_df = _research_decision(
        release_obj, state_obj, signal_date_ts, future_pricing_df, universe_df, vix_close_ser, full_signal_bool=True,
    )
    assert original_plan_obj == perturbed_plan_obj
    pd.testing.assert_frame_equal(prefix_signal_df, full_signal_df.loc[:signal_date_ts])
    _assert_intent_parity(perturbed_plan_obj, full_research_obj, state_obj)


def test_actual_hpi_readiness_still_rejects_insufficient_prior_history(monkeypatch):
    release_obj = parse_release_manifest("docs/live/release_templates/pod_mr_hpi_vote_vix_gated_bil_daily_moo.yaml.example")
    signal_date_ts, pricing_df, universe_df, vix_close_ser = _market_inputs("opening")
    state_obj = _case_state(release_obj, signal_date_ts, pricing_df, "opening")
    pricing_df = pricing_df.iloc[200:]  # Fewer than 1,260 observations before the decision.
    _install_inputs(monkeypatch, release_obj, signal_date_ts, pricing_df, universe_df, vix_close_ser)
    with pytest.raises(RuntimeError, match="no fully ready PIT member"):
        strategy_host.build_decision_plan_for_release(release_obj, state_obj.updated_timestamp_ts.replace(hour=18), state_obj)
