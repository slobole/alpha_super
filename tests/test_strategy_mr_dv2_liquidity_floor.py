"""Focused checks for the standalone DV2 liquidity-floor strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog
from alpha.engine import portfolio_manager
from alpha.engine.strategy import Strategy
from alpha.live import release_manifest
from alpha.strategy_registry import MaturityTier, tier_for
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy
from strategies.dv2.strategy_mr_dv2_liquidity_floor import (
    DVO2LiquidityFloorStrategy,
    get_asof_universe_symbol_list,
)


def _strategy_obj() -> DVO2LiquidityFloorStrategy:
    return DVO2LiquidityFloorStrategy(
        name="dv2_liquidity_test", benchmarks=[], capital_base=100_000.0
    )


def test_standalone_strategy_uses_only_information_through_close_t():
    assert DVO2LiquidityFloorStrategy.__bases__ == (Strategy,)

    date_idx = pd.bdate_range("2023-01-02", periods=260)
    close_vec = np.linspace(100.0, 150.0, len(date_idx))
    pricing_df = pd.DataFrame(
        {
            ("AAA", "Close"): close_vec,
            ("AAA", "High"): close_vec + 1.0,
            ("AAA", "Low"): close_vec - 1.0,
            ("AAA", "Unadjusted Close"): close_vec,
            ("AAA", "Volume"): np.full(len(date_idx), 1_000_000.0),
            ("AAA", "Turnover"): close_vec * 1_000_000.0,
        },
        index=date_idx,
    )
    pricing_df.columns = pd.MultiIndex.from_tuples(pricing_df.columns)

    signal_df = _strategy_obj().compute_signals(pricing_df)
    decision_ts = date_idx[-2]
    expected_adv_float = float(close_vec[-64:-1].mean() * 1_000_000.0)
    assert signal_df.loc[decision_ts, ("AAA", "adv_63")] == pytest.approx(
        expected_adv_float
    )
    assert pd.isna(signal_df.loc[date_idx[61], ("AAA", "adv_63")])

    changed_pricing_df = pricing_df.copy()
    changed_pricing_df.loc[date_idx[-1], ("AAA", "Turnover")] = 10_000_000_000.0
    changed_signal_df = _strategy_obj().compute_signals(changed_pricing_df)
    for feature_str in ("adv_63", "p126d_return", "natr", "dv2", "sma_200"):
        assert changed_signal_df.loc[decision_ts, ("AAA", feature_str)] == pytest.approx(
            signal_df.loc[decision_ts, ("AAA", feature_str)]
        )


def test_liquidity_median_uses_all_pit_members_before_signal_filtering():
    decision_ts = pd.Timestamp("2024-02-01")
    strategy_obj = _strategy_obj()
    strategy_obj.previous_bar = decision_ts
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1, 0], "BBB": [1, 0], "CCC": [1, 0], "DDD": [1, 0], "ZZZ": [0, 1]},
        index=[decision_ts - pd.Timedelta(days=1), decision_ts + pd.Timedelta(days=1)],
    )

    row_dict: dict[tuple[str, str], float] = {}
    for symbol_str, adv_float, raw_price_float in (
        ("AAA", 100.0, 10.0),
        ("BBB", 200.0, 10.0),
        ("CCC", 300.0, 10.0),
        ("DDD", 400.0, 4.0),
        ("ZZZ", 10_000.0, 10.0),
    ):
        row_dict.update(
            {
                (symbol_str, "Open"): 100.0,
                (symbol_str, "Close"): 100.0,
                (symbol_str, "dv2"): float("nan") if symbol_str == "AAA" else 5.0,
                (symbol_str, "sma_200"): 90.0,
                (symbol_str, "p126d_return"): 0.10,
                (symbol_str, "natr"): 10.0,
                (symbol_str, "raw_price"): raw_price_float,
                (symbol_str, "adv_63"): adv_float,
            }
        )
    close_row_ser = pd.Series(row_dict)

    assert get_asof_universe_symbol_list(strategy_obj.universe_df, decision_ts) == [
        "AAA", "BBB", "CCC", "DDD"
    ]
    # PIT median is 250 despite AAA's missing DV2 and DDD's raw price <= $5.
    assert strategy_obj.get_opportunities(close_row_ser) == ["CCC"]
    # DV2 requires a complete trading row, even when that field is not an
    # explicit entry filter.
    close_row_ser[("CCC", "Open")] = np.nan
    assert strategy_obj.get_opportunities(close_row_ser) == []


def test_missing_liquidity_source_field_fails_closed():
    date_idx = pd.bdate_range("2024-01-02", periods=2)
    pricing_df = pd.DataFrame(
        {
            ("AAA", "Close"): [100.0, 101.0],
            ("AAA", "High"): [101.0, 102.0],
            ("AAA", "Low"): [99.0, 100.0],
            ("AAA", "Unadjusted Close"): [100.0, 101.0],
        },
        index=date_idx,
    )
    pricing_df.columns = pd.MultiIndex.from_tuples(pricing_df.columns)
    with pytest.raises(ValueError, match="Turnover"):
        _strategy_obj().compute_signals(pricing_df)


def test_missing_turnover_invalidates_the_trailing_adv_window():
    date_idx = pd.bdate_range("2024-01-02", periods=85)
    close_vec = np.full(len(date_idx), 100.0)
    turnover_vec = np.full(len(date_idx), 100_000_000.0)
    turnover_vec[20] = np.nan
    pricing_df = pd.DataFrame(
        {
            ("AAA", "Close"): close_vec,
            ("AAA", "High"): close_vec + 1.0,
            ("AAA", "Low"): close_vec - 1.0,
            ("AAA", "Unadjusted Close"): close_vec,
            ("AAA", "Turnover"): turnover_vec,
        },
        index=date_idx,
    )
    pricing_df.columns = pd.MultiIndex.from_tuples(pricing_df.columns)

    signal_df = _strategy_obj().compute_signals(pricing_df)
    assert pd.isna(signal_df.loc[date_idx[82], ("AAA", "adv_63")])
    assert signal_df.loc[date_idx[83], ("AAA", "adv_63")] == 100_000_000.0


def test_research_entry_does_not_open_live_or_portfolio_routes():
    strategy_import_str = (
        "strategies.dv2.strategy_mr_dv2_liquidity_floor:DVO2LiquidityFloorStrategy"
    )
    strategy_entry_obj = catalog.get_strategy_by_module(
        "strategies.dv2.strategy_mr_dv2_liquidity_floor"
    )
    assert strategy_entry_obj is not None
    assert not strategy_entry_obj.is_wired_bool
    assert not strategy_entry_obj.is_pm_ready_bool
    assert strategy_entry_obj.has_run_variant_bool
    assert strategy_entry_obj.has_capacity_analysis_bool
    assert strategy_entry_obj.has_timing_analysis_bool
    assert tier_for(strategy_import_str) is MaturityTier.RESEARCH
    assert strategy_import_str not in release_manifest.SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert strategy_import_str not in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE


def test_standalone_order_intent_matches_dv2_for_the_same_candidates():
    order_log_list: list[list[tuple[str, str, float, int]]] = []
    for strategy_obj in (
        DVO2Strategy(name="dv2_reference", benchmarks=[], capital_base=100_000.0),
        _strategy_obj(),
    ):
        strategy_obj.max_positions = 2
        strategy_obj.trade_id = 7
        strategy_obj.current_trade = {"OLD": 7}
        strategy_obj._position_amount_map = {"OLD": 10.0}
        strategy_obj._total_value_history_list = [100_000.0]
        strategy_obj.get_opportunities = lambda close_row_ser: ["NEW"]
        local_order_list: list[tuple[str, str, float, int]] = []
        strategy_obj.order_target_value = lambda symbol_str, value_float, trade_id: local_order_list.append(
            ("target", symbol_str, float(value_float), trade_id)
        )
        strategy_obj.order_value = lambda symbol_str, value_float, trade_id: local_order_list.append(
            ("value", symbol_str, float(value_float), trade_id)
        )

        data_df = pd.DataFrame({("OLD", "High"): [98.0, 99.0]})
        close_row_ser = pd.Series({("OLD", "Close"): 101.0})
        strategy_obj.iterate(data_df, close_row_ser, pd.Series(dtype=float))
        order_log_list.append(local_order_list)

    assert order_log_list == [
        [("target", "OLD", 0.0, 7), ("value", "NEW", 50_000.0, 8)]
    ] * 2
