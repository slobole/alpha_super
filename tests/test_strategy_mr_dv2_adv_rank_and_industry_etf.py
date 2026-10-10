"""Checks for the ADV-ranked DV2 floor and the industry-ETF DV2 modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.bench import catalog
from alpha.engine import portfolio_manager
from alpha.live import release_manifest
from alpha.strategy_registry import MaturityTier, tier_for
from strategies.dv2.strategy_mr_dv2_industry_etf import (
    MIN_ADV_DOLLAR_FLOAT,
    MIN_HISTORY_SESSION_INT,
    DVO2IndustryEtfStrategy,
    build_history_universe_df,
)
from strategies.dv2.strategy_mr_dv2_liquidity_floor import DVO2LiquidityFloorStrategy
from strategies.dv2.strategy_mr_dv2_liquidity_floor_adv_rank import (
    DVO2LiquidityFloorAdvRankStrategy,
)


def _row(spec: dict[str, dict[str, float]]) -> pd.Series:
    row_dict: dict[tuple[str, str], float] = {}
    for symbol_str, field_dict in spec.items():
        base_dict = {"Open": 100.0, "Close": 100.0, "dv2": 5.0, "sma_200": 90.0, "p126d_return": 0.10,
                     "natr": 2.0, "raw_price": 50.0, "adv_63": 1e8}
        base_dict.update(field_dict)
        row_dict.update({(symbol_str, k): v for k, v in base_dict.items()})
    return pd.Series(row_dict)


def _universe(symbol_list, decision_ts):
    return pd.DataFrame({s: [1] for s in symbol_list}, index=[decision_ts - pd.Timedelta(days=1)])


def test_adv_rank_orders_by_dollar_volume_and_keeps_the_floor():
    decision_ts = pd.Timestamp("2024-03-01")
    close_row_ser = _row({
        "LOW": {"adv_63": 100.0, "natr": 9.0},     # below the member median -> excluded
        "LOW2": {"adv_63": 50.0, "dv2": 50.0},
        "MID": {"adv_63": 300.0, "natr": 1.0},
        "TOP": {"adv_63": 900.0, "natr": 2.0},
        "HIGH_NATR": {"adv_63": 700.0, "natr": 8.0},
        "NOSIG": {"adv_63": 800.0, "dv2": 50.0},   # liquid but not oversold
    })
    for cls, expected_list in ((DVO2LiquidityFloorAdvRankStrategy, ["TOP", "HIGH_NATR"]),
                               (DVO2LiquidityFloorStrategy, ["HIGH_NATR", "TOP"])):
        strategy_obj = cls(name="t", benchmarks=[], capital_base=100_000.0)
        strategy_obj.previous_bar = decision_ts
        strategy_obj.universe_df = _universe(["LOW", "LOW2", "MID", "TOP", "HIGH_NATR", "NOSIG"], decision_ts)
        # member ADV median = (300 + 700) / 2 = 500 -> only ADV > 500 pass the floor
        assert strategy_obj.get_opportunities(close_row_ser.copy()) == expected_list


def test_adv_rank_signals_and_orders_are_the_floor_modules():
    assert DVO2LiquidityFloorAdvRankStrategy.compute_signals is DVO2LiquidityFloorStrategy.compute_signals
    assert DVO2LiquidityFloorAdvRankStrategy.iterate is DVO2LiquidityFloorStrategy.iterate


def test_industry_etf_eligibility_needs_history_and_liquidity():
    decision_ts = pd.Timestamp("2024-03-01")
    strategy_obj = DVO2IndustryEtfStrategy(name="t", benchmarks=[], capital_base=100_000.0)
    strategy_obj.previous_bar = decision_ts
    strategy_obj.universe_df = _universe(["SMH", "GDX", "XPH"], decision_ts)
    close_row_ser = _row({
        "SMH": {"adv_63": MIN_ADV_DOLLAR_FLOAT * 20, "natr": 3.0},
        "GDX": {"adv_63": MIN_ADV_DOLLAR_FLOAT * 10, "natr": 4.0},
        "XPH": {"adv_63": MIN_ADV_DOLLAR_FLOAT * 0.5, "natr": 9.0},   # too thin
        "IBB": {"adv_63": MIN_ADV_DOLLAR_FLOAT * 5, "natr": 5.0},     # not yet eligible (history)
    })
    assert strategy_obj.get_opportunities(close_row_ser) == ["GDX", "SMH"]


def test_history_universe_uses_only_past_closes():
    date_idx = pd.bdate_range("2020-01-01", periods=MIN_HISTORY_SESSION_INT + 5)
    close_vec = np.arange(len(date_idx), dtype=float) + 10.0
    late_vec = close_vec.copy()
    late_vec[:10] = np.nan
    pricing_df = pd.DataFrame({("AAA", "Close"): close_vec, ("BBB", "Close"): late_vec, ("$SPX", "Close"): close_vec}, index=date_idx)
    pricing_df.columns = pd.MultiIndex.from_tuples(pricing_df.columns)
    universe_df = build_history_universe_df(pricing_df)
    assert "$SPX" not in universe_df.columns
    assert universe_df["AAA"].iloc[MIN_HISTORY_SESSION_INT - 2] == 0
    assert universe_df["AAA"].iloc[MIN_HISTORY_SESSION_INT - 1] == 1
    assert universe_df["BBB"].iloc[MIN_HISTORY_SESSION_INT - 1] == 0


def test_research_variants_have_no_live_or_portfolio_route():
    for module_str, class_str in (
        ("strategies.dv2.strategy_mr_dv2_liquidity_floor_adv_rank", "DVO2LiquidityFloorAdvRankStrategy"),
        ("strategies.dv2.strategy_mr_dv2_industry_etf", "DVO2IndustryEtfStrategy"),
    ):
        entry_obj = catalog.get_strategy_by_module(module_str)
        assert entry_obj is not None
        assert entry_obj.maturity_key_str == "research"
        assert not entry_obj.is_wired_bool
        assert entry_obj.has_run_variant_bool
        assert entry_obj.has_capacity_analysis_bool
        assert entry_obj.has_timing_analysis_bool
        import_str = f"{module_str}:{class_str}"
        assert tier_for(import_str) is MaturityTier.RESEARCH
        assert import_str not in release_manifest.SUPPORTED_STRATEGY_IMPORT_TUPLE
        assert import_str not in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE
