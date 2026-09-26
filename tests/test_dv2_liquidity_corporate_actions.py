"""Provider-consistent future corporate actions must not change DV2 liquidity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.dv2.strategy_mr_dv2_industry_etf import DVO2IndustryEtfStrategy
from strategies.dv2.strategy_mr_dv2_liquidity_floor import DVO2LiquidityFloorStrategy
from strategies.dv2.strategy_mr_dv2_liquidity_floor_adv_rank import DVO2LiquidityFloorAdvRankStrategy


STRATEGY_CLASS_TUPLE = (
    DVO2LiquidityFloorStrategy,
    DVO2LiquidityFloorAdvRankStrategy,
    DVO2IndustryEtfStrategy,
)


def _pricing_df() -> pd.DataFrame:
    date_idx = pd.bdate_range("2023-01-02", periods=260)
    close_vec = np.linspace(20.0, 30.0, len(date_idx))
    close_vec[-2:] = [29.0, 28.0]
    field_dict = {}
    for symbol_int, (symbol_str, turnover_float) in enumerate(
        (("AAA", 10_000_000.0), ("BBB", 60_000_000.0), ("CCC", 100_000_000.0))
    ):
        high_vec = close_vec + 1.0 + 0.25 * symbol_int
        high_vec[-2:] += 3.0
        field_dict.update(
            {
                (symbol_str, "Open"): close_vec.copy(),
                (symbol_str, "High"): high_vec,
                (symbol_str, "Low"): close_vec - 0.1,
                (symbol_str, "Close"): close_vec.copy(),
                (symbol_str, "Unadjusted Close"): close_vec.copy(),
                (symbol_str, "Volume"): turnover_float / close_vec,
                (symbol_str, "Turnover"): np.full(len(date_idx), turnover_float),
            }
        )
    return pd.DataFrame(field_dict, index=date_idx)


def _strategy_obj(strategy_class, date_idx: pd.DatetimeIndex):
    strategy_obj = strategy_class("liquidity_corporate_action_test", [], capital_base=100_000.0)
    strategy_obj.previous_bar = date_idx[-1]
    strategy_obj.universe_df = pd.DataFrame(1, index=date_idx, columns=["AAA", "BBB", "CCC"])
    return strategy_obj


@pytest.mark.parametrize("strategy_class", STRATEGY_CLASS_TUPLE)
@pytest.mark.parametrize("future_split_factor_float", [40.0, 0.1])
def test_future_split_preserves_native_adv_and_actual_candidate_selection(
    strategy_class, future_split_factor_float: float
):
    pricing_df = _pricing_df()
    revised_df = pricing_df.copy()
    # *** CRITICAL *** Provider-consistent vintage change after the last decision:
    # historical adjusted OHLC/F and Volume*F; native Turnover/raw close unchanged.
    for field_str in ("Open", "High", "Low", "Close"):
        revised_df[("AAA", field_str)] /= future_split_factor_float
    revised_df[("AAA", "Volume")] *= future_split_factor_float
    strategy_obj = _strategy_obj(strategy_class, pricing_df.index)
    baseline_signal_df = strategy_obj.compute_signals(pricing_df)
    revised_signal_df = strategy_obj.compute_signals(revised_df)
    for symbol_str in ("AAA", "BBB", "CCC"):
        pd.testing.assert_series_equal(
            baseline_signal_df[(symbol_str, "adv_63")],
            revised_signal_df[(symbol_str, "adv_63")],
        )
    baseline_symbol_list = strategy_obj.get_opportunities(baseline_signal_df.iloc[-1])
    assert "CCC" in baseline_symbol_list  # Ensure this exercises actual eligible entries.
    assert baseline_symbol_list == strategy_obj.get_opportunities(revised_signal_df.iloc[-1])


def test_native_turnover_is_required_and_volume_is_not_a_fallback():
    pricing_df = _pricing_df()
    strategy_obj = _strategy_obj(DVO2LiquidityFloorStrategy, pricing_df.index)
    with pytest.raises(ValueError, match="Turnover"):
        strategy_obj.compute_signals(pricing_df.drop(columns=[("AAA", "Turnover")]))
    signal_df = strategy_obj.compute_signals(pricing_df.drop(columns=[("AAA", "Volume")]))
    assert signal_df[("AAA", "adv_63")].iloc[-1] == 10_000_000.0


@pytest.mark.parametrize("strategy_class", STRATEGY_CLASS_TUPLE)
@pytest.mark.parametrize("invalid_value_obj", [np.nan, np.inf, -np.inf, -1.0, 0.0, "invalid"])
def test_invalid_native_turnover_blocks_that_symbol_for_the_full_window(
    strategy_class, invalid_value_obj
):
    pricing_df = _pricing_df()
    pricing_df[("CCC", "Turnover")] = pricing_df[("CCC", "Turnover")].astype(object)
    pricing_df.loc[pricing_df.index[-20], ("CCC", "Turnover")] = invalid_value_obj
    strategy_obj = _strategy_obj(strategy_class, pricing_df.index)
    signal_df = strategy_obj.compute_signals(pricing_df)
    assert pd.isna(signal_df[("CCC", "adv_63")].iloc[-1])
    assert "CCC" not in strategy_obj.get_opportunities(signal_df.iloc[-1])


@pytest.mark.parametrize("invalid_value_obj", [np.nan, np.inf, -np.inf, -1.0, 0.0, "invalid"])
def test_invalid_raw_price_cannot_pass_the_current_price_floor(invalid_value_obj):
    pricing_df = _pricing_df()
    pricing_df[("CCC", "Unadjusted Close")] = pricing_df[("CCC", "Unadjusted Close")].astype(object)
    pricing_df.loc[pricing_df.index[-1], ("CCC", "Unadjusted Close")] = invalid_value_obj
    strategy_obj = _strategy_obj(DVO2LiquidityFloorStrategy, pricing_df.index)
    signal_df = strategy_obj.compute_signals(pricing_df)
    assert "CCC" not in strategy_obj.get_opportunities(signal_df.iloc[-1])


def test_adv_uses_native_dollars_even_if_adjusted_price_times_volume_differs():
    pricing_df = _pricing_df()
    pricing_df[("CCC", "Volume")] *= 3.0
    signal_df = _strategy_obj(DVO2LiquidityFloorStrategy, pricing_df.index).compute_signals(pricing_df)
    assert signal_df[("CCC", "adv_63")].iloc[-1] == 100_000_000.0
