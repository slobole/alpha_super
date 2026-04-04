"""
Monthly ATR-adjusted momentum rotation for point-in-time Nasdaq 100 members.

Core formulas
-------------
For stock i on month-end decision date t:

    monthly_roc_{i,t}^{(L)}
        = Close_ME_{i,t} / Close_ME_{i,t-L} - 1

    ATR20_{i,t}
        = mean(TR_{i,t-19:t})

    risk_adj_score_{i,t}
        = monthly_roc_{i,t}^{(L)} / ATR20_{i,t}

    selected_t
        = top max_positions eligible symbols by risk_adj_score_{i,t}

    target_weight_{i,t}
        = 1 / max_positions    if i in selected_t
        = 0                    otherwise
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_radge_ndx import (
    ATR_WINDOW_INT,
    RadgeMomentumNdxConfig,
    RadgeMomentumNdxStrategy,
    audit_pit_universe_df,
    compute_radge_signal_tables,
    get_monthly_decision_close_df,
    get_radge_momentum_ndx_data,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)

__all__ = [
    "ATR_WINDOW_INT",
    "AtrNormalizedNdxConfig",
    "AtrNormalizedNdxStrategy",
    "DEFAULT_CONFIG",
    "audit_pit_universe_df",
    "compute_radge_signal_tables",
    "get_atr_normalized_ndx_data",
    "get_monthly_decision_close_df",
    "map_month_end_decision_dates_to_rebalance_schedule_df",
]


@dataclass(frozen=True)
class AtrNormalizedNdxConfig(RadgeMomentumNdxConfig):
    indexname_str: str = "Nasdaq 100"


DEFAULT_CONFIG = AtrNormalizedNdxConfig()


class AtrNormalizedNdxStrategy(RadgeMomentumNdxStrategy):
    """Canonical NDX preset for the ATR-adjusted monthly rotation."""


def get_atr_normalized_ndx_data(
    config: AtrNormalizedNdxConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return get_radge_momentum_ndx_data(config=config)


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(config)

    strategy = AtrNormalizedNdxStrategy(
        name="strategy_mo_atr_normalized_ndx",
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config.backtest_start_date_str)]
    run_daily(strategy, pricing_data_df, calendar=calendar_idx, audit_override_bool=None)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
