"""
Score-2 SPY/IEF variant of the Financial Hacker market-regime filter.

Only the exposure map changes from the baseline:

    score_t = 3  -> 100% SPY
    score_t = 2  ->  50% SPY + 50% IEF
    score_t < 2  -> cash

The detector math, data sources, costs, and next-open execution timing are
inherited unchanged from `strategy_taa_market_regime_filter`.
"""

from __future__ import annotations

from strategies.taa_traditional.strategy_taa_market_regime_filter import (
    DEFAULT_CONFIG,
    MarketRegimeFilterConfig,
    MarketRegimeFilterStrategy,
    run_variant as run_market_regime_variant,
)


SCORE2_SPY_IEF_CONFIG = MarketRegimeFilterConfig(
    **{
        **DEFAULT_CONFIG.__dict__,
        "strategy_name_str": "strategy_taa_market_regime_filter_score2_spy_ief",
        "caution_defensive_trade_symbol_str": "IEF",
        "caution_defensive_target_weight_float": 0.5,
    }
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> MarketRegimeFilterStrategy:
    return run_market_regime_variant(
        config=SCORE2_SPY_IEF_CONFIG,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )


if __name__ == "__main__":
    run_variant()
