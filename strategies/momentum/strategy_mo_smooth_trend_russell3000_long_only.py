"""
Research-only Russell 3000 smooth-trend long-only variant.

This removes the short side from Option A while preserving the same data,
signal, price filter, and next-open execution semantics:

    eligible_t
        = active PIT members with Close_t > 1.00

    RQ5_t
        = eligible_t names in the highest trend_r2 quintile

    SQ5_t
        = highest slope quintile inside RQ5_t

    long_t
        = top n names from RQ5/SQ5 by slope
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_smooth_trend_russell3000_long_short import (
    DEFAULT_CONFIG as LONG_SHORT_DEFAULT_CONFIG,
    SmoothTrendRussell3000LongShortConfig,
    SmoothTrendRussell3000LongShortStrategy,
    get_smooth_trend_russell3000_long_short_data,
)


DEFAULT_CONFIG = replace(
    LONG_SHORT_DEFAULT_CONFIG,
    variant_key_str="russell3000_option_a_n20_long_only_close_gt_1",
    max_short_positions_int=0,
    short_gross_exposure_float=0.0,
)


__all__ = [
    "DEFAULT_CONFIG",
    "SmoothTrendRussell3000LongOnlyStrategy",
    "run_variant",
]


class SmoothTrendRussell3000LongOnlyStrategy(SmoothTrendRussell3000LongShortStrategy):
    """
    Research-only monthly Russell 3000 Option A long-only selector.

    The strategy keeps only the high-R2/high-slope corner and targets 100% long
    gross exposure across the selected names.
    """


def _write_assumptions_md(
    output_path: Path,
    strategy: SmoothTrendRussell3000LongOnlyStrategy,
) -> None:
    config = strategy.config
    assumption_md_str = f"""# Smooth Trend Russell 3000 Long-Only Assumptions

- Research-only strategy; no live/release wiring.
- Universe: `{config.indexname_str}` point-in-time membership through Norgate.
- Benchmark list: `{list(config.benchmark_list)}`.
- Stock price basis: Norgate `CAPITALSPECIAL` OHLC loaded through repo `load_raw_prices`.
- Decision cadence: actual last tradable close of each month.
- Execution cadence: next tradable open after the decision close under the Vanilla engine.
- Formation window: `Close_(t-{config.lookback_trading_day_int})` through `Close_(t-{config.skip_trading_day_int})`.
- Signal math: regress cumulative simple returns `S_k = sum(r_1..r_k)` on time `k`; use slope as direction and unsigned `R2` as smoothness.
- Price eligibility: stock `Close_T > {config.minimum_close_price_float:.2f}` at the decision close, using the same `CAPITALSPECIAL` price basis.
- Sort: `R2` quintile first, then slope quintile within the selected `R2` bucket.
- Long leg: top `{config.max_long_positions_int}` names from RQ5/SQ5 by slope.
- Short leg: disabled.
- Long gross exposure: `{config.long_gross_exposure_float:.4f}`.
- Short gross exposure: `{config.short_gross_exposure_float:.4f}`.
- Slippage: `{config.slippage_float:.6f}` per side.
- Commission: `{config.commission_per_share_float:.6f}` per share, minimum `{config.commission_minimum_float:.2f}`.
"""
    (output_path / "smooth_trend_russell3000_long_only_assumptions.md").write_text(
        assumption_md_str,
        encoding="utf-8",
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = False,
) -> SmoothTrendRussell3000LongOnlyStrategy:
    config_obj: SmoothTrendRussell3000LongShortConfig = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
    ):
        config_obj = replace(
            DEFAULT_CONFIG,
            backtest_start_date_str=(
                DEFAULT_CONFIG.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                DEFAULT_CONFIG.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    (
        pricing_data_df,
        universe_df,
        rebalance_schedule_df,
        trend_slope_df,
        trend_r2_df,
    ) = get_smooth_trend_russell3000_long_short_data(config=config_obj)
    strategy_obj = SmoothTrendRussell3000LongOnlyStrategy(
        name="strategy_mo_smooth_trend_russell3000_long_only",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
        precomputed_trend_slope_df=trend_slope_df,
        precomputed_trend_r2_df=trend_r2_df,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep full pre-start history for the skipped OLS
    # formation window, but execute/report only from backtest_start_date_str.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        display(strategy_obj.rebalance_selection_df.tail(40))

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.rebalance_selection_df.to_csv(
            output_path / "rebalance_selection.csv",
            index=False,
        )
        _write_assumptions_md(output_path=output_path, strategy=strategy_obj)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
