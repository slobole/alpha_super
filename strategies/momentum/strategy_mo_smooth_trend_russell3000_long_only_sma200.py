"""
Research-only Russell 3000 smooth-trend long-only SMA200 regime variant.

This keeps the N=20 long-only Option A construction, but allows new long
targets only when the benchmark index is above its 200-day simple moving
average at the month-end decision close.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_smooth_trend_russell3000_long_only import (
    DEFAULT_CONFIG as LONG_ONLY_DEFAULT_CONFIG,
    SmoothTrendRussell3000LongOnlyStrategy,
)
from strategies.momentum.strategy_mo_smooth_trend_russell3000_long_short import (
    SmoothTrendRussell3000LongShortConfig,
    get_smooth_trend_russell3000_long_short_data,
)


REGIME_BENCHMARK_SYMBOL_STR = "$RUA"
REGIME_SMA_DAY_INT = 200
REGIME_SMA_FIELD_STR = "regime_sma_200_ser"

DEFAULT_CONFIG = replace(
    LONG_ONLY_DEFAULT_CONFIG,
    variant_key_str="russell3000_option_a_n20_long_only_close_gt_1_rua_above_sma200",
)


__all__ = [
    "DEFAULT_CONFIG",
    "REGIME_BENCHMARK_SYMBOL_STR",
    "REGIME_SMA_DAY_INT",
    "REGIME_SMA_FIELD_STR",
    "SmoothTrendRussell3000LongOnlySMA200Strategy",
    "run_variant",
]


class SmoothTrendRussell3000LongOnlySMA200Strategy(SmoothTrendRussell3000LongOnlyStrategy):
    """
    Research-only monthly Russell 3000 long-only selector with SMA200 gate.
    """

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data=pricing_data)
        if (REGIME_BENCHMARK_SYMBOL_STR, "Close") not in signal_data_df.columns:
            raise RuntimeError(f"Missing regime benchmark close for {REGIME_BENCHMARK_SYMBOL_STR}.")

        benchmark_close_ser = signal_data_df[(REGIME_BENCHMARK_SYMBOL_STR, "Close")].astype(float)
        # *** CRITICAL *** lookahead-sensitive regime filter: the SMA200 value
        # is sampled only at the month-end decision close T and executed at the
        # next tradable open. Do not use current_bar open or post-T data here.
        benchmark_sma_ser = benchmark_close_ser.rolling(
            REGIME_SMA_DAY_INT,
            min_periods=REGIME_SMA_DAY_INT,
        ).mean()
        signal_data_df[(REGIME_BENCHMARK_SYMBOL_STR, REGIME_SMA_FIELD_STR)] = benchmark_sma_ser
        return signal_data_df

    def get_selection_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        candidate_feature_df = close_row_ser.unstack()
        if REGIME_BENCHMARK_SYMBOL_STR not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime benchmark row for {REGIME_BENCHMARK_SYMBOL_STR}.")
        if REGIME_SMA_FIELD_STR not in candidate_feature_df.columns:
            raise RuntimeError(f"Missing {REGIME_SMA_FIELD_STR} for {REGIME_BENCHMARK_SYMBOL_STR}.")

        benchmark_close_float = float(candidate_feature_df.loc[REGIME_BENCHMARK_SYMBOL_STR, "Close"])
        benchmark_sma_float = float(candidate_feature_df.loc[REGIME_BENCHMARK_SYMBOL_STR, REGIME_SMA_FIELD_STR])
        # *** CRITICAL *** regime gate uses the same previous_bar decision
        # close as the stock selection. If this is false, the strategy targets
        # cash at the next open instead of holding stale longs.
        bull_market_bool = (
            np.isfinite(benchmark_close_float)
            and np.isfinite(benchmark_sma_float)
            and benchmark_close_float > benchmark_sma_float
        )
        if not bull_market_bool:
            return pd.DataFrame()

        selection_df = super().get_selection_df(close_row_ser=close_row_ser)
        if len(selection_df) > 0:
            selection_df = selection_df.copy()
            selection_df["regime_benchmark_symbol_str"] = REGIME_BENCHMARK_SYMBOL_STR
            selection_df["regime_benchmark_close_float"] = benchmark_close_float
            selection_df["regime_benchmark_sma_float"] = benchmark_sma_float
            selection_df["regime_bull_market_bool"] = True
        return selection_df


def _write_assumptions_md(
    output_path: Path,
    strategy_obj: SmoothTrendRussell3000LongOnlySMA200Strategy,
) -> None:
    config_obj = strategy_obj.config
    assumption_md_str = f"""# Smooth Trend Russell 3000 Long-Only SMA200 Assumptions

- Research-only strategy; no live/release wiring.
- Universe: `{config_obj.indexname_str}` point-in-time membership through Norgate.
- Benchmark list: `{list(config_obj.benchmark_list)}`.
- Stock price basis: Norgate `CAPITALSPECIAL` OHLC loaded through repo `load_raw_prices`.
- Decision cadence: actual last tradable close of each month.
- Execution cadence: next tradable open after the decision close under the Vanilla engine.
- Formation window: `Close_(t-{config_obj.lookback_trading_day_int})` through `Close_(t-{config_obj.skip_trading_day_int})`.
- Signal math: regress cumulative simple returns `S_k = sum(r_1..r_k)` on time `k`; use slope as direction and unsigned `R2` as smoothness.
- Price eligibility: stock `Close_T > {config_obj.minimum_close_price_float:.2f}` at the decision close, using the same `CAPITALSPECIAL` price basis.
- Sort: `R2` quintile first, then slope quintile within the selected `R2` bucket.
- Long leg: top `{config_obj.max_long_positions_int}` names from RQ5/SQ5 by slope.
- Short leg: disabled.
- Regime gate: only hold longs when `{REGIME_BENCHMARK_SYMBOL_STR} Close_T > SMA{REGIME_SMA_DAY_INT}_T` at the same decision close.
- If the regime gate is false or the SMA is unavailable, target cash at the next tradable open.
- Long gross exposure when invested: `{config_obj.long_gross_exposure_float:.4f}`.
- Slippage: `{config_obj.slippage_float:.6f}` per side.
- Commission: `{config_obj.commission_per_share_float:.6f}` per share, minimum `{config_obj.commission_minimum_float:.2f}`.
"""
    (output_path / "smooth_trend_russell3000_long_only_sma200_assumptions.md").write_text(
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
) -> SmoothTrendRussell3000LongOnlySMA200Strategy:
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
    strategy_obj = SmoothTrendRussell3000LongOnlySMA200Strategy(
        name="strategy_mo_smooth_trend_russell3000_long_only_sma200",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
        precomputed_trend_slope_df=trend_slope_df,
        precomputed_trend_r2_df=trend_r2_df,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL *** Keep full pre-start history for the skipped OLS
    # formation window and benchmark SMA200, but execute/report only from
    # backtest_start_date_str.
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
        _write_assumptions_md(output_path=output_path, strategy_obj=strategy_obj)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
