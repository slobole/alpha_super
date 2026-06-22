"""
Research-only IWC gapper close-to-open strategy.

TL;DR: This is the single-ETF IWC variant of the GAPPERS close-to-open rule.
It reuses the existing GAPPERS execution harness but removes the Russell 3000
universe, top-N ranking, and low-price stock filter. The rule is:

    gap_z_t = (Open_t / Close_{t-1} - 1) / std(overnight_gap_{t-252:t-1})

    trend_pass_t = 1[Close_t > SMA200_t]

There is no mean subtraction. If gap_z_t > 2 and trend_pass_t is true, buy
IWC at Close_t and sell IWC at Open_{t+1}.

This is research-only. Same-day close entry is a daily-data MOC approximation
and is not wired to live execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.report import save_results
from data.norgate_loader import load_raw_prices
from strategies.momentum.strategy_mo_gappers_russell3000_close_to_open import (
    GAP_VOL_LOOKBACK_DAY_INT,
    GAP_Z_MIN_FLOAT,
    GappersRussell3000CloseToOpenStrategy,
    GappersRussell3000Config,
    run_gappers_close_to_open_backtest,
)


TREND_SMA_DAY_INT = 200


@dataclass(frozen=True)
class IwcGappersCloseToOpenConfig:
    symbol_str: str = "IWC"
    benchmark_symbol_str: str = "SPY"
    history_start_date_str: str = "2000-01-01"
    backtest_start_date_str: str = "2002-01-01"
    end_date_str: str | None = None
    gap_vol_lookback_day_int: int = GAP_VOL_LOOKBACK_DAY_INT
    gap_z_min_float: float = GAP_Z_MIN_FLOAT
    trend_sma_day_int: int = TREND_SMA_DAY_INT
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol_str:
            raise ValueError("symbol_str must not be empty.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if self.symbol_str == self.benchmark_symbol_str:
            raise ValueError("symbol_str and benchmark_symbol_str must differ.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.gap_vol_lookback_day_int <= 1:
            raise ValueError("gap_vol_lookback_day_int must be greater than 1.")
        if self.gap_z_min_float <= 0.0:
            raise ValueError("gap_z_min_float must be positive.")
        if self.trend_sma_day_int <= 1:
            raise ValueError("trend_sma_day_int must be greater than 1.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = IwcGappersCloseToOpenConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "IwcGappersCloseToOpenConfig",
    "IwcGappersCloseToOpenStrategy",
    "TREND_SMA_DAY_INT",
    "build_base_gappers_config",
    "get_gappers_iwc_data",
    "run_variant",
]


def build_base_gappers_config(
    config_obj: IwcGappersCloseToOpenConfig = DEFAULT_CONFIG,
) -> GappersRussell3000Config:
    """
    Adapt the single-ETF rule to the existing GAPPERS execution harness.

    IWC is not a low-price stock selection strategy, so the inherited price
    filter is widened to any positive tradable close.
    """
    return GappersRussell3000Config(
        indexname_str="Single ETF",
        benchmark_symbol_str=config_obj.benchmark_symbol_str,
        history_start_date_str=config_obj.history_start_date_str,
        backtest_start_date_str=config_obj.backtest_start_date_str,
        end_date_str=config_obj.end_date_str,
        gap_vol_lookback_day_int=config_obj.gap_vol_lookback_day_int,
        gap_z_min_float=config_obj.gap_z_min_float,
        min_entry_price_float=0.01,
        max_entry_price_float=1_000_000.0,
        max_positions_int=1,
        capital_base_float=config_obj.capital_base_float,
        slippage_float=config_obj.slippage_float,
        commission_per_share_float=config_obj.commission_per_share_float,
        commission_minimum_float=config_obj.commission_minimum_float,
    )


class IwcGappersCloseToOpenStrategy(GappersRussell3000CloseToOpenStrategy):
    """
    Single-ETF IWC GAPPERS wrapper around the close-to-open research harness.
    """

    def __init__(
        self,
        name: str,
        config_obj: IwcGappersCloseToOpenConfig = DEFAULT_CONFIG,
    ):
        self.iwc_config = config_obj
        super().__init__(
            name=name,
            benchmarks=[config_obj.benchmark_symbol_str],
            config=build_base_gappers_config(config_obj=config_obj),
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        symbol_str = self.iwc_config.symbol_str
        close_price_ser = signal_data_df[(symbol_str, "Close")].astype(float)

        # *** CRITICAL *** rolling-window timing: the 200-day trend filter is
        # evaluated at Close_T because this research rule also enters at
        # Close_T. This remains a same-close/MOC approximation, not a signal
        # that can be known before the live auction cutoff without a separate
        # pre-close snapshot.
        trend_sma_ser = close_price_ser.rolling(
            window=int(self.iwc_config.trend_sma_day_int),
            min_periods=int(self.iwc_config.trend_sma_day_int),
        ).mean()
        trend_pass_ser = close_price_ser > trend_sma_ser

        trend_pass_field_str = f"above_sma_{self.iwc_config.trend_sma_day_int}_bool"
        signal_data_df[(symbol_str, f"sma_{self.iwc_config.trend_sma_day_int}_ser")] = trend_sma_ser
        signal_data_df[(symbol_str, trend_pass_field_str)] = trend_pass_ser.fillna(False).astype(bool)
        signal_data_df[(symbol_str, "price_filter_pass_bool")] = (
            signal_data_df[(symbol_str, "price_filter_pass_bool")].fillna(False).astype(bool)
            & signal_data_df[(symbol_str, trend_pass_field_str)]
        )
        return signal_data_df


def get_gappers_iwc_data(
    config_obj: IwcGappersCloseToOpenConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pricing_data_df = load_raw_prices(
        symbols=[config_obj.symbol_str],
        benchmarks=[config_obj.benchmark_symbol_str],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    required_column_list = [
        (config_obj.symbol_str, "Open"),
        (config_obj.symbol_str, "Close"),
        (config_obj.benchmark_symbol_str, "Close"),
    ]
    missing_column_list = [
        column_tuple for column_tuple in required_column_list if column_tuple not in pricing_data_df.columns
    ]
    if len(missing_column_list) > 0:
        raise RuntimeError(f"Missing required IWC GAPPERS data columns: {missing_column_list}")

    open_price_ser = pricing_data_df[(config_obj.symbol_str, "Open")].astype(float)
    close_price_ser = pricing_data_df[(config_obj.symbol_str, "Close")].astype(float)
    tradable_bool_ser = (
        open_price_ser.replace([np.inf, -np.inf], np.nan).notna()
        & close_price_ser.replace([np.inf, -np.inf], np.nan).notna()
        & open_price_ser.gt(0.0)
        & close_price_ser.gt(0.0)
    )
    universe_df = pd.DataFrame(
        {config_obj.symbol_str: tradable_bool_ser.astype(int)},
        index=pricing_data_df.index,
    )
    return pricing_data_df.sort_index(), universe_df.sort_index()


def _write_assumptions_md(
    output_path: Path,
    strategy_obj: IwcGappersCloseToOpenStrategy,
) -> None:
    config_obj = strategy_obj.iwc_config
    assumption_md_str = f"""# IWC GAPPERS Assumptions

- Research-only strategy; no live/release wiring.
- Instrument: `{config_obj.symbol_str}`.
- Benchmark: `{config_obj.benchmark_symbol_str}`.
- Signal: `gap_z = (Open_t / Close_(t-1) - 1) / std_{config_obj.gap_vol_lookback_day_int}(overnight_gap ending at t-1)`.
- No mean subtraction is used in `gap_z`.
- Entry filter: `gap_z > {config_obj.gap_z_min_float:.4f}`.
- Trend filter: `Close_t > SMA_{config_obj.trend_sma_day_int}(Close_t)`, using a trailing simple moving average through the same close `t`.
- No Russell universe, top-N cross-sectional ranking, or low-price stock filter is used.
- Entry fill: `Close_t * (1 + slippage)`.
- Exit fill: `Open_(t+1) * (1 - slippage)`.
- Slippage: `{config_obj.slippage_float:.6f}` per side.
- Commission: `{config_obj.commission_per_share_float:.6f}` per share, minimum `{config_obj.commission_minimum_float:.2f}`.
- Missing next-open exits are fatal because the rule explicitly requires `Open_(t+1)`.
- Same-close entry is a daily-data MOC approximation and is not live-clean.
"""
    (output_path / "iwc_gappers_assumptions.md").write_text(
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
    pricing_data_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    audit_override_bool: bool | None = None,
) -> IwcGappersCloseToOpenStrategy:
    config_obj = DEFAULT_CONFIG
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

    if pricing_data_df is None or universe_df is None:
        pricing_data_df, universe_df = get_gappers_iwc_data(config_obj=config_obj)

    strategy_obj = IwcGappersCloseToOpenStrategy(
        name="strategy_mo_gappers_iwc_close_to_open",
        config_obj=config_obj,
    )
    strategy_obj.universe_df = universe_df

    signal_data_df = strategy_obj.compute_signals(pricing_data_df)
    if audit_override_bool is None or audit_override_bool:
        strategy_obj.audit_signals(pricing_data_df, signal_data_df)

    run_gappers_close_to_open_backtest(
        strategy=strategy_obj,
        pricing_data_df=pricing_data_df,
        signal_data_df=signal_data_df,
        backtest_start_date_str=config_obj.backtest_start_date_str,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        display(strategy_obj.daily_selection_df.tail(20))

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.daily_selection_df.to_csv(output_path / "daily_selection.csv", index=False)
        strategy_obj.daily_target_weights.to_csv(output_path / "daily_target_weights.csv")
        _write_assumptions_md(output_path=output_path, strategy_obj=strategy_obj)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
