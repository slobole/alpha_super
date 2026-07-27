"""
Research-only mean-reversion strategy for the 11 US sector ETFs.

For ETF i on completed decision bar T:

    IBS_i,T = (Close_i,T - Low_i,T) / (High_i,T - Low_i,T)

    DownShockATR_i,T
        = (Close_i,T / Close_i,T-1 - 1)
        / (ATR14_i,T-1 / Close_i,T-1)

    RangeRatio_i,T
        = ln(High_i,T / Low_i,T)
        / Median(ln(High_i / Low_i) from T-21 through T-1)

Enter when IBS < 0.05 and DownShockATR < -0.5. Rank qualifying
candidates by prior-day NATR14, highest first. Exit when IBS > 0.90 and
RangeRatio > 1. Process exits before entries, hold at most five positions,
and size each new position to 1.5 / 11 of prior AUM without rebalancing
existing positions.

The supplied 15:45/MOC rule is deliberately not simulated. The executable
daily-bar fallback is:

    completed Close_T signal -> market order filled at Open_(T+1)

This module is research-only and has no LIVE or release wiring.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import ibs_indicator
from data.norgate_loader import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    load_raw_prices,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    resolve_effective_backtest_start_date_str,
    resolve_full_basket_calendar_idx,
    resolve_history_start_date_str,
)


STRATEGY_NAME_STR = "strategy_mr_us_sector_etf_ibs_downshock"
SECTOR_ETF_SYMBOL_TUPLE = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
    "XLC",
)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class UsSectorEtfIbsDownshockConfig:
    symbol_tuple: tuple[str, ...] = SECTOR_ETF_SYMBOL_TUPLE
    benchmark_symbol_str: str = "$SPX"
    history_start_date_str: str = "2017-01-01"
    backtest_start_date_str: str = "2018-01-01"
    end_date_str: str | None = None
    entry_ibs_max_float: float = 0.05
    downshock_atr_max_float: float = -0.5
    exit_ibs_min_float: float = 0.90
    atr_lookback_day_int: int = 14
    range_median_lookback_day_int: int = 21
    max_positions_int: int = 5
    sizing_multiplier_float: float = 1.5
    sizing_universe_count_int: int = 11
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if len(self.symbol_tuple) == 0:
            raise ValueError("symbol_tuple must not be empty.")
        if len(set(self.symbol_tuple)) != len(self.symbol_tuple):
            raise ValueError("symbol_tuple must not contain duplicates.")
        if self.benchmark_symbol_str in self.symbol_tuple:
            raise ValueError("benchmark_symbol_str must differ from tradable symbols.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(
            self.backtest_start_date_str
        ):
            raise ValueError(
                "history_start_date_str must be earlier than backtest_start_date_str."
            )
        if not 0.0 <= self.entry_ibs_max_float <= 1.0:
            raise ValueError("entry_ibs_max_float must lie in [0, 1].")
        if not 0.0 <= self.exit_ibs_min_float <= 1.0:
            raise ValueError("exit_ibs_min_float must lie in [0, 1].")
        if self.entry_ibs_max_float >= self.exit_ibs_min_float:
            raise ValueError("entry_ibs_max_float must be below exit_ibs_min_float.")
        if self.downshock_atr_max_float >= 0.0:
            raise ValueError("downshock_atr_max_float must be negative.")
        if self.atr_lookback_day_int <= 1:
            raise ValueError("atr_lookback_day_int must be greater than 1.")
        if self.range_median_lookback_day_int <= 1:
            raise ValueError("range_median_lookback_day_int must be greater than 1.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.max_positions_int > len(self.symbol_tuple):
            raise ValueError("max_positions_int must not exceed the universe size.")
        if self.sizing_multiplier_float <= 0.0:
            raise ValueError("sizing_multiplier_float must be positive.")
        if self.sizing_universe_count_int <= 0:
            raise ValueError("sizing_universe_count_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")


DEFAULT_CONFIG = UsSectorEtfIbsDownshockConfig()


def _symbol_field_df(
    pricing_data_df: pd.DataFrame,
    symbol_tuple: tuple[str, ...],
    field_str: str,
) -> pd.DataFrame:
    missing_column_list = [
        (symbol_str, field_str)
        for symbol_str in symbol_tuple
        if (symbol_str, field_str) not in pricing_data_df.columns
    ]
    if missing_column_list:
        raise RuntimeError(
            f"Missing required US-sector ETF columns: {missing_column_list}"
        )

    return pd.DataFrame(
        {
            symbol_str: pd.to_numeric(
                pricing_data_df[(symbol_str, field_str)],
                errors="coerce",
            )
            for symbol_str in symbol_tuple
        },
        index=pricing_data_df.index,
        dtype=float,
    )


def compute_us_sector_etf_ibs_downshock_signal_df(
    pricing_data_df: pd.DataFrame,
    config_obj: UsSectorEtfIbsDownshockConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add the causal daily-fallback entry, exit, and ranking fields."""
    signal_data_df = pricing_data_df.copy()
    symbol_tuple = tuple(config_obj.symbol_tuple)
    close_price_df = _symbol_field_df(signal_data_df, symbol_tuple, "Close")
    high_price_df = _symbol_field_df(signal_data_df, symbol_tuple, "High")
    low_price_df = _symbol_field_df(signal_data_df, symbol_tuple, "Low")

    ibs_value_df = ibs_indicator(close_price_df, high_price_df, low_price_df)
    valid_range_bool_df = (
        high_price_df.gt(0.0)
        & low_price_df.gt(0.0)
        & high_price_df.gt(low_price_df)
    )
    log_range_df = np.log(high_price_df / low_price_df).where(valid_range_bool_df)

    # *** CRITICAL*** The exit denominator excludes decision bar T. The
    # rolling median is built through T and then shifted one full session so
    # RangeRatio_T uses only ranges T-21 through T-1.
    prior_range_median_df = (
        log_range_df.rolling(
            window=config_obj.range_median_lookback_day_int,
            min_periods=config_obj.range_median_lookback_day_int,
        )
        .median()
        .shift(1)
    )
    range_ratio_df = log_range_df / prior_range_median_df.replace(0.0, np.nan)

    prior_atr_df = pd.DataFrame(np.nan, index=signal_data_df.index, columns=symbol_tuple)
    prior_natr_df = pd.DataFrame(
        np.nan,
        index=signal_data_df.index,
        columns=symbol_tuple,
    )
    downshock_atr_df = pd.DataFrame(
        np.nan,
        index=signal_data_df.index,
        columns=symbol_tuple,
    )

    for symbol_str in symbol_tuple:
        symbol_ohlc_df = pd.DataFrame(
            {
                "High": high_price_df[symbol_str],
                "Low": low_price_df[symbol_str],
                "Close": close_price_df[symbol_str],
            }
        ).dropna()
        if symbol_ohlc_df.empty:
            continue

        symbol_close_ser = symbol_ohlc_df["Close"].astype(float)
        atr_value_ser = pd.Series(
            talib.ATR(
                symbol_ohlc_df["High"].to_numpy(dtype=float),
                symbol_ohlc_df["Low"].to_numpy(dtype=float),
                symbol_close_ser.to_numpy(dtype=float),
                timeperiod=config_obj.atr_lookback_day_int,
            ),
            index=symbol_ohlc_df.index,
            dtype=float,
        )

        # *** CRITICAL*** Entry and ranking on decision date T use ATR14 and
        # NATR14 from T-1 only. Current High_T/Low_T must not affect either
        # volatility input.
        prior_atr_ser = atr_value_ser.shift(1)
        prior_close_ser = symbol_close_ser.shift(1)
        natr_value_ser = (
            100.0 * atr_value_ser / symbol_close_ser.replace(0.0, np.nan)
        )
        prior_natr_ser = natr_value_ser.shift(1)

        close_return_ser = symbol_close_ser / prior_close_ser - 1.0
        prior_atr_fraction_ser = (
            prior_atr_ser / prior_close_ser.replace(0.0, np.nan)
        )
        downshock_atr_ser = (
            close_return_ser / prior_atr_fraction_ser.replace(0.0, np.nan)
        )

        prior_atr_df[symbol_str] = prior_atr_ser.reindex(signal_data_df.index)
        prior_natr_df[symbol_str] = prior_natr_ser.reindex(signal_data_df.index)
        downshock_atr_df[symbol_str] = downshock_atr_ser.reindex(
            signal_data_df.index
        )

    entry_signal_df = (
        ibs_value_df.lt(config_obj.entry_ibs_max_float)
        & downshock_atr_df.lt(config_obj.downshock_atr_max_float)
    )
    exit_signal_df = (
        ibs_value_df.gt(config_obj.exit_ibs_min_float)
        & range_ratio_df.gt(1.0)
    )

    feature_map_dict = {
        "ibs_value_ser": ibs_value_df,
        f"prior_atr_{config_obj.atr_lookback_day_int}_ser": prior_atr_df,
        f"prior_natr_{config_obj.atr_lookback_day_int}_ser": prior_natr_df,
        "downshock_atr_ser": downshock_atr_df,
        "log_range_ser": log_range_df,
        (
            f"prior_range_median_"
            f"{config_obj.range_median_lookback_day_int}_ser"
        ): prior_range_median_df,
        "range_ratio_ser": range_ratio_df,
        "entry_signal_bool": entry_signal_df.fillna(False).astype(bool),
        "exit_signal_bool": exit_signal_df.fillna(False).astype(bool),
    }

    feature_frame_list: list[pd.DataFrame] = []
    for field_str, feature_df in feature_map_dict.items():
        output_feature_df = feature_df.copy()
        output_feature_df.columns = pd.MultiIndex.from_tuples(
            [
                (symbol_str, field_str)
                for symbol_str in output_feature_df.columns
            ]
        )
        feature_frame_list.append(output_feature_df)

    return pd.concat([signal_data_df] + feature_frame_list, axis=1)


class UsSectorEtfIbsDownshockStrategy(Strategy):
    """Stateful five-slot sector ETF mean-reversion strategy."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        config_obj: UsSectorEtfIbsDownshockConfig = DEFAULT_CONFIG,
    ):
        benchmark_list = list(benchmarks)
        # Costs intentionally inherit Strategy defaults. Do not add strategy-
        # specific slippage or commission arguments here.
        super().__init__(
            name=name,
            benchmarks=benchmark_list,
            capital_base=config_obj.capital_base_float,
            performance_benchmark_symbol_str=(
                config_obj.benchmark_symbol_str if benchmark_list else None
            ),
            performance_benchmark_adjustment_str=(
                TOTALRETURN_ADJUSTMENT_STR if benchmark_list else None
            ),
        )
        self._data_adjustment_policy_dict.update(
            {
                "stock_signal_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "execution_and_marks_adjustment_str": (
                    CAPITALSPECIAL_ADJUSTMENT_STR
                ),
                "performance_benchmark_adjustment_str": (
                    TOTALRETURN_ADJUSTMENT_STR
                ),
            }
        )
        self.config_obj = config_obj
        self.symbol_tuple = tuple(config_obj.symbol_tuple)
        self.max_positions_int = int(config_obj.max_positions_int)
        self.target_weight_float = (
            float(config_obj.sizing_multiplier_float)
            / float(config_obj.sizing_universe_count_int)
        )
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(
            default_trade_id_int
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return compute_us_sector_etf_ibs_downshock_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )

    def get_entry_candidate_list(self, close_row_ser: pd.Series) -> list[str]:
        prior_natr_field_str = (
            f"prior_natr_{self.config_obj.atr_lookback_day_int}_ser"
        )
        ranked_candidate_list: list[tuple[str, float]] = []
        for symbol_str in self.symbol_tuple:
            if self.get_position(symbol_str) != 0:
                continue
            entry_signal_bool = bool(
                close_row_ser.get((symbol_str, "entry_signal_bool"), False)
            )
            prior_natr_float = close_row_ser.get(
                (symbol_str, prior_natr_field_str),
                np.nan,
            )
            if (
                not entry_signal_bool
                or pd.isna(prior_natr_float)
                or not np.isfinite(float(prior_natr_float))
            ):
                continue
            ranked_candidate_list.append(
                (symbol_str, float(prior_natr_float))
            )

        # Python sorting is stable, so exact NATR ties retain the declared
        # fixed-basket order.
        ranked_candidate_list.sort(
            key=lambda candidate_tuple: candidate_tuple[1],
            reverse=True,
        )
        return [
            symbol_str
            for symbol_str, _prior_natr_float in ranked_candidate_list
        ]

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if data_df is None or close_row_ser is None:
            return

        position_ser = self.get_positions()
        held_symbol_set = {
            str(symbol_str)
            for symbol_str, position_float in position_ser.items()
            if str(symbol_str) in self.symbol_tuple and float(position_float) > 0.0
        }
        available_slot_count_int = self.max_positions_int - len(
            held_symbol_set
        )

        for symbol_str in self.symbol_tuple:
            exit_signal_bool = bool(
                close_row_ser.get((symbol_str, "exit_signal_bool"), False)
            )
            if symbol_str not in held_symbol_set or not exit_signal_bool:
                continue
            self.order_target(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )
            held_symbol_set.remove(symbol_str)
            available_slot_count_int += 1

        for symbol_str in self.get_entry_candidate_list(close_row_ser):
            if available_slot_count_int <= 0:
                break
            if symbol_str in held_symbol_set or self.get_position(symbol_str) != 0:
                continue

            target_share_float = self._entry_target_share_float(
                symbol_str=symbol_str,
                close_row_ser=close_row_ser,
            )
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target(
                symbol_str,
                target_share_float,
                trade_id=self.trade_id_int,
            )
            available_slot_count_int -= 1

    def _entry_target_share_float(
        self,
        symbol_str: str,
        close_row_ser: pd.Series,
    ) -> float:
        close_price_float = float(
            close_row_ser.get((symbol_str, "Close"), np.nan)
        )
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError(
                f"Cannot size {symbol_str} entry without a valid decision-bar close."
            )
        return (
            float(self.previous_total_value)
            * self.target_weight_float
            / close_price_float
        )


def get_us_sector_etf_ibs_downshock_data(
    config_obj: UsSectorEtfIbsDownshockConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    pricing_data_df = load_raw_prices(
        symbols=list(config_obj.symbol_tuple),
        benchmarks=[config_obj.benchmark_symbol_str],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    required_column_list = [
        (symbol_str, field_str)
        for symbol_str in config_obj.symbol_tuple
        for field_str in ("Open", "High", "Low", "Close")
    ]
    required_column_list.append((config_obj.benchmark_symbol_str, "Close"))
    missing_column_list = [
        column_tuple
        for column_tuple in required_column_list
        if column_tuple not in pricing_data_df.columns
    ]
    if missing_column_list:
        raise RuntimeError(
            f"Missing required US-sector ETF data columns: {missing_column_list}"
        )
    return pricing_data_df.sort_index()


def _required_history_observation_count_int(
    config_obj: UsSectorEtfIbsDownshockConfig,
) -> int:
    return max(
        config_obj.range_median_lookback_day_int + 1,
        config_obj.atr_lookback_day_int + 2,
    )


def resolve_us_sector_etf_execution_calendar_idx(
    pricing_data_df: pd.DataFrame,
    config_obj: UsSectorEtfIbsDownshockConfig = DEFAULT_CONFIG,
) -> pd.DatetimeIndex:
    ready_calendar_idx = resolve_full_basket_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        required_history_observation_count_int=(
            _required_history_observation_count_int(config_obj)
        ),
    )
    if len(ready_calendar_idx) < 2:
        raise ValueError(
            "No US-sector ETF execution session remains after the first "
            "full-basket-ready decision close."
        )

    # *** CRITICAL*** Full-basket readiness is known only after the first
    # ready bar closes. The first possible fill is therefore the following
    # session's open, never the open of the readiness-certifying bar itself.
    return pd.DatetimeIndex(ready_calendar_idx[1:])


def _run_strategy(
    config_obj: UsSectorEtfIbsDownshockConfig,
    pricing_data_df: pd.DataFrame,
    show_display_bool: bool,
    audit_override_bool: bool | None,
) -> UsSectorEtfIbsDownshockStrategy:
    strategy_obj = UsSectorEtfIbsDownshockStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )
    # *** CRITICAL*** Signals use completed daily bar T. The engine calls
    # iterate on the following session and fills generated orders only at
    # Open_(T+1). No same-close or MOC execution is introduced here.
    calendar_idx = resolve_us_sector_etf_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )
    return strategy_obj


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=DEFAULT_CONFIG,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    config_obj = replace(
        DEFAULT_CONFIG,
        history_start_date_str=resolve_history_start_date_str(
            config_obj=DEFAULT_CONFIG,
            backtest_start_date_str=effective_backtest_start_date_str,
        ),
        backtest_start_date_str=effective_backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_us_sector_etf_ibs_downshock_data(config_obj)
    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        show_display_bool=show_display_bool,
        audit_override_bool=None,
    )
    strategy_obj._performance_benchmark_symbol_str = (
        config_obj.benchmark_symbol_str
    )
    strategy_obj._performance_benchmark_adjustment_str = (
        TOTALRETURN_ADJUSTMENT_STR
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def _write_assumptions_md(
    output_path: Path,
    strategy_obj: UsSectorEtfIbsDownshockStrategy,
) -> None:
    config_obj = strategy_obj.config_obj
    assumptions_md_str = f"""# US Sector ETF IBS Downshock Assumptions

- Research-only strategy; no live/release wiring.
- Fixed ETF basket: `{", ".join(config_obj.symbol_tuple)}`.
- Benchmark: `{config_obj.benchmark_symbol_str}` using `TOTALRETURN`.
- The requested 15:45 snapshot and MOC auction fill are not modeled.
- Daily fallback: completed `Close_T` decision, filled at `Open_(T+1)`.
- `IBS_T = (Close_T - Low_T) / (High_T - Low_T)`.
- Entry: `IBS_T < {config_obj.entry_ibs_max_float}` and `DownShockATR_T < {config_obj.downshock_atr_max_float}`.
- `DownShockATR_T = (Close_T / Close_(T-1) - 1) / (ATR14_(T-1) / Close_(T-1))`.
- Entry ranking: prior-day NATR{config_obj.atr_lookback_day_int}, highest first.
- Exit: `IBS_T > {config_obj.exit_ibs_min_float}` and `RangeRatio_T > 1`.
- `RangeRatio_T = ln(High_T / Low_T) / median(ln(High / Low)_(T-21:T-1))`.
- Exits are ordered before entries; maximum positions: `{config_obj.max_positions_int}`.
- New-position target weight: `{strategy_obj.target_weight_float:.8f}` from `{config_obj.sizing_multiplier_float} / {config_obj.sizing_universe_count_int}`.
- Shares are fixed from `Close_T`, so realized opening weight is `target_weight * Open_(T+1) / Close_T`; overnight gaps can move actual initial exposure away from 13.64% and the five-position total away from 68.18%.
- Existing positions are not rebalanced and there is no time stop.
- Costs inherit the engine defaults unchanged: slippage `{strategy_obj._slippage:.6f}` per side, commission `{strategy_obj._commission_per_share:.6f}` per share, minimum `{strategy_obj._commission_minimum:.2f}` per order.
- Engine cash policy is unchanged: positive cash earns 0%; negative-cash financing is not modeled.
- Execution starts only after every configured ETF has real OHLC data and the full causal warm-up. No proxy or synthetic pre-inception history is used.
"""
    (output_path / "us_sector_etf_ibs_downshock_assumptions.md").write_text(
        assumptions_md_str,
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
    audit_override_bool: bool | None = None,
) -> UsSectorEtfIbsDownshockStrategy:
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=DEFAULT_CONFIG,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    config_obj = replace(
        DEFAULT_CONFIG,
        history_start_date_str=resolve_history_start_date_str(
            config_obj=DEFAULT_CONFIG,
            backtest_start_date_str=effective_backtest_start_date_str,
        ),
        backtest_start_date_str=effective_backtest_start_date_str,
        capital_base_float=(
            DEFAULT_CONFIG.capital_base_float
            if capital_base_float is None
            else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )
    if pricing_data_df is None:
        pricing_data_df = get_us_sector_etf_ibs_downshock_data(config_obj)

    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        show_display_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        _write_assumptions_md(
            output_path=output_path,
            strategy_obj=strategy_obj,
        )
    return strategy_obj


__all__ = [
    "DEFAULT_CONFIG",
    "SECTOR_ETF_SYMBOL_TUPLE",
    "STRATEGY_NAME_STR",
    "UsSectorEtfIbsDownshockConfig",
    "UsSectorEtfIbsDownshockStrategy",
    "build_capacity_analysis_inputs",
    "compute_us_sector_etf_ibs_downshock_signal_df",
    "get_us_sector_etf_ibs_downshock_data",
    "resolve_us_sector_etf_execution_calendar_idx",
    "run_variant",
]


if __name__ == "__main__":
    run_variant()
