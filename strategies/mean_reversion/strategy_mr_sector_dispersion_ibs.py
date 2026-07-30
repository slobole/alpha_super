"""
Research-only Concretum sector-dispersion IBS strategy.

TL;DR: This implements the Concretum Research "Profiting From Sector
Dispersion" rule on SOXX, IGV, and IBB, mapped to the repo's standard
next-open execution convention instead of the paper's MOC implementation.

Paper rule, adapted to daily bars
---------------------------------
For ETF i on decision date t:

    IBS_{i,t} = (Close_{i,t} - Low_{i,t}) / (High_{i,t} - Low_{i,t})

    Range_{i,t} = ln(High_{i,t} / Low_{i,t})

    RelativeRange_{i,t}
        = Range_{i,t} / StdDev(Range_{i,t-1}, ..., Range_{i,t-21})

    entry_{i,t}
        = 1[IBS_{i,t} < 0.10] * 1[RelativeRange_{i,t} > 1.0]

    exit_{i,t}
        = 1[IBS_{i,t} > 0.90] * 1[RelativeRange_{i,t} > 1.0]

    paper_target_weight_i = 1.5 / N

    implemented_target_weight_i = 1.0 / N

    target_shares_{i,t}
        = previous_total_value_t * implemented_target_weight_i / Close_{i,t}

The signal follows the paper, but the default sizing is deliberately
unlevered. The paper's 1.5 / N sizing remains documented above for comparison.

Execution mapping
-----------------
The paper computes OHLC snapshots at 15:45 and submits MOC orders. This repo
does not currently support MOC in the normal engine path, so this module uses
the house Vanilla convention:

    signal from daily bar T -> fractional-share market order filled at Open_{T+1}

This is not an exact article replication. It is the requested next-open
translation of the article signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import ibs_indicator
from data.norgate_loader import (
    INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT,
    TOTALRETURN_ADJUSTMENT_STR,
    load_raw_prices,
)


def default_trade_id_int() -> int:
    return -1


ORIGINAL_SYMBOL_TUPLE = ("SOXX", "IGV", "IBB")
UNIVERSE_A_SYMBOL_TUPLE = (
    "SOXX",
    "IGV",
    "IBB",
    "XLF",
    "XLE",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLRE",
    "XLB",
    "XLC",
)
UNIVERSE_B_SYMBOL_TUPLE = UNIVERSE_A_SYMBOL_TUPLE + (
    "KRE",
    "XOP",
    "ITA",
    "XRT",
    "ITB",
    "XME",
    "IHI",
)
UNIVERSE_C_SYMBOL_TUPLE = UNIVERSE_B_SYMBOL_TUPLE + (
    "XBI",
    "KIE",
    "IAI",
    "IYT",
    "IHF",
    "IHE",
    "XHB",
    "XAR",
    "XES",
)

UNIVERSE_SYMBOL_TUPLE_BY_NAME_DICT = {
    "original": ORIGINAL_SYMBOL_TUPLE,
    "a": UNIVERSE_A_SYMBOL_TUPLE,
    "b": UNIVERSE_B_SYMBOL_TUPLE,
    "c": UNIVERSE_C_SYMBOL_TUPLE,
}


def normalize_universe_name_str(universe_name_str: str) -> str:
    normalized_name_str = str(universe_name_str).strip().lower()
    normalized_name_str = normalized_name_str.replace("universe", "").replace("_", "").replace("-", "")
    normalized_name_str = normalized_name_str.strip()
    if normalized_name_str in ("", "base", "paper", "article", "orig"):
        return "original"
    if normalized_name_str in UNIVERSE_SYMBOL_TUPLE_BY_NAME_DICT:
        return normalized_name_str
    allowed_name_str = ", ".join(["original", "A", "B", "C"])
    raise ValueError(f"Unknown universe_name_str={universe_name_str!r}. Expected one of: {allowed_name_str}.")


def resolve_universe_symbol_tuple(universe_name_str: str) -> tuple[str, ...]:
    normalized_name_str = normalize_universe_name_str(universe_name_str)
    return UNIVERSE_SYMBOL_TUPLE_BY_NAME_DICT[normalized_name_str]


def build_strategy_name_str(config_obj: "SectorDispersionIbsConfig") -> str:
    if config_obj.universe_name_str == "original":
        return "strategy_mr_sector_dispersion_ibs"
    return f"strategy_mr_sector_dispersion_ibs_universe_{config_obj.universe_name_str}"


@dataclass(frozen=True)
class SectorDispersionIbsConfig:
    symbol_tuple: tuple[str, ...] = ORIGINAL_SYMBOL_TUPLE
    universe_name_str: str = "original"
    benchmark_symbol_str: str = "$SPX"
    history_start_date_str: str = "2003-01-01"
    backtest_start_date_str: str = "2004-01-01"
    end_date_str: str | None = None
    entry_ibs_max_float: float = 0.10
    exit_ibs_min_float: float = 0.90
    range_vol_lookback_day_int: int = 21
    min_relative_range_float: float = 1.0
    portfolio_leverage_float: float = 1.0
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.00525
    commission_minimum_float: float = 0.0

    def __post_init__(self) -> None:
        normalized_universe_name_str = normalize_universe_name_str(self.universe_name_str)
        object.__setattr__(self, "universe_name_str", normalized_universe_name_str)
        if normalized_universe_name_str != "original" and tuple(self.symbol_tuple) == ORIGINAL_SYMBOL_TUPLE:
            object.__setattr__(
                self,
                "symbol_tuple",
                resolve_universe_symbol_tuple(normalized_universe_name_str),
            )

        if len(self.symbol_tuple) == 0:
            raise ValueError("symbol_tuple must not be empty.")
        if len(set(self.symbol_tuple)) != len(self.symbol_tuple):
            raise ValueError("symbol_tuple must not contain duplicates.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if self.benchmark_symbol_str in self.symbol_tuple:
            raise ValueError("benchmark_symbol_str must differ from tradable symbols.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if not 0.0 <= self.entry_ibs_max_float <= 1.0:
            raise ValueError("entry_ibs_max_float must lie in [0, 1].")
        if not 0.0 <= self.exit_ibs_min_float <= 1.0:
            raise ValueError("exit_ibs_min_float must lie in [0, 1].")
        if self.entry_ibs_max_float >= self.exit_ibs_min_float:
            raise ValueError("entry_ibs_max_float must be below exit_ibs_min_float.")
        if self.range_vol_lookback_day_int <= 1:
            raise ValueError("range_vol_lookback_day_int must be greater than 1.")
        if self.min_relative_range_float <= 0.0:
            raise ValueError("min_relative_range_float must be positive.")
        if self.portfolio_leverage_float <= 0.0:
            raise ValueError("portfolio_leverage_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = SectorDispersionIbsConfig()
DEFAULT_WARMUP_CALENDAR_DAY_INT = 365


__all__ = [
    "DEFAULT_WARMUP_CALENDAR_DAY_INT",
    "DEFAULT_CONFIG",
    "ORIGINAL_SYMBOL_TUPLE",
    "SectorDispersionIbsConfig",
    "SectorDispersionIbsStrategy",
    "UNIVERSE_A_SYMBOL_TUPLE",
    "UNIVERSE_B_SYMBOL_TUPLE",
    "UNIVERSE_C_SYMBOL_TUPLE",
    "build_capacity_analysis_inputs",
    "build_sector_dispersion_capacity_analysis_inputs",
    "compute_sector_dispersion_ibs_signal_df",
    "get_sector_dispersion_ibs_data",
    "resolve_history_start_date_str",
    "resolve_effective_backtest_start_date_str",
    "resolve_full_basket_calendar_idx",
    "resolve_universe_symbol_tuple",
    "run_variant",
]


def resolve_effective_backtest_start_date_str(
    config_obj: SectorDispersionIbsConfig,
    requested_backtest_start_date_str: str | None,
) -> str:
    """Honor an explicit caller boundary; basket readiness sets the actual start."""
    configured_start_ts = pd.Timestamp(config_obj.backtest_start_date_str)
    if requested_backtest_start_date_str is None:
        return configured_start_ts.date().isoformat()

    requested_start_ts = pd.Timestamp(requested_backtest_start_date_str)
    return requested_start_ts.date().isoformat()


def resolve_history_start_date_str(
    config_obj: SectorDispersionIbsConfig,
    backtest_start_date_str: str | None,
) -> str:
    """Keep the data warmup window before a caller-provided execution start."""
    if backtest_start_date_str is None:
        return config_obj.history_start_date_str

    existing_history_start_ts = pd.Timestamp(config_obj.history_start_date_str)
    requested_backtest_start_ts = pd.Timestamp(backtest_start_date_str)
    if existing_history_start_ts < requested_backtest_start_ts:
        return config_obj.history_start_date_str

    warmup_start_ts = requested_backtest_start_ts - pd.DateOffset(
        days=DEFAULT_WARMUP_CALENDAR_DAY_INT
    )
    return warmup_start_ts.date().isoformat()


def resolve_full_basket_calendar_idx(
    pricing_data_df: pd.DataFrame,
    config_obj: SectorDispersionIbsConfig,
    required_history_observation_count_int: int | None = None,
    required_close_history_observation_count_int: int | None = None,
) -> pd.DatetimeIndex:
    """Return the execution calendar after every fixed-basket ETF is signal-ready."""
    if required_history_observation_count_int is None:
        required_history_observation_count_int = (
            int(config_obj.range_vol_lookback_day_int) + 1
        )
    if required_history_observation_count_int <= 0:
        raise ValueError("required_history_observation_count_int must be positive.")
    if (
        required_close_history_observation_count_int is not None
        and required_close_history_observation_count_int <= 0
    ):
        raise ValueError(
            "required_close_history_observation_count_int must be positive."
        )

    symbol_ready_bool_df = pd.DataFrame(index=pricing_data_df.index)
    symbol_tradable_ohlc_bool_df = pd.DataFrame(index=pricing_data_df.index)
    for symbol_str in config_obj.symbol_tuple:
        field_ser_dict = {
            field_str: pd.to_numeric(
                pricing_data_df[(symbol_str, field_str)],
                errors="coerce",
            )
            for field_str in ("Open", "High", "Low", "Close")
        }
        tradable_ohlc_bool_ser = pd.Series(
            np.isfinite(field_ser_dict["Open"])
            & np.isfinite(field_ser_dict["High"])
            & np.isfinite(field_ser_dict["Low"])
            & np.isfinite(field_ser_dict["Close"])
            & field_ser_dict["Open"].gt(0.0)
            & field_ser_dict["High"].gt(0.0)
            & field_ser_dict["Low"].gt(0.0)
            & field_ser_dict["Close"].gt(0.0)
            & field_ser_dict["High"].ge(field_ser_dict["Open"])
            & field_ser_dict["High"].ge(field_ser_dict["Close"])
            & field_ser_dict["High"].ge(field_ser_dict["Low"])
            & field_ser_dict["Low"].le(field_ser_dict["Open"])
            & field_ser_dict["Low"].le(field_ser_dict["Close"])
            & field_ser_dict["Low"].le(field_ser_dict["High"]),
            index=pricing_data_df.index,
            dtype=bool,
        )
        symbol_tradable_ohlc_bool_df[symbol_str] = tradable_ohlc_bool_ser
        positive_range_bool_ser = (
            tradable_ohlc_bool_ser
            & field_ser_dict["High"].gt(field_ser_dict["Low"])
        )
        # *** CRITICAL*** readiness at T uses only the current completed OHLC
        # row and the preceding fixed lookback. It never backfills pre-inception
        # history or uses observations after T.
        symbol_ready_bool_ser = (
            positive_range_bool_ser.astype(int)
            .rolling(
                window=required_history_observation_count_int,
                min_periods=required_history_observation_count_int,
            )
            .sum()
            .eq(required_history_observation_count_int)
        )
        if required_close_history_observation_count_int is not None:
            valid_close_bool_ser = pd.Series(
                np.isfinite(field_ser_dict["Close"])
                & field_ser_dict["Close"].gt(0.0),
                index=pricing_data_df.index,
                dtype=bool,
            )
            # *** CRITICAL*** Close-history readiness at T uses only closes up
            # to and including completed decision bar T. It does not require
            # unrelated historical Open/High/Low rows outside the range window.
            close_ready_bool_ser = (
                valid_close_bool_ser.astype(int)
                .rolling(
                    window=required_close_history_observation_count_int,
                    min_periods=required_close_history_observation_count_int,
                )
                .sum()
                .eq(required_close_history_observation_count_int)
            )
            symbol_ready_bool_ser = symbol_ready_bool_ser & close_ready_bool_ser
        symbol_ready_bool_df[symbol_str] = symbol_ready_bool_ser

    full_basket_ready_bool_ser = symbol_ready_bool_df.all(axis=1)
    ready_date_index = pricing_data_df.index[full_basket_ready_bool_ser]
    if len(ready_date_index) == 0:
        symbol_list_str = ", ".join(config_obj.symbol_tuple)
        close_requirement_str = (
            ""
            if required_close_history_observation_count_int is None
            else (
                f" and {required_close_history_observation_count_int} consecutive "
                "valid closes"
            )
        )
        raise ValueError(
            "No full-basket Sector Dispersion start is available: "
            f"{required_history_observation_count_int} consecutive positive-range OHLC observations"
            f"{close_requirement_str} "
            f"are required for every ETF in [{symbol_list_str}]."
        )

    configured_start_ts = pd.Timestamp(config_obj.backtest_start_date_str)
    eligible_ready_date_index = ready_date_index[
        ready_date_index >= configured_start_ts
    ]
    if len(eligible_ready_date_index) == 0:
        raise ValueError(
            "No full-basket Sector Dispersion start remains on or after the configured "
            f"start {configured_start_ts.date()}."
        )
    effective_start_ts = pd.Timestamp(
        eligible_ready_date_index[0]
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= effective_start_ts]
    if len(calendar_idx) == 0:
        raise ValueError(
            "No Sector Dispersion observations remain on or after the effective "
            f"full-basket start {effective_start_ts.date()}."
        )
    invalid_after_start_bool_df = ~symbol_tradable_ohlc_bool_df.loc[calendar_idx]
    if invalid_after_start_bool_df.any(axis=None):
        first_invalid_ts = invalid_after_start_bool_df.any(axis=1).idxmax()
        invalid_symbol_list = invalid_after_start_bool_df.columns[
            invalid_after_start_bool_df.loc[first_invalid_ts]
        ].to_list()
        raise ValueError(
            "Sector Dispersion fixed-basket OHLC became invalid after the effective "
            f"start on {pd.Timestamp(first_invalid_ts).date()}: "
            f"{', '.join(invalid_symbol_list)}."
        )
    return pd.DatetimeIndex(calendar_idx)


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
    if len(missing_column_list) > 0:
        raise RuntimeError(f"Missing required sector-dispersion columns: {missing_column_list}")

    return pd.DataFrame(
        {
            symbol_str: pd.to_numeric(pricing_data_df[(symbol_str, field_str)], errors="coerce")
            for symbol_str in symbol_tuple
        },
        index=pricing_data_df.index,
        dtype=float,
    )


def compute_sector_dispersion_ibs_signal_df(
    pricing_data_df: pd.DataFrame,
    config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Add article signal fields to a daily OHLC pricing frame.

    Daily bars are an explicit approximation for the paper's 15:45 OHLC
    snapshot. The lagged range denominator preserves the paper's no-current-day
    scaling rule.
    """
    signal_data_df = pricing_data_df.copy()
    symbol_tuple = tuple(config_obj.symbol_tuple)
    close_price_df = _symbol_field_df(signal_data_df, symbol_tuple, "Close")
    high_price_df = _symbol_field_df(signal_data_df, symbol_tuple, "High")
    low_price_df = _symbol_field_df(signal_data_df, symbol_tuple, "Low")

    ibs_value_df = ibs_indicator(close_price_df, high_price_df, low_price_df)

    valid_range_bool_df = high_price_df.gt(0.0) & low_price_df.gt(0.0) & high_price_df.gt(low_price_df)
    log_range_df = np.log(high_price_df / low_price_df).where(valid_range_bool_df)

    # *** CRITICAL*** range volatility is lagged by one full trading day:
    # RelativeRange_t may use Range_t in the numerator, but the denominator
    # must use only Range_{t-1} through Range_{t-L}. Do not remove shift(1).
    range_vol_df = log_range_df.rolling(
        window=int(config_obj.range_vol_lookback_day_int),
        min_periods=int(config_obj.range_vol_lookback_day_int),
    ).std().shift(1)
    relative_range_df = log_range_df / range_vol_df.replace(0.0, np.nan)

    entry_signal_df = (
        ibs_value_df.lt(float(config_obj.entry_ibs_max_float))
        & relative_range_df.gt(float(config_obj.min_relative_range_float))
    )
    exit_signal_df = (
        ibs_value_df.gt(float(config_obj.exit_ibs_min_float))
        & relative_range_df.gt(float(config_obj.min_relative_range_float))
    )

    feature_map = {
        "ibs_value_ser": ibs_value_df,
        "log_range_ser": log_range_df,
        f"range_vol_{config_obj.range_vol_lookback_day_int}_ser": range_vol_df,
        "relative_range_ser": relative_range_df,
        "entry_signal_bool": entry_signal_df.fillna(False).astype(bool),
        "exit_signal_bool": exit_signal_df.fillna(False).astype(bool),
    }

    feature_frame_list: list[pd.DataFrame] = []
    for field_str, feature_df in feature_map.items():
        output_feature_df = feature_df.copy()
        output_feature_df.columns = pd.MultiIndex.from_tuples(
            [(symbol_str, field_str) for symbol_str in output_feature_df.columns]
        )
        feature_frame_list.append(output_feature_df)

    return pd.concat([signal_data_df] + feature_frame_list, axis=1)


class SectorDispersionIbsStrategy(Strategy):
    """
    Three-ETF IBS mean-reversion sleeve with independent entries and exits.

    Each ETF is sized to:

        target_weight_i = portfolio_leverage / N

        target_shares_i = previous_total_value * target_weight_i / Close_i

    Existing positions are not resized while held. They are liquidated only
    when their own exit signal fires.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config_obj.capital_base_float,
            slippage=config_obj.slippage_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
            performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        )
        # Report-only provenance: benchmark metrics keep the familiar label
        # while reading the total-return series the loader stored for it.
        self._benchmark_data_symbol_map_dict = {
            str(benchmark_str): INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT.get(
                str(benchmark_str), str(benchmark_str)
            )
            for benchmark_str in benchmarks
        }
        self.config_obj = config_obj
        self.symbol_tuple = tuple(config_obj.symbol_tuple)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.target_weight_float = float(config_obj.portfolio_leverage_float) / float(len(self.symbol_tuple))

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return compute_sector_dispersion_ibs_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        position_ser = self.get_positions()
        held_symbol_set = {
            str(symbol_str)
            for symbol_str, position_float in position_ser.items()
            if str(symbol_str) in self.symbol_tuple and float(position_float) > 0.0
        }

        for symbol_str in self.symbol_tuple:
            exit_signal_bool = bool(close_row_ser.get((symbol_str, "exit_signal_bool"), False))
            if symbol_str in held_symbol_set and exit_signal_bool:
                self.order_target(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                held_symbol_set.remove(symbol_str)

        for symbol_str in self.symbol_tuple:
            if symbol_str in held_symbol_set:
                continue

            if self.get_position(symbol_str) != 0:
                continue

            entry_signal_bool = bool(close_row_ser.get((symbol_str, "entry_signal_bool"), False))
            if not entry_signal_bool:
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

    def _entry_target_share_float(
        self,
        symbol_str: str,
        close_row_ser: pd.Series,
    ) -> float:
        close_price_float = float(close_row_ser.get((symbol_str, "Close"), np.nan))
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError(f"Cannot size {symbol_str} entry without a valid decision-bar close.")

        return float(self.previous_total_value) * self.target_weight_float / close_price_float


def get_sector_dispersion_ibs_data(
    config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    # *** CRITICAL*** Norgate's TOTALRETURN adjustment back-adjusts dividends
    # on stocks and ETFs but does nothing on an index symbol, so loading '$SPX'
    # returns the PRICE index. The genuine total-return index is a separate
    # data symbol; the report keeps the familiar label via the strategy's
    # benchmark data-symbol map.
    benchmark_data_symbol_str = INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT.get(
        config_obj.benchmark_symbol_str, config_obj.benchmark_symbol_str
    )
    pricing_data_df = load_raw_prices(
        symbols=list(config_obj.symbol_tuple),
        benchmarks=[benchmark_data_symbol_str],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )

    required_column_list = [
        (symbol_str, field_str)
        for symbol_str in config_obj.symbol_tuple
        for field_str in ("Open", "High", "Low", "Close")
    ]
    required_column_list.append((benchmark_data_symbol_str, "Close"))
    missing_column_list = [
        column_tuple for column_tuple in required_column_list if column_tuple not in pricing_data_df.columns
    ]
    if len(missing_column_list) > 0:
        raise RuntimeError(f"Missing required sector-dispersion data columns: {missing_column_list}")

    return pricing_data_df.sort_index()


def build_sector_dispersion_capacity_analysis_inputs(
    strategy_class: type[SectorDispersionIbsStrategy],
    strategy_name_str: str,
    config_obj: SectorDispersionIbsConfig,
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    required_close_history_observation_count_int: int | None = None,
) -> dict[str, object]:
    """Rerun one fixed-basket sector variant for the Capacity analyzer."""
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=config_obj,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    capacity_config_obj = replace(
        config_obj,
        history_start_date_str=resolve_history_start_date_str(
            config_obj=config_obj,
            backtest_start_date_str=effective_backtest_start_date_str,
        ),
        backtest_start_date_str=effective_backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_sector_dispersion_ibs_data(config_obj=capacity_config_obj)
    strategy_obj = strategy_class(
        name=strategy_name_str,
        benchmarks=[capacity_config_obj.benchmark_symbol_str],
        config_obj=capacity_config_obj,
    )

    # *** CRITICAL *** lookahead-sensitive: the execution calendar keeps the
    # full causal warmup and orders generated from completed bar T still fill
    # at Open_(T+1), exactly as in each normal sector strategy run.
    calendar_idx = resolve_full_basket_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=capacity_config_obj,
        required_close_history_observation_count_int=(
            required_close_history_observation_count_int
        ),
    )
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )
    strategy_obj._performance_benchmark_symbol_str = (
        capacity_config_obj.benchmark_symbol_str
    )
    strategy_obj._performance_benchmark_adjustment_str = "TOTALRETURN"

    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    return build_sector_dispersion_capacity_analysis_inputs(
        strategy_class=SectorDispersionIbsStrategy,
        strategy_name_str=build_strategy_name_str(DEFAULT_CONFIG),
        config_obj=DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        end_date_str=end_date_str,
    )


def _write_assumptions_md(
    output_path: Path,
    strategy_obj: SectorDispersionIbsStrategy,
) -> None:
    config_obj = strategy_obj.config_obj
    assumption_md_str = f"""# Sector Dispersion IBS Assumptions

- Research-only strategy; no live/release wiring.
- Source: Concretum Research, "Profiting From Sector Dispersion", July 3, 2026.
- Universe label: `{config_obj.universe_name_str}`.
- Tradable ETF basket: `{", ".join(config_obj.symbol_tuple)}`.
- Benchmark: `{config_obj.benchmark_symbol_str}`.
- Paper source data used 15:45 OHLC snapshots and MOC orders.
- This implementation uses daily OHLC bars and the repo's normal next-open execution:
  signal at daily bar `T`, fill at `Open_(T+1)`.
- Entry signal: `IBS_T < {config_obj.entry_ibs_max_float:.4f}` and `RelativeRange_T > {config_obj.min_relative_range_float:.4f}`.
- Exit signal: `IBS_T > {config_obj.exit_ibs_min_float:.4f}` and `RelativeRange_T > {config_obj.min_relative_range_float:.4f}`.
- `IBS_T = (Close_T - Low_T) / (High_T - Low_T)`.
- `Range_T = ln(High_T / Low_T)`.
- `RelativeRange_T = Range_T / std(Range_(T-1) ... Range_(T-{config_obj.range_vol_lookback_day_int}))`.
- Target weight per active ETF: `{strategy_obj.target_weight_float:.6f}` from leverage `{config_obj.portfolio_leverage_float:.4f}` / N `{len(config_obj.symbol_tuple)}`.
- Entry share target: `previous_total_value_T * target_weight / Close_T`; fractional shares are allowed in this research path.
- Existing positions are not resized while held; they are touched only on exit.
- Slippage: `{config_obj.slippage_float:.6f}` per side.
- Commission: `{config_obj.commission_per_share_float:.6f}` per share, minimum `{config_obj.commission_minimum_float:.2f}`.
- The commission is the symmetric per-side equivalent of the paper's `$0.0035/share` buy fee and doubled sell fee.
- Norgate tradable ETF OHLC is loaded through `load_raw_prices`, so this follows the repo's tradable-price convention rather than a separate Massive total-return feed.
"""
    (output_path / "sector_dispersion_ibs_assumptions.md").write_text(
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
    universe_name_str: str | None = None,
    pricing_data_df: pd.DataFrame | None = None,
    audit_override_bool: bool | None = None,
) -> SectorDispersionIbsStrategy:
    config_obj = DEFAULT_CONFIG
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=config_obj,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    if universe_name_str is not None:
        normalized_universe_name_str = normalize_universe_name_str(universe_name_str)
        config_obj = replace(
            config_obj,
            universe_name_str=normalized_universe_name_str,
            symbol_tuple=resolve_universe_symbol_tuple(normalized_universe_name_str),
        )

    if backtest_start_date_str is not None or capital_base_float is not None or end_date_str is not None:
        config_obj = replace(
            config_obj,
            history_start_date_str=resolve_history_start_date_str(
                config_obj=config_obj,
                backtest_start_date_str=effective_backtest_start_date_str,
            ),
            backtest_start_date_str=effective_backtest_start_date_str,
            capital_base_float=(
                config_obj.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    if pricing_data_df is None:
        pricing_data_df = get_sector_dispersion_ibs_data(config_obj=config_obj)

    strategy_obj = SectorDispersionIbsStrategy(
        name=build_strategy_name_str(config_obj),
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for the lagged range scale. The
    # execution calendar begins only when every fixed-basket ETF has enough
    # causal history; pre-inception rows are never treated as a partial basket.
    calendar_idx = resolve_full_basket_calendar_idx(
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

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        _write_assumptions_md(output_path=output_path, strategy_obj=strategy_obj)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
