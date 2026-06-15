"""
Weekly risk-managed sector momentum rotation.

TL;DR: every Monday close, rank the 11 Select Sector ETFs by a blended
momentum score divided by trailing 63-day volatility, buy up to the top 3
positive-momentum sectors, keep an existing sector if it remains in the top 5,
and cut total sector exposure to 50% when SPY is below its trailing 200-day SMA.

Core formulas
-------------
For sector ETF i on Monday decision date t:

    R_{i,t}^{(L)}
        = SignalClose_{i,t} / SignalClose_{i,t-L} - 1

    r_{i,t}
        = SignalClose_{i,t} / SignalClose_{i,t-1} - 1

    sigma_{i,t}^{(63)}
        = std(r_{i,t-62}, ..., r_{i,t})

    score_{i,t}
        = (0.5 * R_{i,t}^{(63)}
           + 0.3 * R_{i,t}^{(126)}
           + 0.2 * R_{i,t}^{(252)})
          / sigma_{i,t}^{(63)}

Eligibility:

    eligible_{i,t}
        = 1[R_{i,t}^{(126)} > 0 and score_{i,t} is finite]

Selection with turnover buffer:

    keep_{i,t}
        = 1[currently held i and rank_{i,t} <= 5]

    buy_{i,t}
        = top 3 eligible names by score, filling open slots after keeps

Sizing:

    raw_weight_{i,t}
        = (1 / sigma_{i,t}^{(63)}) / sum_j(1 / sigma_{j,t}^{(63)})

    exposure_t
        = 0.5 if SPY_t < SMA200_t else 1.0

    target_weight_{i,t}
        = exposure_t * raw_weight_{i,t}

Execution mapping:

    decision_date_t
        = actual Monday trading close

    execution_date_t
        = next tradable open after decision_date_t

This module is research/backtest-only. It is not wired to live execution.
The default backtest starts in 2019-07 so the 11-sector ETF set has enough
history for the 252-day return and 63-day volatility windows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import (
    TOTALRETURN_ADJUSTMENT_STR,
    load_price_timeseries,
    load_raw_prices,
)


MODEL_SYMBOL_STR = "_MODEL"
CASH_COLUMN_STR = "Cash"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class WeeklySectorMomentumConfig:
    sector_symbol_tuple: tuple[str, ...] = (
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    )
    regime_symbol_str: str = "SPY"
    benchmark_list: tuple[str, ...] = ("SPY",)
    history_start_date_str: str = "2018-06-01"
    backtest_start_date_str: str = "2019-07-01"
    end_date_str: str | None = None
    return_short_window_int: int = 63
    return_mid_window_int: int = 126
    return_long_window_int: int = 252
    volatility_window_int: int = 63
    regime_sma_window_int: int = 200
    buy_rank_int: int = 3
    hold_rank_int: int = 5
    risk_off_exposure_float: float = 0.5
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if len(self.sector_symbol_tuple) == 0:
            raise ValueError("sector_symbol_tuple must not be empty.")
        if len(set(self.sector_symbol_tuple)) != len(self.sector_symbol_tuple):
            raise ValueError("sector_symbol_tuple must not contain duplicates.")
        if not self.regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if self.regime_symbol_str in self.sector_symbol_tuple:
            raise ValueError("regime_symbol_str must not be part of sector_symbol_tuple.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        for field_name_str in (
            "return_short_window_int",
            "return_mid_window_int",
            "return_long_window_int",
            "volatility_window_int",
            "regime_sma_window_int",
            "buy_rank_int",
            "hold_rank_int",
        ):
            if int(getattr(self, field_name_str)) <= 0:
                raise ValueError(f"{field_name_str} must be positive.")
        if self.hold_rank_int < self.buy_rank_int:
            raise ValueError("hold_rank_int must be greater than or equal to buy_rank_int.")
        if not 0.0 <= self.risk_off_exposure_float <= 1.0:
            raise ValueError("risk_off_exposure_float must be between 0.0 and 1.0.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = WeeklySectorMomentumConfig()


__all__ = [
    "CASH_COLUMN_STR",
    "DEFAULT_CONFIG",
    "MODEL_SYMBOL_STR",
    "WeeklyRiskManagedSectorMomentumStrategy",
    "WeeklySectorMomentumConfig",
    "build_daily_target_weight_df",
    "compute_weekly_sector_momentum_signal_tables",
    "extract_sector_signal_close_df",
    "get_monday_decision_close_df",
    "get_weekly_sector_momentum_data",
    "load_sector_signal_close_df",
    "map_monday_decision_dates_to_rebalance_schedule_df",
    "merge_signal_close_into_pricing_data_df",
    "run_variant",
]


def get_monday_decision_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows whose timestamp is an actual Monday trading close.

    Monday holidays are skipped because there is no Monday close to act on.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")

    sorted_price_close_df = price_close_df.sort_index()
    monday_decision_mask_vec = sorted_price_close_df.index.dayofweek == 0
    monday_decision_close_df = sorted_price_close_df.loc[monday_decision_mask_vec].copy()
    monday_decision_close_df.index.name = "decision_date_ts"
    return monday_decision_close_df


def map_monday_decision_dates_to_rebalance_schedule_df(
    decision_date_index: pd.DatetimeIndex,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each Monday decision close to the strictly next tradable open.
    """
    if len(execution_index) < 2:
        raise ValueError("execution_index must contain at least two trading dates.")
    if len(decision_date_index) == 0:
        raise ValueError("decision_date_index must not be empty.")

    sorted_execution_index = pd.DatetimeIndex(execution_index).sort_values()
    sorted_decision_date_index = pd.DatetimeIndex(decision_date_index).sort_values()

    rebalance_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date_ts in sorted_decision_date_index:
        execution_insert_int = int(
            sorted_execution_index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
        )
        if execution_insert_int >= len(sorted_execution_index):
            continue

        # *** CRITICAL*** Monday-close decisions must execute strictly after
        # the decision bar. A same-bar fill would use prices unavailable when
        # the signal is formed.
        execution_date_ts = pd.Timestamp(sorted_execution_index[execution_insert_int])
        rebalance_schedule_map[execution_date_ts] = pd.Timestamp(decision_date_ts)

    if len(rebalance_schedule_map) == 0:
        raise RuntimeError("No Monday rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def _rank_score_df(score_df: pd.DataFrame) -> pd.DataFrame:
    rank_df = pd.DataFrame(np.nan, index=score_df.index, columns=score_df.columns, dtype=float)
    for decision_date_ts, score_row_ser in score_df.iterrows():
        score_candidate_ser = pd.to_numeric(score_row_ser, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        score_candidate_ser = score_candidate_ser.dropna()
        if len(score_candidate_ser) == 0:
            continue

        rank_frame_df = pd.DataFrame(
            {
                "score_float": score_candidate_ser.astype(float),
                "symbol_str": score_candidate_ser.index.astype(str),
            },
            index=score_candidate_ser.index.astype(str),
        )
        ranked_symbol_list = (
            rank_frame_df.sort_values(
                by=["score_float", "symbol_str"],
                ascending=[False, True],
                kind="mergesort",
            )
            .index.astype(str)
            .tolist()
        )
        for rank_int, symbol_str in enumerate(ranked_symbol_list, start=1):
            rank_df.loc[decision_date_ts, symbol_str] = float(rank_int)
    return rank_df


def compute_weekly_sector_momentum_signal_tables(
    sector_signal_close_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: WeeklySectorMomentumConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Compute Monday-only sector momentum features and the SPY exposure scaler.
    """
    missing_sector_symbol_list = [
        sector_symbol_str
        for sector_symbol_str in config.sector_symbol_tuple
        if sector_symbol_str not in sector_signal_close_df.columns
    ]
    if len(missing_sector_symbol_list) > 0:
        raise RuntimeError(f"Missing sector signal-close columns: {missing_sector_symbol_list}")

    sector_signal_close_df = (
        sector_signal_close_df.loc[:, list(config.sector_symbol_tuple)]
        .astype(float)
        .sort_index()
    )
    regime_close_ser = regime_close_ser.astype(float).sort_index()

    # *** CRITICAL*** Return features are trailing close-to-close returns.
    # For a Monday decision at t, the oldest input is t-L and no observation
    # after Monday close can enter the score.
    return_short_df = (
        sector_signal_close_df / sector_signal_close_df.shift(config.return_short_window_int)
    ) - 1.0
    return_mid_df = (
        sector_signal_close_df / sector_signal_close_df.shift(config.return_mid_window_int)
    ) - 1.0
    return_long_df = (
        sector_signal_close_df / sector_signal_close_df.shift(config.return_long_window_int)
    ) - 1.0

    # *** CRITICAL*** Daily returns use shift(1); the volatility estimate is a
    # backward-only rolling standard deviation ending at the decision close.
    daily_return_df = sector_signal_close_df / sector_signal_close_df.shift(1) - 1.0
    volatility_df = daily_return_df.rolling(
        window=config.volatility_window_int,
        min_periods=config.volatility_window_int,
    ).std()

    raw_score_df = (
        0.5 * return_short_df
        + 0.3 * return_mid_df
        + 0.2 * return_long_df
    ) / volatility_df
    raw_score_df = raw_score_df.replace([np.inf, -np.inf], np.nan)
    eligible_score_df = raw_score_df.where((return_mid_df > 0.0) & (volatility_df > 0.0))

    # *** CRITICAL*** The SPY SMA200 market filter is a backward-only rolling
    # average of SPY closes available at the Monday close.
    regime_sma_ser = regime_close_ser.rolling(
        window=config.regime_sma_window_int,
        min_periods=config.regime_sma_window_int,
    ).mean()
    risk_off_bool_ser = (regime_close_ser < regime_sma_ser).where(regime_sma_ser.notna())
    exposure_multiplier_ser = pd.Series(np.nan, index=regime_close_ser.index, dtype=float)
    exposure_multiplier_ser.loc[risk_off_bool_ser == False] = 1.0  # noqa: E712
    exposure_multiplier_ser.loc[risk_off_bool_ser == True] = float(config.risk_off_exposure_float)  # noqa: E712
    exposure_multiplier_ser.name = "exposure_multiplier_float"

    monday_decision_close_df = get_monday_decision_close_df(
        price_close_df=sector_signal_close_df,
    )
    decision_index = monday_decision_close_df.index
    return_short_decision_df = return_short_df.reindex(decision_index)
    return_mid_decision_df = return_mid_df.reindex(decision_index)
    return_long_decision_df = return_long_df.reindex(decision_index)
    volatility_decision_df = volatility_df.reindex(decision_index)
    score_decision_df = eligible_score_df.reindex(decision_index)
    rank_decision_df = _rank_score_df(score_decision_df)
    regime_sma_decision_ser = regime_sma_ser.reindex(decision_index)
    exposure_decision_ser = exposure_multiplier_ser.reindex(decision_index)

    mature_signal_mask_ser = (
        return_long_decision_df.notna().any(axis=1)
        & volatility_decision_df.notna().any(axis=1)
    )
    valid_decision_mask_ser = mature_signal_mask_ser & exposure_decision_ser.notna()
    valid_decision_index = pd.DatetimeIndex(score_decision_df.index[valid_decision_mask_ser])

    return (
        return_short_decision_df.reindex(valid_decision_index),
        return_mid_decision_df.reindex(valid_decision_index),
        return_long_decision_df.reindex(valid_decision_index),
        volatility_decision_df.reindex(valid_decision_index),
        score_decision_df.reindex(valid_decision_index),
        rank_decision_df.reindex(valid_decision_index),
        regime_sma_decision_ser.reindex(valid_decision_index),
        exposure_decision_ser.reindex(valid_decision_index),
    )


def load_sector_signal_close_df(
    config: WeeklySectorMomentumConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load total-return sector closes for signal formation.
    """
    signal_close_map: dict[str, pd.Series] = {}
    for sector_symbol_str in config.sector_symbol_tuple:
        signal_price_df = load_price_timeseries(
            sector_symbol_str,
            adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
            start_date_str=config.history_start_date_str,
            end_date_str=config.end_date_str,
        )
        if len(signal_price_df) == 0:
            continue
        signal_close_map[sector_symbol_str] = signal_price_df["Close"].astype(float)

    missing_sector_symbol_list = [
        sector_symbol_str
        for sector_symbol_str in config.sector_symbol_tuple
        if sector_symbol_str not in signal_close_map
    ]
    if len(missing_sector_symbol_list) > 0:
        raise RuntimeError(f"Missing total-return signal data for: {missing_sector_symbol_list}")

    return pd.DataFrame(signal_close_map).sort_index()


def load_execution_price_df(
    config: WeeklySectorMomentumConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load CAPITALSPECIAL OHLC bars for fills, valuation, and the SPY price filter.
    """
    execution_symbol_list = list(
        dict.fromkeys(list(config.sector_symbol_tuple) + [config.regime_symbol_str])
    )
    return load_raw_prices(
        symbols=execution_symbol_list,
        benchmarks=[],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )


def merge_signal_close_into_pricing_data_df(
    execution_price_df: pd.DataFrame,
    sector_signal_close_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach total-return sector SignalClose fields beside execution OHLC bars.
    """
    pricing_data_df = execution_price_df.copy()
    for sector_symbol_str in sector_signal_close_df.columns.astype(str):
        pricing_data_df[(sector_symbol_str, "SignalClose")] = sector_signal_close_df[
            sector_symbol_str
        ].reindex(pricing_data_df.index)

    pricing_data_df = pricing_data_df.sort_index(axis=1)
    return pricing_data_df


def extract_sector_signal_close_df(
    pricing_data_df: pd.DataFrame,
    sector_symbol_tuple: Sequence[str],
) -> pd.DataFrame:
    """
    Extract sector total-return signal closes from required SignalClose fields.
    """
    signal_close_map: dict[str, pd.Series] = {}
    for sector_symbol_str in sector_symbol_tuple:
        signal_close_key = (sector_symbol_str, "SignalClose")
        if signal_close_key not in pricing_data_df.columns:
            raise RuntimeError(
                f"Missing total-return SignalClose history for {sector_symbol_str}. "
                "Do not silently fall back to execution Close for this strategy."
            )
        signal_close_map[sector_symbol_str] = pricing_data_df.loc[:, signal_close_key].astype(float)

    return pd.DataFrame(signal_close_map, index=pricing_data_df.index, dtype=float)


def extract_regime_close_ser(
    pricing_data_df: pd.DataFrame,
    regime_symbol_str: str,
) -> pd.Series:
    regime_close_key = (regime_symbol_str, "Close")
    if regime_close_key not in pricing_data_df.columns:
        raise RuntimeError(f"Missing regime close history for {regime_symbol_str}.")
    return pricing_data_df.loc[:, regime_close_key].astype(float)


def build_daily_target_weight_df(
    rebalance_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
    sector_symbol_tuple: Sequence[str],
) -> pd.DataFrame:
    """
    Build report-ready daily target weights including residual cash.
    """
    report_column_list = list(sector_symbol_tuple) + [CASH_COLUMN_STR]
    if len(rebalance_weight_df) == 0:
        return pd.DataFrame(columns=report_column_list, dtype=float)

    aligned_rebalance_weight_df = (
        rebalance_weight_df.reindex(columns=list(sector_symbol_tuple), fill_value=0.0)
        .sort_index()
        .astype(float)
    )
    daily_target_weight_df = aligned_rebalance_weight_df.reindex(pd.DatetimeIndex(execution_index)).ffill()
    daily_target_weight_df = daily_target_weight_df.loc[
        daily_target_weight_df.index >= pd.Timestamp(aligned_rebalance_weight_df.index[0])
    ]
    daily_target_weight_df = daily_target_weight_df.fillna(0.0)
    daily_target_weight_df[CASH_COLUMN_STR] = 1.0 - daily_target_weight_df.sum(axis=1)

    weight_sum_ser = daily_target_weight_df.sum(axis=1)
    if not np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise ValueError("Daily target weights, including cash, must sum to 1.0.")
    if (daily_target_weight_df[CASH_COLUMN_STR] < -1e-12).any():
        raise ValueError("Daily target weights imply negative cash.")

    return daily_target_weight_df.loc[:, report_column_list]


def get_weekly_sector_momentum_data(
    config: WeeklySectorMomentumConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_signal_close_df = load_sector_signal_close_df(config=config)
    execution_price_df = load_execution_price_df(config=config)
    pricing_data_df = merge_signal_close_into_pricing_data_df(
        execution_price_df=execution_price_df,
        sector_signal_close_df=sector_signal_close_df,
    )
    regime_close_ser = extract_regime_close_ser(
        pricing_data_df=pricing_data_df,
        regime_symbol_str=config.regime_symbol_str,
    )
    (
        _return_short_df,
        _return_mid_df,
        _return_long_df,
        _volatility_df,
        _score_df,
        _rank_df,
        _regime_sma_ser,
        exposure_multiplier_ser,
    ) = compute_weekly_sector_momentum_signal_tables(
        sector_signal_close_df=extract_sector_signal_close_df(
            pricing_data_df=pricing_data_df,
            sector_symbol_tuple=config.sector_symbol_tuple,
        ),
        regime_close_ser=regime_close_ser,
        config=config,
    )

    valid_decision_index = pd.DatetimeIndex(exposure_multiplier_ser.index[exposure_multiplier_ser.notna()])
    rebalance_schedule_df = map_monday_decision_dates_to_rebalance_schedule_df(
        decision_date_index=valid_decision_index,
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df.sort_index(), rebalance_schedule_df


class WeeklyRiskManagedSectorMomentumStrategy(Strategy):
    """
    Long-only weekly sector momentum rotation with rank-buffer turnover control.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        config: WeeklySectorMomentumConfig = DEFAULT_CONFIG,
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")

        self.config = config
        self.sector_symbol_tuple = tuple(config.sector_symbol_tuple)
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.trade_id_int = 0
        self.current_trade_id_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.rebalance_weight_df = pd.DataFrame(columns=list(self.sector_symbol_tuple), dtype=float)
        self.show_taa_weights_report = True
        self.daily_target_weights = pd.DataFrame(
            columns=list(self.sector_symbol_tuple) + [CASH_COLUMN_STR],
            dtype=float,
        )

    @property
    def return_short_field_str(self) -> str:
        return f"return_{self.config.return_short_window_int}d_ser"

    @property
    def return_mid_field_str(self) -> str:
        return f"return_{self.config.return_mid_window_int}d_ser"

    @property
    def return_long_field_str(self) -> str:
        return f"return_{self.config.return_long_window_int}d_ser"

    @property
    def volatility_field_str(self) -> str:
        return f"volatility_{self.config.volatility_window_int}d_ser"

    @property
    def regime_sma_field_str(self) -> str:
        return f"regime_sma_{self.config.regime_sma_window_int}d_ser"

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        sector_signal_close_df = extract_sector_signal_close_df(
            pricing_data_df=pricing_data_df,
            sector_symbol_tuple=self.sector_symbol_tuple,
        )
        regime_close_ser = extract_regime_close_ser(
            pricing_data_df=pricing_data_df,
            regime_symbol_str=self.config.regime_symbol_str,
        )
        (
            return_short_df,
            return_mid_df,
            return_long_df,
            volatility_df,
            score_df,
            rank_df,
            regime_sma_ser,
            exposure_multiplier_ser,
        ) = compute_weekly_sector_momentum_signal_tables(
            sector_signal_close_df=sector_signal_close_df,
            regime_close_ser=regime_close_ser,
            config=self.config,
        )

        feature_map: dict[str, pd.DataFrame] = {
            self.return_short_field_str: return_short_df,
            self.return_mid_field_str: return_mid_df,
            self.return_long_field_str: return_long_df,
            self.volatility_field_str: volatility_df,
            "score_ser": score_df,
            "rank_float": rank_df,
        }
        for field_str, feature_value_df in feature_map.items():
            aligned_feature_df = feature_value_df.reindex(signal_data_df.index)
            for sector_symbol_str in self.sector_symbol_tuple:
                signal_data_df[(sector_symbol_str, field_str)] = aligned_feature_df[sector_symbol_str]

        signal_data_df[(MODEL_SYMBOL_STR, self.regime_sma_field_str)] = regime_sma_ser.reindex(
            signal_data_df.index
        )
        signal_data_df[(MODEL_SYMBOL_STR, "exposure_multiplier_float")] = (
            exposure_multiplier_ser.reindex(signal_data_df.index)
        )
        return signal_data_df

    def _ensure_trade_id_int(self, sector_symbol_str: str) -> int:
        if self.current_trade_id_map[sector_symbol_str] == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_map[sector_symbol_str] = self.trade_id_int
        return int(self.current_trade_id_map[sector_symbol_str])

    def _current_held_sector_list(self) -> list[str]:
        current_position_ser = self.get_positions().reindex(
            list(self.sector_symbol_tuple),
            fill_value=0.0,
        )
        return [
            sector_symbol_str
            for sector_symbol_str, share_float in current_position_ser.items()
            if float(share_float) > 0.0
        ]

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        feature_df = close_row_ser.unstack()
        if MODEL_SYMBOL_STR not in feature_df.index:
            return pd.Series(dtype=float)

        exposure_multiplier_float = float(
            pd.to_numeric(
                pd.Series([feature_df.loc[MODEL_SYMBOL_STR].get("exposure_multiplier_float", np.nan)]),
                errors="coerce",
            ).iloc[0]
        )
        if not np.isfinite(exposure_multiplier_float) or exposure_multiplier_float <= 0.0:
            return pd.Series(dtype=float)

        candidate_row_list: list[dict[str, object]] = []
        for sector_symbol_str in self.sector_symbol_tuple:
            if sector_symbol_str not in feature_df.index:
                continue
            sector_feature_ser = feature_df.loc[sector_symbol_str]
            score_float = float(
                pd.to_numeric(pd.Series([sector_feature_ser.get("score_ser", np.nan)]), errors="coerce").iloc[0]
            )
            rank_float = float(
                pd.to_numeric(pd.Series([sector_feature_ser.get("rank_float", np.nan)]), errors="coerce").iloc[0]
            )
            volatility_float = float(
                pd.to_numeric(
                    pd.Series([sector_feature_ser.get(self.volatility_field_str, np.nan)]),
                    errors="coerce",
                ).iloc[0]
            )
            if not (
                np.isfinite(score_float)
                and np.isfinite(rank_float)
                and np.isfinite(volatility_float)
                and volatility_float > 0.0
            ):
                continue

            candidate_row_list.append(
                {
                    "symbol_str": sector_symbol_str,
                    "score_float": score_float,
                    "rank_float": rank_float,
                    "volatility_float": volatility_float,
                }
            )

        if len(candidate_row_list) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = pd.DataFrame(candidate_row_list).sort_values(
            by=["rank_float", "symbol_str"],
            ascending=[True, True],
            kind="mergesort",
        )
        held_sector_set = set(self._current_held_sector_list())
        kept_symbol_list = (
            candidate_feature_df[
                candidate_feature_df["symbol_str"].isin(held_sector_set)
                & (candidate_feature_df["rank_float"] <= float(self.config.hold_rank_int))
            ]["symbol_str"]
            .astype(str)
            .tolist()
        )

        slot_count_int = max(0, self.config.buy_rank_int - len(kept_symbol_list))
        buy_symbol_list = (
            candidate_feature_df[
                (~candidate_feature_df["symbol_str"].isin(kept_symbol_list))
                & (candidate_feature_df["rank_float"] <= float(self.config.buy_rank_int))
            ]["symbol_str"]
            .astype(str)
            .tolist()[:slot_count_int]
        )
        selected_symbol_list = (kept_symbol_list + buy_symbol_list)[: self.config.buy_rank_int]
        if len(selected_symbol_list) == 0:
            return pd.Series(dtype=float)

        selected_feature_df = candidate_feature_df.set_index("symbol_str").loc[selected_symbol_list]
        inverse_volatility_ser = 1.0 / selected_feature_df["volatility_float"].astype(float)
        inverse_volatility_sum_float = float(inverse_volatility_ser.sum())
        if not np.isfinite(inverse_volatility_sum_float) or inverse_volatility_sum_float <= 0.0:
            return pd.Series(dtype=float)

        target_weight_ser = inverse_volatility_ser / inverse_volatility_sum_float
        target_weight_ser = target_weight_ser * exposure_multiplier_float
        target_weight_ser.index = target_weight_ser.index.astype(str)
        return target_weight_ser.astype(float)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The Monday decision close must be the engine's
        # previous_bar. If this fails, the strategy is no longer doing
        # Monday-close to next-open execution.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close_row_ser)
        full_target_weight_ser = target_weight_ser.reindex(
            list(self.sector_symbol_tuple),
            fill_value=0.0,
        ).astype(float)
        if float(full_target_weight_ser.sum()) > 1.0 + 1e-12:
            raise RuntimeError(f"Target weights exceed 100% on {self.previous_bar}.")

        self.rebalance_weight_df.loc[pd.Timestamp(self.current_bar), list(self.sector_symbol_tuple)] = (
            full_target_weight_ser.reindex(list(self.sector_symbol_tuple)).to_numpy(dtype=float)
        )

        target_symbol_set = set(full_target_weight_ser[full_target_weight_ser > 0.0].index.astype(str))
        current_position_ser = self.get_positions().reindex(
            list(self.sector_symbol_tuple),
            fill_value=0.0,
        )

        for sector_symbol_str in self.sector_symbol_tuple:
            current_share_float = float(current_position_ser.loc[sector_symbol_str])
            if current_share_float == 0.0 or sector_symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                sector_symbol_str,
                0.0,
                trade_id=self._ensure_trade_id_int(sector_symbol_str),
            )
            self.current_trade_id_map[sector_symbol_str] = default_trade_id_int()

        for sector_symbol_str in full_target_weight_ser[full_target_weight_ser > 0.0].index.astype(str):
            open_price_float = float(open_price_ser.get(sector_symbol_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for target asset {sector_symbol_str} on {self.current_bar}.")

            self._ensure_trade_id_int(sector_symbol_str)
            self.order_target_percent(
                sector_symbol_str,
                float(full_target_weight_ser.loc[sector_symbol_str]),
                trade_id=self.current_trade_id_map[sector_symbol_str],
            )

    def finalize(self, current_data_df: pd.DataFrame):
        self.daily_target_weights = build_daily_target_weight_df(
            rebalance_weight_df=self.rebalance_weight_df,
            execution_index=pd.DatetimeIndex(current_data_df.index),
            sector_symbol_tuple=self.sector_symbol_tuple,
        )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> WeeklyRiskManagedSectorMomentumStrategy:
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

    pricing_data_df, rebalance_schedule_df = get_weekly_sector_momentum_data(config=config_obj)
    strategy_obj = WeeklyRiskManagedSectorMomentumStrategy(
        name="strategy_mo_weekly_sector_momentum",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
    )

    # *** CRITICAL*** Keep pre-start history for returns, volatility, and
    # SPY SMA200, but execute only from the requested backtest start date.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
