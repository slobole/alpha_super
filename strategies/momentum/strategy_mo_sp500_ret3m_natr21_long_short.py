"""
Research-only SP500 cross-sectional 3-month return / NATR21 long-short model.

Core formulas
-------------
For stock i on decision date t:

    return_3m_{i,t}
        = Close_{i,t} / Close_{i,t-63} - 1

    prior_close_{i,d}
        = Close_{i,d-1}

    TR_{i,d}
        = max(
            High_{i,d} - Low_{i,d},
            abs(High_{i,d} - prior_close_{i,d}),
            abs(Low_{i,d} - prior_close_{i,d})
        )

    ATR21_{i,t}
        = mean(TR_{i,t-20:t})

    NATR21_{i,t}
        = 100 * ATR21_{i,t} / Close_{i,t}

    rank_score_{i,t}
        = return_3m_{i,t} / NATR21_{i,t}

Selection on decision date t:

    eligible_{i,t}
        = 1[PIT_SP500_{i,t} = 1 and rank_score_{i,t} is finite]

    long_set_t
        = top quantile_fraction of eligible names by rank_score

    short_set_t
        = bottom quantile_fraction of eligible names by rank_score

    target_weight_{i,t}
        = +(gross_exposure / 2) / len(long_set_t)    if i in long_set_t
        = -(gross_exposure / 2) / len(short_set_t)   if i in short_set_t
        = 0                                          otherwise

Execution mapping:

    decision_date_t
        = actual last tradable close of the month, quarter, or year

    execution_date_t
        = next tradable open after decision_date_t

This module is intentionally research-only. It does not model borrow costs,
locates, recalls, or margin constraints for short positions.
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
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import audit_pit_universe_df


RETURN_LOOKBACK_DAY_INT = 63
NATR_WINDOW_INT = 21
DEFAULT_QUANTILE_FRACTION_FLOAT = 0.20
DEFAULT_GROSS_EXPOSURE_FLOAT = 1.0
REBALANCE_FREQUENCY_MONTHLY_STR = "monthly"
REBALANCE_FREQUENCY_QUARTERLY_STR = "quarterly"
REBALANCE_FREQUENCY_ANNUAL_STR = "annual"
VALID_REBALANCE_FREQUENCY_SET = frozenset(
    {
        REBALANCE_FREQUENCY_MONTHLY_STR,
        REBALANCE_FREQUENCY_QUARTERLY_STR,
        REBALANCE_FREQUENCY_ANNUAL_STR,
    }
)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class Sp500Ret3mNatr21LongShortConfig:
    indexname_str: str = "S&P 500"
    benchmark_symbol_str: str = "SPY"
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    return_lookback_day_int: int = RETURN_LOOKBACK_DAY_INT
    natr_window_int: int = NATR_WINDOW_INT
    quantile_fraction_float: float = DEFAULT_QUANTILE_FRACTION_FLOAT
    gross_exposure_float: float = DEFAULT_GROSS_EXPOSURE_FLOAT
    rebalance_frequency_str: str = REBALANCE_FREQUENCY_MONTHLY_STR
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.return_lookback_day_int <= 0:
            raise ValueError("return_lookback_day_int must be positive.")
        if self.natr_window_int <= 0:
            raise ValueError("natr_window_int must be positive.")
        if not 0.0 < self.quantile_fraction_float <= 0.5:
            raise ValueError("quantile_fraction_float must be in (0.0, 0.5].")
        if self.gross_exposure_float <= 0.0:
            raise ValueError("gross_exposure_float must be positive.")
        if self.rebalance_frequency_str not in VALID_REBALANCE_FREQUENCY_SET:
            raise ValueError(
                "rebalance_frequency_str must be one of "
                f"{sorted(VALID_REBALANCE_FREQUENCY_SET)}, got {self.rebalance_frequency_str!r}."
            )
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = Sp500Ret3mNatr21LongShortConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_GROSS_EXPOSURE_FLOAT",
    "DEFAULT_QUANTILE_FRACTION_FLOAT",
    "NATR_WINDOW_INT",
    "REBALANCE_FREQUENCY_ANNUAL_STR",
    "REBALANCE_FREQUENCY_MONTHLY_STR",
    "REBALANCE_FREQUENCY_QUARTERLY_STR",
    "RETURN_LOOKBACK_DAY_INT",
    "Sp500Ret3mNatr21LongShortConfig",
    "Sp500Ret3mNatr21LongShortStrategy",
    "VALID_REBALANCE_FREQUENCY_SET",
    "compute_ret3m_natr21_signal_tables",
    "get_asof_universe_membership_ser",
    "get_rebalance_decision_close_df",
    "get_sp500_ret3m_natr21_long_short_data",
    "map_decision_dates_to_next_open_schedule_df",
    "run_frequency_comparison",
    "run_variant",
]


def get_asof_universe_membership_ser(
    universe_df: pd.DataFrame,
    decision_date_ts: pd.Timestamp,
) -> pd.Series:
    if len(universe_df) == 0:
        raise RuntimeError("universe_df is empty.")

    sorted_universe_df = universe_df.sort_index()
    # *** CRITICAL *** PIT universe membership may lag the newest price date.
    # Use only the latest universe row available on or before decision_t.
    universe_row_int = int(
        sorted_universe_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if universe_row_int < 0:
        raise RuntimeError(f"universe_df has no row on or before decision date {decision_date_ts}.")
    return sorted_universe_df.iloc[universe_row_int]


def _period_code_str(rebalance_frequency_str: str) -> str:
    if rebalance_frequency_str == REBALANCE_FREQUENCY_MONTHLY_STR:
        return "M"
    if rebalance_frequency_str == REBALANCE_FREQUENCY_QUARTERLY_STR:
        return "Q"
    if rebalance_frequency_str == REBALANCE_FREQUENCY_ANNUAL_STR:
        return "Y"
    raise ValueError(
        "rebalance_frequency_str must be one of "
        f"{sorted(VALID_REBALANCE_FREQUENCY_SET)}, got {rebalance_frequency_str!r}."
    )


def _expected_period_end_ts(last_available_ts: pd.Timestamp, rebalance_frequency_str: str) -> pd.Timestamp:
    if rebalance_frequency_str == REBALANCE_FREQUENCY_MONTHLY_STR:
        return pd.Timestamp(last_available_ts + pd.offsets.BMonthEnd(0)).normalize()
    if rebalance_frequency_str == REBALANCE_FREQUENCY_QUARTERLY_STR:
        return pd.Timestamp(last_available_ts + pd.offsets.BQuarterEnd(0)).normalize()
    if rebalance_frequency_str == REBALANCE_FREQUENCY_ANNUAL_STR:
        return pd.Timestamp(last_available_ts + pd.offsets.BYearEnd(0)).normalize()
    raise ValueError(
        "rebalance_frequency_str must be one of "
        f"{sorted(VALID_REBALANCE_FREQUENCY_SET)}, got {rebalance_frequency_str!r}."
    )


def get_rebalance_decision_close_df(
    price_close_df: pd.DataFrame,
    rebalance_frequency_str: str = REBALANCE_FREQUENCY_MONTHLY_STR,
) -> pd.DataFrame:
    """
    Collapse daily closes to the actual last tradable close of each rebalance period.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")

    period_code_str = _period_code_str(rebalance_frequency_str=rebalance_frequency_str)
    # *** CRITICAL *** Rebalance decisions must use the actual last tradable
    # close in each completed period, not a synthetic calendar timestamp.
    decision_date_ser = pd.Series(
        price_close_df.index,
        index=price_close_df.index.to_period(period_code_str),
    ).groupby(level=0).max()

    last_available_ts = pd.Timestamp(price_close_df.index[-1])
    expected_period_end_ts = _expected_period_end_ts(
        last_available_ts=last_available_ts,
        rebalance_frequency_str=rebalance_frequency_str,
    )
    if (
        len(decision_date_ser) > 0
        and pd.Timestamp(decision_date_ser.iloc[-1]) == last_available_ts
        and expected_period_end_ts != last_available_ts.normalize()
    ):
        decision_date_ser = decision_date_ser.iloc[:-1]

    decision_date_idx = pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")
    decision_close_df = price_close_df.loc[decision_date_idx].copy()
    decision_close_df.index = decision_date_idx
    return decision_close_df


def compute_ret3m_natr21_signal_tables(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    config: Sp500Ret3mNatr21LongShortConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # *** CRITICAL *** The 3-month return is a trailing 63-trading-day
    # close-to-close return known at decision_t.
    return_3m_df = (
        price_close_df / price_close_df.shift(config.return_lookback_day_int)
    ) - 1.0

    # *** CRITICAL *** prior close alignment for true range must use shift(1)
    # so ATR/NATR is strictly trailing.
    prior_close_df = price_close_df.shift(1)
    true_range_df = (price_high_df - price_low_df).combine(
        (price_high_df - prior_close_df).abs(),
        np.maximum,
    )
    true_range_df = true_range_df.combine(
        (price_low_df - prior_close_df).abs(),
        np.maximum,
    )

    # *** CRITICAL *** NATR21 must remain a trailing rolling mean of past true
    # range values divided by the decision close. No future bars are allowed.
    atr_21_df = true_range_df.rolling(
        window=config.natr_window_int,
        min_periods=config.natr_window_int,
    ).mean()
    natr_21_df = 100.0 * atr_21_df / price_close_df

    rank_score_df = return_3m_df / natr_21_df
    rank_score_df = rank_score_df.replace([np.inf, -np.inf], np.nan)

    decision_close_df = get_rebalance_decision_close_df(
        price_close_df=price_close_df,
        rebalance_frequency_str=config.rebalance_frequency_str,
    )
    decision_date_index = pd.DatetimeIndex(decision_close_df.index)
    return_3m_decision_df = return_3m_df.reindex(decision_date_index)
    natr_21_decision_df = natr_21_df.reindex(decision_date_index)
    rank_score_decision_df = rank_score_df.reindex(decision_date_index)

    valid_score_bool_ser = rank_score_decision_df.notna().any(axis=1)
    valid_decision_index = decision_date_index[valid_score_bool_ser]
    decision_close_df = decision_close_df.reindex(valid_decision_index)
    return_3m_decision_df = return_3m_decision_df.reindex(valid_decision_index)
    natr_21_decision_df = natr_21_decision_df.reindex(valid_decision_index)
    rank_score_decision_df = rank_score_decision_df.reindex(valid_decision_index)
    return (
        decision_close_df,
        return_3m_decision_df,
        natr_21_decision_df,
        rank_score_decision_df,
    )


def map_decision_dates_to_next_open_schedule_df(
    decision_date_index: pd.DatetimeIndex,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(execution_index) < 2:
        raise ValueError("execution_index must contain at least two trading dates.")
    if len(decision_date_index) == 0:
        raise ValueError("decision_date_index must not be empty.")

    execution_index = pd.DatetimeIndex(execution_index).sort_values()
    decision_date_index = pd.DatetimeIndex(decision_date_index).sort_values()

    rebalance_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date_ts in decision_date_index:
        execution_insert_int = int(execution_index.searchsorted(pd.Timestamp(decision_date_ts), side="right"))
        if execution_insert_int >= len(execution_index):
            continue

        # *** CRITICAL *** Decisions execute strictly on the next tradable
        # open after the decision close, never on the same bar.
        execution_date_ts = pd.Timestamp(execution_index[execution_insert_int])
        rebalance_schedule_map[execution_date_ts] = pd.Timestamp(decision_date_ts)

    if len(rebalance_schedule_map) == 0:
        raise RuntimeError("No rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def get_sp500_ret3m_natr21_long_short_data(
    config: Sp500Ret3mNatr21LongShortConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        end_date_ts = pd.Timestamp(config.end_date_str)
        active_universe_df = active_universe_df.loc[active_universe_df.index <= end_date_ts]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError("No active S&P 500 universe symbols were found for the requested backtest window.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(audited_universe_df.columns.tolist() + [config.benchmark_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    close_symbol_list = audited_universe_df.columns.tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "High")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Low")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)

    (
        decision_close_df,
        _return_3m_decision_df,
        _natr_21_decision_df,
        _rank_score_decision_df,
    ) = compute_ret3m_natr21_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        config=config,
    )
    rebalance_schedule_df = map_decision_dates_to_next_open_schedule_df(
        decision_date_index=pd.DatetimeIndex(decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class Sp500Ret3mNatr21LongShortStrategy(Strategy):
    """
    SP500 PIT long-short rank strategy using return_3m / NATR21.

    The default exposure is conservative for a short-capable research path:
    +50% long and -50% short, for 100% gross and 0% net target exposure.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        benchmark_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        return_lookback_day_int: int = RETURN_LOOKBACK_DAY_INT,
        natr_window_int: int = NATR_WINDOW_INT,
        quantile_fraction_float: float = DEFAULT_QUANTILE_FRACTION_FLOAT,
        gross_exposure_float: float = DEFAULT_GROSS_EXPOSURE_FLOAT,
        rebalance_frequency_str: str = REBALANCE_FREQUENCY_MONTHLY_STR,
    ):
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
        if benchmark_symbol_str not in benchmarks:
            raise ValueError("benchmarks must include benchmark_symbol_str.")
        if return_lookback_day_int <= 0:
            raise ValueError("return_lookback_day_int must be positive.")
        if natr_window_int <= 0:
            raise ValueError("natr_window_int must be positive.")
        if not 0.0 < quantile_fraction_float <= 0.5:
            raise ValueError("quantile_fraction_float must be in (0.0, 0.5].")
        if gross_exposure_float <= 0.0:
            raise ValueError("gross_exposure_float must be positive.")
        if rebalance_frequency_str not in VALID_REBALANCE_FREQUENCY_SET:
            raise ValueError(
                "rebalance_frequency_str must be one of "
                f"{sorted(VALID_REBALANCE_FREQUENCY_SET)}, got {rebalance_frequency_str!r}."
            )

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.benchmark_symbol_str = str(benchmark_symbol_str)
        self.return_lookback_day_int = int(return_lookback_day_int)
        self.natr_window_int = int(natr_window_int)
        self.quantile_fraction_float = float(quantile_fraction_float)
        self.gross_exposure_float = float(gross_exposure_float)
        self.rebalance_frequency_str = str(rebalance_frequency_str)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in signal_data_df.columns.get_level_values(0).unique()
            if str(symbol_str) not in self._benchmarks
        ]
        if len(tradeable_symbol_list) == 0:
            raise RuntimeError("No tradeable stock symbols were found in pricing_data.")

        price_close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        price_high_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        price_low_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)

        helper_config = replace(
            DEFAULT_CONFIG,
            benchmark_symbol_str=self.benchmark_symbol_str,
            return_lookback_day_int=self.return_lookback_day_int,
            natr_window_int=self.natr_window_int,
            quantile_fraction_float=self.quantile_fraction_float,
            gross_exposure_float=self.gross_exposure_float,
            rebalance_frequency_str=self.rebalance_frequency_str,
        )
        (
            _decision_close_df,
            return_3m_decision_df,
            natr_21_decision_df,
            rank_score_decision_df,
        ) = compute_ret3m_natr21_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            config=helper_config,
        )

        return_3m_aligned_df = return_3m_decision_df.reindex(signal_data_df.index)
        natr_21_aligned_df = natr_21_decision_df.reindex(signal_data_df.index)
        rank_score_aligned_df = rank_score_decision_df.reindex(signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            f"return_{self.return_lookback_day_int}d_ser": return_3m_aligned_df,
            f"natr_{self.natr_window_int}_ser": natr_21_aligned_df,
            "rank_score_ser": rank_score_aligned_df,
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before rebalances.")

        candidate_feature_df = close_row_ser.unstack()
        if "rank_score_ser" not in candidate_feature_df.columns:
            return pd.Series(dtype=float)

        universe_member_ser = get_asof_universe_membership_ser(
            universe_df=self.universe_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) < 2:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.assign(
            rank_score_float=pd.to_numeric(candidate_feature_df["rank_score_ser"], errors="coerce"),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_score_mask_vec = np.isfinite(
            candidate_feature_df["rank_score_float"].to_numpy(dtype=float)
        )
        candidate_feature_df = candidate_feature_df.loc[finite_score_mask_vec]
        if len(candidate_feature_df) < 2:
            return pd.Series(dtype=float)

        selection_count_int = int(np.floor(len(candidate_feature_df) * self.quantile_fraction_float))
        selection_count_int = max(1, selection_count_int)
        selection_count_int = min(selection_count_int, len(candidate_feature_df) // 2)
        if selection_count_int < 1:
            return pd.Series(dtype=float)

        ranked_feature_df = candidate_feature_df.sort_values(
            by=["rank_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        long_symbol_list = ranked_feature_df.iloc[:selection_count_int].index.astype(str).tolist()
        short_symbol_list = ranked_feature_df.iloc[-selection_count_int:].index.astype(str).tolist()

        long_weight_float = (self.gross_exposure_float / 2.0) / float(len(long_symbol_list))
        short_weight_float = -(self.gross_exposure_float / 2.0) / float(len(short_symbol_list))
        target_weight_ser = pd.concat(
            [
                pd.Series(long_weight_float, index=long_symbol_list, dtype=float),
                pd.Series(short_weight_float, index=short_symbol_list, dtype=float),
            ]
        )
        return target_weight_ser.sort_index()

    @staticmethod
    def _position_sign_int(position_float: float) -> int:
        if position_float > 0.0:
            return 1
        if position_float < 0.0:
            return -1
        return 0

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL *** The scheduled decision close must equal
        # previous_bar exactly, otherwise signals and next-open execution drift.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close)
        target_symbol_set = set(target_weight_ser.index.astype(str))

        current_position_ser = self.get_positions()
        active_position_ser = current_position_ser[current_position_ser != 0]
        for symbol_str, position_float in active_position_ser.items():
            target_weight_float = float(target_weight_ser.get(symbol_str, 0.0))
            current_sign_int = self._position_sign_int(float(position_float))
            target_sign_int = self._position_sign_int(target_weight_float)
            if symbol_str in target_symbol_set and current_sign_int == target_sign_int:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )
            if symbol_str not in target_symbol_set:
                self.current_trade_map.pop(symbol_str, None)

        for symbol_str, target_weight_float in target_weight_ser.items():
            current_share_float = float(current_position_ser.get(symbol_str, 0.0))
            current_sign_int = self._position_sign_int(current_share_float)
            target_sign_int = self._position_sign_int(float(target_weight_float))
            if current_sign_int != target_sign_int:
                self.trade_id_int += 1
                self.current_trade_map[str(symbol_str)] = self.trade_id_int
            elif str(symbol_str) not in self.current_trade_map:
                self.trade_id_int += 1
                self.current_trade_map[str(symbol_str)] = self.trade_id_int

            self.order_target_percent(
                str(symbol_str),
                float(target_weight_float),
                trade_id=self.current_trade_map[str(symbol_str)],
            )


def _build_config(
    rebalance_frequency_str: str | None = None,
    quantile_fraction_float: float | None = None,
    gross_exposure_float: float | None = None,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> Sp500Ret3mNatr21LongShortConfig:
    return replace(
        DEFAULT_CONFIG,
        rebalance_frequency_str=(
            DEFAULT_CONFIG.rebalance_frequency_str
            if rebalance_frequency_str is None
            else str(rebalance_frequency_str)
        ),
        quantile_fraction_float=(
            DEFAULT_CONFIG.quantile_fraction_float
            if quantile_fraction_float is None
            else float(quantile_fraction_float)
        ),
        gross_exposure_float=(
            DEFAULT_CONFIG.gross_exposure_float
            if gross_exposure_float is None
            else float(gross_exposure_float)
        ),
        backtest_start_date_str=(
            DEFAULT_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else str(backtest_start_date_str)
        ),
        capital_base_float=(
            DEFAULT_CONFIG.capital_base_float
            if capital_base_float is None
            else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )


def _make_strategy(
    config_obj: Sp500Ret3mNatr21LongShortConfig,
    rebalance_schedule_df: pd.DataFrame,
) -> Sp500Ret3mNatr21LongShortStrategy:
    return Sp500Ret3mNatr21LongShortStrategy(
        name=f"strategy_mo_sp500_ret3m_natr21_long_short_{config_obj.rebalance_frequency_str}",
        benchmarks=[config_obj.benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        benchmark_symbol_str=config_obj.benchmark_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        return_lookback_day_int=config_obj.return_lookback_day_int,
        natr_window_int=config_obj.natr_window_int,
        quantile_fraction_float=config_obj.quantile_fraction_float,
        gross_exposure_float=config_obj.gross_exposure_float,
        rebalance_frequency_str=config_obj.rebalance_frequency_str,
    )


def run_variant(
    rebalance_frequency_str: str = REBALANCE_FREQUENCY_MONTHLY_STR,
    quantile_fraction_float: float | None = None,
    gross_exposure_float: float | None = None,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> Sp500Ret3mNatr21LongShortStrategy:
    config_obj = _build_config(
        rebalance_frequency_str=rebalance_frequency_str,
        quantile_fraction_float=quantile_fraction_float,
        gross_exposure_float=gross_exposure_float,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_sp500_ret3m_natr21_long_short_data(config_obj)

    strategy_obj = _make_strategy(
        config_obj=config_obj,
        rebalance_schedule_df=rebalance_schedule_df,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL *** Research backtests keep full pre-start history for
    # trailing 3-month return and NATR21, while execution starts at the
    # requested comparison date.
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


def run_frequency_comparison(
    frequency_list: Sequence[str] = (
        REBALANCE_FREQUENCY_MONTHLY_STR,
        REBALANCE_FREQUENCY_QUARTERLY_STR,
        REBALANCE_FREQUENCY_ANNUAL_STR,
    ),
    show_display_bool: bool = False,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    result_row_list: list[dict[str, object]] = []
    for frequency_str in frequency_list:
        strategy_obj = run_variant(
            rebalance_frequency_str=str(frequency_str),
            show_display_bool=show_display_bool,
            save_results_bool=save_results_bool,
            output_dir_str=output_dir_str,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
        )
        strategy_summary_ser = strategy_obj.summary["Strategy"].copy()
        transaction_df = strategy_obj.get_transactions()
        result_row_list.append(
            {
                "rebalance_frequency": str(frequency_str),
                "strategy_name": strategy_obj.name,
                "final_equity": float(strategy_summary_ser.get("Final [$]", np.nan)),
                "return_ann_pct": strategy_summary_ser.get("Return (Ann.) [%]", np.nan),
                "sharpe": strategy_summary_ser.get("Sharpe Ratio", np.nan),
                "max_drawdown_pct": strategy_summary_ser.get("Max. Drawdown [%]", np.nan),
                "mar": strategy_summary_ser.get("MAR Ratio", np.nan),
                "trade_count": int(len(transaction_df)),
            }
        )

    comparison_df = pd.DataFrame(result_row_list)
    return comparison_df.set_index("rebalance_frequency")


if __name__ == "__main__":
    run_variant()
