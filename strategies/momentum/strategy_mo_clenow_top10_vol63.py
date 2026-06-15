"""
Clenow-style Top 10 momentum selector with 63-day volatility normalization.

Core formulas
-------------
For stock i on decision date t:

    log_price_{i,j}
        = alpha_i + beta_i * j + epsilon_{i,j}, over j = t-89:t

    annualized_slope_{i,t}
        = exp(beta_i * 252) - 1

    clenow_score_{i,t}
        = annualized_slope_{i,t} * R2_{i,t}

    vol63_{i,t}
        = stdev(return_{i,t-62:t}) * sqrt(252)

    final_score_{i,t}
        = clenow_score_{i,t} / vol63_{i,t}

Selection:

    selected_t
        = top 10 eligible PIT index members by final_score_{i,t}

    target_weight_{i,t}
        = 1 / 10 if i in selected_t and the matching index regime passes
        = unchanged for already-held selected names if the regime fails

Execution:

    decision_date_t
        = actual last tradable close of the month

    execution_date_t
        = next tradable open after the month-end decision close
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
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
)


TRADING_DAYS_PER_YEAR_FLOAT = 252.0
CLENOW_LOOKBACK_INT = 90
VOL_WINDOW_INT = 63
STOCK_TREND_WINDOW_INT = 100
REGIME_TREND_WINDOW_INT = 200
GAP_WINDOW_INT = 90
GAP_THRESHOLD_FLOAT = 0.15
MAX_POSITIONS_INT = 10


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class ClenowTop10Vol63Config:
    variant_key_str: str
    indexname_str: str
    regime_symbol_str: str
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    max_positions_int: int = MAX_POSITIONS_INT
    clenow_lookback_int: int = CLENOW_LOOKBACK_INT
    vol_window_int: int = VOL_WINDOW_INT
    stock_trend_window_int: int = STOCK_TREND_WINDOW_INT
    regime_trend_window_int: int = REGIME_TREND_WINDOW_INT
    gap_window_int: int = GAP_WINDOW_INT
    gap_threshold_float: float = GAP_THRESHOLD_FLOAT
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.variant_key_str:
            raise ValueError("variant_key_str must not be empty.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.clenow_lookback_int <= 2:
            raise ValueError("clenow_lookback_int must be greater than 2.")
        if self.vol_window_int <= 1:
            raise ValueError("vol_window_int must be greater than 1.")
        if self.stock_trend_window_int <= 0:
            raise ValueError("stock_trend_window_int must be positive.")
        if self.regime_trend_window_int <= 0:
            raise ValueError("regime_trend_window_int must be positive.")
        if self.gap_window_int <= 0:
            raise ValueError("gap_window_int must be positive.")
        if self.gap_threshold_float <= 0.0:
            raise ValueError("gap_threshold_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


SP500_CONFIG = ClenowTop10Vol63Config(
    variant_key_str="sp500",
    indexname_str="S&P 500",
    regime_symbol_str="$SPX",
)
NASDAQ100_CONFIG = ClenowTop10Vol63Config(
    variant_key_str="nasdaq100",
    indexname_str="Nasdaq 100",
    regime_symbol_str="$NDX",
)
RUSSELL1000_CONFIG = ClenowTop10Vol63Config(
    variant_key_str="russell1000",
    indexname_str="Russell 1000",
    regime_symbol_str="$RUI",
)
DEFAULT_CONFIG = SP500_CONFIG
CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_CONFIG.variant_key_str: SP500_CONFIG,
    NASDAQ100_CONFIG.variant_key_str: NASDAQ100_CONFIG,
    RUSSELL1000_CONFIG.variant_key_str: RUSSELL1000_CONFIG,
}


__all__ = [
    "CLENOW_LOOKBACK_INT",
    "CONFIG_BY_VARIANT_KEY_DICT",
    "DEFAULT_CONFIG",
    "GAP_THRESHOLD_FLOAT",
    "GAP_WINDOW_INT",
    "MAX_POSITIONS_INT",
    "NASDAQ100_CONFIG",
    "REGIME_TREND_WINDOW_INT",
    "RUSSELL1000_CONFIG",
    "SP500_CONFIG",
    "STOCK_TREND_WINDOW_INT",
    "VOL_WINDOW_INT",
    "ClenowTop10Vol63Config",
    "ClenowTop10Vol63Strategy",
    "compute_clenow_top10_vol63_signal_tables",
    "get_clenow_top10_vol63_data",
    "map_month_end_rebalance_schedule_df",
    "run_all_variants",
    "run_variant",
]


def map_month_end_rebalance_schedule_df(
    decision_date_index: pd.DatetimeIndex,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(execution_index) < 2:
        raise ValueError("execution_index must contain at least two trading dates.")
    if len(decision_date_index) == 0:
        raise ValueError("decision_date_index must not be empty.")

    execution_index = pd.DatetimeIndex(execution_index).sort_values()
    decision_date_idx = pd.DatetimeIndex(decision_date_index).sort_values()
    decision_date_set = set(decision_date_idx)
    month_end_decision_ser = pd.Series(
        decision_date_idx,
        index=decision_date_idx.to_period("M"),
    ).groupby(level=0).max()
    month_end_decision_set = set(pd.DatetimeIndex(month_end_decision_ser.to_numpy()))
    rebalance_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date_ts in month_end_decision_set:
        execution_insert_int = int(execution_index.searchsorted(pd.Timestamp(decision_date_ts), side="right"))
        if execution_insert_int >= len(execution_index):
            continue
        if decision_date_ts not in decision_date_set:
            continue

        execution_date_ts = pd.Timestamp(execution_index[execution_insert_int])
        # *** CRITICAL*** Monthly rebalance intent is formed on the actual
        # last tradable close of the month and executed strictly at the next
        # tradable open. Same-bar execution would be a lookahead bug.
        rebalance_schedule_map[pd.Timestamp(execution_date_ts)] = decision_date_ts

    if len(rebalance_schedule_map) == 0:
        raise RuntimeError("No month-end rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def compute_clenow_top10_vol63_signal_tables(
    price_close_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: ClenowTop10Vol63Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")

    decision_date_index = pd.DatetimeIndex(price_close_df.index, name="decision_date_ts")
    annualized_slope_df = pd.DataFrame(
        np.nan,
        index=decision_date_index,
        columns=price_close_df.columns,
        dtype=float,
    )
    r2_df = annualized_slope_df.copy()

    log_price_df = np.log(price_close_df.where(price_close_df > 0.0))
    for decision_date_ts in decision_date_index:
        decision_pos_int = int(price_close_df.index.get_loc(decision_date_ts))
        start_pos_int = decision_pos_int - int(config.clenow_lookback_int) + 1
        if start_pos_int < 0:
            continue

        # *** CRITICAL*** The 90-day Clenow regression window ends at the
        # decision close. It is tradable only because execution is next open.
        log_window_df = log_price_df.iloc[start_pos_int : decision_pos_int + 1]
        valid_symbol_list = log_window_df.columns[log_window_df.notna().all(axis=0)].astype(str).tolist()
        if len(valid_symbol_list) == 0:
            continue

        log_window_mat = log_window_df.loc[:, valid_symbol_list].to_numpy(dtype=float)
        time_vec = np.arange(config.clenow_lookback_int, dtype=float)
        centered_time_vec = time_vec - float(time_vec.mean())
        time_ss_float = float(np.dot(centered_time_vec, centered_time_vec))

        slope_vec = centered_time_vec @ log_window_mat / time_ss_float
        centered_log_window_mat = log_window_mat - log_window_mat.mean(axis=0)
        total_ss_vec = np.sum(centered_log_window_mat * centered_log_window_mat, axis=0)
        regression_ss_vec = (slope_vec * slope_vec) * time_ss_float
        r2_vec = np.divide(
            regression_ss_vec,
            total_ss_vec,
            out=np.full_like(regression_ss_vec, np.nan, dtype=float),
            where=total_ss_vec > 0.0,
        )

        annualized_slope_vec = np.exp(slope_vec * TRADING_DAYS_PER_YEAR_FLOAT) - 1.0
        annualized_slope_df.loc[decision_date_ts, valid_symbol_list] = annualized_slope_vec
        r2_df.loc[decision_date_ts, valid_symbol_list] = np.clip(r2_vec, 0.0, 1.0)

    clenow_score_df = annualized_slope_df * r2_df

    # *** CRITICAL*** 63-day volatility uses trailing close-to-close returns
    # through decision close only. It must not include the execution open.
    daily_return_df = price_close_df.pct_change(fill_method=None)
    vol63_df = daily_return_df.rolling(
        window=config.vol_window_int,
        min_periods=config.vol_window_int,
    ).std() * np.sqrt(TRADING_DAYS_PER_YEAR_FLOAT)

    # *** CRITICAL*** The stock trend filter is a trailing close>SMA test at
    # the decision close.
    stock_sma_df = price_close_df.rolling(
        window=config.stock_trend_window_int,
        min_periods=config.stock_trend_window_int,
    ).mean()
    stock_trend_pass_df = price_close_df > stock_sma_df

    # *** CRITICAL*** The gap filter uses only trailing close-to-close moves
    # known at the decision close. The 15% threshold is a research
    # interpretation of the book's recent-gap exclusion.
    max_abs_return_df = daily_return_df.abs().rolling(
        window=config.gap_window_int,
        min_periods=config.gap_window_int,
    ).max()
    gap_pass_df = max_abs_return_df < float(config.gap_threshold_float)

    # *** CRITICAL*** The regime filter is matched to the traded universe's
    # index and is used only to block new buys, not to force liquidation.
    regime_sma_ser = regime_close_ser.rolling(
        window=config.regime_trend_window_int,
        min_periods=config.regime_trend_window_int,
    ).mean()
    regime_pass_ser = regime_close_ser > regime_sma_ser

    vol_norm_score_df = clenow_score_df / vol63_df
    vol_norm_score_df = vol_norm_score_df.replace([np.inf, -np.inf], np.nan)

    valid_signal_bool_ser = (
        vol_norm_score_df.notna().any(axis=1)
        & stock_trend_pass_df.notna().any(axis=1)
        & gap_pass_df.notna().any(axis=1)
        & regime_sma_ser.notna()
        & regime_pass_ser.notna()
    )
    valid_decision_index = decision_date_index[valid_signal_bool_ser]
    return (
        annualized_slope_df.reindex(valid_decision_index),
        r2_df.reindex(valid_decision_index),
        vol63_df.reindex(valid_decision_index),
        stock_trend_pass_df.reindex(valid_decision_index),
        regime_sma_ser.reindex(valid_decision_index),
        regime_pass_ser.reindex(valid_decision_index),
        gap_pass_df.reindex(valid_decision_index),
    )


def get_clenow_top10_vol63_data(
    config: ClenowTop10Vol63Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        end_date_ts = pd.Timestamp(config.end_date_str)
        active_universe_df = active_universe_df.loc[active_universe_df.index <= end_date_ts]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(f"No active {config.indexname_str} symbols were found for the requested window.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=[config.regime_symbol_str],
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

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + [config.regime_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    close_symbol_list = audited_universe_df.columns.astype(str).tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    regime_close_ser = pricing_data_df[(config.regime_symbol_str, "Close")].astype(float)

    (
        annualized_slope_df,
        _r2_df,
        _vol63_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _gap_pass_df,
    ) = compute_clenow_top10_vol63_signal_tables(
        price_close_df=price_close_df,
        regime_close_ser=regime_close_ser,
        config=config,
    )
    rebalance_schedule_df = map_month_end_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(annualized_slope_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class ClenowTop10Vol63Strategy(Strategy):
    """
    Long-only monthly Top 10 Clenow momentum strategy.

    New buys are allowed only when the matching index close is above its
    200-day SMA. Existing names are still sold if they leave the Top 10 or
    fail stock-level filters.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        config: ClenowTop10Vol63Config,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")
        if config.regime_symbol_str not in benchmarks:
            raise ValueError("benchmarks must include config.regime_symbol_str.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.config = config
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    @property
    def annualized_slope_field_str(self) -> str:
        return f"annualized_slope_{self.config.clenow_lookback_int}_float"

    @property
    def r2_field_str(self) -> str:
        return f"trend_r2_{self.config.clenow_lookback_int}_float"

    @property
    def vol_field_str(self) -> str:
        return f"vol_{self.config.vol_window_int}_float"

    @property
    def final_score_field_str(self) -> str:
        return f"clenow_vol_norm_score_{self.config.clenow_lookback_int}_{self.config.vol_window_int}_float"

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        benchmark_symbol_set = {str(symbol_str) for symbol_str in self._benchmarks}
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in signal_data_df.columns.get_level_values(0).unique()
            if str(symbol_str) not in benchmark_symbol_set
        ]
        if len(tradeable_symbol_list) == 0:
            raise RuntimeError("No tradeable stock symbols were found in pricing_data.")

        price_close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        regime_close_key = (self.config.regime_symbol_str, "Close")
        if regime_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing regime close data for {self.config.regime_symbol_str}.")
        regime_close_ser = signal_data_df[regime_close_key].astype(float)

        (
            annualized_slope_df,
            r2_df,
            vol63_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            gap_pass_df,
        ) = compute_clenow_top10_vol63_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=regime_close_ser,
            config=self.config,
        )
        final_score_df = (annualized_slope_df * r2_df) / vol63_df
        final_score_df = final_score_df.replace([np.inf, -np.inf], np.nan)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            self.annualized_slope_field_str: annualized_slope_df.reindex(signal_data_df.index),
            self.r2_field_str: r2_df.reindex(signal_data_df.index),
            self.vol_field_str: vol63_df.reindex(signal_data_df.index),
            self.final_score_field_str: final_score_df.reindex(signal_data_df.index),
            "stock_trend_pass_bool": stock_trend_pass_df.reindex(signal_data_df.index),
            "gap_pass_bool": gap_pass_df.reindex(signal_data_df.index),
        }
        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (
                    self.config.regime_symbol_str,
                    f"regime_sma_{self.config.regime_trend_window_int}_ser",
                ): regime_sma_ser.reindex(signal_data_df.index),
                (self.config.regime_symbol_str, "regime_pass_bool"): regime_pass_ser.reindex(signal_data_df.index),
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_selected_symbol_list(self, close_row_ser: pd.Series) -> list[str]:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before rebalances.")

        candidate_feature_df = close_row_ser.unstack()
        required_field_list = [
            self.final_score_field_str,
            "stock_trend_pass_bool",
            "gap_pass_bool",
        ]
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return []

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return []

        stock_trend_pass_ser = candidate_feature_df["stock_trend_pass_bool"].where(
            candidate_feature_df["stock_trend_pass_bool"].notna(),
            False,
        ).astype(bool)
        gap_pass_ser = candidate_feature_df["gap_pass_bool"].where(
            candidate_feature_df["gap_pass_bool"].notna(),
            False,
        ).astype(bool)
        candidate_feature_df = candidate_feature_df.assign(
            final_score_float=pd.to_numeric(candidate_feature_df[self.final_score_field_str], errors="coerce"),
            stock_trend_pass_bool=stock_trend_pass_ser,
            gap_pass_bool=gap_pass_ser,
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_score_mask_vec = np.isfinite(candidate_feature_df["final_score_float"].to_numpy(dtype=float))
        eligibility_mask_vec = (
            finite_score_mask_vec
            & (candidate_feature_df["final_score_float"].to_numpy(dtype=float) > 0.0)
            & candidate_feature_df["stock_trend_pass_bool"].to_numpy(dtype=bool)
            & candidate_feature_df["gap_pass_bool"].to_numpy(dtype=bool)
        )
        candidate_feature_df = candidate_feature_df.loc[eligibility_mask_vec].copy()
        if len(candidate_feature_df) == 0:
            return []

        selected_feature_df = candidate_feature_df.sort_values(
            by=["final_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[: self.config.max_positions_int]
        return selected_feature_df.index.astype(str).tolist()

    def is_regime_pass_bool(self, close_row_ser: pd.Series) -> bool:
        candidate_feature_df = close_row_ser.unstack()
        if self.config.regime_symbol_str not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime feature row for {self.config.regime_symbol_str}.")

        regime_pass_value = candidate_feature_df.loc[self.config.regime_symbol_str].get("regime_pass_bool", np.nan)
        return bool(False if pd.isna(regime_pass_value) else regime_pass_value)

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** Monthly schedule must align the previous completed
        # close with this rebalance's decision date. Otherwise the month-end
        # next-open execution contract is broken.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        selected_symbol_list = self.get_selected_symbol_list(close_row_ser=close)
        selected_symbol_set = set(selected_symbol_list)
        regime_pass_bool = self.is_regime_pass_bool(close_row_ser=close)
        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]
        held_symbol_set = set(long_position_ser.index.astype(str).tolist())

        for symbol_str in long_position_ser.index.astype(str):
            if symbol_str in selected_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        if not regime_pass_bool:
            return

        target_weight_float = 1.0 / float(self.config.max_positions_int)
        for symbol_str in selected_symbol_list:
            if symbol_str not in held_symbol_set:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_map[symbol_str],
            )


def _with_run_overrides(
    config: ClenowTop10Vol63Config,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> ClenowTop10Vol63Config:
    if backtest_start_date_str is None and capital_base_float is None and end_date_str is None:
        return config

    return replace(
        config,
        backtest_start_date_str=(
            config.backtest_start_date_str if backtest_start_date_str is None else backtest_start_date_str
        ),
        capital_base_float=(
            config.capital_base_float if capital_base_float is None else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )


def run_variant(
    variant_key_str: str = DEFAULT_CONFIG.variant_key_str,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = None,
) -> ClenowTop10Vol63Strategy:
    if variant_key_str not in CONFIG_BY_VARIANT_KEY_DICT:
        raise ValueError(f"Unknown variant_key_str: {variant_key_str}")

    config_obj = _with_run_overrides(
        config=CONFIG_BY_VARIANT_KEY_DICT[variant_key_str],
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_clenow_top10_vol63_data(config=config_obj)
    strategy_obj = ClenowTop10Vol63Strategy(
        name=f"strategy_mo_clenow_top10_vol63_monthly_{config_obj.variant_key_str}",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep full pre-start history for regression, SMA, gap,
    # and volatility features, but execute only from the configured start.
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

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


def run_all_variants(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = None,
) -> dict[str, ClenowTop10Vol63Strategy]:
    strategy_dict: dict[str, ClenowTop10Vol63Strategy] = {}
    for variant_key_str in CONFIG_BY_VARIANT_KEY_DICT:
        strategy_dict[variant_key_str] = run_variant(
            variant_key_str=variant_key_str,
            show_display_bool=show_display_bool,
            save_results_bool=save_results_bool,
            output_dir_str=output_dir_str,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
            audit_override_bool=audit_override_bool,
        )
    return strategy_dict


if __name__ == "__main__":
    run_all_variants()
