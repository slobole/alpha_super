"""
Long-only smooth-trend momentum selector for point-in-time S&P 500 members.

Core formulas
-------------
For stock i on month-end decision date t:

    formation prices
        = Close_{i,t-252:t-21}

    r_{i,j}
        = Close_{i,j} / Close_{i,j-1} - 1

    S_{i,k}
        = sum_{j=1:k} r_{i,j}

    S_{i,k}
        = alpha_i + beta_i * k + epsilon_{i,k}

Selection on decision date t:

    RQ5_t
        = active PIT members in the highest trend_r2 quintile

    selected_t
        = highest slope quintile inside RQ5_t, keeping only positive slopes

    target_weight_{i,t}
        = 1 / len(selected_t)    if i in selected_t
        = 0                      otherwise

Execution mapping:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t
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
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


def default_trade_id_int() -> int:
    return -1


def get_trend_slope_field_str(
    lookback_trading_day_int: int,
    skip_trading_day_int: int,
) -> str:
    return f"trend_slope_{lookback_trading_day_int}_{skip_trading_day_int}_ser"


def get_trend_r2_field_str(
    lookback_trading_day_int: int,
    skip_trading_day_int: int,
) -> str:
    return f"trend_r2_{lookback_trading_day_int}_{skip_trading_day_int}_ser"


@dataclass(frozen=True)
class SmoothTrendLongSp500Config:
    indexname_str: str = "S&P 500"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    lookback_trading_day_int: int = 252
    skip_trading_day_int: int = 21
    quintile_count_int: int = 5
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.lookback_trading_day_int <= 0:
            raise ValueError("lookback_trading_day_int must be positive.")
        if self.skip_trading_day_int <= 0:
            raise ValueError("skip_trading_day_int must be positive.")
        if self.lookback_trading_day_int <= self.skip_trading_day_int + 2:
            raise ValueError("lookback_trading_day_int must exceed skip_trading_day_int by more than 2.")
        if self.quintile_count_int <= 1:
            raise ValueError("quintile_count_int must be greater than 1.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = SmoothTrendLongSp500Config()


__all__ = [
    "DEFAULT_CONFIG",
    "SmoothTrendLongSp500Config",
    "SmoothTrendLongSp500Strategy",
    "compute_smooth_trend_signal_tables",
    "get_smooth_trend_long_sp500_data",
    "get_trend_r2_field_str",
    "get_trend_slope_field_str",
    "run_variant",
]


def _top_ranked_fraction_df(
    candidate_feature_df: pd.DataFrame,
    score_column_str: str,
    quintile_count_int: int,
) -> pd.DataFrame:
    if len(candidate_feature_df) == 0:
        return candidate_feature_df.copy()

    selected_count_int = max(1, int(np.ceil(len(candidate_feature_df) / float(quintile_count_int))))
    ranked_feature_df = candidate_feature_df.sort_values(
        by=[score_column_str, "symbol_str"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked_feature_df.iloc[:selected_count_int].copy()


def compute_smooth_trend_signal_tables(
    price_close_df: pd.DataFrame,
    lookback_trading_day_int: int = DEFAULT_CONFIG.lookback_trading_day_int,
    skip_trading_day_int: int = DEFAULT_CONFIG.skip_trading_day_int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute paper-style trend slope and unsigned R2 on skipped daily returns.

    For decision date t, the signal uses closes from t-252 through t-21.
    That produces 231 simple daily returns under the default parameters.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")
    if lookback_trading_day_int <= skip_trading_day_int + 2:
        raise ValueError("lookback_trading_day_int must exceed skip_trading_day_int by more than 2.")

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    trend_slope_df = pd.DataFrame(
        np.nan,
        index=monthly_decision_close_df.index,
        columns=price_close_df.columns,
        dtype=float,
    )
    trend_r2_df = trend_slope_df.copy()

    for decision_date_ts in monthly_decision_close_df.index:
        decision_pos_int = int(price_close_df.index.get_loc(decision_date_ts))
        start_pos_int = decision_pos_int - int(lookback_trading_day_int)
        end_pos_int = decision_pos_int - int(skip_trading_day_int)
        if start_pos_int < 0 or end_pos_int <= start_pos_int:
            continue

        # *** CRITICAL*** The formation window ends at t-skip. It must not
        # include the most recent skipped days or the decision-date close.
        formation_price_df = price_close_df.iloc[start_pos_int : end_pos_int + 1]
        finite_price_mask_ser = formation_price_df.notna().all(axis=0)
        positive_price_mask_ser = (formation_price_df > 0.0).all(axis=0)
        valid_price_symbol_list = (
            finite_price_mask_ser[finite_price_mask_ser & positive_price_mask_ser]
            .index.astype(str)
            .tolist()
        )
        if len(valid_price_symbol_list) == 0:
            continue

        # *** CRITICAL*** Daily returns are computed only inside the already
        # skipped formation window. No return after t-skip enters the signal.
        return_window_df = formation_price_df.loc[:, valid_price_symbol_list].pct_change().iloc[1:]
        return_window_df = return_window_df.replace([np.inf, -np.inf], np.nan)
        valid_return_symbol_list = (
            return_window_df.notna().all(axis=0)
            .loc[lambda valid_ser: valid_ser]
            .index.astype(str)
            .tolist()
        )
        if len(valid_return_symbol_list) == 0:
            continue

        cumulative_return_df = return_window_df.loc[:, valid_return_symbol_list].cumsum()
        formation_return_count_int = len(cumulative_return_df.index)
        time_vec = np.arange(1, formation_return_count_int + 1, dtype=float)
        centered_time_vec = time_vec - float(time_vec.mean())
        time_ss_float = float(np.dot(centered_time_vec, centered_time_vec))

        cumulative_return_mat = cumulative_return_df.to_numpy(dtype=float)
        slope_vec = centered_time_vec @ cumulative_return_mat / time_ss_float
        centered_cumulative_return_mat = cumulative_return_mat - cumulative_return_mat.mean(axis=0)
        total_ss_vec = np.sum(centered_cumulative_return_mat * centered_cumulative_return_mat, axis=0)
        regression_ss_vec = (slope_vec * slope_vec) * time_ss_float
        r2_vec = np.divide(
            regression_ss_vec,
            total_ss_vec,
            out=np.full_like(regression_ss_vec, np.nan, dtype=float),
            where=total_ss_vec > 0.0,
        )
        r2_vec = np.clip(r2_vec, 0.0, 1.0)

        trend_slope_df.loc[decision_date_ts, valid_return_symbol_list] = slope_vec
        trend_r2_df.loc[decision_date_ts, valid_return_symbol_list] = r2_vec

    valid_decision_mask_ser = trend_slope_df.notna().any(axis=1) & trend_r2_df.notna().any(axis=1)
    valid_decision_index = trend_slope_df.index[valid_decision_mask_ser]
    return (
        monthly_decision_close_df.reindex(valid_decision_index),
        trend_slope_df.reindex(valid_decision_index),
        trend_r2_df.reindex(valid_decision_index),
    )


def get_smooth_trend_long_sp500_data(
    config: SmoothTrendLongSp500Config = DEFAULT_CONFIG,
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
        raise RuntimeError("No active S&P 500 universe symbols were found for the requested backtest window.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=list(config.benchmark_list),
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

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + list(config.benchmark_list))
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in audited_universe_df.columns.astype(str).tolist()
        },
        index=pricing_data_df.index,
    ).astype(float)
    monthly_decision_close_df, _trend_slope_df, _trend_r2_df = compute_smooth_trend_signal_tables(
        price_close_df=price_close_df,
        lookback_trading_day_int=config.lookback_trading_day_int,
        skip_trading_day_int=config.skip_trading_day_int,
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class SmoothTrendLongSp500Strategy(Strategy):
    """
    Long-only monthly smooth-uptrend selector.

    The strategy uses point-in-time index membership and selects the top R2
    quintile first, then the top slope quintile inside that smoothness bucket.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        lookback_trading_day_int: int = DEFAULT_CONFIG.lookback_trading_day_int,
        skip_trading_day_int: int = DEFAULT_CONFIG.skip_trading_day_int,
        quintile_count_int: int = DEFAULT_CONFIG.quintile_count_int,
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
        if lookback_trading_day_int <= skip_trading_day_int + 2:
            raise ValueError("lookback_trading_day_int must exceed skip_trading_day_int by more than 2.")
        if quintile_count_int <= 1:
            raise ValueError("quintile_count_int must be greater than 1.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.lookback_trading_day_int = int(lookback_trading_day_int)
        self.skip_trading_day_int = int(skip_trading_day_int)
        self.quintile_count_int = int(quintile_count_int)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    @property
    def trend_slope_field_str(self) -> str:
        return get_trend_slope_field_str(
            lookback_trading_day_int=self.lookback_trading_day_int,
            skip_trading_day_int=self.skip_trading_day_int,
        )

    @property
    def trend_r2_field_str(self) -> str:
        return get_trend_r2_field_str(
            lookback_trading_day_int=self.lookback_trading_day_int,
            skip_trading_day_int=self.skip_trading_day_int,
        )

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

        _monthly_decision_close_df, trend_slope_df, trend_r2_df = compute_smooth_trend_signal_tables(
            price_close_df=price_close_df,
            lookback_trading_day_int=self.lookback_trading_day_int,
            skip_trading_day_int=self.skip_trading_day_int,
        )
        trend_slope_aligned_df = trend_slope_df.reindex(signal_data_df.index)
        trend_r2_aligned_df = trend_r2_df.reindex(signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            self.trend_slope_field_str: trend_slope_aligned_df,
            self.trend_r2_field_str: trend_r2_aligned_df,
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
            raise RuntimeError("universe_df must be set before monthly rebalances.")

        candidate_feature_df = close_row_ser.unstack()
        required_field_list = [self.trend_slope_field_str, self.trend_r2_field_str]
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return pd.Series(dtype=float)

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.assign(
            trend_slope_float=pd.to_numeric(
                candidate_feature_df[self.trend_slope_field_str],
                errors="coerce",
            ),
            trend_r2_float=pd.to_numeric(
                candidate_feature_df[self.trend_r2_field_str],
                errors="coerce",
            ),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_mask_vec = np.isfinite(
            candidate_feature_df[["trend_slope_float", "trend_r2_float"]].to_numpy(dtype=float)
        ).all(axis=1)
        candidate_feature_df = candidate_feature_df.loc[finite_mask_vec].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        r2_bucket_df = _top_ranked_fraction_df(
            candidate_feature_df=candidate_feature_df,
            score_column_str="trend_r2_float",
            quintile_count_int=self.quintile_count_int,
        )
        slope_bucket_df = _top_ranked_fraction_df(
            candidate_feature_df=r2_bucket_df,
            score_column_str="trend_slope_float",
            quintile_count_int=self.quintile_count_int,
        )
        selected_feature_df = slope_bucket_df.loc[slope_bucket_df["trend_slope_float"] > 0.0].copy()
        if len(selected_feature_df) == 0:
            return pd.Series(dtype=float)

        target_weight_float = 1.0 / float(len(selected_feature_df))
        target_weight_ser = pd.Series(
            target_weight_float,
            index=selected_feature_df.index.astype(str),
            dtype=float,
        )
        return target_weight_ser

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The scheduled month-end decision close must equal
        # previous_bar exactly. Otherwise the skipped-window signal and
        # next-open execution no longer describe the same rebalance.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close)
        target_symbol_set = set(target_weight_ser.index.astype(str).tolist())

        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]
        for symbol_str in long_position_ser.index.astype(str):
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_weight_float in target_weight_ser.items():
            current_share_float = float(current_position_ser.get(symbol_str, 0.0))
            if current_share_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                float(target_weight_float),
                trade_id=self.current_trade_map[symbol_str],
            )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> SmoothTrendLongSp500Strategy:
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

    pricing_data_df, universe_df, rebalance_schedule_df = get_smooth_trend_long_sp500_data(config=config_obj)
    strategy_obj = SmoothTrendLongSp500Strategy(
        name="strategy_mo_smooth_trend_long_sp500",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_trading_day_int=config_obj.lookback_trading_day_int,
        skip_trading_day_int=config_obj.skip_trading_day_int,
        quintile_count_int=config_obj.quintile_count_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep full pre-start history for skipped OLS formation,
    # but only execute the strategy from the configured backtest start.
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
