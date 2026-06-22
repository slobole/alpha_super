"""
Research-only monthly Nasdaq-100 rotation ranked by 252-day Ev.

This variant keeps the deployment-reference NDX strategy shape from
``strategy_mo_atr_normalized_ndx`` and replaces only the stock ranking score.
It is intentionally not wired into live release manifests.

Core formulas
-------------
For stock i on decision close T:

    r_{i,t}
        = ln(Close_{i,t} / Close_{i,t-1})

    P_{i,w}(T)
        = sum(max(r_{i,t}, 0)) over the last w observations ending at T

    N_{i,w}(T)
        = sum(max(-r_{i,t}, 0)) over the last w observations ending at T

    LRB_{i,w}(T)
        = P_{i,w}(T) / N_{i,w}(T)

    Ev_{i,w}(T)
        = (P_{i,w}(T) - N_{i,w}(T)) / (P_{i,w}(T) + N_{i,w}(T))

Selection on decision date T:

    eligible_{i,T}
        = 1[PIT_NDX_{i,T} = 1 and Close_{i,T} > SMA100_{i,T}]

    selected_T
        = top max_positions eligible symbols by Ev_{i,252}(T)

    target_weight_{i,T}
        = 1 / max_positions    if i in selected_T
        = 0                    otherwise

Execution mapping is unchanged:

    decision_date_T
        = actual last tradable close of month T

    execution_date_T
        = next tradable open after decision_date_T
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxConfig,
    AtrNormalizedNdxStrategy,
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


PN_WINDOW_INT = 252


@dataclass(frozen=True)
class EvLrb252NdxConfig(AtrNormalizedNdxConfig):
    pn_window_int: int = PN_WINDOW_INT
    require_positive_ev_bool: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pn_window_int <= 1:
            raise ValueError("pn_window_int must be greater than 1.")


DEFAULT_CONFIG = EvLrb252NdxConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "EvLrb252NdxConfig",
    "EvLrb252NdxStrategy",
    "PN_WINDOW_INT",
    "compute_ev_lrb_indicator_tables",
    "compute_ev_lrb_signal_tables",
    "get_ev_lrb_252_ndx_data",
    "run_variant",
]


def compute_ev_lrb_indicator_tables(
    price_close_df: pd.DataFrame,
    window_int: int = PN_WINDOW_INT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")
    if window_int <= 1:
        raise ValueError("window_int must be greater than 1.")

    clean_price_close_df = price_close_df.astype(float).where(price_close_df.astype(float) > 0.0)
    log_price_df = np.log(clean_price_close_df)

    # *** CRITICAL *** lookahead-sensitive P/N indicator: the diff and rolling
    # windows end at close T. The indicator is only known after close T, and
    # this strategy maps that decision to the next tradable open.
    log_return_df = log_price_df.diff()
    positive_return_df = log_return_df.clip(lower=0.0)
    negative_return_df = (-log_return_df).clip(lower=0.0)
    positive_return_sum_df = positive_return_df.rolling(
        window=window_int,
        min_periods=window_int,
    ).sum()
    negative_return_sum_df = negative_return_df.rolling(
        window=window_int,
        min_periods=window_int,
    ).sum()

    linear_return_balance_df = positive_return_sum_df - negative_return_sum_df
    variation_df = positive_return_sum_df + negative_return_sum_df
    # LRB is audit-only in this strategy. When N_w(T) is zero, LRB is
    # undefined here, while Ev remains defined as 1.0 if P_w(T) > 0.
    lrb_df = positive_return_sum_df.div(negative_return_sum_df.replace(0.0, np.nan))
    ev_df = linear_return_balance_df.div(variation_df.replace(0.0, np.nan))

    return (
        positive_return_sum_df.replace([np.inf, -np.inf], np.nan),
        negative_return_sum_df.replace([np.inf, -np.inf], np.nan),
        lrb_df.replace([np.inf, -np.inf], np.nan),
        ev_df.replace([np.inf, -np.inf], np.nan),
    )


def compute_ev_lrb_signal_tables(
    price_close_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: EvLrb252NdxConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    (
        positive_return_sum_df,
        negative_return_sum_df,
        lrb_df,
        ev_df,
    ) = compute_ev_lrb_indicator_tables(
        price_close_df=price_close_df,
        window_int=config.pn_window_int,
    )

    positive_return_sum_decision_df = positive_return_sum_df.reindex(monthly_decision_close_df.index)
    negative_return_sum_decision_df = negative_return_sum_df.reindex(monthly_decision_close_df.index)
    lrb_decision_df = lrb_df.reindex(monthly_decision_close_df.index)
    ev_decision_df = ev_df.reindex(monthly_decision_close_df.index)

    # *** CRITICAL*** The stock trend filter uses the close at T and the
    # trailing SMA ending at T. Orders still execute only at T+1 open.
    stock_trend_sma_df = price_close_df.rolling(
        window=config.stock_trend_window_int,
        min_periods=config.stock_trend_window_int,
    ).mean()
    stock_trend_pass_df = (price_close_df > stock_trend_sma_df).reindex(monthly_decision_close_df.index)

    # *** CRITICAL*** The regime filter is known only after the benchmark
    # close at T and is mapped to the next tradable open by the schedule.
    regime_close_decision_ser = regime_close_ser.reindex(monthly_decision_close_df.index)
    regime_sma_ser = regime_close_ser.rolling(
        window=config.index_trend_window_int,
        min_periods=config.index_trend_window_int,
    ).mean().reindex(monthly_decision_close_df.index)
    regime_pass_ser = regime_close_decision_ser > regime_sma_ser

    ev_score_df = ev_decision_df.replace([np.inf, -np.inf], np.nan)
    valid_ev_bool_ser = ev_score_df.notna().any(axis=1)
    valid_stock_trend_bool_ser = stock_trend_pass_df.notna().any(axis=1)
    valid_regime_bool_ser = regime_close_decision_ser.notna() & regime_sma_ser.notna()
    valid_decision_index = monthly_decision_close_df.index[
        valid_ev_bool_ser
        & valid_stock_trend_bool_ser
        & valid_regime_bool_ser
    ]

    monthly_decision_close_df = monthly_decision_close_df.reindex(valid_decision_index)
    positive_return_sum_decision_df = positive_return_sum_decision_df.reindex(valid_decision_index)
    negative_return_sum_decision_df = negative_return_sum_decision_df.reindex(valid_decision_index)
    lrb_decision_df = lrb_decision_df.reindex(valid_decision_index)
    ev_decision_df = ev_decision_df.reindex(valid_decision_index)
    stock_trend_pass_df = stock_trend_pass_df.reindex(valid_decision_index)
    regime_sma_ser = regime_sma_ser.reindex(valid_decision_index)
    regime_pass_ser = regime_pass_ser.reindex(valid_decision_index)
    ev_score_df = ev_score_df.reindex(valid_decision_index)

    return (
        monthly_decision_close_df,
        positive_return_sum_decision_df,
        negative_return_sum_decision_df,
        lrb_decision_df,
        ev_decision_df,
        stock_trend_pass_df,
        regime_sma_ser,
        regime_pass_ser,
        ev_score_df,
    )


def get_ev_lrb_252_ndx_data(
    config: EvLrb252NdxConfig = DEFAULT_CONFIG,
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
        raise RuntimeError("No active Nasdaq-100 universe symbols were found for the requested backtest window.")

    price_symbol_list = list(dict.fromkeys(active_symbol_list + [config.regime_symbol_str]))
    pricing_data_df = load_raw_prices(
        symbols=price_symbol_list,
        benchmarks=[],
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

    keep_symbol_set = set(audited_universe_df.columns.tolist() + [config.regime_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    close_symbol_list = audited_universe_df.columns.tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    regime_close_ser = pricing_data_df[(config.regime_symbol_str, "Close")].astype(float)

    (
        monthly_decision_close_df,
        _positive_return_sum_decision_df,
        _negative_return_sum_decision_df,
        _lrb_decision_df,
        _ev_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _ev_score_df,
    ) = compute_ev_lrb_signal_tables(
        price_close_df=price_close_df,
        regime_close_ser=regime_close_ser,
        config=config,
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class EvLrb252NdxStrategy(AtrNormalizedNdxStrategy):
    """
    Long-only monthly NDX selector ranked by Ev_252, with LRB_252 audited.
    """

    @property
    def ev_score_field_str(self) -> str:
        return f"ev_{self.pn_window_int}_float"

    @property
    def lrb_field_str(self) -> str:
        return f"lrb_{self.pn_window_int}_float"

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        regime_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
        pn_window_int: int = PN_WINDOW_INT,
        require_positive_ev_bool: bool = False,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_month_int=1,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        if pn_window_int <= 1:
            raise ValueError("pn_window_int must be greater than 1.")

        self.pn_window_int = int(pn_window_int)
        self.require_positive_ev_bool = bool(require_positive_ev_bool)

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

        regime_close_key = (self.regime_symbol_str, "Close")
        if regime_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing regime close data for {self.regime_symbol_str}.")
        regime_close_ser = signal_data_df[regime_close_key].astype(float)

        helper_config = EvLrb252NdxConfig(
            regime_symbol_str=self.regime_symbol_str,
            index_trend_window_int=self.index_trend_window_int,
            stock_trend_window_int=self.stock_trend_window_int,
            max_positions_int=self.max_positions_int,
            pn_window_int=self.pn_window_int,
            require_positive_ev_bool=self.require_positive_ev_bool,
        )
        (
            _monthly_decision_close_df,
            _positive_return_sum_decision_df,
            _negative_return_sum_decision_df,
            lrb_decision_df,
            ev_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            _ev_score_df,
        ) = compute_ev_lrb_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=regime_close_ser,
            config=helper_config,
        )

        feature_map: dict[str, pd.DataFrame] = {
            self.lrb_field_str: lrb_decision_df.reindex(signal_data_df.index),
            self.ev_score_field_str: ev_decision_df.reindex(signal_data_df.index),
            "stock_trend_pass_bool": stock_trend_pass_df.reindex(signal_data_df.index),
        }
        feature_frame_list: list[pd.DataFrame] = []
        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (self.regime_symbol_str, f"regime_sma_{self.index_trend_window_int}_ser"): regime_sma_ser.reindex(signal_data_df.index),
                (self.regime_symbol_str, "regime_pass_bool"): regime_pass_ser.reindex(signal_data_df.index),
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")
        candidate_feature_df = close_row_ser.unstack()
        if self.regime_symbol_str not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime feature row for {self.regime_symbol_str}.")

        regime_pass_value = candidate_feature_df.loc[self.regime_symbol_str].get("regime_pass_bool", np.nan)
        if pd.isna(regime_pass_value) or not bool(regime_pass_value):
            return pd.Series(dtype=float)

        required_field_list = ["stock_trend_pass_bool", self.ev_score_field_str]
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

        stock_trend_raw_ser = candidate_feature_df["stock_trend_pass_bool"]
        stock_trend_pass_ser = stock_trend_raw_ser.where(stock_trend_raw_ser.notna(), False).astype(bool)
        candidate_feature_df = candidate_feature_df.assign(
            ev_score_float=pd.to_numeric(candidate_feature_df[self.ev_score_field_str], errors="coerce"),
            stock_trend_pass_bool=stock_trend_pass_ser,
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_ev_mask_vec = np.isfinite(
            candidate_feature_df["ev_score_float"].to_numpy(dtype=float)
        )
        stock_trend_pass_mask_vec = candidate_feature_df["stock_trend_pass_bool"].to_numpy(dtype=bool)
        candidate_feature_df = candidate_feature_df.loc[
            finite_ev_mask_vec & stock_trend_pass_mask_vec
        ].copy()
        if self.require_positive_ev_bool:
            candidate_feature_df = candidate_feature_df.loc[candidate_feature_df["ev_score_float"] > 0.0].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.sort_values(
            by=["ev_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_feature_df = candidate_feature_df.iloc[: self.max_positions_int].copy()

        target_weight_float = 1.0 / float(self.max_positions_int)
        target_weight_ser = pd.Series(
            target_weight_float,
            index=selected_feature_df.index,
            dtype=float,
        )
        return target_weight_ser


def _with_run_overrides(
    config_obj: EvLrb252NdxConfig,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    max_positions_int: int | None = None,
    pn_window_int: int | None = None,
    require_positive_ev_bool: bool | None = None,
) -> EvLrb252NdxConfig:
    if (
        backtest_start_date_str is None
        and capital_base_float is None
        and end_date_str is None
        and max_positions_int is None
        and pn_window_int is None
        and require_positive_ev_bool is None
    ):
        return config_obj

    return replace(
        config_obj,
        backtest_start_date_str=(
            config_obj.backtest_start_date_str
            if backtest_start_date_str is None
            else backtest_start_date_str
        ),
        capital_base_float=(
            config_obj.capital_base_float if capital_base_float is None else float(capital_base_float)
        ),
        end_date_str=end_date_str,
        max_positions_int=(
            config_obj.max_positions_int if max_positions_int is None else int(max_positions_int)
        ),
        pn_window_int=config_obj.pn_window_int if pn_window_int is None else int(pn_window_int),
        require_positive_ev_bool=(
            config_obj.require_positive_ev_bool
            if require_positive_ev_bool is None
            else bool(require_positive_ev_bool)
        ),
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    max_positions_int: int | None = None,
    pn_window_int: int | None = None,
    require_positive_ev_bool: bool | None = None,
    audit_override_bool: bool | None = None,
) -> EvLrb252NdxStrategy:
    config_obj = _with_run_overrides(
        config_obj=DEFAULT_CONFIG,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        max_positions_int=max_positions_int,
        pn_window_int=pn_window_int,
        require_positive_ev_bool=require_positive_ev_bool,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_ev_lrb_252_ndx_data(config_obj)

    strategy_name_str = f"strategy_mo_ev_lrb_{config_obj.pn_window_int}_ndx"
    if config_obj.require_positive_ev_bool:
        strategy_name_str = f"{strategy_name_str}_positive_ev"
    if config_obj.max_positions_int != DEFAULT_CONFIG.max_positions_int:
        strategy_name_str = f"{strategy_name_str}_top{config_obj.max_positions_int}"

    strategy_obj = EvLrb252NdxStrategy(
        name=strategy_name_str,
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
        pn_window_int=config_obj.pn_window_int,
        require_positive_ev_bool=config_obj.require_positive_ev_bool,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep full pre-start history for Ev/LRB and trend
    # features, but execute only from the configured backtest start.
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


if __name__ == "__main__":
    run_variant()
