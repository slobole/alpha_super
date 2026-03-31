"""
Nick Radge-style monthly Nasdaq-100 momentum rotation.

Core formulas
-------------
For stock i on month-end decision date t:

    monthly_roc_{i,t}^{(L)}
        = Close_ME_{i,t} / Close_ME_{i,t-L} - 1

    prior_close_{i,d}
        = Close_{i,d-1}

    TR_{i,d}
        = max(
            High_{i,d} - Low_{i,d},
            abs(High_{i,d} - prior_close_{i,d}),
            abs(Low_{i,d} - prior_close_{i,d})
        )

    ATR20_{i,t}
        = mean(TR_{i,t-19:t})

    stock_trend_pass_{i,t}
        = 1[Close_{i,t} > SMA100_{i,t}]

    regime_pass_t
        = 1[SPY_t > SMA200_t]

    risk_adj_score_{i,t}
        = monthly_roc_{i,t}^{(L)} / ATR20_{i,t}

Selection on decision date t:

    eligible_{i,t}
        = 1[PIT_NDX_{i,t} = 1 and stock_trend_pass_{i,t} = 1]

    selected_t
        = top max_positions eligible symbols by risk_adj_score_{i,t}

    target_weight_{i,t}
        = 1 / max_positions    if i in selected_t
        = 0                    otherwise

Execution mapping:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices


ATR_WINDOW_INT = 20


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class RadgeMomentumNdxConfig:
    indexname_str: str = "Nasdaq 100"
    regime_symbol_str: str = "SPY"
    history_start_date_str: str = "1999-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    lookback_month_int: int = 12
    index_trend_window_int: int = 200
    stock_trend_window_int: int = 100
    max_positions_int: int = 10
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.lookback_month_int <= 0:
            raise ValueError("lookback_month_int must be positive.")
        if self.index_trend_window_int <= 0:
            raise ValueError("index_trend_window_int must be positive.")
        if self.stock_trend_window_int <= 0:
            raise ValueError("stock_trend_window_int must be positive.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = RadgeMomentumNdxConfig()


def audit_pit_universe_df(
    universe_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
    tradeable_symbol_list: Sequence[str],
) -> pd.DataFrame:
    if not universe_df.index.is_monotonic_increasing:
        raise ValueError("universe_df index must be sorted.")
    if universe_df.index.has_duplicates:
        raise ValueError("universe_df index must not contain duplicates.")

    aligned_symbol_list = [
        symbol_str for symbol_str in tradeable_symbol_list if symbol_str in universe_df.columns
    ]
    if len(aligned_symbol_list) == 0:
        raise RuntimeError("No tradeable symbols overlap between pricing data and universe_df.")

    aligned_universe_df = universe_df.loc[universe_df.index.isin(execution_index), aligned_symbol_list].copy()
    missing_execution_index = execution_index.difference(aligned_universe_df.index)
    if len(missing_execution_index) > 0:
        missing_date_preview_list = [pd.Timestamp(date_ts).strftime("%Y-%m-%d") for date_ts in missing_execution_index[:5]]
        raise RuntimeError(
            "PIT universe is missing execution dates after loader alignment. "
            f"First missing dates: {missing_date_preview_list}"
        )

    return aligned_universe_df.astype(int).sort_index()


def get_monthly_decision_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily closes to the actual last tradable close of each month.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")

    # *** CRITICAL*** Monthly decisions must use the actual last tradable
    # close in each month, not a synthetic calendar month-end timestamp.
    decision_date_ser = pd.Series(
        price_close_df.index,
        index=price_close_df.index.to_period("M"),
    ).groupby(level=0).max()

    last_available_ts = pd.Timestamp(price_close_df.index[-1])
    expected_business_month_end_ts = pd.Timestamp(last_available_ts + pd.offsets.BMonthEnd(0))
    if (
        len(decision_date_ser) > 0
        and pd.Timestamp(decision_date_ser.iloc[-1]) == last_available_ts
        and expected_business_month_end_ts.normalize() != last_available_ts.normalize()
    ):
        decision_date_ser = decision_date_ser.iloc[:-1]

    decision_date_idx = pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")
    monthly_decision_close_df = price_close_df.loc[decision_date_idx].copy()
    monthly_decision_close_df.index = decision_date_idx
    return monthly_decision_close_df


def compute_radge_signal_tables(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: RadgeMomentumNdxConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** Monthly ROC must use only actual month-end decision
    # closes and trailing month-end history.
    monthly_roc_df = (
        monthly_decision_close_df / monthly_decision_close_df.shift(config.lookback_month_int)
    ) - 1.0

    # *** CRITICAL*** prior close alignment for true range must use shift(1)
    # so ATR is strictly trailing.
    prior_close_df = price_close_df.shift(1)
    true_range_df = (price_high_df - price_low_df).combine(
        (price_high_df - prior_close_df).abs(),
        np.maximum,
    )
    true_range_df = true_range_df.combine(
        (price_low_df - prior_close_df).abs(),
        np.maximum,
    )

    # *** CRITICAL*** ATR20 must remain a trailing rolling mean of past true
    # range values only.
    atr_value_df = true_range_df.rolling(
        window=ATR_WINDOW_INT,
        min_periods=ATR_WINDOW_INT,
    ).mean()
    atr_decision_df = atr_value_df.reindex(monthly_decision_close_df.index)

    # *** CRITICAL*** The stock trend filter must remain a trailing rolling
    # average on past closes only.
    stock_trend_sma_df = price_close_df.rolling(
        window=config.stock_trend_window_int,
        min_periods=config.stock_trend_window_int,
    ).mean()
    stock_trend_pass_df = (price_close_df > stock_trend_sma_df).reindex(monthly_decision_close_df.index)

    # *** CRITICAL*** The regime SMA filter must remain a trailing rolling
    # average on past SPY closes only.
    regime_close_decision_ser = regime_close_ser.reindex(monthly_decision_close_df.index)
    regime_sma_ser = regime_close_ser.rolling(
        window=config.index_trend_window_int,
        min_periods=config.index_trend_window_int,
    ).mean().reindex(monthly_decision_close_df.index)
    regime_pass_ser = regime_close_decision_ser > regime_sma_ser

    risk_adj_score_df = monthly_roc_df / atr_decision_df
    risk_adj_score_df = risk_adj_score_df.replace([np.inf, -np.inf], np.nan)

    valid_monthly_roc_bool_ser = monthly_roc_df.notna().any(axis=1)
    valid_atr_bool_ser = atr_decision_df.notna().any(axis=1)
    valid_stock_trend_bool_ser = stock_trend_pass_df.notna().any(axis=1)
    valid_regime_bool_ser = regime_close_decision_ser.notna() & regime_sma_ser.notna()
    valid_decision_index = monthly_decision_close_df.index[
        valid_monthly_roc_bool_ser
        & valid_atr_bool_ser
        & valid_stock_trend_bool_ser
        & valid_regime_bool_ser
    ]

    monthly_decision_close_df = monthly_decision_close_df.reindex(valid_decision_index)
    monthly_roc_df = monthly_roc_df.reindex(valid_decision_index)
    atr_decision_df = atr_decision_df.reindex(valid_decision_index)
    stock_trend_pass_df = stock_trend_pass_df.reindex(valid_decision_index)
    regime_sma_ser = regime_sma_ser.reindex(valid_decision_index)
    regime_pass_ser = regime_pass_ser.reindex(valid_decision_index)
    risk_adj_score_df = risk_adj_score_df.reindex(valid_decision_index)
    return (
        monthly_decision_close_df,
        monthly_roc_df,
        atr_decision_df,
        stock_trend_pass_df,
        regime_sma_ser,
        regime_pass_ser,
        risk_adj_score_df,
    )


def map_month_end_decision_dates_to_rebalance_schedule_df(
    decision_date_index: pd.DatetimeIndex,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each month-end decision close to the next tradable open.
    """
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

        # *** CRITICAL*** Month-end decisions must execute strictly on the
        # next tradable open after the decision close, never on the same bar.
        execution_date_ts = pd.Timestamp(execution_index[execution_insert_int])
        rebalance_schedule_map[execution_date_ts] = pd.Timestamp(decision_date_ts)

    if len(rebalance_schedule_map) == 0:
        raise RuntimeError("No month-end rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def get_radge_momentum_ndx_data(
    config: RadgeMomentumNdxConfig = DEFAULT_CONFIG,
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
    price_high_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "High")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Low")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    regime_close_ser = pricing_data_df[(config.regime_symbol_str, "Close")].astype(float)

    (
        monthly_decision_close_df,
        _monthly_roc_df,
        _atr_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = compute_radge_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        regime_close_ser=regime_close_ser,
        config=config,
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class RadgeMomentumNdxStrategy(Strategy):
    """
    Long-only monthly Nasdaq-100 momentum rotation with fixed slot sizing.

    For selected stock i at rebalance open t:

        q^{intent}_{i,t}
            = floor(V_{t-1} * (1 / max_positions) / Open_{i,t})
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

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
        lookback_month_int: int = 12,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
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
        if not regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if regime_symbol_str not in benchmarks:
            raise ValueError("benchmarks must include regime_symbol_str.")
        if lookback_month_int <= 0:
            raise ValueError("lookback_month_int must be positive.")
        if index_trend_window_int <= 0:
            raise ValueError("index_trend_window_int must be positive.")
        if stock_trend_window_int <= 0:
            raise ValueError("stock_trend_window_int must be positive.")
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.regime_symbol_str = str(regime_symbol_str)
        self.lookback_month_int = int(lookback_month_int)
        self.index_trend_window_int = int(index_trend_window_int)
        self.stock_trend_window_int = int(stock_trend_window_int)
        self.max_positions_int = int(max_positions_int)
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

        regime_close_key = (self.regime_symbol_str, "Close")
        if regime_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing regime close data for {self.regime_symbol_str}.")
        regime_close_ser = signal_data_df[regime_close_key].astype(float)

        helper_config = RadgeMomentumNdxConfig(
            regime_symbol_str=self.regime_symbol_str,
            lookback_month_int=self.lookback_month_int,
            index_trend_window_int=self.index_trend_window_int,
            stock_trend_window_int=self.stock_trend_window_int,
            max_positions_int=self.max_positions_int,
        )
        (
            _monthly_decision_close_df,
            monthly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        ) = compute_radge_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=helper_config,
        )

        monthly_roc_aligned_df = monthly_roc_df.reindex(signal_data_df.index)
        atr_aligned_df = atr_decision_df.reindex(signal_data_df.index)
        stock_trend_pass_aligned_df = stock_trend_pass_df.reindex(signal_data_df.index)
        risk_adj_score_aligned_df = risk_adj_score_df.reindex(signal_data_df.index)
        regime_sma_aligned_ser = regime_sma_ser.reindex(signal_data_df.index)
        regime_pass_aligned_ser = regime_pass_ser.reindex(signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            f"monthly_roc_{self.lookback_month_int}_ser": monthly_roc_aligned_df,
            f"atr_{ATR_WINDOW_INT}_ser": atr_aligned_df,
            "stock_trend_pass_bool": stock_trend_pass_aligned_df,
            "risk_adj_score_ser": risk_adj_score_aligned_df,
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (self.regime_symbol_str, f"regime_sma_{self.index_trend_window_int}_ser"): regime_sma_aligned_ser,
                (self.regime_symbol_str, "regime_pass_bool"): regime_pass_aligned_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")
        if self.previous_bar not in self.universe_df.index:
            raise RuntimeError(f"universe_df is missing decision date {self.previous_bar}.")

        candidate_feature_df = close_row_ser.unstack()
        if self.regime_symbol_str not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime feature row for {self.regime_symbol_str}.")

        regime_pass_value = candidate_feature_df.loc[self.regime_symbol_str].get("regime_pass_bool", np.nan)
        if pd.isna(regime_pass_value) or not bool(regime_pass_value):
            return pd.Series(dtype=float)

        required_field_list = ["stock_trend_pass_bool", "risk_adj_score_ser"]
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return pd.Series(dtype=float)

        universe_member_ser = self.universe_df.loc[self.previous_bar]
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        stock_trend_raw_ser = candidate_feature_df["stock_trend_pass_bool"]
        stock_trend_pass_ser = stock_trend_raw_ser.where(stock_trend_raw_ser.notna(), False).astype(bool)
        candidate_feature_df = candidate_feature_df.assign(
            risk_adj_score_float=pd.to_numeric(candidate_feature_df["risk_adj_score_ser"], errors="coerce"),
            stock_trend_pass_bool=stock_trend_pass_ser,
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_risk_adj_mask_vec = np.isfinite(
            candidate_feature_df["risk_adj_score_float"].to_numpy(dtype=float)
        )
        stock_trend_pass_mask_vec = candidate_feature_df["stock_trend_pass_bool"].to_numpy(dtype=bool)
        candidate_feature_df = candidate_feature_df.loc[
            finite_risk_adj_mask_vec & stock_trend_pass_mask_vec
        ]
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.sort_values(
            by=["risk_adj_score_float", "symbol_str"],
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

    def get_target_share_int_map(
        self,
        target_weight_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> dict[str, int]:
        target_share_int_map: dict[str, int] = {}
        if len(target_weight_ser) == 0:
            return target_share_int_map

        budget_value_float = float(self.previous_total_value)
        for symbol_str, target_weight_float in target_weight_ser.items():
            open_price_float = float(open_price_ser.get(symbol_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for target asset {symbol_str} on {self.current_bar}.")

            target_share_int = int(np.floor(budget_value_float * float(target_weight_float) / open_price_float))
            if target_share_int > 0:
                target_share_int_map[str(symbol_str)] = target_share_int

        return target_share_int_map

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The scheduled month-end decision close must equal
        # previous_bar exactly, otherwise signals and next-open execution drift.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close)
        target_share_int_map = self.get_target_share_int_map(
            target_weight_ser=target_weight_ser,
            open_price_ser=open_prices,
        )
        target_symbol_set = set(target_share_int_map)

        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]
        for symbol_str in long_position_ser.index:
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_share_int in target_share_int_map.items():
            current_share_int = int(current_position_ser.get(symbol_str, 0.0))
            if current_share_int == target_share_int:
                continue

            if current_share_int == 0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            target_weight_float = float(target_weight_ser.loc[symbol_str])
            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_map[symbol_str],
            )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df, universe_df, rebalance_schedule_df = get_radge_momentum_ndx_data(config)

    strategy = RadgeMomentumNdxStrategy(
        name="strategy_mo_radge_ndx",
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config.backtest_start_date_str)]
    run_daily(strategy, pricing_data_df, calendar=calendar_idx, audit_override_bool=None)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
