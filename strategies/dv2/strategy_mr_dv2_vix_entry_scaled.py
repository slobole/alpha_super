from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices


VIX_SIGNAL_SYMBOL_STR = "$VIX"
BENCHMARK_SYMBOL_STR = "$SPX"
DV2_LOOKBACK_DAY_INT = 126
NATR_LOOKBACK_DAY_INT = 14
SMA_WINDOW_DAY_INT = 200
P126_RETURN_MIN_FLOAT = 0.05
DV2_ENTRY_MAX_FLOAT = 10.0
TARGET_VIX_PCT_FLOAT = 18.0
MIN_ENTRY_SCALE_FLOAT = 0.50
MAX_ENTRY_SCALE_FLOAT = 1.00
DEFAULT_MAX_POSITIONS_INT = 10


def default_trade_id_int() -> int:
    return -1


def get_asof_universe_symbol_list(
    universe_df: pd.DataFrame | None,
    decision_date_ts: pd.Timestamp,
) -> list[str]:
    if universe_df is None or len(universe_df) == 0:
        return []

    sorted_universe_df = universe_df.sort_index()
    # *** CRITICAL*** PIT universe membership may lag the newest price date.
    # Use only the latest universe row available on or before decision_t; never
    # use a later row, because that would leak future index membership.
    universe_row_int = int(
        sorted_universe_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if universe_row_int < 0:
        return []

    universe_membership_ser = sorted_universe_df.iloc[universe_row_int]
    return universe_membership_ser[universe_membership_ser == 1].index.astype(str).tolist()


def get_dv2_vix_entry_scaled_prices(
    symbol_iter: Iterable[str],
    benchmarks: list[str],
    start_date_str: str = "1998-01-01",
    end_date_str: str | None = None,
) -> pd.DataFrame:
    signal_symbol_list = list(dict.fromkeys([*symbol_iter, VIX_SIGNAL_SYMBOL_STR]))
    return load_raw_prices(signal_symbol_list, benchmarks, start_date_str, end_date_str)


def compute_vix_entry_scale_signal_df(
    vix_close_ser: pd.Series,
    target_vix_pct_float: float = TARGET_VIX_PCT_FLOAT,
    min_entry_scale_float: float = MIN_ENTRY_SCALE_FLOAT,
    max_entry_scale_float: float = MAX_ENTRY_SCALE_FLOAT,
) -> pd.DataFrame:
    """
    Compute the daily VIX entry-size multiplier.

    Formula:

        vix_entry_scale_t
            = clip(VIX_t / target_vix_pct, min_scale, max_scale)
    """
    if target_vix_pct_float <= 0.0:
        raise ValueError("target_vix_pct_float must be positive.")
    if min_entry_scale_float < 0.0:
        raise ValueError("min_entry_scale_float must be non-negative.")
    if min_entry_scale_float > max_entry_scale_float:
        raise ValueError("min_entry_scale_float must be <= max_entry_scale_float.")
    if max_entry_scale_float > 1.0:
        raise ValueError("max_entry_scale_float must be <= 1.0 for this no-leverage variant.")

    clean_vix_close_ser = pd.Series(vix_close_ser, copy=True).astype(float).sort_index()
    clean_vix_close_ser.name = "vix_close_ser"

    # *** CRITICAL*** The VIX entry scaler uses only same-row VIX close values.
    # In the engine this row is previous_bar, so orders submitted from it fill
    # at the next open. A future VIX value here would be look-ahead bias.
    raw_entry_scale_ser = clean_vix_close_ser / float(target_vix_pct_float)
    entry_scale_ser = raw_entry_scale_ser.replace([np.inf, -np.inf], np.nan).clip(
        lower=float(min_entry_scale_float),
        upper=float(max_entry_scale_float),
    )

    return pd.DataFrame(
        {
            "vix_close_ser": clean_vix_close_ser,
            "vix_entry_scale_float": entry_scale_ser,
        },
        index=clean_vix_close_ser.index,
    )


class DV2VixEntryScaledStrategy(Strategy):
    """
    DV2 stock mean-reversion variant with VIX-scaled entry size.

    Core entry rule is unchanged:

        entry_t
            = 1[dv2_t < 10]
            * 1[Close_t > SMA200_t]
            * 1[p126d_return_t > 0.05]

    New entry order value is scaled by:

        entry_value_t
            = (previous_total_value / max_positions) * clip(VIX_t / 18, 0.50, 1.00)

    Exit rule is intentionally unchanged from the legacy DV2 variant.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = DEFAULT_MAX_POSITIONS_INT,
        target_vix_pct_float: float = TARGET_VIX_PCT_FLOAT,
        min_entry_scale_float: float = MIN_ENTRY_SCALE_FLOAT,
        max_entry_scale_float: float = MAX_ENTRY_SCALE_FLOAT,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if target_vix_pct_float <= 0.0:
            raise ValueError("target_vix_pct_float must be positive.")
        if min_entry_scale_float < 0.0:
            raise ValueError("min_entry_scale_float must be non-negative.")
        if min_entry_scale_float > max_entry_scale_float:
            raise ValueError("min_entry_scale_float must be <= max_entry_scale_float.")
        if max_entry_scale_float > 1.0:
            raise ValueError("max_entry_scale_float must be <= 1.0 for this no-leverage variant.")

        self.max_positions_int = int(max_positions_int)
        self.target_vix_pct_float = float(target_vix_pct_float)
        self.min_entry_scale_float = float(min_entry_scale_float)
        self.max_entry_scale_float = float(max_entry_scale_float)
        self.trade_id_int = 0
        self.current_trade_id_map = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        feature_col_map: dict[tuple[str, str], pd.Series] = {}
        symbol_list = signal_data_df.columns.get_level_values(0).unique().tolist()

        for symbol_str in self.signal_progress(
            symbol_list,
            desc_str="dv2 vix-scaled signal precompute",
            total_int=len(symbol_list),
        ):
            if str(symbol_str).startswith("$") or (symbol_str, "Close") not in signal_data_df.columns:
                continue

            close_price_ser = signal_data_df[(symbol_str, "Close")].astype(float)
            high_price_ser = signal_data_df[(symbol_str, "High")].astype(float)
            low_price_ser = signal_data_df[(symbol_str, "Low")].astype(float)

            # *** CRITICAL*** The 126-day return must use a lagged close.
            # Using current-to-future prices here would create look-ahead bias
            # inside compute_signals().
            p126d_return_ser = close_price_ser / close_price_ser.shift(DV2_LOOKBACK_DAY_INT) - 1.0
            natr_value_ser = pd.Series(
                talib.NATR(
                    high_price_ser.to_numpy(dtype=float),
                    low_price_ser.to_numpy(dtype=float),
                    close_price_ser.to_numpy(dtype=float),
                    NATR_LOOKBACK_DAY_INT,
                ),
                index=signal_data_df.index,
                dtype=float,
            )
            dv2_value_ser = dv2_indicator(
                close_price_ser,
                high_price_ser,
                low_price_ser,
                length_int=DV2_LOOKBACK_DAY_INT,
            )

            # *** CRITICAL*** The 200-day SMA must stay strictly trailing.
            # A forward or centered window would leak future trend information
            # into the current entry filter.
            sma_200_price_ser = close_price_ser.rolling(SMA_WINDOW_DAY_INT).mean()

            feature_col_map[(symbol_str, "p126d_return_ser")] = p126d_return_ser
            feature_col_map[(symbol_str, "natr_value_ser")] = natr_value_ser
            feature_col_map[(symbol_str, "dv2_value_ser")] = dv2_value_ser
            feature_col_map[(symbol_str, "sma_200_price_ser")] = sma_200_price_ser

        vix_close_key = (VIX_SIGNAL_SYMBOL_STR, "Close")
        if vix_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing required VIX close column: {vix_close_key}")

        vix_scale_signal_df = compute_vix_entry_scale_signal_df(
            vix_close_ser=signal_data_df[vix_close_key],
            target_vix_pct_float=self.target_vix_pct_float,
            min_entry_scale_float=self.min_entry_scale_float,
            max_entry_scale_float=self.max_entry_scale_float,
        )
        for field_str in vix_scale_signal_df.columns:
            feature_col_map[(VIX_SIGNAL_SYMBOL_STR, field_str)] = vix_scale_signal_df[field_str]

        if len(feature_col_map) == 0:
            return signal_data_df

        feature_data_df = pd.DataFrame(feature_col_map, index=signal_data_df.index)
        return pd.concat([signal_data_df, feature_data_df], axis=1)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        if data_df is None or close_row_ser is None:
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slot_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            close_price_float = float(close_row_ser.loc[(symbol_str, "Close")])
            prior_high_float = float(data_df[(symbol_str, "High")].iloc[-2])
            if close_price_float > prior_high_float:
                self.order_target_value(
                    symbol_str,
                    0,
                    trade_id=self.current_trade_id_map[symbol_str],
                )
                long_slot_int += 1

        vix_scale_key = (VIX_SIGNAL_SYMBOL_STR, "vix_entry_scale_float")
        if vix_scale_key not in close_row_ser.index:
            return

        entry_scale_float = float(close_row_ser.loc[vix_scale_key])
        if not np.isfinite(entry_scale_float) or entry_scale_float <= 0.0:
            return

        capital_to_allocate_per_trade_float = (
            self.previous_total_value / self.max_positions_int * entry_scale_float
        )
        opportunity_list = self.get_opportunity_list(close_row_ser)

        while long_slot_int > 0 and len(opportunity_list) > 0:
            symbol_str = opportunity_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_id_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                capital_to_allocate_per_trade_float,
                trade_id=self.trade_id_int,
            )
            long_slot_int -= 1

    def get_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        """
        Rank eligible stock pullback candidates in descending NATR order.
        """
        if self.universe_df is None:
            raise RuntimeError("universe_df must be assigned before get_opportunity_list().")

        stock_mask = ~close_row_ser.index.get_level_values(0).astype(str).str.startswith("$")
        stock_row_ser = close_row_ser.loc[stock_mask]
        if len(stock_row_ser) == 0:
            return []

        stock_feature_df = stock_row_ser.unstack()
        required_col_list = [
            "Close",
            "p126d_return_ser",
            "natr_value_ser",
            "dv2_value_ser",
            "sma_200_price_ser",
        ]
        if any(field_str not in stock_feature_df.columns for field_str in required_col_list):
            return []

        stock_feature_df = stock_feature_df.dropna(subset=required_col_list)
        if len(stock_feature_df) == 0:
            return []

        eligible_feature_df = stock_feature_df[
            (stock_feature_df["dv2_value_ser"] < DV2_ENTRY_MAX_FLOAT)
            & (stock_feature_df["Close"] > stock_feature_df["sma_200_price_ser"])
            & (stock_feature_df["p126d_return_ser"] > P126_RETURN_MIN_FLOAT)
        ].sort_values("natr_value_ser", ascending=False)

        tradable_symbol_list = get_asof_universe_symbol_list(
            universe_df=self.universe_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        return eligible_feature_df[eligible_feature_df.index.isin(tradable_symbol_list)].index.tolist()


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> DV2VixEntryScaledStrategy:
    benchmark_list = [BENCHMARK_SYMBOL_STR]
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_dv2_vix_entry_scaled_prices(
        index_symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )

    strategy_obj = DV2VixEntryScaledStrategy(
        name="strategy_mr_dv2_vix_entry_scaled",
        benchmarks=benchmark_list,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for indicators, but the executable calendar starts at the first
    # deployment fill session.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    strategy_obj.universe_df = None

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
