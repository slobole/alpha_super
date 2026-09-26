"""Standalone DV2 mean reversion with a point-in-time liquidity floor.

Decide after Close_T and fill orders at Open_(T+1). The universe is the
point-in-time S&P 500. The trade and signal price basis is CAPITALSPECIAL;
the $SPX performance benchmark is loaded as TOTALRETURN. At most ten long
positions receive one tenth of prior portfolio value each. Slippage is 2.5
basis points per side; commission is $0.005/share with a $1 minimum.

The only change from DV2's trade rules is which members may be bought:

    ADV63_i,t = mean over [t-62, t] of native Norgate Turnover_i
    median_t  = median of available ADV63 over that day's PIT S&P 500 members

    eligible_i,t = 1[UnadjustedClose_i,t > 5] * 1[ADV63_i,t > median_t]

    entry_i,t = eligible_i,t
        * 1[DV2_i,t < 10]
        * 1[Close_i,t > SMA200_i,t]
        * 1[Return126_i,t > 0.05]

    exit_i,t = 1[Close_i,t > High_i,t-1]

The research predecessor is dv2_liq_floor in
scripts/research/growth_shelf_20260924/dv2_liquidity_variants.py. This
standalone module computes the median before removing members with incomplete
DV2 features, matching the written PIT-universe rule. Its ADV requires 63
consecutive panel rows with finite positive native dollar Turnover. A missing
Turnover field raises; an invalid observation invalidates that ADV window.
Raw price must be finite and positive for eligibility. No extra forward-fill
is applied here. Native Turnover avoids multiplying raw historical dollars
by share volume adjusted for later splits. Prior research figures using that
mixed-unit product are not parity evidence for this corrected version.

Bench displays WIRED for this module. That display does not authorize a LIVE
release, broker route, or portfolio allocation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from IPython.display import display
import talib

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator
from data.norgate_loader import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    build_index_constituent_matrix,
    load_raw_prices,
)

STRATEGY_NAME_STR = "strategy_mr_dv2_liquidity_floor"
ADV_WINDOW_INT = 63
RAW_PRICE_MIN_FLOAT = 5.0


def get_prices(
    symbol_list: list[str],
    benchmark_list: list[str],
    start_date: str = "1998-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(symbol_list, benchmark_list, start_date, end_date)


def default_trade_id_int() -> int:
    return -1


def get_asof_universe_symbol_list(
    universe_df: pd.DataFrame | None,
    decision_date_ts: pd.Timestamp,
) -> list[str]:
    if universe_df is None or len(universe_df) == 0:
        return []
    sorted_universe_df = universe_df.sort_index()
    # *** CRITICAL*** Use the latest PIT membership on or before Close_T.
    universe_row_int = int(
        sorted_universe_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if universe_row_int < 0:
        return []
    universe_membership_ser = sorted_universe_df.iloc[universe_row_int]
    return universe_membership_ser[universe_membership_ser == 1].index.astype(str).tolist()


class DVO2LiquidityFloorStrategy(Strategy):
    max_positions = 10
    trade_id = 0
    current_trade = defaultdict(default_trade_id_int)
    universe_df = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        self._data_adjustment_policy_dict.update(
            {
                "stock_signal_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "execution_and_marks_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "performance_benchmark_adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
            }
        )
        signal_df = pricing_data.copy()
        feature_map_dict: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in pricing_data.columns.get_level_values(0).unique():
            if str(symbol_str).startswith("$") or (symbol_str, "Close") not in pricing_data.columns:
                continue
            required_field_list = ["High", "Low", "Unadjusted Close", "Turnover"]
            missing_field_list = [
                field_str for field_str in required_field_list
                if (symbol_str, field_str) not in pricing_data.columns
            ]
            if missing_field_list:
                raise ValueError(
                    f"{symbol_str} is missing liquidity/DV2 fields {missing_field_list}."
                )
            close_ser = pricing_data[(symbol_str, "Close")]
            high_ser = pricing_data[(symbol_str, "High")]
            low_ser = pricing_data[(symbol_str, "Low")]
            raw_close_ser = pd.to_numeric(
                pricing_data[(symbol_str, "Unadjusted Close")], errors="coerce"
            )
            raw_close_ser = raw_close_ser.where(np.isfinite(raw_close_ser) & raw_close_ser.gt(0.0))
            # *** CRITICAL *** Corporate-action and decision-time boundary:
            # ADV63_T = mean(native dollar Turnover[T-62:T]), used at Open_(T+1).
            # CAPITALSPECIAL Volume includes future splits; raw Close * Volume
            # would leak those splits into historical dollar liquidity.
            dollar_volume_ser = pd.to_numeric(
                pricing_data[(symbol_str, "Turnover")], errors="coerce"
            )
            dollar_volume_ser = dollar_volume_ser.where(
                np.isfinite(dollar_volume_ser) & dollar_volume_ser.gt(0.0)
            )
            # *** CRITICAL*** Every feature ends at Close_T; run_daily executes
            # orders from this decision at Open_(T+1). No later bar may enter.
            feature_map_dict[(symbol_str, "p126d_return")] = close_ser / close_ser.shift(126) - 1
            feature_map_dict[(symbol_str, "natr")] = talib.NATR(high_ser, low_ser, close_ser, 14)
            feature_map_dict[(symbol_str, "dv2")] = dv2_indicator(close_ser, high_ser, low_ser, length_int=126)
            feature_map_dict[(symbol_str, "sma_200")] = close_ser.rolling(200).mean()
            feature_map_dict[(symbol_str, "adv_63")] = dollar_volume_ser.rolling(ADV_WINDOW_INT, min_periods=ADV_WINDOW_INT).mean()
            feature_map_dict[(symbol_str, "raw_price")] = raw_close_ser
        if not feature_map_dict:
            return signal_df
        return pd.concat([signal_df, pd.DataFrame(feature_map_dict, index=signal_df.index)], axis=1).copy()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series) -> None:
        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        available_slot_int = self.max_positions - len(long_position_ser)

        # Exit is identical to DV2: Close_T > High_(T-1).
        for symbol_str in long_position_ser.index:
            close_price_float = close[(symbol_str, "Close")]
            prior_high_float = data[(symbol_str, "High")].iloc[-2]
            if close_price_float > prior_high_float:
                self.order_target_value(symbol_str, 0, trade_id=self.current_trade[symbol_str])
                available_slot_int += 1

        # Each new entry targets one tenth of prior portfolio value, as in DV2.
        capital_per_trade_float = self.previous_total_value / self.max_positions
        opportunity_list = self.get_opportunities(close)
        while available_slot_int > 0 and opportunity_list:
            symbol_str = opportunity_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue
            self.trade_id += 1
            self.current_trade[symbol_str] = self.trade_id
            self.order_value(symbol_str, capital_per_trade_float, trade_id=self.trade_id)
            available_slot_int -= 1

    def get_opportunities(self, close: pd.Series) -> list[str]:
        candidate_df = close.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
        member_list = get_asof_universe_symbol_list(self.universe_df, pd.Timestamp(self.previous_bar))
        member_df = candidate_df[candidate_df.index.isin(member_list)]
        # *** CRITICAL*** The median is taken over that day's point-in-time index members,
        # before requiring DV2 features or applying entry rules. Missing ADV
        # stays missing; it is never forward-filled across decision boundaries.
        median_adv_float = float(member_df["adv_63"].dropna().median())
        if not pd.notna(median_adv_float):
            return []
        # Match DV2's complete-row check for tradable candidates. The median
        # above must still include PIT members missing unrelated DV2 fields.
        member_df = member_df.dropna()
        member_df = member_df[(member_df["raw_price"] > RAW_PRICE_MIN_FLOAT) & (member_df["adv_63"] > median_adv_float)]
        member_df = member_df[
            (member_df["dv2"] < 10)
            & (member_df["Close"] > member_df["sma_200"])
            & (member_df["p126d_return"] > 0.05)
        ]
        return member_df.sort_values("natr", ascending=False).index.tolist()


def _new_strategy_obj(capital_base_float: float, universe_df: pd.DataFrame) -> DVO2LiquidityFloorStrategy:
    strategy_obj = DVO2LiquidityFloorStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=["$SPX"],
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
        performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(default_trade_id_int)
    return strategy_obj


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """Inputs for ExecutionTimingAnalysis; same calendar and timing grid as the WIRED DV2."""
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(symbol_list, benchmark_list, start_date="1998-01-01", end_date=None)
    # *** CRITICAL*** Keep the same post-warmup calendar as the WIRED DV2 timing study.
    calendar_idx = pricing_df.index[pricing_df.index.year >= 2004]
    return {
        "strategy_factory_fn": lambda: _new_strategy_obj(100_000.0, universe_df),
        "pricing_data_df": pricing_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "daily_ohlc_signal",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One completed run for CapacityAnalysis, on the same path as run_variant."""
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(symbol_list, ["$SPX"], start_date="1998-01-01", end_date=end_date_str)
    strategy_obj = _new_strategy_obj(capital_base_float, universe_df)
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy_obj, pricing_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy_obj.universe_df = None
    strategy_obj._performance_benchmark_symbol_str = "$SPX"
    strategy_obj._performance_benchmark_adjustment_str = "TOTALRETURN"
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    benchmark_list = ["$SPX"]
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(index_symbol_list, benchmark_list, start_date="1998-01-01", end_date=end_date_str)

    strategy_obj = _new_strategy_obj(capital_base_float, universe_df)

    # *** CRITICAL*** Keep full pre-start history for indicators; execute only on the
    # configured calendar, as the WIRED DV2 does.
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(
        strategy_obj,
        pricing_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
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
