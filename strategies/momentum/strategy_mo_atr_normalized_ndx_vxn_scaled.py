"""
Monthly ATR-adjusted Nasdaq-100 momentum rotation with a VXN exposure scaler.

Core formulas
-------------
This variant keeps the original stock-selection logic unchanged, then scales
the whole selected basket by the latest known Nasdaq-100 implied-volatility
index close.

For stock i on month-end decision date t:

    base_weight_{i,t}
        = 1 / max_positions    if i is selected by the base model
        = 0                    otherwise

    vxn_scale_t
        = clip(target_vxn_pct / VXN_t, min_scale, max_scale)

    target_weight_{i,t}
        = base_weight_{i,t} * vxn_scale_t

Execution mapping is unchanged:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import norgatedata
import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.friction_analysis import FrictionAnalysis
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    ATR_WINDOW_INT,
    AtrNormalizedNdxConfig,
    AtrNormalizedNdxStrategy,
    audit_pit_universe_df,
    compute_atr_normalized_signal_tables,
    get_atr_normalized_ndx_data,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


@dataclass(frozen=True)
class VxnScaledAtrNormalizedNdxConfig(AtrNormalizedNdxConfig):
    vxn_symbol_str: str = "$VXN"
    target_vxn_pct_float: float = 22.0
    min_exposure_scale_float: float = 0.25
    max_exposure_scale_float: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.vxn_symbol_str:
            raise ValueError("vxn_symbol_str must not be empty.")
        if self.target_vxn_pct_float <= 0.0:
            raise ValueError("target_vxn_pct_float must be positive.")
        if self.min_exposure_scale_float < 0.0:
            raise ValueError("min_exposure_scale_float must be non-negative.")
        if self.max_exposure_scale_float <= 0.0:
            raise ValueError("max_exposure_scale_float must be positive.")
        if self.min_exposure_scale_float > self.max_exposure_scale_float:
            raise ValueError("min_exposure_scale_float must be <= max_exposure_scale_float.")
        if self.max_exposure_scale_float > 1.0:
            raise ValueError("max_exposure_scale_float must be <= 1.0 for this no-leverage variant.")


DEFAULT_CONFIG = VxnScaledAtrNormalizedNdxConfig()

__all__ = [
    "ATR_WINDOW_INT",
    "DEFAULT_CONFIG",
    "VxnScaledAtrNormalizedNdxConfig",
    "VxnScaledAtrNormalizedNdxStrategy",
    "audit_pit_universe_df",
    "compute_atr_normalized_signal_tables",
    "compute_vxn_scale_signal_df",
    "get_asof_vxn_scale_float",
    "get_monthly_decision_close_df",
    "get_vxn_scaled_atr_normalized_ndx_data",
    "load_vxn_close_ser",
    "map_month_end_decision_dates_to_rebalance_schedule_df",
    "run_variant",
]


def load_vxn_close_ser(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None,
) -> pd.Series:
    """
    Load the VXN close series from Norgate.
    """
    vxn_price_df = norgatedata.price_timeseries(
        symbol_str,
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if len(vxn_price_df) == 0:
        raise RuntimeError(f"{symbol_str} returned no VXN helper data.")

    vxn_close_ser = vxn_price_df["Close"].astype(float).sort_index()
    vxn_close_ser.name = symbol_str
    return vxn_close_ser


def compute_vxn_scale_signal_df(
    vxn_close_ser: pd.Series,
    target_vxn_pct_float: float = DEFAULT_CONFIG.target_vxn_pct_float,
    min_exposure_scale_float: float = DEFAULT_CONFIG.min_exposure_scale_float,
    max_exposure_scale_float: float = DEFAULT_CONFIG.max_exposure_scale_float,
) -> pd.DataFrame:
    """
    Compute the daily VXN exposure scale.

    Formula:

        vxn_scale_t = clip(target_vxn_pct / VXN_t, min_scale, max_scale)
    """
    if target_vxn_pct_float <= 0.0:
        raise ValueError("target_vxn_pct_float must be positive.")
    if min_exposure_scale_float < 0.0:
        raise ValueError("min_exposure_scale_float must be non-negative.")
    if min_exposure_scale_float > max_exposure_scale_float:
        raise ValueError("min_exposure_scale_float must be <= max_exposure_scale_float.")
    if max_exposure_scale_float > 1.0:
        raise ValueError("max_exposure_scale_float must be <= 1.0 for this no-leverage variant.")

    clean_vxn_close_ser = vxn_close_ser.astype(float).sort_index().dropna()
    if len(clean_vxn_close_ser) == 0:
        raise ValueError("vxn_close_ser must contain at least one non-null close.")

    vxn_scale_signal_df = pd.DataFrame({"vxn_close": clean_vxn_close_ser})

    # *** CRITICAL*** VXN scaling uses only the VXN close observed on or before
    # the month-end decision close. No future VXN value may enter this series.
    raw_exposure_scale_ser = float(target_vxn_pct_float) / vxn_scale_signal_df["vxn_close"]
    exposure_scale_ser = raw_exposure_scale_ser.replace([np.inf, -np.inf], np.nan).clip(
        lower=float(min_exposure_scale_float),
        upper=float(max_exposure_scale_float),
    )
    vxn_scale_signal_df["vxn_exposure_scale_float"] = exposure_scale_ser
    return vxn_scale_signal_df.dropna(subset=["vxn_exposure_scale_float"])


def get_asof_vxn_scale_float(
    vxn_scale_signal_df: pd.DataFrame,
    decision_date_ts: pd.Timestamp,
) -> float:
    """
    Return the latest VXN exposure scale known on or before decision_date_ts.
    """
    if len(vxn_scale_signal_df) == 0:
        raise RuntimeError("vxn_scale_signal_df must not be empty.")
    if "vxn_exposure_scale_float" not in vxn_scale_signal_df.columns:
        raise RuntimeError("vxn_scale_signal_df must contain vxn_exposure_scale_float.")

    sorted_vxn_scale_signal_df = vxn_scale_signal_df.sort_index()
    if sorted_vxn_scale_signal_df.index.has_duplicates:
        raise RuntimeError("vxn_scale_signal_df index must not contain duplicates.")

    # *** CRITICAL*** This is an as-of lookup. If VXN has no row on the exact
    # stock decision date, use only the latest prior VXN close, never a later
    # row that would leak future volatility information into the rebalance.
    vxn_row_int = int(
        sorted_vxn_scale_signal_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if vxn_row_int < 0:
        raise RuntimeError(f"No VXN scale exists on or before decision date {decision_date_ts}.")

    exposure_scale_float = float(
        sorted_vxn_scale_signal_df.iloc[vxn_row_int]["vxn_exposure_scale_float"]
    )
    if not np.isfinite(exposure_scale_float) or exposure_scale_float < 0.0:
        raise RuntimeError(f"Invalid VXN exposure scale for decision date {decision_date_ts}.")
    return exposure_scale_float


class VxnScaledAtrNormalizedNdxStrategy(AtrNormalizedNdxStrategy):
    """
    ATR-normalized NDX momentum with a VXN total-exposure scaler.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        vxn_scale_signal_df: pd.DataFrame,
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
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_month_int=lookback_month_int,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        if len(vxn_scale_signal_df) == 0:
            raise ValueError("vxn_scale_signal_df must not be empty.")
        if "vxn_exposure_scale_float" not in vxn_scale_signal_df.columns:
            raise ValueError("vxn_scale_signal_df must contain vxn_exposure_scale_float.")

        self.vxn_scale_signal_df = vxn_scale_signal_df.copy().sort_index()

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        base_target_weight_ser = super().get_target_weight_ser(close_row_ser=close_row_ser)
        if len(base_target_weight_ser) == 0:
            return base_target_weight_ser

        exposure_scale_float = get_asof_vxn_scale_float(
            vxn_scale_signal_df=self.vxn_scale_signal_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        scaled_target_weight_ser = base_target_weight_ser * exposure_scale_float
        return scaled_target_weight_ser


def get_vxn_scaled_atr_normalized_ndx_data(
    config: VxnScaledAtrNormalizedNdxConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(config=config)
    vxn_close_ser = load_vxn_close_ser(
        symbol_str=config.vxn_symbol_str,
        start_date_str=config.history_start_date_str,
        end_date_str=config.end_date_str,
    )
    vxn_scale_signal_df = compute_vxn_scale_signal_df(
        vxn_close_ser=vxn_close_ser,
        target_vxn_pct_float=config.target_vxn_pct_float,
        min_exposure_scale_float=config.min_exposure_scale_float,
        max_exposure_scale_float=config.max_exposure_scale_float,
    )
    return pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> VxnScaledAtrNormalizedNdxStrategy:
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
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_data(config_obj)
    )

    strategy_obj = VxnScaledAtrNormalizedNdxStrategy(
        name="strategy_mo_atr_normalized_ndx_vxn_scaled",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for monthly ATR and trend features, but the executable calendar
    # starts at the first deployment fill session.
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


def build_friction_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
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
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_data(config_obj)
    )

    strategy_obj = VxnScaledAtrNormalizedNdxStrategy(
        name="strategy_mo_atr_normalized_ndx_vxn_scaled",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL *** FrictionAnalysis must assess the same completed order
    # ledger as the deployment-reference VXN-scaled NDX momentum backtest.
    # Keep pre-start history for monthly features, but execute only on the
    # configured calendar.
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

    strategy_obj.universe_df = None
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
    }


def run_friction_analysis(
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
):
    friction_input_dict = build_friction_analysis_inputs(
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    friction_analysis_obj = FrictionAnalysis(
        strategy_obj=friction_input_dict["strategy_obj"],
        pricing_data_df=friction_input_dict["pricing_data_df"],
        execution_policy_str=friction_input_dict["execution_policy_str"],
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
    )
    return friction_analysis_obj.run()


if __name__ == "__main__":
    run_variant()
