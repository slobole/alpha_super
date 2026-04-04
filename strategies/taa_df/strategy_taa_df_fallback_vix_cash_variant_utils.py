"""
Utilities for Defense First fallback variants with a VIX cash gate.

Overlay rule
------------
Let:

    ret_spy_t = C_spy_t / C_spy_{t-1} - 1

    rv20_t = std(ret_spy_{t-19:t}) * sqrt(252) * 100

At month-end m:

    if rv20_m < VIX_m:
        keep the fallback sleeve invested
    else:
        set fallback weight to 0 and leave the residual as cash

The base strategy signal and next-open execution timing remain unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable
import sys

import numpy as np
import pandas as pd
import norgatedata
from IPython.display import display

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    from strategies.taa_df.strategy_taa_df import (
        DefenseFirstConfig,
        map_month_end_weights_to_rebalance_open_df,
    )
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import (
        _build_defense_first_strategy,
        _run_strategy_from_weight_df,
    )
except ModuleNotFoundError:
    from strategy_taa_df import (
        DefenseFirstConfig,
        map_month_end_weights_to_rebalance_open_df,
    )
    from strategy_taa_df_fallback_variant_utils import (
        _build_defense_first_strategy,
        _run_strategy_from_weight_df,
    )

from alpha.engine.report import save_results


spy_realized_vol_symbol_str = "SPY"
vix_symbol_str = "$VIX"
rv_lookback_day_int = 20
trading_day_count_per_year_float = 252.0
vix_inception_date_str = "1990-01-02"


def build_vix_cash_variant_config(
    base_config: DefenseFirstConfig,
) -> DefenseFirstConfig:
    """
    Clip the start date so the VIX helper series exists.

    Formula:

        start_date^{variant}
            = max(start_date^{base}, first_VIX_date)
    """
    base_start_timestamp = pd.Timestamp(base_config.start_date_str)
    vix_start_timestamp = pd.Timestamp(vix_inception_date_str)
    effective_start_timestamp = max(base_start_timestamp, vix_start_timestamp)

    return replace(
        base_config,
        start_date_str=effective_start_timestamp.strftime("%Y-%m-%d"),
    )


def load_helper_close_ser(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None,
) -> pd.Series:
    """
    Load helper close data for the VIX cash gate.
    """
    helper_price_df = norgatedata.price_timeseries(
        symbol_str,
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if len(helper_price_df) == 0:
        raise RuntimeError(f"{symbol_str} returned no helper close data.")

    helper_close_ser = helper_price_df["Close"].astype(float).sort_index()
    helper_close_ser.name = symbol_str
    return helper_close_ser


def compute_daily_vrp_signal_df(
    spy_close_ser: pd.Series,
    vix_close_ser: pd.Series,
    realized_vol_lookback_day_int: int = rv_lookback_day_int,
) -> pd.DataFrame:
    """
    Compute the daily realized-volatility versus VIX gate table.

    Formulas:

        ret_spy_t = C_spy_t / C_spy_{t-1} - 1

        rv20_t = std(ret_spy_{t-19:t}) * sqrt(252) * 100

        vrp_gate_t = 1 if rv20_t < VIX_t else 0
    """
    helper_signal_df = pd.concat([spy_close_ser, vix_close_ser], axis=1, join="inner").dropna()
    helper_signal_df.columns = ["spy_close", "vix_close"]

    # *** CRITICAL*** The realized-volatility return series must use strictly
    # backward-looking close-to-close information. Any forward return would
    # leak future volatility into the current decision date.
    spy_ret_ser = helper_signal_df["spy_close"] / helper_signal_df["spy_close"].shift(1) - 1.0

    # *** CRITICAL*** The rolling realized-volatility window must be strictly
    # backward-looking. Centered or forward windows would leak future path
    # information into the month-end gate.
    rv20_ann_pct_ser = (
        spy_ret_ser.rolling(realized_vol_lookback_day_int).std(ddof=0)
        * float(np.sqrt(trading_day_count_per_year_float))
        * 100.0
    )
    vrp_gate_ser = (rv20_ann_pct_ser < helper_signal_df["vix_close"]).astype(float)

    daily_vrp_signal_df = pd.DataFrame(
        {
            "spy_close": helper_signal_df["spy_close"],
            "vix_close": helper_signal_df["vix_close"],
            "spy_ret": spy_ret_ser,
            "rv20_ann_pct": rv20_ann_pct_ser,
            "vrp_gate": vrp_gate_ser,
        }
    ).dropna()
    return daily_vrp_signal_df


def sample_month_end_vrp_signal_df(
    daily_vrp_signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sample the VRP helper table at month-end.
    """
    # *** CRITICAL*** Month-end sampling must use the last available trading day
    # in each month. Using any earlier observation would change the information
    # set available at the rebalance decision point.
    month_end_vrp_signal_df = daily_vrp_signal_df.resample("ME").last().dropna(how="any")
    return month_end_vrp_signal_df


def apply_vrp_cash_gate_to_month_end_weight_df(
    base_month_end_weight_df: pd.DataFrame,
    month_end_vrp_signal_df: pd.DataFrame,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the VIX cash gate to the fallback sleeve only.
    """
    fallback_asset_str = config.fallback_asset
    if fallback_asset_str not in base_month_end_weight_df.columns:
        raise ValueError(
            f"Fallback asset {fallback_asset_str} is missing from base_month_end_weight_df."
        )

    aligned_month_end_vrp_signal_df = month_end_vrp_signal_df.reindex(base_month_end_weight_df.index).dropna(how="any")
    if len(aligned_month_end_vrp_signal_df) == 0:
        raise RuntimeError("No overlapping month-end VRP signals were available for the base month-end weights.")

    aligned_base_month_end_weight_df = base_month_end_weight_df.reindex(aligned_month_end_vrp_signal_df.index).copy()
    final_month_end_weight_df = aligned_base_month_end_weight_df.copy()
    fallback_alias_str = fallback_asset_str.lower()
    diagnostic_row_list: list[dict[str, float | bool | pd.Timestamp]] = []

    for decision_date_ts in aligned_base_month_end_weight_df.index:
        target_weight_ser = aligned_base_month_end_weight_df.loc[decision_date_ts].copy()
        base_fallback_weight_float = float(target_weight_ser.loc[fallback_asset_str])
        rv20_ann_pct_float = float(aligned_month_end_vrp_signal_df.loc[decision_date_ts, "rv20_ann_pct"])
        vix_close_float = float(aligned_month_end_vrp_signal_df.loc[decision_date_ts, "vix_close"])
        vrp_gate_bool = bool(aligned_month_end_vrp_signal_df.loc[decision_date_ts, "vrp_gate"])

        if not vrp_gate_bool:
            target_weight_ser.loc[fallback_asset_str] = 0.0

        final_month_end_weight_df.loc[decision_date_ts] = target_weight_ser

        final_fallback_weight_float = float(target_weight_ser.loc[fallback_asset_str])
        tradeable_weight_sum_float = float(target_weight_ser.sum())
        cash_weight_float = float(1.0 - tradeable_weight_sum_float)
        if cash_weight_float < -1e-12:
            raise ValueError(
                f"Residual cash weight must be non-negative. Found {cash_weight_float:.12f} on {decision_date_ts}."
            )

        diagnostic_row_list.append(
            {
                "decision_date": decision_date_ts,
                "rv20_ann_pct": rv20_ann_pct_float,
                "vix_close": vix_close_float,
                "vrp_gate": vrp_gate_bool,
                "base_fallback_weight": base_fallback_weight_float,
                "final_fallback_weight": final_fallback_weight_float,
                f"base_{fallback_alias_str}_weight": base_fallback_weight_float,
                f"final_{fallback_alias_str}_weight": final_fallback_weight_float,
                "cash_weight": cash_weight_float,
                "tradeable_weight_sum": tradeable_weight_sum_float,
            }
        )

    month_end_vrp_diagnostic_df = pd.DataFrame(diagnostic_row_list).set_index("decision_date")
    return final_month_end_weight_df, month_end_vrp_diagnostic_df


def build_vrp_cash_overlay_weight_frames(
    base_month_end_weight_df: pd.DataFrame,
    month_end_vrp_signal_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the month-end VIX gate and map weights to next-month open dates.
    """
    month_end_weight_df, month_end_vrp_diagnostic_df = apply_vrp_cash_gate_to_month_end_weight_df(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_vrp_signal_df=month_end_vrp_signal_df,
        config=config,
    )

    # *** CRITICAL*** Month-end decisions must map to the first tradable open in
    # the following month. Same-bar execution would create look-ahead bias.
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df,
        execution_index,
    )
    return month_end_weight_df, rebalance_weight_df, month_end_vrp_diagnostic_df


def _load_vrp_overlay_signal_frames(
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy_close_ser = load_helper_close_ser(
        symbol_str=spy_realized_vol_symbol_str,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    vix_close_ser = load_helper_close_ser(
        symbol_str=vix_symbol_str,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    daily_vrp_signal_df = compute_daily_vrp_signal_df(
        spy_close_ser=spy_close_ser,
        vix_close_ser=vix_close_ser,
        realized_vol_lookback_day_int=rv_lookback_day_int,
    )
    month_end_vrp_signal_df = sample_month_end_vrp_signal_df(daily_vrp_signal_df)
    return daily_vrp_signal_df, month_end_vrp_signal_df


def get_standard_fallback_vix_cash_data(
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a standard fallback variant and apply the VIX cash overlay.
    """
    execution_price_df, momentum_score_df, base_month_end_weight_df, _ = base_data_loader_fn(config)
    daily_vrp_signal_df, month_end_vrp_signal_df = _load_vrp_overlay_signal_frames(config)
    month_end_weight_df, rebalance_weight_df, month_end_vrp_diagnostic_df = build_vrp_cash_overlay_weight_frames(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_vrp_signal_df=month_end_vrp_signal_df,
        execution_index=execution_price_df.index,
        config=config,
    )

    return (
        execution_price_df,
        momentum_score_df,
        daily_vrp_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    )


def get_linearity_1n_fallback_vix_cash_data(
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a linearity 1/n fallback variant and apply the VIX cash overlay.
    """
    (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        base_month_end_weight_df,
        _,
    ) = base_data_loader_fn(config)
    daily_vrp_signal_df, month_end_vrp_signal_df = _load_vrp_overlay_signal_frames(config)
    month_end_weight_df, rebalance_weight_df, month_end_vrp_diagnostic_df = build_vrp_cash_overlay_weight_frames(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_vrp_signal_df=month_end_vrp_signal_df,
        execution_index=execution_price_df.index,
        config=config,
    )

    return (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        daily_vrp_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    )


def run_standard_fallback_vix_cash_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    """
    Run a standard fallback variant with the VIX cash overlay.
    """
    (
        execution_price_df,
        momentum_score_df,
        daily_vrp_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    ) = get_standard_fallback_vix_cash_data(
        config=config,
        base_data_loader_fn=base_data_loader_fn,
    )

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
    )
    strategy.daily_vrp_signal_df = daily_vrp_signal_df.copy()
    strategy.month_end_vrp_diagnostic_df = month_end_vrp_diagnostic_df.copy()

    _run_strategy_from_weight_df(
        strategy=strategy,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("First momentum scores:")
        display(momentum_score_df.dropna().head())

        print("First daily VRP signals:")
        display(daily_vrp_signal_df.head())

        print("First month-end VRP diagnostics:")
        display(month_end_vrp_diagnostic_df.head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy


def run_linearity_1n_fallback_vix_cash_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    """
    Run a linearity 1/n fallback variant with the VIX cash overlay.
    """
    (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        daily_vrp_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    ) = get_linearity_1n_fallback_vix_cash_data(
        config=config,
        base_data_loader_fn=base_data_loader_fn,
    )

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
    )
    strategy.daily_vrp_signal_df = daily_vrp_signal_df.copy()
    strategy.month_end_vrp_diagnostic_df = month_end_vrp_diagnostic_df.copy()

    _run_strategy_from_weight_df(
        strategy=strategy,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("First daily linearity scores:")
        display(daily_linearity_score_df.dropna().head())

        print("First month-end linearity scores:")
        display(month_end_score_df.head())

        print("First daily VRP signals:")
        display(daily_vrp_signal_df.head())

        print("First month-end VRP diagnostics:")
        display(month_end_vrp_diagnostic_df.head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy
