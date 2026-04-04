"""
Utilities for Defense First fallback variants with a multi-lookback VIX cash split.

Overlay rule
------------
Let:

    ret_spy_t = C_spy_t / C_spy_{t-1} - 1

    rvL_t = std(ret_spy_{t-L+1:t}) * sqrt(252) * 100

for:

    L in {10, 15, 20}

Define the month-end breach count:

    breach_count_m = sum_L 1[rvL_m >= VIX_m]

Then the fallback sleeve split is:

    cash_frac_m = breach_count_m / 3
    fallback_frac_m = 1 - cash_frac_m

So:

    final_fallback_weight_m = base_fallback_weight_m * fallback_frac_m
    residual_cash_weight_m = base_fallback_weight_m * cash_frac_m

The base strategy signal and next-open execution timing remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import sys

import numpy as np
import pandas as pd
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
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        load_helper_close_ser,
        spy_realized_vol_symbol_str,
        trading_day_count_per_year_float,
        vix_symbol_str,
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
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        load_helper_close_ser,
        spy_realized_vol_symbol_str,
        trading_day_count_per_year_float,
        vix_symbol_str,
    )

from alpha.engine.report import save_results


rv_lookback_day_tuple: tuple[int, ...] = (10, 15, 20)


def compute_daily_multi_rv_signal_df(
    spy_close_ser: pd.Series,
    vix_close_ser: pd.Series,
    realized_vol_lookback_day_tuple: tuple[int, ...] = rv_lookback_day_tuple,
) -> pd.DataFrame:
    """
    Compute the daily multi-lookback realized-volatility versus VIX helper table.

    Formulas:

        ret_spy_t = C_spy_t / C_spy_{t-1} - 1

        rvL_t = std(ret_spy_{t-L+1:t}) * sqrt(252) * 100

        breach_count_t = sum_L 1[rvL_t >= VIX_t]

        cash_frac_t = breach_count_t / K
        fallback_frac_t = 1 - cash_frac_t

    where:

        K = len(realized_vol_lookback_day_tuple)
    """
    helper_signal_df = pd.concat([spy_close_ser, vix_close_ser], axis=1, join="inner").dropna()
    helper_signal_df.columns = ["spy_close", "vix_close"]

    # *** CRITICAL*** The realized-volatility return series must use strictly
    # backward-looking close-to-close information. Any forward return would
    # leak future volatility into the current decision date.
    spy_ret_ser = helper_signal_df["spy_close"] / helper_signal_df["spy_close"].shift(1) - 1.0

    daily_signal_dict: dict[str, pd.Series] = {
        "spy_close": helper_signal_df["spy_close"],
        "vix_close": helper_signal_df["vix_close"],
        "spy_ret": spy_ret_ser,
    }
    rv_column_name_list: list[str] = []

    for realized_vol_lookback_day_int in realized_vol_lookback_day_tuple:
        rv_column_name_str = f"rv{realized_vol_lookback_day_int}_ann_pct"
        rv_column_name_list.append(rv_column_name_str)

        # *** CRITICAL*** Each rolling realized-volatility window must be
        # strictly backward-looking. Centered or forward windows would leak
        # future path information into the month-end gate.
        rv_ann_pct_ser = (
            spy_ret_ser.rolling(realized_vol_lookback_day_int).std(ddof=0)
            * float(np.sqrt(trading_day_count_per_year_float))
            * 100.0
        )
        daily_signal_dict[rv_column_name_str] = rv_ann_pct_ser

    daily_multi_rv_signal_df = pd.DataFrame(daily_signal_dict).dropna()
    breach_flag_df = daily_multi_rv_signal_df[rv_column_name_list].ge(
        daily_multi_rv_signal_df["vix_close"],
        axis=0,
    )
    breach_count_ser = breach_flag_df.sum(axis=1).astype(int)
    cash_frac_ser = breach_count_ser / float(len(realized_vol_lookback_day_tuple))
    fallback_frac_ser = 1.0 - cash_frac_ser

    daily_multi_rv_signal_df["breach_count"] = breach_count_ser
    daily_multi_rv_signal_df["cash_frac"] = cash_frac_ser
    daily_multi_rv_signal_df["fallback_frac"] = fallback_frac_ser
    return daily_multi_rv_signal_df


def sample_month_end_multi_rv_signal_df(
    daily_multi_rv_signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sample the multi-lookback VRP helper table at month-end.
    """
    # *** CRITICAL*** Month-end sampling must use the last available trading day
    # in each month. Using any earlier observation would change the information
    # set available at the rebalance decision point.
    month_end_multi_rv_signal_df = daily_multi_rv_signal_df.resample("ME").last().dropna(how="any")
    return month_end_multi_rv_signal_df


def apply_multi_rv_cash_gate_to_month_end_weight_df(
    base_month_end_weight_df: pd.DataFrame,
    month_end_multi_rv_signal_df: pd.DataFrame,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the multi-lookback VIX cash split to the fallback sleeve only.
    """
    fallback_asset_str = config.fallback_asset
    if fallback_asset_str not in base_month_end_weight_df.columns:
        raise ValueError(
            f"Fallback asset {fallback_asset_str} is missing from base_month_end_weight_df."
        )

    aligned_month_end_multi_rv_signal_df = month_end_multi_rv_signal_df.reindex(
        base_month_end_weight_df.index
    ).dropna(how="any")
    if len(aligned_month_end_multi_rv_signal_df) == 0:
        raise RuntimeError("No overlapping month-end multi-RV signals were available for the base month-end weights.")

    aligned_base_month_end_weight_df = base_month_end_weight_df.reindex(
        aligned_month_end_multi_rv_signal_df.index
    ).copy()
    final_month_end_weight_df = aligned_base_month_end_weight_df.copy()
    fallback_alias_str = fallback_asset_str.lower()
    fallback_frac_column_str = "fallback_frac"
    alias_fallback_frac_column_str = f"{fallback_alias_str}_frac"
    if fallback_frac_column_str not in aligned_month_end_multi_rv_signal_df.columns:
        if alias_fallback_frac_column_str in aligned_month_end_multi_rv_signal_df.columns:
            fallback_frac_column_str = alias_fallback_frac_column_str
        else:
            raise ValueError(
                "month_end_multi_rv_signal_df must contain either fallback_frac "
                f"or {alias_fallback_frac_column_str}."
            )
    rv_column_name_list = [
        f"rv{realized_vol_lookback_day_int}_ann_pct"
        for realized_vol_lookback_day_int in rv_lookback_day_tuple
    ]
    diagnostic_row_list: list[dict[str, float | int | bool | pd.Timestamp]] = []

    for decision_date_ts in aligned_base_month_end_weight_df.index:
        target_weight_ser = aligned_base_month_end_weight_df.loc[decision_date_ts].copy()
        base_fallback_weight_float = float(target_weight_ser.loc[fallback_asset_str])
        base_tradeable_weight_sum_float = float(target_weight_ser.sum())
        base_cash_weight_float = float(1.0 - base_tradeable_weight_sum_float)
        fallback_frac_float = float(aligned_month_end_multi_rv_signal_df.loc[decision_date_ts, fallback_frac_column_str])
        cash_frac_float = float(aligned_month_end_multi_rv_signal_df.loc[decision_date_ts, "cash_frac"])
        breach_count_int = int(aligned_month_end_multi_rv_signal_df.loc[decision_date_ts, "breach_count"])
        vix_close_float = float(aligned_month_end_multi_rv_signal_df.loc[decision_date_ts, "vix_close"])

        target_weight_ser.loc[fallback_asset_str] = base_fallback_weight_float * fallback_frac_float
        final_month_end_weight_df.loc[decision_date_ts] = target_weight_ser

        final_fallback_weight_float = float(target_weight_ser.loc[fallback_asset_str])
        tradeable_weight_sum_float = float(target_weight_ser.sum())
        cash_weight_float = float(1.0 - tradeable_weight_sum_float)
        overlay_cash_weight_increment_float = float(cash_weight_float - base_cash_weight_float)
        if cash_weight_float < -1e-12:
            raise ValueError(
                f"Residual cash weight must be non-negative. Found {cash_weight_float:.12f} on {decision_date_ts}."
            )
        if abs((final_fallback_weight_float + overlay_cash_weight_increment_float) - base_fallback_weight_float) > 1e-12:
            raise ValueError(
                "Fallback sleeve split must preserve the original fallback weight. "
                f"Found base={base_fallback_weight_float:.12f}, final_fallback={final_fallback_weight_float:.12f}, "
                f"overlay_cash_increment={overlay_cash_weight_increment_float:.12f} on {decision_date_ts}."
            )

        diagnostic_row_dict: dict[str, float | int | bool | pd.Timestamp] = {
            "decision_date": decision_date_ts,
            "vix_close": vix_close_float,
            "breach_count": breach_count_int,
            "cash_frac": cash_frac_float,
            "fallback_frac": fallback_frac_float,
            "base_cash_weight": base_cash_weight_float,
            alias_fallback_frac_column_str: fallback_frac_float,
            "base_fallback_weight": base_fallback_weight_float,
            "final_fallback_weight": final_fallback_weight_float,
            f"base_{fallback_alias_str}_weight": base_fallback_weight_float,
            f"final_{fallback_alias_str}_weight": final_fallback_weight_float,
            "cash_weight": cash_weight_float,
            "overlay_cash_weight_increment": overlay_cash_weight_increment_float,
            "tradeable_weight_sum": tradeable_weight_sum_float,
        }
        for rv_column_name_str in rv_column_name_list:
            diagnostic_row_dict[rv_column_name_str] = float(
                aligned_month_end_multi_rv_signal_df.loc[decision_date_ts, rv_column_name_str]
            )
        diagnostic_row_list.append(diagnostic_row_dict)

    month_end_multi_rv_diagnostic_df = pd.DataFrame(diagnostic_row_list).set_index("decision_date")
    return final_month_end_weight_df, month_end_multi_rv_diagnostic_df


def build_multi_rv_cash_overlay_weight_frames(
    base_month_end_weight_df: pd.DataFrame,
    month_end_multi_rv_signal_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the month-end multi-lookback VIX split and map weights to next-month open dates.
    """
    month_end_weight_df, month_end_multi_rv_diagnostic_df = apply_multi_rv_cash_gate_to_month_end_weight_df(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_multi_rv_signal_df=month_end_multi_rv_signal_df,
        config=config,
    )

    # *** CRITICAL*** Month-end decisions must map to the first tradable open in
    # the following month. Same-bar execution would create look-ahead bias.
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df,
        execution_index,
    )
    return month_end_weight_df, rebalance_weight_df, month_end_multi_rv_diagnostic_df


def _load_multi_rv_overlay_signal_frames(
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
    daily_multi_rv_signal_df = compute_daily_multi_rv_signal_df(
        spy_close_ser=spy_close_ser,
        vix_close_ser=vix_close_ser,
        realized_vol_lookback_day_tuple=rv_lookback_day_tuple,
    )
    month_end_multi_rv_signal_df = sample_month_end_multi_rv_signal_df(daily_multi_rv_signal_df)
    return daily_multi_rv_signal_df, month_end_multi_rv_signal_df


def get_standard_fallback_vix_cash_multi_rv_data(
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a standard fallback variant and apply the multi-lookback VIX cash split.
    """
    execution_price_df, momentum_score_df, base_month_end_weight_df, _ = base_data_loader_fn(config)
    daily_multi_rv_signal_df, month_end_multi_rv_signal_df = _load_multi_rv_overlay_signal_frames(config)
    month_end_weight_df, rebalance_weight_df, month_end_multi_rv_diagnostic_df = build_multi_rv_cash_overlay_weight_frames(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_multi_rv_signal_df=month_end_multi_rv_signal_df,
        execution_index=execution_price_df.index,
        config=config,
    )

    return (
        execution_price_df,
        momentum_score_df,
        daily_multi_rv_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_multi_rv_diagnostic_df,
    )


def get_linearity_1n_fallback_vix_cash_multi_rv_data(
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a linearity 1/n fallback variant and apply the multi-lookback VIX cash split.
    """
    (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        base_month_end_weight_df,
        _,
    ) = base_data_loader_fn(config)
    daily_multi_rv_signal_df, month_end_multi_rv_signal_df = _load_multi_rv_overlay_signal_frames(config)
    month_end_weight_df, rebalance_weight_df, month_end_multi_rv_diagnostic_df = build_multi_rv_cash_overlay_weight_frames(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_multi_rv_signal_df=month_end_multi_rv_signal_df,
        execution_index=execution_price_df.index,
        config=config,
    )

    return (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        daily_multi_rv_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_multi_rv_diagnostic_df,
    )


def run_standard_fallback_vix_cash_multi_rv_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    """
    Run a standard fallback variant with the multi-lookback VIX cash split.
    """
    (
        execution_price_df,
        momentum_score_df,
        daily_multi_rv_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_multi_rv_diagnostic_df,
    ) = get_standard_fallback_vix_cash_multi_rv_data(
        config=config,
        base_data_loader_fn=base_data_loader_fn,
    )

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
    )
    strategy.daily_multi_rv_signal_df = daily_multi_rv_signal_df.copy()
    strategy.month_end_multi_rv_diagnostic_df = month_end_multi_rv_diagnostic_df.copy()

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

        print("First daily multi-RV signals:")
        display(daily_multi_rv_signal_df.head())

        print("First month-end multi-RV diagnostics:")
        display(month_end_multi_rv_diagnostic_df.head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy


def run_linearity_1n_fallback_vix_cash_multi_rv_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    """
    Run a linearity 1/n fallback variant with the multi-lookback VIX cash split.
    """
    (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        daily_multi_rv_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_multi_rv_diagnostic_df,
    ) = get_linearity_1n_fallback_vix_cash_multi_rv_data(
        config=config,
        base_data_loader_fn=base_data_loader_fn,
    )

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
    )
    strategy.daily_multi_rv_signal_df = daily_multi_rv_signal_df.copy()
    strategy.month_end_multi_rv_diagnostic_df = month_end_multi_rv_diagnostic_df.copy()

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

        print("First daily multi-RV signals:")
        display(daily_multi_rv_signal_df.head())

        print("First month-end multi-RV diagnostics:")
        display(month_end_multi_rv_diagnostic_df.head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy
