"""
Utilities for Defense First fallback variants with an inverse-volatility
Risk Parity overlay.

The overlay is applied after the base TAA/VIX-cash target weights are already
known. It preserves the existing monthly signal, VIX cash gate, and next-open
execution timing. Only the active tradeable weights are resized.

Core formulas
-------------
For tradeable asset i:

    r_{i,t} = Close_{i,t} / Close_{i,t-1} - 1

    sigma_{i,t} = std(r_{i,t-L+1:t}) * sqrt(252)

At month-end decision date m:

    active_m = {i | base_weight_{i,m} > 0}

    raw_{i,m} = 1 / sigma_{i,m}

    rp_weight_{i,m}
        = active_budget_m * raw_{i,m} / sum_j raw_{j,m}

where active_budget_m is the sum of base weights for active assets that have
valid volatility. Assets with missing, zero, or non-finite volatility are
skipped and their original base weight remains cash.
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
        get_standard_fallback_vix_cash_data,
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
        get_standard_fallback_vix_cash_data,
    )

from alpha.engine.report import save_results


risk_parity_lookback_day_int = 63
trading_day_count_per_year_float = 252.0


def extract_tradeable_close_df(
    execution_price_df: pd.DataFrame,
    tradeable_asset_list: tuple[str, ...],
) -> pd.DataFrame:
    """
    Extract tradeable close prices from the execution OHLC frame.
    """
    close_ser_map: dict[str, pd.Series] = {}
    for asset_str in tradeable_asset_list:
        close_key_tuple = (asset_str, "Close")
        if close_key_tuple not in execution_price_df.columns:
            raise RuntimeError(f"Missing execution close column: {close_key_tuple}")
        close_ser_map[asset_str] = execution_price_df.loc[:, close_key_tuple].astype(float)

    tradeable_close_df = pd.DataFrame(close_ser_map).sort_index()
    return tradeable_close_df


def compute_daily_risk_parity_volatility_df(
    tradeable_close_df: pd.DataFrame,
    lookback_day_int: int = risk_parity_lookback_day_int,
) -> pd.DataFrame:
    """
    Compute trailing annualized volatility for inverse-volatility weighting.

    Formula:

        r_{i,t} = Close_{i,t} / Close_{i,t-1} - 1

        sigma_{i,t} = std(r_{i,t-L+1:t}) * sqrt(252)
    """
    if lookback_day_int <= 1:
        raise ValueError("lookback_day_int must be greater than 1.")

    # *** CRITICAL*** Risk Parity returns must be close-to-close returns known
    # by the month-end decision close. A forward return here would leak future
    # volatility into the next-month rebalance decision.
    tradeable_return_df = tradeable_close_df / tradeable_close_df.shift(1) - 1.0

    # *** CRITICAL*** The rolling volatility window is strictly backward
    # looking. Centered windows or future-filled data would create lookahead.
    daily_volatility_df = (
        tradeable_return_df.rolling(int(lookback_day_int)).std(ddof=0)
        * float(np.sqrt(trading_day_count_per_year_float))
    )
    return daily_volatility_df


def sample_month_end_risk_parity_volatility_df(
    daily_volatility_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sample the latest available volatility in each calendar month.
    """
    # *** CRITICAL*** Month-end sampling uses the last available trading-day
    # observation inside each month, matching the TAA decision information set.
    month_end_volatility_df = daily_volatility_df.resample("ME").last()
    return month_end_volatility_df


def _valid_volatility_asset_list(
    active_asset_list: list[str],
    volatility_ser: pd.Series,
) -> list[str]:
    valid_asset_list: list[str] = []
    for asset_str in active_asset_list:
        volatility_float = float(volatility_ser.get(asset_str, np.nan))
        if np.isfinite(volatility_float) and volatility_float > 0.0:
            valid_asset_list.append(asset_str)
    return valid_asset_list


def apply_risk_parity_to_month_end_weight_df(
    base_month_end_weight_df: pd.DataFrame,
    month_end_volatility_df: pd.DataFrame,
    config: DefenseFirstConfig,
    lookback_day_int: int = risk_parity_lookback_day_int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reweight active tradeable assets by inverse trailing volatility.

    Existing cash is preserved. If an active asset has invalid volatility, that
    asset is skipped and its original base weight remains cash.
    """
    tradeable_asset_list = list(config.tradeable_asset_list)
    missing_asset_list = [
        asset_str
        for asset_str in tradeable_asset_list
        if asset_str not in base_month_end_weight_df.columns
    ]
    if len(missing_asset_list) > 0:
        raise ValueError(f"Missing base weight columns: {missing_asset_list}")

    aligned_volatility_df = month_end_volatility_df.reindex(base_month_end_weight_df.index)
    if aligned_volatility_df[tradeable_asset_list].dropna(how="all").empty:
        raise RuntimeError("No overlapping month-end Risk Parity volatility values were available.")

    final_month_end_weight_df = pd.DataFrame(
        0.0,
        index=base_month_end_weight_df.index,
        columns=tradeable_asset_list,
        dtype=float,
    )
    diagnostic_row_list: list[dict[str, float | int | pd.Timestamp]] = []

    for decision_date_ts in base_month_end_weight_df.index:
        base_weight_ser = base_month_end_weight_df.loc[decision_date_ts, tradeable_asset_list].fillna(0.0).astype(float)
        volatility_ser = aligned_volatility_df.loc[decision_date_ts, tradeable_asset_list].astype(float)
        active_asset_list = [
            asset_str
            for asset_str in tradeable_asset_list
            if float(base_weight_ser.loc[asset_str]) > 1e-12
        ]
        valid_asset_list = _valid_volatility_asset_list(active_asset_list, volatility_ser)

        target_weight_ser = pd.Series(0.0, index=tradeable_asset_list, dtype=float)
        valid_active_budget_float = float(base_weight_ser.loc[valid_asset_list].sum()) if valid_asset_list else 0.0
        skipped_weight_float = float(base_weight_ser.loc[active_asset_list].sum() - valid_active_budget_float)

        if valid_asset_list:
            inverse_volatility_ser = 1.0 / volatility_ser.loc[valid_asset_list]
            inverse_volatility_sum_float = float(inverse_volatility_ser.sum())
            if inverse_volatility_sum_float <= 0.0 or not np.isfinite(inverse_volatility_sum_float):
                raise RuntimeError(f"Invalid inverse-volatility sum on {decision_date_ts}.")
            target_weight_ser.loc[valid_asset_list] = (
                valid_active_budget_float
                * inverse_volatility_ser
                / inverse_volatility_sum_float
            )

        tradeable_weight_sum_float = float(target_weight_ser.sum())
        cash_weight_float = float(1.0 - tradeable_weight_sum_float)
        if cash_weight_float < -1e-12:
            raise ValueError(
                f"Residual cash weight must be non-negative. Found {cash_weight_float:.12f} on {decision_date_ts}."
            )

        final_month_end_weight_df.loc[decision_date_ts] = target_weight_ser

        diagnostic_row_dict: dict[str, float | int | pd.Timestamp] = {
            "decision_date": decision_date_ts,
            "lookback_day_int": int(lookback_day_int),
            "base_tradeable_weight": float(base_weight_ser.sum()),
            "final_tradeable_weight": tradeable_weight_sum_float,
            "cash_weight": max(cash_weight_float, 0.0),
            "active_asset_count": int(len(active_asset_list)),
            "valid_asset_count": int(len(valid_asset_list)),
            "skipped_weight": skipped_weight_float,
        }
        for asset_str in tradeable_asset_list:
            asset_alias_str = asset_str.lower()
            diagnostic_row_dict[f"base_{asset_alias_str}_weight"] = float(base_weight_ser.loc[asset_str])
            diagnostic_row_dict[f"final_{asset_alias_str}_weight"] = float(target_weight_ser.loc[asset_str])
            diagnostic_row_dict[f"{asset_alias_str}_volatility"] = float(volatility_ser.loc[asset_str])
        diagnostic_row_list.append(diagnostic_row_dict)

    month_end_risk_parity_diagnostic_df = pd.DataFrame(diagnostic_row_list).set_index("decision_date")
    return final_month_end_weight_df, month_end_risk_parity_diagnostic_df


def build_risk_parity_overlay_weight_frames(
    base_month_end_weight_df: pd.DataFrame,
    execution_price_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
    config: DefenseFirstConfig,
    lookback_day_int: int = risk_parity_lookback_day_int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the month-end Risk Parity overlay and map to next-month opens.
    """
    tradeable_close_df = extract_tradeable_close_df(
        execution_price_df=execution_price_df,
        tradeable_asset_list=config.tradeable_asset_list,
    )
    daily_volatility_df = compute_daily_risk_parity_volatility_df(
        tradeable_close_df=tradeable_close_df,
        lookback_day_int=lookback_day_int,
    )
    month_end_volatility_df = sample_month_end_risk_parity_volatility_df(daily_volatility_df)
    month_end_weight_df, month_end_risk_parity_diagnostic_df = apply_risk_parity_to_month_end_weight_df(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_volatility_df=month_end_volatility_df,
        config=config,
        lookback_day_int=lookback_day_int,
    )

    # *** CRITICAL*** Month-end Risk Parity weights are still decisions at
    # month-end close and execute only at the first tradable open of next month.
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_index,
    )
    return (
        daily_volatility_df,
        month_end_volatility_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_risk_parity_diagnostic_df,
    )


def get_standard_fallback_vix_cash_risk_parity_data(
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    lookback_day_int: int = risk_parity_lookback_day_int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load a standard fallback VIX-cash variant and apply Risk Parity sizing.
    """
    (
        execution_price_df,
        momentum_score_df,
        daily_vrp_signal_df,
        base_month_end_weight_df,
        _base_rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    ) = get_standard_fallback_vix_cash_data(
        config=config,
        base_data_loader_fn=base_data_loader_fn,
    )

    (
        daily_risk_parity_volatility_df,
        month_end_risk_parity_volatility_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_risk_parity_diagnostic_df,
    ) = build_risk_parity_overlay_weight_frames(
        base_month_end_weight_df=base_month_end_weight_df,
        execution_price_df=execution_price_df,
        execution_index=pd.DatetimeIndex(execution_price_df.index),
        config=config,
        lookback_day_int=lookback_day_int,
    )

    return (
        execution_price_df,
        momentum_score_df,
        daily_vrp_signal_df,
        daily_risk_parity_volatility_df,
        month_end_risk_parity_volatility_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
        month_end_risk_parity_diagnostic_df,
    )


def run_standard_fallback_vix_cash_risk_parity_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    base_data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    lookback_day_int: int = risk_parity_lookback_day_int,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
):
    """
    Run a standard fallback VIX-cash variant with Risk Parity sizing.
    """
    (
        execution_price_df,
        momentum_score_df,
        daily_vrp_signal_df,
        daily_risk_parity_volatility_df,
        month_end_risk_parity_volatility_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
        month_end_risk_parity_diagnostic_df,
    ) = get_standard_fallback_vix_cash_risk_parity_data(
        config=config,
        base_data_loader_fn=base_data_loader_fn,
        lookback_day_int=lookback_day_int,
    )

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
        capital_base_float=capital_base_float,
    )
    strategy.daily_vrp_signal_df = daily_vrp_signal_df.copy()
    strategy.daily_risk_parity_volatility_df = daily_risk_parity_volatility_df.copy()
    strategy.month_end_risk_parity_volatility_df = month_end_risk_parity_volatility_df.copy()
    strategy.month_end_vrp_diagnostic_df = month_end_vrp_diagnostic_df.copy()
    strategy.month_end_risk_parity_diagnostic_df = month_end_risk_parity_diagnostic_df.copy()
    strategy.risk_parity_lookback_day_int = int(lookback_day_int)

    _run_strategy_from_weight_df(
        strategy=strategy,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=backtest_start_date_str,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("First momentum scores:")
        display(momentum_score_df.dropna().head())

        print("First daily VRP signals:")
        display(daily_vrp_signal_df.head())

        print("First month-end Risk Parity diagnostics:")
        display(month_end_risk_parity_diagnostic_df.head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy
