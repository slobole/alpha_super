"""Family C: residual-momentum score tables in the L shell (PREREG section 4, market-only model per decision A8).

At each month-end decision t (a row of the cache's month-end schedule), for stock i:

    r_{i,m}      = Close_ME(m) / Close_ME(m-1) - 1        monthly CAPITALSPECIAL price returns (extended history from 1996)
    x_m          = SPY monthly price return on the same schedule
    over the W months ending at t (at least 0.8 W valid pairs):
        r_{i,m} = a_i + b_i x_m + e_{i,m}                 OLS with intercept
    RES12-1_i(t) = mean(e_{i,t-11..t-1}) / std(e_{i,t-11..t-1})     (11 residuals, skips the last month)
    RES12-0_i(t) = mean(e_{i,t-11..t})   / std(e_{i,t-11..t})       (12 residuals)
    TOT12-1_i(t) = mean(r_{i,t-11..t-1}) / std(r_{i,t-11..t-1})     (control, no regression)

std uses ddof = 1 (sample standard deviation; an implementation reading, recorded in the amendments file). A score is
NaN unless every residual (return) in its 11- or 12-month window is finite and the standard deviation is positive.
Everything is a ratio of returns, so it is invariant to a constant rescaling of a stock's prices.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

import ndx_param_robustness_core as core  # noqa: E402

from trend_breakout_20260927.cells import CCell


def c_core_cell(cell: CCell, liquidity_str: str) -> core.Cell:
    """The L shell (N 10 EW, SMA100, SPY gate, VXN 22/0.25, month-end) ranked on the override score."""
    return core.Cell(
        numerator_str=cell.numerator_key_str,
        denominator_str="none",
        n_int=10,
        weight_str="EW",
        stock_filter_int=100,
        regime_str="SPY",
        buffer_int=0,
        offset_int=0,
        vxn_target_float=22.0,
        vxn_floor_float=0.25,
        liquidity_str=liquidity_str,
    )


def a0_reference_cell(liquidity_str: str) -> core.Cell:
    return dataclasses.replace(core.ANCHOR_CELL, liquidity_str=liquidity_str)


def _standardised_mean(value_arr: np.ndarray) -> np.ndarray:
    """mean / sample std over axis 0; NaN unless every value is finite and std > 0."""
    all_finite_vec = np.isfinite(value_arr).all(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vec = value_arr.mean(axis=0)
        std_vec = value_arr.std(axis=0, ddof=1)
        score_vec = mean_vec / std_vec
    return np.where(all_finite_vec & np.isfinite(score_vec) & (std_vec > 0), score_vec, np.nan)


def build_c_score_tables(universe_dict: dict, monthly_close_df: pd.DataFrame, cell_list: list[CCell]) -> dict[str, np.ndarray]:
    """Score tables (schedule rows x symbols) keyed by CCell.numerator_key_str, aligned to the cache's month-end index."""
    symbol_list = list(universe_dict["symbol_list"])
    month_end_index = pd.DatetimeIndex(universe_dict["month_end_index"])
    close_df = monthly_close_df.reindex(columns=["SPY"] + symbol_list)
    # *** CRITICAL *** month-end to month-end returns on the decision schedule; nothing after the decision month-end.
    return_df = close_df / close_df.shift(1) - 1.0
    return_arr = return_df[symbol_list].to_numpy(dtype=np.float64)
    market_vec = return_df["SPY"].to_numpy(dtype=np.float64)
    ext_index = pd.DatetimeIndex(monthly_close_df.index)
    ext_pos_vec = ext_index.get_indexer(month_end_index)
    if (ext_pos_vec < 0).any():
        raise RuntimeError("cache month-ends missing from the extended monthly close panel")

    table_dict: dict[str, np.ndarray] = {}
    for cell in cell_list:
        table_arr = np.full((len(month_end_index), len(symbol_list)), np.nan)
        for row_int, j_int in enumerate(ext_pos_vec):
            if cell.score_str == "TOT12-1":
                if j_int - 11 < 0:
                    continue
                table_arr[row_int] = _standardised_mean(return_arr[j_int - 11 : j_int])
                continue
            window_int = cell.window_int
            start_int = j_int - window_int + 1
            if start_int < 0:
                continue
            y_arr = return_arr[start_int : j_int + 1]
            x_vec = market_vec[start_int : j_int + 1]
            valid_arr = np.isfinite(y_arr) & np.isfinite(x_vec)[:, None]
            count_vec = valid_arr.sum(axis=0)
            ok_vec = count_vec >= 0.8 * window_int
            with np.errstate(invalid="ignore", divide="ignore"):
                x_masked_arr = np.where(valid_arr, x_vec[:, None], 0.0)
                y_masked_arr = np.where(valid_arr, y_arr, 0.0)
                x_mean_vec = x_masked_arr.sum(axis=0) / count_vec
                y_mean_vec = y_masked_arr.sum(axis=0) / count_vec
                x_dev_arr = np.where(valid_arr, x_vec[:, None] - x_mean_vec, 0.0)
                y_dev_arr = np.where(valid_arr, y_arr - y_mean_vec, 0.0)
                beta_vec = (x_dev_arr * y_dev_arr).sum(axis=0) / (x_dev_arr**2).sum(axis=0)
                alpha_vec = y_mean_vec - beta_vec * x_mean_vec
                residual_arr = y_arr - alpha_vec - beta_vec * x_vec[:, None]
            residual_arr = np.where(valid_arr, residual_arr, np.nan)
            if cell.score_str == "RES12-1":
                score_vec = _standardised_mean(residual_arr[-12:-1])
            elif cell.score_str == "RES12-0":
                score_vec = _standardised_mean(residual_arr[-12:])
            else:
                raise ValueError(cell.score_str)
            table_arr[row_int] = np.where(ok_vec, score_vec, np.nan)
        table_dict[cell.numerator_key_str] = table_arr
    return table_dict
