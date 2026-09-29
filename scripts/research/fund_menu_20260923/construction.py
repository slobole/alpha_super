"""Weight construction for the fund product menu: risk budgeting, no return forecasts.

A product is described by risk budgets, never by expected returns:

    budget_i = group_share(g(i)) * engine_share(e(i)) * member_share(i)

where group_share is the product's dial (risk share of the stock-market-driven
engine group vs the diversifier group), engine_share splits a group's budget
across its engines, and member_share splits an engine across implementations.

Capital weights solve the classic risk-budgeting problem (Roncalli 2013): each
sleeve's Euler risk contribution equals its budget,

    w_i * (Sigma w)_i / (w' Sigma w) = budget_i,  sum_i w_i = 1,  w_i > 0,

obtained from the strictly convex program

    y* = argmin_y  0.5 * y' Sigma y - sum_i budget_i * ln(y_i),   w = y* / sum(y*).

Only the covariance matrix enters, so the weights cannot be steered by the
historical returns being evaluated. The dial is then set by a one-dimensional
search so the book's ex-ante volatility equals the product's target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize


def risk_budget_weight_ser(covariance_df: pd.DataFrame, budget_ser: pd.Series) -> pd.Series:
    """Long-only weights whose Euler risk contributions equal the given budgets."""
    alias_list = list(budget_ser.index)
    budget_arr = budget_ser.to_numpy(dtype=float)
    if (budget_arr <= 0).any() or abs(budget_arr.sum() - 1.0) > 1e-9:
        raise ValueError("Budgets must be positive and sum to 1.")
    sigma_mat = covariance_df.loc[alias_list, alias_list].to_numpy(dtype=float)

    def objective_float(y_arr: np.ndarray) -> float:
        return 0.5 * float(y_arr @ sigma_mat @ y_arr) - float(budget_arr @ np.log(y_arr))

    def gradient_arr(y_arr: np.ndarray) -> np.ndarray:
        return sigma_mat @ y_arr - budget_arr / y_arr

    start_arr = budget_arr / np.sqrt(np.diag(sigma_mat))
    start_arr = start_arr / np.sqrt(start_arr @ sigma_mat @ start_arr)
    result_obj = minimize(
        objective_float,
        start_arr,
        jac=gradient_arr,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * len(alias_list),
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    if not result_obj.success:
        raise RuntimeError(f"Risk-budget solve failed: {result_obj.message}")
    weight_arr = result_obj.x / result_obj.x.sum()
    realised_share_arr = risk_share_arr(sigma_mat, weight_arr)
    if np.max(np.abs(realised_share_arr - budget_arr)) > 1e-6:
        raise RuntimeError("Risk-budget solution does not reproduce its budgets.")
    return pd.Series(weight_arr, index=alias_list)


def risk_share_arr(sigma_mat: np.ndarray, weight_arr: np.ndarray) -> np.ndarray:
    """Euler risk contribution shares: w_i (Sigma w)_i / (w' Sigma w)."""
    marginal_arr = sigma_mat @ weight_arr
    return weight_arr * marginal_arr / float(weight_arr @ marginal_arr)


def budget_ser_for_dial(
    dial_float: float,
    group_engine_share_dict: dict[str, dict[str, float]],
    engine_member_share_dict: dict[str, dict[str, float]],
) -> pd.Series:
    """Sleeve budgets from the group dial, engine shares and member shares.

    group_engine_share_dict = {"equity": {engine: share}, "diversifier": {engine: share}}
    The dial is the equity group's share; the diversifier group gets 1 - dial.
    """
    group_share_dict = {"equity": dial_float, "diversifier": 1.0 - dial_float}
    budget_dict: dict[str, float] = {}
    for group_str, engine_share_dict in group_engine_share_dict.items():
        if group_share_dict[group_str] <= 0:
            continue
        for engine_str, engine_share_float in engine_share_dict.items():
            for alias_str, member_share_float in engine_member_share_dict[engine_str].items():
                budget_dict[alias_str] = budget_dict.get(alias_str, 0.0) + (
                    group_share_dict[group_str] * engine_share_float * member_share_float
                )
    budget_ser = pd.Series(budget_dict)
    return budget_ser / budget_ser.sum()


def book_volatility_float(covariance_df: pd.DataFrame, weight_ser: pd.Series) -> float:
    sigma_mat = covariance_df.loc[weight_ser.index, weight_ser.index].to_numpy(dtype=float)
    weight_arr = weight_ser.to_numpy(dtype=float)
    return float(np.sqrt(weight_arr @ sigma_mat @ weight_arr))


def solve_dial_for_target_volatility(
    covariance_df: pd.DataFrame,
    target_volatility_float: float,
    group_engine_share_dict: dict[str, dict[str, float]],
    engine_member_share_dict: dict[str, dict[str, float]],
    dial_bound_tuple: tuple[float, float] = (0.01, 0.99),
) -> tuple[float, pd.Series]:
    """Dial whose risk-budget book has the target ex-ante volatility (annualised covariance)."""

    def excess_volatility_float(dial_float: float) -> float:
        budget_ser = budget_ser_for_dial(dial_float, group_engine_share_dict, engine_member_share_dict)
        weight_ser = risk_budget_weight_ser(covariance_df, budget_ser)
        return book_volatility_float(covariance_df, weight_ser) - target_volatility_float

    low_float, high_float = dial_bound_tuple
    low_excess_float = excess_volatility_float(low_float)
    high_excess_float = excess_volatility_float(high_float)
    if low_excess_float > 0 or high_excess_float < 0:
        raise ValueError(
            f"Target volatility {target_volatility_float:.4f} unreachable: dial range gives "
            f"{low_excess_float + target_volatility_float:.4f}..{high_excess_float + target_volatility_float:.4f}"
        )
    dial_float = brentq(excess_volatility_float, low_float, high_float, xtol=1e-10)
    budget_ser = budget_ser_for_dial(dial_float, group_engine_share_dict, engine_member_share_dict)
    return dial_float, risk_budget_weight_ser(covariance_df, budget_ser)


def round_weight_ser(weight_ser: pd.Series, step_float: float = 0.01) -> pd.Series:
    """Round to a step while keeping the sum at exactly 1 (largest-remainder method)."""
    unit_int = int(round(1.0 / step_float))
    raw_unit_ser = weight_ser / weight_ser.sum() * unit_int
    floor_ser = np.floor(raw_unit_ser).astype(int)
    remainder_ser = raw_unit_ser - floor_ser
    missing_int = unit_int - int(floor_ser.sum())
    for alias_str in remainder_ser.sort_values(ascending=False).index[:missing_int]:
        floor_ser[alias_str] += 1
    return floor_ser.astype(float) / unit_int
