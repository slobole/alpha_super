"""Robustness and stress helpers for the fund product menu study.

Everything here takes daily return series that already exist; nothing chooses
weights. The tests answer "how much would the conclusion move if ...", never
"which variant looks best".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common

TRADING_DAY_COUNT_INT = common.TRADING_DAY_COUNT_INT


# ─── stationary bootstrap ───────────────────────────────────────────────────


def stationary_bootstrap_index_mat(
    observation_count_int: int,
    replication_count_int: int,
    mean_block_length_float: float,
    seed_int: int,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap row indices, shape (reps, n).

    Each path starts at a uniform random row; with probability p = 1/L a new
    block starts at a fresh uniform row, otherwise the next row (wrapping) is
    taken. Rows of all series are resampled together, so cross-series
    dependence (book vs benchmark vs T-bill) is preserved.
    """
    rng = np.random.default_rng(seed_int)
    restart_probability_float = 1.0 / mean_block_length_float
    index_mat = np.empty((replication_count_int, observation_count_int), dtype=np.int64)
    index_mat[:, 0] = rng.integers(0, observation_count_int, size=replication_count_int)
    restart_mat = rng.random((replication_count_int, observation_count_int)) < restart_probability_float
    fresh_mat = rng.integers(0, observation_count_int, size=(replication_count_int, observation_count_int))
    for column_int in range(1, observation_count_int):
        continued_arr = (index_mat[:, column_int - 1] + 1) % observation_count_int
        index_mat[:, column_int] = np.where(restart_mat[:, column_int], fresh_mat[:, column_int], continued_arr)
    return index_mat


def bootstrap_summary_df(
    return_df: pd.DataFrame,
    tbill_ser: pd.Series,
    comparison_column_str: str,
    replication_count_int: int = 2000,
    mean_block_length_float: float = 63.0,
    seed_int: int = 20260923,
) -> pd.DataFrame:
    """Per column: 5/50/95% of CAGR, Sharpe (rf 0) and MaxDD across bootstrap paths,
    P(excess CAGR over T-bills > 0) and P(Sharpe > comparison column's Sharpe) on the same paths."""
    value_mat = return_df.to_numpy(dtype=float)
    tbill_arr = tbill_ser.reindex(return_df.index).fillna(0.0).to_numpy(dtype=float)
    observation_count_int = value_mat.shape[0]
    year_count_float = observation_count_int / TRADING_DAY_COUNT_INT
    index_mat = stationary_bootstrap_index_mat(
        observation_count_int, replication_count_int, mean_block_length_float, seed_int
    )
    comparison_position_int = list(return_df.columns).index(comparison_column_str)
    sharpe_mat = np.empty((replication_count_int, value_mat.shape[1]))
    cagr_mat = np.empty_like(sharpe_mat)
    maxdd_mat = np.empty_like(sharpe_mat)
    excess_cagr_mat = np.empty_like(sharpe_mat)
    for replication_int in range(replication_count_int):
        sample_mat = value_mat[index_mat[replication_int]]
        sample_tbill_arr = tbill_arr[index_mat[replication_int]]
        mean_arr = sample_mat.mean(axis=0)
        std_arr = sample_mat.std(axis=0, ddof=1)
        sharpe_mat[replication_int] = mean_arr / std_arr * np.sqrt(TRADING_DAY_COUNT_INT)
        growth_arr = np.prod(1.0 + sample_mat, axis=0)
        cagr_mat[replication_int] = growth_arr ** (1.0 / year_count_float) - 1.0
        tbill_growth_float = float(np.prod(1.0 + sample_tbill_arr))
        excess_cagr_mat[replication_int] = cagr_mat[replication_int] - (tbill_growth_float ** (1.0 / year_count_float) - 1.0)
        nav_mat = np.cumprod(1.0 + sample_mat, axis=0)
        maxdd_mat[replication_int] = (nav_mat / np.maximum.accumulate(nav_mat, axis=0) - 1.0).min(axis=0)
    row_list = []
    for column_position_int, column_str in enumerate(return_df.columns):
        row_list.append(
            {
                "series_str": column_str,
                "cagr_p05_float": np.quantile(cagr_mat[:, column_position_int], 0.05),
                "cagr_p50_float": np.quantile(cagr_mat[:, column_position_int], 0.50),
                "cagr_p95_float": np.quantile(cagr_mat[:, column_position_int], 0.95),
                "sharpe_p05_float": np.quantile(sharpe_mat[:, column_position_int], 0.05),
                "sharpe_p50_float": np.quantile(sharpe_mat[:, column_position_int], 0.50),
                "sharpe_p95_float": np.quantile(sharpe_mat[:, column_position_int], 0.95),
                "maxdd_p05_float": np.quantile(maxdd_mat[:, column_position_int], 0.05),
                "maxdd_p50_float": np.quantile(maxdd_mat[:, column_position_int], 0.50),
                "maxdd_p95_float": np.quantile(maxdd_mat[:, column_position_int], 0.95),
                "prob_excess_cagr_positive_float": float((excess_cagr_mat[:, column_position_int] > 0).mean()),
                f"prob_sharpe_above_{comparison_column_str}_float": float(
                    (sharpe_mat[:, column_position_int] > sharpe_mat[:, comparison_position_int]).mean()
                ),
            }
        )
    return pd.DataFrame(row_list).set_index("series_str")


# ─── stresses on sleeve returns ─────────────────────────────────────────────


def extra_slippage_cost_ser(
    transaction_df: pd.DataFrame,
    nav_ser: pd.Series,
    extra_cost_per_side_float: float,
) -> pd.Series:
    """Daily return drag from charging extra_cost on every traded dollar.

    cost_t = extra_cost * sum(|signed notional traded on t|) / NAV_{t-1}
    Using the prior close NAV keeps the drag in return units of that day.
    """
    traded_notional_ser = transaction_df.groupby("date")["signed_notional_float"].apply(lambda s: s.abs().sum())
    prior_nav_ser = nav_ser.shift(1)
    drag_ser = (traded_notional_ser * extra_cost_per_side_float).reindex(nav_ser.index).fillna(0.0) / prior_nav_ser
    return drag_ser.fillna(0.0)


def financing_cost_ser(path_df: pd.DataFrame, tbill_annual_rate_ser: pd.Series, spread_float: float) -> pd.Series:
    """Daily drag if negative cash were financed at T-bill + spread (IBKR-style margin).

    *** CRITICAL*** the deficit is the prior close cash balance; the rate is the
    lagged T-bill rate already aligned by the caller. The engine charges nothing
    for negative cash (gap G-023), so this is an upper-bound style stress.
    """
    nav_ser = path_df["total_value_float"]
    deficit_weight_ser = (-path_df["cash_float"]).clip(lower=0.0).shift(1) / nav_ser.shift(1)
    calendar_day_ser = pd.Series(nav_ser.index, index=nav_ser.index).diff().dt.days.fillna(0.0)
    rate_ser = tbill_annual_rate_ser.reindex(nav_ser.index).ffill().fillna(0.0) + spread_float
    return (deficit_weight_ser * rate_ser * calendar_day_ser / 360.0).fillna(0.0)


def cash_interest_uplift_ser(path_df: pd.DataFrame, tbill_annual_rate_ser: pd.Series, haircut_float: float) -> pd.Series:
    """Daily return the engine forgoes by crediting idle cash at 0%.

    uplift_t = max(cash_{t-1}, 0) / NAV_{t-1} * max(rate_t - haircut, 0) * days / 360
    (IBKR pays roughly the benchmark rate minus 0.5% on USD balances.)
    """
    nav_ser = path_df["total_value_float"]
    cash_weight_ser = path_df["cash_float"].clip(lower=0.0).shift(1) / nav_ser.shift(1)
    calendar_day_ser = pd.Series(nav_ser.index, index=nav_ser.index).diff().dt.days.fillna(0.0)
    rate_ser = (tbill_annual_rate_ser.reindex(nav_ser.index).ffill().fillna(0.0) - haircut_float).clip(lower=0.0)
    return (cash_weight_ser * rate_ser * calendar_day_ser / 360.0).fillna(0.0)


def lagged_tbill_annual_rate_ser(session_index: pd.DatetimeIndex) -> pd.Series:
    """DTB3 in decimal, lagged one observation (see common.load_tbill_return_ser)."""
    dtb3_df = pd.read_csv(common.DTB3_CSV_PATH, parse_dates=["observation_date"], na_values=["."])
    dtb3_ser = dtb3_df.set_index("observation_date")["DTB3"].astype(float).dropna().sort_index()
    rate_ser = dtb3_ser.reindex(dtb3_ser.index.union(session_index)).ffill().shift(1).reindex(session_index)
    return (rate_ser / 100.0).fillna(0.0)
