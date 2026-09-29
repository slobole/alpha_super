"""Retrospective, transfer-adjusted attribution of a saved portfolio path.

This explains the aggregation model; it does not replay executions or add a
cost model. All attribution must reconcile to the saved NAV before use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_attribution_dict(portfolio_obj) -> dict:
    equity_df = getattr(portfolio_obj, "_pod_equities", None)
    return_df = getattr(portfolio_obj, "_daily_rets", None)
    result_df = getattr(portfolio_obj, "results", None)
    if any(frame_obj is None or frame_obj.empty for frame_obj in (equity_df, return_df, result_df)):
        raise ValueError("Saved sleeve equity or daily returns are unavailable.")
    nav_series = result_df["total_value"].astype(float)
    date_index = nav_series.index
    if len(date_index) < 2 or not date_index.is_unique or not date_index.is_monotonic_increasing:
        raise ValueError("Attribution needs at least two unique, ordered dates.")
    if not equity_df.index.equals(date_index) or not return_df.index.equals(date_index) or not equity_df.columns.equals(return_df.columns):
        raise ValueError("Sleeve and portfolio dates or columns do not match.")
    if not equity_df.columns.is_unique or not all(np.isfinite(frame_obj.to_numpy(dtype=float)).all() for frame_obj in (equity_df, return_df)) or not np.isfinite(nav_series.to_numpy()).all() or (nav_series <= 0).any():
        raise ValueError("Attribution inputs must be finite with positive portfolio NAV.")
    if not np.allclose(equity_df.sum(axis=1), nav_series, rtol=1e-9, atol=1e-6):
        raise ValueError("Saved sleeve equities do not reconcile to portfolio NAV.")
    # *** CRITICAL *** retrospective accounting boundary: capital on day T is
    # prior-close sleeve equity, except when SAVED applied targets redistribute
    # prior-close total NAV before T's return. Never infer targets from future
    # observations or treat the capital transfer as investment profit.
    # C_i,T = E_i,T-1, or V_T-1 * w_i,T on applied rebalance dates.
    opening_capital_df = equity_df.shift(1)
    opening_capital_df.iloc[0] = 0.0
    applied_index = pd.DatetimeIndex(getattr(portfolio_obj, "_rebalance_date_index", []))
    target_df = getattr(portfolio_obj, "rebalance_target_weight_df", None)
    if getattr(portfolio_obj, "_rebalance", None):
        if target_df is None or not hasattr(portfolio_obj, "_rebalance_date_index"):
            raise ValueError("Saved applied rebalance targets are unavailable.")
        if not target_df.index.equals(applied_index) or not applied_index.is_unique or not applied_index.isin(date_index[1:]).all():
            raise ValueError("Saved rebalance targets do not match applied dates.")
        if len(applied_index):
            if not target_df.columns.equals(equity_df.columns) or not np.isfinite(target_df.to_numpy(dtype=float)).all() or (target_df < 0).any().any() or not np.allclose(target_df.sum(axis=1), 1.0):
                raise ValueError("Saved rebalance target weights are invalid.")
            opening_capital_df.loc[applied_index] = target_df.mul(nav_series.shift(1).loc[applied_index], axis=0)
    # PnL_i,T = C_i,T * r_i,T. Sum_i PnL_i,T must equal V_T - V_T-1.
    daily_pnl_df = opening_capital_df * return_df
    daily_pnl_df.iloc[0] = 0.0
    if not np.allclose((opening_capital_df + daily_pnl_df).iloc[1:], equity_df.iloc[1:], rtol=1e-9, atol=1e-6):
        raise ValueError("Individual sleeve P&L does not reconcile to saved sleeve equity.")
    if not np.allclose(daily_pnl_df.sum(axis=1).iloc[1:], nav_series.diff().iloc[1:], rtol=1e-9, atol=1e-6):
        raise ValueError("Sleeve P&L does not reconcile to the saved portfolio path.")
    full_pnl_series = daily_pnl_df.sum()
    contribution_df = pd.DataFrame({
        "pnl": full_pnl_series,
        "return_pp": 100.0 * full_pnl_series / float(nav_series.iloc[0]),
    })
    # *** CRITICAL *** retrospective episode selection only, never a signal:
    # select the first deepest trough and its latest preceding high-water mark.
    drawdown_series = nav_series / nav_series.cummax() - 1.0
    trough_obj = drawdown_series.idxmin()
    peak_obj = None
    episode_count_int = 0
    if float(drawdown_series.loc[trough_obj]) < 0.0:
        pre_trough_series = nav_series.loc[:trough_obj]
        peak_obj = pre_trough_series[pre_trough_series == pre_trough_series.max()].index[-1]
        episode_mask = (date_index > peak_obj) & (date_index <= trough_obj)
        episode_count_int = int(episode_mask.sum())
        episode_pnl_series = daily_pnl_df.loc[episode_mask].sum()
        contribution_df["drawdown_pnl"] = episode_pnl_series
        contribution_df["drawdown_pp"] = 100.0 * episode_pnl_series / float(nav_series.loc[peak_obj])
        contribution_df["episode_standalone_pct"] = 100.0 * ((1.0 + return_df.loc[episode_mask]).prod() - 1.0)
    return {
        "contribution_df": contribution_df,
        "daily_pnl_df": daily_pnl_df,
        "peak_obj": peak_obj,
        "trough_obj": trough_obj if peak_obj is not None else None,
        "episode_count_int": episode_count_int,
        "observation_count_int": len(date_index) - 1,
    }
