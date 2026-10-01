"""Fast, vectorised replicas of the LIVE families' searches, for the S5 MCPT (thousands of re-runs).

The engine (`alpha.scout.family`) is authoritative for S4 numbers; these replicas only need to rank configurations
the way the engine does on real and permuted histories. Simplifications, identical on real and permuted data:
gross returns, cash at zero, weights decided at the close of T held at constant weight from the close of T+1
(the engine fills at the open of T+1 and lets weights drift within the month). `tests/test_scout_p5_stations.py`
(Norgate) checks the TAA replica against the engine; the P5 report records both replicas' agreement and where the
replica's plateau pick differs from the engine's (NDX: roc 15 vs roc 6; the MCPT FAIL holds for every pick).

TAA 3x (plain date-row shuffle; ETF timing family):
    matrix columns: total-return daily returns of GLD UUP TLT DBC BTAL TQQQ, SPY daily return, VIX level, DTB3 level.
NDX VXN selection (per-asset null on the Nasdaq-100 panel; cross-sectional ranking family):
    the strategy and its baseline share the overlay (SPY regime and VXN scale, on real dates), so the active return
    isolates stock selection: active = strategy − overlay × equal-weight members.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.stats.selection import neighbourhood_median_vec, plateau_choice

DAYS_INT = 252


def _sharpe(daily_vec: np.ndarray) -> float:
    sd_float = daily_vec.std(ddof=1)
    return float(daily_vec.mean() / sd_float * np.sqrt(DAYS_INT)) if sd_float > 0 else 0.0


def _hold_daily(weight_mat: np.ndarray, decision_row_vec: np.ndarray, return_mat: np.ndarray) -> np.ndarray:
    """Daily return of weights decided at the close of decision rows, held from the close of the next row."""
    date_count_int = return_mat.shape[0]
    row_of_day_vec = np.full(date_count_int, -1)
    for idx_int, row_int in enumerate(decision_row_vec):
        end_int = decision_row_vec[idx_int + 1] + 2 if idx_int + 1 < decision_row_vec.size else date_count_int
        row_of_day_vec[row_int + 2 : end_int] = idx_int
    held_mat = np.where(row_of_day_vec[:, None] >= 0, weight_mat[np.maximum(row_of_day_vec, 0)], 0.0)
    return (held_mat * np.nan_to_num(return_mat)).sum(axis=1)


# ---------------------------------------------------------------- TAA 3x
TAA_ASSET_TUPLE = ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ")


def taa_matrix(date_index: pd.DatetimeIndex, total_return_close_df: pd.DataFrame, spy_close_ser: pd.Series, vix_close_ser: pd.Series, dtb3_ser: pd.Series) -> np.ndarray:
    return_df = total_return_close_df[list(TAA_ASSET_TUPLE)].reindex(date_index).ffill().pct_change().fillna(0.0)
    spy_return_ser = spy_close_ser.reindex(date_index).ffill().pct_change().fillna(0.0)
    vix_ser = vix_close_ser.reindex(date_index).ffill()
    dtb3_level_ser = dtb3_ser.reindex(dtb3_ser.index.union(date_index)).ffill().reindex(date_index)
    return np.column_stack([return_df.to_numpy(), spy_return_ser.to_numpy(), vix_ser.to_numpy(), dtb3_level_ser.to_numpy()])


def taa_config_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, grid_config_list: list[dict]) -> list[np.ndarray]:
    return_mat, spy_return_vec, vix_vec, dtb3_vec = matrix[:, :6], matrix[:, 6], matrix[:, 7], matrix[:, 8]
    price_mat = np.cumprod(1.0 + return_mat[:, :5], axis=0)
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    decision_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]  # last month: no next fill
    month_price_mat = price_mat[decision_row_vec]
    hurdle_vec = (1.0 + dtb3_vec[decision_row_vec] / 100.0) ** (1.0 / 12.0) - 1.0
    daily_list = []
    for config_dict in grid_config_list:
        k_tuple = config_dict["momentum_month_tuple"]
        score_mat = np.full(month_price_mat.shape, np.nan)
        first_int = max(k_tuple)
        score_mat[first_int:] = np.mean([month_price_mat[first_int:] / month_price_mat[first_int - k : len(month_price_mat) - k] - 1.0 for k in k_tuple], axis=0)
        window_int = config_dict["realized_vol_window_int"]
        realized_vec = pd.Series(spy_return_vec).rolling(window_int).std(ddof=0).to_numpy() * np.sqrt(DAYS_INT) * 100.0
        weight_mat = np.zeros((decision_row_vec.size, 6))
        for m_int in range(first_int, decision_row_vec.size):
            order_vec = np.argsort(-score_mat[m_int], kind="stable")
            for slot_int, asset_int in enumerate(order_vec):
                rank_weight_float = (5 - slot_int) / 15.0
                if score_mat[m_int, asset_int] > hurdle_vec[m_int]:
                    weight_mat[m_int, asset_int] = rank_weight_float
                else:
                    weight_mat[m_int, 5] += rank_weight_float
            row_int = decision_row_vec[m_int]
            if not realized_vec[row_int] < vix_vec[row_int]:
                weight_mat[m_int, 5] = 0.0
        daily_list.append(_hold_daily(weight_mat, decision_row_vec, return_mat))
    return daily_list


# ---------------------------------------------------------------- NDX VXN selection
def ndx_selection_daily(
    open_mat, high_mat, low_mat, close_mat, raw_close_mat, member_mat,
    date_index: pd.DatetimeIndex, overlay_scale_ser: pd.Series, grid_config_list: list[dict],
) -> tuple[list[np.ndarray], np.ndarray]:
    """(daily return per configuration, overlay × equal-weight members baseline). Arrays are dates × stocks."""
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    decision_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return_mat = close_mat / np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]]) - 1.0
        previous_close_mat = np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]])
        true_range_mat = np.maximum(high_mat - low_mat, np.maximum(np.abs(high_mat - previous_close_mat), np.abs(low_mat - previous_close_mat)))
        atr_mat = pd.DataFrame(true_range_mat).rolling(20, min_periods=20).mean().to_numpy()
        atr_dollar_mat = atr_mat[decision_row_vec] * (raw_close_mat[decision_row_vec] / close_mat[decision_row_vec])
    scale_vec = overlay_scale_ser.reindex(date_index[decision_row_vec]).fillna(0.0).to_numpy()
    member_decision_mat = member_mat[decision_row_vec] & np.isfinite(close_mat[decision_row_vec])
    baseline_weight_mat = np.where(member_decision_mat, 1.0, 0.0)
    baseline_weight_mat = baseline_weight_mat / np.maximum(baseline_weight_mat.sum(axis=1, keepdims=True), 1.0) * scale_vec[:, None]
    baseline_vec = _hold_daily(baseline_weight_mat, decision_row_vec, return_mat)

    sma_cache_dict, daily_list = {}, []
    month_close_mat = close_mat[decision_row_vec]
    for config_dict in grid_config_list:
        sma_int = config_dict["stock_sma_int"]
        if sma_int not in sma_cache_dict:
            sma_cache_dict[sma_int] = pd.DataFrame(close_mat).rolling(sma_int, min_periods=sma_int).mean().to_numpy()[decision_row_vec]
        roc_int, top_int = config_dict["roc_month_int"], config_dict["top_count_int"]
        weight_mat = np.zeros_like(month_close_mat)
        with np.errstate(invalid="ignore", divide="ignore"):
            roc_mat = np.full_like(month_close_mat, np.nan)
            roc_mat[roc_int:] = month_close_mat[roc_int:] / month_close_mat[:-roc_int] - 1.0
            score_mat = roc_mat / atr_dollar_mat
        eligible_mat = member_decision_mat & (month_close_mat > sma_cache_dict[sma_int]) & np.isfinite(score_mat)
        ranked_mat = np.where(eligible_mat, score_mat, -np.inf)
        top_idx_mat = np.argsort(-ranked_mat, axis=1, kind="stable")[:, :top_int]
        chosen_mat = np.take_along_axis(eligible_mat, top_idx_mat, axis=1)
        row_idx_mat = np.broadcast_to(np.arange(month_close_mat.shape[0])[:, None], top_idx_mat.shape)
        weight_mat[row_idx_mat[chosen_mat], top_idx_mat[chosen_mat]] = 1.0 / top_int
        weight_mat *= scale_vec[:, None]
        daily_list.append(_hold_daily(weight_mat, decision_row_vec, return_mat))
    return daily_list, baseline_vec


def plateau_score(sharpe_vec: np.ndarray, grid_shape_tuple: tuple[int, ...]) -> float:
    return plateau_choice(np.asarray(sharpe_vec, dtype=float), grid_shape_tuple).own_sharpe_float


__all__ = ["ndx_selection_daily", "neighbourhood_median_vec", "plateau_score", "taa_config_daily_list", "taa_matrix"]
