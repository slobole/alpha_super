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


def taa_matrix(date_index: pd.DatetimeIndex, total_return_close_df: pd.DataFrame, spy_close_ser: pd.Series, vix_close_ser: pd.Series,
               dtb3_ser: pd.Series, asset_tuple: tuple = TAA_ASSET_TUPLE) -> np.ndarray:
    """Columns: TR daily returns of `asset_tuple` (defensive ETFs, then the fallback last), SPY return, VIX, DTB3."""
    return_df = total_return_close_df[list(asset_tuple)].reindex(date_index).ffill().pct_change().fillna(0.0)
    spy_return_ser = spy_close_ser.reindex(date_index).ffill().pct_change().fillna(0.0)
    vix_ser = vix_close_ser.reindex(date_index).ffill()
    dtb3_level_ser = dtb3_ser.reindex(dtb3_ser.index.union(date_index)).ffill().reindex(date_index)
    return np.column_stack([return_df.to_numpy(), spy_return_ser.to_numpy(), vix_ser.to_numpy(), dtb3_level_ser.to_numpy()])


def taa_config_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, grid_config_list: list[dict], asset_tuple: tuple = TAA_ASSET_TUPLE,
                          slot_weight_str: str = "rank", score_str: str = "momentum") -> list[np.ndarray]:
    """Daily gross returns of each configuration. The TAA family (alpha/scout/specs/taa_3x.py): defensive slots by
    momentum (vs the DTB3 hurdle) or linearity (vs a threshold), rank (5..1)/15 or equal 1/N slot weights, failed
    slots to the fallback (last asset), the fallback gated to cash unless SPY realised volatility < VIX."""
    from alpha.scout.specs.taa_3x import _linearity_lookback_df

    refuse_ablation_switches(grid_config_list)

    asset_count_int = len(asset_tuple)
    defensive_count_int = asset_count_int - 1
    return_mat = matrix[:, :asset_count_int]
    spy_return_vec, vix_vec, dtb3_vec = matrix[:, asset_count_int], matrix[:, asset_count_int + 1], matrix[:, asset_count_int + 2]
    price_mat = np.cumprod(1.0 + return_mat[:, :defensive_count_int], axis=0)
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    decision_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]  # last month: no next fill
    month_price_mat = price_mat[decision_row_vec]
    hurdle_vec = (1.0 + dtb3_vec[decision_row_vec] / 100.0) ** (1.0 / 12.0) - 1.0
    slot_weight_vec = np.arange(defensive_count_int, 0, -1) / (defensive_count_int * (defensive_count_int + 1) / 2) if slot_weight_str == "rank" else np.full(defensive_count_int, 1.0 / defensive_count_int)
    linearity_cache_dict: dict = {}
    daily_list = []
    for config_dict in grid_config_list:
        if score_str == "momentum":
            k_tuple = config_dict["momentum_month_tuple"]
            first_int = max(k_tuple)
            score_mat = np.full(month_price_mat.shape, np.nan)
            score_mat[first_int:] = np.mean([month_price_mat[first_int:] / month_price_mat[first_int - k : len(month_price_mat) - k] - 1.0 for k in k_tuple], axis=0)
            threshold_vec = hurdle_vec
        else:
            day_tuple = tuple(config_dict["linearity_day_tuple"])
            for day_int in day_tuple:
                if day_int not in linearity_cache_dict:
                    linearity_cache_dict[day_int] = _linearity_lookback_df(pd.DataFrame(np.log(price_mat)), day_int).to_numpy()[decision_row_vec]
            score_mat = np.mean([linearity_cache_dict[d] for d in day_tuple], axis=0)
            first_int = int(np.argmax(np.isfinite(score_mat).all(axis=1)))
            threshold_vec = np.full(decision_row_vec.size, float(config_dict.get("linearity_threshold_float", 0.0)))
        window_int = config_dict["realized_vol_window_int"]
        realized_vec = pd.Series(spy_return_vec).rolling(window_int).std(ddof=0).to_numpy() * np.sqrt(DAYS_INT) * 100.0
        weight_mat = np.zeros((decision_row_vec.size, asset_count_int))
        for m_int in range(first_int, decision_row_vec.size):
            order_vec = np.argsort(-score_mat[m_int], kind="stable")
            for slot_int, asset_int in enumerate(order_vec):
                if score_mat[m_int, asset_int] > threshold_vec[m_int]:
                    weight_mat[m_int, asset_int] = slot_weight_vec[slot_int]
                else:
                    weight_mat[m_int, -1] += slot_weight_vec[slot_int]
            row_int = decision_row_vec[m_int]
            if not realized_vec[row_int] < vix_vec[row_int]:
                weight_mat[m_int, -1] = 0.0
        daily_list.append(_hold_daily(weight_mat, decision_row_vec, return_mat))
    return daily_list


# A15 ablation switches (spec config fields) that the fast replicas do NOT implement: a replica given one would
# silently run the live rule, so it refuses them (parity review 2026-10-02). Ablations run through the full engine.
ABLATION_SWITCH_DEFAULT_DICT = {
    "cash_hurdle_bool": True, "vix_gate_bool": True, "defensive_hold_str": "assets", "fallback_hold_str": "asset",
    "cash_asset_tuple": (), "regime_filter_bool": True, "stock_trend_filter_bool": True, "adaptive_speed_bool": True,
    "trend_rule_bool": True, "trend_fast_sma_int": 0, "trend_threshold_float": 0.0, "trend_filter_str": "sma",
    "cmma_threshold_float": 0.0, "cmma_atr_int": 252, "lt_lookback_int": 252, "lt_atr_int": 20, "lt_rsq_bool": True,
    "sector_cap_int": 0, "sector_level_int": 1,
}


def refuse_ablation_switches(config_list: list) -> None:
    """Raise if any configuration (a dict or a spec config) sets an A15 ablation switch away from the engine rule."""
    for config in config_list:
        value_dict = config if isinstance(config, dict) else {k: getattr(config, k) for k in ABLATION_SWITCH_DEFAULT_DICT if hasattr(config, k)}
        for name_str, default_obj in ABLATION_SWITCH_DEFAULT_DICT.items():
            if name_str in value_dict and value_dict[name_str] != default_obj:
                raise ValueError(f"The fast replica does not implement the ablation switch {name_str}={value_dict[name_str]!r}.")


# ---------------------------------------------------------------- NDX VXN selection
def ndx_selection_daily(
    open_mat, high_mat, low_mat, close_mat, raw_close_mat, member_mat,
    date_index: pd.DatetimeIndex, overlay_scale_ser: pd.Series, grid_config_list: list[dict], atr_unit_str: str = "dollar",
) -> tuple[list[np.ndarray], np.ndarray]:
    """(daily return per configuration, overlay × equal-weight members baseline). Arrays are dates × stocks.

    atr_unit_str "dollar": score = ROC / ATR20 in dollars of day T (NDX VXN, NDX ATR); "percent": score = ROC / (ATR20 /
    Close), the NATR20 siblings (unit-free, so adjusted and raw prices give the same number)."""
    refuse_ablation_switches(grid_config_list)
    if atr_unit_str not in ("dollar", "percent"):
        raise ValueError(f"The NDX replica ranks by ROC / ATR in dollars or percent, not {atr_unit_str!r}.")
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    decision_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return_mat = close_mat / np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]]) - 1.0
        previous_close_mat = np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]])
        true_range_mat = np.maximum(high_mat - low_mat, np.maximum(np.abs(high_mat - previous_close_mat), np.abs(low_mat - previous_close_mat)))
        atr_mat = pd.DataFrame(true_range_mat).rolling(20, min_periods=20).mean().to_numpy()
        if atr_unit_str == "dollar":
            atr_dollar_mat = atr_mat[decision_row_vec] * (raw_close_mat[decision_row_vec] / close_mat[decision_row_vec])
        else:
            atr_dollar_mat = atr_mat[decision_row_vec] / close_mat[decision_row_vec]  # NATR: the "denominator" is ATR / Close
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
