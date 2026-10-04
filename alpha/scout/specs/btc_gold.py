"""Scout spec of a NEW family (discovery, not an engine pod): Bitcoin + gold trend with a per-asset volatility cap.

Source: GQResearch, "Dual Momentum Without the Dual" (2026), audited in Pakal (reports/gqr_dual_momentum_audit,
866 engine runs, verdict: diagnostic - "risk management, not alpha"). Scout's questions (registered before this run):
(1) does the trend signal add anything beyond the volatility cap (MCPT, A9 score), and (2) does a small Bitcoin + gold
sleeve improve the live book against T-bills in the same sleeve (S6)?

Rule ("hold both", the article's improved version):
    weekly decision at the close of the week's first session (+ decision_offset_int sessions: the luck band)
    per lookback L: both returns over L sessions > 0 -> 50/50; only one -> 100% that one; neither -> cash
    base_i  = mean over L in lookback_tuple of that allocation      (signal_bool False: base_i = 0.5, the control)
    cap_i,t = min(1, cap_float / sigma_i,t), sigma = max(vol21, vol63) ("max21_63") or vol63, annualised, at close t
    weight_i,t = base_i x cap_i,t; BIL holds the rest. The cap is re-applied every session (the article: "adjusted daily").
Execution (Scout's fast-replica convention): weights decided at the close of T are held from the close of T+1.
Costs: per side on traded value, `cost_model.slippage_float` (default 10 bp, Pakal's central layer); no per-share fee.

Data (local, read-only, no download): Pakal's cached panel research_cache/gqr_dual_momentum_audit/panel.parquet
(Norgate TOTALRETURN closes of GBTC, IBIT, GLD, BIL; Binance BTCUSDT sampled at 16:00 ET as SPOT). Bitcoin legs:
"splice" = GBTC until 2024-01-10, IBIT from 2024-01-11 (the tradeable leg, as the article); "spot" = real BTC.
*** CRITICAL*** every feature at T uses closes up to T; the panel's SPOT 16:00 ET sample matches the ETF close time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult

PANEL_PATH = Path(r"C:\Users\User\Documents\workspace\Pakal\pakal-research\research_cache\gqr_dual_momentum_audit\panel.parquet")
SPLICE_TS = pd.Timestamp("2024-01-11")
START_STR = "2017-10-02"
ASSET_TUPLE = ("BTC", "GLD", "BIL")
DEFAULT_COST_MODEL = CostModel(slippage_float=0.0010, fee_per_share_float=0.0, min_fee_float=0.0)


@dataclass(frozen=True)
class BtcGoldConfig:
    lookback_tuple: tuple = (21, 42, 63)
    cap_float: float = 0.20
    vol_mode_str: str = "max21_63"
    signal_bool: bool = True
    btc_leg_str: str = "splice"
    decision_offset_int: int = 0


LIVE_CONFIG = BtcGoldConfig()


@dataclass(frozen=True)
class BtcGoldInputs:
    return_df: pd.DataFrame  # daily total returns: BTC_splice, BTC_spot, GLD, BIL


def load_inputs() -> BtcGoldInputs:
    panel_df = pd.read_parquet(PANEL_PATH)
    splice_ser = pd.concat([panel_df["GBTC_close"].pct_change().loc[: SPLICE_TS - pd.Timedelta(days=1)],
                            panel_df["IBIT_close"].pct_change().loc[SPLICE_TS + pd.Timedelta(days=1):]])
    splice_ser.loc[SPLICE_TS] = panel_df.loc[SPLICE_TS, "GBTC_close"] / panel_df["GBTC_close"].shift(1).loc[SPLICE_TS] - 1.0
    return_df = pd.DataFrame({
        "BTC_splice": splice_ser.sort_index(), "BTC_spot": panel_df["SPOT_close"].pct_change(fill_method=None),
        "GLD": panel_df["GLD_close"].pct_change(), "BIL": panel_df["BIL_close"].pct_change(),
    }).loc["2017-06-02":]
    return BtcGoldInputs(return_df=return_df.fillna(0.0))


def _decision_rows(date_index: pd.DatetimeIndex, offset_int: int) -> np.ndarray:
    week_vec = date_index.to_period("W-SUN")
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    first_ser = position_ser.groupby(week_vec).min() + offset_int
    last_ser = position_ser.groupby(week_vec).max()
    return first_ser[first_ser <= last_ser].to_numpy()


def daily_weight_mat(return_mat: np.ndarray, date_index: pd.DatetimeIndex, config: BtcGoldConfig) -> np.ndarray:
    """(dates x 3) weights of BTC, GLD, BIL held on each date (decided at T, held from the close of T+1)."""
    risk_mat = return_mat[:, :2]
    log_price_mat = np.cumsum(np.log1p(risk_mat), axis=0)
    allocation_mat = np.zeros_like(risk_mat)
    for lookback_int in config.lookback_tuple:
        change_mat = np.full_like(risk_mat, np.nan)
        change_mat[lookback_int:] = log_price_mat[lookback_int:] - log_price_mat[:-lookback_int]
        up_mat = (np.nan_to_num(change_mat, nan=-1.0) > 0.0).astype(float)
        count_vec = up_mat.sum(axis=1, keepdims=True)
        allocation_mat += np.divide(up_mat, count_vec, out=np.zeros_like(up_mat), where=count_vec > 0)  # 50/50, 100/0 or cash
    allocation_mat /= len(config.lookback_tuple)
    vol21_mat = pd.DataFrame(risk_mat).rolling(21).std().to_numpy() * np.sqrt(252.0)
    vol63_mat = pd.DataFrame(risk_mat).rolling(63).std().to_numpy() * np.sqrt(252.0)
    sigma_mat = np.fmax(vol21_mat, vol63_mat) if config.vol_mode_str == "max21_63" else vol63_mat
    cap_mat = np.where(np.isfinite(sigma_mat) & (sigma_mat > 0), np.minimum(1.0, config.cap_float / sigma_mat), 0.0)

    base_mat = np.zeros_like(risk_mat)
    decision_vec = _decision_rows(date_index, config.decision_offset_int)
    for idx_int, row_int in enumerate(decision_vec):
        end_int = decision_vec[idx_int + 1] + 1 if idx_int + 1 < decision_vec.size else len(date_index)
        base_mat[row_int:end_int] = allocation_mat[row_int] if config.signal_bool else 0.5
    decided_mat = base_mat * cap_mat  # at the close of T: this week's base x today's cap
    decided_mat[: max(config.lookback_tuple + (63,))] = 0.0  # warm-up: hold cash
    weight_mat = np.zeros((len(date_index), 3))
    weight_mat[2:, :2] = decided_mat[:-2]  # *** CRITICAL*** decided at close T, earns from T+2 (held from close T+1)
    weight_mat[:, 2] = 1.0 - weight_mat[:, :2].sum(axis=1)
    return weight_mat


def simulate_config(inputs: BtcGoldInputs, config: BtcGoldConfig = LIVE_CONFIG, cost_model: CostModel = DEFAULT_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    frame = inputs.return_df.loc[START_STR:]
    return_mat = frame[[f"BTC_{config.btc_leg_str}", "GLD", "BIL"]].to_numpy()
    weight_mat = daily_weight_mat(return_mat, frame.index, config)
    gross_vec = (weight_mat * return_mat).sum(axis=1)
    drifted_mat = weight_mat * (1.0 + return_mat) / (1.0 + gross_vec)[:, None]
    turnover_vec = np.zeros(len(frame))
    turnover_vec[1:] = np.abs(weight_mat[1:] - drifted_mat[:-1]).sum(axis=1)
    net_vec = gross_vec - turnover_vec * cost_model.slippage_float
    value_ser = pd.Series(capital_float * np.cumprod(1.0 + net_vec), index=frame.index, name="total_value")
    trade_row_list = []
    for t_int in np.flatnonzero(turnover_vec > 1e-6):
        for a_int, asset_str in enumerate(ASSET_TUPLE):
            delta_float = weight_mat[t_int, a_int] - (drifted_mat[t_int - 1, a_int] if t_int else 0.0)
            if abs(delta_float) > 1e-9:
                notional_float = delta_float * float(value_ser.iloc[t_int - 1] if t_int else capital_float)
                trade_row_list.append((frame.index[t_int], asset_str, notional_float, 1.0, 0.0, "rebalance"))
    return WeightsResult(
        total_value_ser=value_ser, daily_return_ser=pd.Series(net_vec, index=frame.index),
        position_after_rebalance_df=pd.DataFrame(), trade_df=pd.DataFrame(trade_row_list, columns=["date", "asset", "delta_float", "price_float", "fee_float", "kind_str"]),
        daily_position_df=pd.DataFrame(weight_mat, index=frame.index, columns=list(ASSET_TUPLE)),
    )


# ---------------------------------------------------------------- MCPT contract
def mcpt_matrix(inputs: BtcGoldInputs, end_date_str: str = "2022-12-30", btc_leg_str: str = "splice") -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = inputs.return_df.loc[START_STR:end_date_str, [f"BTC_{btc_leg_str}", "GLD", "BIL"]]
    return frame.index, frame.to_numpy()


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list, base_config: BtcGoldConfig = LIVE_CONFIG) -> list[np.ndarray]:
    """Gross daily returns of each configuration; the BIL column moves with its dates under a row shuffle."""
    return [(daily_weight_mat(matrix, date_index, replace(base_config, **c)) * matrix).sum(axis=1) for c in config_list]
