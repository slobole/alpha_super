"""Scout spec: a volatility-sized Bitcoin sleeve (IBIT + BIL), the forward shadow of the 5% BTC slot beside GR1.

Source: Pakal `pakal-research/reports/btc_sleeve_candidates_study` (2026-11-06, forward_hypothesis) and the
2026-11-07 recheck (`btc_sleeve_candidates/followup_gr1_recent.py`). 5% of GR1 in this sleeve lifted GR1's excess
Sharpe from 1.35 to 1.47 over 2011-08..2026-08 at the same drawdown, but added nothing since 2021 (+0.01) and slightly
less than nothing since IBIT listed (-0.02). The shadow asks the forward question only.

Rule ("SIMPLE"):
    decision at the close of the last XNYS session of each ISO week (T); fill at Open(T+1)
    target w* = min(1, 0.20 / sigma60), sigma60 = std of the last 60 IBIT close-to-close returns x sqrt(252)
    no-trade band: if |w - w*| <= 0.10 x w*, no trade; otherwise trade to the nearest band edge (the first trade goes to w*)
    BIL holds the rest (traded, so idle cash earns T-bill income).
Vehicle: IBIT (Norgate CAPITALSPECIAL Open / Close / Dividend) from its 2024-01-11 listing; signal on the same series.
Engine: Scout weights engine, adjusted units, fractional shares, CostModel(slippage 5 bp, $0.005/share, min $1).
*** CRITICAL*** the decision for a fill at t reads closes up to T = t-1 and the ledger as it stood at the close of T.

The book (95% GR1 + 5% sleeve) is not simulated here: the fund-products Lab that builds GR1 ends 2026-08-19. At a
review the sleeve's forward returns are blended with GR1 rebuilt on the newest data (monthly rebalance, 5%).
Research only: nothing is wired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, simulate

REGISTRATION_ID_STR = "btc_vol_sleeve_gr1_slot_20261107"
ENGINE_COST_MODEL = CostModel(slippage_float=0.0005)


@dataclass(frozen=True)
class BtcVolSleeveConfig:
    symbol_str: str = "IBIT"
    reserve_symbol_str: str = "BIL"
    vol_target_float: float = 0.20
    vol_window_int: int = 60
    band_float: float = 0.10
    cap_float: float = 1.0
    capital_float: float = 1_000_000.0


LIVE_CONFIG = BtcVolSleeveConfig()


def frozen_config_dict(config: BtcVolSleeveConfig = LIVE_CONFIG) -> dict:
    return {**asdict(config), "cost_model": asdict(ENGINE_COST_MODEL), "decision_str": "last XNYS session of each ISO week, fill next open"}


@dataclass(frozen=True)
class BtcVolSleeveInputs:
    open_df: pd.DataFrame
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame


def load_inputs(config: BtcVolSleeveConfig = LIVE_CONFIG) -> BtcVolSleeveInputs:
    from data.norgate_loader import load_price_timeseries

    frame_dict = {s: load_price_timeseries(s, adjustment_str="CAPITALSPECIAL", start_date_str="2024-01-01") for s in (config.symbol_str, config.reserve_symbol_str)}
    index = frame_dict[config.symbol_str].index
    index = pd.DatetimeIndex(index).tz_localize(None).normalize() if getattr(index, "tz", None) else pd.DatetimeIndex(index).normalize()
    out = {}
    for field_str in ("Open", "Close", "Dividend"):
        frame = pd.DataFrame({s: f[field_str].set_axis(pd.DatetimeIndex(f.index).normalize()) for s, f in frame_dict.items()}).reindex(index)
        out[field_str] = frame[[config.symbol_str, config.reserve_symbol_str]]
    return BtcVolSleeveInputs(out["Open"], out["Close"], out["Dividend"].fillna(0.0))


def target_ser(inputs: BtcVolSleeveInputs, config: BtcVolSleeveConfig = LIVE_CONFIG) -> pd.Series:
    """w* at each close T (NaN until 60 returns exist). *** CRITICAL*** rolling window ends at T."""
    close = inputs.close_df[config.symbol_str]
    sigma = close.pct_change(fill_method=None).rolling(config.vol_window_int, min_periods=config.vol_window_int).std() * np.sqrt(252.0)
    return (config.vol_target_float / sigma).clip(upper=config.cap_float)


def week_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    iso = index.isocalendar()
    key = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    return np.r_[key[1:] != key[:-1], False]  # the last row of the data is not a known week end


def decision_fn_factory(inputs: BtcVolSleeveInputs, config: BtcVolSleeveConfig = LIVE_CONFIG):
    target_vec = target_ser(inputs, config).to_numpy()
    close_mat = inputs.close_df.to_numpy(dtype=float)
    week_end_vec = week_end_mask(inputs.close_df.index)
    band = config.band_float
    state = {"started": False}

    def decide(t_idx: int, position_vec: np.ndarray, total_T: float):
        T = t_idx - 1
        if T < 0 or not week_end_vec[T] or not np.isfinite(target_vec[T]) or not np.isfinite(close_mat[T]).all():
            return None
        tgt = float(target_vec[T])
        if not state["started"]:
            state["started"] = True
            return np.array([tgt, 1.0 - tgt])
        w_now = float(position_vec[0] * close_mat[T, 0] / total_T) if total_T > 0 else 0.0
        lo, hi = tgt * (1.0 - band), tgt * (1.0 + band)
        if lo <= w_now <= hi:
            return None
        new = min(max(w_now, lo), hi)
        return np.array([new, 1.0 - new])

    return decide


def simulate_config(inputs: BtcVolSleeveInputs, config: BtcVolSleeveConfig = LIVE_CONFIG):
    empty_df = pd.DataFrame(columns=list(inputs.close_df.columns), dtype=float)
    first_ts = target_ser(inputs, config).first_valid_index()
    return simulate(inputs.open_df, inputs.close_df, inputs.dividend_df, empty_df, start_date=first_ts, capital_float=config.capital_float,
                    share_unit_mode_str="adjusted", cost_model=ENGINE_COST_MODEL, fractional_shares_bool=True,
                    decision_fn=decision_fn_factory(inputs, config))


def shadow_daily_return_ser(shadow_row_dict: dict) -> pd.Series:
    """S8 provider: the frozen rule on the newest Norgate data. Refuses when the rule in code differs from the frozen row."""
    if shadow_row_dict["frozen_config_dict"] != frozen_config_dict():
        raise RuntimeError("The BTC vol sleeve rule in code differs from the frozen shadow row; a changed rule is a new registration.")
    daily_ser = simulate_config(load_inputs()).daily_return_ser.astype(float)
    today_ts = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)  # drop a partial bar of today
    return daily_ser.loc[daily_ser.index < today_ts]
