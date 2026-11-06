"""Scout spec of CORE6: the CORE5 adaptive macro rule plus a volatility-sized Bitcoin overlay funded from BIL.

Source: Pakal study `pakal-research/reports/core6_qualification_study` (2026-11-06, status research_candidate), ledger
registration `core6_btc_sleeve_qualification_20261106` (row 597, family time_series_trend_and_breakout, retro). That
study compared three ways to add BTC to CORE5 in this engine and chose D3 (the default here):

    D3 "overlay_half" (LIVE_CONFIG)  CORE5 sleeves unchanged (0.20 x long_i for SPY IEF GLD DBC UUP, the DBC short as CORE5);
                                     BTC weight = (1/6) x min(1, 0.10 / sigma60_BTC) x long_BTC, taken from BIL
    D2 "overlay"                     the same with 0.20 (rejected: DEF -10% breach probability 14.8% -> 19.4%)
    D1 "dilute"                      six sleeves at 1/6, BTC on-weight x min(1, 0.20 / sigma60) (rejected: sleeping-BTC tail)
In the overlay modes, when BIL is short of the BTC weight (all five sleeves long), the five are scaled down pro rata:
the book never levers.

Signals (CORE5 LIVE_CONFIG, unchanged, alpha/scout/specs/core5.py): per asset, SMA10 > adaptive MA (drawdown-percentile
speed between 50 and 200 sessions). BTC's signal close is the CoinMetrics PriceUSD stamp of session date D (end of the
UTC day, after the 16:00 ET equity close); sigma60 = std of the last 60 of those returns x sqrt(252).
Rebalance: any long-state change (BTC included) or the last XNYS session of the month; fill at Open(T+1) (core5 rule).

Execution: the Scout weights engine (parity mode) with adjusted fractional units, the default CostModel, the DBC short
with split sign flips and the 1% borrow model. BTC pays an extra 12.5 bp on traded notional at the fill price (15 bp a
side in total), applied to the daily returns after the run.

Data (local, read-only, no download): Pakal's cached BTC unit
`research_cache/core6_qualification_study/btc_unit.parquet` (built by core6_qualification/c6q_data.py):
    signal  CoinMetrics PriceUSD on the XNYS session grid;
    unit    Binance BTCUSDT 5m bars ending 09:30 ET (Open) and 16:00 ET (Close) from 2017-08-17; before that
            Open = Close = the CoinMetrics stamp of the session (a later, conservative fill);
    vehicle "spot" (default) or "ibit": IBIT CAPITALSPECIAL Open/Close from 2024-01-11, level-spliced.
*** CRITICAL*** weights for execution date t use closes <= t-1 only; the decision is taken after 00:00 UTC following the
close of t-1 (when the BTC daily stamp exists) and fills at Open(t). Tests: tests/test_scout_core6.py.

Research only: core5.py, the CORE5 strategy and every live path are untouched; this module is not wired to anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import BorrowModel, CostModel, simulate
from alpha.scout.specs import core5

REGISTRATION_ID_STR = "core6_btc_sleeve_qualification_20261106"
UNIT_PATH = Path(r"C:\Users\User\Documents\workspace\pakal\pakal-research\research_cache\core6_qualification_study\btc_unit.parquet")
CRYPTO_STR = "BTC"
FIVE_TUPLE = core5.RISK_ASSET_TUPLE
TRADED_TUPLE = core5.TRADED_TUPLE + (CRYPTO_STR,)
IBIT_LISTING_TS = pd.Timestamp("2024-01-11")
MODE_TUPLE = ("overlay_half", "overlay", "dilute", "core5")


@dataclass(frozen=True)
class Core6Config:
    mode_str: str = "overlay_half"
    crypto_sleeve_fraction_float: float = 1.0 / 6.0
    crypto_vol_window_int: int = 60
    crypto_extra_slippage_float: float = 0.0015 - 0.00025
    vehicle_str: str = "spot"
    backtest_start_date_str: str = "2011-08-01"
    capital_float: float = 1_000_000.0

    @property
    def crypto_vol_target_float(self) -> float:
        return {"overlay_half": 0.10, "overlay": 0.20, "dilute": 0.20, "core5": 0.0}[self.mode_str]

    def __post_init__(self):
        if self.mode_str not in MODE_TUPLE:
            raise ValueError(f"mode_str must be one of {MODE_TUPLE}.")
        if self.vehicle_str not in ("spot", "ibit"):
            raise ValueError("vehicle_str must be 'spot' or 'ibit'.")


LIVE_CONFIG = Core6Config()


@dataclass(frozen=True)
class Core6Inputs:
    core5_inputs: core5.Core5Inputs
    crypto_signal_ser: pd.Series   # CoinMetrics end-of-day close on the core5 calendar
    open_df: pd.DataFrame          # core5 traded ETFs + BTC unit
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame


def load_inputs(vehicle_str: str = "spot", unit_path: Path = UNIT_PATH) -> Core6Inputs:
    base = core5.load_inputs()
    unit_df = pd.read_parquet(unit_path)
    date_index = base.open_df.index
    signal_ser = unit_df["BTC_sig"].reindex(date_index)
    open_ser = unit_df["BTC_open"].reindex(date_index)
    close_ser = unit_df["BTC_close"].reindex(date_index)
    if vehicle_str == "ibit":
        from data.norgate_loader import load_price_timeseries

        ibit_df = load_price_timeseries("IBIT", adjustment_str="CAPITALSPECIAL", start_date_str="2024-01-01")
        ibit_df.index = pd.DatetimeIndex(ibit_df.index).tz_localize(None).normalize()
        level_float = float(close_ser.loc[:IBIT_LISTING_TS].dropna().iloc[-1] / ibit_df["Close"].loc[IBIT_LISTING_TS])
        open_ser = open_ser.where(date_index < IBIT_LISTING_TS, ibit_df["Open"].reindex(date_index) * level_float)
        close_ser = close_ser.where(date_index < IBIT_LISTING_TS, ibit_df["Close"].reindex(date_index) * level_float)
    column_list = list(TRADED_TUPLE)
    return Core6Inputs(
        core5_inputs=base, crypto_signal_ser=signal_ser,
        open_df=base.open_df.assign(**{CRYPTO_STR: open_ser})[column_list],
        close_df=base.close_df.assign(**{CRYPTO_STR: close_ser})[column_list],
        dividend_df=base.dividend_df.assign(**{CRYPTO_STR: 0.0})[column_list],
    )


def rebalance_weight_df(inputs: Core6Inputs, config: Core6Config = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights indexed by execution date (the session after each decision)."""
    c5_inputs, c5_config = inputs.core5_inputs, core5.LIVE_CONFIG
    column_list = list(TRADED_TUPLE)
    if config.mode_str == "core5":
        return core5.rebalance_weight_df(c5_inputs, c5_config).reindex(columns=column_list).fillna(0.0)
    date_index = c5_inputs.open_df.index
    signal_dict = {s: core5.asset_signal_df(c5_inputs.signal_close_df[s], c5_config) for s in FIVE_TUPLE}
    crypto_signal_df = core5.asset_signal_df(inputs.crypto_signal_ser, c5_config)
    # *** CRITICAL*** volatility of the last 60 observed crypto returns ending at the decision close
    crypto_vol_ser = (inputs.crypto_signal_ser.dropna().pct_change().rolling(config.crypto_vol_window_int, min_periods=config.crypto_vol_window_int)
                      .std().mul(np.sqrt(252.0)).reindex(date_index))
    long_mat = np.column_stack([signal_dict[s]["long"].to_numpy() for s in FIVE_TUPLE] + [crypto_signal_df["long"].to_numpy()])
    dbc_short_vec, dbc_vol_vec = signal_dict["DBC"]["short"].to_numpy(), signal_dict["DBC"]["volatility"].to_numpy()
    changed_vec = np.zeros(len(date_index), dtype=bool)
    changed_vec[1:] = np.nan_to_num(np.abs(np.diff(long_mat, axis=0)), nan=0.0).max(axis=1) > 0.0
    month_end_vec = core5.month_end_flag_vec(date_index, 0)
    valid_vec = np.isfinite(long_mat).all(axis=1) & np.isfinite(dbc_vol_vec) & np.isfinite(crypto_vol_ser.to_numpy())
    open_mat = c5_inputs.open_df[list(core5.TRADED_TUPLE)].to_numpy(dtype=float)
    valid_open_vec = (np.isfinite(open_mat) & (open_mat > 0.0)).all(axis=1)
    candidate_vec = np.flatnonzero(valid_vec[:-1] & valid_open_vec[1:]) + 1
    if candidate_vec.size == 0:
        raise RuntimeError("No actionable CORE6 execution date.")
    start_int = int(date_index.searchsorted(max(date_index[candidate_vec[0]], pd.Timestamp(config.backtest_start_date_str))))
    row_dict = {}
    for decision_int in range(start_int - 1, len(date_index) - 1):
        if not (decision_int == start_int - 1 or changed_vec[decision_int] or month_end_vec[decision_int]):
            continue
        long_vec = long_mat[decision_int]
        crypto_size_float = min(1.0, config.crypto_vol_target_float / crypto_vol_ser.iloc[decision_int]) * long_vec[5]
        crypto_weight_float = config.crypto_sleeve_fraction_float * crypto_size_float
        if config.mode_str == "dilute":
            five_vec = long_vec[:5] / 6.0
        else:
            five_vec = long_vec[:5] * c5_config.sleeve_weight_float
            if five_vec.sum() + crypto_weight_float > 1.0:  # no leverage: make room inside the five sleeves
                five_vec = five_vec * (1.0 - crypto_weight_float) / five_vec.sum()
        weight_vec = five_vec.copy()
        if dbc_short_vec[decision_int] == 1.0:
            weight_vec[FIVE_TUPLE.index("DBC")] = -min(c5_config.commodity_short_cap_float,
                                                       c5_config.commodity_short_vol_target_float / dbc_vol_vec[decision_int])
        reserve_float = 1.0 - five_vec.sum() - crypto_weight_float
        row_dict[date_index[decision_int + 1]] = list(weight_vec) + [reserve_float, crypto_weight_float]
    return pd.DataFrame.from_dict(row_dict, orient="index", columns=list(FIVE_TUPLE) + [core5.RESERVE_ASSET_STR, CRYPTO_STR])[column_list]


def simulate_config(inputs: Core6Inputs, config: Core6Config = LIVE_CONFIG) -> tuple[pd.Series, pd.Series, object]:
    """(daily return after the crypto extra slippage, BTC share of NAV at each close, raw WeightsResult)."""
    weight_df = rebalance_weight_df(inputs, config)
    weight_df = weight_df[weight_df.index >= config.backtest_start_date_str]
    result = simulate(inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=weight_df.index[0],
                      capital_float=config.capital_float, share_unit_mode_str="adjusted", cost_model=CostModel(),
                      allow_short_bool=True, split_sign_flip_bool=True, borrow_model=BorrowModel(annual_rate_float=0.01),
                      fractional_shares_bool=True)
    return_ser = result.daily_return_ser.copy()
    position_df = result.daily_position_df.reindex(columns=list(TRADED_TUPLE)).fillna(0.0)
    traded_units_ser = position_df[CRYPTO_STR].diff().abs().fillna(position_df[CRYPTO_STR].abs())
    extra_cost_ser = (traded_units_ser * inputs.open_df[CRYPTO_STR].reindex(traded_units_ser.index) * config.crypto_extra_slippage_float).reindex(return_ser.index).fillna(0.0)
    value_ser = result.total_value_ser.reindex(return_ser.index)
    return_ser = return_ser - extra_cost_ser / value_ser.shift(1).fillna(config.capital_float)
    crypto_share_ser = (position_df[CRYPTO_STR] * inputs.close_df[CRYPTO_STR].reindex(position_df.index) / result.total_value_ser.reindex(position_df.index)).reindex(return_ser.index)
    return return_ser, crypto_share_ser, result


def daily_return_ser(config: Core6Config = LIVE_CONFIG) -> pd.Series:
    return simulate_config(load_inputs(config.vehicle_str), config)[0]


def shadow_daily_return_ser(shadow_row_dict: dict) -> pd.Series:
    """S8 provider for a future forward shadow: the frozen configuration on the newest cached data. The BTC unit cache
    must be refreshed (core6_qualification/c6q_data.py) before each read; ETF data is not panel-sealed."""
    config = Core6Config(**shadow_row_dict.get("frozen_config_dict", {}))
    return daily_return_ser(config).loc[shadow_row_dict["start_date_str"]:]
