"""CORE5 with another equity vehicle in the SPY sleeve (core5_equity_vehicle_v2, ledger row 45): the S8 shadow source.

The 2026-10-11 study (pakal-research/reports/core5_equity_vehicle_v2) recommended QQQ with its own QQQ signal (variant
A2) as a forward line instead of the August QQQ-with-SPY-signal line. This module runs the CORE5 spec
(alpha/scout/specs/core5.py, unchanged) with the chosen symbol's data placed in the equity column: the column keeps the
name "SPY" inside the spec, but its signal closes, opens, closes and dividends are the chosen symbol's. Signal and
vehicle are the same symbol (A2); the other four sleeves, BIL and every parameter are CORE5's live configuration.

Research only: core5.py and the CORE5 strategy are not touched.
"""

from __future__ import annotations

import pandas as pd

from alpha.scout.specs import core5

REGISTRATION_ID_STR = "core5_equity_vehicle_v2_retro_20261011"
EQUITY_COLUMN_STR = "SPY"


def load_inputs(equity_symbol_str: str = "QQQ") -> core5.Core5Inputs:
    from data.norgate_loader import load_price_timeseries
    from data.norgate_snapshot_store import CORE5_PROFILE_STR, use_norgate_data_profile

    base = core5.load_inputs()
    if equity_symbol_str == EQUITY_COLUMN_STR:
        return base
    with use_norgate_data_profile(CORE5_PROFILE_STR):
        capital_frame = load_price_timeseries(equity_symbol_str, adjustment_str="CAPITALSPECIAL", start_date_str=core5.HISTORY_START_DATE_STR)
        total_return_ser = load_price_timeseries(equity_symbol_str, adjustment_str="TOTALRETURN", start_date_str=core5.HISTORY_START_DATE_STR)["Close"]
    date_index = base.open_df.index

    def swapped(frame: pd.DataFrame, value_ser: pd.Series) -> pd.DataFrame:
        frame = frame.copy()
        frame[EQUITY_COLUMN_STR] = value_ser.reindex(date_index)
        return frame

    return core5.Core5Inputs(
        signal_close_df=swapped(base.signal_close_df, total_return_ser),
        reserve_total_return_close_ser=base.reserve_total_return_close_ser,
        open_df=swapped(base.open_df, capital_frame["Open"]),
        close_df=swapped(base.close_df, capital_frame["Close"]),
        dividend_df=swapped(base.dividend_df, capital_frame["Dividend"]).fillna(0.0),
    )


def daily_return_ser(equity_symbol_str: str = "QQQ") -> pd.Series:
    return core5.simulate_config(load_inputs(equity_symbol_str), core5.LIVE_CONFIG, capital_float=1_000_000.0).daily_return_ser


def shadow_daily_return_ser(shadow_row_dict: dict) -> pd.Series:
    """S8 provider: the frozen vehicle on the newest data (ETF data is not panel-sealed; the monitor keeps only the
    sessions from the shadow's start)."""
    return daily_return_ser(shadow_row_dict["frozen_config_dict"]["equity_symbol_str"]).loc[shadow_row_dict["start_date_str"]:]
