from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.taa_df.strategy_taa_df import DefenseFirstConfig
from strategies.taa_df.strategy_taa_df_btal_fallback_upro_vix_cash import (
    apply_vrp_cash_gate_to_month_end_weight_df,
    compute_daily_vrp_signal_df,
)


def test_compute_daily_vrp_signal_df_near_constant_path_has_near_zero_realized_vol():
    daily_index = pd.bdate_range("2020-01-01", periods=40)
    spy_close_ser = pd.Series(np.exp(0.001 * np.arange(len(daily_index))), index=daily_index, name="SPY")
    vix_close_ser = pd.Series(20.0, index=daily_index, name="$VIX")

    daily_vrp_signal_df = compute_daily_vrp_signal_df(
        spy_close_ser=spy_close_ser,
        vix_close_ser=vix_close_ser,
    )

    assert float(daily_vrp_signal_df["rv20_ann_pct"].iloc[-1]) < 1e-9


def test_compute_daily_vrp_signal_df_choppy_path_has_higher_realized_vol():
    daily_index = pd.bdate_range("2020-01-01", periods=40)
    calm_ret_vec = np.repeat(0.001, len(daily_index))
    choppy_ret_vec = np.resize(np.array([0.01, -0.008, 0.012, -0.009], dtype=float), len(daily_index))
    calm_spy_close_ser = pd.Series(np.cumprod(1.0 + calm_ret_vec), index=daily_index, name="SPY")
    choppy_spy_close_ser = pd.Series(np.cumprod(1.0 + choppy_ret_vec), index=daily_index, name="SPY")
    vix_close_ser = pd.Series(20.0, index=daily_index, name="$VIX")

    calm_daily_vrp_signal_df = compute_daily_vrp_signal_df(
        spy_close_ser=calm_spy_close_ser,
        vix_close_ser=vix_close_ser,
    )
    choppy_daily_vrp_signal_df = compute_daily_vrp_signal_df(
        spy_close_ser=choppy_spy_close_ser,
        vix_close_ser=vix_close_ser,
    )

    assert float(choppy_daily_vrp_signal_df["rv20_ann_pct"].iloc[-1]) > float(calm_daily_vrp_signal_df["rv20_ann_pct"].iloc[-1])


def test_apply_vrp_cash_gate_to_month_end_weight_df_only_modifies_upro_sleeve():
    month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    base_month_end_weight_df = pd.DataFrame(
        {
            "GLD": [0.2, 0.2],
            "UUP": [0.0, 0.0],
            "TLT": [0.2, 0.2],
            "DBC": [0.0, 0.0],
            "BTAL": [0.0, 0.0],
            "UPRO": [0.6, 0.6],
        },
        index=month_end_index,
        dtype=float,
    )
    month_end_vrp_signal_df = pd.DataFrame(
        {
            "rv20_ann_pct": [12.0, 22.0],
            "vix_close": [18.0, 18.0],
            "vrp_gate": [1.0, 0.0],
        },
        index=month_end_index,
        dtype=float,
    )

    config = DefenseFirstConfig(
        defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
        fallback_asset="UPRO",
        rank_weight_vec=(5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0),
        start_date_str="2011-09-13",
    )

    month_end_weight_df, month_end_vrp_diagnostic_df = apply_vrp_cash_gate_to_month_end_weight_df(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_vrp_signal_df=month_end_vrp_signal_df,
        config=config,
    )

    january_weight_ser = month_end_weight_df.loc[pd.Timestamp("2020-01-31")]
    february_weight_ser = month_end_weight_df.loc[pd.Timestamp("2020-02-29")]

    assert np.isclose(float(january_weight_ser["UPRO"]), 0.6)
    assert np.isclose(float(february_weight_ser["UPRO"]), 0.0)
    assert np.isclose(float(february_weight_ser["GLD"]), 0.2)
    assert np.isclose(float(february_weight_ser["TLT"]), 0.2)
    assert np.isclose(float(month_end_vrp_diagnostic_df.loc[pd.Timestamp("2020-01-31"), "cash_weight"]), 0.0)
    assert np.isclose(float(month_end_vrp_diagnostic_df.loc[pd.Timestamp("2020-02-29"), "cash_weight"]), 0.6)
    assert np.isclose(float(month_end_vrp_diagnostic_df.loc[pd.Timestamp("2020-01-31"), "tradeable_weight_sum"]), 1.0)
    assert np.isclose(float(month_end_vrp_diagnostic_df.loc[pd.Timestamp("2020-02-29"), "tradeable_weight_sum"]), 0.4)


def test_apply_vrp_cash_gate_to_month_end_weight_df_tie_goes_to_cash():
    month_end_index = pd.to_datetime(["2020-01-31"])
    base_month_end_weight_df = pd.DataFrame(
        {
            "GLD": [0.2],
            "UUP": [0.0],
            "TLT": [0.2],
            "DBC": [0.0],
            "BTAL": [0.0],
            "UPRO": [0.6],
        },
        index=month_end_index,
        dtype=float,
    )
    month_end_vrp_signal_df = pd.DataFrame(
        {
            "rv20_ann_pct": [18.0],
            "vix_close": [18.0],
            "vrp_gate": [0.0],
        },
        index=month_end_index,
        dtype=float,
    )

    config = DefenseFirstConfig(
        defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
        fallback_asset="UPRO",
        rank_weight_vec=(5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0),
        start_date_str="2011-09-13",
    )

    month_end_weight_df, month_end_vrp_diagnostic_df = apply_vrp_cash_gate_to_month_end_weight_df(
        base_month_end_weight_df=base_month_end_weight_df,
        month_end_vrp_signal_df=month_end_vrp_signal_df,
        config=config,
    )

    assert np.isclose(float(month_end_weight_df.loc[pd.Timestamp("2020-01-31"), "UPRO"]), 0.0)
    assert np.isclose(float(month_end_vrp_diagnostic_df.loc[pd.Timestamp("2020-01-31"), "cash_weight"]), 0.6)
