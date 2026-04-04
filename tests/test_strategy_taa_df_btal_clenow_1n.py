from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.taa_df.strategy_taa_df import DefenseFirstConfig
from strategies.taa_df.strategy_taa_df_btal_clenow_1n import (
    compute_clenow_score_from_log_price_window_vec,
    compute_month_end_weight_df_from_daily_clenow_score_df,
)
from strategies.taa_df.strategy_taa_df_btal_clenow import (
    compute_month_end_rank_weight_df_from_daily_clenow_score_df,
)


def test_compute_clenow_score_from_log_price_window_vec_positive_linear_path():
    lookback_day_int = 21
    time_index_vec = np.arange(lookback_day_int, dtype=float)
    time_centered_vec = time_index_vec - float(np.mean(time_index_vec))
    sxx_float = float(np.dot(time_centered_vec, time_centered_vec))
    slope_float = 0.002
    log_price_window_vec = 1.0 + slope_float * time_index_vec

    score_float = compute_clenow_score_from_log_price_window_vec(
        log_price_window_vec=log_price_window_vec,
        x_centered_vec=time_centered_vec,
        sxx_float=sxx_float,
    )

    expected_score_float = float(np.expm1(252.0 * slope_float))

    assert score_float > 0.0
    assert np.isclose(score_float, expected_score_float, rtol=1e-10, atol=1e-12)


def test_compute_clenow_score_from_log_price_window_vec_flat_path():
    lookback_day_int = 21
    time_index_vec = np.arange(lookback_day_int, dtype=float)
    time_centered_vec = time_index_vec - float(np.mean(time_index_vec))
    sxx_float = float(np.dot(time_centered_vec, time_centered_vec))
    log_price_window_vec = np.repeat(1.0, lookback_day_int)

    score_float = compute_clenow_score_from_log_price_window_vec(
        log_price_window_vec=log_price_window_vec,
        x_centered_vec=time_centered_vec,
        sxx_float=sxx_float,
    )

    assert score_float == 0.0


def test_compute_clenow_score_from_log_price_window_vec_penalizes_nonlinearity():
    lookback_day_int = 63
    time_index_vec = np.arange(lookback_day_int, dtype=float)
    time_centered_vec = time_index_vec - float(np.mean(time_index_vec))
    sxx_float = float(np.dot(time_centered_vec, time_centered_vec))
    clean_log_price_window_vec = 1.0 + 0.001 * time_index_vec
    noisy_log_price_window_vec = clean_log_price_window_vec + 0.08 * np.sin(time_index_vec)

    clean_score_float = compute_clenow_score_from_log_price_window_vec(
        log_price_window_vec=clean_log_price_window_vec,
        x_centered_vec=time_centered_vec,
        sxx_float=sxx_float,
    )
    noisy_score_float = compute_clenow_score_from_log_price_window_vec(
        log_price_window_vec=noisy_log_price_window_vec,
        x_centered_vec=time_centered_vec,
        sxx_float=sxx_float,
    )

    assert clean_score_float > noisy_score_float


def test_compute_month_end_weight_df_from_daily_clenow_score_df_uses_equal_slots():
    daily_index = pd.bdate_range("2020-01-01", "2020-02-28")
    daily_clenow_score_df = pd.DataFrame(
        {
            "GLD": -1.0,
            "UUP": -1.0,
            "TLT": -1.0,
            "DBC": -1.0,
            "BTAL": -1.0,
        },
        index=daily_index,
        dtype=float,
    )
    daily_clenow_score_df.loc[pd.Timestamp("2020-01-31"), ["GLD", "TLT", "DBC"]] = 0.5
    daily_clenow_score_df.loc[pd.Timestamp("2020-02-28"), :] = -0.5

    config = DefenseFirstConfig(
        defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
        fallback_asset="SPY",
        rank_weight_vec=(0.20, 0.20, 0.20, 0.20, 0.20),
        start_date_str="2011-09-13",
    )

    month_end_score_df, month_end_weight_df = compute_month_end_weight_df_from_daily_clenow_score_df(
        daily_clenow_score_df=daily_clenow_score_df,
        config=config,
    )

    january_weight_ser = month_end_weight_df.loc[pd.Timestamp("2020-01-31")]
    february_weight_ser = month_end_weight_df.loc[pd.Timestamp("2020-02-29")]

    assert np.allclose(month_end_weight_df.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-12)
    assert set(np.unique(month_end_weight_df[list(config.defensive_asset_list)].to_numpy())) == {0.0, 0.2}
    assert set(np.unique(month_end_weight_df[config.fallback_asset].to_numpy())) == {0.4, 1.0}
    assert np.isclose(float(january_weight_ser["GLD"]), 0.2)
    assert np.isclose(float(january_weight_ser["TLT"]), 0.2)
    assert np.isclose(float(january_weight_ser["DBC"]), 0.2)
    assert np.isclose(float(january_weight_ser["SPY"]), 0.4)
    assert np.isclose(float(february_weight_ser["SPY"]), 1.0)
    assert month_end_score_df.index.tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]


def test_compute_month_end_rank_weight_df_from_daily_clenow_score_df_uses_rank_slots():
    daily_index = pd.bdate_range("2020-01-01", "2020-02-28")
    daily_clenow_score_df = pd.DataFrame(
        {
            "GLD": -1.0,
            "UUP": -1.0,
            "TLT": -1.0,
            "DBC": -1.0,
            "BTAL": -1.0,
        },
        index=daily_index,
        dtype=float,
    )
    daily_clenow_score_df.loc[pd.Timestamp("2020-01-31"), :] = {
        "GLD": 0.50,
        "UUP": 0.40,
        "TLT": 0.30,
        "DBC": -0.20,
        "BTAL": -0.10,
    }
    daily_clenow_score_df.loc[pd.Timestamp("2020-02-28"), :] = -0.50

    config = DefenseFirstConfig(
        defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
        fallback_asset="SPY",
        rank_weight_vec=(5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0),
        start_date_str="2011-09-13",
    )

    month_end_score_df, month_end_weight_df = compute_month_end_rank_weight_df_from_daily_clenow_score_df(
        daily_clenow_score_df=daily_clenow_score_df,
        config=config,
    )

    january_weight_ser = month_end_weight_df.loc[pd.Timestamp("2020-01-31")]
    february_weight_ser = month_end_weight_df.loc[pd.Timestamp("2020-02-29")]

    assert np.allclose(month_end_weight_df.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-12)
    assert np.isclose(float(january_weight_ser["GLD"]), 5.0 / 15.0)
    assert np.isclose(float(january_weight_ser["UUP"]), 4.0 / 15.0)
    assert np.isclose(float(january_weight_ser["TLT"]), 3.0 / 15.0)
    assert np.isclose(float(january_weight_ser["SPY"]), 3.0 / 15.0)
    assert np.isclose(float(february_weight_ser["SPY"]), 1.0)
    assert month_end_score_df.index.tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
