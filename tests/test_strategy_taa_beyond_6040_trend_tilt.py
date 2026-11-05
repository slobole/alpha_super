import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.bench import catalog
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module


MODULE_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_beyond_6040_trend_tilt"
ASSET_LIST = ["VTI", "GLD", "TLT"]


def make_pricing_data_df(num_days_int: int = 420, bil_inception_str: str | None = None) -> pd.DataFrame:
    """Synthetic VTI/GLD/TLT frame with a TotalReturnClose field per asset.

    VTI trends up (above its SMA), TLT trends down (below its SMA), GLD wobbles.
    With ``bil_inception_str`` a BIL series exists from that date on (NaN before it, as Norgate's unpadded history).
    """
    date_index = pd.date_range("2022-01-03", periods=num_days_int, freq="B")
    bar_idx_vec = np.arange(num_days_int, dtype=float)

    vti_return_vec = 0.0010 + 0.0080 * np.sin(bar_idx_vec / 4.0)
    gld_return_vec = 0.0002 + 0.0040 * np.sin(bar_idx_vec / 6.0 + 0.5)
    tlt_return_vec = -0.0008 + 0.0030 * np.cos(bar_idx_vec / 7.0)
    benchmark_return_vec = 0.0004 + 0.0070 * np.sin(bar_idx_vec / 5.0 + 0.25)

    close_map_dict = {
        "VTI": 100.0 * np.cumprod(1.0 + vti_return_vec),
        "GLD": 120.0 * np.cumprod(1.0 + gld_return_vec),
        "TLT": 110.0 * np.cumprod(1.0 + tlt_return_vec),
        "$SPX": 4000.0 * np.cumprod(1.0 + benchmark_return_vec),
    }
    if bil_inception_str is not None:
        bil_close_vec = 91.0 * np.cumprod(1.0 + np.full(num_days_int, 0.00015))
        close_map_dict["BIL"] = np.where(date_index < pd.Timestamp(bil_inception_str), np.nan, bil_close_vec)

    pricing_data_dict: dict[tuple[str, str], np.ndarray] = {}
    for symbol_str, close_vec in close_map_dict.items():
        open_vec = close_vec * 0.999
        pricing_data_dict[(symbol_str, "Open")] = open_vec
        pricing_data_dict[(symbol_str, "High")] = np.maximum(open_vec, close_vec) * 1.001
        pricing_data_dict[(symbol_str, "Low")] = np.minimum(open_vec, close_vec) * 0.999
        pricing_data_dict[(symbol_str, "Close")] = close_vec
        if symbol_str in ASSET_LIST:
            # Total return drifts above the price close by a small accrued yield.
            pricing_data_dict[(symbol_str, "TotalReturnClose")] = close_vec * (1.0 + 0.0001 * bar_idx_vec)

    pricing_data_df = pd.DataFrame(pricing_data_dict, index=date_index, dtype=float)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def run_fixture_strategy(pricing_data_df: pd.DataFrame, **run_kwargs):
    with patch.object(tilt_module, "get_trend_tilt_data", return_value=pricing_data_df):
        return tilt_module.run_variant(show_display_bool=False, save_results_bool=False, **run_kwargs)


def test_default_config_is_t1():
    assert tilt_module.DEFAULT_CONFIG.target_portfolio_vol_float == 0.08
    assert tilt_module.DEFAULT_CONFIG.trigger_portfolio_vol_float == 0.085
    assert tilt_module.DEFAULT_CONFIG.trend_sma_lookback_int == 210
    assert tilt_module.DEFAULT_CONFIG.tilt_strength_float == 0.5
    assert tilt_module.DEFAULT_CONFIG.asset_list == ("VTI", "GLD", "TLT")
    assert tilt_module.DEFAULT_CONFIG.reserve_asset_str == "BIL"
    assert tilt_module.DEFAULT_CONFIG.tradable_list == ["VTI", "GLD", "TLT", "BIL"]
    assert tilt_module.DEFAULT_CONFIG.start_date_str == "2001-05-31"
    with pytest.raises(ValueError):
        replace(tilt_module.DEFAULT_CONFIG, reserve_asset_str="GLD")


BIL_INCEPTION_STR = "2023-03-01"


def test_bil_takes_the_residual_once_it_exists_and_cash_before():
    pricing_data_df = make_pricing_data_df(bil_inception_str=BIL_INCEPTION_STR)
    strategy_obj = run_fixture_strategy(pricing_data_df)
    daily_weight_df = strategy_obj.daily_target_weights
    assert list(daily_weight_df.columns) == ASSET_LIST + ["BIL", "Cash"]
    assert np.allclose(daily_weight_df.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-12)
    risky_sum_ser = daily_weight_df[ASSET_LIST].sum(axis=1)
    # The decision at Close_T applies to bar T+1: BIL's first observed close is the inception bar itself.
    first_bil_bar_ts = pricing_data_df.index[pricing_data_df.index.get_loc(pd.Timestamp(BIL_INCEPTION_STR)) + 1]
    before_df = daily_weight_df.loc[: pd.Timestamp(BIL_INCEPTION_STR)]
    after_df = daily_weight_df.loc[first_bil_bar_ts:]
    assert len(before_df) > 20 and len(after_df) > 20
    assert (before_df["BIL"] == 0.0).all()
    assert np.allclose(before_df["Cash"].to_numpy(dtype=float), (1.0 - risky_sum_ser.loc[before_df.index]).to_numpy(dtype=float), atol=1e-12)
    assert (after_df["Cash"] == 0.0).all()
    assert np.allclose(after_df["BIL"].to_numpy(dtype=float), (1.0 - risky_sum_ser.loc[after_df.index]).to_numpy(dtype=float), atol=1e-12)
    assert (after_df["BIL"] > 0.0).any()
    transaction_df = strategy_obj.get_transactions()
    bil_transaction_df = transaction_df.loc[transaction_df["asset"] == "BIL"]
    assert len(bil_transaction_df) > 0
    assert pd.to_datetime(bil_transaction_df["bar"]).min() >= first_bil_bar_ts
    assert (strategy_obj.results["total_value"] > 0).all()


def test_bil_is_never_in_the_vol_weights_or_the_tilt():
    with_bil_df = make_pricing_data_df(bil_inception_str=BIL_INCEPTION_STR)
    without_bil_df = make_pricing_data_df()
    with_bil_obj = run_fixture_strategy(with_bil_df)
    signal_df = with_bil_obj.compute_signals(with_bil_df)
    assert not any(col[0] == "BIL" and col[1] in ("base_weight_ser", "tilt_multiplier_ser", "volatility_ser", "trend_sma_ser") for col in signal_df.columns)
    # The research convention (no reserve) on the same risk assets: identical risky weights and exposure every day.
    with patch.object(tilt_module, "DEFAULT_CONFIG", replace(tilt_module.DEFAULT_CONFIG, reserve_asset_str=None)):
        no_reserve_obj = run_fixture_strategy(without_bil_df)
    assert list(no_reserve_obj.daily_target_weights.columns) == ASSET_LIST + ["Cash"]
    pd.testing.assert_frame_equal(with_bil_obj.daily_target_weights[ASSET_LIST], no_reserve_obj.daily_target_weights[ASSET_LIST])
    pd.testing.assert_series_equal(with_bil_obj.daily_exposure_ser, no_reserve_obj.daily_exposure_ser)
    assert (no_reserve_obj.daily_target_weights["Cash"] > 0.0).any()
    assert "BIL" not in set(no_reserve_obj.get_transactions()["asset"])


def test_tilt_multiplier_is_zero_below_sma_and_one_above():
    date_index = pd.date_range("2020-01-01", periods=300, freq="B")
    up_vec = np.linspace(100.0, 200.0, 300)          # always above its own SMA
    down_vec = np.linspace(200.0, 100.0, 300)        # always below its own SMA
    trend_price_df = pd.DataFrame({"UP": up_vec, "DOWN": down_vec}, index=date_index)

    trend_sma_df, trend_state_df, multiplier_df = tilt_module.compute_month_end_tilt_multiplier_df(
        trend_price_df=trend_price_df,
        trend_sma_lookback_int=210,
        tilt_strength_float=0.5,
    )

    assert trend_sma_df.iloc[:209].isna().all().all()
    assert trend_state_df.iloc[:209].isna().all().all()
    valid_state_df = trend_state_df.dropna()
    assert (valid_state_df["UP"] == 1.0).all()
    assert (valid_state_df["DOWN"] == -1.0).all()
    assert len(multiplier_df) > 0
    assert (multiplier_df["UP"] == 1.0).all()
    assert (multiplier_df["DOWN"] == 0.0).all()
    # Month-ends before the SMA exists are dropped, not filled.
    assert multiplier_df.index.min() >= date_index[209]


def test_final_weights_equal_exposure_times_base_times_multiplier():
    pricing_data_df = make_pricing_data_df()
    strategy_obj = run_fixture_strategy(pricing_data_df)
    signal_df = strategy_obj.compute_signals(pricing_data_df)

    daily_weight_df = strategy_obj.daily_target_weights
    exposure_ser = strategy_obj.daily_exposure_ser
    assert len(daily_weight_df) > 20

    invested_count_int = 0
    for bar_ts in daily_weight_df.index:
        bar_pos_int = pricing_data_df.index.get_loc(bar_ts)
        previous_bar_ts = pricing_data_df.index[bar_pos_int - 1]
        exposure_float = float(exposure_ser.loc[bar_ts])
        if np.isnan(exposure_float):
            assert np.isclose(float(daily_weight_df.loc[bar_ts, "Cash"]), 1.0)
            continue
        invested_count_int += 1
        for asset_str in ASSET_LIST:
            base_float = float(signal_df.loc[previous_bar_ts, (asset_str, "base_weight_ser")])
            multiplier_float = float(signal_df.loc[previous_bar_ts, (asset_str, "tilt_multiplier_ser")])
            assert multiplier_float in (0.0, 1.0)
            expected_float = exposure_float * base_float * multiplier_float
            assert np.isclose(float(daily_weight_df.loc[bar_ts, asset_str]), expected_float, atol=1e-12)
    assert invested_count_int > 0

    # The downtrending asset is tilted out, the uptrending one stays in.
    assert (daily_weight_df["TLT"].dropna() == 0.0).all()
    assert (daily_weight_df["VTI"].dropna() > 0.0).any()


def test_total_weights_never_exceed_one_and_no_shorts():
    pricing_data_df = make_pricing_data_df()
    strategy_obj = run_fixture_strategy(pricing_data_df)

    daily_weight_df = strategy_obj.daily_target_weights
    risky_weight_df = daily_weight_df[ASSET_LIST]
    assert (risky_weight_df >= -1e-12).all().all()
    assert (risky_weight_df.sum(axis=1) <= 1.0 + 1e-12).all()
    assert (daily_weight_df["Cash"] >= -1e-12).all()
    assert np.allclose(daily_weight_df.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-12)
    assert (strategy_obj.daily_exposure_ser.dropna() <= 1.0 + 1e-12).all()


def test_monthly_tilt_does_not_change_between_month_ends():
    pricing_data_df = make_pricing_data_df()
    strategy_obj = run_fixture_strategy(pricing_data_df)
    signal_df = strategy_obj.compute_signals(pricing_data_df)

    index = signal_df.index
    next_month_differs_vec = np.r_[index[1:].month != index[:-1].month, True]
    for asset_str in ASSET_LIST:
        multiplier_ser = signal_df[(asset_str, "tilt_multiplier_ser")].dropna()
        changed_mask_ser = multiplier_ser.ne(multiplier_ser.shift(1)) & multiplier_ser.shift(1).notna()
        changed_bar_list = multiplier_ser.index[changed_mask_ser.to_numpy()]
        for bar_ts in changed_bar_list:
            # A change may sit only on the last bar of a month (read before the first open of the next).
            assert next_month_differs_vec[index.get_loc(bar_ts)]

    # Held weights: the on/off state per asset is constant within each calendar month.
    daily_weight_df = strategy_obj.daily_target_weights.dropna()
    held_state_df = (daily_weight_df[ASSET_LIST] > 0.0)
    invested_mask_ser = daily_weight_df["Cash"] < 1.0
    monthly_unique_df = held_state_df[invested_mask_ser].groupby(held_state_df[invested_mask_ser].index.to_period("M")).nunique()
    assert (monthly_unique_df <= 1).all().all()


def test_run_variant_honors_bench_run_contract():
    captured_config_list = []
    pricing_data_df = make_pricing_data_df()

    def loader_fn(config):
        captured_config_list.append(config)
        return pricing_data_df

    with patch.object(tilt_module, "get_trend_tilt_data", side_effect=loader_fn):
        strategy_obj = tilt_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2023-02-01",
            capital_base_float=12_345.0,
            end_date_str="2023-08-01",
        )

    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert captured_config_list[0].end_date_str == "2023-08-01"
    assert captured_config_list[0].trend_sma_lookback_int == 210
    assert strategy_obj.name == tilt_module.STRATEGY_NAME_STR
    assert strategy_obj._capital_base == 12_345.0
    assert strategy_obj.trend_sma_lookback_int == 210
    assert strategy_obj.tilt_strength_float == 0.5
    assert len(strategy_obj.results) > 0
    assert strategy_obj.results.index.min() >= pd.Timestamp("2023-02-01")
