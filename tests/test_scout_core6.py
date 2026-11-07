"""CORE6 spec (alpha/scout/specs/core6.py): parity with the Pakal qualification run, the live IBIT-close signal, timing guards.

Skipped when the Pakal BTC unit cache or the qualification engine output is not on this machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.scout.specs import core5, core6

ENGINE_PATH = Path(r"C:\Users\User\Documents\workspace\pakal\pakal-research\research_cache\core6_qualification_study\engine_btc.parquet")
pytestmark = [pytest.mark.data_integration,
              pytest.mark.skipif(not (core6.UNIT_PATH.exists() and ENGINE_PATH.exists()), reason="Pakal CORE6 caches not present")]

STUDY_COLUMN = {"core5": "D0_CORE5", "dilute": "D1_DIL", "overlay": "D2_OVL", "overlay_half": "D3_OVL_HALF"}


@pytest.fixture(scope="module")
def inputs():
    return core6.load_inputs(core6.STUDY_CONFIG.vehicle_str, core6.STUDY_CONFIG.signal_str)


@pytest.fixture(scope="module")
def live_inputs():
    return core6.load_inputs()


@pytest.mark.parametrize("mode_str", ["overlay_half", "overlay", "dilute", "core5"])
def test_returns_match_the_qualification_study(inputs, mode_str):
    reference = pd.read_parquet(ENGINE_PATH)[STUDY_COLUMN[mode_str]].dropna()
    mine = core6.simulate_config(inputs, core6.Core6Config(mode_str=mode_str, vehicle_str="spot", signal_str="coinmetrics"))[0]
    common = mine.index.intersection(reference.index)
    assert len(common) > 3_500
    assert np.allclose(mine.reindex(common).to_numpy(), reference.reindex(common).to_numpy(), atol=1e-12)


def test_core5_mode_is_the_house_core5_weights(inputs):
    mine = core6.rebalance_weight_df(inputs, core6.Core6Config(mode_str="core5"))
    house = core5.rebalance_weight_df(inputs.core5_inputs, core5.LIVE_CONFIG)
    assert np.allclose(mine[list(core5.TRADED_TUPLE)].to_numpy(), house[list(core5.TRADED_TUPLE)].to_numpy())


def test_weights_ignore_future_btc_prices(inputs):
    cut = pd.Timestamp("2020-03-02")
    base = core6.rebalance_weight_df(inputs)
    shocked_signal = inputs.crypto_signal_ser.copy()
    shocked_signal[shocked_signal.index > cut] *= 3.0
    shocked = core6.Core6Inputs(inputs.core5_inputs, shocked_signal, inputs.open_df, inputs.close_df, inputs.dividend_df)
    alt = core6.rebalance_weight_df(shocked)
    keep = base.index[base.index <= cut]
    assert np.allclose(base.loc[keep].to_numpy(), alt.loc[keep].to_numpy())


def test_overlay_never_levers_and_btc_stays_small(inputs):
    weights = core6.rebalance_weight_df(inputs)
    long_sum = weights.clip(lower=0.0).drop(columns=[core5.RESERVE_ASSET_STR]).sum(axis=1)
    assert (long_sum <= 1.0 + 1e-12).all()
    assert (weights[core5.RESERVE_ASSET_STR] >= -1e-12).all()
    assert weights[core6.CRYPTO_STR].max() <= 1.0 / 6.0 + 1e-12


def test_live_signal_is_the_traded_ibit_close(live_inputs):
    assert (core6.LIVE_CONFIG.vehicle_str, core6.LIVE_CONFIG.signal_str) == ("ibit", "close")
    signal, close = live_inputs.crypto_signal_ser, live_inputs.close_df[core6.CRYPTO_STR]
    assert signal.dropna().equals(close.dropna())
    ibit_era = signal.loc[core6.IBIT_LISTING_TS:].dropna()
    assert len(ibit_era) > 400 and ibit_era.index[0] == core6.IBIT_LISTING_TS


def test_live_signal_tracks_the_study(inputs, live_inputs):
    """The 16:00 close switch agrees with the CoinMetrics switch on >= 99% of sessions since Binance fills (2017-08-17)
    and the CORE6 daily returns stay almost identical (measured 2026-11-07: 99.7%, corr 0.9999)."""
    since = pd.Timestamp("2017-08-17")
    study_long = core5.asset_signal_df(inputs.crypto_signal_ser, core5.LIVE_CONFIG)["long"].loc[since:]
    live_long = core5.asset_signal_df(live_inputs.crypto_signal_ser, core5.LIVE_CONFIG)["long"].loc[since:]
    both = study_long.notna() & live_long.notna()
    assert (study_long[both] == live_long[both]).mean() >= 0.99
    study_ret = core6.simulate_config(inputs, core6.STUDY_CONFIG)[0]
    live_ret = core6.simulate_config(live_inputs, core6.LIVE_CONFIG)[0]
    common = study_ret.dropna().index.intersection(live_ret.dropna().index)
    assert np.corrcoef(study_ret.reindex(common), live_ret.reindex(common))[0, 1] > 0.999
