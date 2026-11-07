"""BTC vol sleeve spec (alpha/scout/specs/btc_vol_sleeve.py): timing, band and the shadow guard. Data lane only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.specs import btc_vol_sleeve as S

pytestmark = pytest.mark.data_integration


@pytest.fixture(scope="module")
def inputs():
    return S.load_inputs()


def test_target_ignores_future_prices(inputs):
    cut = pd.Timestamp("2025-06-13")
    base = S.target_ser(inputs).loc[:cut]
    factor = pd.Series(np.where(inputs.close_df.index <= cut, 1.0, 2.0), index=inputs.close_df.index)
    shocked = S.BtcVolSleeveInputs(inputs.open_df, inputs.close_df.mul(factor, axis=0), inputs.dividend_df)
    assert np.allclose(base.dropna().to_numpy(), S.target_ser(shocked).loc[:cut].dropna().to_numpy())


def test_trades_only_after_week_ends_and_inside_band(inputs):
    res = S.simulate_config(inputs)
    decided = res.decided_weight_df
    idx = inputs.close_df.index
    week_end = pd.Series(S.week_end_mask(idx), idx)
    target = S.target_ser(inputs)
    for t in decided.index:
        T = idx[idx.get_loc(t) - 1]
        assert week_end.loc[T]
        w, tgt = float(decided.loc[t, "IBIT"]), float(target.loc[T])
        assert abs(w - tgt) <= 0.10 * tgt + 1e-12
        assert abs(decided.loc[t].sum() - 1.0) < 1e-12


def test_provider_refuses_a_changed_rule():
    with pytest.raises(RuntimeError, match="new registration"):
        S.shadow_daily_return_ser({"frozen_config_dict": {**S.frozen_config_dict(), "vol_target_float": 0.25}, "start_date_str": "2026-11-09"})
