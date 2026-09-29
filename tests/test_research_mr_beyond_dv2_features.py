"""Causality and timing checks for the MR-beyond-DV2 research features (synthetic data, no Norgate).

Features on row t must use data through Close_t only: perturbing every price after row t must leave rows <= t
unchanged. Forward returns must enter at Open_{t+1} and exit at Open_{t+1+h} (or the last close if delisted).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STUDY_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "mr_beyond_dv2_20260926"


def _features_module():
    if str(STUDY_DIR_PATH) not in sys.path:
        sys.path.insert(0, str(STUDY_DIR_PATH))
    spec = importlib.util.spec_from_file_location("mr_beyond_features", STUDY_DIR_PATH / "features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ft = _features_module()


class FakePanel:
    def __init__(self, dates, close_arr, open_arr):
        self.label_str = "fake"
        self.dates = dates
        self.symbols = [f"S{i}" for i in range(close_arr.shape[1])]
        self.C, self.O = close_arr, open_arr
        self.H, self.L = np.maximum(close_arr, open_arr) * 1.01, np.minimum(close_arr, open_arr) * 0.99
        self.V = np.full(close_arr.shape, 1e6)
        self.RAW, self.DIV = close_arr.copy(), np.zeros(close_arr.shape)
        self.valid = np.isfinite(close_arr)
        self.member = np.ones(close_arr.shape, dtype=bool)
        self.eng, self.extra, self._cache = {}, {}, {}

    def feat(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def col(self, symbol_str):
        return self.symbols.index(symbol_str)


class FakeHedges:
    def __init__(self, close_dict, open_dict):
        self.close, self.open = close_dict, open_dict
        self.vix = np.full(len(next(iter(close_dict.values()))), 20.0)


def _world(seed_int=7, n_days=700, n_stocks=6):
    rng = np.random.default_rng(seed_int)
    dates = pd.bdate_range("2001-01-01", periods=n_days)
    etf_list = ["SPY"] + ft.SECTOR_SPDR_LIST
    etf_ret = rng.normal(0.0003, 0.01, size=(n_days, len(etf_list)))
    etf_close = 100 * np.cumprod(1 + etf_ret, axis=0)
    loading = np.zeros((n_stocks, len(etf_list)))
    for i in range(n_stocks):
        loading[i, 1 + (i % 3)] = 1.0
    stock_ret = etf_ret @ loading.T + rng.normal(0, 0.015, size=(n_days, n_stocks))
    close_arr = 50 * np.cumprod(1 + stock_ret, axis=0)
    open_arr = close_arr * (1 + rng.normal(0, 0.002, size=close_arr.shape))
    etf_open = etf_close * (1 + rng.normal(0, 0.001, size=etf_close.shape))
    close_dict = {s: etf_close[:, j].copy() for j, s in enumerate(etf_list)}
    open_dict = {s: etf_open[:, j].copy() for j, s in enumerate(etf_list)}
    return dates, close_arr, open_arr, close_dict, open_dict


def _perturbed_after(t0_int, close_arr, open_arr, close_dict, open_dict, seed_int=99):
    rng = np.random.default_rng(seed_int)
    c2, o2 = close_arr.copy(), open_arr.copy()
    shock = np.exp(rng.normal(0, 0.2, size=c2[t0_int + 1:].shape))
    c2[t0_int + 1:] *= shock
    o2[t0_int + 1:] *= shock
    cd2 = {k: v.copy() for k, v in close_dict.items()}
    od2 = {k: v.copy() for k, v in open_dict.items()}
    for k in cd2:
        s = np.exp(rng.normal(0, 0.2, size=len(cd2[k]) - t0_int - 1))
        cd2[k][t0_int + 1:] *= s
        od2[k][t0_int + 1:] *= s
    return c2, o2, cd2, od2


def _same(a, b):
    return np.allclose(np.nan_to_num(a, nan=-999.0), np.nan_to_num(b, nan=-999.0), rtol=0, atol=1e-7)


def test_residual_and_raw_signals_use_no_future_data():
    dates, c, o, cd, od = _world()
    t0 = 520
    p1, h1 = FakePanel(dates, c, o), FakeHedges(cd, od)
    c2, o2, cd2, od2 = _perturbed_after(t0, c, o, cd, od)
    p2, h2 = FakePanel(dates, c2, o2), FakeHedges(cd2, od2)
    for hedge_str in ("sec", "spy"):
        z1, z2 = ft.residual_z(p1, h1, hedge_str), ft.residual_z(p2, h2, hedge_str)
        for k in z1:
            assert np.isfinite(z1[k][t0]).any()
            assert _same(z1[k][: t0 + 1], z2[k][: t0 + 1])
    r1, r2 = ft.raw_z(p1), ft.raw_z(p2)
    for k in r1:
        assert _same(r1[k][: t0 + 1], r2[k][: t0 + 1])
    a1, _ = ft.sector_assignment(p1, h1)
    a2, _ = ft.sector_assignment(p2, h2)
    assert (a1[: t0 + 1] == a2[: t0 + 1]).all()


def test_sector_assignment_changes_only_after_a_month_end_and_finds_the_true_sector():
    dates, c, o, cd, od = _world()
    p, h = FakePanel(dates, c, o), FakeHedges(cd, od)
    assign, _ = ft.sector_assignment(p, h)
    me_rows = set(ft.month_end_rows(dates).tolist())
    change_rows = np.nonzero((assign[1:] != assign[:-1]).any(axis=1))[0] + 1
    # *** CRITICAL*** a new assignment first applies on the session after a month-end row
    assert all((r - 1) in me_rows for r in change_rows)
    last = assign[-1]
    assert [ft.SECTOR_SPDR_LIST[x] for x in last] == [ft.SECTOR_SPDR_LIST[i % 3] for i in range(len(last))]


def test_forward_open_returns_enter_next_open_and_liquidate_at_last_close():
    dates, c, o, cd, od = _world()
    c, o = c.copy(), o.copy()
    c[650:, 0] = np.nan  # symbol 0 delists after row 649
    o[650:, 0] = np.nan
    p = FakePanel(dates, c, o)
    h = 5
    F = ft.forward_open_returns(p, h)
    t = 100
    assert np.isclose(F[t, 1], o[t + 1 + h, 1] / o[t + 1, 1] - 1.0, atol=1e-6)
    t = 646  # exit row 652 is after delisting -> last finite close (row 649)
    assert np.isclose(F[t, 0], c[649, 0] / o[t + 1, 0] - 1.0, atol=1e-6)
    assert np.isnan(F[-1, 1])
