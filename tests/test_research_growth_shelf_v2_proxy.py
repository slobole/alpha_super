"""Causality and splice tests for the growth-shelf-v2 2008 proxy (research-only code).

scripts/research/growth_shelf_v2_20260926/proxy_instruments.py builds synthetic leveraged ETFs and a synthetic
BTAL; proxy_runs.py splices them in front of the real funds. These tests pin the contracts the results rely on:
no future row can move a beta or a leg choice, a name without a return keeps its value, the real bars survive the
splice unchanged, and the drag calibration hits its target.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

STUDY_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "growth_shelf_v2_20260926"
sys.path.insert(0, str(STUDY_DIR_PATH))

import proxy_instruments as pi  # noqa: E402
import proxy_runs as pr  # noqa: E402


def test_levered_gross_formula():
    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 101.0, 99.99, 100.99], index=index)
    financing = pd.Series(0.0001, index=index)
    gross = pi.levered_gross(close, 3.0, financing)
    expected = 3.0 * close.pct_change() - 2.0 * 0.0001
    pd.testing.assert_series_equal(gross.iloc[1:], expected.iloc[1:], check_names=False)


def test_levered_bars_open_uses_prior_close_and_overnight_move_only():
    index = pd.bdate_range("2020-01-01", periods=3)
    und = pd.DataFrame({"Open": [100.0, 102.0, 99.0], "High": [101.0, 103.0, 100.0], "Low": [99.0, 101.0, 97.0],
                        "Close": [100.0, 101.0, 98.0], "Volume": [1.0, 1.0, 1.0]}, index=index)
    r_syn = 3.0 * und["Close"].pct_change()
    out = pi.levered_bars(und, 3.0, r_syn)
    # Day 2 open: prior synthetic close times 1 + 3 x the underlying's overnight move (102 / 100 - 1).
    assert out["Open"].iloc[1] == pytest.approx(out["Close"].iloc[0] * (1 + 3 * 0.02))
    assert (out["High"] >= out[["Open", "Close"]].max(axis=1) - 1e-12).all()
    assert (out["Low"] <= out[["Open", "Close"]].min(axis=1) + 1e-12).all()
    assert (out["Dividend"] == 0.0).all()


def test_calibrate_drag_recovers_known_drag():
    index = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(0)
    gross = pd.Series(rng.normal(0.0004, 0.01, len(index)), index=index)
    true_drag = 0.015
    # The first close is the base; its own return is not used.
    real_close = 100.0 * pd.concat([pd.Series([1.0], index=index[:1]), (1.0 + (gross - true_drag / 252.0).iloc[1:]).cumprod()])
    drag = pi.calibrate_drag(gross, real_close, index[0], index[-1])
    assert drag == pytest.approx(true_drag, abs=1e-9)


def test_betas_ignore_rows_after_the_decision_close():
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, (300, 4))
    mkt = rng.normal(0, 0.01, 300)
    ret[:, 0] = 2.0 * mkt  # beta 2 by construction
    base, _ = pi.betas_at(ret, mkt, 260, 252)
    shocked = ret.copy()
    shocked[261:] = 5.0  # anything after the decision row
    after, _ = pi.betas_at(shocked, mkt, 260, 252)
    np.testing.assert_allclose(base, after)
    assert base[0] == pytest.approx(2.0)


def test_select_legs_is_sector_neutral_and_picks_low_beta_long():
    beta = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 0.3, 0.6, 0.9, 1.4, 2.0])
    sectors = np.array(["A"] * 5 + ["B"] * 5)
    eligible = np.ones(10, dtype=bool)
    long_idx, short_idx = pi.select_legs(beta, eligible, sectors)
    assert sorted(long_idx.tolist()) == [0, 5]
    assert sorted(short_idx.tolist()) == [4, 9]


def test_anti_beta_legs_are_chosen_from_data_up_to_the_decision_only():
    """Perturbing returns after a decision changes realised returns but never which names were chosen."""
    dates = pd.bdate_range("2019-01-01", periods=320)
    rng = np.random.default_rng(2)
    n = 60  # the builder skips months with fewer than 50 eligible names
    mkt = rng.normal(0.0003, 0.01, len(dates))
    ret = mkt[:, None] * np.linspace(0.2, 2.0, n)[None, :] + rng.normal(0, 0.005, (len(dates), n))
    data = {"dates": dates, "ret": ret, "overnight": ret * 0.3, "unadj": np.full((len(dates), n), 50.0),
            "member": np.ones((len(dates), n), dtype=bool), "symbols": np.array([f"S{i}" for i in range(n)])}
    spec = {"sector": False, "window": 252, "min_obs": 200}
    sectors = np.array(["X"] * n)
    _, log_a = pi.anti_beta_returns(data, mkt, sectors, spec)
    shocked = dict(data)
    ret_b = ret.copy()
    first_decision = pd.Timestamp(log_a["decision"].iloc[0])
    row = dates.get_loc(first_decision)
    ret_b[row + 1:, :] = ret_b[row + 1:, ::-1]  # scramble the future
    shocked["ret"] = ret_b
    _, log_b = pi.anti_beta_returns(shocked, mkt, sectors, spec)
    assert log_a.iloc[0]["beta_long"] == pytest.approx(log_b.iloc[0]["beta_long"])
    assert log_a.iloc[0]["beta_short"] == pytest.approx(log_b.iloc[0]["beta_short"])


def test_name_without_a_return_keeps_its_value():
    dates = pd.bdate_range("2019-01-01", periods=300)
    n = 60
    rng = np.random.default_rng(3)
    mkt = rng.normal(0.0, 0.01, len(dates))
    ret = mkt[:, None] * np.linspace(0.2, 2.0, n)[None, :]
    month = dates.to_period("M")
    decision_rows = np.nonzero(np.r_[month[1:] != month[:-1], True])[0]
    m = int(decision_rows[decision_rows >= 257][0])
    ret[m + 1:, 0] = np.nan  # the lowest-beta name stops trading right after the decision
    data = {"dates": dates, "ret": ret, "overnight": np.zeros_like(ret), "unadj": np.full((len(dates), n), 50.0),
            "member": np.ones((len(dates), n), dtype=bool), "symbols": np.array([f"S{i}" for i in range(n)])}
    frame, _ = pi.anti_beta_returns(data, mkt, np.array(["X"] * n), {"sector": False, "window": 252, "min_obs": 200})
    assert np.isfinite(frame["gross"].iloc[m + 1])


def test_splice_keeps_real_bars_and_rescales_synthetic_history():
    real_index = pd.bdate_range("2011-09-13", periods=3)
    syn_index = pd.bdate_range("2011-09-07", periods=7)
    columns = ["Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend"]
    real = pd.DataFrame(np.arange(24, dtype=float).reshape(3, 8) + 10.0, index=real_index, columns=columns)
    syn = pd.DataFrame(1.0, index=syn_index, columns=columns)
    syn["Close"] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    out = pr.splice(real, syn)
    pd.testing.assert_frame_equal(out.loc[real_index], real)
    scale = real["Close"].iloc[0] / syn.loc[real_index[0], "Close"]
    pre = out.loc[out.index < real_index[0]]
    np.testing.assert_allclose(pre["Close"].to_numpy(), syn.loc[syn.index < real_index[0], "Close"].to_numpy() * scale)
    # The synthetic return into the first real bar is preserved.
    assert real["Close"].iloc[0] / pre["Close"].iloc[-1] == pytest.approx(1.4 / 1.3)


def _bars(index, close):
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0,
                          "Turnover": close, "Unadjusted Close": close, "Dividend": 0.0}, index=index)
    frame.index.name = "Date"
    return frame


def test_patched_loader_passes_other_symbols_and_serves_synthetic_only_where_asked():
    real_index = pd.bdate_range("2011-09-13", periods=4)
    syn_index = pd.bdate_range("2011-09-01", periods=12)
    calls = []

    def original(symbol_str, **kwargs):
        calls.append((symbol_str, kwargs.get("start_date_str")))
        return _bars(real_index, np.array([10.0, 11.0, 12.0, 13.0]))

    def bars_fn(symbol_str, label_str):
        return _bars(syn_index, np.linspace(1.0, 2.1, len(syn_index)))

    passthrough = pr.make_patched_loader(original, "splice_scaled", bars_fn)("SPY", start_date_str="2011-01-01")
    assert calls[-1] == ("SPY", "2011-01-01") and len(passthrough) == 4

    spliced = pr.make_patched_loader(original, "splice_scaled", bars_fn)("BTAL", start_date_str="2011-09-05", end_date_str=None)
    # the real bars come back untouched from the first real date on
    np.testing.assert_allclose(spliced.loc[real_index, "Close"].to_numpy(), [10.0, 11.0, 12.0, 13.0])
    assert spliced.index[0] == pd.Timestamp("2011-09-05")  # start filter applied
    assert (spliced.index < real_index[0]).sum() > 0  # synthetic history in front
    # the full-history request to Norgate asks for an explicit start (Norgate returns nothing for None)
    assert calls[-1] == ("BTAL", "1990-01-01")

    full = pr.make_patched_loader(original, "syn_scaled", bars_fn)("TQQQ", end_date_str="2011-09-14")
    np.testing.assert_allclose(full["Close"].to_numpy(), bars_fn("TQQQ", "scaled").loc[:"2011-09-14", "Close"].to_numpy())


def test_fill_before_leaves_real_returns_untouched():
    import shelf_books as sb

    index = pd.bdate_range("2012-09-25", periods=10)
    real = pd.Series(np.nan, index=index)
    real[index >= pd.Timestamp("2012-10-02")] = 0.01
    proxy = pd.Series(0.05, index=index)
    out = sb.fill_before(real, proxy, pd.Timestamp("2012-10-02"))
    assert (out[index < pd.Timestamp("2012-10-02")] == 0.05).all()
    assert (out[index >= pd.Timestamp("2012-10-02")] == 0.01).all()


def test_commission_fix_reprices_on_real_share_count():
    import commission_fix as cf

    engine = np.array([384.0, 1.0, 50.0])
    ratio = np.array([384.0, 384.0, 1.0])
    np.testing.assert_allclose(cf.real_commission(engine, ratio), [1.0, 1.0, 50.0])
    index = pd.bdate_range("2013-01-01", periods=3)
    nav = pd.Series([1000.0, 1000.0, 2000.0], index=index)
    tx = pd.DataFrame({"date": [index[1], index[1], index[2]], "commission_float": [101.0, 1.0, 10.0]})
    add = cf.add_back_ser(tx, nav, np.array([101.0, 101.0, 1.0]))
    # day 2: (101 - 1) + (1 - 1) saved over the prior NAV of 1000; day 3: nothing saved
    np.testing.assert_allclose(add.to_numpy(), [0.0, 0.1, 0.0])
