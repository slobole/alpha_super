"""Tier C hedge readiness audit (2026-09-28): synthetic tests that pin the audited behaviour.

Strategies: Crisis Trend Core (CTC), VIXM Backwardation (VIXM), Trinity vol control (TRIN). Each test documents one
finding of results/research/strategy_readiness_audit_20260928/tierc_hedge/TIERC_HEDGE_FINDINGS.md. No production code
is modified; these tests read the committed modules only.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily  # noqa: E402
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040 as b6040  # noqa: E402
from strategies.taa_beyond_6040 import strategy_taa_trinity_vol_control_8_bil as trin  # noqa: E402
from strategies.tail_hedge import strategy_crisis_trend_core as ctc  # noqa: E402
from strategies.tail_hedge import strategy_vixm_backwardation as vixm  # noqa: E402


# ----------------------------------------------------------------------------- fixtures
def _ctc_tr_close_df(sessions_int: int = 900, seed_int: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed_int)
    idx = pd.bdate_range("2015-01-02", periods=sessions_int)
    data = {}
    for asset_str in ctc.TRADEABLE_ASSET_TUPLE:
        drift = 0.0001 if asset_str == ctc.RESERVE_ASSET_STR else 0.0
        vol = 0.0005 if asset_str == ctc.RESERVE_ASSET_STR else 0.01
        data[asset_str] = 100.0 * np.cumprod(1.0 + drift + vol * rng.standard_normal(sessions_int))
    return pd.DataFrame(data, index=idx)


def _trin_pricing_df(num_days_int: int = 260, seed_int: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed_int)
    idx = pd.date_range("2023-01-02", periods=num_days_int, freq="B")
    fields = {}
    for sym, base, vol in (("VTI", 100.0, 0.012), ("GLD", 120.0, 0.009), ("TLT", 110.0, 0.008),
                           ("BIL", 90.0, 0.0002), ("$SPX", 4000.0, 0.01)):
        close = base * np.cumprod(1.0 + 0.0003 + vol * rng.standard_normal(num_days_int))
        fields[(sym, "Open")] = close * 0.999
        fields[(sym, "High")] = close * 1.002
        fields[(sym, "Low")] = close * 0.997
        fields[(sym, "Close")] = close
        fields[(sym, "Dividend")] = np.zeros(num_days_int)
    df = pd.DataFrame(fields, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ----------------------------------------------------------------------------- CTC
def test_ctc_reserve_starting_after_history_start_raises_as_head_does_on_real_data():
    """C-CTC-01: at HEAD the default loader starts 2002-01-01 but SHY starts 2002-07-26 (142 sessions later), so the
    first eligible row needs SHY at T-252 and the whole run raises. Trimming the frame to SHY's first bar runs."""
    tr = _ctc_tr_close_df()
    tr.loc[tr.index[:142], ctc.RESERVE_ASSET_STR] = np.nan
    with pytest.raises(ValueError, match="SHY TOTALRETURN signal endpoints"):
        ctc.compute_crisis_trend_signal_bundle(tr)
    trimmed = tr.iloc[142:]
    bundle = ctc.compute_crisis_trend_signal_bundle(trimmed)
    assert bundle.desired_weight_df.abs().sum(axis=1).gt(0).any()


def test_ctc_last_row_of_any_frame_is_flagged_month_end():
    """C-CTC-05: the month-end flag uses shift(-1) of the frame's own index, so the final row is always a
    'month-end'. The engine never iterates on it, but a live caller with data ending mid-month would re-target."""
    idx = pd.bdate_range("2026-09-01", "2026-09-25")
    flag = ctc.month_end_decision_bool_ser(idx)
    assert bool(flag.iloc[-1])
    assert int(flag.sum()) == 1


def test_ctc_long_gross_above_one_gets_no_reserve_and_is_financed_by_cash():
    """C-CTC-03: the reserve is max(0, 1 - long gross); a 1.3 long safe-haven book holds 130% of NAV long with no SHY
    and relies on unfinanced negative cash / short proceeds."""
    risk = pd.Series(0.0, index=ctc.UNIVERSE_ASSET_TUPLE)
    risk["TLT"] = 0.8
    risk["GLD"] = 0.5
    risk["SPY"] = -0.2
    tgt = ctc.build_tradeable_target_weight_ser(risk)
    assert tgt[ctc.RESERVE_ASSET_STR] == 0.0
    assert np.isclose(tgt.clip(lower=0).sum(), 1.3)
    assert np.isclose(1.0 - tgt.sum(), 1.0 - 1.1)  # implied engine cash = -0.1 of NAV


@pytest.mark.parametrize("cut_int", [300, 420, 555, 700, 899])
def test_ctc_row_t_truncation_invariance_and_planted_leak_is_caught(cut_int):
    tr = _ctc_tr_close_df()
    full = ctc.compute_crisis_trend_signal_bundle(tr)
    T = tr.index[cut_int - 1]
    pref = ctc.compute_crisis_trend_signal_bundle(tr.loc[:T])
    pd.testing.assert_series_equal(pref.desired_weight_df.loc[T], full.desired_weight_df.loc[T])
    pd.testing.assert_series_equal(pref.realized_volatility_df.loc[T], full.realized_volatility_df.loc[T])
    assert pref.exante_volatility_ser.loc[T] == full.exante_volatility_ser.loc[T] or (
        np.isnan(pref.exante_volatility_ser.loc[T]) and np.isnan(full.exante_volatility_ser.loc[T])
    )
    if cut_int == len(tr):
        return
    leak = lambda d: d.shift(-1).fillna(d)  # noqa: E731  reads Close_(T+1) when it exists
    full_leak = ctc.compute_crisis_trend_signal_bundle(leak(tr))
    pref_leak = ctc.compute_crisis_trend_signal_bundle(leak(tr.loc[:T]))
    assert not np.allclose(
        pref_leak.realized_volatility_df.loc[T].to_numpy(dtype=float),
        full_leak.realized_volatility_df.loc[T].to_numpy(dtype=float),
        equal_nan=True,
    )


# ----------------------------------------------------------------------------- VIXM
def _vix_pair(n: int = 300, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    v3 = pd.Series(20 + np.cumsum(rng.normal(0, 0.3, n)), index=idx).clip(lower=10)
    v = v3 * (1 + rng.normal(-0.05, 0.06, n))
    return v, v3


@pytest.mark.parametrize("cut_int", [50, 120, 233, 299])
def test_vixm_state_row_t_invariance_and_planted_leak_is_caught(cut_int):
    v, v3 = _vix_pair()
    full = vixm.compute_backwardation_state_ser(v, v3)
    T = v.index[cut_int - 1]
    pref = vixm.compute_backwardation_state_ser(v.loc[:T], v3.loc[:T])
    assert pref.loc[T] == full.loc[T]
    if cut_int == len(v):
        return
    leak = lambda x: x.shift(-1).fillna(x)  # noqa: E731  reads Close_(T+1) when it exists
    ratio_full = (leak(v) / leak(v3)).loc[T]
    ratio_pref = (leak(v.loc[:T]) / leak(v3.loc[:T])).loc[T]
    assert ratio_full != ratio_pref  # the row-T harness sees the planted leak


def test_vixm_stale_padded_vix3m_passes_silently():
    """C-VIXM-06: the fail-loud policy only covers NaN/non-positive inputs. A padded (repeated) VIX3M close, which
    Norgate's ALLMARKETDAYS padding produces for a missing print, yields a normal 0/1 state with no warning."""
    idx = pd.bdate_range("2024-08-01", periods=4)
    v = pd.Series([16.0, 23.4, 38.6, 27.7], index=idx)
    v3_true = pd.Series([18.0, 21.0, 29.0, 24.0], index=idx)
    v3_stale = v3_true.copy()
    v3_stale.iloc[2] = v3_true.iloc[1]  # one-day-stale helper
    st_true = vixm.compute_backwardation_state_ser(v, v3_true)
    st_stale = vixm.compute_backwardation_state_ser(v, v3_stale)
    assert st_stale.notna().all()
    assert (st_true == st_stale).all()  # here same state; the audit counts 258 flips on real data 2011-2026
    v_small = v.copy()
    v_small.iloc[2] = 25.0
    assert vixm.compute_backwardation_state_ser(v_small, v3_true).iloc[2] == 0.0
    assert vixm.compute_backwardation_state_ser(v_small, v3_stale).iloc[2] == 1.0  # silent flip


# ----------------------------------------------------------------------------- TRINITY
def test_trinity_vol_window_ends_at_close_t_and_includes_return_t():
    """C-TRIN-A1: the 63-return window ends at Close_T (return T included); orders fill at Open_(T+1)."""
    idx = pd.bdate_range("2024-01-02", periods=80)
    rr = pd.DataFrame({"VTI": 0.001, "GLD": 0.001, "TLT": 0.001}, index=idx)
    w = pd.Series({"VTI": 0.4, "GLD": 0.3, "TLT": 0.3})
    base = trin.compute_base_portfolio_return_ser(rr, w, 63)
    assert base.index[-1] == idx[-1]
    rr_spike = rr.copy()
    rr_spike.iloc[-1] = -0.08
    m_calm = b6040.compute_gross_exposure_float(trin.compute_base_portfolio_return_ser(rr, w, 63), 63, 0.08, 0.085)
    m_spike = b6040.compute_gross_exposure_float(trin.compute_base_portfolio_return_ser(rr_spike, w, 63), 63, 0.08, 0.085)
    assert m_calm == 1.0 and m_spike < 1.0


def test_trinity_truncated_prefix_at_month_end_has_no_monthly_flag_or_new_weights():
    """C-TRIN-05: new base weights and the monthly override are attached to the session before next month's first
    session, which a frame ending at the month-end close cannot contain. A live route must supply the calendar."""
    df = _trin_pricing_df()
    s = trin.TrinityVolControlStrategy(name="t", benchmarks=["$SPX"])
    full = s.compute_signals(df)
    month_end = pd.Timestamp("2023-09-29")
    assert bool(full.loc[month_end, trin.MONTHLY_REBALANCE_FIELD_TUPLE])
    pref = s.compute_signals(df.loc[:month_end])
    assert not bool(pref.loc[month_end, trin.MONTHLY_REBALANCE_FIELD_TUPLE])
    assert not np.isclose(float(pref.loc[month_end, ("VTI", "base_weight_ser")]),
                          float(full.loc[month_end, ("VTI", "base_weight_ser")]))
    # the return / volatility features themselves are row-T invariant
    for a in trin.RISK_ASSET_TUPLE:
        assert pref.loc[month_end, (a, "return_ser")] == full.loc[month_end, (a, "return_ser")]
        assert pref.loc[month_end, (a, "volatility_ser")] == full.loc[month_end, (a, "volatility_ser")]


def test_trinity_decisions_never_read_the_realized_return_history_used_by_the_timing_adapter():
    """C-TRIN-04: the timing adapter's realized-return bookkeeping (the share-units handoff's 'double-subtract')
    cannot move Trinity decisions, because Trinity's overlay reads base-portfolio returns only."""
    df = _trin_pricing_df()
    df.attrs["norgate_adjustment_by_symbol_dict"] = {"VTI": "CAPITALSPECIAL", "GLD": "CAPITALSPECIAL",
                                                     "TLT": "CAPITALSPECIAL", "BIL": "CAPITALSPECIAL",
                                                     "$SPX": "TOTALRETURN"}
    cfg = trin.DEFAULT_CONFIG
    cal = trin._execution_calendar_index(df, cfg, None)

    def boom(self):
        raise AssertionError("realized strategy returns must not drive Trinity decisions")

    s = trin._build_trinity_strategy(cfg, 100_000.0)
    with patch.object(b6040.Beyond6040Strategy, "_realized_strategy_return_ser", boom):
        run_daily(s, df, calendar=cal, show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    assert len(s.get_transactions()) > 0
