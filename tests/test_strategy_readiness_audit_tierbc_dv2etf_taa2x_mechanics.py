"""Strategy readiness audit 2026-09-28, Tier B/C: Industry-ETF DV2 and the TAA 2x / linearity variants.

Synthetic pins of the mechanics the audit relied on or found. They do not change production code.
Real-data evidence: results/research/strategy_readiness_audit_20260928/tierbc_dv2etf_taa2x/.
"""

import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.live import scheduler_utils  # noqa: E402
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR  # noqa: E402
from strategies.dv2 import strategy_mr_dv2_industry_etf as etf_mod  # noqa: E402
from strategies.dv2.strategy_mr_dv2_liquidity_floor import default_trade_id_int  # noqa: E402
from strategies.taa_df import strategy_taa_df as taa_base  # noqa: E402
from strategies.taa_df import strategy_taa_df_btal_linearity as lin_mod  # noqa: E402
from strategies.taa_df import strategy_taa_df_fallback_vix_cash_variant_utils as vix_mod  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _sessions(n, start="2019-01-02"):
    cal = xcals.get_calendar("XNYS", start="2015-01-02")
    s = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6)))
    return pd.DatetimeIndex(s[:n]).tz_localize(None)


def _etf_panel(n=520, symbols=("AAA", "BBB", "CCC"), seed=5):
    rng = np.random.default_rng(seed)
    idx = _sessions(n)
    cols = {}
    for i, s in enumerate(symbols):
        close = (30.0 + 10 * i) * np.exp(np.cumsum(rng.normal(0.0008, 0.018, n)))
        high = close * (1 + rng.uniform(0.001, 0.02, n))
        low = close * (1 - rng.uniform(0.001, 0.02, n))
        vol = rng.uniform(1e6, 3e6, n)
        cols.update({(s, "Open"): close, (s, "High"): high, (s, "Low"): low, (s, "Close"): close,
                     (s, "Volume"): vol, (s, "Turnover"): vol * close * (0.5 + i), (s, "Unadjusted Close"): close,
                     (s, "Dividend"): np.zeros(n)})
    spx = 3000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    cols.update({("$SPX", "Open"): spx, ("$SPX", "High"): spx, ("$SPX", "Low"): spx, ("$SPX", "Close"): spx})
    prices = pd.DataFrame(cols, index=idx)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns)
    return prices


def _etf_strategy(universe_df):
    s = etf_mod.DVO2IndustryEtfStrategy(name="t", benchmarks=["$SPX"], capital_base=100_000.0, slippage=0.00025,
                                        commission_per_share=0.005, commission_minimum=1.0,
                                        performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
    s.universe_df = universe_df
    s.trade_id = 0
    s.current_trade = defaultdict(default_trade_id_int)
    return s


def _rescale(prices, sym, k):
    out = prices.copy()
    for f in ("Open", "High", "Low", "Close", "Dividend"):
        out[(sym, f)] = out[(sym, f)] / k
    out[(sym, "Volume")] = out[(sym, "Volume")] * k
    return out


FEATURES = ("p126d_return", "natr", "dv2", "sma_200", "adv_63", "raw_price")


def _features(signal_df):
    return signal_df.loc[:, [c for c in signal_df.columns if c[1] in FEATURES]]


# --------------------------------------------------------------------------- Industry-ETF DV2
def test_etf_history_gate_is_causal_and_needs_252_own_sessions():
    prices = _etf_panel()
    prices.loc[prices.index[:100], [("CCC", f) for f in ("Open", "High", "Low", "Close")]] = np.nan  # late listing
    gate = etf_mod.build_history_universe_df(prices)
    t = prices.index[300]
    assert gate.loc[:t].equals(etf_mod.build_history_universe_df(prices.loc[:t]))
    assert gate["AAA"].iloc[250] == 0 and gate["AAA"].iloc[251] == 1
    assert gate["CCC"].iloc[350] == 0 and gate["CCC"].iloc[351] == 1
    assert "$SPX" not in gate.columns


def test_etf_adv63_is_native_turnover_and_future_split_invariant_legacy_formula_is_not():
    prices = _etf_panel()
    universe = etf_mod.build_history_universe_df(prices)
    ref = _etf_strategy(universe).compute_signals(prices.copy())
    turnover = prices[("BBB", "Turnover")].where(prices[("BBB", "Turnover")] > 0)
    expected = turnover.rolling(63, min_periods=63).mean()
    pd.testing.assert_series_equal(ref[("BBB", "adv_63")], expected, check_names=False)
    scale_free = [c for c in _features(ref).columns if c[1] not in ("sma_200", "raw_price")]
    for k in (40.0, 0.1, 1.5):
        cand = _etf_strategy(universe).compute_signals(_rescale(prices, "BBB", k))
        a, b = ref.loc[:, scale_free].to_numpy(float), cand.loc[:, scale_free].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        # SMA200 is in price units but is only compared with Close in the same units
        ratio_ref = ref[("BBB", "Close")] / ref[("BBB", "sma_200")]
        ratio_cand = cand[("BBB", "Close")] / cand[("BBB", "sma_200")]
        assert np.allclose(ratio_ref, ratio_cand, rtol=1e-9, equal_nan=True)
    # positive control: E-01 legacy liquidity (raw Close x split-adjusted Volume) moves by k
    legacy = prices[("BBB", "Unadjusted Close")] * prices[("BBB", "Volume")]
    legacy_k = prices[("BBB", "Unadjusted Close")] * _rescale(prices, "BBB", 40.0)[("BBB", "Volume")]
    assert np.allclose(legacy_k.rolling(63).mean().dropna() / legacy.rolling(63).mean().dropna(), 40.0)


def test_etf_one_zero_turnover_session_blocks_the_liquidity_gate_for_63_sessions():
    prices = _etf_panel()
    d = 300
    prices.loc[prices.index[d], ("AAA", "Turnover")] = 0.0  # padded / no-trade session
    sig = _etf_strategy(etf_mod.build_history_universe_df(prices)).compute_signals(prices.copy())
    adv = sig[("AAA", "adv_63")]
    assert adv.iloc[d:d + 63].isna().all()
    assert np.isfinite(adv.iloc[d - 1]) and np.isfinite(adv.iloc[d + 63])


def test_etf_features_row_t_equal_full_history_and_one_session_leak_is_visible():
    prices = _etf_panel()
    universe = etf_mod.build_history_universe_df(prices)
    full = _etf_strategy(universe).compute_signals(prices.copy())
    for t in prices.index[[260, 333, 400, 519]]:
        trunc = _etf_strategy(universe.loc[:t]).compute_signals(prices.loc[:t].copy())
        assert np.allclose(_features(full).loc[:t].to_numpy(float), _features(trunc).to_numpy(float), rtol=0, atol=0, equal_nan=True)
    t = prices.index[400]
    leak_full = full[("AAA", "dv2")].shift(-1)
    leak_trunc = _etf_strategy(universe.loc[:t]).compute_signals(prices.loc[:t].copy())[("AAA", "dv2")].shift(-1)
    assert leak_full.loc[t] != leak_trunc.loc[t]  # NaN at T in the truncated view vs DV2_(T+1) in the full view


def test_etf_opportunities_apply_50m_gate_rules_and_natr_rank():
    idx = _sessions(3)
    row = {}
    spec = {"AAA": (60e6, 5.0, 2.0), "BBB": (49.9e6, 5.0, 3.0), "CCC": (80e6, 5.0, 4.0), "DDD": (90e6, 12.0, 9.0)}
    for s, (adv, dv2, natr) in spec.items():
        row.update({(s, "Close"): 110.0, (s, "sma_200"): 100.0, (s, "p126d_return"): 0.10, (s, "dv2"): dv2,
                    (s, "natr"): natr, (s, "adv_63"): adv, (s, "raw_price"): 110.0})
    close_row = pd.Series(row)
    close_row.index = pd.MultiIndex.from_tuples(close_row.index)
    universe = pd.DataFrame(1, index=idx, columns=list(spec))
    strat = _etf_strategy(universe)
    strat.previous_bar = idx[-1]
    assert strat.get_opportunities(close_row) == ["CCC", "AAA"]  # BBB fails $50M, DDD fails DV2 < 10


# --------------------------------------------------------------------------- TAA 2x / linearity configs
def test_taa2x_variant_configs_match_their_declared_semantics():
    from strategies.taa_df import strategy_taa_df_1n_fallback_qld_vix_cash as qld
    from strategies.taa_df import strategy_taa_df_1n_fallback_sso_vix_cash as sso
    from strategies.taa_df import strategy_taa_df_btal_1n_fallback_qld_vix_cash as bqld
    from strategies.taa_df import strategy_taa_df_linearity_1n_fallback_qqq_vix_cash as lin

    for mod, fb in ((qld, "QLD"), (sso, "SSO")):
        c = mod.DEFAULT_CONFIG
        assert c.fallback_asset == fb and c.defensive_asset_list == ("GLD", "UUP", "TLT", "DBC")
        assert c.rank_weight_vec == (0.25, 0.25, 0.25, 0.25) and c.start_date_str == "2006-06-21"
    c = bqld.DEFAULT_CONFIG
    assert c.fallback_asset == "QLD" and "BTAL" in c.defensive_asset_list and len(set(c.rank_weight_vec)) == 1
    assert pd.Timestamp(c.start_date_str) >= pd.Timestamp("2011-09-13")
    c = lin.DEFAULT_CONFIG
    assert c.fallback_asset == "QQQ" and c.defensive_asset_list == ("GLD", "UUP", "TLT", "DBC")
    assert c.start_date_str == "2000-01-01"


def _taa_closes(n=900, seed=3):
    rng = np.random.default_rng(seed)
    idx = _sessions(n, start="2016-01-04")
    data = {s: 50 * np.exp(np.cumsum(rng.normal(m, 0.01, n))) for s, m in
            (("GLD", 0.0004), ("UUP", -0.0001), ("TLT", 0.0), ("DBC", 0.0002))}
    return pd.DataFrame(data, index=idx)


def test_taa_month_end_weights_row_t_equal_full_history():
    closes = _taa_closes()
    cash = pd.Series(0.001, index=closes.index, name="cash_return")
    cfg = taa_base.DefenseFirstConfig(defensive_asset_list=("GLD", "UUP", "TLT", "DBC"), fallback_asset="QLD",
                                      rank_weight_vec=(0.25, 0.25, 0.25, 0.25))
    _, full_w = taa_base.compute_month_end_weight_df(closes, cash, cfg)
    month_last = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    for t in month_last.iloc[13:]:
        _, w_t = taa_base.compute_month_end_weight_df(closes.loc[:t], cash.loc[:t], cfg)
        label = (t + pd.offsets.MonthEnd(0)).normalize()
        pd.testing.assert_series_equal(w_t.loc[label], full_w.loc[label])


def test_taa_one_session_leak_changes_the_row_t_momentum_score():
    closes = _taa_closes()
    cash = pd.Series(0.001, index=closes.index, name="cash_return")
    cfg = taa_base.DefenseFirstConfig(defensive_asset_list=("GLD", "UUP", "TLT", "DBC"), fallback_asset="QLD",
                                      rank_weight_vec=(0.25, 0.25, 0.25, 0.25))
    t = closes.index.to_series().groupby(closes.index.to_period("M")).max().iloc[20]
    label = (t + pd.offsets.MonthEnd(0)).normalize()
    full_score, _ = taa_base.compute_month_end_weight_df(closes.shift(-1), cash, cfg)
    trunc_score, _ = taa_base.compute_month_end_weight_df(closes.loc[:t].shift(-1), cash.loc[:t], cfg)
    assert (full_score.loc[label] - trunc_score.loc[label]).abs().max() > 0


def test_vrp_gate_month_end_row_t_equal_full_history():
    rng = np.random.default_rng(9)
    idx = _sessions(400, start="2018-01-02")
    spy = pd.Series(250 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 400))), index=idx)
    vix = pd.Series(np.clip(15 + np.cumsum(rng.normal(0, 0.8, 400)), 9, 60), index=idx)
    full = vix_mod.sample_month_end_vrp_signal_df(vix_mod.compute_daily_vrp_signal_df(spy, vix))
    for t in idx.to_series().groupby(idx.to_period("M")).max().iloc[2:]:
        trunc = vix_mod.sample_month_end_vrp_signal_df(vix_mod.compute_daily_vrp_signal_df(spy.loc[:t], vix.loc[:t]))
        label = (t + pd.offsets.MonthEnd(0)).normalize()
        pd.testing.assert_series_equal(trunc.loc[label], full.loc[label])


def test_linearity_score_is_invariant_to_a_future_split():
    closes = _taa_closes(n=400)
    ref = lin_mod.compute_daily_linearity_score_df(closes, (21, 63))
    scaled = closes.copy()
    scaled["TLT"] = scaled["TLT"] / 40.0
    cand = lin_mod.compute_daily_linearity_score_df(scaled, (21, 63))
    assert np.allclose(ref.to_numpy(), cand.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- live-route prerequisite (C-TAA2X-LP)
def test_host_month_end_resolution_requires_history_inside_the_xnys_calendar_window():
    """Reproducer for the audit's LP finding: the TAA host resolves the decision session by checking EVERY available
    price date against exchange_calendars' default XNYS window (20 years back from today). A history that starts
    before that window (QLD/SSO variants: 2006-06-21; linearity no-BTAL: 2000-01-01) makes the host raise.
    If this test starts failing because no error is raised, the host was fixed: update finding C-TAA2X-05."""
    first_session_ts = pd.Timestamp(scheduler_utils.get_exchange_calendar_obj("XNYS").first_session)
    label_ts = pd.Timestamp("2026-08-31")
    as_of_ts = datetime(2026, 8, 31, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    inside = pd.bdate_range(first_session_ts + pd.Timedelta(days=7), "2026-08-31")
    resolved = scheduler_utils.resolve_calendar_month_end_label_to_last_tradable_session(
        raw_month_end_label_ts=label_ts, available_date_index=inside, session_calendar_id_str="XNYS", as_of_ts=as_of_ts)
    assert resolved == pd.Timestamp("2026-08-31")
    before = pd.bdate_range(first_session_ts - pd.Timedelta(days=120), "2026-08-31")
    with pytest.raises(Exception):
        scheduler_utils.resolve_calendar_month_end_label_to_last_tradable_session(
            raw_month_end_label_ts=label_ts, available_date_index=before, session_calendar_id_str="XNYS", as_of_ts=as_of_ts)
