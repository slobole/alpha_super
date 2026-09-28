"""Strategy readiness audit 2026-09-28, Tier B EOM flow + sector-ETF IBS pods: synthetic pins of audited mechanics.

These tests pin behaviour the audit relied on or found; they do not change production code.
Real-data evidence lives in results/research/strategy_readiness_audit_20260928/tierb_etf_mr/.
"""

import os
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.taa_beyond_6040 import strategy_taa_month_end_rebalancing_flow as eom_mod  # noqa: E402
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (  # noqa: E402
    SectorDispersionIbsConfig,
    compute_sector_dispersion_ibs_signal_df,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 import (  # noqa: E402
    compute_asset_sma200_filtered_signal_df,
)
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock import (  # noqa: E402
    UsSectorEtfIbsDownshockConfig,
    compute_us_sector_etf_ibs_downshock_signal_df,
)


def _xnys(start, end):
    cal = xcals.get_calendar("XNYS", start="2002-07-01", end="2027-12-31")
    return pd.DatetimeIndex(cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))).tz_localize(None)


def _tr_panel(end="2006-12-29", seed=7):
    idx = _xnys("2002-07-26", end)
    rng = np.random.default_rng(seed)
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
    ief = 80.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, len(idx))))
    return pd.DataFrame({"SPY": spy, "IEF": ief}, index=idx)


def _leak_month_table(tr):
    """Planted leak: pressure read at the FILL close [-6] with an as-of fallback (never crashes)."""
    sess = eom_mod.exchange_session_idx(tr.index[-1])
    per = sess.to_period("M")
    out, prior = [], []
    for m in pd.period_range("2002-08", tr.index[-1].to_period("M"), freq="M"):
        msi = sess[per == m]
        if len(msi) < 9 or msi[-7] > tr.index[-1]:
            continue
        prev = sess[per == m - 1][-1]
        row = tr.loc[: msi[-6]].iloc[-1]
        g_s, g_i = row["SPY"] / tr.loc[prev, "SPY"], row["IEF"] / tr.loc[prev, "IEF"]
        p = 10_000.0 * (0.4 - 0.4 * g_i / (0.6 * g_s + 0.4 * g_i))
        out.append({"measure_date": msi[-7], "pressure": p})
        prior.append(p)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- EOM
def test_eom_month_table_row_t_truncation_is_exact_at_measure_dates():
    tr = _tr_panel()
    full = eom_mod.build_month_table_df(tr)
    for measure_ts in full["measure_date"].iloc[[30, 31, 40, -2]]:
        trunc = eom_mod.build_month_table_df(tr.loc[:measure_ts])
        a = full[full["measure_date"] <= measure_ts].reset_index(drop=True)
        b = trunc.reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_eom_planted_fill_close_leak_is_caught_at_row_t():
    tr = _tr_panel()
    full = _leak_month_table(tr)
    caught = 0
    for measure_ts in full["measure_date"].iloc[[30, 31, 40, -2]]:
        trunc = _leak_month_table(tr.loc[:measure_ts])
        caught += int(abs(full[full["measure_date"] == measure_ts]["pressure"].iloc[0]
                          - trunc["pressure"].iloc[-1]) > 0)
    assert caught == 4


def test_eom_pressure_and_bucket_are_split_invariant_but_dollar_pressure_is_not():
    tr = _tr_panel()
    base = eom_mod.build_month_table_df(tr)
    for k in (40.0, 0.1, 1.5):
        scaled = tr.copy()
        scaled["SPY"] = scaled["SPY"] / k
        cand = eom_mod.build_month_table_df(scaled)
        np.testing.assert_allclose(cand["pressure_ief_measure_bps_float"], base["pressure_ief_measure_bps_float"],
                                   rtol=1e-9, atol=1e-9)
        assert cand["bucket_ief_measure_causal_int"].equals(base["bucket_ief_measure_causal_int"])
    # positive control: growth from a dollar difference depends on the price scale
    def dollar_growth(frame):
        synth = (1.0 + frame.diff().fillna(0.0) / 100.0).cumprod()
        return eom_mod.build_month_table_df(synth)
    scaled = tr.copy()
    scaled["SPY"] = scaled["SPY"] / 40.0
    assert not dollar_growth(scaled)["bucket_ief_measure_causal_int"].equals(dollar_growth(tr)["bucket_ief_measure_causal_int"])


def test_eom_bucket_uses_prior_months_only_and_missing_bucket_trades_default_legs():
    prior = list(np.linspace(-100, 100, 24))
    assert np.isnan(eom_mod.causal_bucket_float(0.0, prior[:23]))
    # the current value is not in the prior list: max(prior) -> F = 1 -> bucket 5
    assert eom_mod.causal_bucket_float(100.0, prior) == 5
    assert eom_mod.causal_bucket_float(-1000.0, prior) == 1
    # documented default (docs/research/month_end_rebalancing_flow.md:20-24): missing bucket == bucket 2/3 legs
    nan = float("nan")
    assert eom_mod.target_weight_tuple(nan, "final") == (0.0, 1.0)
    assert eom_mod.target_weight_tuple(nan, "early") == (0.0, -1.0)


def test_eom_schedule_decision_is_the_session_before_every_fill():
    tr = _tr_panel()
    table = eom_mod.build_month_table_df(tr)
    sched = eom_mod.build_order_schedule_df(table)
    sess = eom_mod.exchange_session_idx(tr.index[-1])
    for fill_ts, row in sched.iterrows():
        decision_ts = sess[sess.get_loc(fill_ts) - 1]
        assert row["measure_date"] <= decision_ts < fill_ts


def test_eom_sandy_schedule_depends_on_closures_known_only_afterwards():
    """XNYS 2012-10-29/30 (announced 2012-10-28/29) move the October 2012 measure date back from 10-23 to 10-19."""
    tr = _tr_panel(end="2012-12-31")
    table = eom_mod.build_month_table_df(tr).set_index("month_period")
    assert str(table.loc["2012-10", "measure_date"].date()) == "2012-10-19"
    assert str(table.loc["2012-10", "final_fill_date"].date()) == "2012-10-22"
    original = eom_mod.exchange_session_idx
    try:
        eom_mod.exchange_session_idx = lambda end_ts: original(end_ts).union(
            pd.DatetimeIndex(["2012-10-29", "2012-10-30"]))
        notice = eom_mod.build_month_table_df(tr).set_index("month_period")
    finally:
        eom_mod.exchange_session_idx = original
    assert str(notice.loc["2012-10", "measure_date"].date()) == "2012-10-23"
    assert str(notice.loc["2012-10", "final_fill_date"].date()) == "2012-10-24"


# --------------------------------------------------------------------------- sector IBS pods
def _ohlc_panel(symbols, n=320, seed=3, start="2020-01-02"):
    idx = _xnys(start, "2023-12-29")[:n]
    rng = np.random.default_rng(seed)
    frames = []
    for i, s in enumerate(symbols):
        close = 50.0 * (i + 1) * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
        hi = close * (1 + rng.uniform(0.001, 0.03, n))
        lo = close * (1 - rng.uniform(0.001, 0.03, n))
        op = lo + (hi - lo) * rng.uniform(0, 1, n)
        cl = lo + (hi - lo) * rng.uniform(0, 1, n)
        f = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": cl, "Volume": 1e6,
                          "Unadjusted Close": cl * 2.0, "Dividend": 0.0}, index=idx)
        f.columns = pd.MultiIndex.from_product([[s], f.columns])
        frames.append(f)
    return pd.concat(frames, axis=1)


def _feature_cols(frame, symbols):
    raw = {"Open", "High", "Low", "Close", "Volume", "Unadjusted Close", "Dividend"}
    return [c for c in frame.columns if c[0] in symbols and c[1] not in raw]


def test_dispersion_and_sma200_features_row_t_truncation_is_exact():
    syms = ("AAA", "BBB", "CCC")
    cfg = SectorDispersionIbsConfig(symbol_tuple=syms)
    px = _ohlc_panel(syms)
    for fn in (compute_sector_dispersion_ibs_signal_df, compute_asset_sma200_filtered_signal_df):
        full = fn(px, cfg)
        cols = _feature_cols(full, syms)
        for cut in px.index[[60, 150, 230, 319]]:
            trunc = fn(px.loc[:cut], cfg)
            pd.testing.assert_frame_equal(full.loc[:cut, cols], trunc.loc[:, cols])


def test_downshock_features_row_t_truncation_is_exact_and_one_session_leak_is_caught():
    syms = ("AAA", "BBB", "CCC")
    cfg = UsSectorEtfIbsDownshockConfig(symbol_tuple=syms, max_positions_int=2, history_start_date_str="2019-01-01",
                                        backtest_start_date_str="2020-01-01")
    px = _ohlc_panel(syms)
    full = compute_us_sector_etf_ibs_downshock_signal_df(px, cfg)
    cols = _feature_cols(full, syms)
    caught = 0
    for cut in px.index[[60, 150, 230, 318]]:
        trunc = compute_us_sector_etf_ibs_downshock_signal_df(px.loc[:cut], cfg)
        pd.testing.assert_frame_equal(full.loc[:cut, cols], trunc.loc[:, cols])
        # positive control: IBS from Close_(T+1)
        leak_full = (px[("AAA", "Close")].shift(-1) - px[("AAA", "Low")]) / (px[("AAA", "High")] - px[("AAA", "Low")])
        cut_px = px.loc[:cut]
        leak_trunc = (cut_px[("AAA", "Close")].shift(-1) - cut_px[("AAA", "Low")]) / (
            cut_px[("AAA", "High")] - cut_px[("AAA", "Low")])
        caught += int(not np.isclose(leak_full.loc[cut], leak_trunc.loc[cut], equal_nan=True))
    assert caught == 4


def test_padded_bar_blanks_ibs_and_suppresses_range_signal_for_the_lookback():
    """ALLMARKETDAYS padding (O=H=L=C=prior close) gives NaN IBS and NaN range scale for 21 sessions after it."""
    syms = ("AAA",)
    cfg = SectorDispersionIbsConfig(symbol_tuple=syms)
    px = _ohlc_panel(syms, n=120)
    pad = px.index[60]
    prev_close = px.loc[px.index[59], ("AAA", "Close")]
    for f in ("Open", "High", "Low", "Close"):
        px.loc[pad, ("AAA", f)] = prev_close
    sig = compute_sector_dispersion_ibs_signal_df(px, cfg)
    assert np.isnan(sig.loc[pad, ("AAA", "ibs_value_ser")])
    scale = sig[("AAA", "range_vol_21_ser")]
    blanked = scale.iloc[61:82]  # the 21 sessions whose lagged window contains the padded bar
    assert len(blanked) == 21 and blanked.isna().all() and not np.isnan(scale.iloc[82])
    assert not sig.loc[px.index[60]:px.index[81], ("AAA", "entry_signal_bool")].any()


def test_float32_rounding_flips_an_exact_ibs_exit_tie():
    """A bar whose IBS is exactly 0.90 in decimal exits in float64 (> 0.90) but not in float32 storage."""
    ibs64 = (10.9 - 10.0) / (11.0 - 10.0)
    c32, h32, l32 = np.float32(10.9), np.float32(11.0), np.float32(10.0)
    ibs32 = (c32 - l32) / (h32 - l32)
    assert ibs64 > 0.90
    assert not (float(ibs32) > 0.90)
    assert abs(float(ibs32) - 0.90) < 1e-6
