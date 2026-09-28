"""Tier B macro audit (2026-09-28): Inflation Compass row-T, publication-lag and split harness checks.

Synthetic and offline. They pin the audit's method (protocol A2/A3/A4/A5) against the committed Compass module:
the row-T harness passes HEAD, catches a planted same-date T5YIE read and a planted Close_(T+1) read, extra FRED
rows dated on or after T (for example from the fred_loader UTC as-of rounding) cannot change the decision at T, and a
future split leaves the target weights unchanged while a planted nominal-price rule is caught.
"""

from __future__ import annotations

from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from alpha.data import fred_loader
from strategies.taa_df import strategy_taa_inflation_compass as cmp_mod

FEATURE_COLS = ["growth_on_bool", "t5yie_float", "t5yie_prior_float", "asset_up_bool", "inflation_on_bool"]


def _signal_close_df(num_day_int: int = 420, seed_int: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("2021-01-04", periods=num_day_int)
    data = {}
    for i, sym in enumerate(cmp_mod.SIGNAL_ASSET_TUPLE):
        ret = rng.normal(0.0003 * (1 + i % 3), 0.012, size=num_day_int)
        data[sym] = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame(data, index=date_index)


def _t5yie_ser(date_index: pd.DatetimeIndex, seed_int: int = 9) -> pd.Series:
    rng = np.random.default_rng(seed_int)
    # Values cross the 2.0 gate often and change every day, so a same-date read is visible.
    values = 2.0 + 0.25 * np.sin(np.arange(len(date_index)) / 9.0) + rng.normal(0.0, 0.03, len(date_index))
    return pd.Series(np.round(values, 2), index=date_index, name="T5YIE")


def _same_date_align(orig_align):
    """Planted pre-b7a0018 rule: the level reads the observation dated T (allow_exact_matches=True)."""

    def leaky(fred_value_ser, session_date_index, tolerance_day_int=7, include_same_date_bool=False):
        return orig_align(fred_value_ser, session_date_index, tolerance_day_int, True)

    return leaky


def _all_rows(sig, t5, cfg=cmp_mod.DEFAULT_CONFIG, same_date_level_bool=False):
    orig_idx = cmp_mod.get_month_end_session_index
    orig_align = cmp_mod.align_fred_to_session_ser
    cmp_mod.get_month_end_session_index = lambda idx: pd.DatetimeIndex(idx).tz_localize(None).normalize()
    if same_date_level_bool:
        cmp_mod.align_fred_to_session_ser = _same_date_align(orig_align)
    try:
        return cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cfg)
    finally:
        cmp_mod.get_month_end_session_index = orig_idx
        cmp_mod.align_fred_to_session_ser = orig_align


def _row_t(sig, t5, T, same_date_level_bool=False):
    """Decision at T from data ending at T: the truncated run's last row."""
    orig_align = cmp_mod.align_fred_to_session_ser
    if same_date_level_bool:
        cmp_mod.align_fred_to_session_ser = _same_date_align(orig_align)
    try:
        feat, w = cmp_mod.compute_month_end_signal_and_weight_df(sig.loc[:T], t5, cmp_mod.DEFAULT_CONFIG)
    finally:
        cmp_mod.align_fred_to_session_ser = orig_align
    assert feat.index[-1] == T
    return feat.loc[T], w.loc[T]


def _rows_equal(a: pd.Series, b: pd.Series) -> bool:
    for c in FEATURE_COLS:
        if isinstance(a[c], (bool, np.bool_)):
            if bool(a[c]) != bool(b[c]):
                return False
        elif not np.isclose(float(a[c]), float(b[c]), rtol=0.0, atol=1e-12):
            return False
    return True


def _cutoffs(sig: pd.DataFrame) -> list[pd.Timestamp]:
    return [pd.Timestamp(x) for x in sig.index[300::12][:8]]


def test_row_t_harness_passes_head_with_publication_aware_fred_truncation():
    sig = _signal_close_df()
    t5 = _t5yie_ser(sig.index)
    full_feat, full_w = _all_rows(sig, t5)
    for T in _cutoffs(sig):
        feat_T, w_T = _row_t(sig, t5[t5.index < T], T)
        assert _rows_equal(feat_T, full_feat.loc[T]), T
        assert np.allclose(w_T.to_numpy(float), full_w.loc[T].to_numpy(float))


def test_row_t_harness_catches_planted_same_date_t5yie_read():
    sig = _signal_close_df()
    t5 = _t5yie_ser(sig.index)
    full_feat, _full_w = _all_rows(sig, t5, same_date_level_bool=True)
    caught = 0
    for T in _cutoffs(sig):
        feat_T, _w = _row_t(sig, t5[t5.index < T], T, same_date_level_bool=True)
        caught += not _rows_equal(feat_T, full_feat.loc[T])
    assert caught >= 6  # T5YIE changes almost every day in the fixture


def test_row_t_harness_catches_planted_close_t_plus_1_read():
    sig = _signal_close_df()
    t5 = _t5yie_ser(sig.index)
    leaky = sig.copy()
    leaky["SPY"] = leaky["SPY"].shift(-1).fillna(leaky["SPY"])
    full_feat, _w = _all_rows(leaky, t5)
    caught = 0
    for T in _cutoffs(sig):
        trunc = sig.loc[:T].copy()
        trunc["SPY"] = trunc["SPY"].shift(-1).fillna(trunc["SPY"])
        feat_T, _wt = cmp_mod.compute_month_end_signal_and_weight_df(trunc, t5[t5.index < T], cmp_mod.DEFAULT_CONFIG)
        caught += not np.isclose(float(feat_T.loc[T, "spy_close_float"]), float(full_feat.loc[T, "spy_close_float"]))
    assert caught == len(_cutoffs(sig))


def test_fred_rows_dated_on_or_after_t_cannot_change_decision_t():
    sig = _signal_close_df()
    t5 = _t5yie_ser(sig.index)
    for T in _cutoffs(sig):
        base_feat, base_w = _row_t(sig, t5[t5.index < T], T)
        # Observation T and three later calendar days present (e.g. via the UTC as-of rounding in fred_loader).
        lenient_feat, lenient_w = _row_t(sig, t5[t5.index <= T + pd.Timedelta(days=3)], T)
        assert _rows_equal(base_feat, lenient_feat)
        assert np.allclose(base_w.to_numpy(float), lenient_w.to_numpy(float))


def test_weekend_month_end_decides_on_friday_with_thursday_t5yie():
    sig = _signal_close_df()
    t5 = _t5yie_ser(sig.index)
    feat, _w = cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cmp_mod.DEFAULT_CONFIG)
    # 2022-04-30 is a Saturday; the last session is Friday 2022-04-29.
    friday = pd.Timestamp("2022-04-29")
    assert friday in feat.index
    assert float(feat.loc[friday, "t5yie_float"]) == float(t5.loc[pd.Timestamp("2022-04-28")])
    assert float(feat.loc[friday, "t5yie_observation_age_day_float"]) == 1.0


def test_future_split_leaves_weights_unchanged_and_planted_nominal_rule_is_caught():
    sig = _signal_close_df()
    t5 = _t5yie_ser(sig.index)
    _f0, w0 = cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cmp_mod.DEFAULT_CONFIG)
    for sym in ("SPY", "XLE", "XLU"):
        for k in (40.0, 0.1, 1.5):
            s = sig.copy()
            s[sym] = s[sym] / k
            _f1, w1 = cmp_mod.compute_month_end_signal_and_weight_df(s, t5, cmp_mod.DEFAULT_CONFIG)
            pd.testing.assert_frame_equal(w0, w1)

    threshold_float = float(sig["SPY"].median())

    def planted(sig_df):
        feat, _w = cmp_mod.compute_month_end_signal_and_weight_df(sig_df, t5, cmp_mod.DEFAULT_CONFIG)
        return (feat["spy_close_float"] > threshold_float).astype(int)

    s40 = sig.copy()
    s40["SPY"] = s40["SPY"] / 40.0
    assert (planted(sig) != planted(s40)).any()


def test_fred_loader_utc_as_of_admits_next_day_but_compass_alignment_neutralises_it(tmp_path, monkeypatch):
    """Documents alpha/data/fred_loader.py:61-66,130-131: a 20:00 New York as-of becomes the next UTC date."""
    cache_path = tmp_path / "T5YIE.csv"
    pd.DataFrame(
        {"observation_date": ["2024-03-26", "2024-03-27", "2024-03-28"], "T5YIE": [2.10, 2.20, 2.30]}
    ).to_csv(cache_path, index=False)

    def _offline(*_args, **_kwargs):
        raise URLError("offline test")

    monkeypatch.setattr(fred_loader, "urlopen", _offline)
    as_of = pd.Timestamp("2024-03-27 20:00", tz="America/New_York").to_pydatetime()
    snap = fred_loader.load_daily_fred_series_snapshot("T5YIE", str(cache_path), as_of, "backtest")
    # The bug: the observation dated 2024-03-28 is admitted for a 2024-03-27 New York evening as-of.
    assert snap.value_ser.index[-1] == pd.Timestamp("2024-03-28")
    # Compass reads only observations dated strictly before the session, so session 2024-03-27 still sees 03-26.
    aligned, _age = cmp_mod.align_fred_to_session_ser(snap.value_ser, pd.DatetimeIndex([pd.Timestamp("2024-03-27")]))
    assert float(aligned.iloc[0]) == pytest.approx(2.10)
