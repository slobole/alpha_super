"""Tier B macro audit (2026-09-28): Tactical FI row-T truncation, release model and cash-accrual lag.

Synthetic and offline, against the committed module. The row-T harness truncates the FRED panel by the module's
own release model (observations up to session T-1) and the session index at T plus the next session; HEAD must
match its full-history decisions and a planted one-session leak (observation T read through row T-1) must be caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.taa_beyond_6040 import strategy_taa_tactical_fixed_income_ief_lqd as tfi

SIGNAL_COLS = ["observation_date", "term_spread_float", "credit_spread_float", "term_threshold_float",
               "credit_threshold_float", "term_state_float", "credit_state_float"]


def _sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2014-01-02", "2015-02-06")


def _yield_df(seed_int: int = 11) -> pd.DataFrame:
    idx = pd.bdate_range("2012-06-01", "2015-02-06")
    rng = np.random.default_rng(seed_int)
    steps = rng.normal(0.0, 0.04, size=(len(idx), 4)).cumsum(axis=0)
    return pd.DataFrame(np.round(np.array([2.5, 0.1, 4.2, 5.0]) + steps, 2), index=idx,
                        columns=list(tfi.FRED_SERIES_ID_TUPLE))


def _month_end_cutoffs(sessions: pd.DatetimeIndex) -> list[pd.Timestamp]:
    ser = pd.Series(sessions, index=sessions)
    me = ser.groupby(sessions.to_period("M")).max()
    return [pd.Timestamp(x) for x in me.values[2:12]]


def _truncated(yield_df, sessions, T):
    prior = sessions[sessions < T][-1]
    nxt = sessions[sessions > T][0]
    sess_T = sessions[sessions <= T].append(pd.DatetimeIndex([nxt]))
    return yield_df[yield_df.index <= prior], sess_T, str(T.to_period("M"))


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if not a.index.equals(b.index):
        return False
    if not (a["observation_date"] == b["observation_date"]).all():
        return False
    return bool(np.allclose(a[SIGNAL_COLS[1:]].to_numpy(float), b[SIGNAL_COLS[1:]].to_numpy(float), atol=1e-12))


def test_row_t_truncation_head_matches_full_history():
    sessions, y = _sessions(), _yield_df()
    full_sig, _w = tfi.build_month_end_signal_and_weight_df(y, sessions, "2015-01")
    for T in _month_end_cutoffs(sessions):
        y_T, s_T, last_month = _truncated(y, sessions, T)
        sig_T, _wt = tfi.build_month_end_signal_and_weight_df(y_T, s_T, last_month)
        assert _equal(sig_T[SIGNAL_COLS], full_sig.loc[full_sig.index <= T, SIGNAL_COLS]), T


def test_row_t_truncation_catches_planted_one_session_leak():
    sessions, y = _sessions(), _yield_df()
    leak = y.shift(-1).dropna(how="all")  # row T-1 now holds observation T
    full_sig, _w = tfi.build_month_end_signal_and_weight_df(leak, sessions, "2015-01")
    caught = 0
    for T in _month_end_cutoffs(sessions):
        y_T, s_T, last_month = _truncated(y, sessions, T)
        sig_T, _wt = tfi.build_month_end_signal_and_weight_df(y_T.shift(-1).dropna(how="all"), s_T, last_month)
        caught += not _equal(sig_T[SIGNAL_COLS], full_sig.loc[full_sig.index <= T, SIGNAL_COLS])
    assert caught == len(_month_end_cutoffs(sessions))


def test_observations_after_t_minus_1_never_enter_decision_t():
    sessions, y = _sessions(), _yield_df()
    base_sig, _w = tfi.build_month_end_signal_and_weight_df(y, sessions, "2015-01")
    noisy = y.copy()
    T = _month_end_cutoffs(sessions)[5]
    rng = np.random.default_rng(3)
    after = noisy.index >= T
    noisy.loc[after] = noisy.loc[after] + rng.normal(0.0, 5.0, size=(int(after.sum()), 4))
    noisy_sig, _w2 = tfi.build_month_end_signal_and_weight_df(noisy, sessions, "2015-01")
    assert _equal(base_sig.loc[:T, SIGNAL_COLS], noisy_sig.loc[:T, SIGNAL_COLS])
    assert not _equal(base_sig.loc[T:, SIGNAL_COLS].iloc[1:], noisy_sig.loc[T:, SIGNAL_COLS].iloc[1:])


def test_bond_holiday_before_decision_uses_older_common_row_with_age_one():
    sessions, y = _sessions(), _yield_df()
    sessions = sessions.drop(pd.Timestamp("2014-11-27"))  # NYSE closed on Thanksgiving
    T = pd.Timestamp("2014-11-28")  # day after Thanksgiving, a month-end session
    y = y.copy()
    y.loc[pd.Timestamp("2014-11-27")] = np.nan  # Thanksgiving: no FRED row
    obs = tfi.select_publication_safe_observation_date(T, y, sessions)
    assert obs == pd.Timestamp("2014-11-26")
    # Norgate has no session on Thanksgiving, so observation T-1 session (Wednesday) is age 0.
    assert tfi.observation_age_sessions_int(obs, T, sessions) == 0
    y.loc[pd.Timestamp("2014-11-26"), "DAAA"] = np.nan  # a Moody's miss like 2014-11-28 in ALFRED
    obs2 = tfi.select_publication_safe_observation_date(T, y, sessions)
    assert obs2 == pd.Timestamp("2014-11-25")
    assert tfi.observation_age_sessions_int(obs2, T, sessions) == 1  # not stale (limit 2)


def test_cash_accrual_on_session_t_uses_observation_no_later_than_t_minus_2():
    sessions = pd.bdate_range("2020-03-02", "2020-03-13")
    dgs3mo = pd.Series(1.0, index=pd.bdate_range("2020-02-24", "2020-03-13"))
    spike_day = sessions[5]
    dgs3mo.loc[spike_day] = 50.0  # a large value dated T-1 for session sessions[6]
    ret = tfi.build_causal_cash_return_ser(sessions, dgs3mo)
    days = (sessions[6] - sessions[5]).days
    assert ret.loc[sessions[6]] == np.float64(0.01 * days / 365.0)  # still the old value
    days7 = (sessions[7] - sessions[6]).days
    assert ret.loc[sessions[7]] == np.float64(0.50 * days7 / 365.0)  # the spike arrives two sessions later
