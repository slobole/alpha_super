"""Strategy readiness audit 2026-09-28, CORE5: fast synthetic regression locks for the audited properties.

Real-data evidence lives in results/research/strategy_readiness_audit_20260928/core5/. These tests only lock the
code properties the audit relied on: scale-free signals, causal prefixes, calendar month-end on partial data,
the target-book contract, the LIVE-mode block, fail-closed session gaps, and that the harness catches a leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.live import core5_adapter as adapter_module, scheduler_utils
from alpha.live.models import LiveRelease, PodState
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5


def _sessions(start_str="2018-01-02", end_str="2021-06-30"):
    return pd.DatetimeIndex(scheduler_utils.get_exchange_calendar_obj("XNYS").sessions_in_range(start_str, end_str))


def _synthetic_close(n_int, seed_int=7):
    rng = np.random.default_rng(seed_int)
    return 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_int)))


@pytest.fixture
def price_df():
    idx = _sessions()
    data = {}
    for i, asset in enumerate(adapter_module.CORE5_ASSET_TUPLE):
        close = _synthetic_close(len(idx), 11 + i)
        for field, mult in (("Open", 1.001), ("High", 1.004), ("Low", 0.996), ("Close", 1.0)):
            data[(asset, field)] = close * mult
        data[(asset, "Dividend")] = np.zeros(len(idx))
        if asset != "BIL":
            data[(core5.signal_namespace_str(asset), "Close")] = close * 1.2
    return pd.DataFrame(data, index=idx)


def _states(pdf):
    sig = core5.AdaptiveMacroCore5Strategy().compute_signals(pdf)
    cols = [(core5.signal_namespace_str(a), f) for a in core5.RISK_ASSET_TUPLE for f in ("long_state_ser", "short_state_ser")]
    cols += [(core5.PORTFOLIO_NAMESPACE_STR, core5.MONTH_END_REBALANCE_FIELD_STR)]
    return sig.loc[:, cols].astype(float)


@pytest.mark.parametrize("k", [40.0, 0.1, 1.5])
def test_signal_states_invariant_to_future_split(price_df, k):
    base = _states(price_df)
    scaled = price_df.copy()
    for field in ("Open", "High", "Low", "Close"):
        scaled[("DBC", field)] = scaled[("DBC", field)] / k
    scaled[(core5.signal_namespace_str("DBC"), "Close")] = scaled[(core5.signal_namespace_str("DBC"), "Close")] / k
    assert base.equals(_states(scaled))


@pytest.mark.parametrize("cut_str", ["2019-08-30", "2020-05-29", "2020-03-16", "2021-05-27", "2021-06-15"])
def test_signal_prefix_matches_full_history(price_df, cut_str):
    cut = pd.Timestamp(cut_str)
    full = _states(price_df).loc[:cut]
    pre = _states(price_df.loc[:cut])
    assert full.equals(pre)


def test_partial_month_last_row_is_not_month_end(price_df):
    pre = price_df.loc[:"2021-05-27"]
    flag = core5._month_end_rebalance_ser(pre.index)
    assert not bool(flag.iloc[-1])
    assert bool(core5._month_end_rebalance_ser(price_df.index).loc["2021-05-28"])


def test_positive_control_prefix_harness_catches_close_t_plus_1(price_df, monkeypatch):
    orig = core5.compute_adaptive_asset_signal_df

    def leak(ser, config_obj=core5.DEFAULT_CONFIG):
        return orig(pd.Series(ser, copy=True).astype(float).shift(-1), config_obj)

    monkeypatch.setattr(core5, "compute_adaptive_asset_signal_df", leak)
    cut = pd.Timestamp("2020-03-16")
    assert not _states(price_df).loc[:cut].equals(_states(price_df.loc[:cut]))


def test_target_book_contract_short_cap_and_restricted_cash():
    long_state = pd.Series({"SPY": 1.0, "IEF": 0.0, "GLD": 1.0, "DBC": 0.0, "UUP": 0.0})
    w = core5.build_target_weight_ser(long_state, True, 0.10)
    assert w["DBC"] == pytest.approx(-0.10)  # 2.5% / 10% vol = 25% -> capped at 10%
    assert w["BIL"] == pytest.approx(0.60)
    assert w["Cash"] == pytest.approx(0.10)
    assert w.sum() == pytest.approx(1.0)
    w2 = core5.build_target_weight_ser(long_state, True, 0.50)
    assert w2["DBC"] == pytest.approx(-0.05)


def _release(mode_str):
    return LiveRelease(
        release_id_str="core5.audit.v1", user_id_str="audit", pod_id_str="core5_audit",
        account_route_str="SIM_CORE5_AUDIT", strategy_import_str=adapter_module.CORE5_STRATEGY_IMPORT_STR,
        mode_str=mode_str, session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str="norgate_eod_core5", params_dict={},
        risk_profile_str="audit", enabled_bool=False, source_path_str="in_memory",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
    )


def test_live_mode_is_blocked_incubation_allowed():
    with pytest.raises(ValueError, match="LIVE"):
        adapter_module.validate_core5_release(_release("live"))
    adapter_module.validate_core5_release(_release("incubation"))


def test_missed_session_fails_closed(price_df):
    rel = _release("incubation")
    T = pd.Timestamp("2021-05-27")
    as_of = (T.tz_localize("America/New_York") + pd.Timedelta(hours=18)).to_pydatetime()
    stale_prev = "2021-05-25"  # two sessions back: 2021-05-26 was never committed
    state = {
        "core5_state_version_int": 1, "initialized_bool": True, "last_signal_date_str": stale_prev,
        "last_long_state_map_dict": {a: 0 for a in core5.RISK_ASSET_TUPLE},
        "last_target_weight_map_dict": {"SPY": 0.0, "IEF": 0.0, "GLD": 0.0, "DBC": 0.0, "UUP": 0.0, "BIL": 1.0, "Cash": 0.0},
        "last_rebalance_date_str": stale_prev,
    }
    pod = PodState(rel.pod_id_str, rel.user_id_str, rel.account_route_str, {"BIL": 100.0}, 1_000.0, 10_000.0, state, as_of,
                   snapshot_stage_str="eod", snapshot_source_str="virtual_broker")
    with pytest.raises(ValueError, match="missed decision session"):
        adapter_module.build_core5_decision_from_prices(rel, as_of, pod, price_df, {
            "norgate_snapshot_date_str": T.date().isoformat(), "norgate_data_profile_str": "norgate_eod_core5",
            "norgate_manifest_hash_str": "audit"})


def test_intraday_rerun_of_committed_session_fails_closed(price_df):
    # Invoked at 15:00 on 2021-05-28 the latest completed session is 2021-05-27; memory already committed it.
    rel = _release("incubation")
    T = pd.Timestamp("2021-05-27")
    as_of = (pd.Timestamp("2021-05-28").tz_localize("America/New_York") + pd.Timedelta(hours=15)).to_pydatetime()
    assert scheduler_utils.get_latest_completed_session_label_ts(as_of, "XNYS").date() == T.date()
    state = {
        "core5_state_version_int": 1, "initialized_bool": True, "last_signal_date_str": T.date().isoformat(),
        "last_long_state_map_dict": {a: 0 for a in core5.RISK_ASSET_TUPLE},
        "last_target_weight_map_dict": {"SPY": 0.0, "IEF": 0.0, "GLD": 0.0, "DBC": 0.0, "UUP": 0.0, "BIL": 1.0, "Cash": 0.0},
        "last_rebalance_date_str": T.date().isoformat(),
    }
    stamp = (T.tz_localize("America/New_York") + pd.Timedelta(hours=18)).to_pydatetime()
    pod = PodState(rel.pod_id_str, rel.user_id_str, rel.account_route_str, {"BIL": 100.0}, 1_000.0, 10_000.0, state, stamp,
                   snapshot_stage_str="eod", snapshot_source_str="virtual_broker")
    with pytest.raises(ValueError, match="duplicate or missed"):
        adapter_module.build_core5_decision_from_prices(rel, as_of, pod, price_df, {
            "norgate_snapshot_date_str": T.date().isoformat(), "norgate_data_profile_str": "norgate_eod_core5",
            "norgate_manifest_hash_str": "audit"})
