"""MR capsule: VIX stress gate and SPMO parking weight (strategies/mr_capsule/vix_stress_gate.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.mr_capsule import vix_stress_gate as gate_mod

REPO_PATH = Path(__file__).resolve().parents[1]


def _vix_ser(value_list, start_str="2000-01-03"):
    return pd.Series(value_list, index=pd.bdate_range(start_str, periods=len(value_list)), dtype=float)


def test_threshold_is_expanding_mean_of_closes_up_to_t():
    vix_ser = _vix_ser([10.0, 20.0, 30.0, 40.0])
    threshold_ser = gate_mod.vix_threshold_ser(vix_ser, min_history_session_int=2)
    assert np.isnan(threshold_ser.iloc[0])
    assert threshold_ser.iloc[1:].tolist() == pytest.approx([15.0, 20.0, 25.0])


def test_threshold_has_no_lookahead():
    rng = np.random.default_rng(0)
    vix_ser = _vix_ser(15 + 5 * rng.random(800))
    full_ser = gate_mod.vix_threshold_ser(vix_ser)
    for cut_int in (600, 700, 799):
        prefix_ser = gate_mod.vix_threshold_ser(vix_ser.iloc[:cut_int])
        pd.testing.assert_series_equal(full_ser.iloc[:cut_int], prefix_ser)


def test_gate_opens_above_threshold_and_holds_memory_sessions():
    # 3 closes of history at 10 (threshold 10), then a spike, then calm again.
    value_list = [10.0, 10.0, 10.0, 30.0] + [5.0] * 8
    gate_ser = gate_mod.stress_gate_open_ser(_vix_ser(value_list), memory_session_int=3, min_history_session_int=3)
    # opens at the spike close (row 3); held count 1, 2 at rows 4, 5 (below threshold but memory not met);
    # closes at row 6 (third session after the opening, still below threshold).
    assert gate_ser.tolist() == [False, False, False, True, True, True, False, False, False, False, False, False]


def test_gate_stays_open_while_vix_above_threshold():
    value_list = [10.0, 10.0, 10.0] + [50.0] * 10 + [1.0]
    gate_ser = gate_mod.stress_gate_open_ser(_vix_ser(value_list), memory_session_int=2, min_history_session_int=3)
    assert gate_ser.iloc[3:13].all()
    assert not gate_ser.iloc[13]


def test_gate_carries_state_over_missing_vix():
    value_list = [10.0, 10.0, 10.0, 30.0, np.nan, np.nan, 5.0, 5.0, 5.0]
    gate_ser = gate_mod.stress_gate_open_ser(_vix_ser(value_list), memory_session_int=2, min_history_session_int=3)
    assert gate_ser.iloc[3] and gate_ser.iloc[4] and gate_ser.iloc[5]  # NaN rows: no transition, no memory count
    assert gate_ser.iloc[6]  # held 1 < 2
    assert not gate_ser.iloc[7]  # held 2: closes


def test_gate_has_no_lookahead():
    rng = np.random.default_rng(1)
    vix_ser = _vix_ser(12 + 20 * rng.random(900))
    full_ser = gate_mod.stress_gate_open_ser(vix_ser)
    for cut_int in (600, 750, 899):
        pd.testing.assert_series_equal(full_ser.iloc[:cut_int], gate_mod.stress_gate_open_ser(vix_ser.iloc[:cut_int]))


def _research_gate(vix_ser):
    sys.path.insert(0, str(REPO_PATH / "scripts" / "research" / "mr_gate_selfcal_20261003"))
    run_selfcal = pytest.importorskip("run_selfcal")
    return np.asarray(run_selfcal.gate_mem(vix_ser, run_selfcal.selfcal_params(vix_ser)[0].to_numpy(), 15), dtype=bool)


def test_gate_matches_the_research_state_machine():
    rng = np.random.default_rng(2)
    vix_ser = _vix_ser(12 + 20 * rng.random(1500))
    build_arr = gate_mod.stress_gate_open_ser(vix_ser).to_numpy()
    opening_int = int(np.sum(build_arr[1:] & ~build_arr[:-1]))
    assert opening_int >= 20 and 0.2 < build_arr[500:].mean() < 0.95  # real episodes, not an all-False path
    assert np.array_equal(build_arr, _research_gate(vix_ser))


def test_gate_closes_exactly_at_the_memory_boundary():
    vix_ser = _vix_ser(np.r_[np.full(600, 20.0), 40.0, np.full(30, 10.0)])
    gate_arr = gate_mod.stress_gate_open_ser(vix_ser).to_numpy()
    assert not gate_arr[:600].any()
    assert gate_arr[600:615].all()  # open on the spike and for the next 14 sessions: 15 sessions in all
    assert not gate_arr[615:].any()  # first close at or below the threshold once 15 sessions have passed
    assert np.array_equal(gate_arr, _research_gate(vix_ser))


def test_gate_state_at_uses_latest_row_on_or_before_decision():
    gate_ser = pd.Series([False, True, False], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    assert gate_mod.gate_state_at(gate_ser, pd.Timestamp("2020-01-03")) is True
    assert gate_mod.gate_state_at(gate_ser, pd.Timestamp("2020-01-05")) is True  # weekend: Friday's state
    assert gate_mod.gate_state_at(gate_ser, pd.Timestamp("2020-01-01")) is False  # before history: closed


def test_spmo_weight_is_vol_target_capped_at_one_and_zero_before_data():
    close_arr = np.r_[np.full(5, np.nan), 100 * np.cumprod(1 + np.r_[0.0, np.tile([0.01, -0.01], 30)])]
    weight_ser = gate_mod.spmo_target_weight_ser(_vix_ser(close_arr))
    assert (weight_ser.iloc[:25] == 0.0).all()  # pre-inception and warm-up: no SPMO
    realised_vol_float = float(pd.Series(close_arr).pct_change(fill_method=None).iloc[-20:].std() * np.sqrt(252))
    assert weight_ser.iloc[-1] == pytest.approx(min(1.0, 0.08 / realised_vol_float))
    calm_weight_ser = gate_mod.spmo_target_weight_ser(_vix_ser(100 * np.cumprod(1 + np.tile([0.0005, -0.0004], 20))))
    assert calm_weight_ser.iloc[-1] == 1.0


def test_spmo_weight_has_no_lookahead():
    rng = np.random.default_rng(3)
    close_ser = _vix_ser(100 * np.cumprod(1 + rng.normal(0, 0.015, 300)))
    full_ser = gate_mod.spmo_target_weight_ser(close_ser)
    pd.testing.assert_series_equal(full_ser.iloc[:200], gate_mod.spmo_target_weight_ser(close_ser.iloc[:200]))


def test_spmo_tradability_guard_needs_a_trade_on_each_of_the_last_20_sessions():
    rng = np.random.default_rng(5)
    close_ser = _vix_ser(100 * np.cumprod(1 + rng.normal(0, 0.01, 120)))
    volume_ser = pd.Series(1_000.0, index=close_ser.index)
    volume_ser.iloc[60] = 0.0  # one session without a trade
    guarded_ser = gate_mod.spmo_target_weight_ser(close_ser, spmo_volume_ser=volume_ser)
    plain_ser = gate_mod.spmo_target_weight_ser(close_ser)
    assert (guarded_ser.iloc[60:80] == 0.0).all()  # the no-trade session sits inside the 20-session window
    pd.testing.assert_series_equal(guarded_ser.iloc[80:], plain_ser.iloc[80:])
    pd.testing.assert_series_equal(guarded_ser.iloc[20:60], plain_ser.iloc[20:60])
    # no lookahead: a later no-trade session does not change earlier weights
    later_volume_ser = volume_ser.copy()
    later_volume_ser.iloc[100] = 0.0
    later_ser = gate_mod.spmo_target_weight_ser(close_ser, spmo_volume_ser=later_volume_ser)
    pd.testing.assert_series_equal(later_ser.iloc[:100], guarded_ser.iloc[:100])


def test_spmo_weight_from_history_equals_the_series_value_at_each_decision():
    rng = np.random.default_rng(7)
    close_ser = _vix_ser(100 * np.cumprod(1 + rng.normal(0, 0.012, 200)))
    volume_ser = pd.Series(1_000.0, index=close_ser.index)
    volume_ser.iloc[120] = 0.0
    series_ser = gate_mod.spmo_target_weight_ser(close_ser, spmo_volume_ser=volume_ser)
    for end_int in (10, 21, 22, 60, 121, 139, 141, 199):
        value_float = gate_mod.spmo_weight_from_history(close_ser.iloc[: end_int + 1], volume_ser.iloc[: end_int + 1])
        assert value_float == pytest.approx(float(series_ser.iloc[end_int]), abs=1e-12)
