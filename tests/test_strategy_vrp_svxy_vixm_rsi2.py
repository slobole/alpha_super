"""Rule and causality tests for strategies/vix_stuff/strategy_vrp_svxy_vixm_rsi2.py (synthetic data, no Norgate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.vix_stuff import strategy_vrp_svxy_vixm_rsi2 as m


def _signal_df(n_int: int = 120, seed_int: int = 7, start_str: str = "2019-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed_int)
    idx = pd.bdate_range(start_str, periods=n_int)
    spy = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n_int))
    vix = np.clip(16.0 + np.cumsum(rng.normal(0.0, 1.0, n_int)), 9.0, 60.0)
    vix3m = np.clip(vix * rng.uniform(0.9, 1.08, n_int), 9.0, 60.0)
    spx = 3000.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n_int))
    return pd.DataFrame({"vix_close": vix, "vix3m_close": vix3m, "spx_close": spx, "spy_tr_close": spy}, index=idx)


def _one_row(vix_float: float, vix3m_float: float, erv_target_float: float, rsi_low_bool: bool, date_str: str = "2020-06-01") -> pd.DataFrame:
    """Twelve sessions whose last row has the given VIX, VIX3M and roughly the given 10-day realised volatility."""
    idx = pd.bdate_range(end=date_str, periods=12)
    daily_float = erv_target_float / 100.0 / np.sqrt(252.0)
    spy = 100.0 * np.cumprod(1.0 + np.array([0.0] + [daily_float if i % 2 else -daily_float for i in range(11)]))
    spx = np.linspace(3000.0, 3100.0, 12) if not rsi_low_bool else np.linspace(3100.0, 2900.0, 12)
    return pd.DataFrame({"vix_close": vix_float, "vix3m_close": vix3m_float, "spx_close": spx, "spy_tr_close": spy}, index=idx)


@pytest.mark.parametrize("vix,vix3m,erv,rsi_low,regime,svxy,vixm", [
    (20.0, 22.0, 10.0, False, 1, 0.40, 0.0),           # contango, premium positive: 2 x VIX / 100
    (20.0, 22.0, 10.0, True, 1, 0.50, 0.0),            # RSI2 below 25: x1.25
    (20.0, 22.0, 40.0, False, 2, 0.20, 0.0),           # contango, premium negative: 1 x VIX / 100
    (20.0, 22.0, 40.0, True, 2, 0.30, 0.0),            # RSI2 below 20: 1.5 x VIX / 100
    (20.0, 20.2, 10.0, False, 3, 0.0, 0.0),            # ratio 0.990: the 0.98-1.00 buffer is cash
    (30.0, 25.0, 50.0, False, 4, 0.0, 0.375),          # inverted, premium negative: VIXM 1.25 x VIX / 100
    (30.0, 25.0, 10.0, False, 5, 0.0, 0.0),            # inverted, premium positive: cash
    (60.0, 70.0, 10.0, True, 1, 1.0, 0.0),             # caps at 1
])
def test_regimes(vix, vix3m, erv, rsi_low, regime, svxy, vixm):
    feature_df = m.compute_feature_df(_one_row(vix, vix3m, erv, rsi_low))
    last = feature_df.iloc[-1]
    assert last["regime_int"] == regime
    assert last["target_SVXY"] == pytest.approx(svxy, abs=1e-12)
    assert last["target_VIXM"] == pytest.approx(vixm, abs=1e-12)


def test_svxy_halved_before_cutover_only():
    before = m.compute_feature_df(_one_row(20.0, 22.0, 10.0, False, "2018-02-27")).iloc[-1]
    after = m.compute_feature_df(_one_row(20.0, 22.0, 10.0, False, "2018-02-28")).iloc[-1]
    assert before["target_SVXY"] == pytest.approx(0.20) and after["target_SVXY"] == pytest.approx(0.40)


def test_threshold_edges_are_strict():
    assert 49.0 / 50.0 == 0.98
    at_contango = m.compute_feature_df(_one_row(49.0, 50.0, 10.0, False)).iloc[-1]  # ratio exactly 0.98: not contango
    at_one = m.compute_feature_df(_one_row(20.0, 20.0, 50.0, False)).iloc[-1]  # ratio exactly 1.00: not inverted
    assert at_contango["regime_int"] == 3 and at_one["regime_int"] == 3


def test_band_trades_on_switch_and_five_points_only():
    target_df = pd.DataFrame({"a": [np.nan, 0.30, 0.33, 0.349, 0.351, 0.0, 0.0, 0.02], "b": 0.0})
    assert m.rebalance_flag_ser(target_df, 0.05).tolist() == [False, True, False, False, True, True, False, True]


def test_targets_are_causal_under_truncation():
    signal_df = _signal_df(160)
    full_df = m.compute_feature_df(signal_df)
    for cut_int in (40, 80, 120, 159):
        part_df = m.compute_feature_df(signal_df.iloc[:cut_int])
        pd.testing.assert_frame_equal(full_df.iloc[:cut_int], part_df, check_exact=True)


def test_first_decision_starts_flat():
    signal_df = _signal_df(80)
    first_ts = signal_df.index[40]
    feature_df = m.compute_feature_df(signal_df, first_decision_date=first_ts)
    assert feature_df.loc[: signal_df.index[39], "target_SVXY"].isna().all()
    first_valid = feature_df.loc[first_ts]
    assert bool(first_valid["rebalance_bool"]) == bool(first_valid[["target_SVXY", "target_VIXM"]].sum() > 0)


def test_missing_input_gives_nan_targets():
    signal_df = _signal_df(60)
    signal_df.iloc[50, signal_df.columns.get_loc("vix3m_close")] = np.nan
    feature_df = m.compute_feature_df(signal_df)
    assert np.isnan(feature_df.iloc[50]["target_SVXY"]) and not feature_df.iloc[50]["rebalance_bool"]


def test_config_guards():
    with pytest.raises(ValueError):
        m.VrpRsi2Config(contango_threshold_float=1.01)
    with pytest.raises(ValueError):
        m.VrpRsi2Config(capital_base_float=0.0)


def test_band_on_vixm_alone_direct_switch_and_exact_five_points():
    target_df = pd.DataFrame({"svxy": [0.30, 0.30, 0.0, 0.0, 0.0], "vixm": [0.0, 0.0, 0.40, 0.4375, 0.50]})
    # SVXY on; unchanged; direct switch to VIXM; VIXM +3.75 points (no trade); VIXM +10 points from the traded 0.40.
    assert m.rebalance_flag_ser(target_df, 0.05).tolist() == [True, False, True, False, True]
    exact_df = pd.DataFrame({"svxy": [0.25, 0.50], "vixm": 0.0})
    assert m.rebalance_flag_ser(exact_df, 0.25).tolist() == [True, True]  # a move equal to the band trades


@pytest.mark.parametrize("invalid_close_float", [np.nan, np.inf, -np.inf, 0.0, -1.0])
@pytest.mark.parametrize("invalid_position_int", [0, 30, 59])
def test_invalid_spx_close_stops_feature_computation(invalid_close_float, invalid_position_int):
    signal_df = _signal_df(60)
    invalid_date_ts = signal_df.index[invalid_position_int]
    signal_df.loc[invalid_date_ts, "spx_close"] = invalid_close_float
    with pytest.raises(ValueError, match=f"Invalid SPX close on {invalid_date_ts.date()}"):
        m.compute_feature_df(signal_df, first_decision_date=signal_df.index[40])


def test_valid_zero_loss_rsi_retains_frozen_fallback():
    close_ser = pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2020-01-02", periods=3))
    assert m.wilder_rsi_ser(close_ser, 2).tolist() == [50.0, 50.0, 50.0]


def _close_row_ser(svxy_float: float, vixm_float: float, rebalance_bool: bool, close_float: float = 50.0) -> pd.Series:
    row_dict = {(m.SIGNAL_NAMESPACE_STR, "target_SVXY"): svxy_float, (m.SIGNAL_NAMESPACE_STR, "target_VIXM"): vixm_float,
                (m.SIGNAL_NAMESPACE_STR, "rebalance_bool"): rebalance_bool}
    row_dict.update({(asset_str, "Close"): close_float for asset_str in m.TRADED_ASSET_TUPLE})
    return pd.Series(row_dict)


def _order_amount_dict(strategy_obj) -> dict[str, float]:
    return {order_obj.asset: float(order_obj.amount) for order_obj in strategy_obj.get_orders()}


def test_late_start_trades_to_target_although_the_precomputed_band_is_off():
    strategy_obj = m.VrpRsi2Strategy()
    strategy_obj.previous_bar, strategy_obj.current_bar = pd.Timestamp("2020-02-03"), pd.Timestamp("2020-02-04")
    strategy_obj.iterate(pd.DataFrame(), _close_row_ser(0.40, 0.0, rebalance_bool=False), pd.Series(dtype=float))
    assert _order_amount_dict(strategy_obj) == {"SVXY": 800.0}  # 100,000 x 0.40 / 50


def test_timing_strategy_orders_the_change_from_the_last_traded_shares():
    strategy_obj = m.VrpRsi2TimingStrategy()
    strategy_obj.previous_bar, strategy_obj.current_bar = pd.Timestamp("2020-02-03"), pd.Timestamp("2020-02-04")
    strategy_obj.iterate(pd.DataFrame(), _close_row_ser(0.40, 0.0, True), pd.Series(dtype=float))
    strategy_obj.clear_orders()
    # The 800-share fill is still pending (holding 0): the next decision orders only the 200-share change.
    strategy_obj.iterate(pd.DataFrame(), _close_row_ser(0.50, 0.0, True), pd.Series(dtype=float))
    assert _order_amount_dict(strategy_obj) == {"SVXY": 200.0}


def test_vanilla_strategy_refuses_a_lost_fill():
    strategy_obj = m.VrpRsi2Strategy()
    strategy_obj.previous_bar, strategy_obj.current_bar = pd.Timestamp("2020-02-03"), pd.Timestamp("2020-02-04")
    strategy_obj.iterate(pd.DataFrame(), _close_row_ser(0.40, 0.0, True), pd.Series(dtype=float))
    with pytest.raises(RuntimeError, match="holding differs"):
        strategy_obj.iterate(pd.DataFrame(), _close_row_ser(0.40, 0.0, False), pd.Series(dtype=float))


def test_analysis_hooks_and_stress_key_exist():
    from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP

    for hook_str in ("build_capacity_analysis_inputs", "build_execution_timing_analysis_inputs", "build_stress_test_context_dict", "build_stress_test_strategy_obj"):
        assert callable(getattr(m, hook_str))
    assert SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[m.STRATEGY_NAME_STR].full_history_replay_bool
