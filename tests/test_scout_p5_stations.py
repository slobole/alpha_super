"""Scout P5: S4 strategy build, S5 gate and diagnostics, S6 book value, the card (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.card import grade_str, render_card
from alpha.scout.engines.weights import CostModel, simulate
from alpha.scout.family import FamilyRunner
from alpha.scout.searches import _hold_daily
from alpha.scout.specs.taa_3x import offset_decision_index
from alpha.scout.stations.s4_strategy import run_s4
from alpha.scout.stations.s5_overfit import McptComponent, run_s5
from alpha.scout.stations.s6_book import (
    _monthly,
    capacity,
    diversification,
    finish_s6,
    spanning_table,
    tbill_slot_test,
)

DATE_INDEX = pd.bdate_range("2012-01-02", "2022-12-30")


def _synthetic_family() -> FamilyRunner:
    """Two assets; hold asset A when its L-day return is positive (else B), weight w. A trends mildly."""
    rng_obj = np.random.default_rng(0)
    trend_vec = np.zeros(len(DATE_INDEX))
    for t_int in range(1, len(DATE_INDEX)):
        trend_vec[t_int] = 0.97 * trend_vec[t_int - 1] + rng_obj.normal(0, 0.0004)
    return_mat = np.column_stack([0.0003 + trend_vec + rng_obj.normal(0, 0.01, len(DATE_INDEX)), 0.0001 + rng_obj.normal(0, 0.004, len(DATE_INDEX))])
    close_df = pd.DataFrame(100 * np.cumprod(1 + return_mat, axis=0), index=DATE_INDEX, columns=["A", "B"])
    open_df = close_df.shift(1).fillna(close_df.iloc[0])
    dividend_df = close_df * 0.0

    def simulate_fn(config_dict, cost_model, capital_float):
        offset_int = config_dict.get("decision_offset_int", 0)
        decision_index = offset_decision_index(DATE_INDEX, offset_int)
        trailing_ser = close_df["A"] / close_df["A"].shift(config_dict["lookback_int"]) - 1.0
        row_dict = {}
        for decision_ts in decision_index[:-1]:
            position_int = DATE_INDEX.get_loc(decision_ts)
            weight_ser = pd.Series(0.0, index=["A", "B"])
            weight_ser["A" if trailing_ser.loc[decision_ts] > 0 else "B"] = config_dict["weight_float"]
            row_dict[DATE_INDEX[position_int + 1]] = weight_ser
        weight_df = pd.DataFrame(row_dict).T
        return simulate(open_df, close_df, dividend_df, weight_df, start_date=weight_df.index[0], capital_float=capital_float, cost_model=cost_model)

    return FamilyRunner("synthetic", "time_series_trend_and_breakout", {"lookback_int": (21, 63, 126), "weight_float": (0.5, 1.0)},
                        {"lookback_int": 63, "weight_float": 1.0}, simulate_fn, offset_count_int=3)


def test_offset_decision_index_steps_back_from_month_end():
    index = pd.bdate_range("2020-01-01", "2020-03-31")
    np.testing.assert_array_equal(offset_decision_index(index, 0), pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"]))
    np.testing.assert_array_equal(offset_decision_index(index, 2), pd.DatetimeIndex(["2020-01-29", "2020-02-26", "2020-03-27"]))


def test_hold_daily_earns_from_the_second_session_after_the_decision():
    return_mat = np.zeros((10, 1))
    return_mat[:, 0] = np.arange(10) / 100.0
    daily_vec = _hold_daily(np.array([[1.0], [0.0]]), np.array([2, 6]), return_mat)
    np.testing.assert_allclose(daily_vec, [0, 0, 0, 0, 0.04, 0.05, 0.06, 0.07, 0, 0])


def test_mcpt_component_gate_and_marginal_band():
    def component(p_float, ranking_bool):
        return McptComponent("x", "per-asset", "s", 1.0, np.zeros(10), p_float, ranking_family_bool=ranking_bool)

    assert component(0.01, True).verdict_str == "PASS"
    assert component(0.04, True).verdict_str == "PASS (marginal)"
    assert component(0.04, False).verdict_str == "PASS"
    assert component(0.06, False).verdict_str == "FAIL"


def test_spanning_recovers_a_planted_alpha_and_beta():
    rng_obj = np.random.default_rng(1)
    factor_df = pd.DataFrame({"M": rng_obj.normal(0.0004, 0.01, len(DATE_INDEX))}, index=DATE_INDEX)
    pod_ser = pd.Series(0.0004 + 0.5 * factor_df["M"].to_numpy() + rng_obj.normal(0, 0.002, len(DATE_INDEX)), index=DATE_INDEX)
    row = spanning_table(pod_ser, pod_ser + 0.0001, factor_df, pd.Series(0.0, index=DATE_INDEX), {"market": ["M"]})[0]
    assert row["net_alpha_annual_float"] == pytest.approx(1.0004**252 - 1, abs=0.03)
    assert row["beta_dict"]["M"] == pytest.approx(0.5, abs=0.05) and row["net_alpha_t_float"] > 3
    assert row["gross_alpha_annual_float"] > row["net_alpha_annual_float"]


def test_spanning_drops_months_before_a_factor_exists():
    rng_obj = np.random.default_rng(3)
    late_ser = pd.Series(rng_obj.normal(0.0004, 0.01, len(DATE_INDEX)), index=DATE_INDEX)
    late_ser.loc[:"2016-12-31"] = np.nan  # this factor starts in 2017
    assert np.isnan(_monthly(late_ser).loc["2015-06-30"])
    factor_df = pd.DataFrame({"LATE": late_ser}, index=DATE_INDEX)
    pod_ser = pd.Series(rng_obj.normal(0.0003, 0.005, len(DATE_INDEX)), index=DATE_INDEX)
    row = spanning_table(pod_ser, pod_ser, factor_df, pd.Series(0.0, index=DATE_INDEX), {"late": ["LATE"]})[0]
    assert row["months_int"] == 72  # 2017-2022 only, not 132 months with fake 0% factor returns


def test_grade_follows_d22():
    from types import SimpleNamespace

    def bundle(stress_fail_bool, mcpt_verdict_str):
        s4 = SimpleNamespace(cost_dict={"live": {"fail_bool": stress_fail_bool}}, check_list=[("plateau", "PASS", "")])
        return {"s4": s4, "s5": SimpleNamespace(check_list=[("MCPT", mcpt_verdict_str, "")]), "s6": SimpleNamespace(check_list=[("slot", "PASS", "")])}

    assert grade_str(bundle(False, "PASS")) == "CANDIDATE (S3 pending)"
    assert grade_str(bundle(False, "PASS (marginal)")) == "CANDIDATE (S3 pending)"
    assert grade_str(bundle(False, "FAIL")) == "WATCHLIST"
    assert grade_str(bundle(True, "PASS")) == "REJECTED"


@pytest.mark.skipif("not _norgate_ready()")
def test_taa_replica_tracks_the_engine():
    from alpha.scout import searches
    from alpha.scout.family import taa_3x_family
    from alpha.scout.specs import taa_3x
    from data.norgate_loader import load_price_timeseries

    inputs = taa_3x.load_inputs()
    family = taa_3x_family(inputs)
    tqqq_ser = load_price_timeseries("TQQQ", adjustment_str="TOTALRETURN", start_date_str=taa_3x.START_DATE_STR)["Close"]
    matrix = searches.taa_matrix(inputs.open_df.index, inputs.total_return_close_df.assign(TQQQ=tqqq_ser), inputs.spy_close_ser, inputs.vix_close_ser, inputs.dtb3_ser)
    fast_vec = searches.taa_config_daily_list(matrix, inputs.open_df.index, [family.live_config_dict])[0]
    engine_ser = family.run_config(family.live_config_dict).daily_return_ser.loc["2013-01-01":"2022-12-30"]
    assert np.corrcoef(engine_ser, pd.Series(fast_vec, index=inputs.open_df.index).reindex(engine_ser.index))[0, 1] > 0.98


def _norgate_ready() -> bool:
    try:
        import norgatedata

        return bool(norgatedata.status())
    except Exception:  # noqa: BLE001
        return False


def test_capacity_scales_with_volume():
    trade_df = pd.DataFrame({"date": [DATE_INDEX[100]], "asset": ["A"], "delta_float": [100.0], "price_float": [10.0], "fee_float": [1.0], "kind_str": ["rebalance"]})
    value_ser = pd.Series(10_000.0, index=DATE_INDEX)  # the order is 10% of the account
    volume_df = pd.DataFrame({"A": 1_000_000.0}, index=DATE_INDEX)  # $1M a day
    assert capacity(trade_df, value_ser, volume_df)["recent_3y_aum_float"] == pytest.approx(0.01 * 1_000_000 / 0.1)


def test_s4_to_card_end_to_end():
    family = _synthetic_family()
    s4 = run_s4(family, cost_model=CostModel())
    assert s4.sharpe_ser.size == 6 and s4.live_dict["label_str"] == "lookback_int=63|weight_float=1.0"
    assert set(s4.luck_dict["live"]["sharpe_by_offset"]) == {0, 1, 2}
    assert s4.cost_dict["live"]["stressed_sharpe_float"] < s4.cost_dict["live"]["net_sharpe_float"] < s4.cost_dict["live"]["gross_sharpe_float"]
    in_sample_df = s4.grid_df.loc[:"2022-12-30"]
    components = [McptComponent("whole strategy", "date shuffle", "SD", 0.5, np.linspace(-1, 1, 200), 0.03)]
    s5 = run_s5(in_sample_df, family.grid_shape_tuple, s4.chosen_label_str, s4.live_label_str, components, prior_trial_count_int=10)
    assert s5.check_list[0][1] == "PASS" and 0 <= s5.pbo_dict["pbo_float"] <= 1
    live_ser = s4.grid_df[s4.live_label_str]
    other_ser = pd.Series(np.random.default_rng(2).normal(0.0002, 0.005, len(DATE_INDEX)), index=DATE_INDEX)
    factor_df = pd.DataFrame({"M": other_ser * 2}, index=DATE_INDEX)
    tbill_ser = pd.Series(0.00005, index=DATE_INDEX)
    s6 = finish_s6(
        spanning_table(live_ser, live_ser, factor_df, tbill_ser, {"market": ["M"]}),
        tbill_slot_test(live_ser, {"synthetic": 0.5, "other": 0.5}, {"other": other_ser}, "synthetic", tbill_ser, draw_count_int=500),
        diversification(live_ser, {"other": other_ser}, 0.5 * live_ser + 0.5 * other_ser),
        {"participation_float": 0.01, "full_history_aum_float": 5e6, "recent_3y_aum_float": 4e6, "binding_asset_str": "A", "trade_count_int": 10},
    )
    bundle = {"pod_str": "synthetic", "family": {"grid": family.param_grid_dict, "live": family.live_config_dict, "family_id_str": family.family_id_str},
              "prior_trial_count_int": 10, "s4": s4, "s5": s5, "s6": s6,
              "post_adoption": {"adoption_date_str": "2022-06-01", "sessions_int": 150, "performance": {}, "min_track_record_months_float": 60.0}}
    html_str = render_card(bundle)
    assert grade_str(bundle) in {"CANDIDATE", "WATCHLIST", "REJECTED"}
    assert "Verdict strip" in html_str and "MCPT" in html_str and "data:image/png;base64" in html_str
