"""A15 robustness diagnostics (alpha/scout/stations/robustness.py) and the spec ablation switches they use."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.engines.weights import CostModel, simulate
from alpha.scout.stations.robustness import (
    AblationStep,
    RandomParameterResult,
    ablation_dict,
    contribution_dict,
    paired_sharpe_probability,
    random_parameter_summary,
    timing_dict,
)

NO_COST = CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0)


def _three_asset_result(seed_int: int = 0, star_drift_float: float = 0.004):
    """A buy-and-hold of three assets; asset A drifts up strongly, B and C are noise."""
    date_index = pd.bdate_range("2010-01-01", periods=800)
    rng_obj = np.random.default_rng(seed_int)
    return_mat = rng_obj.normal(0.0, 0.01, size=(len(date_index), 3))
    return_mat[:, 0] += star_drift_float
    close_df = pd.DataFrame(100.0 * np.cumprod(1.0 + return_mat, axis=0), index=date_index, columns=["A", "B", "C"])
    weight_df = pd.DataFrame({"A": [1 / 3], "B": [1 / 3], "C": [1 / 3]}, index=[date_index[1]])
    return simulate(close_df, close_df, close_df * 0.0, weight_df, date_index[1], capital_float=1e6, cost_model=NO_COST)


def test_contribution_adds_up_and_flags_a_one_asset_book():
    result = _three_asset_result()
    contribution = contribution_dict(result, "2010-01-01", "2015-12-31")
    assert contribution["contribution_ser"].index[0] == "A"
    assert contribution["total_contribution_float"] == pytest.approx(result.daily_return_ser.iloc[:].sum())
    assert contribution["pod_kind_str"] == "ETF" and contribution["key_k_int"] == 1
    # Removing A leaves two noise assets: the Sharpe collapses -> CONCENTRATED.
    assert contribution["strip_list"][0]["asset_list"] == ["A"]
    assert contribution["verdict_str"] == "CONCENTRATED"
    # Three equal drifts: no single asset carries the book.
    date_index = pd.bdate_range("2010-01-01", periods=800)
    rng_obj = np.random.default_rng(1)
    close_df = pd.DataFrame(100.0 * np.cumprod(1.0 + rng_obj.normal(0.001, 0.01, size=(800, 3)), axis=0), index=date_index, columns=["A", "B", "C"])
    spread = simulate(close_df, close_df, close_df * 0.0, pd.DataFrame({"A": [1 / 3], "B": [1 / 3], "C": [1 / 3]}, index=[date_index[1]]),
                      date_index[1], capital_float=1e6, cost_model=NO_COST)
    assert contribution_dict(spread, "2010-01-01", "2015-12-31")["verdict_str"] == "SPREAD"


def test_timing_sees_a_call_option_as_convex():
    date_index = pd.bdate_range("2000-01-01", "2019-12-31")
    rng_obj = np.random.default_rng(3)
    month_index = pd.date_range("2000-01-31", "2019-12-31", freq="ME")
    market_month_vec = rng_obj.normal(0.006, 0.045, size=len(month_index))
    # Spread each month's return over its first session so that monthly compounding returns it exactly.
    first_session_index = pd.Series(date_index, index=date_index).groupby(date_index.to_period("M")).min()
    market_ser = pd.Series(0.0, index=date_index)
    market_ser.loc[first_session_index.to_numpy()] = market_month_vec
    noise_vec = rng_obj.normal(0.0, 0.002, size=len(month_index))
    call_ser, linear_ser = pd.Series(0.0, index=date_index), pd.Series(0.0, index=date_index)
    call_ser.loc[first_session_index.to_numpy()] = np.maximum(market_month_vec, 0.0) + noise_vec
    linear_ser.loc[first_session_index.to_numpy()] = 0.5 * market_month_vec + noise_vec
    tbill_ser = pd.Series(0.0, index=date_index)
    call_row = timing_dict(call_ser, {"MKT": market_ser}, tbill_ser, "2000-01-01", "2019-12-31")[0]
    linear_row = timing_dict(linear_ser, {"MKT": market_ser}, tbill_ser, "2000-01-01", "2019-12-31")[0]
    assert call_row["verdict_str"] == "CONVEX" and call_row["down_beta_float"] == pytest.approx(0.0, abs=0.05)
    assert call_row["up_beta_float"] == pytest.approx(1.0, abs=0.05)
    assert linear_row["verdict_str"] == "LINEAR" and linear_row["down_beta_float"] == pytest.approx(0.5, abs=0.05)


def test_ablation_verdicts_and_minimum_spec():
    date_index = pd.bdate_range("2005-01-01", periods=2500)
    rng_obj = np.random.default_rng(5)
    noise_vec = rng_obj.normal(0.0, 0.01, size=len(date_index))
    edge_vec = np.full(len(date_index), 0.0008)

    def run_fn(override_dict: dict) -> pd.Series:
        value_vec = noise_vec + (0.0 if override_dict.get("core") else edge_vec) + (0.0 if override_dict.get("useless") else 0.0)
        return pd.Series(value_vec, index=date_index)

    step_list = [AblationStep("useless", {"useless": True}), AblationStep("core", {"core": True})]
    # A tiny but certain effect (a nearly identical series) is SMALL, not "earns its place".
    tiny_dict = ablation_dict(lambda o: run_fn(o) - (0.00002 if o.get("tiny") else 0.0), [AblationStep("tiny", {"tiny": True})], "2005-01-01", "2015-12-31")
    assert tiny_dict["single_list"][0]["probability_live_better_float"] > 0.95 and tiny_dict["single_list"][0]["verdict_str"] == "SMALL"
    result_dict = ablation_dict(run_fn, step_list, "2005-01-01", "2015-12-31")
    useless_row, core_row = result_dict["single_list"]
    assert useless_row["delta_float"] == pytest.approx(0.0) and useless_row["verdict_str"] == "NEVER BINDS"
    assert core_row["verdict_str"] == "EARNS ITS PLACE" and core_row["probability_live_better_float"] > 0.95
    assert result_dict["min_spec_step_int"] == 1 and result_dict["min_spec_str"] == "useless"
    assert [r["name_str"] for r in result_dict["cumulative_list"]] == ["useless", "useless + core"]
    # A single-drop-only step stays out of the cumulative path.
    only_single = ablation_dict(run_fn, [AblationStep("core", {"core": True}, cumulative_bool=False)], "2005-01-01", "2015-12-31")
    assert only_single["cumulative_list"] == [] and only_single["min_spec_step_int"] == 0


def test_paired_probability_is_symmetric_and_random_summary_rules():
    date_index = pd.bdate_range("2005-01-01", periods=1500)
    rng_obj = np.random.default_rng(7)
    a_ser = pd.Series(rng_obj.normal(0.001, 0.01, size=1500), index=date_index)
    b_ser = pd.Series(rng_obj.normal(0.0, 0.01, size=1500), index=date_index)
    assert paired_sharpe_probability(a_ser, b_ser) + paired_sharpe_probability(b_ser, a_ser) == pytest.approx(1.0, abs=0.01)

    def summary(sharpe_list, live_float):
        return random_parameter_summary(RandomParameterResult(live_float, [{"config": {}, "sharpe_float": v} for v in sharpe_list]))

    assert summary(np.linspace(0.6, 1.2, 50), 1.0)["verdict_str"] == "ROBUST TO VALUES"
    assert summary(np.linspace(-0.4, 0.8, 50), 1.0)["verdict_str"] == "DEPENDS ON VALUES"
    partly = summary(list(np.linspace(0.1, 1.1, 50)) + [float("nan")], 1.0)
    assert partly["verdict_str"] == "PARTLY" and partly["failed_count_int"] == 1 and partly["draw_count_int"] == 51
    assert summary(np.linspace(0.0, 2.0, 101), 1.0)["share_at_or_above_live_float"] == pytest.approx(51 / 101)


# ---------------------------------------------------------------------------------------------- spec switches (real data)
def _norgate_running_bool() -> bool:
    try:
        import norgatedata

        return bool(norgatedata.status())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _norgate_running_bool(), reason="Norgate data not available")
def test_taa_switches_move_only_their_own_weights():
    import dataclasses

    from alpha.scout.specs import taa_3x

    inputs = taa_3x.load_inputs()
    live_df = taa_3x.month_end_weight_df(inputs)
    defensive_list = list(taa_3x.DEFENSIVE_TUPLE)

    def weights(**override_dict):
        return taa_3x.month_end_weight_df(inputs, dataclasses.replace(taa_3x.LIVE_CONFIG, **override_dict))

    assert weights().equals(live_df)
    cash_defensive_df = weights(defensive_hold_str="cash")
    assert (cash_defensive_df[defensive_list] == 0.0).all().all() and cash_defensive_df["TQQQ"].equals(live_df["TQQQ"])
    cash_fallback_df = weights(fallback_hold_str="cash")
    assert (cash_fallback_df["TQQQ"] == 0.0).all() and cash_fallback_df[defensive_list].equals(live_df[defensive_list])
    no_gate_df = weights(vix_gate_bool=False)
    assert (no_gate_df["TQQQ"] >= live_df["TQQQ"] - 1e-12).all() and (no_gate_df["TQQQ"] > live_df["TQQQ"]).any()
    assert no_gate_df.sum(axis=1).round(12).eq(1.0).all()
    assert not weights(cash_hurdle_bool=False).equals(live_df)
    with pytest.raises(ValueError):
        taa_3x.TaaConfig(defensive_hold_str="bonds")


@pytest.mark.skipif(not _norgate_running_bool(), reason="Norgate data not available")
def test_core5_switches():
    import dataclasses

    from alpha.scout.specs import core5

    inputs = core5.load_inputs()
    live_df = core5.rebalance_weight_df(inputs)
    assert core5.rebalance_weight_df(inputs, core5.LIVE_CONFIG).equals(live_df)
    static_df = core5.rebalance_weight_df(inputs, dataclasses.replace(core5.LIVE_CONFIG, trend_rule_bool=False))
    np.testing.assert_allclose(static_df[list(core5.RISK_ASSET_TUPLE)].to_numpy(), 0.2)
    assert static_df.index[0] == live_df.index[0]
    no_short_df = core5.rebalance_weight_df(inputs, dataclasses.replace(core5.LIVE_CONFIG, commodity_short_cap_float=0.0))
    assert (no_short_df["DBC"] >= 0.0).all() and (live_df["DBC"] < 0.0).any()
    fixed_df = core5.rebalance_weight_df(inputs, dataclasses.replace(core5.LIVE_CONFIG, adaptive_speed_bool=False, price_filter_lookback_int=1))
    assert not fixed_df.equals(live_df)
    with pytest.raises(ValueError):
        core5.Core5Config(price_filter_lookback_int=0)



# ---------------------------------------------------------------------------------------------- edge cases (review)
def _hold(weight_dict: dict, drift_dict: dict, asset_count_int: int = 3, seed_int: int = 2):
    date_index = pd.bdate_range("2010-01-01", periods=800)
    columns = [f"S{i:03d}" for i in range(asset_count_int)] if asset_count_int > 3 else ["A", "B", "C"]
    rng_obj = np.random.default_rng(seed_int)
    return_mat = rng_obj.normal(0.0, 0.01, size=(len(date_index), len(columns)))
    for asset_str, drift_float in drift_dict.items():
        return_mat[:, columns.index(asset_str)] += drift_float
    close_df = pd.DataFrame(100.0 * np.cumprod(1.0 + return_mat, axis=0), index=date_index, columns=columns)
    weight_df = pd.DataFrame({c: [weight_dict.get(c, 0.0)] for c in columns}, index=[date_index[1]])
    return simulate(close_df, close_df, close_df * 0.0, weight_df, date_index[1], capital_float=1e7, cost_model=NO_COST)


def test_one_asset_no_trade_and_losing_books():
    one = contribution_dict(_hold({"A": 1.0}, {"A": 0.002}), "2010-01-01", "2015-12-31")
    assert one["verdict_str"] == "CONCENTRATED" and one["strip_list"][0]["asset_list"] == ["A"]
    assert one["top_share_dict"][1] == pytest.approx(1.0)
    assert contribution_dict(_hold({}, {}), "2010-01-01", "2015-12-31")["verdict_str"] == "NO TRADES"
    losing = contribution_dict(_hold({"A": 0.5, "B": 0.5}, {"A": -0.002, "B": -0.002}), "2010-01-01", "2015-12-31")
    assert losing["verdict_str"] == "NO EDGE"


def test_stock_pod_strips_the_top_one_percent():
    weight_dict = {f"S{i:03d}": 1.0 / 150 for i in range(150)}
    star = contribution_dict(_hold(weight_dict, {"S007": 0.02}, asset_count_int=150), "2010-01-01", "2015-12-31")
    assert star["pod_kind_str"] == "stock" and star["key_k_int"] == 2 and 2 in [s["k_int"] for s in star["strip_list"]]
    assert star["verdict_str"] == "CONCENTRATED" and star["contribution_ser"].index[0] == "S007"
    broad = contribution_dict(_hold(weight_dict, {f"S{i:03d}": 0.001 for i in range(150)}, asset_count_int=150), "2010-01-01", "2015-12-31")
    assert broad["verdict_str"] == "SPREAD"


def test_timing_without_down_months_is_insufficient():
    date_index = pd.bdate_range("2010-01-01", "2014-12-31")
    up_ser = pd.Series(0.001, index=date_index)
    zero_ser = pd.Series(0.0, index=date_index)
    assert timing_dict(up_ser, {"MKT": up_ser}, zero_ser, "2010-01-01", "2014-12-31")[0]["verdict_str"] == "INSUFFICIENT"
    empty = timing_dict(up_ser, {"MKT": up_ser.loc["2020":]}, zero_ser, "2010-01-01", "2014-12-31")[0]
    assert empty["verdict_str"] == "INSUFFICIENT" and empty["month_count_int"] == 0


def test_all_draws_failed_first_step_fails_and_a_switch_that_never_binds():
    failed = random_parameter_summary(RandomParameterResult(1.0, [{"config": {}, "sharpe_float": float("nan")}] * 3))
    assert failed["verdict_str"] == "NO VALID DRAWS" and failed["failed_count_int"] == 3
    date_index = pd.bdate_range("2005-01-01", periods=1500)
    noise_vec = np.random.default_rng(9).normal(0.0, 0.01, size=1500)

    def run_fn(override_dict: dict) -> pd.Series:
        return pd.Series(noise_vec + (0.0 if override_dict.get("core") else 0.001), index=date_index)

    result_dict = ablation_dict(run_fn, [AblationStep("core", {"core": True}), AblationStep("idle", {"idle": True})], "2005-01-01", "2012-12-31",
                                alt_run_fn=lambda o: run_fn(o) - 0.0001)
    assert result_dict["min_spec_step_int"] == 0 and result_dict["min_spec_str"].startswith("(none")
    assert result_dict["single_list"][1]["verdict_str"] == "NEVER BINDS"
    assert np.isfinite(result_dict["single_list"][0]["alt_sharpe_float"]) and np.isfinite(result_dict["live_alt_sharpe_float"])


def test_card_renders_the_robustness_section():
    from alpha.scout.card import robustness_html

    result = _hold({f"S{i:03d}": 1.0 / 30 for i in range(30)}, {"S001": 0.002}, asset_count_int=30)
    daily_ser = result.daily_return_ser
    zero_ser = pd.Series(0.0, index=daily_ser.index)
    rob = {
        "contribution": contribution_dict(result, "2010-01-01", "2015-12-31"),
        "timing": timing_dict(daily_ser, {"MKT": daily_ser * 0.5 + 0.0001}, zero_ser, "2010-01-01", "2015-12-31")
                  + timing_dict(daily_ser, {"UP": pd.Series(0.001, index=daily_ser.index)}, zero_ser, "2010-01-01", "2015-12-31"),
        "ablation": ablation_dict(lambda o: daily_ser - (0.0005 if o.get("x") else 0.0), [AblationStep("x", {"x": True}, "a test step")],
                                  "2010-01-01", "2015-12-31"),
        "random": {"summary": random_parameter_summary(RandomParameterResult(1.0, [{"config": {}, "sharpe_float": float("nan")}] * 2)),
                   "sharpe_list": [float("nan")] * 2, "box_str": "test box"},
    }
    page_str = robustness_html(rob)
    assert "Robustness diagnostics" in page_str and "<img" in page_str and "NO VALID DRAWS" in page_str and "INSUFFICIENT" in page_str
    assert rob["contribution"]["verdict_str"] in page_str and "Minimum spec" in page_str


def test_ndx_replica_refuses_an_unknown_ranking():
    from alpha.scout.searches import ndx_selection_daily

    date_index = pd.bdate_range("2020-01-01", periods=60)
    mat = np.ones((60, 2))
    with pytest.raises(ValueError, match="ROC / ATR"):
        ndx_selection_daily(mat, mat, mat, mat, mat, mat.astype(bool), date_index, pd.Series(1.0, index=date_index), [], "none")


@pytest.mark.skipif(not _norgate_running_bool(), reason="Norgate data not available")
def test_ndx_switches():
    import dataclasses

    from alpha.scout.specs import ndx_vxn

    inputs = ndx_vxn.load_inputs()
    live_df = ndx_vxn.rebalance_weight_df(inputs)

    def weights(**override_dict):
        return ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(ndx_vxn.LIVE_CONFIG, **override_dict))

    no_regime_df = weights(regime_filter_bool=False)
    assert no_regime_df.index.equals(live_df.index)
    cash_month_mask = live_df.sum(axis=1) == 0.0
    assert cash_month_mask.any() and (no_regime_df.loc[cash_month_mask].sum(axis=1) > 0).all()
    assert no_regime_df.loc[~cash_month_mask].equals(live_df.loc[~cash_month_mask])
    no_trend_df = weights(stock_trend_filter_bool=False)
    assert no_trend_df.index.equals(live_df.index) and not no_trend_df.equals(live_df)
    roc_df = weights(atr_unit_str="none")
    assert not roc_df.equals(live_df) and ((roc_df > 0).sum(axis=1) <= ndx_vxn.TOP_COUNT_INT).all()


@pytest.mark.skipif(not _norgate_running_bool(), reason="Norgate data not available")
def test_taa_cash_asset_switch_isolates_one_asset():
    import dataclasses

    from alpha.scout.specs import taa_3x

    config = taa_3x.VARIANT_DICT["taa_lin_1n_qqq"].config
    inputs = taa_3x.load_inputs(config=config)
    live_df = taa_3x.month_end_weight_df(inputs, config)
    no_btal_df = taa_3x.month_end_weight_df(inputs, dataclasses.replace(config, cash_asset_tuple=("BTAL",)))
    assert (no_btal_df["BTAL"] == 0.0).all() and (live_df["BTAL"] > 0).any()
    other_list = [c for c in live_df.columns if c != "BTAL"]
    assert no_btal_df[other_list].equals(live_df[other_list])



def test_fast_replicas_refuse_ablation_switches():
    from alpha.scout.searches import refuse_ablation_switches
    from alpha.scout.specs.core5 import Core5Config

    refuse_ablation_switches([{"roc_month_int": 12}, {"trend_rule_bool": True}, Core5Config()])
    for config in ({"vix_gate_bool": False}, {"cash_asset_tuple": ("BTAL",)}, Core5Config(adaptive_speed_bool=False)):
        with pytest.raises(ValueError, match="ablation switch"):
            refuse_ablation_switches([config])


@pytest.mark.skipif(not _norgate_running_bool(), reason="Norgate data not available")
def test_ndx_trend_filter_variants():
    import dataclasses

    from alpha.scout.specs import ndx_vxn

    inputs = ndx_vxn.load_inputs()
    live_df = ndx_vxn.rebalance_weight_df(inputs)

    def weights(**override_dict):
        return ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(ndx_vxn.LIVE_CONFIG, **override_dict))

    # A threshold just below zero is the live filter up to floating-point ties; a crossover and a +10% bar change it.
    assert (weights(trend_threshold_float=-1e-12) - live_df).abs().to_numpy().max() < 1e-12
    strict_df, cross_df = weights(trend_threshold_float=0.10), weights(trend_fast_sma_int=50)
    assert strict_df.index.equals(live_df.index) and not strict_df.equals(live_df) and not cross_df.equals(live_df)
    with pytest.raises(ValueError, match="ablation switch"):
        from alpha.scout.searches import refuse_ablation_switches

        refuse_ablation_switches([{"trend_fast_sma_int": 50}])


def test_cmma_is_causal_scale_free_and_bounded():
    from alpha.scout.specs import ndx_vxn

    date_index = pd.bdate_range("2010-01-01", periods=400)
    rng_obj = np.random.default_rng(11)
    close_df = pd.DataFrame(100.0 * np.exp(np.cumsum(rng_obj.normal(0.0005, 0.02, size=(400, 2)), axis=0)), index=date_index, columns=["A", "B"])
    high_df, low_df = close_df * 1.01, close_df * 0.99
    empty_df = close_df * np.nan
    inputs = ndx_vxn.NdxInputs(close_df, high_df, low_df, close_df, close_df, close_df * 0.0, close_df * 0 + 1, pd.Series(dtype=float))
    full_df = ndx_vxn.cmma_frame(inputs, 50, 100)
    # Causal: truncating the future changes nothing up to the cut.
    cut_inputs = ndx_vxn.NdxInputs(*(f.loc[:date_index[250]] for f in (close_df, high_df, low_df, close_df, close_df)), (close_df * 0.0).loc[:date_index[250]],
                                   (close_df * 0 + 1).loc[:date_index[250]], pd.Series(dtype=float))
    np.testing.assert_allclose(ndx_vxn.cmma_frame(cut_inputs, 50, 100).to_numpy(), full_df.loc[:date_index[250]].to_numpy(), atol=1e-12)
    # A price scale (a back-adjustment factor) cancels; bounded in (-50, 50); NaN until both windows are full.
    scaled_inputs = ndx_vxn.NdxInputs(close_df * 7, high_df * 7, low_df * 7, close_df * 7, close_df * 7, close_df * 0.0, close_df * 0 + 1, pd.Series(dtype=float))
    np.testing.assert_allclose(ndx_vxn.cmma_frame(scaled_inputs, 50, 100).to_numpy(), full_df.to_numpy(), atol=1e-6)
    assert full_df.iloc[:100].isna().all().all() and full_df.iloc[101:].notna().all().all()
    assert full_df.abs().max().max() < 50.0
    assert empty_df.isna().all().all()
    # Hand check at one date: x = (ln C - mean of the previous 50 ln C) / (mean of the last 100 log true ranges * sqrt(51)).
    from scipy.special import ndtr

    t = 300
    log_close_vec = np.log(close_df["A"].to_numpy())
    log_tr_vec = np.maximum(np.log(1.01 / 0.99), np.maximum(np.abs(np.log(close_df["A"] * 1.01).to_numpy()[1:] - log_close_vec[:-1]),
                                                             np.abs(np.log(close_df["A"] * 0.99).to_numpy()[1:] - log_close_vec[:-1])))
    x_float = (log_close_vec[t] - log_close_vec[t - 50:t].mean()) / (log_tr_vec[t - 100:t].mean() * np.sqrt(51.0))
    assert full_df["A"].iloc[t] == pytest.approx(100.0 * ndtr(x_float) - 50.0, abs=1e-9)


def test_linear_trend_matches_ols_and_is_causal():
    from alpha.scout.specs import ndx_vxn

    date_index = pd.bdate_range("2010-01-01", periods=300)
    rng_obj = np.random.default_rng(13)
    close_df = pd.DataFrame(100.0 * np.exp(np.cumsum(rng_obj.normal(0.001, 0.02, size=(300, 2)), axis=0)), index=date_index, columns=["A", "B"])
    close_df.iloc[150, 1] = np.nan  # a hole: every window that holds it is NaN

    def make(c_df):
        return ndx_vxn.NdxInputs(c_df, c_df * 1.01, c_df * 0.99, c_df, c_df, c_df * 0.0, c_df * 0 + 1, pd.Series(dtype=float))

    lt_df, compressed_df = ndx_vxn.linear_trend_frame(make(close_df), 60, 20)
    y_vec = np.log(close_df["A"].to_numpy())
    for t in (59, 120, 299):
        slope_float, _ = np.polyfit(np.arange(60), y_vec[t - 59:t + 1], 1)
        rsq_float = np.corrcoef(np.arange(60), y_vec[t - 59:t + 1])[0, 1] ** 2
        previous_vec = y_vec[t - 20:t]
        tr_vec = np.maximum(np.log(1.01 / 0.99), np.maximum(np.abs(y_vec[t - 19:t + 1] + np.log(1.01) - previous_vec),
                                                             np.abs(y_vec[t - 19:t + 1] + np.log(0.99) - previous_vec)))
        assert lt_df["A"].iloc[t] == pytest.approx(slope_float * 59 / tr_vec.mean() * rsq_float, rel=1e-9)
    assert lt_df["A"].iloc[:59].isna().all() and lt_df["B"].iloc[150:210].isna().all() and np.isfinite(lt_df["B"].iloc[210])
    assert compressed_df.abs().max().max() < 50.0
    # Causal (a truncated future changes nothing) and scale-free (a back-adjustment factor cancels).
    np.testing.assert_allclose(ndx_vxn.linear_trend_frame(make(close_df.iloc[:200]), 60, 20)[0].to_numpy(), lt_df.iloc[:200].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(ndx_vxn.linear_trend_frame(make(close_df * 5.0), 60, 20)[0].to_numpy(), lt_df.to_numpy(), rtol=1e-9)
    no_rsq_df = ndx_vxn.linear_trend_frame(make(close_df), 60, 20, rsq_bool=False)[0]
    assert (no_rsq_df["A"].abs().iloc[59:] >= lt_df["A"].abs().iloc[59:] - 1e-12).all()


def test_romano_wolf_finds_the_one_real_improvement():
    from alpha.scout.stations.robustness import paired_sharpe_difference_draws, romano_wolf_stepdown

    rng_obj = np.random.default_rng(21)
    base_vec = rng_obj.normal(0.0004, 0.01, size=3000)
    variant_mat = np.column_stack([base_vec + rng_obj.normal(0.0, 0.004, size=3000) for _ in range(4)])
    variant_mat[:, 0] += 0.0008  # the only real improvement (about +1.2 Sharpe on the difference)
    observed_vec, draw_mat = paired_sharpe_difference_draws(base_vec, variant_mat, draw_count_int=1000, random_seed_int=3)
    adjusted_vec = romano_wolf_stepdown(observed_vec, draw_mat)
    assert draw_mat.shape == (1000, 4) and observed_vec[0] > 0.5
    assert adjusted_vec[0] < 0.01 and (adjusted_vec[1:] > 0.10).all()
    # Monotone in t, and a variant identical to the base gets p = 1.
    order_vec = np.argsort(-(observed_vec / draw_mat.std(axis=0, ddof=1)))
    assert (np.diff(adjusted_vec[order_vec]) >= -1e-12).all()
    same_observed_vec, same_draw_mat = paired_sharpe_difference_draws(base_vec, np.column_stack([base_vec, variant_mat[:, 0]]), draw_count_int=200)
    assert romano_wolf_stepdown(same_observed_vec, same_draw_mat)[0] == 1.0
