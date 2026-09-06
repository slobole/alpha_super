"""Frozen allocation accounting and PM capital forwarding regressions."""
from unittest.mock import Mock
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha.strategy_registry import MaturityTier, tier_for, wired_import_tuple
from alpha.engine.portfolio_manager import PortfolioManager
from strategies.tail_hedge import strategy_crisis_trend_core as core_module
from strategies.tail_hedge import strategy_vixm_backwardation as vixm_module
from strategies.tail_hedge.run_ladder_tail_hedge_study import (
    allocate_path, funded_weights, exact_common_calendar, validate_saved_config,
)
from strategies.tail_hedge.build_ladder_tail_evidence import dividend_tax_penalty


def test_drift_is_independent_compounding_not_daily_weight_mix():
    return_df = pd.DataFrame({"a": [0., 1., -.5], "b": [0., 0., 0.]},
                             index=pd.bdate_range("2020-01-02", periods=3))
    return_ser, drift_df = allocate_path(return_df, pd.Series({"a": .5, "b": .5}), False)
    np.testing.assert_allclose(return_ser, [0., .5, -1/3])
    assert drift_df.iloc[1]["a"] == pytest.approx(2/3)


def test_annual_reset_uses_previous_close_and_charges_both_transfer_sides():
    return_df = pd.DataFrame({"a": [0., 1., -.5], "b": [0., 0., 0.]},
                            index=pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04"]))
    return_ser, _ = allocate_path(return_df, pd.Series({"a": .5, "b": .5}), True, 10.)
    assert return_ser.iloc[2] == pytest.approx((1-.001/3)*.75-1)


def test_missing_returns_and_nonzero_anchor_fail_loud():
    return_df = pd.DataFrame({"a": [0., np.nan]}, index=pd.bdate_range("2020-01-02", periods=2))
    with pytest.raises(ValueError, match="Missing"):
        allocate_path(return_df, pd.Series({"a": 1.}), False)
    return_df["a"] = [.01, 0.]
    with pytest.raises(ValueError, match="anchor"):
        allocate_path(return_df, pd.Series({"a": 1.}), False)


def test_common_calendar_cannot_erase_internal_loss_session():
    session_idx = pd.bdate_range("2020-01-06", periods=3)
    with pytest.raises(ValueError, match="Internal session"):
        exact_common_calendar(session_idx, session_idx[[0, 2]])


@pytest.mark.parametrize("changed_field_str", ["strategy_import_str", "capital_base_float"])
def test_saved_config_identity_and_capital_must_match(changed_field_str):
    pod_dict = {"pod_id_str": "a", "strategy_import_str": "strategies.a", "weight_float": 1.}
    portfolio_obj = SimpleNamespace(pod_info_list=[pod_dict], weights=[1.], _capital_base=100_000.)
    config_dict = {"pods": [dict(pod_dict)], "capital_base_float": 100_000.}
    if changed_field_str == "strategy_import_str":
        config_dict["pods"][0][changed_field_str] = "strategies.b"
    else:
        config_dict[changed_field_str] = 200_000.
    with pytest.raises(ValueError, match="identity or capital"):
        validate_saved_config(portfolio_obj, config_dict)


def test_future_return_cannot_change_past_allocations():
    return_df = pd.DataFrame({"a": [0., .1, -.2, .3], "b": [0., -.1, .2, -.3]},
                            index=pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"]))
    weight_ser = pd.Series({"a": .5, "b": .5})
    first_ser, first_df = allocate_path(return_df, weight_ser, True)
    return_df.iloc[-1] = [3., -.9]
    repeat_ser, repeat_df = allocate_path(return_df, weight_ser, True)
    pd.testing.assert_series_equal(first_ser.iloc[:-1], repeat_ser.iloc[:-1])
    pd.testing.assert_frame_equal(first_df.iloc[:-1], repeat_df.iloc[:-1])


def test_tax_sensitivity_uses_ex_date_and_only_positive_dividends():
    session_idx = pd.bdate_range("2020-01-06", periods=3)
    nav_ser = pd.Series([100., 200., 300.], index=session_idx)
    ledger_df = pd.DataFrame({"ex_date": ["2020-01-07", "2020-01-07"],
                              "gross_dividend_cash_float": [20., -10.]})
    penalty_ser = dividend_tax_penalty(ledger_df, nav_ser, 100., .25)
    np.testing.assert_allclose(penalty_ser, [0., .05, 0.])


def test_ladder3_ten_percent_preserves_mr_funding_floors():
    weight_ser = pd.Series({"taa": .32, "mo": .32, "dv2": .18, "hpi": .18})
    import_dict = {"taa": "strategies.taa.a", "mo": "strategies.mo.a",
                   "dv2": "strategies.dv2.a", "hpi": "strategies.hpi.a"}
    result_ser = funded_weights(weight_ser, import_dict, 150_000., {"Core": .05, "VIXM": .05})
    np.testing.assert_allclose(result_ser, [.27, .27, .18, .18, .05, .05])
    result_ser = funded_weights(weight_ser, import_dict, 150_000., {"Core": .05})
    np.testing.assert_allclose(result_ser, [.304, .304, .171, .171, .05])


@pytest.mark.parametrize("strategy_module", [core_module, vixm_module])
def test_run_variant_forwards_allocated_capital_and_dates(monkeypatch, strategy_module):
    captured_list = []
    loader_str = ("get_crisis_trend_core_data" if strategy_module is core_module
                  else "get_vixm_backwardation_data")
    constructor_str = ("CrisisTrendCoreStrategy" if strategy_module is core_module
                       else "VixmBackwardationStrategy")
    monkeypatch.setattr(strategy_module, loader_str, lambda config_obj: captured_list.append(config_obj))
    monkeypatch.setattr(strategy_module, "build_execution_calendar_idx", Mock(return_value=[]))
    monkeypatch.setattr(strategy_module, constructor_str, Mock())
    monkeypatch.setattr(strategy_module, "run_daily", Mock())
    strategy_module.run_variant(save_results_bool=False, capital_base_float=5_000.,
                                backtest_start_date_str="2015-01-01", end_date_str="2026-08-31")
    assert captured_list[0].capital_base_float == 5_000.
    assert captured_list[0].backtest_start_date_str == "2015-01-01"
    assert captured_list[0].end_date_str == "2026-08-31"


def test_hedges_are_pm_only_never_live_wired():
    for module_str in (core_module.__name__, vixm_module.__name__):
        assert tier_for(module_str) == MaturityTier.PM_READY
        assert module_str not in wired_import_tuple()


@pytest.mark.parametrize("book_str", ["ladder_3_growth", "ladder_3b_growth_2x", "ladder_3c_growth_2x_btal", "ladder_4_growth"])
def test_research_candidate_configs_parse_with_real_funding_floors(book_str):
    candidate_obj = PortfolioManager.from_yaml(Path("portfolios") / f"{book_str}_tail_vixm_10_research.yaml")
    assert candidate_obj.config.end_date_str == "2026-08-31"
    assert candidate_obj.config.rebalance is None
    assert candidate_obj.config.pod_config_list[-1].weight_float == .10
    assert sum(candidate_obj.config.weight_list) == pytest.approx(1.)
