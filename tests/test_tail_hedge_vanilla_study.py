"""Regression tests for portfolio and crisis report accounting."""

import numpy as np
import pandas as pd
import pytest

from strategies.tail_hedge.run_tail_hedge_vanilla_study import (
    combine_sleeves,
    crisis_tables,
    drawdown_ser,
    metric_dict,
)


def test_combination_compounds_sleeves_independently_and_preserves_first_day() -> None:
    session_idx = pd.bdate_range("2020-01-02", periods=2)
    return_df = pd.DataFrame({"Core": [1.0, -0.5], "VIXM": [0.0, 0.0]}, index=session_idx)
    portfolio_ser, weight_df = combine_sleeves(return_df, {"Core": 0.5, "VIXM": 0.5})
    np.testing.assert_allclose(portfolio_ser, [0.5, -1.0 / 3.0])
    assert (1 + portfolio_ser).prod() == pytest.approx(1.0)
    assert weight_df.loc[session_idx[0], "Core"] == pytest.approx(2 / 3)
    assert weight_df.loc[session_idx[1], "Core"] == pytest.approx(0.5)


def test_missing_return_is_rejected_not_filled() -> None:
    return_df = pd.DataFrame({"Core": [0.0, np.nan], "VIXM": [0.0, 0.1]}, index=pd.bdate_range("2020-01-02", periods=2))
    with pytest.raises(ValueError, match="Missing"):
        combine_sleeves(return_df, {"Core": 0.5, "VIXM": 0.5})


def test_drawdown_includes_starting_capital_peak() -> None:
    return_ser = pd.Series([-0.1, 0.0, -0.1])
    assert drawdown_ser(return_ser).min() == pytest.approx(-0.19)


def test_crisis_excludes_peak_close_return_and_requires_full_coverage() -> None:
    session_idx = pd.bdate_range("2020-02-19", "2020-03-23")
    market_ser = pd.Series(-0.01, index=session_idx)
    market_ser.iloc[0] = 0.5
    return_df = pd.DataFrame({"Core": market_ser, "VIXM": market_ser}, index=session_idx)
    return_df.loc[session_idx[4], "VIXM"] = np.nan
    crisis_df = crisis_tables(return_df, market_ser)
    covid_df = crisis_df.loc[crisis_df["crisis"] == "2020_Covid"].set_index("series")
    assert covid_df.loc["Core", "return"] == pytest.approx(0.99 ** (len(session_idx) - 1) - 1)
    assert covid_df.loc["VIXM", "status"] == "unavailable"


def test_market_beta_and_correlation_use_exact_dates() -> None:
    session_idx = pd.bdate_range("2020-01-01", periods=80)
    market_ser = pd.Series(np.tile([-0.01, 0.02, 0.001, -0.005], 20), index=session_idx)
    metric_map = metric_dict(-0.5 * market_ser, market_ser)
    assert metric_map["beta"] == pytest.approx(-0.5)
    assert metric_map["corr_daily"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="exact market calendar"):
        metric_dict(market_ser.iloc[1:], market_ser.iloc[:-1])


def test_crisis_partial_history_is_not_complete() -> None:
    session_idx = pd.bdate_range("2020-02-20", "2020-03-02")
    return_ser = pd.Series(-0.01, index=session_idx)
    crisis_df = crisis_tables(pd.DataFrame({"Core": return_ser}), return_ser)
    assert crisis_df.loc[crisis_df["crisis"] == "2020_Covid", "status"].item() == "unavailable"


def test_monthly_correlation_does_not_insert_missing_year_as_flat_returns() -> None:
    session_idx = pd.to_datetime(["2019-01-31", "2019-02-28", "2021-01-29", "2021-02-26"])
    market_ser = pd.Series([0.01, 0.03, -0.02, -0.01], index=session_idx)
    strategy_ser = pd.Series([-0.01, -0.02, 0.05, 0.04], index=session_idx)
    assert metric_dict(strategy_ser, market_ser)["corr_monthly"] == pytest.approx(strategy_ser.corr(market_ser))
