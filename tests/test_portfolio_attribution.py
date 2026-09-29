"""Attribution must reconcile at rebalance boundaries without booking transfers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.engine.portfolio_attribution import portfolio_attribution_dict


def _portfolio_obj(return_list, rebalance_dict=None):
    date_index = pd.bdate_range("2024-01-30", periods=len(return_list))
    return_df = pd.DataFrame(return_list, index=date_index, columns=["A", "B"], dtype=float)
    equity_df = return_df.copy()
    capital_vec = np.array([50.0, 50.0])
    target_dict = {}
    for position_int, return_vec in enumerate(return_df.to_numpy()):
        if position_int in (rebalance_dict or {}):
            weight_vec = np.array(rebalance_dict[position_int])
            capital_vec = capital_vec.sum() * weight_vec
            target_dict[date_index[position_int]] = weight_vec
        capital_vec = capital_vec * (1.0 + return_vec)
        equity_df.iloc[position_int] = capital_vec
    target_df = pd.DataFrame.from_dict(target_dict, orient="index", columns=["A", "B"])
    target_df.index = pd.DatetimeIndex(target_df.index)
    return SimpleNamespace(
        _daily_rets=return_df, _pod_equities=equity_df,
        results=pd.DataFrame({"total_value": equity_df.sum(axis=1)}),
        _rebalance="monthly" if rebalance_dict is not None else None,
        _rebalance_date_index=target_df.index, rebalance_target_weight_df=target_df,
    )


def test_rebalance_day_uses_applied_targets_and_excludes_transfers():
    portfolio_obj = _portfolio_obj([[0, 0], [1, 0], [-0.5, 0]], {2: [0.5, 0.5]})
    result_dict = portfolio_attribution_dict(portfolio_obj)
    contribution_df = result_dict["contribution_df"]
    assert contribution_df.loc["A", "pnl"] == pytest.approx(12.5)
    assert contribution_df.loc["B", "pnl"] == pytest.approx(0)
    assert contribution_df["drawdown_pp"].sum() == pytest.approx(-25)
    assert contribution_df.loc["A", "episode_standalone_pct"] == pytest.approx(-50)
    assert result_dict["daily_pnl_df"].iloc[-1].sum() == pytest.approx(-37.5)


@pytest.mark.parametrize("rebalance_dict", [None, {}, {2: [0.7, 0.3], 4: [0.2, 0.8]}])
def test_all_days_reconcile_with_multiple_or_skipped_rebalances(rebalance_dict):
    portfolio_obj = _portfolio_obj([[0, 0], [.1, -.1], [-.2, .05], [.3, .1], [-.05, .02]], rebalance_dict)
    result_dict = portfolio_attribution_dict(portfolio_obj)
    np.testing.assert_allclose(result_dict["daily_pnl_df"].sum(axis=1).iloc[1:], portfolio_obj.results.total_value.diff().iloc[1:])
    assert result_dict["contribution_df"].pnl.sum() == pytest.approx(portfolio_obj.results.total_value.iloc[-1] - 100)


def test_latest_repeated_peak_and_no_drawdown():
    portfolio_obj = _portfolio_obj([[0, 0], [.1, .1], [0, 0], [-.1, -.1]])
    result_dict = portfolio_attribution_dict(portfolio_obj)
    assert result_dict["peak_obj"] == portfolio_obj.results.index[2]
    assert result_dict["episode_count_int"] == 1
    monotonic_obj = _portfolio_obj([[0, 0], [.1, .2]])
    assert portfolio_attribution_dict(monotonic_obj)["peak_obj"] is None


@pytest.mark.parametrize("damage_str", ["missing_targets", "wrong_nav", "wrong_returns", "different_dates", "nonfinite"])
def test_incomplete_or_inconsistent_artifacts_are_unavailable(damage_str):
    portfolio_obj = _portfolio_obj([[0, 0], [.1, 0], [-.1, 0]], {2: [.5, .5]})
    if damage_str == "missing_targets":
        del portfolio_obj.rebalance_target_weight_df
    elif damage_str == "wrong_nav":
        portfolio_obj.results.iloc[-1, 0] += 1
    elif damage_str == "wrong_returns":
        portfolio_obj._daily_rets.iloc[-1, 0] += .01
    elif damage_str == "different_dates":
        portfolio_obj._daily_rets = portfolio_obj._daily_rets.iloc[1:]
    else:
        portfolio_obj._daily_rets.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError):
        portfolio_attribution_dict(portfolio_obj)


def test_offsetting_return_errors_cannot_hide_wrong_sleeve_attribution():
    portfolio_obj = _portfolio_obj([[0, 0], [.1, -.1]])
    portfolio_obj._daily_rets.iloc[1] = [.2, -.2]
    with pytest.raises(ValueError, match="Individual sleeve"):
        portfolio_attribution_dict(portfolio_obj)


@pytest.mark.parametrize("policy_str", ["fixed", "equal", "inverse_volatility"])
def test_real_portfolio_policies_and_unequal_inception_dates(policy_str):
    from alpha.engine.portfolio import Portfolio
    from test_portfolio import make_strategy

    date_index = pd.bdate_range("2024-01-15", periods=75)
    left_list = [0] + [.012 if index_int % 3 else -.025 for index_int in range(1, 75)]
    right_list = [0] + [.004 if index_int % 2 else -.008 for index_int in range(1, 70)]
    portfolio_obj = Portfolio(
        strategies=[make_strategy("A", date_index, left_list), make_strategy("B", date_index[5:], right_list)],
        weights=[.6, .4], capital_base=1000, rebalance="monthly",
        rebalance_policy_str=policy_str, rebalance_inverse_volatility_lookback_day_int=20,
    )
    attribution_dict = portfolio_attribution_dict(portfolio_obj)
    assert attribution_dict["contribution_df"].drawdown_pp.sum() == pytest.approx(100 * portfolio_obj.results.drawdown.min())
    assert attribution_dict["contribution_df"].pnl.sum() == pytest.approx(portfolio_obj.results.total_value.iloc[-1] - 1000)
