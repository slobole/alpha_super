"""Synthetic accounting checks for the additional exposure diagnostic."""
import numpy as np
import pandas as pd
import pytest

from scripts.research.portfolio_family_20260923 import exposures


def test_overlap_excludes_cash_and_separates_cash_scaling_from_name_diversity():
    calendar_idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    left_df = pd.DataFrame({"A": [.2, 0.], "B": [.3, 0.], "Cash": [.5, 1.]}, index=calendar_idx)
    right_df = pd.DataFrame({"A": [.4, .4], "C": [.6, .6], "Cash": [0., 0.]}, index=calendar_idx)
    daily_df, asset_df = exposures.holdings_overlap(left_df, right_df)
    assert daily_df.long_nav_overlap.tolist() == [.2, 0.]
    assert daily_df.invested_long_overlap.iloc[0] == pytest.approx(.4)
    assert np.isnan(daily_df.invested_long_overlap.iloc[1])
    assert "Cash" not in asset_df.asset.to_list()
    assert asset_df.iloc[0].asset == "A"


def test_asset_gross_uses_eod_sleeve_nav_and_keeps_opposing_legs():
    calendar_idx = pd.to_datetime(["2020-01-02"])
    begin_df = pd.DataFrame({"a": [.5], "b": [.5]}, index=calendar_idx)
    return_df = pd.DataFrame({"a": [1.], "b": [0.]}, index=calendar_idx)
    end_df = exposures.end_weights(begin_df, return_df)
    holdings_dict = {
        "a": pd.DataFrame({"Asset": [2.], "Cash": [-1.]}, index=calendar_idx),
        "b": pd.DataFrame({"Asset": [-1.], "Cash": [2.]}, index=calendar_idx),
    }
    daily_df, asset_df = exposures.aggregate_holdings(end_df, holdings_dict)
    assert end_df.a.iloc[0] == pytest.approx(2 / 3)
    assert daily_df.long_nav.iloc[0] == pytest.approx(4 / 3)
    assert daily_df.short_nav.iloc[0] == pytest.approx(1 / 3)
    assert daily_df.gross_unnetted_nav.iloc[0] == pytest.approx(5 / 3)
    assert daily_df.gross_if_same_ticker_netted_nav.iloc[0] == pytest.approx(1.)
    assert daily_df.top1_fraction_long.iloc[0] == pytest.approx(1.)
    assert daily_df.effective_long_names.iloc[0] == pytest.approx(1.)
    assert asset_df.mean_short_nav.iloc[0] == pytest.approx(1 / 3)


def test_sparse_native_unheld_cells_are_zero_but_missing_cash_and_sessions_fail():
    calendar_idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    weight_df = pd.DataFrame({"A": [.4, np.nan, .2], "Cash": [.6, 1., .8]}, index=calendar_idx)
    clean_df = exposures.clean_holdings(weight_df, calendar_idx)
    assert clean_df.A.iloc[1] == 0.
    with pytest.raises(ValueError, match="every exact session"):
        exposures.clean_holdings(weight_df.iloc[[0, 2]], calendar_idx)
    weight_df.loc[calendar_idx[1], "Cash"] = np.nan
    with pytest.raises(ValueError, match="Cash observations"):
        exposures.clean_holdings(weight_df, calendar_idx)


def test_joint_loss_statistics_use_declared_subset_denominators():
    return_df = pd.DataFrame({"a": [-.02, -.01, .03, .01], "b": [-.01, .02, -.02, .01]})
    result_dict = exposures.pair_statistics(return_df, "a", "b")
    assert result_dict["joint_loss_fraction"] == .25
    assert result_dict["right_loss_given_left_loss"] == .5
    assert result_dict["left_loss_given_right_loss"] == .5
    subset_dict = exposures.pair_statistics(return_df.iloc[[0, 2]], "a", "b")
    assert subset_dict["joint_loss_fraction"] == .5
    assert subset_dict["right_loss_given_left_loss"] == 1.
    assert subset_dict["left_loss_given_right_loss"] == .5


def test_cash_only_concentration_and_loss_conditioning_are_undefined():
    calendar_idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    weight_df = pd.DataFrame({"a": [1., 1.]}, index=calendar_idx)
    holdings_dict = {"a": pd.DataFrame({"Asset": [0., 0.], "Cash": [1., 1.]}, index=calendar_idx)}
    daily_df, _ = exposures.aggregate_holdings(weight_df, holdings_dict)
    assert daily_df.long_hhi.isna().all()
    assert daily_df.effective_long_names.isna().all()
    assert daily_df.top1_long_nav.eq(0).all()
    assert daily_df.top1_fraction_long.isna().all()
    result_dict = exposures.pair_statistics(pd.DataFrame({"a": [.01, .01], "b": [.01, .02]}), "a", "b")
    assert np.isnan(result_dict["correlation"])
    assert np.isnan(result_dict["right_loss_given_left_loss"])
