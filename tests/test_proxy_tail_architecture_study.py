"""Research allocation accounting tests; no market connection."""
import numpy as np
import pandas as pd
import pytest

from strategies.tail_hedge.run_ladder_tail_hedge_study import allocate_path, exact_common_calendar
from strategies.tail_hedge.run_proxy_tail_architecture_study import policy_path, longest_underwater_int, bounded_event_dict
from strategies.tail_hedge.run_vxz_history_proxy import trim_proxy_boundaries


def return_fixture_df():
    return pd.DataFrame({"base_a": [0., 1., 0., -.2], "base_b": [0., 0., 0., .1],
        "Core": [0., 0., 0., .2]},
        index=pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"]))


@pytest.mark.parametrize("policy_str,annual_bool", [("drift", False), ("annual_all", True)])
def test_old_allocation_accounting_parity(policy_str, annual_bool):
    return_df = return_fixture_df()
    weight_ser = pd.Series({"base_a": .6, "base_b": .3, "Core": .1})
    actual_ser, actual_df = policy_path(return_df, weight_ser, policy_str, ["Core"])
    expected_ser, expected_df = allocate_path(return_df, weight_ser, annual_bool)
    pd.testing.assert_series_equal(actual_ser, expected_ser)
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_hedge_only_restores_budget_without_rebalancing_base_relative_weights():
    return_df = return_fixture_df()
    weight_ser = pd.Series({"base_a": .6, "base_b": .3, "Core": .1})
    return_ser, drift_df = policy_path(return_df, weight_ser, "annual_hedge_only", ["Core"])
    assert drift_df.iloc[2].Core == pytest.approx(.1)
    assert drift_df.iloc[2].base_a / drift_df.iloc[2].base_b == pytest.approx(4.)
    # Prior hedge=0.1/1.6=6.25%; buy3.75pp and sell3.75pp, 10bps/side.
    assert return_ser.iloc[2] == pytest.approx(-.001*.075)


def test_no_hedge_policy_equals_unrebalanced_baseline():
    return_df = return_fixture_df()[["base_a", "base_b"]]
    weight_ser = pd.Series({"base_a": 2/3, "base_b": 1/3})
    actual_ser, _ = policy_path(return_df, weight_ser, "annual_hedge_only", [])
    expected_ser, _ = policy_path(return_df, weight_ser, "drift", [])
    np.testing.assert_allclose(actual_ser, expected_ser, atol=1e-15)


def test_future_change_leaves_past_weights_and_returns_unchanged():
    return_df = return_fixture_df()
    weight_ser = pd.Series({"base_a": .6, "base_b": .3, "Core": .1})
    before_ser, before_df = policy_path(return_df, weight_ser, "annual_hedge_only", ["Core"])
    return_df.iloc[-1] = [2., -.9, 4.]
    after_ser, after_df = policy_path(return_df, weight_ser, "annual_hedge_only", ["Core"])
    pd.testing.assert_series_equal(before_ser.iloc[:-1], after_ser.iloc[:-1])
    pd.testing.assert_frame_equal(before_df.iloc[:-1], after_df.iloc[:-1])


def test_missing_internal_returns_and_nonzero_anchor_rejected():
    return_df = return_fixture_df()
    weight_ser = pd.Series({"base_a": .6, "base_b": .3, "Core": .1})
    return_df.loc[return_df.index[2], "Core"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        policy_path(return_df, weight_ser, "annual_hedge_only", ["Core"])
    return_df = return_fixture_df()
    return_df.iloc[0, 0] = .1
    with pytest.raises(ValueError, match="anchor"):
        policy_path(return_df, weight_ser, "annual_hedge_only", ["Core"])


def test_core_calendar_does_not_depend_on_missing_preinception_vixm():
    session_idx = pd.bdate_range("2008-03-03", periods=800)
    hedge_df = pd.DataFrame({"Core": .01, "VIXM": np.nan, "SHY": 0., "SPY": -.01}, index=session_idx)
    hedge_df.loc[session_idx[750]:, "VIXM"] = .02
    core_panel_df = hedge_df[["Core", "SHY", "SPY"]].dropna(how="all")
    actual_idx = exact_common_calendar(session_idx, core_panel_df.index)
    assert actual_idx[0] == pd.Timestamp("2008-03-03")
    assert len(actual_idx) == 800


def test_underwater_includes_unrecovered_terminal_run():
    assert longest_underwater_int(pd.Series([.1, -.1, .02, .02, .3])) == 3
    assert longest_underwater_int(pd.Series([-.1, .01, .01])) == 3


def test_proxy_only_trims_outer_missing_prices_not_internal_losses():
    price_df = pd.DataFrame({("VIXM", "Close"): [np.nan, 10., 9., 8.],
        ("SHY", "Close"): 100., ("$SPX", "Close"): 100.}, index=pd.bdate_range("2009-01-29", periods=4))
    assert len(trim_proxy_boundaries(price_df)) == 3
    price_df.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="internal"):
        trim_proxy_boundaries(price_df)


def test_event_must_cover_both_boundaries():
    return_ser = pd.Series([0., -.1, -.2], index=pd.to_datetime(["2008-03-03", "2008-03-04", "2009-03-09"]))
    assert bounded_event_dict(return_ser, "2007-10-09", "2009-03-09")["status"] == "unavailable"
    assert bounded_event_dict(return_ser, "2008-03-03", "2009-03-10")["status"] == "unavailable"
    assert bounded_event_dict(return_ser, "2008-03-03", "2009-03-09")["return"] == pytest.approx(-.28)
