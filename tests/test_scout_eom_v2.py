"""The month-end flow split into rules: identity with the gated spec, causality of every signal, and rule semantics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from alpha.scout.specs import eom, eom_v2


def _synthetic_inputs(end_str: str = "2006-06-30", seed_int: int = 7) -> eom_v2.EomV2Inputs:
    """Random prices on real XNYS sessions; the spec only needs observed closes, opens and dividends."""
    session_index = eom.xnys_session_index(pd.Timestamp(end_str))
    data_index = session_index[(session_index >= "2002-07-26") & (session_index <= end_str)]
    rng_obj = np.random.default_rng(seed_int)
    frame_dict = {}
    for symbol_str in ("SPY", "TLT", "IEF"):
        frame_dict[symbol_str] = 100.0 * np.cumprod(1.0 + rng_obj.normal(0.0002, 0.01, len(data_index)))
    close_df = pd.DataFrame(frame_dict, index=data_index)
    open_df = close_df.shift(1).fillna(close_df) * (1.0 + rng_obj.normal(0.0, 0.002, close_df.shape))
    dividend_df = pd.DataFrame(0.0, index=data_index, columns=["SPY", "TLT"])
    base = eom.EomInputs(open_df=open_df[["SPY", "TLT"]], close_df=close_df[["SPY", "TLT"]], dividend_df=dividend_df,
                         total_return_close_df=close_df[["SPY", "IEF", "TLT"]], session_index=session_index)
    return eom_v2.EomV2Inputs(base, open_df[["SPY", "TLT"]])


def _truncated(inputs: eom_v2.EomV2Inputs, end_ts: pd.Timestamp) -> eom_v2.EomV2Inputs:
    base = inputs.base
    return eom_v2.EomV2Inputs(
        eom.EomInputs(open_df=base.open_df.loc[:end_ts], close_df=base.close_df.loc[:end_ts], dividend_df=base.dividend_df.loc[:end_ts],
                      total_return_close_df=base.total_return_close_df.loc[:end_ts], session_index=base.session_index),
        inputs.total_return_open_df.loc[:end_ts])


def test_default_rank_main_close_is_the_gated_spec_exactly():
    inputs = _synthetic_inputs()
    config = eom_v2.EomV2Config(final_rule_str="rank", early_rule_str="main", fill_str="close")
    old_ser = eom.simulate_config(inputs.base).daily_return_ser
    new_ser = eom_v2.simulate_config(inputs, config).daily_return_ser
    assert (old_ser - new_ser).abs().max() == 0.0
    pd.testing.assert_frame_equal(eom.rebalance_weight_df(inputs.base), eom_v2.rebalance_weight_df(inputs, config), check_names=False)


@pytest.mark.parametrize("state_str,pressure_float,half_float,expected_int", [
    ("low", -80.0, -0.02, 3), ("mid", -10.0, -0.01, 2), ("mid", -10.0, 0.01, 1), ("high", 40.0, -0.01, 1), ("high", 40.0, 0.01, 0),
    ("missing", -10.0, 0.01, 1), ("missing", 10.0, 0.01, 0),
])
def test_vote_counts_and_final_weights_sum_to_full_gross(state_str, pressure_float, half_float, expected_int):
    assert eom_v2.spy_vote_count_int(state_str, pressure_float, half_float) == expected_int
    spy_float, tlt_float = eom_v2.final_weight_tuple("vote", state_str, pressure_float, half_float)
    assert spy_float == pytest.approx(expected_int / 3.0)
    assert spy_float + tlt_float == pytest.approx(1.0)


def test_early_rules():
    assert eom_v2.early_weight_tuple("flat", "high", 50.0, 0.01) == (0.0, 0.0)
    assert eom_v2.early_weight_tuple("pair_top", "high", 50.0, 0.01) == (0.5, -0.5)
    assert eom_v2.early_weight_tuple("pair_top", "mid", 50.0, 0.01) == (0.0, 0.0)
    assert eom_v2.early_weight_tuple("pair_top", "missing", 50.0, 0.01) == (0.0, 0.0)
    assert eom_v2.early_weight_tuple("pair_sign", "mid", 5.0, -0.01) == (0.5, -0.5)
    assert eom_v2.early_weight_tuple("pair_sign", "low", -5.0, -0.01) == (0.0, 0.0)
    assert eom_v2.early_weight_tuple("spy_led", "low", -5.0, 0.01) == (1.0, 0.0)
    assert eom_v2.early_weight_tuple("spy_led", "high", 5.0, -0.01) == (0.0, 0.0)
    assert eom_v2.early_weight_tuple("main", "mid", 5.0, 0.01) == (0.0, -1.0)


@pytest.mark.parametrize("fill_str", eom_v2.FILL_TUPLE)
def test_every_decision_is_prefix_invariant(fill_str):
    """*** CRITICAL*** A target for fill session t is the same when the data stop at the close before t."""
    inputs = _synthetic_inputs()
    config = eom_v2.EomV2Config(final_rule_str="vote", early_rule_str="pair_top", fill_str=fill_str)
    full_df = eom_v2.rebalance_weight_df(inputs, config)
    session_index = inputs.base.session_index
    checked_int = 0
    for fill_ts in full_df.index[60::5]:
        decision_ts = session_index[session_index.get_loc(fill_ts) - 1]
        truncated_df = eom_v2.rebalance_weight_df(_truncated(inputs, fill_ts), config)
        assert fill_ts in truncated_df.index
        # The fill row is kept only because its session is in the data; its target must not move when later data vanish.
        pd.testing.assert_series_equal(full_df.loc[fill_ts], truncated_df.loc[fill_ts], check_names=False)
        assert decision_ts < fill_ts
        checked_int += 1
    assert checked_int > 10


@pytest.mark.parametrize("fill_str", eom_v2.FILL_TUPLE)
def test_shocking_the_fill_session_cannot_change_its_target(fill_str):
    """*** CRITICAL*** A wild close on the fill session and after it leaves that session's target unchanged."""
    inputs = _synthetic_inputs()
    config = eom_v2.EomV2Config(final_rule_str="vote", early_rule_str="pair_sign", fill_str=fill_str)
    full_df = eom_v2.rebalance_weight_df(inputs, config)
    fill_ts = full_df.index[len(full_df) // 2]
    base = inputs.base
    shocked_df = base.total_return_close_df.copy()
    shocked_df.loc[fill_ts:] = shocked_df.loc[fill_ts:] * np.array([3.0, 0.2, 1.7])
    shocked = eom_v2.EomV2Inputs(dataclasses.replace(base, total_return_close_df=shocked_df, cache_dict={}), inputs.total_return_open_df)
    shocked_weight_df = eom_v2.rebalance_weight_df(shocked, config)
    pd.testing.assert_series_equal(full_df.loc[fill_ts], shocked_weight_df.loc[fill_ts], check_names=False)
    earlier_df = full_df.loc[:fill_ts]
    pd.testing.assert_frame_equal(earlier_df, shocked_weight_df.loc[:fill_ts])


def test_open_late_ends_the_final_week_at_the_first_open_of_the_new_month():
    inputs = _synthetic_inputs()
    table_df = eom_v2.month_table_df(inputs)
    weight_df = eom_v2.rebalance_weight_df(inputs, eom_v2.EomV2Config(final_rule_str="tlt", early_rule_str="flat", fill_str="open_late"))
    row = table_df.iloc[10]
    assert weight_df.loc[row.final_fill_date, "TLT"] == 1.0
    assert row.early_fill_date not in weight_df.index
    assert weight_df.loc[row.next_first_date].abs().sum() == 0.0


def test_open_fill_earns_the_intraday_of_the_fill_session_and_close_fill_does_not():
    inputs = _synthetic_inputs()
    table_df = eom_v2.month_table_df(inputs)
    row = table_df.iloc[10]
    free_cost = eom_v2.CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0, dividend_withholding_float=0.0)
    open_ser = eom_v2.simulate_config(inputs, eom_v2.EomV2Config(final_rule_str="tlt", early_rule_str="flat", fill_str="open"), free_cost, 10_000_000.0).daily_return_ser
    close_ser = eom_v2.simulate_config(inputs, eom_v2.EomV2Config(final_rule_str="tlt", early_rule_str="flat", fill_str="close"), free_cost, 10_000_000.0).daily_return_ser
    tlt_open_float = inputs.base.open_df.at[row.final_fill_date, "TLT"]
    tlt_close_float = inputs.base.close_df.at[row.final_fill_date, "TLT"]
    # Shares are sized on the prior close, so the fill session earns (Close - Open) per prior-close dollar.
    tlt_prior_close_float = inputs.base.close_df["TLT"].shift(1).at[row.final_fill_date]
    assert open_ser.loc[row.final_fill_date] == pytest.approx((tlt_close_float - tlt_open_float) / tlt_prior_close_float, rel=1e-3)
    assert close_ser.loc[row.final_fill_date] == 0.0


def test_fast_replica_matches_the_engine_without_costs():
    inputs = _synthetic_inputs("2010-12-31")
    date_index, matrix = eom_v2.mcpt_matrix(inputs, None)
    free_cost = eom_v2.CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0, dividend_withholding_float=0.0)
    for final_str, early_str, fill_str in (("vote", "pair_top", "close"), ("vote", "pair_top", "open"), ("sign", "main", "open_late"), ("half", "spy_led", "close")):
        config = eom_v2.EomV2Config(final_rule_str=final_str, early_rule_str=early_str, fill_str=fill_str)
        fast_ser = pd.Series(eom_v2.fast_daily_list(matrix, date_index, [config])[0], index=date_index)
        engine_ser = eom_v2.simulate_config(inputs, config, free_cost, 10_000_000.0).daily_return_ser
        joined_df = pd.concat([fast_ser.rename("fast"), engine_ser.rename("engine")], axis=1).dropna()
        assert joined_df.corr().iloc[0, 1] > 0.995


def test_a_negative_offset_or_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        eom_v2.EomV2Config(final_rule_str="best")
    with pytest.raises(ValueError):
        eom_v2.EomV2Config(decision_offset_int=1)
