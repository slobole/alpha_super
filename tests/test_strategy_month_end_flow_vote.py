"""Month-End Flow Vote on the real engine: parity with the Scout spec, signal causality, and fill semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.scout.specs import eom, eom_v2
from strategies.taa_beyond_6040 import strategy_taa_month_end_flow_vote as vote_module


def _synthetic_frames(end_str: str = "2006-06-30", seed_int: int = 11, dividend_float: float = 0.0):
    """Random observed prices on real XNYS sessions, in the two shapes the engine and the spec read."""
    session_index = eom.xnys_session_index(pd.Timestamp(end_str))
    data_index = session_index[(session_index >= "2002-07-26") & (session_index <= end_str)]
    rng_obj = np.random.default_rng(seed_int)
    close_dict = {s: 100.0 * np.cumprod(1.0 + rng_obj.normal(0.0002, 0.01, len(data_index))) for s in ("SPY", "TLT", "IEF")}
    close_df = pd.DataFrame(close_dict, index=data_index)
    open_df = close_df.shift(1).fillna(close_df) * (1.0 + rng_obj.normal(0.0, 0.002, close_df.shape))
    dividend_df = pd.DataFrame(0.0, index=data_index, columns=["SPY", "TLT"])
    dividend_df.iloc[20::21] = dividend_float
    field_dict = {}
    for asset_str in ("SPY", "TLT"):
        field_dict[(asset_str, "Open")] = open_df[asset_str]
        field_dict[(asset_str, "High")] = np.maximum(open_df[asset_str], close_df[asset_str])
        field_dict[(asset_str, "Low")] = np.minimum(open_df[asset_str], close_df[asset_str])
        field_dict[(asset_str, "Close")] = close_df[asset_str]
        field_dict[(asset_str, "Dividend")] = dividend_df[asset_str]
        field_dict[(asset_str, "Volume")] = pd.Series(1e7, index=data_index)
    for asset_str in ("SPY", "IEF", "TLT"):
        field_dict[(f"FLOW_TR_{asset_str}", "Close")] = close_df[asset_str]
    for field_str in ("Open", "High", "Low", "Close"):
        field_dict[("$SPX", field_str)] = pd.Series(100.0 + np.arange(len(data_index)) * 0.01, index=data_index)
    pricing_data_df = pd.DataFrame(field_dict, index=data_index)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "SPY": "CAPITALSPECIAL", "TLT": "CAPITALSPECIAL", "$SPX": "TOTALRETURN", "FLOW_TR_SPY": "TOTALRETURN", "FLOW_TR_IEF": "TOTALRETURN", "FLOW_TR_TLT": "TOTALRETURN"}
    pricing_data_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    pricing_data_df.attrs["price_padding_policy_str"] = "NONE"
    inputs = eom_v2.EomV2Inputs(
        eom.EomInputs(open_df=open_df[["SPY", "TLT"]], close_df=close_df[["SPY", "TLT"]], dividend_df=dividend_df,
                      total_return_close_df=close_df[["SPY", "IEF", "TLT"]], session_index=session_index),
        open_df[["SPY", "TLT"]])
    return pricing_data_df, inputs


def _run_engine(pricing_data_df: pd.DataFrame, config_obj: vote_module.MonthEndVoteFlowConfig) -> vote_module.MonthEndVoteFlowStrategy:
    strategy_obj = vote_module.MonthEndVoteFlowStrategy(config_obj)
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=False, show_signal_progress_bool=False)
    return strategy_obj


@pytest.mark.parametrize("early_str", vote_module.EARLY_RULE_TUPLE)
@pytest.mark.parametrize("fill_str", vote_module.FILL_TUPLE)
def test_engine_equals_the_scout_spec(early_str, fill_str):
    pricing_data_df, inputs = _synthetic_frames(dividend_float=0.3)
    config_obj = vote_module.MonthEndVoteFlowConfig(early_rule_str=early_str, fill_str=fill_str)
    engine_value_ser = _run_engine(pricing_data_df, config_obj).results["total_value"]
    spec_value_ser = eom_v2.simulate_config(inputs, eom_v2.EomV2Config(final_rule_str="vote", early_rule_str=early_str, fill_str=fill_str)).total_value_ser
    assert (engine_value_ser.reindex(spec_value_ser.index) / spec_value_ser - 1.0).abs().max() < 1e-9


def test_month_table_and_schedule_equal_the_scout_spec():
    pricing_data_df, inputs = _synthetic_frames()
    total_return_df = pd.DataFrame({a: pricing_data_df[(f"FLOW_TR_{a}", "Close")] for a in vote_module.SIGNAL_ASSET_TUPLE})
    engine_table_df = vote_module.build_month_table_df(total_return_df)
    spec_table_df = eom_v2.month_table_df(inputs)
    assert len(engine_table_df) == len(spec_table_df)
    assert np.allclose(engine_table_df["pressure_bps_float"], spec_table_df["pressure_bps_float"])
    assert np.allclose(engine_table_df["half_spread_float"], spec_table_df["half_spread_float"])
    votes_vec = np.array([eom_v2.spy_vote_count_int(r.state_str, r.pressure_bps_float, r.half_spread_float) for r in spec_table_df.itertuples(index=False)])
    assert (engine_table_df["spy_vote_count_int"].to_numpy() == votes_vec).all()


@pytest.mark.parametrize("bucket_float,pressure_float,half_float,expected_int", [
    (1.0, -80.0, -0.02, 3), (2.0, -10.0, -0.01, 2), (2.0, -10.0, 0.01, 1), (5.0, 40.0, -0.01, 1), (5.0, 40.0, 0.01, 0), (float("nan"), -5.0, 0.01, 1),
])
def test_vote_and_weights(bucket_float, pressure_float, half_float, expected_int):
    assert vote_module.spy_vote_count_int(bucket_float, pressure_float, half_float) == expected_int
    spy_float, tlt_float = vote_module.final_weight_tuple(bucket_float, pressure_float, half_float)
    assert spy_float == pytest.approx(expected_int / 3.0) and spy_float + tlt_float == pytest.approx(1.0)


def test_next_month_rule():
    assert vote_module.early_weight_tuple("flat", 5.0) == (0.0, 0.0)
    assert vote_module.early_weight_tuple("pair_top", 4.0) == (0.5, -0.5)
    assert vote_module.early_weight_tuple("pair_top", 5.0) == (0.5, -0.5)
    for bucket_float in (1.0, 2.0, 3.0, float("nan")):
        assert vote_module.early_weight_tuple("pair_top", bucket_float) == (0.0, 0.0)


def test_future_prices_cannot_change_a_month_row():
    """*** CRITICAL*** Prefix recomputation: each month's P, rank and half spread are final at the measure close."""
    pricing_data_df, _ = _synthetic_frames()
    total_return_df = pd.DataFrame({a: pricing_data_df[(f"FLOW_TR_{a}", "Close")] for a in vote_module.SIGNAL_ASSET_TUPLE})
    full_table_df = vote_module.build_month_table_df(total_return_df).set_index("month_period")
    for measure_ts in full_table_df["measure_date"].iloc[[3, 12, 25, 40]]:
        prefix_table_df = vote_module.build_month_table_df(total_return_df.loc[:measure_ts]).set_index("month_period")
        row_str = prefix_table_df.index[-1]
        assert prefix_table_df.at[row_str, "measure_date"] == measure_ts
        for column_str in ("pressure_bps_float", "half_spread_float", "spy_vote_count_int"):
            assert prefix_table_df.at[row_str, column_str] == full_table_df.at[row_str, column_str]


def test_a_fill_session_shock_leaves_its_own_target_unchanged():
    pricing_data_df, _ = _synthetic_frames()
    total_return_df = pd.DataFrame({a: pricing_data_df[(f"FLOW_TR_{a}", "Close")] for a in vote_module.SIGNAL_ASSET_TUPLE})
    table_df = vote_module.build_month_table_df(total_return_df)
    row = table_df.iloc[20]
    shocked_df = total_return_df.copy()
    shocked_df.loc[row.final_fill_date:] *= np.array([2.0, 0.7, 0.3])
    shocked_table_df = vote_module.build_month_table_df(shocked_df)
    shocked_row = shocked_table_df.iloc[20]
    assert shocked_row.pressure_bps_float == row.pressure_bps_float
    assert shocked_row.half_spread_float == row.half_spread_float
    assert shocked_row.spy_vote_count_int == row.spy_vote_count_int


def test_quantities_are_sized_on_the_prior_close_for_either_fill():
    pricing_data_df, _ = _synthetic_frames()
    for fill_str in ("close", "open"):
        strategy_obj = _run_engine(pricing_data_df, vote_module.MonthEndVoteFlowConfig(early_rule_str="flat", fill_str=fill_str))
        decision_df = strategy_obj.decision_df.copy()
        row = decision_df[decision_df["fill_date"] >= "2004-09-01"].iloc[0]
        close_ser = pricing_data_df[("TLT", "Close")]
        expected_int = int(np.trunc(strategy_obj.order_schedule_df.loc[row["fill_date"], "TLT"] * row["sizing_nav_float"] / close_ser.loc[row["decision_date"]]))
        assert row["TLT"] == expected_int


def test_open_fill_is_at_the_session_open_and_close_fill_at_the_session_close():
    pricing_data_df, _ = _synthetic_frames()
    for fill_str, field_str in (("close", "Close"), ("open", "Open")):
        strategy_obj = _run_engine(pricing_data_df, vote_module.MonthEndVoteFlowConfig(early_rule_str="flat", fill_str=fill_str))
        transaction_df = strategy_obj.get_transactions() if hasattr(strategy_obj, "get_transactions") else pd.DataFrame(strategy_obj._transactions)
        assert len(transaction_df) > 20
        row = transaction_df.iloc[10]
        asset_str, date_ts = row["asset"], pd.Timestamp(row["bar"]) if "bar" in row else pd.Timestamp(row["date"])
        reference_float = pricing_data_df.loc[date_ts, (asset_str, field_str)]
        assert float(row["price"]) == pytest.approx(reference_float, rel=5e-4)


def test_no_short_and_no_borrow_when_the_next_month_leg_is_flat():
    pricing_data_df, _ = _synthetic_frames()
    strategy_obj = _run_engine(pricing_data_df, vote_module.MonthEndVoteFlowConfig(early_rule_str="flat", fill_str="close"))
    assert (strategy_obj.held_share_df >= 0).all().all()
    assert strategy_obj.borrow_fee_total_float == 0.0


def test_pair_leg_shorts_tlt_and_accrues_borrow():
    pricing_data_df, _ = _synthetic_frames()
    strategy_obj = _run_engine(pricing_data_df, vote_module.MonthEndVoteFlowConfig(early_rule_str="pair_top", fill_str="close"))
    assert (strategy_obj.held_share_df["TLT"] < 0).any()
    assert strategy_obj.borrow_fee_total_float > 0.0


def test_refuses_padded_or_mislabelled_data():
    pricing_data_df, _ = _synthetic_frames()
    bad_df = pricing_data_df.copy()
    bad_df.attrs = {**pricing_data_df.attrs, "price_padding_policy_str": "ALLMARKETDAYS"}
    with pytest.raises(ValueError):
        vote_module.MonthEndVoteFlowStrategy().compute_signals(bad_df)
    bad_df = pricing_data_df.copy()
    bad_df.attrs = {**pricing_data_df.attrs, "norgate_adjustment_by_symbol_dict": {**pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"], "FLOW_TR_TLT": "CAPITALSPECIAL"}}
    with pytest.raises(ValueError):
        vote_module.MonthEndVoteFlowStrategy().compute_signals(bad_df)


def test_config_guards():
    with pytest.raises(ValueError):
        vote_module.MonthEndVoteFlowConfig(fill_str="next_week")
    with pytest.raises(ValueError):
        vote_module.MonthEndVoteFlowConfig(early_rule_str="main")
    with pytest.raises(ValueError):
        replace(vote_module.DEFAULT_CONFIG, backtest_start_date_str="2002-12-31")
