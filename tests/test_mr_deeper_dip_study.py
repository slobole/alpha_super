"""Behavioral accounting and timing regression tests for isolated dip research."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.engine.strategy import Strategy
from scripts.research.run_mr_deeper_dip_study import (
    DeeperDipResearchMixin, limit_fill_price,
)


class NativeFixtureStrategy(Strategy):
    def compute_signals(self, pricing_data):
        return pricing_data

    def iterate(self, data, close, open_prices):
        pass


class ResearchFixtureStrategy(DeeperDipResearchMixin, NativeFixtureStrategy):
    pass


def prices_fixture():
    date_idx = pd.date_range("2020-01-02", periods=4, freq="B")
    price_df = pd.DataFrame(index=date_idx)
    for symbol_str in ("AAA", "BBB"):
        for field_str, values_list in {
            "Open": [100., 100., 102., 101.],
            "High": [101., 102., 104., 103.],
            "Low": [99., 98., 99., 100.],
            "Close": [100., 101., 102., 102.],
            "Volume": [100000., 100000., 100000., 100000.],
            "Dividend": [0., 1., 0., 0.],
        }.items():
            price_df[(symbol_str, field_str)] = values_list
    price_df.columns = pd.MultiIndex.from_tuples(price_df.columns)
    price_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "AAA": "CAPITALSPECIAL", "BBB": "CAPITALSPECIAL"}
    return price_df


def make_fixture(native_bool=False, depth_float=None, stress_bool=False):
    strategy_class = NativeFixtureStrategy if native_bool else ResearchFixtureStrategy
    strategy_obj = strategy_class("fixture", [], capital_base=10000.,
                                  slippage=.00025, commission_per_share=.005,
                                  commission_minimum=1.)
    if not native_bool:
        strategy_obj.configure_research(depth_float, stress_bool)
    return strategy_obj


def process_session(strategy_obj, price_df, row_int):
    strategy_obj.previous_bar = price_df.index[row_int-1]
    strategy_obj.current_bar = price_df.index[row_int]
    strategy_obj.process_orders(price_df)


def test_limit_trigger_and_penetration():
    assert limit_fill_price(100., 99., .01) == 99.
    assert limit_fill_price(100., 99.001, .01) is None
    assert limit_fill_price(100., 98.96, .01, .0005) is None
    assert limit_fill_price(100., 98.9505, .01, .0005) == 99.
    assert limit_fill_price(100., np.nan, .01) is None
    assert limit_fill_price(np.nan, 99., .01) is None
    assert limit_fill_price(100., 101., .01) is None
    with pytest.raises(ValueError):
        limit_fill_price(100., 98., 0.)


def test_limit_cash_ledger_price_and_close_mark_agree():
    price_df = prices_fixture()
    strategy_obj = make_fixture(depth_float=.01)
    strategy_obj.order_value("AAA", 1000., trade_id=1)
    process_session(strategy_obj, price_df, 1)
    transaction_df = strategy_obj.get_transactions()
    assert transaction_df.iloc[0]["amount"] == 10.
    assert transaction_df.iloc[0]["price"] == 99.
    assert transaction_df.iloc[0]["total_value"] == 990.
    expected_cash_float = 10000.-990.-1.-990.*.00025
    assert strategy_obj.cash == pytest.approx(expected_cash_float)
    assert strategy_obj.total_value == pytest.approx(expected_cash_float+10.*101.)
    assert strategy_obj.entry_row_list[0]["research_friction"] == pytest.approx(.2475)


def test_miss_creates_no_position_cost_or_persistent_order():
    price_df = prices_fixture()
    price_df.loc[price_df.index[1], ("AAA", "Low")] = 99.5
    strategy_obj = make_fixture(depth_float=.01)
    strategy_obj.order_value("AAA", 1000., trade_id=1)
    process_session(strategy_obj, price_df, 1)
    assert strategy_obj.cash == 10000.
    assert strategy_obj.get_position("AAA") == 0.
    assert len(strategy_obj.get_orders()) == 0
    assert len(strategy_obj.get_transactions()) == 0
    assert strategy_obj.entry_row_list[0]["status"] == "expired_or_invalid_bar"
    process_session(strategy_obj, price_df, 2)
    assert strategy_obj.cash == 10000.


def test_moo_parity_with_native_across_dividend_and_exit():
    price_df = prices_fixture()
    native_obj = make_fixture(native_bool=True)
    research_obj = make_fixture()
    for strategy_obj in (native_obj, research_obj):
        strategy_obj.order_value("AAA", 1000., trade_id=1)
        process_session(strategy_obj, price_df, 1)
        strategy_obj.order_target_value("AAA", 0., trade_id=1)
        strategy_obj.order_value("BBB", 1000., trade_id=2)
        process_session(strategy_obj, price_df, 2)
    pd.testing.assert_frame_equal(
        native_obj.get_transactions().drop(columns="order_id"),
        research_obj.get_transactions().drop(columns="order_id"))
    pd.testing.assert_frame_equal(native_obj.get_dividend_ledger(), research_obj.get_dividend_ledger())
    assert research_obj.cash == native_obj.cash
    assert research_obj.total_value == native_obj.total_value
    assert research_obj.get_dividend_ledger()["net_dividend_cash_float"].sum() == pytest.approx(7.5)


def test_mixed_exit_entry_stress_and_dividend_entitlement():
    price_df = prices_fixture()
    strategy_obj = make_fixture(depth_float=.01, stress_bool=True)
    strategy_obj.order_value("AAA", 1000., trade_id=1)
    process_session(strategy_obj, price_df, 1)
    strategy_obj.order_target_value("AAA", 0., trade_id=1)
    strategy_obj.order_value("BBB", 1000., trade_id=2)
    process_session(strategy_obj, price_df, 2)
    transaction_df = strategy_obj.get_transactions()
    friction_float = sum(row_dict["research_friction"] for row_dict in strategy_obj.friction_row_list)
    dividend_float = strategy_obj.get_dividend_ledger()["net_dividend_cash_float"].sum()
    assert dividend_float == pytest.approx(7.5)  # new BBB buyer gets no ex-date dividend
    expected_cash_float = 10000.+dividend_float-transaction_df["total_value"].sum()-transaction_df["commission"].sum()-friction_float
    assert strategy_obj.cash == pytest.approx(expected_cash_float)
    assert strategy_obj.get_position("AAA") == 0.
    assert strategy_obj.get_position("BBB") == 9.
    # Every executed side gets stress; only positive limit entries get .025% hurdle.
    expected_friction_float = (transaction_df["total_value"].abs().sum()*.001
                              +transaction_df.loc[transaction_df["amount"]>0, "total_value"].sum()*.00025)
    assert friction_float == pytest.approx(expected_friction_float)
    assert strategy_obj.total_value == pytest.approx(strategy_obj.cash+9.*102.)


def test_zero_sized_order_has_no_fill_or_fee():
    price_df = prices_fixture()
    strategy_obj = make_fixture(depth_float=.01)
    strategy_obj.order_value("AAA", 1., trade_id=1)
    process_session(strategy_obj, price_df, 1)
    assert len(strategy_obj.get_transactions()) == 0
    assert strategy_obj.cash == 10000.


def test_higher_daily_high_close_cannot_change_entry_fill_or_quantity():
    first_df = prices_fixture()
    second_df = first_df.copy()
    second_df.loc[second_df.index[1], ("AAA", "High")] = 150.
    second_df.loc[second_df.index[1], ("AAA", "Close")] = 149.
    result_list = []
    for price_df in (first_df, second_df):
        strategy_obj = make_fixture(depth_float=.01)
        strategy_obj.order_value("AAA", 1000., trade_id=1)
        process_session(strategy_obj, price_df, 1)
        result_list.append(strategy_obj.get_transactions().drop(columns="order_id"))
    pd.testing.assert_frame_equal(*result_list)


def test_unknown_low_is_not_a_known_missed_trade():
    price_df = prices_fixture()
    price_df.loc[price_df.index[1], ("AAA", "Low")] = np.nan
    strategy_obj = make_fixture(depth_float=.01)
    strategy_obj.order_value("AAA", 1000., trade_id=1)
    with pytest.raises(AssertionError, match="Unknown limit eligibility"):
        process_session(strategy_obj, price_df, 1)
    assert len(strategy_obj.get_transactions()) == 0
    assert strategy_obj.cash == 10000.

def test_analysis_first_day_and_cash_reconciliation():
    from scripts.research.analyze_mr_deeper_dip_study import nav_returns, performance, reconcile_cash
    date_idx = pd.date_range("2020-01-02", periods=2, freq="B")
    nav_ser = pd.Series([99000., 100000.], index=date_idx)
    return_ser = nav_returns(nav_ser)
    assert return_ser.iloc[0] == pytest.approx(-.01)
    assert (1.+return_ser).prod() == pytest.approx(1.)
    assert performance(return_ser)["max_drawdown"] == pytest.approx(-.01)
    daily_df = pd.DataFrame({"cash":[98998.,99005.5],"invested":[1000.,1000.],
                            "nav":[99998.,100005.5]},index=date_idx)
    transaction_df = pd.DataFrame({"bar":[date_idx[0]],"total_value":[1000.],"commission":[1.]})
    dividend_df = pd.DataFrame({"ex_date":[date_idx[1]],"net_dividend_cash_float":[7.5]})
    friction_df = pd.DataFrame({"date":[date_idx[0]],"research_friction":[1.]})
    assert reconcile_cash(daily_df, transaction_df, dividend_df, friction_df) < 1e-9


def test_paired_opportunities_include_missed_winners_and_friction():
    from scripts.research.analyze_mr_deeper_dip_study import paired_opportunities
    price_df = prices_fixture()
    baseline_entry_df = pd.DataFrame({
        "trade_id":[1,2],"open":[100.,100.],"low":[98.,99.8],
        "sizing_price":[100.,100.],"decision_date":[price_df.index[0]]*2})
    baseline_trade_df = pd.DataFrame({
        "trade_id":[1,2],"shares":[10.,10.],"entry_price":[100.025,100.025],
        "exit_equivalent_price":[105.,110.],"net_pnl":[47.75,97.75],
        "dividends":[0.,0.],"native_commission":[2.,2.],
        "asset":["AAA","BBB"],"entry_date":[price_df.index[1]]*2,
        "exit_date":[price_df.index[2]]*2,"closed":[True,True]})
    paired_df = paired_opportunities("fixture",price_df,baseline_entry_df,baseline_trade_df)
    one_pct_df = paired_df[paired_df["policy"]=="limit_1pct"].set_index("trade_id")
    assert one_pct_df.loc[1,"limit_net_pnl"] == pytest.approx(60.-2.-.2475)
    assert one_pct_df.loc[2,"limit_net_pnl"] == 0.
    assert one_pct_df.loc[2,"delta_net_pnl"] == pytest.approx(-97.75)
    assert one_pct_df["delta_net_pnl"].sum() < 0.  # Better fill, worse all-opportunity PnL.


def test_trade_attribution_closed_and_terminal_marked():
    from scripts.research.analyze_mr_deeper_dip_study import make_trade_table
    price_df=prices_fixture()
    transaction_df=pd.DataFrame({
        "trade_id":[1,1,2],"bar":[price_df.index[1],price_df.index[2],price_df.index[2]],
        "asset":["AAA","AAA","BBB"],"amount":[10.,-10.,9.],
        "price":[99.,102.,100.98],"total_value":[990.,-1020.,908.82],
        "commission":[1.,1.,1.]})
    dividend_df=pd.DataFrame({
        "asset_str":["AAA"],"ex_date":[price_df.index[2]],"net_dividend_cash_float":[7.5]})
    friction_df=pd.DataFrame({"trade_id":[1,2],"research_friction":[.2475,.227205]})
    trade_df=make_trade_table(transaction_df,dividend_df,friction_df,price_df).set_index("trade_id")
    assert bool(trade_df.loc[1,"closed"])
    assert not bool(trade_df.loc[2,"closed"])
    assert trade_df.loc[1,"net_pnl"] == pytest.approx(30.+7.5-2.-.2475)
    assert trade_df.loc[2,"net_pnl"] == pytest.approx(9.*102.-908.82-1.-.227205)


def test_metadata_acceleration_is_immutable_and_economically_identical():
    import copy
    from scripts.research.run_mr_deeper_dip_fast import freeze_frame_metadata, FrozenMetadata
    source_dict = {"AAA": "CAPITALSPECIAL", "BBB": "CAPITALSPECIAL"}
    frozen_map = FrozenMetadata(source_dict)
    source_dict["AAA"] = "TOTALRETURN"
    assert frozen_map["AAA"] == "CAPITALSPECIAL"
    assert copy.deepcopy(frozen_map) is frozen_map
    with pytest.raises(TypeError):
        frozen_map["AAA"] = "TOTALRETURN"
    with pytest.raises(TypeError):
        frozen_map._value_map = {}
    with pytest.raises(TypeError):
        frozen_map._value_map["AAA"] = "TOTALRETURN"
    with pytest.raises(TypeError):
        FrozenMetadata({"invalid": []})
    native_df = prices_fixture()
    fast_df = freeze_frame_metadata(native_df.copy())
    assert fast_df.attrs == native_df.attrs
    assert fast_df.loc[:, [("AAA", "Close")]].attrs == native_df.attrs
    assert pd.concat([fast_df.iloc[:2], fast_df.iloc[2:]]).attrs == native_df.attrs
    native_obj, fast_obj = make_fixture(native_bool=True), make_fixture(native_bool=True)
    for strategy_obj, price_df in ((native_obj,native_df),(fast_obj,fast_df)):
        strategy_obj.order("AAA", 10, target=False, trade_id=1)
        process_session(strategy_obj,price_df,1)
        strategy_obj.order("AAA", -10, target=False, trade_id=1)
        process_session(strategy_obj,price_df,2)
    pd.testing.assert_frame_equal(
        native_obj.get_transactions().drop(columns="order_id"),
        fast_obj.get_transactions().drop(columns="order_id"))
    pd.testing.assert_frame_equal(native_obj.get_dividend_ledger(),fast_obj.get_dividend_ledger())
    assert native_obj.cash == fast_obj.cash
    assert native_obj.total_value == fast_obj.total_value


def test_independent_daily_marks_catch_offsetting_intermediate_error():
    from scripts.research.analyze_mr_deeper_dip_study import reconcile_daily_marks
    price_df=prices_fixture()
    transaction_df=pd.DataFrame({"bar":[price_df.index[1]],"asset":["AAA"],
        "amount":[10.],"price":[100.],"total_value":[1000.]})
    daily_df=pd.DataFrame({"invested":[1010.,1020.],"cash":[8999.,8999.],
                           "nav":[10009.,10019.]},index=price_df.index[1:3])
    assert reconcile_daily_marks(daily_df,transaction_df,price_df)==0.
    corrupted_df=daily_df.copy()
    corrupted_df.loc[price_df.index[1],["invested","nav"]]+=1.
    with pytest.raises(AssertionError,match="daily close marks"):
        reconcile_daily_marks(corrupted_df,transaction_df,price_df)
    bad_transaction_df=transaction_df.copy()
    bad_transaction_df["total_value"]+=1.
    with pytest.raises(AssertionError,match="notional"):
        reconcile_daily_marks(daily_df,bad_transaction_df,price_df)


def test_dividend_reconstruction_rejects_duplicate_and_wrong_entitlement():
    from scripts.research.analyze_mr_deeper_dip_study import verify_dividend_entitlements
    price_df=prices_fixture()
    strategy_obj=make_fixture(native_bool=True)
    strategy_obj.order("AAA",10,target=False,trade_id=1)
    process_session(strategy_obj,price_df,1)
    strategy_obj.order("AAA",-10,target=False,trade_id=1)
    process_session(strategy_obj,price_df,2)
    transaction_df=strategy_obj.get_transactions()
    dividend_df=strategy_obj.get_dividend_ledger()
    assert verify_dividend_entitlements(price_df.index[1:3],price_df,transaction_df,dividend_df)==1
    duplicate_df=pd.concat([dividend_df,dividend_df],ignore_index=True)
    with pytest.raises(AssertionError,match="duplicated or missing"):
        verify_dividend_entitlements(price_df.index[1:3],price_df,transaction_df,duplicate_df)
    wrong_date_df=dividend_df.copy()
    wrong_date_df["entitlement_date"]=price_df.index[0]
    with pytest.raises(AssertionError,match="entitlement fields"):
        verify_dividend_entitlements(price_df.index[1:3],price_df,transaction_df,wrong_date_df)
