"""Causal decision sizing and execution fee units across timing scenarios."""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.engine.execution_timing import (
    ExecutionTimingAnalysis,
    ScheduledOrder,
    _execute_scheduled_order,
    _process_scheduled_order_list,
)
from alpha.engine.strategy import Strategy


def pricing_df(scale_float=1.0, dividend_float=1.0):
    calendar_idx = pd.bdate_range("2024-01-02", periods=7)
    raw_close_vec = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    raw_open_vec = raw_close_vec + 0.4
    price_df = pd.DataFrame({
        ("AAA", "Open"): raw_open_vec / scale_float,
        ("AAA", "High"): (raw_close_vec + 0.7) / scale_float,
        ("AAA", "Low"): (raw_close_vec - 0.3) / scale_float,
        ("AAA", "Close"): raw_close_vec / scale_float,
        ("AAA", "Unadjusted Close"): raw_close_vec,
        ("AAA", "Dividend"): np.array([0.0, dividend_float, 0.0, 0.0, 0.0, 0.0, 0.0]) / scale_float,
    }, index=calendar_idx)
    price_df.columns = pd.MultiIndex.from_tuples(price_df.columns)
    price_df.attrs["norgate_adjustment_by_symbol_dict"] = {"AAA": "CAPITALSPECIAL"}
    return price_df


class HistoricalTimingStrategy(Strategy):
    def __init__(self, decision_amount_dict, unit_str="percent", enabled_bool=True):
        super().__init__(name="HistoricalTiming", benchmarks=[], capital_base=1003.0,
            slippage=0.001, commission_per_share=0.17, commission_minimum=0.1)
        self.decision_amount_dict = decision_amount_dict
        self.unit_str = unit_str
        self.historical_share_units_bool = enabled_bool

    def compute_signals(self, pricing_data):
        return pricing_data.copy()

    def iterate(self, data, close, open_prices):
        if self.previous_bar not in self.decision_amount_dict:
            return
        amount_float = self.decision_amount_dict[self.previous_bar]
        if self.unit_str == "percent":
            self.order_target_percent("AAA", amount_float, trade_id=1)
        elif self.unit_str == "value":
            self.order_target_value("AAA", amount_float, trade_id=1)
        else:
            self.order_target("AAA", amount_float, trade_id=1)


def factory_fn(price_df, unit_str="percent", enabled_bool=True, sign_float=1.0):
    amount_vec = [0.5, 0.8, 0.0] if unit_str == "percent" else [505.0, 803.0, 0.0]
    decision_amount_dict = {price_df.index[position_int]: amount_float * sign_float
        for position_int, amount_float in zip((0, 2, 4), amount_vec)}
    return lambda: HistoricalTimingStrategy(decision_amount_dict, unit_str, enabled_bool)


def timing_strategy_obj(price_df, build_fn, timing_str="next_open", mode_str="signal_bar"):
    with contextlib.redirect_stdout(io.StringIO()):
        analysis_obj = ExecutionTimingAnalysis(
            strategy_factory_fn=build_fn, pricing_data_df=price_df,
            calendar_idx=price_df.index, entry_timing_str_tuple=(timing_str,),
            exit_timing_str_tuple=(timing_str,), order_generation_mode_str=mode_str,
            save_output_bool=False, audit_override_bool=False,
        ).run()
    return analysis_obj.strategy_map[(timing_str, timing_str)]


def economic_transactions_df(strategy_obj, scale_float=1.0):
    transaction_df = strategy_obj.get_transactions().drop(columns=["order_id"]).reset_index(drop=True).copy()
    transaction_df["amount"] /= scale_float
    transaction_df["price"] *= scale_float
    return transaction_df


@pytest.mark.parametrize("scale_float", [1.0, 40.0, 0.025])
@pytest.mark.parametrize("mode_str,timing_str", [("signal_bar", "next_open"), ("vanilla_current_bar", "same_open")])
@pytest.mark.parametrize("unit_str", ["percent", "value"])
def test_default_timing_matches_vanilla_with_nonzero_dividends(scale_float, mode_str, timing_str, unit_str):
    price_df = pricing_df(scale_float)
    build_fn = factory_fn(price_df, unit_str)
    vanilla_obj = build_fn()
    with contextlib.redirect_stdout(io.StringIO()):
        run_daily(vanilla_obj, price_df, show_progress=False, show_signal_progress_bool=False)
    timing_obj = timing_strategy_obj(price_df, build_fn, timing_str, mode_str)
    pd.testing.assert_frame_equal(
        vanilla_obj.results[["cash", "portfolio_value", "total_value"]],
        timing_obj.results[["cash", "portfolio_value", "total_value"]],
        check_exact=True, check_freq=False,
    )
    pd.testing.assert_frame_equal(economic_transactions_df(vanilla_obj), economic_transactions_df(timing_obj), check_exact=True)
    assert timing_obj.dividend_cash_net_total_float > 0


@pytest.mark.parametrize("timing_str", ["same_open", "same_close_moc", "next_open", "next_close"])
@pytest.mark.parametrize("scale_float", [40.0, 0.025, 1e-10])
@pytest.mark.parametrize("sign_float", [1.0, -1.0])
def test_all_modes_economic_scale_invariance(timing_str, scale_float, sign_float):
    original_df = pricing_df()
    adjusted_df = pricing_df(scale_float)
    original_obj = timing_strategy_obj(original_df, factory_fn(original_df, sign_float=sign_float), timing_str)
    adjusted_obj = timing_strategy_obj(adjusted_df, factory_fn(adjusted_df, sign_float=sign_float), timing_str)
    np.testing.assert_allclose(adjusted_obj.results["total_value"], original_obj.results["total_value"], rtol=1e-13, atol=1e-10)
    pd.testing.assert_frame_equal(economic_transactions_df(adjusted_obj, scale_float), economic_transactions_df(original_obj), rtol=1e-13, atol=1e-10)


def test_split_between_decision_and_fill_preserves_decision_quantity_and_execution_fee():
    price_df = pricing_df(40.0, 0.0)
    # The economic adjusted path is unchanged; raw units halve at execution.
    price_df.loc[price_df.index[1]:, ("AAA", "Unadjusted Close")] /= 2.0
    build_fn = lambda: HistoricalTimingStrategy({price_df.index[0]: 505.0}, "value")
    strategy_obj = timing_strategy_obj(price_df, build_fn)
    transaction_ser = strategy_obj.get_transactions().iloc[0]
    assert transaction_ser["amount"] == 50 * 40.0
    assert transaction_ser["commission"] == 100 * 0.17
    # Sizing must not be redone using the fill's raw close (5.50 dollars).
    assert transaction_ser["amount"] != int(505.0 / 5.5) * 20.0


@pytest.mark.parametrize("scale_float", [40.0, 0.025, 1e-10])
def test_missing_close_liquidation_uses_last_observed_fee_units(scale_float):
    price_df = pricing_df(scale_float, 0.0)
    price_df.loc[price_df.index[3]:, pd.IndexSlice["AAA", ["Open", "High", "Low", "Close", "Unadjusted Close"]]] = np.nan
    build_fn = lambda: HistoricalTimingStrategy({price_df.index[0]: 505.0}, "value")
    strategy_obj = timing_strategy_obj(price_df, build_fn)
    transaction_df = strategy_obj.get_transactions()
    assert len(transaction_df) == 2
    assert transaction_df.iloc[-1]["amount"] == -50 * scale_float
    assert transaction_df.iloc[-1]["commission"] == pytest.approx(50 * 0.17)
    assert transaction_df.iloc[-1]["price"] == 12.0 / scale_float
    assert strategy_obj.get_position("AAA") == 0.0


def scheduled_order_obj(strategy_obj, price_df, unit_str="value", ledger_amount_float=2000.0):
    if unit_str == "shares":
        strategy_obj.order("AAA", ledger_amount_float, trade_id=1)
    else:
        strategy_obj.order_value("AAA", 505.0, trade_id=1)
    order_obj = strategy_obj.get_orders()[-1]
    return ScheduledOrder(order_obj=order_obj, order_kind_str="entry", sequence_int=0,
        signal_bar_ts=price_df.index[0], fill_bar_ts=price_df.index[1],
        fill_price_field_str="Open", fill_phase_str="open",
        sizing_price_float=float(price_df.loc[price_df.index[0], ("AAA", "Close")]),
        sizing_portfolio_value_float=1003.0, sizing_ledger_amount_float=ledger_amount_float)


@pytest.mark.parametrize("invalid_float", [0.0, -1.0, np.inf, -np.inf])
def test_invalid_present_fill_fails_before_transactions(invalid_float):
    price_df = pricing_df(40.0)
    strategy_obj = HistoricalTimingStrategy({})
    intent_obj = scheduled_order_obj(strategy_obj, price_df)
    price_df.loc[price_df.index[1], ("AAA", "Open")] = invalid_float
    with pytest.raises(ValueError, match="Invalid historical timing fill"):
        _process_scheduled_order_list(strategy_obj, price_df, [intent_obj])
    assert strategy_obj.cash == 1003.0
    assert len(strategy_obj.get_transactions()) == 0


@pytest.mark.parametrize("invalid_float", [0.0, -1.0, np.inf, np.nan])
def test_invalid_execution_raw_anchor_fails_before_transactions(invalid_float):
    price_df = pricing_df(40.0)
    strategy_obj = HistoricalTimingStrategy({})
    intent_obj = scheduled_order_obj(strategy_obj, price_df)
    price_df.loc[price_df.index[1], ("AAA", "Unadjusted Close")] = invalid_float
    with pytest.raises(ValueError, match="Historical share sizing requires"):
        _process_scheduled_order_list(strategy_obj, price_df, [intent_obj])
    assert strategy_obj.cash == 1003.0
    assert len(strategy_obj.get_transactions()) == 0


def test_missing_decision_anchor_fails_closed():
    price_df = pricing_df(40.0).drop(columns=[("AAA", "Unadjusted Close")])
    with pytest.raises(ValueError, match="Missing historical share price anchors"):
        timing_strategy_obj(price_df, factory_fn(price_df))


def test_explicit_share_orders_remain_ledger_units():
    price_df = pricing_df(40.0)
    strategy_obj = HistoricalTimingStrategy({})
    strategy_obj._commission_minimum = 0.0
    intent_obj = scheduled_order_obj(strategy_obj, price_df, "shares", 3.0)
    _execute_scheduled_order(strategy_obj, price_df, intent_obj)
    transaction_ser = strategy_obj.get_transactions().iloc[0]
    assert transaction_ser["amount"] == 3.0
    assert transaction_ser["commission"] == 3.0 / 40.0 * 0.17


def test_flag_off_needs_no_raw_anchor_and_preserves_adjusted_lots():
    price_df = pricing_df(40.0, 0.0).drop(columns=[("AAA", "Unadjusted Close")])
    strategy_obj = timing_strategy_obj(price_df, factory_fn(price_df, "value", enabled_bool=False))
    transaction_ser = strategy_obj.get_transactions().iloc[0]
    assert transaction_ser["amount"] == int(505.0 / 0.25)
    assert transaction_ser["commission"] == int(505.0 / 0.25) * 0.17


@pytest.mark.parametrize("timing_str", ["same_open", "same_close_moc", "next_open", "next_close"])
def test_factor_one_is_noop_against_legacy_for_all_modes(timing_str):
    price_df = pricing_df(1.0, 0.0)
    original_obj = timing_strategy_obj(price_df, factory_fn(price_df, enabled_bool=False), timing_str)
    corrected_obj = timing_strategy_obj(price_df, factory_fn(price_df), timing_str)
    pd.testing.assert_frame_equal(corrected_obj.results, original_obj.results, check_exact=True)
    pd.testing.assert_frame_equal(economic_transactions_df(corrected_obj), economic_transactions_df(original_obj), check_exact=True)


def test_target_below_one_raw_share_does_not_trade_or_charge():
    price_df = pricing_df(40.0, 0.0)
    build_fn = lambda: HistoricalTimingStrategy({price_df.index[0]: 5.0}, "value")
    strategy_obj = timing_strategy_obj(price_df, build_fn)
    assert len(strategy_obj.get_transactions()) == 0
    assert strategy_obj.cash == 1003.0


def test_invalid_second_fill_anchor_preflights_entire_batch():
    price_df = pricing_df(40.0)
    for field_str in ("Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"):
        price_df[("BBB", field_str)] = price_df[("AAA", field_str)]
    price_df.loc[price_df.index[1], ("BBB", "Unadjusted Close")] = 0.0
    strategy_obj = HistoricalTimingStrategy({})
    first_intent_obj = scheduled_order_obj(strategy_obj, price_df)
    strategy_obj.order_value("BBB", 505.0, trade_id=2)
    second_intent_obj = ScheduledOrder(order_obj=strategy_obj.get_orders()[-1],
        order_kind_str="entry", sequence_int=1, signal_bar_ts=price_df.index[0],
        fill_bar_ts=price_df.index[1], fill_price_field_str="Open", fill_phase_str="open",
        sizing_price_float=0.25, sizing_portfolio_value_float=1003.0, sizing_ledger_amount_float=2000.0)
    with pytest.raises(ValueError, match="Historical share sizing requires"):
        _process_scheduled_order_list(strategy_obj, price_df, [first_intent_obj, second_intent_obj])
    assert len(strategy_obj.get_transactions()) == 0
    assert strategy_obj.cash == 1003.0


def test_invalid_queued_asset_preflights_before_other_assets_dividend():
    price_df = pricing_df(1.0, 1.0)
    for field_str in ("Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"):
        price_df[("BBB", field_str)] = price_df[("AAA", field_str)]
    price_df.attrs["norgate_adjustment_by_symbol_dict"]["BBB"] = "CAPITALSPECIAL"
    price_df.loc[price_df.index[2], ("BBB", "Unadjusted Close")] = 0.0
    instance_list = []

    class QueuedDividendStrategy(HistoricalTimingStrategy):
        def iterate(self, data, close, open_prices):
            if self.previous_bar == price_df.index[0]:
                self.order_target_percent("AAA", 0.5, trade_id=1)
            elif self.previous_bar == price_df.index[1]:
                self.order_target_percent("BBB", 0.5, trade_id=2)

    def build_fn():
        strategy_obj = QueuedDividendStrategy({})
        instance_list.append(strategy_obj)
        return strategy_obj

    with pytest.raises(ValueError, match="Historical share sizing requires"):
        timing_strategy_obj(price_df, build_fn)
    failed_strategy_obj = instance_list[-1]
    transaction_df = failed_strategy_obj.get_transactions()
    assert len(transaction_df) == 1
    assert failed_strategy_obj.dividend_cash_net_total_float == 0.0
    assert failed_strategy_obj.cash == 1003.0 - float(transaction_df["total_value"].sum()) - float(transaction_df["commission"].sum())
