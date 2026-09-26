"""Historical whole-share sizing and fee units; synthetic, no vendor access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pickle

from alpha.engine.strategy import Strategy


class HistoricalUnitsStrategy(Strategy):
    def compute_signals(self, pricing_data_df):
        return pricing_data_df

    def iterate(self, data_df, close_row_ser, open_price_ser):
        pass


def pricing_frame_df(price_scale_float=1.0):
    session_idx = pd.bdate_range("2013-06-27", periods=4)
    raw_close_vec = np.array([14.04, 15.0, 16.0, 17.0])
    raw_open_vec = np.array([14.04, 14.8, 15.9, 16.9])
    return pd.DataFrame({
        ("ASSET", "Open"): raw_open_vec / price_scale_float,
        ("ASSET", "High"): (raw_close_vec + 1.0) / price_scale_float,
        ("ASSET", "Low"): (raw_open_vec - 1.0) / price_scale_float,
        ("ASSET", "Close"): raw_close_vec / price_scale_float,
        ("ASSET", "Unadjusted Close"): raw_close_vec,
    }, index=session_idx)


def strategy_obj(historical_bool=True):
    result_obj = HistoricalUnitsStrategy(
        "historical_units", [], capital_base=10_000.0,
        slippage=0.00025, commission_per_share=0.005, commission_minimum=1.0,
    )
    result_obj.historical_share_units_bool = historical_bool
    result_obj.configure_dividend_cash_ledger(enabled_bool=False)
    return result_obj


def set_bar(strategy_instance, pricing_df, position_int):
    strategy_instance.previous_bar = pricing_df.index[position_int - 1]
    strategy_instance.current_bar = pricing_df.index[position_int]
    strategy_instance._total_value_history_list.append(strategy_instance.total_value)


def round_trip_obj(price_scale_float, historical_bool=True):
    pricing_df = pricing_frame_df(price_scale_float)
    run_obj = strategy_obj(historical_bool)
    nav_list = []
    for position_int in (1, 2, 3):
        set_bar(run_obj, pricing_df, position_int)
        if position_int == 1:
            run_obj.order_target_value("ASSET", 5_000.0, trade_id=1)
        elif position_int == 2:
            run_obj.order_target_percent("ASSET", 0.5, trade_id=1)
        else:
            run_obj.order_target("ASSET", 0.0, trade_id=1)
        run_obj.process_orders(pricing_df)
        nav_list.append(run_obj.total_value)
    return run_obj, np.array(nav_list)


@pytest.mark.parametrize("price_scale_float", [40.0, 0.025, 0.1, 3.7])
def test_future_adjustment_scale_preserves_orders_nav_and_commission(price_scale_float):
    base_obj, base_nav_vec = round_trip_obj(1.0)
    scaled_obj, scaled_nav_vec = round_trip_obj(price_scale_float)
    base_transaction_df = base_obj.get_transactions()
    scaled_transaction_df = scaled_obj.get_transactions()
    np.testing.assert_allclose(scaled_nav_vec, base_nav_vec, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        scaled_transaction_df.amount / price_scale_float, base_transaction_df.amount,
        rtol=1e-12, atol=1e-10,
    )
    np.testing.assert_allclose(scaled_transaction_df.commission, base_transaction_df.commission, rtol=1e-12)
    np.testing.assert_allclose(scaled_transaction_df.total_value, base_transaction_df.total_value, rtol=1e-12)
    assert base_transaction_df.iloc[0].amount == 356.0
    assert base_transaction_df.iloc[0].commission == pytest.approx(1.78)
    assert scaled_obj.get_position("ASSET") == pytest.approx(0.0, abs=1e-10)
    assert scaled_obj._accounting_policy_dict["special_distribution_share_policy_str"] == "embedded_adjusted_reinvestment_not_physical_share_replay"


@pytest.mark.parametrize("unit_str", ["value", "percent"])
@pytest.mark.parametrize("target_bool", [True, False])
@pytest.mark.parametrize("direction_float", [1.0, -1.0])
def test_value_percent_target_and_delta_units(unit_str, target_bool, direction_float):
    pricing_df = pricing_frame_df(0.025)
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.add_transaction(1, pricing_df.index[0], "ASSET", 10.0 * 0.025, 14.04 / 0.025, 140.4, 0)
    if unit_str == "value":
        run_obj.order_value("ASSET", direction_float * 5_000.0, target=target_bool, trade_id=1)
    else:
        run_obj.order_percent("ASSET", direction_float * 0.5, target=target_bool, trade_id=1)
    run_obj.process_orders(pricing_df)
    expected_amount_float = direction_float * 356.0 * 0.025 - (0.25 if target_bool else 0.0)
    transaction_row_ser = run_obj.get_transactions().iloc[-1]
    assert transaction_row_ser.amount == pytest.approx(expected_amount_float)
    assert transaction_row_ser.commission == pytest.approx(max(1.0, abs(expected_amount_float / 0.025) * 0.005))


@pytest.mark.parametrize("execution_scale_float,expected_fee_float", [(20.0, 3.56), (80.0, 1.0)])
def test_split_between_decision_and_fill_uses_prior_whole_shares(execution_scale_float, expected_fee_float):
    pricing_df = pricing_frame_df(40.0)
    # Keep the adjusted economic path fixed. Raw fill-day units reflect a
    # 2-for-1 or 1-for-2 split; quantities were fixed from the prior raw close.
    pricing_df.loc[pricing_df.index[1], ("ASSET", "Unadjusted Close")] = 15.0 / 40.0 * execution_scale_float
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    run_obj.process_orders(pricing_df)
    transaction_row_ser = run_obj.get_transactions().iloc[0]
    assert transaction_row_ser.amount == 356.0 * 40.0
    assert transaction_row_ser.commission == pytest.approx(expected_fee_float)
    assert transaction_row_ser.total_value == pytest.approx(356.0 * 14.8 * 1.00025)


def test_current_open_gap_does_not_change_sized_quantity():
    amount_list = []
    for open_multiplier_float in (0.25, 4.0):
        pricing_df = pricing_frame_df(40.0)
        pricing_df.loc[pricing_df.index[1], ("ASSET", "Open")] *= open_multiplier_float
        run_obj = strategy_obj()
        set_bar(run_obj, pricing_df, 1)
        run_obj.order_percent("ASSET", 0.5, trade_id=1)
        run_obj.process_orders(pricing_df)
        amount_list.append(run_obj.get_transactions().iloc[0].amount)
    assert amount_list == [14_240.0, 14_240.0]


def test_unit_scale_is_noop_against_default_path():
    legacy_obj, legacy_nav_vec = round_trip_obj(1.0, historical_bool=False)
    corrected_obj, corrected_nav_vec = round_trip_obj(1.0)
    np.testing.assert_array_equal(legacy_nav_vec, corrected_nav_vec)
    comparison_column_list = ["amount", "price", "total_value", "commission"]
    pd.testing.assert_frame_equal(
        legacy_obj.get_transactions()[comparison_column_list],
        corrected_obj.get_transactions()[comparison_column_list], check_dtype=False,
    )


def test_default_path_requires_no_raw_anchor_and_preserves_legacy_rounding():
    pricing_df = pricing_frame_df(40.0).drop(columns=[("ASSET", "Unadjusted Close")])
    run_obj = strategy_obj(historical_bool=False)
    assert HistoricalUnitsStrategy("default", []).historical_share_units_bool is False
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    run_obj.process_orders(pricing_df)
    transaction_row_ser = run_obj.get_transactions().iloc[0]
    assert transaction_row_ser.amount == int(5_000.0 / (14.04 / 40.0))
    assert transaction_row_ser.commission == pytest.approx(transaction_row_ser.amount * 0.005)


def test_old_pickled_strategy_without_instance_flag_retains_legacy_execution():
    pricing_df = pricing_frame_df(40.0).drop(columns=[("ASSET", "Unadjusted Close")])
    legacy_obj = strategy_obj(historical_bool=False)
    del legacy_obj.historical_share_units_bool
    restored_obj = pickle.loads(pickle.dumps(legacy_obj))
    assert "historical_share_units_bool" not in restored_obj.__dict__
    assert restored_obj.historical_share_units_bool is False
    set_bar(restored_obj, pricing_df, 1)
    restored_obj.order_value("ASSET", 5_000.0, trade_id=1)
    restored_obj.process_orders(pricing_df)
    transaction_row_ser = restored_obj.get_transactions().iloc[0]
    expected_amount_float = int(5_000.0 / (14.04 / 40.0))
    assert transaction_row_ser.amount == expected_amount_float
    assert transaction_row_ser.commission == pytest.approx(expected_amount_float * 0.005)


@pytest.mark.parametrize("order_kind_str", ["market", "limit", "stop"])
def test_share_orders_keep_adjusted_ledger_semantics_and_correct_fees(order_kind_str):
    pricing_df = pricing_frame_df(40.0)
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    order_keyword_dict = {}
    if order_kind_str == "limit":
        order_keyword_dict["limit_price"] = 15.0 / 40.0
    elif order_kind_str == "stop":
        order_keyword_dict["stop_price"] = 14.5 / 40.0
    run_obj.order("ASSET", 14_240.0, trade_id=1, **order_keyword_dict)
    run_obj.process_orders(pricing_df)
    transaction_row_ser = run_obj.get_transactions().iloc[0]
    assert transaction_row_ser.amount == 14_240.0
    assert transaction_row_ser.commission == pytest.approx(1.78)


@pytest.mark.parametrize("field_str,position_int,bad_value_float", [
    ("Unadjusted Close", 0, 0.0), ("Unadjusted Close", 0, -1.0),
    ("Unadjusted Close", 0, np.nan), ("Unadjusted Close", 1, np.inf),
    ("Unadjusted Close", 1, 0.0), ("Close", 0, 0.0),
    ("Close", 0, np.nan), ("Close", 1, np.nan),
])
def test_invalid_anchors_fail_before_transactions_or_cash_mutate(field_str, position_int, bad_value_float):
    pricing_df = pricing_frame_df(40.0)
    pricing_df.loc[pricing_df.index[position_int], ("ASSET", field_str)] = bad_value_float
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    with pytest.raises(ValueError, match="anchor|scale|finite"):
        run_obj.process_orders(pricing_df)
    assert run_obj.get_transactions().empty
    assert run_obj.cash == 10_000.0
    assert run_obj.get_position("ASSET") == 0.0


def test_all_order_anchors_are_checked_before_any_fill_or_dividend():
    pricing_df = pricing_frame_df(40.0)
    for field_str in ("Open", "High", "Low", "Close", "Unadjusted Close"):
        pricing_df[("BROKEN", field_str)] = pricing_df[("ASSET", field_str)]
    pricing_df.loc[pricing_df.index[1], ("BROKEN", "Close")] = np.nan
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    run_obj.order_value("BROKEN", 5_000.0, trade_id=2)
    dividend_call_list = []
    run_obj._credit_dividend_cash_before_open = lambda price_df: dividend_call_list.append(True)
    with pytest.raises(ValueError, match="anchor"):
        run_obj.process_orders(pricing_df)
    assert not dividend_call_list
    assert run_obj.get_transactions().empty
    assert run_obj.cash == 10_000.0


def test_missing_anchor_column_and_decision_date_fail_closed():
    pricing_df = pricing_frame_df(40.0)
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    with pytest.raises(ValueError, match="Missing historical share price anchors"):
        run_obj.process_orders(pricing_df.drop(columns=[("ASSET", "Unadjusted Close")]))
    run_obj.previous_bar = None
    with pytest.raises(ValueError, match="anchor date"):
        run_obj.process_orders(pricing_df)
    assert run_obj.get_transactions().empty


@pytest.mark.parametrize("price_scale_float", [40.0, 0.025, 1e-10])
def test_stale_liquidation_uses_last_observed_anchor(price_scale_float):
    pricing_df = pricing_frame_df(price_scale_float)
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    run_obj.process_orders(pricing_df)
    set_bar(run_obj, pricing_df, 2)
    pricing_df.loc[pricing_df.index[2], ("ASSET", "Open")] = np.nan
    pricing_df.loc[pricing_df.index[2], ("ASSET", "Close")] = np.nan
    pricing_df.loc[pricing_df.index[2], ("ASSET", "Unadjusted Close")] = np.nan
    run_obj.process_orders(pricing_df)
    transaction_df = run_obj.get_transactions()
    assert len(transaction_df) == 2
    assert transaction_df.iloc[1].amount == pytest.approx(-356.0 * price_scale_float)
    assert transaction_df.iloc[1].commission == pytest.approx(1.78)
    assert transaction_df.iloc[1].total_value == pytest.approx(-356.0 * 15.0)
    assert run_obj.get_position("ASSET") == 0.0


def test_below_one_raw_share_is_canceled_without_commission():
    pricing_df = pricing_frame_df(40.0)
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 10.0, trade_id=1)
    run_obj.process_orders(pricing_df)
    assert run_obj.get_transactions().empty
    assert not run_obj.get_orders()
    assert run_obj.cash == 10_000.0


@pytest.mark.parametrize("price_scale_float", [1.0, 1e-10, 1e10])
@pytest.mark.parametrize("raw_share_float,expected_cash_float", [(1.0, 0.75), (-1.0, -1.0)])
def test_material_dividend_is_not_discarded_when_one_factor_is_tiny(
    price_scale_float, raw_share_float, expected_cash_float,
):
    pricing_df = pricing_frame_df(price_scale_float)
    pricing_df[("ASSET", "Dividend")] = 0.0
    pricing_df.loc[pricing_df.index[0], ("ASSET", "Dividend")] = 1.0 / price_scale_float
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"ASSET": "CAPITALSPECIAL"}
    run_obj = strategy_obj()
    run_obj.configure_dividend_cash_ledger(enabled_bool=True, withholding_rate_float=0.25)
    set_bar(run_obj, pricing_df, 1)
    run_obj.add_transaction(
        1, pricing_df.index[0], "ASSET", raw_share_float * price_scale_float,
        14.04 / price_scale_float, raw_share_float * 14.04, 0,
    )
    initial_cash_float = run_obj.cash
    run_obj.process_orders(pricing_df)
    assert run_obj.cash - initial_cash_float == pytest.approx(expected_cash_float)
    assert len(run_obj._dividend_ledger_row_dict_list) == 1


@pytest.mark.parametrize("invalid_open_float", [np.inf, -np.inf, 0.0, -1.0])
def test_present_invalid_execution_open_fails_before_any_cash_or_position_mutation(invalid_open_float):
    pricing_df = pricing_frame_df(40.0)
    pricing_df.loc[pricing_df.index[1], ("ASSET", "Open")] = invalid_open_float
    run_obj = strategy_obj()
    set_bar(run_obj, pricing_df, 1)
    run_obj.order_value("ASSET", 5_000.0, trade_id=1)
    dividend_call_list = []
    run_obj._credit_dividend_cash_before_open = lambda price_df: dividend_call_list.append(True)
    with pytest.raises(ValueError, match="execution Open"):
        run_obj.process_orders(pricing_df)
    assert not dividend_call_list
    assert run_obj.get_transactions().empty
    assert run_obj.get_position("ASSET") == 0.0
    assert run_obj.cash == 10_000.0
