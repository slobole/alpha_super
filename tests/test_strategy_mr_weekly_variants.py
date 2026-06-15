from __future__ import annotations

from collections import defaultdict

import pandas as pd

from alpha.engine.order import MarketOrder
from strategies.dv2.strategy_mr_dv2 import default_trade_id_int as dv2_default_trade_id_int
from strategies.dv2.strategy_mr_dv2_weekly import (
    WeeklyDVO2Strategy,
    weekly_decision_marker_field_str as dv2_weekly_marker_field_str,
    weekly_dv2_field_str,
    weekly_natr_field_str,
    weekly_previous_high_field_str,
    weekly_return_field_str,
    weekly_sma_field_str,
)
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import (
    default_trade_id_int as qpi_default_trade_id_int,
)
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit_weekly import (
    WeeklyQPIIbsRsiExitStrategy,
    three_week_return_field_str,
    weekly_decision_marker_field_str as qpi_weekly_marker_field_str,
    weekly_ibs_field_str,
    weekly_qpi_field_str,
    weekly_rsi2_field_str,
    weekly_sma_field_str as qpi_weekly_sma_field_str,
    weekly_turnover_field_str,
)
from strategies.weekly_bar_utils import build_completed_week_ohlcv_df


def make_close_row_ser(row_dict: dict[tuple[str, str], object]) -> pd.Series:
    close_row_ser = pd.Series(row_dict)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def test_build_completed_week_ohlcv_uses_actual_completed_week_dates():
    date_index = pd.to_datetime(
        [
            "2024-03-25",
            "2024-03-26",
            "2024-03-27",
            "2024-03-28",
            "2024-04-01",
            "2024-04-02",
            "2024-04-03",
            "2024-04-04",
            "2024-04-05",
        ]
    )
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            ("AAA", "High"): [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            ("AAA", "Low"): [9.0, 8.0, 10.0, 11.0, 19.0, 18.0, 20.0, 21.0, 22.0],
            ("AAA", "Close"): [10.5, 11.5, 12.5, 13.5, 20.5, 21.5, 22.5, 23.5, 24.5],
            ("AAA", "Turnover"): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

    weekly_bar_df = build_completed_week_ohlcv_df(pricing_data_df)

    assert weekly_bar_df.index.tolist() == [pd.Timestamp("2024-03-28"), pd.Timestamp("2024-04-05")]
    assert float(weekly_bar_df.loc["2024-03-28", ("AAA", "Open")]) == 10.0
    assert float(weekly_bar_df.loc["2024-03-28", ("AAA", "High")]) == 14.0
    assert float(weekly_bar_df.loc["2024-03-28", ("AAA", "Low")]) == 8.0
    assert float(weekly_bar_df.loc["2024-03-28", ("AAA", "Close")]) == 13.5
    assert float(weekly_bar_df.loc["2024-03-28", ("AAA", "Turnover")]) == 10.0


def test_dv2_weekly_iterate_ignores_non_weekly_decision_rows():
    strategy_obj = WeeklyDVO2Strategy(
        name="WeeklyDV2Test",
        benchmarks=[],
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-04-04")
    strategy_obj.current_bar = pd.Timestamp("2024-04-05")
    strategy_obj.current_trade = defaultdict(dv2_default_trade_id_int)
    strategy_obj.add_transaction(3, pd.Timestamp("2024-04-01"), "AAA", 10.0, 100.0, 1_000.0, 1, 0.0)
    strategy_obj.current_trade["AAA"] = 3

    close_row_ser = make_close_row_ser(
        {
            ("AAA", "Close"): 110.0,
            ("AAA", weekly_previous_high_field_str): 100.0,
        }
    )

    strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_row_ser, pd.Series(dtype=float))

    assert strategy_obj.get_orders() == []


def test_dv2_weekly_iterate_uses_previous_week_high_for_exit():
    strategy_obj = WeeklyDVO2Strategy(
        name="WeeklyDV2Test",
        benchmarks=[],
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-04-05")
    strategy_obj.current_bar = pd.Timestamp("2024-04-08")
    strategy_obj.current_trade = defaultdict(dv2_default_trade_id_int)
    strategy_obj.add_transaction(7, pd.Timestamp("2024-04-01"), "AAA", 10.0, 100.0, 1_000.0, 1, 0.0)
    strategy_obj.current_trade["AAA"] = 7
    strategy_obj.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy_obj.previous_bar])

    close_row_ser = make_close_row_ser(
        {
            ("AAA", "Close"): 110.0,
            ("AAA", weekly_previous_high_field_str): 105.0,
            ("AAA", dv2_weekly_marker_field_str): True,
        }
    )

    strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), close_row_ser, pd.Series(dtype=float))

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 1
    assert isinstance(order_list[0], MarketOrder)
    assert order_list[0].asset == "AAA"
    assert order_list[0].amount == 0.0
    assert order_list[0].target is True
    assert order_list[0].trade_id == 7


def test_dv2_weekly_opportunities_use_weekly_fields_and_pit_universe():
    strategy_obj = WeeklyDVO2Strategy(
        name="WeeklyDV2Test",
        benchmarks=[],
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-04-05")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1], "OUT": [0]},
        index=[strategy_obj.previous_bar],
    )
    close_row_ser = make_close_row_ser(
        {
            ("AAA", "Close"): 105.0,
            ("AAA", weekly_return_field_str): 0.08,
            ("AAA", weekly_natr_field_str): 2.0,
            ("AAA", weekly_dv2_field_str): 9.0,
            ("AAA", weekly_sma_field_str): 100.0,
            ("BBB", "Close"): 106.0,
            ("BBB", weekly_return_field_str): 0.09,
            ("BBB", weekly_natr_field_str): 3.0,
            ("BBB", weekly_dv2_field_str): 8.0,
            ("BBB", weekly_sma_field_str): 100.0,
            ("OUT", "Close"): 107.0,
            ("OUT", weekly_return_field_str): 0.10,
            ("OUT", weekly_natr_field_str): 9.0,
            ("OUT", weekly_dv2_field_str): 7.0,
            ("OUT", weekly_sma_field_str): 100.0,
        }
    )

    opportunity_list = strategy_obj.get_opportunities(close_row_ser)

    assert opportunity_list == ["BBB", "AAA"]


def test_qpi_weekly_opportunities_use_weekly_fields_and_turnover_ranking():
    strategy_obj = WeeklyQPIIbsRsiExitStrategy(
        name="WeeklyQPITest",
        benchmarks=[],
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
        qpi_lookback_years_int=1,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-04-05")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1], "HIGHQ": [1], "OUT": [0]},
        index=[strategy_obj.previous_bar],
    )
    close_row_ser = make_close_row_ser(
        {
            ("AAA", "Close"): 105.0,
            ("AAA", weekly_turnover_field_str): 30_000_000.0,
            ("AAA", weekly_qpi_field_str): 10.0,
            ("AAA", qpi_weekly_sma_field_str): 100.0,
            ("AAA", three_week_return_field_str): -0.03,
            ("AAA", weekly_ibs_field_str): 0.05,
            ("BBB", "Close"): 106.0,
            ("BBB", weekly_turnover_field_str): 50_000_000.0,
            ("BBB", weekly_qpi_field_str): 8.0,
            ("BBB", qpi_weekly_sma_field_str): 100.0,
            ("BBB", three_week_return_field_str): -0.04,
            ("BBB", weekly_ibs_field_str): 0.08,
            ("HIGHQ", "Close"): 107.0,
            ("HIGHQ", weekly_turnover_field_str): 90_000_000.0,
            ("HIGHQ", weekly_qpi_field_str): 40.0,
            ("HIGHQ", qpi_weekly_sma_field_str): 100.0,
            ("HIGHQ", three_week_return_field_str): -0.05,
            ("HIGHQ", weekly_ibs_field_str): 0.05,
            ("OUT", "Close"): 108.0,
            ("OUT", weekly_turnover_field_str): 80_000_000.0,
            ("OUT", weekly_qpi_field_str): 5.0,
            ("OUT", qpi_weekly_sma_field_str): 100.0,
            ("OUT", three_week_return_field_str): -0.06,
            ("OUT", weekly_ibs_field_str): 0.05,
        }
    )

    opportunity_list = strategy_obj.get_opportunity_list(close_row_ser)

    assert opportunity_list == ["BBB", "AAA"]


def test_qpi_weekly_iterate_submits_exit_only_on_weekly_decision_row():
    strategy_obj = WeeklyQPIIbsRsiExitStrategy(
        name="WeeklyQPITest",
        benchmarks=[],
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-04-05")
    strategy_obj.current_bar = pd.Timestamp("2024-04-08")
    strategy_obj.current_trade_map = defaultdict(qpi_default_trade_id_int)
    strategy_obj.add_transaction(11, pd.Timestamp("2024-04-01"), "AAA", 10.0, 100.0, 1_000.0, 1, 0.0)
    strategy_obj.current_trade_map["AAA"] = 11

    non_decision_close_row_ser = make_close_row_ser(
        {
            ("AAA", weekly_ibs_field_str): 0.95,
            ("AAA", weekly_rsi2_field_str): 50.0,
        }
    )
    strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), non_decision_close_row_ser, pd.Series(dtype=float))
    assert strategy_obj.get_orders() == []

    decision_close_row_ser = make_close_row_ser(
        {
            ("AAA", weekly_ibs_field_str): 0.95,
            ("AAA", weekly_rsi2_field_str): 50.0,
            ("AAA", qpi_weekly_marker_field_str): True,
        }
    )
    strategy_obj.iterate(pd.DataFrame(index=[strategy_obj.previous_bar]), decision_close_row_ser, pd.Series(dtype=float))

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 1
    assert order_list[0].asset == "AAA"
    assert order_list[0].amount == 0.0
    assert order_list[0].target is True
    assert order_list[0].trade_id == 11
