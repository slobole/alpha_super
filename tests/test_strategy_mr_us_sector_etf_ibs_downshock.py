from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock import (
    DEFAULT_CONFIG,
    SECTOR_ETF_SYMBOL_TUPLE,
    UsSectorEtfIbsDownshockConfig,
    UsSectorEtfIbsDownshockStrategy,
    _run_strategy,
    compute_us_sector_etf_ibs_downshock_signal_df,
    resolve_us_sector_etf_execution_calendar_idx,
)


def make_pricing_data_df(
    price_map_dict: dict[str, dict[str, list[float]]],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    column_map_dict: dict[tuple[str, str], pd.Series] = {}
    for symbol_str, field_map_dict in price_map_dict.items():
        for field_str, value_list in field_map_dict.items():
            column_map_dict[(symbol_str, field_str)] = pd.Series(
                value_list,
                index=date_index,
                dtype=float,
            )
    pricing_data_df = pd.DataFrame(column_map_dict, index=date_index)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def make_close_row_ser(
    row_map_dict: dict[tuple[str, str], object],
) -> pd.Series:
    close_row_ser = pd.Series(row_map_dict)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def make_config(
    symbol_tuple: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    max_positions_int: int = 2,
    capital_base_float: float = 1_000.0,
) -> UsSectorEtfIbsDownshockConfig:
    return replace(
        DEFAULT_CONFIG,
        symbol_tuple=symbol_tuple,
        history_start_date_str="2023-01-01",
        backtest_start_date_str="2024-01-01",
        max_positions_int=max_positions_int,
        capital_base_float=capital_base_float,
    )


def make_strategy(
    config_obj: UsSectorEtfIbsDownshockConfig | None = None,
) -> UsSectorEtfIbsDownshockStrategy:
    if config_obj is None:
        config_obj = make_config()
    return UsSectorEtfIbsDownshockStrategy(
        name="UsSectorEtfIbsDownshockTest",
        benchmarks=[],
        config_obj=config_obj,
    )


def test_default_config_matches_requested_rules_and_engine_cost_defaults():
    strategy_obj = make_strategy(DEFAULT_CONFIG)

    assert DEFAULT_CONFIG.symbol_tuple == SECTOR_ETF_SYMBOL_TUPLE
    assert DEFAULT_CONFIG.entry_ibs_max_float == pytest.approx(0.05)
    assert DEFAULT_CONFIG.downshock_atr_max_float == pytest.approx(-0.5)
    assert DEFAULT_CONFIG.exit_ibs_min_float == pytest.approx(0.90)
    assert DEFAULT_CONFIG.atr_lookback_day_int == 14
    assert DEFAULT_CONFIG.range_median_lookback_day_int == 21
    assert DEFAULT_CONFIG.max_positions_int == 5
    assert strategy_obj.target_weight_float == pytest.approx(1.5 / 11.0)
    assert strategy_obj._slippage == pytest.approx(0.00025)
    assert strategy_obj._commission_per_share == pytest.approx(0.005)
    assert strategy_obj._commission_minimum == pytest.approx(1.0)


def test_strategy_declares_price_and_benchmark_adjustment_provenance():
    strategy_obj = UsSectorEtfIbsDownshockStrategy(
        name="UsSectorEtfAdjustmentProvenanceTest",
        benchmarks=["$SPX"],
        config_obj=DEFAULT_CONFIG,
    )

    assert strategy_obj._performance_benchmark_adjustment_str == "TOTALRETURN"
    assert strategy_obj._data_adjustment_policy_dict == {
        "stock_signal_adjustment_str": "CAPITALSPECIAL",
        "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
        "performance_benchmark_adjustment_str": "TOTALRETURN",
    }


def test_signal_formulas_use_prior_atr_natr_and_prior_21_range_median():
    date_index = pd.bdate_range("2024-01-02", periods=24)
    high_value_list = [101.0] * len(date_index)
    low_value_list = [99.0] * len(date_index)
    close_value_list = [100.0] * len(date_index)
    open_value_list = [100.0] * len(date_index)

    entry_position_int = 22
    high_value_list[entry_position_int] = 100.0
    low_value_list[entry_position_int] = 97.9
    close_value_list[entry_position_int] = 98.0

    exit_position_int = 23
    high_value_list[exit_position_int] = 101.0
    low_value_list[exit_position_int] = 98.0
    close_value_list[exit_position_int] = 100.9

    pricing_data_df = make_pricing_data_df(
        {
            "AAA": {
                "Open": open_value_list,
                "High": high_value_list,
                "Low": low_value_list,
                "Close": close_value_list,
            }
        },
        date_index=date_index,
    )
    config_obj = make_config(symbol_tuple=("AAA",), max_positions_int=1)

    signal_data_df = compute_us_sector_etf_ibs_downshock_signal_df(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )

    entry_date_ts = date_index[entry_position_int]
    prior_log_range_float = float(np.log(101.0 / 99.0))
    expected_current_log_range_float = float(np.log(100.0 / 97.9))

    # *** CRITICAL*** Expected entry volatility is frozen at T-1. The wide
    # current range is intentionally excluded from ATR14 and the range median.
    assert signal_data_df.loc[
        entry_date_ts,
        ("AAA", "prior_atr_14_ser"),
    ] == pytest.approx(2.0)
    assert signal_data_df.loc[
        entry_date_ts,
        ("AAA", "prior_natr_14_ser"),
    ] == pytest.approx(2.0)
    assert signal_data_df.loc[
        entry_date_ts,
        ("AAA", "downshock_atr_ser"),
    ] == pytest.approx(-1.0)
    assert signal_data_df.loc[
        entry_date_ts,
        ("AAA", "prior_range_median_21_ser"),
    ] == pytest.approx(prior_log_range_float)
    assert signal_data_df.loc[
        entry_date_ts,
        ("AAA", "range_ratio_ser"),
    ] == pytest.approx(expected_current_log_range_float / prior_log_range_float)
    assert bool(
        signal_data_df.loc[entry_date_ts, ("AAA", "entry_signal_bool")]
    )

    exit_date_ts = date_index[exit_position_int]
    assert bool(
        signal_data_df.loc[exit_date_ts, ("AAA", "exit_signal_bool")]
    )


def test_iterate_processes_exit_then_uses_prior_natr_ranking_for_one_slot():
    strategy_obj = make_strategy(
        make_config(
            symbol_tuple=("AAA", "BBB", "CCC"),
            max_positions_int=1,
        )
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-08")
    strategy_obj.current_bar = pd.Timestamp("2024-01-09")
    strategy_obj.add_transaction(
        7,
        pd.Timestamp("2024-01-05"),
        "AAA",
        1.0,
        100.0,
        100.0,
        1,
        0.0,
    )
    strategy_obj.current_trade_map["AAA"] = 7

    close_row_ser = make_close_row_ser(
        {
            ("AAA", "exit_signal_bool"): True,
            ("AAA", "entry_signal_bool"): False,
            ("BBB", "entry_signal_bool"): True,
            ("BBB", "prior_natr_14_ser"): 3.0,
            ("BBB", "Close"): 100.0,
            ("CCC", "entry_signal_bool"): True,
            ("CCC", "prior_natr_14_ser"): 5.0,
            ("CCC", "Close"): 80.0,
        }
    )

    strategy_obj.iterate(
        pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser,
        pd.Series(dtype=float),
    )

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 2
    assert all(isinstance(order_obj, MarketOrder) for order_obj in order_list)
    assert order_list[0].asset == "AAA"
    assert order_list[0].amount == pytest.approx(0.0)
    assert order_list[0].trade_id == 7
    assert order_list[1].asset == "CCC"
    assert order_list[1].amount == pytest.approx(
        strategy_obj.previous_total_value * (1.5 / 11.0) / 80.0
    )


def test_execution_calendar_starts_after_first_ready_decision_close():
    date_index = pd.bdate_range("2024-01-02", periods=24)
    pricing_data_df = make_pricing_data_df(
        {
            "AAA": {
                "Open": [100.0] * len(date_index),
                "High": [101.0] * len(date_index),
                "Low": [99.0] * len(date_index),
                "Close": [100.0] * len(date_index),
            }
        },
        date_index=date_index,
    )
    config_obj = make_config(symbol_tuple=("AAA",), max_positions_int=1)

    execution_calendar_idx = resolve_us_sector_etf_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )

    # *** CRITICAL*** Date 21 is the first close with 21 prior ranges. Its
    # OHLC cannot authorize a fill at that same session's already-passed open.
    assert execution_calendar_idx[0] == date_index[22]


def test_production_run_fills_first_ready_close_signal_at_next_open():
    date_index = pd.bdate_range("2024-01-02", periods=24)
    high_value_list = [101.0] * len(date_index)
    low_value_list = [99.0] * len(date_index)
    close_value_list = [100.0] * len(date_index)
    open_value_list = [100.0] * len(date_index)

    first_ready_signal_position_int = 21
    first_fill_position_int = 22
    high_value_list[first_ready_signal_position_int] = 100.0
    low_value_list[first_ready_signal_position_int] = 97.9
    close_value_list[first_ready_signal_position_int] = 98.0
    open_value_list[first_fill_position_int] = 110.0
    high_value_list[first_fill_position_int] = 111.0
    pricing_data_df = make_pricing_data_df(
        {
            "AAA": {
                "Open": open_value_list,
                "High": high_value_list,
                "Low": low_value_list,
                "Close": close_value_list,
            },
            "$SPX": {
                "Close": [4_700.0] * len(date_index),
            },
        },
        date_index=date_index,
    )
    config_obj = make_config(
        symbol_tuple=("AAA",),
        max_positions_int=1,
        capital_base_float=10_000.0,
    )

    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        show_display_bool=False,
        audit_override_bool=False,
    )

    transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
    assert len(transaction_df) == 1
    assert pd.Timestamp(transaction_df.loc[0, "bar"]) == date_index[
        first_fill_position_int
    ]


def test_iterate_does_not_rebalance_an_existing_position():
    strategy_obj = make_strategy()
    strategy_obj.previous_bar = pd.Timestamp("2024-01-08")
    strategy_obj.current_bar = pd.Timestamp("2024-01-09")
    strategy_obj.add_transaction(
        11,
        pd.Timestamp("2024-01-05"),
        "AAA",
        1.0,
        100.0,
        100.0,
        1,
        0.0,
    )
    strategy_obj.current_trade_map["AAA"] = 11
    close_row_ser = make_close_row_ser(
        {
            ("AAA", "exit_signal_bool"): False,
            ("AAA", "entry_signal_bool"): True,
            ("AAA", "prior_natr_14_ser"): 9.0,
        }
    )

    strategy_obj.iterate(
        pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser,
        pd.Series(dtype=float),
    )

    assert strategy_obj.get_orders() == []


def test_run_daily_fills_close_t_signal_at_open_t_plus_1_with_default_costs():
    date_index = pd.bdate_range("2024-01-02", periods=24)
    high_value_list = [101.0] * len(date_index)
    low_value_list = [99.0] * len(date_index)
    close_value_list = [100.0] * len(date_index)
    open_value_list = [100.0] * len(date_index)

    signal_position_int = 22
    fill_position_int = 23
    high_value_list[signal_position_int] = 100.0
    low_value_list[signal_position_int] = 97.9
    close_value_list[signal_position_int] = 98.0
    open_value_list[fill_position_int] = 110.0

    pricing_data_df = make_pricing_data_df(
        {
            "AAA": {
                "Open": open_value_list,
                "High": high_value_list,
                "Low": low_value_list,
                "Close": close_value_list,
            }
        },
        date_index=date_index,
    )
    strategy_obj = make_strategy(
        make_config(
            symbol_tuple=("AAA",),
            max_positions_int=1,
            capital_base_float=10_000.0,
        )
    )

    run_daily(
        strategy_obj,
        pricing_data_df,
        date_index,
        show_progress=False,
        show_signal_progress_bool=False,
        audit_override_bool=False,
    )

    transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
    assert len(transaction_df) == 1
    entry_row_ser = transaction_df.iloc[0]
    expected_target_share_float = (
        strategy_obj._capital_base * (1.5 / 11.0) / 98.0
    )
    assert pd.Timestamp(entry_row_ser["bar"]) == date_index[fill_position_int]
    assert entry_row_ser["asset"] == "AAA"
    assert float(entry_row_ser["price"]) == pytest.approx(
        110.0 * (1.0 + 0.00025)
    )
    assert float(entry_row_ser["amount"]) == pytest.approx(
        expected_target_share_float
    )
