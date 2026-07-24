import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.bench import catalog
from alpha.engine.backtest import run_daily
from alpha.engine.execution_timing import ExecutionTimingAnalysis
from alpha.engine.order import MarketOrder
from strategies.all_weather import strategy_taa_levered_all_weather as strategy_module
from strategies.all_weather.strategy_taa_levered_all_weather import (
    DEFAULT_CONFIG,
    STRATEGY_NAME_STR,
    LeveredAllWeatherConfig,
    LeveredAllWeatherStrategy,
    build_conservative_covariance_df,
    build_backtest_calendar_idx,
    compute_daily_target_weight_df,
    compute_risk_share_ser,
    compute_target_weight_ser,
    solve_risk_budget_weight_ser,
)


MODULE_IMPORT_STR = (
    "strategies.all_weather.strategy_taa_levered_all_weather"
)


def make_price_close_df(num_day_int: int = 220) -> pd.DataFrame:
    date_index = pd.date_range("2023-01-02", periods=num_day_int, freq="B")
    bar_index_vec = np.arange(num_day_int, dtype=float)
    return_df = pd.DataFrame(
        {
            "SPY": 0.0004 + 0.0100 * np.sin(bar_index_vec / 4.0),
            "TLT": 0.0001 + 0.0060 * np.cos(bar_index_vec / 6.0),
            "DBC": 0.0002 + 0.0080 * np.sin(bar_index_vec / 7.0 + 0.4),
            "GLD": 0.0002 + 0.0050 * np.cos(bar_index_vec / 9.0 + 0.2),
        },
        index=date_index,
        dtype=float,
    )
    return 100.0 * (1.0 + return_df).cumprod()


def make_pricing_data_df(num_day_int: int = 220) -> pd.DataFrame:
    price_close_df = make_price_close_df(num_day_int=num_day_int)
    benchmark_close_ser = 4_000.0 * (
        1.0
        + pd.Series(
            0.0003
            + 0.0090
            * np.sin(np.arange(num_day_int, dtype=float) / 5.0 + 0.1),
            index=price_close_df.index,
            dtype=float,
        )
    ).cumprod()

    close_map = {
        **{
            asset_str: price_close_df[asset_str].to_numpy(dtype=float)
            for asset_str in price_close_df.columns
        },
        "$SPX": benchmark_close_ser.to_numpy(dtype=float),
    }
    pricing_data_dict: dict[tuple[str, str], np.ndarray] = {}
    for symbol_str, close_vec in close_map.items():
        open_vec = close_vec * 0.999
        pricing_data_dict[(symbol_str, "Open")] = open_vec
        pricing_data_dict[(symbol_str, "High")] = np.maximum(open_vec, close_vec) * 1.001
        pricing_data_dict[(symbol_str, "Low")] = np.minimum(open_vec, close_vec) * 0.999
        pricing_data_dict[(symbol_str, "Close")] = close_vec

    pricing_data_df = pd.DataFrame(
        pricing_data_dict,
        index=price_close_df.index,
        dtype=float,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def make_strategy(**override_kwarg_dict) -> LeveredAllWeatherStrategy:
    strategy_kwarg_dict = {
        "name": STRATEGY_NAME_STR,
        "benchmarks": [],
        "asset_tuple": DEFAULT_CONFIG.asset_tuple,
        "risk_budget_tuple": DEFAULT_CONFIG.risk_budget_tuple,
        "covariance_lookback_day_int": 63,
        "target_annualized_volatility_float": 0.15,
        "max_gross_exposure_float": 2.0,
        "max_asset_weight_float": 0.80,
        "annual_financing_rate_float": 0.024,
        "capital_base": 100_000.0,
        "slippage": 0.0,
        "commission_per_share": 0.0,
        "commission_minimum": 0.0,
    }
    strategy_kwarg_dict.update(override_kwarg_dict)
    return LeveredAllWeatherStrategy(**strategy_kwarg_dict)


def make_signal_close_row_ser(
    target_weight_tuple: tuple[float, float, float, float],
    previous_close_float: float = 100.0,
) -> pd.Series:
    row_dict: dict[tuple[str, str], float] = {}
    for asset_str, target_weight_float in zip(
        DEFAULT_CONFIG.asset_tuple,
        target_weight_tuple,
    ):
        row_dict[(asset_str, "Close")] = previous_close_float
        row_dict[(asset_str, "target_weight_ser")] = target_weight_float
    close_row_ser = pd.Series(row_dict, dtype=float)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def test_config_preserves_article_parameters_and_explicit_lookback_assumption():
    assert DEFAULT_CONFIG.asset_tuple == ("SPY", "TLT", "DBC", "GLD")
    assert DEFAULT_CONFIG.risk_budget_tuple == (0.30, 0.30, 0.20, 0.20)
    assert DEFAULT_CONFIG.covariance_lookback_day_int == 63
    assert DEFAULT_CONFIG.target_annualized_volatility_float == pytest.approx(0.15)
    assert DEFAULT_CONFIG.max_gross_exposure_float == pytest.approx(2.0)
    assert DEFAULT_CONFIG.max_asset_weight_float == pytest.approx(0.80)
    assert DEFAULT_CONFIG.annual_financing_rate_float == pytest.approx(0.024)


def test_risk_budget_solver_matches_requested_risk_contributions():
    asset_index = pd.Index(DEFAULT_CONFIG.asset_tuple)
    covariance_df = pd.DataFrame(
        np.diag([0.0004, 0.0001, 0.0009, 0.000225]),
        index=asset_index,
        columns=asset_index,
        dtype=float,
    )

    base_weight_ser = solve_risk_budget_weight_ser(
        covariance_df=covariance_df,
        risk_budget_ser=DEFAULT_CONFIG.risk_budget_ser,
    )
    risk_share_ser = compute_risk_share_ser(
        base_weight_ser=base_weight_ser,
        covariance_df=covariance_df,
    )

    assert float(base_weight_ser.sum()) == pytest.approx(1.0)
    assert (base_weight_ser > 0.0).all()
    np.testing.assert_allclose(
        risk_share_ser.to_numpy(dtype=float),
        DEFAULT_CONFIG.risk_budget_ser.to_numpy(dtype=float),
        atol=1e-8,
    )


def test_conservative_covariance_floors_negative_correlations():
    asset_index = pd.Index(["AAA", "BBB"])
    covariance_df = pd.DataFrame(
        [[0.0004, -0.0001], [-0.0001, 0.0001]],
        index=asset_index,
        columns=asset_index,
        dtype=float,
    )

    conservative_covariance_df = build_conservative_covariance_df(covariance_df)

    assert float(conservative_covariance_df.loc["AAA", "BBB"]) == pytest.approx(0.0)
    assert float(conservative_covariance_df.loc["BBB", "AAA"]) == pytest.approx(0.0)
    assert float(conservative_covariance_df.loc["AAA", "AAA"]) == pytest.approx(0.0004)
    assert float(conservative_covariance_df.loc["BBB", "BBB"]) == pytest.approx(0.0001)


def test_target_weights_respect_gross_and_single_asset_caps():
    asset_index = pd.Index(DEFAULT_CONFIG.asset_tuple)
    covariance_df = pd.DataFrame(
        np.diag([0.0004, 0.00001, 0.0009, 0.000225]),
        index=asset_index,
        columns=asset_index,
        dtype=float,
    )

    target_weight_ser, _base_weight_ser, _base_volatility_float, gross_exposure_float = (
        compute_target_weight_ser(
            covariance_df=covariance_df,
            risk_budget_ser=DEFAULT_CONFIG.risk_budget_ser,
            target_annualized_volatility_float=0.50,
            max_gross_exposure_float=2.0,
            max_asset_weight_float=0.80,
        )
    )

    assert gross_exposure_float <= 2.0
    assert float(target_weight_ser.sum()) <= 2.0 + 1e-12
    assert float(target_weight_ser.max()) <= 0.80 + 1e-12
    assert float(target_weight_ser.max()) == pytest.approx(0.80)


def test_daily_target_weights_do_not_change_when_future_prices_are_appended():
    price_close_df = make_price_close_df(num_day_int=140)
    cutoff_ts = price_close_df.index[109]

    _return_short_df, target_short_df = compute_daily_target_weight_df(
        price_close_df=price_close_df.loc[:cutoff_ts],
        risk_budget_ser=DEFAULT_CONFIG.risk_budget_ser,
        covariance_lookback_day_int=20,
        target_annualized_volatility_float=0.15,
        max_gross_exposure_float=2.0,
        max_asset_weight_float=0.80,
    )
    _return_full_df, target_full_df = compute_daily_target_weight_df(
        price_close_df=price_close_df,
        risk_budget_ser=DEFAULT_CONFIG.risk_budget_ser,
        covariance_lookback_day_int=20,
        target_annualized_volatility_float=0.15,
        max_gross_exposure_float=2.0,
        max_asset_weight_float=0.80,
    )

    pd.testing.assert_series_equal(
        target_short_df.loc[cutoff_ts],
        target_full_df.loc[cutoff_ts],
    )


def test_pre_inception_asset_history_stays_unactionable_without_audit_failure():
    price_close_df = make_price_close_df(num_day_int=80)
    price_close_df.loc[:, "DBC"] = np.nan

    asset_return_df, target_weight_df = compute_daily_target_weight_df(
        price_close_df=price_close_df,
        risk_budget_ser=DEFAULT_CONFIG.risk_budget_ser,
        covariance_lookback_day_int=20,
        target_annualized_volatility_float=0.15,
        max_gross_exposure_float=2.0,
        max_asset_weight_float=0.80,
    )

    assert asset_return_df["DBC"].isna().all()
    assert target_weight_df.isna().all(axis=None)


def test_iterate_rebalances_only_at_quarter_turn_and_sizes_from_previous_close():
    strategy_obj = make_strategy()
    strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
    strategy_obj.current_bar = pd.Timestamp("2024-04-01")
    strategy_obj._total_value_history_list = [100_000.0]
    strategy_obj.total_value = 100_000.0
    close_row_ser = make_signal_close_row_ser(
        (0.40, 0.60, 0.30, 0.20),
        previous_close_float=100.0,
    )
    open_price_ser = pd.Series(
        {asset_str: 200.0 for asset_str in DEFAULT_CONFIG.asset_tuple},
        dtype=float,
    )

    strategy_obj.iterate(
        pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser,
        open_price_ser,
    )

    order_list = strategy_obj.get_orders()
    assert len(order_list) == 4
    assert all(isinstance(order_obj, MarketOrder) for order_obj in order_list)
    assert [order_obj.amount for order_obj in order_list] == [400, 600, 300, 200]
    assert all(order_obj.target for order_obj in order_list)
    assert all(order_obj.unit == "shares" for order_obj in order_list)
    assert float(strategy_obj.current_target_weight_ser["Cash"]) == pytest.approx(-0.50)

    strategy_obj.clear_orders()
    strategy_obj.previous_bar = pd.Timestamp("2024-04-01")
    strategy_obj.current_bar = pd.Timestamp("2024-04-02")
    strategy_obj.iterate(
        pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser,
        open_price_ser,
    )
    assert strategy_obj.get_orders() == []


def test_daily_financing_cost_is_charged_on_actual_negative_cash():
    strategy_obj = make_strategy()
    strategy_obj.previous_bar = pd.Timestamp("2024-04-01")
    strategy_obj.current_bar = pd.Timestamp("2024-04-02")
    strategy_obj.cash = -50_000.0
    strategy_obj.portfolio_value = 150_000.0
    strategy_obj.total_value = 100_000.0
    close_row_ser = make_signal_close_row_ser((0.40, 0.60, 0.30, 0.20))
    expected_cost_float = 50_000.0 * 0.024 / 252.0

    strategy_obj.iterate(
        pd.DataFrame(index=[strategy_obj.previous_bar]),
        close_row_ser,
        pd.Series(dtype=float),
    )

    assert float(strategy_obj.financing_cost_map[strategy_obj.current_bar]) == pytest.approx(
        expected_cost_float
    )
    assert strategy_obj.cash == pytest.approx(-50_000.0 - expected_cost_float)
    assert strategy_obj.total_value == pytest.approx(100_000.0 - expected_cost_float)


def test_run_daily_smoke_generates_quarterly_ledger_and_target_weights():
    pricing_data_df = make_pricing_data_df(num_day_int=220)
    strategy_obj = make_strategy(benchmarks=["$SPX"])
    config_obj = LeveredAllWeatherConfig(
        covariance_lookback_day_int=63,
        start_date_str="2023-01-01",
        capital_base_float=100_000.0,
        slippage_float=0.0,
        commission_per_share_float=0.0,
        commission_minimum_float=0.0,
    )
    first_rebalance_ts = strategy_module.get_first_actionable_rebalance_ts(
        pricing_data_df=pricing_data_df,
        config=config_obj,
    )

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=build_backtest_calendar_idx(
            pricing_data_df=pricing_data_df,
            first_rebalance_ts=first_rebalance_ts,
        ),
        show_progress=False,
        show_signal_progress_bool=False,
        audit_override_bool=None,
    )

    assert strategy_obj.summary is not None
    assert "Strategy" in strategy_obj.summary.columns
    assert len(strategy_obj.get_transactions()) > 0
    assert len(strategy_obj.daily_target_weights) > 0
    target_sum_ser = strategy_obj.daily_target_weights.sum(axis=1)
    np.testing.assert_allclose(
        target_sum_ser.to_numpy(dtype=float),
        np.ones(len(target_sum_ser)),
        atol=1e-12,
    )
    assert (strategy_obj.daily_target_weights["Cash"] < 0.0).any()
    assert float(strategy_obj.financing_cost_ser.sum()) > 0.0
    assert float(strategy_obj.summary.loc["Exposure Time [%]", "Strategy"]) > 90.0
    assert float(
        strategy_obj.summary.loc[
            "Exposure-Adjusted Return (Ann.) [%]",
            "Strategy",
        ]
    ) != 0.0
    assert float(
        strategy_obj.summary.loc["Modeled Financing Cost [$]", "Strategy"]
    ) == pytest.approx(float(strategy_obj.financing_cost_ser.sum()))


def test_run_variant_and_standard_analyzer_hooks_use_same_research_path():
    pricing_data_df = make_pricing_data_df(num_day_int=220)
    test_config_obj = replace_config_for_test()

    with (
        patch.object(strategy_module, "DEFAULT_CONFIG", test_config_obj),
        patch.object(
            strategy_module,
            "get_levered_all_weather_data",
            return_value=pricing_data_df,
        ),
    ):
        strategy_obj = strategy_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
        )
        capacity_input_dict = strategy_module.build_capacity_analysis_inputs(
            show_display_bool=False,
        )
        timing_input_dict = strategy_module.build_execution_timing_analysis_inputs()
        timing_result_obj = ExecutionTimingAnalysis(
            **{
                **timing_input_dict,
                "entry_timing_str_tuple": ("same_open",),
                "exit_timing_str_tuple": ("same_open",),
            },
            save_output_bool=False,
            audit_override_bool=False,
        ).run()

    assert strategy_obj.name == STRATEGY_NAME_STR
    assert capacity_input_dict["strategy_obj"].name == STRATEGY_NAME_STR
    assert capacity_input_dict["execution_policy_str"] == "MOO"
    assert capacity_input_dict["impact_profile_str"] == "MOO_ETF_PROXY"
    assert timing_input_dict["order_generation_mode_str"] == "vanilla_current_bar"
    assert timing_input_dict["risk_model_str"] == "taa_rebalance"
    assert timing_input_dict["default_entry_timing_str"] == "same_open"
    assert timing_input_dict["default_exit_timing_str"] == "same_open"
    timing_strategy_obj = timing_result_obj.strategy_map[("same_open", "same_open")]
    pd.testing.assert_series_equal(
        strategy_obj.results["total_value"],
        timing_strategy_obj.results["total_value"],
        check_names=False,
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        strategy_obj.results["daily_returns"],
        timing_strategy_obj.results["daily_returns"],
        check_names=False,
        check_freq=False,
    )
    assert float(
        timing_strategy_obj.summary.loc["Modeled Financing Cost [$]", "Strategy"]
    ) == pytest.approx(
        float(strategy_obj.summary.loc["Modeled Financing Cost [$]", "Strategy"])
    )


def replace_config_for_test() -> LeveredAllWeatherConfig:
    return LeveredAllWeatherConfig(
        covariance_lookback_day_int=63,
        start_date_str="2023-01-01",
        capital_base_float=100_000.0,
        slippage_float=0.0,
        commission_per_share_float=0.0,
        commission_minimum_float=0.0,
    )


def test_bench_catalog_discovers_all_weather_as_runnable_and_research_only():
    catalog.list_strategies.cache_clear() if hasattr(catalog.list_strategies, "cache_clear") else None
    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)

    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert strategy_entry_obj.has_capacity_analysis_bool is True
    assert strategy_entry_obj.is_wired_bool is False
    assert strategy_entry_obj.category_str == "all_weather"
    assert strategy_entry_obj.category_label_str == "All Weather"
