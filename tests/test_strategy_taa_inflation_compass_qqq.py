from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpha import strategy_registry
from alpha.bench import catalog
from alpha.data import FredSeriesSnapshot
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from scripts.research import run_strategy_analysis as analysis_runner
from strategies.taa_df import strategy_taa_inflation_compass as base_module
from strategies.taa_df import strategy_taa_inflation_compass_qqq as variant_module
from strategies.taa_df.strategy_taa_df import map_month_end_weights_to_rebalance_open_df


MODULE_IMPORT_STR = "strategies.taa_df.strategy_taa_inflation_compass_qqq"


def make_signal_close_df(num_day_int: int = 280) -> pd.DataFrame:
    date_index = pd.bdate_range("2023-01-02", periods=num_day_int)
    price_step_vec = np.arange(num_day_int, dtype=float)
    signal_close_dict = {"SPY": 100.0 + 0.20 * price_step_vec}
    for asset_str in ("XLE", "XLI", "XLF", "XLB"):
        signal_close_dict[asset_str] = 100.0 * np.cumprod(np.full(num_day_int, 0.9990))
    for asset_str in ("XLU", "XLV", "XLP"):
        signal_close_dict[asset_str] = 100.0 * np.cumprod(np.full(num_day_int, 1.0010))
    return pd.DataFrame(signal_close_dict, index=date_index)


def make_execution_price_df(num_day_int: int = 100) -> pd.DataFrame:
    date_index = pd.bdate_range("2023-01-02", periods=num_day_int)
    day_vec = np.arange(num_day_int, dtype=float)
    base_price_dict = {"XLE": 80.0, "QQQ": 350.0, "XLU": 70.0, "XLP": 18.0, "IEF": 85.0, "$SPXTR": 4000.0}
    pricing_data_dict: dict[tuple[str, str], np.ndarray] = {}
    for symbol_str, base_price_float in base_price_dict.items():
        return_vec = 0.0002 + 0.0010 * np.sin(day_vec / 7.0 + base_price_float / 100.0)
        close_vec = base_price_float * np.cumprod(1.0 + return_vec)
        open_vec = close_vec * 0.9995
        pricing_data_dict[(symbol_str, "Open")] = open_vec
        pricing_data_dict[(symbol_str, "High")] = np.maximum(open_vec, close_vec) * 1.001
        pricing_data_dict[(symbol_str, "Low")] = np.minimum(open_vec, close_vec) * 0.999
        pricing_data_dict[(symbol_str, "Close")] = close_vec
        dividend_vec = np.zeros(num_day_int, dtype=float)
        if symbol_str in variant_module.TRADEABLE_ASSET_TUPLE:
            dividend_vec[np.flatnonzero(date_index.is_month_end)] = 0.20
        pricing_data_dict[(symbol_str, "Dividend")] = dividend_vec
    pricing_data_df = pd.DataFrame(pricing_data_dict, index=date_index)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def make_strategy_data_tuple():
    execution_price_df = make_execution_price_df()
    decision_index = pd.DatetimeIndex([pd.Timestamp("2023-01-31"), pd.Timestamp("2023-02-28")])
    month_end_feature_df = pd.DataFrame(
        {"growth_on_bool": [True, False], "inflation_on_bool": [False, False],
         "regime_label_str": ["growth_up__inflation_off", "growth_down__inflation_off"]},
        index=decision_index,
    )
    month_end_weight_df = pd.DataFrame(
        {"XLE": [0.0, 0.0], "QQQ": [1.0, 0.0], "XLU": [0.0, 0.0], "XLP": [0.0, 0.5], "IEF": [0.0, 0.5]},
        index=decision_index,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df, execution_index=execution_price_df.index
    )
    value_ser = pd.Series([2.10], index=pd.DatetimeIndex([pd.Timestamp("2023-01-31")]), name="T5YIE")
    snapshot_obj = FredSeriesSnapshot(
        value_ser=value_ser, source_name_str="FRED", series_id_str="T5YIE",
        download_attempt_timestamp_ts=datetime(2026, 9, 28, tzinfo=UTC), download_status_str="test_snapshot",
        latest_observation_date_ts=pd.Timestamp("2023-01-31"), used_cache_bool=True, freshness_business_days_int=0,
    )
    return execution_price_df, month_end_feature_df, month_end_weight_df, rebalance_weight_df, snapshot_obj


def test_contract_differs_from_parent_only_in_the_goldilocks_holding():
    assert variant_module.TRADEABLE_ASSET_TUPLE == ("XLE", "QQQ", "XLU", "XLP", "IEF")
    variant_config_obj = variant_module.DEFAULT_CONFIG
    parent_config_obj = base_module.DEFAULT_CONFIG
    assert variant_config_obj.goldilocks_asset_str == "QQQ"
    assert variant_config_obj.strategy_name_str == variant_module.STRATEGY_NAME_STR
    changed_field_set = {
        field_str
        for field_str in parent_config_obj.__dataclass_fields__
        if getattr(parent_config_obj, field_str) != getattr(variant_config_obj, field_str)
    }
    assert changed_field_set == {"tradeable_asset_tuple", "goldilocks_asset_str", "strategy_name_str"}
    # The parent keeps the literal source rule.
    assert parent_config_obj.goldilocks_asset_str == "XLK"
    assert parent_config_obj.strategy_name_str == "strategy_taa_inflation_compass"


def test_registry_marks_variant_pm_ready_and_out_of_live():
    assert strategy_registry.tier_for(MODULE_IMPORT_STR) is strategy_registry.MaturityTier.PM_READY
    assert MODULE_IMPORT_STR not in strategy_registry.wired_import_tuple()


def test_goldilocks_regime_holds_qqq_and_other_cells_are_unchanged():
    config_obj = variant_module.DEFAULT_CONFIG
    expected_weight_dict = {
        (True, True): {"XLE": 1.0},
        (True, False): {"QQQ": 1.0},
        (False, True): {"XLU": 1.0},
        (False, False): {"XLP": 0.5, "IEF": 0.5},
    }
    for regime_bool_tuple, nonzero_weight_dict in expected_weight_dict.items():
        _label_str, target_weight_ser = base_module._regime_target_weight_ser(
            growth_on_bool=regime_bool_tuple[0],
            inflation_on_bool=regime_bool_tuple[1],
            tradeable_asset_tuple=config_obj.tradeable_asset_tuple,
            goldilocks_asset_str=config_obj.goldilocks_asset_str,
        )
        assert np.isclose(target_weight_ser.sum(), 1.0)
        assert target_weight_ser[target_weight_ser > 0.0].to_dict() == nonzero_weight_dict


def test_same_signal_as_parent_with_qqq_in_place_of_xlk():
    signal_close_df = make_signal_close_df()
    t5yie_value_ser = pd.Series(2.20, index=signal_close_df.index, name="T5YIE")
    small_window_dict = dict(growth_sma_session_int=20, breakeven_lookback_session_int=5,
                             asset_slope_lookback_session_int=5)
    parent_config_obj = base_module.InflationCompassConfig(**small_window_dict)
    variant_config_obj = base_module.InflationCompassConfig(
        **small_window_dict,
        tradeable_asset_tuple=variant_module.TRADEABLE_ASSET_TUPLE,
        goldilocks_asset_str="QQQ",
    )
    parent_feature_df, parent_weight_df = base_module.compute_month_end_signal_and_weight_df(
        signal_close_df, t5yie_value_ser, parent_config_obj
    )
    variant_feature_df, variant_weight_df = base_module.compute_month_end_signal_and_weight_df(
        signal_close_df, t5yie_value_ser, variant_config_obj
    )
    pd.testing.assert_frame_equal(parent_feature_df, variant_feature_df)
    pd.testing.assert_frame_equal(
        parent_weight_df.rename(columns={"XLK": "QQQ"})[list(variant_module.TRADEABLE_ASSET_TUPLE)],
        variant_weight_df,
    )
    assert variant_weight_df.iloc[-1].to_dict() == {"XLE": 0.0, "QQQ": 1.0, "XLU": 0.0, "XLP": 0.0, "IEF": 0.0}


def test_config_rejects_a_tradeable_set_without_the_goldilocks_asset():
    with pytest.raises(ValueError, match="four regime sleeves"):
        base_module.InflationCompassConfig(goldilocks_asset_str="QQQ")


def test_config_rejects_a_goldilocks_asset_that_collides_with_another_sleeve():
    with pytest.raises(ValueError, match="must differ"):
        base_module.InflationCompassConfig(
            tradeable_asset_tuple=("XLE", "XLU", "XLP", "IEF"), goldilocks_asset_str="XLE"
        )


def test_regime_weights_fail_loud_when_goldilocks_asset_is_not_tradeable():
    with pytest.raises(ValueError, match="not in tradeable_asset_tuple"):
        base_module._regime_target_weight_ser(
            growth_on_bool=True,
            inflation_on_bool=False,
            tradeable_asset_tuple=base_module.TRADEABLE_ASSET_TUPLE,
            goldilocks_asset_str="QQQ",
        )


def test_run_variant_honors_pm_contract_and_uses_variant_name():
    with patch.object(base_module, "get_inflation_compass_data", return_value=make_strategy_data_tuple()) as loader_mock:
        strategy_obj = variant_module.run_variant(
            show_display_bool=False, save_results_bool=False, backtest_start_date_str="2023-02-01",
            capital_base_float=12_345.0, end_date_str="2023-05-19",
        )
    passed_config_obj = loader_mock.call_args.args[0]
    assert passed_config_obj.goldilocks_asset_str == "QQQ"
    assert passed_config_obj.capital_base_float == 12_345.0
    assert passed_config_obj.end_date_str == "2023-05-19"
    assert strategy_obj.name == "strategy_taa_inflation_compass_qqq"
    assert strategy_obj._capital_base == 12_345.0
    assert strategy_obj.tradeable_asset_list == list(variant_module.TRADEABLE_ASSET_TUPLE)
    assert strategy_obj.results.index.min() >= pd.Timestamp("2023-02-01")
    # The first rebalance (growth up, inflation off) buys QQQ, never XLK.
    traded_asset_set = set(strategy_obj._transactions["asset"])
    assert "QQQ" in traded_asset_set
    assert "XLK" not in traded_asset_set


def test_capacity_builder_uses_variant_config_and_moo_contract():
    with patch.object(base_module, "get_inflation_compass_data", return_value=make_strategy_data_tuple()) as loader_mock:
        capacity_input_dict = variant_module.build_capacity_analysis_inputs(capital_base_float=25_000.0)
    assert loader_mock.call_args.args[0].goldilocks_asset_str == "QQQ"
    assert capacity_input_dict["strategy_obj"]._capital_base == 25_000.0
    assert capacity_input_dict["strategy_obj"].name == "strategy_taa_inflation_compass_qqq"
    assert capacity_input_dict["execution_policy_str"] == "MOO"


def test_timing_default_cell_matches_vanilla_next_open_contract():
    with patch.object(base_module, "get_inflation_compass_data", return_value=make_strategy_data_tuple()):
        vanilla_strategy_obj = variant_module.run_variant(show_display_bool=False, save_results_bool=False)
        timing_input_dict = variant_module.build_execution_timing_analysis_inputs()
    timing_result_obj = ExecutionTimingAnalyzer(
        strategy_factory_fn=timing_input_dict["strategy_factory_fn"],
        pricing_data_df=timing_input_dict["pricing_data_df"],
        calendar_idx=timing_input_dict["calendar_idx"],
        entry_timing_str_tuple=("same_open",),
        exit_timing_str_tuple=("same_open",),
        save_output_bool=False,
        order_generation_mode_str=timing_input_dict["order_generation_mode_str"],
        risk_model_str=timing_input_dict["risk_model_str"],
        default_entry_timing_str=timing_input_dict["default_entry_timing_str"],
        default_exit_timing_str=timing_input_dict["default_exit_timing_str"],
    ).run()
    timing_strategy_obj = timing_result_obj.strategy_map[("same_open", "same_open")]
    assert timing_strategy_obj.name == "strategy_taa_inflation_compass_qqq"
    pd.testing.assert_series_equal(
        timing_strategy_obj.results["total_value"], vanilla_strategy_obj.results["total_value"],
        check_names=False, check_freq=False, rtol=0.0, atol=1e-8,
    )


def test_bench_hooks_resolve_and_stress_is_reported_unsupported():
    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    for analysis_str in ("vanilla", "capacity", "timing", "risk"):
        assert analysis_runner._missing_hook_detail_str(variant_module, analysis_str) is None
    assert "unsupported stress strategy key" in analysis_runner._missing_hook_detail_str(variant_module, "stress")
