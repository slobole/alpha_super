from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpha import strategy_registry
from alpha.bench import catalog
from alpha.data import FredSeriesSnapshot
from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from alpha.engine.report import _strategy_metadata_dict
from scripts.research import run_strategy_analysis as analysis_runner
from strategies.taa_df import strategy_taa_inflation_compass as variant_module
from strategies.taa_df.strategy_taa_df import map_month_end_weights_to_rebalance_open_df


MODULE_IMPORT_STR = "strategies.taa_df.strategy_taa_inflation_compass"


def make_signal_close_df(num_day_int: int = 280) -> pd.DataFrame:
    date_index = pd.bdate_range("2023-01-02", periods=num_day_int)
    price_step_vec = np.arange(num_day_int, dtype=float)
    signal_close_dict: dict[str, np.ndarray] = {
        "SPY": 100.0 + 0.20 * price_step_vec,
        "XLE": 100.0 * np.cumprod(np.full(num_day_int, 1.0010)),
        "XLI": 100.0 * np.cumprod(np.full(num_day_int, 1.0008)),
        "XLF": 100.0 * np.cumprod(np.full(num_day_int, 1.0007)),
        "XLB": 100.0 * np.cumprod(np.full(num_day_int, 1.0006)),
        "XLU": 100.0 * np.cumprod(np.full(num_day_int, 1.0001)),
        "XLV": 100.0 * np.cumprod(np.full(num_day_int, 1.0001)),
        "XLP": 100.0 * np.cumprod(np.full(num_day_int, 1.0001)),
    }
    return pd.DataFrame(signal_close_dict, index=date_index)


def make_execution_price_df(num_day_int: int = 100) -> pd.DataFrame:
    date_index = pd.bdate_range("2023-01-02", periods=num_day_int)
    day_vec = np.arange(num_day_int, dtype=float)
    base_price_dict = {
        "XLE": 80.0,
        "XLK": 7.5,
        "XLU": 70.0,
        "XLP": 18.0,
        "IEF": 85.0,
        "$SPXTR": 4000.0,
    }
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
            # Put the entitlement on month-end so the next-open rebalance
            # must size from the prior close value before that cash is posted.
            dividend_vec[np.flatnonzero(date_index.is_month_end)] = 0.20
        pricing_data_dict[(symbol_str, "Dividend")] = dividend_vec
    pricing_data_df = pd.DataFrame(pricing_data_dict, index=date_index)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def make_fred_snapshot(value_ser: pd.Series | None = None) -> FredSeriesSnapshot:
    if value_ser is None:
        value_ser = pd.Series(
            [2.10],
            index=pd.DatetimeIndex([pd.Timestamp("2023-01-31")]),
            name="T5YIE",
        )
    return FredSeriesSnapshot(
        value_ser=value_ser,
        source_name_str="FRED",
        series_id_str="T5YIE",
        download_attempt_timestamp_ts=datetime(2026, 8, 20, tzinfo=UTC),
        download_status_str="test_snapshot",
        latest_observation_date_ts=pd.Timestamp(value_ser.index[-1]),
        used_cache_bool=True,
        freshness_business_days_int=0,
    )


def make_strategy_data_tuple():
    execution_price_df = make_execution_price_df()
    decision_index = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-31"), pd.Timestamp("2023-02-28")]
    )
    month_end_feature_df = pd.DataFrame(
        {
            "growth_on_bool": [True, False],
            "inflation_on_bool": [False, False],
            "regime_label_str": [
                "growth_up__inflation_off",
                "growth_down__inflation_off",
            ],
        },
        index=decision_index,
    )
    month_end_weight_df = pd.DataFrame(
        {
            "XLE": [0.0, 0.0],
            "XLK": [1.0, 0.0],
            "XLU": [0.0, 0.0],
            "XLP": [0.0, 0.5],
            "IEF": [0.0, 0.5],
        },
        index=decision_index,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_price_df.index,
    )
    return (
        execution_price_df,
        month_end_feature_df,
        month_end_weight_df,
        rebalance_weight_df,
        make_fred_snapshot(),
    )


def test_default_contract_matches_literal_source_and_stays_out_of_live():
    assert variant_module.SIGNAL_ASSET_TUPLE == (
        "SPY",
        "XLE",
        "XLI",
        "XLF",
        "XLB",
        "XLU",
        "XLV",
        "XLP",
    )
    assert variant_module.TRADEABLE_ASSET_TUPLE == ("XLE", "XLK", "XLU", "XLP", "IEF")
    assert variant_module.GROWTH_SMA_SESSION_INT == 200
    assert variant_module.BREAKEVEN_LOOKBACK_SESSION_INT == 60
    assert variant_module.ASSET_SLOPE_LOOKBACK_SESSION_INT == 60
    assert variant_module.INFLATION_THRESHOLD_FLOAT == 2.0
    assert variant_module.SLIPPAGE_PER_SIDE_FLOAT == 0.0005
    assert variant_module.COMMISSION_PER_SHARE_FLOAT == 0.0
    assert variant_module.POSITIVE_BASKET_WEIGHT_DICT == {
        "XLE": 0.5,
        "XLI": 1.0 / 6.0,
        "XLF": 1.0 / 6.0,
        "XLB": 1.0 / 6.0,
    }
    assert variant_module.NEGATIVE_BASKET_WEIGHT_DICT == {
        "XLU": 1.0 / 3.0,
        "XLV": 1.0 / 3.0,
        "XLP": 1.0 / 3.0,
    }
    assert strategy_registry.tier_for(MODULE_IMPORT_STR) is strategy_registry.MaturityTier.PM_READY
    assert MODULE_IMPORT_STR not in strategy_registry.wired_import_tuple()


def test_regime_map_is_literal_and_always_fully_invested():
    expected_weight_dict = {
        (True, True): {"XLE": 1.0},
        (True, False): {"XLK": 1.0},
        (False, True): {"XLU": 1.0},
        (False, False): {"XLP": 0.5, "IEF": 0.5},
    }
    for regime_bool_tuple, nonzero_weight_dict in expected_weight_dict.items():
        _regime_label_str, target_weight_ser = variant_module._regime_target_weight_ser(
            growth_on_bool=regime_bool_tuple[0],
            inflation_on_bool=regime_bool_tuple[1],
        )
        assert np.isclose(target_weight_ser.sum(), 1.0)
        assert target_weight_ser[target_weight_ser > 0.0].to_dict() == nonzero_weight_dict


def test_fred_alignment_is_backward_as_of_and_never_uses_future_value():
    fred_value_ser = pd.Series(
        [1.90, 2.10],
        index=pd.DatetimeIndex(["2023-01-06", "2023-01-10"]),
        name="T5YIE",
    )
    session_date_index = pd.DatetimeIndex(["2023-01-06", "2023-01-09", "2023-01-10"])

    aligned_value_ser, observation_age_day_ser = variant_module.align_fred_to_session_ser(
        fred_value_ser=fred_value_ser,
        session_date_index=session_date_index,
    )

    assert aligned_value_ser.tolist() == [1.90, 1.90, 2.10]
    assert observation_age_day_ser.astype(float).tolist() == [0.0, 3.0, 0.0]


def test_stale_fred_after_signal_warmup_fails_loud_instead_of_holding_old_sleeve():
    signal_close_df = make_signal_close_df()
    t5yie_value_ser = pd.Series(
        2.20,
        index=signal_close_df.index[:180],
        name="T5YIE",
    )
    config_obj = variant_module.InflationCompassConfig(
        growth_sma_session_int=20,
        breakeven_lookback_session_int=5,
        asset_slope_lookback_session_int=5,
    )

    with pytest.raises(RuntimeError, match="refusing to silently hold"):
        variant_module.compute_month_end_signal_and_weight_df(
            signal_close_df=signal_close_df,
            t5yie_value_ser=t5yie_value_ser,
            config_obj=config_obj,
        )


def test_rolling_ols_slope_uses_exact_trailing_window():
    value_ser = pd.Series([1.0, 2.0, 3.0, 4.0, 8.0])
    slope_ser = variant_module.compute_rolling_ols_slope_ser(
        value_ser=value_ser,
        lookback_session_int=3,
    )

    assert slope_ser.iloc[:2].isna().all()
    assert np.isclose(slope_ser.iloc[2], 1.0)
    assert np.isclose(slope_ser.iloc[3], 1.0)
    assert np.isclose(slope_ser.iloc[4], 2.5)


def test_basket_returns_use_literal_unequal_weights():
    date_index = pd.bdate_range("2023-01-27", periods=5)
    signal_close_df = pd.DataFrame(
        100.0,
        index=date_index,
        columns=list(variant_module.SIGNAL_ASSET_TUPLE),
    )
    month_end_ts = pd.Timestamp("2023-01-31")
    distinct_month_end_price_dict = {
        "XLE": 110.0,
        "XLI": 102.0,
        "XLF": 104.0,
        "XLB": 106.0,
        "XLU": 103.0,
        "XLV": 106.0,
        "XLP": 109.0,
    }
    for asset_str, price_float in distinct_month_end_price_dict.items():
        signal_close_df.loc[month_end_ts:, asset_str] = price_float
    t5yie_value_ser = pd.Series(2.20, index=date_index, name="T5YIE")
    config_obj = variant_module.InflationCompassConfig(
        growth_sma_session_int=2,
        breakeven_lookback_session_int=1,
        asset_slope_lookback_session_int=2,
    )

    month_end_feature_df, _month_end_weight_df = (
        variant_module.compute_month_end_signal_and_weight_df(
            signal_close_df=signal_close_df,
            t5yie_value_ser=t5yie_value_ser,
            config_obj=config_obj,
        )
    )

    month_end_feature_ser = month_end_feature_df.loc[month_end_ts]
    assert np.isclose(
        month_end_feature_ser["positive_basket_return_float"],
        0.50 * 0.10 + (0.02 + 0.04 + 0.06) / 6.0,
    )
    assert np.isclose(
        month_end_feature_ser["negative_basket_return_float"],
        (0.03 + 0.06 + 0.09) / 3.0,
    )
    assert np.isclose(
        month_end_feature_ser["asset_ratio_float"],
        1.07 / 1.06,
    )


def test_literal_inflation_or_rule_uses_breakeven_and_asset_confirmation():
    signal_close_df = make_signal_close_df()
    date_index = signal_close_df.index
    t5yie_value_ser = pd.Series(2.20, index=date_index, name="T5YIE")
    config_obj = variant_module.InflationCompassConfig(
        growth_sma_session_int=20,
        breakeven_lookback_session_int=5,
        asset_slope_lookback_session_int=5,
    )

    month_end_feature_df, month_end_weight_df = (
        variant_module.compute_month_end_signal_and_weight_df(
            signal_close_df=signal_close_df,
            t5yie_value_ser=t5yie_value_ser,
            config_obj=config_obj,
        )
    )

    last_feature_ser = month_end_feature_df.iloc[-1]
    assert bool(last_feature_ser["growth_on_bool"])
    assert bool(last_feature_ser["inflation_level_on_bool"])
    assert not bool(last_feature_ser["breakeven_up_bool"])
    assert bool(last_feature_ser["asset_up_bool"])
    assert bool(last_feature_ser["inflation_on_bool"])
    assert month_end_weight_df.iloc[-1].to_dict() == {
        "XLE": 1.0,
        "XLK": 0.0,
        "XLU": 0.0,
        "XLP": 0.0,
        "IEF": 0.0,
    }


def test_literal_inflation_or_rule_accepts_breakeven_confirmation_alone():
    signal_close_df = make_signal_close_df()
    price_step_vec = np.arange(len(signal_close_df), dtype=float)
    for asset_str in variant_module.POSITIVE_BASKET_WEIGHT_DICT:
        signal_close_df[asset_str] = 100.0 * np.cumprod(
            np.full(len(signal_close_df), 0.9990)
        )
    for asset_str in variant_module.NEGATIVE_BASKET_WEIGHT_DICT:
        signal_close_df[asset_str] = 100.0 * np.cumprod(
            np.full(len(signal_close_df), 1.0010)
        )
    t5yie_value_ser = pd.Series(
        2.10 + 0.001 * price_step_vec,
        index=signal_close_df.index,
        name="T5YIE",
    )
    config_obj = variant_module.InflationCompassConfig(
        growth_sma_session_int=20,
        breakeven_lookback_session_int=5,
        asset_slope_lookback_session_int=5,
    )

    month_end_feature_df, _month_end_weight_df = (
        variant_module.compute_month_end_signal_and_weight_df(
            signal_close_df=signal_close_df,
            t5yie_value_ser=t5yie_value_ser,
            config_obj=config_obj,
        )
    )

    last_feature_ser = month_end_feature_df.iloc[-1]
    assert bool(last_feature_ser["inflation_level_on_bool"])
    assert bool(last_feature_ser["breakeven_up_bool"])
    assert not bool(last_feature_ser["asset_up_bool"])
    assert bool(last_feature_ser["inflation_on_bool"])


def test_growth_momentum_and_slope_equality_boundaries_are_strict():
    signal_close_df = make_signal_close_df()
    signal_close_df.loc[:, :] = 100.0
    t5yie_value_ser = pd.Series(
        2.20,
        index=signal_close_df.index,
        name="T5YIE",
    )
    config_obj = variant_module.InflationCompassConfig(
        growth_sma_session_int=20,
        breakeven_lookback_session_int=5,
        asset_slope_lookback_session_int=5,
    )

    month_end_feature_df, month_end_weight_df = (
        variant_module.compute_month_end_signal_and_weight_df(
            signal_close_df=signal_close_df,
            t5yie_value_ser=t5yie_value_ser,
            config_obj=config_obj,
        )
    )

    last_feature_ser = month_end_feature_df.iloc[-1]
    assert not bool(last_feature_ser["growth_on_bool"])
    assert bool(last_feature_ser["inflation_level_on_bool"])
    assert not bool(last_feature_ser["breakeven_up_bool"])
    assert not bool(last_feature_ser["asset_up_bool"])
    assert not bool(last_feature_ser["inflation_on_bool"])
    assert month_end_weight_df.iloc[-1][["XLP", "IEF"]].tolist() == [0.5, 0.5]


def test_two_percent_gate_is_strict_even_when_confirmation_is_positive():
    signal_close_df = make_signal_close_df()
    t5yie_value_ser = pd.Series(2.00, index=signal_close_df.index, name="T5YIE")
    config_obj = variant_module.InflationCompassConfig(
        growth_sma_session_int=20,
        breakeven_lookback_session_int=5,
        asset_slope_lookback_session_int=5,
    )

    month_end_feature_df, month_end_weight_df = (
        variant_module.compute_month_end_signal_and_weight_df(
            signal_close_df=signal_close_df,
            t5yie_value_ser=t5yie_value_ser,
            config_obj=config_obj,
        )
    )

    assert not bool(month_end_feature_df.iloc[-1]["inflation_level_on_bool"])
    assert not bool(month_end_feature_df.iloc[-1]["inflation_on_bool"])
    assert month_end_weight_df.iloc[-1]["XLK"] == 1.0


def test_month_end_decision_maps_to_first_next_month_open():
    decision_weight_df = pd.DataFrame(
        {"XLE": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2003-03-31")]),
    )
    execution_index = pd.DatetimeIndex(
        [pd.Timestamp("2003-03-31"), pd.Timestamp("2003-04-01"), pd.Timestamp("2003-04-02")]
    )

    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=decision_weight_df,
        execution_index=execution_index,
    )

    assert rebalance_weight_df.index.tolist() == [pd.Timestamp("2003-04-01")]


def test_shared_fred_loader_receives_t5yie_contract():
    config_obj = variant_module.InflationCompassConfig(
        end_date_str="2026-06-30",
        t5yie_csv_path_str="T5YIE-test.csv",
    )
    expected_snapshot_obj = make_fred_snapshot()
    with patch.object(
        variant_module,
        "load_daily_fred_series_snapshot",
        return_value=expected_snapshot_obj,
    ) as loader_mock:
        actual_snapshot_obj = variant_module.load_t5yie_snapshot(config_obj)

    assert actual_snapshot_obj is expected_snapshot_obj
    loader_mock.assert_called_once_with(
        series_id_str="T5YIE",
        cache_csv_path_str="T5YIE-test.csv",
        as_of_ts=datetime(2026, 6, 30),
        mode_str="backtest",
    )


def test_run_variant_honors_pm_contract_and_attaches_fred_provenance():
    strategy_data_tuple = make_strategy_data_tuple()
    with patch.object(
        variant_module,
        "get_inflation_compass_data",
        return_value=strategy_data_tuple,
    ):
        strategy_obj = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2023-02-01",
            capital_base_float=12_345.0,
            end_date_str="2023-05-19",
        )

    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert strategy_obj.name == variant_module.STRATEGY_NAME_STR
    assert strategy_obj._capital_base == 12_345.0
    assert strategy_obj.tradeable_asset_list == list(variant_module.TRADEABLE_ASSET_TUPLE)
    assert strategy_obj.t5yie_snapshot_obj.series_id_str == "T5YIE"
    fred_provenance_dict = strategy_obj._data_adjustment_policy_dict[
        "fred_series_provenance_dict"
    ]
    assert fred_provenance_dict == {
        "source_name_str": "FRED",
        "series_id_str": "T5YIE",
        "download_attempt_timestamp_str": "2026-08-20T00:00:00+00:00",
        "download_status_str": "test_snapshot",
        "latest_observation_date_str": "2023-01-31",
        "used_cache_bool": True,
        "freshness_business_days_int": 0,
        "vintage_policy_str": "current_vintage_not_alfred",
    }
    metadata_dict = _strategy_metadata_dict(
        strategy_obj,
        variant_module.REPO_ROOT_PATH / "test-placeholder.pkl",
    )
    assert metadata_dict["data_adjustment_policy"][
        "fred_series_provenance_dict"
    ] == fred_provenance_dict
    assert len(strategy_obj.results) > 0
    assert strategy_obj.results.index.min() >= pd.Timestamp("2023-02-01")


def test_capacity_builder_preserves_moo_contract():
    strategy_data_tuple = make_strategy_data_tuple()
    with patch.object(
        variant_module,
        "get_inflation_compass_data",
        return_value=strategy_data_tuple,
    ):
        capacity_input_dict = variant_module.build_capacity_analysis_inputs(
            capital_base_float=25_000.0,
        )

    assert capacity_input_dict["strategy_obj"]._capital_base == 25_000.0
    assert capacity_input_dict["execution_policy_str"] == "MOO"
    assert capacity_input_dict["impact_profile_str"] == "MOO_ETF_PROXY"


def test_timing_default_cell_matches_vanilla_next_open_contract():
    strategy_data_tuple = make_strategy_data_tuple()
    with patch.object(
        variant_module,
        "get_inflation_compass_data",
        return_value=strategy_data_tuple,
    ):
        vanilla_strategy_obj = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
        )
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
    assert isinstance(timing_strategy_obj, variant_module.InflationCompassTimingStrategy)

    pd.testing.assert_series_equal(
        timing_strategy_obj.results["total_value"],
        vanilla_strategy_obj.results["total_value"],
        check_names=False,
        check_freq=False,
        rtol=0.0,
        atol=1e-8,
    )
    pd.testing.assert_series_equal(
        timing_strategy_obj.results["cash"],
        vanilla_strategy_obj.results["cash"],
        check_names=False,
        check_freq=False,
        rtol=0.0,
        atol=1e-8,
    )
    transaction_column_list = [
        "trade_id",
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "commission",
    ]
    pd.testing.assert_frame_equal(
        timing_strategy_obj._transactions[transaction_column_list].reset_index(drop=True),
        vanilla_strategy_obj._transactions[transaction_column_list].reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=1e-8,
    )
    pd.testing.assert_frame_equal(
        timing_strategy_obj.get_dividend_ledger().reset_index(drop=True),
        vanilla_strategy_obj.get_dividend_ledger().reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=1e-8,
    )


def test_stress_registry_builds_vanilla_strategy_and_calendar():
    strategy_data_tuple = make_strategy_data_tuple()
    strategy_spec_obj = SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[variant_module.STRATEGY_NAME_STR]
    with patch.object(
        variant_module,
        "get_inflation_compass_data",
        return_value=strategy_data_tuple,
    ):
        context_dict = strategy_spec_obj.load_context_fn()

    strategy_obj = strategy_spec_obj.build_strategy_fn(context_dict)
    assert type(strategy_obj) is variant_module.InflationCompassStrategy
    assert context_dict["calendar_idx"][0] == pd.Timestamp("2023-02-01")
    assert strategy_obj.t5yie_snapshot_obj.series_id_str == "T5YIE"


def test_all_five_bench_analyzers_resolve_without_skip():
    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj.has_capacity_analysis_bool is True
    assert strategy_entry_obj.has_timing_analysis_bool is True

    for analysis_str in ("vanilla", "capacity", "timing", "risk", "stress"):
        assert analysis_runner._missing_hook_detail_str(
            variant_module,
            analysis_str,
        ) is None
