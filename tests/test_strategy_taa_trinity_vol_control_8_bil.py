import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha import strategy_registry
from alpha.bench import catalog
from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from scripts.research import run_strategy_analysis as analysis_runner
from strategies.taa_beyond_6040 import strategy_taa_trinity_vol_control_8_bil as variant_module


MODULE_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil"


def make_pricing_data_df(num_days_int: int = 220) -> pd.DataFrame:
    date_index = pd.date_range("2023-01-02", periods=num_days_int, freq="B")
    bar_index_vec = np.arange(num_days_int, dtype=float)
    return_vec_by_symbol_dict = {
        "VTI": 0.0005 + 0.0080 * np.sin(bar_index_vec / 4.0),
        "GLD": 0.0002 + 0.0040 * np.sin(bar_index_vec / 6.0 + 0.5),
        "TLT": 0.0001 + 0.0030 * np.cos(bar_index_vec / 7.0),
        "BIL": 0.0001 + 0.0001 * np.sin(bar_index_vec / 8.0),
        "$SPX": 0.0004 + 0.0070 * np.sin(bar_index_vec / 5.0 + 0.25),
    }
    base_price_by_symbol_dict = {
        "VTI": 100.0,
        "GLD": 120.0,
        "TLT": 110.0,
        "BIL": 90.0,
        "$SPX": 4000.0,
    }

    pricing_data_dict: dict[tuple[str, str], np.ndarray] = {}
    for symbol_str, return_vec in return_vec_by_symbol_dict.items():
        close_vec = base_price_by_symbol_dict[symbol_str] * np.cumprod(1.0 + return_vec)
        open_vec = close_vec * 0.999
        high_vec = np.maximum(open_vec, close_vec) * 1.001
        low_vec = np.minimum(open_vec, close_vec) * 0.999
        dividend_vec = np.zeros(num_days_int, dtype=float)
        if symbol_str in variant_module.TRADEABLE_ASSET_TUPLE:
            dividend_vec[80::40] = 0.20
        pricing_data_dict[(symbol_str, "Open")] = open_vec
        pricing_data_dict[(symbol_str, "High")] = high_vec
        pricing_data_dict[(symbol_str, "Low")] = low_vec
        pricing_data_dict[(symbol_str, "Close")] = close_vec
        pricing_data_dict[(symbol_str, "Dividend")] = dividend_vec

    pricing_data_df = pd.DataFrame(pricing_data_dict, index=date_index, dtype=float)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def test_default_contract_uses_three_risk_assets_bil_and_requested_thresholds():
    assert variant_module.RISK_ASSET_TUPLE == ("VTI", "GLD", "TLT")
    assert variant_module.CASH_SUBSTITUTE_ASSET_STR == "BIL"
    assert variant_module.DEFAULT_CONFIG.asset_list == ("VTI", "GLD", "TLT", "BIL")
    assert variant_module.DEFAULT_CONFIG.target_portfolio_vol_float == 0.08
    assert variant_module.DEFAULT_CONFIG.trigger_portfolio_vol_float == 0.085
    assert variant_module.EXPOSURE_REBALANCE_BAND_FLOAT == 0.05


def test_five_point_band_and_monthly_override_are_literal():
    assert not variant_module.should_rebalance_exposure_bool(0.80, 0.751)
    assert variant_module.should_rebalance_exposure_bool(0.80, 0.75)
    assert variant_module.should_rebalance_exposure_bool(0.70, 0.65)
    assert not variant_module.should_rebalance_exposure_bool(0.65, 0.699)
    assert variant_module.should_rebalance_exposure_bool(0.65, 0.70)
    assert variant_module.should_rebalance_exposure_bool(
        0.80,
        0.80,
        monthly_rebalance_bool=True,
    )
    assert variant_module.should_rebalance_exposure_bool(1.0, 1.001)


def test_base_portfolio_volatility_uses_unscaled_risk_assets_only():
    return_index = pd.bdate_range("2023-01-02", periods=63)
    vti_return_vec = np.linspace(-0.02, 0.02, 63)
    risk_return_df = pd.DataFrame(
        {
            "VTI": vti_return_vec,
            "GLD": -0.5 * vti_return_vec,
            "TLT": 0.25 * vti_return_vec,
        },
        index=return_index,
    )
    base_weight_ser = pd.Series({"VTI": 0.50, "GLD": 0.30, "TLT": 0.20})

    base_portfolio_return_ser = variant_module.compute_base_portfolio_return_ser(
        risk_return_df=risk_return_df,
        base_weight_ser=base_weight_ser,
        portfolio_vol_lookback_int=63,
    )

    expected_return_ser = (
        risk_return_df["VTI"] * 0.50
        + risk_return_df["GLD"] * 0.30
        + risk_return_df["TLT"] * 0.20
    )
    pd.testing.assert_series_equal(base_portfolio_return_ser, expected_return_ser)

    expected_volatility_float = float(expected_return_ser.std(ddof=1) * np.sqrt(252.0))
    expected_exposure_float = (
        1.0
        if expected_volatility_float <= variant_module.TRIGGER_PORTFOLIO_VOL_FLOAT
        else variant_module.TARGET_PORTFOLIO_VOL_FLOAT / expected_volatility_float
    )
    actual_exposure_float = variant_module.compute_gross_exposure_float(
        realized_return_ser=base_portfolio_return_ser,
        portfolio_vol_lookback_int=63,
        target_portfolio_vol_float=variant_module.TARGET_PORTFOLIO_VOL_FLOAT,
        trigger_portfolio_vol_float=variant_module.TRIGGER_PORTFOLIO_VOL_FLOAT,
    )
    assert np.isclose(actual_exposure_float, expected_exposure_float)


def test_base_portfolio_volatility_fails_loud_on_incomplete_window():
    risk_return_df = pd.DataFrame(
        {
            "VTI": np.zeros(63),
            "GLD": np.zeros(63),
            "TLT": np.zeros(63),
        },
        index=pd.bdate_range("2023-01-02", periods=63),
    )
    risk_return_df.iloc[-1, 0] = np.nan

    with np.testing.assert_raises_regex(
        RuntimeError,
        "complete trailing return window",
    ):
        variant_module.compute_base_portfolio_return_ser(
            risk_return_df=risk_return_df,
            base_weight_ser=pd.Series({"VTI": 0.50, "GLD": 0.30, "TLT": 0.20}),
            portfolio_vol_lookback_int=63,
        )


def test_portfolio_volatility_trigger_is_inclusive_at_eight_point_five_percent():
    raw_return_ser = pd.Series(np.linspace(-1.0, 1.0, 63), dtype=float)

    def exposure_for_vol_float(annualized_volatility_float: float) -> float:
        scaled_return_ser = raw_return_ser * (
            annualized_volatility_float
            / (float(raw_return_ser.std(ddof=1)) * np.sqrt(252.0))
        )
        return variant_module.compute_gross_exposure_float(
            realized_return_ser=scaled_return_ser,
            portfolio_vol_lookback_int=63,
            target_portfolio_vol_float=variant_module.TARGET_PORTFOLIO_VOL_FLOAT,
            trigger_portfolio_vol_float=variant_module.TRIGGER_PORTFOLIO_VOL_FLOAT,
        )

    assert exposure_for_vol_float(0.085 - 1e-10) == 1.0
    assert exposure_for_vol_float(0.085) == 1.0
    assert exposure_for_vol_float(0.085 + 1e-10) < 1.0


def test_target_weights_put_the_unexposed_share_in_bil_without_leverage():
    base_weight_ser = pd.Series({"VTI": 0.50, "GLD": 0.30, "TLT": 0.20})
    target_weight_ser = variant_module.build_target_weight_ser(
        base_weight_ser=base_weight_ser,
        gross_exposure_float=0.70,
    )

    assert np.isclose(target_weight_ser["VTI"], 0.35)
    assert np.isclose(target_weight_ser["GLD"], 0.21)
    assert np.isclose(target_weight_ser["TLT"], 0.14)
    assert np.isclose(target_weight_ser["BIL"], 0.30)
    assert target_weight_ser["Cash"] == 0.0
    assert np.isclose(target_weight_ser.sum(), 1.0)

    full_risk_weight_ser = variant_module.build_target_weight_ser(
        base_weight_ser=base_weight_ser,
        gross_exposure_float=1.0,
    )
    assert np.isclose(
        full_risk_weight_ser[list(variant_module.RISK_ASSET_TUPLE)].sum(),
        1.0,
    )
    assert full_risk_weight_ser["BIL"] == 0.0

    all_bil_weight_ser = variant_module.build_target_weight_ser(
        base_weight_ser=base_weight_ser,
        gross_exposure_float=0.0,
    )
    assert all_bil_weight_ser[list(variant_module.RISK_ASSET_TUPLE)].sum() == 0.0
    assert all_bil_weight_ser["BIL"] == 1.0

    for invalid_exposure_float in (-0.01, 1.01):
        with np.testing.assert_raises_regex(ValueError, "non-negative and sum to 1.0"):
            variant_module.build_target_weight_ser(
                base_weight_ser=base_weight_ser,
                gross_exposure_float=invalid_exposure_float,
            )


def test_monthly_override_is_marked_on_close_before_next_month_open():
    execution_index = pd.bdate_range("2023-01-02", "2023-02-03")
    month_end_weight_df = pd.DataFrame(
        {"VTI": [0.5], "GLD": [0.3], "TLT": [0.2]},
        index=[pd.Timestamp("2023-01-31")],
    )

    monthly_rebalance_ser = variant_module.build_monthly_rebalance_signal_ser(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_index,
    )

    assert monthly_rebalance_ser.loc[pd.Timestamp("2023-01-31")]
    assert int(monthly_rebalance_ser.sum()) == 1


def test_iterate_monthly_override_rebalances_inside_band():
    strategy_obj = variant_module.TrinityVolControlStrategy(
        name="monthly_override_test",
        benchmarks=[],
        capital_base=100_000.0,
    )
    strategy_obj.current_bar = pd.Timestamp("2023-04-03")
    return_index = pd.bdate_range(end="2023-03-31", periods=63)
    data_df = pd.DataFrame(
        {
            ("VTI", "return_ser"): np.zeros(63),
            ("GLD", "return_ser"): np.zeros(63),
            ("TLT", "return_ser"): np.zeros(63),
        },
        index=return_index,
    )
    data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)
    close_row_ser = pd.Series(
        {
            ("VTI", "base_weight_ser"): 0.50,
            ("GLD", "base_weight_ser"): 0.30,
            ("TLT", "base_weight_ser"): 0.20,
            variant_module.MONTHLY_REBALANCE_FIELD_TUPLE: True,
        }
    )
    current_weight_ser = pd.Series(
        {"VTI": 0.49, "GLD": 0.29, "TLT": 0.20, "BIL": 0.02, "Cash": 0.0}
    )

    with (
        patch.object(
            strategy_obj,
            "_current_close_weight_ser",
            return_value=current_weight_ser,
        ),
        patch.object(strategy_obj, "_submit_target_orders") as submit_target_orders_mock,
    ):
        strategy_obj.iterate(data_df, close_row_ser, pd.Series(dtype=float))

    submit_target_orders_mock.assert_called_once()
    submitted_target_weight_ser = submit_target_orders_mock.call_args.kwargs[
        "target_weight_ser"
    ]
    assert np.isclose(
        submitted_target_weight_ser[list(variant_module.RISK_ASSET_TUPLE)].sum(),
        1.0,
    )
    assert submitted_target_weight_ser["BIL"] == 0.0


def test_iterate_de_risks_from_base_portfolio_not_realized_strategy_returns():
    strategy_obj = variant_module.TrinityVolControlStrategy(
        name="base_portfolio_vol_test",
        benchmarks=[],
        capital_base=100_000.0,
    )
    strategy_obj.current_bar = pd.Timestamp("2023-04-03")
    strategy_obj._daily_return_history_list = [0.0] * 100
    return_index = pd.bdate_range(end="2023-03-31", periods=63)
    alternating_return_vec = np.resize(np.array([-0.02, 0.02]), 63)
    data_df = pd.DataFrame(
        {
            ("VTI", "return_ser"): alternating_return_vec,
            ("GLD", "return_ser"): alternating_return_vec,
            ("TLT", "return_ser"): alternating_return_vec,
        },
        index=return_index,
    )
    data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)
    close_row_ser = pd.Series(
        {
            ("VTI", "base_weight_ser"): 0.50,
            ("GLD", "base_weight_ser"): 0.30,
            ("TLT", "base_weight_ser"): 0.20,
            variant_module.MONTHLY_REBALANCE_FIELD_TUPLE: False,
        }
    )
    current_weight_ser = pd.Series(
        {"VTI": 0.50, "GLD": 0.30, "TLT": 0.20, "BIL": 0.0, "Cash": 0.0}
    )
    expected_volatility_float = float(
        pd.Series(alternating_return_vec).std(ddof=1) * np.sqrt(252.0)
    )
    expected_exposure_float = variant_module.TARGET_PORTFOLIO_VOL_FLOAT / (
        expected_volatility_float
    )

    with (
        patch.object(
            strategy_obj,
            "_current_close_weight_ser",
            return_value=current_weight_ser,
        ),
        patch.object(strategy_obj, "_submit_target_orders") as submit_target_orders_mock,
    ):
        strategy_obj.iterate(data_df, close_row_ser, pd.Series(dtype=float))

    submit_target_orders_mock.assert_called_once()
    submitted_target_weight_ser = submit_target_orders_mock.call_args.kwargs[
        "target_weight_ser"
    ]
    assert np.isclose(
        submitted_target_weight_ser[list(variant_module.RISK_ASSET_TUPLE)].sum(),
        expected_exposure_float,
    )
    assert np.isclose(submitted_target_weight_ser["BIL"], 1.0 - expected_exposure_float)


def test_first_actionable_rebalance_waits_until_bil_is_tradable():
    pricing_data_df = make_pricing_data_df()
    pricing_data_df.loc[
        pricing_data_df.index < pd.Timestamp("2023-05-15"),
        "BIL",
    ] = np.nan

    first_rebalance_ts = variant_module.get_first_actionable_trinity_rebalance_ts(
        pricing_data_df=pricing_data_df
    )

    assert first_rebalance_ts == pd.Timestamp("2023-06-01")


def test_order_decision_uses_close_t_share_count_not_next_open_gap():
    strategy_obj = variant_module.TrinityVolControlStrategy(
        name="close_t_order_decision_test",
        benchmarks=[],
        capital_base=1_000.0,
    )
    strategy_obj._position_amount_map = {"VTI": 1.0}
    target_weight_ser = pd.Series(
        {"VTI": 0.10, "GLD": 0.0, "TLT": 0.0, "BIL": 0.0, "Cash": 0.90}
    )
    close_row_ser = pd.Series(
        {
            ("VTI", "Close"): 200.0,
            ("GLD", "Close"): 100.0,
            ("TLT", "Close"): 100.0,
            ("BIL", "Close"): 100.0,
        }
    )

    strategy_obj._submit_target_orders(
        target_weight_ser=target_weight_ser,
        close_row_ser=close_row_ser,
    )

    assert len(strategy_obj.get_orders()) == 1
    assert strategy_obj.get_orders()[0].asset == "VTI"
    assert strategy_obj.get_orders()[0].target is True


def test_run_variant_is_pm_ready_and_keeps_bil_out_of_inverse_volatility_weights():
    captured_config_list = []
    pricing_data_df = make_pricing_data_df()

    def loader_fn(config):
        captured_config_list.append(config)
        return pricing_data_df

    with (
        patch.object(variant_module, "get_beyond_6040_data", side_effect=loader_fn),
        patch.object(variant_module, "compute_gross_exposure_float", return_value=0.70),
    ):
        strategy_obj = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2023-05-01",
            capital_base_float=12_345.0,
            end_date_str="2023-09-01",
        )

    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert strategy_registry.tier_for(MODULE_IMPORT_STR) is strategy_registry.MaturityTier.PM_READY
    assert captured_config_list[0].asset_list == ("VTI", "GLD", "TLT", "BIL")
    assert captured_config_list[0].end_date_str == "2023-09-01"
    assert strategy_obj.name == variant_module.STRATEGY_NAME_STR
    assert strategy_obj.risk_asset_list == ["VTI", "GLD", "TLT"]
    assert strategy_obj.asset_list == ["VTI", "GLD", "TLT", "BIL"]
    assert list(strategy_obj.month_end_weight_df.columns) == ["VTI", "GLD", "TLT"]
    assert strategy_obj._capital_base == 12_345.0
    assert {"VTI", "GLD", "TLT", "BIL", "Cash"}.issubset(
        strategy_obj.daily_target_weights.columns
    )
    assert np.isclose(strategy_obj.daily_target_weights["BIL"], 0.30).any()
    assert len(strategy_obj.results) > 0
    assert strategy_obj.results.index.min() >= pd.Timestamp("2023-05-01")


def test_capacity_builder_preserves_strategy_and_moo_contract():
    pricing_data_df = make_pricing_data_df()
    with patch.object(
        variant_module,
        "get_beyond_6040_data",
        return_value=pricing_data_df,
    ):
        capacity_input_dict = variant_module.build_capacity_analysis_inputs(
            show_display_bool=False,
            backtest_start_date_str="2023-05-01",
            capital_base_float=25_000.0,
            end_date_str="2023-09-01",
        )

    strategy_obj = capacity_input_dict["strategy_obj"]
    assert strategy_obj._capital_base == 25_000.0
    assert capacity_input_dict["pricing_data_df"] is pricing_data_df
    assert capacity_input_dict["execution_policy_str"] == "MOO"
    assert capacity_input_dict["impact_profile_str"] == "MOO_ETF_PROXY"
    assert strategy_obj.results.index.min() >= pd.Timestamp("2023-05-01")


def test_timing_default_cell_matches_vanilla_next_open_contract():
    pricing_data_df = make_pricing_data_df()
    with patch.object(
        variant_module,
        "get_beyond_6040_data",
        return_value=pricing_data_df,
    ):
        vanilla_strategy_obj = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
        )
        timing_input_dict = variant_module.build_execution_timing_analysis_inputs()

    timing_strategy_obj = timing_input_dict["strategy_factory_fn"]()
    assert isinstance(timing_strategy_obj, variant_module.TrinityVolControlTimingStrategy)
    assert timing_input_dict["order_generation_mode_str"] == "vanilla_current_bar"
    assert timing_input_dict["default_entry_timing_str"] == "same_open"
    assert timing_input_dict["default_exit_timing_str"] == "same_open"
    assert timing_input_dict["entry_timing_str_tuple"] == (
        "same_open",
        "same_close_moc",
        "next_open",
        "next_close",
    )
    assert timing_input_dict["exit_timing_str_tuple"] == (
        "same_open",
        "same_close_moc",
        "next_open",
        "next_close",
    )

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
    default_timing_strategy_obj = timing_result_obj.strategy_map[("same_open", "same_open")]

    pd.testing.assert_series_equal(
        default_timing_strategy_obj.results["total_value"],
        vanilla_strategy_obj.results["total_value"],
        check_names=False,
        check_freq=False,
        rtol=0.0,
        atol=1e-8,
    )


def test_stress_registry_uses_vanilla_strategy_and_calendar():
    pricing_data_df = make_pricing_data_df()
    strategy_spec_obj = SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[variant_module.STRATEGY_NAME_STR]
    with patch.object(
        variant_module,
        "get_beyond_6040_data",
        return_value=pricing_data_df,
    ):
        context_dict = strategy_spec_obj.load_context_fn()

    strategy_obj = strategy_spec_obj.build_strategy_fn(context_dict)
    assert type(strategy_obj) is variant_module.TrinityVolControlStrategy
    assert context_dict["calendar_idx"][0] == pd.Timestamp("2023-04-03")
    assert strategy_obj._capital_base == variant_module.DEFAULT_CONFIG.capital_base_float


def test_all_five_bench_analyzers_resolve_without_skip():
    strategy_entry_obj = catalog.get_strategy_by_module(MODULE_IMPORT_STR)
    assert strategy_entry_obj.has_capacity_analysis_bool is True
    assert strategy_entry_obj.has_timing_analysis_bool is True

    for analysis_str in ("vanilla", "capacity", "timing", "risk", "stress"):
        assert analysis_runner._missing_hook_detail_str(
            variant_module,
            analysis_str,
        ) is None
