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
    assert variant_module.should_rebalance_exposure_bool(
        0.80,
        0.80,
        monthly_rebalance_bool=True,
    )


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
