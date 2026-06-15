from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.taa_df.strategy_taa_df import (
    DefenseFirstConfig,
    map_month_end_weights_to_rebalance_open_df,
)


MODULE_IMPORT_STR = "strategies.taa_df.strategy_taa_df_btal_fallback_spy"


def make_execution_price_df(symbol_list: list[str]) -> pd.DataFrame:
    execution_index = pd.bdate_range("2020-01-01", periods=100)
    execution_frame_list: list[pd.DataFrame] = []

    for symbol_idx_int, symbol_str in enumerate(symbol_list):
        base_price_vec = 100.0 + float(symbol_idx_int * 5) + np.arange(len(execution_index), dtype=float) * 0.1
        open_price_vec = base_price_vec
        close_price_vec = base_price_vec * 1.001
        high_price_vec = np.maximum(open_price_vec, close_price_vec) * 1.001
        low_price_vec = np.minimum(open_price_vec, close_price_vec) * 0.999

        price_df = pd.DataFrame(
            {
                "Open": open_price_vec,
                "High": high_price_vec,
                "Low": low_price_vec,
                "Close": close_price_vec,
            },
            index=execution_index,
        )
        price_df.columns = pd.MultiIndex.from_product([[symbol_str], price_df.columns])
        execution_frame_list.append(price_df)

    return pd.concat(execution_frame_list, axis=1).sort_index()


def make_standard_loader_output(
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol_list = list(config.tradeable_asset_list) + list(config.benchmark_list)
    execution_price_df = make_execution_price_df(symbol_list)
    month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])

    momentum_score_df = pd.DataFrame(
        {
            asset_str: np.linspace(0.1, 0.2, len(month_end_index))
            for asset_str in config.defensive_asset_list
        },
        index=month_end_index,
        dtype=float,
    )
    month_end_weight_df = pd.DataFrame(
        0.0,
        index=month_end_index,
        columns=list(config.tradeable_asset_list),
        dtype=float,
    )
    month_end_weight_df.loc[:, config.defensive_asset_list[0]] = 0.4
    month_end_weight_df.loc[:, config.fallback_asset] = 0.6

    # *** CRITICAL*** The plain fallback variant decides at month-end and
    # executes on the first tradable open of the following month.
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df,
        execution_price_df.index,
    )
    return execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df


def test_btal_fallback_spy_honors_bench_run_contract():
    variant_module = importlib.import_module(MODULE_IMPORT_STR)
    captured_config_list: list[DefenseFirstConfig] = []

    def standard_loader_fn(config: DefenseFirstConfig):
        captured_config_list.append(config)
        return make_standard_loader_output(config)

    with patch.object(variant_module, "get_defense_first_data", side_effect=standard_loader_fn):
        strategy_obj = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2020-03-02",
            capital_base_float=12345.0,
            end_date_str="2020-03-31",
        )

    assert captured_config_list[0].end_date_str == "2020-03-31"
    assert strategy_obj.name == "strategy_taa_df_btal_fallback_spy"
    assert strategy_obj._capital_base == 12345.0
    assert strategy_obj.results.index.min() >= pd.Timestamp("2020-03-02")


def test_btal_fallback_spy_exposes_friction_inputs_for_bench():
    variant_module = importlib.import_module(MODULE_IMPORT_STR)

    with patch.object(
        variant_module,
        "get_defense_first_data",
        side_effect=make_standard_loader_output,
    ):
        friction_input_dict = variant_module.build_friction_analysis_inputs(
            show_display_bool=False,
            backtest_start_date_str="2020-03-02",
            capital_base_float=12345.0,
            end_date_str="2020-03-31",
        )

    strategy_obj = friction_input_dict["strategy_obj"]
    assert strategy_obj.name == "strategy_taa_df_btal_fallback_spy"
    assert strategy_obj._capital_base == 12345.0
    assert strategy_obj.results.index.min() >= pd.Timestamp("2020-03-02")
    assert friction_input_dict["execution_policy_str"] == "MOO"
    assert isinstance(friction_input_dict["pricing_data_df"], pd.DataFrame)


def test_btal_fallback_spy_exposes_execution_timing_inputs_for_bench():
    variant_module = importlib.import_module(MODULE_IMPORT_STR)

    with patch.object(
        variant_module,
        "get_defense_first_data",
        side_effect=make_standard_loader_output,
    ):
        strategy_input_dict = variant_module.build_execution_timing_analysis_inputs()

    assert strategy_input_dict["order_generation_mode_str"] == "signal_bar"
    assert strategy_input_dict["risk_model_str"] == "taa_rebalance"
    assert strategy_input_dict["entry_timing_str_tuple"] == ("same_close_moc", "next_open", "next_close")
    assert strategy_input_dict["default_entry_timing_str"] == "next_open"

    strategy_obj = strategy_input_dict["strategy_factory_fn"]()
    assert strategy_obj.name == "strategy_taa_df_btal_fallback_spy"
    assert pd.Timestamp("2020-01-31") in strategy_obj.rebalance_weight_df.index
    assert pd.Timestamp("2020-02-28") in strategy_obj.rebalance_weight_df.index
    assert strategy_input_dict["calendar_idx"].min() == pd.Timestamp("2020-02-03")
    assert hasattr(strategy_obj, "daily_target_weights")
