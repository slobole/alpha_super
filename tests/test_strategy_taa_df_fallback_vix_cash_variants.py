from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.taa_df.run_taa_df_fallback_vix_cash_variant_suite import _variant_module_name_tuple
from strategies.taa_df.strategy_taa_df import DefenseFirstConfig
import strategies.taa_df.strategy_taa_df_btal_fallback_upro_vix_cash as reference_vix_module
import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as shared_vix_helper


def make_execution_price_df(symbol_list: list[str]) -> pd.DataFrame:
    execution_index = pd.bdate_range("2020-01-01", periods=80)
    execution_frame_list: list[pd.DataFrame] = []

    for symbol_idx_int, symbol_str in enumerate(symbol_list):
        base_price_vec = 100.0 + float(symbol_idx_int * 10) + np.arange(len(execution_index), dtype=float) * 0.1
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

    execution_price_df = pd.concat(execution_frame_list, axis=1).sort_index()
    return execution_price_df


def make_base_month_end_weight_df(config: DefenseFirstConfig) -> pd.DataFrame:
    month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    base_month_end_weight_df = pd.DataFrame(
        0.0,
        index=month_end_index,
        columns=list(config.tradeable_asset_list),
        dtype=float,
    )
    base_month_end_weight_df.loc[pd.Timestamp("2020-01-31"), config.defensive_asset_list[0]] = 0.4
    base_month_end_weight_df.loc[pd.Timestamp("2020-01-31"), config.fallback_asset] = 0.6
    return base_month_end_weight_df


def make_standard_loader_output(config: DefenseFirstConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    base_month_end_weight_df = make_base_month_end_weight_df(config)
    return execution_price_df, momentum_score_df, base_month_end_weight_df, pd.DataFrame()


def make_linearity_loader_output(
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol_list = list(config.tradeable_asset_list) + list(config.benchmark_list)
    execution_price_df = make_execution_price_df(symbol_list)
    execution_index = execution_price_df.index
    month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])

    daily_linearity_score_df = pd.DataFrame(
        {
            asset_str: np.linspace(0.01, 0.02, len(execution_index))
            for asset_str in config.defensive_asset_list
        },
        index=execution_index,
        dtype=float,
    )
    month_end_score_df = daily_linearity_score_df.resample("ME").last().reindex(month_end_index)
    base_month_end_weight_df = make_base_month_end_weight_df(config)
    return execution_price_df, daily_linearity_score_df, month_end_score_df, base_month_end_weight_df, pd.DataFrame()


def make_helper_close_ser_side_effect(index: pd.DatetimeIndex):
    spy_close_ser = pd.Series(
        np.cumprod(1.0 + np.repeat(0.001, len(index))),
        index=index,
        name="SPY",
    )
    vix_close_ser = pd.Series(30.0, index=index, name="$VIX")

    def _side_effect(symbol_str: str, start_date_str: str, end_date_str: str | None):
        if symbol_str == "SPY":
            return spy_close_ser
        if symbol_str == "$VIX":
            return vix_close_ser
        raise AssertionError(f"Unexpected helper symbol {symbol_str}.")

    return _side_effect


class DefenseFirstFallbackVixCashVariantTests(unittest.TestCase):
    def test_apply_vrp_cash_gate_to_month_end_weight_df_supports_generic_fallback_asset(self):
        month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])
        base_month_end_weight_df = pd.DataFrame(
            {
                "GLD": [0.4, 0.4],
                "UUP": [0.0, 0.0],
                "TLT": [0.0, 0.0],
                "DBC": [0.0, 0.0],
                "QQQ": [0.6, 0.6],
            },
            index=month_end_index,
            dtype=float,
        )
        month_end_vrp_signal_df = pd.DataFrame(
            {
                "rv20_ann_pct": [12.0, 22.0],
                "vix_close": [18.0, 18.0],
                "vrp_gate": [1.0, 0.0],
            },
            index=month_end_index,
            dtype=float,
        )
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD", "UUP", "TLT", "DBC"),
            fallback_asset="QQQ",
            rank_weight_vec=(0.4, 0.3, 0.2, 0.1),
            start_date_str="2012-01-01",
        )

        month_end_weight_df, month_end_vrp_diagnostic_df = (
            shared_vix_helper.apply_vrp_cash_gate_to_month_end_weight_df(
                base_month_end_weight_df=base_month_end_weight_df,
                month_end_vrp_signal_df=month_end_vrp_signal_df,
                config=config,
            )
        )

        self.assertAlmostEqual(float(month_end_weight_df.loc[pd.Timestamp("2020-01-31"), "QQQ"]), 0.6)
        self.assertAlmostEqual(float(month_end_weight_df.loc[pd.Timestamp("2020-02-29"), "QQQ"]), 0.0)
        self.assertAlmostEqual(float(month_end_vrp_diagnostic_df.loc[pd.Timestamp("2020-02-29"), "cash_weight"]), 0.6)

    def test_apply_vrp_cash_gate_to_month_end_weight_df_negative_residual_cash_raises(self):
        month_end_index = pd.to_datetime(["2020-01-31"])
        base_month_end_weight_df = pd.DataFrame(
            {
                "GLD": [0.7],
                "TLT": [0.4],
                "UPRO": [0.1],
            },
            index=month_end_index,
            dtype=float,
        )
        month_end_vrp_signal_df = pd.DataFrame(
            {
                "rv20_ann_pct": [25.0],
                "vix_close": [18.0],
                "vrp_gate": [0.0],
            },
            index=month_end_index,
            dtype=float,
        )
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD", "TLT"),
            fallback_asset="UPRO",
            rank_weight_vec=(0.5, 0.5),
            start_date_str="2012-01-01",
        )

        with self.assertRaisesRegex(ValueError, "Residual cash weight must be non-negative"):
            shared_vix_helper.apply_vrp_cash_gate_to_month_end_weight_df(
                base_month_end_weight_df=base_month_end_weight_df,
                month_end_vrp_signal_df=month_end_vrp_signal_df,
                config=config,
            )

    def test_sample_month_end_vrp_signal_df_uses_last_available_trading_day(self):
        daily_vrp_signal_df = pd.DataFrame(
            {
                "rv20_ann_pct": [10.0, 11.0, 12.0],
                "vix_close": [20.0, 21.0, 22.0],
                "vrp_gate": [1.0, 1.0, 1.0],
            },
            index=pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-28"]),
            dtype=float,
        )

        month_end_vrp_signal_df = shared_vix_helper.sample_month_end_vrp_signal_df(daily_vrp_signal_df)

        self.assertAlmostEqual(float(month_end_vrp_signal_df.loc[pd.Timestamp("2020-01-31"), "vix_close"]), 21.0)
        self.assertAlmostEqual(float(month_end_vrp_signal_df.loc[pd.Timestamp("2020-02-29"), "vix_close"]), 22.0)

    def test_build_vrp_cash_overlay_weight_frames_uses_first_next_month_open(self):
        month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])
        base_month_end_weight_df = pd.DataFrame(
            {
                "GLD": [0.4, 0.4],
                "SPY": [0.6, 0.6],
            },
            index=month_end_index,
            dtype=float,
        )
        month_end_vrp_signal_df = pd.DataFrame(
            {
                "rv20_ann_pct": [12.0, 22.0],
                "vix_close": [18.0, 18.0],
                "vrp_gate": [1.0, 0.0],
            },
            index=month_end_index,
            dtype=float,
        )
        execution_index = pd.to_datetime(["2020-02-03", "2020-02-04", "2020-03-02", "2020-03-03"])
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD",),
            fallback_asset="SPY",
            rank_weight_vec=(1.0,),
            start_date_str="2012-01-01",
        )

        _, rebalance_weight_df, _ = shared_vix_helper.build_vrp_cash_overlay_weight_frames(
            base_month_end_weight_df=base_month_end_weight_df,
            month_end_vrp_signal_df=month_end_vrp_signal_df,
            execution_index=execution_index,
            config=config,
        )

        expected_index = pd.to_datetime(["2020-02-03", "2020-03-02"])
        self.assertTrue(rebalance_weight_df.index.equals(expected_index))

    def test_reference_upro_vix_module_reexports_shared_helpers(self):
        self.assertIs(
            reference_vix_module.compute_daily_vrp_signal_df,
            shared_vix_helper.compute_daily_vrp_signal_df,
        )
        self.assertIs(
            reference_vix_module.apply_vrp_cash_gate_to_month_end_weight_df,
            shared_vix_helper.apply_vrp_cash_gate_to_month_end_weight_df,
        )

    def test_vix_cash_variant_suite_enumerates_24_modules(self):
        self.assertEqual(len(_variant_module_name_tuple()), 24)

    def _run_smoke_variant(self, module_name_str: str, loader_attr_str: str, lineary_bool: bool) -> None:
        variant_module = importlib.import_module(module_name_str)
        config = variant_module.DEFAULT_CONFIG

        if lineary_bool:
            base_loader_output = make_linearity_loader_output(config)
            execution_price_df = base_loader_output[0]
        else:
            base_loader_output = make_standard_loader_output(config)
            execution_price_df = base_loader_output[0]

        with patch.object(variant_module, loader_attr_str, return_value=base_loader_output):
            with patch.object(
                shared_vix_helper,
                "load_helper_close_ser",
                side_effect=make_helper_close_ser_side_effect(execution_price_df.index),
            ):
                strategy = variant_module.run_variant(
                    show_display_bool=False,
                    save_results_bool=False,
                )

        self.assertIsNotNone(strategy.summary)
        self.assertGreater(len(strategy.summary.index), 0)
        self.assertTrue(hasattr(strategy, "daily_vrp_signal_df"))
        self.assertTrue(hasattr(strategy, "month_end_vrp_diagnostic_df"))

    def test_standard_vix_cash_variant_smoke(self):
        self._run_smoke_variant(
            module_name_str="strategies.taa_df.strategy_taa_df_fallback_spy_vix_cash",
            loader_attr_str="get_defense_first_data",
            lineary_bool=False,
        )

    def test_btal_1n_vix_cash_variant_smoke(self):
        self._run_smoke_variant(
            module_name_str="strategies.taa_df.strategy_taa_df_btal_1n_fallback_upro_vix_cash",
            loader_attr_str="get_defense_first_data",
            lineary_bool=False,
        )

    def test_btal_linearity_1n_vix_cash_variant_smoke(self):
        self._run_smoke_variant(
            module_name_str="strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
            loader_attr_str="get_defense_first_linearity_1n_data",
            lineary_bool=True,
        )


if __name__ == "__main__":
    unittest.main()
