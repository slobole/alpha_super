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

from strategies.taa_df.strategy_taa_df import DefenseFirstConfig
import strategies.taa_df.strategy_taa_df_fallback_risk_parity_variant_utils as shared_risk_parity_helper
import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as shared_vix_helper


def make_execution_price_df(symbol_list: list[str]) -> pd.DataFrame:
    execution_index = pd.bdate_range("2020-01-01", periods=90)
    execution_frame_list: list[pd.DataFrame] = []

    for symbol_idx_int, symbol_str in enumerate(symbol_list):
        return_scale_float = 0.001 + 0.0005 * float(symbol_idx_int)
        return_vec = np.resize(
            np.array([return_scale_float, -return_scale_float * 0.5, return_scale_float * 1.5], dtype=float),
            len(execution_index),
        )
        base_price_vec = 100.0 + float(symbol_idx_int * 10)
        close_price_vec = base_price_vec * np.cumprod(1.0 + return_vec)
        open_price_vec = close_price_vec * 0.999
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
    base_month_end_weight_df = pd.DataFrame(
        0.0,
        index=month_end_index,
        columns=list(config.tradeable_asset_list),
        dtype=float,
    )
    base_month_end_weight_df.loc[pd.Timestamp("2020-01-31"), config.defensive_asset_list[0]] = 0.4
    base_month_end_weight_df.loc[pd.Timestamp("2020-01-31"), config.fallback_asset] = 0.6
    return execution_price_df, momentum_score_df, base_month_end_weight_df, pd.DataFrame()


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


class DefenseFirstRiskParityVariantTests(unittest.TestCase):
    def test_compute_daily_risk_parity_volatility_df_prefers_lower_volatility_asset(self):
        daily_index = pd.bdate_range("2020-01-01", periods=80)
        low_return_vec = np.resize(np.array([0.001, -0.001], dtype=float), len(daily_index))
        high_return_vec = np.resize(np.array([0.02, -0.02], dtype=float), len(daily_index))
        tradeable_close_df = pd.DataFrame(
            {
                "LOW": 100.0 * np.cumprod(1.0 + low_return_vec),
                "HIGH": 100.0 * np.cumprod(1.0 + high_return_vec),
            },
            index=daily_index,
        )

        daily_volatility_df = shared_risk_parity_helper.compute_daily_risk_parity_volatility_df(
            tradeable_close_df=tradeable_close_df,
            lookback_day_int=20,
        )

        self.assertLess(
            float(daily_volatility_df["LOW"].iloc[-1]),
            float(daily_volatility_df["HIGH"].iloc[-1]),
        )

    def test_apply_risk_parity_weights_sum_to_tradeable_budget_and_prefer_low_vol(self):
        month_end_index = pd.to_datetime(["2020-01-31"])
        base_month_end_weight_df = pd.DataFrame(
            {"GLD": [0.5], "SPY": [0.5]},
            index=month_end_index,
            dtype=float,
        )
        month_end_volatility_df = pd.DataFrame(
            {"GLD": [0.10], "SPY": [0.20]},
            index=month_end_index,
            dtype=float,
        )
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD",),
            fallback_asset="SPY",
            rank_weight_vec=(1.0,),
            start_date_str="2012-01-01",
        )

        final_month_end_weight_df, diagnostic_df = (
            shared_risk_parity_helper.apply_risk_parity_to_month_end_weight_df(
                base_month_end_weight_df=base_month_end_weight_df,
                month_end_volatility_df=month_end_volatility_df,
                config=config,
            )
        )

        self.assertAlmostEqual(float(final_month_end_weight_df.loc[month_end_index[0]].sum()), 1.0)
        self.assertGreater(
            float(final_month_end_weight_df.loc[month_end_index[0], "GLD"]),
            float(final_month_end_weight_df.loc[month_end_index[0], "SPY"]),
        )
        self.assertAlmostEqual(float(diagnostic_df.loc[month_end_index[0], "cash_weight"]), 0.0)

    def test_apply_risk_parity_preserves_existing_cash_weight(self):
        month_end_index = pd.to_datetime(["2020-01-31"])
        base_month_end_weight_df = pd.DataFrame(
            {"GLD": [0.4], "SPY": [0.0]},
            index=month_end_index,
            dtype=float,
        )
        month_end_volatility_df = pd.DataFrame(
            {"GLD": [0.10], "SPY": [0.20]},
            index=month_end_index,
            dtype=float,
        )
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD",),
            fallback_asset="SPY",
            rank_weight_vec=(1.0,),
            start_date_str="2012-01-01",
        )

        final_month_end_weight_df, diagnostic_df = (
            shared_risk_parity_helper.apply_risk_parity_to_month_end_weight_df(
                base_month_end_weight_df=base_month_end_weight_df,
                month_end_volatility_df=month_end_volatility_df,
                config=config,
            )
        )

        self.assertAlmostEqual(float(final_month_end_weight_df.loc[month_end_index[0], "GLD"]), 0.4)
        self.assertAlmostEqual(float(final_month_end_weight_df.loc[month_end_index[0], "SPY"]), 0.0)
        self.assertAlmostEqual(float(diagnostic_df.loc[month_end_index[0], "cash_weight"]), 0.6)

    def test_apply_risk_parity_skips_invalid_volatility_and_leaves_weight_as_cash(self):
        month_end_index = pd.to_datetime(["2020-01-31"])
        base_month_end_weight_df = pd.DataFrame(
            {"GLD": [0.4], "SPY": [0.6]},
            index=month_end_index,
            dtype=float,
        )
        month_end_volatility_df = pd.DataFrame(
            {"GLD": [0.10], "SPY": [0.0]},
            index=month_end_index,
            dtype=float,
        )
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD",),
            fallback_asset="SPY",
            rank_weight_vec=(1.0,),
            start_date_str="2012-01-01",
        )

        final_month_end_weight_df, diagnostic_df = (
            shared_risk_parity_helper.apply_risk_parity_to_month_end_weight_df(
                base_month_end_weight_df=base_month_end_weight_df,
                month_end_volatility_df=month_end_volatility_df,
                config=config,
            )
        )

        self.assertAlmostEqual(float(final_month_end_weight_df.loc[month_end_index[0], "GLD"]), 0.4)
        self.assertAlmostEqual(float(final_month_end_weight_df.loc[month_end_index[0], "SPY"]), 0.0)
        self.assertAlmostEqual(float(diagnostic_df.loc[month_end_index[0], "skipped_weight"]), 0.6)
        self.assertAlmostEqual(float(diagnostic_df.loc[month_end_index[0], "cash_weight"]), 0.6)

    def test_build_risk_parity_overlay_weight_frames_uses_first_next_month_open(self):
        execution_index = pd.bdate_range("2020-01-01", periods=90)
        execution_price_df = make_execution_price_df(["GLD", "SPY"])
        month_end_index = pd.to_datetime(["2020-01-31", "2020-02-29"])
        base_month_end_weight_df = pd.DataFrame(
            {"GLD": [0.4, 0.4], "SPY": [0.6, 0.6]},
            index=month_end_index,
            dtype=float,
        )
        config = DefenseFirstConfig(
            defensive_asset_list=("GLD",),
            fallback_asset="SPY",
            rank_weight_vec=(1.0,),
            start_date_str="2012-01-01",
        )

        _, _, _, rebalance_weight_df, _ = shared_risk_parity_helper.build_risk_parity_overlay_weight_frames(
            base_month_end_weight_df=base_month_end_weight_df,
            execution_price_df=execution_price_df,
            execution_index=execution_index,
            config=config,
            lookback_day_int=5,
        )

        expected_index = pd.to_datetime(["2020-02-03", "2020-03-02"])
        self.assertTrue(rebalance_weight_df.index.equals(expected_index))

    def _run_smoke_variant(self, module_name_str: str) -> None:
        variant_module = importlib.import_module(module_name_str)
        config = variant_module.DEFAULT_CONFIG
        base_loader_output = make_standard_loader_output(config)
        execution_price_df = base_loader_output[0]

        with patch.object(variant_module, "get_defense_first_data", return_value=base_loader_output):
            with patch.object(
                shared_vix_helper,
                "load_helper_close_ser",
                side_effect=make_helper_close_ser_side_effect(execution_price_df.index),
            ):
                strategy = variant_module.run_variant(
                    risk_parity_lookback_day_int=5,
                    show_display_bool=False,
                    save_results_bool=False,
                )

        self.assertIsNotNone(strategy.summary)
        self.assertGreater(len(strategy.summary.index), 0)
        self.assertTrue(hasattr(strategy, "daily_vrp_signal_df"))
        self.assertTrue(hasattr(strategy, "daily_risk_parity_volatility_df"))
        self.assertTrue(hasattr(strategy, "month_end_risk_parity_diagnostic_df"))
        self.assertEqual(strategy.risk_parity_lookback_day_int, 5)

    def test_btal_spy_vix_cash_risk_parity_variant_smoke(self):
        self._run_smoke_variant(
            "strategies.taa_df.strategy_taa_df_btal_fallback_spy_vix_cash_risk_parity"
        )

    def test_btal_tqqq_vix_cash_risk_parity_variant_smoke(self):
        self._run_smoke_variant(
            "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash_risk_parity"
        )


if __name__ == "__main__":
    unittest.main()
