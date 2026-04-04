import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.strategy_taa_baa import (
    BoldAssetAllocationConfig,
    BoldAssetAllocationStrategy,
    compute_bold_asset_allocation_month_end_weight_df,
    map_month_end_weights_to_rebalance_open_df,
)


class BoldAssetAllocationStrategyTests(unittest.TestCase):
    def setUp(self):
        self.config = BoldAssetAllocationConfig()
        self.month_end_index = pd.date_range("2023-01-31", periods=14, freq="ME")

    def make_signal_close_df(self, growth_rate_map: dict[str, float]) -> pd.DataFrame:
        month_step_vec = np.arange(len(self.month_end_index), dtype=float)
        close_map: dict[str, np.ndarray] = {}

        for asset_str in self.config.signal_asset_list:
            growth_rate_float = growth_rate_map[asset_str]
            close_map[asset_str] = 100.0 * np.power(1.0 + growth_rate_float, month_step_vec)

        return pd.DataFrame(close_map, index=self.month_end_index)

    def make_growth_rate_map(self) -> dict[str, float]:
        return {
            "SPY": 0.010,
            "QQQ": 0.030,
            "IWM": 0.028,
            "VGK": 0.026,
            "EWJ": 0.024,
            "EEM": 0.022,
            "VNQ": 0.020,
            "DBC": 0.018,
            "GLD": 0.016,
            "TLT": 0.014,
            "HYG": 0.012,
            "LQD": 0.011,
            "EFA": 0.009,
            "AGG": 0.008,
            "TIP": 0.007,
            "BIL": 0.004,
            "IEF": 0.006,
        }

    def make_pricing_data(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-29", periods=8, freq="D")
        close_vec = np.linspace(100.0, 107.0, len(date_index))

        pricing_data = pd.DataFrame(
            {
                ("SPY", "Open"): close_vec - 0.5,
                ("SPY", "High"): close_vec + 0.5,
                ("SPY", "Low"): close_vec - 1.0,
                ("SPY", "Close"): close_vec,
            },
            index=date_index,
        )
        pricing_data.columns = pd.MultiIndex.from_tuples(pricing_data.columns)
        return pricing_data

    def test_compute_month_end_weights_selects_top_offensive_assets_when_all_canaries_positive(self):
        growth_rate_map = self.make_growth_rate_map()
        signal_close_df = self.make_signal_close_df(growth_rate_map)

        canary_score_df, relative_score_df, month_end_weight_df, regime_ser = (
            compute_bold_asset_allocation_month_end_weight_df(signal_close_df, self.config)
        )

        expected_asset_list = ["QQQ", "IWM", "VGK", "EWJ", "EEM", "VNQ"]
        target_weight_ser = month_end_weight_df.iloc[-1]

        self.assertEqual(regime_ser.iloc[-1], "offensive")
        self.assertTrue((canary_score_df.iloc[-1] > 0.0).all())
        self.assertTrue(np.isclose(target_weight_ser.sum(), 1.0, atol=1e-12))
        self.assertEqual(relative_score_df.index[-1], month_end_weight_df.index[-1])

        for asset_str in expected_asset_list:
            self.assertAlmostEqual(target_weight_ser.loc[asset_str], 1.0 / 6.0)

        zero_weight_asset_list = sorted(set(self.config.tradeable_asset_list) - set(expected_asset_list))
        for asset_str in zero_weight_asset_list:
            self.assertAlmostEqual(target_weight_ser.loc[asset_str], 0.0)

    def test_compute_month_end_weights_uses_defensive_universe_when_any_canary_is_negative(self):
        growth_rate_map = self.make_growth_rate_map()
        growth_rate_map["TIP"] = 0.025
        growth_rate_map["DBC"] = 0.023
        growth_rate_map["IEF"] = 0.021
        signal_close_df = self.make_signal_close_df(growth_rate_map)

        last_date = signal_close_df.index[-1]
        previous_date = signal_close_df.index[-2]
        signal_close_df.loc[last_date, "EFA"] = signal_close_df.loc[previous_date, "EFA"] * 0.70

        _, _, month_end_weight_df, regime_ser = compute_bold_asset_allocation_month_end_weight_df(
            signal_close_df,
            self.config,
        )

        target_weight_ser = month_end_weight_df.iloc[-1]

        self.assertEqual(regime_ser.iloc[-1], "defensive")
        self.assertAlmostEqual(target_weight_ser.loc["TIP"], 1.0 / 3.0)
        self.assertAlmostEqual(target_weight_ser.loc["DBC"], 1.0 / 3.0)
        self.assertAlmostEqual(target_weight_ser.loc["IEF"], 1.0 / 3.0)
        self.assertAlmostEqual(target_weight_ser.loc["BIL"], 0.0)
        self.assertTrue(np.isclose(target_weight_ser.sum(), 1.0, atol=1e-12))

    def test_compute_month_end_weights_redirects_multiple_defensive_slots_to_bil(self):
        growth_rate_map = self.make_growth_rate_map()
        growth_rate_map["BIL"] = 0.020
        growth_rate_map["TIP"] = 0.019
        growth_rate_map["IEF"] = 0.018
        growth_rate_map["DBC"] = 0.010
        growth_rate_map["TLT"] = 0.009
        growth_rate_map["LQD"] = 0.008
        growth_rate_map["AGG"] = 0.007
        signal_close_df = self.make_signal_close_df(growth_rate_map)

        last_date = signal_close_df.index[-1]
        previous_date = signal_close_df.index[-2]
        signal_close_df.loc[last_date, "EFA"] = signal_close_df.loc[previous_date, "EFA"] * 0.60

        _, _, month_end_weight_df, regime_ser = compute_bold_asset_allocation_month_end_weight_df(
            signal_close_df,
            self.config,
        )

        target_weight_ser = month_end_weight_df.iloc[-1]

        self.assertEqual(regime_ser.iloc[-1], "defensive")
        self.assertAlmostEqual(target_weight_ser.loc["BIL"], 1.0)
        non_bil_weight_float = float(target_weight_ser.drop(labels=["BIL"]).sum())
        self.assertAlmostEqual(non_bil_weight_float, 0.0)

    def test_map_month_end_weights_to_rebalance_open_df_uses_first_trading_day_of_next_month(self):
        month_end_weight_df = pd.DataFrame(
            {"SPY": [1.0, 0.0], "BIL": [0.0, 1.0]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        execution_index = pd.to_datetime(["2024-02-01", "2024-02-05", "2024-03-01", "2024-03-04"])

        rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_index)

        expected_weight_df = pd.DataFrame(
            {"SPY": [1.0, 0.0], "BIL": [0.0, 1.0]},
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
        )
        expected_weight_df.index.name = "rebalance_date"

        pd.testing.assert_frame_equal(rebalance_weight_df, expected_weight_df)

    def test_compute_signals_passes_signal_audit(self):
        rebalance_weight_df = pd.DataFrame(
            {"SPY": [1.0], "BIL": [0.0]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        strategy = BoldAssetAllocationStrategy(
            name="BAATest",
            benchmarks=[],
            rebalance_weight_df=rebalance_weight_df,
            tradeable_asset_list=["SPY", "BIL"],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        pricing_data = self.make_pricing_data()

        signal_data = strategy.compute_signals(pricing_data)

        self.assertIn(("SPY", "target_weight"), signal_data.columns)
        self.assertIn(("BIL", "target_weight"), signal_data.columns)

        strategy.audit_signals(pricing_data, signal_data)


if __name__ == "__main__":
    unittest.main()
