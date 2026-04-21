import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.taa_growth_inflation.strategy_taa_growth_inflation_sector_timing import (
    GrowthInflationSectorConfig,
    GrowthInflationSectorTimingStrategy,
    compute_growth_inflation_signal_tables,
)


class GrowthInflationSectorTimingStrategyTests(unittest.TestCase):
    def make_config(
        self,
        growth_sma_window_int: int = 3,
        inflation_median_window_int: int = 3,
    ) -> GrowthInflationSectorConfig:
        return GrowthInflationSectorConfig(
            growth_sma_window_int=growth_sma_window_int,
            inflation_median_window_int=inflation_median_window_int,
            benchmark_list=(),
            fallback_asset_str="BIL",
        )

    def make_strategy(self, config: GrowthInflationSectorConfig) -> GrowthInflationSectorTimingStrategy:
        return GrowthInflationSectorTimingStrategy(
            name="GrowthInflationSectorTest",
            benchmarks=[],
            config=config,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_signal_close_df(self) -> pd.DataFrame:
        trading_index = pd.date_range("2024-01-01", periods=8, freq="B")
        growth_close_vec = np.array([100.0, 100.0, 100.0, 105.0, 110.0, 90.0, 80.0, 120.0], dtype=float)
        ratio_vec = np.array([1.0, 1.0, 1.0, 1.2, 1.3, 0.8, 1.4, 0.7], dtype=float)
        positive_close_vec = 100.0 * ratio_vec
        negative_close_vec = np.full(len(trading_index), 100.0, dtype=float)

        signal_close_df = pd.DataFrame(
            {
                "SPY": growth_close_vec,
                "XLE": positive_close_vec,
                "XLI": positive_close_vec,
                "XLF": positive_close_vec,
                "XLB": positive_close_vec,
                "XLU": negative_close_vec,
                "XLP": negative_close_vec,
                "XLV": negative_close_vec,
            },
            index=trading_index,
            dtype=float,
        )
        return signal_close_df

    def make_pricing_data_df(self, periods_int: int = 40) -> pd.DataFrame:
        trading_index = pd.date_range("2024-01-01", periods=periods_int, freq="B")
        bar_idx_vec = np.arange(periods_int, dtype=float)

        growth_close_vec = 100.0 + 0.25 * bar_idx_vec + 4.0 * np.sin(bar_idx_vec / 2.5)
        ratio_vec = 1.0 + 0.18 * np.sin(bar_idx_vec / 3.0) + 0.05 * np.cos(bar_idx_vec / 7.0)
        positive_close_vec = 110.0 * ratio_vec
        negative_close_vec = 100.0 + 0.8 * np.sin(bar_idx_vec / 9.0)

        close_map = {
            "SPY": growth_close_vec,
            "XLE": 90.0 + 0.35 * bar_idx_vec + 0.6 * np.sin(bar_idx_vec / 4.0),
            "XLK": 95.0 + 0.40 * bar_idx_vec + 0.8 * np.cos(bar_idx_vec / 5.0),
            "XLV": 88.0 + 0.28 * bar_idx_vec + 0.5 * np.sin(bar_idx_vec / 6.0),
            "XLP": 86.0 + 0.20 * bar_idx_vec + 0.4 * np.cos(bar_idx_vec / 7.0),
            "XLI": 92.0 + 0.24 * bar_idx_vec + 0.4 * np.sin(bar_idx_vec / 8.0),
            "XLF": 84.0 + 0.22 * bar_idx_vec + 0.4 * np.cos(bar_idx_vec / 8.5),
            "XLB": 82.0 + 0.21 * bar_idx_vec + 0.4 * np.sin(bar_idx_vec / 9.0),
            "XLU": 78.0 + 0.10 * bar_idx_vec + 0.2 * np.cos(bar_idx_vec / 10.0),
            "BIL": 100.0 + 0.01 * bar_idx_vec,
        }

        signal_close_map = {
            "SPY": growth_close_vec,
            "XLE": positive_close_vec,
            "XLI": positive_close_vec,
            "XLF": positive_close_vec,
            "XLB": positive_close_vec,
            "XLU": negative_close_vec,
            "XLP": negative_close_vec,
            "XLV": negative_close_vec,
        }

        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}
        for asset_str, close_vec in close_map.items():
            open_vec = close_vec * 0.999
            high_vec = np.maximum(open_vec, close_vec) * 1.001
            low_vec = np.minimum(open_vec, close_vec) * 0.999
            pricing_data_map[(asset_str, "Open")] = open_vec
            pricing_data_map[(asset_str, "High")] = high_vec
            pricing_data_map[(asset_str, "Low")] = low_vec
            pricing_data_map[(asset_str, "Close")] = close_vec

            if asset_str in signal_close_map:
                pricing_data_map[(asset_str, "SignalClose")] = signal_close_map[asset_str]

        pricing_data_df = pd.DataFrame(pricing_data_map, index=trading_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool | str]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_growth_inflation_signal_tables_matches_formula_and_regime_map(self):
        config = self.make_config()
        signal_close_df = self.make_signal_close_df()

        (
            growth_price_ser,
            growth_sma_ser,
            growth_rising_bool_ser,
            inflation_ratio_ser,
            inflation_median_ser,
            inflation_rising_bool_ser,
            regime_name_ser,
            target_asset_ser,
            target_weight_df,
        ) = compute_growth_inflation_signal_tables(
            signal_close_df=signal_close_df,
            config=config,
        )

        reflation_ts = pd.Timestamp("2024-01-05")
        deflation_ts = pd.Timestamp("2024-01-08")
        stagflation_ts = pd.Timestamp("2024-01-09")
        goldilocks_ts = pd.Timestamp("2024-01-10")

        self.assertAlmostEqual(float(growth_price_ser.loc[reflation_ts]), 110.0)
        self.assertAlmostEqual(float(growth_sma_ser.loc[reflation_ts]), (100.0 + 105.0 + 110.0) / 3.0)
        self.assertTrue(bool(growth_rising_bool_ser.loc[reflation_ts]))

        self.assertAlmostEqual(float(inflation_ratio_ser.loc[reflation_ts]), 1.30, places=12)
        self.assertAlmostEqual(float(inflation_median_ser.loc[reflation_ts]), 1.20, places=12)
        self.assertTrue(bool(inflation_rising_bool_ser.loc[reflation_ts]))

        self.assertEqual(str(regime_name_ser.loc[reflation_ts]), "Reflation")
        self.assertEqual(str(target_asset_ser.loc[reflation_ts]), "XLE")
        self.assertAlmostEqual(float(target_weight_df.loc[reflation_ts, "XLE"]), 1.0, places=12)

        self.assertEqual(str(regime_name_ser.loc[deflation_ts]), "Deflation")
        self.assertEqual(str(target_asset_ser.loc[deflation_ts]), "XLP")
        self.assertAlmostEqual(float(target_weight_df.loc[deflation_ts, "XLP"]), 1.0, places=12)

        self.assertEqual(str(regime_name_ser.loc[stagflation_ts]), "Stagflation")
        self.assertEqual(str(target_asset_ser.loc[stagflation_ts]), "XLV")
        self.assertAlmostEqual(float(target_weight_df.loc[stagflation_ts, "XLV"]), 1.0, places=12)

        self.assertEqual(str(regime_name_ser.loc[goldilocks_ts]), "Goldilocks")
        self.assertEqual(str(target_asset_ser.loc[goldilocks_ts]), "XLK")
        self.assertAlmostEqual(float(target_weight_df.loc[goldilocks_ts, "XLK"]), 1.0, places=12)
        self.assertTrue(np.allclose(target_weight_df.sum(axis=1).iloc[3:].to_numpy(dtype=float), 1.0, atol=1e-12))

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        config = self.make_config(growth_sma_window_int=5, inflation_median_window_int=5)
        strategy = self.make_strategy(config=config)
        pricing_data_df = self.make_pricing_data_df(periods_int=30)

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("SPY", "growth_sma_5d_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "growth_rising_bool"), signal_data_df.columns)
        self.assertIn(("_MODEL", "inflation_ratio_ser"), signal_data_df.columns)
        self.assertIn(("_MODEL", "inflation_median_5d_ser"), signal_data_df.columns)
        self.assertIn(("_MODEL", "regime_name_str"), signal_data_df.columns)
        self.assertIn(("XLK", "target_weight_ser"), signal_data_df.columns)
        self.assertIn(("BIL", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_rotates_from_old_sector_into_new_sector(self):
        config = self.make_config()
        strategy = self.make_strategy(config=config)
        strategy.previous_bar = pd.Timestamp("2024-01-05")
        strategy.current_bar = pd.Timestamp("2024-01-08")
        strategy.trade_id_int = 4
        strategy.add_transaction(4, strategy.previous_bar, "XLE", 100, 100.0, 10_000.0, 1, 0.0)
        strategy.current_trade_id_map["XLE"] = 4

        close_row_ser = self.make_close_row_ser(
            {
                ("XLK", "target_weight_ser"): 0.0,
                ("XLE", "target_weight_ser"): 0.0,
                ("XLV", "target_weight_ser"): 1.0,
                ("XLP", "target_weight_ser"): 0.0,
                ("BIL", "target_weight_ser"): 0.0,
            }
        )
        open_price_ser = pd.Series(
            {"XLK": 100.0, "XLE": 100.0, "XLV": 100.0, "XLP": 100.0, "BIL": 100.0},
            dtype=float,
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertEqual(order_list[0].asset, "XLE")
        self.assertEqual(order_list[0].amount, 0.0)
        self.assertEqual(order_list[0].trade_id, 4)
        self.assertEqual(order_list[1].asset, "XLV")
        self.assertEqual(order_list[1].amount, 1.0)
        self.assertEqual(order_list[1].trade_id, 5)
        self.assertEqual(strategy.current_trade_id_map["XLV"], 5)

    def test_run_daily_smoke_generates_summary_and_daily_target_weights(self):
        config = self.make_config(growth_sma_window_int=5, inflation_median_window_int=5)
        strategy = self.make_strategy(config=config)
        pricing_data_df = self.make_pricing_data_df(periods_int=60)

        signal_data_df = strategy.compute_signals(pricing_data_df)
        target_weight_df = pd.DataFrame(
            {
                asset_str: signal_data_df[(asset_str, "target_weight_ser")].astype(float)
                for asset_str in config.trade_asset_list
            },
            index=signal_data_df.index,
            dtype=float,
        )
        actionable_weight_mask_ser = target_weight_df.sum(axis=1) > 0.0
        first_actionable_ts = pd.Timestamp(actionable_weight_mask_ser[actionable_weight_mask_ser].index[0])
        calendar_index = pricing_data_df.index[pricing_data_df.index >= first_actionable_ts]

        run_daily(
            strategy,
            pricing_data_df,
            calendar=calendar_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
            audit_sample_size_int=5,
        )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)
        self.assertGreater(len(strategy.daily_target_weights), 0)
        self.assertTrue({"XLK", "XLE", "XLV", "XLP", "BIL", "Cash"}.issubset(strategy.daily_target_weights.columns))

        weight_sum_ser = strategy.daily_target_weights.sum(axis=1)
        self.assertTrue(np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
