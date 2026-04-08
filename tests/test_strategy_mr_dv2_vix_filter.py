import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.dv2.strategy_mr_dv2_vix_filter import (
    BENCHMARK_SYMBOL_STR,
    DV2VixFilterStrategy,
    VIX_BEAR_BUFFER_FLOAT,
    VIX_SIGNAL_SYMBOL_STR,
    VIX_SMA_WINDOW_DAY_INT,
    compute_vix_regime_signal_df,
)


class DV2VixFilterStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> DV2VixFilterStrategy:
        base_kwargs = dict(
            name="DV2VixFilterTest",
            benchmarks=[],
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=2,
        )
        base_kwargs.update(kwargs)
        return DV2VixFilterStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=260, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 50.0 + 0.18 * step_vec + 1.8 * np.sin(step_vec * 0.05)
        bbb_close_vec = 40.0 + 0.14 * step_vec + 1.5 * np.cos(step_vec * 0.06)
        spx_close_vec = 4000.0 + 3.5 * step_vec + 18.0 * np.sin(step_vec * 0.03)
        vix_close_vec = 19.0 + 1.6 * np.sin(step_vec * 0.07) + 0.8 * np.cos(step_vec * 0.03)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.25,
                ("AAA", "High"): aaa_close_vec + 0.60,
                ("AAA", "Low"): aaa_close_vec - 0.60,
                ("AAA", "Close"): aaa_close_vec,
                ("BBB", "Open"): bbb_close_vec - 0.20,
                ("BBB", "High"): bbb_close_vec + 0.55,
                ("BBB", "Low"): bbb_close_vec - 0.55,
                ("BBB", "Close"): bbb_close_vec,
                (VIX_SIGNAL_SYMBOL_STR, "Open"): vix_close_vec + 0.10,
                (VIX_SIGNAL_SYMBOL_STR, "High"): vix_close_vec + 0.45,
                (VIX_SIGNAL_SYMBOL_STR, "Low"): vix_close_vec - 0.45,
                (VIX_SIGNAL_SYMBOL_STR, "Close"): vix_close_vec,
                (BENCHMARK_SYMBOL_STR, "Open"): spx_close_vec - 4.0,
                (BENCHMARK_SYMBOL_STR, "High"): spx_close_vec + 8.0,
                (BENCHMARK_SYMBOL_STR, "Low"): spx_close_vec - 8.0,
                (BENCHMARK_SYMBOL_STR, "Close"): spx_close_vec,
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_vix_regime_signal_df_matches_formula_and_strict_threshold(self):
        date_index = pd.date_range("2024-01-02", periods=17, freq="B")
        boundary_vix_close_float = 322.0 / 13.85
        below_boundary_vix_close_float = boundary_vix_close_float - 1e-9
        vix_close_ser = pd.Series(
            [20.0] * 15 + [below_boundary_vix_close_float, 25.0],
            index=date_index,
            dtype=float,
        )

        signal_df = compute_vix_regime_signal_df(vix_close_ser)

        first_ready_timestamp = date_index[VIX_SMA_WINDOW_DAY_INT - 1]
        expected_first_threshold_float = 20.0 * (1.0 + VIX_BEAR_BUFFER_FLOAT)
        self.assertAlmostEqual(
            float(signal_df.loc[first_ready_timestamp, "vix_sma_15_ser"]),
            20.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(signal_df.loc[first_ready_timestamp, "vix_bear_threshold_ser"]),
            expected_first_threshold_float,
            places=12,
        )
        self.assertFalse(bool(signal_df.loc[first_ready_timestamp, "vix_bear_bool"]))
        self.assertTrue(bool(signal_df.loc[first_ready_timestamp, "vix_bull_bool"]))

        boundary_timestamp = date_index[VIX_SMA_WINDOW_DAY_INT]
        expected_boundary_threshold_float = (1.0 + VIX_BEAR_BUFFER_FLOAT) * (
            ((14.0 * 20.0) + below_boundary_vix_close_float) / 15.0
        )
        self.assertAlmostEqual(
            float(signal_df.loc[boundary_timestamp, "vix_bear_threshold_ser"]),
            expected_boundary_threshold_float,
            places=12,
        )
        threshold_gap_float = float(
            signal_df.loc[boundary_timestamp, "vix_bear_threshold_ser"]
            - signal_df.loc[boundary_timestamp, "vix_close_ser"]
        )
        self.assertGreater(threshold_gap_float, 0.0)
        self.assertLess(threshold_gap_float, 1e-8)
        self.assertFalse(bool(signal_df.loc[boundary_timestamp, "vix_bear_bool"]))

        bear_timestamp = date_index[VIX_SMA_WINDOW_DAY_INT + 1]
        self.assertTrue(bool(signal_df.loc[bear_timestamp, "vix_bear_bool"]))
        self.assertFalse(bool(signal_df.loc[bear_timestamp, "vix_bull_bool"]))

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "p126d_return_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "natr_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "dv2_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "sma_200_price_ser"), signal_data_df.columns)
        self.assertIn((VIX_SIGNAL_SYMBOL_STR, "vix_sma_15_ser"), signal_data_df.columns)
        self.assertIn((VIX_SIGNAL_SYMBOL_STR, "vix_bear_threshold_ser"), signal_data_df.columns)
        self.assertIn((VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"), signal_data_df.columns)
        self.assertIn((VIX_SIGNAL_SYMBOL_STR, "vix_bull_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_blocks_entry_when_vix_regime_is_bear(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 110.0,
                ("AAA", "p126d_return_ser"): 0.12,
                ("AAA", "natr_value_ser"): 4.0,
                ("AAA", "dv2_value_ser"): 6.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"): True,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=pd.bdate_range("2024-03-06", periods=3)),
            close_row_ser,
            pd.Series(dtype=float),
        )

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_allows_entry_when_vix_regime_is_bull(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 110.0,
                ("AAA", "p126d_return_ser"): 0.12,
                ("AAA", "natr_value_ser"): 4.0,
                ("AAA", "dv2_value_ser"): 6.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"): False,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=pd.bdate_range("2024-03-06", periods=3)),
            close_row_ser,
            pd.Series(dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "AAA")
        self.assertEqual(entry_order.unit, "value")
        self.assertAlmostEqual(entry_order.amount, 50_000.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_keeps_legacy_exit_rule_active_in_bear_regime(self):
        strategy = self.make_strategy()
        strategy.trade_id_int = 4
        strategy.current_trade_id_map["AAA"] = 4
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(4, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)

        data_df = pd.DataFrame(
            {
                ("AAA", "High"): [100.0, 101.0, 102.0],
            },
            index=pd.bdate_range("2024-03-06", periods=3),
            dtype=float,
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 105.0,
                (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"): True,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "AAA")
        self.assertEqual(exit_order.amount, 0.0)
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.trade_id, 4)

    def test_run_daily_smoke_generates_summary(self):
        strategy = self.make_strategy(benchmarks=[BENCHMARK_SYMBOL_STR])
        pricing_data_df = self.make_pricing_data_df()
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": np.ones(len(pricing_data_df.index), dtype=int),
                "BBB": np.ones(len(pricing_data_df.index), dtype=int),
            },
            index=pricing_data_df.index,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=pricing_data_df.index,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)


if __name__ == "__main__":
    unittest.main()
