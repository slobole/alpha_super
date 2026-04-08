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
from strategies.dv2.strategy_mr_dv2_short_vix_filter import (
    BENCHMARK_SYMBOL_STR,
    DV2ShortVixFilterStrategy,
    DV2_SHORT_ENTRY_MIN_FLOAT,
    P126_SHORT_RETURN_MAX_FLOAT,
    VIX_SIGNAL_SYMBOL_STR,
)


class DV2ShortVixFilterStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> DV2ShortVixFilterStrategy:
        base_kwargs = dict(
            name="DV2ShortVixFilterTest",
            benchmarks=[],
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=2,
        )
        base_kwargs.update(kwargs)
        return DV2ShortVixFilterStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=260, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 120.0 - 0.18 * step_vec + 1.6 * np.sin(step_vec * 0.05)
        bbb_close_vec = 100.0 - 0.15 * step_vec + 1.3 * np.cos(step_vec * 0.06)
        spx_close_vec = 4200.0 - 2.8 * step_vec + 18.0 * np.sin(step_vec * 0.03)
        vix_close_vec = 24.0 + 1.8 * np.sin(step_vec * 0.07) + 0.9 * np.cos(step_vec * 0.03)

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

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "p126d_return_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "natr_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "dv2_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "sma_200_price_ser"), signal_data_df.columns)
        self.assertIn((VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"), signal_data_df.columns)
        self.assertIn((VIX_SIGNAL_SYMBOL_STR, "vix_bull_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_get_opportunity_list_enforces_mirrored_short_filters(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "LOWDV2": [1],
                "ABOVEMA": [1],
                "POSRET": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 95.0,
                ("AAA", "p126d_return_ser"): -0.08,
                ("AAA", "natr_value_ser"): 4.0,
                ("AAA", "dv2_value_ser"): 95.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                ("BBB", "Close"): 94.0,
                ("BBB", "p126d_return_ser"): -0.09,
                ("BBB", "natr_value_ser"): 5.0,
                ("BBB", "dv2_value_ser"): 97.0,
                ("BBB", "sma_200_price_ser"): 100.0,
                ("CCC", "Close"): 93.0,
                ("CCC", "p126d_return_ser"): -0.07,
                ("CCC", "natr_value_ser"): 4.5,
                ("CCC", "dv2_value_ser"): 96.0,
                ("CCC", "sma_200_price_ser"): 100.0,
                ("LOWDV2", "Close"): 92.0,
                ("LOWDV2", "p126d_return_ser"): -0.08,
                ("LOWDV2", "natr_value_ser"): 6.0,
                ("LOWDV2", "dv2_value_ser"): 80.0,
                ("LOWDV2", "sma_200_price_ser"): 100.0,
                ("ABOVEMA", "Close"): 105.0,
                ("ABOVEMA", "p126d_return_ser"): -0.08,
                ("ABOVEMA", "natr_value_ser"): 6.0,
                ("ABOVEMA", "dv2_value_ser"): 98.0,
                ("ABOVEMA", "sma_200_price_ser"): 100.0,
                ("POSRET", "Close"): 90.0,
                ("POSRET", "p126d_return_ser"): -0.01,
                ("POSRET", "natr_value_ser"): 6.0,
                ("POSRET", "dv2_value_ser"): 98.0,
                ("POSRET", "sma_200_price_ser"): 100.0,
                ("OUT", "Close"): 89.0,
                ("OUT", "p126d_return_ser"): -0.10,
                ("OUT", "natr_value_ser"): 7.0,
                ("OUT", "dv2_value_ser"): 99.0,
                ("OUT", "sma_200_price_ser"): 100.0,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, ["BBB", "CCC", "AAA"])

    def test_short_threshold_constants_match_spec(self):
        self.assertEqual(DV2_SHORT_ENTRY_MIN_FLOAT, 90.0)
        self.assertEqual(P126_SHORT_RETURN_MAX_FLOAT, -0.05)

    def test_iterate_blocks_entry_when_vix_regime_is_not_bear(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 95.0,
                ("AAA", "p126d_return_ser"): -0.08,
                ("AAA", "natr_value_ser"): 4.0,
                ("AAA", "dv2_value_ser"): 95.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"): False,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=pd.bdate_range("2024-03-06", periods=3)),
            close_row_ser,
            pd.Series(dtype=float),
        )

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_allows_entry_when_vix_regime_is_bear(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 95.0,
                ("AAA", "p126d_return_ser"): -0.08,
                ("AAA", "natr_value_ser"): 4.0,
                ("AAA", "dv2_value_ser"): 95.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"): True,
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
        self.assertFalse(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, -50_000.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_keeps_cover_rule_active_even_when_vix_is_not_bear(self):
        strategy = self.make_strategy()
        strategy.trade_id_int = 4
        strategy.current_trade_id_map["AAA"] = 4
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(4, pd.Timestamp("2024-03-07"), "AAA", -10, 100.0, -1_000.0, 1, 0.0)

        data_df = pd.DataFrame(
            {
                ("AAA", "Low"): [100.0, 99.0, 98.0],
            },
            index=pd.bdate_range("2024-03-06", periods=3),
            dtype=float,
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 97.0,
                (VIX_SIGNAL_SYMBOL_STR, "vix_bear_bool"): False,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        cover_order = order_list[0]
        self.assertIsInstance(cover_order, MarketOrder)
        self.assertEqual(cover_order.asset, "AAA")
        self.assertEqual(cover_order.amount, 0.0)
        self.assertTrue(cover_order.target)
        self.assertEqual(cover_order.trade_id, 4)

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
