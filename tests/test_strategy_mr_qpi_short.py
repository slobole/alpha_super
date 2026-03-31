import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_mr_qpi_short import QPIShortStrategy


class QPIShortStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> QPIShortStrategy:
        base_kwargs = dict(
            name="QPIShortTest",
            benchmarks=["$SPX"],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            sma_window_int=20,
            qpi_lookback_years_int=1,
        )
        base_kwargs.update(kwargs)
        return QPIShortStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=280, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 50.0 + 0.10 * step_vec + 1.5 * np.sin(step_vec * 0.07)
        bbb_close_vec = 70.0 + 0.08 * step_vec + 1.2 * np.cos(step_vec * 0.05)
        ccc_close_vec = 40.0 + 0.06 * step_vec + 1.8 * np.sin(step_vec * 0.09 + 0.4)
        spx_close_vec = 4200.0 - 0.80 * step_vec + 12.0 * np.sin(step_vec * 0.03)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.25,
                ("AAA", "High"): aaa_close_vec + 0.60,
                ("AAA", "Low"): aaa_close_vec - 0.60,
                ("AAA", "Close"): aaa_close_vec,
                ("AAA", "Turnover"): 35_000_000.0 + 100_000.0 * step_vec,
                ("BBB", "Open"): bbb_close_vec - 0.20,
                ("BBB", "High"): bbb_close_vec + 0.55,
                ("BBB", "Low"): bbb_close_vec - 0.55,
                ("BBB", "Close"): bbb_close_vec,
                ("BBB", "Turnover"): 30_000_000.0 + 90_000.0 * step_vec,
                ("CCC", "Open"): ccc_close_vec - 0.15,
                ("CCC", "High"): ccc_close_vec + 0.50,
                ("CCC", "Low"): ccc_close_vec - 0.50,
                ("CCC", "Close"): ccc_close_vec,
                ("CCC", "Turnover"): 25_000_000.0 + 80_000.0 * step_vec,
                ("$SPX", "Open"): spx_close_vec - 8.0,
                ("$SPX", "High"): spx_close_vec + 10.0,
                ("$SPX", "Low"): spx_close_vec - 10.0,
                ("$SPX", "Close"): spx_close_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "three_day_return_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "qpi_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "sma_200_price_ser"), signal_data_df.columns)
        self.assertIn(("$SPX", "benchmark_sma_200_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

        last_bar_ts = pricing_data_df.index[-1]
        expected_three_day_return_float = float(
            pricing_data_df[("AAA", "Close")].pct_change(periods=3, fill_method=None).loc[last_bar_ts]
        )
        actual_three_day_return_float = float(
            signal_data_df.loc[last_bar_ts, ("AAA", "three_day_return_ser")]
        )
        self.assertAlmostEqual(actual_three_day_return_float, expected_three_day_return_float)

    def test_get_opportunity_list_enforces_short_filters_and_ranking(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "HIGHQ": [1],
                "ABOVEMA": [1],
                "DOWN3D": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("$SPX", "Close"): 4_400.0,
                ("$SPX", "benchmark_sma_200_ser"): 4_500.0,
                ("AAA", "Close"): 95.0,
                ("AAA", "Turnover"): 35_000_000.0,
                ("AAA", "qpi_value_ser"): 10.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                ("AAA", "three_day_return_ser"): 0.04,
                ("BBB", "Close"): 94.0,
                ("BBB", "Turnover"): 50_000_000.0,
                ("BBB", "qpi_value_ser"): 8.0,
                ("BBB", "sma_200_price_ser"): 100.0,
                ("BBB", "three_day_return_ser"): 0.03,
                ("CCC", "Close"): 93.0,
                ("CCC", "Turnover"): 45_000_000.0,
                ("CCC", "qpi_value_ser"): 7.0,
                ("CCC", "sma_200_price_ser"): 100.0,
                ("CCC", "three_day_return_ser"): 0.02,
                ("HIGHQ", "Close"): 92.0,
                ("HIGHQ", "Turnover"): 60_000_000.0,
                ("HIGHQ", "qpi_value_ser"): 20.0,
                ("HIGHQ", "sma_200_price_ser"): 100.0,
                ("HIGHQ", "three_day_return_ser"): 0.05,
                ("ABOVEMA", "Close"): 105.0,
                ("ABOVEMA", "Turnover"): 55_000_000.0,
                ("ABOVEMA", "qpi_value_ser"): 9.0,
                ("ABOVEMA", "sma_200_price_ser"): 100.0,
                ("ABOVEMA", "three_day_return_ser"): 0.06,
                ("DOWN3D", "Close"): 91.0,
                ("DOWN3D", "Turnover"): 65_000_000.0,
                ("DOWN3D", "qpi_value_ser"): 6.0,
                ("DOWN3D", "sma_200_price_ser"): 100.0,
                ("DOWN3D", "three_day_return_ser"): -0.04,
                ("OUT", "Close"): 89.0,
                ("OUT", "Turnover"): 90_000_000.0,
                ("OUT", "qpi_value_ser"): 4.0,
                ("OUT", "sma_200_price_ser"): 100.0,
                ("OUT", "three_day_return_ser"): 0.05,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, ["BBB", "CCC", "AAA"])

    def test_get_opportunity_list_returns_empty_when_benchmark_is_not_in_bear_market(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("$SPX", "Close"): 4_550.0,
                ("$SPX", "benchmark_sma_200_ser"): 4_500.0,
                ("AAA", "Close"): 95.0,
                ("AAA", "Turnover"): 35_000_000.0,
                ("AAA", "qpi_value_ser"): 10.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                ("AAA", "three_day_return_ser"): 0.04,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, [])

    def test_iterate_submits_cover_order_when_close_breaks_prior_low(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(7, pd.Timestamp("2024-03-07"), "AAA", -10, 100.0, -1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7

        data_df = pd.DataFrame(
            {
                ("AAA", "Low"): [92.0, 91.5, 91.0, 90.5, 90.0],
            },
            index=pd.bdate_range("2024-03-04", periods=5),
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 89.0,
                ("$SPX", "Close"): 4_400.0,
                ("$SPX", "benchmark_sma_200_ser"): 4_500.0,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        self.assertIsInstance(order_list[0], MarketOrder)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[0].amount, 0.0)
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 7)

    def test_iterate_enters_highest_ranked_short_candidate_when_no_cover_is_triggered(self):
        strategy = self.make_strategy(max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.trade_id_int = 10
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1], "CCC": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(10, pd.Timestamp("2024-03-07"), "AAA", -10, 100.0, -1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 10

        data_df = pd.DataFrame(
            {
                ("AAA", "Low"): [99.0, 99.0, 99.0, 99.0, 99.0],
            },
            index=pd.bdate_range("2024-03-04", periods=5),
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)

        close_row_ser = self.make_close_row_ser(
            {
                ("$SPX", "Close"): 4_400.0,
                ("$SPX", "benchmark_sma_200_ser"): 4_500.0,
                ("AAA", "Close"): 100.0,
                ("AAA", "Turnover"): 35_000_000.0,
                ("AAA", "qpi_value_ser"): 40.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                ("AAA", "three_day_return_ser"): 0.02,
                ("BBB", "Close"): 94.0,
                ("BBB", "Turnover"): 50_000_000.0,
                ("BBB", "qpi_value_ser"): 8.0,
                ("BBB", "sma_200_price_ser"): 100.0,
                ("BBB", "three_day_return_ser"): 0.04,
                ("CCC", "Close"): 93.0,
                ("CCC", "Turnover"): 40_000_000.0,
                ("CCC", "qpi_value_ser"): 9.0,
                ("CCC", "sma_200_price_ser"): 100.0,
                ("CCC", "three_day_return_ser"): 0.03,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "BBB")
        self.assertEqual(entry_order.unit, "value")
        self.assertFalse(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, -50_000.0)
        self.assertEqual(entry_order.trade_id, 11)


if __name__ == "__main__":
    unittest.main()
