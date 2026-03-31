import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_mr_qpi_long_short import QPILongShortStrategy


class QPILongShortStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> QPILongShortStrategy:
        base_kwargs = dict(
            name="QPILongShortTest",
            benchmarks=[],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            sma_window_int=20,
            qpi_lookback_years_int=1,
        )
        base_kwargs.update(kwargs)
        return QPILongShortStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=280, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 50.0 + 0.10 * step_vec + 1.5 * np.sin(step_vec * 0.07)
        bbb_close_vec = 70.0 + 0.08 * step_vec + 1.2 * np.cos(step_vec * 0.05)
        ccc_close_vec = 40.0 + 0.06 * step_vec + 1.8 * np.sin(step_vec * 0.09 + 0.4)

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

        strategy.audit_signals(pricing_data_df, signal_data_df)

        last_bar_ts = pricing_data_df.index[-1]
        expected_three_day_return_float = float(
            pricing_data_df[("AAA", "Close")].pct_change(periods=3, fill_method=None).loc[last_bar_ts]
        )
        actual_three_day_return_float = float(
            signal_data_df.loc[last_bar_ts, ("AAA", "three_day_return_ser")]
        )
        self.assertAlmostEqual(actual_three_day_return_float, expected_three_day_return_float)

    def test_long_and_short_opportunity_lists_enforce_filters_and_ranking(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "HIGHQ": [1],
                "WRONGSIGN": [1],
                "ABOVEMA": [1],
                "BELOWMA": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 105.0,
                ("AAA", "Turnover"): 30_000_000.0,
                ("AAA", "qpi_value_ser"): 10.0,
                ("AAA", "sma_200_price_ser"): 100.0,
                ("AAA", "three_day_return_ser"): -0.04,
                ("BBB", "Close"): 104.0,
                ("BBB", "Turnover"): 45_000_000.0,
                ("BBB", "qpi_value_ser"): 12.0,
                ("BBB", "sma_200_price_ser"): 100.0,
                ("BBB", "three_day_return_ser"): -0.03,
                ("CCC", "Close"): 95.0,
                ("CCC", "Turnover"): 35_000_000.0,
                ("CCC", "qpi_value_ser"): 11.0,
                ("CCC", "sma_200_price_ser"): 100.0,
                ("CCC", "three_day_return_ser"): 0.05,
                ("DDD", "Close"): 90.0,
                ("DDD", "Turnover"): 50_000_000.0,
                ("DDD", "qpi_value_ser"): 5.0,
                ("DDD", "sma_200_price_ser"): 100.0,
                ("DDD", "three_day_return_ser"): 0.04,
                ("HIGHQ", "Close"): 106.0,
                ("HIGHQ", "Turnover"): 60_000_000.0,
                ("HIGHQ", "qpi_value_ser"): 20.0,
                ("HIGHQ", "sma_200_price_ser"): 100.0,
                ("HIGHQ", "three_day_return_ser"): -0.05,
                ("WRONGSIGN", "Close"): 108.0,
                ("WRONGSIGN", "Turnover"): 55_000_000.0,
                ("WRONGSIGN", "qpi_value_ser"): 8.0,
                ("WRONGSIGN", "sma_200_price_ser"): 100.0,
                ("WRONGSIGN", "three_day_return_ser"): 0.06,
                ("ABOVEMA", "Close"): 110.0,
                ("ABOVEMA", "Turnover"): 48_000_000.0,
                ("ABOVEMA", "qpi_value_ser"): 7.0,
                ("ABOVEMA", "sma_200_price_ser"): 100.0,
                ("ABOVEMA", "three_day_return_ser"): 0.07,
                ("BELOWMA", "Close"): 92.0,
                ("BELOWMA", "Turnover"): 38_000_000.0,
                ("BELOWMA", "qpi_value_ser"): 6.0,
                ("BELOWMA", "sma_200_price_ser"): 100.0,
                ("BELOWMA", "three_day_return_ser"): -0.02,
                ("OUT", "Close"): 85.0,
                ("OUT", "Turnover"): 80_000_000.0,
                ("OUT", "qpi_value_ser"): 3.0,
                ("OUT", "sma_200_price_ser"): 100.0,
                ("OUT", "three_day_return_ser"): 0.05,
            }
        )

        long_opportunity_list = strategy.get_long_opportunity_list(close_row_ser)
        short_opportunity_list = strategy.get_short_opportunity_list(close_row_ser)

        self.assertEqual(long_opportunity_list, ["BBB", "AAA"])
        self.assertEqual(short_opportunity_list, ["DDD", "CCC"])

    def test_iterate_enters_long_and_short_positions_with_separate_budgets(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame(
            {"LONG1": [1], "SHORT1": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("LONG1", "Close"): 101.0,
                ("LONG1", "Turnover"): 40_000_000.0,
                ("LONG1", "qpi_value_ser"): 8.0,
                ("LONG1", "sma_200_price_ser"): 100.0,
                ("LONG1", "three_day_return_ser"): -0.03,
                ("SHORT1", "Close"): 99.0,
                ("SHORT1", "Turnover"): 42_000_000.0,
                ("SHORT1", "qpi_value_ser"): 7.0,
                ("SHORT1", "sma_200_price_ser"): 100.0,
                ("SHORT1", "three_day_return_ser"): 0.04,
            }
        )

        data_df = pd.DataFrame(
            {
                ("LONG_HELD", "High"): [101.0, 101.0, 101.0, 101.0, 101.0],
                ("SHORT_HELD", "Low"): [99.0, 99.0, 99.0, 99.0, 99.0],
            },
            index=pd.bdate_range("2024-03-04", periods=5),
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)

        strategy.iterate(
            data_df,
            close_row_ser,
            pd.Series(dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        order_map = {order.asset: order for order in order_list}
        self.assertEqual(set(order_map), {"LONG1", "SHORT1"})

        long_order = order_map["LONG1"]
        short_order = order_map["SHORT1"]

        self.assertIsInstance(long_order, MarketOrder)
        self.assertEqual(long_order.unit, "value")
        self.assertFalse(long_order.target)
        self.assertAlmostEqual(long_order.amount, 25_000.0)

        self.assertIsInstance(short_order, MarketOrder)
        self.assertEqual(short_order.unit, "value")
        self.assertFalse(short_order.target)
        self.assertAlmostEqual(short_order.amount, -(100_000.0 / 12.0))

    def test_iterate_submits_symmetric_exit_orders_for_long_and_short_positions(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(11, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(12, pd.Timestamp("2024-03-07"), "BBB", -8, 100.0, -800.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 11
        strategy.current_trade_map["BBB"] = 12

        data_df = pd.DataFrame(
            {
                ("AAA", "High"): [100.0, 101.0, 102.0, 103.0, 104.0],
                ("BBB", "Low"): [92.0, 91.5, 91.0, 90.5, 90.0],
            },
            index=pd.bdate_range("2024-03-04", periods=5),
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 105.0,
                ("BBB", "Close"): 89.0,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        order_map = {order.asset: order for order in order_list}
        self.assertEqual(order_map["AAA"].amount, 0.0)
        self.assertEqual(order_map["BBB"].amount, 0.0)
        self.assertTrue(order_map["AAA"].target)
        self.assertTrue(order_map["BBB"].target)
        self.assertEqual(order_map["AAA"].trade_id, 11)
        self.assertEqual(order_map["BBB"].trade_id, 12)

    def test_iterate_respects_independent_long_and_short_slot_limits(self):
        strategy = self.make_strategy(long_max_positions_int=1, short_max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.trade_id_int = 20
        strategy.universe_df = pd.DataFrame(
            {"LONG_HELD": [1], "SHORT_HELD": [1], "LONG_NEW": [1], "SHORT_A": [1], "SHORT_B": [1]},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(10, pd.Timestamp("2024-03-07"), "LONG_HELD", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(11, pd.Timestamp("2024-03-07"), "SHORT_HELD", -10, 100.0, -1_000.0, 2, 0.0)
        strategy.current_trade_map["LONG_HELD"] = 10
        strategy.current_trade_map["SHORT_HELD"] = 11

        close_row_ser = self.make_close_row_ser(
            {
                ("LONG_HELD", "Close"): 100.0,
                ("LONG_HELD", "Turnover"): 10_000_000.0,
                ("LONG_HELD", "qpi_value_ser"): 30.0,
                ("LONG_HELD", "sma_200_price_ser"): 100.0,
                ("LONG_HELD", "three_day_return_ser"): 0.00,
                ("SHORT_HELD", "Close"): 100.0,
                ("SHORT_HELD", "Turnover"): 10_000_000.0,
                ("SHORT_HELD", "qpi_value_ser"): 30.0,
                ("SHORT_HELD", "sma_200_price_ser"): 100.0,
                ("SHORT_HELD", "three_day_return_ser"): 0.00,
                ("LONG_NEW", "Close"): 101.0,
                ("LONG_NEW", "Turnover"): 60_000_000.0,
                ("LONG_NEW", "qpi_value_ser"): 5.0,
                ("LONG_NEW", "sma_200_price_ser"): 100.0,
                ("LONG_NEW", "three_day_return_ser"): -0.05,
                ("SHORT_A", "Close"): 95.0,
                ("SHORT_A", "Turnover"): 50_000_000.0,
                ("SHORT_A", "qpi_value_ser"): 6.0,
                ("SHORT_A", "sma_200_price_ser"): 100.0,
                ("SHORT_A", "three_day_return_ser"): 0.04,
                ("SHORT_B", "Close"): 94.0,
                ("SHORT_B", "Turnover"): 45_000_000.0,
                ("SHORT_B", "qpi_value_ser"): 7.0,
                ("SHORT_B", "sma_200_price_ser"): 100.0,
                ("SHORT_B", "three_day_return_ser"): 0.05,
            }
        )

        data_df = pd.DataFrame(
            {
                ("LONG_HELD", "High"): [101.0, 101.0, 101.0, 101.0, 101.0],
                ("SHORT_HELD", "Low"): [99.0, 99.0, 99.0, 99.0, 99.0],
            },
            index=pd.bdate_range("2024-03-04", periods=5),
        )
        data_df.columns = pd.MultiIndex.from_tuples(data_df.columns)

        strategy.iterate(
            data_df,
            close_row_ser,
            pd.Series(dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        short_entry_order = order_list[0]
        self.assertEqual(short_entry_order.asset, "SHORT_A")
        self.assertAlmostEqual(short_entry_order.amount, -50_000.0)
        self.assertEqual(short_entry_order.trade_id, 21)


if __name__ == "__main__":
    unittest.main()
