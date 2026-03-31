import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import talib

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_mr_qpi_ibs_rsi_exit_price_adv import QPIIbsRsiExitPriceAdvStrategy


class QPIIbsRsiExitPriceAdvStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> QPIIbsRsiExitPriceAdvStrategy:
        base_kwargs = dict(
            name="QPIIbsRsiExitPriceAdvTest",
            benchmarks=[],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            sma_window_int=20,
            qpi_lookback_years_int=1,
            adv_window_int=20,
        )
        base_kwargs.update(kwargs)
        return QPIIbsRsiExitPriceAdvStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=280, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 60.0 + 0.10 * step_vec + 1.4 * np.sin(step_vec * 0.06)
        bbb_close_vec = 50.0 + 0.09 * step_vec + 1.1 * np.cos(step_vec * 0.05)
        ccc_close_vec = 45.0 + 0.07 * step_vec + 1.3 * np.sin(step_vec * 0.08 + 0.3)

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

    def test_compute_signals_adds_expected_features_including_adv20_and_rsi2(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "three_day_return_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "qpi_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "sma_200_price_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "adv20_dollar_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "ibs_value_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "rsi2_value_ser"), signal_data_df.columns)

        last_bar_ts = pricing_data_df.index[-1]
        expected_adv20_float = float(
            pricing_data_df[("AAA", "Turnover")].rolling(window=20, min_periods=20).mean().loc[last_bar_ts]
        )
        actual_adv20_float = float(signal_data_df.loc[last_bar_ts, ("AAA", "adv20_dollar_ser")])
        self.assertAlmostEqual(actual_adv20_float, expected_adv20_float)

        expected_rsi2_float = float(
            talib.RSI(
                pricing_data_df[("AAA", "Close")].to_numpy(dtype=float),
                timeperiod=2,
            )[-1]
        )
        actual_rsi2_float = float(signal_data_df.loc[last_bar_ts, ("AAA", "rsi2_value_ser")])
        self.assertAlmostEqual(actual_rsi2_float, expected_rsi2_float)

    def test_get_opportunity_list_enforces_price_adv_and_ibs_filters_and_ranking(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "LOWP": [1],
                "ILLIQ": [1],
                "HIGHQ": [1],
                "BELOWMA": [1],
                "UP3D": [1],
                "HIGHIBS": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 12.0,
                ("AAA", "Turnover"): 35_000_000.0,
                ("AAA", "qpi_value_ser"): 10.0,
                ("AAA", "sma_200_price_ser"): 11.0,
                ("AAA", "three_day_return_ser"): -0.04,
                ("AAA", "adv20_dollar_ser"): 25_000_000.0,
                ("AAA", "ibs_value_ser"): 0.05,
                ("BBB", "Close"): 13.0,
                ("BBB", "Turnover"): 50_000_000.0,
                ("BBB", "qpi_value_ser"): 8.0,
                ("BBB", "sma_200_price_ser"): 11.0,
                ("BBB", "three_day_return_ser"): -0.03,
                ("BBB", "adv20_dollar_ser"): 30_000_000.0,
                ("BBB", "ibs_value_ser"): 0.08,
                ("CCC", "Close"): 14.0,
                ("CCC", "Turnover"): 45_000_000.0,
                ("CCC", "qpi_value_ser"): 7.0,
                ("CCC", "sma_200_price_ser"): 12.0,
                ("CCC", "three_day_return_ser"): -0.02,
                ("CCC", "adv20_dollar_ser"): 22_000_000.0,
                ("CCC", "ibs_value_ser"): 0.09,
                ("LOWP", "Close"): 9.0,
                ("LOWP", "Turnover"): 80_000_000.0,
                ("LOWP", "qpi_value_ser"): 5.0,
                ("LOWP", "sma_200_price_ser"): 8.0,
                ("LOWP", "three_day_return_ser"): -0.05,
                ("LOWP", "adv20_dollar_ser"): 50_000_000.0,
                ("LOWP", "ibs_value_ser"): 0.04,
                ("ILLIQ", "Close"): 15.0,
                ("ILLIQ", "Turnover"): 55_000_000.0,
                ("ILLIQ", "qpi_value_ser"): 6.0,
                ("ILLIQ", "sma_200_price_ser"): 12.0,
                ("ILLIQ", "three_day_return_ser"): -0.04,
                ("ILLIQ", "adv20_dollar_ser"): 10_000_000.0,
                ("ILLIQ", "ibs_value_ser"): 0.04,
                ("HIGHQ", "Close"): 16.0,
                ("HIGHQ", "Turnover"): 60_000_000.0,
                ("HIGHQ", "qpi_value_ser"): 40.0,
                ("HIGHQ", "sma_200_price_ser"): 12.0,
                ("HIGHQ", "three_day_return_ser"): -0.05,
                ("HIGHQ", "adv20_dollar_ser"): 35_000_000.0,
                ("HIGHQ", "ibs_value_ser"): 0.05,
                ("BELOWMA", "Close"): 10.5,
                ("BELOWMA", "Turnover"): 65_000_000.0,
                ("BELOWMA", "qpi_value_ser"): 9.0,
                ("BELOWMA", "sma_200_price_ser"): 12.0,
                ("BELOWMA", "three_day_return_ser"): -0.06,
                ("BELOWMA", "adv20_dollar_ser"): 30_000_000.0,
                ("BELOWMA", "ibs_value_ser"): 0.04,
                ("UP3D", "Close"): 17.0,
                ("UP3D", "Turnover"): 70_000_000.0,
                ("UP3D", "qpi_value_ser"): 6.0,
                ("UP3D", "sma_200_price_ser"): 12.0,
                ("UP3D", "three_day_return_ser"): 0.04,
                ("UP3D", "adv20_dollar_ser"): 40_000_000.0,
                ("UP3D", "ibs_value_ser"): 0.05,
                ("HIGHIBS", "Close"): 18.0,
                ("HIGHIBS", "Turnover"): 75_000_000.0,
                ("HIGHIBS", "qpi_value_ser"): 5.0,
                ("HIGHIBS", "sma_200_price_ser"): 12.0,
                ("HIGHIBS", "three_day_return_ser"): -0.05,
                ("HIGHIBS", "adv20_dollar_ser"): 45_000_000.0,
                ("HIGHIBS", "ibs_value_ser"): 0.15,
                ("OUT", "Close"): 19.0,
                ("OUT", "Turnover"): 90_000_000.0,
                ("OUT", "qpi_value_ser"): 4.0,
                ("OUT", "sma_200_price_ser"): 12.0,
                ("OUT", "three_day_return_ser"): -0.05,
                ("OUT", "adv20_dollar_ser"): 50_000_000.0,
                ("OUT", "ibs_value_ser"): 0.05,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, ["BBB", "CCC", "AAA"])

    def test_iterate_submits_exit_order_when_ibs_crosses_threshold(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(7, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 12.0,
                ("AAA", "Turnover"): 30_000_000.0,
                ("AAA", "qpi_value_ser"): 50.0,
                ("AAA", "sma_200_price_ser"): 11.0,
                ("AAA", "three_day_return_ser"): 0.03,
                ("AAA", "adv20_dollar_ser"): 25_000_000.0,
                ("AAA", "ibs_value_ser"): 0.95,
                ("AAA", "rsi2_value_ser"): 50.0,
            }
        )

        strategy.iterate(pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        self.assertIsInstance(order_list[0], MarketOrder)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[0].amount, 0.0)
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 7)

    def test_iterate_submits_exit_order_when_rsi2_crosses_threshold(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(8, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 12.0,
                ("AAA", "Turnover"): 30_000_000.0,
                ("AAA", "qpi_value_ser"): 50.0,
                ("AAA", "sma_200_price_ser"): 11.0,
                ("AAA", "three_day_return_ser"): 0.03,
                ("AAA", "adv20_dollar_ser"): 25_000_000.0,
                ("AAA", "ibs_value_ser"): 0.50,
                ("AAA", "rsi2_value_ser"): 95.0,
            }
        )

        strategy.iterate(pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[0].trade_id, 8)
        self.assertEqual(order_list[0].amount, 0.0)
        self.assertTrue(order_list[0].target)

    def test_iterate_enters_highest_ranked_candidate_that_passes_price_adv_filters(self):
        strategy = self.make_strategy(max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.trade_id_int = 10
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1], "CCC": [1], "LOWP": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(10, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 10

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 12.0,
                ("AAA", "Turnover"): 35_000_000.0,
                ("AAA", "qpi_value_ser"): 40.0,
                ("AAA", "sma_200_price_ser"): 11.0,
                ("AAA", "three_day_return_ser"): 0.02,
                ("AAA", "adv20_dollar_ser"): 25_000_000.0,
                ("AAA", "ibs_value_ser"): 0.40,
                ("AAA", "rsi2_value_ser"): 60.0,
                ("BBB", "Close"): 13.0,
                ("BBB", "Turnover"): 50_000_000.0,
                ("BBB", "qpi_value_ser"): 8.0,
                ("BBB", "sma_200_price_ser"): 11.0,
                ("BBB", "three_day_return_ser"): -0.04,
                ("BBB", "adv20_dollar_ser"): 30_000_000.0,
                ("BBB", "ibs_value_ser"): 0.07,
                ("BBB", "rsi2_value_ser"): 30.0,
                ("CCC", "Close"): 14.0,
                ("CCC", "Turnover"): 40_000_000.0,
                ("CCC", "qpi_value_ser"): 9.0,
                ("CCC", "sma_200_price_ser"): 11.0,
                ("CCC", "three_day_return_ser"): -0.03,
                ("CCC", "adv20_dollar_ser"): 22_000_000.0,
                ("CCC", "ibs_value_ser"): 0.08,
                ("CCC", "rsi2_value_ser"): 40.0,
                ("LOWP", "Close"): 9.0,
                ("LOWP", "Turnover"): 80_000_000.0,
                ("LOWP", "qpi_value_ser"): 5.0,
                ("LOWP", "sma_200_price_ser"): 8.0,
                ("LOWP", "three_day_return_ser"): -0.05,
                ("LOWP", "adv20_dollar_ser"): 50_000_000.0,
                ("LOWP", "ibs_value_ser"): 0.04,
                ("LOWP", "rsi2_value_ser"): 25.0,
            }
        )

        strategy.iterate(pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "BBB")
        self.assertEqual(entry_order.unit, "value")
        self.assertFalse(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 50_000.0)
        self.assertEqual(entry_order.trade_id, 11)


if __name__ == "__main__":
    unittest.main()
