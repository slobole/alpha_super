import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

import norgatedata

from alpha.engine.order import MarketOrder
from data.norgate_loader import load_raw_prices
from strategies.strategy_mo_squeeze import SqueezeMomentumBreakoutStrategy


class SqueezeMomentumBreakoutStrategyTests(unittest.TestCase):
    def make_strategy(self) -> SqueezeMomentumBreakoutStrategy:
        return SqueezeMomentumBreakoutStrategy(
            name="SqueezeMomentumBreakoutTest",
            benchmarks=[],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_pricing_data(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-01", periods=80, freq="D")
        step_vec = np.arange(len(date_index), dtype=float)

        close_vec = 100.0 + 0.25 * step_vec + np.sin(step_vec * 0.35)
        open_vec = close_vec - 0.30 + 0.15 * np.cos(step_vec * 0.20)
        high_vec = np.maximum(open_vec, close_vec) + 1.00
        low_vec = np.minimum(open_vec, close_vec) - 1.00
        turnover_vec = 1_000_000.0 + 10_000.0 * step_vec

        pricing_data = pd.DataFrame(
            {
                ("TEST", "Open"): open_vec,
                ("TEST", "High"): high_vec,
                ("TEST", "Low"): low_vec,
                ("TEST", "Close"): close_vec,
                ("TEST", "Turnover"): turnover_vec,
            },
            index=date_index,
        )
        pricing_data.columns = pd.MultiIndex.from_tuples(pricing_data.columns)
        return pricing_data

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_signals_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data = self.make_pricing_data()

        signal_data = strategy.compute_signals(pricing_data)

        self.assertIn(("TEST", "ibs"), signal_data.columns)
        self.assertIn(("TEST", "long_breakout"), signal_data.columns)
        self.assertIn(("TEST", "squeeze_momentum"), signal_data.columns)

        strategy.audit_signals(pricing_data, signal_data)

    @patch("strategies.strategy_mo_squeeze.talib.LINEARREG")
    def test_compute_signals_uses_negative_to_positive_histogram_cross_and_green_candle_for_entry(self, mock_linearreg):
        strategy = self.make_strategy()
        pricing_data = self.make_pricing_data()

        pricing_data.loc[pricing_data.index[-3], ("TEST", "Open")] = 140.0
        pricing_data.loc[pricing_data.index[-3], ("TEST", "Close")] = 139.0
        pricing_data.loc[pricing_data.index[-1], ("TEST", "Open")] = 133.0
        pricing_data.loc[pricing_data.index[-1], ("TEST", "Close")] = 134.0

        histogram_vec = np.full(len(pricing_data), -1.0, dtype=float)
        histogram_vec[-3] = 0.60
        histogram_vec[-2] = -0.20
        histogram_vec[-1] = 0.90
        mock_linearreg.return_value = histogram_vec

        signal_data = strategy.compute_signals(pricing_data)

        self.assertFalse(signal_data.loc[pricing_data.index[-3], ("TEST", "long_breakout")])
        self.assertFalse(signal_data.loc[pricing_data.index[-2], ("TEST", "long_breakout")])
        self.assertTrue(signal_data.loc[pricing_data.index[-1], ("TEST", "long_breakout")])

    def test_get_opportunities_filters_to_entry_names_and_ranks_by_turnover(self):
        strategy = self.make_strategy()
        previous_bar = pd.Timestamp("2024-03-08")
        strategy.previous_bar = previous_bar
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "$SPX": [0]},
            index=[previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Turnover"): 200.0,
                ("AAA", "long_breakout"): True,
                ("BBB", "Turnover"): 400.0,
                ("BBB", "long_breakout"): True,
                ("CCC", "Turnover"): 900.0,
                ("CCC", "long_breakout"): False,
                ("$SPX", "Turnover"): 1_000.0,
                ("$SPX", "long_breakout"): True,
            }
        )

        opportunity_list = strategy.get_opportunities(close_row_ser)

        self.assertEqual(opportunity_list, ["BBB", "AAA"])

    def test_iterate_submits_exit_order_when_prior_bar_ibs_is_below_threshold(self):
        strategy = self.make_strategy()
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "ibs"): 0.20,
                ("AAA", "Turnover"): 300.0,
                ("AAA", "long_breakout"): False,
            }
        )
        open_price_ser = pd.Series({"AAA": 100.0})

        strategy.iterate(pd.DataFrame(), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.asset, "AAA")
        self.assertEqual(liquidation_order.amount, 0)
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.trade_id, 7)

    def test_iterate_does_not_reenter_already_held_name_when_scanning_candidates(self):
        strategy = self.make_strategy()
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1]}, index=[strategy.previous_bar])

        strategy.add_transaction(3, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 3

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "ibs"): 0.60,
                ("AAA", "Turnover"): 800.0,
                ("AAA", "long_breakout"): True,
                ("BBB", "ibs"): 0.60,
                ("BBB", "Turnover"): 500.0,
                ("BBB", "long_breakout"): True,
            }
        )
        open_price_ser = pd.Series({"AAA": 100.0, "BBB": 80.0})

        strategy.iterate(pd.DataFrame(), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "BBB")
        self.assertEqual(entry_order.unit, "value")
        self.assertFalse(entry_order.target)


class LoadRawPricesAdjustmentTests(unittest.TestCase):
    @patch("data.norgate_loader.norgatedata.price_timeseries")
    def test_load_raw_prices_uses_capitalspecial_for_tradeable_symbols_and_totalreturn_for_benchmark(
        self,
        mock_price_timeseries,
    ):
        adjustment_map: dict[str, object] = {}

        def price_timeseries_side_effect(
            symbol,
            stock_price_adjustment_setting,
            padding_setting,
            start_date,
            end_date,
            timeseriesformat,
        ):
            adjustment_map[symbol] = stock_price_adjustment_setting
            return pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.5],
                    "Turnover": [1_000_000.0],
                },
                index=pd.to_datetime(["2024-01-02"]),
            )

        mock_price_timeseries.side_effect = price_timeseries_side_effect

        pricing_data = load_raw_prices(["AAA", "BBB"], ["$SPX"], start_date="2024-01-01", end_date="2024-01-31")

        self.assertEqual(
            adjustment_map["AAA"],
            norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        )
        self.assertEqual(
            adjustment_map["BBB"],
            norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        )
        self.assertEqual(
            adjustment_map["$SPX"],
            norgatedata.StockPriceAdjustmentType.TOTALRETURN,
        )
        self.assertIn(("AAA", "Close"), pricing_data.columns)
        self.assertIn(("BBB", "Close"), pricing_data.columns)
        self.assertIn(("$SPX", "Close"), pricing_data.columns)


if __name__ == "__main__":
    unittest.main()
