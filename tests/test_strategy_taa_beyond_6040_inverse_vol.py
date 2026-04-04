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
from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import get_first_actionable_rebalance_ts
from strategies.taa_beyond_6040.strategy_taa_beyond_6040_inverse_vol import (
    Beyond6040InverseVolStrategy,
)


class Beyond6040InverseVolStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> Beyond6040InverseVolStrategy:
        base_kwargs = dict(
            name="Beyond6040InverseVolTest",
            benchmarks=[],
            asset_list=["VTI", "GLD", "TLT"],
            asset_vol_lookback_int=63,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return Beyond6040InverseVolStrategy(**base_kwargs)

    def make_pricing_data_df(self, num_days_int: int = 180) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=num_days_int, freq="B")
        bar_idx_vec = np.arange(num_days_int, dtype=float)

        vti_return_vec = 0.0005 + 0.0080 * np.sin(bar_idx_vec / 4.0)
        gld_return_vec = 0.0002 + 0.0040 * np.sin(bar_idx_vec / 6.0 + 0.5)
        tlt_return_vec = 0.0001 + 0.0030 * np.cos(bar_idx_vec / 7.0)
        benchmark_return_vec = 0.0004 + 0.0070 * np.sin(bar_idx_vec / 5.0 + 0.25)

        close_map = {
            "VTI": 100.0 * np.cumprod(1.0 + vti_return_vec),
            "GLD": 120.0 * np.cumprod(1.0 + gld_return_vec),
            "TLT": 110.0 * np.cumprod(1.0 + tlt_return_vec),
            "$SPX": 4000.0 * np.cumprod(1.0 + benchmark_return_vec),
        }

        pricing_data_dict: dict[tuple[str, str], np.ndarray] = {}
        for symbol_str, close_vec in close_map.items():
            open_vec = close_vec * 0.999
            high_vec = np.maximum(open_vec, close_vec) * 1.001
            low_vec = np.minimum(open_vec, close_vec) * 0.999
            pricing_data_dict[(symbol_str, "Open")] = open_vec
            pricing_data_dict[(symbol_str, "High")] = high_vec
            pricing_data_dict[(symbol_str, "Low")] = low_vec
            pricing_data_dict[(symbol_str, "Close")] = close_vec

        pricing_data_df = pd.DataFrame(pricing_data_dict, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, base_weight_tuple: tuple[float, float, float]) -> pd.Series:
        close_row_ser = pd.Series(
            {
                ("VTI", "base_weight_ser"): base_weight_tuple[0],
                ("GLD", "base_weight_ser"): base_weight_tuple[1],
                ("TLT", "base_weight_ser"): base_weight_tuple[2],
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_iterate_submits_no_orders_inside_month(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-04-12")
        strategy.current_bar = pd.Timestamp("2024-04-15")
        strategy._total_value_history_list = [100_000.0]

        close_row_ser = self.make_close_row_ser((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
        open_price_ser = pd.Series({"VTI": 100.0, "GLD": 100.0, "TLT": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)
        self.assertAlmostEqual(float(strategy.daily_target_weights.loc[strategy.current_bar, "Cash"]), 0.0)

    def test_iterate_rebalances_on_first_trading_day_of_month(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-04-30")
        strategy.current_bar = pd.Timestamp("2024-05-01")
        strategy._total_value_history_list = [100_000.0]

        close_row_ser = self.make_close_row_ser((0.30, 0.30, 0.40))
        open_price_ser = pd.Series({"VTI": 100.0, "GLD": 100.0, "TLT": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual([order.asset for order in order_list], ["VTI", "GLD", "TLT"])
        for order_obj, expected_weight_float, expected_trade_id_int in zip(
            order_list,
            [0.30, 0.30, 0.40],
            [1, 2, 3],
        ):
            self.assertIsInstance(order_obj, MarketOrder)
            self.assertEqual(order_obj.unit, "percent")
            self.assertTrue(order_obj.target)
            self.assertAlmostEqual(float(order_obj.amount), expected_weight_float, places=12)
            self.assertEqual(order_obj.trade_id, expected_trade_id_int)

        self.assertAlmostEqual(float(strategy.daily_target_weights.loc[strategy.current_bar, "Cash"]), 0.0)

    def test_run_daily_smoke_generates_summary_and_zero_cash_weights(self):
        strategy = self.make_strategy(benchmarks=["$SPX"])
        pricing_data_df = self.make_pricing_data_df(num_days_int=220)
        relevant_start_ts = get_first_actionable_rebalance_ts(
            pricing_data_df=pricing_data_df,
            asset_list=strategy.asset_list,
            asset_vol_lookback_int=strategy.asset_vol_lookback_int,
        )
        calendar_index = pricing_data_df.index[pricing_data_df.index >= relevant_start_ts]

        run_daily(
            strategy,
            pricing_data_df,
            calendar=calendar_index,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
        )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.daily_target_weights), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)
        self.assertTrue({"VTI", "GLD", "TLT", "Cash"}.issubset(strategy.daily_target_weights.columns))
        weight_sum_ser = strategy.daily_target_weights.sum(axis=1)
        self.assertTrue(np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12))
        self.assertTrue(np.allclose(strategy.daily_target_weights["Cash"].to_numpy(dtype=float), 0.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
