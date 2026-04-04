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
from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import (
    Beyond6040Strategy,
    compute_gross_exposure_float,
    compute_month_end_inverse_vol_weight_df,
    map_month_end_weights_to_rebalance_open_df,
)


class Beyond6040StrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> Beyond6040Strategy:
        base_kwargs = dict(
            name="Beyond6040Test",
            benchmarks=[],
            asset_list=["VTI", "GLD", "TLT"],
            asset_vol_lookback_int=63,
            portfolio_vol_lookback_int=63,
            target_portfolio_vol_float=0.08,
            trigger_portfolio_vol_float=0.085,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return Beyond6040Strategy(**base_kwargs)

    def make_price_close_df(self, num_days_int: int = 90) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=num_days_int, freq="B")
        bar_idx_vec = np.arange(num_days_int, dtype=float)

        vti_return_vec = 0.0006 + 0.0200 * np.where((bar_idx_vec % 2) == 0, 1.0, -1.0)
        gld_return_vec = 0.0002 + 0.0050 * np.where((bar_idx_vec % 2) == 0, 1.0, -1.0)
        tlt_return_vec = 0.0003 + 0.0100 * np.where((bar_idx_vec % 2) == 0, 1.0, -1.0)

        price_close_df = pd.DataFrame(
            {
                "VTI": 100.0 * np.cumprod(1.0 + vti_return_vec),
                "GLD": 100.0 * np.cumprod(1.0 + gld_return_vec),
                "TLT": 100.0 * np.cumprod(1.0 + tlt_return_vec),
            },
            index=date_index,
            dtype=float,
        )
        return price_close_df

    def make_pricing_data_df(self, num_days_int: int = 140) -> pd.DataFrame:
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

    def test_compute_month_end_inverse_vol_weight_df_prefers_lower_vol_asset_and_sums_to_one(self):
        price_close_df = self.make_price_close_df()

        _, _, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
            price_close_df=price_close_df,
            asset_vol_lookback_int=63,
        )

        self.assertGreater(len(month_end_weight_df), 0)
        last_weight_ser = month_end_weight_df.iloc[-1]
        self.assertAlmostEqual(float(last_weight_ser.sum()), 1.0, places=12)
        self.assertGreater(float(last_weight_ser["GLD"]), float(last_weight_ser["TLT"]))
        self.assertGreater(float(last_weight_ser["TLT"]), float(last_weight_ser["VTI"]))

    def test_map_month_end_weights_to_rebalance_open_df_uses_first_trading_day_of_next_month(self):
        month_end_weight_df = pd.DataFrame(
            {
                "VTI": [0.30, 0.35],
                "GLD": [0.30, 0.25],
                "TLT": [0.40, 0.40],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
            dtype=float,
        )
        execution_index = pd.to_datetime(
            ["2024-02-01", "2024-02-02", "2024-03-01", "2024-03-04", "2024-03-05"]
        )

        rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=execution_index,
        )

        expected_weight_df = pd.DataFrame(
            {
                "VTI": [0.30, 0.35],
                "GLD": [0.30, 0.25],
                "TLT": [0.40, 0.40],
            },
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
            dtype=float,
        )
        expected_weight_df.index.name = "rebalance_date"

        pd.testing.assert_frame_equal(rebalance_weight_df, expected_weight_df)

    def test_compute_gross_exposure_float_returns_one_with_insufficient_history(self):
        realized_return_ser = pd.Series([0.01, -0.01] * 20, dtype=float)

        gross_exposure_float = compute_gross_exposure_float(
            realized_return_ser=realized_return_ser,
            portfolio_vol_lookback_int=63,
        )

        self.assertEqual(gross_exposure_float, 1.0)

    def test_compute_gross_exposure_float_returns_one_below_trigger(self):
        realized_return_ser = pd.Series([0.002, -0.002] * 32, dtype=float)

        gross_exposure_float = compute_gross_exposure_float(
            realized_return_ser=realized_return_ser,
            portfolio_vol_lookback_int=63,
            target_portfolio_vol_float=0.08,
            trigger_portfolio_vol_float=0.085,
        )

        self.assertEqual(gross_exposure_float, 1.0)

    def test_compute_gross_exposure_float_scales_down_above_trigger(self):
        realized_return_ser = pd.Series([0.02, -0.02] * 32, dtype=float)
        trailing_return_ser = realized_return_ser.iloc[-63:]
        expected_portfolio_vol_float = float(trailing_return_ser.std(ddof=1) * np.sqrt(252.0))
        expected_gross_exposure_float = 0.08 / expected_portfolio_vol_float

        gross_exposure_float = compute_gross_exposure_float(
            realized_return_ser=realized_return_ser,
            portfolio_vol_lookback_int=63,
            target_portfolio_vol_float=0.08,
            trigger_portfolio_vol_float=0.085,
        )

        self.assertAlmostEqual(gross_exposure_float, expected_gross_exposure_float, places=12)
        self.assertLess(gross_exposure_float, 1.0)

    def test_iterate_skips_no_op_when_target_shares_match_current_position(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-04-29")
        strategy.current_bar = pd.Timestamp("2024-04-30")
        strategy._total_value_history_list = [100_000.0]
        strategy._daily_return_history_list = [0.0] + [0.002, -0.002] * 32

        strategy.add_transaction(4, strategy.previous_bar, "VTI", 300, 100.0, 30_000.0, 1, 0.0)
        strategy.add_transaction(5, strategy.previous_bar, "GLD", 300, 100.0, 30_000.0, 2, 0.0)
        strategy.add_transaction(6, strategy.previous_bar, "TLT", 400, 100.0, 40_000.0, 3, 0.0)
        strategy.current_trade_id_map["VTI"] = 4
        strategy.current_trade_id_map["GLD"] = 5
        strategy.current_trade_id_map["TLT"] = 6

        close_row_ser = self.make_close_row_ser((0.30, 0.30, 0.40))
        open_price_ser = pd.Series({"VTI": 100.0, "GLD": 100.0, "TLT": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)
        self.assertAlmostEqual(float(strategy.daily_target_weights.loc[strategy.current_bar, "Cash"]), 0.0)

    def test_iterate_scales_existing_positions_down_when_overlay_reduces_exposure(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-04-29")
        strategy.current_bar = pd.Timestamp("2024-04-30")
        strategy._total_value_history_list = [100_000.0]
        strategy._daily_return_history_list = [0.0] + [0.02, -0.02] * 32
        strategy.trade_id_int = 10

        strategy.add_transaction(7, strategy.previous_bar, "VTI", 300, 100.0, 30_000.0, 1, 0.0)
        strategy.add_transaction(8, strategy.previous_bar, "GLD", 300, 100.0, 30_000.0, 2, 0.0)
        strategy.add_transaction(9, strategy.previous_bar, "TLT", 400, 100.0, 40_000.0, 3, 0.0)
        strategy.current_trade_id_map["VTI"] = 7
        strategy.current_trade_id_map["GLD"] = 8
        strategy.current_trade_id_map["TLT"] = 9

        close_row_ser = self.make_close_row_ser((0.30, 0.30, 0.40))
        open_price_ser = pd.Series({"VTI": 100.0, "GLD": 100.0, "TLT": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        gross_exposure_float = compute_gross_exposure_float(
            realized_return_ser=pd.Series(strategy._daily_return_history_list[1:], dtype=float),
            portfolio_vol_lookback_int=63,
            target_portfolio_vol_float=0.08,
            trigger_portfolio_vol_float=0.085,
        )
        expected_weight_vec = np.array([0.30, 0.30, 0.40], dtype=float) * gross_exposure_float

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        for order_obj, expected_weight_float, trade_id_int in zip(order_list, expected_weight_vec, [7, 8, 9]):
            self.assertIsInstance(order_obj, MarketOrder)
            self.assertEqual(order_obj.unit, "percent")
            self.assertTrue(order_obj.target)
            self.assertAlmostEqual(float(order_obj.amount), float(expected_weight_float), places=12)
            self.assertEqual(order_obj.trade_id, trade_id_int)

        self.assertGreater(float(strategy.daily_target_weights.loc[strategy.current_bar, "Cash"]), 0.0)

    def test_iterate_reuses_existing_trade_id_and_opens_new_trade_ids(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-04-29")
        strategy.current_bar = pd.Timestamp("2024-04-30")
        strategy._total_value_history_list = [100_000.0]
        strategy._daily_return_history_list = [0.0] + [0.002, -0.002] * 32
        strategy.trade_id_int = 11

        strategy.add_transaction(4, strategy.previous_bar, "VTI", 300, 100.0, 30_000.0, 1, 0.0)
        strategy.current_trade_id_map["VTI"] = 4

        close_row_ser = self.make_close_row_ser((0.50, 0.25, 0.25))
        open_price_ser = pd.Series({"VTI": 100.0, "GLD": 100.0, "TLT": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual([order.asset for order in order_list], ["VTI", "GLD", "TLT"])
        self.assertEqual(order_list[0].trade_id, 4)
        self.assertEqual(order_list[1].trade_id, 12)
        self.assertEqual(order_list[2].trade_id, 13)
        self.assertEqual(strategy.current_trade_id_map["GLD"], 12)
        self.assertEqual(strategy.current_trade_id_map["TLT"], 13)
        self.assertEqual(strategy.trade_id_int, 13)

    def test_run_daily_smoke_generates_summary_and_daily_target_weights(self):
        strategy = self.make_strategy(benchmarks=["$SPX"])
        pricing_data_df = self.make_pricing_data_df(num_days_int=180)

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
        self.assertGreater(len(strategy.daily_target_weights), 0)
        self.assertTrue({"VTI", "GLD", "TLT", "Cash"}.issubset(strategy.daily_target_weights.columns))
        weight_sum_ser = strategy.daily_target_weights.sum(axis=1)
        self.assertTrue(np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
