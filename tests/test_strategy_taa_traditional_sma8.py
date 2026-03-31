import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_taa_traditional_sma8 import (
    AssetSignalSpec,
    TraditionalSma8TrendStrategy,
    compute_month_end_trend_weight_df,
    load_signal_close_df,
    map_month_end_weights_to_rebalance_open_df,
)


class TraditionalSma8TrendStrategyTests(unittest.TestCase):
    def make_strategy(
        self,
        rebalance_weight_df: pd.DataFrame,
        trade_symbol_list: list[str],
        benchmarks: list[str] | None = None,
    ) -> TraditionalSma8TrendStrategy:
        return TraditionalSma8TrendStrategy(
            name="TraditionalSma8Test",
            benchmarks=[] if benchmarks is None else benchmarks,
            trade_symbol_list=trade_symbol_list,
            rebalance_weight_df=rebalance_weight_df,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_signal_close_df(self) -> pd.DataFrame:
        month_end_index = pd.date_range("2023-01-31", periods=10, freq="ME")
        signal_close_df = pd.DataFrame(
            {
                "VTI": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0],
                "TLT": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0],
            },
            index=month_end_index,
            dtype=float,
        )
        return signal_close_df

    def make_pricing_data_df(self, num_days_int: int = 320) -> pd.DataFrame:
        trading_index = pd.date_range("2023-01-02", periods=num_days_int, freq="B")
        bar_idx_vec = np.arange(num_days_int, dtype=float)

        close_map = {
            "VTI": 100.0 * np.cumprod(1.0 + 0.0007 + 0.0010 * np.sin(bar_idx_vec / 17.0)),
            "GLD": 120.0 * np.cumprod(1.0 - 0.0002 + 0.0008 * np.sin(bar_idx_vec / 23.0 + 0.3)),
            "TLT": 110.0 * np.cumprod(1.0 + 0.0001 + 0.0015 * np.cos(bar_idx_vec / 19.0)),
            "$SPXTR": 4000.0 * np.cumprod(1.0 + 0.0005 + 0.0009 * np.sin(bar_idx_vec / 21.0 + 0.5)),
        }

        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}
        for symbol_str, close_vec in close_map.items():
            open_vec = close_vec * 0.999
            high_vec = np.maximum(open_vec, close_vec) * 1.001
            low_vec = np.minimum(open_vec, close_vec) * 0.999
            pricing_data_map[(symbol_str, "Open")] = open_vec
            pricing_data_map[(symbol_str, "High")] = high_vec
            pricing_data_map[(symbol_str, "Low")] = low_vec
            pricing_data_map[(symbol_str, "Close")] = close_vec

        pricing_data_df = pd.DataFrame(pricing_data_map, index=trading_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_month_end_trend_weight_df_uses_fixed_equal_sleeves(self):
        signal_close_df = self.make_signal_close_df()

        monthly_close_df, monthly_sma_df, signal_state_df, month_end_weight_df = compute_month_end_trend_weight_df(
            signal_close_df=signal_close_df,
            sma_month_lookback_int=8,
        )

        self.assertEqual(list(monthly_close_df.index), list(pd.date_range("2023-08-31", periods=3, freq="ME")))
        self.assertEqual(list(monthly_sma_df.index), list(monthly_close_df.index))
        self.assertEqual(list(signal_state_df.index), list(monthly_close_df.index))

        first_weight_ser = month_end_weight_df.iloc[0]
        self.assertAlmostEqual(float(first_weight_ser["VTI"]), 0.5, places=12)
        self.assertAlmostEqual(float(first_weight_ser["TLT"]), 0.0, places=12)
        self.assertAlmostEqual(float(first_weight_ser.sum()), 0.5, places=12)

    def test_load_signal_close_df_falls_back_to_next_available_symbol(self):
        trading_index = pd.to_datetime(["2024-01-31", "2024-02-29"])

        def mocked_price_timeseries(symbol_str, **kwargs):
            if symbol_str == "$MISSING":
                raise ValueError
            if symbol_str == "VEA":
                return pd.DataFrame({"Close": [100.0, 101.0]}, index=trading_index, dtype=float)
            raise AssertionError(f"Unexpected symbol request: {symbol_str}")

        asset_spec_tuple = (
            AssetSignalSpec(
                asset_class_str="Developed Intl.",
                trade_symbol_str="VEA",
                signal_symbol_candidate_tuple=("$MISSING", "VEA"),
            ),
        )

        with patch(
            "strategies.strategy_taa_traditional_sma8.load_available_signal_symbol_set",
            return_value={"VEA"},
        ), patch(
            "strategies.strategy_taa_traditional_sma8.norgatedata.price_timeseries",
            side_effect=mocked_price_timeseries,
        ):
            signal_close_df, selected_signal_symbol_map = load_signal_close_df(
                asset_spec_tuple=asset_spec_tuple,
                start_date_str="2024-01-01",
                end_date_str="2024-02-29",
            )

        self.assertEqual(selected_signal_symbol_map["VEA"], "VEA")
        self.assertEqual(list(signal_close_df.columns), ["VEA"])
        self.assertEqual(list(signal_close_df.index), list(trading_index))
        self.assertTrue(np.allclose(signal_close_df["VEA"].to_numpy(dtype=float), np.array([100.0, 101.0])))

    def test_map_month_end_weights_to_rebalance_open_df_uses_first_trading_day_of_next_month(self):
        month_end_weight_df = pd.DataFrame(
            {
                "VTI": [1.0 / 3.0, 0.0],
                "GLD": [0.0, 1.0 / 3.0],
                "TLT": [1.0 / 3.0, 1.0 / 3.0],
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
                "VTI": [1.0 / 3.0, 0.0],
                "GLD": [0.0, 1.0 / 3.0],
                "TLT": [1.0 / 3.0, 1.0 / 3.0],
            },
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
            dtype=float,
        )
        expected_weight_df.index.name = "rebalance_date"

        pd.testing.assert_frame_equal(rebalance_weight_df, expected_weight_df)

    def test_iterate_liquidates_inactive_sleeves_and_opens_new_positions(self):
        rebalance_ts = pd.Timestamp("2024-02-01")
        rebalance_weight_df = pd.DataFrame(
            {
                "VTI": [1.0 / 3.0],
                "VEA": [0.0],
                "GLD": [1.0 / 3.0],
            },
            index=pd.to_datetime([rebalance_ts]),
            dtype=float,
        )
        strategy = self.make_strategy(rebalance_weight_df=rebalance_weight_df, trade_symbol_list=["VTI", "VEA", "GLD"])
        strategy.current_bar = rebalance_ts
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy._total_value_history_list = [90_000.0]
        strategy.trade_id_int = 5

        strategy.add_transaction(4, strategy.previous_bar, "VTI", 300, 100.0, 30_000.0, 1, 0.0)
        strategy.add_transaction(5, strategy.previous_bar, "VEA", 150, 100.0, 15_000.0, 2, 0.0)
        strategy.current_trade_id_map["VTI"] = 4
        strategy.current_trade_id_map["VEA"] = 5

        open_price_ser = pd.Series({"VTI": 100.0, "VEA": 100.0, "GLD": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), pd.Series(dtype=float), open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)
        self.assertEqual([order.asset for order in order_list], ["VEA", "GLD"])

        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.amount, 0)
        self.assertEqual(liquidation_order.trade_id, 5)

        new_position_order = order_list[1]
        self.assertIsInstance(new_position_order, MarketOrder)
        self.assertEqual(new_position_order.unit, "percent")
        self.assertTrue(new_position_order.target)
        self.assertAlmostEqual(float(new_position_order.amount), 1.0 / 3.0, places=12)
        self.assertEqual(new_position_order.trade_id, 6)

        self.assertEqual(strategy.current_trade_id_map["GLD"], 6)
        self.assertEqual(strategy.trade_id_int, 6)

    def test_run_daily_smoke_generates_summary_and_daily_target_weights(self):
        pricing_data_df = self.make_pricing_data_df()
        signal_close_df = pricing_data_df.loc[:, [("VTI", "Close"), ("GLD", "Close"), ("TLT", "Close")]].copy()
        signal_close_df.columns = signal_close_df.columns.get_level_values(0)

        _, _, _, month_end_weight_df = compute_month_end_trend_weight_df(
            signal_close_df=signal_close_df,
            sma_month_lookback_int=8,
        )
        rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=pricing_data_df.index,
        )

        strategy = self.make_strategy(
            rebalance_weight_df=rebalance_weight_df,
            trade_symbol_list=["VTI", "GLD", "TLT"],
            benchmarks=["$SPXTR"],
        )

        calendar_index = pricing_data_df.index[pricing_data_df.index >= rebalance_weight_df.index[0]]
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
        self.assertTrue({"VTI", "GLD", "TLT", "Cash"}.issubset(strategy.daily_target_weights.columns))

        weight_sum_ser = strategy.daily_target_weights.sum(axis=1)
        self.assertTrue(np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
