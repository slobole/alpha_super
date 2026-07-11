import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.momentum.strategy_mo_smooth_trend_russell3000_long_only_sma200 import (
    DEFAULT_CONFIG,
    REGIME_BENCHMARK_SYMBOL_STR,
    REGIME_SMA_FIELD_STR,
    SmoothTrendRussell3000LongOnlySMA200Strategy,
)


class SmoothTrendRussell3000LongOnlySMA200Tests(unittest.TestCase):
    def make_config(self):
        return replace(
            DEFAULT_CONFIG,
            lookback_trading_day_int=20,
            skip_trading_day_int=2,
            max_long_positions_int=2,
            capital_base_float=10_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-29")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self) -> SmoothTrendRussell3000LongOnlySMA200Strategy:
        return SmoothTrendRussell3000LongOnlySMA200Strategy(
            name="SmoothTrendRussell3000LongOnlySMA200Test",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=self.make_config(),
        )

    def make_close_row_ser(
        self,
        benchmark_close_float: float,
        benchmark_sma_float: float,
    ) -> pd.Series:
        row_map: dict[tuple[str, str], float] = {}
        for symbol_int in range(1, 51):
            symbol_str = f"S{symbol_int:03d}"
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)
        row_map[(REGIME_BENCHMARK_SYMBOL_STR, "Close")] = benchmark_close_float
        row_map[(REGIME_BENCHMARK_SYMBOL_STR, REGIME_SMA_FIELD_STR)] = benchmark_sma_float
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def attach_universe(self, strategy_obj: SmoothTrendRussell3000LongOnlySMA200Strategy) -> None:
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )

    def test_default_config_is_long_only_sma200_variant(self):
        self.assertEqual(
            DEFAULT_CONFIG.variant_key_str,
            "russell3000_option_a_n20_long_only_close_gt_1_rua_above_sma200",
        )
        self.assertEqual(DEFAULT_CONFIG.max_long_positions_int, 20)
        self.assertEqual(DEFAULT_CONFIG.max_short_positions_int, 0)
        self.assertAlmostEqual(DEFAULT_CONFIG.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.short_gross_exposure_float, 0.0)

    def test_compute_signals_adds_causal_benchmark_sma200(self):
        strategy_obj = self.make_strategy()
        date_index = pd.date_range("2023-01-02", periods=220, freq="B")
        benchmark_close_vec = np.arange(1.0, 221.0)
        pricing_data_df = pd.DataFrame(
            {
                (REGIME_BENCHMARK_SYMBOL_STR, "Open"): benchmark_close_vec,
                (REGIME_BENCHMARK_SYMBOL_STR, "High"): benchmark_close_vec,
                (REGIME_BENCHMARK_SYMBOL_STR, "Low"): benchmark_close_vec,
                (REGIME_BENCHMARK_SYMBOL_STR, "Close"): benchmark_close_vec,
                ("S001", "Open"): np.full(len(date_index), 10.0),
                ("S001", "High"): np.full(len(date_index), 10.1),
                ("S001", "Low"): np.full(len(date_index), 9.9),
                ("S001", "Close"): np.full(len(date_index), 10.0),
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        signal_data_df = strategy_obj.compute_signals(pricing_data=pricing_data_df)

        self.assertIn(
            (REGIME_BENCHMARK_SYMBOL_STR, REGIME_SMA_FIELD_STR),
            signal_data_df.columns,
        )
        self.assertTrue(
            pd.isna(signal_data_df[(REGIME_BENCHMARK_SYMBOL_STR, REGIME_SMA_FIELD_STR)].iloc[198])
        )
        self.assertAlmostEqual(
            float(signal_data_df[(REGIME_BENCHMARK_SYMBOL_STR, REGIME_SMA_FIELD_STR)].iloc[199]),
            100.5,
        )

    def test_get_selection_df_passes_long_targets_when_benchmark_above_sma200(self):
        strategy_obj = self.make_strategy()
        self.attach_universe(strategy_obj=strategy_obj)

        selection_df = strategy_obj.get_selection_df(
            close_row_ser=self.make_close_row_ser(
                benchmark_close_float=110.0,
                benchmark_sma_float=100.0,
            )
        )

        self.assertEqual(selection_df["side_str"].tolist(), ["long", "long"])
        self.assertEqual(selection_df["symbol_str"].tolist(), ["S050", "S049"])
        self.assertEqual(selection_df["regime_bull_market_bool"].tolist(), [True, True])
        self.assertAlmostEqual(float(selection_df["target_weight_float"].sum()), 1.0)

    def test_get_selection_df_returns_cash_when_benchmark_below_sma200(self):
        strategy_obj = self.make_strategy()
        self.attach_universe(strategy_obj=strategy_obj)

        selection_df = strategy_obj.get_selection_df(
            close_row_ser=self.make_close_row_ser(
                benchmark_close_float=99.0,
                benchmark_sma_float=100.0,
            )
        )

        self.assertEqual(len(selection_df), 0)

    def test_iterate_liquidates_old_positions_when_benchmark_below_sma200(self):
        strategy_obj = self.make_strategy()
        self.attach_universe(strategy_obj=strategy_obj)
        strategy_obj.current_bar = pd.Timestamp("2024-04-01")
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "OLDL", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["OLDL"] = 7

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            self.make_close_row_ser(
                benchmark_close_float=99.0,
                benchmark_sma_float=100.0,
            ),
            pd.Series({"OLDL": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertEqual(len(order_list), 1)
        self.assertEqual(order_list[0].asset, "OLDL")
        self.assertEqual(order_list[0].unit, "shares")
        self.assertTrue(order_list[0].target)


if __name__ == "__main__":
    unittest.main()
