import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_smooth_trend_long_sp500 import (
    compute_smooth_trend_signal_tables,
)
from strategies.momentum.strategy_mo_smooth_trend_russell3000_long_short import (
    DEFAULT_CONFIG,
    SmoothTrendRussell3000LongShortStrategy,
)


class SmoothTrendRussell3000LongShortTests(unittest.TestCase):
    def make_config(self):
        return replace(
            DEFAULT_CONFIG,
            lookback_trading_day_int=20,
            skip_trading_day_int=2,
            max_long_positions_int=2,
            max_short_positions_int=2,
            capital_base_float=10_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-04-01",
        decision_date_str: str = "2024-03-29",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> SmoothTrendRussell3000LongShortStrategy:
        config_obj = kwargs.pop("config", self.make_config())
        base_kwargs = dict(
            name="SmoothTrendRussell3000LongShortTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )
        base_kwargs.update(kwargs)
        return SmoothTrendRussell3000LongShortStrategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    @staticmethod
    def make_close_vec(base_price_float: float, daily_return_vec: np.ndarray) -> np.ndarray:
        return base_price_float * np.cumprod(1.0 + daily_return_vec)

    def make_pricing_data_df(self, periods_int: int = 120, symbol_count_int: int = 50) -> pd.DataFrame:
        date_index = pd.date_range("2023-11-01", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}
        for symbol_int in range(1, symbol_count_int + 1):
            symbol_str = f"S{symbol_int:03d}"
            drift_float = (float(symbol_int) - 25.0) * 0.00002
            wiggle_vec = 0.00015 * np.sin(step_vec * (0.03 + symbol_int * 0.001))
            close_vec = self.make_close_vec(
                base_price_float=20.0 + symbol_int * 0.1,
                daily_return_vec=drift_float + wiggle_vec,
            )
            pricing_data_map[(symbol_str, "Open")] = close_vec * 0.999
            pricing_data_map[(symbol_str, "High")] = close_vec * 1.010
            pricing_data_map[(symbol_str, "Low")] = close_vec * 0.990
            pricing_data_map[(symbol_str, "Close")] = close_vec

        spy_close_vec = self.make_close_vec(
            base_price_float=400.0,
            daily_return_vec=0.0002 + 0.00005 * np.sin(step_vec * 0.02),
        )
        pricing_data_map[("SPY", "Open")] = spy_close_vec * 0.999
        pricing_data_map[("SPY", "High")] = spy_close_vec * 1.010
        pricing_data_map[("SPY", "Low")] = spy_close_vec * 0.990
        pricing_data_map[("SPY", "Close")] = spy_close_vec

        pricing_data_df = pd.DataFrame(pricing_data_map, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_default_config_is_russell3000_option_a_n20_long_short(self):
        self.assertEqual(DEFAULT_CONFIG.variant_key_str, "russell3000_option_a_n20_long_short_close_gt_1")
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(DEFAULT_CONFIG.benchmark_list, ("$RUA",))
        self.assertEqual(DEFAULT_CONFIG.lookback_trading_day_int, 252)
        self.assertEqual(DEFAULT_CONFIG.skip_trading_day_int, 21)
        self.assertEqual(DEFAULT_CONFIG.quintile_count_int, 5)
        self.assertEqual(DEFAULT_CONFIG.max_long_positions_int, 20)
        self.assertEqual(DEFAULT_CONFIG.max_short_positions_int, 20)
        self.assertAlmostEqual(DEFAULT_CONFIG.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.short_gross_exposure_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.minimum_close_price_float, 1.0)

    def test_get_selection_df_uses_r2_first_then_slope_for_both_legs(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")

        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.universe_df["OUT"] = 0

        row_map: dict[tuple[str, str], float] = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)
        row_map[("OUT", "Close")] = 10.0
        row_map[("OUT", "trend_r2_20_2_ser")] = 999.0
        row_map[("OUT", "trend_slope_20_2_ser")] = 999.0

        selection_df = strategy_obj.get_selection_df(close_row_ser=self.make_close_row_ser(row_map))

        long_df = selection_df.loc[selection_df["side_str"] == "long"]
        short_df = selection_df.loc[selection_df["side_str"] == "short"]

        self.assertEqual(long_df["symbol_str"].tolist(), ["S050", "S049"])
        self.assertEqual(short_df["symbol_str"].tolist(), ["S001", "S002"])
        self.assertEqual(selection_df["symbol_str"].tolist(), ["S050", "S049", "S001", "S002"])
        self.assertNotIn("OUT", selection_df["symbol_str"].tolist())
        self.assertEqual(long_df["target_weight_float"].tolist(), [0.5, 0.5])
        self.assertEqual(short_df["target_weight_float"].tolist(), [-0.5, -0.5])

    def test_get_target_weight_ser_is_dollar_neutral_when_both_legs_exist(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")

        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        row_map: dict[tuple[str, str], float] = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser(row_map)
        )

        self.assertAlmostEqual(float(target_weight_ser[target_weight_ser > 0].sum()), 1.0)
        self.assertAlmostEqual(float(target_weight_ser[target_weight_ser < 0].sum()), -1.0)
        self.assertAlmostEqual(float(target_weight_ser.abs().sum()), 2.0)

    def test_iterate_liquidates_old_names_and_submits_long_short_targets(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        strategy_obj.current_bar = pd.Timestamp("2024-04-01")

        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "OLDL", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.add_transaction(8, strategy_obj.previous_bar, "OLDS", -10, 100.0, -1_000.0, 2, 0.0)
        strategy_obj.current_trade_map["OLDL"] = 7
        strategy_obj.current_trade_map["OLDS"] = 8

        row_map: dict[tuple[str, str], float] = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)
        close_row_ser = self.make_close_row_ser(row_map)

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({symbol_str: 100.0 for symbol_str in symbol_list}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        liquidation_asset_set = {
            order_obj.asset
            for order_obj in order_list
            if order_obj.unit == "shares" and order_obj.target
        }
        target_order_map = {
            order_obj.asset: order_obj
            for order_obj in order_list
            if order_obj.unit == "percent" and order_obj.target
        }

        self.assertEqual(liquidation_asset_set, {"OLDL", "OLDS"})
        self.assertAlmostEqual(float(target_order_map["S050"].amount), 0.5)
        self.assertAlmostEqual(float(target_order_map["S049"].amount), 0.5)
        self.assertAlmostEqual(float(target_order_map["S001"].amount), -0.5)
        self.assertAlmostEqual(float(target_order_map["S002"].amount), -0.5)
        self.assertEqual(len(strategy_obj.rebalance_selection_row_list), 4)

    def test_get_selection_df_filters_decision_close_at_one_dollar_or_less(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")

        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )

        row_map: dict[tuple[str, str], float] = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)
        row_map[("S050", "Close")] = 1.0
        row_map[("S001", "Close")] = 0.99

        selection_df = strategy_obj.get_selection_df(close_row_ser=self.make_close_row_ser(row_map))

        long_df = selection_df.loc[selection_df["side_str"] == "long"]
        short_df = selection_df.loc[selection_df["side_str"] == "short"]

        self.assertEqual(long_df["symbol_str"].tolist(), ["S049", "S048"])
        self.assertEqual(short_df["symbol_str"].tolist(), ["S002", "S003"])
        self.assertNotIn("S050", selection_df["symbol_str"].tolist())
        self.assertNotIn("S001", selection_df["symbol_str"].tolist())
        self.assertTrue((selection_df["decision_close_price_float"] > 1.0).all())

    def test_compute_signals_passes_signal_audit_on_small_fixture(self):
        config_obj = self.make_config()
        strategy_obj = self.make_strategy(config=config_obj)
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        self.assertIn(("S001", "trend_slope_20_2_ser"), signal_data_df.columns)
        self.assertIn(("S001", "trend_r2_20_2_ser"), signal_data_df.columns)
        strategy_obj.audit_signals(pricing_data_df, signal_data_df, sample_size=3)

    def test_compute_signals_uses_precomputed_trend_tables(self):
        config_obj = self.make_config()
        pricing_data_df = self.make_pricing_data_df()
        tradeable_symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        decision_date_ts = pricing_data_df.index[-5]
        precomputed_trend_slope_df = pd.DataFrame(
            123.0,
            index=[decision_date_ts],
            columns=tradeable_symbol_list,
        )
        precomputed_trend_r2_df = pd.DataFrame(
            0.75,
            index=[decision_date_ts],
            columns=tradeable_symbol_list,
        )
        strategy_obj = self.make_strategy(
            config=config_obj,
            precomputed_trend_slope_df=precomputed_trend_slope_df,
            precomputed_trend_r2_df=precomputed_trend_r2_df,
        )

        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        self.assertAlmostEqual(
            float(signal_data_df.loc[decision_date_ts, ("S001", "trend_slope_20_2_ser")]),
            123.0,
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[decision_date_ts, ("S001", "trend_r2_20_2_ser")]),
            0.75,
        )

    def test_run_daily_smoke_generates_summary_and_selection_log(self):
        config_obj = self.make_config()
        pricing_data_df = self.make_pricing_data_df(periods_int=160)
        tradeable_symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in tradeable_symbol_list
            },
            index=pricing_data_df.index,
        )
        monthly_decision_close_df, _trend_slope_df, _trend_r2_df = compute_smooth_trend_signal_tables(
            price_close_df=price_close_df,
            lookback_trading_day_int=config_obj.lookback_trading_day_int,
            skip_trading_day_int=config_obj.skip_trading_day_int,
        )
        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
            execution_index=pricing_data_df.index,
        )
        strategy_obj = self.make_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
        )
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: 1 for symbol_str in tradeable_symbol_list},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[pricing_data_df.index >= rebalance_schedule_df.index[0]]
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=calendar_idx,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )

        self.assertIsNotNone(strategy_obj.summary)
        self.assertGreater(len(strategy_obj.results), 0)
        self.assertGreater(len(strategy_obj.get_transactions()), 0)
        self.assertGreater(len(strategy_obj.rebalance_selection_df), 0)


if __name__ == "__main__":
    unittest.main()
