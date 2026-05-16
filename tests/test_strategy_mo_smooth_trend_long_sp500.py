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
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_smooth_trend_long_sp500 import (
    DEFAULT_CONFIG,
    SmoothTrendLongSp500Strategy,
    compute_smooth_trend_signal_tables,
    get_trend_r2_field_str,
    get_trend_slope_field_str,
)


class SmoothTrendLongSp500StrategyTests(unittest.TestCase):
    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-04-01",
        decision_date_str: str = "2024-03-28",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> SmoothTrendLongSp500Strategy:
        base_kwargs = dict(
            name="SmoothTrendLongSp500Test",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_trading_day_int=252,
            skip_trading_day_int=21,
            quintile_count_int=5,
        )
        base_kwargs.update(kwargs)
        return SmoothTrendLongSp500Strategy(**base_kwargs)

    @staticmethod
    def make_close_vec(base_price_float: float, daily_return_vec: np.ndarray) -> np.ndarray:
        return base_price_float * np.cumprod(1.0 + daily_return_vec)

    def make_pricing_data_df(self, periods_int: int = 520) -> pd.DataFrame:
        date_index = pd.date_range("2022-01-03", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_return_vec = 0.0012 + 0.00002 * np.sin(step_vec * 0.03)
        bbb_return_vec = 0.0010 + 0.00002 * np.cos(step_vec * 0.04)
        ccc_return_vec = 0.0007 + 0.00070 * np.where((step_vec.astype(int) % 10) < 5, 1.0, -1.0)
        ddd_return_vec = 0.0005 + 0.00100 * np.sin(step_vec * 0.22)
        eee_return_vec = -0.0004 + 0.00002 * np.cos(step_vec * 0.05)
        fff_return_vec = 0.0003 + 0.00050 * np.sin(step_vec * 0.11)
        spy_return_vec = 0.0005 + 0.00003 * np.sin(step_vec * 0.02)

        close_map = {
            "AAA": self.make_close_vec(100.0, aaa_return_vec),
            "BBB": self.make_close_vec(95.0, bbb_return_vec),
            "CCC": self.make_close_vec(90.0, ccc_return_vec),
            "DDD": self.make_close_vec(85.0, ddd_return_vec),
            "EEE": self.make_close_vec(80.0, eee_return_vec),
            "FFF": self.make_close_vec(75.0, fff_return_vec),
            "SPY": self.make_close_vec(300.0, spy_return_vec),
        }

        pricing_data_map: dict[tuple[str, str], np.ndarray] = {}
        for symbol_str, close_vec in close_map.items():
            pricing_data_map[(symbol_str, "Open")] = close_vec * 0.999
            pricing_data_map[(symbol_str, "High")] = close_vec * 1.010
            pricing_data_map[(symbol_str, "Low")] = close_vec * 0.990
            pricing_data_map[(symbol_str, "Close")] = close_vec

        pricing_data_df = pd.DataFrame(pricing_data_map, index=date_index, dtype=float)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_config_matches_first_practical_paper_window(self):
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(DEFAULT_CONFIG.benchmark_list, ("$SPX",))
        self.assertEqual(DEFAULT_CONFIG.lookback_trading_day_int, 252)
        self.assertEqual(DEFAULT_CONFIG.skip_trading_day_int, 21)
        self.assertEqual(DEFAULT_CONFIG.quintile_count_int, 5)
        self.assertEqual(get_trend_slope_field_str(252, 21), "trend_slope_252_21_ser")
        self.assertEqual(get_trend_r2_field_str(252, 21), "trend_r2_252_21_ser")

    def test_compute_smooth_trend_signal_tables_matches_manual_ols_window(self):
        pricing_data_df = self.make_pricing_data_df()
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
            },
            index=pricing_data_df.index,
        )

        _monthly_decision_close_df, trend_slope_df, trend_r2_df = compute_smooth_trend_signal_tables(
            price_close_df=price_close_df,
            lookback_trading_day_int=252,
            skip_trading_day_int=21,
        )

        decision_date_ts = pd.Timestamp(trend_slope_df.index[-1])
        decision_pos_int = int(price_close_df.index.get_loc(decision_date_ts))
        # *** CRITICAL*** This manual check must use the same t-252 through
        # t-21 skipped formation window as the production signal.
        formation_price_ser = price_close_df["AAA"].iloc[decision_pos_int - 252 : decision_pos_int - 21 + 1]
        return_window_ser = formation_price_ser.pct_change().iloc[1:]
        cumulative_return_vec = return_window_ser.cumsum().to_numpy(dtype=float)
        time_vec = np.arange(1, len(cumulative_return_vec) + 1, dtype=float)
        centered_time_vec = time_vec - float(time_vec.mean())
        time_ss_float = float(np.dot(centered_time_vec, centered_time_vec))
        expected_slope_float = float(centered_time_vec @ cumulative_return_vec / time_ss_float)
        centered_cumulative_return_vec = cumulative_return_vec - float(cumulative_return_vec.mean())
        expected_r2_float = float(
            (expected_slope_float * expected_slope_float * time_ss_float)
            / np.dot(centered_cumulative_return_vec, centered_cumulative_return_vec)
        )

        self.assertEqual(len(return_window_ser), 231)
        self.assertAlmostEqual(float(trend_slope_df.loc[decision_date_ts, "AAA"]), expected_slope_float, places=14)
        self.assertAlmostEqual(float(trend_r2_df.loc[decision_date_ts, "AAA"]), expected_r2_float, places=14)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "trend_slope_252_21_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "trend_r2_252_21_ser"), signal_data_df.columns)
        self.assertIn(("FFF", "trend_slope_252_21_ser"), signal_data_df.columns)
        self.assertIn(("FFF", "trend_r2_252_21_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df, sample_size=5)

    def test_get_target_weight_ser_uses_pit_r2_first_then_slope(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "FFF": [1],
                "GGG": [1],
                "HHH": [1],
                "III": [1],
                "JJJ": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "trend_r2_252_21_ser"): 0.99,
                ("AAA", "trend_slope_252_21_ser"): 0.020,
                ("BBB", "trend_r2_252_21_ser"): 0.98,
                ("BBB", "trend_slope_252_21_ser"): 0.030,
                ("CCC", "trend_r2_252_21_ser"): 0.97,
                ("CCC", "trend_slope_252_21_ser"): 0.090,
                ("DDD", "trend_r2_252_21_ser"): 0.80,
                ("DDD", "trend_slope_252_21_ser"): 0.080,
                ("EEE", "trend_r2_252_21_ser"): 0.70,
                ("EEE", "trend_slope_252_21_ser"): 0.070,
                ("FFF", "trend_r2_252_21_ser"): 0.60,
                ("FFF", "trend_slope_252_21_ser"): 0.060,
                ("GGG", "trend_r2_252_21_ser"): 0.50,
                ("GGG", "trend_slope_252_21_ser"): 0.050,
                ("HHH", "trend_r2_252_21_ser"): 0.40,
                ("HHH", "trend_slope_252_21_ser"): 0.040,
                ("III", "trend_r2_252_21_ser"): 0.30,
                ("III", "trend_slope_252_21_ser"): 0.030,
                ("JJJ", "trend_r2_252_21_ser"): 0.20,
                ("JJJ", "trend_slope_252_21_ser"): 0.020,
                ("OUT", "trend_r2_252_21_ser"): 1.00,
                ("OUT", "trend_slope_252_21_ser"): 1.000,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["BBB"])
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 1.0)

    def test_get_target_weight_ser_stays_cash_when_smooth_bucket_has_negative_slope(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "FFF": [1],
                "GGG": [1],
                "HHH": [1],
                "III": [1],
                "JJJ": [1],
            },
            index=[strategy.previous_bar],
        )
        close_row_map: dict[tuple[str, str], float] = {}
        for symbol_index_int, symbol_str in enumerate(strategy.universe_df.columns.astype(str).tolist()):
            close_row_map[(symbol_str, "trend_r2_252_21_ser")] = 1.00 - symbol_index_int * 0.05
            close_row_map[(symbol_str, "trend_slope_252_21_ser")] = -0.010 - symbol_index_int * 0.001
        close_row_ser = self.make_close_row_ser(close_row_map)

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(len(target_weight_ser), 0)

    def test_iterate_submits_liquidation_and_new_target_order_on_rebalance(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.current_bar = pd.Timestamp("2024-04-01")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "FFF": [1],
                "GGG": [1],
                "HHH": [1],
                "III": [1],
                "JJJ": [1],
            },
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "trend_r2_252_21_ser"): 0.99,
                ("AAA", "trend_slope_252_21_ser"): 0.020,
                ("BBB", "trend_r2_252_21_ser"): 0.98,
                ("BBB", "trend_slope_252_21_ser"): 0.030,
                ("CCC", "trend_r2_252_21_ser"): 0.97,
                ("CCC", "trend_slope_252_21_ser"): 0.090,
                ("DDD", "trend_r2_252_21_ser"): 0.80,
                ("DDD", "trend_slope_252_21_ser"): 0.080,
                ("EEE", "trend_r2_252_21_ser"): 0.70,
                ("EEE", "trend_slope_252_21_ser"): 0.070,
                ("FFF", "trend_r2_252_21_ser"): 0.60,
                ("FFF", "trend_slope_252_21_ser"): 0.060,
                ("GGG", "trend_r2_252_21_ser"): 0.50,
                ("GGG", "trend_slope_252_21_ser"): 0.050,
                ("HHH", "trend_r2_252_21_ser"): 0.40,
                ("HHH", "trend_slope_252_21_ser"): 0.040,
                ("III", "trend_r2_252_21_ser"): 0.30,
                ("III", "trend_slope_252_21_ser"): 0.030,
                ("JJJ", "trend_r2_252_21_ser"): 0.20,
                ("JJJ", "trend_slope_252_21_ser"): 0.020,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0}, dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "BBB"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))

        liquidation_order = order_list[0]
        self.assertEqual(liquidation_order.amount, 0)
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.trade_id, 7)

        new_order = order_list[1]
        self.assertEqual(new_order.unit, "percent")
        self.assertTrue(new_order.target)
        self.assertAlmostEqual(float(new_order.amount), 1.0)
        self.assertEqual(new_order.trade_id, 1)
        self.assertEqual(strategy.current_trade_map["BBB"], 1)

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df()
        tradeable_symbol_list = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")].astype(float)
                for symbol_str in tradeable_symbol_list
            },
            index=pricing_data_df.index,
        )
        monthly_decision_close_df, _trend_slope_df, _trend_r2_df = compute_smooth_trend_signal_tables(
            price_close_df=price_close_df,
            lookback_trading_day_int=252,
            skip_trading_day_int=21,
        )
        rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
            execution_index=pricing_data_df.index,
        )

        strategy = self.make_strategy(rebalance_schedule_df=rebalance_schedule_df)
        strategy.universe_df = pd.DataFrame(
            {symbol_str: 1 for symbol_str in tradeable_symbol_list},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[pricing_data_df.index >= rebalance_schedule_df.index[0]]
        run_daily(
            strategy,
            pricing_data_df,
            calendar=calendar_idx,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
            audit_sample_size_int=5,
        )

        self.assertIsNotNone(strategy.summary)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)


if __name__ == "__main__":
    unittest.main()
