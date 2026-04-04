import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_mo_ndx_rotation import (
    NdxTrendRotationStrategy,
    map_weekly_decision_dates_to_execution_date_df,
)


class NdxTrendRotationStrategyTests(unittest.TestCase):
    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-03-11",
        decision_date_str: str = "2024-03-08",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> NdxTrendRotationStrategy:
        base_kwargs = dict(
            name="NdxTrendRotationTest",
            benchmarks=["QQQ"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=5,
            require_scaled_rs_positive_bool=True,
            require_scaled_rs_above_benchmark_bool=True,
        )
        base_kwargs.update(kwargs)
        return NdxTrendRotationStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=260, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 100.0 + 0.60 * step_vec + 1.5 * np.sin(step_vec * 0.07)
        bbb_close_vec = 90.0 + 0.45 * step_vec + 1.2 * np.cos(step_vec * 0.05)
        ccc_close_vec = 150.0 - 0.20 * step_vec + 1.0 * np.sin(step_vec * 0.03)
        ndx_close_vec = 10_000.0 + 8.0 * step_vec + 12.0 * np.sin(step_vec * 0.04)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.5,
                ("AAA", "High"): aaa_close_vec + 1.0,
                ("AAA", "Low"): aaa_close_vec - 1.0,
                ("AAA", "Close"): aaa_close_vec,
                ("AAA", "Turnover"): 30_000_000.0 + 50_000.0 * step_vec,
                ("BBB", "Open"): bbb_close_vec - 0.5,
                ("BBB", "High"): bbb_close_vec + 1.0,
                ("BBB", "Low"): bbb_close_vec - 1.0,
                ("BBB", "Close"): bbb_close_vec,
                ("BBB", "Turnover"): 28_000_000.0 + 40_000.0 * step_vec,
                ("CCC", "Open"): ccc_close_vec - 0.5,
                ("CCC", "High"): ccc_close_vec + 1.0,
                ("CCC", "Low"): ccc_close_vec - 1.0,
                ("CCC", "Close"): ccc_close_vec,
                ("CCC", "Turnover"): 27_000_000.0 + 30_000.0 * step_vec,
                ("QQQ", "Open"): ndx_close_vec - 5.0,
                ("QQQ", "High"): ndx_close_vec + 10.0,
                ("QQQ", "Low"): ndx_close_vec - 10.0,
                ("QQQ", "Close"): ndx_close_vec,
                ("QQQ", "Turnover"): 100_000_000.0 + 75_000.0 * step_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_map_weekly_decision_dates_uses_next_tradable_open_for_holidays(self):
        execution_index = pd.to_datetime(
            [
                "2024-02-12",
                "2024-02-13",
                "2024-02-14",
                "2024-02-15",
                "2024-02-16",
                "2024-02-20",
                "2024-02-21",
                "2024-02-22",
                "2024-02-23",
            ]
        )

        rebalance_schedule_df = map_weekly_decision_dates_to_execution_date_df(execution_index)

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-02-20"), "decision_date_ts"]),
            pd.Timestamp("2024-02-16"),
        )

    def test_map_weekly_decision_dates_uses_thursday_close_when_friday_is_closed(self):
        execution_index = pd.to_datetime(
            [
                "2024-03-25",
                "2024-03-26",
                "2024-03-27",
                "2024-03-28",
                "2024-04-01",
                "2024-04-02",
            ]
        )

        rebalance_schedule_df = map_weekly_decision_dates_to_execution_date_df(execution_index)

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-04-01"), "decision_date_ts"]),
            pd.Timestamp("2024-03-28"),
        )

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        slope_field_prefix_str = f"slope_{strategy.slope_window_int}"
        drawdown_window_int = strategy.benchmark_drawdown_window_int

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "adv20_dollar_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "turnover_dollar_ser"), signal_data_df.columns)
        self.assertIn(("AAA", f"atr_{strategy.atr_window_int}_ser"), signal_data_df.columns)
        self.assertIn(("AAA", f"{slope_field_prefix_str}_log_beta_ser"), signal_data_df.columns)
        self.assertIn(("AAA", f"{slope_field_prefix_str}_ann_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "rs_factor_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "scaled_rs_factor_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", "benchmark_sma200_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", f"benchmark_peak_{drawdown_window_int}_high_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", f"benchmark_drawdown_{drawdown_window_int}_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", "benchmark_scaled_rs_factor_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", "market_regime_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_get_selected_symbol_list_filters_pit_universe_and_ranks_by_rs_factor(self):
        strategy = self.make_strategy(max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "OUT": [0]},
            index=[strategy.previous_bar],
        )
        slope_field_str = f"slope_{strategy.slope_window_int}_ann_ser"

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 120.0,
                ("AAA", "turnover_dollar_ser"): 25_000_000.0,
                ("AAA", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("AAA", slope_field_str): 0.20,
                ("AAA", "rs_factor_ser"): 0.50,
                ("AAA", "scaled_rs_factor_ser"): 0.40,
                ("BBB", "Close"): 130.0,
                ("BBB", "turnover_dollar_ser"): 24_000_000.0,
                ("BBB", f"atr_{strategy.atr_window_int}_ser"): 9.0,
                ("BBB", slope_field_str): 0.15,
                ("BBB", "rs_factor_ser"): 0.60,
                ("BBB", "scaled_rs_factor_ser"): 0.50,
                ("CCC", "Close"): 8.0,
                ("CCC", "turnover_dollar_ser"): 30_000_000.0,
                ("CCC", f"atr_{strategy.atr_window_int}_ser"): 8.0,
                ("CCC", slope_field_str): 0.25,
                ("CCC", "rs_factor_ser"): 0.90,
                ("CCC", "scaled_rs_factor_ser"): 0.60,
                ("OUT", "Close"): 200.0,
                ("OUT", "turnover_dollar_ser"): 50_000_000.0,
                ("OUT", f"atr_{strategy.atr_window_int}_ser"): 7.0,
                ("OUT", slope_field_str): 0.30,
                ("OUT", "rs_factor_ser"): 0.95,
                ("OUT", "scaled_rs_factor_ser"): 0.80,
                ("QQQ", "benchmark_scaled_rs_factor_ser"): 0.30,
                ("QQQ", "market_regime_bool"): True,
            }
        )

        selected_symbol_list = strategy.get_selected_symbol_list(close_row_ser)

        self.assertEqual(selected_symbol_list, ["BBB", "AAA"])

    def test_get_selected_symbol_list_requires_scaled_rs_to_exceed_zero_and_benchmark(self):
        strategy = self.make_strategy(max_positions_int=3)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[strategy.previous_bar],
        )
        slope_field_str = f"slope_{strategy.slope_window_int}_ann_ser"

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 120.0,
                ("AAA", "turnover_dollar_ser"): 25_000_000.0,
                ("AAA", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("AAA", slope_field_str): 0.20,
                ("AAA", "rs_factor_ser"): 0.70,
                ("AAA", "scaled_rs_factor_ser"): 0.25,
                ("BBB", "Close"): 130.0,
                ("BBB", "turnover_dollar_ser"): 25_000_000.0,
                ("BBB", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("BBB", slope_field_str): 0.20,
                ("BBB", "rs_factor_ser"): 0.60,
                ("BBB", "scaled_rs_factor_ser"): -0.05,
                ("CCC", "Close"): 140.0,
                ("CCC", "turnover_dollar_ser"): 25_000_000.0,
                ("CCC", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("CCC", slope_field_str): 0.20,
                ("CCC", "rs_factor_ser"): 0.80,
                ("CCC", "scaled_rs_factor_ser"): 0.10,
                ("QQQ", "benchmark_scaled_rs_factor_ser"): 0.15,
                ("QQQ", "market_regime_bool"): True,
            }
        )

        selected_symbol_list = strategy.get_selected_symbol_list(close_row_ser)

        self.assertEqual(selected_symbol_list, ["AAA"])

    def test_drawdown_regime_turns_off_after_twelve_percent_drop_from_rolling_peak(self):
        strategy = self.make_strategy()
        drawdown_window_int = strategy.benchmark_drawdown_window_int
        date_index = pd.date_range("2024-01-02", periods=140, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)
        benchmark_close_vec = np.linspace(100.0, 140.0, len(date_index))
        benchmark_high_vec = benchmark_close_vec + 1.0
        benchmark_close_vec[-1] = 120.0
        benchmark_high_vec[-1] = 121.0

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): 50.0 + 0.2 * step_vec,
                ("AAA", "High"): 51.0 + 0.2 * step_vec,
                ("AAA", "Low"): 49.0 + 0.2 * step_vec,
                ("AAA", "Close"): 50.5 + 0.2 * step_vec,
                ("AAA", "Turnover"): 30_000_000.0 + 1_000.0 * step_vec,
                ("QQQ", "Open"): benchmark_close_vec - 0.5,
                ("QQQ", "High"): benchmark_high_vec,
                ("QQQ", "Low"): benchmark_close_vec - 1.0,
                ("QQQ", "Close"): benchmark_close_vec,
                ("QQQ", "Turnover"): 100_000_000.0 + 2_000.0 * step_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        signal_data_df = strategy.compute_signals(pricing_data_df)
        last_bar_ts = date_index[-1]
        benchmark_peak_high_float = float(
            signal_data_df.loc[last_bar_ts, ("QQQ", f"benchmark_peak_{drawdown_window_int}_high_ser")]
        )
        benchmark_drawdown_float = float(
            signal_data_df.loc[last_bar_ts, ("QQQ", f"benchmark_drawdown_{drawdown_window_int}_ser")]
        )
        market_regime_bool = bool(signal_data_df.loc[last_bar_ts, ("QQQ", "market_regime_bool")])

        expected_peak_high_float = float(np.max(benchmark_high_vec[-drawdown_window_int:]))
        expected_drawdown_float = (120.0 / benchmark_peak_high_float) - 1.0
        self.assertAlmostEqual(benchmark_peak_high_float, expected_peak_high_float)
        self.assertAlmostEqual(benchmark_drawdown_float, expected_drawdown_float)
        self.assertLess(benchmark_drawdown_float, -0.12)
        self.assertFalse(market_regime_bool)

    def test_iterate_liquidates_all_positions_when_regime_is_false(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7
        slope_field_str = f"slope_{strategy.slope_window_int}_ann_ser"

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 120.0,
                ("AAA", "turnover_dollar_ser"): 25_000_000.0,
                ("AAA", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("AAA", slope_field_str): 0.20,
                ("AAA", "rs_factor_ser"): 0.50,
                ("AAA", "scaled_rs_factor_ser"): 0.40,
                ("QQQ", "benchmark_scaled_rs_factor_ser"): 0.20,
                ("QQQ", "market_regime_bool"): False,
            }
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.asset, "AAA")
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertEqual(liquidation_order.amount, 0)
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.trade_id, 7)

    def test_iterate_uses_atr_risk_target_shares(self):
        strategy = self.make_strategy(max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1]}, index=[strategy.previous_bar])
        slope_field_str = f"slope_{strategy.slope_window_int}_ann_ser"

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 120.0,
                ("AAA", "turnover_dollar_ser"): 25_000_000.0,
                ("AAA", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("AAA", slope_field_str): 0.20,
                ("AAA", "rs_factor_ser"): 0.50,
                ("AAA", "scaled_rs_factor_ser"): 0.40,
                ("BBB", "Close"): 130.0,
                ("BBB", "turnover_dollar_ser"): 25_000_000.0,
                ("BBB", f"atr_{strategy.atr_window_int}_ser"): 10.0,
                ("BBB", slope_field_str): -0.10,
                ("BBB", "rs_factor_ser"): 0.40,
                ("BBB", "scaled_rs_factor_ser"): -0.20,
                ("QQQ", "benchmark_scaled_rs_factor_ser"): 0.10,
                ("QQQ", "market_regime_bool"): True,
            }
        )
        open_price_ser = pd.Series({"AAA": 100.0})

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "AAA")
        self.assertEqual(entry_order.unit, "shares")
        self.assertTrue(entry_order.target)
        self.assertEqual(entry_order.amount, 50)
        self.assertEqual(entry_order.trade_id, 1)
        self.assertEqual(strategy.current_trade_map["AAA"], 1)

    def test_iterate_skips_resize_when_share_difference_is_within_rebalance_band(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(9, strategy.previous_bar, "AAA", 50, 100.0, 5_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 9
        slope_field_str = f"slope_{strategy.slope_window_int}_ann_ser"

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 120.0,
                ("AAA", "turnover_dollar_ser"): 25_000_000.0,
                ("AAA", f"atr_{strategy.atr_window_int}_ser"): 9.8,
                ("AAA", slope_field_str): 0.20,
                ("AAA", "rs_factor_ser"): 0.50,
                ("AAA", "scaled_rs_factor_ser"): 0.40,
                ("QQQ", "benchmark_scaled_rs_factor_ser"): 0.10,
                ("QQQ", "market_regime_bool"): True,
            }
        )

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, pd.Series({"AAA": 100.0}))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df().copy()
        late_regime_break_index = pricing_data_df.index[-20:]
        late_ndx_close_vec = np.linspace(9_000.0, 8_200.0, len(late_regime_break_index))
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "Open")] = late_ndx_close_vec - 5.0
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "High")] = late_ndx_close_vec + 10.0
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "Low")] = late_ndx_close_vec - 10.0
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "Close")] = late_ndx_close_vec

        rebalance_schedule_df = map_weekly_decision_dates_to_execution_date_df(pricing_data_df.index)
        strategy = self.make_strategy(
            rebalance_schedule_df=rebalance_schedule_df,
            max_positions_int=2,
        )
        strategy.universe_df = pd.DataFrame(
            {"AAA": 1, "BBB": 1, "CCC": 1},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[200:]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=calendar_idx,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=True,
                audit_sample_size_int=5,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)


if __name__ == "__main__":
    unittest.main()
