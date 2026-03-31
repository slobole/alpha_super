import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from alpha.engine.backtest import run_daily
from strategies.strategy_mo_alpha23_breakout import Alpha23BreakoutStrategy


class Alpha23BreakoutStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> Alpha23BreakoutStrategy:
        base_kwargs = dict(
            name="Alpha23BreakoutTest",
            benchmarks=[],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return Alpha23BreakoutStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=35, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_high_vec = np.linspace(50.0, 54.0, len(date_index))
        aaa_high_vec[-3:] = np.array([54.0, 54.5, 56.0])
        aaa_open_vec = aaa_high_vec - 1.0
        aaa_low_vec = aaa_high_vec - 2.0
        aaa_close_vec = aaa_high_vec - 0.5

        bbb_high_vec = np.linspace(35.0, 37.0, len(date_index))
        bbb_high_vec[-3:] = np.array([37.2, 37.3, 38.0])
        bbb_open_vec = bbb_high_vec - 0.8
        bbb_low_vec = bbb_high_vec - 1.6
        bbb_close_vec = bbb_high_vec - 0.3

        ccc_high_vec = np.linspace(25.0, 27.0, len(date_index))
        ccc_high_vec[-3:] = np.array([28.0, 28.5, 27.0])
        ccc_open_vec = ccc_high_vec - 0.7
        ccc_low_vec = ccc_high_vec - 1.4
        ccc_close_vec = ccc_high_vec - 0.2

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_open_vec,
                ("AAA", "High"): aaa_high_vec,
                ("AAA", "Low"): aaa_low_vec,
                ("AAA", "Close"): aaa_close_vec,
                ("BBB", "Open"): bbb_open_vec,
                ("BBB", "High"): bbb_high_vec,
                ("BBB", "Low"): bbb_low_vec,
                ("BBB", "Close"): bbb_close_vec,
                ("CCC", "Open"): ccc_open_vec,
                ("CCC", "High"): ccc_high_vec,
                ("CCC", "Low"): ccc_low_vec,
                ("CCC", "Close"): ccc_close_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "prior_avg_high_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "high_return_2_day_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "alpha23_breakout_bool"), signal_data_df.columns)
        self.assertIn(("AAA", "alpha23_score_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_breakout_gate_uses_prior_window_average_only(self):
        strategy = self.make_strategy(
            breakout_window_int=3,
            score_lookback_days_int=2,
        )
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")
        high_price_vec = np.array([100.0, 100.0, 1.0, 1.0, 60.0], dtype=float)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): high_price_vec - 1.0,
                ("AAA", "High"): high_price_vec,
                ("AAA", "Low"): high_price_vec - 2.0,
                ("AAA", "Close"): high_price_vec - 0.5,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)

        signal_data_df = strategy.compute_signals(pricing_data_df)

        last_bar_ts = date_index[-1]
        prior_avg_high_float = float(signal_data_df.loc[last_bar_ts, ("AAA", "prior_avg_high_ser")])
        breakout_bool = bool(signal_data_df.loc[last_bar_ts, ("AAA", "alpha23_breakout_bool")])

        self.assertAlmostEqual(prior_avg_high_float, (100.0 + 1.0 + 1.0) / 3.0)
        self.assertTrue(breakout_bool)

    def test_alpha23_score_matches_negative_two_day_high_return(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        signal_data_df = strategy.compute_signals(pricing_data_df)

        last_bar_ts = pricing_data_df.index[-1]
        high_price_ser = pricing_data_df[("AAA", "High")]
        expected_score_float = -(
            float(high_price_ser.loc[last_bar_ts]) / float(high_price_ser.shift(2).loc[last_bar_ts]) - 1.0
        )
        actual_score_float = float(signal_data_df.loc[last_bar_ts, ("AAA", "alpha23_score_ser")])

        self.assertTrue(bool(signal_data_df.loc[last_bar_ts, ("AAA", "alpha23_breakout_bool")]))
        self.assertAlmostEqual(actual_score_float, expected_score_float)

    def test_get_opportunity_list_filters_to_breakouts_inside_pit_universe_and_ranks_by_score(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "OUT": [0]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "alpha23_breakout_bool"): True,
                ("AAA", "alpha23_score_ser"): -0.020,
                ("BBB", "alpha23_breakout_bool"): True,
                ("BBB", "alpha23_score_ser"): -0.010,
                ("CCC", "alpha23_breakout_bool"): False,
                ("CCC", "alpha23_score_ser"): 0.000,
                ("OUT", "alpha23_breakout_bool"): True,
                ("OUT", "alpha23_score_ser"): 0.500,
            }
        )

        opportunity_symbol_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_symbol_list, ["BBB", "AAA"])

    def test_iterate_submits_exit_order_on_max_holding_days(self):
        strategy = self.make_strategy(max_holding_days_int=5)
        strategy.previous_bar = pd.Timestamp("2024-02-06")
        strategy.current_bar = pd.Timestamp("2024-02-07")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(7, pd.Timestamp("2024-01-31"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7

        data_df = pd.DataFrame(index=pd.bdate_range("2024-01-31", periods=5))
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "alpha23_breakout_bool"): True,
                ("AAA", "alpha23_score_ser"): -0.010,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.asset, "AAA")
        self.assertEqual(liquidation_order.amount, 0)
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.trade_id, 7)

    def test_iterate_skips_held_name_and_enters_next_ranked_candidate(self):
        strategy = self.make_strategy(max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.trade_id_int = 11
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(11, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 11

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "alpha23_breakout_bool"): True,
                ("AAA", "alpha23_score_ser"): -0.005,
                ("BBB", "alpha23_breakout_bool"): True,
                ("BBB", "alpha23_score_ser"): -0.010,
                ("CCC", "alpha23_breakout_bool"): True,
                ("CCC", "alpha23_score_ser"): -0.020,
            }
        )

        strategy.iterate(
            pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)),
            close_row_ser,
            pd.Series(dtype=float),
        )

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "BBB")
        self.assertEqual(entry_order.unit, "value")
        self.assertFalse(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 50_000.0)
        self.assertEqual(entry_order.trade_id, 12)

    def test_run_daily_smoke_generates_summary(self):
        strategy = self.make_strategy(
            benchmarks=["$SPX"],
            max_positions_int=1,
            breakout_window_int=3,
            score_lookback_days_int=2,
            max_holding_days_int=3,
        )
        pricing_data_df = self.make_pricing_data_df().copy()
        benchmark_close_vec = np.linspace(4_000.0, 4_100.0, len(pricing_data_df.index))
        pricing_data_df[("$SPX", "Open")] = benchmark_close_vec - 5.0
        pricing_data_df[("$SPX", "High")] = benchmark_close_vec + 10.0
        pricing_data_df[("$SPX", "Low")] = benchmark_close_vec - 10.0
        pricing_data_df[("$SPX", "Close")] = benchmark_close_vec
        pricing_data_df = pricing_data_df.sort_index(axis=1)

        strategy.universe_df = pd.DataFrame(
            {"AAA": 1, "BBB": 1, "CCC": 1},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[5:]
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


if __name__ == "__main__":
    unittest.main()
