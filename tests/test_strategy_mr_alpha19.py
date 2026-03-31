import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_mr_alpha19 import Alpha19PullbackStrategy


class Alpha19PullbackStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> Alpha19PullbackStrategy:
        base_kwargs = dict(
            name="Alpha19PullbackTest",
            benchmarks=[],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return Alpha19PullbackStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=275, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 50.0 + 0.28 * step_vec + 2.0 * np.sin(step_vec * 0.05)
        bbb_close_vec = 40.0 + 0.18 * step_vec + 1.5 * np.cos(step_vec * 0.03)
        ccc_close_vec = 30.0 + 0.08 * step_vec + 1.2 * np.sin(step_vec * 0.08)

        aaa_close_vec[-7:] = np.array([130.0, 129.0, 128.0, 127.0, 126.0, 125.0, 124.0])
        bbb_close_vec[-7:] = np.array([88.0, 88.5, 89.0, 89.5, 90.0, 90.5, 91.0])
        ccc_close_vec[-7:] = np.array([52.0, 51.8, 51.6, 51.4, 51.2, 51.0, 50.8])

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.4,
                ("AAA", "High"): aaa_close_vec + 0.8,
                ("AAA", "Low"): aaa_close_vec - 0.8,
                ("AAA", "Close"): aaa_close_vec,
                ("AAA", "Turnover"): 35_000_000.0 + 50_000.0 * step_vec,
                ("BBB", "Open"): bbb_close_vec - 0.3,
                ("BBB", "High"): bbb_close_vec + 0.7,
                ("BBB", "Low"): bbb_close_vec - 0.7,
                ("BBB", "Close"): bbb_close_vec,
                ("BBB", "Turnover"): 28_000_000.0 + 40_000.0 * step_vec,
                ("CCC", "Open"): ccc_close_vec - 0.2,
                ("CCC", "High"): ccc_close_vec + 0.6,
                ("CCC", "Low"): ccc_close_vec - 0.6,
                ("CCC", "Close"): ccc_close_vec,
                ("CCC", "Turnover"): 24_000_000.0 + 30_000.0 * step_vec,
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

        self.assertIn(("AAA", "return_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "alpha19_signal_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "trailing_250_return_sum_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "winner_rank_pct_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "adv20_dollar_ser"), signal_data_df.columns)
        self.assertIn(("AAA", "ibs_value_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_alpha19_signal_sign_matches_trailing_seven_day_pullback_term(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()
        signal_data_df = strategy.compute_signals(pricing_data_df)

        last_bar_ts = pricing_data_df.index[-1]
        close_ser = pricing_data_df[("AAA", "Close")]

        close_today_float = float(close_ser.loc[last_bar_ts])
        close_lagged_float = float(close_ser.shift(strategy.alpha_delay_days_int).loc[last_bar_ts])
        expected_sign_float = -np.sign(close_today_float - close_lagged_float)
        actual_signal_float = float(signal_data_df.loc[last_bar_ts, ("AAA", "alpha19_signal_ser")])

        self.assertNotEqual(expected_sign_float, 0.0)
        self.assertEqual(np.sign(actual_signal_float), expected_sign_float)

    def test_get_opportunity_list_enforces_filters_and_deterministic_ranking(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "LOWP": [1],
                "ILLIQ": [1],
                "LOSER": [1],
                "NEG": [1],
                "BELOWMA": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 20.0,
                ("AAA", "alpha19_signal_ser"): 1.5,
                ("AAA", "winner_rank_pct_ser"): 0.80,
                ("AAA", "adv20_dollar_ser"): 30_000_000.0,
                ("AAA", "ibs_value_ser"): 0.10,
                ("BBB", "Close"): 21.0,
                ("BBB", "alpha19_signal_ser"): 1.5,
                ("BBB", "winner_rank_pct_ser"): 0.85,
                ("BBB", "adv20_dollar_ser"): 30_000_000.0,
                ("BBB", "ibs_value_ser"): 0.15,
                ("CCC", "Close"): 22.0,
                ("CCC", "alpha19_signal_ser"): 1.8,
                ("CCC", "winner_rank_pct_ser"): 0.90,
                ("CCC", "adv20_dollar_ser"): 40_000_000.0,
                ("CCC", "ibs_value_ser"): 0.05,
                ("LOWP", "Close"): 9.0,
                ("LOWP", "alpha19_signal_ser"): 2.0,
                ("LOWP", "winner_rank_pct_ser"): 0.95,
                ("LOWP", "adv20_dollar_ser"): 50_000_000.0,
                ("LOWP", "ibs_value_ser"): 0.05,
                ("ILLIQ", "Close"): 20.0,
                ("ILLIQ", "alpha19_signal_ser"): 2.0,
                ("ILLIQ", "winner_rank_pct_ser"): 0.95,
                ("ILLIQ", "adv20_dollar_ser"): 10_000_000.0,
                ("ILLIQ", "ibs_value_ser"): 0.05,
                ("LOSER", "Close"): 20.0,
                ("LOSER", "alpha19_signal_ser"): 2.0,
                ("LOSER", "winner_rank_pct_ser"): 0.60,
                ("LOSER", "adv20_dollar_ser"): 50_000_000.0,
                ("LOSER", "ibs_value_ser"): 0.05,
                ("NEG", "Close"): 20.0,
                ("NEG", "alpha19_signal_ser"): -0.5,
                ("NEG", "winner_rank_pct_ser"): 0.95,
                ("NEG", "adv20_dollar_ser"): 50_000_000.0,
                ("NEG", "ibs_value_ser"): 0.05,
                ("BELOWMA", "Close"): 20.0,
                ("BELOWMA", "alpha19_signal_ser"): 1.7,
                ("BELOWMA", "winner_rank_pct_ser"): 0.92,
                ("BELOWMA", "adv20_dollar_ser"): 55_000_000.0,
                ("BELOWMA", "ibs_value_ser"): 0.05,
                ("OUT", "Close"): 30.0,
                ("OUT", "alpha19_signal_ser"): 2.0,
                ("OUT", "winner_rank_pct_ser"): 0.99,
                ("OUT", "adv20_dollar_ser"): 80_000_000.0,
                ("OUT", "ibs_value_ser"): 0.05,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, ["CCC", "BELOWMA", "AAA", "BBB"])

    def test_get_opportunity_list_does_not_require_sma_feature(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame({"BELOWMA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("BELOWMA", "Close"): 20.0,
                ("BELOWMA", "alpha19_signal_ser"): 1.2,
                ("BELOWMA", "winner_rank_pct_ser"): 0.80,
                ("BELOWMA", "adv20_dollar_ser"): 30_000_000.0,
                ("BELOWMA", "ibs_value_ser"): 0.10,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, ["BELOWMA"])

    def test_get_opportunity_list_requires_ibs_below_threshold(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 20.0,
                ("AAA", "alpha19_signal_ser"): 1.2,
                ("AAA", "winner_rank_pct_ser"): 0.80,
                ("AAA", "adv20_dollar_ser"): 30_000_000.0,
                ("AAA", "ibs_value_ser"): 0.19,
                ("BBB", "Close"): 21.0,
                ("BBB", "alpha19_signal_ser"): 1.4,
                ("BBB", "winner_rank_pct_ser"): 0.85,
                ("BBB", "adv20_dollar_ser"): 35_000_000.0,
                ("BBB", "ibs_value_ser"): 0.25,
            }
        )

        opportunity_list = strategy.get_opportunity_list(close_row_ser)

        self.assertEqual(opportunity_list, ["AAA"])

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
                ("AAA", "Close"): 50.0,
                ("AAA", "alpha19_signal_ser"): 1.0,
                ("AAA", "winner_rank_pct_ser"): 0.80,
                ("AAA", "adv20_dollar_ser"): 30_000_000.0,
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

    def test_iterate_submits_exit_order_when_alpha19_is_non_positive(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(9, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 9

        data_df = pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5))
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 50.0,
                ("AAA", "alpha19_signal_ser"): 0.0,
                ("AAA", "winner_rank_pct_ser"): 0.80,
                ("AAA", "adv20_dollar_ser"): 30_000_000.0,
            }
        )

        strategy.iterate(data_df, close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[0].trade_id, 9)
        self.assertEqual(order_list[0].amount, 0)
        self.assertTrue(order_list[0].target)

    def test_iterate_skips_held_name_and_enters_next_ranked_candidate(self):
        strategy = self.make_strategy(max_positions_int=2)
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.trade_id_int = 11
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1], "CCC": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(11, pd.Timestamp("2024-03-07"), "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 11

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 20.0,
                ("AAA", "alpha19_signal_ser"): 2.0,
                ("AAA", "winner_rank_pct_ser"): 0.90,
                ("AAA", "adv20_dollar_ser"): 35_000_000.0,
                ("AAA", "ibs_value_ser"): 0.10,
                ("BBB", "Close"): 21.0,
                ("BBB", "alpha19_signal_ser"): 1.8,
                ("BBB", "winner_rank_pct_ser"): 0.85,
                ("BBB", "adv20_dollar_ser"): 40_000_000.0,
                ("BBB", "ibs_value_ser"): 0.15,
                ("CCC", "Close"): 22.0,
                ("CCC", "alpha19_signal_ser"): 1.7,
                ("CCC", "winner_rank_pct_ser"): 0.83,
                ("CCC", "adv20_dollar_ser"): 30_000_000.0,
                ("CCC", "ibs_value_ser"): 0.18,
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
        self.assertEqual(entry_order.trade_id, 12)

    def test_iterate_submits_no_orders_when_features_are_not_ready(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "Close"): 20.0,
                ("AAA", "alpha19_signal_ser"): np.nan,
                ("AAA", "winner_rank_pct_ser"): np.nan,
                ("AAA", "adv20_dollar_ser"): np.nan,
            }
        )

        strategy.iterate(pd.DataFrame(index=pd.bdate_range("2024-03-04", periods=5)), close_row_ser, pd.Series(dtype=float))

        self.assertEqual(len(strategy.get_orders()), 0)


if __name__ == "__main__":
    unittest.main()
