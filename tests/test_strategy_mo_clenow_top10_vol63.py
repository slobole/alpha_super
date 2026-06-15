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
from strategies.momentum.strategy_mo_clenow_top10_vol63 import (
    DEFAULT_CONFIG,
    NASDAQ100_CONFIG,
    RUSSELL1000_CONFIG,
    SP500_CONFIG,
    ClenowTop10Vol63Strategy,
    compute_clenow_top10_vol63_signal_tables,
    map_month_end_rebalance_schedule_df,
)


class ClenowTop10Vol63Tests(unittest.TestCase):
    def make_test_config(self):
        return replace(
            DEFAULT_CONFIG,
            variant_key_str="test",
            regime_symbol_str="SPY",
            max_positions_int=2,
            clenow_lookback_int=5,
            vol_window_int=3,
            stock_trend_window_int=3,
            regime_trend_window_int=3,
            gap_window_int=3,
        )

    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-01-10",
        decision_date_str: str = "2024-01-09",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> ClenowTop10Vol63Strategy:
        config_obj = kwargs.pop("config", self.make_test_config())
        base_kwargs = dict(
            name="ClenowTop10Vol63Test",
            benchmarks=[config_obj.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )
        base_kwargs.update(kwargs)
        return ClenowTop10Vol63Strategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_configs_use_matching_regime_indexes(self):
        self.assertEqual(SP500_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(SP500_CONFIG.regime_symbol_str, "$SPX")
        self.assertEqual(NASDAQ100_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(NASDAQ100_CONFIG.regime_symbol_str, "$NDX")
        self.assertEqual(RUSSELL1000_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(RUSSELL1000_CONFIG.regime_symbol_str, "$RUI")

    def test_map_month_end_rebalance_schedule_uses_next_tradable_open(self):
        execution_index = pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-28",
                "2024-02-29",
                "2024-03-01",
            ]
        )
        decision_date_index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"])

        rebalance_schedule_df = map_month_end_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        expected_schedule_df = pd.DataFrame(
            {"decision_date_ts": pd.to_datetime(["2024-01-31", "2024-02-29"])},
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
        )
        expected_schedule_df.index.name = "execution_date_ts"
        pd.testing.assert_frame_equal(rebalance_schedule_df, expected_schedule_df)

    def test_compute_signal_tables_match_manual_log_regression_and_vol63(self):
        config_obj = self.make_test_config()
        date_index = pd.bdate_range("2024-01-01", periods=8)
        price_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 102.0, 104.0, 107.0, 111.0, 116.0, 122.0],
                "BBB": [100.0, 100.5, 101.0, 101.3, 101.6, 101.8, 102.0, 102.2],
            },
            index=date_index,
        )
        regime_close_ser = pd.Series(
            [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0],
            index=date_index,
        )

        annualized_slope_df, r2_df, vol63_df, *_ = compute_clenow_top10_vol63_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=regime_close_ser,
            config=config_obj,
        )

        decision_date_ts = pd.Timestamp(date_index[-1])
        # *** CRITICAL*** This manual check uses the same trailing decision
        # close window as the production Clenow signal.
        log_price_vec = np.log(price_close_df["AAA"].iloc[-5:].to_numpy(dtype=float))
        time_vec = np.arange(5, dtype=float)
        centered_time_vec = time_vec - float(time_vec.mean())
        time_ss_float = float(np.dot(centered_time_vec, centered_time_vec))
        slope_float = float(centered_time_vec @ log_price_vec / time_ss_float)
        expected_annualized_slope_float = float(np.exp(slope_float * 252.0) - 1.0)
        centered_log_price_vec = log_price_vec - float(log_price_vec.mean())
        expected_r2_float = float(
            (slope_float * slope_float * time_ss_float)
            / np.dot(centered_log_price_vec, centered_log_price_vec)
        )
        expected_vol_float = float(price_close_df["AAA"].pct_change().iloc[-3:].std() * np.sqrt(252.0))

        self.assertAlmostEqual(
            float(annualized_slope_df.loc[decision_date_ts, "AAA"]),
            expected_annualized_slope_float,
        )
        self.assertAlmostEqual(float(r2_df.loc[decision_date_ts, "AAA"]), expected_r2_float)
        self.assertAlmostEqual(float(vol63_df.loc[decision_date_ts, "AAA"]), expected_vol_float)

    def test_get_selected_symbol_list_uses_top_scores_and_filters(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "OUT": [0],
            },
            index=[strategy_obj.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.final_score_field_str): 4.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("AAA", "gap_pass_bool"): True,
                ("BBB", strategy_obj.final_score_field_str): 5.0,
                ("BBB", "stock_trend_pass_bool"): True,
                ("BBB", "gap_pass_bool"): True,
                ("CCC", strategy_obj.final_score_field_str): 6.0,
                ("CCC", "stock_trend_pass_bool"): False,
                ("CCC", "gap_pass_bool"): True,
                ("DDD", strategy_obj.final_score_field_str): -1.0,
                ("DDD", "stock_trend_pass_bool"): True,
                ("DDD", "gap_pass_bool"): True,
                ("OUT", strategy_obj.final_score_field_str): 99.0,
                ("OUT", "stock_trend_pass_bool"): True,
                ("OUT", "gap_pass_bool"): True,
            }
        )

        selected_symbol_list = strategy_obj.get_selected_symbol_list(close_row_ser=close_row_ser)

        self.assertEqual(selected_symbol_list, ["BBB", "AAA"])

    def test_regime_failure_blocks_new_buys_but_sells_failed_holdings(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.current_bar = pd.Timestamp("2024-01-10")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.add_transaction(8, strategy_obj.previous_bar, "CCC", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["AAA"] = 7
        strategy_obj.current_trade_map["CCC"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.final_score_field_str): 5.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("AAA", "gap_pass_bool"): True,
                ("BBB", strategy_obj.final_score_field_str): 4.0,
                ("BBB", "stock_trend_pass_bool"): True,
                ("BBB", "gap_pass_bool"): True,
                ("CCC", strategy_obj.final_score_field_str): 1.0,
                ("CCC", "stock_trend_pass_bool"): False,
                ("CCC", "gap_pass_bool"): True,
                ("SPY", "regime_pass_bool"): False,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["CCC"])
        self.assertTrue(isinstance(order_list[0], MarketOrder))
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[0].unit, "shares")
        self.assertTrue(order_list[0].target)
        self.assertEqual(order_list[0].trade_id, 8)

    def test_regime_pass_allows_top_selection_target_orders(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-09")
        strategy_obj.current_bar = pd.Timestamp("2024-01-10")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.final_score_field_str): 5.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("AAA", "gap_pass_bool"): True,
                ("BBB", strategy_obj.final_score_field_str): 4.0,
                ("BBB", "stock_trend_pass_bool"): True,
                ("BBB", "gap_pass_bool"): True,
                ("CCC", strategy_obj.final_score_field_str): 3.0,
                ("CCC", "stock_trend_pass_bool"): True,
                ("CCC", "gap_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "BBB"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertTrue(all(order_obj.target for order_obj in order_list))
        self.assertTrue(all(order_obj.unit == "percent" for order_obj in order_list))
        self.assertTrue(all(float(order_obj.amount) == 0.5 for order_obj in order_list))


if __name__ == "__main__":
    unittest.main()
