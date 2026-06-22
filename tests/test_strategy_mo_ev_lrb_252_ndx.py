import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_ev_lrb_252_ndx import (
    DEFAULT_CONFIG,
    EvLrb252NdxConfig,
    EvLrb252NdxStrategy,
    compute_ev_lrb_indicator_tables,
    compute_ev_lrb_signal_tables,
)


class EvLrb252NdxTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], object]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_strategy(self, **kwargs) -> EvLrb252NdxStrategy:
        base_kwargs = dict(
            name="EvLrb252NdxTest",
            benchmarks=[DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=DEFAULT_CONFIG.regime_symbol_str,
            capital_base=DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            index_trend_window_int=DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=2,
            pn_window_int=DEFAULT_CONFIG.pn_window_int,
            require_positive_ev_bool=False,
        )
        base_kwargs.update(kwargs)
        return EvLrb252NdxStrategy(**base_kwargs)

    def test_default_config_preserves_ndx_research_shape(self):
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(DEFAULT_CONFIG.pn_window_int, 252)
        self.assertEqual(DEFAULT_CONFIG.max_positions_int, 10)
        self.assertFalse(DEFAULT_CONFIG.require_positive_ev_bool)

    def test_signal_table_matches_lrb_and_ev_formulas_at_month_end_close(self):
        date_index = pd.bdate_range("2024-01-01", "2024-02-29")
        aaa_price_vec = np.array([100.0 + 0.1 * day_int for day_int in range(len(date_index))])
        aaa_price_vec[-4:] = np.array([100.0, 110.0, 99.0, 108.9])
        price_close_df = pd.DataFrame(
            {
                "AAA": aaa_price_vec,
                "BBB": np.array([50.0 + 0.2 * day_int for day_int in range(len(date_index))]),
            },
            index=date_index,
        )
        regime_close_ser = pd.Series(
            np.array([200.0 + float(day_int) for day_int in range(len(date_index))]),
            index=date_index,
        )
        config_obj = EvLrb252NdxConfig(
            pn_window_int=3,
            index_trend_window_int=2,
            stock_trend_window_int=2,
        )

        (
            _monthly_decision_close_df,
            positive_return_sum_decision_df,
            negative_return_sum_decision_df,
            lrb_decision_df,
            ev_decision_df,
            _stock_trend_pass_df,
            _regime_sma_ser,
            regime_pass_ser,
            ev_score_df,
        ) = compute_ev_lrb_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=regime_close_ser,
            config=config_obj,
        )

        decision_date_ts = pd.Timestamp("2024-02-29")
        # *** CRITICAL*** Expected returns use the last 3 close-to-close log
        # returns ending exactly at the month-end decision close.
        log_return_vec = np.log(np.array([110.0 / 100.0, 99.0 / 110.0, 108.9 / 99.0]))
        expected_positive_float = float(np.maximum(log_return_vec, 0.0).sum())
        expected_negative_float = float(np.maximum(-log_return_vec, 0.0).sum())
        expected_lrb_float = expected_positive_float / expected_negative_float
        expected_ev_float = (
            (expected_positive_float - expected_negative_float)
            / (expected_positive_float + expected_negative_float)
        )

        self.assertAlmostEqual(
            float(positive_return_sum_decision_df.loc[decision_date_ts, "AAA"]),
            expected_positive_float,
        )
        self.assertAlmostEqual(
            float(negative_return_sum_decision_df.loc[decision_date_ts, "AAA"]),
            expected_negative_float,
        )
        self.assertAlmostEqual(float(lrb_decision_df.loc[decision_date_ts, "AAA"]), expected_lrb_float)
        self.assertAlmostEqual(float(ev_decision_df.loc[decision_date_ts, "AAA"]), expected_ev_float)
        self.assertAlmostEqual(float(ev_score_df.loc[decision_date_ts, "AAA"]), expected_ev_float)
        self.assertTrue(bool(regime_pass_ser.loc[decision_date_ts]))

    def test_decision_close_indicator_is_invariant_to_future_prices(self):
        date_index = pd.bdate_range("2024-01-01", "2024-02-29")
        future_date_index = pd.bdate_range("2024-03-01", "2024-03-08")
        price_close_df = pd.DataFrame(
            {
                "AAA": np.array([100.0 + 0.5 * day_int for day_int in range(len(date_index))]),
                "BBB": np.array([80.0 + 0.1 * day_int for day_int in range(len(date_index))]),
            },
            index=date_index,
        )
        future_price_close_df = pd.DataFrame(
            {
                "AAA": np.array([1000.0, 10.0, 1000.0, 10.0, 1000.0, 10.0]),
                "BBB": np.array([5.0, 500.0, 5.0, 500.0, 5.0, 500.0]),
            },
            index=future_date_index,
        )
        regime_close_ser = pd.Series(
            np.array([200.0 + float(day_int) for day_int in range(len(date_index))]),
            index=date_index,
        )
        future_regime_close_ser = pd.Series(
            np.array([500.0, 100.0, 500.0, 100.0, 500.0, 100.0]),
            index=future_date_index,
        )
        config_obj = EvLrb252NdxConfig(
            pn_window_int=5,
            index_trend_window_int=2,
            stock_trend_window_int=2,
        )

        base_result_tuple = compute_ev_lrb_signal_tables(
            price_close_df=price_close_df,
            regime_close_ser=regime_close_ser,
            config=config_obj,
        )
        extended_result_tuple = compute_ev_lrb_signal_tables(
            price_close_df=pd.concat([price_close_df, future_price_close_df]),
            regime_close_ser=pd.concat([regime_close_ser, future_regime_close_ser]),
            config=config_obj,
        )

        decision_date_ts = pd.Timestamp("2024-02-29")
        base_ev_score_df = base_result_tuple[-1]
        extended_ev_score_df = extended_result_tuple[-1]
        # *** CRITICAL*** Future March prices must not alter the signal known
        # after the February month-end close.
        pd.testing.assert_series_equal(
            base_ev_score_df.loc[decision_date_ts],
            extended_ev_score_df.loc[decision_date_ts],
            check_names=False,
        )

    def test_lrb_is_nan_when_negative_return_sum_is_zero_but_ev_is_one(self):
        date_index = pd.bdate_range("2024-01-01", periods=4)
        price_close_df = pd.DataFrame(
            {"AAA": [100.0, 101.0, 102.0, 103.0]},
            index=date_index,
        )

        (
            positive_return_sum_df,
            negative_return_sum_df,
            lrb_df,
            ev_df,
        ) = compute_ev_lrb_indicator_tables(
            price_close_df=price_close_df,
            window_int=3,
        )

        latest_date_ts = date_index[-1]
        self.assertGreater(float(positive_return_sum_df.loc[latest_date_ts, "AAA"]), 0.0)
        self.assertAlmostEqual(float(negative_return_sum_df.loc[latest_date_ts, "AAA"]), 0.0)
        self.assertTrue(pd.isna(lrb_df.loc[latest_date_ts, "AAA"]))
        self.assertAlmostEqual(float(ev_df.loc[latest_date_ts, "AAA"]), 1.0)

    def test_target_selection_ranks_by_ev_and_uses_lrb_only_as_audit_field(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "OUT": [0],
            },
            index=[strategy_obj.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.ev_score_field_str): 0.30,
                ("AAA", strategy_obj.lrb_field_str): 0.01,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", strategy_obj.ev_score_field_str): 0.20,
                ("BBB", strategy_obj.lrb_field_str): 999.0,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", strategy_obj.ev_score_field_str): 0.10,
                ("CCC", strategy_obj.lrb_field_str): 10.0,
                ("CCC", "stock_trend_pass_bool"): True,
                ("OUT", strategy_obj.ev_score_field_str): 1.00,
                ("OUT", strategy_obj.lrb_field_str): 9999.0,
                ("OUT", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(dtype=float), 0.50))

    def test_positive_ev_gate_leaves_unused_slots_in_cash(self):
        strategy_obj = self.make_strategy(
            max_positions_int=3,
            require_positive_ev_bool=True,
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
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
                ("AAA", strategy_obj.ev_score_field_str): 0.10,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", strategy_obj.ev_score_field_str): 0.00,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", strategy_obj.ev_score_field_str): -0.10,
                ("CCC", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA"])
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
