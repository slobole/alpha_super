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
from strategies.strategy_mo_nasdaq_katsanos_rotation import (
    NasdaqKatsanosRotationStrategy,
    compute_capped_score_weight_ser,
    compute_ehlers_zero_lag_ma_ser,
    map_periodic_decision_dates_to_execution_date_df,
)


def manual_zero_lag_ma_ser(close_ser: pd.Series, window_int: int) -> pd.Series:
    alpha_float = 2.0 / float(window_int + 1)
    close_vec = close_ser.to_numpy(dtype=float)
    ema_vec = np.full(len(close_vec), np.nan, dtype=float)
    zma_vec = np.full(len(close_vec), np.nan, dtype=float)
    gain_float_vec = np.round(np.arange(-5.0, 5.0 + 0.1, 0.1), 10)

    for bar_int, close_float in enumerate(close_vec):
        if bar_int == 0:
            ema_vec[bar_int] = close_float
            zma_vec[bar_int] = close_float
            continue

        ema_float = alpha_float * close_float + (1.0 - alpha_float) * ema_vec[bar_int - 1]
        ema_vec[bar_int] = ema_float

        prior_zma_float = zma_vec[bar_int - 1]
        least_error_float = np.inf
        best_zma_float = np.nan
        for gain_float in gain_float_vec:
            zma_candidate_float = (
                alpha_float * (ema_float + gain_float * (close_float - prior_zma_float))
                + (1.0 - alpha_float) * prior_zma_float
            )
            error_float = abs(close_float - zma_candidate_float)
            if error_float < least_error_float:
                least_error_float = error_float
                best_zma_float = zma_candidate_float

        zma_vec[bar_int] = best_zma_float

    return pd.Series(zma_vec, index=close_ser.index, dtype=float, name=f"zma_{window_int}_ser")


class NasdaqKatsanosRotationStrategyTests(unittest.TestCase):
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

    def make_strategy(self, **kwargs) -> NasdaqKatsanosRotationStrategy:
        base_kwargs = dict(
            name="NasdaqKatsanosRotationTest",
            benchmarks=["QQQ"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="QQQ",
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=3,
            roc_lookback_day_int=20,
            rebalance_interval_day_int=10,
            zma_window_int=30,
            exclude_top_n_int=1,
            max_asset_weight_float=0.5,
        )
        base_kwargs.update(kwargs)
        return NasdaqKatsanosRotationStrategy(**base_kwargs)

    def make_pricing_data_df(self, periods_int: int = 180) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=periods_int, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 50.0 + 0.60 * step_vec + 0.80 * np.sin(step_vec * 0.05)
        bbb_close_vec = 60.0 + 0.55 * step_vec + 0.70 * np.cos(step_vec * 0.04)
        ccc_close_vec = 40.0 + 0.40 * step_vec + 0.50 * np.sin(step_vec * 0.03)
        ddd_close_vec = 30.0 + 0.28 * step_vec + 0.35 * np.cos(step_vec * 0.06)
        eee_close_vec = 25.0 + 0.18 * step_vec + 0.25 * np.sin(step_vec * 0.02)
        fff_close_vec = 80.0 - 0.08 * step_vec + 0.20 * np.cos(step_vec * 0.03)
        qqq_close_vec = 100.0 + 0.22 * step_vec + 0.50 * np.sin(step_vec * 0.02)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.4,
                ("AAA", "High"): aaa_close_vec + 0.8,
                ("AAA", "Low"): aaa_close_vec - 0.8,
                ("AAA", "Close"): aaa_close_vec,
                ("BBB", "Open"): bbb_close_vec - 0.4,
                ("BBB", "High"): bbb_close_vec + 0.8,
                ("BBB", "Low"): bbb_close_vec - 0.8,
                ("BBB", "Close"): bbb_close_vec,
                ("CCC", "Open"): ccc_close_vec - 0.4,
                ("CCC", "High"): ccc_close_vec + 0.8,
                ("CCC", "Low"): ccc_close_vec - 0.8,
                ("CCC", "Close"): ccc_close_vec,
                ("DDD", "Open"): ddd_close_vec - 0.4,
                ("DDD", "High"): ddd_close_vec + 0.8,
                ("DDD", "Low"): ddd_close_vec - 0.8,
                ("DDD", "Close"): ddd_close_vec,
                ("EEE", "Open"): eee_close_vec - 0.4,
                ("EEE", "High"): eee_close_vec + 0.8,
                ("EEE", "Low"): eee_close_vec - 0.8,
                ("EEE", "Close"): eee_close_vec,
                ("FFF", "Open"): fff_close_vec - 0.4,
                ("FFF", "High"): fff_close_vec + 0.8,
                ("FFF", "Low"): fff_close_vec - 0.8,
                ("FFF", "Close"): fff_close_vec,
                ("QQQ", "Open"): qqq_close_vec - 0.4,
                ("QQQ", "High"): qqq_close_vec + 0.8,
                ("QQQ", "Low"): qqq_close_vec - 0.8,
                ("QQQ", "Close"): qqq_close_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_map_periodic_decision_dates_uses_next_tradable_open(self):
        execution_index = pd.to_datetime(
            [
                "2024-02-12",
                "2024-02-13",
                "2024-02-14",
                "2024-02-15",
                "2024-02-16",
                "2024-02-20",
            ]
        )

        rebalance_schedule_df = map_periodic_decision_dates_to_execution_date_df(
            execution_index=execution_index,
            interval_day_int=5,
        )

        self.assertEqual(
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-02-20"), "decision_date_ts"]),
            pd.Timestamp("2024-02-16"),
        )

    def test_zero_lag_ma_matches_manual_formula(self):
        close_ser = pd.Series(
            [100.0, 101.0, 103.0, 104.0, 102.0, 105.0, 107.0, 106.0],
            index=pd.date_range("2024-01-02", periods=8, freq="B"),
            dtype=float,
        )

        actual_zma_ser = compute_ehlers_zero_lag_ma_ser(close_ser=close_ser, window_int=5)
        expected_zma_ser = manual_zero_lag_ma_ser(close_ser=close_ser, window_int=5)

        pd.testing.assert_series_equal(actual_zma_ser, expected_zma_ser)

    def test_compute_capped_score_weight_ser_caps_and_redistributes(self):
        score_ser = pd.Series({"CCC": 0.70, "DDD": 0.10, "EEE": 0.10}, dtype=float)

        target_weight_ser = compute_capped_score_weight_ser(score_ser=score_ser, max_weight_float=0.5)

        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.loc["DDD"]), 0.25)
        self.assertAlmostEqual(float(target_weight_ser.loc["EEE"]), 0.25)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df(periods_int=120)

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "roc_20_day_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", "zma_30_ser"), signal_data_df.columns)
        self.assertIn(("QQQ", "bear_market_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_get_target_weight_ser_excludes_top_scores_and_caps_weights(self):
        strategy = self.make_strategy(
            max_positions_int=3,
            exclude_top_n_int=2,
            max_asset_weight_float=0.5,
            roc_lookback_day_int=20,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1], "EEE": [1], "FFF": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "roc_20_day_ser"): 0.90,
                ("BBB", "roc_20_day_ser"): 0.80,
                ("CCC", "roc_20_day_ser"): 0.70,
                ("DDD", "roc_20_day_ser"): 0.10,
                ("EEE", "roc_20_day_ser"): 0.10,
                ("FFF", "roc_20_day_ser"): -0.05,
                ("QQQ", "bear_market_bool"): False,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["CCC", "DDD", "EEE"])
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.loc["DDD"]), 0.25)
        self.assertAlmostEqual(float(target_weight_ser.loc["EEE"]), 0.25)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_get_target_weight_ser_leaves_residual_cash_when_too_few_positive_scores(self):
        strategy = self.make_strategy(
            max_positions_int=3,
            exclude_top_n_int=2,
            max_asset_weight_float=0.5,
            roc_lookback_day_int=20,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1], "DDD": [1]},
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "roc_20_day_ser"): 0.90,
                ("BBB", "roc_20_day_ser"): 0.80,
                ("CCC", "roc_20_day_ser"): 0.20,
                ("DDD", "roc_20_day_ser"): -0.10,
                ("QQQ", "bear_market_bool"): False,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["CCC"])
        self.assertAlmostEqual(float(target_weight_ser.loc["CCC"]), 0.5)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 0.5)

    def test_get_target_weight_ser_returns_empty_in_bear_market(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "roc_20_day_ser"): 0.20,
                ("BBB", "roc_20_day_ser"): 0.10,
                ("QQQ", "bear_market_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(len(target_weight_ser), 0)

    def test_iterate_submits_liquidations_before_new_buys_and_preserves_trade_ids(self):
        strategy = self.make_strategy(
            max_positions_int=1,
            exclude_top_n_int=0,
            max_asset_weight_float=1.0,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-11")
        strategy.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[strategy.previous_bar],
        )
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(8, strategy.previous_bar, "BBB", 5, 100.0, 500.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 7
        strategy.current_trade_map["BBB"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "roc_20_day_ser"): -0.10,
                ("BBB", "roc_20_day_ser"): 0.10,
                ("CCC", "roc_20_day_ser"): 0.50,
                ("QQQ", "bear_market_bool"): False,
            }
        )
        open_price_ser = pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 3)
        self.assertEqual(order_list[0].asset, "AAA")
        self.assertEqual(order_list[1].asset, "BBB")
        self.assertEqual(order_list[2].asset, "CCC")
        self.assertEqual(order_list[0].trade_id, 7)
        self.assertEqual(order_list[1].trade_id, 8)
        self.assertEqual(order_list[2].trade_id, 1)
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[1].amount, 0)
        self.assertAlmostEqual(float(order_list[2].amount), 1.0)
        self.assertEqual(strategy.current_trade_map["CCC"], 1)

    def test_iterate_skips_non_rebalance_dates(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-08")
        strategy.current_bar = pd.Timestamp("2024-03-12")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "roc_20_day_ser"): 0.20,
                ("QQQ", "bear_market_bool"): False,
            }
        )
        strategy.iterate(
            pd.DataFrame(index=[strategy.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0}, dtype=float),
        )

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df(periods_int=180).copy()
        late_regime_break_index = pricing_data_df.index[-25:]
        late_qqq_close_vec = np.linspace(130.0, 110.0, len(late_regime_break_index))
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "Open")] = late_qqq_close_vec - 0.4
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "High")] = late_qqq_close_vec + 0.8
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "Low")] = late_qqq_close_vec - 0.8
        pricing_data_df.loc[late_regime_break_index, ("QQQ", "Close")] = late_qqq_close_vec

        rebalance_schedule_df = map_periodic_decision_dates_to_execution_date_df(
            execution_index=pricing_data_df.index,
            interval_day_int=10,
        )
        strategy = self.make_strategy(
            rebalance_schedule_df=rebalance_schedule_df,
            max_positions_int=3,
            roc_lookback_day_int=20,
            rebalance_interval_day_int=10,
            zma_window_int=30,
            exclude_top_n_int=1,
            max_asset_weight_float=0.5,
        )
        strategy.universe_df = pd.DataFrame(
            {"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1, "EEE": 1, "FFF": 1},
            index=pricing_data_df.index,
        )

        calendar_idx = pricing_data_df.index[pricing_data_df.index >= pricing_data_df.index[40]]
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
                audit_override_bool=None,
                audit_sample_size_int=5,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertGreater(len(strategy.results), 0)
        self.assertGreater(len(strategy.get_transactions()), 0)


if __name__ == "__main__":
    unittest.main()
