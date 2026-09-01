import unittest
from inspect import signature

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from alpha.engine.strategy import Strategy
from strategies.momentum.strategy_mo_spy_adaptive_momentum_regime import (
    DEFAULT_CONFIG,
    SIGNAL_NAMESPACE_STR,
    SpyAdaptiveMomentumRegimeConfig,
    SpyAdaptiveMomentumRegimeStrategy,
    compute_spy_adaptive_momentum_signal_df,
    compute_strict_trailing_percentile_ser,
    compute_weak_trailing_percentile_ser,
    run_variant,
)


class SpyAdaptiveMomentumRegimeTests(unittest.TestCase):
    def make_config(self, **override_dict) -> SpyAdaptiveMomentumRegimeConfig:
        base_dict = {
            "percentile_lookback_int": 3,
            "fast_lookback_int": 2,
            "slow_lookback_int": 5,
            "percentile_power_float": 2.0,
            "price_filter_lookback_int": 2,
            "history_start_date_str": "2023-01-01",
            "backtest_start_date_str": "2024-01-01",
            "slippage_float": 0.0,
            "commission_per_share_float": 0.0,
            "commission_minimum_float": 0.0,
        }
        base_dict.update(override_dict)
        return SpyAdaptiveMomentumRegimeConfig(**base_dict)

    def make_strategy(self, **override_dict) -> SpyAdaptiveMomentumRegimeStrategy:
        config_obj = self.make_config(**override_dict)
        return SpyAdaptiveMomentumRegimeStrategy(
            name="SpyAdaptiveMomentumRegimeTest",
            benchmarks=[config_obj.benchmark_symbol_str],
            config=config_obj,
        )

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=8, freq="B")
        execution_close_vec = np.array(
            [100.0, 101.0, 99.0, 98.0, 103.0, 104.0, 105.0, 106.0],
            dtype=float,
        )
        signal_close_vec = np.array(
            [100.0, 102.0, 96.0, 94.0, 101.0, 104.0, 107.0, 109.0],
            dtype=float,
        )
        benchmark_close_vec = np.linspace(4_000.0, 4_070.0, len(date_index))
        pricing_data_df = pd.DataFrame(
            {
                ("SPY", "Open"): execution_close_vec - 0.5,
                ("SPY", "High"): execution_close_vec + 1.0,
                ("SPY", "Low"): execution_close_vec - 1.0,
                ("SPY", "Close"): execution_close_vec,
                ("SPY_TR_SIGNAL", "Close"): signal_close_vec,
                ("$SPX", "Open"): benchmark_close_vec - 5.0,
                ("$SPX", "High"): benchmark_close_vec + 5.0,
                ("$SPX", "Low"): benchmark_close_vec - 5.0,
                ("$SPX", "Close"): benchmark_close_vec,
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(
            pricing_data_df.columns
        )
        return pricing_data_df

    def test_strict_percentile_maps_ties_to_zero_and_new_maximum_to_one(self):
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")
        severity_ser = pd.Series(
            [0.0, 0.0, 0.0, 0.5, 0.5],
            index=date_index,
            dtype=float,
        )
        percentile_ser = compute_strict_trailing_percentile_ser(
            severity_ser=severity_ser,
            lookback_int=3,
        )

        self.assertTrue(np.isnan(percentile_ser.iloc[1]))
        self.assertEqual(float(percentile_ser.iloc[2]), 0.0)
        self.assertEqual(float(percentile_ser.iloc[3]), 1.0)
        self.assertEqual(float(percentile_ser.iloc[4]), 0.5)

        weak_percentile_ser = compute_weak_trailing_percentile_ser(
            severity_ser=severity_ser,
            lookback_int=3,
        )
        self.assertEqual(float(weak_percentile_ser.iloc[2]), 1.0)
        self.assertEqual(float(weak_percentile_ser.iloc[3]), 1.0)
        self.assertEqual(float(weak_percentile_ser.iloc[4]), 1.0)

    def test_signal_formula_matches_explicit_recursive_calculation(self):
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")
        price_close_ser = pd.Series(
            [100.0, 100.0, 90.0, 95.0, 101.0],
            index=date_index,
            dtype=float,
        )
        signal_df = compute_spy_adaptive_momentum_signal_df(
            signal_price_close_ser=price_close_ser,
            percentile_lookback_int=3,
            fast_lookback_int=2,
            slow_lookback_int=5,
            percentile_power_float=2.0,
            price_filter_lookback_int=2,
        )

        slow_alpha_float = 2.0 / 6.0
        fast_alpha_float = 2.0 / 3.0
        expected_ama_1_float = 100.0
        expected_ama_2_float = 100.0
        expected_ama_3_float = fast_alpha_float * 90.0 + (1.0 - fast_alpha_float) * expected_ama_2_float
        expected_alpha_4_float = 0.25 * fast_alpha_float + 0.75 * slow_alpha_float
        expected_ama_4_float = expected_alpha_4_float * 95.0 + (1.0 - expected_alpha_4_float) * expected_ama_3_float

        self.assertAlmostEqual(
            float(signal_df.iloc[0]["adaptive_alpha_ser"]),
            slow_alpha_float,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[2]["drawdown_percentile_ser"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[3]["drawdown_percentile_ser"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[2]["adaptive_moving_average_ser"]),
            expected_ama_3_float,
        )
        self.assertAlmostEqual(
            float(signal_df.iloc[3]["adaptive_moving_average_ser"]),
            expected_ama_4_float,
        )
        self.assertEqual(float(signal_df.iloc[-1]["target_weight_ser"]), 1.0)

    def test_signal_prefix_is_unchanged_by_future_prices(self):
        date_index = pd.date_range("2024-01-02", periods=12, freq="B")
        prefix_price_ser = pd.Series(
            [100.0, 101.0, 99.0, 95.0, 96.0, 98.0, 102.0, 103.0, 97.0, 94.0],
            index=date_index[:10],
            dtype=float,
        )
        extended_price_ser = pd.concat(
            [
                prefix_price_ser,
                pd.Series([1_000.0, 1.0], index=date_index[10:], dtype=float),
            ]
        )

        prefix_signal_df = compute_spy_adaptive_momentum_signal_df(
            prefix_price_ser,
            percentile_lookback_int=3,
            fast_lookback_int=2,
            slow_lookback_int=5,
            price_filter_lookback_int=2,
        )
        extended_signal_df = compute_spy_adaptive_momentum_signal_df(
            extended_price_ser,
            percentile_lookback_int=3,
            fast_lookback_int=2,
            slow_lookback_int=5,
            price_filter_lookback_int=2,
        )

        pd.testing.assert_frame_equal(
            prefix_signal_df,
            extended_signal_df.loc[prefix_signal_df.index],
        )

    def test_signal_allows_pre_inception_nan_prefix(self):
        pre_inception_index = pd.date_range("1986-01-02", periods=5, freq="B")
        signal_df = compute_spy_adaptive_momentum_signal_df(
            pd.Series(np.nan, index=pre_inception_index),
            percentile_lookback_int=3,
            fast_lookback_int=2,
            slow_lookback_int=4,
            price_filter_lookback_int=2,
        )

        self.assertTrue(signal_df.index.equals(pre_inception_index))
        self.assertTrue(signal_df["target_weight_ser"].isna().all())

    def test_compute_signals_uses_total_return_signal_namespace(self):
        pricing_data_df = self.make_pricing_data_df()
        strategy_obj = self.make_strategy()
        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        self.assertIn(
            (SIGNAL_NAMESPACE_STR, "target_weight_ser"),
            signal_data_df.columns,
        )
        self.assertAlmostEqual(
            float(
                strategy_obj.regime_signal_df.iloc[2][
                    "signal_price_close_ser"
                ]
            ),
            96.0,
        )
        strategy_obj.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_sizes_from_previous_close_and_fills_at_current_open(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-03")
        strategy_obj.current_bar = pd.Timestamp("2024-01-04")
        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 1.0,
                ("SPY", "Close"): 80.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SPY": 125.0}, dtype=float)

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            open_price_ser,
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 1)
        order_obj = order_list[0]
        self.assertIsInstance(order_obj, MarketOrder)
        self.assertEqual(order_obj.asset, "SPY")
        self.assertEqual(order_obj.amount, 1250)
        self.assertTrue(order_obj.target)
        self.assertEqual(order_obj.trade_id, 1)

    def test_iterate_exits_with_existing_trade_id(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-03")
        strategy_obj.current_bar = pd.Timestamp("2024-01-04")
        strategy_obj.trade_id_int = 4
        strategy_obj.current_trade_id_int = 4
        strategy_obj.add_transaction(
            4,
            strategy_obj.previous_bar,
            "SPY",
            500,
            100.0,
            50_000.0,
            1,
            0.0,
        )
        close_row_ser = pd.Series(
            {(SIGNAL_NAMESPACE_STR, "target_weight_ser"): 0.0}
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"SPY": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 1)
        self.assertEqual(order_list[0].amount, 0)
        self.assertEqual(order_list[0].trade_id, 4)
        self.assertEqual(strategy_obj.current_trade_id_int, -1)

    def test_iterate_does_not_resize_existing_risk_on_position(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-03")
        strategy_obj.current_bar = pd.Timestamp("2024-01-04")
        strategy_obj.current_trade_id_int = 2
        strategy_obj.add_transaction(
            2,
            strategy_obj.previous_bar,
            "SPY",
            500,
            100.0,
            50_000.0,
            1,
            0.0,
        )
        close_row_ser = pd.Series(
            {(SIGNAL_NAMESPACE_STR, "target_weight_ser"): 1.0}
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"SPY": 110.0}, dtype=float),
        )

        self.assertEqual(strategy_obj.get_orders(), [])

    def test_run_daily_smoke_generates_summary(self):
        pricing_data_df = self.make_pricing_data_df()
        strategy_obj = self.make_strategy()
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=pricing_data_df.index[2:],
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
        )

        self.assertIsNotNone(strategy_obj.summary)
        self.assertIn("Strategy", strategy_obj.summary.columns)
        self.assertGreater(len(strategy_obj.results), 0)

    def test_default_costs_match_engine_defaults(self):
        engine_signature_obj = signature(Strategy.__init__)

        self.assertEqual(
            DEFAULT_CONFIG.slippage_float,
            engine_signature_obj.parameters["slippage"].default,
        )
        self.assertEqual(
            DEFAULT_CONFIG.commission_per_share_float,
            engine_signature_obj.parameters["commission_per_share"].default,
        )
        self.assertEqual(
            DEFAULT_CONFIG.commission_minimum_float,
            engine_signature_obj.parameters["commission_minimum"].default,
        )

    def test_run_variant_accepts_injected_pricing_data(self):
        strategy_obj = run_variant(
            show_display_bool=False,
            save_results_bool=False,
            pricing_data_df=self.make_pricing_data_df(),
            config_obj=self.make_config(),
        )

        self.assertIsInstance(strategy_obj, SpyAdaptiveMomentumRegimeStrategy)
        self.assertIsNotNone(strategy_obj.summary)

    def test_config_rejects_noncausal_or_invalid_parameter_order(self):
        with self.assertRaisesRegex(
            ValueError,
            "slow_lookback_int must exceed fast_lookback_int",
        ):
            self.make_config(fast_lookback_int=5, slow_lookback_int=5)
        with self.assertRaisesRegex(
            ValueError,
            "history_start_date_str must be earlier",
        ):
            self.make_config(history_start_date_str="2024-01-01")


if __name__ == "__main__":
    unittest.main()
