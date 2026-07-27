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
from strategies.momentum.adaptive_moving_average_factor import (
    TARGET_WEIGHT_FIELD_STR,
    AdaptiveMovingAverageFactorConfig,
    AdaptiveMovingAverageFactorSignalBundle,
    AdaptiveMovingAverageFactorStrategy,
    align_pit_universe_with_unavailable_prefix_df,
    assign_stable_quintile_ser,
    build_adaptive_moving_average_factor_signal_bundle,
    build_monthly_sma_ratio_by_lookback_dict,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_amaf_nasdaq100 import (
    DEFAULT_CONFIG as NASDAQ100_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_amaf_russell1000 import (
    DEFAULT_CONFIG as RUSSELL1000_DEFAULT_CONFIG,
)


class AdaptiveMovingAverageFactorTests(unittest.TestCase):
    def make_config(self) -> AdaptiveMovingAverageFactorConfig:
        return replace(
            RUSSELL1000_DEFAULT_CONFIG,
            history_start_date_str="2023-01-01",
            backtest_start_date_str="2023-03-01",
            sma_lookback_tuple=(2, 3),
            smoothing_month_int=2,
            min_eligible_count_int=5,
            capital_base_float=10_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def make_price_fixture(
        self,
        month_count_int: int = 8,
        symbol_count_int: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        date_index = pd.bdate_range(
            "2023-01-02",
            periods=month_count_int * 23,
        )
        step_vec = np.arange(len(date_index), dtype=float)
        price_by_symbol_dict: dict[str, np.ndarray] = {}
        for symbol_int in range(1, symbol_count_int + 1):
            symbol_str = f"S{symbol_int:03d}"
            daily_return_vec = (
                0.0001
                + symbol_int * 0.00003
                + 0.0002
                * np.sin(step_vec * (0.02 + symbol_int * 0.001))
            )
            price_by_symbol_dict[symbol_str] = (
                (10.0 + symbol_int)
                * np.cumprod(1.0 + daily_return_vec)
            )
        price_close_df = pd.DataFrame(
            price_by_symbol_dict,
            index=date_index,
            dtype=float,
        )
        raw_close_df = price_close_df.copy()
        universe_df = pd.DataFrame(
            1,
            index=date_index,
            columns=price_close_df.columns,
            dtype=int,
        )
        decision_date_index = pd.DatetimeIndex(
            pd.Series(
                date_index,
                index=date_index.to_period("M"),
            )
            .groupby(level=0)
            .max()
            .to_numpy()
        )
        return (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        )

    def make_signal_bundle(self) -> AdaptiveMovingAverageFactorSignalBundle:
        (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        ) = self.make_price_fixture()
        return build_adaptive_moving_average_factor_signal_bundle(
            price_close_df=price_close_df,
            raw_close_df=raw_close_df,
            universe_df=universe_df,
            decision_date_index=decision_date_index,
            config_obj=self.make_config(),
        )

    def test_default_variants_freeze_the_requested_universes(self):
        self.assertEqual(
            RUSSELL1000_DEFAULT_CONFIG.strategy_name_str,
            "strategy_mo_amaf_russell1000",
        )
        self.assertEqual(
            RUSSELL1000_DEFAULT_CONFIG.indexname_str,
            "Russell 1000",
        )
        self.assertEqual(
            RUSSELL1000_DEFAULT_CONFIG.source_panel_indexname_str,
            "Russell 3000",
        )
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.min_eligible_count_int, 100)
        self.assertEqual(
            NASDAQ100_DEFAULT_CONFIG.strategy_name_str,
            "strategy_mo_amaf_nasdaq100",
        )
        self.assertEqual(
            NASDAQ100_DEFAULT_CONFIG.indexname_str,
            "Nasdaq 100",
        )
        self.assertEqual(
            NASDAQ100_DEFAULT_CONFIG.source_panel_indexname_str,
            "Russell 3000",
        )
        self.assertEqual(NASDAQ100_DEFAULT_CONFIG.min_eligible_count_int, 50)
        self.assertEqual(
            RUSSELL1000_DEFAULT_CONFIG.sma_lookback_tuple,
            (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000),
        )
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.smoothing_month_int, 12)
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.quintile_count_int, 5)
        self.assertEqual(
            RUSSELL1000_DEFAULT_CONFIG.minimum_raw_close_float,
            5.0,
        )

    def test_sma_ratio_uses_observed_closes_and_includes_close_t(self):
        date_index = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )
        price_close_df = pd.DataFrame(
            {"AAA": [1.0, np.nan, 3.0, 6.0]},
            index=date_index,
        )
        feature_by_lookback_dict = build_monthly_sma_ratio_by_lookback_dict(
            price_close_df=price_close_df,
            decision_date_index=pd.DatetimeIndex([date_index[-1]]),
            sma_lookback_tuple=(3,),
        )
        self.assertAlmostEqual(
            float(feature_by_lookback_dict[3].loc[date_index[-1], "AAA"]),
            (1.0 + 3.0 + 6.0) / 3.0 / 6.0,
        )

    def test_price_history_before_first_pit_row_is_ineligible_not_backfilled(self):
        execution_index = pd.bdate_range("2024-01-02", periods=5)
        universe_df = pd.DataFrame(
            {"AAA": [1, 1, 1]},
            index=execution_index[2:],
            dtype=int,
        )
        aligned_universe_df = (
            align_pit_universe_with_unavailable_prefix_df(
                universe_df=universe_df,
                execution_index=execution_index,
                tradeable_symbol_list=["AAA"],
            )
        )
        self.assertEqual(
            aligned_universe_df["AAA"].tolist(),
            [0, 0, 1, 1, 1],
        )

    def test_stable_quintile_uses_symbol_as_tie_breaker(self):
        forecast_ser = pd.Series(
            1.0,
            index=["K", "A", "J", "B", "I", "C", "H", "D", "G", "E", "F"],
            dtype=float,
        )
        quintile_ser = assign_stable_quintile_ser(
            forecast_ser=forecast_ser,
            quintile_count_int=5,
        )
        self.assertEqual(
            set(quintile_ser.loc[quintile_ser.eq(5)].index),
            {"J", "K"},
        )

    def test_signal_bundle_forms_equal_weight_top_quintile(self):
        signal_bundle_obj = self.make_signal_bundle()
        self.assertGreater(len(signal_bundle_obj.target_weight_df), 0)
        np.testing.assert_allclose(
            signal_bundle_obj.target_weight_df.sum(axis=1).to_numpy(),
            1.0,
        )
        valid_coverage_df = signal_bundle_obj.coverage_df.loc[
            signal_bundle_obj.coverage_df["status_str"].eq("valid_target")
        ]
        self.assertTrue(
            (
                valid_coverage_df["selected_count_int"]
                == 2
            ).all()
        )
        self.assertTrue(
            signal_bundle_obj.coefficient_df.filter(
                like="smoothed_"
            ).notna().any(axis=None)
        )

    def test_regression_smoothing_and_forecast_arithmetic_is_exact(self):
        (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        ) = self.make_price_fixture()
        config_obj = self.make_config()
        signal_bundle_obj = (
            build_adaptive_moving_average_factor_signal_bundle(
                price_close_df=price_close_df,
                raw_close_df=raw_close_df,
                universe_df=universe_df,
                decision_date_index=decision_date_index,
                config_obj=config_obj,
            )
        )
        feature_by_lookback_dict = build_monthly_sma_ratio_by_lookback_dict(
            price_close_df=price_close_df,
            decision_date_index=decision_date_index,
            sma_lookback_tuple=config_obj.sma_lookback_tuple,
        )
        first_target_date_ts = pd.Timestamp(
            signal_bundle_obj.target_weight_df.index[0]
        )
        first_target_index_int = decision_date_index.get_loc(
            first_target_date_ts
        )
        expected_beta_row_list: list[np.ndarray] = []
        for beta_index_int in range(
            first_target_index_int - config_obj.smoothing_month_int + 1,
            first_target_index_int + 1,
        ):
            current_date_ts = pd.Timestamp(
                decision_date_index[beta_index_int]
            )
            prior_date_ts = pd.Timestamp(
                decision_date_index[beta_index_int - 1]
            )
            prior_feature_mat = np.column_stack(
                [
                    feature_by_lookback_dict[lookback_int].loc[
                        prior_date_ts
                    ].to_numpy(dtype=float)
                    for lookback_int in config_obj.sma_lookback_tuple
                ]
            )
            current_return_vec = (
                price_close_df.loc[current_date_ts].to_numpy(dtype=float)
                / price_close_df.loc[prior_date_ts].to_numpy(dtype=float)
                - 1.0
            )
            regression_design_mat = np.column_stack(
                [np.ones(len(prior_feature_mat)), prior_feature_mat]
            )
            expected_beta_vec, _, _, _ = np.linalg.lstsq(
                regression_design_mat,
                current_return_vec,
                rcond=None,
            )
            expected_beta_row_list.append(expected_beta_vec)

            actual_monthly_beta_vec = signal_bundle_obj.coefficient_df.loc[
                current_date_ts,
                [
                    "monthly_intercept_float",
                    "monthly_sma_2_ratio_float",
                    "monthly_sma_3_ratio_float",
                ],
            ].to_numpy(dtype=float)
            np.testing.assert_allclose(
                actual_monthly_beta_vec,
                expected_beta_vec,
                rtol=1e-12,
                atol=1e-12,
            )

        expected_smoothed_beta_vec = np.mean(
            np.vstack(expected_beta_row_list),
            axis=0,
        )
        actual_smoothed_beta_vec = signal_bundle_obj.coefficient_df.loc[
            first_target_date_ts,
            [
                "smoothed_intercept_float",
                "smoothed_sma_2_ratio_float",
                "smoothed_sma_3_ratio_float",
            ],
        ].to_numpy(dtype=float)
        np.testing.assert_allclose(
            actual_smoothed_beta_vec,
            expected_smoothed_beta_vec,
            rtol=1e-12,
            atol=1e-12,
        )

        symbol_str = "S001"
        current_feature_vec = np.array(
            [
                feature_by_lookback_dict[lookback_int].loc[
                    first_target_date_ts,
                    symbol_str,
                ]
                for lookback_int in config_obj.sma_lookback_tuple
            ],
            dtype=float,
        )
        expected_forecast_float = float(
            current_feature_vec @ expected_smoothed_beta_vec[1:]
        )
        actual_forecast_float = float(
            signal_bundle_obj.forecast_df.loc[
                signal_bundle_obj.forecast_df[
                    "decision_date_ts"
                ].eq(first_target_date_ts)
                & signal_bundle_obj.forecast_df["symbol_str"].eq(symbol_str),
                "forecast_float",
            ].iloc[0]
        )
        self.assertAlmostEqual(
            actual_forecast_float,
            expected_forecast_float,
            places=12,
        )

    def test_future_price_changes_cannot_change_prior_targets(self):
        (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        ) = self.make_price_fixture()
        config_obj = self.make_config()
        baseline_bundle_obj = (
            build_adaptive_moving_average_factor_signal_bundle(
                price_close_df=price_close_df,
                raw_close_df=raw_close_df,
                universe_df=universe_df,
                decision_date_index=decision_date_index,
                config_obj=config_obj,
            )
        )
        cutoff_date_ts = pd.Timestamp(
            baseline_bundle_obj.target_weight_df.index[-2]
        )
        changed_price_close_df = price_close_df.copy()
        changed_raw_close_df = raw_close_df.copy()
        future_row_ser = changed_price_close_df.index > cutoff_date_ts
        changed_price_close_df.loc[future_row_ser, "S010"] *= 4.0
        changed_raw_close_df.loc[future_row_ser, "S010"] *= 4.0
        changed_bundle_obj = (
            build_adaptive_moving_average_factor_signal_bundle(
                price_close_df=changed_price_close_df,
                raw_close_df=changed_raw_close_df,
                universe_df=universe_df,
                decision_date_index=decision_date_index,
                config_obj=config_obj,
            )
        )
        pd.testing.assert_frame_equal(
            baseline_bundle_obj.target_weight_df.loc[:cutoff_date_ts],
            changed_bundle_obj.target_weight_df.loc[:cutoff_date_ts],
        )

    def test_raw_five_dollar_boundary_is_inclusive_and_stale_row_is_rejected(self):
        (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        ) = self.make_price_fixture()
        final_decision_date_ts = pd.Timestamp(decision_date_index[-1])
        raw_close_df.loc[final_decision_date_ts, "S001"] = 5.0
        raw_close_df.loc[final_decision_date_ts, "S002"] = 4.99
        price_close_df.loc[final_decision_date_ts, "S003"] = np.nan

        signal_bundle_obj = (
            build_adaptive_moving_average_factor_signal_bundle(
                price_close_df=price_close_df,
                raw_close_df=raw_close_df,
                universe_df=universe_df,
                decision_date_index=decision_date_index,
                config_obj=self.make_config(),
            )
        )
        final_forecast_df = signal_bundle_obj.forecast_df.loc[
            signal_bundle_obj.forecast_df["decision_date_ts"].eq(
                final_decision_date_ts
            )
        ]
        self.assertIn("S001", set(final_forecast_df["symbol_str"]))
        self.assertNotIn("S002", set(final_forecast_df["symbol_str"]))
        self.assertNotIn("S003", set(final_forecast_df["symbol_str"]))

    def test_pit_membership_at_prior_close_controls_regression_sample(self):
        (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        ) = self.make_price_fixture()
        prior_decision_date_ts = pd.Timestamp(decision_date_index[-2])
        current_decision_date_ts = pd.Timestamp(decision_date_index[-1])
        universe_df.loc[
            universe_df.index <= prior_decision_date_ts,
            "S010",
        ] = 0
        universe_df.loc[
            universe_df.index > prior_decision_date_ts,
            "S010",
        ] = 1
        signal_bundle_obj = (
            build_adaptive_moving_average_factor_signal_bundle(
                price_close_df=price_close_df,
                raw_close_df=raw_close_df,
                universe_df=universe_df,
                decision_date_index=decision_date_index,
                config_obj=self.make_config(),
            )
        )
        current_coverage_ser = signal_bundle_obj.coverage_df.set_index(
            "decision_date_ts"
        ).loc[current_decision_date_ts]
        self.assertEqual(
            int(current_coverage_ser["regression_count_int"]),
            9,
        )

    def test_iterate_submits_delta_targets_and_liquidates_removed_name(self):
        config_obj = self.make_config()
        decision_date_ts = pd.Timestamp("2024-03-29")
        execution_date_ts = pd.Timestamp("2024-04-01")
        universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1]},
            index=[decision_date_ts],
        )
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [decision_date_ts]},
            index=[execution_date_ts],
        )
        strategy_obj = AdaptiveMovingAverageFactorStrategy(
            name="amaf_test",
            benchmarks=["SPY"],
            universe_df=universe_df,
            rebalance_schedule_df=rebalance_schedule_df,
            config_obj=config_obj,
        )
        self.assertEqual(
            strategy_obj._performance_benchmark_adjustment_str,
            "TOTALRETURN",
        )
        self.assertEqual(
            strategy_obj._data_adjustment_policy_dict,
            {
                "stock_signal_adjustment_str": "CAPITALSPECIAL",
                "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
                "performance_benchmark_adjustment_str": "TOTALRETURN",
            },
        )
        strategy_obj.signal_bundle_obj = AdaptiveMovingAverageFactorSignalBundle(
            target_weight_df=pd.DataFrame(
                {"AAA": [0.5], "BBB": [0.5]},
                index=[decision_date_ts],
            ),
            forecast_df=pd.DataFrame(
                {
                    "decision_date_ts": [decision_date_ts, decision_date_ts],
                    "symbol_str": ["AAA", "BBB"],
                    "forecast_float": [1.0, 2.0],
                    "quintile_int": [5, 5],
                    "selected_bool": [True, True],
                    "target_weight_float": [0.5, 0.5],
                }
            ),
            coefficient_df=pd.DataFrame(),
            coverage_df=pd.DataFrame(),
        )
        strategy_obj.previous_bar = decision_date_ts
        strategy_obj.current_bar = execution_date_ts
        strategy_obj.add_transaction(
            1,
            decision_date_ts,
            "OLD",
            10,
            100.0,
            1_000.0,
            1,
            0.0,
        )
        strategy_obj.current_trade_map["OLD"] = 1
        close_row_ser = pd.Series(
            {
                ("AAA", TARGET_WEIGHT_FIELD_STR): 0.5,
                ("BBB", TARGET_WEIGHT_FIELD_STR): 0.5,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy_obj.iterate(
            data=pd.DataFrame(index=[decision_date_ts]),
            close=close_row_ser,
            open_prices=pd.Series({"AAA": 10.0, "BBB": 10.0}),
        )

        order_list = strategy_obj.get_orders()
        self.assertTrue(
            all(isinstance(order_obj, MarketOrder) for order_obj in order_list)
        )
        liquidation_asset_set = {
            order_obj.asset
            for order_obj in order_list
            if order_obj.unit == "shares" and order_obj.target
        }
        target_order_by_symbol_dict = {
            order_obj.asset: order_obj
            for order_obj in order_list
            if order_obj.unit == "percent" and order_obj.target
        }
        self.assertEqual(liquidation_asset_set, {"OLD"})
        self.assertAlmostEqual(
            float(target_order_by_symbol_dict["AAA"].amount),
            0.5,
        )
        self.assertAlmostEqual(
            float(target_order_by_symbol_dict["BBB"].amount),
            0.5,
        )

    def test_compute_signals_passes_truncated_history_audit(self):
        (
            price_close_df,
            raw_close_df,
            universe_df,
            decision_date_index,
        ) = self.make_price_fixture()
        pricing_column_dict: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in price_close_df.columns:
            pricing_column_dict[(symbol_str, "Open")] = (
                price_close_df[symbol_str] * 0.999
            )
            pricing_column_dict[(symbol_str, "High")] = (
                price_close_df[symbol_str] * 1.01
            )
            pricing_column_dict[(symbol_str, "Low")] = (
                price_close_df[symbol_str] * 0.99
            )
            pricing_column_dict[(symbol_str, "Close")] = price_close_df[
                symbol_str
            ]
            pricing_column_dict[
                (symbol_str, "Unadjusted Close")
            ] = raw_close_df[symbol_str]
            pricing_column_dict[(symbol_str, "Dividend")] = pd.Series(
                0.0,
                index=price_close_df.index,
            )
        benchmark_close_ser = pd.Series(
            np.linspace(100.0, 120.0, len(price_close_df.index)),
            index=price_close_df.index,
        )
        for field_str, multiplier_float in (
            ("Open", 0.999),
            ("High", 1.01),
            ("Low", 0.99),
            ("Close", 1.0),
        ):
            pricing_column_dict[("SPY", field_str)] = (
                benchmark_close_ser * multiplier_float
            )
        pricing_data_df = pd.DataFrame(
            pricing_column_dict,
            index=price_close_df.index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(
            pricing_data_df.columns
        )
        rebalance_schedule_df = (
            map_month_end_decision_dates_to_rebalance_schedule_df(
                decision_date_index=decision_date_index,
                execution_index=pricing_data_df.index,
            )
        )
        strategy_obj = AdaptiveMovingAverageFactorStrategy(
            name="amaf_signal_audit_test",
            benchmarks=["SPY"],
            universe_df=universe_df,
            rebalance_schedule_df=rebalance_schedule_df,
            config_obj=self.make_config(),
        )

        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        self.assertIn(
            ("S001", TARGET_WEIGHT_FIELD_STR),
            signal_data_df.columns,
        )
        strategy_obj.audit_signals(
            pricing_data_df,
            signal_data_df,
            sample_size=3,
        )


if __name__ == "__main__":
    unittest.main()
