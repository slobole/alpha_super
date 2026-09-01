import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from alpha.engine.metrics import (
    EXPECTED_SHORTFALL_METRIC_NAME_STR,
    generate_benchmark_regression_metrics,
    generate_overall_metrics,
    generate_trades,
    sharpe_ratio,
)


class GenerateTradesTests(unittest.TestCase):
    def test_generate_trades_preserves_long_trade_return_math(self):
        transaction_df = pd.DataFrame(
            [
                {
                    "trade_id": 1,
                    "bar": pd.Timestamp("2024-01-02"),
                    "asset": "AAA",
                    "amount": 10,
                    "price": 100.0,
                    "total_value": 1_000.0,
                    "order_id": 1,
                    "commission": 0.0,
                },
                {
                    "trade_id": 1,
                    "bar": pd.Timestamp("2024-01-10"),
                    "asset": "AAA",
                    "amount": -10,
                    "price": 110.0,
                    "total_value": -1_100.0,
                    "order_id": 2,
                    "commission": 0.0,
                },
            ]
        )

        trade_df = generate_trades(transaction_df)

        self.assertAlmostEqual(float(trade_df.loc[1, "capital"]), 1_000.0)
        self.assertAlmostEqual(float(trade_df.loc[1, "profit"]), 100.0)
        self.assertAlmostEqual(float(trade_df.loc[1, "return"]), 0.10)

    def test_stress_correlation_is_unbiased_for_independent_pods(self):
        """Independent pods must not appear to decouple under stress.

        Conditioning on the portfolio's own worst days forces its components to
        offset one another, so genuinely independent pods score a large
        negative correlation. Selecting days by an exogenous benchmark removes
        that selection effect.
        """
        from alpha.engine.metrics import generate_tail_risk_diagnostics

        random_generator = np.random.default_rng(0)
        bar_date_idx = pd.bdate_range('2010-01-04', periods=4000)
        pod_name_list = [f'pod_{idx}' for idx in range(4)]
        pod_daily_return_df = pd.DataFrame(
            random_generator.normal(0.0, 0.01, (len(bar_date_idx), len(pod_name_list))),
            index=bar_date_idx,
            columns=pod_name_list,
        )
        pod_equity_df = (1.0 + pod_daily_return_df).cumprod() * 10_000.0
        portfolio_daily_return_ser = pod_daily_return_df.mean(axis=1)
        # A benchmark unrelated to the pods: the honest stress reference.
        benchmark_return_ser = pd.Series(
            random_generator.normal(0.0, 0.011, len(bar_date_idx)), index=bar_date_idx
        )

        diagnostic_dict = generate_tail_risk_diagnostics(
            pod_daily_return_df=pod_daily_return_df,
            portfolio_daily_return_ser=portfolio_daily_return_ser,
            pod_equity_df=pod_equity_df,
            tail_fraction_float=0.05,
            stress_reference_return_ser=benchmark_return_ser,
        )
        correlation_matrix = diagnostic_dict['tail_correlation_matrix'].to_numpy()
        off_diagonal_mask = ~np.eye(len(pod_name_list), dtype=bool)
        mean_correlation_float = float(np.nanmean(correlation_matrix[off_diagonal_mask]))

        # The biased definition produced roughly -0.27 here; the truth is 0.
        self.assertLess(abs(mean_correlation_float), 0.12)
        self.assertGreater(len(diagnostic_dict['stress_event_date_index']), 0)

    def test_stress_correlation_omitted_without_a_reference(self):
        """No benchmark means no correlation, rather than a biased one."""
        from alpha.engine.metrics import generate_tail_risk_diagnostics

        random_generator = np.random.default_rng(1)
        bar_date_idx = pd.bdate_range('2010-01-04', periods=400)
        pod_daily_return_df = pd.DataFrame(
            random_generator.normal(0.0, 0.01, (len(bar_date_idx), 3)),
            index=bar_date_idx,
            columns=['pod_a', 'pod_b', 'pod_c'],
        )
        diagnostic_dict = generate_tail_risk_diagnostics(
            pod_daily_return_df=pod_daily_return_df,
            portfolio_daily_return_ser=pod_daily_return_df.mean(axis=1),
            pod_equity_df=(1.0 + pod_daily_return_df).cumprod() * 10_000.0,
            tail_fraction_float=0.05,
            stress_reference_return_ser=None,
        )

        self.assertEqual(len(diagnostic_dict['stress_event_date_index']), 0)
        # The diagonal is 1.0 by definition; every pod *pair* must be undefined.
        correlation_matrix = diagnostic_dict['tail_correlation_matrix'].to_numpy()
        off_diagonal_mask = ~np.eye(len(correlation_matrix), dtype=bool)
        self.assertTrue(np.isnan(correlation_matrix[off_diagonal_mask]).all())
        # Attribution still works: it is correctly conditioned on the book's own tail.
        self.assertGreater(len(diagnostic_dict['tail_event_date_index']), 0)

    def test_generate_trades_returns_nan_for_zero_capital_trade(self):
        """A zero-capital entry must not yield -inf and poison the aggregates."""
        transaction_df = pd.DataFrame(
            [
                {
                    "trade_id": 1,
                    "bar": pd.Timestamp("2024-01-02"),
                    "asset": "AAA",
                    "amount": 0.0,
                    "price": 29_912.99,
                    "total_value": 0.0,
                    "order_id": 1,
                    "commission": 1.0,
                },
                {
                    "trade_id": 2,
                    "bar": pd.Timestamp("2024-01-02"),
                    "asset": "BBB",
                    "amount": 10,
                    "price": 100.0,
                    "total_value": 1_000.0,
                    "order_id": 2,
                    "commission": 0.0,
                },
                {
                    "trade_id": 2,
                    "bar": pd.Timestamp("2024-01-10"),
                    "asset": "BBB",
                    "amount": -10,
                    "price": 110.0,
                    "total_value": -1_100.0,
                    "order_id": 3,
                    "commission": 0.0,
                },
            ]
        )

        trade_df = generate_trades(transaction_df)
        return_ser = trade_df["return"].astype(float)

        self.assertTrue(np.isnan(float(trade_df.loc[1, "return"])))
        self.assertFalse(np.isinf(return_ser).any())
        # The healthy trade is unaffected, and the mean skips the undefined row.
        self.assertAlmostEqual(float(trade_df.loc[2, "return"]), 0.10)
        self.assertAlmostEqual(float(return_ser.mean()), 0.10)

    def test_generate_trades_uses_absolute_entry_notional_for_short_trade(self):
        transaction_df = pd.DataFrame(
            [
                {
                    "trade_id": 2,
                    "bar": pd.Timestamp("2024-01-02"),
                    "asset": "BBB",
                    "amount": -10,
                    "price": 100.0,
                    "total_value": -1_000.0,
                    "order_id": 1,
                    "commission": 0.0,
                },
                {
                    "trade_id": 2,
                    "bar": pd.Timestamp("2024-01-10"),
                    "asset": "BBB",
                    "amount": 10,
                    "price": 90.0,
                    "total_value": 900.0,
                    "order_id": 2,
                    "commission": 0.0,
                },
            ]
        )

        trade_df = generate_trades(transaction_df)

        self.assertAlmostEqual(float(trade_df.loc[2, "capital"]), 1_000.0)
        self.assertAlmostEqual(float(trade_df.loc[2, "profit"]), 100.0)
        self.assertAlmostEqual(float(trade_df.loc[2, "return"]), 0.10)


class GenerateOverallMetricsTests(unittest.TestCase):
    def test_expected_shortfall_matches_constant_growth_horizon_return(self):
        date_index = pd.date_range("2020-01-01", periods=130, freq="D")
        total_value_ser = pd.Series(
            10_000.0 * (1.01 ** np.arange(len(date_index))),
            index=date_index,
            dtype=float,
        )

        summary_ser = generate_overall_metrics(total_value_ser, capital_base=10_000.0)

        self.assertAlmostEqual(
            float(summary_ser.loc[EXPECTED_SHORTFALL_METRIC_NAME_STR]),
            (1.01 ** 21 - 1.0) * 100.0,
            places=8,
        )

    def test_expected_shortfall_sits_at_or_below_the_five_percent_quantile(self):
        random_generator_obj = np.random.default_rng(11)
        date_index = pd.date_range("2020-01-01", periods=1_000, freq="D")
        daily_return_ser = pd.Series(
            random_generator_obj.normal(0.0004, 0.011, len(date_index)),
            index=date_index,
        )
        total_value_ser = 10_000.0 * (1.0 + daily_return_ser).cumprod()

        summary_ser = generate_overall_metrics(total_value_ser, capital_base=10_000.0)
        # *** CRITICAL*** This reproduces the report-only backward-looking
        # 21-day horizon exactly; it does not construct a forward return label.
        horizon_return_ser = (
            total_value_ser / total_value_ser.shift(21) - 1.0
        ).dropna()
        quantile_pct_float = float(np.quantile(horizon_return_ser, 0.05)) * 100.0
        expected_tail_mean_pct_float = float(
            horizon_return_ser[horizon_return_ser <= quantile_pct_float / 100.0].mean()
            * 100.0
        )
        expected_shortfall_pct_float = float(
            summary_ser.loc[EXPECTED_SHORTFALL_METRIC_NAME_STR]
        )

        self.assertAlmostEqual(
            expected_shortfall_pct_float,
            expected_tail_mean_pct_float,
        )
        self.assertLessEqual(expected_shortfall_pct_float, quantile_pct_float)
        self.assertLess(expected_shortfall_pct_float, 0.0)

    def test_expected_shortfall_is_withheld_when_the_tail_is_too_small(self):
        date_index = pd.date_range("2020-01-01", periods=60, freq="D")
        total_value_ser = pd.Series(
            10_000.0 * (1.001 ** np.arange(len(date_index))),
            index=date_index,
            dtype=float,
        )

        summary_ser = generate_overall_metrics(total_value_ser, capital_base=10_000.0)

        self.assertTrue(
            np.isnan(float(summary_ser.loc[EXPECTED_SHORTFALL_METRIC_NAME_STR]))
        )

    def test_drawdowns_per_year_preserves_fractional_annualization(self):
        date_index = pd.date_range("2024-01-01", periods=5, freq="D")
        total_value_ser = pd.Series(
            [100.0, 90.0, 100.0, 100.0, 100.0],
            index=date_index,
            dtype=float,
        )

        summary_ser = generate_overall_metrics(
            total_value_ser,
            capital_base=100.0,
            days_in_year=4,
        )

        self.assertEqual(float(summary_ser.loc["# Drawdowns"]), 1.0)
        self.assertAlmostEqual(float(summary_ser.loc["# Drawdowns / year"]), 0.8)

    def test_benchmark_regression_recovers_known_alpha_beta_and_r_squared(self):
        random_generator_obj = np.random.default_rng(7)
        date_index = pd.date_range('2020-01-01', periods=400, freq='B')
        benchmark_return_ser = pd.Series(
            random_generator_obj.normal(0.0004, 0.01, len(date_index)),
            index=date_index,
            dtype=float,
        )
        residual_return_ser = pd.Series(
            random_generator_obj.normal(0.0, 0.002, len(date_index)),
            index=date_index,
            dtype=float,
        )
        residual_return_ser = residual_return_ser - residual_return_ser.mean()
        centered_benchmark_ser = benchmark_return_ser - benchmark_return_ser.mean()
        residual_return_ser = residual_return_ser - (
            residual_return_ser.cov(centered_benchmark_ser)
            / centered_benchmark_ser.var()
        ) * centered_benchmark_ser
        alpha_daily_float = 0.0002
        beta_float = 1.5
        strategy_return_ser = alpha_daily_float + beta_float * benchmark_return_ser + residual_return_ser

        regression_metric_ser, regression_metadata_dict = generate_benchmark_regression_metrics(
            strategy_return_ser,
            benchmark_return_ser,
            benchmark_label_str='$SPX · TOTALRETURN',
        )

        self.assertEqual(regression_metadata_dict['status_str'], 'ok')
        self.assertEqual(regression_metadata_dict['observation_count_int'], 400)
        self.assertEqual(
            regression_metadata_dict['hac_max_lag_int'],
            int(np.floor(4.0 * (400 / 100.0) ** (2.0 / 9.0))),
        )
        expected_r_squared_float = 1.0 - float(
            np.square(residual_return_ser).sum()
            / np.square(strategy_return_ser - strategy_return_ser.mean()).sum()
        )
        self.assertAlmostEqual(float(regression_metric_ser.loc['Beta']), beta_float, places=10)
        self.assertAlmostEqual(
            float(regression_metric_ser.loc['Alpha (Ann.) [%]']),
            alpha_daily_float * 252 * 100.0,
            places=10,
        )
        self.assertAlmostEqual(
            float(regression_metric_ser.loc['R²']),
            expected_r_squared_float,
            places=12,
        )
        expected_hac_lag_int = int(regression_metadata_dict['hac_max_lag_int'])
        expected_regression_result_obj = sm.OLS(
            strategy_return_ser,
            sm.add_constant(benchmark_return_ser.rename('benchmark_return'), has_constant='add'),
        ).fit(cov_type='HAC', cov_kwds={'maxlags': expected_hac_lag_int})
        self.assertAlmostEqual(
            float(regression_metric_ser.loc['Alpha HAC t-stat']),
            float(expected_regression_result_obj.tvalues['const']),
            places=12,
        )

    def test_benchmark_regression_supports_negative_and_zero_beta(self):
        date_index = pd.date_range('2020-01-01', periods=300, freq='B')
        benchmark_return_ser = pd.Series(
            np.linspace(-0.02, 0.02, len(date_index)),
            index=date_index,
            dtype=float,
        )

        negative_metric_ser, _ = generate_benchmark_regression_metrics(
            -0.5 * benchmark_return_ser,
            benchmark_return_ser,
            benchmark_label_str='Benchmark',
        )
        zero_beta_return_ser = pd.Series(
            np.tile([-0.001, 0.001], len(date_index) // 2),
            index=date_index,
            dtype=float,
        )
        centered_benchmark_ser = benchmark_return_ser - benchmark_return_ser.mean()
        zero_beta_return_ser = zero_beta_return_ser - (
            zero_beta_return_ser.cov(centered_benchmark_ser)
            / centered_benchmark_ser.var()
        ) * centered_benchmark_ser
        zero_metric_ser, _ = generate_benchmark_regression_metrics(
            zero_beta_return_ser,
            benchmark_return_ser,
            benchmark_label_str='Benchmark',
        )

        self.assertAlmostEqual(float(negative_metric_ser.loc['Beta']), -0.5, places=12)
        self.assertAlmostEqual(float(negative_metric_ser.loc['R²']), 1.0, places=12)
        self.assertAlmostEqual(float(zero_metric_ser.loc['Beta']), 0.0, places=12)
        self.assertAlmostEqual(float(zero_metric_ser.loc['R²']), 0.0, places=12)

    def test_benchmark_regression_aligns_without_fill_and_handles_unavailable_cases(self):
        strategy_date_index = pd.date_range('2020-01-01', periods=260, freq='B')
        benchmark_date_index = pd.date_range('2020-01-15', periods=260, freq='B')
        strategy_return_ser = pd.Series(0.001, index=strategy_date_index, dtype=float)
        benchmark_return_ser = pd.Series(
            np.linspace(-0.01, 0.01, len(benchmark_date_index)),
            index=benchmark_date_index,
            dtype=float,
        )
        strategy_return_ser.iloc[20] = np.nan
        benchmark_return_ser.iloc[20] = np.inf

        _, aligned_metadata_dict = generate_benchmark_regression_metrics(
            strategy_return_ser,
            benchmark_return_ser,
            benchmark_label_str='Benchmark',
            min_observation_count_int=10,
        )
        missing_metric_ser, missing_metadata_dict = generate_benchmark_regression_metrics(
            strategy_return_ser,
            None,
            benchmark_label_str=None,
        )
        _, short_metadata_dict = generate_benchmark_regression_metrics(
            strategy_return_ser.iloc[:20],
            benchmark_return_ser.iloc[:20],
            benchmark_label_str='Benchmark',
        )
        _, flat_metadata_dict = generate_benchmark_regression_metrics(
            strategy_return_ser,
            pd.Series(0.0, index=strategy_date_index, dtype=float),
            benchmark_label_str='Benchmark',
        )
        _, flat_strategy_metadata_dict = generate_benchmark_regression_metrics(
            pd.Series(0.001, index=strategy_date_index, dtype=float),
            benchmark_return_ser.reindex(strategy_date_index),
            benchmark_label_str='Benchmark',
            min_observation_count_int=10,
        )

        expected_pair_count_int = int(
            pd.concat([strategy_return_ser, benchmark_return_ser], axis=1, join='inner')
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .shape[0]
        )
        self.assertEqual(aligned_metadata_dict['observation_count_int'], expected_pair_count_int)
        self.assertEqual(missing_metadata_dict['reason_str'], 'missing_benchmark')
        self.assertTrue(missing_metric_ser.isna().all())
        self.assertEqual(short_metadata_dict['reason_str'], 'insufficient_observations')
        self.assertEqual(flat_metadata_dict['reason_str'], 'zero_benchmark_variance')
        self.assertEqual(flat_strategy_metadata_dict['reason_str'], 'zero_strategy_variance')

    def test_generate_overall_metrics_adds_l1_mar_and_underwater_metrics(self):
        date_index = pd.date_range("2024-01-01", periods=5, freq="D")
        total_value_ser = pd.Series(
            [100.0, 110.0, 105.0, 115.0, 120.0],
            index=date_index,
            dtype=float,
        )

        summary_ser = generate_overall_metrics(
            total_value_ser,
            capital_base=100.0,
            days_in_year=5,
        )

        daily_return_ser = total_value_ser.pct_change(fill_method=None).dropna()
        drawdown_ser = total_value_ser / total_value_ser.cummax() - 1.0
        max_drawdown_pct_float = float(drawdown_ser.min() * 100)
        annual_return_pct_float = float(((120.0 / 100.0) ** (5 / 5) - 1.0) * 100)

        self.assertAlmostEqual(
            float(summary_ser.loc["AAR [%]"]),
            float(daily_return_ser.abs().mean() * 100),
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Downside L1 [%]"]),
            float(np.maximum(-daily_return_ser, 0.0).mean() * 100),
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Avg. Loss Day [%]"]),
            float((-daily_return_ser[daily_return_ser < 0.0]).mean() * 100),
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Time Under Water [%]"]),
            float((drawdown_ser < 0.0).mean() * 100),
        )
        self.assertAlmostEqual(float(summary_ser.loc["Max. Drawdown [%]"]), max_drawdown_pct_float)
        self.assertAlmostEqual(
            float(summary_ser.loc["MAR Ratio"]),
            annual_return_pct_float / abs(max_drawdown_pct_float),
        )
        expected_drawdowns_per_year_float = (
            float(summary_ser.loc["# Drawdowns"]) / (len(total_value_ser) / 5.0)
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["# Drawdowns / year"]),
            expected_drawdowns_per_year_float,
        )

    def test_generate_overall_metrics_reports_both_sharpe_bases(self):
        date_index = pd.date_range("2024-01-01", periods=6, freq="D")
        daily_return_ser = pd.Series(
            [0.0, 0.01, 0.0, 0.0, -0.005, 0.02],
            index=date_index,
            dtype=float,
        )
        total_value_ser = 100.0 * (1.0 + daily_return_ser).cumprod()
        # invested only on days that produced a return; all-cash (dead) otherwise
        portfolio_value_ser = total_value_ser.where(daily_return_ser != 0.0, 0.0)

        summary_with_pv_ser = generate_overall_metrics(
            total_value_ser,
            portfolio_value=portfolio_value_ser,
            capital_base=100.0,
        )
        summary_without_pv_ser = generate_overall_metrics(
            total_value_ser,
            capital_base=100.0,
        )

        realized_return_ser = total_value_ser.pct_change(fill_method=None)
        expected_all_days_float = float(sharpe_ratio(realized_return_ser))
        expected_active_days_float = float(
            sharpe_ratio(realized_return_ser, portfolio_value_ser)
        )

        # the fixture must actually separate the two bases
        self.assertNotAlmostEqual(expected_all_days_float, expected_active_days_float)

        # headline Sharpe is all-days regardless of whether an invested-value
        # series was supplied — the caller can no longer switch its basis
        self.assertAlmostEqual(
            float(summary_with_pv_ser.loc["Sharpe Ratio"]), expected_all_days_float
        )
        self.assertAlmostEqual(
            float(summary_without_pv_ser.loc["Sharpe Ratio"]), expected_all_days_float
        )

        # active-days Sharpe is present and labeled when computable, NaN otherwise
        self.assertAlmostEqual(
            float(summary_with_pv_ser.loc["Sharpe Ratio (Active Days)"]),
            expected_active_days_float,
        )
        self.assertTrue(
            np.isnan(float(summary_without_pv_ser.loc["Sharpe Ratio (Active Days)"]))
        )

    def test_generate_overall_metrics_adds_turnover_and_cost_drag_from_transactions(self):
        date_index = pd.date_range("2024-01-01", periods=5, freq="D")
        total_value_ser = pd.Series(
            [10_000.0, 10_010.0, 10_000.0, 10_020.0, 10_030.0],
            index=date_index,
            dtype=float,
        )
        transaction_df = pd.DataFrame(
            [
                {
                    "bar": date_index[1],
                    "amount": 10.0,
                    "total_value": 1_001.0,
                    "commission": 2.0,
                },
                {
                    "bar": date_index[3],
                    "amount": -10.0,
                    "total_value": -999.0,
                    "commission": 3.0,
                },
            ]
        )

        summary_ser = generate_overall_metrics(
            total_value_ser,
            capital_base=10_000.0,
            days_in_year=5,
            transactions_df=transaction_df,
            slippage_float=0.001,
        )

        average_equity_float = float(total_value_ser.mean())
        gross_trade_notional_float = 2_000.0
        expected_slippage_float = 2.0
        expected_total_cost_float = 7.0

        self.assertAlmostEqual(
            float(summary_ser.loc["Turnover (Ann.) [%]"]),
            gross_trade_notional_float / average_equity_float * 100,
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Estimated Slippage [$]"]),
            expected_slippage_float,
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Total Trading Costs [$]"]),
            expected_total_cost_float,
        )
        self.assertAlmostEqual(
            float(summary_ser.loc["Cost Drag (Ann.) [%]"]),
            expected_total_cost_float / average_equity_float * 100,
        )


if __name__ == "__main__":
    unittest.main()
