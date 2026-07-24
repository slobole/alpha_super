import unittest

import numpy as np
import pandas as pd

from alpha.engine.metrics import generate_monthly_returns, select_tail_event_date_index
from alpha.engine.portfolio import Portfolio
from alpha.engine.strategy import Strategy


class DummyStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def make_strategy(name: str, dates_index: pd.DatetimeIndex, daily_returns_list: list[float], capital_base: float = 100.0):
    strategy = DummyStrategy(
        name=name,
        benchmarks=[],
        capital_base=capital_base,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    daily_returns_ser = pd.Series(daily_returns_list, index=dates_index, dtype=float)
    total_value_ser = capital_base * (1 + daily_returns_ser).cumprod()
    strategy.results = pd.DataFrame({
        'daily_returns': daily_returns_ser,
        'total_value': total_value_ser,
        'portfolio_value': total_value_ser,
    }, index=dates_index)
    strategy.summary = pd.DataFrame()
    strategy.summary_trades = pd.DataFrame()
    return strategy


class PortfolioTests(unittest.TestCase):
    def test_pooled_trade_diagnostics_use_only_complete_pm_window_lifecycles(self):
        strategy_a_date_index = pd.bdate_range('2024-01-02', periods=7)
        strategy_b_date_index = strategy_a_date_index[2:5]
        strategy_a = make_strategy(
            'StrategyA',
            strategy_a_date_index,
            [0.0, 0.01, -0.01, 0.01, 0.0, 0.01, 0.0],
        )
        strategy_b = make_strategy(
            'StrategyB',
            strategy_b_date_index,
            [0.0, 0.01, 0.0],
        )
        strategy_a._trades = pd.DataFrame(
            {
                'start': [strategy_a_date_index[0], strategy_a_date_index[1], strategy_a_date_index[2], strategy_a_date_index[3]],
                'end': [strategy_a_date_index[1], strategy_a_date_index[3], strategy_a_date_index[3], strategy_a_date_index[5]],
                'return': [0.01, 0.02, 0.03, 0.04],
                'duration': [pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=1), pd.Timedelta(days=2)],
                'profit': [1.0, 2.0, 3.0, 4.0],
                'commission': [0.1, 0.2, 0.3, 0.4],
            }
        )
        strategy_a._transactions = pd.DataFrame(
            {
                'bar': [strategy_a_date_index[0], strategy_a_date_index[2], strategy_a_date_index[4], strategy_a_date_index[5]],
                'commission': [0.1, 0.2, 0.3, 0.4],
            }
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertEqual(portfolio._common_start, strategy_a_date_index[2])
        self.assertEqual(len(portfolio._trades), 1)
        self.assertEqual(float(portfolio._trades.iloc[0]['profit']), 3.0)
        self.assertEqual(
            portfolio._transactions['bar'].to_list(),
            [strategy_a_date_index[2], strategy_a_date_index[4]],
        )

    def test_no_rebalance_compounds_pods_independently(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03', '2024-02-04'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, 0.0, 0.0])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0, 1.0])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertAlmostEqual(portfolio.results.iloc[-1]['total_value'], 200.0)

    def test_monthly_rebalance_redistributes_capital(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03', '2024-02-04'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, 0.0, 0.0])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0, 1.0])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
            rebalance='monthly',
        )

        self.assertAlmostEqual(portfolio.results.iloc[-1]['total_value'], 225.0)

    def test_equal_rebalance_policy_targets_one_over_n(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03', '2024-02-04'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, 0.0, 0.0])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0, 1.0])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.8, 0.2],
            capital_base=100.0,
            rebalance='monthly',
            rebalance_policy_str='equal',
        )

        self.assertAlmostEqual(portfolio.results.iloc[-1]['total_value'], 270.0)
        target_weight_ser = portfolio.rebalance_target_weight_df.loc[pd.Timestamp('2024-02-03')]
        self.assertAlmostEqual(float(target_weight_ser['StrategyA']), 0.5)
        self.assertAlmostEqual(float(target_weight_ser['StrategyB']), 0.5)

    def test_inverse_volatility_rebalance_policy_prefers_lower_vol_pod(self):
        dates_index = pd.bdate_range('2024-01-02', periods=45)
        low_vol_return_list = [0.0] + [0.001 if idx_int % 2 == 0 else -0.001 for idx_int in range(44)]
        high_vol_return_list = [0.0] + [0.02 if idx_int % 2 == 0 else -0.02 for idx_int in range(44)]
        strategy_low = make_strategy('LowVol', dates_index, low_vol_return_list)
        strategy_high = make_strategy('HighVol', dates_index, high_vol_return_list)

        portfolio = Portfolio(
            strategies=[strategy_low, strategy_high],
            weights=[0.5, 0.5],
            capital_base=100.0,
            rebalance='monthly',
            rebalance_policy_str='inverse_volatility',
            rebalance_inverse_volatility_lookback_day_int=4,
        )

        first_target_weight_ser = portfolio.rebalance_target_weight_df.iloc[0]
        self.assertGreater(
            float(first_target_weight_ser['LowVol']),
            float(first_target_weight_ser['HighVol']),
        )
        self.assertAlmostEqual(float(first_target_weight_ser.sum()), 1.0)
        self.assertEqual(
            set(portfolio.rebalance_diagnostic_df['status_str']),
            {'applied'},
        )

    def test_inverse_volatility_rebalance_skips_dates_without_enough_history(self):
        dates_index = pd.bdate_range('2024-01-30', periods=10)
        strategy_a = make_strategy('StrategyA', dates_index, [0.0] + [0.01, -0.01] * 4 + [0.01])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0] + [0.02, -0.02] * 4 + [0.02])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
            rebalance='monthly',
            rebalance_policy_str='inverse_volatility',
            rebalance_inverse_volatility_lookback_day_int=20,
        )

        self.assertEqual(len(portfolio.rebalance_target_weight_df), 0)
        self.assertEqual(
            set(portfolio.rebalance_diagnostic_df['status_str']),
            {'skipped_insufficient_history'},
        )

    def test_common_date_range_uses_overlap_only(self):
        strategy_a = make_strategy(
            'StrategyA',
            pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            [0.0, 0.01, 0.02],
        )
        strategy_b = make_strategy(
            'StrategyB',
            pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']),
            [0.0, -0.01, 0.01],
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertEqual(list(portfolio.results.index), list(pd.to_datetime(['2024-01-02', '2024-01-03'])))

    def test_overlap_start_is_a_clean_capital_anchor(self):
        strategy_a = make_strategy(
            'StrategyA',
            pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            [0.0, 0.10, 0.0],
            capital_base=100.0,
        )
        strategy_b = make_strategy(
            'StrategyB',
            pd.to_datetime(['2024-01-02', '2024-01-03']),
            [0.0, 0.0],
            capital_base=100.0,
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertAlmostEqual(float(portfolio.results.iloc[0]['total_value']), 100.0)
        self.assertAlmostEqual(float(portfolio._pod_equities.iloc[0]['StrategyA']), 50.0)
        self.assertAlmostEqual(float(portfolio._pod_equities.iloc[0]['StrategyB']), 50.0)

    def test_summary_uses_allocated_sleeve_capital_not_standalone_capital(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 0.10, 0.0], capital_base=100.0)
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.10], capital_base=100.0)

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.4, 0.6],
            capital_base=100.0,
        )

        self.assertAlmostEqual(float(portfolio.summary.loc['Start [$]', portfolio.name]), 100.0)
        self.assertAlmostEqual(
            float(portfolio.summary.loc['Start [$]', 'StrategyA Sleeve (40%)']),
            40.0,
        )
        self.assertAlmostEqual(
            float(portfolio.summary.loc['Start [$]', 'StrategyB Sleeve (60%)']),
            60.0,
        )

    def test_portfolio_and_allocated_sleeves_share_explicit_pm_regression_benchmark(self):
        dates_index = pd.bdate_range('2020-01-02', periods=300)
        benchmark_return_ser = pd.Series(
            np.linspace(-0.01, 0.01, len(dates_index)),
            index=dates_index,
            dtype=float,
        )
        benchmark_value_ser = 100.0 * (1.0 + benchmark_return_ser).cumprod()
        strategy_a = make_strategy(
            'StrategyA',
            dates_index,
            [0.0] + (1.2 * benchmark_return_ser.iloc[1:]).tolist(),
        )
        strategy_b = make_strategy(
            'StrategyB',
            dates_index,
            [0.0] + (0.4 * benchmark_return_ser.iloc[1:]).tolist(),
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
            regression_benchmark_value_ser=benchmark_value_ser,
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )

        expected_column_name_list = [
            portfolio.name,
            'StrategyA Sleeve (50%)',
            'StrategyB Sleeve (50%)',
        ]
        for column_name_str in expected_column_name_list:
            metadata_dict = portfolio.benchmark_regression_metadata_by_column_dict[column_name_str]
            self.assertEqual(metadata_dict['status_str'], 'ok')
            self.assertEqual(metadata_dict['benchmark_label_str'], '$SPX · TOTALRETURN')
            self.assertTrue(np.isfinite(float(portfolio.summary.loc['Beta', column_name_str])))
        expected_benchmark_monthly_return_df = generate_monthly_returns(
            benchmark_value_ser.copy(),
            add_sharpe_ratios=True,
            add_max_drawdowns=True,
        )
        pd.testing.assert_frame_equal(
            portfolio.benchmark_monthly_returns,
            expected_benchmark_monthly_return_df,
        )
        self.assertEqual(
            portfolio.standalone_benchmark_regression_metadata_by_column_dict[
                'StrategyA Standalone'
            ]['reason_str'],
            'missing_benchmark',
        )

    def test_portfolio_without_benchmark_has_no_benchmark_monthly_returns(self):
        dates_index = pd.bdate_range('2024-01-02', periods=30)
        portfolio = Portfolio(
            strategies=[
                make_strategy('StrategyA', dates_index, [0.0] + [0.001] * 29),
                make_strategy('StrategyB', dates_index, [0.0] + [0.002] * 29),
            ],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertIsNone(portfolio.benchmark_monthly_returns)

    def test_benchmark_monthly_returns_require_complete_reporting_window(self):
        portfolio_date_index = pd.bdate_range('2024-01-15', periods=40)
        benchmark_date_index = pd.bdate_range('2024-01-02', periods=70)
        benchmark_return_ser = pd.Series(
            np.linspace(-0.002, 0.003, len(benchmark_date_index)),
            index=benchmark_date_index,
            dtype=float,
        )
        benchmark_value_ser = 100.0 * (1.0 + benchmark_return_ser).cumprod()
        benchmark_value_ser.loc[portfolio_date_index[5]] = np.nan
        benchmark_value_ser.loc[portfolio_date_index[10]] = np.inf
        portfolio = Portfolio(
            strategies=[
                make_strategy('StrategyA', portfolio_date_index, [0.0] + [0.001] * 39),
                make_strategy('StrategyB', portfolio_date_index, [0.0] + [0.002] * 39),
            ],
            weights=[0.5, 0.5],
            capital_base=100.0,
            regression_benchmark_value_ser=benchmark_value_ser,
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )
        self.assertIsNone(portfolio.benchmark_monthly_returns)

    def test_benchmark_monthly_returns_are_unavailable_without_usable_overlap(self):
        portfolio_date_index = pd.bdate_range('2024-01-02', periods=30)
        benchmark_date_index = pd.bdate_range('2020-01-02', periods=30)
        portfolio = Portfolio(
            strategies=[
                make_strategy('StrategyA', portfolio_date_index, [0.0] + [0.001] * 29),
                make_strategy('StrategyB', portfolio_date_index, [0.0] + [0.002] * 29),
            ],
            weights=[0.5, 0.5],
            capital_base=100.0,
            regression_benchmark_value_ser=pd.Series(
                np.linspace(100.0, 110.0, len(benchmark_date_index)),
                index=benchmark_date_index,
                dtype=float,
            ),
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )

        self.assertIsNone(portfolio.benchmark_monthly_returns)

    def test_benchmark_monthly_returns_are_unavailable_with_one_overlap_point(self):
        portfolio_date_index = pd.bdate_range('2024-01-02', periods=30)
        portfolio = Portfolio(
            strategies=[
                make_strategy('StrategyA', portfolio_date_index, [0.0] + [0.001] * 29),
                make_strategy('StrategyB', portfolio_date_index, [0.0] + [0.002] * 29),
            ],
            weights=[0.5, 0.5],
            capital_base=100.0,
            regression_benchmark_value_ser=pd.Series(
                [100.0],
                index=pd.DatetimeIndex([portfolio_date_index[10]]),
                dtype=float,
            ),
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )

        self.assertIsNone(portfolio.benchmark_monthly_returns)

    def test_rebalanced_sleeve_regression_excludes_pm_cash_transfers(self):
        dates_index = pd.bdate_range('2020-01-02', periods=320)
        benchmark_return_ser = pd.Series(
            np.linspace(-0.005, 0.005, len(dates_index)),
            index=dates_index,
            dtype=float,
        )
        benchmark_value_ser = 100.0 * (1.0 + benchmark_return_ser).cumprod()
        strategy_a = make_strategy(
            'StrategyA',
            dates_index,
            [0.0] + (1.5 * benchmark_return_ser.iloc[1:]).tolist(),
        )
        strategy_b = make_strategy(
            'StrategyB',
            dates_index,
            [0.0] + (0.25 * benchmark_return_ser.iloc[1:]).tolist(),
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
            rebalance='monthly',
            regression_benchmark_value_ser=benchmark_value_ser,
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )

        sleeve_column_name_str = 'StrategyA Sleeve (50%)'
        sleeve_equity_beta_float = float(
            np.cov(
                portfolio._pod_equities['StrategyA'].pct_change(fill_method=None).iloc[1:],
                benchmark_return_ser.iloc[1:],
                ddof=1,
            )[0, 1]
            / np.var(benchmark_return_ser.iloc[1:], ddof=1)
        )
        self.assertAlmostEqual(
            float(portfolio.summary.loc['Beta', sleeve_column_name_str]),
            1.5,
            places=10,
        )
        self.assertNotAlmostEqual(sleeve_equity_beta_float, 1.5, places=4)

    def test_sleeve_regression_drops_internal_missing_returns_without_filling(self):
        dates_index = pd.bdate_range('2020-01-02', periods=320)
        benchmark_return_ser = pd.Series(
            np.linspace(-0.005, 0.005, len(dates_index)),
            index=dates_index,
            dtype=float,
        )
        benchmark_value_ser = 100.0 * (1.0 + benchmark_return_ser).cumprod()
        strategy_a = make_strategy(
            'StrategyA',
            dates_index,
            [0.0] + (1.5 * benchmark_return_ser.iloc[1:]).tolist(),
        )
        strategy_b = make_strategy(
            'StrategyB',
            dates_index,
            [0.0] + (0.25 * benchmark_return_ser.iloc[1:]).tolist(),
        )
        strategy_a.results.loc[dates_index[100], 'total_value'] = np.nan

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
            regression_benchmark_value_ser=benchmark_value_ser,
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )

        sleeve_column_name_str = 'StrategyA Sleeve (50%)'
        metadata_dict = portfolio.benchmark_regression_metadata_by_column_dict[
            sleeve_column_name_str
        ]
        self.assertEqual(metadata_dict['observation_count_int'], 317)
        self.assertAlmostEqual(
            float(portfolio.summary.loc['Beta', sleeve_column_name_str]),
            1.5,
            places=10,
        )

    def test_buy_and_hold_pod_math_differs_from_daily_rebalanced_shortcut(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03', '2024-02-04'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, 0.0, 0.0])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0, 1.0])

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        shortcut_return_ser = (
            portfolio._daily_rets.mul([0.5, 0.5], axis=1).sum(axis=1)
        )
        shortcut_total_value_ser = 100.0 * (1.0 + shortcut_return_ser).cumprod()

        self.assertAlmostEqual(float(portfolio.results.iloc[-1]['total_value']), 200.0)
        self.assertAlmostEqual(float(shortcut_total_value_ser.iloc[-1]), 225.0)
        self.assertNotAlmostEqual(
            float(portfolio.results.iloc[-1]['total_value']),
            float(shortcut_total_value_ser.iloc[-1]),
        )

    def test_portfolio_equity_equals_sum_of_pod_equities(self):
        dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 0.10, -0.05], capital_base=100.0)
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, -0.02, 0.03], capital_base=100.0)

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        pod_sum_ser = portfolio._pod_equities.sum(axis=1)
        pd.testing.assert_series_equal(
            portfolio.results['total_value'],
            pod_sum_ser,
            check_names=False,
        )

    def test_weight_sum_validation(self):
        dates_index = pd.to_datetime(['2024-01-01', '2024-01-02'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 0.01])
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.01])

        with self.assertRaisesRegex(ValueError, 'Weights must sum to 1.0'):
            Portfolio(
                strategies=[strategy_a, strategy_b],
                weights=[0.6, 0.3],
                capital_base=100.0,
            )

    def test_tail_event_selection_excludes_bootstrap_and_uses_ceiling_count(self):
        dates_index = pd.date_range('2024-01-01', periods=41, freq='B')
        portfolio_daily_return_ser = pd.Series(
            [-0.99] + [0.01] * 38 + [-0.02, -0.03],
            index=dates_index,
            dtype=float,
        )

        tail_event_date_index = select_tail_event_date_index(
            portfolio_daily_return_ser,
            tail_fraction_float=0.05,
            min_tail_days_int=1,
        )

        self.assertEqual(len(tail_event_date_index), 2)
        self.assertNotIn(dates_index[0], tail_event_date_index)
        self.assertEqual(set(tail_event_date_index), set(dates_index[-2:]))

    def test_tail_contributions_sum_to_portfolio_return_using_previous_weights(self):
        dates_index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        strategy_a = make_strategy('StrategyA', dates_index, [0.0, 1.0, -0.5], capital_base=100.0)
        strategy_b = make_strategy('StrategyB', dates_index, [0.0, 0.0, 0.0], capital_base=100.0)

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        tail_date_ts = pd.Timestamp('2024-01-03')
        self.assertEqual(list(portfolio.tail_event_date_index), [tail_date_ts])
        self.assertAlmostEqual(
            float(portfolio.tail_contribution_df.loc[tail_date_ts].sum()),
            float(portfolio.results.loc[tail_date_ts, 'daily_returns']),
        )
        self.assertAlmostEqual(
            float(portfolio.tail_contribution_df.loc[tail_date_ts, 'StrategyA']),
            -1.0 / 3.0,
        )
        self.assertNotAlmostEqual(
            float(portfolio.tail_contribution_df.loc[tail_date_ts, 'StrategyA']),
            -0.25,
        )

    def test_portfolio_exposes_tail_attribution_and_withholds_biased_correlation(self):
        dates_index = pd.date_range('2024-01-01', periods=41, freq='B')
        strategy_a_return_list = [0.0]
        strategy_b_return_list = [0.0]
        for day_idx_int in range(40):
            if day_idx_int == 10:
                strategy_a_return_list.append(-0.05)
                strategy_b_return_list.append(-0.05)
            elif day_idx_int == 30:
                strategy_a_return_list.append(-0.04)
                strategy_b_return_list.append(-0.04)
            elif day_idx_int % 2 == 0:
                strategy_a_return_list.append(0.01)
                strategy_b_return_list.append(-0.005)
            else:
                strategy_a_return_list.append(-0.005)
                strategy_b_return_list.append(0.01)

        strategy_a = make_strategy('StrategyA', dates_index, strategy_a_return_list, capital_base=100.0)
        strategy_b = make_strategy('StrategyB', dates_index, strategy_b_return_list, capital_base=100.0)

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        self.assertEqual(len(portfolio.tail_event_date_index), 2)
        self.assertListEqual(list(portfolio.tail_return_df.columns), ['StrategyA', 'StrategyB'])
        self.assertListEqual(list(portfolio.tail_contribution_df.columns), ['StrategyA', 'StrategyB'])
        self.assertIn('average_loss_contribution_share_float', portfolio.tail_summary_df.columns)

        # Attribution is conditioned on the book's own worst days, which is the
        # right question for "who hurt me" and stays available here.
        #
        # Correlation is not. This portfolio has no benchmark, so there is no
        # exogenous stress reference, and measuring co-movement on the
        # portfolio's own tail would force the pods to offset one another by
        # construction. The estimate is withheld rather than reported biased.
        self.assertEqual(len(portfolio.stress_event_date_index), 0)
        tail_corr_float = float(portfolio.tail_correlation_matrix.loc['StrategyA', 'StrategyB'])
        self.assertTrue(np.isnan(tail_corr_float))
