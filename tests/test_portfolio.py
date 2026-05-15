import unittest

import numpy as np
import pandas as pd

from alpha.engine.metrics import select_tail_event_date_index
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

    def test_portfolio_exposes_tail_diagnostics_and_tail_corr_differs_from_full_sample(self):
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
        tail_corr_float = float(portfolio.tail_correlation_matrix.loc['StrategyA', 'StrategyB'])
        full_corr_float = float(portfolio.correlation_matrix.loc['StrategyA', 'StrategyB'])
        self.assertTrue(np.isfinite(tail_corr_float))
        self.assertNotAlmostEqual(tail_corr_float, full_corr_float)
