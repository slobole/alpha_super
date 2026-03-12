import unittest
from unittest import mock

import numpy as np
import pandas as pd

from alpha.engine.report import (
    _DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT,
    _build_daily_return_distribution_html,
    _build_html,
    _daily_return_histogram_b64,
    _format_summary,
    _prepare_daily_return_distribution_dict,
)
from alpha.engine.strategy import Strategy


class DummyStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def make_strategy(daily_returns_list: list[float]) -> DummyStrategy:
    date_index = pd.date_range('2024-01-02', periods=len(daily_returns_list), freq='D')
    daily_return_ser = pd.Series(daily_returns_list, index=date_index, dtype=float)
    total_value_ser = 100_000.0 * (1.0 + daily_return_ser).cumprod()

    strategy = DummyStrategy(
        name='ReportStrategy',
        benchmarks=[],
        capital_base=100_000.0,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy.results = pd.DataFrame(
        {
            'daily_returns': daily_return_ser,
            'total_value': total_value_ser,
            'portfolio_value': total_value_ser,
        },
        index=date_index,
    )
    strategy.summary = pd.DataFrame(
        {
            'Strategy': [
                pd.Timestamp(date_index[0]),
                pd.Timestamp(date_index[-1]),
                100_000.0,
                float(total_value_ser.iloc[-1]),
                1.23,
            ]
        },
        index=['Start', 'End', 'Start [$]', 'Final [$]', 'Sharpe Ratio'],
    )
    strategy.monthly_returns = pd.DataFrame(
        {'Annual Return': [0.12], 'Sharpe Ratio': [1.23], 'Max Drawdown': [-0.08]},
        index=pd.Index([2024], name='year'),
    )
    strategy.summary_trades = pd.DataFrame(
        {'All Trades': [4, 55.0]},
        index=['# Trades', 'Win Rate [%]'],
    )
    strategy._trades = pd.DataFrame()
    strategy._transactions = pd.DataFrame()
    return strategy


class ReportFormattingTests(unittest.TestCase):
    def test_format_summary_marks_sharpe_ratio_row(self):
        summary_df = pd.DataFrame(
            {
                'Strategy': [0.12, 1.34],
                'Benchmark': [0.08, 0.91],
            },
            index=['Return [%]', 'Sharpe Ratio'],
        )

        summary_html_str = _format_summary(summary_df)

        self.assertIn('summary-row-sharpe', summary_html_str)
        self.assertIn('metric-sharpe', summary_html_str)
        self.assertIn('Sharpe Ratio', summary_html_str)

    def test_prepare_daily_return_distribution_excludes_bootstrap_and_preserves_formulas(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03])

        distribution_dict = _prepare_daily_return_distribution_dict(strategy)
        daily_return_ser = distribution_dict['daily_return_ser']
        return_vec = distribution_dict['return_vec']

        self.assertEqual(len(daily_return_ser), 4)
        self.assertAlmostEqual(float(daily_return_ser.iloc[0]), 0.01)
        np.testing.assert_allclose(return_vec, np.array([0.01, -0.02, 0.0, 0.03]))
        self.assertAlmostEqual(distribution_dict['mean_return_float'], 0.005)
        self.assertAlmostEqual(distribution_dict['std_return_float'], np.sqrt(0.0013 / 3.0))
        self.assertAlmostEqual(distribution_dict['skew_return_float'], 0.0, places=12)
        self.assertAlmostEqual(distribution_dict['negative_rate_float'], 0.25)

    def test_build_html_includes_daily_return_distribution_and_existing_sections(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')
        trade_statistics_idx_int = report_html_str.index('<h2>Trade Statistics</h2>')
        daily_distribution_idx_int = report_html_str.index('<h2>Daily Return Distribution</h2>')
        closed_trades_idx_int = report_html_str.index('<h2>Closed Trades</h2>')

        self.assertIn('<h2>Daily Return Distribution</h2>', report_html_str)
        self.assertIn('alt="Daily Return Distribution"', report_html_str)
        self.assertIn('Mean</th><th>Std. Dev.</th><th>Skew</th><th>Negative Days</th>', report_html_str)
        self.assertIn('<h2>Performance Summary</h2>', report_html_str)
        self.assertIn('<h2>Monthly Returns</h2>', report_html_str)
        self.assertIn('<h2>Trade Statistics</h2>', report_html_str)
        self.assertIn('<h2>Closed Trades</h2>', report_html_str)
        self.assertIn('<h2>All Transactions</h2>', report_html_str)
        self.assertLess(trade_statistics_idx_int, daily_distribution_idx_int)
        self.assertLess(daily_distribution_idx_int, closed_trades_idx_int)

    def test_daily_return_histogram_uses_60_equal_width_bins(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        distribution_dict = _prepare_daily_return_distribution_dict(strategy)

        with mock.patch('alpha.engine.report.np.linspace', wraps=np.linspace) as linspace_mock_obj:
            histogram_b64 = _daily_return_histogram_b64(distribution_dict)

        self.assertIsInstance(histogram_b64, str)
        self.assertGreater(len(histogram_b64), 0)
        histogram_edge_count_list = [call_args.args[2] for call_args in linspace_mock_obj.call_args_list if len(call_args.args) >= 3]
        self.assertIn(_DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT + 1, histogram_edge_count_list)

    def test_daily_return_distribution_falls_back_when_variation_is_degenerate(self):
        strategy = make_strategy([0.0, 0.0, 0.0])

        distribution_html_str = _build_daily_return_distribution_html(strategy)

        self.assertIn('Daily Return Distribution', distribution_html_str)
        self.assertIn('Not enough realized daily return variation', distribution_html_str)
        self.assertNotIn('data:image/png;base64,', distribution_html_str)


if __name__ == '__main__':
    unittest.main()
