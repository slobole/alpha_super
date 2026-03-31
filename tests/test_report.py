import unittest
import tempfile
from unittest import mock
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.axes._axes as maxes
import numpy as np
import pandas as pd

from alpha.engine.report import (
    _DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT,
    _build_daily_return_distribution_html,
    _build_html,
    _build_portfolio_html,
    _corr_color,
    _daily_return_histogram_b64,
    _format_summary,
    _prepare_daily_return_distribution_dict,
    _prepare_trade_distribution_dict,
    _ret_color,
    _trade_return_histogram_b64,
    save_results,
)
from alpha.engine.portfolio import Portfolio
from alpha.engine.strategy import Strategy
from alpha.engine.theme import SEABORN_DEEP_COLOR_LIST, SIGNATURE_PALETTE_DICT, blend_hex_color_str


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


def make_trade_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'return': [0.02, -0.03, 0.015, -0.01],
            'duration': ['2 days', '5 days', '1 days', '3 days'],
        }
    )


def make_portfolio() -> Portfolio:
    strategy_a = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
    strategy_a.name = 'PodA'
    strategy_b = make_strategy([0.0, -0.01, 0.02, 0.01, -0.01, 0.02])
    strategy_b.name = 'PodB'
    return Portfolio(
        strategies=[strategy_a, strategy_b],
        weights=[0.4, 0.6],
        capital_base=100_000.0,
    )


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
        self.assertNotIn('<h2>All Transactions</h2>', report_html_str)
        self.assertLess(trade_statistics_idx_int, daily_distribution_idx_int)
        self.assertLess(daily_distribution_idx_int, closed_trades_idx_int)

    def test_build_html_embeds_signature_css(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')

        self.assertIn(f'--color-strategy: {SIGNATURE_PALETTE_DICT["strategy"]};', report_html_str)
        self.assertIn(f'--color-page: {SIGNATURE_PALETTE_DICT["page"]};', report_html_str)
        self.assertIn('font-family: "Segoe UI", Arial, sans-serif;', report_html_str)
        self.assertIn('box-shadow: none;', report_html_str)

    def test_save_results_writes_transactions_csv(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        strategy._transactions = pd.DataFrame(
            [
                {
                    'trade_id': 1,
                    'bar': pd.Timestamp('2024-01-03'),
                    'asset': 'AAA',
                    'amount': 10,
                    'price': 101.5,
                    'total_value': 1015.0,
                    'order_id': 11,
                    'commission': 1.0,
                },
                {
                    'trade_id': 2,
                    'bar': pd.Timestamp('2024-01-04'),
                    'asset': 'BBB',
                    'amount': -5,
                    'price': 99.0,
                    'total_value': -495.0,
                    'order_id': 12,
                    'commission': 1.0,
                },
            ]
        )

        def write_fake_chart(*, save_to):
            save_to.write(b'fake-chart-bytes')

        strategy.plot = mock.Mock(side_effect=write_fake_chart)
        strategy.to_pickle = mock.Mock(side_effect=lambda path: Path(path).write_bytes(b'pickle-bytes'))

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = save_results(strategy, output_dir=temp_dir_str)

            transaction_csv_path = output_path / 'transactions.csv'
            report_html_path = output_path / 'report.html'

            self.assertTrue(transaction_csv_path.exists())
            transaction_df = pd.read_csv(transaction_csv_path)
            self.assertEqual(len(transaction_df), 2)
            self.assertListEqual(list(transaction_df.columns), list(strategy._transactions.columns))
            self.assertEqual(transaction_df.loc[0, 'asset'], 'AAA')
            self.assertEqual(int(transaction_df.loc[1, 'amount']), -5)

            report_html_str = report_html_path.read_text(encoding='utf-8')
            self.assertNotIn('<h2>All Transactions</h2>', report_html_str)

    def test_daily_return_histogram_uses_60_equal_width_bins(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        distribution_dict = _prepare_daily_return_distribution_dict(strategy)

        with mock.patch('alpha.engine.report.np.linspace', wraps=np.linspace) as linspace_mock_obj:
            histogram_b64 = _daily_return_histogram_b64(distribution_dict)

        self.assertIsInstance(histogram_b64, str)
        self.assertGreater(len(histogram_b64), 0)
        histogram_edge_count_list = [call_args.args[2] for call_args in linspace_mock_obj.call_args_list if len(call_args.args) >= 3]
        self.assertIn(_DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT + 1, histogram_edge_count_list)

    def test_daily_return_histogram_uses_signature_palette(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        distribution_dict = _prepare_daily_return_distribution_dict(strategy)

        with mock.patch('matplotlib.axes._axes.Axes.hist', autospec=True, wraps=maxes.Axes.hist) as hist_mock_obj:
            histogram_b64 = _daily_return_histogram_b64(distribution_dict)

        self.assertIsInstance(histogram_b64, str)
        hist_color_list = [
            call_args.kwargs['color']
            for call_args in hist_mock_obj.call_args_list
            if 'color' in call_args.kwargs
        ]
        self.assertIn(SEABORN_DEEP_COLOR_LIST[0], hist_color_list)

    def test_trade_return_histogram_uses_signature_palette(self):
        distribution_dict = _prepare_trade_distribution_dict(make_trade_df())

        with mock.patch('matplotlib.axes._axes.Axes.hist', autospec=True, wraps=maxes.Axes.hist) as hist_mock_obj:
            histogram_b64 = _trade_return_histogram_b64(distribution_dict)

        self.assertIsInstance(histogram_b64, str)
        hist_color_list = [
            call_args.kwargs['color']
            for call_args in hist_mock_obj.call_args_list
            if 'color' in call_args.kwargs
        ]
        self.assertIn(SEABORN_DEEP_COLOR_LIST[0], hist_color_list)
        self.assertIn(SEABORN_DEEP_COLOR_LIST[1], hist_color_list)

    def test_signature_color_helpers_follow_palette(self):
        positive_style_str = _ret_color(0.15)
        negative_style_str = _ret_color(-0.15)
        low_corr_style_str = _corr_color(0.0)
        high_corr_style_str = _corr_color(1.0)

        expected_positive_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['strategy'],
            0.12 + 0.45 * min(abs(0.15) / 0.30, 1.0),
        )
        expected_negative_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['benchmark'],
            0.12 + 0.45 * min(abs(-0.15) / 0.30, 1.0),
        )
        expected_low_corr_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['strategy'],
            0.30,
        )
        expected_high_corr_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['benchmark'],
            0.52,
        )

        self.assertIn(expected_positive_color_str, positive_style_str)
        self.assertIn(expected_negative_color_str, negative_style_str)
        self.assertIn(expected_low_corr_color_str, low_corr_style_str)
        self.assertIn(expected_high_corr_color_str, high_corr_style_str)

    def test_daily_return_distribution_falls_back_when_variation_is_degenerate(self):
        strategy = make_strategy([0.0, 0.0, 0.0])

        distribution_html_str = _build_daily_return_distribution_html(strategy)

        self.assertIn('Daily Return Distribution', distribution_html_str)
        self.assertIn('Not enough realized daily return variation', distribution_html_str)
        self.assertNotIn('data:image/png;base64,', distribution_html_str)

    def test_build_portfolio_html_includes_pod_drift_sections_and_labels(self):
        portfolio = make_portfolio()

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')

        self.assertIn('<h2>Pod Drift Diagnostics</h2>', report_html_str)
        self.assertIn('Actual Sleeve Weights', report_html_str)
        self.assertIn('Sleeve Equity Contributions', report_html_str)
        self.assertIn('Rolling 63-Day Pairwise Correlations', report_html_str)
        self.assertIn('Rolling 63-Day Diversification Ratio', report_html_str)
        self.assertIn('<h2>Pooled Pod Trade Statistics</h2>', report_html_str)
        self.assertIn('Allocated Sleeve Summary', report_html_str)
        self.assertIn('Standalone Pod Summary', report_html_str)
        self.assertIn('Common Overlap Window', report_html_str)


if __name__ == '__main__':
    unittest.main()
