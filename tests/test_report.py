import unittest
import tempfile
import json
import re
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
    _build_headline_delta_table_html,
    _build_html,
    _display_metric_dict_for_value_ser,
    _build_portfolio_html,
    _corr_color,
    _daily_return_histogram_b64,
    _drawdown_color,
    _format_portfolio_summary,
    _format_summary,
    _format_trades,
    _monthly_returns_html,
    _pm_allocation_snapshot_df,
    _prepare_daily_return_distribution_dict,
    _prepare_trade_distribution_dict,
    _ret_color,
    _strategy_metadata_dict,
    _trade_return_histogram_b64,
    _weight_color_for_asset,
    save_portfolio_results,
    save_results,
)
from alpha.engine.portfolio import Portfolio
from alpha.engine.strategy import Strategy
from alpha.engine.theme import (
    SEABORN_DEEP_COLOR_LIST,
    SIGNATURE_PALETTE_DICT,
    blend_hex_color_str,
    build_report_css,
)


class DummyStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def make_strategy(daily_returns_list: list[float]) -> DummyStrategy:
    date_index = pd.date_range('2024-01-02', periods=len(daily_returns_list), freq='D')
    daily_return_ser = pd.Series(daily_returns_list, index=date_index, dtype=float)
    total_value_ser = 100_000.0 * (1.0 + daily_return_ser).cumprod()
    benchmark_daily_return_ser = daily_return_ser.mul(0.60)
    benchmark_total_value_ser = 100_000.0 * (1.0 + benchmark_daily_return_ser).cumprod()

    strategy = DummyStrategy(
        name='ReportStrategy',
        benchmarks=['$SPX'],
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
            '$SPX': benchmark_total_value_ser,
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
                7.42,
                6.11,
                11.58,
                1.23,
                -8.40,
            ]
        },
        index=[
            'Start',
            'End',
            'Start [$]',
            'Final [$]',
            'Return [%]',
            'Return (Ann.) [%]',
            'Volatility (Ann.) [%]',
            'Sharpe Ratio',
            'Max. Drawdown [%]',
        ],
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
    def test_format_summary_keeps_sharpe_ratio_row_neutral(self):
        summary_df = pd.DataFrame(
            {
                'Strategy': [0.12, 1.34],
                'Benchmark': [0.08, 0.91],
            },
            index=['Return [%]', 'Sharpe Ratio'],
        )

        summary_html_str = _format_summary(summary_df)

        self.assertNotIn('summary-row-sharpe', summary_html_str)
        self.assertNotIn('metric-sharpe', summary_html_str)
        self.assertIn('Sharpe Ratio', summary_html_str)

    def test_format_summary_keeps_drawdown_cells_neutral(self):
        summary_df = pd.DataFrame(
            {'Strategy': [-8.4]},
            index=['Max. Drawdown [%]'],
        )

        summary_html_str = _format_summary(summary_df)

        self.assertNotIn('class="drawdown"', summary_html_str)
        self.assertIn('<td>-8.40%</td>', summary_html_str)

    def test_format_summary_adds_help_for_non_trivial_metrics(self):
        summary_df = pd.DataFrame(
            {'Strategy': [0.42, 100_000.0]},
            index=['Beta', 'Start [$]'],
        )

        summary_html_str = _format_summary(summary_df)

        self.assertIn('<button type="button" class="metric-help"', summary_html_str)
        self.assertIn('How much benchmark exposure', summary_html_str)
        self.assertNotIn('role="button"', summary_html_str)
        self.assertIn('aria-label="Beta: How much benchmark exposure', summary_html_str)
        self.assertIn('data-help="How much benchmark exposure', summary_html_str)
        self.assertNotIn('title="How much benchmark exposure', summary_html_str)
        self.assertNotIn('Start [$] <button', summary_html_str)

    def test_format_portfolio_summary_groups_rows_and_keeps_drawdown_counts_visible(self):
        summary_df = pd.DataFrame(
            {'Portfolio': [100_000.0, 12.0, 1.1, 0.4, 4.2, 1.3, 0.18, 100.0, -8.0, 5, 2, 0.6, 1.0, 20.0]},
            index=[
                'Start [$]',
                'Return (Ann.) [%]',
                'Sharpe Ratio',
                'Beta',
                'Alpha (Ann.) [%]',
                'Alpha HAC t-stat',
                'R²',
                'Exposure Time [%]',
                'Max. Drawdown [%]',
                '# Drawdowns',
                '# Drawdowns / year',
                'Avg. Drawdown [%]',
                'Correlation',
                'Exposure-Adjusted Return (Ann.) [%]',
            ],
        )

        summary_html_str = _format_portfolio_summary(summary_df)

        self.assertIn('Period &amp; Capital', summary_html_str)
        self.assertIn('Return &amp; Risk-Adjusted Performance', summary_html_str)
        self.assertIn('Benchmark Regression', summary_html_str)
        for regression_metric_name_str in ('Beta', 'Alpha (Ann.) / HAC t-stat', 'R²'):
            self.assertEqual(
                summary_html_str.count(f'<td class="metric">{regression_metric_name_str} '),
                1,
            )
        self.assertNotIn('<td class="metric">Alpha HAC t-stat ', summary_html_str)
        self.assertNotIn('<h3>Other Metrics</h3>', summary_html_str)
        self.assertIn('Drawdown &amp; Recovery', summary_html_str)
        self.assertEqual(summary_html_str.count('<td class="metric"># Drawdowns'), 2)
        self.assertIn('<td class="metric"># Drawdowns / year ', summary_html_str)
        self.assertIn('<h3>Extended Risk Diagnostics</h3>', summary_html_str)
        self.assertNotIn('<td class="metric">Correlation ', summary_html_str)
        self.assertNotIn('Exposure-Adjusted Return (Ann.)', summary_html_str)

    def test_benchmark_regression_section_renders_explicit_unavailable_reason(self):
        summary_df = pd.DataFrame(
            {'Strategy': [np.nan, np.nan, np.nan, np.nan]},
            index=['Beta', 'Alpha (Ann.) [%]', 'Alpha HAC t-stat', 'R²'],
        )

        summary_html_str = _format_portfolio_summary(
            summary_df,
            {
                'Strategy': {
                    'status_str': 'unavailable',
                    'reason_str': 'insufficient_observations',
                    'benchmark_label_str': '$SPX',
                    'benchmark_adjustment_str': 'TOTALRETURN',
                    'observation_count_int': 120,
                    'hac_max_lag_int': None,
                }
            },
        )

        self.assertIn('$SPX · TOTALRETURN · N/A: insufficient observations', summary_html_str)
        self.assertEqual(summary_html_str.count('>N/A</td>'), 3)

    def test_format_portfolio_summary_preserves_unknown_metrics_in_other_section(self):
        summary_df = pd.DataFrame({'Portfolio': [3.14]}, index=['Custom Diagnostic'])

        summary_html_str = _format_portfolio_summary(summary_df)

        self.assertIn('<h3>Other Metrics</h3>', summary_html_str)
        self.assertEqual(summary_html_str.count('Custom Diagnostic'), 1)

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

    def test_monthly_heatmap_uses_gain_and_loss_hues(self):
        """Hue must carry the sign: gains green-side, losses brown-side."""
        from alpha.engine.report import _signature_heatmap_background_str
        from alpha.engine.theme import signature_variant_context

        def rgb_tuple(hex_color_str):
            hex_digit_str = hex_color_str.lstrip('#')
            return tuple(int(hex_digit_str[i:i + 2], 16) for i in (0, 2, 4))

        with signature_variant_context('journal'):
            gain_rgb_tuple = rgb_tuple(_signature_heatmap_background_str(0.10, -0.10, 0.10))
            loss_rgb_tuple = rgb_tuple(_signature_heatmap_background_str(-0.10, -0.10, 0.10))
            flat_color_str = _signature_heatmap_background_str(0.0, -0.10, 0.10)
            panel_color_str = str(SIGNATURE_PALETTE_DICT['panel'])

        # A gain leans green (green channel dominates red); a loss leans warm.
        self.assertGreater(gain_rgb_tuple[1], gain_rgb_tuple[0])
        self.assertGreater(loss_rgb_tuple[0], loss_rgb_tuple[1])
        # Zero is bare paper, so the ramp diverges from the page itself.
        self.assertEqual(flat_color_str, panel_color_str)

    def test_monthly_heatmap_intensity_grows_with_magnitude(self):
        from alpha.engine.report import _signature_heatmap_background_str
        from alpha.engine.theme import signature_variant_context

        def luminance_float(hex_color_str):
            hex_digit_str = hex_color_str.lstrip('#')
            return sum(int(hex_digit_str[i:i + 2], 16) for i in (0, 2, 4)) / 3.0

        with signature_variant_context('journal'):
            small_gain_str = _signature_heatmap_background_str(0.02, -0.10, 0.10)
            big_gain_str = _signature_heatmap_background_str(0.10, -0.10, 0.10)
            small_loss_str = _signature_heatmap_background_str(-0.02, -0.10, 0.10)
            big_loss_str = _signature_heatmap_background_str(-0.10, -0.10, 0.10)

        # Within each side, a larger move is a stronger tint.
        self.assertLess(luminance_float(big_gain_str), luminance_float(small_gain_str))
        self.assertLess(luminance_float(big_loss_str), luminance_float(small_loss_str))

    def test_monthly_heatmap_clamps_out_of_range_returns(self):
        from alpha.engine.report import _signature_heatmap_background_str

        self.assertEqual(
            _signature_heatmap_background_str(-5.0, -0.10, 0.10),
            _signature_heatmap_background_str(-0.10, -0.10, 0.10),
        )
        self.assertEqual(
            _signature_heatmap_background_str(5.0, -0.10, 0.10),
            _signature_heatmap_background_str(0.10, -0.10, 0.10),
        )

    def test_display_metrics_sortino_penalizes_only_downside(self):
        """Sortino must exceed Sharpe when upside moves dominate the variance."""
        bar_date_idx = pd.bdate_range('2024-01-02', periods=260)
        # One big up day and one modest down day repeating: high total volatility,
        # small downside deviation.
        daily_return_vec = np.where(np.arange(260) % 2 == 0, 0.02, -0.005)
        value_ser = pd.Series(
            100_000.0 * np.cumprod(1.0 + daily_return_vec), index=bar_date_idx
        )

        display_metric_dict = _display_metric_dict_for_value_ser(value_ser)
        return_ser = value_ser.pct_change(fill_method=None).dropna()
        year_count_float = len(return_ser) / 252.0
        annualised_return_float = (
            float(value_ser.iloc[-1] / value_ser.iloc[0]) ** (1.0 / year_count_float) - 1.0
        )
        sharpe_float = annualised_return_float / float(return_ser.std() * np.sqrt(252.0))

        self.assertIn('Sortino Ratio', display_metric_dict)
        self.assertGreater(display_metric_dict['Sortino Ratio'], sharpe_float)

    def test_display_metrics_ulcer_index_punishes_prolonged_drawdown(self):
        """A book that stays underwater must score worse than one that recovers."""
        bar_date_idx = pd.bdate_range('2024-01-02', periods=120)
        deep_value_vec = np.concatenate([
            np.full(10, 100.0), np.full(110, 70.0),  # drops and never recovers
        ])
        shallow_value_vec = np.concatenate([
            np.full(10, 100.0), np.full(5, 70.0), np.full(105, 100.0),  # quick recovery
        ])

        deep_ulcer_float = _display_metric_dict_for_value_ser(
            pd.Series(deep_value_vec, index=bar_date_idx)
        )['Ulcer Index']
        shallow_ulcer_float = _display_metric_dict_for_value_ser(
            pd.Series(shallow_value_vec, index=bar_date_idx)
        )['Ulcer Index']

        # Same 30% trough depth; only time underwater differs.
        self.assertGreater(deep_ulcer_float, shallow_ulcer_float)

    def test_build_html_sections_and_removed_distributions(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')
        trade_statistics_idx_int = report_html_str.index('<h3>Trade Statistics</h3>')
        open_trades_idx_int = report_html_str.index('<h2>Open Trades</h2>')
        closed_trades_idx_int = report_html_str.index('<h2>Closed Trades</h2>')

        # The trade and daily return distribution plates were removed.
        self.assertNotIn('<h2>Daily Return Distribution</h2>', report_html_str)
        self.assertNotIn('Trade Return Distribution', report_html_str)
        # Core sections remain in order.
        self.assertLess(trade_statistics_idx_int, open_trades_idx_int)
        self.assertLess(open_trades_idx_int, closed_trades_idx_int)
        # Closed trades are folded behind a summary by default.
        self.assertIn('<summary>Show closed trades</summary>', report_html_str)
        # Redundant trade statistics are trimmed from the display.
        self.assertNotIn('CPC Index', report_html_str)
        self.assertIn('class="kpi-grid"', report_html_str)
        self.assertIn('id="metric-help-tooltip"', report_html_str)
        self.assertIn("trigger.addEventListener('mouseenter'", report_html_str)
        self.assertIn("trigger.addEventListener('focus'", report_html_str)
        self.assertIn("trigger.addEventListener('click'", report_html_str)
        self.assertIn("event.key === 'Escape'", report_html_str)
        self.assertIn(
            "document.documentElement.classList.add('metric-tooltip-js-enabled')",
            report_html_str,
        )
        report_css_str = build_report_css()
        self.assertIn(
            'html:not(.metric-tooltip-js-enabled) .metric-help:hover::after',
            report_css_str,
        )
        self.assertIn('content: attr(data-help);', report_css_str)
        self.assertIn('font-size: 1.18rem;', report_css_str)
        self.assertIn('background: var(--color-neutral);', report_css_str)
        self.assertIn('border-left: 4px solid var(--color-strategy-dark);', report_css_str)
        self.assertIn('color: var(--color-strategy-dark);', report_css_str)
        h2_rule_str = report_css_str.split('h2 {', 1)[1].split('}', 1)[0]
        h3_rule_str = report_css_str.split('\nh3 {', 1)[1].split('}', 1)[0]
        self.assertIn('font-size: 1.18rem;', h2_rule_str)
        self.assertIn('background: var(--color-neutral);', h2_rule_str)
        self.assertIn('border-left: 4px solid var(--color-strategy-dark);', h2_rule_str)
        self.assertIn('font-size: 0.94rem;', h3_rule_str)
        self.assertNotIn('background:', h3_rule_str)
        self.assertNotIn('<div class="kpi-label">Final Value</div>', report_html_str)
        self.assertEqual(report_html_str.count('<div class="kpi-card">'), 5)
        # Headline drops Total Return and ends with Beta.
        self.assertNotIn('<div class="kpi-label">Total Return</div>', report_html_str)
        self.assertIn('<div class="kpi-label">Beta</div>', report_html_str)
        self.assertIn('Volatility', report_html_str)
        self.assertIn('11.58%', report_html_str)
        self.assertIn('class="kpi-value pos">+6.11%</div>', report_html_str)
        self.assertIn('<h2>Performance Summary</h2>', report_html_str)
        self.assertIn('<h2>Monthly Returns</h2>', report_html_str)
        self.assertIn('class="card card-monthly-returns"', report_html_str)
        self.assertIn('SPX Ann Ret', report_html_str)
        self.assertIn('SPX Max DD', report_html_str)
        self.assertIn('SPX Sharpe', report_html_str)
        self.assertIn('class="divider-left"', report_html_str)
        self.assertIn('<h3>Trade Statistics</h3>', report_html_str)
        self.assertIn('<h2>Closed Trades</h2>', report_html_str)
        self.assertNotIn('<h2>All Transactions</h2>', report_html_str)

    def test_build_html_groups_strategy_summary_and_adds_alpha_kpi_when_available(self):
        strategy = make_strategy([0.0, 0.01, -0.005, 0.008, -0.002, 0.004])
        strategy.summary.loc['Beta', 'Strategy'] = 0.75
        strategy.summary.loc['Alpha (Ann.) [%]', 'Strategy'] = 4.2
        strategy.summary.loc['Alpha HAC t-stat', 'Strategy'] = 1.35
        strategy.summary.loc['R²', 'Strategy'] = 0.31
        strategy.benchmark_regression_metadata_by_column_dict = {
            'Strategy': {
                'status_str': 'ok',
                'reason_str': None,
                'benchmark_label_str': '$SPX',
                'observation_count_int': 500,
                'hac_max_lag_int': 5,
            }
        }

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')

        self.assertIn('<h3>Period &amp; Capital</h3>', report_html_str)
        self.assertIn('<h3>Benchmark Regression</h3>', report_html_str)
        self.assertIn('Zero-Rate Market Regression', report_html_str)
        self.assertIn('$SPX · N=500 · HAC L=5', report_html_str)
        self.assertIn('<div class="kpi-label">Alpha (Ann.)</div>', report_html_str)
        self.assertIn('Zero-rate vs $SPX · HAC t=1.35', report_html_str)
        self.assertEqual(report_html_str.count('<div class="kpi-card">'), 6)
        self.assertIn('+4.20% / 1.35', report_html_str)

    def test_strategy_appends_regression_metrics_from_declared_stored_benchmark(self):
        daily_return_list = [0.0] + [0.01 if idx_int % 2 == 0 else -0.006 for idx_int in range(299)]
        strategy = make_strategy(daily_return_list)

        strategy._append_benchmark_regression_metrics()

        metadata_dict = strategy.benchmark_regression_metadata_by_column_dict['Strategy']
        self.assertEqual(metadata_dict['status_str'], 'ok')
        self.assertEqual(metadata_dict['benchmark_label_str'], '$SPX')
        self.assertEqual(metadata_dict['benchmark_adjustment_str'], 'not_declared')
        self.assertEqual(metadata_dict['observation_count_int'], 299)
        self.assertTrue(np.isfinite(float(strategy.summary.loc['Beta', 'Strategy'])))
        self.assertTrue(np.isfinite(float(strategy.summary.loc['R²', 'Strategy'])))

    def test_strategy_declared_benchmark_overrides_first_comparison_benchmark(self):
        date_index = pd.bdate_range('2020-01-02', periods=300)
        declared_benchmark_return_ser = pd.Series(
            np.linspace(-0.01, 0.01, len(date_index)),
            index=date_index,
            dtype=float,
        )
        strategy_return_ser = 0.0001 + 0.8 * declared_benchmark_return_ser
        strategy = DummyStrategy(
            name='DeclaredBenchmarkStrategy',
            benchmarks=['$SPX', '$NDX'],
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            performance_benchmark_symbol_str='$NDX',
            performance_benchmark_adjustment_str='TOTALRETURN',
        )
        strategy.results = pd.DataFrame(
            {
                'daily_returns': pd.concat(
                    [pd.Series([0.0], index=date_index[:1]), strategy_return_ser.iloc[1:]]
                ),
                'total_value': 100_000.0 * (1.0 + strategy_return_ser).cumprod(),
                '$SPX': 100_000.0 * (1.0 - 0.2 * declared_benchmark_return_ser).cumprod(),
                '$NDX': 100_000.0 * (1.0 + declared_benchmark_return_ser).cumprod(),
            },
            index=date_index,
        )
        strategy.summary = pd.DataFrame(columns=['Strategy'])

        strategy._append_benchmark_regression_metrics()

        metadata_dict = strategy.benchmark_regression_metadata_by_column_dict['Strategy']
        self.assertEqual(metadata_dict['benchmark_label_str'], '$NDX')
        self.assertEqual(metadata_dict['benchmark_adjustment_str'], 'TOTALRETURN')
        self.assertAlmostEqual(float(strategy.summary.loc['Beta', 'Strategy']), 0.8, places=10)

    def test_strategy_records_realized_weight_snapshot_after_valuation(self):
        strategy = DummyStrategy(
            name='RealizedWeightStrategy',
            benchmarks=['$SPX'],
            capital_base=100.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        current_bar_ts = pd.Timestamp('2024-02-01')
        strategy.current_bar = current_bar_ts
        strategy.cash = 20.0
        strategy.portfolio_value = 80.0
        strategy.total_value = 100.0
        strategy._position_amount_map = {'AAA': 2.0}

        price_col_index = pd.MultiIndex.from_tuples(
            [
                ('AAA', 'Close'),
                ('$SPX', 'Close'),
            ]
        )
        price_df = pd.DataFrame(
            [[40.0, 100.0]],
            index=pd.DatetimeIndex([current_bar_ts]),
            columns=price_col_index,
        )

        strategy.update_metrics(price_df, current_bar_ts)
        strategy._materialize_realized_weight_df()

        self.assertAlmostEqual(float(strategy.realized_weight_df.loc[current_bar_ts, 'AAA']), 0.80)
        self.assertAlmostEqual(float(strategy.realized_weight_df.loc[current_bar_ts, 'Cash']), 0.20)

    def test_strategy_benchmark_can_use_separate_total_return_data_symbol(self):
        strategy = DummyStrategy(
            name='BenchmarkSourceStrategy',
            benchmarks=['SPY'],
            capital_base=100.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        strategy._benchmark_data_symbol_map_dict = {'SPY': 'SPY_TR'}
        date_index = pd.to_datetime(['2024-01-02', '2024-01-03'])
        price_df = pd.DataFrame(
            {
                ('SPY', 'Close'): [100.0, 90.0],
                ('SPY_TR', 'Close'): [100.0, 110.0],
            },
            index=date_index,
        )
        price_df.columns = pd.MultiIndex.from_tuples(price_df.columns)

        strategy.current_bar = date_index[0]
        strategy.update_metrics(price_df, date_index[0])
        strategy.previous_bar = date_index[0]
        strategy.current_bar = date_index[1]
        strategy.update_metrics(price_df, date_index[0])

        self.assertAlmostEqual(float(strategy.results.iloc[-1]['SPY']), 110.0)
        self.assertAlmostEqual(float(strategy.results.iloc[-1]['total_value']), 100.0)

    def test_strategy_metadata_records_benchmark_and_adjustment_provenance(self):
        strategy = DummyStrategy(
            name='MetadataProvenanceStrategy',
            benchmarks=['SPY'],
            capital_base=100.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        strategy._benchmark_data_symbol_map_dict = {'SPY': 'SPY_TR'}
        strategy._data_adjustment_policy_dict = {
            'execution_and_marks_adjustment_str': 'CAPITALSPECIAL',
            'regime_signal_adjustment_str': 'CAPITALSPECIAL',
            'performance_benchmark_adjustment_str': 'TOTALRETURN',
            'performance_benchmark_data_symbol_str': 'SPY_TR',
        }

        metadata_dict = _strategy_metadata_dict(
            strategy,
            Path('metadata-provenance.pkl'),
        )

        self.assertEqual(
            metadata_dict['benchmark_data_symbol_map'],
            {'SPY': 'SPY_TR'},
        )
        self.assertEqual(
            metadata_dict['data_adjustment_policy'],
            strategy._data_adjustment_policy_dict,
        )

    def test_build_html_includes_recent_taa_target_realized_and_drift_weights(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        rebalance_date_index = pd.to_datetime(
            [
                '2024-01-31',
                '2024-02-29',
                '2024-03-29',
                '2024-04-30',
            ]
        )
        strategy.show_taa_weights_report = True
        strategy.rebalance_weight_df = pd.DataFrame(
            {
                'AAPL': [1.00, 0.60, 0.20, 0.00],
                'TLT': [0.00, 0.30, 0.50, 0.70],
            },
            index=rebalance_date_index,
        )
        strategy.daily_target_weights = strategy.rebalance_weight_df.copy()
        strategy.realized_weight_df = pd.DataFrame(
            {
                'AAPL': [0.58, 0.22, 0.00],
                'TLT': [0.31, 0.47, 0.69],
                'Cash': [0.11, 0.31, 0.31],
            },
            index=rebalance_date_index[1:],
        )

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')

        self.assertIn('Recent TAA Weights - Last 3 Rebalances', report_html_str)
        self.assertNotIn('2024-01-31', report_html_str)
        self.assertIn('2024-02-29', report_html_str)
        self.assertIn('2024-03-29', report_html_str)
        self.assertIn('2024-04-30', report_html_str)
        self.assertIn(
            '<td>2024-02-29</td><td>AAPL</td><td>60.00%</td><td>58.00%</td><td class="neg">-2.00%</td>',
            report_html_str,
        )
        self.assertIn(
            '<td>2024-02-29</td><td>Cash</td><td>10.00%</td><td>11.00%</td><td class="pos">1.00%</td>',
            report_html_str,
        )

    def test_build_html_does_not_show_recent_taa_weights_for_non_taa_strategy(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        strategy.rebalance_weight_df = pd.DataFrame(
            {'AAPL': [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp('2024-02-29')]),
        )
        strategy.daily_target_weights = strategy.rebalance_weight_df.copy()
        strategy.realized_weight_df = pd.DataFrame(
            {'AAPL': [1.0], 'Cash': [0.0]},
            index=pd.DatetimeIndex([pd.Timestamp('2024-02-29')]),
        )

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')

        self.assertNotIn('Recent TAA Weights - Last 3 Rebalances', report_html_str)

    def test_build_html_embeds_signature_css(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])

        report_html_str = _build_html(strategy, chart_b64='equity-chart-b64')

        self.assertIn(f'--color-strategy: {SIGNATURE_PALETTE_DICT["strategy"]};', report_html_str)
        self.assertIn(f'--color-page: {SIGNATURE_PALETTE_DICT["page"]};', report_html_str)
        self.assertIn('font-family: "Atlassian Sans", "Segoe UI", Arial, "DejaVu Sans", sans-serif;', report_html_str)
        self.assertIn('https://ds-cdn.prod-east.frontend.public.atl-paas.net/assets/font-rules/v5/atlassian-fonts.css', report_html_str)
        self.assertIn('.kpi-value.pos {', report_html_str)
        self.assertIn('color: var(--color-profit-dark);', report_html_str)
        self.assertIn('box-shadow: none;', report_html_str)
        self.assertIn('td.metric {', report_html_str)
        self.assertNotIn('td.drawdown {', report_html_str)
        self.assertNotIn('tr:nth-child(even) td', report_html_str)
        self.assertIn('class="report-shell"', report_html_str)

    def test_save_results_writes_transactions_csv(self):
        strategy = make_strategy([0.0, 0.01, -0.02, 0.0, 0.03, -0.01])
        strategy._append_benchmark_regression_metrics()
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
        strategy.previous_bar = pd.Timestamp('2024-01-02')
        strategy.current_bar = pd.Timestamp('2024-01-03')
        strategy._position_amount_map = {'AAA': 10.0}
        dividend_pricing_data_df = pd.DataFrame(
            {
                ('AAA', 'Open'): [100.0, 99.0],
                ('AAA', 'High'): [100.0, 99.0],
                ('AAA', 'Low'): [100.0, 99.0],
                ('AAA', 'Close'): [100.0, 99.0],
                ('AAA', 'Dividend'): [1.0, 0.0],
            },
            index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        )
        strategy._credit_dividend_cash_before_open(dividend_pricing_data_df)

        def write_fake_chart(*, save_to):
            save_to.write(b'fake-chart-bytes')

        strategy.plot = mock.Mock(side_effect=write_fake_chart)
        strategy.to_pickle = mock.Mock(side_effect=lambda path: Path(path).write_bytes(b'pickle-bytes'))

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = save_results(strategy, output_dir=temp_dir_str)

            transaction_csv_path = output_path / 'transactions.csv'
            dividend_ledger_csv_path = output_path / 'dividend_ledger.csv'
            report_html_path = output_path / 'report.html'
            run_info_path = output_path / 'run_info.json'
            summary_json_path = output_path / 'summary.json'
            relative_output_path = output_path.relative_to(Path(temp_dir_str))

            self.assertEqual(relative_output_path.parts[:4], (
                'research',
                'strategy',
                'ReportStrategy',
                'vanilla_backtest',
            ))

            self.assertTrue(transaction_csv_path.exists())
            self.assertTrue(dividend_ledger_csv_path.exists())
            self.assertTrue(run_info_path.exists())
            self.assertTrue(summary_json_path.exists())
            transaction_df = pd.read_csv(transaction_csv_path)
            self.assertEqual(len(transaction_df), 2)
            self.assertListEqual(list(transaction_df.columns), list(strategy._transactions.columns))
            self.assertEqual(transaction_df.loc[0, 'asset'], 'AAA')
            self.assertEqual(int(transaction_df.loc[1, 'amount']), -5)
            dividend_ledger_df = pd.read_csv(dividend_ledger_csv_path)
            self.assertListEqual(
                list(dividend_ledger_df.columns),
                list(strategy.get_dividend_ledger().columns),
            )
            self.assertEqual(len(dividend_ledger_df), 1)
            self.assertEqual(
                float(dividend_ledger_df.loc[0, 'gross_dividend_cash_float']),
                10.0,
            )
            self.assertEqual(
                float(dividend_ledger_df.loc[0, 'withholding_cash_float']),
                2.5,
            )
            self.assertEqual(
                float(dividend_ledger_df.loc[0, 'net_dividend_cash_float']),
                7.5,
            )
            run_info_dict = json.loads(run_info_path.read_text(encoding='utf-8'))
            summary_dict = json.loads(summary_json_path.read_text(encoding='utf-8'))
            self.assertEqual(run_info_dict['entity_type'], 'strategy')
            self.assertEqual(run_info_dict['entity_id'], 'ReportStrategy')
            self.assertEqual(run_info_dict['analysis_type'], 'vanilla_backtest')
            self.assertEqual(run_info_dict['parameters']['capital'], 100_000.0)
            self.assertEqual(summary_dict['final_equity'], float(strategy.results['total_value'].iloc[-1]))
            self.assertEqual(summary_dict['trade_count'], 4)
            regression_metadata_dict = summary_dict['benchmark_regression']['Strategy']
            self.assertEqual(regression_metadata_dict['model_str'], 'zero_rate_daily_ols_hac')
            self.assertEqual(regression_metadata_dict['benchmark_label_str'], '$SPX')
            self.assertEqual(regression_metadata_dict['benchmark_adjustment_str'], 'not_declared')
            self.assertEqual(regression_metadata_dict['observation_count_int'], 5)
            self.assertIsNone(regression_metadata_dict['hac_max_lag_int'])
            self.assertEqual(regression_metadata_dict['status_str'], 'unavailable')
            self.assertEqual(regression_metadata_dict['reason_str'], 'insufficient_observations')
            metadata_dict = json.loads((output_path / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(
                metadata_dict['accounting_policy']['accounting_contract_version_str'],
                'net_dividend_cash_ledger_v2',
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['dividend_policy_str'],
                'explicit_entitlement_transition_cash_no_automatic_reinvestment',
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['dividend_event_count_int'],
                1,
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['dividend_cash_net_total_float'],
                7.5,
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['positive_cash_rate_policy_str'],
                'zero_percent_intentional',
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['negative_cash_financing_policy_str'],
                'not_modeled',
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['current_wired_negative_cash_policy_str'],
                'diagnostic_only',
            )
            self.assertEqual(
                metadata_dict['accounting_policy']['negative_cash_enforcement_str'],
                'reported_not_blocked',
            )

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
            SIGNATURE_PALETTE_DICT['profit'],
            0.12 + 0.45 * min(abs(0.15) / 0.30, 1.0),
        )
        expected_negative_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['loss'],
            0.12 + 0.45 * min(abs(-0.15) / 0.30, 1.0),
        )
        # Correlation deepens from bare paper toward the loss tone: zero is the
        # neutral baseline, only rising correlation is a concentration flag.
        expected_low_corr_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'], SIGNATURE_PALETTE_DICT['loss'], 0.0
        )
        expected_high_corr_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'], SIGNATURE_PALETTE_DICT['loss'], 0.62
        )

        self.assertIn(expected_positive_color_str, positive_style_str)
        self.assertIn(expected_negative_color_str, negative_style_str)
        self.assertIn(expected_low_corr_color_str, low_corr_style_str)
        self.assertIn(expected_high_corr_color_str, high_corr_style_str)

    def test_correlation_shift_aligns_pods_before_subtracting(self):
        """Differencing must match pods by name, not by position."""
        from alpha.engine.report import _build_correlation_shift_html

        class _PortfolioStub:
            pass

        portfolio_stub = _PortfolioStub()
        portfolio_stub.correlation_matrix = pd.DataFrame(
            [[1.0, 0.20], [0.20, 1.0]], index=['pod_a', 'pod_b'], columns=['pod_a', 'pod_b']
        )
        # Same pods, reversed order: a positional subtraction would pair the
        # wrong strategies and report a fabricated shift.
        portfolio_stub.tail_correlation_matrix = pd.DataFrame(
            [[1.0, 0.70], [0.70, 1.0]], index=['pod_b', 'pod_a'], columns=['pod_b', 'pod_a']
        )

        shift_html_str = _build_correlation_shift_html(portfolio_stub)

        self.assertIn('+0.500', shift_html_str)
        self.assertIn('Correlation Shift Under Stress', shift_html_str)

    def test_correlation_shift_colors_convergence_and_decoupling(self):
        from alpha.engine.report import _correlation_shift_color_str

        converging_style_str = _correlation_shift_color_str(0.5)
        decoupling_style_str = _correlation_shift_color_str(-0.5)

        def rgb_tuple(style_str):
            hex_str = style_str.split('background-color: ')[1][:7].lstrip('#')
            return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))

        # Convergence reads warm (loss tone), decoupling reads green.
        self.assertGreater(rgb_tuple(converging_style_str)[0], rgb_tuple(converging_style_str)[1])
        self.assertGreater(rgb_tuple(decoupling_style_str)[1], rgb_tuple(decoupling_style_str)[0])

    def test_correlation_shift_omitted_without_both_matrices(self):
        from alpha.engine.report import _build_correlation_shift_html

        class _PortfolioStub:
            correlation_matrix = pd.DataFrame()
            tail_correlation_matrix = pd.DataFrame()

        self.assertEqual(_build_correlation_shift_html(_PortfolioStub()), '')

    def test_drawdown_color_uses_light_red_palette(self):
        drawdown_style_str = _drawdown_color(-0.12)
        expected_drawdown_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['loss'],
            0.18 + 0.28 * min(abs(-0.12) / 0.30, 1.0),
        )

        self.assertIn(expected_drawdown_color_str, drawdown_style_str)
        self.assertIn(SIGNATURE_PALETTE_DICT['loss_dark'], drawdown_style_str)

    def test_weight_color_helper_maps_fallback_to_benchmark_orange(self):
        self.assertEqual(_weight_color_for_asset('SPY'), SIGNATURE_PALETTE_DICT['benchmark'])
        self.assertEqual(_weight_color_for_asset('TQQQ'), SIGNATURE_PALETTE_DICT['benchmark'])
        self.assertEqual(_weight_color_for_asset('GLD'), '#d9a441')
        self.assertEqual(_weight_color_for_asset('BTAL'), '#c251c0')

    def test_format_trades_uses_green_red_sign_classes_for_profit_and_return(self):
        trade_df = pd.DataFrame(
            {
                'start': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-05'), pd.Timestamp('2024-01-07')],
                'end': [pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-06'), pd.Timestamp('2024-01-08')],
                'capital': [1_000.0, 1_000.0, 1_000.0],
                'profit': [25.0, -10.0, 0.0],
                'return': [0.025, -0.010, 0.0],
            },
            index=pd.Index([11, 12, 13], name='trade_id'),
        )

        trades_html_str = _format_trades(trade_df)

        self.assertIn('<td class="pos">$25.00</td>', trades_html_str)
        self.assertIn('<td class="neg">$-10.00</td>', trades_html_str)
        self.assertIn('<td>$0.00</td>', trades_html_str)
        self.assertIn('<td class="pos">2.50%</td>', trades_html_str)
        self.assertIn('<td class="neg">-1.00%</td>', trades_html_str)
        self.assertIn('<td>0.00%</td>', trades_html_str)

    def test_daily_return_distribution_falls_back_when_variation_is_degenerate(self):
        strategy = make_strategy([0.0, 0.0, 0.0])

        distribution_html_str = _build_daily_return_distribution_html(strategy)

        self.assertIn('Daily Return Distribution', distribution_html_str)
        self.assertIn('Not enough realized daily return variation', distribution_html_str)
        self.assertNotIn('data:image/png;base64,', distribution_html_str)

    def test_build_portfolio_html_includes_pod_drift_sections_and_labels(self):
        portfolio = make_portfolio()

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')
        performance_summary_start_int = report_html_str.index('<h2>Performance Summary</h2>')
        performance_summary_end_int = report_html_str.index('<h2>Portfolio Monthly Returns</h2>' if '<h2>Portfolio Monthly Returns</h2>' in report_html_str else '<h2>Monthly Returns</h2>')
        performance_summary_html_str = report_html_str[
            performance_summary_start_int:performance_summary_end_int
        ]

        self.assertIn('<h2>Pod Drift Diagnostics</h2>', report_html_str)
        self.assertIn('Actual Sleeve Weights', report_html_str)
        self.assertIn('Sleeve Equity Contributions', report_html_str)
        self.assertIn('Rolling 63-Day Pairwise Correlations', report_html_str)
        self.assertIn('Rolling 63-Day Diversification Ratio', report_html_str)
        self.assertIn('class="kpi-grid"', report_html_str)
        self.assertIn('class="card-grid"', report_html_str)
        self.assertIn('<button type="button" class="metric-help"', report_html_str)
        self.assertIn('id="metric-help-tooltip"', report_html_str)
        self.assertIn(
            "document.documentElement.classList.add('metric-tooltip-js-enabled')",
            report_html_str,
        )
        self.assertIn('<h3>Period &amp; Capital</h3>', performance_summary_html_str)
        self.assertIn('<h3>Drawdown &amp; Recovery</h3>', performance_summary_html_str)
        self.assertIn('<h3>Extended Risk Diagnostics</h3>', performance_summary_html_str)
        self.assertIn('Final [$]', performance_summary_html_str)
        self.assertNotIn('<td class="metric">Correlation ', performance_summary_html_str)
        self.assertEqual(report_html_str.count('class="summary-section-stack"'), 3)
        self.assertEqual(
            report_html_str.count('<h3>Allocated Sleeve Performance — PM Window</h3>'),
            2,
        )
        self.assertIn('https://ds-cdn.prod-east.frontend.public.atl-paas.net/assets/font-rules/v5/atlassian-fonts.css', report_html_str)
        self.assertNotIn('SPX Ann Ret', report_html_str)
        self.assertIn('Allocated Sleeve Performance — PM Window', report_html_str)
        self.assertNotIn('Standalone Pod Summary', report_html_str)
        self.assertIn('PM-Window Pod Trade Distribution', report_html_str)
        self.assertNotIn('<h3>Monthly Returns</h3>', report_html_str)
        # This fixture has no completed PM-window trades, so the trade
        # statistics sub-section is correctly absent.
        self.assertNotIn('<h3>Trade Statistics</h3>', report_html_str)
        self.assertIn('Common Overlap Window', report_html_str)
        self.assertIn('<h2>PM Allocation Overview</h2>', report_html_str)
        self.assertIn('Construction Policy', report_html_str)
        self.assertIn('Manual IBKR Transfer Guide', report_html_str)
        self.assertIn('Weight Drift = actual end weight - active target weight', report_html_str)
        self.assertIn('Manual Delta = target capital - current sleeve equity', report_html_str)
        self.assertIn('None (buy-and-hold)', report_html_str)
        portfolio_monthly_index_int = report_html_str.index('<h2>Portfolio Monthly Returns</h2>' if '<h2>Portfolio Monthly Returns</h2>' in report_html_str else '<h2>Monthly Returns</h2>')
        benchmark_monthly_index_int = report_html_str.index(
            '<h2>Benchmark Portfolio Monthly Returns — Benchmark</h2>'
        )
        self.assertLess(portfolio_monthly_index_int, benchmark_monthly_index_int)
        self.assertIn(
            'N/A — PM performance benchmark data is unavailable for this reporting window.',
            report_html_str,
        )

    def test_portfolio_html_does_not_render_standalone_pod_sentinels(self):
        portfolio = make_portfolio()
        for strategy_obj in portfolio.strategies:
            strategy_obj.summary.loc['Standalone Sentinel', 'Strategy'] = 'DO_NOT_RENDER'
            strategy_obj.monthly_returns = pd.DataFrame(
                {'Annual Return': [9.99]},
                index=pd.Index([2004], name='year'),
            )
            strategy_obj.summary_trades = pd.DataFrame(
                {'All Trades': [9999]},
                index=['# Trades'],
            )

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')

        self.assertNotIn('Standalone Sentinel', report_html_str)
        self.assertNotIn('DO_NOT_RENDER', report_html_str)
        self.assertNotIn('<td>2004</td>', report_html_str)
        self.assertNotIn('9.99', report_html_str)
        self.assertNotIn('9999', report_html_str)

    def test_portfolio_provenance_distinguishes_requested_and_effective_pod_starts(self):
        portfolio = make_portfolio()
        portfolio.pod_info_list[0]['requested_backtest_start_date_str'] = '2004-01-01'
        portfolio.pod_info_list[0]['effective_backtest_start_date_str'] = '2018-07-19'

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')

        self.assertIn('<th>Requested Start</th>', report_html_str)
        self.assertIn('<th>Effective Pod Start</th>', report_html_str)
        self.assertIn('<td>2004-01-01</td>', report_html_str)
        self.assertIn('<td>2018-07-19</td>', report_html_str)

    def test_pod_links_use_the_pod_id_not_the_strategy_name(self):
        """The artifact directory is the pod id; linking by strategy name 404s."""
        from alpha.engine.report import _build_pod_report_links_html

        class _PortfolioStub:
            pod_info_list = [
                {
                    'pod_id_str': 'pod_taa',
                    'strategy_name': 'strategy_taa_df_btal_fallback_tqqq',
                    'weight': 0.32,
                    'allocated_capital': 64_000.0,
                },
            ]

        links_html_str = _build_pod_report_links_html(_PortfolioStub())

        self.assertIn('href="pods/pod_taa/report.html"', links_html_str)
        self.assertNotIn('pods/strategy_taa_df_btal_fallback_tqqq/', links_html_str)
        self.assertIn('32%', links_html_str)

    def test_pod_links_omitted_without_pod_metadata(self):
        from alpha.engine.report import _build_pod_report_links_html

        class _PortfolioStub:
            pod_info_list = []

        self.assertEqual(_build_pod_report_links_html(_PortfolioStub()), '')

    def test_plate_index_links_every_plate_to_a_real_anchor(self):
        from alpha.engine.theme import signature_variant_context

        portfolio = make_portfolio()
        # Plates only exist under the spec layout; the card layouts have none.
        with signature_variant_context('journal_spec'):
            report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')

        anchor_id_list = re.findall(r'<div class="plate" id="(plate-\d+)"', report_html_str)
        index_target_list = re.findall(r'<li><a href="#(plate-\d+)"', report_html_str)
        self.assertGreater(len(anchor_id_list), 0)
        # Every index entry points at a plate that exists, and vice versa.
        self.assertEqual(index_target_list, anchor_id_list)

    def test_build_portfolio_html_includes_tail_risk_diagnostics(self):
        portfolio = make_portfolio()

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')

        self.assertIn('<h2>Tail Risk Diagnostics</h2>', report_html_str)
        self.assertIn('Tail Summary By Pod', report_html_str)
        self.assertIn('Worst Portfolio Days - Pod Contributions', report_html_str)
        # Attribution needs no benchmark, but correlation does: without one the
        # report says so rather than showing a self-conditioned estimate.
        self.assertNotIn('Correlation on Benchmark Stress Days', report_html_str)
        self.assertIn('No benchmark is attached to this portfolio', report_html_str)

    def test_build_portfolio_html_reports_stress_correlation_with_a_benchmark(self):
        date_index = pd.bdate_range('2024-01-02', periods=120)
        random_generator = np.random.default_rng(3)
        strategy_a = make_strategy(list(random_generator.normal(0.0, 0.01, len(date_index))))
        strategy_a.name = 'PodA'
        strategy_b = make_strategy(list(random_generator.normal(0.0, 0.01, len(date_index))))
        strategy_b.name = 'PodB'
        benchmark_value_ser = pd.Series(
            100.0 * (1.0 + pd.Series(
                random_generator.normal(0.0, 0.011, len(date_index)), index=date_index
            )).cumprod(),
            index=date_index,
        )

        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100_000.0,
            regression_benchmark_value_ser=benchmark_value_ser,
            regression_benchmark_label_str='BENCH',
        )
        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')

        self.assertIn('Correlation on Benchmark Stress Days', report_html_str)
        self.assertIn('worst benchmark days', report_html_str)
        self.assertGreater(len(portfolio.stress_event_date_index), 0)

    def test_build_portfolio_html_renders_full_benchmark_monthly_returns_card(self):
        dates_index = pd.bdate_range('2024-01-02', periods=40)
        benchmark_return_ser = pd.Series(
            np.linspace(-0.003, 0.004, len(dates_index)),
            index=dates_index,
            dtype=float,
        )
        benchmark_value_ser = 100.0 * (1.0 + benchmark_return_ser).cumprod()
        strategy_a = make_strategy([0.0] + [0.001] * 39)
        strategy_a.name = 'PodA'
        strategy_a.results.index = dates_index
        strategy_b = make_strategy([0.0] + [0.002] * 39)
        strategy_b.name = 'PodB'
        strategy_b.results.index = dates_index
        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100_000.0,
            regression_benchmark_value_ser=benchmark_value_ser,
            regression_benchmark_label_str='$SPX · TOTALRETURN',
            regression_benchmark_adjustment_str='TOTALRETURN',
        )

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')
        benchmark_card_start_int = report_html_str.index(
            '<h2>Benchmark Portfolio Monthly Returns — $SPX · TOTALRETURN</h2>'
        )
        benchmark_card_end_int = report_html_str.index(
            '<h2>Diversification</h2>'
        )
        benchmark_card_html_str = report_html_str[
            benchmark_card_start_int:benchmark_card_end_int
        ]
        portfolio_card_start_int = report_html_str.index('<h2>Portfolio Monthly Returns</h2>' if '<h2>Portfolio Monthly Returns</h2>' in report_html_str else '<h2>Monthly Returns</h2>')
        portfolio_card_html_str = report_html_str[
            portfolio_card_start_int:benchmark_card_start_int
        ]

        for expected_header_str in (
            '<th>Jan</th>',
            '<th>Feb</th>',
            '<th>Annual Return</th>',
            '<th>Max Drawdown</th>',
            '<th>Sharpe Ratio</th>',
        ):
            self.assertIn(expected_header_str, benchmark_card_html_str)
        portfolio_header_html_str = portfolio_card_html_str.split('<thead>', 1)[1].split(
            '</thead>',
            1,
        )[0]
        benchmark_header_html_str = benchmark_card_html_str.split('<thead>', 1)[1].split(
            '</thead>',
            1,
        )[0]
        self.assertEqual(benchmark_header_html_str, portfolio_header_html_str)
        self.assertIn(
            _monthly_returns_html(portfolio.benchmark_monthly_returns),
            benchmark_card_html_str,
        )
        self.assertNotIn(
            _monthly_returns_html(portfolio.monthly_returns),
            benchmark_card_html_str,
        )
        self.assertNotIn('N/A — PM performance benchmark data is unavailable', benchmark_card_html_str)

    def test_build_portfolio_html_includes_pm_rebalance_targets(self):
        strategy_a = make_strategy([0.0] + [0.01, -0.005] * 20)
        strategy_a.name = 'PodA'
        strategy_b = make_strategy([0.0] + [-0.005, 0.01] * 20)
        strategy_b.name = 'PodB'
        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.8, 0.2],
            capital_base=100_000.0,
            rebalance='monthly',
            rebalance_policy_str='equal',
        )

        report_html_str = _build_portfolio_html(portfolio, chart_b64='portfolio-chart-b64')
        allocation_snapshot_df = _pm_allocation_snapshot_df(portfolio)

        self.assertIn('Latest applied rebalance target', report_html_str)
        self.assertIn('Recent PM Rebalances', report_html_str)
        self.assertIn('Target Rebalance Weights', report_html_str)
        self.assertIn('PodA 50.00%; PodB 50.00%', report_html_str)
        self.assertIn('<td>equal</td>', report_html_str)
        self.assertAlmostEqual(float(allocation_snapshot_df['manual_delta_float'].sum()), 0.0)
        self.assertAlmostEqual(
            float(
                allocation_snapshot_df
                .set_index('pod_name_str')
                .loc['PodA', 'target_weight_float']
            ),
            0.5,
        )

    def test_save_portfolio_results_writes_tail_csv_artifacts(self):
        portfolio = make_portfolio()

        def write_fake_chart(*, save_to):
            save_to.write(b'fake-portfolio-chart-bytes')

        portfolio.plot = mock.Mock(side_effect=write_fake_chart)
        portfolio.to_pickle = mock.Mock(side_effect=lambda path: Path(path).write_bytes(b'portfolio-pickle-bytes'))

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = save_portfolio_results(portfolio, output_dir=temp_dir_str)

            tail_summary_csv_path = output_path / 'tail_summary.csv'
            tail_contribution_csv_path = output_path / 'tail_contribution.csv'
            tail_returns_csv_path = output_path / 'tail_returns.csv'
            run_info_path = output_path / 'run_info.json'
            summary_json_path = output_path / 'summary.json'
            relative_output_path = output_path.relative_to(Path(temp_dir_str))

            self.assertEqual(relative_output_path.parts[:4], (
                'research',
                'portfolio',
                'Portfolio',
                'vanilla_backtest',
            ))

            self.assertTrue(tail_summary_csv_path.exists())
            self.assertTrue(tail_contribution_csv_path.exists())
            self.assertTrue(tail_returns_csv_path.exists())
            self.assertTrue(run_info_path.exists())
            self.assertTrue(summary_json_path.exists())

            tail_summary_df = pd.read_csv(tail_summary_csv_path)
            tail_contribution_df = pd.read_csv(tail_contribution_csv_path)
            tail_returns_df = pd.read_csv(tail_returns_csv_path)
            run_info_dict = json.loads(run_info_path.read_text(encoding='utf-8'))
            summary_dict = json.loads(summary_json_path.read_text(encoding='utf-8'))

            self.assertIn('average_tail_return_float', tail_summary_df.columns)
            self.assertIn('PodA', tail_contribution_df.columns)
            self.assertIn('PodB', tail_returns_df.columns)
            self.assertEqual(run_info_dict['entity_type'], 'portfolio')
            self.assertEqual(run_info_dict['entity_id'], 'Portfolio')
            self.assertEqual(run_info_dict['analysis_type'], 'vanilla_backtest')
            self.assertEqual(run_info_dict['parameters']['capital'], 100_000.0)
            self.assertEqual(len(run_info_dict['parameters']['pods']), 2)
            self.assertIn('final_equity', summary_dict)
            portfolio_regression_metadata_dict = summary_dict['benchmark_regression']['Portfolio']
            self.assertEqual(portfolio_regression_metadata_dict['model_str'], 'zero_rate_daily_ols_hac')
            self.assertEqual(portfolio_regression_metadata_dict['status_str'], 'unavailable')
            self.assertEqual(portfolio_regression_metadata_dict['reason_str'], 'missing_benchmark')
            self.assertIn('PodA Sleeve (40%)', summary_dict['benchmark_regression'])
            self.assertIn('PodA Standalone', summary_dict['standalone_benchmark_regression'])

    def test_save_portfolio_results_writes_rebalance_csv_artifacts(self):
        strategy_a = make_strategy([0.0] + [0.01, -0.005] * 20)
        strategy_a.name = 'PodA'
        strategy_b = make_strategy([0.0] + [-0.005, 0.01] * 20)
        strategy_b.name = 'PodB'
        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.8, 0.2],
            capital_base=100_000.0,
            rebalance='monthly',
            rebalance_policy_str='equal',
        )

        def write_fake_chart(*, save_to):
            save_to.write(b'fake-portfolio-chart-bytes')

        portfolio.plot = mock.Mock(side_effect=write_fake_chart)
        portfolio.to_pickle = mock.Mock(side_effect=lambda path: Path(path).write_bytes(b'portfolio-pickle-bytes'))

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = save_portfolio_results(portfolio, output_dir=temp_dir_str)

            target_weight_csv_path = output_path / 'rebalance_target_weights.csv'
            diagnostic_csv_path = output_path / 'rebalance_diagnostics.csv'
            metadata_path = output_path / 'metadata.json'

            self.assertTrue(target_weight_csv_path.exists())
            self.assertTrue(diagnostic_csv_path.exists())

            target_weight_df = pd.read_csv(target_weight_csv_path)
            diagnostic_df = pd.read_csv(diagnostic_csv_path)
            metadata_dict = json.loads(metadata_path.read_text(encoding='utf-8'))

            self.assertIn('PodA', target_weight_df.columns)
            self.assertIn('PodB', target_weight_df.columns)
            self.assertAlmostEqual(float(target_weight_df.iloc[0]['PodA']), 0.5)
            self.assertEqual(diagnostic_df.iloc[0]['policy_str'], 'equal')
            self.assertEqual(metadata_dict['rebalance'], 'monthly')
            self.assertEqual(metadata_dict['rebalance_policy'], 'equal')


if __name__ == '__main__':
    unittest.main()


class HeadlineDeltaTableTests(unittest.TestCase):
    """The specimen sheet's headline is the delta table, not the KPI tiles."""

    @staticmethod
    def _summary_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                'Strategy': {
                    'Return (Ann.) [%]': 18.886,
                    'Volatility (Ann.) [%]': 20.458,
                    'Sharpe Ratio': 0.9548,
                    'Max. Drawdown [%]': -30.914,
                    'Correlation': 1.0,
                },
                '$SPX': {
                    'Return (Ann.) [%]': 8.804,
                    'Volatility (Ann.) [%]': 18.747,
                    'Sharpe Ratio': 0.5442,
                    'Max. Drawdown [%]': -56.775,
                    'Correlation': 0.6825,
                },
            }
        )

    def test_correlation_is_read_from_the_benchmark_column(self):
        """*** CRITICAL*** regression.

        The summary stores Correlation per column as that column's correlation
        to the strategy, so the strategy's own cell is always 1.0. Reading it
        would print 1.00 for every strategy ever run, with a delta of zero.
        """
        table_html_str = _build_headline_delta_table_html(self._summary_df(), 'Strategy')
        correlation_row_str = re.search(
            r'<tr><td class="metric">Correlation</td>(.*?)</tr>', table_html_str
        ).group(1)
        cell_list = re.findall(r'<td[^>]*>([^<]*)</td>', correlation_row_str)
        self.assertEqual(cell_list[0], '0.68')
        self.assertEqual(cell_list[1], '1.00')
        self.assertEqual(cell_list[2], '-0.32')

    def test_a_shallower_drawdown_reads_as_favourable_despite_a_negative_delta(self):
        """Depth is what is compared, so the sign of the delta is not the verdict."""
        table_html_str = _build_headline_delta_table_html(self._summary_df(), 'Strategy')
        drawdown_row_str = re.search(
            r'<tr><td class="metric">Max drawdown</td>(.*?)</tr>', table_html_str
        ).group(1)
        self.assertIn('-25.9pp', drawdown_row_str)
        self.assertIn('--color-profit-dark', drawdown_row_str)

    def test_higher_volatility_reads_as_unfavourable(self):
        table_html_str = _build_headline_delta_table_html(self._summary_df(), 'Strategy')
        volatility_row_str = re.search(
            r'<tr><td class="metric">Volatility</td>(.*?)</tr>', table_html_str
        ).group(1)
        self.assertIn('+1.7pp', volatility_row_str)
        self.assertIn('--color-loss-dark', volatility_row_str)

    def test_every_headline_metric_is_present_in_order(self):
        table_html_str = _build_headline_delta_table_html(self._summary_df(), 'Strategy')
        label_list = re.findall(r'<td class="metric">([^<]+)</td>', table_html_str)
        self.assertEqual(
            label_list,
            ['CAGR (net)', 'Volatility', 'Sharpe ratio', 'Max drawdown', 'Correlation'],
        )

    def test_no_benchmark_column_yields_no_table_so_the_caller_can_fall_back(self):
        """A table of em dashes is worse than the tiles it replaced."""
        strategy_only_df = self._summary_df()[['Strategy']]
        self.assertEqual(_build_headline_delta_table_html(strategy_only_df, 'Strategy'), '')
        self.assertEqual(_build_headline_delta_table_html(None, 'Strategy'), '')

    def test_the_headline_table_agrees_with_the_summary_it_came_from(self):
        """The headline must not drift from the Performance Summary below it."""
        summary_df = self._summary_df()
        table_html_str = _build_headline_delta_table_html(summary_df, 'Strategy')
        self.assertIn('18.9%', table_html_str)
        self.assertIn('8.8%', table_html_str)
        self.assertIn('+10.1pp', table_html_str)
        self.assertIn('0.95', table_html_str)
