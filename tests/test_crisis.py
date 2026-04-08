import tempfile
import unittest

import pandas as pd

from alpha.engine.crisis import (
    CrisisAnalyzer,
    CrisisPeriodConfig,
    CrisisStrategySpec,
    resolve_crisis_window,
    run_crisis_replay_suite,
)
from alpha.engine.report import _format_crisis_metric_table_html, save_crisis_replay_results
from alpha.engine.strategy import Strategy


class BuyAndHoldOneShareStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return
        if self.get_position('AAA') != 0:
            return
        self.order('AAA', 1, trade_id=1)


def build_pricing_data_df() -> pd.DataFrame:
    calendar_idx = pd.DatetimeIndex(
        ['2020-02-19', '2020-02-20', '2020-02-21', '2020-02-24']
    )
    pricing_data_df = pd.DataFrame(
        {
            ('AAA', 'Open'): [10.0, 10.0, 8.0, 12.0],
            ('AAA', 'High'): [10.5, 10.5, 8.5, 12.5],
            ('AAA', 'Low'): [9.5, 9.5, 7.5, 11.5],
            ('AAA', 'Close'): [10.0, 10.0, 8.0, 12.0],
            ('$SPX', 'Open'): [100.0, 100.0, 90.0, 95.0],
            ('$SPX', 'High'): [101.0, 101.0, 91.0, 96.0],
            ('$SPX', 'Low'): [99.0, 99.0, 89.0, 94.0],
            ('$SPX', 'Close'): [100.0, 100.0, 90.0, 95.0],
        },
        index=calendar_idx,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def load_buy_hold_context_dict() -> dict[str, object]:
    pricing_data_df = build_pricing_data_df()
    return {
        'strategy_name_str': 'ToyBuyHold',
        'capital_base_float': 100.0,
        'benchmark_list': ['$SPX'],
        'pricing_data_df': pricing_data_df,
        'calendar_idx': pricing_data_df.index,
    }


def build_buy_hold_strategy_obj(context_dict: dict[str, object]) -> Strategy:
    return BuyAndHoldOneShareStrategy(
        name=str(context_dict['strategy_name_str']),
        benchmarks=list(context_dict['benchmark_list']),
        capital_base=float(context_dict['capital_base_float']),
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )


BUY_HOLD_SPEC = CrisisStrategySpec(
    strategy_key_str='toy_buy_hold',
    load_context_fn=load_buy_hold_context_dict,
    build_strategy_fn=build_buy_hold_strategy_obj,
)

class CrisisReplayTests(unittest.TestCase):
    def test_supported_strategy_keys_include_btal_fallback_tqqq_vix_cash_variant(self):
        self.assertIn(
            'strategy_taa_df_btal_fallback_tqqq_vix_cash',
            CrisisAnalyzer.supported_strategy_key_tuple(),
        )
        self.assertIn(
            'strategy_mo_atr_normalized_ndx',
            CrisisAnalyzer.supported_strategy_key_tuple(),
        )

    def test_crisis_analyzer_class_api_runs_suite(self):
        crisis_analyzer = CrisisAnalyzer(
            strategy_spec_obj=BUY_HOLD_SPEC,
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str='toy_crisis',
                    start_date_str='2020-02-20',
                    end_date_str='2020-02-24',
                )
            ],
            save_output_bool=False,
        )

        crisis_replay_result = crisis_analyzer.run()

        self.assertEqual(crisis_analyzer.latest_result_obj, crisis_replay_result)
        self.assertEqual(crisis_replay_result.strategy_key_str, 'toy_buy_hold')
        self.assertEqual(len(crisis_replay_result.crisis_metric_df), 1)

    def test_resolve_crisis_window_snaps_911_closure_to_september_17_2001(self):
        calendar_idx = pd.DatetimeIndex(
            ['2001-09-10', '2001-09-17', '2001-09-18', '2001-10-11']
        )
        crisis_period_config = CrisisPeriodConfig(
            crisis_name_str='911_aftermath',
            start_date_str='2001-09-11',
            end_date_str='2001-10-11',
        )

        effective_start_ts, effective_end_ts, skip_reason_str = resolve_crisis_window(
            crisis_period_config=crisis_period_config,
            calendar_idx=calendar_idx,
        )

        self.assertEqual(skip_reason_str, '')
        self.assertEqual(effective_start_ts, pd.Timestamp('2001-09-17'))
        self.assertEqual(effective_end_ts, pd.Timestamp('2001-10-11'))

    def test_resolve_crisis_window_snaps_february_1_2025_to_february_3_2025(self):
        calendar_idx = pd.DatetimeIndex(['2025-01-31', '2025-02-03', '2025-04-30'])
        crisis_period_config = CrisisPeriodConfig(
            crisis_name_str='trump_tariffs_2025',
            start_date_str='2025-02-01',
            end_date_str='2025-04-30',
        )

        effective_start_ts, effective_end_ts, skip_reason_str = resolve_crisis_window(
            crisis_period_config=crisis_period_config,
            calendar_idx=calendar_idx,
        )

        self.assertEqual(skip_reason_str, '')
        self.assertEqual(effective_start_ts, pd.Timestamp('2025-02-03'))
        self.assertEqual(effective_end_ts, pd.Timestamp('2025-04-30'))

    def test_crisis_replay_starts_fresh_and_marks_open_trade_without_forced_exit(self):
        crisis_period_list = [
            CrisisPeriodConfig(
                crisis_name_str='toy_crisis',
                start_date_str='2020-02-20',
                end_date_str='2020-02-24',
            )
        ]

        crisis_replay_result = run_crisis_replay_suite(
            crisis_period_list=crisis_period_list,
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )
        strategy_obj = crisis_replay_result.crisis_strategy_map['toy_crisis']
        transaction_df = strategy_obj.get_transactions()

        self.assertEqual(strategy_obj.results.index.min(), pd.Timestamp('2020-02-20'))
        self.assertEqual(len(transaction_df), 1)
        self.assertEqual(pd.Timestamp(transaction_df.iloc[0]['bar']), pd.Timestamp('2020-02-20'))
        self.assertEqual(len(strategy_obj._open_trades), 1)
        self.assertEqual(int(strategy_obj.get_position('AAA')), 1)

    def test_unsupported_periods_are_silently_dropped_from_metric_table(self):
        crisis_period_list = [
            CrisisPeriodConfig(
                crisis_name_str='supported_window',
                start_date_str='2020-02-20',
                end_date_str='2020-02-24',
            ),
            CrisisPeriodConfig(
                crisis_name_str='unsupported_window',
                start_date_str='2025-02-01',
                end_date_str='2025-04-30',
            ),
        ]

        crisis_replay_result = run_crisis_replay_suite(
            crisis_period_list=crisis_period_list,
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        self.assertEqual(len(crisis_replay_result.crisis_metric_df), 1)
        self.assertEqual(len(crisis_replay_result.supported_crisis_df), 1)
        self.assertEqual(len(crisis_replay_result.unsupported_crisis_df), 0)
        self.assertEqual(
            crisis_replay_result.crisis_metric_df.iloc[0]['crisis_name_str'],
            'supported_window',
        )
        self.assertNotIn('supported_bool', crisis_replay_result.crisis_metric_df.columns)
        self.assertNotIn('skip_reason_str', crisis_replay_result.crisis_metric_df.columns)
        self.assertNotIn('requested_start_ts', crisis_replay_result.crisis_metric_df.columns)
        self.assertNotIn('requested_end_ts', crisis_replay_result.crisis_metric_df.columns)

    def test_normalized_crisis_curves_start_at_one(self):
        crisis_replay_result = run_crisis_replay_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str='toy_crisis',
                    start_date_str='2020-02-20',
                    end_date_str='2020-02-24',
                )
            ],
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        crisis_path_df = crisis_replay_result.crisis_path_df[
            crisis_replay_result.crisis_path_df['crisis_name_str'] == 'toy_crisis'
        ].copy()
        first_row_ser = crisis_path_df.sort_values('bar_offset_int').iloc[0]

        self.assertEqual(int(first_row_ser['bar_offset_int']), 0)
        self.assertAlmostEqual(float(first_row_ser['normalized_strategy_equity_float']), 1.0)
        self.assertAlmostEqual(float(first_row_ser['normalized_benchmark_equity_float']), 1.0)

    def test_crisis_metrics_match_synthetic_path_formulas(self):
        crisis_replay_result = run_crisis_replay_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str='toy_crisis',
                    start_date_str='2020-02-20',
                    end_date_str='2020-02-24',
                )
            ],
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        metric_row_ser = crisis_replay_result.crisis_metric_df.iloc[0]
        self.assertAlmostEqual(float(metric_row_ser['strategy_return_pct_float']), 2.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser['benchmark_return_pct_float']), -5.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser['relative_return_pct_float']), 7.0, places=12)
        self.assertAlmostEqual(float(metric_row_ser['max_drawdown_pct_float']), -2.0, places=12)
        self.assertEqual(int(metric_row_ser['trade_count_int']), 1)

    def test_crisis_metric_table_keeps_non_negative_relative_return_green(self):
        crisis_metric_df = pd.DataFrame(
            [
                {
                    'crisis_name_str': 'flat_relative',
                    'effective_start_ts': pd.Timestamp('2020-02-20'),
                    'effective_end_ts': pd.Timestamp('2020-02-24'),
                    'strategy_return_pct_float': 1.0,
                    'benchmark_return_pct_float': 1.0,
                    'relative_return_pct_float': 0.0,
                    'max_drawdown_pct_float': -2.0,
                    'volatility_ann_pct_float': 12.0,
                    'sharpe_ratio_float': 0.5,
                    'trade_count_int': 1,
                },
                {
                    'crisis_name_str': 'negative_relative',
                    'effective_start_ts': pd.Timestamp('2020-03-01'),
                    'effective_end_ts': pd.Timestamp('2020-03-10'),
                    'strategy_return_pct_float': -3.0,
                    'benchmark_return_pct_float': -1.75,
                    'relative_return_pct_float': -1.25,
                    'max_drawdown_pct_float': -4.0,
                    'volatility_ann_pct_float': 18.0,
                    'sharpe_ratio_float': -0.3,
                    'trade_count_int': 2,
                },
            ]
        )

        crisis_table_html_str = _format_crisis_metric_table_html(crisis_metric_df)

        self.assertIn('<td class="pos">+0.00%</td>', crisis_table_html_str)
        self.assertIn('<td class="neg">-1.25%</td>', crisis_table_html_str)

    def test_save_crisis_replay_results_writes_expected_artifacts(self):
        crisis_replay_result = run_crisis_replay_suite(
            crisis_period_list=[
                CrisisPeriodConfig(
                    crisis_name_str='toy_crisis',
                    start_date_str='2020-02-20',
                    end_date_str='2020-02-24',
                )
            ],
            strategy_spec_obj=BUY_HOLD_SPEC,
            save_output_bool=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = save_crisis_replay_results(
                crisis_replay_result,
                output_dir=temp_dir_str,
            )

            self.assertTrue((output_path / 'crisis_metrics.csv').exists())
            self.assertTrue((output_path / 'crisis_paths.csv').exists())
            self.assertTrue((output_path / 'metadata.json').exists())
            self.assertTrue((output_path / 'report.html').exists())
            report_html_str = (output_path / 'report.html').read_text(encoding='utf-8')
            self.assertIn('<h2>Crisis Summary</h2>', report_html_str)
            self.assertIn('toy_crisis', report_html_str)
            self.assertIn('class="crisis-chart-grid"', report_html_str)
            self.assertIn('<td class="pos">+7.00%</td>', report_html_str)
            self.assertNotIn('Crisis Windows', report_html_str)
            self.assertNotIn('Unsupported Crisis Windows', report_html_str)
            self.assertNotIn('Skipped', report_html_str)
            self.assertNotIn('Bars Since Crisis Start', report_html_str)


if __name__ == '__main__':
    unittest.main()
