import io
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.axes._axes as maxes
import matplotlib.pyplot as plt
import pandas as pd

from alpha.engine.plot import generate_yearly_returns, plot
from alpha.engine.portfolio import Portfolio
from alpha.engine.strategy import Strategy
from alpha.engine.theme import SEABORN_DEEP_COLOR_LIST, SIGNATURE_PALETTE_DICT


class DummyStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def make_strategy(
        name: str,
        dates_index: pd.DatetimeIndex,
        daily_returns_list: list[float],
        capital_base: float = 100.0,
        benchmark_name: str | None = None,
        benchmark_total_value_list: list[float] | None = None,
):
    benchmark_list = [benchmark_name] if benchmark_name is not None else []
    strategy = DummyStrategy(
        name=name,
        benchmarks=benchmark_list,
        capital_base=capital_base,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    daily_returns_ser = pd.Series(daily_returns_list, index=dates_index, dtype=float)
    total_value_ser = capital_base * (1.0 + daily_returns_ser).cumprod()
    results_dict = {
        'daily_returns': daily_returns_ser,
        'total_value': total_value_ser,
        'portfolio_value': total_value_ser,
    }
    if benchmark_name is not None and benchmark_total_value_list is not None:
        benchmark_total_value_ser = pd.Series(benchmark_total_value_list, index=dates_index, dtype=float)
        results_dict[benchmark_name] = benchmark_total_value_ser
    strategy.results = pd.DataFrame(results_dict, index=dates_index)
    strategy.summary = pd.DataFrame()
    strategy.summary_trades = pd.DataFrame()
    return strategy


def _major_tick_by_label_gid(axis_obj, label_id_str: str):
    return next(
        major_tick_obj
        for major_tick_obj in axis_obj.yaxis.get_major_ticks()
        if major_tick_obj.label2.get_gid() == label_id_str
    )


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_plot_without_benchmark_renders_to_buffer(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 105.0, 103.0, 112.0], index=dates_index)
        plot_buffer = io.BytesIO()

        with mock.patch('matplotlib.pyplot.show'):
            plot(strategy_total_value=strategy_total_value_ser, save_to=plot_buffer)

        self.assertGreater(plot_buffer.getbuffer().nbytes, 0)
        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]

        strategy_line_obj = next(line for line in equity_ax.lines if line.get_gid() == 'strategy_equity_line')
        strategy_fill_collection_obj = next(
            collection_obj
            for collection_obj in equity_ax.collections
            if collection_obj.get_gid() == 'strategy_equity_fill'
        )
        legend_obj = equity_ax.get_legend()
        figure_obj.canvas.draw()
        strategy_major_tick_obj = _major_tick_by_label_gid(equity_ax, 'strategy_growth_of_1_label')
        strategy_label_obj = strategy_major_tick_obj.label2
        strategy_axis_tick_obj = strategy_major_tick_obj.tick2line
        self.assertEqual(strategy_line_obj.get_linestyle(), '-')
        self.assertEqual(strategy_line_obj.get_color(), SIGNATURE_PALETTE_DICT['strategy'])
        self.assertEqual(mcolors.to_hex(strategy_axis_tick_obj.get_color()), SIGNATURE_PALETTE_DICT['strategy'])
        self.assertEqual(
            mcolors.to_hex(strategy_fill_collection_obj.get_facecolor()[0]),
            SIGNATURE_PALETTE_DICT['strategy'],
        )
        self.assertAlmostEqual(float(strategy_fill_collection_obj.get_facecolor()[0][3]), 0.24, places=2)
        self.assertAlmostEqual(float(strategy_major_tick_obj.get_loc()), 1.12)
        self.assertEqual(mcolors.to_hex(equity_ax.get_facecolor()), SIGNATURE_PALETTE_DICT['panel'])
        self.assertEqual(mcolors.to_hex(legend_obj.get_frame().get_facecolor()), SIGNATURE_PALETTE_DICT['legend_face'])
        self.assertEqual(mcolors.to_hex(legend_obj.get_frame().get_edgecolor()), SIGNATURE_PALETTE_DICT['legend_edge'])
        self.assertEqual(equity_ax.get_ylabel(), 'Growth of $1 (log scale)')
        self.assertEqual(strategy_label_obj.get_text(), '$1.12')
        self.assertEqual(len(equity_ax.texts), 0)
        self.assertEqual(equity_ax.yaxis.get_major_formatter()(1.5, 0), '$1.50')
        self.assertEqual(equity_ax.yaxis.get_minor_formatter()(1.5, 0), '')

    def test_plot_with_benchmark_uses_solid_benchmark_and_two_labels(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 102.0, 101.0, 110.0], index=dates_index)
        benchmark_total_value_ser = pd.Series([100.0, 101.0, 100.0, 104.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            plot(
                strategy_total_value=strategy_total_value_ser,
                benchmark_total_value=benchmark_total_value_ser,
                save_to=io.BytesIO(),
            )

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        strategy_line_obj = next(line for line in equity_ax.lines if line.get_gid() == 'strategy_equity_line')
        benchmark_line_obj = next(line for line in equity_ax.lines if line.get_gid() == 'benchmark_equity_line')
        figure_obj.canvas.draw()
        strategy_axis_tick_obj = _major_tick_by_label_gid(equity_ax, 'strategy_growth_of_1_label').tick2line
        benchmark_axis_tick_obj = _major_tick_by_label_gid(equity_ax, 'benchmark_growth_of_1_label').tick2line

        self.assertEqual(strategy_line_obj.get_linestyle(), '-')
        self.assertEqual(benchmark_line_obj.get_linestyle(), '-')
        self.assertEqual(strategy_line_obj.get_color(), SIGNATURE_PALETTE_DICT['strategy'])
        self.assertEqual(benchmark_line_obj.get_color(), SIGNATURE_PALETTE_DICT['benchmark'])
        self.assertEqual(mcolors.to_hex(strategy_axis_tick_obj.get_color()), SIGNATURE_PALETTE_DICT['strategy'])
        self.assertEqual(mcolors.to_hex(benchmark_axis_tick_obj.get_color()), SIGNATURE_PALETTE_DICT['benchmark'])
        self.assertEqual(
            {
                major_tick_obj.label2.get_gid()
                for major_tick_obj in equity_ax.yaxis.get_major_ticks()
                if major_tick_obj.label2.get_gid() is not None
            },
            {'strategy_growth_of_1_label', 'benchmark_growth_of_1_label'},
        )

    def test_plot_fills_only_strategy_outperformance_above_benchmark(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 103.0, 101.0, 110.0], index=dates_index)
        benchmark_total_value_ser = pd.Series([100.0, 102.0, 104.0, 107.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            with mock.patch(
                'matplotlib.axes._axes.Axes.fill_between',
                autospec=True,
                wraps=maxes.Axes.fill_between,
            ) as fill_between_mock_obj:
                plot(
                    strategy_total_value=strategy_total_value_ser,
                    benchmark_total_value=benchmark_total_value_ser,
                    save_to=io.BytesIO(),
                )

        equity_fill_call_obj = fill_between_mock_obj.call_args_list[0]
        benchmark_growth_of_1_ser = benchmark_total_value_ser / benchmark_total_value_ser.iloc[0]
        strategy_growth_of_1_ser = strategy_total_value_ser / strategy_total_value_ser.iloc[0]
        expected_outperformance_mask_vec = (
            strategy_growth_of_1_ser.gt(benchmark_growth_of_1_ser).to_numpy()
        )

        pd.testing.assert_index_equal(equity_fill_call_obj.args[1], strategy_growth_of_1_ser.index)
        pd.testing.assert_series_equal(
            equity_fill_call_obj.args[2],
            benchmark_growth_of_1_ser,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            equity_fill_call_obj.args[3],
            strategy_growth_of_1_ser,
            check_names=False,
        )
        self.assertListEqual(
            equity_fill_call_obj.kwargs['where'].tolist(),
            expected_outperformance_mask_vec.tolist(),
        )
        self.assertTrue(equity_fill_call_obj.kwargs['interpolate'])

    def test_plot_colors_parameter_preserves_strategy_then_benchmark_order(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 102.0, 101.0, 110.0], index=dates_index)
        benchmark_total_value_ser = pd.Series([100.0, 101.0, 100.0, 104.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            plot(
                strategy_total_value=strategy_total_value_ser,
                benchmark_total_value=benchmark_total_value_ser,
                colors=('#123456', '#654321'),
                save_to=io.BytesIO(),
            )

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        strategy_line_obj = next(line for line in equity_ax.lines if line.get_gid() == 'strategy_equity_line')
        benchmark_line_obj = next(line for line in equity_ax.lines if line.get_gid() == 'benchmark_equity_line')

        self.assertEqual(strategy_line_obj.get_color(), '#123456')
        self.assertEqual(benchmark_line_obj.get_color(), '#654321')

    def test_plot_applies_signature_grid_color(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 102.0, 101.0, 105.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            plot(strategy_total_value=strategy_total_value_ser, save_to=io.BytesIO())

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        figure_obj.canvas.draw()
        visible_grid_line_list = [
            grid_line_obj for grid_line_obj in equity_ax.get_ygridlines() if grid_line_obj.get_visible()
        ]
        visible_x_grid_line_list = [
            grid_line_obj for grid_line_obj in equity_ax.get_xgridlines() if grid_line_obj.get_visible()
        ]

        self.assertGreater(len(visible_grid_line_list), 0)
        self.assertEqual(len(visible_x_grid_line_list), 0)
        self.assertTrue(
            all(
                mcolors.to_hex(grid_line_obj.get_color()) == SIGNATURE_PALETTE_DICT['grid']
                for grid_line_obj in visible_grid_line_list
            )
        )
        self.assertFalse(equity_ax.spines['top'].get_visible())
        self.assertEqual(mcolors.to_hex(equity_ax.spines['bottom'].get_edgecolor()), SIGNATURE_PALETTE_DICT['axes_border'])
        self.assertEqual(mcolors.to_hex(equity_ax.spines['right'].get_edgecolor()), SIGNATURE_PALETTE_DICT['axes_border'])

    def test_plot_aligns_axis_labels_with_final_growth_values(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 110.0, 120.0, 130.0], index=dates_index)
        benchmark_total_value_ser = pd.Series([100.0, 130.0, 150.0, 160.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            plot(
                strategy_total_value=strategy_total_value_ser,
                benchmark_total_value=benchmark_total_value_ser,
                save_to=io.BytesIO(),
            )

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        figure_obj.canvas.draw()
        strategy_major_tick_obj = _major_tick_by_label_gid(equity_ax, 'strategy_growth_of_1_label')
        benchmark_major_tick_obj = _major_tick_by_label_gid(equity_ax, 'benchmark_growth_of_1_label')

        self.assertAlmostEqual(float(strategy_major_tick_obj.get_loc()), 1.30)
        self.assertAlmostEqual(float(benchmark_major_tick_obj.get_loc()), 1.60)

    def test_plot_promoted_tick_label_aligns_with_native_right_axis_ticks(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 170.0, 260.0, 420.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            plot(strategy_total_value=strategy_total_value_ser, save_to=io.BytesIO())

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        figure_obj.canvas.draw()
        renderer_obj = figure_obj.canvas.get_renderer()
        strategy_major_tick_obj = _major_tick_by_label_gid(equity_ax, 'strategy_growth_of_1_label')
        strategy_label_obj = strategy_major_tick_obj.label2
        label_bbox_obj = strategy_label_obj.get_window_extent(renderer=renderer_obj)
        reference_label_obj = next(
            major_tick_obj.label2
            for major_tick_obj in equity_ax.yaxis.get_major_ticks()
            if major_tick_obj.label2.get_gid() is None and major_tick_obj.label2.get_text() != ''
        )
        reference_bbox_obj = reference_label_obj.get_window_extent(renderer=renderer_obj)

        self.assertAlmostEqual(float(strategy_major_tick_obj.get_loc()), 4.20)
        self.assertAlmostEqual(label_bbox_obj.x0, reference_bbox_obj.x0, delta=1.0)
        self.assertEqual(len(equity_ax.texts), 0)

    def test_generate_yearly_returns_preserves_return_formula(self):
        total_value_ser = pd.Series(
            [100.0, 120.0, 132.0],
            index=pd.to_datetime(['2023-12-31', '2024-12-31', '2025-12-31']),
        )

        yearly_return_ser = generate_yearly_returns(total_value_ser)

        expected_yearly_return_ser = pd.Series(
            [0.20, 0.10],
            index=pd.to_datetime(['2024-12-31', '2025-12-31']),
        )
        pd.testing.assert_series_equal(yearly_return_ser, expected_yearly_return_ser, check_freq=False)

    def test_portfolio_plot_keeps_overlay_lines_subordinate(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_a = make_strategy(
            name='StrategyA',
            dates_index=dates_index,
            daily_returns_list=[0.0, 0.01, -0.01, 0.02],
            benchmark_name='SPY',
            benchmark_total_value_list=[100.0, 101.0, 100.5, 102.0],
        )
        strategy_b = make_strategy(
            name='StrategyB',
            dates_index=dates_index,
            daily_returns_list=[0.0, -0.005, 0.015, 0.01],
        )
        portfolio = Portfolio(
            strategies=[strategy_a, strategy_b],
            weights=[0.5, 0.5],
            capital_base=100.0,
        )

        with mock.patch('matplotlib.pyplot.show'):
            portfolio.plot(save_to=io.BytesIO())

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        overlay_line_list = [
            line for line in equity_ax.lines
            if line.get_gid() is not None and line.get_gid().startswith('additional_equity_line:')
        ]

        self.assertEqual(len(overlay_line_list), 2)
        self.assertTrue(all(line.get_alpha() <= 0.25 for line in overlay_line_list))
        figure_obj.canvas.draw()
        self.assertIn(
            'strategy_growth_of_1_label',
            {major_tick_obj.label2.get_gid() for major_tick_obj in equity_ax.yaxis.get_major_ticks()},
        )


if __name__ == '__main__':
    unittest.main()

