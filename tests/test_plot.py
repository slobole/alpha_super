import io
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
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
        self.assertEqual(strategy_line_obj.get_linestyle(), '-')
        self.assertEqual(strategy_line_obj.get_color(), SEABORN_DEEP_COLOR_LIST[2])
        self.assertEqual(mcolors.to_hex(equity_ax.get_facecolor()), SIGNATURE_PALETTE_DICT['panel'])
        self.assertEqual({text.get_gid() for text in equity_ax.texts}, {'strategy_total_return_label'})
        strategy_label_obj = next(text for text in equity_ax.texts if text.get_gid() == 'strategy_total_return_label')
        self.assertLess(strategy_label_obj.xyann[0], 0)
        self.assertEqual(strategy_label_obj.get_ha(), 'right')

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

        self.assertEqual(strategy_line_obj.get_linestyle(), '-')
        self.assertEqual(benchmark_line_obj.get_linestyle(), '-')
        self.assertEqual(strategy_line_obj.get_color(), SEABORN_DEEP_COLOR_LIST[2])
        self.assertEqual(benchmark_line_obj.get_color(), SEABORN_DEEP_COLOR_LIST[3])
        self.assertEqual(
            {text.get_gid() for text in equity_ax.texts},
            {'strategy_total_return_label', 'benchmark_total_return_label'},
        )

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

        self.assertGreater(len(visible_grid_line_list), 0)
        self.assertTrue(
            all(
                mcolors.to_hex(grid_line_obj.get_color()) == SIGNATURE_PALETTE_DICT['grid']
                for grid_line_obj in visible_grid_line_list
            )
        )

    def test_plot_offsets_end_labels_away_from_higher_finishing_line(self):
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
        strategy_label_obj = next(text for text in equity_ax.texts if text.get_gid() == 'strategy_total_return_label')
        benchmark_label_obj = next(text for text in equity_ax.texts if text.get_gid() == 'benchmark_total_return_label')

        self.assertLess(strategy_label_obj.xyann[1], 0)
        self.assertGreater(benchmark_label_obj.xyann[1], 0)

    def test_plot_keeps_end_label_inside_axes_and_left_of_endpoint(self):
        dates_index = pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
        strategy_total_value_ser = pd.Series([100.0, 170.0, 260.0, 420.0], index=dates_index)

        with mock.patch('matplotlib.pyplot.show'):
            plot(strategy_total_value=strategy_total_value_ser, save_to=io.BytesIO())

        figure_obj = plt.gcf()
        equity_ax = figure_obj.axes[0]
        strategy_line_obj = next(line for line in equity_ax.lines if line.get_gid() == 'strategy_equity_line')
        strategy_label_obj = next(text for text in equity_ax.texts if text.get_gid() == 'strategy_total_return_label')

        figure_obj.canvas.draw()
        renderer_obj = figure_obj.canvas.get_renderer()
        label_bbox_obj = strategy_label_obj.get_window_extent(renderer=renderer_obj)
        axes_bbox_obj = equity_ax.get_window_extent(renderer=renderer_obj)
        end_point_display_arr = equity_ax.transData.transform(
            (
                strategy_line_obj.get_xdata(orig=False)[-1],
                strategy_line_obj.get_ydata(orig=False)[-1],
            )
        )

        self.assertLessEqual(label_bbox_obj.x1, axes_bbox_obj.x1 + 0.5)
        self.assertLess(label_bbox_obj.x1, end_point_display_arr[0])

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
        self.assertTrue(all(line.get_alpha() <= 0.5 for line in overlay_line_list))
        self.assertIn('strategy_total_return_label', {text.get_gid() for text in equity_ax.texts})


if __name__ == '__main__':
    unittest.main()
