from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from alpha.engine.theme import (
    SIGNATURE_PALETTE_DICT,
    apply_signature_axis_style,
    build_plot_color_dict,
    build_signature_rcparams,
)


_PLOT_X_MARGIN_FLOAT = 0.008
_END_LABEL_X_OFFSET_POINTS_INT = -4


def plot(
        strategy_total_value: pd.Series,
        strategy_drawdown: pd.Series = None,
        benchmark_total_value: pd.Series = None,
        benchmark_drawdown: pd.Series = None,
        benchmark_label: str = 'Benchmark',
        strategy_label: str = 'Strategy',
        additional_returns: pd.DataFrame = None,
        additional_drawdowns: pd.DataFrame = None,
        save_to: str = None,
        to_web: bool = True,
        dpi: int = 150,
        alpha_additional: float = 0.2,
        colors=None,
        vertical_lines=(),
        ylims=(),
        use_log_scale=True
):
    """
    Plot strategy and benchmark performance, drawdowns, and annual returns.

    Formulas preserved by this visualization layer:

    normalized_equity_ser_t = total_value_t / total_value_0

    total_return_float = total_value_T / total_value_0 - 1

    drawdown_ser_t = total_value_t / max(total_value_1, ..., total_value_t) - 1
    """
    strategy_total_value_ser = pd.Series(strategy_total_value, copy=False).astype(float)
    benchmark_total_value_ser = (
        pd.Series(benchmark_total_value, copy=False).astype(float)
        if benchmark_total_value is not None else None
    )
    additional_returns_df = additional_returns.copy() if additional_returns is not None else None
    additional_drawdowns_df = additional_drawdowns.copy() if additional_drawdowns is not None else None

    plot_style_dict = build_signature_rcparams(to_web_bool=to_web)
    plot_color_dict = build_plot_color_dict(colors)
    benchmark_color_str = plot_color_dict['benchmark']
    strategy_color_str = plot_color_dict['strategy']
    additional_color_map: dict[str, str] = {}

    strategy_equity_ser = strategy_total_value_ser / strategy_total_value_ser.iloc[0]
    strategy_peak_equity_ser = strategy_equity_ser.cummax()
    strategy_yearly_return_ser = generate_yearly_returns(strategy_total_value_ser)

    benchmark_equity_ser = None
    benchmark_peak_equity_ser = None
    benchmark_yearly_return_ser = None
    if benchmark_total_value_ser is not None:
        benchmark_equity_ser = benchmark_total_value_ser / benchmark_total_value_ser.iloc[0]
        benchmark_peak_equity_ser = benchmark_equity_ser.cummax()
        benchmark_yearly_return_ser = generate_yearly_returns(benchmark_total_value_ser)

    if strategy_drawdown is None:
        strategy_drawdown_ser = compute_drawdown(strategy_total_value_ser)
    else:
        strategy_drawdown_ser = pd.Series(strategy_drawdown, copy=False).astype(float)

    if benchmark_drawdown is None and benchmark_total_value_ser is not None:
        benchmark_drawdown_ser = compute_drawdown(benchmark_total_value_ser)
    elif benchmark_drawdown is not None:
        benchmark_drawdown_ser = pd.Series(benchmark_drawdown, copy=False).astype(float)
    else:
        benchmark_drawdown_ser = None

    if benchmark_yearly_return_ser is not None:
        annual_return_df = pd.concat(
            [
                strategy_yearly_return_ser.rename(strategy_label),
                benchmark_yearly_return_ser.rename(benchmark_label),
            ],
            axis=1,
        )
        strategy_yearly_return_ser = annual_return_df[strategy_label]
        benchmark_yearly_return_ser = annual_return_df[benchmark_label]

    with plt.rc_context(plot_style_dict):
        figure_obj = plt.figure(figsize=(12, 11.2))
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.16)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, height_ratios=[2, 1], subplot_spec=outer_gs[0], hspace=0.07
        )

        equity_ax = figure_obj.add_subplot(inner_gs[0])
        drawdown_ax = figure_obj.add_subplot(inner_gs[1])
        annual_ax = figure_obj.add_subplot(outer_gs[1])

        benchmark_equity_line_obj = None
        if benchmark_equity_ser is not None:
            benchmark_equity_line_obj, = equity_ax.plot(
                benchmark_equity_ser.index,
                benchmark_equity_ser,
                color=benchmark_color_str,
                linestyle='-',
                linewidth=0.95,
                alpha=0.98,
                label=benchmark_label,
                zorder=4,
            )
            benchmark_equity_line_obj.set_gid('benchmark_equity_line')
            equity_ax.fill_between(
                benchmark_equity_ser.index,
                benchmark_peak_equity_ser,
                benchmark_equity_ser,
                where=benchmark_peak_equity_ser > benchmark_equity_ser,
                color=benchmark_color_str,
                alpha=0.08,
                zorder=1,
            )

        strategy_equity_line_obj, = equity_ax.plot(
            strategy_equity_ser.index,
            strategy_equity_ser,
            color=strategy_color_str,
            linewidth=1.15,
            alpha=1.0,
            label=strategy_label,
            zorder=5,
        )
        strategy_equity_line_obj.set_gid('strategy_equity_line')
        equity_ax.fill_between(
            strategy_equity_ser.index,
            strategy_peak_equity_ser,
            strategy_equity_ser,
            where=strategy_peak_equity_ser > strategy_equity_ser,
            color=strategy_color_str,
            alpha=0.11,
            zorder=2,
        )

        equity_max_float = float(strategy_equity_ser.max())
        if benchmark_equity_ser is not None:
            equity_max_float = max(equity_max_float, float(benchmark_equity_ser.max()))

        if additional_returns_df is not None:
            for column_idx_int, column_name_str in enumerate(additional_returns_df.columns):
                additional_total_value_ser = pd.Series(
                    additional_returns_df[column_name_str], copy=False
                ).astype(float)
                additional_equity_ser = additional_total_value_ser / additional_total_value_ser.iloc[0]
                additional_color_str = plot_color_dict['additional_cycle'][
                    column_idx_int % len(plot_color_dict['additional_cycle'])
                ]

                additional_equity_line_obj, = equity_ax.plot(
                    additional_equity_ser.index,
                    additional_equity_ser,
                    color=additional_color_str,
                    linewidth=0.65,
                    alpha=alpha_additional,
                    zorder=3,
                    label='_nolegend_',
                )
                additional_equity_line_obj.set_gid(f'additional_equity_line:{column_name_str}')
                additional_color_map[column_name_str] = additional_color_str
                equity_max_float = max(equity_max_float, float(additional_equity_ser.max()))

        if use_log_scale:
            equity_ax.set_yscale('log')
            _set_equity_ticks(equity_ax, equity_max_float)

        equity_ax.set_ylabel('Total Return (log scale)' if use_log_scale else 'Total Return')
        equity_ax.yaxis.set_major_formatter(FuncFormatter(percentage_major_formatter))
        equity_ax.yaxis.set_minor_formatter(FuncFormatter(percentage_minor_formatter))
        equity_ax.tick_params(axis='x', labelbottom=False)
        equity_ax.margins(x=_PLOT_X_MARGIN_FLOAT)
        equity_ax.legend(
            loc='upper left',
            frameon=True,
        )

        if len(ylims) > 0:
            equity_ax.set_ylim(ylims)

        if benchmark_drawdown_ser is not None:
            benchmark_drawdown_line_obj, = drawdown_ax.plot(
                benchmark_drawdown_ser.index,
                benchmark_drawdown_ser,
                color=benchmark_color_str,
                linestyle='-',
                linewidth=0.8,
                alpha=0.98,
                zorder=4,
            )
            benchmark_drawdown_line_obj.set_gid('benchmark_drawdown_line')
            drawdown_ax.fill_between(
                benchmark_drawdown_ser.index,
                0.0,
                benchmark_drawdown_ser,
                color=benchmark_color_str,
                alpha=0.10,
                zorder=1,
            )

        strategy_drawdown_line_obj, = drawdown_ax.plot(
            strategy_drawdown_ser.index,
            strategy_drawdown_ser,
            color=strategy_color_str,
            linewidth=0.9,
            alpha=1.0,
            zorder=5,
        )
        strategy_drawdown_line_obj.set_gid('strategy_drawdown_line')
        drawdown_ax.fill_between(
            strategy_drawdown_ser.index,
            0.0,
            strategy_drawdown_ser,
            color=strategy_color_str,
            alpha=0.13,
            zorder=2,
        )

        if additional_drawdowns_df is not None:
            for column_idx_int, column_name_str in enumerate(additional_drawdowns_df.columns):
                additional_drawdown_ser = pd.Series(
                    additional_drawdowns_df[column_name_str], copy=False
                ).astype(float)
                additional_drawdown_line_obj, = drawdown_ax.plot(
                    additional_drawdown_ser.index,
                    additional_drawdown_ser,
                    color=additional_color_map.get(
                        column_name_str,
                        plot_color_dict['additional_cycle'][
                            column_idx_int % len(plot_color_dict['additional_cycle'])
                        ],
                    ),
                    linewidth=0.6,
                    alpha=max(0.12, alpha_additional * 0.8),
                    zorder=3,
                )
                additional_drawdown_line_obj.set_gid(f'additional_drawdown_line:{column_name_str}')

        drawdown_ax.set_ylabel('Drawdown')
        drawdown_ax.yaxis.set_major_formatter(FuncFormatter(percentage_major_formatter_2))
        drawdown_ax.tick_params(axis='x', labelbottom=False)
        drawdown_ax.margins(x=_PLOT_X_MARGIN_FLOAT)

        annual_position_vec = np.arange(len(strategy_yearly_return_ser), dtype=float)
        has_benchmark_bool = benchmark_yearly_return_ser is not None
        bar_width_float = 0.34 if has_benchmark_bool else 0.56
        bar_edge_color_str = SIGNATURE_PALETTE_DICT['bar_edge']

        if has_benchmark_bool:
            strategy_bar_center_vec = annual_position_vec - bar_width_float / 2.0
            benchmark_bar_center_vec = annual_position_vec + bar_width_float / 2.0
        else:
            strategy_bar_center_vec = annual_position_vec
            benchmark_bar_center_vec = None

        strategy_bar_container = annual_ax.bar(
            strategy_bar_center_vec,
            strategy_yearly_return_ser.to_numpy(),
            bar_width_float,
            label=strategy_label,
            color=strategy_color_str,
            alpha=0.88,
            edgecolor=bar_edge_color_str,
            linewidth=0.55,
            zorder=3,
        )
        for bar_patch in strategy_bar_container.patches:
            bar_patch.set_gid('strategy_annual_bar')

        if has_benchmark_bool:
            benchmark_bar_container = annual_ax.bar(
                benchmark_bar_center_vec,
                benchmark_yearly_return_ser.to_numpy(),
                bar_width_float,
                label=benchmark_label,
                color=benchmark_color_str,
                alpha=0.80,
                edgecolor=bar_edge_color_str,
                linewidth=0.55,
                zorder=2,
            )
            for bar_patch in benchmark_bar_container.patches:
                bar_patch.set_gid('benchmark_annual_bar')

        annual_ax.set_ylabel('Annual Return (%)')
        annual_ax.set_xticks(annual_position_vec)
        annual_ax.set_xticklabels(strategy_yearly_return_ser.index.year, rotation=90)
        annual_ax.set_xlim(-0.6, len(strategy_yearly_return_ser) - 0.4)
        annual_ax.yaxis.set_major_formatter(FuncFormatter(percentage_major_formatter_2))

        for separator_float in np.arange(-0.5, len(strategy_yearly_return_ser) + 0.5, 1.0):
            annual_ax.axvline(
                x=separator_float,
                color=SIGNATURE_PALETTE_DICT['grid'],
                linewidth=0.7,
                alpha=0.55,
                zorder=0,
            )

        for axis_obj in (equity_ax, drawdown_ax, annual_ax):
            apply_signature_axis_style(axis_obj, vertical_lines)

        strategy_total_return_float = strategy_total_value_ser.iloc[-1] / strategy_total_value_ser.iloc[0] - 1.0
        label_offset_dict = _resolve_end_label_offset_dict(
            strategy_end_value_float=float(strategy_equity_ser.iloc[-1]),
            benchmark_end_value_float=(
                float(benchmark_equity_ser.iloc[-1]) if benchmark_equity_ser is not None else None
            ),
        )
        add_label(
            equity_ax,
            strategy_equity_line_obj,
            f'{strategy_total_return_float:+,.0%}',
            border_color_str=strategy_color_str,
            label_id_str='strategy_total_return_label',
            x_offset_points_int=_END_LABEL_X_OFFSET_POINTS_INT,
            horizontal_alignment_str='right',
            y_offset_points_int=label_offset_dict['strategy'],
            zorder=10,
        )

        if benchmark_equity_line_obj is not None and benchmark_total_value_ser is not None:
            benchmark_total_return_float = (
                benchmark_total_value_ser.iloc[-1] / benchmark_total_value_ser.iloc[0] - 1.0
            )
            add_label(
                equity_ax,
                benchmark_equity_line_obj,
                f'{benchmark_total_return_float:+,.0%}',
                border_color_str=benchmark_color_str,
                label_id_str='benchmark_total_return_label',
                x_offset_points_int=_END_LABEL_X_OFFSET_POINTS_INT,
                horizontal_alignment_str='right',
                y_offset_points_int=label_offset_dict['benchmark'],
                zorder=9,
            )

        figure_obj.align_ylabels((equity_ax, drawdown_ax, annual_ax))

        if save_to:
            figure_obj.savefig(save_to, dpi=dpi, bbox_inches='tight')

        plt.show()


def _set_equity_ticks(axis_obj, equity_max_float: float) -> None:
    upper_tick_float = max(1.0, float(equity_max_float))
    major_tick_list = [0.5, 1.0]
    candidate_tick_list = [1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    major_tick_list.extend(
        tick_float for tick_float in candidate_tick_list if tick_float <= upper_tick_float * 1.02
    )

    if upper_tick_float > 10.0:
        extended_tick_arr = np.arange(12.0, np.ceil(upper_tick_float) + 2.0, 2.0)
        major_tick_list.extend(extended_tick_arr.tolist())

    axis_obj.set_yticks(sorted(set(major_tick_list)))


def _resolve_end_label_offset_dict(
        strategy_end_value_float: float,
        benchmark_end_value_float: float | None,
) -> dict[str, int]:
    """
    Offset end labels away from each other using the final normalized equity values.

    If benchmark_end_value_float > strategy_end_value_float, then:

    strategy_label_offset_int < 0
    benchmark_label_offset_int > 0
    """
    if benchmark_end_value_float is None:
        return {'strategy': 0, 'benchmark': 0}

    offset_magnitude_int = 12
    if benchmark_end_value_float >= strategy_end_value_float:
        return {'strategy': -offset_magnitude_int, 'benchmark': offset_magnitude_int}

    return {'strategy': offset_magnitude_int, 'benchmark': -offset_magnitude_int}


def generate_yearly_returns(series):
    """Compute annual returns from year-end total values."""
    total_value_ser = pd.Series(series, copy=False).astype(float)
    year_end_total_value_ser = total_value_ser.resample('YE').last()

    if year_end_total_value_ser.index[0] != total_value_ser.index[0]:
        start_anchor_ts = pd.Timestamp(
            year=year_end_total_value_ser.index[0].year - 1,
            month=year_end_total_value_ser.index[0].month,
            day=year_end_total_value_ser.index[0].day,
        )
        year_end_total_value_ser.loc[start_anchor_ts] = total_value_ser.iloc[0]
        year_end_total_value_ser = year_end_total_value_ser.sort_index()

    yearly_return_ser = year_end_total_value_ser.pct_change(fill_method=None).dropna()
    return yearly_return_ser


def compute_drawdown(total_value_ser: pd.Series) -> pd.Series:
    """
    Compute drawdown with:

    drawdown_ser_t = total_value_t / max(total_value_1, ..., total_value_t) - 1
    """
    total_value_ser = pd.Series(total_value_ser, copy=False).astype(float)
    running_peak_ser = total_value_ser.cummax()
    drawdown_ser = total_value_ser / running_peak_ser - 1.0
    return drawdown_ser


def add_label(
        axis_obj,
        line_obj,
        label_str: str,
        border_color_str: str,
        label_id_str: str,
        font_color_str: str | None = None,
        x_offset_points_int: int = _END_LABEL_X_OFFSET_POINTS_INT,
        horizontal_alignment_str: str = 'right',
        y_offset_points_int: int = 0,
        zorder: int = 5,
):
    """Add a compact end-of-line label for the final total return."""
    x_data, y_data = line_obj.get_data()
    font_color_str = SIGNATURE_PALETTE_DICT['ink'] if font_color_str is None else font_color_str

    annotation_obj = axis_obj.annotate(
        label_str,
        xy=(x_data[-1], y_data[-1]),
        xytext=(x_offset_points_int, y_offset_points_int),
        textcoords='offset points',
        ha=horizontal_alignment_str,
        va='center',
        fontsize=7.5,
        color=font_color_str,
        bbox=dict(
            boxstyle='round,pad=0.22,rounding_size=0.18',
            edgecolor=border_color_str,
            facecolor=SIGNATURE_PALETTE_DICT['label_face'],
            linewidth=1.0,
        ),
        zorder=zorder,
    )
    annotation_obj.set_gid(label_id_str)
    return annotation_obj


def percentage_major_formatter(x, pos):
    """Format equity-axis values with:

    displayed_return_pct = (x - 1) * 100
    """
    return f'{(x - 1) * 100:.0f}%'


def percentage_major_formatter_2(x, pos):
    """Format drawdown and annual-return values with:

    displayed_return_pct = x * 100
    """
    return f'{x * 100:.0f}%'


def percentage_minor_formatter(x, pos):
    """Keep minor tick labels empty for a cleaner look."""
    return ''
