from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from alpha.engine.theme import (
    SIGNATURE_PALETTE_DICT,
    apply_signature_axis_style,
    build_plot_color_dict,
    build_signature_rcparams,
)


_PLOT_X_MARGIN_FLOAT = 0.008
_PROMOTED_TICK_MIN_VERTICAL_GAP_PIXELS_FLOAT = 18.0
_SHORT_WINDOW_MAX_DAY_COUNT_INT = 62
_MEDIUM_WINDOW_MAX_DAY_COUNT_INT = 366
_INTERMEDIATE_WINDOW_MAX_DAY_COUNT_INT = 3 * 366
_SHORT_WINDOW_MAX_TICK_COUNT_INT = 8


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
        use_log_scale=True,
        return_bar_frequency_str: str = 'annual',
):
    """
    Plot strategy and benchmark performance, drawdowns, and period returns.

    Formulas preserved by this visualization layer:

    growth_of_1_ser_t = total_value_t / total_value_0

    cumulative_return_float = total_value_T / total_value_0 - 1

    displayed_growth_dollar_float = 1.0 * growth_of_1_float

    drawdown_ser_t = total_value_t / max(total_value_1, ..., total_value_t) - 1

    relative_outperformance_ser_t = strategy_growth_of_1_ser_t - benchmark_growth_of_1_ser_t

    For the bottom return panel:

    R_period_k = V_period_end_k / V_period_end_(k-1) - 1

    with the first partial displayed period anchored to the plotted window:

    R_first = V_first_period_end / V_0 - 1
    """
    strategy_total_value_ser = pd.Series(strategy_total_value, copy=False).astype(float)
    bar_date_idx = pd.DatetimeIndex(strategy_total_value_ser.index)
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

    strategy_growth_of_1_ser = strategy_total_value_ser / strategy_total_value_ser.iloc[0]
    strategy_period_return_ser = generate_period_returns(
        strategy_total_value_ser,
        return_bar_frequency_str=return_bar_frequency_str,
    )
    return_panel_ylabel_str, return_period_label_formatter_fn = (
        _resolve_return_panel_spec(return_bar_frequency_str)
    )

    benchmark_growth_of_1_ser = None
    benchmark_period_return_ser = None
    if benchmark_total_value_ser is not None:
        benchmark_growth_of_1_ser = benchmark_total_value_ser / benchmark_total_value_ser.iloc[0]
        benchmark_period_return_ser = generate_period_returns(
            benchmark_total_value_ser,
            return_bar_frequency_str=return_bar_frequency_str,
        )

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

    if benchmark_period_return_ser is not None:
        period_return_df = pd.concat(
            [
                strategy_period_return_ser.rename(strategy_label),
                benchmark_period_return_ser.rename(benchmark_label),
            ],
            axis=1,
        )
        strategy_period_return_ser = period_return_df[strategy_label]
        benchmark_period_return_ser = period_return_df[benchmark_label]

    with plt.rc_context(plot_style_dict):
        figure_obj = plt.figure(figsize=(12, 11.2))
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.16)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, height_ratios=[2, 1], subplot_spec=outer_gs[0], hspace=0.07
        )

        equity_ax = figure_obj.add_subplot(inner_gs[0])
        drawdown_ax = figure_obj.add_subplot(inner_gs[1])
        annual_ax = figure_obj.add_subplot(outer_gs[1])
        equity_baseline_growth_of_1_float = 1.0

        benchmark_growth_line_obj = None
        if benchmark_growth_of_1_ser is not None:
            benchmark_growth_line_obj, = equity_ax.plot(
                benchmark_growth_of_1_ser.index,
                benchmark_growth_of_1_ser,
                color=benchmark_color_str,
                linestyle='-',
                linewidth=1.0,
                alpha=0.92,
                label=benchmark_label,
                zorder=4,
            )
            benchmark_growth_line_obj.set_gid('benchmark_equity_line')

        strategy_growth_line_obj, = equity_ax.plot(
            strategy_growth_of_1_ser.index,
            strategy_growth_of_1_ser,
            color=strategy_color_str,
            linewidth=1.35,
            alpha=1.0,
            label=strategy_label,
            zorder=5,
        )
        strategy_growth_line_obj.set_gid('strategy_equity_line')
        if benchmark_growth_of_1_ser is not None:
            benchmark_fill_base_ser = benchmark_growth_of_1_ser.reindex(strategy_growth_of_1_ser.index)
            outperformance_mask_ser = strategy_growth_of_1_ser.gt(benchmark_fill_base_ser).fillna(False)
            strategy_equity_fill_collection_obj = equity_ax.fill_between(
                strategy_growth_of_1_ser.index,
                benchmark_fill_base_ser,
                strategy_growth_of_1_ser,
                where=outperformance_mask_ser.to_numpy(),
                interpolate=True,
                color=strategy_color_str,
                alpha=0.24,
                zorder=2,
            )
        else:
            strategy_equity_fill_collection_obj = equity_ax.fill_between(
                strategy_growth_of_1_ser.index,
                equity_baseline_growth_of_1_float,
                strategy_growth_of_1_ser,
                color=strategy_color_str,
                alpha=0.24,
                zorder=2,
            )
        strategy_equity_fill_collection_obj.set_gid('strategy_equity_fill')

        growth_of_1_max_float = float(strategy_growth_of_1_ser.max())
        if benchmark_growth_of_1_ser is not None:
            growth_of_1_max_float = max(
                growth_of_1_max_float,
                float(benchmark_growth_of_1_ser.max()),
            )

        if additional_returns_df is not None:
            for column_idx_int, column_name_str in enumerate(additional_returns_df.columns):
                additional_total_value_ser = pd.Series(
                    additional_returns_df[column_name_str], copy=False
                ).astype(float)
                additional_growth_of_1_ser = (
                    additional_total_value_ser / additional_total_value_ser.iloc[0]
                )
                additional_color_str = plot_color_dict['additional_cycle'][
                    column_idx_int % len(plot_color_dict['additional_cycle'])
                ]

                additional_growth_line_obj, = equity_ax.plot(
                    additional_growth_of_1_ser.index,
                    additional_growth_of_1_ser,
                    color=additional_color_str,
                    linewidth=0.7,
                    alpha=max(0.12, alpha_additional * 0.85),
                    zorder=3,
                    label='_nolegend_',
                )
                additional_growth_line_obj.set_gid(f'additional_equity_line:{column_name_str}')
                additional_color_map[column_name_str] = additional_color_str
                growth_of_1_max_float = max(
                    growth_of_1_max_float,
                    float(additional_growth_of_1_ser.max()),
                )

        if use_log_scale:
            equity_ax.set_yscale('log')
            _set_growth_of_1_ticks(equity_ax, growth_of_1_max_float)

        equity_ax.set_ylabel('Growth of $1 (log scale)' if use_log_scale else 'Growth of $1')
        equity_ax.yaxis.set_major_formatter(FuncFormatter(growth_of_1_major_formatter))
        equity_ax.yaxis.set_minor_formatter(FuncFormatter(blank_minor_formatter))
        equity_ax.tick_params(axis='x', labelbottom=False)
        equity_ax.margins(x=_PLOT_X_MARGIN_FLOAT)
        equity_legend_obj = equity_ax.legend(
            loc='upper left',
            frameon=True,
            fancybox=True,
            borderpad=0.55,
            labelspacing=0.4,
            handlelength=2.2,
        )
        equity_legend_obj.get_frame().set_linewidth(0.9)
        equity_legend_obj.get_frame().set_edgecolor(SIGNATURE_PALETTE_DICT['legend_edge'])
        equity_legend_obj.get_frame().set_facecolor(SIGNATURE_PALETTE_DICT['legend_face'])

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
            alpha=0.07,
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
                    linewidth=0.65,
                    alpha=max(0.10, alpha_additional * 0.65),
                    zorder=3,
                )
                additional_drawdown_line_obj.set_gid(f'additional_drawdown_line:{column_name_str}')

        drawdown_ax.set_ylabel('Drawdown')
        drawdown_ax.yaxis.set_major_formatter(FuncFormatter(fraction_major_formatter))
        drawdown_locator_obj, drawdown_formatter_obj, drawdown_rotation_float = (
            _build_time_axis_locator_formatter_tuple(bar_date_idx)
        )
        drawdown_ax.xaxis.set_major_locator(drawdown_locator_obj)
        drawdown_ax.xaxis.set_major_formatter(drawdown_formatter_obj)
        drawdown_ax.tick_params(axis='x', labelbottom=True, rotation=drawdown_rotation_float)
        drawdown_ax.margins(x=_PLOT_X_MARGIN_FLOAT)

        return_position_vec = np.arange(len(strategy_period_return_ser), dtype=float)
        has_benchmark_bool = benchmark_period_return_ser is not None
        bar_width_float = 0.34 if has_benchmark_bool else 0.56
        bar_edge_color_str = SIGNATURE_PALETTE_DICT['bar_edge']

        if has_benchmark_bool:
            strategy_bar_center_vec = return_position_vec - bar_width_float / 2.0
            benchmark_bar_center_vec = return_position_vec + bar_width_float / 2.0
        else:
            strategy_bar_center_vec = return_position_vec
            benchmark_bar_center_vec = None

        strategy_bar_container = annual_ax.bar(
            strategy_bar_center_vec,
            strategy_period_return_ser.to_numpy(),
            bar_width_float,
            label=strategy_label,
            color=strategy_color_str,
            alpha=0.78,
            edgecolor=bar_edge_color_str,
            linewidth=0.75,
            zorder=3,
        )
        for bar_patch in strategy_bar_container.patches:
            bar_patch.set_gid('strategy_period_return_bar')

        if has_benchmark_bool:
            benchmark_bar_container = annual_ax.bar(
                benchmark_bar_center_vec,
                benchmark_period_return_ser.to_numpy(),
                bar_width_float,
                label=benchmark_label,
                color=benchmark_color_str,
                alpha=0.72,
                edgecolor=bar_edge_color_str,
                linewidth=0.75,
                zorder=2,
            )
            for bar_patch in benchmark_bar_container.patches:
                bar_patch.set_gid('benchmark_period_return_bar')

        return_label_list = [
            return_period_label_formatter_fn(period_end_ts)
            for period_end_ts in pd.DatetimeIndex(strategy_period_return_ser.index)
        ]
        annual_ax.set_ylabel(return_panel_ylabel_str)
        annual_ax.set_xticks(return_position_vec)
        annual_ax.set_xticklabels(return_label_list, rotation=90)
        annual_ax.set_xlim(-0.6, len(strategy_period_return_ser) - 0.4)
        annual_ax.yaxis.set_major_formatter(FuncFormatter(fraction_major_formatter))

        for separator_float in np.arange(-0.5, len(strategy_period_return_ser) + 0.5, 1.0):
            annual_ax.axvline(
                x=separator_float,
                color=SIGNATURE_PALETTE_DICT['grid'],
                linewidth=0.7,
                alpha=0.55,
                zorder=0,
            )

        for axis_obj in (equity_ax, drawdown_ax, annual_ax):
            apply_signature_axis_style(axis_obj, vertical_lines)

        promoted_growth_tick_spec_list = [
            {
                'growth_of_1_float': float(strategy_total_value_ser.iloc[-1] / strategy_total_value_ser.iloc[0]),
                'color_str': strategy_color_str,
                'label_id_str': 'strategy_growth_of_1_label',
                'axis_tick_id_str': 'strategy_growth_of_1_axis_tick',
            }
        ]

        if benchmark_growth_line_obj is not None and benchmark_total_value_ser is not None:
            promoted_growth_tick_spec_list.append(
                {
                    'growth_of_1_float': float(
                        benchmark_total_value_ser.iloc[-1] / benchmark_total_value_ser.iloc[0]
                    ),
                    'color_str': benchmark_color_str,
                    'label_id_str': 'benchmark_growth_of_1_label',
                    'axis_tick_id_str': 'benchmark_growth_of_1_axis_tick',
                }
            )

        promote_right_axis_growth_ticks(
            equity_ax,
            promoted_growth_tick_spec_list=promoted_growth_tick_spec_list,
        )

        figure_obj.align_ylabels((equity_ax, drawdown_ax, annual_ax))

        if save_to:
            figure_obj.savefig(save_to, dpi=dpi, bbox_inches='tight')

        plt.show()


def _build_time_axis_locator_formatter_tuple(
    bar_date_idx: pd.DatetimeIndex,
) -> tuple[object, object, float]:
    """
    Build an adaptive date axis for the lower panel.

    For short windows, the tick dates are sampled from the actual observed bars:

        i_k = round(k * (N - 1) / (K - 1))

        tick_date_k = T[i_k]

    where:

        T = observed bar-date index
        N = len(T)
        K = requested tick count
    """
    normalized_bar_date_idx = pd.DatetimeIndex(bar_date_idx).sort_values().unique()
    if len(normalized_bar_date_idx) == 0:
        return mdates.YearLocator(), mdates.DateFormatter('%Y'), 0.0

    if len(normalized_bar_date_idx) == 1:
        single_tick_float = float(mdates.date2num(normalized_bar_date_idx[0].to_pydatetime()))
        return FixedLocator([single_tick_float]), mdates.DateFormatter('%Y-%m-%d'), 35.0

    span_day_count_int = int(
        (normalized_bar_date_idx[-1] - normalized_bar_date_idx[0]).days
    )

    if span_day_count_int <= _SHORT_WINDOW_MAX_DAY_COUNT_INT:
        tick_count_int = min(_SHORT_WINDOW_MAX_TICK_COUNT_INT, len(normalized_bar_date_idx))
        tick_position_vec = np.linspace(
            0.0,
            float(len(normalized_bar_date_idx) - 1),
            tick_count_int,
        )
        tick_index_vec = np.unique(np.round(tick_position_vec).astype(int))
        # *** CRITICAL*** Sample short-window tick labels from the actual
        # observed bar dates so crisis plots show real tradable dates rather
        # than synthetic calendar interpolation.
        tick_date_idx = normalized_bar_date_idx[tick_index_vec]
        tick_location_vec = mdates.date2num(tick_date_idx.to_pydatetime())
        return FixedLocator(tick_location_vec), mdates.DateFormatter('%Y-%m-%d'), 35.0

    if span_day_count_int <= _MEDIUM_WINDOW_MAX_DAY_COUNT_INT:
        return mdates.MonthLocator(interval=1), mdates.DateFormatter('%Y-%m'), 25.0

    if span_day_count_int <= _INTERMEDIATE_WINDOW_MAX_DAY_COUNT_INT:
        return mdates.MonthLocator(interval=3), mdates.DateFormatter('%Y-%m'), 0.0

    return mdates.YearLocator(), mdates.DateFormatter('%Y'), 0.0


def _set_growth_of_1_ticks(axis_obj, growth_of_1_max_float: float) -> None:
    upper_tick_float = max(1.0, float(growth_of_1_max_float))
    major_tick_list = [0.5, 1.0]
    candidate_tick_list = [1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    major_tick_list.extend(
        tick_float for tick_float in candidate_tick_list if tick_float <= upper_tick_float * 1.02
    )

    if upper_tick_float > 10.0:
        extended_tick_arr = np.arange(12.0, np.ceil(upper_tick_float) + 2.0, 2.0)
        major_tick_list.extend(extended_tick_arr.tolist())

    axis_obj.set_yticks(sorted(set(major_tick_list)))


def _resolve_return_panel_spec(
    return_bar_frequency_str: str,
) -> tuple[str, Callable[[pd.Timestamp], str]]:
    normalized_frequency_str = str(return_bar_frequency_str).strip().lower()
    if normalized_frequency_str == 'annual':
        return 'Annual Return (%)', lambda period_end_ts: f'{pd.Timestamp(period_end_ts).year}'
    if normalized_frequency_str == 'monthly':
        return 'Monthly Return (%)', lambda period_end_ts: pd.Timestamp(period_end_ts).strftime('%Y-%m')
    raise ValueError(
        f'Unsupported return_bar_frequency_str={return_bar_frequency_str!r}. '
        "Expected 'annual' or 'monthly'."
    )


def generate_period_returns(
    series,
    return_bar_frequency_str: str = 'annual',
):
    """
    Compute displayed period returns from realized total value.

    For each displayed period k:

        R_period_k = V_period_end_k / V_period_end_(k-1) - 1

    The first plotted period is anchored to the plotted start:

        R_first = V_first_period_end / V_0 - 1
    """
    total_value_ser = pd.Series(series, copy=False).astype(float)
    if len(total_value_ser) == 0:
        return pd.Series(dtype=float)

    normalized_frequency_str = str(return_bar_frequency_str).strip().lower()
    if normalized_frequency_str == 'annual':
        resample_rule_str = 'YE'
        period_offset_obj = pd.offsets.YearEnd(1)
    elif normalized_frequency_str == 'monthly':
        resample_rule_str = 'ME'
        period_offset_obj = pd.offsets.MonthEnd(1)
    else:
        raise ValueError(
            f'Unsupported return_bar_frequency_str={return_bar_frequency_str!r}. '
            "Expected 'annual' or 'monthly'."
        )

    period_end_total_value_ser = total_value_ser.resample(resample_rule_str).last()
    if len(period_end_total_value_ser) == 0:
        return pd.Series(dtype=float)

    if period_end_total_value_ser.index[0] != total_value_ser.index[0]:
        # *** CRITICAL*** Anchor the first displayed return period to the
        # realized window start so a partial first month/year is measured as
        # V_first_period_end / V_0 - 1 rather than being dropped.
        start_anchor_ts = pd.Timestamp(period_end_total_value_ser.index[0]) - period_offset_obj
        period_end_total_value_ser.loc[start_anchor_ts] = total_value_ser.iloc[0]
        period_end_total_value_ser = period_end_total_value_ser.sort_index()

    period_return_ser = period_end_total_value_ser.pct_change(fill_method=None).dropna()
    return period_return_ser


def generate_yearly_returns(series):
    """Compute annual returns from year-end total values."""
    return generate_period_returns(series, return_bar_frequency_str='annual')


def compute_drawdown(total_value_ser: pd.Series) -> pd.Series:
    """
    Compute drawdown with:

    drawdown_ser_t = total_value_t / max(total_value_1, ..., total_value_t) - 1
    """
    total_value_ser = pd.Series(total_value_ser, copy=False).astype(float)
    running_peak_ser = total_value_ser.cummax()
    drawdown_ser = total_value_ser / running_peak_ser - 1.0
    return drawdown_ser


def promote_right_axis_growth_ticks(
        axis_obj,
        promoted_growth_tick_spec_list: list[dict[str, object]],
) -> None:
    """Promote final Growth of $1 values into native right-axis ticks.

    The promoted tick values are:

    G_T = V_T / V_0

    The promoted tick labels are part of the y-axis tick system itself rather
    than offset annotations, which keeps spacing aligned with the native axis
    labels and avoids covering nearby tick values.
    """
    if len(promoted_growth_tick_spec_list) == 0:
        return

    base_growth_tick_list = [
        float(tick_value_obj)
        for tick_value_obj in axis_obj.get_yticks()
        if np.isfinite(tick_value_obj) and float(tick_value_obj) > 0.0
    ]
    promoted_growth_value_list = [
        float(spec_dict['growth_of_1_float'])
        for spec_dict in promoted_growth_tick_spec_list
        if np.isfinite(spec_dict['growth_of_1_float']) and float(spec_dict['growth_of_1_float']) > 0.0
    ]
    combined_growth_tick_list = sorted(set(base_growth_tick_list + promoted_growth_value_list))

    promoted_growth_pixel_list = [
        float(axis_obj.transData.transform((0.0, promoted_growth_float))[1])
        for promoted_growth_float in promoted_growth_value_list
    ]

    pruned_growth_tick_list: list[float] = []
    for tick_value_float in combined_growth_tick_list:
        tick_is_promoted_bool = any(
            np.isclose(tick_value_float, promoted_growth_float, atol=1e-9, rtol=0.0)
            for promoted_growth_float in promoted_growth_value_list
        )
        if tick_is_promoted_bool:
            pruned_growth_tick_list.append(tick_value_float)
            continue

        tick_pixel_float = float(axis_obj.transData.transform((0.0, tick_value_float))[1])
        crowded_by_promoted_bool = any(
            abs(tick_pixel_float - promoted_pixel_float) < _PROMOTED_TICK_MIN_VERTICAL_GAP_PIXELS_FLOAT
            for promoted_pixel_float in promoted_growth_pixel_list
        )
        if not crowded_by_promoted_bool:
            pruned_growth_tick_list.append(tick_value_float)

    axis_obj.set_yticks(pruned_growth_tick_list)
    axis_obj.figure.canvas.draw()

    for major_tick_obj in axis_obj.yaxis.get_major_ticks():
        tick_value_float = float(major_tick_obj.get_loc())
        major_tick_obj.tick1line.set_visible(False)
        major_tick_obj.tick2line.set_color(SIGNATURE_PALETTE_DICT['ink'])
        major_tick_obj.tick2line.set_markeredgewidth(0.8)
        major_tick_obj.tick2line.set_markersize(3.5)
        major_tick_obj.label2.set_color(SIGNATURE_PALETTE_DICT['ink'])
        major_tick_obj.label2.set_fontweight('normal')
        major_tick_obj.label2.set_gid(None)
        major_tick_obj.tick2line.set_gid(None)

        for promoted_tick_spec_dict in promoted_growth_tick_spec_list:
            promoted_growth_float = float(promoted_tick_spec_dict['growth_of_1_float'])
            if not np.isclose(tick_value_float, promoted_growth_float, atol=1e-9, rtol=0.0):
                continue

            promoted_color_str = str(promoted_tick_spec_dict['color_str'])
            major_tick_obj.tick2line.set_color(promoted_color_str)
            major_tick_obj.tick2line.set_markeredgewidth(1.2)
            major_tick_obj.tick2line.set_markersize(5.0)
            major_tick_obj.tick2line.set_gid(str(promoted_tick_spec_dict['axis_tick_id_str']))
            major_tick_obj.label2.set_color(promoted_color_str)
            major_tick_obj.label2.set_fontweight('semibold')
            major_tick_obj.label2.set_gid(str(promoted_tick_spec_dict['label_id_str']))
            break


def growth_of_1_major_formatter(x, pos):
    """Format Growth of $1 axis values with:

    displayed_growth_dollar_float = 1.0 * growth_of_1_float
    """
    return f'${x:,.2f}'


def fraction_major_formatter(x, pos):
    """Format drawdown and annual-return values with:

    displayed_return_pct = x * 100
    """
    return f'{x * 100:.0f}%'


def blank_minor_formatter(x, pos):
    """Keep minor tick labels empty for a cleaner look."""
    return ''
