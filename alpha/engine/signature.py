"""Signature marks: the recurring devices that make an artifact identifiably ours.

These are deliberately separate from ``theme.py``. The theme decides how things
look; this module supplies structural devices that appear across every report
regardless of which theme is active:

* ``build_title_block_html`` — the provenance block borrowed from engineering
  drawings. It states run identity, data vintage and cost assumptions on the
  face of the artifact rather than in a footnote.
* ``render_sparkline_data_uri_str`` — an axis-free trace sized to sit inline in
  a sentence or a table cell.
* ``render_small_multiples_data_uri_str`` — one grid of panels on shared axes,
  so a family of results is shown whole instead of one selected curve.
* ``apply_figure_stamp`` — the small provenance line on a chart itself.

None of these compute or transform quantitative results. They render values
supplied by the caller.
"""

from __future__ import annotations

import base64
import html
import io
import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.ticker import FixedLocator, FuncFormatter

from alpha.engine.theme import (
    SIGNATURE_PALETTE_DICT,
    apply_signature_time_axis,
    build_signature_rcparams,
)


_SPARKLINE_DPI_INT = 220
_SMALL_MULTIPLE_DPI_INT = 150


def _encode_figure_data_uri_str(
        figure_obj,
        dpi_int: int,
        transparent_bool: bool = False,
        pad_inch_float: float = 0.04,
) -> str:
    png_buffer = io.BytesIO()
    figure_obj.savefig(
        png_buffer,
        format='png',
        dpi=dpi_int,
        bbox_inches='tight',
        # A hairline pad clipped axis labels off the left edge of figures whose
        # tight_layout had already consumed the margin.
        pad_inches=pad_inch_float,
        transparent=transparent_bool,
    )
    plt.close(figure_obj)
    return 'data:image/png;base64,' + base64.b64encode(png_buffer.getvalue()).decode('ascii')


def render_sparkline_data_uri_str(
        value_ser: pd.Series,
        color_str: str | None = None,
        width_px_int: int = 96,
        height_px_int: int = 20,
        show_endpoint_bool: bool = True,
) -> str:
    """Render an axis-free inline trace of a value series.

    A sparkline carries shape only: no axis, no ticks, no gridlines and no
    zero baseline unless the series crosses it. It is meant to be read in
    passing, at the size of surrounding text, not measured against a scale.
    """
    plotted_value_vec = pd.Series(value_ser, copy=False).astype(float).dropna().to_numpy()
    if len(plotted_value_vec) < 2:
        return ''

    resolved_color_str = color_str or str(SIGNATURE_PALETTE_DICT['strategy'])
    figure_obj = plt.figure(
        figsize=(width_px_int / 100.0, height_px_int / 100.0), dpi=100
    )
    axis_obj = figure_obj.add_axes((0.0, 0.0, 1.0, 1.0))
    axis_obj.plot(
        np.arange(len(plotted_value_vec)),
        plotted_value_vec,
        color=resolved_color_str,
        linewidth=0.8,
        solid_capstyle='round',
    )

    if show_endpoint_bool:
        axis_obj.plot(
            [len(plotted_value_vec) - 1],
            [plotted_value_vec[-1]],
            marker='o',
            markersize=1.9,
            color=resolved_color_str,
        )

    value_span_float = float(plotted_value_vec.max() - plotted_value_vec.min()) or 1.0
    axis_obj.set_ylim(
        plotted_value_vec.min() - value_span_float * 0.18,
        plotted_value_vec.max() + value_span_float * 0.18,
    )
    axis_obj.set_axis_off()
    figure_obj.patch.set_alpha(0.0)
    # Sparklines are sized to sit inline in text, so they keep a zero pad.
    return _encode_figure_data_uri_str(
        figure_obj, _SPARKLINE_DPI_INT, transparent_bool=True, pad_inch_float=0.0
    )


def build_sparkline_img_html(
        value_ser: pd.Series,
        color_str: str | None = None,
        width_px_int: int = 96,
        height_px_int: int = 20,
) -> str:
    """Wrap a sparkline in an inline-baseline <img> so it sits inside text."""
    data_uri_str = render_sparkline_data_uri_str(
        value_ser, color_str=color_str, width_px_int=width_px_int, height_px_int=height_px_int
    )
    if not data_uri_str:
        return ''
    return (
        f'<img src="{data_uri_str}" alt="" width="{width_px_int}" height="{height_px_int}" '
        f'style="vertical-align:-0.25em;margin:0 0.15em;display:inline-block">'
    )


def render_small_multiples_data_uri_str(
        panel_ser_dict: dict[str, pd.Series],
        column_count_int: int = 4,
        share_ylim_bool: bool = True,
        panel_width_inch_float: float = 2.05,
        panel_height_inch_float: float = 1.25,
        value_formatter_fn=None,
        overlay_ser_dict: dict[str, pd.Series] | None = None,
) -> str:
    """Render a family of series as one grid of panels on shared axes.

    ``overlay_ser_dict`` draws an optional reference line in each panel, keyed
    by the same panel name — a benchmark against which the panel's own series
    is meant to be read. Panels with no entry simply have no reference line.

    *** CRITICAL*** All panels share a single y range by default. Per-panel
    autoscaling would make unequal moves look equal and is the main way a grid
    of small charts misleads; opt out only when the panels are not comparable.
    """
    if len(panel_ser_dict) == 0:
        raise ValueError('panel_ser_dict must contain at least one panel.')

    panel_name_list = list(panel_ser_dict)
    row_count_int = math.ceil(len(panel_name_list) / column_count_int)
    strategy_color_str = str(SIGNATURE_PALETTE_DICT['strategy'])
    benchmark_color_str = str(SIGNATURE_PALETTE_DICT['benchmark'])
    muted_color_str = str(SIGNATURE_PALETTE_DICT['muted'])
    grid_color_str = str(SIGNATURE_PALETTE_DICT['grid'])
    profit_color_str = str(SIGNATURE_PALETTE_DICT['profit_dark'])
    loss_color_str = str(SIGNATURE_PALETTE_DICT['loss_dark'])

    # *** CRITICAL*** The reference lines join the shared range. Scaling to the
    # panel series alone would clip exactly the case the comparison exists for
    # — a benchmark that fell further than the strategy — and the clipping is
    # silent, so the grid would understate how much was avoided.
    range_source_ser_list = list(panel_ser_dict.values())
    if overlay_ser_dict:
        range_source_ser_list += [
            overlay_value_ser
            for overlay_value_ser in overlay_ser_dict.values()
            if len(pd.Series(overlay_value_ser)) > 0
        ]
    shared_min_float = min(float(pd.Series(s).min()) for s in range_source_ser_list)
    shared_max_float = max(float(pd.Series(s).max()) for s in range_source_ser_list)
    shared_pad_float = (shared_max_float - shared_min_float) * 0.12 or 0.1

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        figure_obj, axis_arr = plt.subplots(
            row_count_int,
            column_count_int,
            figsize=(
                panel_width_inch_float * column_count_int,
                panel_height_inch_float * row_count_int,
            ),
            sharex=False,
            sharey=share_ylim_bool,
        )
        axis_list = np.atleast_1d(axis_arr).ravel().tolist()

        for panel_idx_int, axis_obj in enumerate(axis_list):
            if panel_idx_int >= len(panel_name_list):
                axis_obj.set_visible(False)
                continue

            panel_name_str = panel_name_list[panel_idx_int]
            panel_value_ser = pd.Series(panel_ser_dict[panel_name_str], copy=False).astype(float)

            # Reference first, so the panel's own series is never overdrawn.
            if overlay_ser_dict and panel_name_str in overlay_ser_dict:
                overlay_value_ser = pd.Series(
                    overlay_ser_dict[panel_name_str], copy=False
                ).astype(float)
                if len(overlay_value_ser) > 0:
                    axis_obj.plot(
                        np.arange(len(overlay_value_ser)),
                        overlay_value_ser.to_numpy(),
                        color=benchmark_color_str,
                        linewidth=0.75,
                        linestyle=(0, (2.5, 1.5)),
                    )
            axis_obj.plot(
                np.arange(len(panel_value_ser)),
                panel_value_ser.to_numpy(),
                color=strategy_color_str,
                linewidth=0.85,
            )
            axis_obj.axhline(
                0.0 if shared_min_float < 0.0 < shared_max_float else shared_min_float,
                color=grid_color_str,
                linewidth=0.5,
            )
            if share_ylim_bool:
                axis_obj.set_ylim(shared_min_float - shared_pad_float, shared_max_float + shared_pad_float)

            # Print each panel's own outcome. Without it a grid of small charts
            # shows relative shape but cannot be read for magnitude, which is
            # what turns the device from evidence into decoration.
            #
            # The label is drawn in two parts so the outcome can carry its
            # sign's colour while the panel name stays neutral. The *line* is
            # deliberately left in ink: a year's path is not positive or
            # negative, only its outcome is, and colouring the whole curve by
            # its endpoint would claim the year was never under water.
            axis_obj.set_title(
                panel_name_str,
                fontsize=6.5,
                color=muted_color_str,
                loc='left',
                pad=3.0,
            )
            if value_formatter_fn is not None and len(panel_value_ser) > 0:
                panel_outcome_float = float(panel_value_ser.iloc[-1])
                axis_obj.set_title(
                    value_formatter_fn(panel_outcome_float),
                    fontsize=6.5,
                    color=(
                        loss_color_str if panel_outcome_float < 0.0 else profit_color_str
                    ),
                    loc='right',
                    pad=3.0,
                )
            for spine_name_str in ('top', 'right', 'left'):
                axis_obj.spines[spine_name_str].set_visible(False)
            axis_obj.spines['bottom'].set_color(grid_color_str)
            axis_obj.spines['bottom'].set_linewidth(0.5)
            axis_obj.set_xticks([])
            axis_obj.grid(False)

            # Only the leftmost panel of each row carries a scale; repeating it
            # on every panel is ink that encodes nothing new.
            #
            # *** CRITICAL*** With sharey=True the tick locations belong to the
            # shared axis, so clearing them on one panel clears them on all.
            # Hide the *labels* per panel instead of removing the ticks.
            if value_formatter_fn is not None:
                axis_obj.set_yticks([shared_min_float, shared_max_float])
                axis_obj.set_yticklabels(
                    [value_formatter_fn(shared_min_float), value_formatter_fn(shared_max_float)]
                )
            else:
                axis_obj.set_yticks([])

            panel_is_row_leader_bool = panel_idx_int % column_count_int == 0
            axis_obj.tick_params(
                axis='y',
                labelsize=6.0,
                colors=muted_color_str,
                length=0.0,
                labelleft=panel_is_row_leader_bool and value_formatter_fn is not None,
            )

        figure_obj.tight_layout(pad=0.4, h_pad=1.0, w_pad=0.8)
        return _encode_figure_data_uri_str(figure_obj, _SMALL_MULTIPLE_DPI_INT)


_SLEEVE_MAX_DISTINCT_NAME_COUNT_INT = 12

SLEEVE_COMPOSITION_MODE_STR = 'sleeve'
ROTATION_COMPOSITION_MODE_STR = 'rotation'


def _position_weight_df(holding_weight_df: pd.DataFrame) -> pd.DataFrame:
    """Return investable holdings only; Cash is the undeployed remainder."""
    return holding_weight_df.drop(columns=['Cash'], errors='ignore')


def _is_held_df(holding_weight_df: pd.DataFrame) -> pd.DataFrame:
    position_weight_df = _position_weight_df(holding_weight_df)
    if position_weight_df.shape[1] == 0:
        raise ValueError(
            'holding_weight_df must contain at least one name column after '
            'excluding Cash.'
        )

    is_held_df = position_weight_df.fillna(0.0).abs().gt(0.0)
    if not is_held_df.to_numpy().any():
        raise ValueError('holding_weight_df contains no held positions.')
    return is_held_df


def detect_composition_mode_str(holding_weight_df: pd.DataFrame) -> str:
    """Decide which composition view a book needs, from the book itself.

    The rule is deliberately about *how many distinct names the book ever
    touched*, not about turnover or concurrency:

        mode = sleeve    if distinct names ever held <= 12
        mode = rotation  otherwise

    A stacked area by name is only readable while its legend is readable. A
    persistent-sleeve book (a handful of ETFs held throughout) stays under the
    threshold for its whole life; a slot rotation crosses it almost immediately
    and never comes back, so the classification is stable rather than flapping
    between views as positions change.
    """
    distinct_name_count_int = int(_is_held_df(holding_weight_df).any(axis=0).sum())
    return (
        SLEEVE_COMPOSITION_MODE_STR
        if distinct_name_count_int <= _SLEEVE_MAX_DISTINCT_NAME_COUNT_INT
        else ROTATION_COMPOSITION_MODE_STR
    )


def compute_holding_period_length_list(holding_weight_df: pd.DataFrame) -> list[int]:
    """Length in bars of every completed and open holding spell, per name.

    A name re-entered later contributes a separate spell rather than one merged
    span, because two six-week holds and one twelve-week hold are different
    trading behaviour with different costs.
    """
    is_held_arr = _is_held_df(holding_weight_df).to_numpy()
    holding_period_length_list: list[int] = []

    for column_idx_int in range(is_held_arr.shape[1]):
        held_flag_vec = is_held_arr[:, column_idx_int].astype(int)
        padded_flag_vec = np.concatenate(([0], held_flag_vec, [0]))
        transition_vec = np.diff(padded_flag_vec)
        spell_start_vec = np.flatnonzero(transition_vec == 1)
        spell_end_vec = np.flatnonzero(transition_vec == -1)
        holding_period_length_list.extend((spell_end_vec - spell_start_vec).tolist())

    return holding_period_length_list


# Below this a name is carrying rebalance residue rather than a position.
_MATERIAL_WEIGHT_FLOOR_FLOAT = 0.005


def _weight_axis_tick_list(max_weight_float: float) -> list[float]:
    """Ticks in 20-point steps up to the largest weight the book actually used.

    A fixed 20/40 pair leaves anything above it unlabelled, so a name running
    at 60% is read against the nearest tick below it and understates by a third.
    Steps stop at the last one the data reaches: a tick drawn above every bar
    in the figure labels empty space and costs the rows vertical room.
    """
    if not np.isfinite(max_weight_float) or max_weight_float <= 0.0:
        return [0.2]
    tick_count_int = max(1, int(math.floor(round(max_weight_float, 6) / 0.2)))
    return [round(0.2 * (step_int + 1), 2) for step_int in range(tick_count_int)]


def _render_sleeve_composition_figure(holding_weight_df: pd.DataFrame):
    """One row per name, each drawn from its own zero, plus a gross strip.

    Stacking was the previous device and it is the harder one to read: only the
    bottom band sits on a common baseline, so every other name has to be
    measured by subtracting the band beneath it while also decoding which hatch
    belongs to whom. Given a row apiece the weight is read straight off the
    axis. The gross strip keeps the one thing stacking did show — the book's
    total — which separate rows would otherwise lose.
    """
    ink_color_str = str(SIGNATURE_PALETTE_DICT['ink'])
    muted_color_str = str(SIGNATURE_PALETTE_DICT['muted'])
    grid_color_str = str(SIGNATURE_PALETTE_DICT['grid'])

    weight_df = holding_weight_df.fillna(0.0)
    name_order_index = weight_df.abs().mean().sort_values(ascending=False).index
    gross_exposure_ser = weight_df.abs().sum(axis=1)
    max_weight_float = float(weight_df.abs().to_numpy().max()) if weight_df.size else 0.0
    weight_tick_list = _weight_axis_tick_list(max_weight_float)

    figure_obj, axis_arr = plt.subplots(
        len(name_order_index) + 1,
        1,
        figsize=(9.5, 0.62 + 0.62 * len(name_order_index)),
        sharex=True,
        gridspec_kw={'height_ratios': [0.62] + [1.0] * len(name_order_index)},
    )
    axis_list = np.atleast_1d(axis_arr).ravel().tolist()

    def strip_axis(axis_obj):
        for spine_name_str in ('top', 'right', 'left'):
            axis_obj.spines[spine_name_str].set_visible(False)
        axis_obj.spines['bottom'].set_color(grid_color_str)
        axis_obj.spines['bottom'].set_linewidth(0.5)
        axis_obj.tick_params(length=0.0)

    gross_axis_obj = axis_list[0]
    gross_axis_obj.fill_between(
        weight_df.index, 0.0, gross_exposure_ser.to_numpy(),
        color=muted_color_str, linewidth=0.0, alpha=0.5, step='post',
    )
    gross_axis_obj.set_ylim(0.0, max(1.05, float(gross_exposure_ser.max() or 1.0) * 1.05))
    gross_axis_obj.set_yticks([1.0])
    gross_axis_obj.set_yticklabels(['100%'], fontsize=6.5, color=muted_color_str)
    gross_axis_obj.set_ylabel(
        'Gross', rotation=0, ha='right', va='center', fontsize=7.5,
        color=muted_color_str, labelpad=8,
    )
    strip_axis(gross_axis_obj)

    for axis_obj, name_str in zip(axis_list[1:], name_order_index):
        name_weight_ser = weight_df[name_str].abs()
        axis_obj.fill_between(
            weight_df.index, 0.0, name_weight_ser.to_numpy(),
            color=ink_color_str, linewidth=0.0, alpha=0.82, step='post',
        )
        # The ceiling follows the data, not the last tick, so a bar between two
        # steps is still drawn in full rather than clipped at the gridline.
        axis_obj.set_ylim(0.0, max(max(weight_tick_list), max_weight_float) * 1.08)
        axis_obj.set_yticks(weight_tick_list)
        axis_obj.set_yticklabels(
            [f'{tick_float * 100:.0f}' for tick_float in weight_tick_list],
            fontsize=6.5, color=muted_color_str,
        )
        for tick_float in weight_tick_list:
            axis_obj.axhline(tick_float, color=grid_color_str, linewidth=0.5, zorder=0)
        axis_obj.set_ylabel(
            str(name_str), rotation=0, ha='right', va='center', fontsize=8.5,
            color=ink_color_str, labelpad=8,
        )
        # The share of days held, so the row carries a figure and not only a
        # shape — the same reading Year by Year prints beside each panel.
        #
        # *** CRITICAL*** Counted against a materiality floor, not against
        # zero. Rebalancing leaves fractional residue in a name for long
        # stretches: this book's Cash line is above zero on every single day
        # with a median of 0.23%, so a bare > 0 test reports "100% of days"
        # for what is rounding dust rather than a position.
        held_share_float = float((name_weight_ser > _MATERIAL_WEIGHT_FLOOR_FLOAT).mean())
        axis_obj.text(
            1.005, 0.5, f'{held_share_float * 100:.0f}% of days',
            transform=axis_obj.transAxes, fontsize=6.8,
            color=muted_color_str, va='center',
        )
        strip_axis(axis_obj)
        axis_obj.margins(x=0.008)

    apply_signature_time_axis(axis_list[-1], weight_df.index)
    return figure_obj, tuple(axis_list)


def _render_rotation_composition_figure(
        holding_weight_df: pd.DataFrame,
        slot_capacity_int: int | None,
):
    """Exposure, deployment and holding periods — what a rotating book needs.

    Names are not plotted at all. In an equal-weight slot strategy every held
    name carries the same weight by construction, so a weight chart per name
    encodes nothing. What varies is how much capital is deployed, how many
    slots are filled, and how long positions are kept.
    """
    is_held_df = _is_held_df(holding_weight_df)
    gross_exposure_ser = holding_weight_df.fillna(0.0).abs().sum(axis=1)
    held_name_count_ser = is_held_df.sum(axis=1)
    resolved_capacity_int = int(slot_capacity_int or held_name_count_ser.max())
    holding_period_length_list = compute_holding_period_length_list(holding_weight_df)

    strategy_color_str = str(SIGNATURE_PALETTE_DICT['strategy'])
    ink_color_str = str(SIGNATURE_PALETTE_DICT['ink'])
    muted_color_str = str(SIGNATURE_PALETTE_DICT['muted'])
    grid_color_str = str(SIGNATURE_PALETTE_DICT['grid'])

    figure_obj = plt.figure(figsize=(9.5, 5.6))
    grid_spec_obj = figure_obj.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 1.2], hspace=0.42)
    exposure_ax = figure_obj.add_subplot(grid_spec_obj[0])
    occupancy_ax = figure_obj.add_subplot(grid_spec_obj[1], sharex=exposure_ax)
    holding_ax = figure_obj.add_subplot(grid_spec_obj[2])

    # Deployed capital against the idle remainder. For a volatility-scaled book
    # the gap to 100% is the scaler doing its job, and it is the single most
    # important composition fact — invisible on any per-name chart.
    exposure_ax.fill_between(
        holding_weight_df.index, 0.0, gross_exposure_ser.to_numpy(),
        color=strategy_color_str, alpha=0.30,
    )
    exposure_ax.plot(
        holding_weight_df.index, gross_exposure_ser.to_numpy(),
        color=strategy_color_str, linewidth=0.9,
    )
    exposure_ax.axhline(1.0, color=ink_color_str, linestyle='--', linewidth=0.7)
    exposure_ax.set_ylabel('Gross exposure')
    exposure_ax.set_ylim(0.0, max(1.08, float(gross_exposure_ser.max()) * 1.08))
    exposure_ax.set_title('Deployed capital — the gap to 1.0 is cash')
    exposure_ax.margins(x=0.008)

    occupancy_ax.fill_between(
        holding_weight_df.index, held_name_count_ser.to_numpy(), float(resolved_capacity_int),
        color=muted_color_str, alpha=0.22, step='post',
    )
    occupancy_ax.step(
        holding_weight_df.index, held_name_count_ser.to_numpy(),
        color=strategy_color_str, linewidth=0.8, where='post',
    )
    occupancy_ax.axhline(resolved_capacity_int, color=ink_color_str, linestyle='--', linewidth=0.7)
    occupancy_ax.set_ylabel(f'Slots / {resolved_capacity_int}')
    occupancy_ax.set_ylim(0, resolved_capacity_int * 1.12)
    occupancy_ax.margins(x=0.008)

    holding_period_vec = np.asarray(holding_period_length_list, dtype=float)
    median_holding_period_float = float(np.median(holding_period_vec))
    # *** CRITICAL*** A few names held almost the whole backtest stretch the
    # x-axis to thousands of bars and crush the bulk into one bar. Cap the
    # *display* at the 98th percentile so the mass is readable; the top bin
    # absorbs the long holds. The median annotation stays on the true data.
    display_cap_float = max(float(np.percentile(holding_period_vec, 98.0)), 10.0)
    clipped_holding_period_vec = np.clip(holding_period_vec, None, display_cap_float)
    long_hold_count_int = int((holding_period_vec > display_cap_float).sum())
    holding_ax.hist(
        clipped_holding_period_vec,
        bins=min(30, max(6, len(set(clipped_holding_period_vec.tolist())))),
        color=strategy_color_str, alpha=0.75,
        edgecolor=str(SIGNATURE_PALETTE_DICT['bar_edge']), linewidth=0.5,
    )
    holding_ax.axvline(
        min(median_holding_period_float, display_cap_float),
        color=ink_color_str, linestyle='--', linewidth=0.8,
    )
    holding_ax.set_xlim(0.0, display_cap_float)
    holding_ax.set_ylabel('Spells')
    long_hold_note_str = (
        f', {long_hold_count_int} beyond {display_cap_float:.0f}' if long_hold_count_int else ''
    )
    holding_ax.set_xlabel(
        f'Holding period (bars) — median {median_holding_period_float:.0f}, '
        f'{len(holding_period_length_list)} spells{long_hold_note_str}'
    )

    for axis_obj in (exposure_ax, occupancy_ax, holding_ax):
        for spine_name_str in ('top', 'right', 'left'):
            axis_obj.spines[spine_name_str].set_visible(False)
        axis_obj.spines['bottom'].set_color(grid_color_str)
        axis_obj.spines['bottom'].set_linewidth(0.6)
        axis_obj.grid(False)

    exposure_ax.tick_params(axis='x', labelbottom=False)
    apply_signature_time_axis(occupancy_ax, holding_weight_df.index)
    return figure_obj, (exposure_ax, occupancy_ax, holding_ax)


def render_composition_data_uri_str(
        holding_weight_df: pd.DataFrame,
        slot_capacity_int: int | None = None,
        composition_mode_str: str | None = None,
) -> tuple[str, str]:
    """Render the composition view a book's own shape calls for.

    Returns ``(data_uri_str, resolved_mode_str)`` so the caller can caption the
    figure with the mode that was actually used rather than assuming one.
    Pass ``composition_mode_str`` to override the detection.
    """
    position_weight_df = _position_weight_df(holding_weight_df)
    resolved_mode_str = composition_mode_str or detect_composition_mode_str(
        position_weight_df
    )
    if resolved_mode_str not in (SLEEVE_COMPOSITION_MODE_STR, ROTATION_COMPOSITION_MODE_STR):
        raise ValueError(
            f'Unknown composition mode {resolved_mode_str!r}. Expected '
            f'{SLEEVE_COMPOSITION_MODE_STR!r} or {ROTATION_COMPOSITION_MODE_STR!r}.'
        )

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        if resolved_mode_str == SLEEVE_COMPOSITION_MODE_STR:
            figure_obj, axis_tuple = _render_sleeve_composition_figure(
                position_weight_df
            )
        else:
            figure_obj, axis_tuple = _render_rotation_composition_figure(
                position_weight_df, slot_capacity_int
            )

        for axis_obj in axis_tuple:
            axis_obj.set_axisbelow(True)
        figure_obj.align_ylabels(axis_tuple)
        # The rotation figure sets its own gridspec spacing, which tight_layout
        # would override and warn about; only the single-axes sleeve figure
        # benefits from it.
        if resolved_mode_str == SLEEVE_COMPOSITION_MODE_STR:
            figure_obj.tight_layout(pad=0.5)
        return _encode_figure_data_uri_str(figure_obj, _SMALL_MULTIPLE_DPI_INT), resolved_mode_str


def compute_conditional_beta_dict(
        strategy_return_ser: pd.Series,
        benchmark_return_ser: pd.Series,
        threshold_float: float = 0.0,
) -> dict[str, float]:
    """Split beta, correlation and capture by the sign of the benchmark day.

    A single full-sample beta averages two regimes that a defensive strategy is
    specifically built to separate. What matters is whether the book tracks the
    benchmark less on the way down than on the way up.

    For the down set D = {t : r_b,t < threshold} and the up set U likewise:

        beta_D    = cov(r_s, r_b | D) / var(r_b | D)
        capture_D = mean(r_s | D) / mean(r_b | D)

    *** CRITICAL*** Conditioning is on the **benchmark's** sign, never the
    strategy's. Selecting on the strategy's own bad days would guarantee a
    flattering number by construction.
    """
    aligned_df = pd.concat(
        {'strategy': pd.Series(strategy_return_ser).astype(float),
         'benchmark': pd.Series(benchmark_return_ser).astype(float)},
        axis=1,
    ).dropna()
    if len(aligned_df) < 3:
        raise ValueError('Need at least three overlapping return observations.')

    conditional_metric_dict: dict[str, float] = {}
    for regime_name_str, regime_mask_ser in (
        ('down', aligned_df['benchmark'] < threshold_float),
        ('up', aligned_df['benchmark'] >= threshold_float),
    ):
        regime_df = aligned_df.loc[regime_mask_ser]
        if len(regime_df) < 3:
            raise ValueError(
                f'Only {len(regime_df)} {regime_name_str}-market observations; '
                'conditional beta needs at least three.'
            )
        benchmark_variance_float = float(regime_df['benchmark'].var())
        conditional_metric_dict[f'{regime_name_str}_beta_float'] = (
            float(regime_df['strategy'].cov(regime_df['benchmark'])) / benchmark_variance_float
        )
        conditional_metric_dict[f'{regime_name_str}_correlation_float'] = float(
            regime_df['strategy'].corr(regime_df['benchmark'])
        )
        conditional_metric_dict[f'{regime_name_str}_capture_float'] = (
            float(regime_df['strategy'].mean()) / float(regime_df['benchmark'].mean())
        )
        conditional_metric_dict[f'{regime_name_str}_day_count_float'] = float(len(regime_df))

    conditional_metric_dict['beta_asymmetry_float'] = (
        conditional_metric_dict['up_beta_float'] - conditional_metric_dict['down_beta_float']
    )
    return conditional_metric_dict


_RELATIVE_RATIO_TICK_CANDIDATE_LIST = [
    0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5,
    3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0,
]


def render_relative_performance_data_uri_str(
        strategy_total_value_ser: pd.Series,
        benchmark_total_value_ser: pd.Series,
) -> str:
    """Cumulative relative performance: strategy ÷ benchmark on a log scale.

        ratio_t = (V_s,t / V_s,0) / (V_b,t / V_b,0)

    One line, no parameters. Slope up means beating the benchmark, flat means
    matching it, down means lagging — and edge decay shows as flattening with
    no window or smoothing choice to defend. This is the parameter-free answer
    to the question rolling Sharpe is usually asked.

    *** CRITICAL*** Both series are rebased at their first *common* bar and the
    ratio is computed on aligned dates only. Rebasing each series at its own
    start while the calendars differ would fabricate relative performance from
    the non-overlapping stub.
    """
    aligned_df = pd.concat(
        {'strategy': pd.Series(strategy_total_value_ser).astype(float),
         'benchmark': pd.Series(benchmark_total_value_ser).astype(float)},
        axis=1,
    ).dropna()
    if len(aligned_df) < 2:
        raise ValueError('Need at least two overlapping total-value observations.')
    if bool((aligned_df <= 0.0).to_numpy().any()):
        raise ValueError('Total value must be strictly positive for a log-scale ratio.')

    relative_ratio_ser = (
        (aligned_df['strategy'] / aligned_df['strategy'].iloc[0])
        / (aligned_df['benchmark'] / aligned_df['benchmark'].iloc[0])
    )

    strategy_color_str = str(SIGNATURE_PALETTE_DICT['strategy'])
    zero_line_color_str = str(SIGNATURE_PALETTE_DICT['zero_line'])
    grid_color_str = str(SIGNATURE_PALETTE_DICT['grid'])

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        figure_obj, axis_obj = plt.subplots(figsize=(9.5, 2.6))
        axis_obj.plot(
            relative_ratio_ser.index, relative_ratio_ser.to_numpy(),
            color=strategy_color_str, linewidth=1.0,
        )
        axis_obj.fill_between(
            relative_ratio_ser.index, 1.0, relative_ratio_ser.to_numpy(),
            color=strategy_color_str, alpha=0.10,
        )
        axis_obj.axhline(1.0, color=zero_line_color_str, linestyle='--', linewidth=0.7)
        axis_obj.set_yscale('log')

        ratio_min_float = float(relative_ratio_ser.min())
        ratio_max_float = float(relative_ratio_ser.max())
        ratio_tick_list = [
            tick_float for tick_float in _RELATIVE_RATIO_TICK_CANDIDATE_LIST
            if ratio_min_float / 1.04 <= tick_float <= ratio_max_float * 1.04
        ]
        if 1.0 not in ratio_tick_list:
            ratio_tick_list.append(1.0)
        axis_obj.yaxis.set_major_locator(FixedLocator(sorted(ratio_tick_list)))
        axis_obj.yaxis.set_major_formatter(
            FuncFormatter(lambda tick_float, _pos: f'{tick_float:g}×')
        )
        axis_obj.minorticks_off()

        axis_obj.set_ylabel('Strategy ÷ benchmark')
        axis_obj.set_title('Relative performance — rising = pulling ahead')
        axis_obj.margins(x=0.008)
        for spine_name_str in ('top', 'right', 'left'):
            axis_obj.spines[spine_name_str].set_visible(False)
        axis_obj.spines['bottom'].set_color(grid_color_str)
        axis_obj.spines['bottom'].set_linewidth(0.6)
        axis_obj.grid(False)
        apply_signature_time_axis(axis_obj, relative_ratio_ser.index)
        figure_obj.tight_layout(pad=0.5)
        return _encode_figure_data_uri_str(figure_obj, _SMALL_MULTIPLE_DPI_INT)


def apply_figure_stamp(figure_obj, stamp_str: str) -> None:
    """Print a small provenance line along the bottom edge of a figure."""
    figure_obj.text(
        0.995,
        0.004,
        stamp_str,
        ha='right',
        va='bottom',
        fontsize=5.5,
        color=str(SIGNATURE_PALETTE_DICT['muted']),
    )


def _metric_domain_span_float(metric_spec_dict: dict[str, object]) -> float:
    domain_span_float = (
        float(metric_spec_dict['domain_max_float']) - float(metric_spec_dict['domain_min_float'])
    )
    if domain_span_float <= 0.0:
        raise ValueError(
            f'Metric {metric_spec_dict["label_str"]!r} has a non-positive scale domain.'
        )
    return domain_span_float


def _metric_position_pct_float(value_float: float, metric_spec_dict: dict[str, object]) -> float:
    """Place a reading on its metric's scale, as a percentage of the track.

    *** CRITICAL*** Readings are clamped to the domain. An out-of-domain value
    must pin to the end of the track rather than overflow it, which would draw
    a bar wider than its own scale and misreport the position.
    """
    domain_min_float = float(metric_spec_dict['domain_min_float'])
    domain_max_float = float(metric_spec_dict['domain_max_float'])
    bounded_float = min(max(float(value_float), domain_min_float), domain_max_float)
    return (bounded_float - domain_min_float) / _metric_domain_span_float(metric_spec_dict) * 100.0


def _metric_is_favourable_bool(metric_spec_dict: dict[str, object]) -> bool | None:
    """Whether the reading beats its benchmark, given the metric's direction."""
    benchmark_value_obj = metric_spec_dict.get('benchmark_float')
    if benchmark_value_obj is None:
        return None
    difference_float = float(metric_spec_dict['value_float']) - float(benchmark_value_obj)
    if difference_float == 0.0:
        return None
    return (difference_float > 0.0) == bool(metric_spec_dict.get('higher_is_better_bool', True))


def _metric_label_html(metric_spec_dict: dict[str, object]) -> str:
    return (
        '<div style="font-family:var(--font-figure);font-size:0.6rem;letter-spacing:0.1em;'
        'text-transform:uppercase;color:var(--color-muted)">'
        f'{html.escape(str(metric_spec_dict["label_str"]))}</div>'
    )


def _metric_caption_html(caption_str: str) -> str:
    return (
        '<div style="font-family:var(--font-figure);font-size:0.58rem;letter-spacing:0.1em;'
        'text-transform:uppercase;color:var(--color-muted);margin-top:10px">'
        f'{html.escape(caption_str)}</div>'
    )


def build_metric_dumbbell_html(metric_spec_list: list[dict[str, object]]) -> str:
    """Paired-dot comparison: strategy and benchmark on one shared track.

    A Cleveland dot plot. The connector between the two dots *is* the edge, so
    the size of the gap is read directly rather than computed from two numbers
    in adjacent columns. Minimal ink, and the direction of every metric is
    visible at a glance down the column of dots.
    """
    if len(metric_spec_list) == 0:
        raise ValueError('metric_spec_list must contain at least one metric.')

    row_html_list = []
    for metric_spec_dict in metric_spec_list:
        value_position_pct_float = _metric_position_pct_float(
            float(metric_spec_dict['value_float']), metric_spec_dict
        )
        benchmark_value_obj = metric_spec_dict.get('benchmark_float')
        connector_html_str = ''
        benchmark_dot_html_str = ''
        benchmark_display_str = '—'

        if benchmark_value_obj is not None:
            benchmark_position_pct_float = _metric_position_pct_float(
                float(benchmark_value_obj), metric_spec_dict
            )
            benchmark_display_str = str(
                metric_spec_dict.get('benchmark_display_str', f'{float(benchmark_value_obj):.2f}')
            )
            connector_left_pct_float = min(value_position_pct_float, benchmark_position_pct_float)
            connector_width_pct_float = abs(value_position_pct_float - benchmark_position_pct_float)
            connector_html_str = (
                '<div style="position:absolute;top:50%;height:1px;transform:translateY(-50%);'
                f'left:{connector_left_pct_float:.2f}%;width:{connector_width_pct_float:.2f}%;'
                'background:var(--color-ink);opacity:0.45"></div>'
            )
            benchmark_dot_html_str = (
                '<div style="position:absolute;top:50%;width:7px;height:7px;border-radius:50%;'
                'transform:translate(-50%,-50%);background:var(--color-page);'
                'border:1px solid var(--color-ink);'
                f'left:{benchmark_position_pct_float:.2f}%"></div>'
            )

        row_html_list.append(
            '<div style="display:grid;grid-template-columns:126px 1fr 66px 66px;gap:14px;'
            'align-items:center;padding:9px 0;border-bottom:1px solid var(--color-border)">'
            + _metric_label_html(metric_spec_dict)
            + '<div style="position:relative;height:14px">'
            '<div style="position:absolute;top:50%;left:0;right:0;height:1px;'
            'transform:translateY(-50%);background:var(--color-grid)"></div>'
            + connector_html_str + benchmark_dot_html_str
            + '<div style="position:absolute;top:50%;width:9px;height:9px;border-radius:50%;'
            'transform:translate(-50%,-50%);background:var(--color-strategy);'
            f'left:{value_position_pct_float:.2f}%"></div></div>'
            '<div style="font-family:var(--font-figure);font-size:0.84rem;text-align:right;'
            'font-variant-numeric:tabular-nums;color:var(--color-ink)">'
            f'{html.escape(str(metric_spec_dict["display_str"]))}</div>'
            '<div style="font-family:var(--font-figure);font-size:0.76rem;text-align:right;'
            'font-variant-numeric:tabular-nums;color:var(--color-muted)">'
            f'{html.escape(benchmark_display_str)}</div>'
            '</div>'
        )

    return (
        '<div>' + ''.join(row_html_list)
        + _metric_caption_html('Filled dot = strategy · hollow dot = benchmark')
        + '</div>'
    )


def build_metric_delta_table_html(metric_spec_list: list[dict[str, object]]) -> str:
    """Pure-typographic comparison: value, benchmark, and the difference.

    No graphics at all. The delta column is the whole design — it states the
    edge rather than leaving the reader to subtract two columns in their head.
    """
    if len(metric_spec_list) == 0:
        raise ValueError('metric_spec_list must contain at least one metric.')

    row_html_list = []
    for metric_spec_dict in metric_spec_list:
        benchmark_value_obj = metric_spec_dict.get('benchmark_float')
        favourable_bool = _metric_is_favourable_bool(metric_spec_dict)
        delta_color_str = {
            True: 'var(--color-profit-dark)',
            False: 'var(--color-loss-dark)',
            None: 'var(--color-muted)',
        }[favourable_bool]

        if benchmark_value_obj is None:
            benchmark_display_str = '—'
            delta_display_str = '—'
        else:
            benchmark_display_str = str(
                metric_spec_dict.get('benchmark_display_str', f'{float(benchmark_value_obj):.2f}')
            )
            delta_display_str = str(metric_spec_dict.get('delta_display_str', ''))

        row_html_list.append(
            f'<tr><td class="metric">{html.escape(str(metric_spec_dict["label_str"]))}</td>'
            f'<td>{html.escape(str(metric_spec_dict["display_str"]))}</td>'
            f'<td>{html.escape(benchmark_display_str)}</td>'
            f'<td style="color:{delta_color_str}">{html.escape(delta_display_str)}</td></tr>'
        )

    # Named, not just a generic scroll container. This table is the report's
    # headline claim, and a layout that wants to place it — beside the equity
    # curve rather than above it — needs a hook that says what it is. The Bench
    # workspace already synthesises the same class name when it rebuilds this
    # table from a saved artifact, so both paths agree.
    return (
        '<div class="scroll headline-comparison"><table class="stats-table">'
        '<thead><tr><th>Metric</th><th>Strategy</th><th>Benchmark</th><th>Delta</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def build_metric_deviation_html(metric_spec_list: list[dict[str, object]]) -> str:
    """Deviation from benchmark on a shared centred axis.

    Every metric is reduced to one question — better or worse than the
    reference, and by how much — so the whole record reads as a single shape.
    Bars extend right when favourable and left when not, with direction taken
    from each metric's ``higher_is_better_bool`` rather than its raw sign.
    """
    if len(metric_spec_list) == 0:
        raise ValueError('metric_spec_list must contain at least one metric.')

    row_html_list = []
    for metric_spec_dict in metric_spec_list:
        benchmark_value_obj = metric_spec_dict.get('benchmark_float')
        bar_html_str = ''

        if benchmark_value_obj is not None:
            deviation_fraction_float = (
                float(metric_spec_dict['value_float']) - float(benchmark_value_obj)
            ) / _metric_domain_span_float(metric_spec_dict)
            favourable_bool = _metric_is_favourable_bool(metric_spec_dict)
            # Half the track represents a full domain-span deviation; clamp so a
            # large gap saturates instead of escaping its own axis.
            bar_width_pct_float = min(abs(deviation_fraction_float) * 100.0, 50.0)
            bar_color_str = (
                'var(--color-loss)' if favourable_bool is False else 'var(--color-strategy)'
            )
            side_style_str = (
                f'left:50%;width:{bar_width_pct_float:.2f}%' if favourable_bool is not False
                else f'right:50%;width:{bar_width_pct_float:.2f}%'
            )
            bar_html_str = (
                f'<div style="position:absolute;top:2px;bottom:2px;{side_style_str};'
                f'background:{bar_color_str}"></div>'
            )

        row_html_list.append(
            '<div style="display:grid;grid-template-columns:126px 66px 1fr;gap:14px;'
            'align-items:center;padding:7px 0;border-bottom:1px solid var(--color-border)">'
            + _metric_label_html(metric_spec_dict)
            + '<div style="font-family:var(--font-figure);font-size:0.84rem;text-align:right;'
            'font-variant-numeric:tabular-nums;color:var(--color-ink)">'
            f'{html.escape(str(metric_spec_dict["display_str"]))}</div>'
            '<div style="position:relative;height:13px;background:var(--color-neutral)">'
            '<div style="position:absolute;top:-2px;bottom:-2px;left:50%;width:1px;'
            'background:var(--color-ink)"></div>'
            f'{bar_html_str}</div>'
            '</div>'
        )

    return (
        '<div>' + ''.join(row_html_list)
        + _metric_caption_html('Centre rule = benchmark · right = better · left = worse')
        + '</div>'
    )


def build_metric_scale_html(metric_spec_list: list[dict[str, object]]) -> str:
    """Draw each metric as a calibrated scale rather than a bare number.

    A stated value answers "what is it"; a position on a scale answers "is that
    a lot", which is the question a reader actually has. Each row shows the
    measured value as a bar against its plausible domain, with the benchmark
    marked on the same axis so the comparison needs no second column.

    Each spec supplies ``label_str``, ``value_float``, ``display_str``, a
    ``domain_min_float``/``domain_max_float`` pair defining the scale, and
    optionally ``benchmark_float`` plus ``is_adverse_bool``.
    """
    if len(metric_spec_list) == 0:
        raise ValueError('metric_spec_list must contain at least one metric.')

    row_html_list = []
    for metric_spec_dict in metric_spec_list:
        value_position_pct_float = _metric_position_pct_float(
            float(metric_spec_dict['value_float']), metric_spec_dict
        )
        bar_color_str = (
            'var(--color-loss)' if bool(metric_spec_dict.get('is_adverse_bool', False))
            else 'var(--color-strategy)'
        )

        benchmark_mark_html_str = ''
        benchmark_value_obj = metric_spec_dict.get('benchmark_float')
        if benchmark_value_obj is not None:
            benchmark_position_pct_float = _metric_position_pct_float(
                float(benchmark_value_obj), metric_spec_dict
            )
            benchmark_mark_html_str = (
                '<div style="position:absolute;top:-3px;bottom:-3px;width:1px;'
                f'left:{benchmark_position_pct_float:.2f}%;background:var(--color-ink)"></div>'
            )

        row_html_list.append(
            '<div style="display:grid;grid-template-columns:132px 78px 1fr;gap:14px;'
            'align-items:center;padding:7px 0;border-bottom:1px solid var(--color-border)">'
            '<div style="font-family:var(--font-figure);font-size:0.6rem;letter-spacing:0.1em;'
            'text-transform:uppercase;color:var(--color-muted)">'
            f'{html.escape(str(metric_spec_dict["label_str"]))}</div>'
            '<div style="font-family:var(--font-figure);font-size:0.86rem;text-align:right;'
            'font-variant-numeric:tabular-nums;color:var(--color-ink)">'
            f'{html.escape(str(metric_spec_dict["display_str"]))}</div>'
            '<div style="position:relative;height:9px;background:var(--color-neutral);'
            'border:1px solid var(--color-border)">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;'
            f'width:{value_position_pct_float:.2f}%;background:{bar_color_str}"></div>'
            f'{benchmark_mark_html_str}</div>'
            '</div>'
        )

    return (
        '<div class="metric-scale-stack">' + ''.join(row_html_list)
        + '<div style="font-family:var(--font-figure);font-size:0.58rem;'
          'letter-spacing:0.1em;text-transform:uppercase;color:var(--color-muted);'
          'margin-top:9px">Bar = measured · vertical rule = benchmark</div></div>'
    )


def build_title_block_html(field_pair_list: list[tuple[str, str]]) -> str:
    """Render the provenance title block.

    Modelled on the title block of an engineering drawing: a ruled grid in
    which every field that identifies *this exact artifact* is stated on its
    face — run identity, data vintage, cost assumptions, revision. Styled with
    the report's CSS variables so it inherits whichever theme is active.
    """
    if len(field_pair_list) == 0:
        raise ValueError('field_pair_list must contain at least one field.')

    cell_html_list = []
    for field_label_str, field_value_str in field_pair_list:
        cell_html_list.append(
            '<div style="padding:7px 11px;border-left:1px solid var(--color-border);'
            'border-top:1px solid var(--color-border);min-width:0">'
            '<div style="font-size:0.56rem;letter-spacing:0.13em;text-transform:uppercase;'
            f'color:var(--color-muted);white-space:nowrap">{html.escape(field_label_str)}</div>'
            '<div style="font-size:0.72rem;color:var(--color-ink);margin-top:3px;'
            f'overflow-wrap:anywhere">{html.escape(field_value_str)}</div>'
            '</div>'
        )

    return (
        '<div class="title-block" style="font-family:var(--font-figure,inherit);'
        'border:1px solid var(--color-ink);border-left:none;border-top:none;'
        'display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));'
        'margin-top:44px">'
        + ''.join(cell_html_list)
        + '</div>'
    )
