from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import colors as mcolors


SEABORN_DEEP_COLOR_LIST: list[str] = [
    '#4c72b0',
    '#dd8452',
    '#55a868',
    '#c44e52',
    '#8172b3',
    '#937860',
    '#da8bc3',
    '#8c8c8c',
    '#ccb974',
    '#64b5cd',
]


SIGNATURE_PALETTE_DICT: dict[str, object] = {
    'ink': '#262626',
    'page': '#ffffff',
    'panel': '#eaeaf2',
    'neutral': '#f7f7fb',
    'grid': '#ffffff',
    'border': '#d8dce6',
    'axes_border': '#ffffff',
    'muted': '#4c4c4c',
    'strategy': '#55a868',
    'strategy_dark': '#387c44',
    'benchmark': '#c44e52',
    'benchmark_dark': '#8f2d39',
    'vertical_line': '#8c8c8c',
    'zero_line': '#7a7a7a',
    'bar_edge': '#ffffff',
    'legend_face': '#ffffff',
    'legend_edge': '#c8c8d3',
    'label_face': '#ffffff',
    'overlay_cycle': list(SEABORN_DEEP_COLOR_LIST),
    'mean_line': '#55a868',
    'shadow_rgba': 'rgba(0, 0, 0, 0.0)',
}

SIGNATURE_ASSET_COLOR_DICT: dict[str, str] = {
    'TLT': SEABORN_DEEP_COLOR_LIST[0],
    'GLD': SEABORN_DEEP_COLOR_LIST[1],
    'DBC': SEABORN_DEEP_COLOR_LIST[2],
    'UUP': SEABORN_DEEP_COLOR_LIST[3],
    'SPY': SEABORN_DEEP_COLOR_LIST[4],
    'UPRO': SEABORN_DEEP_COLOR_LIST[5],
    'BTAL': SEABORN_DEEP_COLOR_LIST[6],
    'DEFAULT': SEABORN_DEEP_COLOR_LIST[7],
}


def get_signature_palette_dict() -> dict[str, object]:
    signature_palette_dict = dict(SIGNATURE_PALETTE_DICT)
    signature_palette_dict['overlay_cycle'] = list(SIGNATURE_PALETTE_DICT['overlay_cycle'])
    return signature_palette_dict


def build_signature_rcparams(to_web_bool: bool) -> dict[str, object]:
    base_style_dict = dict(plt.style.library.get('seaborn-v0_8-darkgrid', {}))
    override_style_dict = {
        'axes.prop_cycle': cycler(color=SEABORN_DEEP_COLOR_LIST),
        'figure.facecolor': SIGNATURE_PALETTE_DICT['page'],
        'axes.facecolor': SIGNATURE_PALETTE_DICT['panel'],
        'axes.edgecolor': SIGNATURE_PALETTE_DICT['axes_border'],
        'axes.labelcolor': SIGNATURE_PALETTE_DICT['ink'],
        'axes.titlecolor': SIGNATURE_PALETTE_DICT['ink'],
        'grid.color': SIGNATURE_PALETTE_DICT['grid'],
        'grid.alpha': 1.0,
        'xtick.color': SIGNATURE_PALETTE_DICT['ink'],
        'ytick.color': SIGNATURE_PALETTE_DICT['ink'],
        'text.color': SIGNATURE_PALETTE_DICT['ink'],
        'font.size': 9.5 if to_web_bool else 10.0,
        'legend.facecolor': SIGNATURE_PALETTE_DICT['legend_face'],
        'legend.edgecolor': SIGNATURE_PALETTE_DICT['legend_edge'],
        'savefig.facecolor': SIGNATURE_PALETTE_DICT['page'],
        'savefig.edgecolor': SIGNATURE_PALETTE_DICT['page'],
    }
    base_style_dict.update(override_style_dict)
    return base_style_dict


def build_plot_color_dict(colors=None) -> dict[str, object]:
    if colors is None:
        strategy_color_str = SEABORN_DEEP_COLOR_LIST[2]
        benchmark_color_str = SEABORN_DEEP_COLOR_LIST[3]
    else:
        strategy_color_str, benchmark_color_str = colors

    return {
        'strategy': strategy_color_str,
        'benchmark': benchmark_color_str,
        'additional_cycle': [
            SEABORN_DEEP_COLOR_LIST[0],
            SEABORN_DEEP_COLOR_LIST[1],
            SEABORN_DEEP_COLOR_LIST[4],
            SEABORN_DEEP_COLOR_LIST[5],
            SEABORN_DEEP_COLOR_LIST[6],
            SEABORN_DEEP_COLOR_LIST[7],
            SEABORN_DEEP_COLOR_LIST[8],
            SEABORN_DEEP_COLOR_LIST[9],
        ],
    }


def apply_signature_axis_style(axis_obj, vertical_line_iterable: Iterable[object] = ()) -> None:
    signature_palette_dict = SIGNATURE_PALETTE_DICT

    for side_name_str in ('top', 'right', 'left', 'bottom'):
        axis_obj.spines[side_name_str].set_visible(True)
        axis_obj.spines[side_name_str].set_color(signature_palette_dict['axes_border'])
        axis_obj.spines[side_name_str].set_linewidth(1.0)

    axis_obj.tick_params(axis='x', labelsize=8, colors=signature_palette_dict['ink'])
    axis_obj.tick_params(axis='y', labelsize=8, colors=signature_palette_dict['ink'])
    axis_obj.grid(
        which='major',
        linestyle='-',
        linewidth=1.0,
        color=signature_palette_dict['grid'],
        alpha=1.0,
    )
    axis_obj.set_axisbelow(True)
    axis_obj.yaxis.tick_right()
    axis_obj.yaxis.set_label_position('left')

    for vertical_line_obj in vertical_line_iterable:
        axis_obj.axvline(
            vertical_line_obj,
            color=signature_palette_dict['vertical_line'],
            linestyle='--',
            linewidth=0.9,
            alpha=0.9,
            zorder=1,
        )


def blend_hex_color_str(
        start_color_str: str,
        end_color_str: str,
        weight_float: float,
) -> str:
    bounded_weight_float = max(0.0, min(1.0, float(weight_float)))
    start_rgb_tuple = mcolors.to_rgb(start_color_str)
    end_rgb_tuple = mcolors.to_rgb(end_color_str)

    channel_value_list: list[int] = []
    for channel_idx_int in range(3):
        blended_channel_float = (
            (1.0 - bounded_weight_float) * start_rgb_tuple[channel_idx_int]
            + bounded_weight_float * end_rgb_tuple[channel_idx_int]
        )
        channel_value_list.append(int(round(blended_channel_float * 255.0)))

    return '#{0:02x}{1:02x}{2:02x}'.format(*channel_value_list)


def build_report_css() -> str:
    signature_palette_dict = SIGNATURE_PALETTE_DICT
    return f'''
:root {{
    --color-ink: {signature_palette_dict["ink"]};
    --color-page: {signature_palette_dict["page"]};
    --color-panel: {signature_palette_dict["panel"]};
    --color-neutral: {signature_palette_dict["neutral"]};
    --color-grid: {signature_palette_dict["grid"]};
    --color-border: {signature_palette_dict["border"]};
    --color-muted: {signature_palette_dict["muted"]};
    --color-strategy: {signature_palette_dict["strategy"]};
    --color-strategy-dark: {signature_palette_dict["strategy_dark"]};
    --color-benchmark: {signature_palette_dict["benchmark"]};
    --color-benchmark-dark: {signature_palette_dict["benchmark_dark"]};
    --color-shadow: {signature_palette_dict["shadow_rgba"]};
}}
body {{
    font-family: "Segoe UI", Arial, sans-serif;
    margin: 0;
    padding: 20px 40px;
    background: var(--color-page);
    color: var(--color-ink);
    line-height: 1.4;
}}
h1, h2, h3 {{
    font-family: "Segoe UI", Arial, sans-serif;
    color: var(--color-ink);
}}
h1 {{
    font-size: 1.6em;
    margin-bottom: 4px;
}}
h2 {{
    font-size: 1.1em;
    margin-top: 32px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--color-border);
    padding-bottom: 4px;
}}
h3 {{
    font-size: 0.98em;
    margin-top: 18px;
    margin-bottom: 8px;
}}
.meta {{
    color: var(--color-muted);
    font-size: 0.9em;
    margin-bottom: 20px;
}}
p {{
    color: var(--color-ink);
}}
table {{
    border-collapse: collapse;
    font-size: 0.85em;
    width: 100%;
    margin-bottom: 16px;
}}
th {{
    background: var(--color-panel);
    padding: 6px 10px;
    text-align: left;
    border: 1px solid var(--color-border);
    font-weight: 600;
}}
td {{
    padding: 5px 10px;
    border: 1px solid var(--color-border);
}}
tr:nth-child(even) td {{
    background: var(--color-neutral);
}}
td.metric {{
    font-weight: 500;
    background: var(--color-panel);
    white-space: nowrap;
}}
tr.summary-row-sharpe td {{
    background: #eef5ef;
}}
tr.summary-row-sharpe td.metric-sharpe {{
    background: #e2efe4;
    border-left: 4px solid var(--color-strategy);
    color: var(--color-strategy-dark);
    font-weight: 700;
}}
tr.summary-row-sharpe td:not(.metric-sharpe) {{
    font-weight: 600;
}}
td.pos {{
    color: var(--color-strategy-dark);
    font-weight: 600;
}}
td.neg {{
    color: var(--color-benchmark-dark);
    font-weight: 600;
}}
.heatmap td {{
    text-align: center;
    font-size: 0.8em;
    min-width: 48px;
    padding: 4px 6px;
}}
.chart-wrap {{
    margin: 16px 0;
}}
.chart-wrap img,
.chart-panel img {{
    max-width: 100%;
    width: 100%;
    border: none;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    margin: 16px 0;
}}
.stats-table {{
    width: auto;
    min-width: 420px;
}}
.scroll {{
    overflow-x: auto;
    border: none;
    border-radius: 0;
    background: transparent;
    padding: 0;
    margin-bottom: 16px;
}}
strong {{
    color: var(--color-ink);
}}
'''
