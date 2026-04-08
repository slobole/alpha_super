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
    'ink': '#172b4d',
    'page': '#ffffff',
    'panel': '#ffffff',
    'neutral': '#f7f8fa',
    'grid': '#dfe1e6',
    'border': '#dfe1e6',
    'axes_border': '#7a869a',
    'muted': '#6b778c',
    'strategy': '#357de8',
    'strategy_dark': '#1d5fdb',
    'benchmark': '#f28c28',
    'benchmark_dark': '#c66a13',
    'profit': '#22a06b',
    'profit_dark': '#216e4e',
    'loss': '#c9372c',
    'loss_dark': '#ae2e24',
    'vertical_line': '#8590a2',
    'zero_line': '#626f86',
    'bar_edge': '#dfe1e6',
    'legend_face': '#ffffff',
    'legend_edge': '#dfe1e6',
    'label_face': '#ffffff',
    'overlay_cycle': list(SEABORN_DEEP_COLOR_LIST),
    'mean_line': '#357de8',
    'shadow_rgba': 'rgba(9, 30, 66, 0.04)',
}

SIGNATURE_ASSET_COLOR_DICT: dict[str, str] = {
    'TLT': '#4f6bed',
    'GLD': '#d9a441',
    'DBC': '#36b37e',
    'UUP': '#00a3bf',
    'SPY': SIGNATURE_PALETTE_DICT['benchmark'],
    'SSO': SIGNATURE_PALETTE_DICT['benchmark'],
    'QQQ': '#6554c0',
    'QLD': '#7a5af8',
    'TQQQ': '#8f63ff',
    'UPRO': '#9b8afb',
    'BTAL': '#c251c0',
    'SH': '#6b778c',
    'CASH': '#b3bac5',
    'DEFAULT': '#7a869a',
}

SIGNATURE_FONT_STACK_LIST: list[str] = [
    'Atlassian Sans',
    'Segoe UI',
    'Arial',
    'DejaVu Sans',
    'sans-serif',
]
SIGNATURE_FONT_STACK_STR: str = '"Atlassian Sans", "Segoe UI", Arial, "DejaVu Sans", sans-serif'
_ATLASSIAN_FONT_CDN_BASE_URL_STR: str = 'https://ds-cdn.prod-east.frontend.public.atl-paas.net'


def get_signature_palette_dict() -> dict[str, object]:
    signature_palette_dict = dict(SIGNATURE_PALETTE_DICT)
    signature_palette_dict['overlay_cycle'] = list(SIGNATURE_PALETTE_DICT['overlay_cycle'])
    return signature_palette_dict


def build_report_font_head_html() -> str:
    """Return the official Atlassian Sans font preload tags for report HTML."""
    return (
        f'<link rel="preconnect" href="{_ATLASSIAN_FONT_CDN_BASE_URL_STR}" crossorigin>\n'
        f'<link rel="preload" href="{_ATLASSIAN_FONT_CDN_BASE_URL_STR}/assets/fonts/atlassian-sans/v3/AtlassianSans-latin.woff2" '
        'as="font" type="font/woff2" crossorigin>\n'
        f'<link rel="preload stylesheet" href="{_ATLASSIAN_FONT_CDN_BASE_URL_STR}/assets/font-rules/v5/atlassian-fonts.css" '
        'as="style" crossorigin>'
    )


def build_signature_rcparams(to_web_bool: bool) -> dict[str, object]:
    base_style_dict = dict(plt.style.library.get('seaborn-v0_8-whitegrid', {}))
    override_style_dict = {
        'axes.prop_cycle': cycler(color=SEABORN_DEEP_COLOR_LIST),
        'figure.facecolor': SIGNATURE_PALETTE_DICT['page'],
        'axes.facecolor': SIGNATURE_PALETTE_DICT['panel'],
        'axes.edgecolor': SIGNATURE_PALETTE_DICT['axes_border'],
        'axes.linewidth': 0.85,
        'axes.labelcolor': SIGNATURE_PALETTE_DICT['ink'],
        'axes.titlecolor': SIGNATURE_PALETTE_DICT['ink'],
        'grid.color': SIGNATURE_PALETTE_DICT['grid'],
        'grid.alpha': 1.0,
        'grid.linewidth': 0.75,
        'xtick.color': SIGNATURE_PALETTE_DICT['ink'],
        'ytick.color': SIGNATURE_PALETTE_DICT['ink'],
        'text.color': SIGNATURE_PALETTE_DICT['ink'],
        'font.family': 'sans-serif',
        'font.sans-serif': list(SIGNATURE_FONT_STACK_LIST),
        'font.size': 9.5 if to_web_bool else 10.0,
        'axes.titlesize': 10.5 if to_web_bool else 11.0,
        'axes.labelsize': 9.0 if to_web_bool else 9.5,
        'legend.framealpha': 1.0,
        'legend.fontsize': 8.0 if to_web_bool else 8.5,
        'legend.facecolor': SIGNATURE_PALETTE_DICT['legend_face'],
        'legend.edgecolor': SIGNATURE_PALETTE_DICT['legend_edge'],
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'savefig.facecolor': SIGNATURE_PALETTE_DICT['page'],
        'savefig.edgecolor': SIGNATURE_PALETTE_DICT['page'],
    }
    base_style_dict.update(override_style_dict)
    return base_style_dict


def build_plot_color_dict(colors=None) -> dict[str, object]:
    if colors is None:
        strategy_color_str = SIGNATURE_PALETTE_DICT['strategy']
        benchmark_color_str = SIGNATURE_PALETTE_DICT['benchmark']
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

    axis_obj.spines['top'].set_visible(False)
    for side_name_str in ('right', 'left', 'bottom'):
        axis_obj.spines[side_name_str].set_visible(True)
        axis_obj.spines[side_name_str].set_color(signature_palette_dict['axes_border'])
        axis_obj.spines[side_name_str].set_linewidth(0.85)

    axis_obj.tick_params(axis='x', labelsize=8, colors=signature_palette_dict['ink'], pad=4)
    axis_obj.tick_params(axis='y', labelsize=8, colors=signature_palette_dict['ink'], pad=4)
    axis_obj.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.75,
        color=signature_palette_dict['grid'],
        alpha=1.0,
    )
    axis_obj.xaxis.grid(False, which='major')
    axis_obj.xaxis.grid(False, which='minor')
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
    --color-profit: {signature_palette_dict["profit"]};
    --color-profit-dark: {signature_palette_dict["profit_dark"]};
    --color-loss: {signature_palette_dict["loss"]};
    --color-loss-dark: {signature_palette_dict["loss_dark"]};
    --color-shadow: {signature_palette_dict["shadow_rgba"]};
}}
body {{
    font-family: {SIGNATURE_FONT_STACK_STR};
    margin: 0;
    padding: 18px 20px 32px;
    background: var(--color-page);
    color: var(--color-ink);
    line-height: 1.45;
}}
.report-shell {{
    max-width: 1480px;
    margin: 0 auto;
}}
.report-header {{
    margin-bottom: 14px;
}}
.report-eyebrow {{
    margin: 0 0 6px;
    color: var(--color-muted);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
h1, h2, h3 {{
    font-family: {SIGNATURE_FONT_STACK_STR};
    color: var(--color-ink);
}}
h1 {{
    font-size: 1.72rem;
    margin: 0 0 4px;
    font-weight: 700;
}}
h2 {{
    font-size: 1.02rem;
    margin: 0 0 12px;
    border-bottom: none;
    padding-bottom: 0;
}}
h3 {{
    font-size: 0.92rem;
    margin-top: 16px;
    margin-bottom: 8px;
}}
.meta {{
    color: var(--color-muted);
    font-size: 0.88rem;
    margin-bottom: 0;
}}
p {{
    color: var(--color-ink);
    margin-top: 0;
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 12px;
    margin: 0 0 14px;
}}
.kpi-card {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 14px 16px;
    box-shadow: none;
}}
.kpi-label {{
    color: var(--color-ink);
    font-size: 0.92rem;
    font-weight: 600;
}}
.kpi-value {{
    margin-top: 6px;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
}}
.kpi-value.pos {{
    color: var(--color-profit-dark);
}}
.kpi-value.neg {{
    color: var(--color-loss-dark);
}}
.kpi-note {{
    margin-top: 4px;
    color: var(--color-muted);
    font-size: 0.84rem;
}}
.card {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 16px 18px;
    box-shadow: none;
    margin-bottom: 12px;
}}
.card-primary {{
    padding-top: 18px;
}}
.card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
    align-items: start;
}}
.card-grid > .card {{
    margin-bottom: 0;
}}
.crisis-chart-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
    align-items: start;
}}
.crisis-chart-grid > .card {{
    margin-bottom: 0;
}}
.crisis-chart-grid .card {{
    padding: 12px 14px;
}}
.crisis-chart-grid .chart-wrap {{
    margin-top: 6px;
}}
.section-stack {{
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.section-stack > .card {{
    margin-bottom: 0;
}}
table {{
    border-collapse: collapse;
    font-size: 0.85em;
    width: 100%;
    margin-bottom: 0;
}}
th {{
    background: var(--color-neutral);
    padding: 8px 10px;
    text-align: left;
    border: 1px solid var(--color-border);
    font-weight: 600;
}}
td {{
    padding: 7px 10px;
    border: 1px solid var(--color-border);
    background: var(--color-panel);
}}
td.metric {{
    font-weight: 500;
    background: var(--color-panel);
    white-space: nowrap;
}}
td.pos {{
    color: var(--color-profit-dark);
    font-weight: 600;
}}
td.neg {{
    color: var(--color-loss-dark);
    font-weight: 600;
}}
.heatmap td {{
    text-align: center;
    font-size: 0.92em;
    min-width: 64px;
    padding: 7px 9px;
}}
.heatmap th {{
    text-align: center;
    font-size: 0.88em;
    padding: 7px 9px;
}}
.heatmap .divider-left {{
    border-left: 3px solid var(--color-muted);
}}
.heatmap {{
    table-layout: fixed;
}}
.card-monthly-returns .scroll {{
    overflow-x: visible;
}}
.chart-wrap {{
    margin: 0;
}}
.chart-wrap img,
.chart-panel img {{
    max-width: 100%;
    width: 100%;
    display: block;
    border: none;
    border-radius: 4px;
    background: var(--color-panel);
    box-shadow: none;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 12px;
    margin-bottom: 12px;
}}
.chart-panel {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 12px;
}}
.stats-table {{
    width: auto;
    min-width: 420px;
}}
.scroll {{
    overflow-x: auto;
    width: 100%;
    border: none;
    border-radius: 0;
    background: transparent;
    padding: 0;
    margin-bottom: 0;
}}
strong {{
    color: var(--color-ink);
}}
@media (max-width: 960px) {{
    body {{
        padding: 16px 14px 24px;
    }}
    .card {{
        padding: 16px;
    }}
    .card-grid {{
        grid-template-columns: 1fr;
    }}
    .crisis-chart-grid {{
        grid-template-columns: 1fr;
    }}
}}
'''
