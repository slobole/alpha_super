"""Render the signature style gallery for every candidate theme variant.

This is a presentation-layer pilot only. It renders the *same* synthetic
fixture through the *same* production plotting and CSS code paths under each
variant in ``alpha.engine.theme``, so any visual difference between the output
pages is attributable to the theme alone.

No strategy, signal, execution, or cost logic is touched or exercised here.

Usage:
    uv run python scripts/dev/preview_theme.py
    uv run python scripts/dev/preview_theme.py --out results/_theme_preview
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import itertools
import sys
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha.engine.plot import compute_drawdown, plot
from alpha.engine.signature import (
    build_metric_delta_table_html,
    build_sparkline_img_html,
    build_title_block_html,
    render_composition_data_uri_str,
    render_relative_performance_data_uri_str,
    render_small_multiples_data_uri_str,
)
from preview_tables import (
    build_conditional_beta_html,
    build_monthly_returns_html,
    build_open_positions_html,
    build_performance_summary_html,
)
from alpha.engine.theme import (
    SIGNATURE_PALETTE_DICT,
    SIGNATURE_VARIANT_NAME_LIST,
    blend_hex_color_str,
    build_report_css,
    build_signature_rcparams,
    signature_variant_context,
)


_FIXTURE_SEED_INT = 20260723
_FIXTURE_START_DATE_STR = '1990-01-01'
_FIXTURE_END_DATE_STR = '2026-06-30'
_FIXTURE_SLEEVE_NAME_LIST = ['SPY', 'TLT', 'DBC', 'GLD']

_VARIANT_BLURB_DICT = {
    'current': 'Baseline — the legacy card grid and blue accent.',
    'desk': 'Desk · specimen sheet — mono figures, numbered plates, colour only for state.',
}

_PROSE_PARAGRAPH_STR = (
    'Every block on this page uses the same HTML classes the report builders already emit. '
    'Nothing about the markup changes between variants — the layout grammar comes entirely '
    'from the stylesheet, which is why the same fixture can read as a control panel or as a '
    'note without touching the report code.'
)


def build_fixture_dict() -> dict[str, object]:
    """Build a deterministic synthetic fixture.

    Synthetic on purpose: this page exists to compare colour, type and chrome,
    and must never be mistaken for a research result.
    """
    random_generator = np.random.default_rng(_FIXTURE_SEED_INT)
    bar_date_idx = pd.bdate_range(_FIXTURE_START_DATE_STR, _FIXTURE_END_DATE_STR)
    bar_count_int = len(bar_date_idx)

    # Drift is calibrated so the benchmark still compounds at a plausible rate
    # *after* variance drag and both crises, rather than limping in near zero.
    benchmark_return_vec = random_generator.normal(0.00047, 0.0100, bar_count_int)
    # Give the benchmark two visible crises so the drawdown panel has structure.
    for crisis_start_str, crisis_day_count_int, crisis_drift_float in (
        ('2000-09-01', 380, -0.0009),
        ('2008-06-01', 240, -0.0018),
    ):
        crisis_start_int = int(bar_date_idx.searchsorted(pd.Timestamp(crisis_start_str)))
        crisis_slice = slice(crisis_start_int, crisis_start_int + crisis_day_count_int)
        benchmark_return_vec[crisis_slice] += crisis_drift_float

    # The strategy is beta-linked to the benchmark plus its own idiosyncratic
    # return. Drawing the two independently produced a zero correlation and a
    # losing benchmark, which made every comparison device demo meaningless.
    #
    # Beta is deliberately *asymmetric* — lower when the benchmark falls than
    # when it rises. A constant-beta fixture reports zero asymmetry by
    # construction, which is the correct answer but demonstrates nothing about
    # the conditional-beta panel. This is also what a defensive book looks like.
    #
    # The idiosyncratic drift is negative on purpose. A *predictably* lower
    # down-beta is an enormous free lunch: left uncorrected this fixture
    # compounded at 40% a year, which would make every number on the page
    # absurd. The negative drift pays for the asymmetry, so the headline metrics
    # stay plausible while the conditional panel still has something to show.
    _FIXTURE_DOWN_BETA_FLOAT = 0.35
    _FIXTURE_UP_BETA_FLOAT = 0.52
    _FIXTURE_IDIOSYNCRATIC_DRIFT_FLOAT = -0.00030
    conditional_beta_vec = np.where(
        benchmark_return_vec < 0.0, _FIXTURE_DOWN_BETA_FLOAT, _FIXTURE_UP_BETA_FLOAT
    )
    strategy_return_vec = (
        conditional_beta_vec * benchmark_return_vec
        + random_generator.normal(_FIXTURE_IDIOSYNCRATIC_DRIFT_FLOAT, 0.0070, bar_count_int)
    )

    strategy_total_value_ser = pd.Series(
        10_000.0 * np.cumprod(1.0 + strategy_return_vec), index=bar_date_idx
    )
    benchmark_total_value_ser = pd.Series(
        10_000.0 * np.cumprod(1.0 + benchmark_return_vec), index=bar_date_idx
    )

    monthly_return_ser = strategy_total_value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    monthly_return_df = pd.DataFrame(
        {
            'year_int': monthly_return_ser.index.year,
            'month_int': monthly_return_ser.index.month,
            'return_float': monthly_return_ser.to_numpy(),
        }
    ).pivot(index='year_int', columns='month_int', values='return_float')

    sleeve_weight_df = pd.DataFrame(
        {
            sleeve_name_str: pd.Series(
                0.32 + 0.14 * np.sin(np.linspace(0, 9 + sleeve_idx_int * 3, bar_count_int)),
                index=bar_date_idx,
            )
            for sleeve_idx_int, sleeve_name_str in enumerate(_FIXTURE_SLEEVE_NAME_LIST)
        }
    )

    sleeve_trace_ser_dict = {
        sleeve_name_str: sleeve_weight_df[sleeve_name_str].tail(90)
        for sleeve_name_str in _FIXTURE_SLEEVE_NAME_LIST
    }

    # Per-calendar-year growth paths, rebased to 0 at each year start. This is
    # the small-multiples fixture: every year is shown, including the bad ones.
    annual_path_ser_dict: dict[str, pd.Series] = {}
    for year_int, year_total_value_ser in strategy_total_value_ser.groupby(
        strategy_total_value_ser.index.year
    ):
        if int(year_int) < 2011:
            continue
        annual_path_ser_dict[str(year_int)] = (
            year_total_value_ser / year_total_value_ser.iloc[0] - 1.0
        ).reset_index(drop=True)

    rotation_holding_df, rotation_price_df = build_rotation_fixture_df()

    return {
        'strategy_total_value_ser': strategy_total_value_ser,
        'benchmark_total_value_ser': benchmark_total_value_ser,
        'monthly_return_df': monthly_return_df.tail(12),
        'sleeve_weight_df': sleeve_weight_df,
        'sleeve_trace_ser_dict': sleeve_trace_ser_dict,
        'annual_path_ser_dict': annual_path_ser_dict,
        'rotation_holding_df': rotation_holding_df,
        'rotation_price_df': rotation_price_df,
    }


def build_rotation_fixture_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A ten-slot monthly rotation, shaped like the real NDX momentum family.

    Modelled on ``strategy_mo_atr_normalized_ndx_vxn_scaled``: ten equal-weight
    slots refreshed at month end, the whole basket scaled by a volatility-driven
    exposure factor, and a trend filter that flattens the book entirely in bad
    regimes. Every held name therefore carries an identical weight, which is
    exactly why a per-name weight chart tells you nothing about this strategy.

    Returns the holding-weight frame and a matching per-name close-price frame,
    so open positions can be marked to the last known close.
    """
    random_generator = np.random.default_rng(_FIXTURE_SEED_INT + 1)
    bar_date_idx = pd.bdate_range('2016-01-01', '2026-06-30')
    slot_capacity_int = 10
    name_list = [f'N{name_idx_int:03d}' for name_idx_int in range(180)]
    month_period_idx = bar_date_idx.to_period('M')

    # Synthetic per-name close prices: independent geometric random walks off a
    # spread of starting levels. Only used to mark open positions to market —
    # they feed no signal, so their statistical realism does not matter.
    #
    # *** CRITICAL*** Prices draw from their own generator. Sharing the holdings
    # generator would consume from that stream and silently reshuffle the whole
    # rotation, which had already moved the last-bar regime off.
    price_generator = np.random.default_rng(_FIXTURE_SEED_INT + 2)
    daily_price_return_matrix = price_generator.normal(
        0.0002, 0.018, (len(bar_date_idx), len(name_list))
    )
    start_price_vec = price_generator.uniform(15.0, 240.0, len(name_list))
    price_matrix = start_price_vec * np.cumprod(1.0 + daily_price_return_matrix, axis=0)
    rotation_price_df = pd.DataFrame(price_matrix, index=bar_date_idx, columns=name_list)

    # Synthetic implied-volatility path driving the exposure scaler, plus a
    # slow-moving trend filter that switches the book off in sustained stress.
    implied_vol_ser = pd.Series(
        22.0
        + 9.0 * np.sin(np.linspace(0.0, 14.0, len(bar_date_idx)))
        + random_generator.normal(0.0, 2.4, len(bar_date_idx)).cumsum() * 0.05,
        index=bar_date_idx,
    ).clip(11.0, 55.0)
    regime_on_ser = implied_vol_ser.rolling(60, min_periods=1).mean().lt(31.0)

    holding_matrix = np.zeros((len(bar_date_idx), len(name_list)), dtype=float)
    selected_idx_list: list[int] = []
    current_month_period = None

    for bar_idx_int, bar_date_ts in enumerate(bar_date_idx):
        if month_period_idx[bar_idx_int] != current_month_period:
            current_month_period = month_period_idx[bar_idx_int]
            # Month-end reselection: keep part of the roster, replace the rest.
            retained_count_int = int(random_generator.integers(3, 8))
            retained_idx_list = list(random_generator.permutation(selected_idx_list))[:retained_count_int]
            selected_idx_list = list(retained_idx_list)
            while len(selected_idx_list) < slot_capacity_int:
                candidate_idx_int = int(random_generator.integers(0, len(name_list)))
                if candidate_idx_int not in selected_idx_list:
                    selected_idx_list.append(candidate_idx_int)

        if not bool(regime_on_ser.iloc[bar_idx_int]):
            continue

        exposure_scale_float = float(
            np.clip(22.0 / implied_vol_ser.iloc[bar_idx_int], 0.25, 1.0)
        )
        for name_idx_int in selected_idx_list:
            holding_matrix[bar_idx_int, name_idx_int] = exposure_scale_float / slot_capacity_int

    return pd.DataFrame(holding_matrix, index=bar_date_idx, columns=name_list), rotation_price_df


def encode_current_figure_data_uri_str(dpi_int: int = 140) -> str:
    """Serialise the active matplotlib figure to an inline data URI."""
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format='png', dpi=dpi_int, bbox_inches='tight')
    plt.close('all')
    return 'data:image/png;base64,' + base64.b64encode(png_buffer.getvalue()).decode('ascii')


def render_equity_panel_data_uri_str(fixture_dict: dict[str, object]) -> str:
    """Render the flagship three-panel chart through the production plot()."""
    png_buffer = io.BytesIO()
    plot(
        fixture_dict['strategy_total_value_ser'],
        benchmark_total_value=fixture_dict['benchmark_total_value_ser'],
        strategy_label='Signature Demo',
        benchmark_label='Benchmark',
        save_to=png_buffer,
        to_web=True,
        dpi=140,
    )
    plt.close('all')
    return 'data:image/png;base64,' + base64.b64encode(png_buffer.getvalue()).decode('ascii')


def build_composition_html(
        holding_weight_df: pd.DataFrame,
        slot_capacity_int: int | None = None,
) -> str:
    """Render whichever composition view the book's own shape calls for."""
    composition_uri_str, resolved_mode_str = render_composition_data_uri_str(
        holding_weight_df, slot_capacity_int=slot_capacity_int
    )
    distinct_name_count_int = int(
        holding_weight_df.fillna(0.0).abs().gt(0.0).any(axis=0).sum()
    )
    caption_str = {
        'sleeve': (
            f'Auto-detected: SLEEVE ({distinct_name_count_int} distinct names ever held). '
            'Few persistent instruments, so weights by name are the story.'
        ),
        'rotation': (
            f'Auto-detected: ROTATION ({distinct_name_count_int} distinct names ever held). '
            'Equal-weight slots make a per-name chart meaningless, so this shows deployed '
            'capital, slot occupancy and holding periods instead.'
        ),
    }[resolved_mode_str]
    return (
        f'<div class="chart-wrap"><img src="{composition_uri_str}" alt="Composition"></div>'
        f'<p class="metric-context">{html.escape(caption_str)}</p>'
    )


def build_metric_spec_list(fixture_dict: dict[str, object]) -> list[dict[str, object]]:
    """Shared metric inputs, rendered three different ways by the layouts."""
    strategy_total_value_ser = fixture_dict['strategy_total_value_ser']
    benchmark_total_value_ser = fixture_dict['benchmark_total_value_ser']

    year_count_float = len(strategy_total_value_ser) / 252.0
    strategy_growth_float = float(strategy_total_value_ser.iloc[-1] / strategy_total_value_ser.iloc[0])
    benchmark_growth_float = float(
        benchmark_total_value_ser.iloc[-1] / benchmark_total_value_ser.iloc[0]
    )
    strategy_cagr_float = strategy_growth_float ** (1.0 / year_count_float) - 1.0
    benchmark_cagr_float = benchmark_growth_float ** (1.0 / year_count_float) - 1.0

    strategy_return_ser = strategy_total_value_ser.pct_change(fill_method=None).dropna()
    benchmark_return_ser = benchmark_total_value_ser.pct_change(fill_method=None).dropna()
    strategy_vol_float = float(strategy_return_ser.std() * np.sqrt(252.0))
    benchmark_vol_float = float(benchmark_return_ser.std() * np.sqrt(252.0))
    strategy_max_drawdown_float = float(compute_drawdown(strategy_total_value_ser).min())
    benchmark_max_drawdown_float = float(compute_drawdown(benchmark_total_value_ser).min())

    strategy_sharpe_float = strategy_cagr_float / strategy_vol_float
    benchmark_sharpe_float = benchmark_cagr_float / benchmark_vol_float
    correlation_float = float(strategy_return_ser.corr(benchmark_return_ser))

    return [
        {
            'label_str': 'CAGR (net)', 'value_float': strategy_cagr_float,
            'display_str': f'{strategy_cagr_float * 100:.1f}%',
            'benchmark_float': benchmark_cagr_float,
            'benchmark_display_str': f'{benchmark_cagr_float * 100:.1f}%',
            'delta_display_str': f'{(strategy_cagr_float - benchmark_cagr_float) * 100:+.1f}pp',
            'domain_min_float': 0.0, 'domain_max_float': 0.15,
            'higher_is_better_bool': True,
        },
        {
            'label_str': 'Volatility', 'value_float': strategy_vol_float,
            'display_str': f'{strategy_vol_float * 100:.1f}%',
            'benchmark_float': benchmark_vol_float,
            'benchmark_display_str': f'{benchmark_vol_float * 100:.1f}%',
            'delta_display_str': f'{(strategy_vol_float - benchmark_vol_float) * 100:+.1f}pp',
            'domain_min_float': 0.0, 'domain_max_float': 0.30,
            'higher_is_better_bool': False,
        },
        {
            'label_str': 'Sharpe ratio', 'value_float': strategy_sharpe_float,
            'display_str': f'{strategy_sharpe_float:.2f}',
            'benchmark_float': benchmark_sharpe_float,
            'benchmark_display_str': f'{benchmark_sharpe_float:.2f}',
            'delta_display_str': f'{strategy_sharpe_float - benchmark_sharpe_float:+.2f}',
            'domain_min_float': 0.0, 'domain_max_float': 1.5,
            'higher_is_better_bool': True,
        },
        {
            'label_str': 'Max drawdown', 'value_float': abs(strategy_max_drawdown_float),
            'display_str': f'{strategy_max_drawdown_float * 100:.1f}%',
            'benchmark_float': abs(benchmark_max_drawdown_float),
            'benchmark_display_str': f'{benchmark_max_drawdown_float * 100:.1f}%',
            'delta_display_str': (
                f'{(abs(strategy_max_drawdown_float) - abs(benchmark_max_drawdown_float)) * 100:+.1f}pp'
            ),
            'domain_min_float': 0.0, 'domain_max_float': 0.70,
            'higher_is_better_bool': False,
            'is_adverse_bool': True,
        },
        {
            'label_str': 'Correlation', 'value_float': correlation_float,
            'display_str': f'{correlation_float:.2f}',
            'benchmark_float': 1.0,
            'benchmark_display_str': '1.00',
            'delta_display_str': f'{correlation_float - 1.0:+.2f}',
            'domain_min_float': 0.0, 'domain_max_float': 1.0,
            'higher_is_better_bool': False,
        },
    ]


def build_headline_metrics_html(fixture_dict: dict[str, object]) -> str:
    """Headline metrics as a pure-typographic delta table."""
    return build_metric_delta_table_html(build_metric_spec_list(fixture_dict))


def build_kpi_grid_html(fixture_dict: dict[str, object]) -> str:
    strategy_total_value_ser = fixture_dict['strategy_total_value_ser']
    benchmark_total_value_ser = fixture_dict['benchmark_total_value_ser']

    year_count_float = len(strategy_total_value_ser) / 252.0
    strategy_growth_float = float(strategy_total_value_ser.iloc[-1] / strategy_total_value_ser.iloc[0])
    strategy_cagr_float = strategy_growth_float ** (1.0 / year_count_float) - 1.0
    strategy_return_ser = strategy_total_value_ser.pct_change(fill_method=None).dropna()
    strategy_vol_float = float(strategy_return_ser.std() * np.sqrt(252.0))
    strategy_sharpe_float = strategy_cagr_float / strategy_vol_float
    strategy_max_drawdown_float = float(compute_drawdown(strategy_total_value_ser).min())
    benchmark_growth_float = float(
        benchmark_total_value_ser.iloc[-1] / benchmark_total_value_ser.iloc[0]
    )

    kpi_spec_list = [
        ('CAGR (net)', f'{strategy_cagr_float * 100:.1f}%', 'pos', 'Synthetic fixture'),
        ('Volatility', f'{strategy_vol_float * 100:.1f}%', '', 'Annualised, 252d'),
        ('Sharpe ratio', f'{strategy_sharpe_float:.2f}', 'pos', 'Risk-free = 0'),
        ('Max drawdown', f'{strategy_max_drawdown_float * 100:.1f}%', 'neg', 'Peak to trough'),
        ('Growth of $10,000', f'${strategy_growth_float * 10_000:,.0f}', 'pos',
         f'Benchmark ${benchmark_growth_float * 10_000:,.0f}'),
    ]

    kpi_card_html_list = [
        f'<div class="kpi-card"><div class="kpi-label">{html.escape(label_str)}</div>'
        f'<div class="kpi-value {tone_str}">{html.escape(value_str)}</div>'
        f'<div class="kpi-note">{html.escape(note_str)}</div></div>'
        for label_str, value_str, tone_str, note_str in kpi_spec_list
    ]
    return '<div class="kpi-grid">' + ''.join(kpi_card_html_list) + '</div>'


def build_monthly_heatmap_html(fixture_dict: dict[str, object]) -> str:
    monthly_return_df = fixture_dict['monthly_return_df']
    month_label_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    max_abs_return_float = float(np.nanmax(np.abs(monthly_return_df.to_numpy()))) or 1.0

    header_html = ''.join(f'<th>{month_str}</th>' for month_str in month_label_list)
    row_html_list = []
    for year_int, month_return_ser in monthly_return_df.iterrows():
        cell_html_list = []
        for month_int in range(1, 13):
            return_float = month_return_ser.get(month_int, np.nan)
            if pd.isna(return_float):
                cell_html_list.append('<td></td>')
                continue
            tone_color_str = (
                SIGNATURE_PALETTE_DICT['profit'] if return_float >= 0
                else SIGNATURE_PALETTE_DICT['loss']
            )
            blend_weight_float = min(1.0, abs(return_float) / max_abs_return_float) * 0.55
            cell_background_str = blend_hex_color_str(
                SIGNATURE_PALETTE_DICT['panel'], tone_color_str, blend_weight_float
            )
            cell_html_list.append(
                f'<td style="background:{cell_background_str}">{return_float * 100:.1f}</td>'
            )
        row_html_list.append(
            f'<tr><td class="metric">{year_int}</td>' + ''.join(cell_html_list) + '</tr>'
        )

    return (
        '<div class="scroll"><table class="heatmap">'
        f'<thead><tr><th>Year</th>{header_html}</tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def build_stats_table_html(fixture_dict: dict[str, object]) -> str:
    strategy_total_value_ser = fixture_dict['strategy_total_value_ser']
    benchmark_total_value_ser = fixture_dict['benchmark_total_value_ser']
    strategy_return_ser = strategy_total_value_ser.pct_change(fill_method=None).dropna()
    benchmark_return_ser = benchmark_total_value_ser.pct_change(fill_method=None).dropna()

    row_spec_list = [
        ('Volatility',
         f'{strategy_return_ser.std() * np.sqrt(252) * 100:.1f}%',
         f'{benchmark_return_ser.std() * np.sqrt(252) * 100:.1f}%'),
        ('Max drawdown',
         f'{compute_drawdown(strategy_total_value_ser).min() * 100:.1f}%',
         f'{compute_drawdown(benchmark_total_value_ser).min() * 100:.1f}%'),
        ('% positive months',
         f'{(strategy_return_ser > 0).mean() * 100:.1f}%',
         f'{(benchmark_return_ser > 0).mean() * 100:.1f}%'),
        ('Correlation', '0.51', '1.00'),
        ('Beta', '0.45', '1.00'),
    ]

    row_html_list = [
        f'<tr><td class="metric">{html.escape(label_str)}</td>'
        f'<td>{html.escape(strategy_value_str)}</td>'
        f'<td>{html.escape(benchmark_value_str)}</td></tr>'
        for label_str, strategy_value_str, benchmark_value_str in row_spec_list
    ]
    return (
        '<div class="scroll"><table class="stats-table">'
        '<thead><tr><th>Metric</th><th>Signature Demo</th><th>Benchmark</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def build_sparkline_table_html(fixture_dict: dict[str, object]) -> str:
    """Positions table with a 90-bar trace inside each row."""
    sleeve_trace_ser_dict = fixture_dict['sleeve_trace_ser_dict']
    overlay_color_list = list(SIGNATURE_PALETTE_DICT['overlay_cycle'])

    row_html_list = []
    for sleeve_idx_int, (sleeve_name_str, trace_ser) in enumerate(sleeve_trace_ser_dict.items()):
        trace_color_str = overlay_color_list[sleeve_idx_int % len(overlay_color_list)]
        change_float = float(trace_ser.iloc[-1] - trace_ser.iloc[0])
        tone_class_str = 'pos' if change_float >= 0 else 'neg'
        row_html_list.append(
            f'<tr><td class="metric">{html.escape(sleeve_name_str)}</td>'
            f'<td>{trace_ser.iloc[-1] * 100:.1f}%</td>'
            f'<td class="{tone_class_str}">{change_float * 100:+.1f}pp</td></tr>'
        )

    return (
        '<div class="scroll"><table class="stats-table">'
        '<thead><tr><th>Sleeve</th><th>Weight</th><th>90d change</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def build_sparkline_prose_html(fixture_dict: dict[str, object]) -> str:
    """Sparklines set inline in a sentence — Tufte's datawords."""
    sleeve_trace_ser_dict = fixture_dict['sleeve_trace_ser_dict']
    overlay_color_list = list(SIGNATURE_PALETTE_DICT['overlay_cycle'])
    sleeve_name_list = list(sleeve_trace_ser_dict)

    def inline_spark_html(sleeve_idx_int: int) -> str:
        return build_sparkline_img_html(
            sleeve_trace_ser_dict[sleeve_name_list[sleeve_idx_int]],
            color_str=overlay_color_list[sleeve_idx_int % len(overlay_color_list)],
            width_px_int=58,
            height_px_int=13,
        )

    # Extra leading so the inline traces do not push successive lines apart.
    return (
        '<p style="line-height:2.05">Over the trailing quarter the Treasury sleeve '
        + inline_spark_html(1)
        + ' carried the allocation while the commodity sleeve '
        + inline_spark_html(2)
        + ' was cut back; equity exposure '
        + inline_spark_html(0)
        + ' drifted sideways throughout.</p>'
    )


def build_relative_performance_html(fixture_dict: dict[str, object]) -> str:
    """The parameter-free edge-decay view: strategy ÷ benchmark, log scale."""
    relative_uri_str = render_relative_performance_data_uri_str(
        fixture_dict['strategy_total_value_ser'],
        fixture_dict['benchmark_total_value_ser'],
    )
    return (
        f'<div class="chart-wrap"><img src="{relative_uri_str}" '
        'alt="Relative performance — strategy divided by benchmark"></div>'
        '<p class="metric-context">Rising = beating the benchmark, flat = matching it, '
        'falling = lagging. No window parameter — edge decay shows as flattening. '
        'Both series rebased at the first common bar.</p>'
    )


def build_small_multiples_html(fixture_dict: dict[str, object]) -> str:
    small_multiples_uri_str = render_small_multiples_data_uri_str(
        fixture_dict['annual_path_ser_dict'],
        column_count_int=4,
        share_ylim_bool=True,
        value_formatter_fn=lambda value_float: f'{value_float * 100:.0f}%',
    )
    return (
        '<div class="chart-wrap">'
        f'<img src="{small_multiples_uri_str}" alt="Growth path by calendar year">'
        '</div>'
        '<p class="metric-context">Every calendar year on one shared vertical scale — '
        'the losing years are shown at the same size as the winning ones.</p>'
    )


def build_preview_title_block_html(variant_name_str: str) -> str:
    return build_title_block_html([
        ('Artifact', 'Signature style gallery'),
        ('Variant', variant_name_str),
        ('Fixture', f'synthetic · seed {_FIXTURE_SEED_INT}'),
        ('Window', f'{_FIXTURE_START_DATE_STR} → {_FIXTURE_END_DATE_STR}'),
        ('Costs', 'not modelled — demo fixture'),
        ('Status', 'pilot · not a research result'),
    ])


def build_swatch_row_html() -> str:
    swatch_key_list = [
        'ink', 'strategy', 'strategy_dark', 'benchmark', 'profit', 'loss', 'muted', 'grid',
    ]
    swatch_html_list = []
    for swatch_key_str in swatch_key_list:
        swatch_color_str = str(SIGNATURE_PALETTE_DICT[swatch_key_str])
        swatch_html_list.append(
            '<div style="flex:1;min-width:96px">'
            f'<div style="height:44px;border-radius:3px;border:1px solid var(--color-border);'
            f'background:{swatch_color_str}"></div>'
            f'<div style="margin-top:5px;font-size:0.72rem;color:var(--color-muted)">'
            f'{swatch_key_str}<br>{swatch_color_str}</div></div>'
        )
    overlay_html_list = [
        f'<div style="flex:1;height:26px;background:{overlay_color_str};'
        'border:1px solid var(--color-border)"></div>'
        for overlay_color_str in SIGNATURE_PALETTE_DICT['overlay_cycle']
    ]
    return (
        '<div style="display:flex;gap:8px;flex-wrap:wrap">' + ''.join(swatch_html_list) + '</div>'
        '<div style="margin-top:14px;font-size:0.72rem;color:var(--color-muted)">overlay_cycle</div>'
        '<div style="display:flex;gap:3px;margin-top:5px">' + ''.join(overlay_html_list) + '</div>'
    )


def build_document_body_html(
        variant_name_str: str,
        fixture_dict: dict[str, object],
        equity_panel_uri_str: str,
) -> str:
    """Single reading column: lede claim, then evidence, then provenance."""
    return f'''
  <div class="report-header">
    <p class="report-eyebrow">Signature variant — {html.escape(variant_name_str)}</p>
    <h1>Style gallery</h1>
    <p class="meta">{html.escape(_VARIANT_BLURB_DICT.get(variant_name_str, ""))}</p>
    <p class="meta"><strong>Synthetic demo data — not a research result.</strong>
       Deterministic seed {_FIXTURE_SEED_INT}; identical across all variants.</p>
  </div>

  <h2>Headline metrics</h2>
  {build_headline_metrics_html(fixture_dict)}

  <h2>Flagship chart</h2>
  <div class="card card-primary">
    <div class="chart-wrap"><img src="{equity_panel_uri_str}" alt="Equity, drawdown and annual returns"></div>
  </div>

  <h2>Composition — persistent sleeves</h2>
  <div class="card">
    {build_composition_html(fixture_dict['sleeve_weight_df'])}
  </div>

  <h2>Composition — rotating slots</h2>
  <div class="card">{build_composition_html(fixture_dict['rotation_holding_df'], slot_capacity_int=10)}</div>

  <h2>Small multiples</h2>
  <div class="card">{build_small_multiples_html(fixture_dict)}</div>

  <h2>Sparklines</h2>
  <div class="card">
    {build_sparkline_prose_html(fixture_dict)}
    {build_sparkline_table_html(fixture_dict)}
  </div>

  <h2>Performance summary</h2>
  <div class="card">{build_performance_summary_html(fixture_dict)}</div>

  <h2>Monthly returns</h2>
  <div class="card">{build_monthly_returns_html(fixture_dict)}</div>

  <h2>Conditional beta</h2>
  <div class="card">{build_conditional_beta_html(fixture_dict)}</div>

  <h2>Palette</h2>
  <div class="card">{build_swatch_row_html()}</div>

  {build_preview_title_block_html(variant_name_str)}
'''


def build_spec_body_html(
        variant_name_str: str,
        fixture_dict: dict[str, object],
        equity_panel_uri_str: str,
) -> str:
    """Datasheet: provenance masthead, numbered plates, calibrated metrics."""
    masthead_field_list = [
        ('Artifact', 'Signature style gallery'),
        ('Variant', variant_name_str),
        ('Fixture', f'synthetic · seed {_FIXTURE_SEED_INT}'),
        ('Window', f'{_FIXTURE_START_DATE_STR} → {_FIXTURE_END_DATE_STR}'),
        ('Costs', 'not modelled'),
        ('Status', 'pilot'),
    ]
    masthead_html_str = ''.join(
        f'<div class="spec-field"><div class="spec-field-label">{html.escape(label_str)}</div>'
        f'<div class="spec-field-value">{html.escape(value_str)}</div></div>'
        for label_str, value_str in masthead_field_list
    )

    # Plates number themselves in emission order, so inserting or removing one
    # never silently leaves the labels out of sequence.
    plate_number_counter = itertools.count(1)

    def plate_html(title_str: str, body_html_str: str, note_str: str = '') -> str:
        return (
            f'<div class="plate"><div class="plate-label"><span>Plate {next(plate_number_counter):02d} — '
            f'{html.escape(title_str)}</span><span>{html.escape(note_str)}</span></div>'
            f'<div class="plate-body">{body_html_str}</div></div>'
        )

    return f'''
  <div class="report-header">
    <p class="report-eyebrow">Specimen sheet</p>
    <h1>Signature style gallery</h1>
  </div>
  <div class="spec-masthead">{masthead_html_str}</div>

  <h2>Headline metrics</h2>
  {build_headline_metrics_html(fixture_dict)}

  <h2>Plates</h2>
  {plate_html('Growth of $1, drawdown, annual returns',
              f'<div class="chart-wrap"><img src="{equity_panel_uri_str}" alt="Equity panel"></div>',
              'log scale')}
  {plate_html('Capital weights over time',
              build_composition_html(fixture_dict['sleeve_weight_df']),
              'stack height = gross exposure')}
  {plate_html('Rotating book — exposure, occupancy, holding periods',
              build_composition_html(fixture_dict['rotation_holding_df'], slot_capacity_int=10), '10 slots · 120 names')}
  {plate_html('Relative performance',
              build_relative_performance_html(fixture_dict), 'strategy ÷ benchmark · log scale')}
  {plate_html('Sleeve positions',
              build_sparkline_table_html(fixture_dict), '90 trading days')}
  {plate_html('Open positions',
              build_open_positions_html(fixture_dict), 'rotation book · last bar')}
  {plate_html('Performance summary',
              build_performance_summary_html(fixture_dict), 'all sections')}
  {plate_html('Monthly returns',
              build_monthly_returns_html(fixture_dict), 'vs SPY')}
  {plate_html('Conditional beta',
              build_conditional_beta_html(fixture_dict), 'by benchmark direction')}
  {plate_html('Palette', build_swatch_row_html(), 'resolved tokens')}
'''


def build_dashboard_body_html(
        variant_name_str: str,
        fixture_dict: dict[str, object],
        equity_panel_uri_str: str,
) -> str:
    """The card-grid baseline, kept intact so the comparison stays honest.

    This is the only body that still emits the KPI tile row; the structural
    layouts replace it with a lede, a margin stack or calibrated scales.
    """
    return f'''
  <div class="report-header">
    <p class="report-eyebrow">Signature variant — {html.escape(variant_name_str)}</p>
    <h1>Style gallery</h1>
    <p class="meta">{html.escape(_VARIANT_BLURB_DICT.get(variant_name_str, ""))}</p>
    <p class="meta"><strong>Synthetic demo data — not a research result.</strong>
       Deterministic seed {_FIXTURE_SEED_INT}; identical across all variants.</p>
  </div>

  <p>{html.escape(_PROSE_PARAGRAPH_STR)}</p>

  <h2>Headline metrics</h2>
  {build_kpi_grid_html(fixture_dict)}

  <h2>Flagship chart</h2>
  <div class="card card-primary">
    <div class="chart-wrap"><img src="{equity_panel_uri_str}" alt="Equity, drawdown and annual returns"></div>
  </div>

  <h2>Composition — persistent sleeves</h2>
  <div class="card">
    {build_composition_html(fixture_dict['sleeve_weight_df'])}
  </div>

  <h2>Composition — rotating slots</h2>
  <div class="card">{build_composition_html(fixture_dict['rotation_holding_df'], slot_capacity_int=10)}</div>

  <h2>Small multiples</h2>
  <div class="card">{build_small_multiples_html(fixture_dict)}</div>

  <h2>Sparklines</h2>
  <div class="card">
    {build_sparkline_prose_html(fixture_dict)}
    {build_sparkline_table_html(fixture_dict)}
  </div>

  <h2>Performance summary</h2>
  <div class="card">{build_performance_summary_html(fixture_dict)}</div>

  <h2>Monthly returns</h2>
  <div class="card">{build_monthly_returns_html(fixture_dict)}</div>

  <h2>Conditional beta</h2>
  <div class="card">{build_conditional_beta_html(fixture_dict)}</div>

  <h2>Palette</h2>
  <div class="card">{build_swatch_row_html()}</div>

  {build_preview_title_block_html(variant_name_str)}
'''


_LAYOUT_BODY_BUILDER_DICT = {
    'dashboard': build_dashboard_body_html,
    'document': build_document_body_html,
    'spec': build_spec_body_html,
}


def build_variant_page_html(variant_name_str: str, fixture_dict: dict[str, object]) -> str:
    equity_panel_uri_str = render_equity_panel_data_uri_str(fixture_dict)

    layout_str = str(SIGNATURE_PALETTE_DICT['layout_str'])
    body_builder_fn = _LAYOUT_BODY_BUILDER_DICT.get(layout_str, build_document_body_html)
    body_html_str = body_builder_fn(
        variant_name_str, fixture_dict, equity_panel_uri_str
    )

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signature preview — {html.escape(variant_name_str)}</title>
<style>{build_report_css()}</style>
</head><body>
<div class="report-shell">{body_html_str}</div>
</body></html>
'''


def build_index_html(variant_name_list: list[str]) -> str:
    active_class_attr_str = ' class="active"'
    shown_class_attr_str = ' class="shown"'

    tab_html_list = [
        '<button type="button" data-variant="{0}"{1}>{0}</button>'.format(
            html.escape(variant_name_str),
            active_class_attr_str if variant_idx_int == 0 else '',
        )
        for variant_idx_int, variant_name_str in enumerate(variant_name_list)
    ]
    frame_html_list = [
        '<iframe data-variant="{0}" src="preview_{0}.html"{1}></iframe>'.format(
            html.escape(variant_name_str),
            shown_class_attr_str if variant_idx_int == 0 else '',
        )
        for variant_idx_int, variant_name_str in enumerate(variant_name_list)
    ]
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signature style pilot</title>
<style>
  body {{ margin:0; font-family:"Segoe UI",Arial,sans-serif; background:#eef1f5; color:#1f2733; }}
  header {{ padding:14px 18px; background:#fff; border-bottom:1px solid #d3dbe5;
            display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
  h1 {{ font-size:1rem; margin:0; font-weight:700; }}
  .tabs {{ display:flex; gap:6px; }}
  button {{ font:inherit; padding:6px 14px; border:1px solid #d3dbe5; border-radius:6px;
            background:#f7f9fb; color:#1f2733; cursor:pointer; }}
  button.active {{ background:#1f2733; border-color:#1f2733; color:#fff; }}
  label {{ font-size:0.85rem; color:#5b6776; display:flex; gap:6px; align-items:center; }}
  #panes {{ display:grid; grid-template-columns:1fr; gap:0; height:calc(100vh - 57px); }}
  #panes.split {{ grid-template-columns:repeat({len(variant_name_list)}, 1fr); gap:1px; background:#d3dbe5; }}
  iframe {{ width:100%; height:100%; border:0; background:#fff; display:none; }}
  iframe.shown {{ display:block; }}
</style>
</head><body>
<header>
  <h1>Signature style pilot</h1>
  <div class="tabs">{''.join(tab_html_list)}</div>
  <label><input type="checkbox" id="split-toggle"> side by side</label>
</header>
<div id="panes">
  {''.join(frame_html_list)}
</div>
<script>
  const panes = document.getElementById('panes');
  const frames = [...document.querySelectorAll('iframe')];
  const buttons = [...document.querySelectorAll('.tabs button')];
  const splitToggle = document.getElementById('split-toggle');

  function render() {{
    const split = splitToggle.checked;
    panes.classList.toggle('split', split);
    const active = buttons.find(b => b.classList.contains('active')).dataset.variant;
    frames.forEach(f => f.classList.toggle('shown', split || f.dataset.variant === active));
  }}

  buttons.forEach(button => button.addEventListener('click', () => {{
    buttons.forEach(b => b.classList.toggle('active', b === button));
    render();
  }}));
  splitToggle.addEventListener('change', render);
</script>
</body></html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/_theme_preview', help='Output directory')
    parser.add_argument(
        '--variants',
        nargs='*',
        default=['journal_spec'],
        help='Variant names to render',
    )
    parsed_args = parser.parse_args()

    output_dir_path = Path(parsed_args.out)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Clear pages from retired variants. Leaving them behind means the folder
    # accumulates stale designs that look current but are no longer generated.
    for stale_page_path_obj in output_dir_path.glob('preview_*.html'):
        stale_page_path_obj.unlink()

    fixture_dict = build_fixture_dict()
    variant_name_list = list(parsed_args.variants)

    for variant_name_str in variant_name_list:
        with signature_variant_context(variant_name_str):
            page_html_str = build_variant_page_html(variant_name_str, fixture_dict)
        page_path_obj = output_dir_path / f'preview_{variant_name_str}.html'
        page_path_obj.write_text(page_html_str, encoding='utf-8')
        print(f'  rendered {page_path_obj}')

    index_path_obj = output_dir_path / 'index.html'
    index_path_obj.write_text(build_index_html(variant_name_list), encoding='utf-8')
    print(f'\nOpen: {index_path_obj.resolve()}')


if __name__ == '__main__':
    main()
