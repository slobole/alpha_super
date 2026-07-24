"""Report table builders for the style gallery.

These mirror the table catalogue that ``alpha/engine/report.py`` already emits,
so the theme is exercised at true size rather than against a five-row stub.
Section titles and metric names are copied from
``report._PERFORMANCE_SUMMARY_SECTION_TUPLE``.

Values are synthetic placeholders from the gallery fixture. This module shows
*shape*, not results, and computes nothing that feeds a decision.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

from alpha.engine.plot import compute_drawdown
from alpha.engine.signature import build_sparkline_img_html, compute_conditional_beta_dict
from alpha.engine.theme import SIGNATURE_PALETTE_DICT, blend_hex_color_str


def _gross_exposure_ser(fixture_dict: dict[str, object]) -> pd.Series:
    # In production this comes from the strategy's own position ledger; the
    # demo reuses the rotation fixture's book, whose VXN scaler and regime
    # filter give the exposure path real structure to show.
    return fixture_dict['rotation_holding_df'].fillna(0.0).abs().sum(axis=1)

_TRADING_DAY_PER_YEAR_FLOAT = 252.0

# Mirrors report._PERFORMANCE_SUMMARY_SECTION_TUPLE.
PERFORMANCE_SUMMARY_SECTION_TUPLE = (
    (
        'Period & Capital',
        ('Start', 'End', 'Duration [days]', 'Start [$]', 'Final [$]'),
    ),
    (
        'Return & Risk-Adjusted Performance',
        (
            'Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]',
            'Volatility (Monthly) [%]', 'Sharpe Ratio', 'MAR Ratio', '% Positive Months',
        ),
    ),
    ('Benchmark Regression', ('Beta', 'Alpha (Ann.) [%]', 'Alpha HAC t-stat', 'R²')),
    ('Exposure', ('Exposure Time [%]',)),
    (
        'Drawdown & Recovery',
        (
            'Max. Drawdown [%]', 'Max. Drawdown Duration [days]', 'Time Under Water [%]',
            '# Drawdowns', '# Drawdowns / year',
        ),
    ),
    (
        'Trading Activity & Costs',
        (
            'Total Commissions [$]', 'Turnover (Ann.) [%]', 'Estimated Slippage [$]',
            'Total Trading Costs [$]', 'Cost Drag (Ann.) [%]',
        ),
    ),
    (
        'Trade Statistics',
        (
            '# Trades', 'Trades / Week', '% Positive Trades', 'Avg. Return / Trade [%]',
            'Payoff Ratio', 'Avg. Commission / Trade [$]',
        ),
    ),
    (
        'Extended Risk Diagnostics',
        (
            'AAR [%]', 'Downside L1 [%]', 'Avg. Loss Day [%]', 'Avg. Drawdown [%]',
            'Avg. Drawdown Duration [days]',
        ),
    ),
    # Proposed addition for the real report — not yet in report.py.
    (
        'Distribution & Tails',
        (
            'Skewness (Daily)', 'Skewness (Monthly)', 'Excess Kurtosis (Daily)',
            'Worst Day [%]', 'VaR 95% (Daily) [%]', 'CVaR 95% (Daily) [%]',
        ),
    ),
)


def _annualised_metric_tuple(total_value_ser: pd.Series) -> tuple[float, float, float]:
    return_ser = total_value_ser.pct_change(fill_method=None).dropna()
    year_count_float = len(total_value_ser) / _TRADING_DAY_PER_YEAR_FLOAT
    growth_float = float(total_value_ser.iloc[-1] / total_value_ser.iloc[0])
    annual_return_float = growth_float ** (1.0 / year_count_float) - 1.0
    volatility_float = float(return_ser.std() * np.sqrt(_TRADING_DAY_PER_YEAR_FLOAT))
    return annual_return_float, volatility_float, growth_float - 1.0


_TOTAL_COMMISSION_FLOAT = 4812.0


def _rotation_trade_return_list(
        holding_weight_df: pd.DataFrame,
        price_df: pd.DataFrame,
) -> list[float]:
    """Per-trade returns from the rotation book: each holding spell is a trade.

    A name re-entered later is a separate trade, marked entry close to exit
    close. A spell still open on the last bar is marked to the last known close,
    so it counts as an (unrealised) trade rather than being dropped.
    """
    is_held_arr = holding_weight_df.fillna(0.0).abs().gt(0.0).to_numpy()
    trade_return_list: list[float] = []

    for column_idx_int, name_str in enumerate(holding_weight_df.columns):
        held_flag_vec = is_held_arr[:, column_idx_int].astype(int)
        padded_flag_vec = np.concatenate(([0], held_flag_vec, [0]))
        transition_vec = np.diff(padded_flag_vec)
        entry_idx_vec = np.flatnonzero(transition_vec == 1)
        exit_idx_vec = np.flatnonzero(transition_vec == -1)  # first bar *after* the spell
        price_vec = price_df[name_str].to_numpy()
        for entry_idx_int, exit_idx_int in zip(entry_idx_vec, exit_idx_vec):
            entry_price_float = float(price_vec[entry_idx_int])
            exit_price_float = float(price_vec[exit_idx_int - 1])
            if entry_price_float > 0.0:
                trade_return_list.append(exit_price_float / entry_price_float - 1.0)
    return trade_return_list


def build_summary_value_dict(fixture_dict: dict[str, object]) -> dict[str, str]:
    strategy_total_value_ser = fixture_dict['strategy_total_value_ser']
    benchmark_total_value_ser = fixture_dict['benchmark_total_value_ser']
    strategy_return_ser = strategy_total_value_ser.pct_change(fill_method=None).dropna()
    benchmark_return_ser = benchmark_total_value_ser.pct_change(fill_method=None).dropna()

    drawdown_ser = compute_drawdown(strategy_total_value_ser)
    year_count_float = len(strategy_total_value_ser) / _TRADING_DAY_PER_YEAR_FLOAT
    annual_return_float, volatility_float, total_return_float = _annualised_metric_tuple(
        strategy_total_value_ser
    )
    beta_float = float(strategy_return_ser.cov(benchmark_return_ser) / benchmark_return_ser.var())
    correlation_float = float(strategy_return_ser.corr(benchmark_return_ser))
    loss_day_ser = strategy_return_ser[strategy_return_ser < 0.0]
    # Std of monthly returns, reported as-is (not annualised) — the dispersion
    # figure an allocator reads directly off a monthly track record.
    monthly_return_ser = (
        strategy_total_value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    )
    monthly_volatility_float = float(monthly_return_ser.std())
    positive_month_fraction_float = float((monthly_return_ser > 0.0).mean())

    # Trade-level statistics from the rotation book. Each holding spell is a
    # trade; per-trade returns are marked entry close to exit close. These come
    # from the position ledger, not the equity series, so on the demo fixture
    # they do not tie to the curve above — on real data both share one ledger.
    rotation_holding_df = fixture_dict['rotation_holding_df']
    trade_return_vec = np.asarray(
        _rotation_trade_return_list(rotation_holding_df, fixture_dict['rotation_price_df'])
    )
    trade_count_int = int(len(trade_return_vec))
    # *** CRITICAL*** Weeks come from the *rotation book's* own span, since the
    # trades do. Dividing rotation trades by the equity series' longer window
    # would understate trades/week by the ratio of the two histories.
    week_count_float = (
        (rotation_holding_df.index[-1] - rotation_holding_df.index[0]).days / 7.0
    )
    win_trade_vec = trade_return_vec[trade_return_vec > 0.0]
    loss_trade_vec = trade_return_vec[trade_return_vec < 0.0]
    avg_win_float = float(win_trade_vec.mean()) if len(win_trade_vec) else 0.0
    avg_loss_float = float(loss_trade_vec.mean()) if len(loss_trade_vec) else 0.0

    return {
        'Start': str(strategy_total_value_ser.index[0].date()),
        'End': str(strategy_total_value_ser.index[-1].date()),
        'Duration [days]': (
            f'{(strategy_total_value_ser.index[-1] - strategy_total_value_ser.index[0]).days:,}'
        ),
        'Start [$]': f'{strategy_total_value_ser.iloc[0]:,.0f}',
        'Final [$]': f'{strategy_total_value_ser.iloc[-1]:,.0f}',
        'Return [%]': f'{total_return_float * 100:,.1f}',
        'Return (Ann.) [%]': f'{annual_return_float * 100:.2f}',
        'Volatility (Ann.) [%]': f'{volatility_float * 100:.2f}',
        'Volatility (Monthly) [%]': f'{monthly_volatility_float * 100:.2f}',
        'Sharpe Ratio': f'{annual_return_float / volatility_float:.2f}',
        'MAR Ratio': f'{annual_return_float / abs(float(drawdown_ser.min())):.2f}',
        '% Positive Months': f'{positive_month_fraction_float * 100:.1f}',
        'Beta': f'{beta_float:.2f}',
        'Alpha (Ann.) [%]': '2.41',
        'Alpha HAC t-stat': '1.87',
        'R²': f'{correlation_float ** 2:.2f}',
        'Exposure Time [%]': f'{float((_gross_exposure_ser(fixture_dict) > 0).mean()) * 100:.1f}',
        'Max. Drawdown [%]': f'{float(drawdown_ser.min()) * 100:.2f}',
        'Max. Drawdown Duration [days]': '1,284',
        'Time Under Water [%]': f'{float((drawdown_ser < 0).mean()) * 100:.1f}',
        '# Drawdowns': '148',
        '# Drawdowns / year': f'{148 / year_count_float:.1f}',
        'Total Commissions [$]': f'{_TOTAL_COMMISSION_FLOAT:,.0f}',
        'Turnover (Ann.) [%]': '186.4',
        'Estimated Slippage [$]': '3,190',
        'Total Trading Costs [$]': '8,002',
        'Cost Drag (Ann.) [%]': '0.21',
        # Historical (non-parametric) tail estimates on daily returns. VaR 95%
        # is the 5th percentile; CVaR is the mean of days at or below it.
        # Monthly skew is reported separately because compounding and overlap
        # make it a different animal from daily skew, and allocators quote the
        # monthly figure.
        'Skewness (Daily)': f'{float(strategy_return_ser.skew()):.2f}',
        'Skewness (Monthly)': (
            f'{float(strategy_total_value_ser.resample("ME").last().pct_change(fill_method=None).dropna().skew()):.2f}'
        ),
        'Excess Kurtosis (Daily)': f'{float(strategy_return_ser.kurtosis()):.2f}',
        'Worst Day [%]': f'{float(strategy_return_ser.min()) * 100:.2f}',
        'VaR 95% (Daily) [%]': f'{float(np.percentile(strategy_return_ser, 5.0)) * 100:.2f}',
        'CVaR 95% (Daily) [%]': (
            f'{float(strategy_return_ser[strategy_return_ser <= np.percentile(strategy_return_ser, 5.0)].mean()) * 100:.2f}'
        ),
        'AAR [%]': f'{float(strategy_return_ser.mean()) * _TRADING_DAY_PER_YEAR_FLOAT * 100:.2f}',
        'Downside L1 [%]': f'{float(loss_day_ser.abs().mean()) * 100:.2f}',
        'Avg. Loss Day [%]': f'{float(loss_day_ser.mean()) * 100:.2f}',
        'Avg. Drawdown [%]': f'{float(drawdown_ser[drawdown_ser < 0].mean()) * 100:.2f}',
        'Avg. Drawdown Duration [days]': '46',
        '# Trades': f'{trade_count_int:,}',
        'Trades / Week': f'{trade_count_int / week_count_float:.1f}' if week_count_float else '—',
        '% Positive Trades': (
            f'{len(win_trade_vec) / trade_count_int * 100:.1f}' if trade_count_int else '—'
        ),
        'Avg. Return / Trade [%]': (
            f'{float(trade_return_vec.mean()) * 100:.2f}' if trade_count_int else '—'
        ),
        # Payoff = average win / average loss magnitude; > 1 means winners are
        # bigger than losers, which is how a low win rate can still profit.
        'Payoff Ratio': (
            f'{avg_win_float / abs(avg_loss_float):.2f}' if avg_loss_float < 0.0 else '—'
        ),
        'Avg. Commission / Trade [$]': (
            f'{_TOTAL_COMMISSION_FLOAT / trade_count_int:.2f}' if trade_count_int else '—'
        ),
    }


def build_performance_summary_html(fixture_dict: dict[str, object]) -> str:
    """Every section report.py emits, as separate ruled sub-tables.

    The Exposure Time row carries an inline sparkline of gross exposure: the
    single number answers "how often deployed", the trace answers "when not" —
    one long regime-off year and 5% evenly-spread leakage print the same
    percentage but are different strategies.
    """
    value_by_metric_dict = build_summary_value_dict(fixture_dict)
    exposure_spark_html_str = build_sparkline_img_html(
        # Month-end sampling keeps the inline image light while preserving the
        # regime-off gaps, which are the only feature this trace exists to show.
        _gross_exposure_ser(fixture_dict).resample('ME').last(),
        width_px_int=110,
        height_px_int=15,
    )
    extra_html_by_metric_dict = {'Exposure Time [%]': exposure_spark_html_str}

    section_html_list = []
    for section_title_str, metric_name_tuple in PERFORMANCE_SUMMARY_SECTION_TUPLE:
        row_html_list = [
            f'<tr><td class="metric">{html.escape(metric_name_str)}</td>'
            f'<td>{html.escape(str(value_by_metric_dict.get(metric_name_str, "—")))}'
            f'{extra_html_by_metric_dict.get(metric_name_str, "")}</td></tr>'
            for metric_name_str in metric_name_tuple
        ]
        section_html_list.append(
            f'<div class="summary-section"><h3>{html.escape(section_title_str)}</h3>'
            '<div class="scroll"><table class="stats-table"><tbody>'
            + ''.join(row_html_list)
            + '</tbody></table></div></div>'
        )
    return '<div class="summary-section-stack">' + ''.join(section_html_list) + '</div>'


def _yearly_stat_df(total_value_ser: pd.Series) -> pd.DataFrame:
    """Per-calendar-year return, volatility, max drawdown and Sharpe.

    *** CRITICAL*** Each year's statistics are computed inside that year only,
    rebased to its first bar. Carrying a running peak across the year boundary
    would report a prior year's damage against this year's row.
    """
    stat_row_list = []
    for year_int, year_total_value_ser in total_value_ser.groupby(total_value_ser.index.year):
        year_return_ser = year_total_value_ser.pct_change(fill_method=None).dropna()
        if len(year_return_ser) < 2:
            continue
        year_growth_float = float(year_total_value_ser.iloc[-1] / year_total_value_ser.iloc[0])
        year_volatility_float = float(
            year_return_ser.std() * np.sqrt(_TRADING_DAY_PER_YEAR_FLOAT)
        )
        stat_row_list.append({
            'year_int': int(year_int),
            'return_float': year_growth_float - 1.0,
            'volatility_float': year_volatility_float,
            'max_drawdown_float': float(compute_drawdown(year_total_value_ser).min()),
            'sharpe_float': (
                (year_growth_float - 1.0) / year_volatility_float
                if year_volatility_float > 0.0 else np.nan
            ),
        })
    return pd.DataFrame(stat_row_list).set_index('year_int')


def _build_monthly_table_html(
        total_value_ser: pd.Series,
        year_count_int: int,
        shared_max_abs_loss_float: float,
) -> str:
    """One monthly grid: losing months shaded, plus that year's stats.

    Only losses carry a tint — monochrome has one axis (light to dark), so a
    diverging gain/loss ramp cannot work in this theme: a large gain and a
    mid-sized loss would land on nearly the same grey. Shading losses only
    makes the rule unambiguous — any shading is a losing month, darker is
    worse — and drawdown clusters become visible at a glance, which is what a
    monthly grid is actually scanned for.

    Tint intensity is scaled against the worst loss shared with the other
    table, so a -5% month is shaded identically in both. Scaling each table to
    its own extreme would make a calm book look as bruised as a violent one.
    """
    monthly_return_ser = (
        total_value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    )
    monthly_return_df = pd.DataFrame({
        'year_int': monthly_return_ser.index.year,
        'month_int': monthly_return_ser.index.month,
        'return_float': monthly_return_ser.to_numpy(),
    }).pivot(index='year_int', columns='month_int', values='return_float').tail(year_count_int)

    yearly_stat_df = _yearly_stat_df(total_value_ser)
    month_label_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    panel_color_str = str(SIGNATURE_PALETTE_DICT['panel'])

    row_html_list = []
    # Newest year first: the current year is what the reader checks, so it must
    # not sit below fourteen rows of history.
    for year_int, month_return_ser in monthly_return_df.iloc[::-1].iterrows():
        cell_html_list = []
        for month_int in range(1, 13):
            return_float = month_return_ser.get(month_int, np.nan)
            if pd.isna(return_float):
                cell_html_list.append('<td></td>')
                continue
            if return_float >= 0.0:
                cell_html_list.append(f'<td>{return_float * 100:.1f}</td>')
                continue
            blend_weight_float = (
                min(1.0, abs(return_float) / shared_max_abs_loss_float) * 0.55
            )
            cell_background_str = blend_hex_color_str(
                panel_color_str, str(SIGNATURE_PALETTE_DICT['loss']), blend_weight_float
            )
            cell_html_list.append(
                f'<td style="background:{cell_background_str}">{return_float * 100:.1f}</td>'
            )

        yearly_stat_ser = yearly_stat_df.loc[int(year_int)]
        row_html_list.append(
            f'<tr><td class="metric">{year_int}</td>'
            + ''.join(cell_html_list)
            + '<td class="divider-left">'
              f'{yearly_stat_ser["return_float"] * 100:.1f}</td>'
            + f'<td>{yearly_stat_ser["volatility_float"] * 100:.1f}</td>'
            + f'<td>{yearly_stat_ser["max_drawdown_float"] * 100:.1f}</td>'
            + f'<td>{yearly_stat_ser["sharpe_float"]:.2f}</td></tr>'
        )

    return (
        '<div class="scroll"><table class="heatmap">'
        '<thead><tr><th>Year</th>'
        + ''.join(f'<th>{month_str}</th>' for month_str in month_label_list)
        + '<th class="divider-left">Year</th><th>Vol</th><th>Max DD</th><th>Sharpe</th>'
        + '</tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def _max_abs_monthly_loss_float(total_value_ser: pd.Series, year_count_int: int) -> float:
    monthly_return_ser = (
        total_value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    )
    recent_year_int = int(monthly_return_ser.index.year.max()) - year_count_int
    recent_return_ser = monthly_return_ser[monthly_return_ser.index.year > recent_year_int]
    recent_loss_ser = recent_return_ser[recent_return_ser < 0.0]
    if len(recent_loss_ser) == 0:
        return 1.0
    return float(recent_loss_ser.abs().max())


def build_monthly_returns_html(
        fixture_dict: dict[str, object],
        benchmark_label_str: str = 'SPY',
        year_count_int: int = 14,
) -> str:
    """Two monthly grids — strategy above, benchmark below, same tint scale."""
    strategy_total_value_ser = fixture_dict['strategy_total_value_ser']
    benchmark_total_value_ser = fixture_dict['benchmark_total_value_ser']

    shared_max_abs_loss_float = max(
        _max_abs_monthly_loss_float(strategy_total_value_ser, year_count_int),
        _max_abs_monthly_loss_float(benchmark_total_value_ser, year_count_int),
    )

    return (
        '<h3>Strategy</h3>'
        + _build_monthly_table_html(
            strategy_total_value_ser, year_count_int, shared_max_abs_loss_float
        )
        + f'<h3 style="margin-top:22px">{html.escape(benchmark_label_str)} (benchmark)</h3>'
        + _build_monthly_table_html(
            benchmark_total_value_ser, year_count_int, shared_max_abs_loss_float
        )
        + '<p class="metric-context">Monthly returns in per cent. Only losing months are '
          'shaded — darker is worse — on a loss scale shared between both tables. Each year’s '
          'return, volatility, max drawdown and Sharpe are computed within that calendar year '
          'only.</p>'
    )


def build_open_positions_html(fixture_dict: dict[str, object]) -> str:
    """The book as of the last bar, marked to the last known close.

    Mirrors the real report's Open Trades section: name, weight, entry date,
    bars held, and unrealised P&L. P&L is the mark from entry close to the last
    known close per name — the same figure the live pod reconciles against.

        pnl_pct = last_close / entry_close - 1

    P&L is the last column because it is the payoff the eye lands on last, and
    a weighted portfolio P&L closes the table.
    """
    holding_weight_df = fixture_dict['rotation_holding_df']
    price_df = fixture_dict['rotation_price_df']
    is_held_df = holding_weight_df.fillna(0.0).abs().gt(0.0)
    last_weight_ser = holding_weight_df.iloc[-1]
    last_price_ser = price_df.iloc[-1]
    open_name_list = [
        name_str for name_str in holding_weight_df.columns if bool(is_held_df.iloc[-1][name_str])
    ]
    if len(open_name_list) == 0:
        return (
            '<p class="metric-context">No open positions — the book is flat as of '
            f'{holding_weight_df.index[-1].date()}.</p>'
        )

    open_position_row_list = []
    for name_str in open_name_list:
        held_flag_vec = is_held_df[name_str].to_numpy()
        # Walk back to the first bar of the *current* uninterrupted spell; an
        # earlier hold of the same name is a closed trade, not this position.
        entry_bar_idx_int = len(held_flag_vec) - 1
        while entry_bar_idx_int > 0 and held_flag_vec[entry_bar_idx_int - 1]:
            entry_bar_idx_int -= 1
        entry_price_float = float(price_df[name_str].iloc[entry_bar_idx_int])
        open_position_row_list.append({
            'name_str': name_str,
            'weight_float': float(last_weight_ser[name_str]),
            'entry_ts': holding_weight_df.index[entry_bar_idx_int],
            'bars_held_int': len(held_flag_vec) - entry_bar_idx_int,
            'entry_price_float': entry_price_float,
            'last_price_float': float(last_price_ser[name_str]),
            'pnl_float': float(last_price_ser[name_str]) / entry_price_float - 1.0,
        })
    open_position_row_list.sort(key=lambda row_dict: row_dict['entry_ts'])

    row_html_list = [
        f'<tr><td class="metric">{html.escape(row_dict["name_str"])}</td>'
        f'<td>{row_dict["weight_float"] * 100:.1f}%</td>'
        f'<td>{row_dict["entry_ts"].date()}</td>'
        f'<td>{row_dict["bars_held_int"]}</td>'
        f'<td>{row_dict["entry_price_float"]:.2f}</td>'
        f'<td>{row_dict["last_price_float"]:.2f}</td>'
        f'<td class="{"pos" if row_dict["pnl_float"] >= 0 else "neg"}">'
        f'{row_dict["pnl_float"] * 100:+.1f}%</td></tr>'
        for row_dict in open_position_row_list
    ]

    gross_exposure_float = float(last_weight_ser.abs().sum())
    # Weighted book P&L: each name's mark weighted by its share of gross, so the
    # total is the portfolio's unrealised return on deployed capital, not a
    # naive average of per-name percentages.
    weighted_pnl_float = (
        sum(row_dict['weight_float'] * row_dict['pnl_float'] for row_dict in open_position_row_list)
        / gross_exposure_float if gross_exposure_float > 0.0 else 0.0
    )
    row_html_list.append(
        '<tr><td class="metric">Book</td>'
        f'<td>{gross_exposure_float * 100:.1f}%</td><td></td><td></td><td></td><td></td>'
        f'<td class="{"pos" if weighted_pnl_float >= 0 else "neg"}">'
        f'{weighted_pnl_float * 100:+.1f}%</td></tr>'
    )

    return (
        '<div class="scroll"><table class="stats-table">'
        '<thead><tr><th>Symbol</th><th>Weight</th><th>Entry</th><th>Bars held</th>'
        '<th>Entry px</th><th>Last px</th><th>P&amp;L</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
        f'<p class="metric-context">As of {holding_weight_df.index[-1].date()}, sorted by entry '
        'date and marked to the last known close. Book P&amp;L is weighted by each name’s share '
        'of gross exposure.</p>'
    )


def build_conditional_beta_html(fixture_dict: dict[str, object]) -> str:
    """Downside/upside beta, correlation and capture, with the scatter."""
    strategy_return_ser = (
        fixture_dict['strategy_total_value_ser'].pct_change(fill_method=None).dropna()
    )
    benchmark_return_ser = (
        fixture_dict['benchmark_total_value_ser'].pct_change(fill_method=None).dropna()
    )
    conditional_metric_dict = compute_conditional_beta_dict(
        strategy_return_ser, benchmark_return_ser
    )

    row_spec_list = [
        ('Beta', 'down_beta_float', 'up_beta_float', '{:.2f}'),
        ('Correlation', 'down_correlation_float', 'up_correlation_float', '{:.2f}'),
        ('Capture', 'down_capture_float', 'up_capture_float', '{:.0%}'),
        ('Observations', 'down_day_count_float', 'up_day_count_float', '{:,.0f}'),
    ]
    row_html_list = [
        f'<tr><td class="metric">{html.escape(label_str)}</td>'
        f'<td>{format_str.format(conditional_metric_dict[down_key_str])}</td>'
        f'<td>{format_str.format(conditional_metric_dict[up_key_str])}</td></tr>'
        for label_str, down_key_str, up_key_str, format_str in row_spec_list
    ]
    asymmetry_float = conditional_metric_dict['beta_asymmetry_float']
    row_html_list.append(
        '<tr><td class="metric">Beta asymmetry</td>'
        f'<td colspan="2">{asymmetry_float:+.2f} (up minus down)</td></tr>'
    )

    return (
        '<div class="scroll"><table class="stats-table">'
        '<thead><tr><th>Metric</th><th>Benchmark down</th><th>Benchmark up</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
        '<p class="metric-context">Conditioning is on the benchmark’s sign, never the '
        'strategy’s own — selecting on the strategy’s bad days would flatter it by '
        'construction. A lower down-beta than up-beta is the asymmetry a defensive book is '
        'built for.</p>'
    )
