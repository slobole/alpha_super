import io
import base64
import html
import inspect
import json
import os
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from alpha.engine.metrics import generate_monthly_returns, generate_overall_metrics
from alpha.engine.plot import plot as render_strategy_plot
from alpha.engine.signature import (
    build_metric_delta_table_html,
    compute_conditional_beta_dict,
    render_composition_data_uri_str,
    render_relative_performance_data_uri_str,
    render_small_multiples_data_uri_str,
)
from alpha.engine.theme import (
    SIGNATURE_ASSET_COLOR_DICT,
    SIGNATURE_PALETTE_DICT,
    blend_hex_color_str,
    build_report_css,
    build_report_font_head_html,
    build_signature_rcparams,
    signature_variant_context,
)


# The signature variant every report renders under. desk is the shipped look
# and the one Bench renders with, so the console and the artifacts embedded in
# it agree; override with ALPHA_REPORT_VARIANT_STR (e.g. 'current' for the
# legacy card dashboard) without touching code. The report resolves its CSS,
# fonts and charts inside this variant at render time — see _render_report_html.
#
# *** UI*** Charts are rasterised at render time, so this variant is baked into
# every saved PNG. Artifacts written under a previous variant keep that variant
# for good; only a fresh run adopts a change here.
_ACTIVE_REPORT_VARIANT_STR = os.environ.get('ALPHA_REPORT_VARIANT_STR', 'desk')


_MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_METADATA_FILENAME = 'metadata.json'
_RUN_INFO_FILENAME = 'run_info.json'
_SUMMARY_FILENAME = 'summary.json'
_TRANSACTION_CSV_FILENAME = 'transactions.csv'
_DIVIDEND_LEDGER_CSV_FILENAME = 'dividend_ledger.csv'
_CRISIS_METRICS_CSV_FILENAME = 'crisis_metrics.csv'
_CRISIS_PATHS_CSV_FILENAME = 'crisis_paths.csv'
_TAIL_SUMMARY_CSV_FILENAME = 'tail_summary.csv'
_TAIL_CONTRIBUTION_CSV_FILENAME = 'tail_contribution.csv'
_TAIL_RETURNS_CSV_FILENAME = 'tail_returns.csv'
_REBALANCE_TARGET_WEIGHTS_CSV_FILENAME = 'rebalance_target_weights.csv'
_REBALANCE_DIAGNOSTICS_CSV_FILENAME = 'rebalance_diagnostics.csv'
_TRADING_DAY_PER_YEAR_FLOAT = 252.0
_DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT = 60
_TRADE_RETURN_HISTOGRAM_BIN_COUNT_INT = 60
_FALLBACK_ASSET_SET = frozenset({'SPY', 'SSO', 'QQQ', 'QLD', 'TQQQ', 'UPRO'})
_WEIGHT_STACK_EDGE_COLOR_STR = '#101418'


METRIC_HELP_TEXT_DICT = {
    'Duration [days]': 'Number of stored equity observations in the report period.',
    'Peak [$]': 'Highest portfolio equity observed during the report period.',
    'Return [%]': 'Total compounded return from starting capital to ending equity.',
    'Return (Ann.) [%]': 'Compounded annual return using the report\'s 252-trading-day convention.',
    'Volatility (Ann.) [%]': 'Standard deviation of daily returns annualized by the square root of 252.',
    'AAR [%]': 'Mean absolute realized daily return, including both positive and negative days.',
    'Downside L1 [%]': 'Mean daily downside magnitude across all days; non-loss days contribute zero.',
    'Avg. Loss Day [%]': 'Mean loss magnitude conditional on the strategy having a negative day.',
    'Sharpe Ratio': (
        'Mean daily return divided by daily-return standard deviation, multiplied by the square root of 252. '
        'The risk-free rate is zero. Uses every day in the window, including flat all-cash days, '
        'so pod and book figures share one basis.'
    ),
    'Sharpe Ratio (Active Days)': (
        'Same formula, but dead days — invested value zero and return zero — are excluded first. '
        'Shows risk-adjusted return while capital was actually deployed; blank when no '
        'invested-value series is available.'
    ),
    'Exposure Time [%]': (
        'Percentage of stored days covered by at least one closed-trade interval. '
        'The engine assumes 100% when no trade history is supplied.'
    ),
    'Exposure-Adjusted Return (Ann.) [%]': (
        'Annualized return divided by the fraction of days marked exposed. '
        'This binary measure does not account for position size or leverage.'
    ),
    'Correlation': (
        'Pearson correlation of aligned daily returns against the supplied comparison series. '
        'The engine currently reports 1 when no comparison series is supplied.'
    ),
    'Max. Drawdown [%]': 'Largest peak-to-trough decline in the realized equity curve.',
    'MAR Ratio': 'Annualized return divided by the absolute value of maximum drawdown.',
    'Sortino Ratio': (
        'Like Sharpe, but only losing days count as risk. '
        'sortino = annual return / (sqrt(mean(min(r,0)^2)) * sqrt(252)). Higher is better.'
    ),
    'Ulcer Index': (
        'How deep drawdowns go and how long they last, in one number. '
        'ulcer = sqrt(mean(drawdown_t^2)), drawdown in per cent. Lower is better.'
    ),
    'Volatility (Monthly) [%]': (
        'Spread of monthly returns, not annualized — what a monthly track record shows. '
        'vol = std(monthly returns).'
    ),
    'Positive Months [%]': (
        'Share of months that finished up. '
        'positive months = count(monthly return > 0) / count(months).'
    ),
    'Skewness (Daily)': (
        'Whether the tail is on the losing or winning side. Negative means rare large losses '
        'against many small gains; positive is the reverse. Zero is symmetric.'
    ),
    'Skewness (Monthly)': (
        'The same shape measured on monthly returns, which is the figure allocators quote. '
        'Monthly and daily skew can disagree because compounding reshapes the distribution.'
    ),
    'Excess Kurtosis (Daily)': (
        'How fat the tails are versus a normal distribution, which scores zero. '
        'Above zero means extreme days happen more often than a bell curve predicts.'
    ),
    'Worst Day [%]': 'The single worst daily return in the sample.',
    'Worst Month [%]': 'The single worst monthly return in the sample.',
    'VaR 95% (Daily) [%]': (
        'The daily loss only the worst 1-in-20 days exceed. Historical, not modelled: '
        'VaR = 5th percentile of daily returns.'
    ),
    'CVaR 95% (Daily) [%]': (
        'The average loss on the days that breach VaR — how bad it gets once it is bad. '
        'CVaR = mean(returns <= VaR 95%).'
    ),
    'AAR [%]': 'Average absolute daily move, annualized. A size-of-swing measure, not a return.',
    'Downside L1 [%]': 'Average size of a losing day. mean(|r|) over days where r < 0.',
    '# Drawdowns / month': 'Drawdown episodes per month. The annual rate divided by twelve.',
    'Time Under Water [%]': 'Percentage of stored days when equity was below its previous running peak.',
    'Avg. Drawdown [%]': 'Mean trough depth across distinct below-peak drawdown episodes.',
    'Max. Drawdown Duration [days]': 'Longest number of stored observations in one below-peak episode.',
    'Avg. Drawdown Duration [days]': 'Mean number of stored observations across below-peak episodes.',
    '# Drawdowns': 'Number of distinct continuous below-peak episodes in the equity curve.',
    '# Drawdowns / year': 'Drawdown-episode count divided by stored observations and scaled by 252.',
    'Total Commissions [$]': (
        'Modeled commissions already charged to backtest cash and equity; this row attributes that embedded cost.'
    ),
    'Turnover (Ann.) [%]': 'Gross traded notional divided by average equity and annualized.',
    'Estimated Slippage [$]': (
        'Attribution of slippage already embedded in backtest fills using the configured fixed price-penalty model; '
        'it is not a liquidity-aware live estimate.'
    ),
    'Total Trading Costs [$]': 'Attribution of modeled commissions plus slippage already reflected in backtest equity.',
    'Cost Drag (Ann.) [%]': (
        'Modeled trading-cost attribution divided by average equity and annualized; '
        'it is not the exact difference between gross and net CAGR.'
    ),
    'Beta': 'How much benchmark exposure the complete strategy behaved as if it had.',
    'Alpha (Ann.) [%]': 'Annualized return not explained by estimated benchmark exposure.',
    'Alpha HAC t-stat': 'Newey-West t-statistic for the estimated regression alpha.',
    'Alpha (Ann.) / HAC t-stat': (
        'Annualized zero-rate regression intercept with its Newey-West/HAC t-statistic, which measures '
        'sampling uncertainty under this regression model and is not adjusted for strategy selection or '
        'multiple comparisons.'
    ),
    'R²': 'Percentage of daily strategy-return variation explained by the benchmark regression.',
    'Mean Rank IC': (
        'Mean decision-date Spearman correlation between point-in-time signal ranks known at the decision and '
        'returns from the first executable price through the configured forecast horizon.'
    ),
    'ICIR': (
        'Mean decision-date information coefficient divided by its standard deviation and annualized using the '
        'actual decision frequency; interpret together with decision-date and cross-sectional sample counts.'
    ),
}


_PERFORMANCE_SUMMARY_SECTION_TUPLE = (
    (
        'Period & Capital',
        ('Start', 'End', 'Duration [days]', 'Start [$]', 'Final [$]', 'Peak [$]'),
        False,
    ),
    (
        'Return & Risk-Adjusted Performance',
        (
            'Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]',
            'Volatility (Monthly) [%]', 'Sharpe Ratio', 'Sharpe Ratio (Active Days)',
            'MAR Ratio', 'Positive Months [%]',
        ),
        False,
    ),
    (
        'Benchmark Regression',
        ('Beta', 'Alpha (Ann.) [%]', 'Alpha HAC t-stat', 'R²'),
        False,
    ),
    (
        'Drawdown & Recovery',
        (
            'Max. Drawdown [%]',
            'Max. Drawdown Duration [days]',
            'Time Under Water [%]',
            '# Drawdowns',
            '# Drawdowns / year',
            '# Drawdowns / month',
        ),
        False,
    ),
    (
        'Trading Activity & Costs',
        (
            'Total Commissions [$]',
            'Turnover (Ann.) [%]',
            'Estimated Slippage [$]',
            'Total Trading Costs [$]',
            'Cost Drag (Ann.) [%]',
        ),
        False,
    ),
    (
        'Extended Risk Diagnostics',
        (
            'Sortino Ratio',
            'Ulcer Index',
            'Downside L1 [%]',
            'Avg. Loss Day [%]',
            'Avg. Drawdown [%]',
            'Avg. Drawdown Duration [days]',
        ),
        False,
    ),
    (
        'Distribution & Tails',
        (
            'Skewness (Daily)',
            'Skewness (Monthly)',
            'Excess Kurtosis (Daily)',
            'Worst Day [%]',
            'Worst Month [%]',
            'VaR 95% (Daily) [%]',
            'CVaR 95% (Daily) [%]',
        ),
        False,
    ),
)
# AAR is the arithmetic average annual return, which duplicates the compounded
# Return (Ann.) already shown in the headline and summary.
_PERFORMANCE_SUMMARY_HIDDEN_METRIC_SET = frozenset({
    'Correlation',
    'Exposure-Adjusted Return (Ann.) [%]',
    'AAR [%]',
    'Exposure Time [%]',
})

_METRIC_TOOLTIP_HTML_STR = (
    '<div id="metric-help-tooltip" class="metric-help-tooltip" role="tooltip" hidden></div>'
)
_METRIC_TOOLTIP_SCRIPT_STR = r'''
<script>
(() => {
    const tooltip = document.getElementById('metric-help-tooltip');
    if (!tooltip) return;
    document.documentElement.classList.add('metric-tooltip-js-enabled');
    let pinnedTrigger = null;

    const hideTooltip = () => {
        document.querySelectorAll('.metric-help[aria-expanded="true"]').forEach((trigger) => {
            trigger.setAttribute('aria-expanded', 'false');
            trigger.removeAttribute('aria-describedby');
        });
        tooltip.hidden = true;
        pinnedTrigger = null;
    };

    const showTooltip = (trigger) => {
        tooltip.textContent = trigger.dataset.help || '';
        tooltip.hidden = false;
        trigger.setAttribute('aria-expanded', 'true');
        trigger.setAttribute('aria-describedby', 'metric-help-tooltip');

        const triggerRect = trigger.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const margin = 8;
        let left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2;
        left = Math.max(margin, Math.min(left, window.innerWidth - tooltipRect.width - margin));
        let top = triggerRect.bottom + margin;
        if (top + tooltipRect.height > window.innerHeight - margin) {
            top = Math.max(margin, triggerRect.top - tooltipRect.height - margin);
        }
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    };

    document.querySelectorAll('.metric-help').forEach((trigger) => {
        trigger.addEventListener('mouseenter', () => {
            if (!pinnedTrigger) showTooltip(trigger);
        });
        trigger.addEventListener('mouseleave', () => {
            if (!pinnedTrigger) hideTooltip();
        });
        trigger.addEventListener('focus', () => showTooltip(trigger));
        trigger.addEventListener('blur', () => {
            if (pinnedTrigger === trigger) pinnedTrigger = null;
            hideTooltip();
        });
        trigger.addEventListener('click', (event) => {
            event.stopPropagation();
            if (pinnedTrigger === trigger) {
                hideTooltip();
                return;
            }
            hideTooltip();
            pinnedTrigger = trigger;
            showTooltip(trigger);
        });
    });
    document.addEventListener('click', hideTooltip);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') hideTooltip();
    });
})();
</script>
'''


def build_research_output_path(
    output_dir: str | Path,
    entity_type_str: str,
    entity_id_str: str,
    analysis_type_str: str,
    timestamp_str: str | None = None,
) -> Path:
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return (
        Path(output_dir)
        / 'research'
        / str(entity_type_str)
        / str(entity_id_str)
        / str(analysis_type_str)
        / str(timestamp_str)
    )


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _weight_color_for_asset(asset_name_str: str) -> str:
    """Resolve an asset's chart colour from the active signature palette.

    Read from the palette rather than the module-level baseline so a monochrome
    variant maps assets onto its own grey ramp instead of leaving a blue TLT
    and a gold GLD inside an otherwise colourless report.
    """
    normalized_asset_name_str = str(asset_name_str).upper()
    if normalized_asset_name_str in _FALLBACK_ASSET_SET:
        return SIGNATURE_PALETTE_DICT['benchmark']
    active_asset_color_dict = SIGNATURE_PALETTE_DICT['asset_color_dict']
    return active_asset_color_dict.get(
        normalized_asset_name_str,
        active_asset_color_dict['DEFAULT'],
    )


def _write_metadata(metadata_path: Path, metadata_dict: dict):
    metadata_path.write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True, default=_json_default),
        encoding='utf-8',
    )


def _compact_dict(raw_dict: dict) -> dict:
    return {
        key_str: value_obj
        for key_str, value_obj in raw_dict.items()
        if value_obj is not None
    }


def _json_float(value_obj):
    if value_obj is None:
        return None
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_float):
        return None
    return value_float


def _json_date_str(value_obj) -> str | None:
    if value_obj is None:
        return None
    try:
        timestamp_obj = pd.Timestamp(value_obj)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp_obj):
        return None
    return timestamp_obj.date().isoformat()


def _first_summary_value(summary_df: pd.DataFrame | None, metric_name_list: list[str]):
    if summary_df is None:
        return None
    if len(summary_df) == 0:
        return None
    for metric_name_str in metric_name_list:
        if metric_name_str not in summary_df.index:
            continue
        value_obj = summary_df.loc[metric_name_str]
        if isinstance(value_obj, pd.Series):
            for candidate_obj in value_obj:
                if pd.notna(candidate_obj):
                    return candidate_obj
            continue
        if pd.notna(value_obj):
            return value_obj
    return None


def _entity_column_summary_value(summary_df: pd.DataFrame | None, metric_name_str: str):
    """Read a metric from the entity's own (first) summary column only.

    'Sharpe Ratio (Active Days)' is legitimately NaN for benchmark columns, so
    scanning across columns for the first non-NaN value would silently report a
    benchmark's figure as the entity's.
    """
    if summary_df is None or len(summary_df) == 0 or len(summary_df.columns) == 0:
        return None
    if metric_name_str not in summary_df.index:
        return None
    value_obj = summary_df[summary_df.columns[0]].loc[metric_name_str]
    if pd.isna(value_obj):
        return None
    return value_obj


def _trade_count_int(summary_trades_df: pd.DataFrame | None, fallback_trade_df: pd.DataFrame | None = None):
    if summary_trades_df is not None and '# Trades' in summary_trades_df.index:
        value_obj = _first_summary_value(summary_trades_df, ['# Trades'])
        value_float = _json_float(value_obj)
        if value_float is not None:
            return int(value_float)
    if fallback_trade_df is not None:
        return int(len(fallback_trade_df))
    return None


def _result_window_dict(result_df: pd.DataFrame | None) -> dict:
    if result_df is None or len(result_df) == 0:
        return {}
    return {
        'start_date': _json_date_str(result_df.index.min()),
        'end_date': _json_date_str(result_df.index.max()),
    }


def _summary_metrics_dict(result_obj) -> dict:
    summary_df = getattr(result_obj, 'summary', None)
    result_df = getattr(result_obj, 'results', None)
    final_equity_float = _json_float(_first_summary_value(summary_df, ['Final [$]']))
    if final_equity_float is None and result_df is not None and 'total_value' in result_df.columns and len(result_df) > 0:
        final_equity_float = _json_float(result_df['total_value'].iloc[-1])

    return _compact_dict(
        {
            'final_equity': final_equity_float,
            'ann_return_pct': _json_float(_first_summary_value(summary_df, ['Return (Ann.) [%]'])),
            'sharpe': _json_float(_first_summary_value(summary_df, ['Sharpe Ratio'])),
            'sharpe_active_days': _json_float(
                _entity_column_summary_value(summary_df, 'Sharpe Ratio (Active Days)')
            ),
            'max_drawdown_pct': _json_float(_first_summary_value(summary_df, ['Max. Drawdown [%]'])),
            'trade_count': _trade_count_int(
                getattr(result_obj, 'summary_trades', None),
                getattr(result_obj, '_trades', None),
            ),
            'benchmark_regression': getattr(
                result_obj,
                'benchmark_regression_metadata_by_column_dict',
                None,
            ),
            'standalone_benchmark_regression': getattr(
                result_obj,
                'standalone_benchmark_regression_metadata_by_column_dict',
                None,
            ),
        }
    )


def _strategy_run_info_dict(strategy) -> dict:
    result_df = getattr(strategy, 'results', None)
    summary_df = getattr(strategy, 'summary', None)
    window_dict = {
        'start_date': _json_date_str(_first_summary_value(summary_df, ['Start'])),
        'end_date': _json_date_str(_first_summary_value(summary_df, ['End'])),
    }
    window_dict.update({
        key_str: value_obj
        for key_str, value_obj in _result_window_dict(result_df).items()
        if window_dict.get(key_str) is None
    })
    return {
        'entity_type': 'strategy',
        'entity_id': strategy.name,
        'analysis_type': 'vanilla_backtest',
        'parameters': _compact_dict(
            {
                'capital': _json_float(getattr(strategy, '_capital_base', None)),
                **window_dict,
            }
        ),
    }


def _portfolio_run_info_dict(portfolio) -> dict:
    pod_info_list = []
    weight_list = list(getattr(portfolio, '_weights', []))
    for position_int, pod_info_dict in enumerate(getattr(portfolio, 'pod_info_list', []) or []):
        weight_float = pod_info_dict.get('weight_float')
        if weight_float is None and position_int < len(weight_list):
            weight_float = weight_list[position_int]
        pod_info_list.append(
            _compact_dict(
                {
                    'pod_id': pod_info_dict.get('pod_id_str'),
                    'strategy': pod_info_dict.get('strategy_name'),
                    'weight': _json_float(weight_float),
                }
            )
        )

    return {
        'entity_type': 'portfolio',
        'entity_id': portfolio.name,
        'analysis_type': 'vanilla_backtest',
        'parameters': _compact_dict(
            {
                'capital': _json_float(getattr(portfolio, '_capital_base', None)),
                'start_date': _json_date_str(getattr(portfolio, '_common_start', None)),
                'end_date': _json_date_str(getattr(portfolio, '_common_end', None)),
                'config_path': getattr(portfolio, 'source_config_path', None),
                'pods': pod_info_list if len(pod_info_list) > 0 else None,
            }
        ),
    }


def _crisis_run_info_dict(crisis_replay_result) -> dict:
    return {
        'entity_type': 'strategy',
        'entity_id': crisis_replay_result.strategy_key_str,
        'analysis_type': 'stress_analysis',
        'parameters': _compact_dict(
            {
                'stress_type': 'crisis_replay',
                'capital': _json_float(crisis_replay_result.capital_base_float),
                'crisis_windows': [
                    {
                        'name': crisis_period_config.crisis_name_str,
                        'start_date': crisis_period_config.start_date_str,
                        'end_date': crisis_period_config.end_date_str,
                    }
                    for crisis_period_config in crisis_replay_result.crisis_period_config_list
                ],
            }
        ),
    }


def _crisis_summary_dict(crisis_replay_result) -> dict:
    metric_df = getattr(crisis_replay_result, 'crisis_metric_df', pd.DataFrame())
    summary_dict = {'crisis_count': int(len(metric_df))}
    if metric_df is not None and len(metric_df) > 0:
        if 'strategy_return_pct_float' in metric_df.columns:
            summary_dict['worst_strategy_return_pct'] = _json_float(
                metric_df['strategy_return_pct_float'].min()
            )
        if 'relative_return_pct_float' in metric_df.columns:
            summary_dict['worst_relative_return_pct'] = _json_float(
                metric_df['relative_return_pct_float'].min()
            )
    return _compact_dict(summary_dict)


def _write_transaction_csv(transaction_df: pd.DataFrame | None, transaction_csv_path: Path):
    if transaction_df is None:
        transaction_df = pd.DataFrame()
    transaction_export_df = transaction_df.copy()
    transaction_export_df.to_csv(transaction_csv_path, index=False, date_format='%Y-%m-%d')


def _write_portfolio_tail_csvs(portfolio, output_path: Path):
    tail_summary_df = getattr(portfolio, 'tail_summary_df', pd.DataFrame())
    tail_contribution_df = getattr(portfolio, 'tail_contribution_df', pd.DataFrame())
    tail_return_df = getattr(portfolio, 'tail_return_df', pd.DataFrame())
    if tail_summary_df is None:
        tail_summary_df = pd.DataFrame()
    if tail_contribution_df is None:
        tail_contribution_df = pd.DataFrame()
    if tail_return_df is None:
        tail_return_df = pd.DataFrame()

    tail_summary_df.to_csv(output_path / _TAIL_SUMMARY_CSV_FILENAME)
    tail_contribution_df.to_csv(output_path / _TAIL_CONTRIBUTION_CSV_FILENAME, date_format='%Y-%m-%d')
    tail_return_df.to_csv(output_path / _TAIL_RETURNS_CSV_FILENAME, date_format='%Y-%m-%d')


def _write_portfolio_rebalance_csvs(portfolio, output_path: Path):
    target_weight_df = getattr(portfolio, 'rebalance_target_weight_df', pd.DataFrame())
    diagnostic_df = getattr(portfolio, 'rebalance_diagnostic_df', pd.DataFrame())
    if target_weight_df is None:
        target_weight_df = pd.DataFrame()
    if diagnostic_df is None:
        diagnostic_df = pd.DataFrame()

    target_weight_df.to_csv(
        output_path / _REBALANCE_TARGET_WEIGHTS_CSV_FILENAME,
        date_format='%Y-%m-%d',
    )
    diagnostic_df.to_csv(
        output_path / _REBALANCE_DIAGNOSTICS_CSV_FILENAME,
        date_format='%Y-%m-%d',
    )


def _strategy_metadata_dict(strategy, pickle_path: Path) -> dict:
    try:
        class_file = inspect.getfile(strategy.__class__)
    except TypeError:
        class_file = None

    return {
        'artifact_type': 'strategy',
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'pickle_path': pickle_path.resolve(),
        'strategy_name': strategy.name,
        'class_name': strategy.__class__.__name__,
        'class_module': strategy.__class__.__module__,
        'class_file': Path(class_file).resolve() if class_file is not None else None,
        'capital_base': float(strategy._capital_base),
        'benchmarks': list(strategy._benchmarks),
        'benchmark_data_symbol_map': dict(
            getattr(strategy, '_benchmark_data_symbol_map_dict', {})
        ),
        'accounting_policy': dict(getattr(strategy, '_accounting_policy_dict', {})),
        'data_adjustment_policy': dict(
            getattr(strategy, '_data_adjustment_policy_dict', {})
        ),
    }


def _portfolio_metadata_dict(portfolio, pickle_path: Path) -> dict:
    return {
        'artifact_type': 'portfolio',
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'pickle_path': pickle_path.resolve(),
        'portfolio_name': portfolio.name,
        'capital_base': float(portfolio._capital_base),
        'rebalance': portfolio._rebalance,
        'rebalance_policy': getattr(portfolio, '_rebalance_policy', 'fixed'),
        'rebalance_inverse_volatility_lookback_day_int': getattr(
            portfolio,
            '_rebalance_inverse_volatility_lookback_day_int',
            None,
        ),
        'source_config_path': portfolio.source_config_path,
        'common_start': portfolio._common_start,
        'common_end': portfolio._common_end,
        'pods': portfolio.pod_info_list,
    }


def _crisis_replay_metadata_dict(crisis_replay_result) -> dict:
    return {
        'artifact_type': 'crisis_replay',
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'strategy_key': crisis_replay_result.strategy_key_str,
        'strategy_name': crisis_replay_result.strategy_name_str,
        'capital_base': float(crisis_replay_result.capital_base_float),
        'configured_crisis_count': int(len(crisis_replay_result.crisis_period_config_list)),
        'evaluated_crisis_count': int(crisis_replay_result.crisis_metric_df.shape[0]),
        'crisis_periods': [
            {
                'crisis_name_str': crisis_period_config.crisis_name_str,
                'start_date_str': crisis_period_config.start_date_str,
                'end_date_str': crisis_period_config.end_date_str,
            }
            for crisis_period_config in crisis_replay_result.crisis_period_config_list
        ],
    }


def save_results(strategy, output_dir='results', output_path: str | Path | None = None) -> Path:
    """Save strategy results to a structured folder and generate an HTML report.

    Creates:
        {output_dir}/research/strategy/{strategy.name}/vanilla_backtest/{YYYY-MM-DD_HHMMSS}/
            {strategy.name}.pkl
            report.html
            transactions.csv
            dividend_ledger.csv

    Returns the output directory path.
    """
    out = (
        Path(output_path)
        if output_path is not None
        else build_research_output_path(output_dir, 'strategy', strategy.name, 'vanilla_backtest')
    )
    out.mkdir(parents=True, exist_ok=True)

    pickle_path = out / f'{strategy.name}.pkl'
    strategy.to_pickle(pickle_path)
    _write_metadata(out / _METADATA_FILENAME, _strategy_metadata_dict(strategy, pickle_path))
    _write_metadata(out / _RUN_INFO_FILENAME, _strategy_run_info_dict(strategy))
    _write_metadata(out / _SUMMARY_FILENAME, _summary_metrics_dict(strategy))

    # Chart and HTML are rendered inside the active signature variant so the
    # flagship chart and the page styling always agree.
    with signature_variant_context(_ACTIVE_REPORT_VARIANT_STR):
        buf = io.BytesIO()
        strategy.plot(save_to=buf)
        plt.close('all')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('ascii')
        html = _build_html(strategy, chart_b64)
    (out / 'report.html').write_text(html, encoding='utf-8')
    _write_transaction_csv(strategy._transactions, out / _TRANSACTION_CSV_FILENAME)
    strategy.get_dividend_ledger().to_csv(
        out / _DIVIDEND_LEDGER_CSV_FILENAME,
        index=False,
        date_format='%Y-%m-%d',
    )

    print(f'Results saved to: {out.resolve()}')
    return out


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ formatting helpers ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

def save_crisis_replay_results(crisis_replay_result, output_dir='results') -> Path:
    """Save crisis replay artifacts and generate an HTML report."""
    out = build_research_output_path(
        output_dir,
        'strategy',
        crisis_replay_result.strategy_key_str,
        'stress_analysis',
    )
    out.mkdir(parents=True, exist_ok=True)

    _write_metadata(out / _METADATA_FILENAME, _crisis_replay_metadata_dict(crisis_replay_result))
    _write_metadata(out / _RUN_INFO_FILENAME, _crisis_run_info_dict(crisis_replay_result))
    _write_metadata(out / _SUMMARY_FILENAME, _crisis_summary_dict(crisis_replay_result))
    crisis_replay_result.crisis_metric_df.to_csv(
        out / _CRISIS_METRICS_CSV_FILENAME,
        index=False,
        date_format='%Y-%m-%d',
    )
    crisis_replay_result.crisis_path_df.to_csv(
        out / _CRISIS_PATHS_CSV_FILENAME,
        index=False,
        date_format='%Y-%m-%d',
    )

    with signature_variant_context(_ACTIVE_REPORT_VARIANT_STR):
        html = _build_crisis_replay_html(crisis_replay_result)
    (out / 'report.html').write_text(html, encoding='utf-8')

    print(f'Results saved to: {out.resolve()}')
    return out


def _fmt_pct(val):
    try:
        return f'{float(val):.2f}%'
    except (TypeError, ValueError):
        return str(val)


def _fmt_dollar(val):
    try:
        return f'${float(val):,.2f}'
    except (TypeError, ValueError):
        return str(val)


def _fmt_num(val, decimals=2):
    try:
        return f'{float(val):,.{decimals}f}'
    except (TypeError, ValueError):
        return str(val)


def _safe_summary_metric_value(summary_df: pd.DataFrame | None, column_name_str: str, metric_name_str: str):
    """Return a summary-table value when the metric exists; otherwise None."""
    if summary_df is None:
        return None
    if column_name_str not in summary_df.columns or metric_name_str not in summary_df.index:
        return None

    metric_value_obj = summary_df.loc[metric_name_str, column_name_str]
    if metric_value_obj == '' or (isinstance(metric_value_obj, float) and np.isnan(metric_value_obj)):
        return None
    return metric_value_obj


def _fmt_signed_pct(metric_value_obj) -> str:
    try:
        return f'{float(metric_value_obj):+,.2f}%'
    except (TypeError, ValueError):
        return 'N/A'


def _format_kpi_value_str(metric_name_str: str, metric_value_obj) -> str:
    """Format KPI values using the summary-table metric conventions."""
    if metric_value_obj is None:
        return 'N/A'
    if metric_name_str in {'Return [%]', 'Return (Ann.) [%]', 'Max. Drawdown [%]'}:
        return _fmt_signed_pct(metric_value_obj)
    if metric_name_str == 'Volatility (Ann.) [%]':
        return _fmt_pct(metric_value_obj)
    if '[$]' in metric_name_str:
        return _fmt_dollar(metric_value_obj)
    if metric_name_str in {'Sharpe Ratio', 'Beta'}:
        return _fmt_num(metric_value_obj, 2)
    return str(metric_value_obj)


def _kpi_value_class_str(metric_name_str: str, metric_value_obj) -> str:
    if metric_value_obj is None:
        return ''
    if metric_name_str in {'Return [%]', 'Return (Ann.) [%]', 'Alpha (Ann.) [%]'}:
        return _signed_value_class_str(metric_value_obj)
    return ''


def _build_kpi_card_html(
    title_str: str,
    value_str: str,
    note_str: str,
    value_class_str: str = '',
) -> str:
    value_class_attr_str = f'kpi-value {value_class_str}'.strip()
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{title_str}</div>'
        f'<div class="{value_class_attr_str}">{value_str}</div>'
        f'<div class="kpi-note">{note_str}</div>'
        '</div>'
    )


def _build_kpi_grid_html(
    summary_df: pd.DataFrame | None,
    column_name_str: str,
    regression_metadata_by_column_dict: dict[str, dict[str, object]] | None = None,
) -> str:
    """Build the KPI summary row shown beneath the report header."""
    kpi_spec_list = [
        ('Return (Ann.) [%]', 'Annualized Return', '252-day convention'),
        ('Volatility (Ann.) [%]', 'Volatility', 'Annualized sigma'),
        ('Sharpe Ratio', 'Sharpe Ratio', 'All days, risk-free rate = 0'),
        ('Max. Drawdown [%]', 'Max Drawdown', 'Peak to trough'),
        ('Beta', 'Beta', 'vs benchmark'),
    ]
    kpi_card_html_list: list[str] = []
    for metric_name_str, title_str, note_str in kpi_spec_list:
        metric_value_obj = _safe_summary_metric_value(summary_df, column_name_str, metric_name_str)
        kpi_card_html_list.append(
            _build_kpi_card_html(
                title_str=title_str,
                value_str=_format_kpi_value_str(metric_name_str, metric_value_obj),
                note_str=note_str,
                value_class_str=_kpi_value_class_str(metric_name_str, metric_value_obj),
            )
        )
    regression_metadata_dict = (regression_metadata_by_column_dict or {}).get(
        column_name_str,
        {},
    )
    alpha_value_obj = _safe_summary_metric_value(
        summary_df,
        column_name_str,
        'Alpha (Ann.) [%]',
    )
    alpha_t_value_obj = _safe_summary_metric_value(
        summary_df,
        column_name_str,
        'Alpha HAC t-stat',
    )
    if regression_metadata_dict.get('status_str') == 'ok' and alpha_value_obj is not None:
        benchmark_label_str = str(regression_metadata_dict.get('benchmark_label_str') or 'Benchmark')
        alpha_note_str = f'Zero-rate vs {benchmark_label_str}'
        if alpha_t_value_obj is not None:
            alpha_note_str += f' · HAC t={float(alpha_t_value_obj):.2f}'
        kpi_card_html_list.append(
            _build_kpi_card_html(
                title_str='Alpha (Ann.)',
                value_str=_fmt_signed_pct(alpha_value_obj),
                note_str=html.escape(alpha_note_str),
                value_class_str=_kpi_value_class_str('Alpha (Ann.) [%]', alpha_value_obj),
            )
        )
    return f'<div class="kpi-grid">{"".join(kpi_card_html_list)}</div>'


def _build_headline_delta_table_html(
        summary_df: 'pd.DataFrame | None',
        strategy_column_name_str: str,
) -> str:
    """Headline metrics as a strategy / benchmark / delta table.

    Every figure is read out of the same summary frame the Performance Summary
    plate prints, so the headline cannot drift from the table below it.

    Returns '' when there is no benchmark column to compare against, so the
    caller can fall back rather than render a table of em dashes.
    """
    if summary_df is None or len(summary_df) == 0:
        return ''
    if strategy_column_name_str not in summary_df.columns:
        return ''
    benchmark_column_name_list = [
        column_name_str
        for column_name_str in summary_df.columns
        if column_name_str != strategy_column_name_str
    ]
    if len(benchmark_column_name_list) == 0:
        return ''
    benchmark_column_name_str = benchmark_column_name_list[0]

    def metric_float(column_name_str: str, metric_name_str: str):
        value_obj = _safe_summary_metric_value(summary_df, column_name_str, metric_name_str)
        try:
            value_float = float(value_obj)
        except (TypeError, ValueError):
            return None
        return value_float if np.isfinite(value_float) else None

    metric_spec_list: list[dict[str, object]] = []

    for label_str, summary_metric_name_str, higher_is_better_bool in (
        ('CAGR (net)', 'Return (Ann.) [%]', True),
        ('Volatility', 'Volatility (Ann.) [%]', False),
    ):
        strategy_float = metric_float(strategy_column_name_str, summary_metric_name_str)
        benchmark_float = metric_float(benchmark_column_name_str, summary_metric_name_str)
        if strategy_float is None:
            continue
        metric_spec_list.append({
            'label_str': label_str,
            'value_float': strategy_float,
            'display_str': f'{strategy_float:.1f}%',
            'benchmark_float': benchmark_float,
            'benchmark_display_str': None if benchmark_float is None else f'{benchmark_float:.1f}%',
            'delta_display_str': (
                '' if benchmark_float is None else f'{strategy_float - benchmark_float:+.1f}pp'
            ),
            'higher_is_better_bool': higher_is_better_bool,
        })

    sharpe_float = metric_float(strategy_column_name_str, 'Sharpe Ratio')
    benchmark_sharpe_float = metric_float(benchmark_column_name_str, 'Sharpe Ratio')
    if sharpe_float is not None:
        metric_spec_list.append({
            'label_str': 'Sharpe ratio',
            'value_float': sharpe_float,
            'display_str': f'{sharpe_float:.2f}',
            'benchmark_float': benchmark_sharpe_float,
            'benchmark_display_str': (
                None if benchmark_sharpe_float is None else f'{benchmark_sharpe_float:.2f}'
            ),
            'delta_display_str': (
                '' if benchmark_sharpe_float is None
                else f'{sharpe_float - benchmark_sharpe_float:+.2f}'
            ),
            'higher_is_better_bool': True,
        })

    drawdown_float = metric_float(strategy_column_name_str, 'Max. Drawdown [%]')
    benchmark_drawdown_float = metric_float(benchmark_column_name_str, 'Max. Drawdown [%]')
    if drawdown_float is not None:
        # Drawdowns are stored negative. Compare depth, so a shallower drawdown
        # reads as the better one whichever sign convention the source used.
        metric_spec_list.append({
            'label_str': 'Max drawdown',
            'value_float': abs(drawdown_float),
            'display_str': f'{drawdown_float:.1f}%',
            'benchmark_float': None if benchmark_drawdown_float is None else abs(benchmark_drawdown_float),
            'benchmark_display_str': (
                None if benchmark_drawdown_float is None else f'{benchmark_drawdown_float:.1f}%'
            ),
            'delta_display_str': (
                '' if benchmark_drawdown_float is None
                else f'{abs(drawdown_float) - abs(benchmark_drawdown_float):+.1f}pp'
            ),
            'higher_is_better_bool': False,
            'is_adverse_bool': True,
        })

    # *** CRITICAL*** The summary stores Correlation per column as that column's
    # correlation to the strategy, so the strategy's own cell is 1.0 and the
    # figure we want -- how closely the strategy tracks the benchmark -- sits in
    # the benchmark column. Reading the strategy column here would print 1.00
    # for every strategy ever run and a delta of zero.
    correlation_float = metric_float(benchmark_column_name_str, 'Correlation')
    if correlation_float is not None:
        metric_spec_list.append({
            'label_str': 'Correlation',
            'value_float': correlation_float,
            'display_str': f'{correlation_float:.2f}',
            'benchmark_float': 1.0,
            'benchmark_display_str': '1.00',
            'delta_display_str': f'{correlation_float - 1.0:+.2f}',
            'higher_is_better_bool': False,
        })

    if len(metric_spec_list) == 0:
        return ''
    return build_metric_delta_table_html(metric_spec_list)


def _wrap_card_html(card_body_html_str: str, card_class_str: str = '') -> str:
    """Wrap a section for the active layout: a card, or a numbered plate.

    Under the spec layout every section becomes a plate and the CSS counter
    numbers it from its own heading, so the portfolio and crisis reports adopt
    the specimen sheet without restructuring how they assemble their sections.
    Empty bodies are dropped so the plate sequence has no gaps.
    """
    if not card_body_html_str or not card_body_html_str.strip():
        return ''
    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'spec':
        return f'<div class="plate">{card_body_html_str}</div>'

    card_class_attr_str = 'card'
    if card_class_str:
        card_class_attr_str += f' {card_class_str}'
    return f'<section class="{card_class_attr_str}">{card_body_html_str}</section>'


def _build_card_grid_html(card_html_list: list[str]) -> str:
    active_card_html_list = [card_html_str for card_html_str in card_html_list if card_html_str]
    if len(active_card_html_list) == 0:
        return ''
    # Plates own the full measure, so the spec layout stacks them instead of
    # pairing them side by side.
    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'spec':
        return ''.join(active_card_html_list)
    return f'<div class="card-grid">{"".join(active_card_html_list)}</div>'


def _build_report_header_html(
    report_kind_str: str,
    report_name_str: str,
    run_date_str: str,
    start_str: str,
    end_str: str,
    capital_base_obj,
    final_value_obj,
) -> str:
    return (
        '<header class="report-header">'
        f'<div class="report-eyebrow">{report_kind_str}</div>'
        f'<h1>{report_name_str}</h1>'
        '<div class="meta">'
        f'Run: {run_date_str} &nbsp;|&nbsp; Period: {start_str} &rarr; {end_str} &nbsp;|&nbsp; '
        f'Capital: {_fmt_dollar(capital_base_obj)} &rarr; {_fmt_dollar(final_value_obj)}'
        '</div>'
        '</header>'
    )


def _prepare_daily_return_distribution_dict(strategy) -> dict[str, object]:
    """
    Prepare realized daily return data for the strategy HTML histogram.

    The report uses the already-realized strategy return series:

    r_t = V_t / V_{t-1} - 1

    The first stored observation is excluded because it is a bootstrap
    placeholder produced by the reporting lifecycle, not a realized return.

    Compact statistics use:

    mu = (1 / N) * sum_{t=1}^{N} r_t

    sigma = sqrt((1 / (N - 1)) * sum_{t=1}^{N} (r_t - mu)^2)

    skew = (1 / N) * sum_{t=1}^{N} ((r_t - mu) / sigma)^3

    P(r_t < 0) = (1 / N) * sum_{t=1}^{N} 1[r_t < 0]
    """
    daily_return_ser = strategy.results['daily_returns'].astype(float)
    realized_daily_return_ser = daily_return_ser.iloc[1:].dropna()
    return_vec = realized_daily_return_ser.to_numpy(dtype=float)

    distribution_dict: dict[str, object] = {
        'daily_return_ser': realized_daily_return_ser,
        'return_vec': return_vec,
        'mean_return_float': np.nan,
        'std_return_float': np.nan,
        'skew_return_float': np.nan,
        'negative_rate_float': np.nan,
    }

    sample_size_int = int(return_vec.size)
    if sample_size_int == 0:
        return distribution_dict

    mean_return_float = float(return_vec.mean())
    negative_rate_float = float((return_vec < 0.0).mean())

    if sample_size_int >= 2:
        std_return_float = float(return_vec.std(ddof=1))
    else:
        std_return_float = np.nan

    if sample_size_int >= 2 and np.isfinite(std_return_float) and std_return_float > 0.0:
        standardized_return_vec = (return_vec - mean_return_float) / std_return_float
        skew_return_float = float(np.mean(standardized_return_vec ** 3))
    else:
        skew_return_float = np.nan

    distribution_dict['mean_return_float'] = mean_return_float
    distribution_dict['std_return_float'] = std_return_float
    distribution_dict['skew_return_float'] = skew_return_float
    distribution_dict['negative_rate_float'] = negative_rate_float
    return distribution_dict


def _daily_return_histogram_b64(distribution_dict: dict[str, object]) -> str | None:
    """
    Render a daily return histogram to base64.

    Histogram bins follow:

    M = max(|min(r_t)|, |max(r_t)|)

    bins in [-M, M] with 60 equal-width bins.
    """
    return_vec = np.asarray(distribution_dict['return_vec'], dtype=float)
    if return_vec.size < 2:
        return None

    half_range_float = float(np.max(np.abs(return_vec)))
    if not np.isfinite(half_range_float) or half_range_float <= 0.0:
        return None

    histogram_edge_count_int = _DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT + 1
    bin_edge_vec = np.linspace(-half_range_float, half_range_float, histogram_edge_count_int)
    mean_return_float = float(distribution_dict['mean_return_float'])

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        figure_obj, axis_obj = plt.subplots(figsize=(12, 4.2))
        axis_obj.hist(
            return_vec,
            bins=bin_edge_vec,
            color=SIGNATURE_PALETTE_DICT['series_cycle'][0],
            alpha=0.78,
            edgecolor=SIGNATURE_PALETTE_DICT['bar_edge'],
            linewidth=0.65,
        )
        axis_obj.axvline(
            0.0,
            color=SIGNATURE_PALETTE_DICT['zero_line'],
            linestyle='--',
            linewidth=1.0,
            label='Zero return',
        )
        axis_obj.axvline(
            mean_return_float,
            color=SIGNATURE_PALETTE_DICT['series_cycle'][1],
            linestyle='-',
            linewidth=1.1,
            label='Mean return',
        )
        axis_obj.set_title('Daily Return Distribution')
        axis_obj.set_xlabel('Daily Return')
        axis_obj.set_ylabel('Frequency')
        axis_obj.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=1))
        axis_obj.grid(True)
        axis_obj.legend(loc='upper right', fontsize=8)
        figure_obj.tight_layout()

        buffer_obj = io.BytesIO()
        figure_obj.savefig(buffer_obj, format='png', dpi=140, bbox_inches='tight')
        plt.close(figure_obj)
    buffer_obj.seek(0)
    return base64.b64encode(buffer_obj.read()).decode('ascii')


def _build_daily_return_distribution_html(strategy) -> str:
    distribution_dict = _prepare_daily_return_distribution_dict(strategy)
    histogram_b64 = _daily_return_histogram_b64(distribution_dict)

    if histogram_b64 is None:
        return (
            '<h2>Daily Return Distribution</h2>'
            '<p>Not enough realized daily return variation is available to render a meaningful histogram.</p>'
        )

    mean_return_float = float(distribution_dict['mean_return_float'])
    std_return_float = float(distribution_dict['std_return_float'])
    skew_return_float = float(distribution_dict['skew_return_float'])
    negative_rate_float = float(distribution_dict['negative_rate_float'])
    skew_str = _fmt_num(skew_return_float, 2) if np.isfinite(skew_return_float) else 'N/A'

    # Chart plus a one-line caption, not a glued stats table — the fuller
    # distribution stats live in the performance summary's tails section.
    return (
        '<h2>Daily Return Distribution</h2>'
        f'<div class="chart-wrap"><img src="data:image/png;base64,{histogram_b64}" '
        'alt="Daily Return Distribution"></div>'
        f'<p class="metric-context">Realized daily returns — mean {mean_return_float:.3%}, '
        f'sigma {std_return_float:.3%}, skew {skew_str}, {negative_rate_float:.1%} negative days. '
        'Fuller distribution and tail stats are in the performance summary.</p>'
    )


def _tail_mean_float(value_vec: np.ndarray, alpha_float: float) -> float:
    """
    Compute the lower-tail conditional mean:

    q_alpha = Quantile_alpha(x_i)

    tail_mean_alpha = mean(x_i | x_i <= q_alpha)
    """
    clean_value_vec = np.asarray(value_vec, dtype=float)
    clean_value_vec = clean_value_vec[np.isfinite(clean_value_vec)]
    if clean_value_vec.size == 0:
        return np.nan

    tail_cutoff_float = float(np.quantile(clean_value_vec, alpha_float))
    tail_mask_vec = clean_value_vec <= tail_cutoff_float
    return float(clean_value_vec[tail_mask_vec].mean())


def _prepare_trade_distribution_dict(trade_df: pd.DataFrame | None) -> dict[str, object]:
    """
    Prepare trade-level diagnostics for the HTML report.

    The primary trade-level return is:

    trade_return_i = profit_i / capital_i

    Win rate is:

    p_win = (1 / N) * sum_{i=1}^{N} 1[trade_return_i > 0]

    The lower loss tail is summarized with:

    q_alpha = Quantile_alpha(loss_return_i)

    loss_cvar_alpha = mean(loss_return_i | loss_return_i <= q_alpha)

    Trade duration in days is:

    duration_day_i = duration_i / 1 day
    """
    if trade_df is None:
        trade_df = pd.DataFrame()

    trade_df = trade_df.copy()
    distribution_dict: dict[str, object] = {
        'trade_df': trade_df,
        'trade_count_int': 0,
        'trade_return_vec': np.array([], dtype=float),
        'winning_trade_return_vec': np.array([], dtype=float),
        'losing_trade_return_vec': np.array([], dtype=float),
        'trade_duration_day_vec': np.array([], dtype=float),
        'winning_duration_day_vec': np.array([], dtype=float),
        'losing_duration_day_vec': np.array([], dtype=float),
        'win_rate_float': np.nan,
        'mean_trade_return_float': np.nan,
        'median_trade_return_float': np.nan,
        'median_winning_trade_return_float': np.nan,
        'median_losing_trade_return_float': np.nan,
        'worst_trade_return_float': np.nan,
        'loss_quantile_10_float': np.nan,
        'loss_quantile_5_float': np.nan,
        'loss_cvar_10_float': np.nan,
        'median_winning_duration_day_float': np.nan,
        'median_losing_duration_day_float': np.nan,
        'worst_loss_duration_day_float': np.nan,
    }

    if len(trade_df) == 0:
        return distribution_dict

    trade_return_vec = trade_df['return'].astype(float).to_numpy(dtype=float)
    trade_duration_ser = pd.to_timedelta(trade_df['duration'])
    trade_duration_day_vec = trade_duration_ser.dt.total_seconds().to_numpy(dtype=float) / 86400.0
    winning_mask_vec = trade_return_vec > 0.0
    losing_mask_vec = trade_return_vec <= 0.0
    winning_trade_return_vec = trade_return_vec[winning_mask_vec]
    losing_trade_return_vec = trade_return_vec[losing_mask_vec]
    winning_duration_day_vec = trade_duration_day_vec[winning_mask_vec]
    losing_duration_day_vec = trade_duration_day_vec[losing_mask_vec]

    distribution_dict['trade_count_int'] = int(trade_return_vec.size)
    distribution_dict['trade_return_vec'] = trade_return_vec
    distribution_dict['winning_trade_return_vec'] = winning_trade_return_vec
    distribution_dict['losing_trade_return_vec'] = losing_trade_return_vec
    distribution_dict['trade_duration_day_vec'] = trade_duration_day_vec
    distribution_dict['winning_duration_day_vec'] = winning_duration_day_vec
    distribution_dict['losing_duration_day_vec'] = losing_duration_day_vec
    distribution_dict['win_rate_float'] = float(winning_mask_vec.mean())
    distribution_dict['mean_trade_return_float'] = float(trade_return_vec.mean())
    distribution_dict['median_trade_return_float'] = float(np.median(trade_return_vec))
    distribution_dict['worst_trade_return_float'] = float(trade_return_vec.min())

    if winning_trade_return_vec.size > 0:
        distribution_dict['median_winning_trade_return_float'] = float(
            np.median(winning_trade_return_vec)
        )
    if losing_trade_return_vec.size > 0:
        distribution_dict['median_losing_trade_return_float'] = float(
            np.median(losing_trade_return_vec)
        )
        distribution_dict['loss_quantile_10_float'] = float(
            np.quantile(losing_trade_return_vec, 0.10)
        )
        distribution_dict['loss_quantile_5_float'] = float(
            np.quantile(losing_trade_return_vec, 0.05)
        )
        distribution_dict['loss_cvar_10_float'] = _tail_mean_float(
            losing_trade_return_vec,
            alpha_float=0.10,
        )

    if winning_duration_day_vec.size > 0:
        distribution_dict['median_winning_duration_day_float'] = float(
            np.median(winning_duration_day_vec)
        )
    if losing_duration_day_vec.size > 0:
        distribution_dict['median_losing_duration_day_float'] = float(
            np.median(losing_duration_day_vec)
        )
        worst_loss_idx_int = int(np.argmin(trade_return_vec))
        distribution_dict['worst_loss_duration_day_float'] = float(
            trade_duration_day_vec[worst_loss_idx_int]
        )

    return distribution_dict


def _trade_return_histogram_b64(distribution_dict: dict[str, object]) -> str | None:
    """
    Render winning and losing trade returns on a common histogram.

    The symmetric plotting range is:

    M = max(|min(trade_return_i)|, |max(trade_return_i)|)

    bins in [-M, M] with 60 equal-width bins.
    """
    trade_return_vec = np.asarray(distribution_dict['trade_return_vec'], dtype=float)
    if trade_return_vec.size < 2:
        return None

    half_range_float = float(np.max(np.abs(trade_return_vec)))
    if not np.isfinite(half_range_float) or half_range_float <= 0.0:
        return None

    histogram_edge_count_int = _TRADE_RETURN_HISTOGRAM_BIN_COUNT_INT + 1
    bin_edge_vec = np.linspace(-half_range_float, half_range_float, histogram_edge_count_int)
    winning_trade_return_vec = np.asarray(
        distribution_dict['winning_trade_return_vec'],
        dtype=float,
    )
    losing_trade_return_vec = np.asarray(
        distribution_dict['losing_trade_return_vec'],
        dtype=float,
    )
    mean_trade_return_float = float(distribution_dict['mean_trade_return_float'])

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        figure_obj, axis_obj = plt.subplots(figsize=(7.2, 4.0))
        if losing_trade_return_vec.size > 0:
            axis_obj.hist(
            losing_trade_return_vec,
            bins=bin_edge_vec,
            color=SIGNATURE_PALETTE_DICT['series_cycle'][0],
            alpha=0.72,
            edgecolor=SIGNATURE_PALETTE_DICT['bar_edge'],
            linewidth=0.60,
                label='Losing trades',
            )
        if winning_trade_return_vec.size > 0:
            axis_obj.hist(
            winning_trade_return_vec,
            bins=bin_edge_vec,
            color=SIGNATURE_PALETTE_DICT['series_cycle'][1],
            alpha=0.72,
            edgecolor=SIGNATURE_PALETTE_DICT['bar_edge'],
            linewidth=0.60,
                label='Winning trades',
            )

        axis_obj.axvline(
            0.0,
            color=SIGNATURE_PALETTE_DICT['zero_line'],
            linestyle='--',
            linewidth=1.0,
            label='Zero return',
        )
        axis_obj.axvline(
            mean_trade_return_float,
            color=SIGNATURE_PALETTE_DICT['series_cycle'][2],
            linestyle='-',
            linewidth=1.1,
            label='Mean trade return',
        )
        axis_obj.set_title('Winning vs Losing Trade Returns')
        axis_obj.set_xlabel('Trade Return')
        axis_obj.set_ylabel('Frequency')
        axis_obj.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=1))
        axis_obj.grid(True)
        axis_obj.legend(loc='upper right', fontsize=8)
        figure_obj.tight_layout()

        buffer_obj = io.BytesIO()
        figure_obj.savefig(buffer_obj, format='png', dpi=140, bbox_inches='tight')
        plt.close(figure_obj)
    buffer_obj.seek(0)
    return base64.b64encode(buffer_obj.read()).decode('ascii')


def _trade_return_duration_scatter_b64(distribution_dict: dict[str, object]) -> str | None:
    """
    Render trade return against holding duration:

    x_i = duration_day_i

    y_i = trade_return_i
    """
    trade_return_vec = np.asarray(distribution_dict['trade_return_vec'], dtype=float)
    trade_duration_day_vec = np.asarray(distribution_dict['trade_duration_day_vec'], dtype=float)
    if trade_return_vec.size < 2 or trade_duration_day_vec.size != trade_return_vec.size:
        return None

    winning_mask_vec = trade_return_vec > 0.0
    losing_mask_vec = trade_return_vec <= 0.0

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        figure_obj, axis_obj = plt.subplots(figsize=(7.2, 4.0))
        if losing_mask_vec.any():
            axis_obj.scatter(
                trade_duration_day_vec[losing_mask_vec],
                trade_return_vec[losing_mask_vec],
                color=SIGNATURE_PALETTE_DICT['series_cycle'][0],
                alpha=0.58,
                s=20,
                edgecolors='none',
                label='Losing trades',
            )
        if winning_mask_vec.any():
            axis_obj.scatter(
                trade_duration_day_vec[winning_mask_vec],
                trade_return_vec[winning_mask_vec],
                color=SIGNATURE_PALETTE_DICT['series_cycle'][1],
                alpha=0.58,
                s=20,
                edgecolors='none',
                label='Winning trades',
            )

        axis_obj.axhline(0.0, color=SIGNATURE_PALETTE_DICT['zero_line'], linestyle='--', linewidth=1.0)
        axis_obj.set_title('Trade Return vs Holding Duration')
        axis_obj.set_xlabel('Holding Duration [days]')
        axis_obj.set_ylabel('Trade Return')
        axis_obj.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=1))
        axis_obj.grid(True)
        axis_obj.legend(loc='upper right', fontsize=8)
        figure_obj.tight_layout()

        buffer_obj = io.BytesIO()
        figure_obj.savefig(buffer_obj, format='png', dpi=140, bbox_inches='tight')
        plt.close(figure_obj)
    buffer_obj.seek(0)
    return base64.b64encode(buffer_obj.read()).decode('ascii')


def _trade_distribution_summary_table_html(distribution_dict: dict[str, object]) -> str:
    trade_count_int = int(distribution_dict['trade_count_int'])
    if trade_count_int == 0:
        return '<p>No closed trades are available for trade-distribution diagnostics.</p>'

    def _pct_or_na(value_float: float) -> str:
        return f'{value_float:.2%}' if np.isfinite(value_float) else 'N/A'

    return (
        '<table class="stats-table"><thead><tr>'
        '<th>Trades</th><th>Win Rate</th><th>Mean Return</th><th>Median Return</th>'
        '<th>Median Winner</th><th>Median Loser</th>'
        '</tr></thead><tbody><tr>'
        f'<td>{trade_count_int}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["win_rate_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["mean_trade_return_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["median_trade_return_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["median_winning_trade_return_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["median_losing_trade_return_float"]))}</td>'
        '</tr></tbody></table>'
    )


def _loss_tail_summary_table_html(distribution_dict: dict[str, object]) -> str:
    if int(distribution_dict['trade_count_int']) == 0:
        return ''

    def _pct_or_na(value_float: float) -> str:
        return f'{value_float:.2%}' if np.isfinite(value_float) else 'N/A'

    def _day_or_na(value_float: float) -> str:
        return _fmt_num(value_float, 1) if np.isfinite(value_float) else 'N/A'

    return (
        '<table class="stats-table"><thead><tr>'
        '<th>Worst Trade</th><th>Loss q10</th><th>Loss q5</th><th>Loss CVaR 10%</th>'
        '<th>Median Winner Days</th><th>Median Loser Days</th><th>Worst Loss Days</th>'
        '</tr></thead><tbody><tr>'
        f'<td>{_pct_or_na(float(distribution_dict["worst_trade_return_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["loss_quantile_10_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["loss_quantile_5_float"]))}</td>'
        f'<td>{_pct_or_na(float(distribution_dict["loss_cvar_10_float"]))}</td>'
        f'<td>{_day_or_na(float(distribution_dict["median_winning_duration_day_float"]))}</td>'
        f'<td>{_day_or_na(float(distribution_dict["median_losing_duration_day_float"]))}</td>'
        f'<td>{_day_or_na(float(distribution_dict["worst_loss_duration_day_float"]))}</td>'
        '</tr></tbody></table>'
    )


def _build_trade_distribution_html(trade_df: pd.DataFrame | None, section_title_str: str) -> str:
    """
    Build trade-level diagnostics for winners, losers, and the loss tail.

    Core formulas exposed to the report are:

    trade_return_i = profit_i / capital_i

    loss_cvar_10% = mean(trade_return_i | trade_return_i <= q_10%(loss_return_i))
    """
    distribution_dict = _prepare_trade_distribution_dict(trade_df)
    if int(distribution_dict['trade_count_int']) == 0:
        return (
            f'<h2>{section_title_str}</h2>'
            '<p>No closed trades are available for trade-distribution diagnostics.</p>'
        )

    histogram_b64 = _trade_return_histogram_b64(distribution_dict)
    scatter_b64 = _trade_return_duration_scatter_b64(distribution_dict)
    chart_block_list: list[str] = []

    if histogram_b64 is not None:
        chart_block_list.append(
            '<div class="chart-panel">'
            f'<img src="data:image/png;base64,{histogram_b64}" alt="Winning vs Losing Trade Returns">'
            '</div>'
        )
    if scatter_b64 is not None:
        chart_block_list.append(
            '<div class="chart-panel">'
            f'<img src="data:image/png;base64,{scatter_b64}" alt="Trade Return vs Holding Duration">'
            '</div>'
        )

    chart_grid_html = (
        f'<div class="chart-grid">{"".join(chart_block_list)}</div>'
        if len(chart_block_list) > 0 else ''
    )

    # Charts plus a one-line caption, not glued stats tables — win rate and
    # payoff live in the Trade Statistics plate, tail stats in the summary.
    trade_count_int = int(distribution_dict['trade_count_int'])
    win_rate_float = float(distribution_dict.get('win_rate_float', float('nan')))
    win_rate_str = f'{win_rate_float:.1%}' if np.isfinite(win_rate_float) else 'N/A'
    return (
        f'<h2>{section_title_str}</h2>'
        f'{chart_grid_html}'
        f'<p class="metric-context">Each trade\'s return against its holding duration, winners '
        f'and losers separated — {trade_count_int:,} closed trades, {win_rate_str} winners. '
        'Trade-level statistics are in the Trade Statistics plate.</p>'
    )


def _fmt_cell(metric, val):
    """Format a summary metric cell based on the metric name suffix."""
    if val == '' or (isinstance(val, float) and np.isnan(val)):
        return ''
    if isinstance(val, pd.Timestamp):
        return str(val.date())
    if '[%]' in metric:
        return _fmt_pct(val)
    if '[$]' in metric:
        return _fmt_dollar(val)
    if '[days]' in metric:
        return _fmt_num(val, 0)
    return _fmt_num(val, 2)


def _summary_metric_cell_class_str(metric_name_str: str, metric_value_obj) -> str:
    return ''


def _signed_value_class_str(value_obj) -> str:
    """Map a numeric sign to a shared positive/negative table class."""
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ''

    if np.isnan(value_float):
        return ''
    if value_float > 0.0:
        return 'pos'
    if value_float < 0.0:
        return 'neg'
    return ''


def _format_summary(df: pd.DataFrame) -> str:
    """Render strategy summary DataFrame as an HTML table."""
    headers = '<th>Metric</th>' + ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = []
    for metric in df.index:
        metric_name_str = str(metric)
        metric_label_html_str = _metric_label_html(metric_name_str)
        cells = [f'<td class="metric">{metric_label_html_str}</td>']
        for col in df.columns:
            metric_value_obj = df.loc[metric, col]
            cell_class_str = _summary_metric_cell_class_str(metric, metric_value_obj)
            class_attr_str = f' class="{cell_class_str}"' if cell_class_str else ''
            cells.append(f'<td{class_attr_str}>{_fmt_cell(metric, metric_value_obj)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _metric_label_html(metric_name_str: str) -> str:
    metric_help_text_str = METRIC_HELP_TEXT_DICT.get(metric_name_str)
    metric_label_html_str = html.escape(metric_name_str)
    if metric_help_text_str:
        escaped_help_text_str = html.escape(metric_help_text_str, quote=True)
        metric_label_html_str += (
            f' <button type="button" class="metric-help" '
            f'aria-label="{html.escape(metric_name_str, quote=True)}: {escaped_help_text_str}" '
            f'aria-expanded="false" data-help="{escaped_help_text_str}">i</button>'
        )
    return metric_label_html_str


def _format_benchmark_regression_summary(
    summary_df: pd.DataFrame,
    regression_metadata_by_column_dict: dict[str, dict[str, object]] | None,
) -> str:
    regression_metadata_by_column_dict = regression_metadata_by_column_dict or {}
    regression_column_name_list = [
        column_name_str
        for column_name_str in summary_df.columns
        if str(column_name_str) in regression_metadata_by_column_dict
    ]
    if len(regression_column_name_list) == 0:
        regression_column_name_list = list(summary_df.columns)
    header_html_str = '<th>Metric</th>'
    for column_name_str in regression_column_name_list:
        metadata_dict = regression_metadata_by_column_dict.get(str(column_name_str), {})
        benchmark_label_str = metadata_dict.get('benchmark_label_str') or 'No valid benchmark'
        benchmark_adjustment_str = metadata_dict.get('benchmark_adjustment_str')
        benchmark_context_str = (
            f'{benchmark_label_str} · {benchmark_adjustment_str}'
            if benchmark_adjustment_str
            and str(benchmark_adjustment_str) not in str(benchmark_label_str)
            else str(benchmark_label_str)
        )
        if metadata_dict.get('status_str') == 'ok':
            context_str = (
                f'{benchmark_context_str} · N={metadata_dict.get("observation_count_int", 0)} '
                f'· HAC L={metadata_dict.get("hac_max_lag_int", 0)}'
            )
        else:
            reason_str = str(metadata_dict.get('reason_str') or 'unavailable').replace('_', ' ')
            context_str = f'{benchmark_context_str} · N/A: {reason_str}'
        header_html_str += (
            f'<th>{html.escape(str(column_name_str))}'
            f'<div class="metric-context">{html.escape(context_str)}</div></th>'
        )

    regression_row_spec_tuple = (
        ('Beta', 'Beta'),
        ('Alpha (Ann.) / HAC t-stat', 'alpha_combined'),
        ('R²', 'R²'),
    )
    row_html_list: list[str] = []
    for display_metric_name_str, source_metric_name_str in regression_row_spec_tuple:
        cell_html_list = [
            f'<td class="metric">{_metric_label_html(display_metric_name_str)}</td>'
        ]
        for column_name_str in regression_column_name_list:
            if source_metric_name_str == 'alpha_combined':
                alpha_value_obj = (
                    summary_df.loc['Alpha (Ann.) [%]', column_name_str]
                    if 'Alpha (Ann.) [%]' in summary_df.index
                    else np.nan
                )
                alpha_t_value_obj = (
                    summary_df.loc['Alpha HAC t-stat', column_name_str]
                    if 'Alpha HAC t-stat' in summary_df.index
                    else np.nan
                )
                if pd.notna(alpha_value_obj):
                    alpha_t_value_str = (
                        _fmt_num(alpha_t_value_obj, 2)
                        if pd.notna(alpha_t_value_obj)
                        else 'N/A'
                    )
                    value_html_str = f'{_fmt_signed_pct(alpha_value_obj)} / {alpha_t_value_str}'
                else:
                    value_html_str = 'N/A'
            else:
                value_obj = (
                    summary_df.loc[source_metric_name_str, column_name_str]
                    if source_metric_name_str in summary_df.index
                    else np.nan
                )
                value_html_str = _fmt_num(value_obj, 2) if pd.notna(value_obj) else 'N/A'
            cell_html_list.append(f'<td>{value_html_str}</td>')
        row_html_list.append('<tr>' + ''.join(cell_html_list) + '</tr>')

    return (
        '<div class="regression-model-note">Zero-Rate Market Regression</div>'
        f'<table><thead><tr>{header_html_str}</tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table>'
    )


def _display_metric_dict_for_value_ser(value_ser: pd.Series) -> dict[str, float]:
    """Distribution and consistency stats derived from a daily equity curve.

    These are presentation-only: computed from the already-realized wealth
    series for the report, not part of the core metrics contract. Percent
    metrics are returned already in percent units, matching the summary
    formatter's convention. VaR/CVaR are historical (non-parametric): the 5th
    percentile of daily returns and the mean of days at or below it.
    """
    daily_return_ser = value_ser.pct_change(fill_method=None).dropna()
    monthly_return_ser = value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    if len(daily_return_ser) < 2 or len(monthly_return_ser) < 2:
        return {}

    var_95_float = float(np.percentile(daily_return_ser, 5.0))
    cvar_tail_ser = daily_return_ser[daily_return_ser <= var_95_float]
    cvar_95_float = float(cvar_tail_ser.mean()) if len(cvar_tail_ser) else var_95_float

    display_metric_dict = {
        'Volatility (Monthly) [%]': float(monthly_return_ser.std()) * 100.0,
        'Positive Months [%]': float((monthly_return_ser > 0.0).mean()) * 100.0,
        'Skewness (Daily)': float(daily_return_ser.skew()),
        'Skewness (Monthly)': float(monthly_return_ser.skew()),
        'Excess Kurtosis (Daily)': float(daily_return_ser.kurtosis()),
        'Worst Day [%]': float(daily_return_ser.min()) * 100.0,
        'Worst Month [%]': float(monthly_return_ser.min()) * 100.0,
        'VaR 95% (Daily) [%]': var_95_float * 100.0,
        'CVaR 95% (Daily) [%]': cvar_95_float * 100.0,
    }

    # Sortino: like Sharpe, but the denominator counts only downside deviation,
    # so upside volatility is not penalised.
    #
    #     downside_deviation = sqrt(mean(min(r_t, 0)^2)) * sqrt(252)
    #     sortino = annualised_return / downside_deviation
    #
    # *** CRITICAL*** The downside deviation averages over *all* observations,
    # not only the losing ones. Dividing by the loss count instead would inflate
    # the ratio for strategies that lose rarely but severely.
    year_count_float = len(daily_return_ser) / _TRADING_DAY_PER_YEAR_FLOAT
    growth_float = float(value_ser.iloc[-1] / value_ser.iloc[0])
    downside_deviation_float = float(
        np.sqrt(np.mean(np.square(np.minimum(daily_return_ser.to_numpy(), 0.0))))
        * np.sqrt(_TRADING_DAY_PER_YEAR_FLOAT)
    )
    if year_count_float > 0.0 and growth_float > 0.0 and downside_deviation_float > 0.0:
        annualised_return_float = growth_float ** (1.0 / year_count_float) - 1.0
        display_metric_dict['Sortino Ratio'] = annualised_return_float / downside_deviation_float

    # Ulcer Index: RMS of the drawdown path, so depth *and* time underwater are
    # both punished — a single number for how painful the ride was.
    #
    #     ulcer = sqrt(mean(drawdown_t^2)),  drawdown_t = V_t / max(V_1..V_t) - 1
    drawdown_pct_vec = (value_ser / value_ser.cummax() - 1.0).to_numpy() * 100.0
    display_metric_dict['Ulcer Index'] = float(np.sqrt(np.mean(np.square(drawdown_pct_vec))))
    return display_metric_dict


def _strategy_unconditional_beta_float(strategy) -> float | None:
    """Full-sample beta of strategy daily returns to the benchmark's.

        beta = cov(r_strategy, r_benchmark) / var(r_benchmark)

    Returned only when a benchmark series is present with enough overlap; this
    is the raw market beta shown in the headline, distinct from the alpha/beta
    regression (which additionally requires a tradeable benchmark).
    """
    strategy_value_ser, benchmark_value_ser, _label_str = _strategy_benchmark_value_pair(strategy)
    if strategy_value_ser is None or benchmark_value_ser is None:
        return None
    aligned_df = pd.concat(
        {
            'strategy': strategy_value_ser.pct_change(fill_method=None),
            'benchmark': benchmark_value_ser.pct_change(fill_method=None),
        },
        axis=1,
    ).dropna()
    if len(aligned_df) < 3 or float(aligned_df['benchmark'].var()) == 0.0:
        return None
    return float(aligned_df['strategy'].cov(aligned_df['benchmark']) / aligned_df['benchmark'].var())


def _augment_summary_display_metrics(strategy, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary copy with distribution/consistency rows added per column.

    Each column is marked to its own equity curve: the strategy column to
    ``total_value``, a benchmark column to its stored series. Columns with no
    matching series (or too little data) are simply left without the extra rows.
    """
    augmented_summary_df = summary_df.copy()
    primary_value_ser, _b, _l = _strategy_benchmark_value_pair(strategy)
    for column_name_str in augmented_summary_df.columns:
        # The first summary column is the entity itself, whatever it is called
        # ('Strategy' for a strategy, 'Portfolio' for a book); the rest name
        # benchmark series stored alongside it.
        if column_name_str == augmented_summary_df.columns[0] and primary_value_ser is not None:
            value_ser = primary_value_ser.dropna()
        elif str(column_name_str) in strategy.results.columns:
            value_ser = strategy.results[str(column_name_str)].astype(float).dropna()
        else:
            continue
        for metric_name_str, metric_value_float in _display_metric_dict_for_value_ser(value_ser).items():
            augmented_summary_df.loc[metric_name_str, column_name_str] = metric_value_float

    # Full-sample beta: the strategy against its benchmark, and a benchmark
    # column against itself is 1.0 by definition.
    # Drawdown frequency per month, derived from the annual count already in
    # the summary so both rates always agree.
    for column_name_str in augmented_summary_df.columns:
        annual_drawdown_count_obj = _safe_summary_metric_value(
            augmented_summary_df, column_name_str, '# Drawdowns / year'
        )
        if annual_drawdown_count_obj is not None and np.isfinite(float(annual_drawdown_count_obj)):
            augmented_summary_df.loc['# Drawdowns / month', column_name_str] = (
                float(annual_drawdown_count_obj) / 12.0
            )

    strategy_beta_float = _strategy_unconditional_beta_float(strategy)
    if strategy_beta_float is not None and 'Strategy' in augmented_summary_df.columns:
        augmented_summary_df.loc['Beta', 'Strategy'] = strategy_beta_float
        for column_name_str in augmented_summary_df.columns:
            if str(column_name_str) != 'Strategy':
                augmented_summary_df.loc['Beta', column_name_str] = 1.0
    return augmented_summary_df


def _trade_statistics_section_html(entity, intro_html_str: str = '') -> str:
    """Trade statistics as a summary sub-section rather than its own plate.

    These are summary numbers of the same kind as the sections around them, so
    they belong in the one place a reader goes for summary numbers.
    """
    summary_trades_df = getattr(entity, 'summary_trades', None)
    if summary_trades_df is None or len(summary_trades_df) == 0:
        return ''
    return (
        '<div class="summary-section"><h3>Trade Statistics</h3>'
        f'{intro_html_str}'
        f'<div class="scroll">{_format_summary_trades(_curate_summary_trades(summary_trades_df))}</div>'
        f'{_degenerate_trade_note_html(entity)}'
        '</div>'
    )


def _format_performance_summary(
    summary_df: pd.DataFrame,
    regression_metadata_by_column_dict: dict[str, dict[str, object]] | None = None,
    extra_section_html_str: str = '',
) -> str:
    """Render a flat performance summary as consistent named sections."""
    section_html_list: list[str] = []
    assigned_metric_name_set: set[str] = set()

    for section_title_str, metric_name_tuple, collapsed_bool in _PERFORMANCE_SUMMARY_SECTION_TUPLE:
        if section_title_str == 'Benchmark Regression':
            assigned_metric_name_set.update(metric_name_tuple)
            regression_table_html_str = _format_benchmark_regression_summary(
                summary_df,
                regression_metadata_by_column_dict,
            )
            section_html_list.append(
                '<div class="summary-section">'
                '<h3>Benchmark Regression</h3>'
                f'<div class="scroll">{regression_table_html_str}</div>'
                '</div>'
            )
            continue
        present_metric_name_list = [
            metric_name_str
            for metric_name_str in metric_name_tuple
            if metric_name_str in summary_df.index
        ]
        assigned_metric_name_set.update(present_metric_name_list)
        if len(present_metric_name_list) == 0:
            continue

        section_table_html_str = _format_summary(summary_df.loc[present_metric_name_list])
        if collapsed_bool:
            section_html_list.append(
                '<details class="summary-details">'
                f'<summary>{html.escape(section_title_str)}</summary>'
                f'<div class="scroll">{section_table_html_str}</div>'
                '</details>'
            )
        else:
            section_html_list.append(
                '<div class="summary-section">'
                f'<h3>{html.escape(section_title_str)}</h3>'
                f'<div class="scroll">{section_table_html_str}</div>'
                '</div>'
            )

    unassigned_metric_name_list = [
        str(metric_name_str)
        for metric_name_str in summary_df.index
        if str(metric_name_str) not in assigned_metric_name_set
        and str(metric_name_str) not in _PERFORMANCE_SUMMARY_HIDDEN_METRIC_SET
    ]
    if len(unassigned_metric_name_list) > 0:
        section_html_list.append(
            '<div class="summary-section">'
            '<h3>Other Metrics</h3>'
            f'<div class="scroll">{_format_summary(summary_df.loc[unassigned_metric_name_list])}</div>'
            '</div>'
        )

    if extra_section_html_str:
        # Trade activity is the first thing read about a book, so it leads the
        # summary rather than sitting behind seven tables of risk statistics.
        section_html_list.insert(0, extra_section_html_str)
    return '<div class="summary-section-stack">' + ''.join(section_html_list) + '</div>'


def _format_portfolio_summary(
    df: pd.DataFrame,
    regression_metadata_by_column_dict: dict[str, dict[str, object]] | None = None,
) -> str:
    """Backward-compatible wrapper for callers and focused renderer tests."""
    return _format_performance_summary(df, regression_metadata_by_column_dict)


# The report shows only the trade statistics that answer a distinct question:
# how many trades, how often, how often right, and what each one costs. The
# rest (duration spreads, profit factor, expectancy, payoff, CPC) are either
# derivable from these or too specialised for the headline table.
#
# Note this also drops the rows currently poisoned by degenerate zero-capital
# trades (see _TRADE_RETURN_DEGENERATE_NOTE_STR); trimming is not the fix for
# that, it just keeps the visible table honest until the ledger is corrected.
_TRADE_STATISTIC_KEEP_TUPLE = (
    '# Trades',
    '# Trades / week',
    'Win Rate [%]',
    'Avg. Commission / trade [$]',
)


def _curate_summary_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    keep_metric_list = [
        metric_name for metric_name in _TRADE_STATISTIC_KEEP_TUPLE
        if metric_name in df.index
    ]
    return df.loc[keep_metric_list]


def _format_summary_trades(df: pd.DataFrame) -> str:
    """Render trade summary DataFrame as an HTML table."""
    headers = '<th>Metric</th>' + ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = []
    for metric in df.index:
        cells = [f'<td class="metric">{metric}</td>']
        for col in df.columns:
            val = df.loc[metric, col]
            if val == '':
                cells.append('<td></td>')
            elif '[%]' in metric:
                cells.append(f'<td>{_fmt_pct(val)}</td>')
            elif '[$]' in metric:
                cells.append(f'<td>{_fmt_dollar(val)}</td>')
            elif '[days]' in metric:
                cells.append(f'<td>{_fmt_num(val, 0)}</td>')
            else:
                cells.append(f'<td>{_fmt_num(val, 2)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _format_trades(df: pd.DataFrame) -> str:
    """Render closed trades DataFrame as an HTML table."""
    if df is None or len(df) == 0:
        return '<p>No closed trades.</p>'
    headers = '<th>trade_id</th>' + ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = []
    for trade_id, row in df.iterrows():
        cells = [f'<td>{trade_id}</td>']
        for col in df.columns:
            val = row[col]
            if col in ('start', 'end'):
                cells.append(f'<td>{str(val.date()) if hasattr(val, "date") else val}</td>')
            elif col == 'capital':
                cells.append(f'<td>{_fmt_dollar(val)}</td>')
            elif col == 'profit':
                cell_class_str = _signed_value_class_str(val)
                class_attr_str = f' class="{cell_class_str}"' if cell_class_str else ''
                cells.append(f'<td{class_attr_str}>{_fmt_dollar(val)}</td>')
            elif col == 'return':
                cell_class_str = _signed_value_class_str(val)
                class_attr_str = f' class="{cell_class_str}"' if cell_class_str else ''
                cells.append(f'<td{class_attr_str}>{_fmt_pct(val * 100)}</td>')
            else:
                cells.append(f'<td>{val}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _format_open_trades(df: pd.DataFrame) -> str:
    """Render marked open trades as an HTML table."""
    if df is None or len(df) == 0:
        return '<p>No open trades.</p>'

    headers = '<th>trade_id</th>' + ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = []
    for trade_id, row in df.iterrows():
        cells = [f'<td>{trade_id}</td>']
        for col in df.columns:
            val = row[col]
            if col in ('start', 'mark'):
                cells.append(f'<td>{str(val.date()) if hasattr(val, "date") else val}</td>')
            elif col in ('capital', 'market_value', 'commission'):
                if pd.isna(val):
                    cells.append('<td>N/A</td>')
                else:
                    cells.append(f'<td>{_fmt_dollar(val)}</td>')
            elif col == 'unrealized_pnl':
                if pd.isna(val):
                    cells.append('<td>N/A</td>')
                else:
                    cell_class_str = _signed_value_class_str(val)
                    class_attr_str = f' class="{cell_class_str}"' if cell_class_str else ''
                    cells.append(f'<td{class_attr_str}>{_fmt_dollar(val)}</td>')
            elif col == 'return':
                if pd.isna(val):
                    cells.append('<td>N/A</td>')
                else:
                    cell_class_str = _signed_value_class_str(val)
                    class_attr_str = f' class="{cell_class_str}"' if cell_class_str else ''
                    cells.append(f'<td{class_attr_str}>{_fmt_pct(val * 100)}</td>')
            else:
                cells.append(f'<td>{val}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _format_transactions(df: pd.DataFrame) -> str:
    """Render transactions DataFrame as an HTML table."""
    if df is None or len(df) == 0:
        return '<p>No transactions.</p>'
    headers = ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col == 'price':
                cells.append(f'<td>{_fmt_dollar(val)}</td>')
            elif col == 'total_value':
                cells.append(f'<td>{_fmt_dollar(val)}</td>')
            elif col == 'bar':
                cells.append(f'<td>{str(val.date()) if hasattr(val, "date") else val}</td>')
            else:
                cells.append(f'<td>{val}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ monthly returns heatmap ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

def _ret_color(val) -> str:
    """Map a return ratio to a muted signature heatmap color."""
    try:
        value_float = float(val)
    except (TypeError, ValueError):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )
    if np.isnan(value_float):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )

    intensity_float = min(abs(value_float) / 0.30, 1.0)
    fill_weight_float = 0.12 + 0.45 * intensity_float

    if value_float >= 0.0:
        background_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['profit'],
            fill_weight_float,
        )
        font_color_str = (
            SIGNATURE_PALETTE_DICT['profit_dark']
            if intensity_float < 0.65 else SIGNATURE_PALETTE_DICT['page']
        )
    else:
        background_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['loss'],
            fill_weight_float,
        )
        font_color_str = (
            SIGNATURE_PALETTE_DICT['loss_dark']
            if intensity_float < 0.65 else SIGNATURE_PALETTE_DICT['page']
        )

    return f'background-color: {background_color_str}; color: {font_color_str};'


def _monthly_benchmark_label(benchmark_name_str: str) -> str:
    """Return a compact display label for benchmark metric columns."""
    return str(benchmark_name_str).lstrip('$')


def _strategy_monthly_benchmark_metric_bundle(strategy) -> tuple[pd.DataFrame | None, str | None]:
    """Build annual benchmark metrics from the stored benchmark equity curve.

    The comparison block uses the benchmark wealth series already stored in
    `strategy.results`, preserving the annual formulas

        R_ann_y = V_year_end_y / V_year_end_(y-1) - 1

    and

        DD_t = V_t / max(V_1, ..., V_t) - 1.
    """
    benchmark_name_list = list(getattr(strategy, '_benchmarks', []))
    if len(benchmark_name_list) == 0:
        return None, None

    benchmark_name_str = benchmark_name_list[0]
    if benchmark_name_str not in strategy.results.columns:
        return None, None

    benchmark_value_ser = strategy.results[benchmark_name_str].astype(float).copy()
    benchmark_monthly_metric_df = generate_monthly_returns(
        benchmark_value_ser,
        add_sharpe_ratios=True,
        add_max_drawdowns=True,
    )
    return benchmark_monthly_metric_df, _monthly_benchmark_label(benchmark_name_str)


def _strategy_benchmark_value_pair(entity):
    """Return (value_ser, benchmark_value_ser, label) for a strategy or portfolio.

    Both series are stored daily equity curves, so the shared plates measure
    the same wealth the rest of the report does — no re-simulation.

    A strategy keeps its benchmark as a column inside ``results``; a portfolio
    carries an explicit PM benchmark series instead. Resolving both here is
    what lets one set of plate builders serve either entity.
    """
    results_df = getattr(entity, 'results', None)
    if results_df is None or 'total_value' not in results_df.columns:
        return None, None, None
    value_ser = results_df['total_value'].astype(float)

    portfolio_benchmark_value_ser = getattr(entity, 'regression_benchmark_value_ser', None)
    if portfolio_benchmark_value_ser is not None and len(portfolio_benchmark_value_ser) > 0:
        benchmark_label_str = (
            getattr(entity, 'regression_benchmark_label_str', None) or 'Benchmark'
        )
        aligned_benchmark_ser = (
            portfolio_benchmark_value_ser.astype(float).reindex(results_df.index).ffill()
        )
        return value_ser, aligned_benchmark_ser, benchmark_label_str

    benchmark_name_list = list(getattr(entity, '_benchmarks', []))
    if len(benchmark_name_list) > 0 and benchmark_name_list[0] in results_df.columns:
        benchmark_name_str = benchmark_name_list[0]
        return (
            value_ser,
            results_df[benchmark_name_str].astype(float),
            _monthly_benchmark_label(benchmark_name_str),
        )
    return value_ser, None, None


def _build_composition_plate_html(strategy) -> str:
    """Composition of the strategy's own book, in the view its shape calls for.

    Detected from the realized weights: a persistent-sleeve book (few distinct
    names) stacks by weight; a rotating book (many names, e.g. a slot momentum
    strategy) shows deployed capital, slot occupancy and holding periods, since
    a per-name weight chart there encodes nothing. Empty without weight history.
    """
    realized_weight_df = getattr(strategy, 'realized_weight_df', None)
    if realized_weight_df is None or len(realized_weight_df) == 0 or realized_weight_df.shape[1] == 0:
        return ''
    position_weight_df = realized_weight_df.drop(
        columns=['Cash'],
        errors='ignore',
    )
    try:
        composition_uri_str, resolved_mode_str = render_composition_data_uri_str(
            position_weight_df
        )
    except ValueError:
        return ''

    distinct_name_count_int = int(
        position_weight_df.fillna(0.0).abs().gt(0.0).any(axis=0).sum()
    )
    caption_str = {
        'sleeve': (
            f'{distinct_name_count_int} distinct names ever held — few enough that weights '
            'by name are the story, so composition stacks by weight.'
        ),
        'rotation': (
            f'{distinct_name_count_int} distinct names ever held — a per-name weight chart '
            'would be unreadable, so composition shows deployed capital, slot occupancy and '
            'holding periods instead.'
        ),
    }[resolved_mode_str]

    # One plate answers the whole allocation question: the full history, what
    # is held today, and how the last three rebalances landed against target.
    # These used to be split across Composition and a separate Portfolio
    # Weights plate that re-drew the same stack over two arbitrary windows.
    section_html_list = ['<h2>Composition</h2>']

    weight_stack_b64_str = _composition_weights_chart_b64(realized_weight_df)
    if weight_stack_b64_str is not None:
        section_html_list.append(
            '<h3>Portfolio weights</h3>'
            '<div class="chart-wrap">'
            f'<img src="data:image/png;base64,{weight_stack_b64_str}" alt="Portfolio weights">'
            '</div>'
        )
    else:
        section_html_list.append(
            f'<div class="chart-wrap"><img src="{composition_uri_str}" alt="Composition"></div>'
        )
    section_html_list.append(f'<p class="metric-context">{caption_str}</p>')
    section_html_list.append(_current_composition_html(strategy))
    section_html_list.append(_recent_taa_weight_comparison_html(strategy))
    return '\n'.join(part_str for part_str in section_html_list if part_str)


def _signature_within_year_stat_df(total_value_ser: pd.Series) -> pd.DataFrame:
    """Per-calendar-year return, volatility, max drawdown and Sharpe.

    *** CRITICAL*** Each year's statistics are computed inside that year only,
    rebased to its first bar. Carrying a running peak across the year boundary
    would report a prior year's damage against this year's row.
    """
    stat_row_list = []
    for year_int, year_value_ser in total_value_ser.groupby(total_value_ser.index.year):
        year_return_ser = year_value_ser.pct_change(fill_method=None).dropna()
        if len(year_return_ser) < 2:
            continue
        year_growth_float = float(year_value_ser.iloc[-1] / year_value_ser.iloc[0])
        year_volatility_float = float(
            year_return_ser.std() * np.sqrt(_TRADING_DAY_PER_YEAR_FLOAT)
        )
        year_drawdown_float = float((year_value_ser / year_value_ser.cummax() - 1.0).min())
        stat_row_list.append({
            'year_int': int(year_int),
            'return_float': year_growth_float - 1.0,
            'volatility_float': year_volatility_float,
            'max_drawdown_float': year_drawdown_float,
            'sharpe_float': (
                (year_growth_float - 1.0) / year_volatility_float
                if year_volatility_float > 0.0 else np.nan
            ),
        })
    return pd.DataFrame(stat_row_list).set_index('year_int')


_SIGNATURE_HEATMAP_MAX_BLEND_FLOAT = 0.55


def _signature_monthly_return_range_tuple(total_value_ser: pd.Series) -> tuple[float, float]:
    monthly_return_ser = (
        total_value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    )
    if len(monthly_return_ser) == 0:
        return 0.0, 1.0
    return float(monthly_return_ser.min()), float(monthly_return_ser.max())


def _signature_heatmap_background_str(
    value_float: float,
    low_float: float,
    high_float: float,
) -> str:
    """Map a return onto a diverging gain/loss ramp.

    *** CRITICAL*** Hue carries the direction and intensity carries the
    magnitude — gains tint toward the palette's profit tone, losses toward its
    loss tone, both fading to bare paper at zero. Splitting the two roles is
    what makes this readable: a single-hue ramp had to choose between showing
    direction and showing size, and either choice made one of them a guess.

    Each side is scaled against its own extreme, so the worst loss and the best
    gain both reach full intensity regardless of which is larger in the sample.
    """
    if not np.isfinite(value_float):
        return str(SIGNATURE_PALETTE_DICT['panel'])

    if value_float >= 0.0:
        tone_color_str = str(SIGNATURE_PALETTE_DICT['profit'])
        extreme_float = max(high_float, 0.0)
    else:
        tone_color_str = str(SIGNATURE_PALETTE_DICT['loss'])
        extreme_float = abs(min(low_float, 0.0))

    if not np.isfinite(extreme_float) or extreme_float <= 0.0:
        return str(SIGNATURE_PALETTE_DICT['panel'])

    intensity_float = min(abs(value_float) / extreme_float, 1.0)
    return blend_hex_color_str(
        str(SIGNATURE_PALETTE_DICT['panel']),
        tone_color_str,
        intensity_float * _SIGNATURE_HEATMAP_MAX_BLEND_FLOAT,
    )


def _signature_monthly_table_html(
    total_value_ser: pd.Series,
    monthly_range_tuple: tuple[float, float],
) -> str:
    """One monthly grid, newest year on top, shaded light-to-dark by return.

    Every shaded cell in the table — months and the Year column alike — uses
    the same rule: darker means a higher return. The month cells share a scale
    with the benchmark table so the same return is the same shade in both; the
    Year column gets its own scale because annual moves dwarf monthly ones and
    would otherwise saturate.
    """
    monthly_return_ser = (
        total_value_ser.resample('ME').last().pct_change(fill_method=None).dropna()
    )
    monthly_return_df = pd.DataFrame({
        'year_int': monthly_return_ser.index.year,
        'month_int': monthly_return_ser.index.month,
        'return_float': monthly_return_ser.to_numpy(),
    }).pivot(index='year_int', columns='month_int', values='return_float')
    yearly_stat_df = _signature_within_year_stat_df(total_value_ser)
    monthly_low_float, monthly_high_float = monthly_range_tuple
    year_low_float = (
        float(yearly_stat_df['return_float'].min()) if len(yearly_stat_df) else 0.0
    )
    year_high_float = (
        float(yearly_stat_df['return_float'].max()) if len(yearly_stat_df) else 1.0
    )

    row_html_list = []
    # Newest year first: the current year is what the reader checks.
    for year_int, month_return_ser in monthly_return_df.iloc[::-1].iterrows():
        cell_html_list = []
        for month_int in range(1, 13):
            return_float = month_return_ser.get(month_int, np.nan)
            if pd.isna(return_float):
                cell_html_list.append('<td></td>')
                continue
            cell_background_str = _signature_heatmap_background_str(
                float(return_float), monthly_low_float, monthly_high_float
            )
            cell_html_list.append(
                f'<td style="background:{cell_background_str}">{return_float * 100:.1f}</td>'
            )
        if int(year_int) not in yearly_stat_df.index:
            continue
        yearly_stat_ser = yearly_stat_df.loc[int(year_int)]
        year_return_float = float(yearly_stat_ser['return_float'])
        year_background_str = _signature_heatmap_background_str(
            year_return_float, year_low_float, year_high_float
        )
        row_html_list.append(
            f'<tr><td class="metric">{year_int}</td>'
            + ''.join(cell_html_list)
            + f'<td class="divider-left" style="background:{year_background_str}">'
              f'{year_return_float * 100:.1f}</td>'
            + f'<td>{yearly_stat_ser["volatility_float"] * 100:.1f}</td>'
            + f'<td>{yearly_stat_ser["max_drawdown_float"] * 100:.1f}</td>'
            + f'<td>{yearly_stat_ser["sharpe_float"]:.2f}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="heatmap">'
        '<thead><tr><th>Year</th>'
        + ''.join(f'<th>{month_str}</th>' for month_str in _MONTH_NAMES)
        + '<th class="divider-left">Year</th><th>Vol</th><th>Max DD</th><th>Sharpe</th>'
        + '</tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
    )


def _build_signature_monthly_returns_html(strategy) -> str:
    """Strategy monthly grid over benchmark grid, on one shared return scale."""
    strategy_value_ser, benchmark_value_ser, benchmark_label_str = (
        _strategy_benchmark_value_pair(strategy)
    )
    if strategy_value_ser is None:
        return ''

    # One scale spanning both tables, so an identical return is an identical
    # shade in each and the two grids can be compared directly.
    monthly_low_float, monthly_high_float = _signature_monthly_return_range_tuple(
        strategy_value_ser
    )
    if benchmark_value_ser is not None:
        benchmark_low_float, benchmark_high_float = _signature_monthly_return_range_tuple(
            benchmark_value_ser
        )
        monthly_low_float = min(monthly_low_float, benchmark_low_float)
        monthly_high_float = max(monthly_high_float, benchmark_high_float)
    monthly_range_tuple = (monthly_low_float, monthly_high_float)

    html_part_list = [
        '<h3>Strategy</h3>',
        _signature_monthly_table_html(strategy_value_ser, monthly_range_tuple),
    ]
    if benchmark_value_ser is not None:
        html_part_list.append(
            f'<h3 style="margin-top:22px">{html.escape(str(benchmark_label_str))} (benchmark)</h3>'
        )
        html_part_list.append(
            _signature_monthly_table_html(benchmark_value_ser, monthly_range_tuple)
        )
    html_part_list.append(
        '<p class="metric-context">Monthly returns in per cent. Gains tint green and losses '
        'brown, deepening with the size of the move, on one scale shared between both tables. '
        'Each year\'s return, volatility, max drawdown and Sharpe are computed within that '
        'calendar year only.</p>'
    )
    return ''.join(html_part_list)


def _degenerate_trade_note_html(strategy) -> str:
    """Warn on the report when trade returns contain non-finite values.

    A trade recorded with zero capital but a non-zero commission yields
    return = profit / capital = -inf, which silently poisons every aggregate
    computed from trade returns (average return per trade, best/worst trade,
    payoff ratio). Rather than hide the affected rows and move on, the report
    states the count so the ledger gets fixed at the source.
    """
    trade_df = getattr(strategy, '_trades', None)
    if trade_df is None or len(trade_df) == 0 or 'return' not in trade_df.columns:
        return ''
    trade_return_vec = trade_df['return'].astype(float).to_numpy()
    degenerate_count_int = int((~np.isfinite(trade_return_vec)).sum())
    if degenerate_count_int == 0:
        return ''
    return (
        f'<p class="metric-context"><strong>{degenerate_count_int:,} of {len(trade_return_vec):,} '
        'trades have a non-finite return</strong> (zero recorded capital with a non-zero '
        'commission). Statistics derived from trade returns are unreliable until those ledger '
        'entries are corrected.</p>'
    )


def _build_annual_paths_plate_html(strategy) -> str:
    """One mini-chart per calendar year, all on a shared vertical scale.

    Each panel is that year's growth path rebased to 0 at its first bar, so the
    shape of every year is comparable — the good years and the bad ones shown
    at the same size, which is the point of the device.
    """
    total_value_ser, _b, _l = _strategy_benchmark_value_pair(strategy)
    if total_value_ser is None:
        return ''
    total_value_ser = total_value_ser.dropna()
    if len(total_value_ser) < 2:
        return ''

    # Newest year first, matching the monthly tables: the current year is what
    # the reader checks, so it leads rather than trailing twenty panels of history.
    annual_path_ser_dict: dict[str, pd.Series] = {}
    for year_int, year_value_ser in sorted(
        total_value_ser.groupby(total_value_ser.index.year), key=lambda pair: pair[0], reverse=True
    ):
        if len(year_value_ser) < 2:
            continue
        annual_path_ser_dict[str(int(year_int))] = (
            year_value_ser / year_value_ser.iloc[0] - 1.0
        ).reset_index(drop=True)
    if len(annual_path_ser_dict) == 0:
        return ''

    try:
        small_multiples_uri_str = render_small_multiples_data_uri_str(
            annual_path_ser_dict,
            column_count_int=4,
            share_ylim_bool=True,
            value_formatter_fn=lambda value_float: f'{value_float * 100:.0f}%',
        )
    except ValueError:
        return ''
    return f'''
<h2>Year by Year</h2>
<div class="chart-wrap"><img src="{small_multiples_uri_str}" alt="Growth path by calendar year"></div>
<p class="metric-context">Each calendar year&rsquo;s path, rebased to zero at the year&rsquo;s first
bar and drawn on one shared vertical scale — the losing years are shown at the same size as the
winning ones. The figure beside each year is where it ended.</p>
'''


def _build_relative_performance_plate_html(strategy) -> str:
    """Cumulative strategy-over-benchmark ratio (log). Empty without a benchmark."""
    strategy_value_ser, benchmark_value_ser, _label_str = _strategy_benchmark_value_pair(strategy)
    if strategy_value_ser is None or benchmark_value_ser is None:
        return ''
    try:
        relative_uri_str = render_relative_performance_data_uri_str(
            strategy_value_ser, benchmark_value_ser
        )
    except ValueError:
        return ''
    return f'''
<h2>Relative Performance</h2>
<div class="chart-wrap"><img src="{relative_uri_str}" alt="Relative performance"></div>
<p class="metric-context">Strategy &divide; benchmark, log scale. Rising = beating the
benchmark, flat = matching it, falling = lagging; edge decay shows as flattening.
Both series rebased at the first common bar.</p>
'''


def _build_conditional_beta_plate_html(strategy) -> str:
    """Down/up beta, correlation and capture split by benchmark direction.

    Conditioning is on the benchmark's sign, never the strategy's own — the
    computation guards that. Empty without a benchmark or with too few days in
    either regime.
    """
    strategy_value_ser, benchmark_value_ser, _label_str = _strategy_benchmark_value_pair(strategy)
    if strategy_value_ser is None or benchmark_value_ser is None:
        return ''
    strategy_return_ser = strategy_value_ser.pct_change(fill_method=None).dropna()
    benchmark_return_ser = benchmark_value_ser.pct_change(fill_method=None).dropna()
    try:
        conditional_metric_dict = compute_conditional_beta_dict(
            strategy_return_ser, benchmark_return_ser
        )
    except ValueError:
        return ''

    row_spec_list = [
        ('Beta', 'down_beta_float', 'up_beta_float', '{:.2f}'),
        ('Correlation', 'down_correlation_float', 'up_correlation_float', '{:.2f}'),
        ('Capture', 'down_capture_float', 'up_capture_float', '{:.0%}'),
        ('Observations', 'down_day_count_float', 'up_day_count_float', '{:,.0f}'),
    ]
    row_html_list = [
        f'<tr><td class="metric">{label_str}</td>'
        f'<td>{format_str.format(conditional_metric_dict[down_key_str])}</td>'
        f'<td>{format_str.format(conditional_metric_dict[up_key_str])}</td></tr>'
        for label_str, down_key_str, up_key_str, format_str in row_spec_list
    ]
    asymmetry_float = conditional_metric_dict['beta_asymmetry_float']
    row_html_list.append(
        '<tr><td class="metric">Beta asymmetry</td>'
        f'<td colspan="2">{asymmetry_float:+.2f} (up minus down)</td></tr>'
    )
    return f'''
<h2>Conditional Beta</h2>
<div class="scroll"><table class="stats-table">
<thead><tr><th>Metric</th><th>Benchmark down</th><th>Benchmark up</th></tr></thead>
<tbody>{''.join(row_html_list)}</tbody></table></div>
<p class="metric-context">Beta, correlation and capture split by the sign of the
benchmark day — conditioned on the benchmark, never the strategy's own returns.
A lower down-beta than up-beta is the asymmetry a defensive book is built for.</p>
'''


def _monthly_extra_cell_html(
    column_name_str: str,
    value_obj,
    divider_left_bool: bool = False,
) -> str:
    class_attr_str = ' class="divider-left"' if divider_left_bool else ''
    if pd.isna(value_obj) or value_obj == '':
        return f'<td{class_attr_str}></td>'
    if column_name_str == 'Annual Return':
        return f'<td{class_attr_str} style="{_ret_color(value_obj)}">{value_obj:.1%}</td>'
    if column_name_str == 'Sharpe Ratio':
        return f'<td{class_attr_str}>{float(value_obj):.2f}</td>'
    if column_name_str == 'Max Drawdown':
        return f'<td{class_attr_str} style="{_drawdown_color(value_obj)}">{value_obj:.1%}</td>'
    return f'<td{class_attr_str}>{value_obj}</td>'


def _drawdown_color(val) -> str:
    """Map a drawdown ratio to a light red inline CSS background."""
    try:
        value_float = float(val)
    except (TypeError, ValueError):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )
    if np.isnan(value_float):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )

    intensity_float = min(abs(value_float) / 0.30, 1.0)
    fill_weight_float = 0.18 + 0.28 * intensity_float
    background_color_str = blend_hex_color_str(
        SIGNATURE_PALETTE_DICT['page'],
        SIGNATURE_PALETTE_DICT['loss'],
        fill_weight_float,
    )
    font_color_str = (
        SIGNATURE_PALETTE_DICT['loss_dark']
        if intensity_float < 0.72 else SIGNATURE_PALETTE_DICT['page']
    )
    return f'background-color: {background_color_str}; color: {font_color_str};'


def _monthly_returns_html(
    mr: pd.DataFrame,
    benchmark_mr: pd.DataFrame | None = None,
    benchmark_label_str: str | None = None,
) -> str:
    """Build the monthly returns heatmap as a raw HTML string."""
    month_cols = [m for m in range(1, 13) if m in mr.columns]
    extra_cols = [c for c in mr.columns if c not in range(1, 13)]
    benchmark_metric_spec_list: list[tuple[str, str]] = []
    if benchmark_mr is not None and benchmark_label_str:
        benchmark_metric_spec_list = [
            ('Annual Return', f'{benchmark_label_str} Ann Ret'),
            ('Max Drawdown', f'{benchmark_label_str} Max DD'),
            ('Sharpe Ratio', f'{benchmark_label_str} Sharpe'),
        ]

    month_headers = ''.join(f'<th>{_MONTH_NAMES[m - 1]}</th>' for m in month_cols)
    extra_headers = ''.join(f'<th>{c}</th>' for c in extra_cols)
    benchmark_headers = ''.join(
        (
            f'<th class="divider-left">{display_name_str}</th>'
            if metric_idx_int == 0 else f'<th>{display_name_str}</th>'
        )
        for metric_idx_int, (_, display_name_str) in enumerate(benchmark_metric_spec_list)
    )
    header = f'<tr><th>Year</th>{month_headers}{extra_headers}{benchmark_headers}</tr>'

    rows = []
    for year in mr.index:
        cells = [f'<td><strong>{year}</strong></td>']
        for m in month_cols:
            val = mr.loc[year, m]
            if pd.isna(val):
                cells.append(
                    f'<td style="background-color:{SIGNATURE_PALETTE_DICT["neutral"]};"></td>'
                )
            else:
                cells.append(f'<td style="{_ret_color(val)}">{val:.1%}</td>')
        for c in extra_cols:
            val = mr.loc[year, c]
            cells.append(_monthly_extra_cell_html(c, val))
        if benchmark_mr is not None:
            for metric_idx_int, (metric_name_str, _) in enumerate(benchmark_metric_spec_list):
                if year in benchmark_mr.index and metric_name_str in benchmark_mr.columns:
                    benchmark_value_obj = benchmark_mr.loc[year, metric_name_str]
                else:
                    benchmark_value_obj = np.nan
                cells.append(
                    _monthly_extra_cell_html(
                        metric_name_str,
                        benchmark_value_obj,
                        divider_left_bool=(metric_idx_int == 0),
                    )
                )
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    return (f'<table class="heatmap"><thead>{header}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ HTML assembly ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬



def save_portfolio_results(portfolio, output_dir='results', output_path: str | Path | None = None) -> Path:
    """Save portfolio results to a structured folder and generate an HTML report.

    Creates:
        {output_dir}/research/portfolio/{portfolio.name}/vanilla_backtest/{YYYY-MM-DD_HHMMSS}/
            {portfolio.name}.pkl
            report.html

    Returns the output directory path.
    """
    out = (
        Path(output_path)
        if output_path is not None
        else build_research_output_path(output_dir, 'portfolio', portfolio.name, 'vanilla_backtest')
    )
    out.mkdir(parents=True, exist_ok=True)

    pickle_path = out / f'{portfolio.name}.pkl'
    portfolio.to_pickle(pickle_path)
    _write_metadata(out / _METADATA_FILENAME, _portfolio_metadata_dict(portfolio, pickle_path))
    _write_metadata(out / _RUN_INFO_FILENAME, _portfolio_run_info_dict(portfolio))
    _write_metadata(out / _SUMMARY_FILENAME, _summary_metrics_dict(portfolio))

    # Chart and HTML render inside the active signature variant so the charts
    # and the page styling always agree — same contract as save_results.
    with signature_variant_context(_ACTIVE_REPORT_VARIANT_STR):
        buf = io.BytesIO()
        portfolio.plot(save_to=buf)
        plt.close('all')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('ascii')
        html = _build_portfolio_html(portfolio, chart_b64)
    (out / 'report.html').write_text(html, encoding='utf-8')
    _write_portfolio_tail_csvs(portfolio, out)
    _write_portfolio_rebalance_csvs(portfolio, out)

    print(f'Results saved to: {out.resolve()}')
    return out


def _corr_color(val) -> str:
    """Shade a correlation by how much concentration risk it carries.

    A single hue deepening from bare paper: zero correlation is the neutral
    baseline rather than something to celebrate, and only rising correlation
    is a flag. Blending between two hues instead pushed mid-range values —
    where most real pairs sit — through a muddy midpoint that read as neither.

    *** CRITICAL*** The scale is on |rho|. A strongly negative correlation is
    as much a real relationship as a strongly positive one, and colouring it as
    if it were independent would hide a genuine linkage between pods.
    """
    try:
        value_float = float(val)
    except (TypeError, ValueError):
        return ''
    if np.isnan(value_float):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )

    intensity_float = min(abs(value_float), 1.0)
    background_color_str = blend_hex_color_str(
        str(SIGNATURE_PALETTE_DICT['page']),
        str(SIGNATURE_PALETTE_DICT['loss']),
        intensity_float * 0.62,
    )
    font_color_str = (
        SIGNATURE_PALETTE_DICT['ink']
        if intensity_float < 0.82 else SIGNATURE_PALETTE_DICT['page']
    )
    return f'background-color: {background_color_str}; color: {font_color_str};'


def _format_correlation_matrix(corr: 'pd.DataFrame') -> str:
    """Render a correlation matrix as a colour-coded HTML table.

    The diagonal is left blank. Self-correlation is 1.0 by construction, so
    shading it would put the loudest cells in the table on the one quantity
    that carries no information and draw the eye away from the pairs.
    """
    headers = '<th></th>' + ''.join(f'<th>{c}</th>' for c in corr.columns)
    rows = []
    for row_label in corr.index:
        cells = [f'<td class="metric">{row_label}</td>']
        for col_label in corr.columns:
            if row_label == col_label:
                cells.append('<td style="text-align:center;">—</td>')
                continue
            val = corr.loc[row_label, col_label]
            style = _corr_color(val)
            cells.append(f'<td style="{style} text-align:center;">{val:.3f}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _correlation_shift_color_str(shift_float: float) -> str:
    """Colour a correlation shift by whether diversification held or broke.

    A pair that decouples in the tail is genuinely diversifying and reads
    green; a pair that converges is where the benefit disappears at the moment
    it is needed, and reads brown.
    """
    if not np.isfinite(shift_float):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )
    tone_color_str = str(
        SIGNATURE_PALETTE_DICT['loss'] if shift_float > 0.0 else SIGNATURE_PALETTE_DICT['profit']
    )
    # A shift of one full correlation point is the strongest tint on the scale.
    intensity_float = min(abs(shift_float), 1.0)
    background_color_str = blend_hex_color_str(
        str(SIGNATURE_PALETTE_DICT['page']), tone_color_str, intensity_float * 0.62
    )
    font_color_str = (
        SIGNATURE_PALETTE_DICT['ink']
        if intensity_float < 0.78 else SIGNATURE_PALETTE_DICT['page']
    )
    return f'background-color: {background_color_str}; color: {font_color_str};'


def _build_correlation_shift_html(portfolio) -> str:
    """Show how far each pod pair's correlation moves in the tail.

        shift_ij = rho_ij(tail days) - rho_ij(full sample)

    The level of tail correlation is not the insight on its own — the change is.
    A pair sitting at 0.20 normally and 0.70 in the tail has tripled exactly
    when the diversification was supposed to pay, and neither matrix alone
    makes that visible.

    *** CRITICAL*** Both matrices must be aligned on the same pods before
    subtracting. Differencing them positionally would silently pair unrelated
    strategies if either matrix ever ordered or filtered its pods differently.

    The stress correlation is measured on the benchmark's worst days, which is
    exogenous to this book, so the difference reflects genuine co-movement
    rather than the selection effect that conditioning on the portfolio's own
    losses would introduce.
    """
    full_correlation_df = getattr(portfolio, 'correlation_matrix', None)
    tail_correlation_df = getattr(portfolio, 'tail_correlation_matrix', None)
    if full_correlation_df is None or tail_correlation_df is None:
        return ''
    if len(full_correlation_df) == 0 or len(tail_correlation_df) == 0:
        return ''

    shared_pod_name_list = [
        pod_name_str for pod_name_str in full_correlation_df.index
        if pod_name_str in tail_correlation_df.index
        and pod_name_str in full_correlation_df.columns
        and pod_name_str in tail_correlation_df.columns
    ]
    if len(shared_pod_name_list) < 2:
        return ''

    shift_df = (
        tail_correlation_df.loc[shared_pod_name_list, shared_pod_name_list]
        - full_correlation_df.loc[shared_pod_name_list, shared_pod_name_list]
    )

    header_html_str = '<th></th>' + ''.join(f'<th>{c}</th>' for c in shift_df.columns)
    row_html_list = []
    for row_label_str in shift_df.index:
        cell_html_list = [f'<td class="metric">{row_label_str}</td>']
        for column_label_str in shift_df.columns:
            if row_label_str == column_label_str:
                cell_html_list.append('<td style="text-align:center;">—</td>')
                continue
            shift_float = float(shift_df.loc[row_label_str, column_label_str])
            style_str = _correlation_shift_color_str(shift_float)
            cell_html_list.append(
                f'<td style="{style_str} text-align:center;">{shift_float:+.3f}</td>'
            )
        row_html_list.append('<tr>' + ''.join(cell_html_list) + '</tr>')

    off_diagonal_shift_vec = shift_df.to_numpy()[~np.eye(len(shift_df), dtype=bool)]
    finite_shift_vec = off_diagonal_shift_vec[np.isfinite(off_diagonal_shift_vec)]
    worst_shift_note_str = (
        f' Largest convergence: {float(finite_shift_vec.max()):+.2f}.'
        if len(finite_shift_vec) else ''
    )

    return (
        '<h3>Correlation Shift Under Stress</h3>'
        f'<div class="scroll"><table><thead><tr>{header_html_str}</tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
        '<p class="metric-context">Correlation on benchmark stress days minus full-sample '
        'correlation. Brown means a pair converged when it mattered — diversification that was '
        f'not there in the drawdown; green means it decoupled.{worst_shift_note_str}</p>'
    )


def _fmt_decimal_pct(value_obj, signed_bool: bool = False) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ''
    if np.isnan(value_float):
        return ''
    if signed_bool:
        return f'{value_float * 100:+.2f}%'
    return f'{value_float * 100:.2f}%'


def _tail_summary_html(tail_summary_df: pd.DataFrame) -> str:
    if tail_summary_df is None or len(tail_summary_df) == 0:
        return '<p>No tail summary data available.</p>'

    column_spec_list = [
        ('average_tail_return_float', 'Avg Tail Ret', True),
        ('worst_tail_return_float', 'Worst Tail Ret', True),
        ('negative_tail_day_rate_float', 'Neg Tail Days', False),
        ('average_tail_contribution_float', 'Avg Contribution', True),
        ('worst_tail_contribution_float', 'Worst Contribution', True),
        ('average_loss_contribution_share_float', 'Avg Loss Share', True),
    ]
    header_html_str = '<th>Pod</th>' + ''.join(
        f'<th>{header_str}</th>' for _, header_str, _ in column_spec_list
    )
    row_html_list: list[str] = []
    for pod_name_str, tail_summary_ser in tail_summary_df.iterrows():
        cell_html_list = [f'<td class="metric">{html.escape(str(pod_name_str))}</td>']
        for column_name_str, _, signed_bool in column_spec_list:
            value_obj = tail_summary_ser.get(column_name_str, np.nan)
            if column_name_str == 'average_loss_contribution_share_float':
                class_str = _signed_value_class_str(-float(value_obj)) if pd.notna(value_obj) else ''
            elif column_name_str != 'negative_tail_day_rate_float':
                class_str = _signed_value_class_str(value_obj)
            else:
                class_str = ''
            class_attr_str = f' class="{class_str}"' if class_str else ''
            cell_html_list.append(
                f'<td{class_attr_str}>{_fmt_decimal_pct(value_obj, signed_bool=signed_bool)}</td>'
            )
        row_html_list.append('<tr>' + ''.join(cell_html_list) + '</tr>')

    return f'<table><thead><tr>{header_html_str}</tr></thead><tbody>{"".join(row_html_list)}</tbody></table>'


def _tail_event_contribution_html(portfolio) -> str:
    tail_contribution_df = getattr(portfolio, 'tail_contribution_df', pd.DataFrame())
    if tail_contribution_df is None or len(tail_contribution_df) == 0:
        return '<p>No tail contribution data available.</p>'

    portfolio_tail_return_ser = portfolio.results.loc[
        tail_contribution_df.index,
        'daily_returns',
    ].astype(float)
    worst_tail_date_index = portfolio_tail_return_ser.sort_values(kind='mergesort').head(10).index

    header_html_str = '<th>Date</th><th>Portfolio Return</th>' + ''.join(
        f'<th>{html.escape(str(column_str))}</th>' for column_str in tail_contribution_df.columns
    )
    row_html_list: list[str] = []
    for tail_date_ts in worst_tail_date_index:
        portfolio_return_float = float(portfolio_tail_return_ser.loc[tail_date_ts])
        portfolio_class_str = _signed_value_class_str(portfolio_return_float)
        portfolio_class_attr_str = f' class="{portfolio_class_str}"' if portfolio_class_str else ''
        cell_html_list = [
            f'<td>{pd.Timestamp(tail_date_ts).date()}</td>',
            f'<td{portfolio_class_attr_str}>{_fmt_decimal_pct(portfolio_return_float, signed_bool=True)}</td>',
        ]
        for column_str in tail_contribution_df.columns:
            contribution_float = float(tail_contribution_df.loc[tail_date_ts, column_str])
            contribution_class_str = _signed_value_class_str(contribution_float)
            contribution_class_attr_str = f' class="{contribution_class_str}"' if contribution_class_str else ''
            cell_html_list.append(
                f'<td{contribution_class_attr_str}>{_fmt_decimal_pct(contribution_float, signed_bool=True)}</td>'
            )
        row_html_list.append('<tr>' + ''.join(cell_html_list) + '</tr>')

    return f'<table><thead><tr>{header_html_str}</tr></thead><tbody>{"".join(row_html_list)}</tbody></table>'


def _build_tail_risk_html(portfolio) -> str:
    """Build the portfolio tail-risk diagnostics HTML section."""
    parts = ['<h2>Tail Risk Diagnostics</h2>']
    tail_event_date_index = getattr(portfolio, 'tail_event_date_index', pd.DatetimeIndex([]))
    if len(tail_event_date_index) == 0:
        parts.append('<p>No realized tail days are available for this portfolio.</p>')
        return '\n'.join(parts)

    portfolio_tail_return_ser = portfolio.results.loc[tail_event_date_index, 'daily_returns'].astype(float)
    worst_tail_date_ts = portfolio_tail_return_ser.idxmin()
    worst_tail_return_float = float(portfolio_tail_return_ser.loc[worst_tail_date_ts])
    parts.append(
        '<p>'
        f'<strong>Tail Days:</strong> {len(tail_event_date_index)} '
        f'({portfolio._TAIL_FRACTION_FLOAT:.0%} worst realized portfolio-return days). '
        f'<strong>Worst Tail Date:</strong> {pd.Timestamp(worst_tail_date_ts).date()} '
        f'({_fmt_decimal_pct(worst_tail_return_float, signed_bool=True)}).'
        '</p>'
    )

    tail_correlation_matrix = getattr(portfolio, 'tail_correlation_matrix', pd.DataFrame())
    stress_event_date_index = getattr(portfolio, 'stress_event_date_index', None)
    # *** CRITICAL*** Only present the correlation when the run actually keyed
    # it off an exogenous stress reference. Runs stored before that change hold
    # a matrix conditioned on the portfolio's own tail, which is biased;
    # labelling it as benchmark-based would misrepresent it.
    if stress_event_date_index is None:
        parts.append(
            '<p class="metric-context">Cross-pod correlation under stress is not shown: this '
            'run predates keying stress days off the benchmark, and its stored matrix was '
            'conditioned on the portfolio\'s own worst days, which biases the estimate '
            'downward. Re-run the portfolio to populate it.</p>'
        )
    elif len(stress_event_date_index) > 0 and len(tail_correlation_matrix) > 0:
        parts.append('<h3>Correlation on Benchmark Stress Days</h3>')
        parts.append(f'<div class="scroll">{_format_correlation_matrix(tail_correlation_matrix)}</div>')
        parts.append(
            '<p class="metric-context">Measured on the '
            f'{len(stress_event_date_index)} worst benchmark days, not the portfolio\'s own. '
            'Selecting days by the book\'s own losses would force its pods to offset one '
            'another by construction; an exogenous reference carries no such effect.</p>'
        )
        parts.append(_build_correlation_shift_html(portfolio))
    else:
        parts.append(
            '<p class="metric-context">No benchmark is attached to this portfolio, so '
            'cross-pod correlation under stress is not reported: the only alternative would '
            'be to condition on the book\'s own worst days, which biases the estimate.</p>'
        )

    parts.append('<h3>Tail Summary By Pod</h3>')
    parts.append(f'<div class="scroll">{_tail_summary_html(getattr(portfolio, "tail_summary_df", pd.DataFrame()))}</div>')
    parts.append('<h3>Worst Portfolio Days - Pod Contributions</h3>')
    parts.append(f'<div class="scroll">{_tail_event_contribution_html(portfolio)}</div>')
    return '\n'.join(parts)


def _diversification_ratio_table_html(portfolio) -> str:
    """Report each diversification ratio next to what it actually means.

    The ratio on its own is hard to act on. Two readings make it concrete:

        volatility saved   = 1 - 1 / DR
        effective bets     = DR^2

    Effective bets is the useful one: with N equally weighted, equally
    volatile and uncorrelated pods the ratio is sqrt(N), so squaring it says
    how many genuinely independent pods the book behaves like — against the
    pod count as the ceiling.
    """
    ratio_spec_list = [
        ('At target weights', getattr(portfolio, 'target_diversification_ratio', None)),
        ('At end weights', getattr(portfolio, 'realized_diversification_ratio', None)),
        ('Rolling 63-day average', getattr(portfolio, 'average_rolling_diversification_ratio', None)),
    ]
    row_html_list = []
    for label_str, ratio_obj in ratio_spec_list:
        if ratio_obj is None or not np.isfinite(float(ratio_obj)) or float(ratio_obj) <= 0.0:
            continue
        ratio_float = float(ratio_obj)
        row_html_list.append(
            f'<tr><td class="metric">{label_str}</td>'
            f'<td>{ratio_float:.3f}</td>'
            f'<td>{(1.0 - 1.0 / ratio_float) * 100:.1f}%</td>'
            f'<td>{ratio_float ** 2:.2f}</td></tr>'
        )
    if len(row_html_list) == 0:
        return ''

    pod_count_int = len(getattr(portfolio, 'strategies', []) or [])
    ceiling_note_str = (
        f' With {pod_count_int} pods the ceiling is {pod_count_int} effective bets '
        f'(a ratio of {np.sqrt(pod_count_int):.2f}), reached only if they were fully independent.'
        if pod_count_int > 1 else ''
    )
    return (
        '<h3>Diversification Ratio</h3>'
        '<div class="scroll"><table class="stats-table">'
        '<thead><tr><th>Measured</th><th>Ratio</th><th>Volatility saved</th>'
        '<th>Effective bets</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
        '<p class="metric-context">A ratio of 1.00 means the pods move as one and the book is no '
        'calmer than its parts. Volatility saved is how much lower the book\'s volatility is than '
        'the weighted sum of its pods; effective bets is how many genuinely independent pods it '
        f'behaves like.{ceiling_note_str} Measured across the whole sample — whether the benefit '
        'survives a crash is the correlation shift under Tail Risk.</p>'
    )


def _build_diagnostics_html(portfolio) -> str:
    """Build the Cross-Strategy Diagnostics HTML section."""
    parts = ['<h2>Diversification</h2>']

    if portfolio.correlation_matrix is not None and len(portfolio.correlation_matrix) > 0:
        parts.append('<h3>Correlation Matrix</h3>')
        parts.append(f'<div class="scroll">{_format_correlation_matrix(portfolio.correlation_matrix)}</div>')

    parts.append(_diversification_ratio_table_html(portfolio))

    if portfolio._rebalance is not None:
        parts.append(f'<p><strong>Rebalance Frequency:</strong> {portfolio._rebalance}</p>')
        parts.append(
            f'<p><strong>Rebalance Policy:</strong> '
            f'{getattr(portfolio, "_rebalance_policy", "fixed")}</p>'
        )
        if getattr(portfolio, '_rebalance_policy', 'fixed') == 'inverse_volatility':
            parts.append(
                f'<p><strong>Inverse-Vol Lookback:</strong> '
                f'{getattr(portfolio, "_rebalance_inverse_volatility_lookback_day_int", "")} '
                f'trading days</p>'
            )
    else:
        parts.append('<p><strong>Rebalance Frequency:</strong> None (buy-and-hold)</p>')

    return '\n'.join(parts)


def _build_provenance_html(portfolio) -> str:
    """Build a provenance section for portfolio configuration and sources."""
    rows = []
    for pod_info_dict in portfolio.pod_info_list:
        requested_start_date_str = pod_info_dict.get(
            "requested_backtest_start_date_str",
            "",
        )
        effective_start_date_str = pod_info_dict.get(
            "effective_backtest_start_date_str",
            pod_info_dict.get("backtest_start_date_str", ""),
        )
        rows.append(
            '<tr>'
            f'<td>{pod_info_dict.get("strategy_name", "")}</td>'
            f'<td>{pod_info_dict.get("weight", 0):.1%}</td>'
            f'<td>{_fmt_dollar(pod_info_dict.get("allocated_capital", ""))}</td>'
            f'<td>{requested_start_date_str}</td>'
            f'<td>{effective_start_date_str}</td>'
            f'<td>{pod_info_dict.get("source_pkl", "")}</td>'
            '</tr>'
        )

    overlap_start = ''
    if portfolio._common_start is not None:
        overlap_start = str(pd.Timestamp(portfolio._common_start).date())

    overlap_end = ''
    if portfolio._common_end is not None:
        overlap_end = str(pd.Timestamp(portfolio._common_end).date())

    config_path = portfolio.source_config_path or ''
    source_table = (
        '<table><thead><tr><th>Pod</th><th>Weight</th><th>Allocated Capital</th>'
        '<th>Requested Start</th><th>Effective Pod Start</th><th>Source Pickle</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )

    return (
        '<h2>Provenance</h2>'
        f'<p><strong>Config:</strong> {config_path}</p>'
        f'<p><strong>Common Overlap Window:</strong> {overlap_start} &rarr; {overlap_end}</p>'
        f'<div class="scroll">{source_table}</div>'
    )


def _add_vertical_line_markers(ax, vertical_line_index: pd.DatetimeIndex | None):
    if vertical_line_index is None or len(vertical_line_index) == 0:
        return

    for vertical_date in pd.DatetimeIndex(vertical_line_index):
        ax.axvline(
            vertical_date,
            color=SIGNATURE_PALETTE_DICT['benchmark'],
            linestyle='--',
            linewidth=0.8,
            alpha=0.42,
            zorder=1,
        )


def _short_chart_label_str(label_str: str, max_part_len_int: int = 34) -> str:
    """Legend-safe display label: drop the 'strategy_' prefix and cap length.

    Pod labels are file stems (~55 chars) and pairwise-correlation labels join
    two of them with ' vs '. A multi-column legend of such labels grows wider
    than the 12-inch figure, and ``savefig(bbox_inches='tight')`` then expands
    the canvas around the legend — leaving the actual plot as a tiny sliver.
    """
    part_list = [part_str.strip() for part_str in str(label_str).split(' vs ')]
    short_part_list = []
    for part_str in part_list:
        trimmed_str = part_str.removeprefix('strategy_')
        if len(trimmed_str) > max_part_len_int:
            trimmed_str = trimmed_str[: max_part_len_int - 1] + '…'
        short_part_list.append(trimmed_str)
    return ' vs '.join(short_part_list)


def _legend_ncol_int(label_list: list[str]) -> int:
    """One column when any label is long, else up to three side by side."""
    if any(len(label_str) > 30 for label_str in label_list):
        return 1
    return min(3, max(1, len(label_list)))


def _weights_chart_b64(
    weights: pd.DataFrame,
    title: str,
    vertical_line_index: pd.DatetimeIndex | None = None,
) -> str | None:
    """Render a stacked portfolio-weights chart to base64 PNG."""
    if weights is None or len(weights) == 0:
        return None

    weights = weights.copy().sort_index()
    active_cols = [col for col in weights.columns if not np.allclose(weights[col].fillna(0).to_numpy(), 0.0)]
    if not active_cols:
        return None

    weight_color_list = [_weight_color_for_asset(column_name_str) for column_name_str in active_cols]

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.stackplot(
            weights.index,
            [weights[col].fillna(0).to_numpy() for col in active_cols],
            labels=[_short_chart_label_str(col) for col in active_cols],
            colors=weight_color_list,
            alpha=0.88,
            edgecolor=_WEIGHT_STACK_EDGE_COLOR_STR,
            linewidth=0.95,
        )
        ax.set_title(title)
        ax.set_ylabel('Weight')
        ax.set_xlabel('Date')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis='y', alpha=1.0)
        _add_vertical_line_markers(ax, vertical_line_index)
        legend_label_list = [_short_chart_label_str(col) for col in active_cols]
        legend_obj = ax.legend(
            loc='upper left', ncol=_legend_ncol_int(legend_label_list), fontsize=8, frameon=True
        )
        legend_obj.get_frame().set_edgecolor(SIGNATURE_PALETTE_DICT['legend_edge'])
        legend_obj.get_frame().set_facecolor(SIGNATURE_PALETTE_DICT['legend_face'])
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _composition_weights_chart_b64(weights_df) -> str | None:
    """Full-history weight stack, labelled at the right edge.

    Distinct from _weights_chart_b64, which is the windowed chart the portfolio
    pages use. Two differences, both because this one carries a decade rather
    than a two-year window:

    * Sleeves are labelled where they END, not in a legend box. With seven
      bands a top-left legend forces the reader to match a swatch to a band
      by colour alone, which is exactly what fails when two muted tones sit
      next to each other. A label at the band's own height needs no matching.
    * Bands carry a hatch as well as a colour, so a sleeve that narrows to a
      few pixels is still identifiable, and the stack survives printing.
    """
    if weights_df is None or len(weights_df) == 0:
        return None
    weights_df = weights_df.copy().sort_index()
    active_column_name_list = [
        column_name_str for column_name_str in weights_df.columns
        if not np.allclose(weights_df[column_name_str].fillna(0).to_numpy(), 0.0)
    ]
    if not active_column_name_list:
        return None

    hatch_cycle_list = list(SIGNATURE_PALETTE_DICT['hatch_cycle_list']) or ['']
    weight_color_list = [
        _weight_color_for_asset(column_name_str) for column_name_str in active_column_name_list
    ]
    weight_value_list = [
        weights_df[column_name_str].fillna(0).to_numpy()
        for column_name_str in active_column_name_list
    ]

    # The hatch is drawn in the page colour at a thin stroke, so it reads as a
    # light texture over the sleeve's hue rather than as a dark mesh on top of
    # it. The same colour draws the band separators, which is what gives the
    # stack its hairline partings.
    stack_rcparam_dict = dict(build_signature_rcparams(to_web_bool=True))
    stack_rcparam_dict['hatch.linewidth'] = 0.35
    with plt.rc_context(stack_rcparam_dict):
        figure_obj, axis_obj = plt.subplots(figsize=(12, 3.6))
        stack_collection_list = axis_obj.stackplot(
            weights_df.index,
            weight_value_list,
            colors=weight_color_list,
            alpha=0.95,
            edgecolor=str(SIGNATURE_PALETTE_DICT['page']),
            linewidth=0.7,
        )
        for band_index_int, band_collection_obj in enumerate(stack_collection_list):
            band_collection_obj.set_hatch(hatch_cycle_list[band_index_int % len(hatch_cycle_list)])

        # Label each sleeve at its own final height. Bands that end at zero
        # would collide at the baseline, so they are stacked upward by a
        # minimum step rather than overprinting each other.
        cumulative_float = 0.0
        label_position_list: list[tuple[float, str]] = []
        for column_index_int, column_name_str in enumerate(active_column_name_list):
            final_weight_float = float(weight_value_list[column_index_int][-1])
            label_position_list.append((
                cumulative_float + final_weight_float / 2.0,
                _short_chart_label_str(column_name_str),
            ))
            cumulative_float += final_weight_float

        minimum_label_gap_float = 0.045
        previous_y_float = -1.0
        for label_y_float, label_text_str in label_position_list:
            resolved_y_float = max(label_y_float, previous_y_float + minimum_label_gap_float)
            previous_y_float = resolved_y_float
            axis_obj.annotate(
                label_text_str,
                xy=(1.005, resolved_y_float),
                xycoords=('axes fraction', 'data'),
                va='center',
                ha='left',
                fontsize=7.5,
                color=SIGNATURE_PALETTE_DICT['ink'],
                annotation_clip=False,
            )

        axis_obj.set_ylabel('Weight')
        axis_obj.set_ylim(0, 1.0)
        axis_obj.set_xlim(weights_df.index.min(), weights_df.index.max())
        axis_obj.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
        )
        axis_obj.grid(axis='y', alpha=1.0)
        figure_obj.autofmt_xdate()
        figure_obj.tight_layout()

        buffer_obj = io.BytesIO()
        figure_obj.savefig(buffer_obj, format='png', dpi=140, bbox_inches='tight')
        plt.close(figure_obj)
    buffer_obj.seek(0)
    return base64.b64encode(buffer_obj.read()).decode('ascii')


def _stacked_equity_chart_b64(
    pod_equity_df: pd.DataFrame,
    title: str,
    vertical_line_index: pd.DatetimeIndex | None = None,
) -> str | None:
    """Render stacked pod equity contributions in dollars."""
    if pod_equity_df is None or len(pod_equity_df) == 0:
        return None

    pod_equity_df = pod_equity_df.copy().sort_index()
    active_col_list = [
        column_str for column_str in pod_equity_df.columns
        if not np.allclose(pod_equity_df[column_str].fillna(0.0).to_numpy(), 0.0)
    ]
    if not active_col_list:
        return None

    color_list = [
        SIGNATURE_PALETTE_DICT['series_cycle'][column_idx_int % len(SIGNATURE_PALETTE_DICT['series_cycle'])]
        for column_idx_int, _ in enumerate(active_col_list)
    ]

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.stackplot(
            pod_equity_df.index,
            [pod_equity_df[column_str].fillna(0.0).to_numpy() for column_str in active_col_list],
            labels=[_short_chart_label_str(column_str) for column_str in active_col_list],
            colors=color_list,
            alpha=0.84,
            edgecolor=SIGNATURE_PALETTE_DICT['page'],
            linewidth=0.6,
        )
        ax.set_title(title)
        ax.set_ylabel('Equity [$]')
        ax.set_xlabel('Date')
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda value_float, _: f'${value_float:,.0f}')
        )
        ax.grid(True)
        _add_vertical_line_markers(ax, vertical_line_index)
        legend_label_list = [_short_chart_label_str(column_str) for column_str in active_col_list]
        ax.legend(loc='upper left', ncol=_legend_ncol_int(legend_label_list), fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _multi_line_chart_b64(
    value_df: pd.DataFrame,
    title: str,
    y_label_str: str,
    vertical_line_index: pd.DatetimeIndex | None = None,
    ylim_tuple: tuple[float, float] | None = None,
) -> str | None:
    """Render one or more diagnostic time series as a line chart."""
    if value_df is None or len(value_df) == 0:
        return None

    plot_df = value_df.copy().sort_index().dropna(how='all')
    if len(plot_df) == 0:
        return None

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        fig, ax = plt.subplots(figsize=(12, 4.4))
        for column_idx_int, column_str in enumerate(plot_df.columns):
            value_ser = plot_df[column_str].astype(float)
            ax.plot(
                value_ser.index,
                value_ser.to_numpy(),
                label=_short_chart_label_str(column_str),
                color=SIGNATURE_PALETTE_DICT['series_cycle'][column_idx_int % len(SIGNATURE_PALETTE_DICT['series_cycle'])],
                linewidth=1.15,
                alpha=0.95,
            )

        ax.set_title(title)
        ax.set_ylabel(y_label_str)
        ax.set_xlabel('Date')
        if ylim_tuple is not None:
            ax.set_ylim(*ylim_tuple)
        ax.grid(True)
        _add_vertical_line_markers(ax, vertical_line_index)
        legend_label_list = [_short_chart_label_str(column_str) for column_str in plot_df.columns]
        ax.legend(loc='upper left', ncol=_legend_ncol_int(legend_label_list), fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _weights_chart_block(
    weights: pd.DataFrame,
    title: str,
    subtitle: str,
    vertical_line_index: pd.DatetimeIndex | None = None,
) -> str:
    chart_b64 = _weights_chart_b64(weights, title, vertical_line_index=vertical_line_index)
    if chart_b64 is None:
        return f'<h3>{title}</h3><p>{subtitle}</p><p>No weight data available for this window.</p>'
    return (
        f'<h3>{title}</h3>'
        f'<p>{subtitle}</p>'
        f'<div class="chart-wrap"><img src="data:image/png;base64,{chart_b64}" alt="{title}"></div>'
    )


def _chart_block_from_b64(chart_b64: str | None, title: str, subtitle: str) -> str:
    if chart_b64 is None:
        return f'<h3>{title}</h3><p>{subtitle}</p><p>No diagnostic data available for this window.</p>'
    return (
        f'<h3>{title}</h3>'
        f'<p>{subtitle}</p>'
        f'<div class="chart-wrap"><img src="data:image/png;base64,{chart_b64}" alt="{title}"></div>'
    )


def _format_signed_dollar_str(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ''
    if np.isnan(value_float):
        return ''
    if np.isclose(value_float, 0.0, atol=1e-9):
        return '$0.00'
    sign_str = '+' if value_float > 0.0 else '-'
    return f'{sign_str}${abs(value_float):,.2f}'


def _pm_pod_name_list(portfolio) -> list[str]:
    return [str(strategy_obj.name) for strategy_obj in getattr(portfolio, 'strategies', [])]


def _pm_initial_weight_ser(portfolio, pod_name_list: list[str] | None = None) -> pd.Series:
    if pod_name_list is None:
        pod_name_list = _pm_pod_name_list(portfolio)
    weight_float_list = [float(weight_obj) for weight_obj in getattr(portfolio, 'weights', [])]
    return pd.Series(weight_float_list, index=pod_name_list, dtype=float)


def _pm_rebalance_target_weight_df(portfolio, pod_name_list: list[str]) -> pd.DataFrame:
    target_weight_df = getattr(portfolio, 'rebalance_target_weight_df', pd.DataFrame())
    if target_weight_df is None or len(target_weight_df) == 0:
        return pd.DataFrame(columns=pod_name_list, dtype=float)

    normalized_target_weight_df = target_weight_df.copy()
    normalized_target_weight_df.index = pd.to_datetime(normalized_target_weight_df.index).normalize()
    normalized_target_weight_df.columns = [str(column_obj) for column_obj in normalized_target_weight_df.columns]
    normalized_target_weight_df = normalized_target_weight_df.apply(pd.to_numeric, errors='coerce')
    normalized_target_weight_df = normalized_target_weight_df.groupby(normalized_target_weight_df.index).last()
    return normalized_target_weight_df.reindex(columns=pod_name_list)


def _pm_rebalance_diagnostic_df(portfolio) -> pd.DataFrame:
    diagnostic_df = getattr(portfolio, 'rebalance_diagnostic_df', pd.DataFrame())
    if diagnostic_df is None or len(diagnostic_df) == 0:
        return pd.DataFrame()

    normalized_diagnostic_df = diagnostic_df.copy()
    normalized_diagnostic_df.index = pd.to_datetime(normalized_diagnostic_df.index).normalize()
    return normalized_diagnostic_df.groupby(normalized_diagnostic_df.index).last()


def _pm_latest_target_weight_bundle(portfolio) -> tuple[pd.Series, str]:
    pod_name_list = _pm_pod_name_list(portfolio)
    initial_weight_ser = _pm_initial_weight_ser(portfolio, pod_name_list=pod_name_list)
    target_weight_df = _pm_rebalance_target_weight_df(portfolio, pod_name_list)
    if len(target_weight_df) == 0:
        return initial_weight_ser, 'Initial configured weights'

    latest_target_date_ts = pd.Timestamp(target_weight_df.index[-1])
    latest_target_weight_ser = target_weight_df.iloc[-1].reindex(pod_name_list).fillna(0.0).astype(float)
    return latest_target_weight_ser, f'Latest applied rebalance target ({latest_target_date_ts.date()})'


def _pm_common_window_str(portfolio) -> str:
    common_start_obj = getattr(portfolio, '_common_start', None)
    common_end_obj = getattr(portfolio, '_common_end', None)
    if common_start_obj is None or common_end_obj is None:
        return ''
    return f'{pd.Timestamp(common_start_obj).date()} -> {pd.Timestamp(common_end_obj).date()}'


def _pm_policy_summary_html(portfolio, target_source_str: str) -> str:
    rebalance_frequency_str = getattr(portfolio, '_rebalance', None) or 'None (buy-and-hold)'
    rebalance_policy_str = getattr(portfolio, '_rebalance_policy', 'fixed')
    row_tuple_list = [
        ('Source config', getattr(portfolio, 'source_config_path', '') or ''),
        ('Common overlap window', _pm_common_window_str(portfolio)),
        ('Rebalance frequency', rebalance_frequency_str),
        ('Rebalance policy', rebalance_policy_str),
        ('Active target source', target_source_str),
    ]
    if rebalance_policy_str == 'inverse_volatility':
        row_tuple_list.append(
            (
                'Inverse-vol lookback',
                f'{getattr(portfolio, "_rebalance_inverse_volatility_lookback_day_int", "")} completed trading days',
            )
        )
    row_tuple_list.append(
        (
            'Live capital movement',
            'Manual IBKR transfers between pod accounts; this report does not place orders.',
        )
    )

    row_html_list = [
        '<tr>'
        f'<td class="metric">{html.escape(label_str)}</td>'
        f'<td>{html.escape(str(value_obj))}</td>'
        '</tr>'
        for label_str, value_obj in row_tuple_list
    ]
    return (
        '<div class="scroll">'
        '<table><thead><tr><th>Field</th><th>Value</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table>'
        '</div>'
    )


def _pm_allocation_snapshot_df(portfolio) -> pd.DataFrame:
    pod_name_list = _pm_pod_name_list(portfolio)
    pod_equity_df = getattr(portfolio, '_pod_equities', pd.DataFrame())
    if pod_equity_df is None or len(pod_equity_df) == 0 or len(pod_name_list) == 0:
        return pd.DataFrame()

    initial_weight_ser = _pm_initial_weight_ser(portfolio, pod_name_list=pod_name_list).reindex(pod_name_list)
    target_weight_ser, _ = _pm_latest_target_weight_bundle(portfolio)
    target_weight_ser = target_weight_ser.reindex(pod_name_list).fillna(0.0).astype(float)

    current_equity_ser = pod_equity_df.iloc[-1].reindex(pod_name_list).astype(float)
    total_equity_float = float(current_equity_ser.sum())
    if not np.isfinite(total_equity_float) or total_equity_float <= 0.0:
        return pd.DataFrame()

    drift_weight_df = getattr(portfolio, 'drift_weight_df', pd.DataFrame())
    if drift_weight_df is not None and len(drift_weight_df) > 0:
        actual_weight_ser = drift_weight_df.iloc[-1].reindex(pod_name_list).fillna(0.0).astype(float)
    else:
        actual_weight_ser = current_equity_ser / total_equity_float

    target_equity_ser = target_weight_ser * total_equity_float
    transfer_amount_ser = target_equity_ser - current_equity_ser
    drift_weight_ser = actual_weight_ser - target_weight_ser

    row_dict_list: list[dict[str, object]] = []
    for pod_name_str in pod_name_list:
        row_dict_list.append(
            {
                'pod_name_str': pod_name_str,
                'initial_weight_float': float(initial_weight_ser.loc[pod_name_str]),
                'target_weight_float': float(target_weight_ser.loc[pod_name_str]),
                'actual_weight_float': float(actual_weight_ser.loc[pod_name_str]),
                'drift_weight_float': float(drift_weight_ser.loc[pod_name_str]),
                'current_equity_float': float(current_equity_ser.loc[pod_name_str]),
                'target_equity_float': float(target_equity_ser.loc[pod_name_str]),
                'manual_delta_float': float(transfer_amount_ser.loc[pod_name_str]),
            }
        )

    return pd.DataFrame(row_dict_list)


def _pm_allocation_snapshot_html(portfolio) -> str:
    snapshot_df = _pm_allocation_snapshot_df(portfolio)
    if len(snapshot_df) == 0:
        return '<p>No PM allocation snapshot is available.</p>'

    pod_equity_df = getattr(portfolio, '_pod_equities', pd.DataFrame())
    snapshot_date_str = str(pd.Timestamp(pod_equity_df.index[-1]).date())
    header_html_str = (
        '<th>Pod</th>'
        '<th>Initial Target</th>'
        '<th>Active Target</th>'
        '<th>Actual End Weight</th>'
        '<th>Weight Drift</th>'
        '<th>Current Sleeve</th>'
        '<th>Target Capital</th>'
        '<th>Manual Delta</th>'
    )
    row_html_list: list[str] = []
    for _, snapshot_row_ser in snapshot_df.iterrows():
        drift_class_str = _signed_value_class_str(snapshot_row_ser['drift_weight_float'])
        delta_class_str = _signed_value_class_str(snapshot_row_ser['manual_delta_float'])
        drift_class_attr_str = f' class="{drift_class_str}"' if drift_class_str else ''
        delta_class_attr_str = f' class="{delta_class_str}"' if delta_class_str else ''
        row_html_list.append(
            '<tr>'
            f'<td class="metric">{html.escape(str(snapshot_row_ser["pod_name_str"]))}</td>'
            f'<td>{_format_weight_ratio_str(snapshot_row_ser["initial_weight_float"])}</td>'
            f'<td>{_format_weight_ratio_str(snapshot_row_ser["target_weight_float"])}</td>'
            f'<td>{_format_weight_ratio_str(snapshot_row_ser["actual_weight_float"])}</td>'
            f'<td{drift_class_attr_str}>{_format_weight_ratio_str(snapshot_row_ser["drift_weight_float"])}</td>'
            f'<td>{_fmt_dollar(snapshot_row_ser["current_equity_float"])}</td>'
            f'<td>{_fmt_dollar(snapshot_row_ser["target_equity_float"])}</td>'
            f'<td{delta_class_attr_str}>{_format_signed_dollar_str(snapshot_row_ser["manual_delta_float"])}</td>'
            '</tr>'
        )

    return (
        '<h3>Manual IBKR Transfer Guide</h3>'
        f'<p>Close-marked snapshot date: {snapshot_date_str}. '
        'Weight Drift = actual end weight - active target weight. '
        'Manual Delta = target capital - current sleeve equity; positive means add capital to that pod, '
        'negative means remove capital from that pod.</p>'
        '<div class="scroll">'
        f'<table><thead><tr>{header_html_str}</tr></thead><tbody>{"".join(row_html_list)}</tbody></table>'
        '</div>'
    )


def _format_pm_weight_list_str(weight_ser: pd.Series) -> str:
    part_str_list = []
    for pod_name_str, weight_float in weight_ser.items():
        part_str_list.append(f'{pod_name_str} {_format_weight_ratio_str(weight_float)}')
    return '; '.join(part_str_list)


def _recent_pm_rebalance_html(portfolio) -> str:
    pod_name_list = _pm_pod_name_list(portfolio)
    target_weight_df = _pm_rebalance_target_weight_df(portfolio, pod_name_list)
    diagnostic_df = _pm_rebalance_diagnostic_df(portfolio)
    if len(target_weight_df) == 0 and len(diagnostic_df) == 0:
        return ''

    if len(diagnostic_df) > 0:
        recent_date_index = diagnostic_df.tail(6).index
    else:
        recent_date_index = target_weight_df.tail(6).index

    header_html_str = (
        '<th>Date</th>'
        '<th>Status</th>'
        '<th>Policy</th>'
        '<th>Observations</th>'
        '<th>Target Weights</th>'
    )
    row_html_list: list[str] = []
    for rebalance_date_ts in recent_date_index:
        normalized_rebalance_date_ts = pd.Timestamp(rebalance_date_ts).normalize()
        if normalized_rebalance_date_ts in diagnostic_df.index:
            diagnostic_ser = diagnostic_df.loc[normalized_rebalance_date_ts]
            status_str = str(diagnostic_ser.get('status_str', ''))
            policy_str = str(diagnostic_ser.get('policy_str', getattr(portfolio, '_rebalance_policy', 'fixed')))
            observation_count_obj = diagnostic_ser.get('observation_count_int', '')
            observation_count_str = ''
            if pd.notna(observation_count_obj):
                observation_count_str = str(int(float(observation_count_obj)))
        else:
            status_str = 'applied'
            policy_str = getattr(portfolio, '_rebalance_policy', 'fixed')
            observation_count_str = ''

        if normalized_rebalance_date_ts in target_weight_df.index:
            target_weight_ser = target_weight_df.loc[normalized_rebalance_date_ts].reindex(pod_name_list).fillna(0.0)
            target_weight_str = _format_pm_weight_list_str(target_weight_ser)
        else:
            target_weight_str = ''

        row_html_list.append(
            '<tr>'
            f'<td>{normalized_rebalance_date_ts.date()}</td>'
            f'<td>{html.escape(status_str)}</td>'
            f'<td>{html.escape(policy_str)}</td>'
            f'<td>{html.escape(observation_count_str)}</td>'
            f'<td>{html.escape(target_weight_str)}</td>'
            '</tr>'
        )

    return (
        '<h3>Recent PM Rebalances</h3>'
        '<p>Scheduled PM rebalance decisions. Skipped rows have no applied target weights.</p>'
        '<div class="scroll">'
        f'<table><thead><tr>{header_html_str}</tr></thead><tbody>{"".join(row_html_list)}</tbody></table>'
        '</div>'
    )


def _build_pm_allocation_html(portfolio) -> str:
    _, target_source_str = _pm_latest_target_weight_bundle(portfolio)
    return (
        '<h2>PM Allocation Overview</h2>'
        '<p>This is the PM aggregation view for the selected portfolio config. '
        'The output folder may still use the vanilla_backtest label, but the weights, '
        'rebalance policy, and target calculations are specific to this PM run.</p>'
        '<h3>Construction Policy</h3>'
        f'{_pm_policy_summary_html(portfolio, target_source_str)}'
        f'{_pm_allocation_snapshot_html(portfolio)}'
        f'{_recent_pm_rebalance_html(portfolio)}'
    )


def _format_weight_ratio_str(weight_obj) -> str:
    if pd.isna(weight_obj):
        return ''
    try:
        return f'{float(weight_obj):.2%}'
    except (TypeError, ValueError):
        return str(weight_obj)


def _recent_taa_weight_comparison_df(strategy) -> pd.DataFrame:
    """
    Build the recent TAA target-vs-realized comparison table.

    For asset i on rebalance date t:

        target_cash_weight_t = 1 - sum_i target_weight_{i,t}

        drift_weight_{i,t} = realized_weight_{i,t} - target_weight_{i,t}
    """
    target_weight_df = getattr(strategy, 'rebalance_weight_df', None)
    if target_weight_df is None or len(target_weight_df) == 0:
        return pd.DataFrame(columns=['date', 'asset', 'target_weight', 'realized_weight', 'drift_weight'])

    realized_weight_df = getattr(strategy, 'realized_weight_df', pd.DataFrame())
    target_weight_df = target_weight_df.copy().sort_index()
    target_weight_df.index = pd.to_datetime(target_weight_df.index).normalize()
    target_weight_df.columns = [str(column_obj) for column_obj in target_weight_df.columns]
    target_weight_df = target_weight_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    target_weight_df = target_weight_df.groupby(target_weight_df.index).last()

    recent_target_weight_df = target_weight_df.tail(3).copy()
    if len(recent_target_weight_df) == 0:
        return pd.DataFrame(columns=['date', 'asset', 'target_weight', 'realized_weight', 'drift_weight'])

    target_cash_weight_ser = 1.0 - recent_target_weight_df.sum(axis=1)
    target_cash_weight_ser = target_cash_weight_ser.mask(
        np.isclose(target_cash_weight_ser.to_numpy(dtype=float), 0.0, atol=1e-12),
        0.0,
    )
    recent_target_weight_df['Cash'] = target_cash_weight_ser

    if realized_weight_df is None or len(realized_weight_df) == 0:
        realized_weight_df = pd.DataFrame()
    else:
        realized_weight_df = realized_weight_df.copy()
        realized_weight_df.index = pd.to_datetime(realized_weight_df.index).normalize()
        realized_weight_df.columns = [str(column_obj) for column_obj in realized_weight_df.columns]
        realized_weight_df = realized_weight_df.apply(pd.to_numeric, errors='coerce')
        realized_weight_df = realized_weight_df.groupby(realized_weight_df.index).last()

    target_asset_name_list = [
        column_str for column_str in recent_target_weight_df.columns
        if column_str != 'Cash'
    ]
    asset_name_set = set(target_asset_name_list)
    realized_extra_asset_name_list: list[str] = []
    if len(realized_weight_df) > 0:
        recent_realized_weight_df = realized_weight_df.reindex(recent_target_weight_df.index)
        for column_str in recent_realized_weight_df.columns:
            if column_str in asset_name_set or column_str == 'Cash':
                continue
            realized_asset_weight_ser = recent_realized_weight_df[column_str].dropna()
            if len(realized_asset_weight_ser) > 0 and not np.allclose(
                realized_asset_weight_ser.to_numpy(dtype=float),
                0.0,
                atol=1e-12,
            ):
                realized_extra_asset_name_list.append(column_str)

    asset_name_list = target_asset_name_list + realized_extra_asset_name_list + ['Cash']
    realized_date_set = set(realized_weight_df.index) if len(realized_weight_df) > 0 else set()
    row_dict_list: list[dict[str, object]] = []

    for rebalance_date_ts, target_weight_ser in recent_target_weight_df.iterrows():
        if rebalance_date_ts in realized_date_set:
            realized_weight_ser = realized_weight_df.loc[rebalance_date_ts].reindex(asset_name_list).fillna(0.0)
        else:
            realized_weight_ser = pd.Series(np.nan, index=asset_name_list, dtype=float)

        for asset_name_str in asset_name_list:
            target_weight_float = float(target_weight_ser.get(asset_name_str, 0.0))
            realized_weight_obj = realized_weight_ser.get(asset_name_str, np.nan)
            realized_weight_float = float(realized_weight_obj) if pd.notna(realized_weight_obj) else np.nan
            drift_weight_float = (
                np.nan
                if pd.isna(realized_weight_float)
                else realized_weight_float - target_weight_float
            )
            row_dict_list.append(
                {
                    'date': rebalance_date_ts.date(),
                    'asset': asset_name_str,
                    'target_weight': target_weight_float,
                    'realized_weight': realized_weight_float,
                    'drift_weight': drift_weight_float,
                }
            )

    return pd.DataFrame(row_dict_list, columns=['date', 'asset', 'target_weight', 'realized_weight', 'drift_weight'])


def _current_composition_html(strategy) -> str:
    """What the book holds right now, as a labelled bar per sleeve.

    The stacked weight chart above answers "how did this change over a decade";
    it cannot be read for "what do I hold today" without squinting at its right
    edge. This answers the second question directly, and that is the one an
    operator and an investor both ask first.

    Bars are HTML, not a rasterised chart: they stay crisp at any zoom, add no
    bytes to the artifact, and reuse the same per-asset colours as the stack
    above, so a sleeve is the same colour in both.

    *** CRITICAL*** Read from the last row of the REALIZED weight history --
    close-marked weights after execution and valuation, not the target the
    rebalance asked for. Showing targets here would state an intention as
    though it were a position.
    """
    realized_weight_df = getattr(strategy, 'realized_weight_df', None)
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return ''
    latest_weight_ser = realized_weight_df.iloc[-1].apply(
        pd.to_numeric, errors='coerce'
    ).dropna()
    latest_weight_ser = latest_weight_ser[latest_weight_ser.abs() > 1e-6]
    if len(latest_weight_ser) == 0:
        return ''
    latest_weight_ser = latest_weight_ser.sort_values(ascending=False)

    as_of_obj = realized_weight_df.index[-1]
    as_of_str = as_of_obj.date().isoformat() if hasattr(as_of_obj, 'date') else str(as_of_obj)
    # Bars scale to the largest holding, not to 100%: a book whose biggest
    # sleeve is 25% would otherwise render as a row of barely visible slivers.
    scale_weight_float = float(latest_weight_ser.abs().max()) or 1.0

    row_html_list: list[str] = []
    for asset_name_obj, weight_float in latest_weight_ser.items():
        asset_name_str = str(asset_name_obj)
        width_pct_float = abs(float(weight_float)) / scale_weight_float * 100.0
        row_html_list.append(
            '<tr>'
            f'<th>{html.escape(asset_name_str)}</th>'
            '<td class="composition-bar-cell">'
            f'<span class="composition-bar" style="width:{width_pct_float:.1f}%;'
            f'background:{_weight_color_for_asset(asset_name_str)}"></span>'
            '</td>'
            f'<td class="composition-bar-value">{_format_weight_ratio_str(weight_float)}</td>'
            '</tr>'
        )
    return (
        '<h3>Current Composition</h3>'
        f'<p class="metric-context">Close-marked realized weights as of {html.escape(as_of_str)}. '
        'Bar length is relative to the largest sleeve, not to 100%.</p>'
        '<table class="composition-bars"><tbody>'
        f'{"".join(row_html_list)}'
        '</tbody></table>'
    )


def _recent_taa_weight_comparison_html(strategy) -> str:
    """The last three rebalances, pivoted so one asset is one row.

    Emitted long (date, asset, target, realized, drift) this is 21 rows for a
    seven-sleeve book, and comparing one asset across three dates means finding
    three rows scattered through it. Pivoted to assets-by-date it is seven rows
    and the comparison runs horizontally -- which is the whole reason to show
    three rebalances rather than one.
    """
    recent_weight_df = _recent_taa_weight_comparison_df(strategy)
    if len(recent_weight_df) == 0:
        return ''

    rebalance_date_list = list(dict.fromkeys(recent_weight_df['date'].tolist()))
    asset_name_list = list(dict.fromkeys(recent_weight_df['asset'].tolist()))
    weight_row_by_key_dict = {
        (str(row_ser['date']), str(row_ser['asset'])): row_ser
        for _, row_ser in recent_weight_df.iterrows()
    }

    group_header_html_str = '<th></th>' + ''.join(
        f'<th colspan="3" class="rebalance-group">{html.escape(str(date_obj))}</th>'
        for date_obj in rebalance_date_list
    )
    column_header_html_str = '<th>Asset</th>' + ''.join(
        '<th>Target</th><th>Realized</th><th>Drift</th>'
        for _ in rebalance_date_list
    )

    body_row_html_list: list[str] = []
    for asset_name_str in asset_name_list:
        cell_html_list: list[str] = []
        for date_obj in rebalance_date_list:
            weight_row_ser = weight_row_by_key_dict.get((str(date_obj), str(asset_name_str)))
            if weight_row_ser is None:
                cell_html_list.append('<td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>')
                continue
            drift_weight_obj = weight_row_ser['drift_weight']
            drift_class_str = _signed_value_class_str(drift_weight_obj)
            drift_class_attr_str = f' class="{drift_class_str}"' if drift_class_str else ''
            cell_html_list.append(
                f'<td>{_format_weight_ratio_str(weight_row_ser["target_weight"])}</td>'
                f'<td>{_format_weight_ratio_str(weight_row_ser["realized_weight"])}</td>'
                f'<td{drift_class_attr_str}>{_format_weight_ratio_str(drift_weight_obj)}</td>'
            )
        body_row_html_list.append(
            f'<tr><th>{html.escape(str(asset_name_str))}</th>{"".join(cell_html_list)}</tr>'
        )

    return (
        '<h3>Recent TAA Weights - Last 3 Rebalances</h3>'
        '<p>Close-marked realized weights after execution and valuation; drift = realized_weight - target_weight.</p>'
        '<div class="scroll">'
        '<table class="rebalance-table"><thead>'
        f'<tr>{group_header_html_str}</tr>'
        f'<tr>{column_header_html_str}</tr>'
        f'</thead><tbody>{"".join(body_row_html_list)}</tbody></table>'
        '</div>'
    )


def _portfolio_weights_html(strategy) -> str:
    weights = getattr(strategy, 'daily_target_weights', None)
    if not getattr(strategy, 'show_taa_weights_report', False) or weights is None or len(weights) == 0:
        return ''

    weights = weights.copy().sort_index()
    weights.index = pd.to_datetime(weights.index)

    bear_start = pd.Timestamp('2021-01-01')
    bear_end = pd.Timestamp('2023-12-31')
    bear_weights = weights.loc[(weights.index >= bear_start) & (weights.index <= bear_end)]

    end_date = pd.Timestamp(weights.index.max()).normalize()
    trailing_start = end_date - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    trailing_weights = weights.loc[weights.index >= trailing_start]

    parts = ['<h2>Portfolio Weights</h2>']
    # Today's book first. The rebalance history and the multi-year stacks below
    # answer how it got here; this answers what it is, which is the question
    # asked first and was previously not answered anywhere in the report.
    parts.append(_current_composition_html(strategy))
    parts.append(_recent_taa_weight_comparison_html(strategy))
    parts.append(
        _weights_chart_block(
            bear_weights,
            'Portfolio Weights: 2021-2023 (Bear Market of 2022)',
            'Target allocation schedule for this corrected TAA strategy during the 2022 bear-market window.',
        )
    )
    parts.append(
        _weights_chart_block(
            trailing_weights,
            'Portfolio Weights: Last 2 Years',
            f'Trailing 24 months ending on {end_date.date()}.',
        )
    )
    return ''.join(parts)


def _portfolio_pod_drift_html(portfolio) -> str:
    vertical_line_index = getattr(portfolio, '_rebalance_date_index', pd.DatetimeIndex([]))
    parts = ['<h2>Pod Drift Diagnostics</h2>']
    rebalance_target_weight_df = getattr(portfolio, 'rebalance_target_weight_df', pd.DataFrame())
    if rebalance_target_weight_df is not None and len(rebalance_target_weight_df) > 0:
        parts.append(
            _weights_chart_block(
                rebalance_target_weight_df,
                'Target Rebalance Weights',
                'PM target weights applied on rebalance dates before subsequent pod drift.',
                vertical_line_index=vertical_line_index,
            )
        )
    parts.append(
        _weights_chart_block(
            portfolio.drift_weight_df,
            'Actual Sleeve Weights',
            'Realized pod drift weights with w_{i,t} = pod_equity_{i,t} / portfolio_equity_t.',
            vertical_line_index=vertical_line_index,
        )
    )
    parts.append(
        _chart_block_from_b64(
            _stacked_equity_chart_b64(
                portfolio._pod_equities,
                'Sleeve Equity Contributions',
                vertical_line_index=vertical_line_index,
            ),
            'Sleeve Equity Contributions',
            'Dollar sleeve contributions that sum to total portfolio equity.',
        )
    )
    parts.append(
        _chart_block_from_b64(
            _multi_line_chart_b64(
                portfolio.rolling_pairwise_correlation_df,
                'Rolling 63-Day Pairwise Correlations',
                'Correlation',
                vertical_line_index=vertical_line_index,
                ylim_tuple=(-1.05, 1.05),
            ),
            'Rolling 63-Day Pairwise Correlations',
            'Pairwise pod correlations over a 63-trading-day window.',
        )
    )

    rolling_diversification_ratio_df = None
    if portfolio.rolling_diversification_ratio_ser is not None:
        rolling_diversification_ratio_df = portfolio.rolling_diversification_ratio_ser.to_frame(
            name='Rolling Diversification Ratio'
        )

    parts.append(
        _chart_block_from_b64(
            _multi_line_chart_b64(
                rolling_diversification_ratio_df,
                'Rolling 63-Day Diversification Ratio',
                'Diversification Ratio',
                vertical_line_index=vertical_line_index,
            ),
            'Rolling 63-Day Diversification Ratio',
            'Uses realized drift weights and rolling covariance estimates.',
        )
    )
    return ''.join(parts)

def _portfolio_headline_summary_df(portfolio, summary_df) -> 'pd.DataFrame | None':
    """Two-column frame for the headline delta: portfolio vs the PM benchmark.

    The full portfolio summary's columns are ``[portfolio, sleeve1, sleeve2,
    ...]`` and the delta builder treats "the first other column" as the
    benchmark — which rendered pod #1 as BENCHMARK in the report header. Build
    the comparison frame explicitly from the attached PM benchmark instead;
    without one, return only the portfolio column so the caller falls back to
    the metric tiles rather than comparing against a sleeve.
    """
    if summary_df is None or portfolio.name not in getattr(summary_df, 'columns', []):
        return None
    headline_df = summary_df[[portfolio.name]].copy()

    benchmark_value_ser = getattr(portfolio, 'regression_benchmark_value_ser', None)
    if benchmark_value_ser is None:
        return headline_df
    aligned_benchmark_ser = (
        benchmark_value_ser.astype(float).reindex(portfolio.results.index).dropna()
    )
    if len(aligned_benchmark_ser) < 2:
        return headline_df

    benchmark_label_str = (
        getattr(portfolio, 'regression_benchmark_label_str', None) or 'Benchmark'
    )
    # Mirrors the strategy report: the benchmark column's Correlation is its
    # correlation to the entity under review.
    headline_df[benchmark_label_str] = generate_overall_metrics(
        aligned_benchmark_ser,
        series_to_correlate=(
            portfolio.results['total_value'].astype(float).pct_change(fill_method=None)
        ),
    )
    return headline_df


def _build_portfolio_html(portfolio, chart_b64: str) -> str:
    summ = portfolio.summary
    start_val = summ.loc['Start', portfolio.name]
    end_val = summ.loc['End', portfolio.name]
    start_str = str(start_val.date()) if isinstance(start_val, pd.Timestamp) else str(start_val)
    end_str = str(end_val.date()) if isinstance(end_val, pd.Timestamp) else str(end_val)
    capital_base = summ.loc['Start [$]', portfolio.name]
    final_val = summ.loc['Final [$]', portfolio.name]
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # weight allocation table
    weight_rows = ''
    for pod_info_dict in portfolio.pod_info_list:
        weight_rows += (
            f'<tr><td>{pod_info_dict.get("strategy_name", "")}</td>'
            f'<td>{pod_info_dict.get("weight", 0):.1%}</td>'
            f'<td>{_fmt_dollar(pod_info_dict.get("allocated_capital", ""))}</td></tr>'
        )
    weight_table = (
        '<table><thead><tr><th>Pod</th><th>Weight</th><th>Capital</th></tr></thead>'
        f'<tbody>{weight_rows}</tbody></table>'
    )

    # per-pod sections
    pod_sections = ''
    for i, s in enumerate(portfolio.strategies):
        pct_label = f"{portfolio.weights[i]:.0%}"
        pod_name = f"{s.name} ({pct_label})"
        sleeve_col_name_str = f"{s.name} Sleeve ({pct_label})"

        allocated_summary_html = ''
        if (
            hasattr(portfolio, 'sleeve_summary')
            and portfolio.sleeve_summary is not None
            and sleeve_col_name_str in portfolio.sleeve_summary.columns
        ):
            allocated_regression_metadata_by_column_dict = {
                sleeve_col_name_str: portfolio.benchmark_regression_metadata_by_column_dict.get(
                    sleeve_col_name_str,
                    {},
                )
            }
            allocated_summary_table_html_str = _format_performance_summary(
                portfolio.sleeve_summary[[sleeve_col_name_str]],
                allocated_regression_metadata_by_column_dict,
            )
            allocated_summary_html = (
                '<h3>Allocated Sleeve Performance — PM Window</h3>'
                f'{allocated_summary_table_html_str}'
            )

        pod_sections += _wrap_card_html(
            f'''
<h2>Allocated Sleeve — {pod_name}</h2>
{allocated_summary_html}
''',
            card_class_str='card-pod',
        )

    header_html_str = _build_report_header_html(
        report_kind_str='Portfolio Report',
        report_name_str=portfolio.name,
        run_date_str=run_date,
        start_str=start_str,
        end_str=end_str,
        capital_base_obj=capital_base,
        final_value_obj=final_val,
    )
    kpi_grid_html_str = _build_kpi_grid_html(
        summ,
        portfolio.name,
        portfolio.benchmark_regression_metadata_by_column_dict,
    )
    spec_headline_metrics_html_str = (
        _build_headline_delta_table_html(
            _portfolio_headline_summary_df(portfolio, summ), portfolio.name
        )
        or kpi_grid_html_str
    )
    # Raw section content, wrapped as cards below or emitted as plates by the
    # spec body builder.
    pm_allocation_content_html_str = _build_pm_allocation_html(portfolio)
    equity_content_html_str = f'''
<h2>Equity Curve</h2>
<div class="chart-wrap">
  <img src="data:image/png;base64,{chart_b64}" alt="Portfolio Equity Curve">
</div>
'''
    weight_allocation_content_html_str = f'''
<h2>Weight Allocation</h2>
{weight_table}
'''
    provenance_content_html_str = _build_provenance_html(portfolio)
    performance_summary_content_html_str = f'''
<h2>Performance Summary</h2>
{_format_performance_summary(
    _augment_summary_display_metrics(portfolio, summ),
    portfolio.benchmark_regression_metadata_by_column_dict,
    extra_section_html_str=_trade_statistics_section_html(
        portfolio,
        '<p class="metric-context">Completed pod trades whose full entry-to-exit lifecycle '
        'falls inside the common PM reporting window.</p>',
    ),
)}
'''
    pm_allocation_card_html_str = _wrap_card_html(pm_allocation_content_html_str)
    equity_card_html_str = _wrap_card_html(
        equity_content_html_str, card_class_str='card-primary'
    )
    weight_allocation_card_html_str = _wrap_card_html(weight_allocation_content_html_str)
    provenance_card_html_str = _wrap_card_html(provenance_content_html_str)
    performance_summary_card_html_str = _wrap_card_html(performance_summary_content_html_str)
    monthly_returns_card_html_str = _wrap_card_html(
        f'''
<h2>Portfolio Monthly Returns</h2>
<div class="scroll">{_monthly_returns_html(portfolio.monthly_returns)}</div>
''',
        card_class_str='card-monthly-returns',
    )
    benchmark_monthly_returns_df = getattr(portfolio, 'benchmark_monthly_returns', None)
    benchmark_monthly_label_str = str(
        getattr(portfolio, 'regression_benchmark_label_str', None)
        or 'Benchmark'
    )
    benchmark_monthly_body_html_str = (
        f'<div class="scroll">{_monthly_returns_html(benchmark_monthly_returns_df)}</div>'
        if benchmark_monthly_returns_df is not None and len(benchmark_monthly_returns_df) > 0
        else '<p>N/A — PM performance benchmark data is unavailable for this reporting window.</p>'
    )
    benchmark_monthly_returns_card_html_str = _wrap_card_html(
        f'''
<h2>Benchmark Portfolio Monthly Returns — {html.escape(benchmark_monthly_label_str)}</h2>
{benchmark_monthly_body_html_str}
''',
        card_class_str='card-monthly-returns',
    )
    diagnostics_card_html_str = _wrap_card_html(_build_diagnostics_html(portfolio))
    tail_risk_card_html_str = _wrap_card_html(_build_tail_risk_html(portfolio))
    pod_drift_card_html_str = _wrap_card_html(_portfolio_pod_drift_html(portfolio))
    pooled_trade_distribution_card_html_str = _wrap_card_html(
        _build_trade_distribution_html(
            portfolio._trades,
            'PM-Window Pod Trade Distribution',
        )
    )

    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'spec':
        # A portfolio is a strategy that happens to be made of pods, so it
        # leads with the same devices in the same order, then adds the
        # book-level ones. Per-pod detail is linked, not restated: each pod
        # already has its own full report beside this file.
        portfolio_monthly_content_html_str = _build_signature_monthly_returns_html(portfolio)
        body = _build_spec_report_body_html(
            strategy=portfolio,
            run_date_str=run_date,
            start_str=start_str,
            end_str=end_str,
            capital_base_obj=capital_base,
            final_value_obj=final_val,
            headline_metrics_html_str=spec_headline_metrics_html_str,
            report_kind_str='Portfolio Report',
            plate_content_html_list=[
                equity_content_html_str,
                _build_annual_paths_plate_html(portfolio),
                (
                    f'<h2>Monthly Returns</h2>{portfolio_monthly_content_html_str}'
                    if portfolio_monthly_content_html_str else ''
                ),
                _build_relative_performance_plate_html(portfolio),
                performance_summary_content_html_str,
                _build_conditional_beta_plate_html(portfolio),
                _build_diagnostics_html(portfolio),
                _build_tail_risk_html(portfolio),
                # Allocation and drift are one operational question: what the
                # book targets, what it actually holds, and the gap between.
                _merge_plate_sections_html('Allocation & Drift', [
                    pm_allocation_content_html_str,
                    weight_allocation_content_html_str,
                    _portfolio_pod_drift_html(portfolio),
                ]),
                _build_pod_report_links_html(portfolio),
                _collapse_plate_body_html(
                    provenance_content_html_str, 'Show data sources and run windows'
                ),
            ],
        )
    else:
        body = f'''<div class="report-shell">
{header_html_str}
{kpi_grid_html_str}
{pm_allocation_card_html_str}
{equity_card_html_str}
{_build_card_grid_html([weight_allocation_card_html_str, provenance_card_html_str])}
{performance_summary_card_html_str}
{monthly_returns_card_html_str}
{benchmark_monthly_returns_card_html_str}
{diagnostics_card_html_str}
{tail_risk_card_html_str}
{pod_drift_card_html_str}
{pooled_trade_distribution_card_html_str}
<div class="section-stack">{pod_sections}</div>
</div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{portfolio.name} \u2014 Portfolio Report</title>
{build_report_font_head_html()}
<style>{build_report_css()}</style>
</head>
<body>
{body}
{_METRIC_TOOLTIP_HTML_STR}
{_METRIC_TOOLTIP_SCRIPT_STR}
</body>
</html>'''


_PLATE_CAPTION_PATTERN = re.compile(r'<p class="metric-context">(.*?)</p>', re.S)


def _hoist_plate_captions_html(plate_content_html_str: str) -> str:
    """Move a plate's explanatory captions into a hover marker on its heading.

    Long captions under every figure cost more vertical space than the charts
    they describe. The text is not deleted — it moves into the same info marker
    the metric tables already use, so it is one hover away and the figures get
    the room back.

    *** CRITICAL*** Captions carrying emphasis are left in place. Those are the
    caveats and data warnings — a non-finite trade ledger, a biased estimate —
    and a warning that has to be hovered to be discovered is not a warning.
    """
    caption_match_list = _PLATE_CAPTION_PATTERN.findall(plate_content_html_str)
    hoistable_caption_list = [
        caption_html_str for caption_html_str in caption_match_list
        if '<strong>' not in caption_html_str
    ]
    if len(hoistable_caption_list) == 0:
        return plate_content_html_str

    for caption_html_str in hoistable_caption_list:
        plate_content_html_str = plate_content_html_str.replace(
            f'<p class="metric-context">{caption_html_str}</p>', '', 1
        )

    caption_text_str = ' '.join(
        ' '.join(html.unescape(re.sub(r'<[^>]+>', '', caption_html_str)).split())
        for caption_html_str in hoistable_caption_list
    )
    escaped_caption_str = html.escape(caption_text_str, quote=True)
    marker_html_str = (
        f' <button type="button" class="metric-help" aria-label="{escaped_caption_str}" '
        f'aria-expanded="false" data-help="{escaped_caption_str}">i</button>'
    )
    return re.sub(
        r'(<h2>.*?)(</h2>)',
        lambda m: f'{m.group(1)}{marker_html_str}{m.group(2)}',
        plate_content_html_str,
        count=1,
        flags=re.S,
    )


def _plate_anchor_id_str(plate_index_int: int) -> str:
    return f'plate-{plate_index_int:02d}'


def _plate_title_str(plate_content_html_str: str) -> str:
    """Pull a plate's heading out of its own content for the index."""
    heading_match = re.search(r'<h2>(.*?)</h2>', plate_content_html_str, re.S)
    if heading_match is None:
        return ''
    return html.unescape(re.sub(r'<[^>]+>', '', heading_match.group(1))).strip()


def _build_plate_index_html(plate_content_html_list: list[str]) -> str:
    """List the plates up front so a long sheet can be navigated.

    A specimen sheet is read by scrolling, not by hiding sections behind tabs;
    an index gives the document a visible shape without putting any evidence
    out of sight, and it survives printing because it is plain anchors.
    """
    entry_html_list = []
    for plate_index_int, plate_content_html_str in enumerate(plate_content_html_list, start=1):
        plate_title_str = _plate_title_str(plate_content_html_str)
        if not plate_title_str:
            continue
        entry_html_list.append(
            f'<li><a href="#{_plate_anchor_id_str(plate_index_int)}">'
            f'{html.escape(plate_title_str)}</a></li>'
        )
    if len(entry_html_list) == 0:
        return ''
    return (
        '<nav class="plate-index"><ol>' + ''.join(entry_html_list) + '</ol></nav>'
    )


def _collapse_plate_body_html(section_html_str: str, summary_label_str: str) -> str:
    """Keep a plate's heading visible but fold its body behind a summary.

    Used for reference material — an audit trail is something you go looking
    for, not something that should sit between two things you read.
    """
    if not section_html_str or not section_html_str.strip():
        return ''
    heading_match = re.search(r'(<h2>.*?</h2>)', section_html_str, re.S)
    if heading_match is None:
        return section_html_str
    heading_html_str = heading_match.group(1)
    body_html_str = section_html_str.replace(heading_html_str, '', 1)
    return (
        f'{heading_html_str}'
        f'<details class="summary-details"><summary>{html.escape(summary_label_str)}</summary>'
        f'{body_html_str}</details>'
    )


def _merge_plate_sections_html(plate_title_str: str, section_html_list: list[str]) -> str:
    """Combine related sections into one plate under a single heading.

    Each source section carries its own ``<h2>``; those are demoted to ``<h3>``
    so the plate has exactly one title. Without this the plate index — which
    reads the first heading — would silently label the plate after whichever
    section happened to come first.
    """
    demoted_html_list = []
    for section_html_str in section_html_list:
        if not section_html_str or not section_html_str.strip():
            continue
        demoted_html_list.append(
            section_html_str.replace('<h2>', '<h3>').replace('</h2>', '</h3>')
        )
    if len(demoted_html_list) == 0:
        return ''
    return f'<h2>{html.escape(plate_title_str)}</h2>' + ''.join(demoted_html_list)


def _build_pod_report_links_html(portfolio) -> str:
    """Link each pod to its own full report rather than restating it here.

    ``save_portfolio_results`` already writes a complete report for every pod
    beside the portfolio's own, and each of those carries the full device set.
    Summarising them again in the book-level sheet duplicates a better
    artifact, so this points at them instead.
    """
    pod_info_list = list(getattr(portfolio, 'pod_info_list', None) or [])
    if len(pod_info_list) == 0:
        return ''

    row_html_list = []
    for pod_info_dict in pod_info_list:
        # *** CRITICAL*** The artifact directory is the pod id, not the
        # strategy name. Linking by strategy name produces a dead link for
        # every pod whose id differs from the strategy it runs.
        pod_id_str = str(pod_info_dict.get('pod_id_str') or '').strip()
        if not pod_id_str:
            continue
        strategy_name_str = str(pod_info_dict.get('strategy_name') or pod_id_str)
        weight_obj = pod_info_dict.get('weight')
        weight_str = f'{float(weight_obj) * 100:.0f}%' if weight_obj is not None else ''
        capital_obj = pod_info_dict.get('allocated_capital')
        capital_str = _fmt_dollar(capital_obj) if capital_obj is not None else ''
        # The effective start is what the pod actually traded from, which can
        # lag the requested start when its data begins later.
        start_str = str(
            pod_info_dict.get('effective_backtest_start_date_str')
            or pod_info_dict.get('backtest_start_date_str')
            or ''
        )
        pod_report_path_str = f'pods/{pod_id_str}/report.html'
        row_html_list.append(
            f'<tr><td class="metric">{html.escape(pod_id_str)}</td>'
            f'<td>{html.escape(strategy_name_str)}</td>'
            f'<td>{weight_str}</td><td>{capital_str}</td>'
            f'<td>{html.escape(start_str)}</td>'
            f'<td><a href="{html.escape(pod_report_path_str)}">open report</a></td></tr>'
        )
    if len(row_html_list) == 0:
        return ''

    return (
        '<h2>Pods</h2>'
        '<div class="scroll"><table class="stats-table">'
        '<thead><tr><th>Pod</th><th>Strategy</th><th>Weight</th><th>Capital</th>'
        '<th>Start</th><th>Full report</th></tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table></div>'
        '<p class="metric-context">The one place each pod is identified. Every pod is also '
        'written as its own complete report beside this one, carrying the same devices at '
        'strategy level, so per-pod detail is one click away rather than restated here.</p>'
    )


def _compact_policy_str(policy_obj: object) -> str:
    if not isinstance(policy_obj, dict) or len(policy_obj) == 0:
        return 'Not recorded'
    return ' · '.join(
        f'{key_str}={value_obj}' for key_str, value_obj in sorted(policy_obj.items())
    )


def _build_strategy_audit_html(strategy) -> str:
    """Human-readable provenance already saved beside the report as metadata."""
    benchmark_list = list(getattr(strategy, '_benchmarks', []) or [])
    audit_pair_list = [
        ('Implementation', f'{strategy.__class__.__module__}.{strategy.__class__.__name__}'),
        (
            'Execution provenance',
            'Read from the strategy implementation; this report does not infer fill timing',
        ),
        ('Benchmarks', ', '.join(str(item_obj) for item_obj in benchmark_list) or 'None'),
        (
            'Data adjustment',
            _compact_policy_str(getattr(strategy, '_data_adjustment_policy_dict', {})),
        ),
        (
            'Accounting',
            _compact_policy_str(getattr(strategy, '_accounting_policy_dict', {})),
        ),
        ('Scope', 'Research artifact only; this report does not assert LIVE readiness'),
    ]
    row_html_str = ''.join(
        '<tr><td class="metric">'
        f'{html.escape(label_str)}</td><td>{html.escape(value_str)}</td></tr>'
        for label_str, value_str in audit_pair_list
    )
    return (
        '<h2>Audit &amp; Provenance</h2>'
        '<div class="scroll"><table class="stats-table"><tbody>'
        f'{row_html_str}</tbody></table></div>'
        '<p class="metric-context">These fields describe the saved research run. '
        'They do not change strategy behavior or execution semantics.</p>'
    )


def _build_spec_report_body_html(
    strategy,
    run_date_str: str,
    start_str: str,
    end_str: str,
    capital_base_obj,
    final_value_obj,
    headline_metrics_html_str: str,
    plate_content_html_list: list[str],
    report_kind_str: str = 'Strategy Report',
) -> str:
    """Assemble the specimen-sheet body: provenance masthead, then numbered plates.

    Reuses each section's existing content unchanged; the plate frame and its
    number are supplied by CSS (see theme._build_spec_layout_css). Empty
    sections (e.g. no weights for a single-asset book) are dropped so the plate
    sequence has no gaps.
    """
    # The strategy name is the page title above, so the masthead carries only
    # what the title cannot: the window, the capital path, and the run stamp.
    masthead_field_list = [
        ('Period', f'{start_str} → {end_str}'),
        ('Capital', f'{_fmt_dollar(capital_base_obj)} → {_fmt_dollar(final_value_obj)}'),
        ('Run', run_date_str),
    ]
    masthead_html_str = ''.join(
        f'<div class="spec-field"><div class="spec-field-label">{html.escape(str(label_str))}</div>'
        f'<div class="spec-field-value">{html.escape(str(value_str))}</div></div>'
        for label_str, value_str in masthead_field_list
    )
    active_plate_content_html_list = [
        content_html_str for content_html_str in plate_content_html_list
        if content_html_str and content_html_str.strip()
    ]
    plate_html_str = ''.join(
        f'<div class="plate" id="{_plate_anchor_id_str(plate_index_int)}">'
        f'{_hoist_plate_captions_html(content_html_str)}</div>'
        for plate_index_int, content_html_str in enumerate(active_plate_content_html_list, start=1)
    )
    return f'''<div class="report-shell">
<header class="report-header">
  <div class="report-eyebrow">{html.escape(report_kind_str)}</div>
  <h1>{html.escape(str(strategy.name))}</h1>
</header>
<div class="spec-masthead">{masthead_html_str}</div>
{_build_plate_index_html(active_plate_content_html_list)}
{headline_metrics_html_str}
{plate_html_str}
</div>'''


def _build_html(strategy, chart_b64: str) -> str:
    summ = strategy.summary
    start_val = summ.loc['Start', 'Strategy']
    end_val = summ.loc['End', 'Strategy']
    start_str = str(start_val.date()) if isinstance(start_val, pd.Timestamp) else str(start_val)
    end_str = str(end_val.date()) if isinstance(end_val, pd.Timestamp) else str(end_val)
    capital_base = summ.loc['Start [$]', 'Strategy']
    final_val = summ.loc['Final [$]', 'Strategy']
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header_html_str = _build_report_header_html(
        report_kind_str='Vanilla Backtest',
        report_name_str=strategy.name,
        run_date_str=run_date,
        start_str=start_str,
        end_str=end_str,
        capital_base_obj=capital_base,
        final_value_obj=final_val,
    )
    strategy_regression_metadata_by_column_dict = getattr(
        strategy,
        'benchmark_regression_metadata_by_column_dict',
        {},
    )
    # Augment once with the display-only rows (beta, monthly vol, tails, …) so
    # both the headline and the summary see the same values.
    augmented_summary_df = _augment_summary_display_metrics(strategy, summ)
    kpi_grid_html_str = _build_kpi_grid_html(
        augmented_summary_df,
        'Strategy',
        strategy_regression_metadata_by_column_dict,
    )
    # The specimen sheet leads with the delta table: the same five figures the
    # tiles carried, but next to the benchmark and the gap, which is the
    # question the tiles left the reader to do in their head. Falls back to the
    # tiles when there is no benchmark to compare against.
    spec_headline_metrics_html_str = (
        _build_headline_delta_table_html(augmented_summary_df, 'Strategy')
        or kpi_grid_html_str
    )
    benchmark_monthly_metric_df, benchmark_label_str = _strategy_monthly_benchmark_metric_bundle(strategy)

    # Raw section content (each carrying its own <h2>), assembled below either
    # as cards (dashboard/document) or as numbered plates (spec).
    equity_content_html_str = f'''
<h2>Equity Curve</h2>
<div class="chart-wrap">
  <img src="data:image/png;base64,{chart_b64}" alt="Equity Curve">
</div>
'''
    weights_content_html_str = _portfolio_weights_html(strategy)
    performance_summary_content_html_str = f'''
<h2>Statistics</h2>
{_format_performance_summary(
    augmented_summary_df,
    strategy_regression_metadata_by_column_dict,
    extra_section_html_str=_trade_statistics_section_html(strategy),
)}
'''
    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'dashboard':
        monthly_returns_body_html_str = (
            f'<div class="scroll">{_monthly_returns_html(strategy.monthly_returns, benchmark_monthly_metric_df, benchmark_label_str)}</div>'
        )
    else:
        monthly_returns_body_html_str = _build_signature_monthly_returns_html(strategy)
    monthly_returns_content_html_str = f'''
<h2>Monthly Returns</h2>
{monthly_returns_body_html_str}
'''
    open_trades_content_html_str = f'''
<h2>Open Trades</h2>
<div class="scroll">{_format_open_trades(strategy._open_trades)}</div>
'''
    # Closed trades are long and rarely the first thing read, so the table is
    # folded behind a summary by default.
    closed_trades_content_html_str = f'''
<h2>Closed Trades</h2>
<details class="summary-details"><summary>Show closed trades</summary>
<div class="scroll">{_format_trades(strategy._trades)}</div>
</details>
'''

    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'spec':
        body = _build_spec_report_body_html(
            strategy=strategy,
            run_date_str=run_date,
            start_str=start_str,
            end_str=end_str,
            capital_base_obj=capital_base,
            final_value_obj=final_val,
            headline_metrics_html_str=spec_headline_metrics_html_str,
            plate_content_html_list=[
                equity_content_html_str,
                _build_annual_paths_plate_html(strategy),
                # Year by year and monthly returns are siblings — both are
                # performance over time — so they sit together.
                monthly_returns_content_html_str,
                _build_relative_performance_plate_html(strategy),
                # Composition owns the whole allocation story now. The
                # Portfolio Weights plate re-drew the same stack over two
                # arbitrary windows and carried the tables Composition now
                # holds, so it was answering a question already answered.
                _build_composition_plate_html(strategy),
                performance_summary_content_html_str,
                _build_conditional_beta_plate_html(strategy),
                open_trades_content_html_str,
                closed_trades_content_html_str,
                _build_strategy_audit_html(strategy),
            ],
            report_kind_str='Vanilla Backtest',
        )
    else:
        weights_card_html_str = (
            _wrap_card_html(weights_content_html_str) if weights_content_html_str else ''
        )
        body = f'''<div class="report-shell">
{header_html_str}
{kpi_grid_html_str}
{_wrap_card_html(equity_content_html_str, card_class_str='card-primary')}
{weights_card_html_str}
{_wrap_card_html(performance_summary_content_html_str)}
{_wrap_card_html(monthly_returns_content_html_str, card_class_str='card-monthly-returns')}
{_wrap_card_html(open_trades_content_html_str)}
{_wrap_card_html(closed_trades_content_html_str)}
{_wrap_card_html(_build_strategy_audit_html(strategy))}
</div>'''

    # Resolve styling at render time so it reflects whichever signature variant
    # is active \u2014 not the one frozen into module constants at import.
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy.name} \u2014 Vanilla Backtest</title>
{build_report_font_head_html()}
<style>{build_report_css()}</style>
</head>
<body>
{body}
{_METRIC_TOOLTIP_HTML_STR}
{_METRIC_TOOLTIP_SCRIPT_STR}
</body>
</html>'''


def _format_crisis_metric_table_html(crisis_metric_df: pd.DataFrame) -> str:
    if crisis_metric_df is None or len(crisis_metric_df) == 0:
        return '<p>No crisis periods were evaluated.</p>'

    display_column_spec_list = [
        ('crisis_name_str', 'Crisis'),
        ('effective_start_ts', 'Start'),
        ('effective_end_ts', 'End'),
        ('strategy_return_pct_float', 'Strategy Return'),
        ('benchmark_return_pct_float', 'Benchmark Return'),
        ('relative_return_pct_float', 'Relative Return'),
        ('max_drawdown_pct_float', 'Max Drawdown'),
        ('volatility_ann_pct_float', 'Volatility (Ann.)'),
        ('sharpe_ratio_float', 'Sharpe'),
        ('trade_count_int', 'Trades'),
    ]
    header_html_str = ''.join(
        f'<th>{header_label_str}</th>'
        for _, header_label_str in display_column_spec_list
    )
    row_html_list: list[str] = []

    for _, row_ser in crisis_metric_df[[column_name_str for column_name_str, _ in display_column_spec_list]].iterrows():
        cell_html_list: list[str] = []
        for column_name_str, _header_label_str in display_column_spec_list:
            cell_value_obj = row_ser[column_name_str]
            cell_class_str = ''
            if column_name_str.endswith('_ts'):
                cell_text_str = (
                    ''
                    if pd.isna(cell_value_obj)
                    else str(pd.Timestamp(cell_value_obj).date())
                )
            elif column_name_str.endswith('_pct_float'):
                cell_text_str = (
                    ''
                    if pd.isna(cell_value_obj)
                    else f'{float(cell_value_obj):+,.2f}%'
                )
                if column_name_str == 'relative_return_pct_float' and not pd.isna(cell_value_obj):
                    cell_class_str = 'pos' if float(cell_value_obj) >= 0.0 else 'neg'
            elif column_name_str == 'sharpe_ratio_float':
                cell_text_str = '' if pd.isna(cell_value_obj) else f'{float(cell_value_obj):,.2f}'
            elif column_name_str == 'trade_count_int':
                cell_text_str = '' if pd.isna(cell_value_obj) else str(int(cell_value_obj))
            else:
                cell_text_str = '' if pd.isna(cell_value_obj) else str(cell_value_obj)
            class_attr_str = f' class="{cell_class_str}"' if cell_class_str else ''
            cell_html_list.append(f'<td{class_attr_str}>{cell_text_str}</td>')
        row_html_list.append('<tr>' + ''.join(cell_html_list) + '</tr>')

    return (
        f'<table><thead><tr>{header_html_str}</tr></thead>'
        f'<tbody>{"".join(row_html_list)}</tbody></table>'
    )


def _crisis_path_chart_b64(
    strategy_obj,
    crisis_name_str: str,
    effective_start_ts: pd.Timestamp,
    effective_end_ts: pd.Timestamp,
) -> str | None:
    if strategy_obj is None or strategy_obj.results is None or len(strategy_obj.results) == 0:
        return None

    benchmark_name_str = None
    benchmark_drawdown_column_name_str = None
    if hasattr(strategy_obj, '_benchmarks') and len(strategy_obj._benchmarks) > 0:
        candidate_benchmark_name_str = str(strategy_obj._benchmarks[0])
        candidate_drawdown_column_name_str = f'{candidate_benchmark_name_str}_drawdown'
        if (
            candidate_benchmark_name_str in strategy_obj.results.columns
            and candidate_drawdown_column_name_str in strategy_obj.results.columns
        ):
            benchmark_name_str = candidate_benchmark_name_str
            benchmark_drawdown_column_name_str = candidate_drawdown_column_name_str

    buffer_obj = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='FigureCanvasAgg is non-interactive, and thus cannot be shown',
            category=UserWarning,
        )
        render_strategy_plot(
            strategy_total_value=strategy_obj.results['total_value'],
            strategy_drawdown=strategy_obj.results['drawdown'],
            benchmark_total_value=(
                strategy_obj.results[benchmark_name_str]
                if benchmark_name_str is not None
                else None
            ),
            benchmark_drawdown=(
                strategy_obj.results[benchmark_drawdown_column_name_str]
                if benchmark_drawdown_column_name_str is not None
                else None
            ),
            benchmark_label=benchmark_name_str or 'Benchmark',
            strategy_label='Strategy',
            save_to=buffer_obj,
            to_web=True,
            dpi=160,
            return_bar_frequency_str='monthly',
        )
    plt.close('all')
    buffer_obj.seek(0)
    return base64.b64encode(buffer_obj.read()).decode('ascii')


def _build_crisis_chart_cards_html(crisis_replay_result) -> str:
    card_html_list: list[str] = []
    for _, metric_row_ser in crisis_replay_result.crisis_metric_df.iterrows():
        crisis_name_str = str(metric_row_ser['crisis_name_str'])
        strategy_obj = crisis_replay_result.crisis_strategy_map.get(crisis_name_str)
        chart_b64 = _crisis_path_chart_b64(
            strategy_obj=strategy_obj,
            crisis_name_str=crisis_name_str,
            effective_start_ts=pd.Timestamp(metric_row_ser['effective_start_ts']),
            effective_end_ts=pd.Timestamp(metric_row_ser['effective_end_ts']),
        )
        if chart_b64 is None:
            continue

        card_html_list.append(
            _wrap_card_html(
                f'''
<h2>{crisis_name_str}</h2>
<div class="meta">
  {pd.Timestamp(metric_row_ser['effective_start_ts']).date()} &rarr; {pd.Timestamp(metric_row_ser['effective_end_ts']).date()}
</div>
<div class="chart-wrap">
  <img src="data:image/png;base64,{chart_b64}" alt="{crisis_name_str} crisis path">
</div>
''',
                card_class_str='card-primary',
            )
        )

    if len(card_html_list) == 0:
        return ''
    # Plates own the full measure, so the spec layout stacks the crisis charts
    # rather than pairing them two-up.
    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'spec':
        return ''.join(card_html_list)
    return f'<div class="crisis-chart-grid">{"".join(card_html_list)}</div>'


def _build_crisis_replay_html(crisis_replay_result) -> str:
    run_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_card_html_str = _wrap_card_html(
        f'''
<h2>Crisis Summary</h2>
<div class="scroll">{_format_crisis_metric_table_html(crisis_replay_result.crisis_metric_df)}</div>
''',
    )
    chart_cards_html_str = _build_crisis_chart_cards_html(crisis_replay_result)

    body = f'''<div class="report-shell">
<header class="report-header">
  <div class="report-eyebrow">Crisis Replay Report</div>
  <h1>{crisis_replay_result.strategy_name_str}</h1>
  <div class="meta">
    Run: {run_date_str} &nbsp;|&nbsp;
    Strategy Key: {crisis_replay_result.strategy_key_str} &nbsp;|&nbsp;
    Capital: {_fmt_dollar(crisis_replay_result.capital_base_float)}
  </div>
</header>
{summary_card_html_str}
{chart_cards_html_str}
</div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{crisis_replay_result.strategy_name_str} - Crisis Replay Report</title>
{build_report_font_head_html()}
<style>{build_report_css()}</style>
</head>
<body>
{body}
</body>
</html>'''

