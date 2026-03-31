import io
import base64
import inspect
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from alpha.engine.theme import (
    SEABORN_DEEP_COLOR_LIST,
    SIGNATURE_PALETTE_DICT,
    blend_hex_color_str,
    build_report_css,
    build_signature_rcparams,
)


_MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_METADATA_FILENAME = 'metadata.json'
_TRANSACTION_CSV_FILENAME = 'transactions.csv'
_DAILY_RETURN_HISTOGRAM_BIN_COUNT_INT = 60
_TRADE_RETURN_HISTOGRAM_BIN_COUNT_INT = 60


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_metadata(metadata_path: Path, metadata_dict: dict):
    metadata_path.write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True, default=_json_default),
        encoding='utf-8',
    )


def _write_transaction_csv(transaction_df: pd.DataFrame | None, transaction_csv_path: Path):
    if transaction_df is None:
        transaction_df = pd.DataFrame()
    transaction_export_df = transaction_df.copy()
    transaction_export_df.to_csv(transaction_csv_path, index=False, date_format='%Y-%m-%d')


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
    }


def _portfolio_metadata_dict(portfolio, pickle_path: Path) -> dict:
    return {
        'artifact_type': 'portfolio',
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'pickle_path': pickle_path.resolve(),
        'portfolio_name': portfolio.name,
        'capital_base': float(portfolio._capital_base),
        'rebalance': portfolio._rebalance,
        'source_config_path': portfolio.source_config_path,
        'common_start': portfolio._common_start,
        'common_end': portfolio._common_end,
        'pods': portfolio.pod_info_list,
    }


def save_results(strategy, output_dir='results') -> Path:
    """Save strategy results to a structured folder and generate an HTML report.

    Creates:
        {output_dir}/{strategy.name}/{YYYY-MM-DD_HHMMSS}/
            {strategy.name}.pkl
            report.html
            transactions.csv

    Returns the output directory path.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out = Path(output_dir) / strategy.name / timestamp
    out.mkdir(parents=True, exist_ok=True)

    pickle_path = out / f'{strategy.name}.pkl'
    strategy.to_pickle(pickle_path)
    _write_metadata(out / _METADATA_FILENAME, _strategy_metadata_dict(strategy, pickle_path))

    buf = io.BytesIO()
    strategy.plot(save_to=buf)
    plt.close('all')
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('ascii')

    html = _build_html(strategy, chart_b64)
    (out / 'report.html').write_text(html, encoding='utf-8')
    _write_transaction_csv(strategy._transactions, out / _TRANSACTION_CSV_FILENAME)

    print(f'Results saved to: {out.resolve()}')
    return out


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ formatting helpers ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

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
            color=SEABORN_DEEP_COLOR_LIST[0],
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
            color=SEABORN_DEEP_COLOR_LIST[1],
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

    stats_table_html = (
        '<table class="stats-table"><thead><tr>'
        '<th>Mean</th><th>Std. Dev.</th><th>Skew</th><th>Negative Days</th>'
        '</tr></thead><tbody><tr>'
        f'<td>{mean_return_float:.3%}</td>'
        f'<td>{std_return_float:.3%}</td>'
        f'<td>{_fmt_num(skew_return_float, 3) if np.isfinite(skew_return_float) else "N/A"}</td>'
        f'<td>{negative_rate_float:.2%}</td>'
        '</tr></tbody></table>'
    )

    return (
        '<h2>Daily Return Distribution</h2>'
        f'<div class="chart-wrap"><img src="data:image/png;base64,{histogram_b64}" '
        'alt="Daily Return Distribution"></div>'
        f'{stats_table_html}'
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
            color=SEABORN_DEEP_COLOR_LIST[0],
            alpha=0.72,
            edgecolor=SIGNATURE_PALETTE_DICT['bar_edge'],
            linewidth=0.60,
                label='Losing trades',
            )
        if winning_trade_return_vec.size > 0:
            axis_obj.hist(
            winning_trade_return_vec,
            bins=bin_edge_vec,
            color=SEABORN_DEEP_COLOR_LIST[1],
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
            color=SEABORN_DEEP_COLOR_LIST[2],
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
                color=SEABORN_DEEP_COLOR_LIST[0],
                alpha=0.58,
                s=20,
                edgecolors='none',
                label='Losing trades',
            )
        if winning_mask_vec.any():
            axis_obj.scatter(
                trade_duration_day_vec[winning_mask_vec],
                trade_return_vec[winning_mask_vec],
                color=SEABORN_DEEP_COLOR_LIST[1],
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

    return (
        f'<h2>{section_title_str}</h2>'
        f'{chart_grid_html}'
        f'<div class="scroll">{_trade_distribution_summary_table_html(distribution_dict)}</div>'
        f'<div class="scroll">{_loss_tail_summary_table_html(distribution_dict)}</div>'
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


def _format_summary(df: pd.DataFrame) -> str:
    """Render strategy summary DataFrame as an HTML table."""
    headers = '<th>Metric</th>' + ''.join(f'<th>{c}</th>' for c in df.columns)
    rows = []
    for metric in df.index:
        row_class_str = ' class="summary-row-sharpe"' if metric == 'Sharpe Ratio' else ''
        metric_class_str = 'metric metric-sharpe' if metric == 'Sharpe Ratio' else 'metric'
        cells = [f'<td class="{metric_class_str}">{metric}</td>']
        for col in df.columns:
            cells.append(f'<td>{_fmt_cell(metric, df.loc[metric, col])}</td>')
        rows.append(f'<tr{row_class_str}>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


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
                cls = 'pos' if val >= 0 else 'neg'
                cells.append(f'<td class="{cls}">{_fmt_dollar(val)}</td>')
            elif col == 'return':
                cls = 'pos' if val >= 0 else 'neg'
                cells.append(f'<td class="{cls}">{_fmt_pct(val * 100)}</td>')
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
            SIGNATURE_PALETTE_DICT['strategy'],
            fill_weight_float,
        )
        font_color_str = (
            SIGNATURE_PALETTE_DICT['strategy_dark']
            if intensity_float < 0.65 else SIGNATURE_PALETTE_DICT['page']
        )
    else:
        background_color_str = blend_hex_color_str(
            SIGNATURE_PALETTE_DICT['page'],
            SIGNATURE_PALETTE_DICT['benchmark'],
            fill_weight_float,
        )
        font_color_str = (
            SIGNATURE_PALETTE_DICT['benchmark_dark']
            if intensity_float < 0.65 else SIGNATURE_PALETTE_DICT['page']
        )

    return f'background-color: {background_color_str}; color: {font_color_str};'


def _monthly_returns_html(mr: pd.DataFrame) -> str:
    """Build the monthly returns heatmap as a raw HTML string."""
    month_cols = [m for m in range(1, 13) if m in mr.columns]
    extra_cols = [c for c in mr.columns if c not in range(1, 13)]

    month_headers = ''.join(f'<th>{_MONTH_NAMES[m - 1]}</th>' for m in month_cols)
    extra_headers = ''.join(f'<th>{c}</th>' for c in extra_cols)
    header = f'<tr><th>Year</th>{month_headers}{extra_headers}</tr>'

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
            if pd.isna(val) or val == '':
                cells.append('<td></td>')
            elif c == 'Annual Return':
                cells.append(f'<td style="{_ret_color(val)}">{val:.1%}</td>')
            elif c == 'Sharpe Ratio':
                cells.append(f'<td>{float(val):.2f}</td>')
            elif c == 'Max Drawdown':
                cells.append(f'<td>{val:.1%}</td>')
            else:
                cells.append(f'<td>{val}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    return (f'<table class="heatmap"><thead>{header}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ HTML assembly ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

_CSS = build_report_css()


def save_portfolio_results(portfolio, output_dir='results') -> Path:
    """Save portfolio results to a structured folder and generate an HTML report.

    Creates:
        {output_dir}/{portfolio.name}/{YYYY-MM-DD_HHMMSS}/
            {portfolio.name}.pkl
            report.html

    Returns the output directory path.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out = Path(output_dir) / portfolio.name / timestamp
    out.mkdir(parents=True, exist_ok=True)

    pickle_path = out / f'{portfolio.name}.pkl'
    portfolio.to_pickle(pickle_path)
    _write_metadata(out / _METADATA_FILENAME, _portfolio_metadata_dict(portfolio, pickle_path))

    buf = io.BytesIO()
    portfolio.plot(save_to=buf)
    plt.close('all')
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('ascii')

    html = _build_portfolio_html(portfolio, chart_b64)
    (out / 'report.html').write_text(html, encoding='utf-8')

    print(f'Results saved to: {out.resolve()}')
    return out


def _corr_color(val) -> str:
    """Map a correlation value to an inline CSS background.
    Low correlation (near 0) = teal, high (near 1) = muted rose."""
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
    low_corr_color_str = blend_hex_color_str(
        SIGNATURE_PALETTE_DICT['page'],
        SIGNATURE_PALETTE_DICT['strategy'],
        0.30,
    )
    high_corr_color_str = blend_hex_color_str(
        SIGNATURE_PALETTE_DICT['page'],
        SIGNATURE_PALETTE_DICT['benchmark'],
        0.52,
    )
    background_color_str = blend_hex_color_str(
        low_corr_color_str,
        high_corr_color_str,
        intensity_float,
    )
    font_color_str = (
        SIGNATURE_PALETTE_DICT['ink']
        if intensity_float < 0.72 else SIGNATURE_PALETTE_DICT['page']
    )
    return f'background-color: {background_color_str}; color: {font_color_str};'


def _format_correlation_matrix(corr: 'pd.DataFrame') -> str:
    """Render a correlation matrix as a color-coded HTML table."""
    headers = '<th></th>' + ''.join(f'<th>{c}</th>' for c in corr.columns)
    rows = []
    for row_label in corr.index:
        cells = [f'<td class="metric">{row_label}</td>']
        for col_label in corr.columns:
            val = corr.loc[row_label, col_label]
            style = _corr_color(val)
            cells.append(f'<td style="{style} text-align:center;">{val:.3f}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _build_diagnostics_html(portfolio) -> str:
    """Build the Cross-Strategy Diagnostics HTML section."""
    parts = ['<h2>Cross-Strategy Diagnostics</h2>']

    if portfolio.correlation_matrix is not None and len(portfolio.correlation_matrix) > 0:
        parts.append('<h3>Correlation Matrix</h3>')
        parts.append(f'<div class="scroll">{_format_correlation_matrix(portfolio.correlation_matrix)}</div>')

    if portfolio.target_diversification_ratio is not None and np.isfinite(portfolio.target_diversification_ratio):
        parts.append(
            f'<p><strong>Target-Weight Diversification Ratio:</strong> '
            f'{portfolio.target_diversification_ratio:.3f}</p>'
        )

    if portfolio.realized_diversification_ratio is not None and np.isfinite(portfolio.realized_diversification_ratio):
        parts.append(
            f'<p><strong>End-Weight Diversification Ratio:</strong> '
            f'{portfolio.realized_diversification_ratio:.3f}</p>'
        )

    if (
        portfolio.average_rolling_diversification_ratio is not None
        and np.isfinite(portfolio.average_rolling_diversification_ratio)
    ):
        parts.append(
            f'<p><strong>Average Rolling Diversification Ratio (63d):</strong> '
            f'{portfolio.average_rolling_diversification_ratio:.3f}</p>'
        )

    if portfolio._rebalance is not None:
        parts.append(f'<p><strong>Rebalance Frequency:</strong> {portfolio._rebalance}</p>')
    else:
        parts.append('<p><strong>Rebalance Frequency:</strong> None (buy-and-hold)</p>')

    return '\n'.join(parts)


def _build_provenance_html(portfolio) -> str:
    """Build a provenance section for portfolio configuration and sources."""
    rows = []
    for pod_info_dict in portfolio.pod_info_list:
        rows.append(
            '<tr>'
            f'<td>{pod_info_dict.get("strategy_name", "")}</td>'
            f'<td>{pod_info_dict.get("weight", 0):.1%}</td>'
            f'<td>{_fmt_dollar(pod_info_dict.get("allocated_capital", ""))}</td>'
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
        '<table><thead><tr><th>Pod</th><th>Weight</th><th>Allocated Capital</th><th>Source Pickle</th></tr></thead>'
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

    color_list = [
        SEABORN_DEEP_COLOR_LIST[column_idx_int % len(SEABORN_DEEP_COLOR_LIST)]
        for column_idx_int, _ in enumerate(active_cols)
    ]

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.stackplot(
            weights.index,
            [weights[col].fillna(0).to_numpy() for col in active_cols],
            labels=active_cols,
            colors=color_list,
            alpha=0.82,
            edgecolor=SIGNATURE_PALETTE_DICT['page'],
            linewidth=0.6,
        )
        ax.set_title(title)
        ax.set_ylabel('Weight')
        ax.set_xlabel('Date')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True)
        _add_vertical_line_markers(ax, vertical_line_index)
        ax.legend(loc='upper left', ncol=min(3, len(active_cols)), fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


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
        SEABORN_DEEP_COLOR_LIST[column_idx_int % len(SEABORN_DEEP_COLOR_LIST)]
        for column_idx_int, _ in enumerate(active_col_list)
    ]

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.stackplot(
            pod_equity_df.index,
            [pod_equity_df[column_str].fillna(0.0).to_numpy() for column_str in active_col_list],
            labels=active_col_list,
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
        ax.legend(loc='upper left', ncol=min(3, len(active_col_list)), fontsize=8)
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
                label=column_str,
                color=SEABORN_DEEP_COLOR_LIST[column_idx_int % len(SEABORN_DEEP_COLOR_LIST)],
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
        ax.legend(loc='upper left', ncol=min(3, len(plot_df.columns)), fontsize=8)
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

        pod_monthly = ''
        if hasattr(s, 'monthly_returns') and s.monthly_returns is not None:
            pod_monthly = f'<h3>Monthly Returns</h3><div class="scroll">{_monthly_returns_html(s.monthly_returns)}</div>'

        pod_trade_stats = ''
        if hasattr(s, 'summary_trades') and s.summary_trades is not None and len(s.summary_trades) > 0:
            pod_trade_stats = f'<h3>Trade Statistics</h3><div class="scroll">{_format_summary_trades(s.summary_trades)}</div>'

        allocated_summary_html = ''
        if (
            hasattr(portfolio, 'sleeve_summary')
            and portfolio.sleeve_summary is not None
            and sleeve_col_name_str in portfolio.sleeve_summary.columns
        ):
            allocated_summary_html = (
                '<h3>Allocated Sleeve Summary</h3>'
                f'<div class="scroll">{_format_summary(portfolio.sleeve_summary[[sleeve_col_name_str]])}</div>'
            )

        standalone_summary_html = ''
        if hasattr(s, 'summary') and s.summary is not None:
            standalone_summary_html = (
                '<h3>Standalone Pod Summary</h3>'
                f'<div class="scroll">{_format_summary(s.summary)}</div>'
            )

        pod_sections += f'''
<h2>{pod_name}</h2>
{allocated_summary_html}
{standalone_summary_html}
{pod_monthly}
{pod_trade_stats}
'''

    body = f'''<h1>{portfolio.name}</h1>
<div class="meta">
  Run: {run_date} &nbsp;|&nbsp; Period: {start_str} &rarr; {end_str} &nbsp;|&nbsp;
  Capital: {_fmt_dollar(capital_base)} &rarr; {_fmt_dollar(final_val)}
</div>

<h2>Weight Allocation</h2>
{weight_table}

<h2>Equity Curve</h2>
<div class="chart-wrap">
  <img src="data:image/png;base64,{chart_b64}" alt="Portfolio Equity Curve">
</div>

<h2>Portfolio Performance Summary</h2>
<div class="scroll">{_format_summary(summ)}</div>

{_build_provenance_html(portfolio)}

{_build_diagnostics_html(portfolio)}

{_portfolio_pod_drift_html(portfolio)}

<h2>Portfolio Monthly Returns</h2>
<div class="scroll">{_monthly_returns_html(portfolio.monthly_returns)}</div>

<h2>Pooled Pod Trade Statistics</h2>
<div class="scroll">{_format_summary_trades(portfolio.summary_trades) if portfolio.summary_trades is not None and len(portfolio.summary_trades) > 0 else '<p>No trades.</p>'}</div>

{_build_trade_distribution_html(portfolio._trades, 'Pooled Pod Trade Distribution')}

{pod_sections}'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{portfolio.name} \u2014 Portfolio Report</title>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>'''


def _build_html(strategy, chart_b64: str) -> str:
    summ = strategy.summary
    start_val = summ.loc['Start', 'Strategy']
    end_val = summ.loc['End', 'Strategy']
    start_str = str(start_val.date()) if isinstance(start_val, pd.Timestamp) else str(start_val)
    end_str = str(end_val.date()) if isinstance(end_val, pd.Timestamp) else str(end_val)
    capital_base = summ.loc['Start [$]', 'Strategy']
    final_val = summ.loc['Final [$]', 'Strategy']
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    body = f'''<h1>{strategy.name}</h1>
<div class="meta">
  Run: {run_date} &nbsp;|&nbsp; Period: {start_str} &rarr; {end_str} &nbsp;|&nbsp;
  Capital: {_fmt_dollar(capital_base)} &rarr; {_fmt_dollar(final_val)}
</div>

<h2>Equity Curve</h2>
<div class="chart-wrap">
  <img src="data:image/png;base64,{chart_b64}" alt="Equity Curve">
</div>

 {_portfolio_weights_html(strategy)}

<h2>Performance Summary</h2>
<div class="scroll">{_format_summary(summ)}</div>

<h2>Monthly Returns</h2>
<div class="scroll">{_monthly_returns_html(strategy.monthly_returns)}</div>

<h2>Trade Statistics</h2>
<div class="scroll">{_format_summary_trades(strategy.summary_trades)}</div>

{_build_trade_distribution_html(strategy._trades, 'Trade Return Distribution')}

{_build_daily_return_distribution_html(strategy)}

<h2>Closed Trades</h2>
<div class="scroll">{_format_trades(strategy._trades)}</div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy.name} \u2014 Strategy Report</title>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>'''

