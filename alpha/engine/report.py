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


_MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_METADATA_FILENAME = 'metadata.json'


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
        cells = [f'<td class="metric">{metric}</td>']
        for col in df.columns:
            cells.append(f'<td>{_fmt_cell(metric, df.loc[metric, col])}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
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
    """Map a return ratio to an inline CSS background. ????????????????????30% = full green/red."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 'background-color: #f5f5f5;'
    if np.isnan(v):
        return 'background-color: #f5f5f5;'
    intensity = min(abs(v) / 0.30, 1.0)
    if v >= 0:
        r = int(255 - intensity * 155)
        g = int(255 - intensity * 55)
        b = int(255 - intensity * 155)
    else:
        r = int(255 - intensity * 55)
        g = int(255 - intensity * 155)
        b = int(255 - intensity * 155)
    return f'background-color: rgb({r},{g},{b});'


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
                cells.append('<td style="background-color:#f5f5f5;"></td>')
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

_CSS = '''
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 20px 40px; background: #fafafa; color: #222;
}
h1 { font-size: 1.6em; margin-bottom: 4px; }
h2 {
    font-size: 1.1em; margin-top: 32px; margin-bottom: 8px;
    border-bottom: 1px solid #ddd; padding-bottom: 4px; color: #444;
}
.meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
table { border-collapse: collapse; font-size: 0.85em; width: 100%; margin-bottom: 16px; }
th { background: #f0f0f0; padding: 6px 10px; text-align: left; border: 1px solid #ddd; }
td { padding: 5px 10px; border: 1px solid #e8e8e8; }
tr:nth-child(even) td { background: #f9f9f9; }
td.metric { font-weight: 500; background: #f5f5f5; white-space: nowrap; }
td.pos { color: #1a7a1a; }
td.neg { color: #c0392b; }
.heatmap td { text-align: center; font-size: 0.8em; min-width: 48px; padding: 4px 6px; }
.chart-wrap { margin: 16px 0; }
.chart-wrap img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.scroll { overflow-x: auto; }
'''


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
    Low correlation (near 0) = green, high (near 1) = red."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ''
    if np.isnan(v):
        return 'background-color: #f5f5f5;'
    # abs correlation: 0 = full green, 1 = full red
    intensity = min(abs(v), 1.0)
    r = int(100 + intensity * 155)
    g = int(200 - intensity * 100)
    b = int(100 + intensity * 55)
    return f'background-color: rgb({r},{g},{b}); color: #fff;'


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

    if portfolio.correlation_matrix is not None:
        parts.append('<h3>Correlation Matrix</h3>')
        parts.append(f'<div class="scroll">{_format_correlation_matrix(portfolio.correlation_matrix)}</div>')

    if portfolio.diversification_ratio is not None:
        parts.append(f'<p><strong>Diversification Ratio:</strong> {portfolio.diversification_ratio:.3f}</p>')

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



def _weights_chart_b64(weights: pd.DataFrame, title: str) -> str | None:
    """Render a stacked portfolio-weights chart to base64 PNG."""
    if weights is None or len(weights) == 0:
        return None

    weights = weights.copy().sort_index()
    active_cols = [col for col in weights.columns if not np.allclose(weights[col].fillna(0).to_numpy(), 0.0)]
    if not active_cols:
        return None

    color_map = {
        'TLT': '#1f77b4',
        'GLD': '#ff7f0e',
        'DBC': '#2ca02c',
        'UUP': '#d62728',
        'SPY': '#d4a900',
        'UPRO': '#9467bd',
        'BTAL': '#8c564b',
    }
    colors = [color_map.get(col, '#7f7f7f') for col in active_cols]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.stackplot(
        weights.index,
        [weights[col].fillna(0).to_numpy() for col in active_cols],
        labels=active_cols,
        colors=colors,
        alpha=0.8,
        edgecolor='#111111',
        linewidth=0.6,
    )
    ax.set_title(title)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Date')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', ncol=min(3, len(active_cols)), fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _weights_chart_block(weights: pd.DataFrame, title: str, subtitle: str) -> str:
    chart_b64 = _weights_chart_b64(weights, title)
    if chart_b64 is None:
        return f'<h3>{title}</h3><p>{subtitle}</p><p>No weight data available for this window.</p>'
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

        pod_monthly = ''
        if hasattr(s, 'monthly_returns') and s.monthly_returns is not None:
            pod_monthly = f'<h3>Monthly Returns</h3><div class="scroll">{_monthly_returns_html(s.monthly_returns)}</div>'

        pod_trade_stats = ''
        if hasattr(s, 'summary_trades') and s.summary_trades is not None and len(s.summary_trades) > 0:
            pod_trade_stats = f'<h3>Trade Statistics</h3><div class="scroll">{_format_summary_trades(s.summary_trades)}</div>'

        pod_summary_html = ''
        if hasattr(s, 'summary') and s.summary is not None:
            pod_summary_html = f'<h3>Performance Summary</h3><div class="scroll">{_format_summary(s.summary)}</div>'

        pod_sections += f'''
<h2>{pod_name}</h2>
{pod_summary_html}
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

<h2>Portfolio Monthly Returns</h2>
<div class="scroll">{_monthly_returns_html(portfolio.monthly_returns)}</div>

<h2>Aggregated Trade Statistics</h2>
<div class="scroll">{_format_summary_trades(portfolio.summary_trades) if portfolio.summary_trades is not None and len(portfolio.summary_trades) > 0 else '<p>No trades.</p>'}</div>

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

<h2>Closed Trades</h2>
<div class="scroll">{_format_trades(strategy._trades)}</div>

<h2>All Transactions</h2>
<div class="scroll">{_format_transactions(strategy._transactions)}</div>'''

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

