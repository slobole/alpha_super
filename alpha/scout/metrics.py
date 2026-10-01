"""The S4 metric set on a daily return series (zero risk-free rate for Sharpe, per QUANT_PHILOSOPHY.md).

Definitions follow alpha/engine/metrics.py conventions: 252 sessions a year, Sharpe = mean / std (ddof = 1) × √252,
drawdown on the compounded value, 95% expected shortfall = mean of the worst 5% of daily returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_INT = 252


def sharpe_float(daily_ser: pd.Series) -> float:
    clean_ser = daily_ser.dropna()
    sd_float = clean_ser.std(ddof=1)
    return float(clean_ser.mean() / sd_float * np.sqrt(DAYS_INT)) if sd_float > 0 else float("nan")


def performance_dict(daily_ser: pd.Series, tbill_daily_ser: pd.Series | None = None) -> dict:
    clean_ser = daily_ser.dropna()
    if len(clean_ser) < 20:
        return {}
    value_ser = (1.0 + clean_ser).cumprod()
    years_float = len(clean_ser) / DAYS_INT
    drawdown_ser = value_ser / value_ser.cummax() - 1.0
    underwater_vec = (drawdown_ser < 0).to_numpy()
    longest_int, run_int = 0, 0
    for flag_bool in underwater_vec:
        run_int = run_int + 1 if flag_bool else 0
        longest_int = max(longest_int, run_int)
    downside_float = np.sqrt(np.mean(np.minimum(clean_ser.to_numpy(), 0.0) ** 2)) * np.sqrt(DAYS_INT)
    cagr_float = float(value_ser.iloc[-1] ** (1.0 / years_float) - 1.0) if years_float > 0 else float("nan")
    max_drawdown_float = float(drawdown_ser.min())
    worst_count_int = max(1, int(np.ceil(0.05 * len(clean_ser))))
    result_dict = {
        "start_str": str(clean_ser.index[0].date()),
        "end_str": str(clean_ser.index[-1].date()),
        "cagr_float": cagr_float,
        "volatility_float": float(clean_ser.std(ddof=1) * np.sqrt(DAYS_INT)),
        "sharpe_float": sharpe_float(clean_ser),
        "sortino_float": float(clean_ser.mean() * DAYS_INT / downside_float) if downside_float > 0 else float("nan"),
        "max_drawdown_float": max_drawdown_float,
        "calmar_float": cagr_float / abs(max_drawdown_float) if max_drawdown_float < 0 else float("nan"),
        "longest_underwater_days_int": int(longest_int),
        "skew_float": float(clean_ser.skew()),
        "kurtosis_float": float(clean_ser.kurt()),
        "expected_shortfall_95_float": float(np.sort(clean_ser.to_numpy())[:worst_count_int].mean()),
        "positive_month_share_float": float((clean_ser.resample("ME").apply(lambda s: (1 + s).prod() - 1) > 0).mean()),
        "year_return_dict": {int(y): float((1 + s).prod() - 1) for y, s in clean_ser.groupby(clean_ser.index.year)},
    }
    if tbill_daily_ser is not None:
        excess_ser = clean_ser - tbill_daily_ser.reindex(clean_ser.index).fillna(0.0)
        result_dict["excess_sharpe_float"] = sharpe_float(excess_ser)
    return result_dict


def tbill_daily_ser(dtb3_ser: pd.Series, date_index: pd.DatetimeIndex) -> pd.Series:
    """Daily T-bill return from DTB3 (percent, annual); the rate known at the previous session."""
    rate_ser = dtb3_ser.reindex(dtb3_ser.index.union(date_index)).ffill().reindex(date_index).shift(1) / 100.0
    return ((1.0 + rate_ser) ** (1.0 / DAYS_INT) - 1.0).fillna(0.0)
