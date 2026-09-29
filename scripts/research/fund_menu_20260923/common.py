"""Shared loading, benchmark and metric helpers for the fund product menu study.

Conventions (match QUANT_PHILOSOPHY.md and the house reports):
- daily simple returns r_t = V_t / V_{t-1} - 1 on the trading-day calendar;
- volatility and Sharpe annualise with 252 sessions; the headline Sharpe uses a
  zero risk-free rate, and an excess-over-T-bill Sharpe is reported beside it;
- CAGR uses calendar time: (V_end / V_start) ** (365.25 / calendar_days) - 1;
- drawdown_t = V_t / max(V_1..V_t) - 1.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "fund_product_menu_20260923"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
DTB3_CSV_PATH = REPO_ROOT_PATH.parent / "1_data" / "DTB3.csv"
TRADING_DAY_COUNT_INT = 252


# ─── sleeve sources ──────────────────────────────────────────────────────────


def load_sleeve_metadata_dict() -> dict[str, dict]:
    metadata_by_alias_dict = {}
    for metadata_path in sorted(SOURCE_DIR_PATH.glob("*__metadata.json")):
        metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_by_alias_dict[metadata_dict["alias_str"]] = metadata_dict
    return metadata_by_alias_dict


def load_sleeve_path_dict() -> dict[str, pd.DataFrame]:
    path_by_alias_dict = {}
    for path_file_path in sorted(SOURCE_DIR_PATH.glob("*__path.csv.gz")):
        alias_str = path_file_path.name.split("__", maxsplit=1)[0]
        path_df = pd.read_csv(path_file_path, index_col="date", parse_dates=True)
        path_by_alias_dict[alias_str] = path_df
    return path_by_alias_dict


def sleeve_nav_df(path_by_alias_dict: dict[str, pd.DataFrame], metadata_by_alias_dict: dict[str, dict]) -> pd.DataFrame:
    """NAV per sleeve from the day before its first position onward (NaN before).

    Leading all-cash warmup days are dropped, never filled: before a sleeve has
    ever held a position its flat NAV says nothing about the strategy. One cash
    day is kept in front of the first invested day as the NAV base, so the first
    fill's slippage, commission and open-to-close P&L stay in the record.
    """
    nav_column_dict = {}
    for alias_str, path_df in path_by_alias_dict.items():
        nav_ser = path_df["total_value_float"].astype(float)
        first_invested_str = metadata_by_alias_dict[alias_str]["first_invested_date_str"]
        first_invested_position_int = nav_ser.index.get_loc(pd.Timestamp(first_invested_str))
        base_position_int = max(first_invested_position_int - 1, 0)
        nav_column_dict[alias_str] = nav_ser.iloc[base_position_int:]
    return pd.DataFrame(nav_column_dict).sort_index()


# ─── benchmarks ──────────────────────────────────────────────────────────────


def load_total_return_close_ser(symbol_str: str, start_date_str: str, end_date_str: str) -> pd.Series:
    from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, load_price_timeseries

    price_df = load_price_timeseries(
        symbol_str,
        adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    close_ser = price_df["Close"].astype(float)
    close_ser.index = pd.to_datetime(close_ser.index).normalize()
    close_ser.name = symbol_str
    return close_ser


def load_tbill_return_ser(session_index: pd.DatetimeIndex) -> pd.Series:
    """Daily T-bill accrual from FRED DTB3 (3-month bill, discount basis, percent).

    *** CRITICAL*** publication lag: the accrual from Close_(t-1) to Close_t uses
    the latest DTB3 observation dated strictly before session t, so no same-day
    rate is used. r_t = DTB3/100 * calendar_days(t-1, t) / 360.
    This is a benchmark/hurdle only; the engine itself credits idle cash at 0%.
    """
    dtb3_df = pd.read_csv(DTB3_CSV_PATH, parse_dates=["observation_date"], na_values=["."])
    dtb3_ser = dtb3_df.set_index("observation_date")["DTB3"].astype(float).dropna().sort_index()
    rate_ser = dtb3_ser.reindex(dtb3_ser.index.union(session_index)).ffill()
    rate_ser = rate_ser.shift(1).reindex(session_index)  # *** CRITICAL*** prior observation only
    calendar_day_ser = pd.Series(session_index, index=session_index).diff().dt.days.fillna(0.0)
    tbill_return_ser = rate_ser / 100.0 * calendar_day_ser / 360.0
    return tbill_return_ser.fillna(0.0).rename("TBILL")


def build_benchmark_return_df(session_index: pd.DatetimeIndex, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """S&P 500 TR, 60/40 SPY/AGG (monthly rebalanced, TR), Nasdaq-100 (QQQ TR) and T-bills."""
    spxtr_ser = load_total_return_close_ser("$SPXTR", start_date_str, end_date_str)
    spy_ser = load_total_return_close_ser("SPY", start_date_str, end_date_str)
    agg_ser = load_total_return_close_ser("AGG", start_date_str, end_date_str)
    qqq_ser = load_total_return_close_ser("QQQ", start_date_str, end_date_str)

    price_df = pd.concat([spxtr_ser, spy_ser, agg_ser, qqq_ser], axis=1).reindex(session_index)
    return_df = price_df.pct_change(fill_method=None)

    # 60/40: weights reset to 60/40 at each month's last session close, drift in between.
    sixty_forty_value_list = []
    value_float = 1.0
    spy_weight_float, agg_weight_float = 0.6, 0.4
    for position_int, session_ts in enumerate(session_index):
        if position_int > 0:
            spy_return_float = return_df["SPY"].iloc[position_int]
            agg_return_float = return_df["AGG"].iloc[position_int]
            if np.isnan(spy_return_float) or np.isnan(agg_return_float):
                sixty_forty_value_list.append(np.nan)
                continue
            spy_leg_float = value_float * spy_weight_float * (1.0 + spy_return_float)
            agg_leg_float = value_float * agg_weight_float * (1.0 + agg_return_float)
            value_float = spy_leg_float + agg_leg_float
            spy_weight_float = spy_leg_float / value_float
            agg_weight_float = agg_leg_float / value_float
        is_month_end_bool = position_int == len(session_index) - 1 or session_index[position_int + 1].month != session_ts.month
        if is_month_end_bool:
            spy_weight_float, agg_weight_float = 0.6, 0.4
        sixty_forty_value_list.append(value_float)
    sixty_forty_ser = pd.Series(sixty_forty_value_list, index=session_index)

    benchmark_return_df = pd.DataFrame(
        {
            "SPXTR": return_df["$SPXTR"],
            "SIXTY_FORTY": sixty_forty_ser.pct_change(fill_method=None),
            "QQQ": return_df["QQQ"],
            "TBILL": load_tbill_return_ser(session_index),
        },
        index=session_index,
    )
    return benchmark_return_df


# ─── metrics ─────────────────────────────────────────────────────────────────


def nav_from_return_ser(return_ser: pd.Series) -> pd.Series:
    return (1.0 + return_ser.fillna(0.0)).cumprod()


def drawdown_ser(nav_ser: pd.Series) -> pd.Series:
    return nav_ser / nav_ser.cummax() - 1.0


def max_drawdown_detail_dict(nav_ser: pd.Series) -> dict:
    """Worst peak-to-trough loss, its dates, and the longest time under water."""
    dd_ser = drawdown_ser(nav_ser)
    trough_ts = dd_ser.idxmin()
    peak_ts = nav_ser.loc[:trough_ts].idxmax()
    recovered_ser = nav_ser.loc[trough_ts:]
    recovery_candidates = recovered_ser[recovered_ser >= nav_ser.loc[peak_ts]]
    recovery_ts = recovery_candidates.index[0] if len(recovery_candidates) else pd.NaT
    underwater_bool_ser = dd_ser < -1e-12
    longest_underwater_day_int = 0
    run_start_ts = None
    for session_ts, underwater_bool in underwater_bool_ser.items():
        if underwater_bool and run_start_ts is None:
            run_start_ts = session_ts
        if not underwater_bool and run_start_ts is not None:
            longest_underwater_day_int = max(longest_underwater_day_int, (session_ts - run_start_ts).days)
            run_start_ts = None
    if run_start_ts is not None:
        longest_underwater_day_int = max(longest_underwater_day_int, (underwater_bool_ser.index[-1] - run_start_ts).days)
    return {
        "max_drawdown_float": float(dd_ser.min()),
        "max_dd_peak_date_str": peak_ts.date().isoformat(),
        "max_dd_trough_date_str": trough_ts.date().isoformat(),
        "max_dd_recovery_date_str": None if pd.isna(recovery_ts) else recovery_ts.date().isoformat(),
        "longest_underwater_days_int": int(longest_underwater_day_int),
    }


def cagr_float(nav_ser: pd.Series) -> float:
    calendar_day_int = (nav_ser.index[-1] - nav_ser.index[0]).days
    return float((nav_ser.iloc[-1] / nav_ser.iloc[0]) ** (365.25 / calendar_day_int) - 1.0)


def metric_dict(
    return_ser: pd.Series,
    spx_return_ser: pd.Series,
    tbill_return_ser: pd.Series,
    base_date_ts: pd.Timestamp | None = None,
) -> dict:
    """Headline metrics for one daily return series on its own (already sliced) window.

    return_ser must have no NaN inside the window; the first element is the
    first realised return and base_date_ts is the prior session whose close is
    the NAV base (used for calendar-time CAGR).
    """
    return_ser = return_ser.astype(float)
    if return_ser.isna().any():
        raise ValueError("metric_dict needs a gap-free return series.")
    base_ts = pd.Timestamp(base_date_ts) if base_date_ts is not None else return_ser.index[0] - pd.Timedelta(days=1)
    nav_ser = pd.concat([pd.Series([1.0], index=[base_ts]), nav_from_return_ser(return_ser)])
    excess_ser = return_ser - tbill_return_ser.reindex(return_ser.index).fillna(0.0)
    spx_ser = spx_return_ser.reindex(return_ser.index)
    monthly_return_ser = (1.0 + return_ser).resample("ME").prod() - 1.0
    monthly_spx_ser = (1.0 + spx_ser.fillna(0.0)).resample("ME").prod() - 1.0
    yearly_return_ser = (1.0 + return_ser).resample("YE").prod() - 1.0
    rolling_21_ser = (1.0 + return_ser).rolling(21).apply(np.prod, raw=True) - 1.0
    tail_cut_float = return_ser.quantile(0.05)
    down_month_mask = monthly_spx_ser < 0
    up_month_mask = monthly_spx_ser > 0
    covariance_float = return_ser.cov(spx_ser)
    tbill_nav_ser = nav_from_return_ser(tbill_return_ser.reindex(return_ser.index).fillna(0.0))

    detail_dict = max_drawdown_detail_dict(nav_ser)
    out_dict = {
        "start_date_str": return_ser.index[0].date().isoformat(),
        "end_date_str": return_ser.index[-1].date().isoformat(),
        "observation_count_int": int(len(return_ser)),
        "cagr_float": cagr_float(nav_ser),
        "tbill_cagr_float": float(tbill_nav_ser.iloc[-1] ** (365.25 / (return_ser.index[-1] - base_ts).days) - 1.0),
        "volatility_float": float(return_ser.std() * np.sqrt(TRADING_DAY_COUNT_INT)),
        "sharpe_rf0_float": float(return_ser.mean() / return_ser.std() * np.sqrt(TRADING_DAY_COUNT_INT)),
        "sharpe_excess_float": float(excess_ser.mean() / excess_ser.std() * np.sqrt(TRADING_DAY_COUNT_INT)),
        **detail_dict,
        "mar_float": float(cagr_float(nav_ser) / abs(detail_dict["max_drawdown_float"]))
        if detail_dict["max_drawdown_float"] < 0
        else np.nan,
        "es95_daily_float": float(return_ser[return_ser <= tail_cut_float].mean()),
        "es95_21d_float": float(rolling_21_ser[rolling_21_ser <= rolling_21_ser.quantile(0.05)].mean()),
        "worst_month_float": float(monthly_return_ser.min()),
        "worst_year_float": float(yearly_return_ser.min()),
        "best_year_float": float(yearly_return_ser.max()),
        "positive_month_share_float": float((monthly_return_ser > 0).mean()),
        "beta_spx_float": float(covariance_float / spx_ser.var()),
        "corr_spx_daily_float": float(return_ser.corr(spx_ser)),
        "corr_spx_monthly_float": float(monthly_return_ser.corr(monthly_spx_ser)),
        "down_capture_float": float(monthly_return_ser[down_month_mask].mean() / monthly_spx_ser[down_month_mask].mean())
        if down_month_mask.any()
        else np.nan,
        "up_capture_float": float(monthly_return_ser[up_month_mask].mean() / monthly_spx_ser[up_month_mask].mean())
        if up_month_mask.any()
        else np.nan,
        "skew_monthly_float": float(monthly_return_ser.skew()),
    }
    return out_dict


def equity_drawdown_episode_list(spx_return_ser: pd.Series, threshold_float: float = -0.10) -> list[dict]:
    """Objective crisis list: every S&P 500 TR peak-to-trough decline of at least 10%.

    An episode starts at a running peak and ends at the lowest close before the
    index regains that peak. Dates come from the benchmark alone, so the list
    cannot be tuned to any book.
    """
    nav_ser = nav_from_return_ser(spx_return_ser.dropna())
    episode_list = []
    peak_ts = nav_ser.index[0]
    peak_float = nav_ser.iloc[0]
    trough_ts = peak_ts
    trough_float = peak_float
    for session_ts, value_float in nav_ser.items():
        if value_float >= peak_float:
            if trough_float / peak_float - 1.0 <= threshold_float:
                episode_list.append(
                    {"peak_date_str": peak_ts.date().isoformat(), "trough_date_str": trough_ts.date().isoformat(),
                     "spx_drawdown_float": float(trough_float / peak_float - 1.0)}
                )
            peak_ts, peak_float = session_ts, value_float
            trough_ts, trough_float = session_ts, value_float
        elif value_float < trough_float:
            trough_ts, trough_float = session_ts, value_float
    if trough_float / peak_float - 1.0 <= threshold_float:
        episode_list.append(
            {"peak_date_str": peak_ts.date().isoformat(), "trough_date_str": trough_ts.date().isoformat(),
             "spx_drawdown_float": float(trough_float / peak_float - 1.0), "open_bool": True}
        )
    return episode_list


def window_return_float(return_ser: pd.Series, start_date_str: str, end_date_str: str) -> float:
    """Compounded return from the close of start_date to the close of end_date."""
    window_ser = return_ser.loc[(return_ser.index > pd.Timestamp(start_date_str)) & (return_ser.index <= pd.Timestamp(end_date_str))]
    if window_ser.isna().any() or len(window_ser) == 0:
        return np.nan
    return float((1.0 + window_ser).prod() - 1.0)


# ─── book construction (pod model) ──────────────────────────────────────────


def book_return_ser(
    sleeve_return_df: pd.DataFrame,
    weight_by_alias_dict: dict[str, float],
    rebalance_str: str = "none",
) -> tuple[pd.Series, pd.DataFrame]:
    """Sum of independently compounded pods, optionally reset to target weights.

    Pod model: E_book,t = sum_i E_i,t with E_i,0 = w_i * capital, each pod
    compounding its own daily return. With rebalance "annual", the pod values
    are reset to w_i * E_book at the last session of each calendar year (a
    frictionless reallocation between pod accounts; each pod re-sizes at its next
    own decision). Returns the book daily return and the prior-close pod weights.

    *** CRITICAL*** the reset happens AFTER the year-end close is booked, so the
    first return that uses the new weights is the first session of the new year.
    """
    alias_list = list(weight_by_alias_dict)
    weight_arr = np.array([weight_by_alias_dict[alias_str] for alias_str in alias_list], dtype=float)
    if abs(weight_arr.sum() - 1.0) > 1e-9:
        raise ValueError("Book weights must sum to 1.")
    window_return_df = sleeve_return_df[alias_list]
    if window_return_df.isna().any().any():
        raise ValueError("Book window has missing sleeve returns; slice to a common window first.")
    return_mat = window_return_df.to_numpy(dtype=float)
    index = window_return_df.index
    pod_value_arr = weight_arr.copy()
    book_return_list = []
    prior_weight_list = []
    for position_int in range(len(index)):
        book_value_before_float = pod_value_arr.sum()
        prior_weight_list.append(pod_value_arr / book_value_before_float)
        pod_value_arr = pod_value_arr * (1.0 + return_mat[position_int])
        book_return_list.append(pod_value_arr.sum() / book_value_before_float - 1.0)
        is_year_end_bool = position_int + 1 < len(index) and index[position_int + 1].year != index[position_int].year
        if rebalance_str == "annual" and is_year_end_bool:
            pod_value_arr = weight_arr * pod_value_arr.sum()
        elif rebalance_str not in ("none", "annual"):
            raise ValueError(f"Unsupported rebalance_str {rebalance_str!r}.")
    book_ser = pd.Series(book_return_list, index=index, name="book")
    prior_weight_df = pd.DataFrame(prior_weight_list, index=index, columns=alias_list)
    return book_ser, prior_weight_df


def risk_share_df(sleeve_return_df: pd.DataFrame, prior_weight_df: pd.DataFrame, book_ser: pd.Series) -> pd.DataFrame:
    """Exact realised variance and tail-loss decomposition of a pod book.

    Contribution c_i,t = w_i,t-1 * r_i,t sums to the book return, so
        Var(r_book) = sum_i Cov(c_i, r_book)
    and each pod's variance share is Cov(c_i, r_book) / Var(r_book).
    Tail share = pod contribution on the worst 5% of book days / book loss on those days.
    """
    contribution_df = prior_weight_df * sleeve_return_df[prior_weight_df.columns]
    variance_float = book_ser.var()
    tail_mask = book_ser <= book_ser.quantile(0.05)
    return pd.DataFrame(
        {
            "start_weight_float": prior_weight_df.iloc[0],
            "end_weight_float": prior_weight_df.iloc[-1],
            "variance_share_float": contribution_df.apply(lambda col: col.cov(book_ser)) / variance_float,
            "tail_loss_share_float": contribution_df[tail_mask].sum() / book_ser[tail_mask].sum(),
        }
    )
