"""Station S3: does the signal carry information? (design section 9, S3, as amended by A5).

The owner's Pakal edge notebooks, kept in structure and corrected in three ways:
1. Point-in-time membership: a (date, stock) counts only if the stock was an index member that day.
2. The unit of inference is the DATE: event excess returns are averaged per date, and the date series is tested
   with a Newey-West t (lag h − 1). P2 showed per-event tests reject a true null 19% of the time on panels with
   persistent signals and sector co-movement; the date-level test held 5%.
   A second estimator weights every event equally (a ratio of sums over dates, with the same date-level Newey-West
   error, so dependence within a date is still respected). It is more powerful when event counts vary a lot, but
   P4b measured 9.5% false positives at nominal 5% with sector factors and a persistent signal, so it is
   informational: it can save an idea from a hard fail, never cause one.
3. Everything is broken down: eras, years, volatility regimes, liquidity terciles, lag decay, crisis windows, the
   most extreme event-dates removed, and costs.

4. Placebo: the event mask is shifted in time by random offsets of at least a year, which keeps its persistence and
   date clustering and breaks its link with returns; p = the share of shifted date-mean excesses at least as good
   as the real one. P4b calibrated it at 4.3-4.5% false positives on panels where every stock is eligible; here the
   shifted events are re-intersected with eligibility (regime and membership), which thins them, so its size on
   real panels is not calibrated. Diagnostic.
5. Replication (`add_replication`): the same sign in at least one sibling universe (soft check).

Diagnostics, not trading rules: the volatility terciles and indicator deciles use full-sample cut points.

Labels (h = horizon in sessions, lag = extra entry delay):
    forward_{i,t}(lag) = Close_i(t + h + lag) / Open_i(t + 1 + lag) − 1        (entry at the next open, as the notebooks)
    excess_{i,t}       = forward_{i,t} − mean of forward over the eligible (regime, member) stocks that date
*** CRITICAL*** Forward shifts are labels only; nothing computed from them feeds back into a feature or a signal.
Labels that would need bars after the panel's end (the vault seal) are NaN, so the seal purges them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from alpha.scout.panel import Panel
from alpha.scout.stations.s1_causality import non_member_share
from alpha.stats.newey_west import newey_west_mean_t_stat

ERA_TUPLE = (("1998-2007", "1998-01-01", "2007-12-31"), ("2008-2015", "2008-01-01", "2015-12-31"), ("2016-2022", "2016-01-01", "2022-12-31"))
CRISIS_WINDOW_TUPLE = (("2000-03-01", "2002-12-31"), ("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-04-30"))
LIQUIDITY_SLIPPAGE_BPS_TUPLE = (10.0, 5.0, 2.5)  # per side, bottom / middle / top turnover tercile (design 6.2)
LAG_TUPLE = (0, 1, 2, 3, 5)


@dataclass
class EdgeReport:
    name_str: str
    horizon_int: int
    headline_dict: dict
    table_dict: dict = field(default_factory=dict)
    check_list: list = field(default_factory=list)
    verdict_str: str = ""
    expected_sign_int: int = 1


def forward_return_df(panel: Panel, horizon_int: int, lag_int: int = 0) -> pd.DataFrame:
    open_df, close_df = panel.field("Open"), panel.field("Close")
    # *** CRITICAL*** forward labels: entry Open(t + 1 + lag), exit Close(t + h + lag).
    return close_df.shift(-(horizon_int + lag_int)) / open_df.shift(-(1 + lag_int)) - 1.0


def _date_series(value_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.Series:
    """Mean of the masked values per date (dates with at least one masked, finite value)."""
    masked_df = value_df.where(mask_df)
    count_ser = masked_df.notna().sum(axis=1)
    return masked_df.sum(axis=1)[count_ser > 0] / count_ser[count_ser > 0]


def _nw(date_ser: pd.Series, horizon_int: int) -> dict:
    if date_ser.size < max(30, horizon_int + 2):
        return {"mean_float": float(date_ser.mean()) if date_ser.size else float("nan"), "t_float": float("nan"), "dates_int": int(date_ser.size)}
    result = newey_west_mean_t_stat(date_ser.to_numpy(), horizon_int - 1)
    return {"mean_float": result.mean_float, "t_float": result.t_stat_float, "dates_int": result.observation_count_int}


def _per_event_nw(value_df: pd.DataFrame, mask_df: pd.DataFrame, horizon_int: int) -> dict:
    """Every event weighted equally: m = sum of excess / number of events, with a Newey-West error on the date
    series u_t = s_t − m·n_t (s_t, n_t: sum and count of events on date t), so se(m) = se(mean u) / mean(n)."""
    masked_df = value_df.where(mask_df)
    count_ser = masked_df.notna().sum(axis=1)
    count_ser = count_ser[count_ser > 0]
    sum_ser = masked_df.sum(axis=1).loc[count_ser.index]
    if count_ser.size < max(30, horizon_int + 2):
        return {"mean_float": float("nan"), "t_float": float("nan")}
    mean_float = float(sum_ser.sum() / count_ser.sum())
    residual_nw = newey_west_mean_t_stat((sum_ser - mean_float * count_ser).to_numpy(), horizon_int - 1)
    return {"mean_float": mean_float, "t_float": mean_float * float(count_ser.mean()) / residual_nw.standard_error_float}


def _shift_placebo_p(
    excess_df: pd.DataFrame, event_mask_df: pd.DataFrame, eligible_mask_df: pd.DataFrame, expected_sign_int: int,
    draw_count_int: int = 200, min_shift_int: int = 252, random_seed_int: int = 0,
) -> float:
    date_count_int = len(event_mask_df.index)
    if date_count_int <= 2 * min_shift_int:
        return float("nan")
    excess_mat, eligible_mat = excess_df.to_numpy(), eligible_mask_df.to_numpy()
    event_mat = event_mask_df.to_numpy()

    def statistic(mask_mat: np.ndarray) -> float:
        count_vec = (mask_mat & np.isfinite(excess_mat)).sum(axis=1)
        sum_vec = np.where(mask_mat, np.nan_to_num(excess_mat), 0.0).sum(axis=1)
        return float(np.mean(sum_vec[count_vec > 0] / count_vec[count_vec > 0])) if (count_vec > 0).any() else float("nan")

    observed_float = expected_sign_int * statistic(event_mat)
    rng_obj = np.random.default_rng(random_seed_int)
    shift_vec = rng_obj.integers(min_shift_int, date_count_int - min_shift_int, draw_count_int)
    null_vec = np.array([expected_sign_int * statistic(np.roll(event_mat, shift_int, axis=0) & eligible_mat) for shift_int in shift_vec])
    null_vec = null_vec[np.isfinite(null_vec)]
    return float((1 + np.sum(null_vec >= observed_float)) / (1 + null_vec.size))


def _set_verdict(report: EdgeReport) -> None:
    hard_fail_bool = any(not passed for _, passed, kind in report.check_list if kind == "hard")
    soft_fail_bool = any(not passed for _, passed, kind in report.check_list if kind == "soft")
    report.verdict_str = "REJECTED (hard fail)" if hard_fail_bool else ("WATCHLIST (soft fail)" if soft_fail_bool else "PASS")


def add_replication(report: EdgeReport, sibling_report_list: list[EdgeReport]) -> EdgeReport:
    """Design S3 replication: the same sign (date-level mean excess) in at least one sibling universe. Soft check."""
    expected_sign_int = report.expected_sign_int
    row_list = [
        {"universe_str": sibling.name_str, "date_mean_excess_float": sibling.headline_dict["date_mean_excess_float"],
         "nw_t_float": sibling.headline_dict["nw_t_float"], "events_int": sibling.headline_dict["events_int"]}
        for sibling in sibling_report_list
    ]
    report.table_dict["replication"] = row_list
    replicated_bool = any(expected_sign_int * row["date_mean_excess_float"] > 0 for row in row_list)
    report.check_list = [row for row in report.check_list if row[0] != "same sign in >= 1 sibling universe"]
    report.check_list.append(("same sign in >= 1 sibling universe", replicated_bool, "soft"))
    _set_verdict(report)
    return report


def run_s3(
    name_str: str,
    panel: Panel,
    regime_mask_df: pd.DataFrame,
    event_mask_df: pd.DataFrame,
    horizon_int: int,
    indicator_df: pd.DataFrame | None = None,
    liquidity_rank_df: pd.DataFrame | None = None,
    expected_sign_int: int = 1,
) -> EdgeReport:
    member_mask_df = panel.member_df == 1
    regime_df = regime_mask_df.fillna(False).astype(bool)
    eligible_mask_df = regime_df & member_mask_df
    non_member_share_float = non_member_share(event_mask_df.fillna(False).astype(bool) & regime_df, panel)
    event_mask_df = event_mask_df.fillna(False).astype(bool) & eligible_mask_df

    forward_df = forward_return_df(panel, horizon_int)
    eligible_forward_df = forward_df.where(eligible_mask_df)
    baseline_ser = eligible_forward_df.mean(axis=1)
    excess_df = forward_df.sub(baseline_ser, axis=0)
    event_date_ser = _date_series(excess_df, event_mask_df)
    headline_nw = _nw(event_date_ser, horizon_int)
    one_sided_p_float = float(stats.norm.sf(expected_sign_int * headline_nw["t_float"])) if np.isfinite(headline_nw["t_float"]) else float("nan")

    per_event_nw = _per_event_nw(excess_df, event_mask_df, horizon_int)
    event_excess_vec = excess_df.where(event_mask_df).stack().to_numpy()
    naive_t_float = float(event_excess_vec.mean() / (event_excess_vec.std(ddof=1) / np.sqrt(event_excess_vec.size))) if event_excess_vec.size > 2 else float("nan")

    report = EdgeReport(
        name_str=name_str,
        horizon_int=horizon_int,
        expected_sign_int=expected_sign_int,
        headline_dict={
            "events_int": int(event_excess_vec.size),
            "event_dates_int": headline_nw["dates_int"],
            "mean_event_excess_float": float(event_excess_vec.mean()) if event_excess_vec.size else float("nan"),
            "date_mean_excess_float": headline_nw["mean_float"],
            "nw_t_float": headline_nw["t_float"],
            "nw_one_sided_p_float": one_sided_p_float,
            "nw_lag_int": horizon_int - 1,
            "per_event_mean_excess_float": per_event_nw["mean_float"],
            "per_event_nw_t_float": per_event_nw["t_float"],
            "naive_event_t_float": naive_t_float,
            "non_member_event_share_float": non_member_share_float,
            "panel_snapshot_str": panel.snapshot_id_str,
            "panel_last_date_str": str(panel.date_index[-1].date()),
        },
    )

    # Eras and years.
    era_row_list = []
    for era_str, start_str, end_str in ERA_TUPLE:
        era_ser = event_date_ser.loc[start_str:end_str]
        era_row_list.append({"era_str": era_str, **_nw(era_ser, horizon_int)})
    report.table_dict["eras"] = era_row_list
    year_mean_ser = event_date_ser.groupby(event_date_ser.index.year).mean()
    positive_year_share_float = float((expected_sign_int * year_mean_ser > 0).mean()) if year_mean_ser.size else float("nan")
    report.table_dict["years"] = {int(year): float(value) for year, value in year_mean_ser.items()}

    # Lag decay.
    lag_row_list = []
    for lag_int in LAG_TUPLE:
        lag_forward_df = forward_return_df(panel, horizon_int, lag_int)
        lag_excess_df = lag_forward_df.sub(lag_forward_df.where(eligible_mask_df).mean(axis=1), axis=0)
        lag_row_list.append({"entry_delay_sessions_int": lag_int, **_nw(_date_series(lag_excess_df, event_mask_df), horizon_int)})
    report.table_dict["lag_decay"] = lag_row_list

    # Volatility regimes: terciles of trailing 63-session volatility of the equal-weight eligible universe.
    universe_return_ser = (panel.field("Close") / panel.field("Close").shift(1) - 1.0).where(member_mask_df).mean(axis=1)
    universe_vol_ser = universe_return_ser.rolling(63, min_periods=40).std()
    vol_tercile_ser = pd.qcut(universe_vol_ser.dropna(), 3, labels=["calm", "normal", "stressed"])
    report.table_dict["volatility_regimes"] = [
        {"regime_str": str(label), **_nw(event_date_ser[event_date_ser.index.isin(vol_tercile_ser.index[vol_tercile_ser == label])], horizon_int)}
        for label in ["calm", "normal", "stressed"]
    ]

    # Concentration and crisis windows. Both tails are trimmed: the date means are fat-tailed, so trimming only the
    # favourable tail drags t down even for pure noise (t about −3 on DV2's demeaned date series).
    ranked_date_ser = event_date_ser.sort_values()
    drop_count_int = max(1, round(0.01 * ranked_date_ser.size))
    without_extreme_ser = event_date_ser.drop(ranked_date_ser.index[:drop_count_int].union(ranked_date_ser.index[-drop_count_int:]))
    crisis_mask = np.zeros(event_date_ser.size, dtype=bool)
    for start_str, end_str in CRISIS_WINDOW_TUPLE:
        crisis_mask |= (event_date_ser.index >= start_str) & (event_date_ser.index <= end_str)
    report.table_dict["concentration"] = {
        "without_extreme_1pct_dates": _nw(without_extreme_ser, horizon_int),
        "without_crisis_windows": _nw(event_date_ser[~crisis_mask], horizon_int),
    }

    # Liquidity terciles and cost coverage.
    tradeable_tercile_int = None
    if liquidity_rank_df is not None:
        tercile_df = np.ceil(liquidity_rank_df.where(eligible_mask_df) * 3).clip(1, 3)
        liquidity_row_list = []
        for tercile_int, label_str in ((1, "bottom"), (2, "middle"), (3, "top")):
            tercile_nw = _nw(_date_series(excess_df, event_mask_df & (tercile_df == tercile_int)), horizon_int)
            liquidity_row_list.append({"tercile_str": label_str, **tercile_nw})
        report.table_dict["liquidity_terciles"] = liquidity_row_list
        event_tercile_vec = tercile_df.where(event_mask_df).stack().to_numpy()
        tradeable_tercile_int = int(np.median(event_tercile_vec)) if event_tercile_vec.size else 2
    round_trip_cost_float = 2.0 * LIQUIDITY_SLIPPAGE_BPS_TUPLE[(tradeable_tercile_int or 2) - 1] / 1e4
    cost_coverage_float = expected_sign_int * headline_nw["mean_float"] / round_trip_cost_float

    # Deciles of the indicator inside the eligible regime, ranked per date (pooled cut points mix eras when the
    # indicator drifts).
    decile_spearman_float = float("nan")
    if indicator_df is not None:
        eligible_indicator_df = indicator_df.where(eligible_mask_df)
        decile_df = np.ceil(eligible_indicator_df.rank(axis=1, pct=True) * 10).clip(1, 10)
        decile_row_list = []
        for decile_int in range(1, 11):
            decile_mask_df = eligible_mask_df & (decile_df == decile_int)
            decile_row_list.append({
                "decile_int": decile_int,
                "indicator_median_float": float(np.nanmedian(eligible_indicator_df.where(decile_mask_df).to_numpy())),
                **_nw(_date_series(excess_df, decile_mask_df), horizon_int),
            })
        report.table_dict["deciles"] = decile_row_list
        decile_spearman_float = float(stats.spearmanr(range(1, 11), [row["mean_float"] for row in decile_row_list]).statistic)

    # Checks (A5: the date-level NW t is the significance test; the ledger FDR is applied when a ledger is used).
    era_sign_count_int = sum(1 for row in era_row_list if np.isfinite(row["mean_float"]) and expected_sign_int * row["mean_float"] > 0)
    liquidity_only_bottom_bool = False
    if "liquidity_terciles" in report.table_dict:
        tercile_mean_list = [row["mean_float"] for row in report.table_dict["liquidity_terciles"]]
        liquidity_only_bottom_bool = expected_sign_int * tercile_mean_list[0] > 0 and all(
            not (expected_sign_int * value > 0) for value in tercile_mean_list[1:]
        )
    # D22: a hard fail needs evidence against the idea: the sign is wrong under both estimators, or the calibrated
    # date-level test says the effect is significantly the wrong way (the per-event t is over-sized, P4b).
    signed_t_list = [expected_sign_int * headline_nw["t_float"], expected_sign_int * per_event_nw["t_float"]]
    report.check_list = [
        ("sign as expected under at least one estimator", any(t_float > 0 for t_float in signed_t_list if np.isfinite(t_float)), "hard"),
        ("not significantly against (date-level t > −2)", not signed_t_list[0] <= -2.0, "hard"),
        ("date-level Newey-West t >= 2", signed_t_list[0] >= 2.0, "soft"),
        ("correct sign in >= 2 of 3 eras", era_sign_count_int >= 2, "soft"),
        ("positive years >= 60%", positive_year_share_float >= 0.60, "soft"),
        ("t >= 2 without the most extreme 1% of event-dates (both tails)", expected_sign_int * report.table_dict["concentration"]["without_extreme_1pct_dates"]["t_float"] >= 2.0, "soft"),
        ("cost coverage >= 2 (tradeable liquidity tercile)", cost_coverage_float >= 2.0, "soft"),
        ("edge not only in the bottom liquidity tercile", not liquidity_only_bottom_bool, "soft"),
    ]
    report.headline_dict.update({
        "positive_year_share_float": positive_year_share_float,
        "cost_coverage_float": cost_coverage_float,
        "decile_spearman_float": decile_spearman_float,
    })
    report.headline_dict["placebo_p_float"] = _shift_placebo_p(excess_df, event_mask_df, eligible_mask_df, expected_sign_int)
    _set_verdict(report)
    return report
