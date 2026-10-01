"""Station S3 for allocation (class W) and monthly ranking (class X) families: does the score carry information?

Design section 9, S3: class W has too little data for an event study, so S3 runs per-asset predictive regressions
and the signal-on against signal-off spread, with the bar at t >= 2.0. Class X re-auditions (a monthly ranking of
stocks) use the cross-sectional analogue. Every test is on the DATE series (one number per month), with a
Newey-West t (lag 2), the unit of inference P2 calibrated for S3.

- `predictive_tests` (W): for each month m, the cross-sectional slope of next-month excess return on the standardised
  score (Fama-MacBeth); per-asset time-series slopes; and the on/off spread: mean next-month return of assets whose
  score clears the hurdle minus that of the others.
- `gate_split` (W overlays such as a VIX gate): next-month mean return and volatility when the gate is on vs off. A
  risk gate is judged on the volatility ratio, not on the mean.
- `ranking_tests` (X): per month, the Spearman rank IC of the score against next-month return among eligible members,
  and the spread of the top-N over the equal-weight eligible mean.

*** CRITICAL*** Scores are read at the month-end decision; returns are the NEXT month's (labels only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from alpha.stats.newey_west import newey_west_mean_t_stat

LAG_INT = 2
T_BAR_FLOAT = 2.0


def _nw(series: pd.Series) -> dict:
    clean_ser = series.dropna()
    if clean_ser.size < 24:
        return {"mean_float": float(clean_ser.mean()) if clean_ser.size else float("nan"), "t_float": float("nan"), "months_int": int(clean_ser.size)}
    result = newey_west_mean_t_stat(clean_ser.to_numpy(), LAG_INT)
    return {"mean_float": result.mean_float, "t_float": result.t_stat_float, "months_int": result.observation_count_int}


def predictive_tests(score_df: pd.DataFrame, next_return_df: pd.DataFrame, hurdle_ser: pd.Series | None = None, min_asset_int: int = 3) -> dict:
    """score_df and next_return_df: month-end index x assets; next_return_df[m] is the return over month m+1."""
    aligned_score_df, aligned_return_df = score_df.align(next_return_df, join="inner")
    slope_list, spread_list = [], []
    for month_ts in aligned_score_df.index:
        frame = pd.DataFrame({"z": aligned_score_df.loc[month_ts], "r": aligned_return_df.loc[month_ts]}).dropna()
        if len(frame) < min_asset_int or frame["z"].std() == 0:
            continue
        z_vec = (frame["z"] - frame["z"].mean()) / frame["z"].std()
        slope_list.append((month_ts, float(np.polyfit(z_vec, frame["r"], 1)[0])))
        threshold_float = 0.0 if hurdle_ser is None else float(hurdle_ser.get(month_ts, 0.0))
        on_mask = frame["z"] > threshold_float
        if on_mask.any() and (~on_mask).any():
            spread_list.append((month_ts, float(frame.loc[on_mask, "r"].mean() - frame.loc[~on_mask, "r"].mean())))
    asset_row_list = []
    for asset_str in aligned_score_df.columns:
        frame = pd.DataFrame({"z": aligned_score_df[asset_str], "r": aligned_return_df[asset_str]}).dropna()
        if len(frame) < 24:
            continue
        z_ser = (frame["z"] - frame["z"].mean()) / frame["z"].std()
        # Slope on the demeaned score = mean of z x r / mean of z^2; its NW t is the t of the product series.
        product_ser = z_ser * (frame["r"] - frame["r"].mean())
        asset_row_list.append({"asset_str": asset_str, **{k: v for k, v in _nw(product_ser).items() if k != "mean_float"},
                               "slope_float": float(product_ser.mean() / (z_ser**2).mean())})
    fama_macbeth = _nw(pd.Series(dict(slope_list)))
    spread = _nw(pd.Series(dict(spread_list)))
    return {
        "fama_macbeth_slope": fama_macbeth,
        "on_off_spread": spread,
        "per_asset_list": asset_row_list,
        "check_list": [
            ("cross-sectional slope t >= 2 (Fama-MacBeth)", "PASS" if fama_macbeth["t_float"] >= T_BAR_FLOAT else "WARN",
             f"{fama_macbeth['mean_float'] * 100:+.2f}% a month per score sd, t {fama_macbeth['t_float']:.2f}, {fama_macbeth['months_int']} months"),
            ("signal-on minus signal-off t >= 2", "PASS" if spread["t_float"] >= T_BAR_FLOAT else "WARN",
             f"{spread['mean_float'] * 100:+.2f}% a month, t {spread['t_float']:.2f}"),
        ],
    }


def gate_split(next_return_ser: pd.Series, gate_on_ser: pd.Series) -> dict:
    frame = pd.DataFrame({"r": next_return_ser, "on": gate_on_ser}).dropna()
    on_ser, off_ser = frame.loc[frame["on"].astype(bool), "r"], frame.loc[~frame["on"].astype(bool), "r"]
    vol_ratio_float = float(off_ser.std() / on_ser.std()) if len(on_ser) > 2 and len(off_ser) > 2 else float("nan")
    # Volatility ratio test: Levene (robust to fat tails) on the two groups.
    levene_p_float = float(stats.levene(on_ser, off_ser).pvalue) if len(on_ser) > 2 and len(off_ser) > 2 else float("nan")
    return {
        "months_on_int": len(on_ser), "months_off_int": len(off_ser),
        "mean_on_float": float(on_ser.mean()), "mean_off_float": float(off_ser.mean()),
        "vol_on_float": float(on_ser.std() * np.sqrt(12)), "vol_off_float": float(off_ser.std() * np.sqrt(12)),
        "vol_ratio_off_over_on_float": vol_ratio_float, "levene_p_float": levene_p_float,
        "check": ("gate predicts risk: next-month volatility higher when off (Levene p <= 0.05)",
                  "PASS" if np.isfinite(levene_p_float) and levene_p_float <= 0.05 and vol_ratio_float > 1 else "WARN",
                  f"vol off/on {vol_ratio_float:.2f} ({frame['on'].astype(bool).sum()} months on, {(~frame['on'].astype(bool)).sum()} off), p {levene_p_float:.3f}"),
    }


def ranking_tests(score_df: pd.DataFrame, next_return_df: pd.DataFrame, eligible_df: pd.DataFrame, top_int: int, min_eligible_int: int = 20) -> dict:
    """Monthly rank IC of the score among eligible members, and the top-N spread over the eligible mean."""
    ic_list, spread_list = [], []
    for month_ts in score_df.index:
        if month_ts not in next_return_df.index or month_ts not in eligible_df.index:
            continue
        frame = pd.DataFrame({"s": score_df.loc[month_ts], "r": next_return_df.loc[month_ts], "e": eligible_df.loc[month_ts]}).dropna()
        frame = frame[frame["e"].astype(bool)]
        if len(frame) < min_eligible_int:
            continue
        ic_list.append((month_ts, float(stats.spearmanr(frame["s"], frame["r"]).statistic)))
        top_mean_float = frame.sort_values("s", ascending=False)["r"].iloc[:top_int].mean()
        spread_list.append((month_ts, float(top_mean_float - frame["r"].mean())))
    ic, spread = _nw(pd.Series(dict(ic_list))), _nw(pd.Series(dict(spread_list)))
    return {
        "rank_ic": ic,
        "top_spread": spread,
        "check_list": [
            ("rank IC t >= 2", "PASS" if ic["t_float"] >= T_BAR_FLOAT else "WARN", f"mean IC {ic['mean_float']:.3f}, t {ic['t_float']:.2f}, {ic['months_int']} months"),
            (f"top-{top_int} minus eligible mean t >= 2", "PASS" if spread["t_float"] >= T_BAR_FLOAT else "WARN",
             f"{spread['mean_float'] * 100:+.2f}% a month, t {spread['t_float']:.2f}"),
        ],
    }
