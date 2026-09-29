"""TAA leakage hunt - (d) FRED publication-lag and ALFRED vintage study (DTB3 hurdle; T5YIE for Compass).

Run:  uv run python scripts/research/leakage_hunt_20260927/taa_macro_vintage.py

Facts used
----------
* The backtest hurdle is ``cash_return_ser.resample('ME').last()`` = the last DTB3 observation DATED within the
  calendar month (strategy_taa_df.py:299).  At the month's last session T that is normally DTB3_T.
* H.15 / FRED publish DTB3_t on t+1 (ALFRED vintage 2020-03-31 ends at 2020-03-30; verified again below for every
  decision), i.e. after the Close_T decision and, for the live evening build, not available either.
* Compass aligns T5YIE by observation date with allow_exact_matches=True (strategy_taa_inflation_compass.py:185),
  so it uses T5YIE_T at Close_T; T5YIE (= DGS5 - DFII5, H.15) has the same t+1 publication.

Variants per decision month (label L, last XNYS session T):
  base   : last obs dated <= L (reproduces the code; validated to give identical weights)
  lag1   : last obs dated <  T  (causal prior-business-day value)
  alfred : last obs in the ALFRED vintage as of T (what was actually downloadable on T)
Compass: lag1 = every T5YIE obs made available one XNYS session after its date (affects T and T-60 values);
         alfred = features at T recomputed with the vintage-as-of-T series.

Outputs: results/research/leakage_hunt_20260927/taa/taa_macro_vintage.json, taa_macro_vintage_flips.csv,
         taa_macro_vintage_rerun.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import compare_weight_frames
from taa_alfred import vintage_frame
from taa_common import (BOOK_END_STR, BOOK_START_STR, OUT_DIR, compute_decisions, load_norgate_full,
                        metrics_from_strategy, patched, read_frozen_series, run_backtest_from_weights)
from taa_accounting_rerun import run_compass


def xnys_sessions() -> pd.DatetimeIndex:
    spy = load_norgate_full("SPY", "CAPITALSPECIAL")
    return pd.DatetimeIndex(spy.index[spy["Volume"].fillna(0) > 0])


def month_last_session(sessions: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()


def last_obs_before(ser: pd.Series, ts: pd.Timestamp, strict: bool) -> tuple[pd.Timestamp, float]:
    s = ser[ser.index < ts] if strict else ser[ser.index <= ts]
    return (pd.Timestamp(s.index[-1]), float(s.iloc[-1])) if len(s) else (pd.NaT, np.nan)


def diff_rows(ref: pd.DataFrame, cand: pd.DataFrame, lo: str | None = None, hi: str | None = None) -> list:
    idx = ref.index.intersection(cand.index)
    if lo:
        idx = idx[idx >= pd.Timestamp(lo)]
    if hi:
        idx = idx[idx <= pd.Timestamp(hi)]
    cols = ref.columns.union(cand.columns)
    a = ref.reindex(index=idx, columns=cols).fillna(0)
    b = cand.reindex(index=idx, columns=cols).fillna(0)
    bad = (a - b).abs().max(axis=1) > 1e-9
    return [{"label": str(d.date()), "base": {c: round(v, 4) for c, v in a.loc[d].items() if v},
             "variant": {c: round(v, 4) for c, v in b.loc[d].items() if v}} for d in idx[bad]]


# ------------------------------------------------------------------------------------------------ DTB3
def dtb3_study(sessions, rerun_rows, report):
    dtb3 = read_frozen_series("DTB3")
    mls = month_last_session(sessions)
    with patched():
        base_dec = {k: compute_decisions(k) for k in ("taa3x_rank", "taa3x_1n")}
    labels = base_dec["taa3x_rank"]["month_end_weight_df"].index
    complete = [L for L in labels if L.to_period("M") in mls.index and mls.loc[L.to_period("M")] <= sessions[-1]
                and mls.loc[L.to_period("M")] < sessions[-1] + pd.Timedelta(days=0) or L.to_period("M") < sessions[-1].to_period("M")]
    complete = [L for L in labels if L.to_period("M") < sessions[-1].to_period("M")]  # drop the partial Sept 2026
    T_list = [mls.loc[L.to_period("M")] for L in complete]
    alf = vintage_frame("DTB3", T_list)

    table = []
    for L, T in zip(complete, T_list):
        b_d, b_v = last_obs_before(dtb3, L, strict=False)
        l_d, l_v = last_obs_before(dtb3, T, strict=True)
        vs = alf[pd.Timestamp(T)]
        a_d, a_v = (pd.Timestamp(vs.index[-1]), float(vs.iloc[-1])) if len(vs) else (pd.NaT, np.nan)
        # revision check: vintage-as-of-T value vs current-vintage value for the same observation date
        cur_same = float(dtb3.get(a_d, np.nan)) if a_d is not pd.NaT else np.nan
        table.append({"label": L, "T": T, "base_obs": b_d, "base": b_v, "lag1_obs": l_d, "lag1": l_v,
                      "alfred_obs": a_d, "alfred": a_v, "current_value_of_alfred_obs": cur_same,
                      "alfred_last_obs_is_T_minus": int(((sessions > a_d) & (sessions <= T)).sum()) if a_d is not pd.NaT else None})
    tab = pd.DataFrame(table)
    tab.to_csv(OUT_DIR / "taa_dtb3_month_end_table.csv", index=False)

    rev = tab.dropna(subset=["alfred", "current_value_of_alfred_obs"])
    # broader revision check: every overlapping observation of every fetched vintage vs current series
    n_cmp = n_rev = 0
    max_rev = 0.0
    for v, s in alf.items():
        common = s.index.intersection(dtb3.index)
        common = common[common >= pd.Timestamp("2011-01-01")]
        d = (s.loc[common] - dtb3.loc[common]).abs()
        n_cmp += len(d)
        n_rev += int((d > 1e-9).sum())
        max_rev = max(max_rev, float(d.max()) if len(d) else 0.0)
    report["dtb3"] = {
        "n_complete_decisions": len(tab),
        "n_base_obs_dated_T": int((tab["base_obs"] == tab["T"]).sum()),
        "n_base_obs_dated_after_T": int((tab["base_obs"] > tab["T"]).sum()),
        "n_alfred_vintage_contains_obs_T": int((tab["alfred_obs"] >= tab["T"]).sum()),
        "alfred_lag_sessions_distribution": tab["alfred_last_obs_is_T_minus"].value_counts().to_dict(),
        "n_months_base_ne_lag1_value": int((tab["base"] - tab["lag1"]).abs().gt(1e-12).sum()),
        "n_months_lag1_ne_alfred_value": int((tab["lag1"] - tab["alfred"]).abs().gt(1e-12).sum()),
        "max_abs_base_minus_lag1_pct_pts": float((tab["base"] - tab["lag1"]).abs().max()),
        "revision_check_month_end": {"n": int(len(rev)), "n_revised": int((rev["alfred"] - rev["current_value_of_alfred_obs"]).abs().gt(1e-9).sum())},
        "revision_check_all_obs_since_2011": {"n_compared": n_cmp, "n_revised": n_rev, "max_abs_revision": max_rev},
    }

    flips_all = []
    for variant in ("base", "lag1", "alfred"):
        ser = pd.Series(tab[variant].values, index=pd.DatetimeIndex(tab["T"]), name="DTB3").dropna()
        for key in ("taa3x_rank", "taa3x_1n"):
            with patched(macro_override={"DTB3": ser}):
                dec = compute_decisions(key)
            ref = base_dec[key]["month_end_weight_df"].loc[: complete[-1]]
            flips = diff_rows(ref, dec["month_end_weight_df"].loc[: complete[-1]])
            book_flips = [f for f in flips if pd.Timestamp(f["label"]) >= pd.Timestamp("2012-09-30")
                          and pd.Timestamp(f["label"]) <= pd.Timestamp("2026-07-31")]
            report.setdefault("dtb3_flips", {})[f"{key}:{variant}"] = {
                "n_flips_all": len(flips), "n_flips_book_window": len(book_flips), "flips": flips}
            for f in flips:
                flips_all.append({"strategy": key, "variant": variant, **{k: json.dumps(v) if isinstance(v, dict) else v for k, v in f.items()}})
            print(key, variant, "flips", len(flips), flush=True)
            if variant != "base" and len(book_flips) > 0:
                for hsu in (False,):
                    with patched(macro_override={"DTB3": ser}):
                        dec_b = compute_decisions(key, end_date_str=BOOK_END_STR)
                        strat = run_backtest_from_weights(key, dec_b, start_str=BOOK_START_STR)
                    m = metrics_from_strategy(strat)
                    m.update({"strategy": key, "variant": f"dtb3_{variant}"})
                    rerun_rows.append(m)
                    print(m, flush=True)

    # closeness of scores to the hurdle (how fragile the absolute filter is)
    for key in ("taa3x_rank",):
        sc = base_dec[key]["score_df"]
        hurdle = (1 + tab.set_index("label")["base"] / 100) ** (1 / 12) - 1
        margin = sc.reindex(hurdle.index)[["GLD", "UUP", "TLT", "DBC", "BTAL"]].sub(hurdle, axis=0).abs()
        report["dtb3_score_margin_bp"] = {
            "min_margin_bp": float(margin.min().min() * 1e4),
            "n_asset_months_within_1bp": int((margin < 1e-4).sum().sum()),
            "n_asset_months_within_0.1bp": int((margin < 1e-5).sum().sum()),
            "monthly_hurdle_change_from_one_day_lag_max_bp": float(
                ((1 + tab["base"] / 100) ** (1 / 12) - (1 + tab["lag1"] / 100) ** (1 / 12)).abs().max() * 1e4),
        }
    return flips_all


# ------------------------------------------------------------------------------------------------ T5YIE
def t5yie_study(sessions, rerun_rows, report):
    import strategies.taa_df.strategy_taa_inflation_compass as compass
    t5 = read_frozen_series("T5YIE")
    with patched():
        base = compute_decisions("compass")
    ref = base["month_end_weight_df"]
    feat = base["score_df"]

    # lag1: obs dated d becomes usable at the next XNYS session after d
    pos = sessions.searchsorted(t5.index, side="right")
    ok = pos < len(sessions)
    lag = pd.Series(t5.values[ok], index=sessions[pos[ok]], name="T5YIE")
    lag = lag[~lag.index.duplicated(keep="last")]
    with patched(macro_override={"T5YIE": lag}):
        dec_lag = compute_decisions("compass")
    flips_lag = diff_rows(ref, dec_lag["month_end_weight_df"])

    # ALFRED vintage as of each decision session T
    T_list = [pd.Timestamp(t) for t in ref.index if pd.Timestamp(t) < sessions[-1]]
    alf = vintage_frame("T5YIE", T_list)
    with patched():
        sig = compass.load_signal_close_df(symbol_list=compass.DEFAULT_CONFIG.signal_asset_tuple,
                                           start_date_str=compass.DEFAULT_CONFIG.start_date_str)
    rows, flips_alf, no_vintage = [], [], 0
    n_rev = n_cmp = 0
    for T in T_list:
        vs = alf[T]
        if len(vs) == 0:
            no_vintage += 1
            continue
        common = vs.index.intersection(t5.index)
        dd = (vs.loc[common] - t5.loc[common]).abs()
        n_cmp += len(dd)
        n_rev += int((dd > 1e-9).sum())
        try:
            f_df, w_df = compass.compute_month_end_signal_and_weight_df(sig.loc[:T], vs, compass.DEFAULT_CONFIG)
        except RuntimeError as exc:
            rows.append({"T": T, "error": str(exc)[:200]})
            continue
        if T not in w_df.index:
            rows.append({"T": T, "error": "no row at T"})
            continue
        same = bool(np.allclose(w_df.loc[T].values, ref.loc[T].reindex(w_df.columns).values))
        rows.append({"T": T, "t5yie_used_base": float(feat.loc[T, "t5yie_float"]),
                     "t5yie_used_alfred": float(f_df.loc[T, "t5yie_float"]),
                     "alfred_last_obs": vs.index[-1], "same_weights": same})
        if not same:
            flips_alf.append({"label": str(T.date()), "base": {c: v for c, v in ref.loc[T].items() if v},
                              "variant": {c: v for c, v in w_df.loc[T].items() if v}})
    alf_tab = pd.DataFrame(rows)
    alf_tab.to_csv(OUT_DIR / "taa_t5yie_alfred_table.csv", index=False)
    report["t5yie"] = {
        "n_decisions": int(len(ref)),
        "n_decisions_using_same_date_obs": int((feat["t5yie_observation_age_day_float"] == 0).sum()),
        "lag1_n_flips": len(flips_lag), "lag1_flips": flips_lag,
        "alfred_n_decisions_tested": int(alf_tab["same_weights"].notna().sum()) if "same_weights" in alf_tab else 0,
        "alfred_n_no_vintage_available": no_vintage,
        "alfred_n_errors": int(alf_tab["error"].notna().sum()) if "error" in alf_tab else 0,
        "alfred_first_vintage_decision": str(alf_tab.dropna(subset=["same_weights"])["T"].min()) if "same_weights" in alf_tab else None,
        "alfred_n_flips": len(flips_alf), "alfred_flips": flips_alf,
        "alfred_revision_check": {"n_compared": n_cmp, "n_revised": n_rev},
    }
    print("compass lag1 flips", len(flips_lag), "alfred flips", len(flips_alf), "no vintage", no_vintage, flush=True)
    for variant, ser in (("lag1", lag),):
        if flips_lag:
            with patched(macro_override={"T5YIE": ser}):
                dec_b = compute_decisions("compass", end_date_str=BOOK_END_STR)
                strat = run_compass(dec_b, hsu=False)
            m = metrics_from_strategy(strat)
            m.update({"strategy": "compass", "variant": f"t5yie_{variant}"})
            rerun_rows.append(m)
            print(m, flush=True)
    with patched():
        dec_b = compute_decisions("compass", end_date_str=BOOK_END_STR)
        strat = run_compass(dec_b, hsu=False)
    m = metrics_from_strategy(strat)
    m.update({"strategy": "compass", "variant": "base"})
    rerun_rows.append(m)


def main():
    t0 = time.time()
    sessions = xnys_sessions()
    report, rerun_rows = {}, []
    flips = dtb3_study(sessions, rerun_rows, report)
    # baselines for the DTB3 reruns
    for key in ("taa3x_rank", "taa3x_1n"):
        with patched():
            dec_b = compute_decisions(key, end_date_str=BOOK_END_STR)
            strat = run_backtest_from_weights(key, dec_b, start_str=BOOK_START_STR)
        m = metrics_from_strategy(strat)
        m.update({"strategy": key, "variant": "base"})
        rerun_rows.append(m)
    t5yie_study(sessions, rerun_rows, report)
    pd.DataFrame(flips).to_csv(OUT_DIR / "taa_macro_vintage_flips.csv", index=False)
    pd.DataFrame(rerun_rows).to_csv(OUT_DIR / "taa_macro_vintage_rerun.csv", index=False)
    report["runtime_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "taa_macro_vintage.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("dtb3_flips",)}, indent=1, default=str)[:6000])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "flips"} for k, v in report["dtb3_flips"].items()}, indent=1))
    print(pd.DataFrame(rerun_rows).to_string())


if __name__ == "__main__":
    main()
