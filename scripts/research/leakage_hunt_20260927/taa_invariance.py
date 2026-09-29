"""TAA leakage hunt - tests (b) future-action invariance and (c) truncation/prefix on REAL Norgate data.

Run:  uv run python scripts/research/leakage_hunt_20260927/taa_invariance.py

(b) For each audited strategy, rescale ONE symbol's whole loaded history (TOTALRETURN signal frame and
    CAPITALSPECIAL execution/helper frame consistently, per harness rules) by k in {40, 0.1, 1.5} and require the
    month-end target weights (after the VRP gate) to be identical at every decision date.  Also one joint case with
    independent factors on all symbols at once.
    Plus: TOTALRETURN multiplicativity on real data: TR_t/TR_{t-1}-1 vs (C_t + D_t)/C_{t-1}-1 from CAPITALSPECIAL.
(c) Truncation: decisions computed with config.end_date_str = T (provider data, DTB3/T5YIE as-of, VIX helper all
    cut at T) must equal full-history decisions for every COMPLETED month <= T.  Provisional rows (the month that
    contains T when T is not its last session) are reported separately and are expected to differ.

Outputs: results/research/leakage_hunt_20260927/taa/taa_invariance_results.json, taa_tr_multiplicativity.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import ResultLog, compare_weight_frames
from taa_common import (OUT_DIR, STRATEGY_KEYS, compute_decisions, load_norgate_full, patched, warm_cache)

FACTORS = (40.0, 0.1, 1.5)
SYMBOLS_BY_KEY = {
    "taa3x_rank": ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ", "SPY"),
    "taa3x_1n": ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ", "SPY"),
    "btal_qqq_linearity": ("GLD", "UUP", "TLT", "DBC", "BTAL", "QQQ", "SPY"),
    "compass": ("SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP", "XLK", "IEF"),
}
# Cutoffs: mid-month; month whose calendar last day is a weekend (2020-05-31 Sun, 2022-04-30 Sat);
# Good-Friday month-end holiday (2024-03-29 -> last session 2024-03-28); a plain month-end; book end (mid-month);
# current partial month (Sept 2026, data ends 2026-09-25).
CUTOFFS = ("2015-06-15", "2020-05-29", "2022-04-29", "2024-03-28", "2023-06-30", "2026-08-19", "2026-09-15",
           "2026-08-31")


def xnys_sessions() -> pd.DatetimeIndex:
    spy = load_norgate_full("SPY", "CAPITALSPECIAL")
    return pd.DatetimeIndex(spy.index[spy["Volume"].fillna(0) > 0])


def invariance_tests(log: ResultLog) -> dict:
    summary = {}
    for key in STRATEGY_KEYS:
        with patched():
            base = compute_decisions(key)
        ref = base["month_end_weight_df"]
        n_pass = n_fail = 0
        for sym in SYMBOLS_BY_KEY[key]:
            for k in FACTORS:
                with patched(rescale={sym: k}):
                    cand = compute_decisions(key)["month_end_weight_df"]
                res = compare_weight_frames(ref, cand)
                ok = res["n_dates_different"] == 0 and res["reference_only_dates"] == 0 and res["candidate_only_dates"] == 0
                n_pass += ok
                n_fail += (not ok)
                log.add("future_action_invariance", f"{key}:{sym}x{k}", ok, res)
        # joint independent factors on every symbol
        rng = np.random.default_rng(20260927)
        joint = {s: float(f) for s, f in zip(SYMBOLS_BY_KEY[key], rng.choice([40.0, 0.1, 1.5, 7.0, 0.125], len(SYMBOLS_BY_KEY[key])))}
        with patched(rescale=joint):
            cand = compute_decisions(key)["month_end_weight_df"]
        res = compare_weight_frames(ref, cand)
        ok = res["n_dates_different"] == 0 and res["reference_only_dates"] == 0 and res["candidate_only_dates"] == 0
        n_pass += ok
        n_fail += (not ok)
        log.add("future_action_invariance", f"{key}:joint{joint}", ok, res)
        summary[key] = {"n_pass": int(n_pass), "n_fail": int(n_fail), "n_decisions": int(len(ref))}
    return summary


def truncation_tests(log: ResultLog, sessions: pd.DatetimeIndex) -> dict:
    summary = {}
    for key in STRATEGY_KEYS:
        with patched():
            full = compute_decisions(key)["month_end_weight_df"]
        n_pass = n_fail = 0
        for cut in CUTOFFS:
            cut_ts = pd.Timestamp(cut)
            with patched(hard_end_str=cut):
                trunc = compute_decisions(key, end_date_str=cut)["month_end_weight_df"]
            # completed months: month whose last XNYS session <= cut
            month_last = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()

            def completed(ts):
                p = pd.Timestamp(ts).to_period("M")
                return p in month_last.index and month_last.loc[p] <= cut_ts

            trunc_c = trunc.loc[[completed(t) for t in trunc.index]]
            full_c = full.loc[[(completed(t) and pd.Timestamp(t).to_period("M") <= cut_ts.to_period("M")) for t in full.index]]
            res = compare_weight_frames(full_c, trunc_c)
            prov = trunc.loc[[not completed(t) for t in trunc.index]]
            prov_rows = []
            for t, row in prov.iterrows():
                full_row = full.loc[[i for i in full.index if pd.Timestamp(i).to_period("M") == pd.Timestamp(t).to_period("M")]]
                prov_rows.append({
                    "label": str(pd.Timestamp(t).date()),
                    "provisional_weights": {c: round(float(v), 6) for c, v in row.items() if abs(v) > 1e-12},
                    "final_weights": ({c: round(float(v), 6) for c, v in full_row.iloc[0].items() if abs(v) > 1e-12}
                                      if len(full_row) else None),
                    "equal_to_final": bool(len(full_row) and np.allclose(full_row.iloc[0].reindex(row.index).fillna(0).values,
                                                                        row.fillna(0).values)),
                })
            ok = (res["n_dates_different"] == 0 and res["reference_only_dates"] == 0
                  and res["candidate_only_dates"] == 0 and res["n_dates_compared"] > 0)
            res["provisional_rows"] = prov_rows
            n_pass += ok
            n_fail += (not ok)
            log.add("truncation_prefix", f"{key}:end={cut}", ok, res)
        summary[key] = {"n_pass": int(n_pass), "n_fail": int(n_fail)}
    return summary


def tr_multiplicativity(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """TR daily return vs CAPITALSPECIAL (C_t + D_t)/C_{t-1} - 1.  If Norgate TR is a multiplicative back-adjustment
    of the CAPITALSPECIAL series by dividends, the two agree to float precision and every TR return ratio is
    vintage-invariant (a future dividend multiplies the whole history by one constant)."""
    rows = []
    for sym in ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ", "QQQ", "SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV",
                "XLP", "XLK", "IEF"):
        tr = load_norgate_full(sym, "TOTALRETURN")["Close"].astype(float)
        cs = load_norgate_full(sym, "CAPITALSPECIAL")
        c = cs["Close"].astype(float)
        d = cs["Dividend"].astype(float).fillna(0.0)
        idx = tr.index.intersection(c.index)
        tr, c, d = tr.loc[idx], c.loc[idx], d.loc[idx]
        r_tr = tr / tr.shift(1) - 1
        # Norgate places Dividend on the LAST CUM session (row t); the price drop is at t+1 (ex-date).
        # (verified: SPY Dividend on 2008-12-18, ex-date 2008-12-19).  So r_TR(t+1) = (C_{t+1} + D_t)/C_t - 1.
        r_cs_same_row = (c + d) / c.shift(1) - 1
        r_cs = (c + d.shift(1).fillna(0.0)) / c.shift(1) - 1
        diff = (r_tr - r_cs).abs().dropna()
        diff_same_row = (r_tr - r_cs_same_row).abs().dropna()
        # Norgate-style factor convention: TR back-adjusts by (1 - D_t / C_t) -> r_TR(t+1) = C_{t+1}/(C_t - D_t) - 1
        r_cs_factor = c / (c.shift(1) - d.shift(1).fillna(0.0)) - 1
        diff_factor = (r_tr - r_cs_factor).abs().dropna()
        div_days = d.shift(1)[d.shift(1) > 0].index
        rows.append({
            "symbol": sym, "n_days": int(len(diff)), "n_dividend_days": int(len(div_days)),
            "max_abs_dev": float(diff.max()),
            "max_abs_dev_if_dividend_on_same_row": float(diff_same_row.max()),
            "max_abs_dev_factor_convention": float(diff_factor.max()), "p99_abs_dev": float(diff.quantile(0.99)),
            "max_abs_dev_on_dividend_days": float(diff.reindex(div_days).max()) if len(div_days) else 0.0,
            "max_abs_dev_non_dividend_days": float(diff.drop(div_days, errors="ignore").max()),
            "worst_date": str(diff.idxmax().date()),
            "tr_over_cs_ratio_first": float(tr.iloc[0] / c.iloc[0]), "tr_over_cs_ratio_last": float(tr.iloc[-1] / c.iloc[-1]),
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    if "--tr-only" in sys.argv:
        trm = tr_multiplicativity(None)
        trm.to_csv(OUT_DIR / "taa_tr_multiplicativity.csv", index=False)
        jp = OUT_DIR / "taa_invariance_results.json"
        if jp.exists():
            payload = json.loads(jp.read_text(encoding="utf-8"))
            payload["tr_multiplicativity"] = trm.to_dict("records")
            jp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(trm.to_string(index=False))
        return
    warm_cache()
    sessions = xnys_sessions()
    log = ResultLog("taa")
    inv = invariance_tests(log)
    trn = truncation_tests(log, sessions)
    trm = tr_multiplicativity(sessions)
    trm.to_csv(OUT_DIR / "taa_tr_multiplicativity.csv", index=False)
    payload = {"invariance_summary": inv, "truncation_summary": trn,
               "tr_multiplicativity": trm.to_dict("records"), "rows": log.rows,
               "runtime_sec": round(time.time() - t0, 1)}
    (OUT_DIR / "taa_invariance_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"invariance_summary": inv, "truncation_summary": trn}, indent=1))
    print(trm.to_string(index=False))


if __name__ == "__main__":
    main()
