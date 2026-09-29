"""Tactical FI L14: ALFRED real-time replay of the month-end decisions (research-only; no source edits).

ALFRED coverage (probed 2026-09-27): DGS10 / DGS3MO vintages from ~2005-06-28, DAAA / DBAA from 2014-04-02.
Before those dates only today's vintage exists, so the replay has two scopes:
  all4          decisions >= 2014-04-30: every series real-time.
  treasury_rt   decisions >= 2005-06-30: DGS10/DGS3MO real-time, DAAA/DBAA current vintage (credit sleeve's
                Moody's inputs cannot be replayed before 2014).
For each decision T and vintage date vd in {T (same-day vintage, optimistic about the time of day),
prev session (conservative)}:
  * last 45 days of each series as known on vd (batched window requests);
  * older history from the latest annual full-history vintage <= vd (history revisions);
  * the module's own build_month_end_signal_and_weight_df recomputes the whole rule on that panel; row T is
    compared with the frozen current-vintage contract.
Also: publication-day evidence (is obs T-1 in the vintage dated T / T-1?), revision statistics, engine re-run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import BOOK_END, BOOK_START, OUT, dump_json, harness, metric_rows
from def_tfi_alfred import full_vintage, window_panel
import def_tfi as t

import numpy as np
import pandas as pd

from strategies.taa_beyond_6040 import strategy_taa_tactical_fixed_income_ief_lqd as m

SERIES = t.SERIES
FIRST_VINTAGE = {"DGS10": pd.Timestamp("2005-06-28"), "DGS3MO": pd.Timestamp("2005-06-30"),
                 "DAAA": pd.Timestamp("2014-04-02"), "DBAA": pd.Timestamp("2014-04-02")}
WINDOW = 45


def main():
    t0 = time.time()
    log = harness.ResultLog("tactical_fi_vintage")
    px, yield_df, snaps = t.load_inputs()
    sessions = pd.DatetimeIndex(px.index)
    sig, w = t.weights_from(yield_df, sessions, m.DEFAULT_CONFIG.last_complete_signal_month_str)
    dec = pd.DatetimeIndex(sig.index)
    prev = pd.DatetimeIndex([sessions[sessions < d][-1] for d in dec])
    cur = {s: yield_df[s].dropna() for s in SERIES}

    win = {}
    for s in SERIES:
        vds = [d for d in list(dec) + list(prev) if d >= FIRST_VINTAGE[s]]
        win[s] = window_panel(s, vds, window_days=WINDOW, batch=10)
        print(s, "window vintages", len(win[s]), f"{time.time()-t0:.0f}s", flush=True)
    # annual full-history grid (December decision dates) for history revisions
    grid = {}
    for s in SERIES:
        gd = [d for d in dec if d.month == 12 and d >= FIRST_VINTAGE[s]]
        grid[s] = {}
        for d in gd:
            ser = full_vintage(s, d)
            if ser is not None:
                grid[s][d] = ser
        print(s, "full vintages", len(grid[s]), f"{time.time()-t0:.0f}s", flush=True)

    # ---- revision statistics (window values vs current vintage) ----
    rev = {}
    for s in SERIES:
        diffs = []
        for vd, ser in win[s].items():
            c = cur[s].reindex(ser.index)
            dd = (ser - c).dropna()
            diffs.append(dd)
        allv = pd.concat(diffs) if diffs else pd.Series(dtype=float)
        hist_rev = []
        for vd, ser in grid[s].items():
            c = cur[s].reindex(ser.index)
            hist_rev.append((ser - c).dropna())
        hv = pd.concat(hist_rev) if hist_rev else pd.Series(dtype=float)
        rev[s] = {"window_obs_compared": int(len(allv)), "window_n_revised": int((allv.abs() > 1e-9).sum()),
                  "window_max_abs_revision_pp": float(allv.abs().max()) if len(allv) else None,
                  "full_history_obs_compared": int(len(hv)), "full_history_n_revised": int((hv.abs() > 1e-9).sum()),
                  "full_history_max_abs_revision_pp": float(hv.abs().max()) if len(hv) else None}
    print("revisions", rev, flush=True)

    # ---- publication-day evidence ----
    pub_rows = []
    for d, p in zip(dec, prev):
        rec = {"decision_date": d.date(), "prev_session": p.date()}
        for s in SERIES:
            vT, vP = win[s].get(d), win[s].get(p)
            rec[f"{s}_last_obs_in_vintage_T"] = None if vT is None or vT.empty else vT.index[-1].date()
            rec[f"{s}_obs_prev_in_vintage_T"] = None if vT is None else bool(p in vT.index)
            rec[f"{s}_obs_prev_in_vintage_prev"] = None if vP is None else bool(p in vP.index)
        pub_rows.append(rec)
    pub = pd.DataFrame(pub_rows)
    pub.to_csv(OUT / "tfi_alfred_publication_day_evidence.csv", index=False)
    pub_summary = {}
    for s in SERIES:
        a = pub[f"{s}_obs_prev_in_vintage_T"].dropna().astype(bool)
        b = pub[f"{s}_obs_prev_in_vintage_prev"].dropna().astype(bool)
        pub_summary[s] = {"n_decisions": int(len(a)), "obs_Tminus1_in_vintage_T_share": float(a.mean()) if len(a) else None,
                          "obs_Tminus1_missing_from_vintage_T": [str(x) for x in pub.loc[a.index[~a], "decision_date"]][:20],
                          "obs_Tminus1_in_vintage_Tminus1_share": float(b.mean()) if len(b) else None}
    print("publication", pub_summary, flush=True)

    # ---- replay ----
    def hist_source(s, vd):
        ds = [g for g in grid[s] if g <= vd]
        return grid[s][max(ds)] if ds else cur[s]

    rows = []
    for mode, vlist in (("vintage_T", dec), ("vintage_prev_session", prev)):
        for d, vd in zip(dec, vlist):
            have = {s: vd in win[s] for s in SERIES}
            if not (have["DGS10"] and have["DGS3MO"]):
                continue
            scope = "all4" if all(have.values()) else "treasury_rt"
            for hist_mode in ("history_realtime", "history_current"):
                parts = []
                for s in SERIES:
                    if not have[s]:
                        parts.append(cur[s].loc[:d].rename(s))
                        continue
                    h = hist_source(s, vd) if hist_mode == "history_realtime" else cur[s]
                    h = h[h.index <= vd - pd.Timedelta(days=WINDOW)]
                    parts.append(pd.concat([h, win[s][vd]]).sort_index().rename(s))
                pan = pd.concat(parts, axis=1).sort_index().loc[:d]
                try:
                    sv, _ = t.weights_from(pan, sessions, str(d.to_period("M")))
                except Exception as exc:
                    rows.append({"mode": mode, "hist": hist_mode, "scope": scope, "decision_date": d, "error": str(exc)})
                    continue
                r, f = sv.loc[d], sig.loc[d]
                rows.append({"mode": mode, "hist": hist_mode, "scope": scope, "decision_date": d, "vintage_date": vd,
                             "obs_frozen": f["observation_date"], "obs_rt": r["observation_date"],
                             "term_frozen": f["term_spread_float"], "term_rt": r["term_spread_float"],
                             "credit_frozen": f["credit_spread_float"], "credit_rt": r["credit_spread_float"],
                             "term_thr_frozen": f["term_threshold_float"], "term_thr_rt": r["term_threshold_float"],
                             "credit_thr_frozen": f["credit_threshold_float"], "credit_thr_rt": r["credit_threshold_float"],
                             "term_state_frozen": f["term_state_float"], "term_state_rt": r["term_state_float"],
                             "credit_state_frozen": f["credit_state_float"], "credit_state_rt": r["credit_state_float"]})
        print(mode, "replayed", f"{time.time()-t0:.0f}s", flush=True)
    rt = pd.DataFrame(rows)
    rt.to_csv(OUT / "tfi_fred_vintage_comparison.csv", index=False)

    # ---- flips + engine impact ----
    cash = m.build_causal_cash_return_ser(sessions, snaps[SERIES.index("DGS3MO")].value_ser)
    base = t.run_engine(px, sig, w, cash, snaps)
    windows = {"full": (None, None), "book": (BOOK_START, BOOK_END), "since_2005-07": ("2005-07-01", BOOK_END),
               "since_2014-05": ("2014-05-01", BOOK_END)}
    metrics = metric_rows("baseline_current_vintage", base.results["daily_returns"], windows)
    summary = {}
    ok_rows = rt[rt.get("term_state_rt").notna()] if "term_state_rt" in rt else rt
    for (mode, hist_mode), sub in ok_rows.groupby(["mode", "hist"]):
        key = f"{mode}__{hist_mode}"
        tf = sub["term_state_rt"] != sub["term_state_frozen"]
        cf = sub["credit_state_rt"] != sub["credit_state_frozen"]
        s_all4 = sub["scope"] == "all4"
        summary[key] = {
            "n_decisions_replayed": int(len(sub)), "n_all4": int(s_all4.sum()), "n_treasury_rt_only": int((~s_all4).sum()),
            "n_term_flips": int(tf.sum()), "n_credit_flips": int(cf.sum()), "n_credit_flips_all4": int((cf & s_all4).sum()),
            "n_decisions_any_flip": int((tf | cf).sum()),
            "flips": [{"date": pd.Timestamp(r.decision_date).date().isoformat(), "term": f"{r.term_state_frozen:.0f}->{r.term_state_rt:.0f}",
                       "credit": f"{r.credit_state_frozen:.0f}->{r.credit_state_rt:.0f}", "scope": r.scope,
                       "obs_frozen": str(pd.Timestamp(r.obs_frozen).date()), "obs_rt": str(pd.Timestamp(r.obs_rt).date())}
                      for r in sub[tf | cf].itertuples()],
            "n_obs_date_differs": int((pd.to_datetime(sub["obs_rt"]) != pd.to_datetime(sub["obs_frozen"])).sum()),
            "max_abs_term_spread_diff": float((sub["term_rt"] - sub["term_frozen"]).abs().max()),
            "max_abs_credit_spread_diff": float((sub["credit_rt"] - sub["credit_frozen"]).abs().max()),
            "max_abs_term_threshold_diff": float((sub["term_thr_rt"] - sub["term_thr_frozen"]).abs().max()),
            "max_abs_credit_threshold_diff": float((sub["credit_thr_rt"] - sub["credit_thr_frozen"]).abs().max()),
        }
        w_rt = w.copy()
        for r in sub.itertuples():
            idx = w_rt.index[w_rt["decision_date"] == r.decision_date]
            ief, lqd = 0.5 * r.term_state_rt, 0.5 * r.credit_state_rt
            w_rt.loc[idx[0], ["IEF", "LQD", "Cash"]] = [ief, lqd, 1.0 - ief - lqd]
        s_rt = t.run_engine(px, sig, w_rt, cash, snaps)
        metrics += metric_rows(f"alfred_{key}", s_rt.results["daily_returns"], windows)
        log.add("fred_vintage_flips", key, summary[key]["n_decisions_any_flip"] == 0, {k: v for k, v in summary[key].items() if k != "flips"})
    errs = rt[rt["error"].notna()] if "error" in rt else pd.DataFrame()
    pd.DataFrame(metrics).to_csv(OUT / "tfi_vintage_metrics.csv", index=False)
    dump_json({"first_valid_alfred_vintage_probe": {k: v.date().isoformat() for k, v in FIRST_VINTAGE.items()},
               "revision_statistics": rev, "publication_day_evidence": pub_summary, "flip_summary": summary,
               "n_replay_errors": int(len(errs)), "replay_error_examples": errs.head(5).astype(str).to_dict("records")},
              "tfi_fred_vintage_summary.json")
    log.save("def_tactical_fi_vintage")
    print(pd.DataFrame(metrics).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
