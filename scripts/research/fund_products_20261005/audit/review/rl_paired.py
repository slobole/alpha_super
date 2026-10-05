"""Reviewer scratch (report lens): paired bootstrap share GR1 vs the monthly book S9 on excess Sharpe and CAGR, at
0 / +5 / +10 bps and in the planning frame. Same index matrices as the study (stationary, block 63, 2,000 x 10 seeds).
Read-only: no Lab.tails, no ledger."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
STUDY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY_DIR))
import g_lib as g
from g_lib import END, LONG_START, TBILL, lib

data = g.load_inputs()
F = g.frames(data)
main, start = F["main"]
f5 = F["s3_plus_5bps"][0]
rf = main[TBILL].loc[start:END].to_numpy()
GR1, S9 = g.PRODUCTS["GR1"], g.INCUMBENT

def frame_plus(n_bps):
    fr = main.copy()
    for c in set(GR1) | set(S9):
        fr[c] = main[c] - (n_bps / 5.0) * (main[c] - f5[c])
    return fr

def decay(fr, k):
    out = fr.copy(); win = fr.loc[LONG_START:END]
    for a in g.CAPSULE_OF:
        if a in out.columns:
            out[a] = fr[a] - (1 - k) * float((win[a] - win[TBILL]).mean())
    return out

cases = {"+0 bps": main, "+4 bps": frame_plus(4), "+5 bps": f5, "+10 bps": frame_plus(10), "planning (k 0.75, +5 bps)": decay(f5, 0.75), "k 0.75, +0 bps": decay(main, 0.75)}
n = len(rf)
idx = [np.ascontiguousarray(lib.evaluation.stationary_bootstrap_index_mat(n, 2000, 63.0, 20260929 + s).T.astype(np.int32)) for s in range(10)]
res = {}
for label, fr in cases.items():
    R = np.column_stack([g.book_returns(fr, w, start).to_numpy() for w in (GR1, S9)])
    xs_all, cg_all = [], []
    for it in idx:
        for a in range(0, it.shape[1], 250):
            ii = it[:, a:a + 250]
            smp = R[ii]
            x = smp - rf[ii][:, :, None]
            xs_all.append(x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(252))
            cg_all.append(np.expm1(np.log1p(smp).sum(axis=0) * 252 / n))
    xs, cg = np.vstack(xs_all), np.vstack(cg_all)
    gap = xs[:, 0] - xs[:, 1]
    res[label] = {"share_xs": float((gap > 0).mean()), "share_cagr": float((cg[:, 0] > cg[:, 1]).mean()), "gap_xs_p5_50_95": [float(v) for v in np.percentile(gap, [5, 50, 95])],
                  "gap_cagr_p50": float(np.median(cg[:, 0] - cg[:, 1])), "paths": int(len(gap))}
    print(label, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v] if isinstance(v, list) else v) for k, v in res[label].items()}, flush=True)
(g.STUDY / "audit" / "review" / "report_lens" / "paired_gr1_vs_s9.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
