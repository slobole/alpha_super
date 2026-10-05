"""Reviewer scratch (report lens): independent bootstrap of P(max DD < limit) for the headline books (own drawdown
loop; the study's index convention: stationary, block 63, 2,000 paths x 10 seeds). Read-only."""
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
def decay(fr, k):
    out = fr.copy(); win = fr.loc[LONG_START:END]
    for a in g.CAPSULE_OF:
        if a in out.columns:
            out[a] = fr[a] - (1 - k) * float((win[a] - win[TBILL]).mean())
    return out
books = {"GR1": g.PRODUCTS["GR1"], "GR2": g.PRODUCTS["GR2"], "GR3": g.PRODUCTS["GR3"], "S9": g.INCUMBENT}
n = len(main.loc[start:END])
idx = [lib.evaluation.stationary_bootstrap_index_mat(n, 2000, 63.0, 20260929 + s) for s in range(10)]   # (paths, days)
res = {}
for label, fr in (("main", main), ("planning", decay(f5, 0.75))):
    R = np.column_stack([g.book_returns(fr, w, start).to_numpy() for w in books.values()])
    acc = {L: [] for L in (-0.20, -0.25, -0.30)}
    for mat in idx:
        dd = np.zeros((mat.shape[0], R.shape[1]))
        for a in range(0, mat.shape[0], 250):
            smp = R[mat[a:a + 250]]                         # (paths, days, books)
            nav = np.cumprod(1.0 + smp, axis=1)
            peak = np.maximum.accumulate(np.maximum(nav, 1.0), axis=1)
            dd[a:a + 250] = (nav / peak - 1.0).min(axis=1)
        for L in acc:
            acc[L].append((dd < L).mean(axis=0))
    res[label] = {name: {f"p{int(-L * 100)}": [round(float(np.mean([s[j] for s in acc[L]])), 4), round(float(np.max([s[j] for s in acc[L]])), 4)] for L in acc} for j, name in enumerate(books)}
    print(label, json.dumps(res[label]), flush=True)
(g.STUDY / "audit" / "review" / "report_lens" / "tails_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
