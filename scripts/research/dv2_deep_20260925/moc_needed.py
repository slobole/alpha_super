"""Which (date, symbol) 15:45 states the MOC test needs (SPEC amendment A3, revised).

needed[t, i] = possible entry at 15:45 on t:  member & close-based DV2(126) < 25 & Close > 0.95 * SMA200
             | held at close t-1 or t by any finalist in its next-open or perfect-close replica run (exits).
A stock outside this set on a day keeps the final-close state in the exact test and is counted.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp
import phase5_finalists as f5
import phase2_mechanism as p2

p = rp.Panel("sp500")
with np.errstate(invalid="ignore"):
    need = np.asarray(p.member) & (np.asarray(p.eng["dv2"]) < 25) & (np.asarray(p.C) > 0.95 * np.asarray(p.eng["sma_200"]))
for alias, rule in f5.FINALISTS.items():
    for r in (rule, rule.with_(timing="moc")):
        res = rp.run(p, r, start="2016-01-04", end="2026-09-24")
        sh = p2.positions_matrix(p, res) != 0
        t0 = p.dates.get_loc(res.dates[0])
        held = np.zeros_like(need)
        held[t0:t0 + len(res.dates)] = sh
        need |= held
        need[1:] |= held[:-1]
out = rp.CACHE_DIR_PATH / "moc_needed.npy"
np.save(out, need)
m = (p.dates >= "2016-01-04")
print("avg needed per day", need[m].sum(axis=1).mean(), "vs members", np.asarray(p.member)[m].sum(axis=1).mean())
