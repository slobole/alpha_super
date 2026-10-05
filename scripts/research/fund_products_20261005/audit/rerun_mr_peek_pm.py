"""Peek at the PM book pickle (read-only)."""
import pickle, sys
from pathlib import Path
import pandas as pd
MAIN = Path(r"C:/Users/User/Documents/workspace/alpha_super")
p = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751/mr_capsule_bil.pkl"
with open(p, "rb") as fh:
    pf = pickle.load(fh)
print(type(pf))
print([a for a in dir(pf) if not a.startswith("__")][:80])
res = pf.results
print(res.columns.tolist()[:40])
print(res.head(3).to_string()); print(res.tail(3).to_string())
for a in ("pod_results_dict", "pods", "pod_list", "strategies", "strategy_list", "pod_strategy_dict", "sleeve_results"):
    if hasattr(pf, a):
        v = getattr(pf, a)
        print(a, type(v), (list(v)[:5] if hasattr(v, "__iter__") else v))
