"""Reviewer scratch (report lens): the embedded /*CHARTS*/ JSON: lengths, nulls, monotone dates, value sanity."""
import json, re, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[5]
src = Path(sys.argv[1])
s = src.read_text(encoding="utf-8")
m = re.search(r'<script type="application/json" id="ch">(.*?)</script>', s, flags=re.S)
CH = json.loads(m.group(1).replace("<\/", "</"))
REP = WT / "results/research/portfolio/fund_products_20261005/report"
S = json.loads((REP / "study.json").read_text(encoding="utf-8"))
for name, ch in CH.items():
    d = ch["dates"]
    print(f"== {name}: {len(d)} dates {d[0]} .. {d[-1]} sorted={d == sorted(d)} unique={len(set(d)) == len(d)}")
    for sr in ch["series"]:
        p = sr["pts"]
        nul = [i for i, v in enumerate(p) if v is None]
        vals = [v for v in p if v is not None]
        inner = [i for i in nul if 0 < i < len(p) - 1 and any(x is not None for x in p[:i]) and any(x is not None for x in p[i + 1:])]
        print(f"   {sr['l']:<14} len={len(p)} same_len={len(p) == len(d)} nulls={len(nul)} (leading={sum(1 for i in nul if all(x is None for x in p[:i + 1]))}, inner={len(inner)}) "
              f"first={vals[0] if vals else None} last={vals[-1] if vals else None} min={min(vals):.3f} max={max(vals):.3f} colour={sr['c']} dash={sr.get('dash', False)}")
# cross-checks: last nav vs wealth implied by CAGR
K = S["books"]
for lab, key in (("Growth", "GR1"), ("Growth Plus", "GR2"), ("Aggressive", "GR3"), ("Monthly", "S9 incumbent launch")):
    q = K[key]["q"]
    wealth = (1 + q["cagr"]) ** (q["n"] / 252)
    sr = next(x for x in CH["nav"]["series"] if x["l"] == lab)
    print(f"{lab}: wealth from CAGR {wealth:.3f} vs last chart nav {sr['pts'][-1]} first {sr['pts'][0]}")
print("nav date sample:", CH["nav"]["dates"][:3], CH["nav"]["dates"][-3:])
print("tick label test: dates ending '-01' with year%3==2:", [d for d in CH["nav"]["dates"] if d.endswith("-01") and int(d[:4]) % 3 == 2][:12])
for nm in ("roll", "corr", "nasdaq"):
    print(nm, "date format sample", CH[nm]["dates"][:2], "labels:", [d for d in CH[nm]["dates"] if d.endswith("-01") and int(d[:4]) % 3 == 2][:8])
