"""Reviewer scratch: print the shape of the report JSONs (read-only)."""
import json, sys
from pathlib import Path
WT = Path(__file__).resolve().parents[5]
REP = WT / "results/research/portfolio/fund_products_20261005/report"
def shape(o, depth=0, maxd=3, pre=""):
    if isinstance(o, dict):
        ks = list(o.keys())
        if depth >= maxd:
            print(pre + f"dict[{len(ks)}] {ks[:8]}"); return
        for k in ks[:40]:
            v = o[k]
            if isinstance(v, (dict, list)):
                print(pre + f"{k}: {type(v).__name__}[{len(v)}]")
                shape(v, depth + 1, maxd, pre + "  ")
            else:
                print(pre + f"{k}: {v!r}"[:160])
        if len(ks) > 40: print(pre + f"... +{len(ks)-40} keys")
    elif isinstance(o, list):
        if o and isinstance(o[0], (dict, list)) and depth < maxd:
            shape(o[0], depth + 1, maxd, pre + "  [0] ")
        else:
            print(pre + repr(o[:6])[:160])
for n in sys.argv[1:]:
    name, maxd = n.split(":")
    print("=" * 20, name)
    shape(json.loads((REP / f"{name}.json").read_text(encoding="utf-8")), maxd=int(maxd))
