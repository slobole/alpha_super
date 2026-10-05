"""Read-only: shape of the newer capsule-page data files in MAIN (page_v2_data.json, book_weights.json)."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
M = Path(r"C:/Users/User/Documents/workspace/alpha_super/results/research/mr_capsule_build_20261004")
P = json.loads((M / "page_v2_data.json").read_text(encoding="utf-8"))
print("page_v2_data top:", list(P))
print("meta:", P["meta"])
print("ndx_legs:", {k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in P["ndx_legs"].items()})
print("capsule keys:", list(P["capsule"]))
print("capsule.bil_full keys:", list(P["capsule"]["bil_full"]), "stats keys:", list(P["capsule"]["bil_full"]["stats"]))
print("capsule.bil_full stats:", {k: round(v, 4) for k, v in P["capsule"]["bil_full"]["stats"].items()})
print("charts keys:", {k: list(v) for k, v in P["charts"].items()})
c = P["charts"]["book_long"]["G3"]
print("chart series shape: keys", list(c), "n", len(c["d"]), c["d"][0], c["d"][-1], c["w"][:2], c["dd"][:2])
print("book keys:", {w: list(v) for w, v in P["book"].items()})
for k, v in P["book"]["long"].items():
    s = v["stats"]
    print(f"  book long {k:22s} cagr {s['cagr']:.4f} sharpe {s['sharpe']:.3f} xs {s['sharpe_excess_tbill']:.3f} maxdd {s['max_dd']:.4f} blocks {v.get('blocks')}")
B = json.loads((M / "book_weights.json").read_text(encoding="utf-8"))
print("book_weights top:", list(B))
for leg, d in B["legs"].items():
    print("  LEG", leg, {w: (round(s["cagr"], 4), round(s["sharpe"], 3), round(s["max_dd"], 4)) for w, s in d.items()})
for taa in ("TAA 3x", "TAA 3x 1N"):
    for wn, d in B[taa].items():
        s = d["2008-26"]
        print(f"  {taa:10s} | {wn:46s} cagr {s['cagr']:.4f} sharpe {s['sharpe']:.3f} maxdd {s['max_dd']:.4f} worst_year {s['worst_year']:.4f}")
