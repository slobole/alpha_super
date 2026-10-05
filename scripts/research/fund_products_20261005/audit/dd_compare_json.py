"""defensive_delta audit: field-by-field comparison of a re-run json with the stored one (numbers, strings, structure).

Usage: python dd_compare_json.py a6d | a6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
OLD = WT / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"
AUD = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "defensive_delta"


def flat(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flat(v, f"{prefix}[{i}]."))
    else:
        out[prefix.rstrip(".")] = x
    return out


def main() -> int:
    which = sys.argv[1]
    old = flat(json.loads((OLD / f"{which}.json").read_text(encoding="utf-8")))
    new = flat(json.loads((AUD / f"rerun_{which}" / "report" / f"{which}.json").read_text(encoding="utf-8")))
    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    n_num = n_other = 0
    diffs = []
    for k in sorted(set(old) & set(new)):
        a, b = old[k], new[k]
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            n_num += 1
            d = abs(float(a) - float(b))
            if d > 0:
                diffs.append((d, k, a, b))
        else:
            n_other += 1
            if a != b:
                diffs.append((float("inf"), k, a, b))
    diffs.sort(key=lambda t: -t[0])
    sections = {}
    for d, k, a, b in diffs:
        sec = k.split(".")[0]
        s = sections.setdefault(sec, {"count": 0, "max": 0.0, "where": ""})
        s["count"] += 1
        if d > s["max"]:
            s["max"], s["where"] = d, k
    res = {"which": which, "fields_old": len(old), "fields_new": len(new), "numeric_compared": n_num, "other_compared": n_other,
           "only_in_old": only_old[:50], "only_in_new": only_new[:50], "n_only_old": len(only_old), "n_only_new": len(only_new),
           "n_different": len(diffs), "max_abs_diff": diffs[0][0] if diffs else 0.0,
           "by_section": sections, "top_diffs": [{"d": d, "field": k, "old": a, "new": b} for d, k, a, b in diffs[:25]]}
    (AUD / f"compare_{which}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("only_in_old", "only_in_new")}, indent=1, default=str))
    if only_old or only_new:
        print("only_in_old", only_old[:20], "only_in_new", only_new[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
