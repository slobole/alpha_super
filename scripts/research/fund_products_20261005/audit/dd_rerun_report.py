"""defensive_delta audit, step 2c: re-run report_a6.py (full statistics, factor alpha, capacity of the A6-d slots, bench)
from the STORED a6.json / a6d.json, redirected to the audit folder, and compare with the stored report_a6.json."""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
sys.dont_write_bytecode = True
import data.norgate_loader  # noqa: F401
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20260930"))
import fp_lib as fp
OLD = fp.STUDY
NEW = WT / "results/research/portfolio/fund_products_20261005/audit/defensive_delta/rerun_report_a6"
(NEW / "report").mkdir(parents=True, exist_ok=True)
for f in ("a6.json", "a6d.json"):
    shutil.copy(OLD / "report" / f, NEW / "report" / f)
fp.STUDY = NEW
import report_a6
rc = report_a6.main()

def flat(x, p=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items(): out.update(flat(v, f"{p}{k}."))
    elif isinstance(x, list):
        for i, v in enumerate(x): out.update(flat(v, f"{p}[{i}]."))
    else: out[p.rstrip(".")] = x
    return out
a = flat(json.loads((OLD / "report" / "report_a6.json").read_text(encoding="utf-8")))
b = flat(json.loads((NEW / "report" / "report_a6.json").read_text(encoding="utf-8")))
diffs, n_num = [], 0
for k in sorted(set(a) & set(b)):
    x, y = a[k], b[k]
    if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool):
        n_num += 1
        if x != y: diffs.append((abs(x - y), k, x, y))
    elif x != y: diffs.append((float("inf"), k, x, y))
diffs.sort(key=lambda t: -t[0])
sec = {}
for d, k, x, y in diffs:
    key = ".".join(k.split(".")[:2]) if k.startswith("rows.") else k.split(".")[0]
    s = sec.setdefault(key, [0, 0.0, ""]); s[0] += 1
    if d > s[1]: s[1], s[2] = d, k
res = {"fields_old": len(a), "fields_new": len(b), "only_old": sorted(set(a) - set(b))[:30], "only_new": sorted(set(b) - set(a))[:30],
       "numeric_compared": n_num, "n_different": len(diffs), "max_abs_diff": diffs[0][0] if diffs else 0.0, "by_section": sec,
       "top": [{"d": d, "field": k, "old": x, "new": y} for d, k, x, y in diffs[:30]]}
(NEW.parent / "compare_report_a6.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print("COMPARE", json.dumps(res, indent=1, default=str)[:6000])
raise SystemExit(rc)
