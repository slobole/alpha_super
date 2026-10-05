"""Read-only smoke test: does the v4 data builder (report_a6.metrics on lib.load_inputs) still reproduce report_a6.json?

Recomputes four rows (d_launch, d_target, g_launch, g_plus) and the S&P 500 benchmark with today's local data and
compares with the stored JSON. Writes only to the audit output folder.
"""
import json
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
SRC = WT / "scripts" / "research" / "fund_products_20260930"
REP = WT / "results" / "research" / "portfolio" / "fund_products_20260930" / "report"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "report_map"
sys.path.insert(0, str(SRC))
t0 = time.time()
import report_a6 as ra  # noqa: E402

lib, ga, Book, TBILL = ra.lib, ra.ga, ra.Book, ra.TBILL
data = lib.load_inputs()
print("load_inputs s", round(time.time() - t0, 1), "index", data["index"][0].date(), data["index"][-1].date(), "cols", len(data["cash_long"].columns), flush=True)
frame, start = ga.frames(data)["main"]
rf = frame[TBILL]
D = json.loads((REP / "report_a6.json").read_text(encoding="utf-8"))
out = {"index_end": str(data["index"][-1].date()), "rows": {}}
for key in ("d_launch", "d_target", "g_launch", "g_plus"):
    w = D["rows"][key]["weights"]
    w = {k: v / sum(w.values()) for k, v in w.items()}
    r = lib.book_returns(frame, Book(key, tuple(w), "EQ", w), start)
    m = ra.metrics(r, data, rf)
    old = D["rows"][key]["m"]
    diff = {k: (m[k], old[k], m[k] - old[k]) for k in ("cagr", "maxdd", "xsharpe", "sharpe", "net_cagr", "sortino", "cvar5_21d", "worst_year", "worst_12m", "beta")}
    nav_same = m["nav"] == old["nav"]
    out["rows"][key] = {"diff": diff, "nav_identical": nav_same, "years_max_abs_diff": max(abs(m["years"][int(y)] - v) for y, v in old["years"].items())}
    print(key, "nav_identical", nav_same, {k: f"{a:.6f} vs {b:.6f}" for k, (a, b, c) in diff.items() if abs(c) > 1e-9} or "all 10 metrics equal to 1e-9", flush=True)
b = ra.metrics(data["bench"]["SPXTR"].loc[ra.LONG_START:ra.fp.END], data, rf)
ob = D["bench"]["S&P 500"]
out["bench_spx"] = {"cagr": [b["cagr"], ob["cagr"]], "maxdd": [b["maxdd"], ob["maxdd"]]}
print("S&P 500 cagr", b["cagr"], ob["cagr"], "maxdd", b["maxdd"], ob["maxdd"], flush=True)
out["seconds"] = round(time.time() - t0, 1)
(OUT / "repro.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print("done s", out["seconds"])
