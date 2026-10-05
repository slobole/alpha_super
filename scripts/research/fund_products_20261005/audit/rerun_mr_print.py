"""Compact tables from rerun_mr_report.json (audit task rerun_mr)."""

import json
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
rep = json.loads((WT / "results/research/portfolio/fund_products_20261005/audit/rerun_mr/rerun_mr_report.json").read_text(encoding="utf-8"))
print("dry", rep["dry_run_cash_from_main_bool"])
print("\n== new vs MAIN stored")
for k, d in rep["new_vs_main"].items():
    print(k, {x: d[x] for x in ("new_first", "new_last", "new_sessions", "old_last", "old_sessions", "common_sessions", "index_identical",
                                 "max_abs_daily_ret_diff", "sessions_with_ret_diff_gt_1e-12", "first_ret_diverging_date", "daily_ret_corr",
                                 "nav_ratio_last_common", "max_abs_nav_rel_diff", "new_final_nav", "old_final_nav", "tx_count_new", "tx_count_old",
                                 "tx_count_new_stock", "tx_count_old_stock", "tx_count_new_bil", "tx_count_old_bil", "tx_only_new", "tx_only_old")})
    if d["first_tx_diff"]:
        print("   first tx diffs:", d["first_tx_diff"][:6])
print("\n== BIL mark check", rep["bil_mark_check"])
print("ev.capsule vs inline:", rep["ev_capsule_vs_inline_max_abs_diff"])
print("\n== PM book")
pm = rep["pm_book"]
for k, v in pm.items():
    if k != "pod_100k_vs_pm_pod_500k":
        print(" ", k, v)
for k, v in pm["pod_100k_vs_pm_pod_500k"].items():
    print(" ", k, v)
print("\n== tidy", rep["tidy_csv"])


def row(s):
    if not s:
        return "n/a"
    return f"n={s['sessions']:5d} {s['first']}..{s['last']}  tot {s['total_return'] * 100:8.2f}%  CAGR {s['cagr'] * 100:6.2f}%  vol {s['vol'] * 100:5.2f}%  Sharpe {s['sharpe']:5.3f}  maxDD {s['max_dd'] * 100:6.2f}%"


print("\n== stats")
for w, d in rep["stats"].items():
    print(w)
    for n, s in d.items():
        print(f"   {n:<24}", row(s))
print("\n== capsule restarted at window start")
for w, d in rep["stats_capsule_restarted_at_window_start"].items():
    for m, s in d.items():
        print(f"   {w:<26} {m:<5}", row(s))
print("\n== corr", json.dumps(rep["corr"], indent=1))
print("\n== exposure", json.dumps(rep["exposure"], indent=1))
print("\n== extra cost")
for lab, d in rep["extra_cost"].items():
    print(lab)
    for w, dd in d.items():
        for n, s in dd.items():
            print(f"   {w:<26} {n:<12}", row(s))
print("\n== calendar years (capsule_bil / capsule_cash / dv2_bil / hpi_bil)")
cy = rep["calendar_years"]
for y in cy["capsule_bil"]:
    print(f"   {y}: " + "  ".join(f"{cy[n][y] * 100:7.2f}%" for n in ("capsule_bil", "capsule_cash", "dv2_bil", "hpi_bil")))
