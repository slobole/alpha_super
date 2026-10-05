"""Collect every number the Hebrew report shows into one JSON file (descriptive; nothing is selected here).

Reads the outputs of part_d.py, part_g.py, part_m.py and friction_runs.py and writes
results/research/portfolio/shelf_rebuild_20260929/report/report_data.json.

Usage: python report_data.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

import lib
from lib import END, EXACT_START, LONG_START, TBILL, Book
import part_m

OUT = lib.STUDY / "report"
RECOMMENDED = {"DEF": "DEF-7 | D' + G3", "BAL": "DEF-10 | D' + G3", "GRO": "GRO | D' + G3",
               "GRO_PLUS": "GRO | D' + G'"}
MECHANICAL = {"DEF": "DEF-7 | D* + G*", "DEF10": "DEF-10 | D* + G*", "BAL": "BAL | D* + G*", "GRO": "GRO | D* + G*"}
# Judgement per existing YAML (labelled as judgement in the report): (chip class, Hebrew verdict).
VERDICTS = {
    "yaml:00_0": ("acc", "מוחלף ב״הגנה״"), "yaml:0_1": ("", "ארכיון"), "yaml:0_3": ("acc", "קרוב ל״מאוזן״"),
    "yaml:0_4": ("warn", "חלופה פשוטה ל״צמיחה+״"), "yaml:0_allin": ("", "ארכיון"),
    "yaml:fund_menu_aggressive": ("", "ארכיון"), "yaml:fund_menu_balanced": ("", "ארכיון (מורכב, EOM)"),
    "yaml:fund_menu_defensive": ("", "ארכיון (מורכב, EOM)"), "yaml:fund_menu_growth": ("", "ארכיון"),
    "yaml:fund_menu_low_touch_balanced": ("", "ארכיון"), "yaml:fund_menu_low_touch_defensive": ("", "ארכיון"),
    "yaml:fund_menu_low_touch_growth": ("", "ארכיון"), "yaml:ladder_1_defensive": ("acc", "מוחלף ב״הגנה״"),
    "yaml:ladder_1_defensive_proxy_2008": ("", "ארכיון"), "yaml:ladder_2_balanced": ("acc", "מוחלף ב״מאוזן״"),
    "yaml:ladder_2_balanced_proxy_2008": ("", "ארכיון"), "yaml:ladder_3_growth": ("", "ארכיון"),
    "yaml:ladder_3b_growth_2x": ("", "ארכיון"), "yaml:ladder_3c_growth_2x_btal": ("", "ארכיון"),
    "yaml:ladder_4_growth": ("good", "דרך שדרוג לצמיחה"), "yaml:ladder_4_growth_rebalance": ("good", "דרך שדרוג לצמיחה"),
    "yaml:ladder_4_growth_1n": ("warn", "צמיחה+ עם MR"), "yaml:ladder_4_growth_1n_rebalance": ("warn", "צמיחה+ עם MR"),
    "yaml:ladder_4_growth_inflation_compass_05_rebalance": ("", "ארכיון (Compass)"),
    "yaml:ladder_4_growth_inflation_compass_10_rebalance": ("", "ארכיון (Compass)"),
    "yaml:loren": ("acc", "קרוב ל״צמיחה״ בלי איזון"),
}


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.date().isoformat()
    return obj


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    present = [c for c in columns if c in frame.columns]
    out = frame[present].copy()
    out.insert(0, "name", out.index)
    return clean(out.to_dict(orient="records"))


def breach_table(series: pd.DataFrame) -> pd.DataFrame:
    idx = lib.boot_index(len(series))
    R = series.to_numpy()
    rows = {}
    for j, name in enumerate(series.columns):
        dd = np.array([lib.maxdd(R[i, j]) for i in idx])
        rows[name] = {"boot_dd_p50": float(np.percentile(dd, 50)), "boot_dd_p05": float(np.percentile(dd, 5)),
                      **{f"p_worse_{b}": float(np.mean(dd < -b / 100)) for b in (7, 10, 15, 20, 25)}}
    return pd.DataFrame(rows).T


def monthly_nav(r: pd.Series) -> list:
    nav = (1.0 + r).cumprod()
    m = nav.resample("ME").last()
    return [[d.strftime("%Y-%m"), round(float(v), 5)] for d, v in m.items()]


def weekly_dd(r: pd.Series) -> list:
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    w = dd.resample("W-FRI").min()
    return [[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in w.items()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    index_all = data["index"]
    tb = data["long"][TBILL]
    base = lib.STUDY

    d_books = pd.read_csv(base / "part_d" / "d_books.csv", index_col=0)
    g_books = pd.read_csv(base / "part_g" / "g_books.csv", index_col=0)
    d_sel = json.loads((base / "part_d" / "d_selection.json").read_text(encoding="utf-8"))
    dp_sel = json.loads((base / "part_d" / "d_prime_selection.json").read_text(encoding="utf-8"))
    g_sel = json.loads((base / "part_g" / "g_selection.json").read_text(encoding="utf-8"))
    products = pd.read_csv(base / "part_m" / "products.csv", index_col=0)
    sens = pd.read_csv(base / "part_m" / "products_sensitivity.csv")
    cap = pd.read_csv(base / "part_m" / "products_capacity.csv", index_col=0)
    refs = pd.read_csv(base / "part_m" / "references.csv", index_col=0)
    bench = pd.read_csv(base / "part_m" / "benchmarks.csv", index_col=0)
    sleeves = pd.read_csv(base / "part_m" / "sleeves.csv", index_col=0)
    comps = json.loads((base / "part_m" / "components.json").read_text(encoding="utf-8"))
    prod_long = pd.read_csv(base / "part_m" / "products_long_returns.csv.gz", index_col=0, parse_dates=True)

    # Reference books and components on LONG, for the breach table and the charts.
    comp = part_m.components()
    ref_series = {"D* (CORE5 + EOM)": lib.book_returns(data["long"], comp["D"], LONG_START),
                  "D' (CORE5 + BTAL_QQQ, IV)": lib.book_returns(data["long"], comp["Dp"], LONG_START),
                  "G* (TAA3x-1N + NDX-ATR)": lib.book_returns(data["long"], comp["G"], LONG_START),
                  "G' (TAA3x-1N + NDX-VXN)": lib.book_returns(data["long"], comp["Gp"], LONG_START),
                  "G3 (TAA3x + NDX-VXN)": lib.book_returns(data["long"], comp["G3"], LONG_START),
                  "CORE5 alone": data["long"]["core5"].loc[LONG_START:END]}
    yaml_books, _ = part_m.yaml_books()
    for book in yaml_books:
        if book.name.replace("yaml:", "") in ("ladder_4_growth", "loren", "0_4", "00_0", "fund_menu_defensive",
                                              "fund_menu_low_touch_defensive", "fund_menu_aggressive",
                                              "fund_menu_growth", "ladder_1_defensive"):
            if max(data["long"][p].first_valid_index() for p in book.pods) <= LONG_START:
                ref_series[book.name] = lib.book_returns(data["long"], book, LONG_START)
    ref_frame = pd.DataFrame(ref_series)
    breach_products = breach_table(prod_long)
    breach_refs = breach_table(ref_frame)
    ref_metrics = {}
    for name, r in ref_series.items():
        m = lib.full_metrics(r, data, "long")
        ref_metrics[name] = {k: m[f"long_{k}"] for k in ("cagr", "maxdd", "sharpe", "calmar", "worst_year",
                                                          "crisis_corr", "dd_trough")}
        for crisis, (lo, hi) in lib.CRISIS_DICT.items():
            ref_metrics[name][f"crisis_{crisis}"] = lib.common.window_return_float(r, lo, hi)

    # Frontier grids.
    grids = {}
    for d_key, g_key in (("Dp", "G3"), ("Dp", "Gp"), ("D", "G")):
        grid = pd.read_csv(base / "part_m" / f"map_grid_{d_key}_{g_key}.csv")
        grids[f"{d_key}+{g_key}"] = clean(grid[["d", "g", "t", "long_cagr", "long_maxdd"]].to_dict(orient="records"))

    # Charts: monthly NAV and weekly drawdown for the recommended products, G' and the benchmarks.
    curves, dds = {}, {}
    for label, name in (("DEF", RECOMMENDED["DEF"]), ("BAL", RECOMMENDED["BAL"]), ("GRO", RECOMMENDED["GRO"]),
                        ("GRO_PLUS", RECOMMENDED["GRO_PLUS"])):
        curves[label] = monthly_nav(prod_long[name])
        dds[label] = weekly_dd(prod_long[name])
    for label, column in (("SPX", "SPXTR"), ("SIXTY_FORTY", "SIXTY_FORTY"), ("TBILL", "BIL")):
        r = data["bench"][column].loc[LONG_START:END]
        curves[label] = monthly_nav(r)
        dds[label] = weekly_dd(r)

    friction = None
    friction_path = base / "friction" / "friction_by_product.csv"
    if friction_path.exists():
        friction = pd.read_csv(friction_path)
        runs = pd.read_csv(base / "friction" / "friction_runs.csv")
    eom = data["sleeve"]["eom_flow"]
    yearly = pd.DataFrame({"eom_flow": (1 + eom).groupby(eom.index.year).prod() - 1,
                           "tbill": (1 + data["sleeve"][TBILL]).groupby(data["sleeve"].index.year).prod() - 1}).loc[2008:]
    ledger = [json.loads(line) for line in (base / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    spec_hashes = sorted({e["spec_sha256_str"] for e in ledger if "spec_sha256_str" in e})
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=lib.REPO, capture_output=True, text=True).stdout.strip()

    product_cols = ["rung", "defensive", "growth", "d", "g", "t", "pod_weights", "long_cagr", "long_maxdd", "long_sharpe",
                    "long_calmar", "long_worst_year", "long_cvar5_21d", "long_crisis_corr", "exact_cagr", "exact_maxdd",
                    "exact_sharpe", "xs_A", "xs_B", "xs_C", "xs_RECENT", "crisis_gfc", "crisis_q4_2018", "crisis_covid",
                    "crisis_bear_2022", "crisis_tariffs_2025", "pods", "trade_days_per_year", "wired_share",
                    "pm_ready_share", "shadow_share", "cash_share", "any_daily", "needs_moc", "needs_short",
                    "neighbour_cagr_range", "neighbour_dd_range", "long_dd_trough"]
    payload = {
        "meta": {"head": head, "end": END.date().isoformat(), "long_start": LONG_START.date().isoformat(),
                 "exact_start": EXACT_START.date().isoformat(), "spec_hashes": spec_hashes,
                 "sleeve_count": len(data["meta"]), "d_family": d_sel["family_size"], "g_family": g_sel["family_size"]},
        "sleeves": records(sleeves, ["tier", "legacy_cagr", "exact_cagr", "legacy_sharpe", "exact_sharpe", "legacy_maxdd",
                                     "exact_maxdd", "long_cagr", "long_sharpe", "long_maxdd", "long_gfc", "xs_RECENT",
                                     "trade_days_per_year", "route", "instruments", "min_cash_weight"]),
        "d": {"selection": d_sel, "prime": dp_sel,
              "books": records(d_books.sort_values("objective", ascending=False),
                               ["objective", "long_cagr", "long_maxdd", "long_sharpe", "exact_cagr", "exact_maxdd",
                                "xs_A", "xs_B", "xs_C", "xs_RECENT", "gate_d1", "gate_d2", "gate_d3", "gates_pass",
                                "slot_long_fail_pods", "slot_recent_pass", "slot_recent_fail_pods", "pods",
                                "trade_days_per_year", "beaten_by_top_share", "in_tie_band", "avg_weights", "rule",
                                "crisis_gfc", "crisis_covid", "crisis_bear_2022", "crisis_tariffs_2025",
                                "long_crisis_corr"]),
              "gate_rates": clean(d_books[["gate_d1", "gate_d2", "gate_d3", "gates_pass"]].mean().to_dict())},
        "g": {"selection": g_sel, "prime": comps["Gp"],
              "books": records(g_books.sort_values("objective", ascending=False),
                               ["objective", "objective_design_16", "s_at_budget", "long_cagr", "long_maxdd",
                                "long_sharpe", "exact_cagr", "exact_maxdd", "xs_B", "xs_C", "xs_RECENT", "gate_g1",
                                "gate_g2", "gate_g3", "gates_pass", "compass_beats_twin_share", "slot_recent_pass",
                                "pods", "any_daily", "trade_days_per_year", "wired_share", "shadow_share",
                                "beaten_by_top_share", "in_tie_band", "taa_leg", "ndx_leg", "third_leg", "mr_option",
                                "crisis_gfc", "crisis_covid", "crisis_bear_2022", "crisis_tariffs_2025"])},
        "components": comps,
        "products": records(products, product_cols),
        "breach_products": records(breach_products, list(breach_products.columns)),
        "breach_refs": records(breach_refs, list(breach_refs.columns)),
        "ref_metrics": clean(ref_metrics),
        "sensitivity": clean(sens.to_dict(orient="records")),
        "capacity": records(cap, [c for c in cap.columns if c.endswith(("_recommended", "_first_fail", "_cost_at_25m"))]),
        "references": records(refs, ["policy", "weights", "exact_cagr", "exact_sharpe", "exact_maxdd", "long_cagr",
                                     "long_sharpe", "long_maxdd", "crisis_gfc", "xs_RECENT", "pods",
                                     "trade_days_per_year", "any_daily", "needs_moc", "long_cagr_at_10",
                                     "long_cagr_at_20"]),
        "benchmarks": records(bench, ["long_cagr", "long_sharpe", "long_maxdd", "exact_cagr", "exact_sharpe",
                                      "exact_maxdd"]),
        "grids": grids, "curves": curves, "drawdowns": dds,
        "eom_yearly": clean({int(y): r for y, r in yearly.to_dict(orient="index").items()}),
        "friction": clean(friction.to_dict(orient="records")) if friction is not None else None,
        "friction_runs": clean(runs.to_dict(orient="records")) if friction is not None else None,
        "recommended": RECOMMENDED, "mechanical": MECHANICAL, "verdicts": VERDICTS,
        "g_breach": records(pd.read_csv(OUT / "breach_frontier_g.csv", index_col=0), ["p_worse_15", "p_worse_20",
                                                                                     "boot_dd_p50"]),
        "d_breach": records(pd.read_csv(OUT / "breach_frontier_d.csv", index_col=0), ["p_worse_7", "p_worse_10",
                                                                                     "boot_dd_p50"]),
        "a3_rows": clean(pd.read_csv(base / "proxy_runs" / "a3_strategy_validation.csv").to_dict(orient="records")),
        "checks": json.loads((base / "checks" / "selection_checks.json").read_text(encoding="utf-8")),
        "product_dd_dates": records(products, ["long_dd_peak", "long_dd_trough", "exact_dd_trough", "exact_maxdd"]),
    }
    (OUT / "report_data.json").write_text(json.dumps(clean(payload), ensure_ascii=False), encoding="utf-8")
    breach_products.to_csv(OUT / "breach_products.csv", float_format="%.4f")
    breach_refs.to_csv(OUT / "breach_refs.csv", float_format="%.4f")
    print(breach_refs.round(3).to_string())
    print("written", (OUT / "report_data.json").stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
