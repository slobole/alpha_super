"""Fund products, final pass: the product YAMLs and the engine confirmation (SPEC 5.12, SPEC 7).

    python pm_confirm.py write     write portfolios/fund_growth*.yaml for the final products (and keep the 2026-10-01
                                   books under *_monthly names); no engine run
    python pm_confirm.py compare   compare the PortfolioManager runs of those YAMLs with the research book model
                                   (house-cash frame, 2013-01-02 -> END); writes <study>/report/pm_confirm.json

Between the two, run each YAML through the PortfolioManager from this worktree:
    PYTHONPATH=. .venv/Scripts/python.exe strategies/run_portfolio_manager.py portfolios/fund_growth.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WT_REPO = HERE.parents[2]
STUDY = WT_REPO / "results" / "research" / "portfolio" / "fund_products_20261005"
PORTFOLIOS = WT_REPO / "portfolios"
POD = {
    "taa3x": ("pod_taa_btal_tqqq", "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"),
    "taa3x_1n": ("pod_taa_btal_1n_tqqq", "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"),
    "ndx_atr_cap": ("pod_ndx_atr_vxn_sector_cap", "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap:SectorCapVxnScaledAtrNormalizedNdxStrategy"),
    "ndx_natr_cap": ("pod_ndx_natr20_vxn_sector_cap", "strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap:SectorCapNatr20VxnScaledNdxStrategy"),
    "dv2_g": ("pod_mr_dv2_gated_bil", "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil"),
    "hpi_g": ("pod_mr_hpi_vote_gated_bil", "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil"),
    "tbill": ("pod_bil", "strategies.portfolio_controls.strategy_passive_bil"),
    "core5": ("pod_core5", "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5"),
}
FILE = {"GR1": "fund_growth", "GR2": "fund_growth_plus", "GR3": "fund_growth_aggressive"}
MONTHLY_FILE = {"M1": ("fund_growth_monthly", "S9 incumbent launch", "Monthly (TAA 3x 1N 60 / CORE5 40)")}
TOL = {"cagr": 0.003, "dd": 0.010, "corr": 0.995}
COMPARE_START = "2013-01-02"


def six(weights: dict) -> dict:
    """Weights to six decimals that sum to exactly 1 (largest remainder)."""
    keys = list(weights)
    total = sum(weights.values())                      # the stored weights are rounded: renormalise first
    raw = np.array([weights[k] / total for k in keys]) * 1e6
    base = np.floor(raw)
    left = int(round(1e6 - base.sum()))
    base[np.argsort(-(raw - base))[:left]] += 1
    return {k: float(b / 1e6) for k, b in zip(keys, base)}


def write() -> int:
    study = json.loads((STUDY / "report" / "study.json").read_text(encoding="utf-8"))
    # Keep the 2026-10-01 books under what they are (SPEC 7).
    # O3: the 2026-10-01 four-pod books move aside; the new two-pod monthly books take the monthly names.
    for old, new in (("fund_growth_monthly", "fund_growth_monthly_20261001"), ("fund_growth_plus_monthly", "fund_growth_plus_monthly_20261001")):
        src, dst = PORTFOLIOS / f"{old}.yaml", PORTFOLIOS / f"{new}.yaml"
        if src.exists() and "ndx_vxn" in src.read_text(encoding="utf-8").lower().replace("-", "_") or (src.exists() and "btal_lin_qqq" in src.read_text(encoding="utf-8")):
            dst.write_text(src.read_text(encoding="utf-8").replace(f"name_str: {old}", f"name_str: {new}"), encoding="utf-8")
            src.unlink()
    for code, (stem, key, label) in MONTHLY_FILE.items():
        b = study["books"][key]
        w = six(b["weights"])
        q, t = b["q"], b["tails"]
        hard = "p25"                      # Monthly 60 / 40 is read on the GROWTH PLUS rung
        lines = [
            f"# GROWTH menu, {label}: the monthly-only growth book (two pods, no daily trading).",
            f"# Study (LONG 2008-03..2026-08, gross, fair cash): {q['cagr'] * 100:.1f}%/yr, max DD {q['dd'] * 100:.1f}%, excess Sharpe {q['xs']:.2f}, "
            f"DD beyond -{hard[1:]}% {t[hard] * 100:.1f}% (block-63 bootstrap yardstick, full backtest edge).",
            "# Source: fund products final pass (2026-10-05), amendment O3 (owner decision), scripts/research/fund_products_20261005.",
            "# Research configuration only: not an allocation approval and not wired to LIVE (CORE5 has no live route yet).",
            "# Simulation, before fees. Starts 2012-10-02 (TQQQ/BTAL real history); the engine pays 0% on idle cash.",
            f"name_str: {stem}", "capital_base_float: 6000000.0", "backtest_start_date_str: '2012-10-02'", "end_date_str: null",
            "allocation_policy_str: fixed", "max_workers_int: 1", "rebalance:", "  frequency_str: annually", "  policy_str: fixed",
            "save_pod_artifacts_bool: true", "regression_benchmark_symbol_str: $SPX", "pods:",
        ]
        for alias, weight in w.items():
            pod_id, imp = POD[alias]
            lines += [f"- pod_id_str: {pod_id}", f"  strategy_import_str: {imp}", f"  weight_float: {weight:.6f}"]
        (PORTFOLIOS / f"{stem}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", stem, w)
    for old, new in ():
        src, dst = PORTFOLIOS / f"{old}.yaml", PORTFOLIOS / f"{new}.yaml"
        if src.exists() and not dst.exists() and "taa_btal_lin_qqq" in src.read_text(encoding="utf-8"):
            text = src.read_text(encoding="utf-8").replace(f"name_str: {old}", f"name_str: {new}")
            head = ("# Kept from the 2026-10-01 fund-products study as the monthly-only growth book: the book that can run before\n"
                    "# the capsule conditions hold and the scalable alternative (docs/research/FUND_PRODUCTS_20261005.md).\n")
            dst.write_text(head + text, encoding="utf-8")
    mr = PORTFOLIOS / "fund_growth_mr.yaml"
    if mr.exists():
        mr.unlink()                      # superseded: its HPI-RSI pod was demoted (SPEC 7)
    for code, stem in FILE.items():
        p = study["products"][code]
        b = study["books"][p["final"]]
        w = six(b["weights"])
        q, t = b["q"], b["tails"]
        hard = {"GROWTH": "p20", "GROWTH PLUS": "p25", "AGGRESSIVE": "p30"}[p["target_rung"]]
        mix = " / ".join(f"{k} {v * 100:.1f}%" for k, v in w.items())
        lines = [
            f"# GROWTH menu, {code} {p['label']} (three capsules: TAA, momentum E2 + sector cap, MR capsule with BIL parking).",
            f"# Weights: {mix}.",
            f"# Study (LONG 2008-03..2026-08, gross, fair cash): {q['cagr'] * 100:.1f}%/yr, max DD {q['dd'] * 100:.1f}%, excess Sharpe {q['xs']:.2f}, "
            f"breach figure P(DD < -{hard[1:]}%) {t[hard] * 100:.1f}%"
            + (f", P(DD < -25%) {t['p25'] * 100:.1f}%" if hard == "p30" else "") + " (block-63 convention, full backtest edge).",
            f"# Rung: target {p['target_rung']}; strictest passed {p['strictest_rung']}. See the conservative case in the report (section how much to believe) before relying on these numbers.",
            "# Source: fund products final pass (2026-10-05), scripts/research/fund_products_20261005/SPEC_FROZEN.md, study.py.",
            "# Research configuration only: not an allocation approval and not wired to LIVE. The momentum and MR capsule pods",
            "# have no live route yet, and the MR capsule waits for the slippage gate (<= 4 bps per side over >= 200 fills).",
            "# Simulation, before fees. Starts 2012-10-02 (TQQQ/BTAL real history); the engine pays 0% on idle cash.",
            f"name_str: {stem}",
            "capital_base_float: 6000000.0",
            "backtest_start_date_str: '2012-10-02'",
            "end_date_str: null",
            "allocation_policy_str: fixed",
            "max_workers_int: 1",
            "rebalance:",
            "  frequency_str: annually",
            "  policy_str: fixed",
            "save_pod_artifacts_bool: true",
            "regression_benchmark_symbol_str: $SPX",
            "pods:",
        ]
        for alias, weight in w.items():
            pod_id, imp = POD[alias]
            lines += [f"- pod_id_str: {pod_id}", f"  strategy_import_str: {imp}", f"  weight_float: {weight:.6f}"]
        (PORTFOLIOS / f"{stem}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", stem, w)
    return 0


def compare() -> int:
    sys.path.insert(0, str(HERE))
    import g_lib as g  # noqa: PLC0415

    lab = g.Lab()
    study = json.loads((g.OUT / "study.json").read_text(encoding="utf-8"))
    out = {"tolerance": TOL, "window": [COMPARE_START, str(g.END.date())], "books": {}}
    key_of = {**{c: study["products"][c]["final"] for c in FILE}, **{c: v[1] for c, v in MONTHLY_FILE.items()}}
    for code, stem in {**FILE, **{c: v[0] for c, v in MONTHLY_FILE.items()}}.items():
        runs = sorted((WT_REPO / "results" / "research" / "portfolio" / stem / "vanilla_backtest").glob("*/" + stem + ".pkl"))
        if not runs:
            out["books"][code] = {"error": "no PortfolioManager run found"}
            continue
        with open(runs[-1], "rb") as fh:
            portfolio = pickle.load(fh)
        tv = portfolio.results["total_value"].astype(float)
        tv.index = pd.to_datetime(tv.index).normalize()
        pm = tv.pct_change().loc[COMPARE_START:g.END]
        w = g.blend((1.0, study["books"][key_of[code]]["weights"]))   # stored weights are rounded
        house = lab.ret(w, "s1_house_cash").loc[COMPARE_START:]
        pm = pm.reindex(house.index)
        a, b = g.stats(pm, lab.rf), g.stats(house, lab.rf)
        corr = float(pm.corr(house))
        ok = bool(abs(a["cagr"] - b["cagr"]) <= TOL["cagr"] and abs(a["dd"] - b["dd"]) <= TOL["dd"] and corr >= TOL["corr"])
        out["books"][code] = {"run": runs[-1].parent.name, "engine": a, "research_house_cash": b, "corr": corr,
                              "max_abs_daily_diff": float((pm - house).abs().max()), "accepted": ok,
                              "main_frame": g.stats(lab.ret(w).loc[COMPARE_START:], lab.rf)}
        print(code, "engine", round(a["cagr"], 4), round(a["dd"], 4), "research", round(b["cagr"], 4), round(b["dd"], 4), "corr", round(corr, 5), "accepted", ok)
    (g.OUT / "pm_confirm.json").write_text(json.dumps(g.r6(out), indent=1), encoding="utf-8")
    g.ledger("pm_confirm_finished", accepted={k: v.get("accepted") for k, v in out["books"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit({"write": write, "compare": compare}[sys.argv[1]]())
