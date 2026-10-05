"""Post-review checks (amendment A6; post-result, labelled): the full selection rule under every SPEC 9 sensitivity.

The independent quant review found that part_d.py / part_g.py reported only the top book and D*'s rank under each
sensitivity, not the rule's output. This script re-runs the complete rule (gates, bootstrap tie band on that
frame's own returns, ease tie-break) for D*, D' (no eom_flow) and G*, defines G' with the Part G tie-break applied
at the -16% design point (the review's fix to A4's raw argmax), measures each fixed book's out-of-sample rank in
the CSCV splits (the PBO in part_d/part_g describes an argmax-over-all-books rule), and records the facts the review
asked for: EOM's RECENT excess under cash realism and the NDX legs' drawdowns before 2008.

Usage: python selection_checks.py
"""

from __future__ import annotations

from itertools import combinations
import json

import numpy as np
import pandas as pd

import family
import lib
import part_d
import part_g
from lib import TBILL

OUT = lib.STUDY / "checks"
D_OPS = ["pods", "shadow_share", "pm_ready_share", "trade_days_per_year"]
G_OPS = ["pods", "shadow_share", "pm_ready_share", "any_daily", "trade_days_per_year"]


def d_gates(table: pd.DataFrame, returns: pd.DataFrame, recent: bool = True) -> pd.Series:
    blocks = ["xs_A", "xs_B", "xs_C"] + (["xs_RECENT"] if recent else [])
    dd = returns.apply(lib.maxdd)
    return (dd.reindex(table.index) >= part_d.BUDGET) & (table[blocks] > 0).all(axis=1) & table["slot_long_pass"]


def run_d(books, frame_data, main_ops: pd.DataFrame) -> dict:
    table, returns = family.evaluate(books, frame_data, part_d.KIND, part_d.BUDGET, with_metrics=False)
    table = table.join(main_ops[D_OPS])
    table["gates_pass"] = d_gates(table, returns)
    relaxed = False
    if not table["gates_pass"].any():
        relaxed = True
        table["gates_pass"] = d_gates(table, returns, recent=False)
    tb = frame_data["long"][TBILL].reindex(returns.index).to_numpy()
    boot = lib.bootstrap_objective(returns.to_numpy(), tb, part_d.KIND)
    top, share = family.tie_band(table, boot, list(returns.columns), "gates_pass")
    band = family.select(table, share, part_d.TIEBREAK)
    return {"pick": band.index[0], "top": top, "band_size": int(len(band)), "gate_passers": int(table["gates_pass"].sum()),
            "recent_relaxed": relaxed, "band": list(band.index)}


def run_g(books, frame_data, main_ops: pd.DataFrame, budget: float) -> dict:
    table, returns = family.evaluate(books, frame_data, part_g.KIND, budget, with_metrics=False)
    table = table.join(main_ops[G_OPS])
    names = list(returns.columns)
    tb = frame_data["long"][TBILL].reindex(returns.index).to_numpy()
    boot = lib.bootstrap_objective(returns.to_numpy(), tb, part_g.KIND, budget)
    compass = {}
    for b in table.index:
        if table.at[b, "third_leg"] == "compass_qqq":
            compass[b] = float(np.mean(boot[:, names.index(b)] > boot[:, names.index(part_g.twin_of(b))]))
    g3 = pd.Series(compass).reindex(table.index)
    table["gates_pass"] = (table[["xs_B", "xs_C", "xs_RECENT"]] > 0).all(axis=1) & table["slot_long_pass"] \
        & (g3.isna() | (g3 >= 0.90))
    top, share = family.tie_band(table, boot, names, "gates_pass")
    band = family.select(table, share, part_g.TIEBREAK)
    return {"pick": band.index[0], "top": top, "band_size": int(len(band)), "gate_passers": int(table["gates_pass"].sum())}


def fixed_book_cscv(returns: pd.DataFrame, tb: np.ndarray, kind: str, budget: float, names: list[str],
                    blocks: int = 16) -> dict:
    """Out-of-sample rank of fixed books across the CSCV splits (omega = rank / (B + 1), as lib.pbo_cscv)."""
    R = returns.to_numpy()
    n = R.shape[0] - R.shape[0] % blocks
    R, tb = R[:n], tb[:n]
    block_idx = np.array_split(np.arange(n), blocks)
    cols = [list(returns.columns).index(b) for b in names]
    omega = {b: [] for b in names}
    for combo in combinations(range(blocks), blocks // 2):
        oos = np.concatenate([block_idx[i] for i in range(blocks) if i not in combo])
        c, d = lib.path_stats(R[oos])
        tbc = float(np.prod(1.0 + tb[oos]) ** (252.0 / len(oos)) - 1.0)
        v = lib.objective_from_stats(kind, c, d, tbc, budget)
        for b, j in zip(names, cols):
            below = np.sum(v < v[j]) + 0.5 * (np.sum(v == v[j]) - 1)
            omega[b].append((below + 1.0) / (R.shape[1] + 1))
    return {b: {"median_oos_omega": float(np.median(o)), "share_below_median": float(np.mean(np.array(o) <= 0.5))}
            for b, o in omega.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lib.ledger("selection_checks_started")
    data = lib.load_inputs()
    d_main = pd.read_csv(lib.STUDY / "part_d" / "d_books.csv", index_col=0)
    g_main = pd.read_csv(lib.STUDY / "part_g" / "g_books.csv", index_col=0)
    d_books = part_d.family_books()
    d_prime_books = [b for b in d_books if "eom_flow" not in b.pods]
    g_books = part_g.family_books()

    swap_tfi = data["long"].copy()
    swap_tfi["tactical_fi"] = data["tfi_frozen"]
    frames = {"main": data["long"], "proxy_unscaled": data["long_unscaled"], "plus_5bps": data["stressed_long"],
              "cash_realism": data["cash_long"], "tfi_frozen_mode": swap_tfi, "etf_dv2_idle_before_2010": data["long_etf_cash"]}
    rows = []
    for label, frame in frames.items():
        alt = dict(data)
        alt["long"] = frame
        row = {"sensitivity": label}
        if label != "etf_dv2_idle_before_2010":
            d_res = run_d(d_books, alt, d_main)
            dp_res = run_d(d_prime_books, alt, d_main)
            row.update({"d_star": d_res["pick"], "d_top": d_res["top"], "d_band": d_res["band_size"],
                        "d_passers": d_res["gate_passers"], "d_recent_relaxed": d_res["recent_relaxed"],
                        "d_prime": dp_res["pick"], "d_prime_band": dp_res["band_size"]})
        if label != "tfi_frozen_mode":
            g_res = run_g(g_books, alt, g_main, part_g.BUDGET)
            row.update({"g_star": g_res["pick"], "g_top": g_res["top"], "g_band": g_res["band_size"],
                        "g_passers": g_res["gate_passers"]})
        rows.append(row)
        print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "selection_under_sensitivities.csv", index=False)

    # G' with the Part G tie-break at the -16% design point (the review's fix to A4).
    g_prime_rule = run_g(g_books, data, g_main, part_g.DESIGN_BUDGET)

    # Fixed-book out-of-sample ranks in the CSCV splits.
    d_r = pd.read_csv(lib.STUDY / "part_d" / "d_long_returns.csv.gz", index_col=0, parse_dates=True)
    g_r = pd.read_csv(lib.STUDY / "part_g" / "g_long_returns.csv.gz", index_col=0, parse_dates=True)
    tb_d = data["long"][TBILL].reindex(d_r.index).to_numpy()
    tb_g = data["long"][TBILL].reindex(g_r.index).to_numpy()
    d_sel = json.loads((lib.STUDY / "part_d" / "d_selection.json").read_text(encoding="utf-8"))
    dp_sel = json.loads((lib.STUDY / "part_d" / "d_prime_selection.json").read_text(encoding="utf-8"))
    g_sel = json.loads((lib.STUDY / "part_g" / "g_selection.json").read_text(encoding="utf-8"))
    cscv_d = fixed_book_cscv(d_r, tb_d, part_d.KIND, part_d.BUDGET, [d_sel["d_star"], dp_sel["d_prime"],
                                                                     "CORE5 + BTAL_QQQ [EQ]"])
    cscv_g = fixed_book_cscv(g_r, tb_g, part_g.KIND, part_g.BUDGET, [g_sel["g_star"], "TAA3x-1N + NDX-VXN",
                                                                     "TAA3x + NDX-VXN", "TAA3x + NDX-VXN | pair"])

    # Facts the review asked for.
    idx = data["index"]
    recent_lo, recent_hi = lib.BLOCK_DICT["RECENT"]
    eom_main = lib.excess_cagr(lib.window(data["sleeve"]["eom_flow"], recent_lo, recent_hi), data["sleeve"][TBILL], idx)
    eom_cash = lib.excess_cagr(lib.window(data["cash_exact"]["eom_flow"], recent_lo, recent_hi), data["sleeve"][TBILL], idx)
    cash_share = {a: float((lib.read_path(lib.SOURCE, a)["cash_float"] / lib.read_path(lib.SOURCE, a)["total_value_float"])
                           .loc[recent_lo:recent_hi].clip(lower=0).mean())
                  for a in ("eom_flow", "etf_dv2", "disp", "downshock", "core5", "btal_qqq", "tactical_fi")}
    pre_2008 = {}
    for a in ("ndx_vxn", "ndx_atr", "ndx_natr20"):
        r = data["sleeve"][a].loc[:"2008-03-03"].dropna()
        nav = (1 + r).cumprod()
        dd = nav / nav.cummax() - 1
        pre_2008[a] = {"maxdd_2000_2008": float(dd.min()), "trough": dd.idxmin().date().isoformat(),
                       "peak": nav.loc[:dd.idxmin()].idxmax().date().isoformat()}
    out = {"selection_under_sensitivities": rows, "g_prime_rule_at_16": g_prime_rule,
           "cscv_fixed_books_d": cscv_d, "cscv_fixed_books_g": cscv_g,
           "eom_recent_excess": {"house_zero_cash": eom_main, "cash_realism": eom_cash},
           "recent_mean_positive_cash_share": cash_share, "ndx_pre_2008": pre_2008}
    (OUT / "selection_checks.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "selection_under_sensitivities"}, indent=2, default=str))
    lib.ledger("selection_checks_finished", g_prime_rule_at_16=g_prime_rule["pick"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
