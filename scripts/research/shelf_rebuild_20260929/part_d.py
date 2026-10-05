"""SPEC 6, Part D: the defensive core.

Family: CORE5 alone and CORE5 + every subset (size 1-3) of {btal_qqq, tactical_fi, trinity, eom_flow, downshock,
disp}, each under EQ and IV (83 books). Objective: LONG excess Calmar. Gates D1 (LONG max DD >= -10%), D2 (excess
CAGR > 0 in A, B, C, RECENT), D3 (T-bill slot test on LONG). Tie band from the paired bootstrap; tie-break by ease.
Sensitivity family (not selectable): the pool without CORE5, EQ.

Usage: python part_d.py
"""

from __future__ import annotations

from itertools import combinations
import json

import numpy as np
import pandas as pd

import family
import lib
from lib import TBILL, Book

OUT = lib.STUDY / "part_d"
LABEL = {"core5": "CORE5", "btal_qqq": "BTAL_QQQ", "tactical_fi": "TFI", "trinity": "TRINITY", "eom_flow": "EOM",
         "downshock": "DOWNSHOCK", "disp": "DISP"}
POOL = ["btal_qqq", "tactical_fi", "trinity", "eom_flow", "downshock", "disp"]
KIND, BUDGET = "excess_calmar", -0.10
TIEBREAK = [("slot_recent_pass", False), ("pods", True), ("shadow_share", True), ("pm_ready_share", True),
            ("trade_days_per_year", True), ("objective", False)]


def name_of(pods: tuple[str, ...], rule: str) -> str:
    return " + ".join(LABEL[p] for p in pods) + f" [{rule}]"


def family_books() -> list[Book]:
    books = [Book(name_of(("core5",), "EQ"), ("core5",), "EQ", family="D")]
    for size in (1, 2, 3):
        for subset in combinations(POOL, size):
            pods = ("core5",) + subset
            for rule in ("EQ", "IV"):
                books.append(Book(name_of(pods, rule), pods, rule, family="D"))
    return books


def core5_free_books() -> list[Book]:
    return [Book(name_of(subset, "EQ"), subset, "EQ", family="D-no-CORE5")
            for size in (1, 2, 3) for subset in combinations(POOL, size)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    books = family_books()
    lib.ledger("part_d_started")
    assert len(books) == 83, len(books)
    table, long_r = family.evaluate(books, data, KIND, BUDGET)
    table["gate_d1"] = table["long_maxdd"] >= BUDGET
    table["gate_d2"] = (table[["xs_A", "xs_B", "xs_C", "xs_RECENT"]] > 0).all(axis=1)
    table["gate_d2_no_recent"] = (table[["xs_A", "xs_B", "xs_C"]] > 0).all(axis=1)
    table["gate_d3"] = table["slot_long_pass"]
    table["gates_pass"] = table["gate_d1"] & table["gate_d2"] & table["gate_d3"]
    recent_relaxed = False
    if not table["gates_pass"].any():
        # SPEC 6: only D2's RECENT part may become a flag, and this is reported as a finding.
        recent_relaxed = True
        table["gates_pass"] = table["gate_d1"] & table["gate_d2_no_recent"] & table["gate_d3"]

    names = list(long_r.columns)
    R = long_r.to_numpy()
    tb = data["long"][TBILL].reindex(long_r.index).to_numpy()
    boot = lib.bootstrap_objective(R, tb, KIND)
    top, share = family.tie_band(table, boot, names, "gates_pass")
    band = family.select(table, share, TIEBREAK)
    d_star = band.index[0]
    pbo = lib.pbo_cscv(R, tb, KIND)

    # Sensitivities (SPEC 9): objective and gates of every book under each alternative input; never used to select.
    sens = {}
    swap_tfi = data["long"].copy()
    if data["tfi_frozen"] is not None:
        swap_tfi["tactical_fi"] = data["tfi_frozen"]
    for label, frame in (("proxy_unscaled", data["long_unscaled"]), ("plus_5bps", data["stressed_long"]),
                         ("cash_realism", data["cash_long"]), ("tfi_frozen_mode", swap_tfi)):
        alt = dict(data)
        alt["long"] = frame
        t_alt, r_alt = family.evaluate(books, alt, KIND, BUDGET, with_metrics=False)
        dd_alt = r_alt.apply(lib.maxdd)
        t_alt["gates_pass"] = (dd_alt >= BUDGET) & (t_alt[["xs_A", "xs_B", "xs_C", "xs_RECENT"]] > 0).all(axis=1) \
            & t_alt["slot_long_pass"]
        passers = t_alt[t_alt["gates_pass"]]
        sens[label] = {"top_gate_passer": passers["objective"].idxmax() if len(passers) else None,
                       "d_star_objective": float(t_alt.at[d_star, "objective"]),
                       "d_star_rank_among_passers": int((passers["objective"] > t_alt.at[d_star, "objective"]).sum() + 1)
                       if d_star in passers.index else None,
                       "d_star_gates_pass": bool(t_alt.at[d_star, "gates_pass"])}
        t_alt.to_csv(OUT / f"sensitivity_{label}.csv", float_format="%.6g")
    exact_rank = table.loc[table["gates_pass"]].sort_values("exact_objective", ascending=False)
    sens["exact_window"] = {"top_gate_passer": exact_rank.index[0],
                            "d_star_rank_among_passers": int(list(exact_rank.index).index(d_star) + 1)}

    free_table, _ = family.evaluate(core5_free_books(), data, KIND, BUDGET)
    free_table["gates_pass"] = (free_table["long_maxdd"] >= BUDGET) & \
        (free_table[["xs_A", "xs_B", "xs_C", "xs_RECENT"]] > 0).all(axis=1) & free_table["slot_long_pass"]

    table["beaten_by_top_share"] = share.reindex(table.index)
    table["in_tie_band"] = table.index.isin(band.index)
    table.to_csv(OUT / "d_books.csv", float_format="%.6g")
    band.to_csv(OUT / "d_tie_band.csv", float_format="%.6g")
    free_table.to_csv(OUT / "d_core5_free.csv", float_format="%.6g")
    long_r.to_csv(OUT / "d_long_returns.csv.gz", float_format="%.10g", compression="gzip")
    np.save(OUT / "d_boot_objective.npy", boot)
    summary = {"family_size": len(books), "gate_passers": int(table["gates_pass"].sum()),
               "recent_gate_relaxed": recent_relaxed, "top_by_objective": top, "tie_band_size": int(len(band)),
               "d_star": d_star, "d_star_pods": list(table.at[d_star, "pods_list"].split("+")),
               "d_star_rule": table.at[d_star, "rule"], "d_star_avg_weights": json.loads(table.at[d_star, "avg_weights"]),
               "pbo": pbo, "sensitivities": sens,
               "core5_free_top": free_table.loc[free_table["gates_pass"], "objective"].idxmax()
               if free_table["gates_pass"].any() else None}
    (OUT / "d_selection.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["objective", "long_cagr", "long_maxdd", "long_sharpe", "exact_cagr", "exact_maxdd", "xs_RECENT",
            "slot_long_pass", "slot_recent_pass", "pods", "trade_days_per_year", "beaten_by_top_share"]
    print(band[show].round(4).to_string())
    print(json.dumps(summary, indent=2, default=str))
    lib.ledger("part_d_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
