"""SPEC 7, Part G: the growth core.

Family (72 books) = TAA leg {taa3x, taa3x_1n, taa2x_1n} x NDX leg {ndx_vxn, ndx_atr, ndx_natr20} x third leg
{none, compass_qqq} x MR option {none, pair, capsule, capsule_adv}. Core legs equal; an MR option takes 36% of the
book and the core legs 64%. Objective: LONG CAGR at the -20% budget (T-bill dilution only, no leverage).
Gates G1 (excess CAGR > 0 in B, C, RECENT), G2 (T-bill slot test on LONG), G3 (a Compass book beats its no-Compass
twin on >= 90% of bootstrap paths). Tie band from the paired bootstrap; tie-break by ease.

Usage: python part_g.py
"""

from __future__ import annotations

from itertools import product
import json

import numpy as np
import pandas as pd

import family
import lib
from lib import TBILL, Book

OUT = lib.STUDY / "part_g"
TAA_LEGS = {"taa3x": "TAA3x", "taa3x_1n": "TAA3x-1N", "taa2x_1n": "TAA2x-1N"}
NDX_LEGS = {"ndx_vxn": "NDX-VXN", "ndx_atr": "NDX-ATR", "ndx_natr20": "NDX-NATR20"}
THIRD_LEGS = {None: None, "compass_qqq": "COMPASS-QQQ"}
MR_OPTIONS = {"none": {}, "pair": {"dv2": 0.18, "hpi_vote": 0.18},
              "capsule": {"dv2": 0.12, "hpi_vote": 0.12, "etf_dv2": 0.12},
              "capsule_adv": {"dv2_adv": 0.12, "hpi_vote": 0.12, "etf_dv2": 0.12}}
KIND, BUDGET, DESIGN_BUDGET = "cagr_at_budget", -0.20, -0.16
TIEBREAK = [("slot_recent_pass", False), ("pods", True), ("shadow_share", True), ("pm_ready_share", True),
            ("any_daily", True), ("trade_days_per_year", True), ("objective", False)]


def family_books() -> list[Book]:
    books = []
    for taa, ndx, third, mr in product(TAA_LEGS, NDX_LEGS, THIRD_LEGS, MR_OPTIONS):
        legs = [taa, ndx] + ([third] if third else [])
        core_share = 1.0 if mr == "none" else 0.64
        weights = {leg: core_share / len(legs) for leg in legs}
        weights.update(MR_OPTIONS[mr])
        label = " + ".join([TAA_LEGS[taa], NDX_LEGS[ndx]] + ([THIRD_LEGS[third]] if third else []))
        name = label + ("" if mr == "none" else f" | {mr}")
        tags = {"taa_leg": taa, "ndx_leg": ndx, "third_leg": third or "none", "mr_option": mr}
        books.append(Book(name, tuple(weights), "EQ", weights, "annual", "G", tags))
    return books


def twin_of(name: str) -> str:
    return name.replace(" + COMPASS-QQQ", "")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    books = family_books()
    lib.ledger("part_g_started")
    assert len(books) == 72, len(books)
    table, long_r = family.evaluate(books, data, KIND, BUDGET)
    index_all = data["index"]
    tb_long = data["long"][TBILL]
    table["objective_design_16"] = [lib.cagr_at_budget(long_r[b], tb_long, index_all, DESIGN_BUDGET)[0]
                                    for b in table.index]
    names = list(long_r.columns)
    R = long_r.to_numpy()
    tb = tb_long.reindex(long_r.index).to_numpy()
    boot = lib.bootstrap_objective(R, tb, KIND, BUDGET)

    table["gate_g1"] = (table[["xs_B", "xs_C", "xs_RECENT"]] > 0).all(axis=1)
    table["gate_g2"] = table["slot_long_pass"]
    compass_share = {}
    for b in table.index:
        if table.at[b, "third_leg"] == "compass_qqq":
            compass_share[b] = float(np.mean(boot[:, names.index(b)] > boot[:, names.index(twin_of(b))]))
    table["compass_beats_twin_share"] = pd.Series(compass_share)
    table["gate_g3"] = table["compass_beats_twin_share"].isna() | (table["compass_beats_twin_share"] >= 0.90)
    table["gates_pass"] = table["gate_g1"] & table["gate_g2"] & table["gate_g3"]

    top, share = family.tie_band(table, boot, names, "gates_pass")
    band = family.select(table, share, TIEBREAK)
    g_star = band.index[0]
    pbo = lib.pbo_cscv(R, tb, KIND, BUDGET)

    sens = {}  # Tactical FI is not in the growth family, so the TFI-mode sensitivity does not apply here.
    for label, frame in (("proxy_unscaled", data["long_unscaled"]), ("plus_5bps", data["stressed_long"]),
                         ("cash_realism", data["cash_long"]), ("etf_dv2_idle_before_2010", data["long_etf_cash"])):
        alt = dict(data)
        alt["long"] = frame
        t_alt, _ = family.evaluate(books, alt, KIND, BUDGET, with_metrics=False)
        t_alt["gates_pass"] = (t_alt[["xs_B", "xs_C", "xs_RECENT"]] > 0).all(axis=1) & t_alt["slot_long_pass"] \
            & table["gate_g3"].reindex(t_alt.index)
        passers = t_alt[t_alt["gates_pass"]]
        sens[label] = {"top_gate_passer": passers["objective"].idxmax() if len(passers) else None,
                       "g_star_objective": float(t_alt.at[g_star, "objective"]),
                       "g_star_rank_among_passers": int((passers["objective"] > t_alt.at[g_star, "objective"]).sum() + 1)
                       if g_star in passers.index else None,
                       "g_star_gates_pass": bool(t_alt.at[g_star, "gates_pass"])}
        t_alt.to_csv(OUT / f"sensitivity_{label}.csv", float_format="%.6g")
    passers = table[table["gates_pass"]]
    for label, column in (("exact_window", "exact_objective"), ("design_budget_16", "objective_design_16")):
        order = passers.sort_values(column, ascending=False)
        sens[label] = {"top_gate_passer": order.index[0],
                       "g_star_rank_among_passers": int(list(order.index).index(g_star) + 1)}

    table["beaten_by_top_share"] = share.reindex(table.index)
    table["in_tie_band"] = table.index.isin(band.index)
    table.to_csv(OUT / "g_books.csv", float_format="%.6g")
    band.to_csv(OUT / "g_tie_band.csv", float_format="%.6g")
    long_r.to_csv(OUT / "g_long_returns.csv.gz", float_format="%.10g", compression="gzip")
    np.save(OUT / "g_boot_objective.npy", boot)
    summary = {"family_size": len(books), "gate_passers": int(table["gates_pass"].sum()), "top_by_objective": top,
               "tie_band_size": int(len(band)), "g_star": g_star,
               "g_star_weights": json.loads(table.at[g_star, "avg_weights"]), "pbo": pbo, "sensitivities": sens}
    (OUT / "g_selection.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["objective", "s_at_budget", "long_cagr", "long_maxdd", "long_sharpe", "exact_cagr", "exact_maxdd",
            "xs_RECENT", "slot_long_pass", "slot_recent_pass", "pods", "any_daily", "beaten_by_top_share"]
    print(band[show].round(4).to_string())
    print(json.dumps(summary, indent=2, default=str))
    lib.ledger("part_g_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
