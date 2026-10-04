"""MR slot, final decision (SPEC_FROZEN_MR_FINAL_SLOT.md): M0 capsule vs DV2-LF-ADV combinations, in the book.

Writes results/research/mr_capsule_build_20261004/mr_final_slot.json and prints the decision table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adv_rank_calm as arc  # noqa: E402  (panel, native ADV, floor, gate, sweep helpers)

cp, ev, st, ll, rp, npc, tbc = arc.cp, arc.ev, arc.st, arc.ll, arc.rp, arc.npc, arc.tbc
OUT = arc.OUT
SLOT_START, SLOT_END = "2004-01-02", "2026-09-24"
BOOK = ("2008-03-04", "2026-08-19")
BLOCKS = {"2008-11": ("2008-03-04", "2011-12-30"), "2012-21": ("2012-01-03", "2021-12-31"), "2022-26": ("2022-01-03", "2026-08-19")}
CANDIDATES = {
    "M0": {"DV2-G": 0.5, "HPI-G": 0.5},
    "M1": {"ADV-U": 0.5, "HPI-G": 0.5},
    "M2": {"DV2-G": 1 / 3, "HPI-G": 1 / 3, "ADV-U": 1 / 3},
    "M3": {"ADV-G": 0.5, "HPI-G": 0.5},
    "M4": {"ADV-U": 1.0},
}


def legs(cost_str: str) -> dict:
    p = rp.Panel("sp500")
    adv_arr = arc.native_adv63(p)
    p._cache["adv63"] = adv_arr
    floor_arr = arc.liquidity_floor_mask(p, adv_arr)
    gate_arr = cp.gate_on(p.dates)
    rate = st.cash_rate(p.dates)
    bps = 5.0 if cost_str == "stress" else 0.0
    out = {}
    e_dv2, s_dv2, x_dv2 = ll.dv2_masks(p, rp.Rule())
    e_adv, s_adv, x_adv = ll.dv2_masks(p, rp.Rule(rank="adv"))
    e_adv = e_adv & floor_arr
    for name, (e, s, x) in {"DV2-G": (e_dv2 & gate_arr[:, None], s_dv2, x_dv2), "ADV-U": (e_adv, s_adv, x_adv),
                            "ADV-G": (e_adv & gate_arr[:, None], s_adv, x_adv)}.items():
        res = ll.run(p, ll.Spec(name, e, s, x, slip_extra_bps=bps), SLOT_START, SLOT_END)
        out[name] = st.swept(res, rate)
    comp = pd.read_parquet(cp.OUT / "components.parquet")
    idx = pd.DatetimeIndex(comp.index)
    out["HPI-G"] = comp[f"HPI-G|{cost_str}|base"] + comp[f"HPI-G|{cost_str}|cw"] * st.cash_rate(idx)
    return out


def main() -> None:
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    report: dict = {"spec": "SPEC_FROZEN_MR_FINAL_SLOT.md"}
    book_by_cost: dict = {}
    for cost_str in ("engine", "stress"):
        leg_dict = legs(cost_str)
        books = {}
        for cand, weights in CANDIDATES.items():
            slot = ev.capsule({k: leg_dict[k] for k in weights}, weights, start=SLOT_START, end=SLOT_END)
            book = tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": slot}, npc.CANDIDATE_WEIGHT_DICT, *BOOK)
            books[cand] = book
            m = tbc.metric_dict(book)
            entry = report.setdefault(cand, {})
            entry[f"book_sharpe_{cost_str}"] = m["sharpe"]
            if cost_str == "engine":
                entry["book_max_dd"] = m["max_dd"]
                entry["book_cagr"] = m["cagr"]
                entry["blocks"] = {k: tbc.metric_dict(book.loc[a:b])["sharpe"] for k, (a, b) in BLOCKS.items()}
                s = tbc.metric_dict(slot.loc[SLOT_START:SLOT_END])
                entry["slot_standalone"] = {"cagr": s["cagr"], "sharpe": s["sharpe"], "max_dd": s["max_dd"]}
                entry["corr_with_ndx_pod"] = float(pd.concat([slot, ndx], axis=1).dropna().loc[BOOK[0]:BOOK[1]].corr().iloc[0, 1])
                entry["corr_with_taa"] = float(pd.concat([slot, taa], axis=1).dropna().loc[BOOK[0]:BOOK[1]].corr().iloc[0, 1])
        book_by_cost[cost_str] = books
    m0 = report["M0"]
    for cand in CANDIDATES:
        if cand == "M0":
            continue
        r = report[cand]
        r["bootstrap_p_beats_M0"] = ev.bootstrap_p(book_by_cost["engine"][cand], book_by_cost["engine"]["M0"])
        blocks_won = sum(r["blocks"][k] > m0["blocks"][k] for k in BLOCKS)
        r["rule"] = {
            "1_sharpe_engine_and_stress": bool(r["book_sharpe_engine"] > m0["book_sharpe_engine"] and r["book_sharpe_stress"] > m0["book_sharpe_stress"]),
            "2_blocks_won_ge_2": bool(blocks_won >= 2), "blocks_won": int(blocks_won),
            "3_bootstrap_ge_0.80": bool(r["bootstrap_p_beats_M0"] >= 0.80),
            "4_dd_within_1pp": bool(r["book_max_dd"] >= m0["book_max_dd"] - 0.01),
        }
        r["passes"] = all(v for k, v in r["rule"].items() if k != "blocks_won")
    passing = [c for c in CANDIDATES if c != "M0" and report[c]["passes"]]
    if passing:
        best = max(passing, key=lambda c: report[c]["book_sharpe_engine"])
        near = [c for c in passing if report[best]["book_sharpe_engine"] - report[c]["book_sharpe_engine"] <= 0.01]
        winner = min(near, key=lambda c: len(CANDIDATES[c]))
    else:
        winner = "M0"
    report["decision"] = winner
    (OUT / "mr_final_slot.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"{'':<4}{'book Sh eng':>12}{'+5bps':>8}{'08-11':>7}{'12-21':>7}{'22-26':>7}{'MaxDD':>8}{'CAGR':>7}{'slot Sh':>8}{'corrNDX':>8}{'P>M0':>6}  pass")
    for cand in CANDIDATES:
        r = report[cand]
        b = r["blocks"]
        print(f"{cand:<4}{r['book_sharpe_engine']:>12.3f}{r['book_sharpe_stress']:>8.3f}{b['2008-11']:>7.3f}{b['2012-21']:>7.3f}{b['2022-26']:>7.3f}"
              f"{r['book_max_dd'] * 100:>7.1f}%{r['book_cagr'] * 100:>6.1f}%{r['slot_standalone']['sharpe']:>8.2f}{r['corr_with_ndx_pod']:>8.2f}"
              f"{r.get('bootstrap_p_beats_M0', float('nan')):>6.2f}  {r.get('passes', '-')} {r.get('rule', '')}")
    print("DECISION:", winner)


if __name__ == "__main__":
    main()
