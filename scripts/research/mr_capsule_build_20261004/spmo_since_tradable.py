"""SPMO vs BIL parking since SPMO became continuously traded (owner request 2026-10-04).

SPMO's last session without a trade was 2017-09-29; the tradability guard B1 is permanently on from the 2017-10-27
close, so the comparison starts 2017-11-01. Engine runs (real BIL/SPMO trades, costs, 25% withholding):
the spec (SPMO while the gate is closed, BIL otherwise) vs BIL only; the research record (free T-bill sweep, SPMO
total return) as a cross-check. Capsule = DV2-G + HPI-G 50/50, annual reset; book = TAA 0.5 + NDX 0.25 + capsule 0.25.
Writes results/research/mr_capsule_build_20261004/spmo_since_tradable.json and prints the tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as cmp  # noqa: E402
import mr_final_slot_stats as mss  # noqa: E402

ev, tbc, npc, st = cmp.ev, cmp.sr.tbc, cmp.npc, mss.st
START, CAP_END, BOOK_END = "2017-11-01", cmp.END, "2026-08-19"
WINDOWS = {"Q4 2018": ("2018-10-01", "2018-12-24"), "COVID 2020": ("2020-02-19", "2020-03-23"),
           "2022": ("2022-01-03", "2022-12-30"), "Feb-Apr 2025": ("2025-02-19", "2025-04-08")}


def main() -> None:
    runs = {m: {p: cmp.load_engine(p, m) for p in ("dv2", "hpi")} for m in ("parked", "bil")}
    idx = runs["parked"]["dv2"][0].index
    pct = lambda nav: nav["total_value"].pct_change().reindex(idx)  # noqa: E731
    research = cmp.research_pods(idx)
    rate = st.cash_rate(idx)
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    variants = {
        "engine_SPMO": {"DV2": pct(runs["parked"]["dv2"][0]), "HPI": pct(runs["parked"]["hpi"][0])},
        "engine_BIL": {"DV2": pct(runs["bil"]["dv2"][0]), "HPI": pct(runs["bil"]["hpi"][0])},
        "research_SPMO": {"DV2": research["DV2-G"], "HPI": research["HPI-G"]},
        "research_Tbills": {"DV2": research["DV2-G|tbill"], "HPI": research["HPI-G|tbill"]},
    }
    report, caps, books = {}, {}, {}
    for name, parts in variants.items():
        cap = ev.capsule(parts, {"DV2": .5, "HPI": .5}, start=START, end=CAP_END)
        book = tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": cap}, npc.CANDIDATE_WEIGHT_DICT, START, BOOK_END)
        caps[name], books[name] = cap, book
        report[name] = {
            "capsule": mss.stats(cap, rate), "book": mss.stats(book, rate),
            "capsule_windows": {k: float((1 + cap.loc[a:b]).prod() - 1) for k, (a, b) in WINDOWS.items()},
            "book_windows": {k: float((1 + book.loc[a:b]).prod() - 1) for k, (a, b) in WINDOWS.items()},
            "capsule_years": {int(y): float(v) for y, v in ((1 + cap).groupby(cap.index.year).prod() - 1).items()},
            "book_years": {int(y): float(v) for y, v in ((1 + book).groupby(book.index.year).prod() - 1).items()},
        }
    report["bootstrap_p_spmo_beats_bil"] = {
        "engine_capsule": ev.bootstrap_p(caps["engine_SPMO"], caps["engine_BIL"]),
        "engine_book": ev.bootstrap_p(books["engine_SPMO"], books["engine_BIL"]),
        "research_capsule": ev.bootstrap_p(caps["research_SPMO"], caps["research_Tbills"]),
        "research_book": ev.bootstrap_p(books["research_SPMO"], books["research_Tbills"]),
    }
    (cmp.OUT / "spmo_since_tradable.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for level in ("capsule", "book"):
        print(f"\n{level.upper()} {START} ..")
        print(f"{'':<16}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Calmar':>7}{'$100K':>8}" + "".join(f"{k:>14}" for k in WINDOWS))
        for name in variants:
            s = report[name][level]
            w = report[name][f"{level}_windows"]
            print(f"{name:<16}{s['cagr'] * 100:>6.1f}%{s['vol'] * 100:>6.1f}%{s['sharpe']:>8.3f}{s['sortino']:>8.2f}{s['max_dd'] * 100:>7.1f}%{s['calmar']:>7.2f}"
                  f"{s['wealth_100k'] / 1e3:>7.0f}K" + "".join(f"{w[k] * 100:>13.1f}%" for k in WINDOWS))
        years = sorted(report["engine_SPMO"][f"{level}_years"])
        print(f"{'year':<16}" + "".join(f"{y:>7}" for y in years))
        for name in variants:
            print(f"{name:<16}" + "".join(f"{report[name][f'{level}_years'][y] * 100:>6.1f}%" for y in years))
        diff = [report["engine_SPMO"][f"{level}_years"][y] - report["engine_BIL"][f"{level}_years"][y] for y in years]
        print(f"{'engine SPMO-BIL':<16}" + "".join(f"{d * 100:>+6.1f}%" for d in diff))
    print("\nP(SPMO Sharpe > BIL Sharpe):", {k: round(v, 2) for k, v in report["bootstrap_p_spmo_beats_bil"].items()})


if __name__ == "__main__":
    main()
