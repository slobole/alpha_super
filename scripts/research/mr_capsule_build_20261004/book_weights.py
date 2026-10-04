"""How to push the book's CAGR (owner question 2026-10-04): weights, 1/N, and the TAA 1N variant.

Legs: TAA = the live TAA 3x pod (taa_btal_tqqq) or its 1N variant (taa_btal_1n_tqqq), both from the portfolio-refresh
sleeve file (synthetic proxy before 2012-10); momentum = the NDX momentum capsule E2 (PortfolioManager run); MR = the
capsule with BIL parking (real engine). Books reset to their weights once a year (tbc.book_window_return_ser).
Windows: 2008-03-04 .. 2026-08-19 and 2017-11-01 .. 2026-08-19. Exploratory; descriptive only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as cmp  # noqa: E402
import mr_final_slot_stats as mss  # noqa: E402
import page_v2_data as pvd  # noqa: E402

ev, tbc, st = cmp.ev, cmp.sr.tbc, mss.st
WINDOWS = {"2008-26": ("2008-03-04", "2026-08-19"), "since Nov 2017": ("2017-11-01", "2026-08-19")}
WEIGHTS = {
    "Base: TAA 0.5 / Mom 0.25 / MR 0.25": (0.5, 0.25, 0.25),
    "1/N over sleeves: 1/3 each": (1 / 3, 1 / 3, 1 / 3),
    "1/N over pods: TAA 0.2 / Mom 0.4 / MR 0.4": (0.2, 0.4, 0.4),
    "More MR: TAA 0.5 / Mom 0.15 / MR 0.35": (0.5, 0.15, 0.35),
    "TAA 0.5 / MR 0.5 (no momentum)": (0.5, 0.0, 0.5),
    "More TAA: TAA 0.6 / Mom 0.2 / MR 0.2": (0.6, 0.2, 0.2),
    "G3 (no MR): TAA 0.5 / Mom 0.5": (0.5, 0.5, 0.0),
}


def main() -> None:
    sleeves = pd.read_csv(cmp.sr.tbc.SLEEVE_SERIES_PATH, index_col=0, parse_dates=True)
    taa = {"TAA 3x": sleeves["taa_btal_tqqq"].astype(float), "TAA 3x 1N": sleeves["taa_btal_1n_tqqq"].astype(float)}
    mom = pvd.load_e2_ret_ser()
    runs = {p: cmp.load_engine(p, "bil") for p in ("dv2", "hpi")}
    idx = runs["dv2"][0].index
    mr = ev.capsule({p.upper(): runs[p][0]["total_value"].pct_change().reindex(idx) for p in ("dv2", "hpi")},
                    {"DV2": .5, "HPI": .5}, start=pvd.FULL_START, end=pvd.END)
    rate = st.cash_rate(idx)
    out = {"legs": {}}
    for name, ser in {**taa, "Momentum E2": mom, "MR capsule BIL": mr}.items():
        out["legs"][name] = {w: mss.stats(ser.loc[a:b].dropna(), rate) for w, (a, b) in WINDOWS.items()}
    for taa_name, taa_ser in taa.items():
        for wname, (wt, wm, wr) in WEIGHTS.items():
            legs, weights = {"taa": taa_ser}, {"taa": wt}
            if wm > 0:
                legs["mom"], weights["mom"] = mom, wm
            if wr > 0:
                legs["mr"], weights["mr"] = mr, wr
            for w, (a, b) in WINDOWS.items():
                book = tbc.book_window_return_ser(legs, weights, a, b)
                out.setdefault(taa_name, {}).setdefault(wname, {})[w] = mss.stats(book, rate)
    (cmp.OUT / "book_weights.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    for name, d in out["legs"].items():
        print(f"LEG {name:<16}" + "  ".join(f"{w}: {s['cagr'] * 100:5.1f}% / {s['sharpe']:.2f} / {s['max_dd'] * 100:5.1f}%" for w, s in d.items()))
    for taa_name in taa:
        print(f"\n{taa_name}")
        for wname in WEIGHTS:
            d = out[taa_name][wname]
            print(f"  {wname:<44}" + "  ".join(f"{w}: {s['cagr'] * 100:5.1f}% / {s['sharpe']:.3f} / {s['max_dd'] * 100:5.1f}% / worst yr {s['worst_year'] * 100:5.1f}%" for w, s in d.items()))


if __name__ == "__main__":
    main()
