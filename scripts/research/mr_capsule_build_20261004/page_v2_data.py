"""Data for the focused MR capsule page (owner request 2026-10-04): SPMO, BIL, the LIQ alternatives, and the G3 book.

Sources and conventions:
- Capsule with SPMO / with BIL: the real-engine runs (run_engine.py parked / bil): real BIL and SPMO trades, costs,
  25% withholding. SPMO is shown only from 2017-11-01 (it traded every session from 2017-10-02).
- LIQ chapter: research conventions (engine-parity replica + HPI engine run, idle cash swept at the T-bill rate),
  with the capsule recomputed under the same conventions as the baseline (mr_final_slot.legs).
- Book: G3 = TAA 0.5 + NDX 0.5 (the research G3); "G3 + slot" = TAA 0.5 + NDX 0.25 + slot 0.25. Each window is its
  own run from target weights with an annual reset (tbc.book_window_return_ser).
Writes results/research/mr_capsule_build_20261004/page_v2_data.json.
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
import mr_final_slot as mfs  # noqa: E402
import mr_final_slot_stats as mss  # noqa: E402

ev, tbc, npc, st = cmp.ev, cmp.sr.tbc, cmp.npc, mss.st
SPMO_START, FULL_START, END, BOOK_START, BOOK_END = "2017-11-01", "2004-01-05", cmp.END, "2008-03-04", "2026-08-19"
G3 = {"taa": 0.5, "L": 0.5}
CRISES = {"GFC (Mar 08–Mar 09)": ("2008-03-04", "2009-03-09"), "Aug 2011": ("2011-07-22", "2011-10-03"),
          "Q4 2018": ("2018-10-01", "2018-12-24"), "COVID 2020": ("2020-02-19", "2020-03-23"),
          "2022": ("2022-01-03", "2022-12-30"), "Feb–Apr 2025": ("2025-02-19", "2025-04-08")}
BLOCKS = {"2008–11": ("2008-03-04", "2011-12-30"), "2012–21": ("2012-01-03", "2021-12-31"), "2022–26": ("2022-01-03", BOOK_END)}


def crises(ret: pd.Series) -> dict:
    out = {}
    for name, (a, b) in CRISES.items():
        part = ret.loc[a:b]
        out[name] = float((1 + part).prod() - 1) if len(part) and part.index[0] <= pd.Timestamp(a) + pd.Timedelta(days=7) else None
    return out


def years(ret: pd.Series) -> dict:
    return {int(y): float(v) for y, v in ((1 + ret).groupby(ret.index.year).prod() - 1).items()}


def weekly(ret: pd.Series) -> dict:
    wealth = 100_000 * (1 + ret).cumprod()
    dd = wealth / wealth.cummax() - 1
    w = wealth.resample("W-FRI").last().dropna()
    d = dd.resample("W-FRI").min().reindex(w.index)
    return {"d": [x.strftime("%Y-%m-%d") for x in w.index], "w": [round(float(v), 0) for v in w], "dd": [round(float(v), 4) for v in d]}


def block_sharpe(ret: pd.Series) -> dict:
    return {k: float(tbc.metric_dict(ret.loc[a:b])["sharpe"]) for k, (a, b) in BLOCKS.items()}


def entry(ret: pd.Series, rate: pd.Series, book: bool = False) -> dict:
    out = {"stats": mss.stats(ret, rate), "crises": crises(ret), "years": years(ret)}
    if book:
        out["blocks"] = block_sharpe(ret)
    return out


def main() -> None:
    runs = {m: {p: cmp.load_engine(p, m) for p in ("dv2", "hpi")} for m in ("parked", "bil")}
    idx = runs["parked"]["dv2"][0].index
    pct = lambda nav: nav["total_value"].pct_change().reindex(idx)  # noqa: E731
    rate = st.cash_rate(idx)
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    spy = npc.load_total_return_ret_ser("SPY", "SPY").reindex(idx)
    eng = {m: {"DV2": pct(runs[m]["dv2"][0]), "HPI": pct(runs[m]["hpi"][0])} for m in ("parked", "bil")}
    cap = {
        "spmo_w": ev.capsule(eng["parked"], {"DV2": .5, "HPI": .5}, start=SPMO_START, end=END),
        "bil_w": ev.capsule(eng["bil"], {"DV2": .5, "HPI": .5}, start=SPMO_START, end=END),
        "bil_full": ev.capsule(eng["bil"], {"DV2": .5, "HPI": .5}, start=FULL_START, end=END),
    }
    data: dict = {"meta": {"spmo_start": SPMO_START, "full_start": FULL_START, "end": END, "book_start": BOOK_START, "book_end": BOOK_END}}
    # ---- 1-2. capsule with SPMO (2017-11 on) and with BIL (same window and full history)
    data["capsule"] = {
        "spmo_w": entry(cap["spmo_w"], rate), "bil_w": entry(cap["bil_w"], rate), "bil_full": entry(cap["bil_full"], rate),
        "spy_w": entry(spy.loc[SPMO_START:END].dropna(), rate), "spy_full": entry(spy.loc[FULL_START:END].dropna(), rate),
        "pods_w": {f"{p}-{m}": mss.stats(eng[m][p].loc[SPMO_START:END], rate) for m in ("parked", "bil") for p in ("DV2", "HPI")},
        "pods_full": {f"{p}-bil": mss.stats(eng["bil"][p].loc[FULL_START:END], rate) for p in ("DV2", "HPI")},
        "boot": {"spmo_beats_bil_sharpe": ev.bootstrap_p(cap["spmo_w"], cap["bil_w"])},
    }
    rel = json.loads((cmp.OUT / "spmo_reliability.json").read_text(encoding="utf-8"))
    data["capsule"]["reliability"] = {"return_difference": rel["return_difference"],
                                      "start_sensitivity": rel["start_sensitivity"],
                                      "leave_one_year_out": rel["capsule_cagr_diff_leave_one_year_out_pp"]}
    data["charts"] = {"cap_w": {"spmo": weekly(cap["spmo_w"]), "bil": weekly(cap["bil_w"])},
                      "cap_full": {"bil": weekly(cap["bil_full"]), "spy": weekly(spy.loc[FULL_START:END].dropna())}}
    # ---- 3. LIQ alternatives, research conventions (with the capsule under the same conventions)
    leg = mfs.legs("engine")
    slots = {k: ev.capsule({n: leg[n] for n in w}, w, start=mfs.SLOT_START, end=mfs.SLOT_END) for k, w in mfs.CANDIDATES.items()}
    rate_r = st.cash_rate(pd.DatetimeIndex(slots["M0"].index))
    data["liq"] = {k: entry(v.loc[FULL_START:END], rate_r) for k, v in slots.items()}
    for k in slots:
        data["liq"][k]["book"] = entry(tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": slots[k]}, npc.CANDIDATE_WEIGHT_DICT, BOOK_START, BOOK_END), rate_r, book=True)
        data["liq"][k]["corr_ndx"] = float(pd.concat([slots[k].loc[BOOK_START:BOOK_END], ndx], axis=1).dropna().corr().iloc[0, 1])
        data["liq"][k]["corr_taa"] = float(pd.concat([slots[k].loc[BOOK_START:BOOK_END], taa], axis=1).dropna().corr().iloc[0, 1])
    data["liq_boot"] = {k: ev.bootstrap_p(tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": slots[k]}, npc.CANDIDATE_WEIGHT_DICT, BOOK_START, BOOK_END),
                                          tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": slots["M0"]}, npc.CANDIDATE_WEIGHT_DICT, BOOK_START, BOOK_END))
                        for k in slots if k != "M0"}
    data["charts"]["liq"] = {k: weekly(slots[k].loc[FULL_START:END]) for k in ("M0", "M4", "M2")}
    # ---- 4. the G3 book
    book = {}
    for window, (a, b) in {"long": (BOOK_START, BOOK_END), "spmo": (SPMO_START, BOOK_END)}.items():
        variants = {"G3": ({"taa": taa, "L": ndx}, G3),
                    "G3 + T-bills": ({"taa": taa, "L": ndx, "X": rate}, npc.CANDIDATE_WEIGHT_DICT),
                    "G3 + capsule BIL": ({"taa": taa, "L": ndx, "X": ev.capsule(eng["bil"], {"DV2": .5, "HPI": .5}, start=FULL_START, end=END)}, npc.CANDIDATE_WEIGHT_DICT),
                    "G3 + LIQ alone": ({"taa": taa, "L": ndx, "X": slots["M4"]}, npc.CANDIDATE_WEIGHT_DICT),
                    "G3 + thirds": ({"taa": taa, "L": ndx, "X": slots["M2"]}, npc.CANDIDATE_WEIGHT_DICT)}
        if window == "spmo":
            variants["G3 + capsule SPMO"] = ({"taa": taa, "L": ndx, "X": cap["spmo_w"]}, npc.CANDIDATE_WEIGHT_DICT)
        series = {k: tbc.book_window_return_ser(legs_, w, a, b) for k, (legs_, w) in variants.items()}
        book[window] = {k: entry(v, rate, book=(window == "long")) for k, v in series.items()}
        g3 = series["G3"]
        for k, v in series.items():
            if k != "G3":
                book[window][k]["boot_beats_g3"] = ev.bootstrap_p(v, g3)
        if window == "long":
            data["charts"]["book_long"] = {k: weekly(series[k]) for k in ("G3", "G3 + capsule BIL", "G3 + T-bills")}
        else:
            data["charts"]["book_spmo"] = {k: weekly(series[k]) for k in ("G3", "G3 + capsule SPMO", "G3 + capsule BIL")}
    data["book"] = book
    out = cmp.OUT / "page_v2_data.json"
    out.write_text(json.dumps(data, default=float), encoding="utf-8")
    print("written", out, out.stat().st_size)
    for w in ("long", "spmo"):
        print(w, {k: (round(v["stats"]["cagr"] * 100, 1), round(v["stats"]["sharpe"], 3), round(v["stats"]["max_dd"] * 100, 1)) for k, v in book[w].items()})


if __name__ == "__main__":
    main()
