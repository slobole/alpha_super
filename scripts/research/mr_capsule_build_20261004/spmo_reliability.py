"""How reliable is "SPMO vs BIL since 2017-11"? (owner question 2026-10-04)

Engine runs (parked = SPMO spec, bil = BIL only); the two share every stock trade, so the paired daily difference
isolates the parking. Reports, for the capsule and the book:
- the annualised return difference with a block-bootstrap 90% interval (20-day blocks, 2,000 draws) and its t-stat;
- start-date sensitivity: CAGR and Sharpe differences for monthly start dates 2017-11 .. 2019-12;
- leave-one-year-out: the CAGR difference with each calendar year removed.
Writes results/research/mr_capsule_build_20261004/spmo_reliability.json and prints a summary.
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

ev, tbc, npc = cmp.ev, cmp.sr.tbc, cmp.npc
START, CAP_END, BOOK_END = "2017-11-01", cmp.END, "2026-08-19"


def cagr(ret: pd.Series) -> float:
    return float((1 + ret).prod() ** (252.0 / len(ret)) - 1)


def sharpe(ret: pd.Series) -> float:
    return float(ret.mean() / ret.std() * np.sqrt(252.0))


def diff_ci(a: pd.Series, b: pd.Series, draws: int = 2000, block: int = 20) -> dict:
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(11)
    n_blocks = len(d) // block
    starts = rng.integers(0, len(d) - block, size=(draws, n_blocks))
    boot = np.array([d[(s[:, None] + np.arange(block)).ravel()].mean() for s in starts]) * 252.0
    return {"mean_diff_pp_per_year": float(d.mean() * 252.0 * 100), "ci90_pp": [float(np.percentile(boot, 5) * 100), float(np.percentile(boot, 95) * 100)],
            "t_stat_iid": float(d.mean() / d.std() * np.sqrt(len(d))), "p_diff_positive": float((boot > 0).mean())}


def main() -> None:
    runs = {m: {p: cmp.load_engine(p, m) for p in ("dv2", "hpi")} for m in ("parked", "bil")}
    idx = runs["parked"]["dv2"][0].index
    pct = lambda nav: nav["total_value"].pct_change().reindex(idx)  # noqa: E731
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    parts = {m: {"DV2": pct(runs[m]["dv2"][0]), "HPI": pct(runs[m]["hpi"][0])} for m in ("parked", "bil")}

    def series(mode: str, start: str):
        cap = ev.capsule(parts[mode], {"DV2": .5, "HPI": .5}, start=start, end=CAP_END)
        book = tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": cap}, npc.CANDIDATE_WEIGHT_DICT, start, BOOK_END)
        return cap, book

    report = {}
    cap_s, book_s = series("parked", START)
    cap_b, book_b = series("bil", START)
    report["return_difference"] = {"capsule": diff_ci(cap_s, cap_b), "book": diff_ci(book_s, book_b)}
    sens = []
    for start in pd.date_range("2017-11-01", "2019-12-01", freq="MS"):
        s = start.strftime("%Y-%m-%d")
        cs, bs = series("parked", s)
        cb, bb = series("bil", s)
        sens.append({"start": s, "capsule_cagr_diff_pp": (cagr(cs) - cagr(cb)) * 100, "capsule_sharpe_diff": sharpe(cs) - sharpe(cb),
                     "book_cagr_diff_pp": (cagr(bs) - cagr(bb)) * 100, "book_sharpe_diff": sharpe(bs) - sharpe(bb)})
    report["start_sensitivity"] = sens
    loyo = {}
    for year in sorted(set(cap_s.index.year)):
        keep = cap_s.index.year != year
        loyo[int(year)] = (cagr(cap_s[keep]) - cagr(cap_b[keep])) * 100
    report["capsule_cagr_diff_leave_one_year_out_pp"] = loyo
    (cmp.OUT / "spmo_reliability.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for level in ("capsule", "book"):
        r = report["return_difference"][level]
        print(f"{level}: SPMO - BIL = {r['mean_diff_pp_per_year']:+.2f} pp/yr, 90% CI [{r['ci90_pp'][0]:+.2f}, {r['ci90_pp'][1]:+.2f}], "
              f"t {r['t_stat_iid']:.2f}, P(>0) {r['p_diff_positive']:.2f}")
    sdf = pd.DataFrame(sens)
    print("start sensitivity (min / median / max):")
    for col in ("capsule_cagr_diff_pp", "capsule_sharpe_diff", "book_cagr_diff_pp", "book_sharpe_diff"):
        print(f"  {col:<22} {sdf[col].min():+.2f} / {sdf[col].median():+.2f} / {sdf[col].max():+.2f}   share > 0: {(sdf[col] > 0).mean():.0%}")
    print("capsule CAGR diff with one year removed:", {y: round(v, 2) for y, v in loyo.items()})


if __name__ == "__main__":
    main()
