"""Amendment A2 (post-result, labelled): R2 breach shares and the beat-G3 share of the key books on 10 bootstrap seeds.

Same frame (fair cash), block 63, 2,000 paths per seed. Nothing is re-selected. Usage: python multiseed.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ga_lib as ga
from ga_lib import Book, lib

SEEDS = [20260929] + [20260930 + k for k in range(9)]
KEYS = ["TAA3x + NDX-VXN", "TAA3x + NDX-VXN | pair_dv2@36", "TAA3x-1N + NDX-VXN | def2@36",
        "TAA3x-1N + NDX-VXN | pair_dv2@36", "TAA3x-1N + NDX-VXN | def2@18", "TAA3x + NDX-ATR | dv2@18"]
LADDER4 = Book("ladder_4_growth (yaml)", ("dv2", "hpi_vote", "ndx_vxn", "taa3x"), "EQ",
               {"dv2": 0.17, "hpi_vote": 0.19, "ndx_vxn": 0.27, "taa3x": 0.37}, "annual", "ref")


def main() -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    books = {b.name: b for b in ga.family_books()}
    books[LADDER4.name] = LADDER4
    names = KEYS + [LADDER4.name]
    R = pd.DataFrame({n: lib.book_returns(frame, books[n], start) for n in names}).to_numpy()
    rows = {n: {k: [] for k in ("pg20", "pn20", "pg25", "pn25", "beats_g3")} for n in names}
    for seed in SEEDS:
        idx = lib.evaluation.stationary_bootstrap_index_mat(R.shape[0], ga.BOOT_REPS, 63.0, seed)
        bp = ga.bootstrap_paths(R, idx)
        for j, n in enumerate(names):
            rows[n]["pg20"].append(float((bp["gross_dd"][:, j] < -0.20).mean()))
            rows[n]["pn20"].append(float((bp["net_dd"][:, j] < -0.20).mean()))
            rows[n]["pg25"].append(float((bp["gross_dd"][:, j] < -0.25).mean()))
            rows[n]["pn25"].append(float((bp["net_dd"][:, j] < -0.25).mean()))
            rows[n]["beats_g3"].append(float((bp["net_cagr"][:, j] > bp["net_cagr"][:, 0]).mean()))
        print("seed", seed, flush=True)
    out = {}
    for n, v in rows.items():
        out[n] = {k: {"mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a)),
                      "frozen_seed": a[0], "seeds_over_15pct": int(sum(x > ga.MAX_BREACH for x in a))}
                  for k, a in v.items()}
        print(n, {k: (round(x["mean"], 4), round(x["min"], 4), round(x["max"], 4), x["seeds_over_15pct"])
                  for k, x in out[n].items()})
    (ga.STUDY / "report" / "multiseed.json").write_text(json.dumps({"seeds": SEEDS, "books": out}, indent=2),
                                                        encoding="utf-8")
    ga.ledger("multiseed_finished", books=list(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
