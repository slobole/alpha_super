"""Extension E6: breach on 10 seeds (block 63) for every pick, top and product of the extension's main frame.

Reported, not a gate. Usage: python seeds_ext.py (after select_ext.py main)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ga_lib as ga
from select_ext import EXT, RUNGS

SEEDS = [20260929] + [20260930 + k for k in range(9)]


def main() -> int:
    sel = json.loads((EXT / "main" / "selection.json").read_text(encoding="utf-8"))
    names = {ga.G3_NAME}
    for mode in sel["modes"].values():
        for rung in RUNGS:
            for line in ("MAIN", "LOW-TOUCH"):
                e = mode[rung][line]
                names |= {e[k] for k in ("pick", "top", "product") if e.get(k)}
    names = [ga.G3_NAME] + sorted(names - {ga.G3_NAME})
    R = pd.read_csv(EXT / "main" / "long_returns.csv.gz", index_col=0, parse_dates=True)[names].to_numpy()
    acc = {n: {f"{c}_{r}": [] for c in ("pg", "pn") for r in RUNGS} | {"beats_g3_gross": [], "beats_g3_net": []} for n in names}
    for seed in SEEDS:
        idx = ga.lib.evaluation.stationary_bootstrap_index_mat(R.shape[0], ga.BOOT_REPS, 63.0, seed)
        bp = ga.bootstrap_paths(R, idx)
        for j, n in enumerate(names):
            for r, (_, lim) in RUNGS.items():
                acc[n][f"pg_{r}"].append(float((bp["gross_dd"][:, j] < lim).mean()))
                acc[n][f"pn_{r}"].append(float((bp["net_dd"][:, j] < lim).mean()))
            acc[n]["beats_g3_gross"].append(float((bp["gross_cagr"][:, j] > bp["gross_cagr"][:, 0]).mean()))
            acc[n]["beats_g3_net"].append(float((bp["net_cagr"][:, j] > bp["net_cagr"][:, 0]).mean()))
    out = {n: {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                   "over_15": int(sum(x > ga.MAX_BREACH for x in v))} for k, v in d.items()} for n, d in acc.items()}
    (EXT / "seeds.json").write_text(json.dumps({"seeds": SEEDS, "books": out}, indent=1), encoding="utf-8")
    for n, d in out.items():
        print(n, {k: (round(v["mean"], 3), v["over_15"]) for k, v in d.items() if k.startswith("p")})
    ga.ledger("ext_seeds_finished", books=names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
