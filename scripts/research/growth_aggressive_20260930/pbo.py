"""SPEC 7: CSCV PBO per rung-line over its gate passers, rule = argmax of gross CAGR; fixed-book OOS ranks.

CAGR of a set of sessions depends only on the sum of log returns, so each of the 12,870 half/half splits is a sum of
16 pre-computed block sums. Usage: python pbo.py (after select_books.py main)
"""

from __future__ import annotations

from itertools import combinations
import json

import numpy as np
import pandas as pd

import ga_lib as ga

BLOCKS = 16


def oos_omega(values: np.ndarray, j: int) -> float:
    """Relative rank of column j in values: (books below + half the ties + 1) / (B + 1), as lib.pbo_cscv."""
    below = np.sum(values < values[j]) + 0.5 * (np.sum(values == values[j]) - 1)
    return float((below + 1.0) / (len(values) + 1))


def cscv(logr: np.ndarray, fixed: dict[str, int], pool: int) -> dict:
    """The argmax rule and PBO use the first `pool` columns (the passers); fixed books rank among all columns."""
    n = logr.shape[0] - logr.shape[0] % BLOCKS
    parts = np.array_split(np.arange(n), BLOCKS)
    block_sum = np.array([logr[p].sum(axis=0) for p in parts])       # BLOCKS x B
    block_len = np.array([len(p) for p in parts])
    logits, ranks, fixed_ranks = [], [], {k: [] for k in fixed}
    for combo in combinations(range(BLOCKS), BLOCKS // 2):
        mask = np.zeros(BLOCKS, dtype=bool)
        mask[list(combo)] = True
        is_val = block_sum[mask].sum(axis=0) / block_len[mask].sum()
        oos_val = block_sum[~mask].sum(axis=0) / block_len[~mask].sum()
        best = int(np.argmax(is_val[:pool]))
        omega = oos_omega(oos_val[:pool], best)
        ranks.append(omega)
        logits.append(np.log(omega / (1.0 - omega)))
        for k, j in fixed.items():
            fixed_ranks[k].append(oos_omega(oos_val, j))
    logits = np.array(logits)
    out = {"pbo_argmax_rule": float(np.mean(logits <= 0.0)), "median_oos_rank_of_is_best": float(np.median(ranks)),
           "splits": len(logits), "books": int(pool)}
    for k, v in fixed_ranks.items():
        arr = np.array(v)
        out[f"fixed::{k}"] = {"median_oos_rank": float(np.median(arr)), "share_top_half": float(np.mean(arr > 0.5))}
    return out


def main() -> int:
    table = pd.read_csv(ga.STUDY / "main" / "books.csv", index_col=0)
    sel = json.loads((ga.STUDY / "main" / "selection.json").read_text(encoding="utf-8"))
    R = pd.read_csv(ga.STUDY / "main" / "long_returns.csv.gz", index_col=0, parse_dates=True)
    out = {}
    for rung in ga.RUNGS:
        for line in ("MAIN", "LOW-TOUCH"):
            fam = table if line == "MAIN" else table[table["low_touch"].astype(bool)]
            passers = list(fam.index[fam[f"pass_{rung}"].astype(bool)])
            entry = sel["rungs"][rung][line]
            fixed_names = {k: entry[k] for k in ("pick", "top") if k in entry}
            fixed_names["G3"] = ga.G3_NAME
            cols = list(dict.fromkeys(passers + list(fixed_names.values())))
            if len(passers) < 2:
                out[f"{rung}|{line}"] = {"books": len(passers), "note": "fewer than two passers"}
                continue
            logr = np.log1p(R[cols].to_numpy(dtype=float))
            res = cscv(logr, {k: cols.index(v) for k, v in fixed_names.items()}, len(passers))
            res["passers"] = len(passers)
            res["note"] = "G3 or a fixed book outside the passers is ranked among passers + fixed books"
            out[f"{rung}|{line}"] = res
            print(rung, line, json.dumps(res))
    (ga.STUDY / "main" / "pbo.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ga.ledger("pbo_finished", pbo=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
