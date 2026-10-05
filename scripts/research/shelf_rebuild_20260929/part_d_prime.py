"""Amendment A3 (post-result, labelled): D' = the Part D rules applied to the subfamily without eom_flow.

Uses Part D's saved table and bootstrap paths (the same paths the main tie band used), so nothing is re-sampled:
gate passers without eom_flow, the top one by objective, the tie band (top beats a book on < 90% of paths) and the
Part D tie-break. Writes part_d/d_prime_selection.json. A6's selection_checks.py re-derives the same pick from
scratch and under every sensitivity.

Usage: python part_d_prime.py   (after part_d.py)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import lib
import part_d


def main() -> int:
    base = lib.STUDY / "part_d"
    lib.ledger("part_d_prime_started")
    table = pd.read_csv(base / "d_books.csv", index_col=0)
    boot = np.load(base / "d_boot_objective.npy")
    names = list(pd.read_csv(base / "d_long_returns.csv.gz", index_col=0, nrows=1).columns)
    sub = table[~table["pods_list"].str.contains("eom_flow") & table["gates_pass"]]
    top = sub["objective"].idxmax()
    j_top = names.index(top)
    share = pd.Series({b: float(np.mean(boot[:, j_top] > boot[:, names.index(b)])) for b in sub.index})
    share[top] = 0.0
    band = sub.loc[share.index[share < lib.TIE_SHARE]].copy()
    band["beaten_by_top_share"] = share
    columns = [c for c, _ in part_d.TIEBREAK]
    ascending = [a for _, a in part_d.TIEBREAK]
    band = band.sort_values(columns, ascending=ascending)
    out = {"d_prime": band.index[0], "top_non_eom": top, "band": list(band.index),
           "d_prime_avg_weights": json.loads(table.at[band.index[0], "avg_weights"])}
    (base / "d_prime_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    lib.ledger("part_d_prime_finished", d_prime=out["d_prime"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
