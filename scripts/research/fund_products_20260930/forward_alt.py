"""A2 follow-up (post-result, labelled): forward split for the alternative TAA blocks (as forward.py)."""

from __future__ import annotations

import json

import pandas as pd

import fp_lib as fp
from fp_lib import Book, ga, lib
import forward
import stage2
from stage2_alt import ALT

RUNS = [("GROWTH", "MAIN", "TAA3X"), ("GROWTH", "LOW-TOUCH", "TAA3X"), ("GROWTH PLUS", "LOW-TOUCH", "TAA1N"),
        ("GROWTH", "MAIN", "TAA1N"), ("DEFENSIVE", "MAIN", "TAA3X")]


def main() -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    base = pd.read_csv(fp.STUDY / "main" / "block_returns.csv.gz", index_col=0, parse_dates=True)
    for tag, w in ALT.items():
        base[f"TAA|{tag}"] = lib.book_returns(frame, Book(tag, tuple(w), "EQ", w), start).reindex(base.index)
    rf = base["CASH"]
    rules = {"GROWTH": ("cagr", -0.17), "GROWTH PLUS": ("cagr", -0.22), "DEFENSIVE": ("xsharpe", -0.07)}
    out = {}
    for prod, line, tag in RUNS:
        cols = {k: (f"TAA|{tag}" if k == "TAA" else v) for k, v in stage2.BLOCK_COL[line].items()}
        names = list(cols)
        W = fp.grid(len(names), fp.GRID_STEP[line])
        for lab, sel in (("select_H1_eval_H2", base.index <= forward.CUT), ("select_H2_eval_H1", base.index > forward.CUT)):
            wv = forward.robust(base.loc[sel, [cols[n] for n in names]].to_numpy(), rf[sel].to_numpy(), W, names, rules[prod], 20261001)
            wd = {cols[n]: float(v) for n, v in zip(names, wv) if v > 1e-9}
            tot = sum(wd.values())
            wd = {k: v / tot for k, v in wd.items()}
            ev = base.index[~sel]
            r_new = lib.book_returns(base, Book("fw", tuple(wd), "EQ", wd), ev[0]).loc[ev]
            row = {"weights": {n: round(float(v), 3) for n, v in zip(names, wv)}, "new": forward.stats(r_new, rf)}
            for cn, cw in forward.CHAMP[prod].items():
                rc = lib.book_returns(frame, Book(cn, tuple(cw), "EQ", cw), start).reindex(ev).dropna()
                row[cn] = forward.stats(rc, rf)
            out[f"{prod}|{line}|{tag}|{lab}"] = row
            print(prod, line, tag, lab, {k: {a: round(b, 3) for a, b in v.items()} for k, v in row.items() if k != "weights"}, flush=True)
    (fp.STUDY / "report" / "forward_alt.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fp.ledger("forward_alt_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
