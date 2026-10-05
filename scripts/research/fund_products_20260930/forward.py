"""SPEC 6 forward split: resampled weights from one half of LONG, evaluated on the other half against the champions.

Usage: python forward.py   (after stage1.py main). Writes <study>/report/forward.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import TBILL, Book, ga, lib
import stage2

CUT = pd.Timestamp("2017-06-30")
RUNS = [("GROWTH", "LOW-TOUCH"), ("GROWTH", "MAIN"), ("GROWTH PLUS", "LOW-TOUCH"), ("DEFENSIVE", "LOW-TOUCH"), ("DEFENSIVE", "MAIN")]
CHAMP = {"GROWTH": {"G3": {"taa3x": 0.5, "ndx_vxn": 0.5}, "GROWTH_VERDICT": {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}},
         "GROWTH PLUS": {"AGGR_VERDICT": {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09}},
         "DEFENSIVE": {"D0": {"core5": 0.6, "btal_qqq": 0.4}}}


def robust(R: np.ndarray, rf: np.ndarray, W: np.ndarray, names: list, rule: tuple, seed: int) -> np.ndarray:
    obj, build = rule
    idx = lib.evaluation.stationary_bootstrap_index_mat(len(R), stage2.N_PATHS, 63.0, seed)
    taa_ok = W[:, names.index("TAA")] <= fp.TAA_CAP + 1e-9
    chosen = []
    for k in range(stage2.N_PATHS):
        m = fp.mix_metrics(R[idx[k]], W, rf[idx[k]])
        feas = (m["dd"] >= build) & taa_ok
        if feas.any():
            chosen.append(W[np.argmax(np.where(feas, m["sharpe"] if obj == "xsharpe" else m["cagr"], -np.inf))])
    return np.mean(chosen, axis=0)


def stats(r: pd.Series, rf: pd.Series) -> dict:
    x = (r - rf.reindex(r.index)).to_numpy()
    nav = np.cumprod(1 + r.to_numpy())
    return {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "xsharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
            "dd": float((np.r_[1, nav] / np.maximum.accumulate(np.r_[1, nav]) - 1).min())}


def main() -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    blocks = pd.read_csv(fp.STUDY / "main" / "block_returns.csv.gz", index_col=0, parse_dates=True)
    rf = blocks["CASH"]
    rules = {"GROWTH": ("cagr", -0.17), "GROWTH PLUS": ("cagr", -0.22), "DEFENSIVE": ("xsharpe", -0.07)}
    out = {}
    for prod, line in RUNS:
        cols = stage2.BLOCK_COL[line]
        names = list(cols)
        W = fp.grid(len(names), fp.GRID_STEP[line])
        for lab, sel_mask in (("select_H1_eval_H2", blocks.index <= CUT), ("select_H2_eval_H1", blocks.index > CUT)):
            Rs = blocks.loc[sel_mask, [cols[n] for n in names]].to_numpy()
            w = robust(Rs, rf[sel_mask].to_numpy(), W, names, rules[prod], 20261001)
            ev = ~sel_mask
            bk = Book("fw", tuple(cols[n] for n in names), "EQ", {cols[n]: float(v) for n, v in zip(names, w)})
            bk = Book("fw", tuple(k for k, v in bk.weights.items() if v > 1e-9), "EQ", {k: v / sum(x for x in bk.weights.values() if x > 1e-9)
                                                                                          for k, v in bk.weights.items() if v > 1e-9})
            r_new = lib.book_returns(blocks, bk, blocks.index[ev][0])
            r_new = r_new.loc[blocks.index[ev]]
            row = {"weights": dict(zip(names, np.round(w, 3).tolist())), "new": stats(r_new, rf)}
            for cn, cw in CHAMP[prod].items():
                rc = lib.book_returns(frame, Book(cn, tuple(cw), "EQ", cw), start).reindex(blocks.index[ev]).dropna()
                row[cn] = stats(rc, rf)
            out[f"{prod}|{line}|{lab}"] = row
            print(prod, line, lab, row["weights"], {k: {a: round(b, 3) for a, b in v.items()} for k, v in row.items() if k != "weights"}, flush=True)
    (fp.STUDY / "report").mkdir(exist_ok=True)
    (fp.STUDY / "report" / "forward.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fp.ledger("forward_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
