"""SPEC 4 + A0: resampled optimisation of family-block weights per line and product/option.

Usage: python stage2.py [frame]   (after stage1.py for the same frame). Writes <study>/<frame>/stage2.json and grid npz.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import ga, lib

N_PATHS, PATH_SEED = 200, 20260930
BLOCK_COL = {"LOW-TOUCH": {"TAA": "TAA|monthly", "NDX": "NDX|monthly", "DEF": "DEF|monthly", "CASH": "CASH"},
             "MAIN": {"TAA": "TAA|deployable", "NDX": "NDX|deployable", "SMR": "SMR|deployable", "EMR": "EMR|deployable",
                      "DEF": "DEF|deployable", "CASH": "CASH"},
             "TARGET": {"TAA": "TAA|target", "NDX": "NDX|target", "SMR": "SMR|target", "EMR": "EMR|target",
                        "DEF": "DEF|target", "EOM": "EOM|target", "CASH": "CASH"}}
# (option, line, objective, build limit on hist DD, hard limit, max breach share, extra)
OPTIONS = [("DEFENSIVE", "LOW-TOUCH", "sharpe", -0.07, -0.10, 0.10, None), ("DEFENSIVE", "MAIN", "sharpe", -0.07, -0.10, 0.10, None),
           ("DEFENSIVE", "TARGET", "sharpe", -0.07, -0.10, 0.10, None),
           ("D-CALMER", "LOW-TOUCH", "sharpe", -0.05, -0.07, 0.10, None), ("D-RICHER", "LOW-TOUCH", "cagr", -0.07, -0.10, 0.10, None),
           ("GROWTH", "LOW-TOUCH", "cagr", -0.17, -0.20, 0.15, None), ("GROWTH", "MAIN", "cagr", -0.17, -0.20, 0.15, None),
           ("GROWTH", "TARGET", "cagr", -0.17, -0.20, 0.15, None),
           ("GROWTH PLUS", "LOW-TOUCH", "cagr", -0.22, -0.25, 0.15, None), ("GROWTH PLUS", "MAIN", "cagr", -0.22, -0.25, 0.15, None),
           ("GROWTH PLUS", "TARGET", "cagr", -0.22, -0.25, 0.15, None),
           ("G-SHARPE", "LOW-TOUCH", "sharpe", -0.17, -0.20, 0.15, "cagr_ge_g3")]


def g3_returns(data: dict, frame: pd.DataFrame, start) -> pd.Series:
    return lib.book_returns(frame, fp.Book("G3", ("taa3x", "ndx_vxn"), "EQ", {"taa3x": 0.5, "ndx_vxn": 0.5}), start)


def main(frame_name: str = "main", block_col=None, options=None, tag: str = "", blocks=None) -> int:
    block_col = block_col or BLOCK_COL
    options = options or OPTIONS
    data = lib.load_inputs()
    frame, start = ga.frames(data)[frame_name]
    if blocks is None:
        blocks = pd.read_csv(fp.STUDY / frame_name / "block_returns.csv.gz", index_col=0, parse_dates=True)
    g3 = g3_returns(data, frame, start).reindex(blocks.index)
    block_len = {"s7_block126": 126.0, "s8_block21": 21.0}.get(frame_name, 63.0)
    idx = lib.evaluation.stationary_bootstrap_index_mat(len(blocks), N_PATHS, block_len, PATH_SEED)
    out = {"frame": frame_name, "lines": {}, "options": {}}
    for line, cols in block_col.items():
        cols = {k: v for k, v in cols.items() if v in blocks.columns}   # a family with no viable variant has no block
        names = list(cols)
        R = blocks[[cols[k] for k in names]].to_numpy()
        W = fp.grid(len(names), fp.GRID_STEP[line])
        taa_ok = W[:, names.index("TAA")] <= fp.TAA_CAP + 1e-9
        rf = blocks["CASH"].to_numpy()
        hist = fp.mix_metrics(R, W, rf)
        n = len(R)
        pc, ps, pd_ = np.empty((N_PATHS, len(W))), np.empty((N_PATHS, len(W))), np.empty((N_PATHS, len(W)))
        g3c = np.empty(N_PATHS)
        g3v = g3.to_numpy()
        for k in range(N_PATHS):
            m = fp.mix_metrics(R[idx[k]], W, rf[idx[k]])
            pc[k], ps[k], pd_[k] = m["cagr"], m["sharpe"], m["dd"]
            g3c[k] = np.prod(1 + g3v[idx[k]]) ** (252.0 / n) - 1.0
        np.savez_compressed(fp.STUDY / frame_name / f"grid_{line}{tag}.npz", W=W, names=np.array(names), hist_cagr=hist["cagr"],
                            hist_sharpe=hist["sharpe"], hist_dd=hist["dd"], path_cagr=pc.astype(np.float32),
                            path_sharpe=ps.astype(np.float32), path_dd=pd_.astype(np.float32))
        out["lines"][line] = {"blocks": names, "block_cols": cols, "mixes": int(len(W))}
        g3_hist_cagr = float(np.prod(1 + g3v) ** (252.0 / n) - 1.0)
        for opt, oline, objective, build, hard, max_breach, extra in options:
            if oline != line:
                continue
            obj_path = ps if objective == "sharpe" else pc
            feas = (pd_ >= build) & taa_ok[None, :]
            if extra == "cagr_ge_g3":
                feas &= pc >= g3c[:, None]
            chosen, n_feas = [], 0
            for k in range(N_PATHS):
                if not feas[k].any():
                    continue
                n_feas += 1
                chosen.append(W[np.argmax(np.where(feas[k], obj_path[k], -np.inf))])
            robust = np.mean(chosen, axis=0) if chosen else np.full(len(names), np.nan)
            incl = (np.array(chosen) > 1e-9).mean(axis=0) if chosen else np.zeros(len(names))
            breach = (pd_ < hard).mean(axis=0)
            hfeas = (hist["dd"] >= build) & (breach <= max_breach) & taa_ok
            if extra == "cagr_ge_g3":
                hfeas &= hist["cagr"] >= g3_hist_cagr
            hobj = hist["sharpe"] if objective == "sharpe" else hist["cagr"]
            if hfeas.any():
                hi = int(np.argmax(np.where(hfeas, hobj, -np.inf)))
                tol = 0.03 if objective == "sharpe" else 0.003
                plateau = np.flatnonzero(hfeas & (hobj >= hobj[hi] - tol))
                hist_opt = {"weights": dict(zip(names, W[hi].round(3).tolist())), "cagr": float(hist["cagr"][hi]),
                            "sharpe": float(hist["sharpe"][hi]), "dd": float(hist["dd"][hi]), "breach200": float(breach[hi]),
                            "plateau_size": int(len(plateau)),
                            "plateau_min": dict(zip(names, W[plateau].min(axis=0).round(2).tolist())),
                            "plateau_max": dict(zip(names, W[plateau].max(axis=0).round(2).tolist()))}
            else:
                hist_opt = None
            out["options"][f"{opt}|{line}"] = {"objective": objective, "build": build, "hard": hard, "max_breach": max_breach,
                                               "robust_weights": dict(zip(names, robust.round(4).tolist())),
                                               "inclusion_share": dict(zip(names, incl.round(3).tolist())),
                                               "feasible_paths": n_feas, "hist_optimum": hist_opt}
            print(frame_name, opt, line, "robust", {k: round(v, 3) for k, v in zip(names, robust)}, "feasible", n_feas,
                  "| hist", hist_opt and hist_opt["weights"], hist_opt and (round(hist_opt["cagr"], 3), round(hist_opt["sharpe"], 2), round(hist_opt["dd"], 3)), flush=True)
    (fp.STUDY / frame_name / f"stage2{tag}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("stage2_finished" if not tag else "stage2_alt_finished", frame=frame_name, tag=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
