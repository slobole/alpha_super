"""A3 (judgement, labelled): clean versions of the chosen options, re-checked. Usage: python simplify.py"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import TBILL, Book, ga, lib
import evaluate as ev

CLEAN = {
    # A4 (post-review): the final products follow the frozen rules.
    "fund_defensive": ({"core5": 0.48, "btal_qqq": 0.32, TBILL: 0.20}, "DEFENSIVE", "D-SIMPLEST"),
    "fund_defensive_growth": ({"core5": 0.24, "btal_qqq": 0.24, "ndx_vxn": 0.12, "taa3x_1n": 0.05, "taa3x": 0.05, TBILL: 0.30}, "DEFENSIVE", "D-SIMPLEST"),
    "fund_defensive_target": ({"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}, "DEFENSIVE", "D-SIMPLEST"),
    "fund_growth": ({"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}, "GROWTH", None),
    "fund_growth_plus": ({"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09}, "GROWTH PLUS", None),
    "fund_growth_mr": ({"taa3x": 0.27, "taa3x_1n": 0.27, "ndx_vxn": 0.07, "dv2": 0.09, "hpi_vote": 0.09, "hpi_ibs_rsi": 0.09,
                        "btal_qqq": 0.03, "core5": 0.03, TBILL: 0.06}, "GROWTH", "REF-GROWTH_VERDICT"),
    "growth_3x_tested": ({"taa3x": 0.27, "taa3x_1n": 0.27, "ndx_vxn": 0.25, "core5": 0.07, "btal_qqq": 0.08, TBILL: 0.06}, "GROWTH", "REF-GROWTH_VERDICT"),
    "ref_growth_verdict": ({"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}, "GROWTH", None),
    "ref_d0": ({"core5": 0.6, "btal_qqq": 0.4}, "DEFENSIVE", None),
}
CUT = pd.Timestamp("2017-06-30")


def main() -> int:
    data = lib.load_inputs()
    frames = ga.frames(data)
    frame, start = frames["main"]
    rf = frame[TBILL]
    out, R = {}, {}
    for name, (w, rule_key, champ) in CLEAN.items():
        assert abs(sum(w.values()) - 1) < 1e-9, name
        r = lib.book_returns(frame, Book(name, tuple(w), "EQ", w), start)
        R[name] = r
        rule = ev.RULES[rule_key]
        tail = ev.seeds_tail(r.to_numpy())
        nav = np.r_[1, np.cumprod(1 + r.to_numpy())]
        dd = float((nav / np.maximum.accumulate(nav) - 1).min())
        st = ga.window_stats(r.to_frame(), data["index"])
        row = {"weights": w, "rule": rule_key, "cagr": float(st["gross_cagr"][0]), "net": float(st["net_cagr"][0]), "dd": dd,
               "xsharpe": ev.xsharpe(r.to_numpy(), rf.reindex(r.index).to_numpy()), "tail": tail,
               "rule_pass": bool(dd >= rule[1] and tail[f"p{int(round(-rule[2] * 100))}"] <= rule[3])}
        for f in ("s3_plus_5bps", "s6_exact", "s1_house_cash"):
            fr, s_ = frames[f]
            rr = lib.book_returns(fr, Book(name, tuple(w), "EQ", w), s_)
            ss = ga.window_stats(rr.to_frame(), data["index"])
            row[f] = {"cagr": float(ss["gross_cagr"][0]), "dd": float(ss["gross_dd"][0]),
                      "xsharpe": ev.xsharpe(rr.to_numpy(), fr[TBILL].reindex(rr.index).to_numpy())}
        for lab, m in (("H1", r.index <= CUT), ("H2", r.index > CUT)):
            x = r[m]
            nv = np.cumprod(1 + x.to_numpy())
            row[lab] = {"cagr": float(nv[-1] ** (252 / len(x)) - 1), "xsharpe": ev.xsharpe(x.to_numpy(), rf.reindex(x.index).to_numpy()),
                        "dd": float((np.r_[1, nv] / np.maximum.accumulate(np.r_[1, nv]) - 1).min())}
        out[name] = row
    for name, (w, rule_key, champ) in CLEAN.items():
        if not champ:
            continue
        cref = {"D-SIMPLEST": "ref_d0", "REF-GROWTH_VERDICT": "ref_growth_verdict"}[champ]
        obj = ev.RULES[rule_key][0]
        A = pd.concat([R[name], R[cref], rf], axis=1).dropna().to_numpy()
        idx = lib.evaluation.stationary_bootstrap_index_mat(len(A), 2000, 63.0, fp.SEED0)
        wins = 0
        for k in range(2000):
            s = A[idx[k]]
            wins += (np.prod(1 + s[:, 0]) > np.prod(1 + s[:, 1])) if obj == "cagr" else (ev.xsharpe(s[:, 0], s[:, 2]) > ev.xsharpe(s[:, 1], s[:, 2]))
        out[name]["beats_champ"] = {"champ": cref, "share": wins / 2000}
    (fp.STUDY / "report" / "clean.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("clean_finished")
    for k, v in out.items():
        print(k, round(v["cagr"], 4), round(v["net"], 4), round(v["dd"], 4), round(v["xsharpe"], 2), "pass", v["rule_pass"],
              {a: round(b, 3) for a, b in v["tail"].items() if not a.endswith("_max")}, "5bps", round(v["s3_plus_5bps"]["cagr"], 4),
              "exact", round(v["s6_exact"]["cagr"], 4), "H1", round(v["H1"]["cagr"], 3), "H2", round(v["H2"]["cagr"], 3), v.get("beats_champ"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
