"""A5 (owner questions 2026-10-01, labelled post-result exploration): defensive third pod and the price of 22% growth.

Part A: CORE5 + BTAL_QQQ (or TAA3x) plus DV2-IND and/or downshock, equal weights, then the smallest T-bill share that
passes the DEFENSIVE rule. Part B: three routes to ~22%/yr gross with wired pods only (more TAA, margin, stock MR),
compared at the same CAGR. Descriptive; nothing here replaces the A4 products by itself.

Usage: python round2.py   Writes <study>/report/round2.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import TBILL, Book, ga, lib
import evaluate as ev

MARGIN_SPREAD = 0.015          # negative cash pays DTB3 + 1.5% (study convention)
CRISES = ("gfc", "covid", "bear_2022", "tariffs_2025")
D0 = {"core5": 0.6, "btal_qqq": 0.4}
DEF_CANDS = {
    "D0": D0,
    "D0+DV2IND": {"core5": 1 / 3, "btal_qqq": 1 / 3, "etf_dv2": 1 / 3},
    "D0+DOWN": {"core5": 1 / 3, "btal_qqq": 1 / 3, "downshock": 1 / 3},
    "D0+DV2IND+DOWN": {"core5": 0.25, "btal_qqq": 0.25, "etf_dv2": 0.25, "downshock": 0.25},
    "D0+EOM+DOWN": {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "downshock": 0.25},
    "D0+EOM+DV2IND": {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25},
    "CORE5+TAA3X+DV2IND": {"core5": 1 / 3, "taa3x": 1 / 3, "etf_dv2": 1 / 3},
    "CORE5+TAA3X+DOWN": {"core5": 1 / 3, "taa3x": 1 / 3, "downshock": 1 / 3},
}
GROWTH_BASE = {"fund_growth": {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18},
               "fund_growth_plus": {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09},
               "fund_growth_mr": {"taa3x": 0.27, "taa3x_1n": 0.27, "ndx_vxn": 0.07, "dv2": 0.09, "hpi_vote": 0.09, "hpi_ibs_rsi": 0.09,
                                  "btal_qqq": 0.03, "core5": 0.03, TBILL: 0.06},
               "fund_defensive_target": {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}}


def lever(r: pd.Series, rf: pd.Series, L: float) -> pd.Series:
    """Daily-rebalanced margin: L x book, borrowing (L-1) at T-bill + spread."""
    return L * r - (L - 1) * (rf.reindex(r.index) + MARGIN_SPREAD / 252)


def describe(r: pd.Series, rf: pd.Series, data: dict, frames: dict, w: dict | None, L: float = 1.0) -> dict:
    nav = np.r_[1, np.cumprod(1 + r.to_numpy())]
    st = ga.window_stats(r.to_frame(), data["index"])
    out = {"cagr": float(st["gross_cagr"][0]), "net": float(st["net_cagr"][0]), "dd": float((nav / np.maximum.accumulate(nav) - 1).min()),
           "vol": float(r.std() * np.sqrt(252)), "xsharpe": ev.xsharpe(r.to_numpy(), rf.reindex(r.index).to_numpy()),
           "tail": ev.seeds_tail(r.to_numpy(), limits=(-0.07, -0.10, -0.15, -0.20, -0.25, -0.30)),
           "crises": {k: lib.common.window_return_float(r, *lib.CRISIS_DICT[k]) for k in CRISES},
           "worst_year": float(((1 + r).groupby(r.index.year).prod() - 1).min()),
           "recent_cagr": float(ga.window_stats(r.loc["2023-08-21":].to_frame(), data["index"])["gross_cagr"][0])}
    if w is not None:
        for f in ("s3_plus_5bps", "s6_exact"):
            fr, s_ = frames[f]
            rr = lib.book_returns(fr, Book("x", tuple(w), "EQ", w), s_)
            if L != 1.0:
                rr = lever(rr, fr[TBILL], L)
            ss = ga.window_stats(rr.to_frame(), data["index"])
            out[f] = {"cagr": float(ss["gross_cagr"][0]), "dd": float(ss["gross_dd"][0]),
                      "xsharpe": ev.xsharpe(rr.to_numpy(), fr[TBILL].reindex(rr.index).to_numpy())}
    return out


def beats(a: pd.Series, b: pd.Series, rf: pd.Series, obj: str) -> float:
    A = pd.concat([a, b, rf], axis=1).dropna().to_numpy()
    idx = lib.evaluation.stationary_bootstrap_index_mat(len(A), 2000, 63.0, fp.SEED0)
    wins = 0
    for k in range(2000):
        s = A[idx[k]]
        wins += (np.prod(1 + s[:, 0]) > np.prod(1 + s[:, 1])) if obj == "cagr" else (ev.xsharpe(s[:, 0], s[:, 2]) > ev.xsharpe(s[:, 1], s[:, 2]))
    return wins / 2000


def main() -> int:
    data = lib.load_inputs()
    frames = ga.frames(data)
    frame, start = frames["main"]
    rf = frame[TBILL]
    out = {"note": {}, "defensive": {}, "growth": {}}
    # DV2-IND starts 2010-01: what stands in for it in 2008-09?
    pre = frame.loc[start:"2009-12-31", "etf_dv2"]
    out["note"]["etf_dv2_pre2010_equals_tbill_share"] = float((pre - rf.loc[pre.index]).abs().lt(1e-12).mean())
    out["note"]["etf_dv2_pre2010_cagr"] = float((1 + pre).prod() ** (252 / len(pre)) - 1)
    print("note", out["note"], flush=True)

    # Part A: defensive third pod.
    rule = ev.RULES["DEFENSIVE"]
    r_d0 = lib.book_returns(frame, Book("D0", tuple(D0), "EQ", D0), start)
    spx = data["bench"]["SPXTR"].reindex(frame.loc[start:fp.END].index)
    worst = spx <= spx.quantile(0.05)
    for name, w0 in DEF_CANDS.items():
        raw_r = lib.book_returns(frame, Book(name, tuple(w0), "EQ", w0), start)
        raw = describe(raw_r, rf, data, frames, w0)
        out["defensive"][name + "|raw"] = {"weights": {k: round(v, 4) for k, v in w0.items()}, **raw}
        chosen = None
        for cash in np.arange(0.0, 0.55, 0.05):
            w = {k: v * (1 - cash) for k, v in w0.items()} | ({TBILL: round(float(cash), 2)} if cash > 0 else {})
            r = raw_r if cash == 0 else lib.book_returns(frame, Book(name, tuple(w), "EQ", w), start)
            nav = np.r_[1, np.cumprod(1 + r.to_numpy())]
            if float((nav / np.maximum.accumulate(nav) - 1).min()) < rule[1]:
                continue
            row = raw if cash == 0 else describe(r, rf, data, frames, w)
            if row["tail"]["p10"] <= rule[3]:
                chosen = (float(cash), w, r, row)
                break
        if chosen is None:
            print("DEF", name, "no cash level up to 50% passes", flush=True)
            continue
        cash, w, r, row = chosen
        row = dict(row)
        row.update({"weights": {k: round(v, 4) for k, v in w.items()}, "cash": round(cash, 2), "rule_pass": True,
                    "beats_d0_xsharpe": beats(r, r_d0, rf, "xsharpe")})
        third = [k for k in w0 if k not in ("core5", "btal_qqq", "taa3x")]
        if third:
            x = frame.loc[r_d0.index, third].mean(axis=1)
            m = worst.reindex(x.index).fillna(False).to_numpy()
            row["corr_third_vs_d0"] = {"all": float(x.corr(r_d0)), "falls": float(np.corrcoef(x[m], r_d0[m])[0, 1])}
        out["defensive"][name] = row
        print("DEF", name, "cash", row["cash"], {k: round(row[k], 4) for k in ("cagr", "net", "dd", "xsharpe")}, "p10", round(row["tail"]["p10"], 4),
              "beatsD0", row["beats_d0_xsharpe"], {k: round(v, 3) for k, v in row["crises"].items()}, row.get("corr_third_vs_d0"), flush=True)

    # Part B: routes to ~22% gross with wired pods.
    R = {}
    for name, w in GROWTH_BASE.items():
        R[name] = lib.book_returns(frame, Book(name, tuple(w), "EQ", w), start)
    # Route 1: more TAA inside the verdict structure (TAA3x-1N up, the rest scaled), no margin.
    rest = {"ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
    routes = {}
    for t in np.arange(0.40, 0.951, 0.05):
        w = {"taa3x_1n": round(float(t), 2)} | {k: v / sum(rest.values()) * (1 - t) for k, v in rest.items()}
        routes[f"more_taa|{t:.2f}"] = ("more_taa", w, 1.0)
    # Route 2: margin on the launch book.  Route 3: margin on the MR target.  Route 4: margin on the defensive target.
    for L in np.arange(1.0, 1.81, 0.05):
        routes[f"margin_growth|{L:.2f}"] = ("margin_growth", GROWTH_BASE["fund_growth"], float(L))
        routes[f"margin_mr|{L:.2f}"] = ("margin_mr", GROWTH_BASE["fund_growth_mr"], float(L))
    for L in np.arange(1.0, 3.01, 0.1):
        routes[f"margin_dtarget|{L:.2f}"] = ("margin_dtarget", GROWTH_BASE["fund_defensive_target"], float(L))
    quick = {}
    for key, (route, w, L) in routes.items():
        r = lib.book_returns(frame, Book(key, tuple(w), "EQ", w), start)
        r = lever(r, rf, L) if L != 1.0 else r
        nv = np.cumprod(1 + r.to_numpy())
        quick[key] = (route, w, L, float(nv[-1] ** (252 / len(r)) - 1), r)
    # For each route: the first point reaching 22% gross, plus the launch book for reference.
    targets = (0.20, 0.22, 0.24)
    for route in ("more_taa", "margin_growth", "margin_mr", "margin_dtarget"):
        pts = sorted([(k, v) for k, v in quick.items() if v[0] == route], key=lambda kv: kv[1][3])
        for T in targets:
            hit = next(((k, v) for k, v in pts if v[3] >= T), None)
            if hit is None:
                continue
            k, (rt, w, L, c, r) = hit
            row = describe(r, rf, data, frames, w, L)
            row.update({"route": rt, "weights": {a: round(b, 4) for a, b in w.items()}, "lever": L, "target": T,
                        "beats_growth_cagr": beats(r, R["fund_growth"], rf, "cagr")})
            out["growth"][f"{route}|{T:.2f}"] = row
            print("GRO", route, T, "L", round(L, 2), {a: round(row[a], 4) for a in ("cagr", "net", "dd", "xsharpe", "vol")},
                  {f"p{p}": round(row["tail"][f"p{p}"], 3) for p in (20, 25, 30)}, "5bps", round(row["s3_plus_5bps"]["cagr"], 4),
                  "exact", round(row["s6_exact"]["cagr"], 4), "recent", round(row["recent_cagr"], 3), flush=True)
    for name in ("fund_growth", "fund_growth_plus"):
        row = describe(R[name], rf, data, frames, GROWTH_BASE[name])
        out["growth"][name] = row | {"route": "reference", "weights": GROWTH_BASE[name], "lever": 1.0}
        print("REF", name, {a: round(row[a], 4) for a in ("cagr", "dd", "xsharpe", "vol")}, {f"p{p}": round(row["tail"][f"p{p}"], 3) for p in (20, 25, 30)}, flush=True)
    (fp.STUDY / "report" / "round2.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("round2_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
