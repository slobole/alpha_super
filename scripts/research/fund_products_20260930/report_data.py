"""Collect the Hebrew report's data (descriptive; nothing selects). Usage: python report_data.py"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import LONG_START, TBILL, Book, ga, lib
import evaluate as ev
import simplify
from round2 import lever

OUT = fp.STUDY / "report"
MENU_ROWS = {
    "DEFENSIVE": [("fund_defensive", "clean"), ("D0+DV2IND", "r2"), ("D0+DV2IND+DOWN", "r2"), ("D0+DOWN", "r2"), ("CORE5+TAA3X+DOWN", "r2"),
                  ("fund_defensive_growth", "clean"), ("fund_defensive_target", "clean"), ("D0+EOM+DOWN", "r2"), ("ref_d0", "clean")],
    "GROWTH": [("fund_growth", "clean"), ("fund_growth_plus", "clean"), ("margin_growth|0.22", "r2"), ("more_taa|0.22", "r2"),
               ("fund_growth_mr", "clean"), ("margin_mr|0.22", "r2"), ("margin_dtarget|0.22", "r2"), ("margin_mix|0.22", "r2"),
               ("growth_3x_tested", "clean"), ("G-SIMPLEST", "eval")],
}


def metrics(r: pd.Series, data: dict, rf: pd.Series) -> dict:
    fm = lib.full_metrics(r, data, "long")
    st = ga.window_stats(r.to_frame(), data["index"])
    yrs = (1 + r).groupby(r.index.year).prod() - 1
    down = r[r < 0]
    out = {k.replace("long_", ""): (v if isinstance(v, (str, type(None))) else float(v)) for k, v in fm.items()}
    out.update({"net_cagr": float(st["net_cagr"][0]), "net_dd": float(st["net_dd"][0]),
                "xsharpe": ev.xsharpe(r.to_numpy(), rf.reindex(r.index).to_numpy()),
                "sortino": float(r.mean() / down.std() * np.sqrt(252)), "years": {int(y): float(v) for y, v in yrs.items()},
                "crises": {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()},
                "nav": [[d.strftime("%Y-%m"), round(float(v), 4)] for d, v in (1 + r).cumprod().resample("ME").last().items()],
                "recent_cagr": float(ga.window_stats(r.loc["2023-08-21":].to_frame(), data["index"])["gross_cagr"][0])})
    for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, fp.END):
        out["crises"][f"cofall|{lo.date()}|{hi.date()}|{ret:.3f}"] = lib.common.window_return_float(r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
    return out


def main() -> int:
    data = lib.load_inputs()
    frames = ga.frames(data)
    frame, start = frames["main"]
    rf = frame[TBILL]
    E = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    C = json.loads((OUT / "clean.json").read_text(encoding="utf-8"))
    R2f = OUT / "round2.json"
    R2 = json.loads(R2f.read_text(encoding="utf-8")) if R2f.exists() else {"defensive": {}, "growth": {}}
    R2 = R2["defensive"] | R2["growth"]
    rows = {}
    series = {}
    for prod, items in MENU_ROWS.items():
        for name, src in items:
            w = C[name]["weights"] if src == "clean" else R2[name]["weights"] if src == "r2" else E["books"][name]["pods"]
            L = float(R2[name].get("lever", 1.0)) if src == "r2" else 1.0
            w = {k: v / sum(w.values()) for k, v in w.items()}   # round2 stores 4-decimal weights
            r = lib.book_returns(frame, Book(name, tuple(w), "EQ", w), start)
            r = lever(r, rf, L) if L != 1.0 else r
            series[name] = r
            m = metrics(r, data, rf)
            tail = C[name]["tail"] if src == "clean" else R2[name]["tail"] if src == "r2" else E["books"][name]["tail"]
            bk = Book(name, tuple(w), "EQ", w)
            ops = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in lib.ops_fields(bk, data).items() if not isinstance(v, str)}
            extra = {}
            for f in ("s3_plus_5bps", "s1_house_cash", "s6_exact", "s5_hpi_live_gap"):
                fr, s_ = frames[f]
                rr = lib.book_returns(fr, bk, s_)
                rr = lever(rr, fr[TBILL], L) if L != 1.0 else rr
                ss = ga.window_stats(rr.to_frame(), data["index"])
                extra[f] = {"cagr": float(ss["gross_cagr"][0]), "dd": float(ss["gross_dd"][0]),
                            "xsharpe": ev.xsharpe(rr.to_numpy(), fr[TBILL].reindex(rr.index).to_numpy())}
            rows[name] = {"product": prod, "weights": {k: round(v, 4) for k, v in w.items()}, "src": src, "lever": L, "m": m, "tail": tail, "ops": ops,
                          "needs_wiring": sorted(p for p in w if p != TBILL and (data["meta"][p]["tier_str"] != "wired" or p in fp.NOT_LIVE_TRADABLE)),
                          "daily_pods": sorted(p for p in w if p in fp.DAILY), "frames": extra}
            print("done", name, round(m["cagr"], 4), round(m["maxdd"], 4), flush=True)
    # Capacity for the clean books (the eval menu books already have it).
    cap_books = {n: {k: v / sum(r["weights"].values()) for k, v in r["weights"].items()} for n, r in rows.items() if r["src"] in ("clean", "r2")}
    excess = {n: float(rows[n]["frames"]["s6_exact"]["cagr"] - 0.0161) for n in cap_books}   # EXACT BIL CAGR 1.61%
    cap = ev.rd.capacity(cap_books, data, excess)
    for n in rows:
        rows[n]["capacity"] = cap.get(n) or E["capacity"].get(n)
        L = rows[n]["lever"]
        if L != 1.0 and rows[n]["capacity"]:   # an L-times book puts L dollars in every pod per NAV dollar
            for rt in rows[n]["capacity"]["routes"].values():
                if isinstance(rt, dict) and isinstance(rt.get("recommended"), (int, float)):
                    rt["recommended"] = rt["recommended"] / L
    # Alpha for the clean books (gross and net).
    closes = pd.concat([lib.common.load_total_return_close_ser(s, "2005-01-01", fp.END.strftime("%Y-%m-%d")) for s in ev.afl.ETF_LIST], axis=1)
    closes.columns = ev.afl.ETF_LIST
    closes = closes.reindex(data["index"]).loc["2005-01-01":]
    naive = ev.afl.naive_rule_return_df(closes, data["sleeve"][TBILL])
    etf_r = closes.pct_change(fill_method=None)
    tb_w = ev.afl.weekly_ser(data["sleeve"][TBILL])
    fac_w = pd.DataFrame({s: ev.afl.weekly_ser(etf_r[s]) for s in ev.afl.FACTOR_M1_LIST}).sub(tb_w, axis=0)
    trend_w = ev.afl.weekly_ser(naive["NAIVE_QQQ_TREND200"]) - tb_w
    alpha = []
    for n in ("fund_defensive", "D0+DV2IND", "D0+DV2IND+DOWN", "fund_defensive_growth", "fund_defensive_target", "fund_growth", "fund_growth_plus", "margin_growth|0.22", "fund_growth_mr", "margin_dtarget|0.22", "margin_mix|0.22", "growth_3x_tested"):
        r = series[n]
        for basis, ser in (("gross", r), ("net", ga.net_return_series(r))):
            y = (ev.afl.weekly_ser(ser) - tb_w).dropna().iloc[1:-1]
            x1 = fac_w.reindex(y.index)
            for model, x in (("M0", x1[["QQQ"]]), ("M1", x1), ("M2", x1.assign(TREND200=trend_w.reindex(y.index)))):
                alpha.append({"book": n, "basis": basis, "model": model, **ev.afl.newey_west_ols(y, x, ev.afl.NW_LAG_INT)})
    bench = {}
    for lab, col in (("S&P 500", "SPXTR"), ("60/40", "SIXTY_FORTY"), ("QQQ", "QQQ")):
        r = data["bench"][col].loc[LONG_START:fp.END]
        bench[lab] = metrics(r, data, rf)
    # Pod correlations inside the products (all days and S&P 500 worst-5% days).
    pods = ["taa3x", "taa3x_1n", "ndx_vxn", "core5", "btal_qqq", "dv2", "hpi_vote", "hpi_ibs_rsi", "eom_flow", "etf_dv2"]
    M = frame.loc[start:fp.END, pods].dropna()
    spx = data["bench"]["SPXTR"].reindex(M.index)
    worst = spx <= spx.quantile(0.05)
    corr = {"names": pods, "all": M.corr().round(2).values.tolist(), "falls": M[worst].corr().round(2).values.tolist()}
    payload = {"rows": rows, "bench": bench, "alpha": alpha, "corr": corr, "tests": E["tests"],
               "forward": json.loads((OUT / "forward.json").read_text(encoding="utf-8")),
               "forward_alt": json.loads((OUT / "forward_alt.json").read_text(encoding="utf-8")),
               "stage1": E["stage1"]["families"], "clean": C,
               "bench_variants": json.loads((OUT / "bench_variants.json").read_text(encoding="utf-8")),
               "sens_own": {f: v["own_weights"] for f, v in E["sens"].items()}}
    (OUT / "report_data.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    fp.ledger("report_data_written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
