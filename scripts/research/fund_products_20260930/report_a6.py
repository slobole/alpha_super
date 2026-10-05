"""Collect the A6 report's data (descriptive; a6.py made every choice). Usage: python report_a6.py"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import LONG_START, TBILL, Book, ga, lib
import evaluate as ev

OUT = fp.STUDY / "report"
# Tier facts newer than the study metadata: main demoted HPI IBS RSI to PM_READY on 2026-09-30 (693ec8e).
TIER_FIX = {"hpi_ibs_rsi": "pm-ready"}
FIN = None    # daily margin cost per borrowed dollar, set in main() (A6-c: DTB3 prior observation + spread, ACT/360)


def metrics(r: pd.Series, data: dict, rf: pd.Series) -> dict:
    """Same row metrics as this study's report_data.py (imported by path there; the module name clashes with lib's)."""
    fm = lib.full_metrics(r, data, "long")
    st = ga.window_stats(r.to_frame(), data["index"])
    yrs = (1 + r).groupby(r.index.year).prod() - 1
    down = r[r < 0]
    out = {k.replace("long_", ""): (v if isinstance(v, (str, type(None))) else float(v)) for k, v in fm.items()}
    out.update({"net_cagr": float(st["net_cagr"][0]), "net_dd": float(st["net_dd"][0]),
                "xsharpe": ev.xsharpe(r.to_numpy(), rf.reindex(r.index).to_numpy()),
                "sortino": float(r.mean() / down.std() * np.sqrt(252)), "years": {int(y): float(v) for y, v in yrs.items()},
                "crises": {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()},
                "crises_dd": {k: float((lambda w: (w / np.maximum.accumulate(w) - 1).min())(np.r_[1.0, np.cumprod(1 + r[(r.index > pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))].to_numpy())]))
                              for k, (lo, hi) in lib.CRISIS_DICT.items()},
                "nav": [[d.strftime("%Y-%m"), round(float(v), 4)] for d, v in (1 + r).cumprod().resample("ME").last().items()],
                "recent_cagr": float(ga.window_stats(r.loc["2023-08-21":].to_frame(), data["index"])["gross_cagr"][0])})
    for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, fp.END):
        out["crises"][f"cofall|{lo.date()}|{hi.date()}|{ret:.3f}"] = lib.common.window_return_float(r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
    return out
SPREAD = 0.015
DEF_SLOTS = ("launch", "launch_gated", "ds_upgrade_10", "ds_upgrade_05", "next", "calm", "rich", "target", "calm_next")
GRO_SLOTS = ("launch", "plus", "g22_launch", "g22_next", "g22_target", "g22_maxlev", "mr", "mr22")
ALPHA_ROWS = ("d_launch", "d_launch_gated", "d_ds_upgrade_10", "d_next", "d_calm", "d_rich", "d_target", "d_ref_d0", "g_launch", "g_plus", "g_g22_launch",
              "g_g22_next", "g_g22_target", "g_g22_maxlev", "g_mr")


def lever(r: pd.Series, rf: pd.Series, L: float) -> pd.Series:
    """Daily margin at the house convention (a6.Lab.fin); rf is unused and kept for the call sites."""
    return L * r - (L - 1) * FIN.reindex(r.index)


def main() -> int:
    global FIN
    data = lib.load_inputs()
    for p_, t_ in TIER_FIX.items():
        data["meta"][p_]["tier_str"] = t_
    frames = ga.frames(data)
    frame, start = frames["main"]
    rf = frame[TBILL]
    days = pd.Series(frame.index, index=frame.index).diff().dt.days
    FIN = (lib.dtb3_annual_rate(frame.index) + SPREAD) * days / 360.0
    A = json.loads((OUT / "a6.json").read_text(encoding="utf-8"))
    Dd = json.loads((OUT / "a6d.json").read_text(encoding="utf-8"))        # A6-d: defensive slots with the +5 bps double test
    A["defensive"] = Dd["defensive"]
    A["challenges"] = {**Dd["challenges"], **{k: v for k, v in A["challenges"].items() if k.startswith("g22")}}
    A["notes"] = {**{k: v for k, v in A["notes"].items() if not k.startswith("rich") and not k.startswith("launch_gated")}, **Dd["notes"]}
    A["next_vs_launch"] = Dd["next_vs_launch"]
    ref = next(s for s in A["sweep"] if s["context"] == "C_L" and s["ds"] == 0.0)
    items = [("DEFENSIVE", f"d_{s}", A["defensive"][s]) for s in DEF_SLOTS if A["defensive"].get(s)]
    items.append(("DEFENSIVE", "d_ref_d0", {"weights": {"core5": 0.6, "btal_qqq": 0.4}, "lever": 1.0, "tails": ref["raw"]["tails"], "capacity": None}))
    items += [("GROWTH", f"g_{s}", A["growth"][s]) for s in GRO_SLOTS]
    rows, series = {}, {}
    for prod, key, src in items:
        w = {k: v / sum(src["weights"].values()) for k, v in src["weights"].items()}
        L = float(src.get("lever", 1.0))
        r = lib.book_returns(frame, Book(key, tuple(w), "EQ", w), start)
        r = lever(r, rf, L) if L != 1.0 else r
        series[key] = r
        m = metrics(r, data, rf)
        bk = Book(key, tuple(w), "EQ", w)
        ops = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in lib.ops_fields(bk, data).items() if not isinstance(v, str)}
        extra = {}
        for f in ("s3_plus_5bps", "s1_house_cash", "s6_exact", "s5_hpi_live_gap"):
            fr, s_ = frames[f]
            rr = lib.book_returns(fr, bk, s_)
            rr = lever(rr, fr[TBILL], L) if L != 1.0 else rr
            ss = ga.window_stats(rr.to_frame(), data["index"])
            extra[f] = {"cagr": float(ss["gross_cagr"][0]), "dd": float(ss["gross_dd"][0]),
                        "xsharpe": ev.xsharpe(rr.to_numpy(), fr[TBILL].reindex(rr.index).to_numpy())}
        rows[key] = {"product": prod, "weights": {k: round(v, 4) for k, v in w.items()}, "lever": L, "m": m, "tail": src.get("tails"), "ops": ops,
                     "needs_wiring": sorted(p for p in w if p != TBILL and (data["meta"][p]["tier_str"] != "wired" or p in fp.NOT_LIVE_TRADABLE)),
                     "daily_pods": sorted(p for p in w if p in fp.DAILY), "frames": extra, "capacity": src.get("capacity"),
                     "cash": src.get("cash"), "g": src.get("g"), "plus10_cagr": src.get("plus10_cagr"), "financing": src.get("financing"),
                     "reg_t": src.get("reg_t"), "reg_t_flag": src.get("reg_t_flag"), "halves_xs": src.get("halves_xs"), "a6_name": src.get("name"),
                     "capacity_top": src.get("capacity_top"), "gated": key == "d_launch_gated", "tails_plus5": src.get("tails_plus5")}
        print("done", key, round(m["cagr"], 4), round(m["maxdd"], 4), flush=True)
    # Capacity for reference rows a6.py did not size (unlevered, same route model).
    miss = {k: {a: b / sum(r["weights"].values()) for a, b in r["weights"].items()} for k, r in rows.items() if r["capacity"] is None}
    if miss:
        cap = ev.rd.capacity(miss, data, {k: rows[k]["frames"]["s6_exact"]["cagr"] - 0.0161 for k in miss})
        for k in miss:
            rec = cap.get(k)
            rows[k]["capacity"] = float(rec["routes"]["worked+blocks"]["recommended"]) if rec else None
            rows[k]["capacity_top"] = bool(rows[k]["capacity"] is not None and rows[k]["capacity"] >= 250e6)
    # Alpha, gross and net, against QQQ (M0), the ETF mix (M1) and the ETF mix + QQQ 200-day rule (M2).
    closes = pd.concat([lib.common.load_total_return_close_ser(s, "2005-01-01", fp.END.strftime("%Y-%m-%d")) for s in ev.afl.ETF_LIST], axis=1)
    closes.columns = ev.afl.ETF_LIST
    closes = closes.reindex(data["index"]).loc["2005-01-01":]
    naive = ev.afl.naive_rule_return_df(closes, data["sleeve"][TBILL])
    etf_r = closes.pct_change(fill_method=None)
    tb_w = ev.afl.weekly_ser(data["sleeve"][TBILL])
    fac_w = pd.DataFrame({s: ev.afl.weekly_ser(etf_r[s]) for s in ev.afl.FACTOR_M1_LIST}).sub(tb_w, axis=0)
    trend_w = ev.afl.weekly_ser(naive["NAIVE_QQQ_TREND200"]) - tb_w
    alpha = []
    for n in ALPHA_ROWS:
        if n not in series:
            continue
        for basis, ser in (("gross", series[n]), ("net", ga.net_return_series(series[n]))):
            y = (ev.afl.weekly_ser(ser) - tb_w).dropna().iloc[1:-1]
            x1 = fac_w.reindex(y.index)
            for model, x in (("M0", x1[["QQQ"]]), ("M1", x1), ("M2", x1.assign(TREND200=trend_w.reindex(y.index)))):
                alpha.append({"book": n, "basis": basis, "model": model, **ev.afl.newey_west_ols(y, x, ev.afl.NW_LAG_INT)})
    bench = {}
    for lab_, col in (("S&P 500", "SPXTR"), ("60/40", "SIXTY_FORTY"), ("QQQ", "QQQ")):
        bench[lab_] = metrics(data["bench"][col].loc[LONG_START:fp.END], data, rf)
    pods = ["taa3x", "taa3x_1n", "ndx_vxn", "core5", "btal_qqq", "etf_dv2", "eom_flow", "downshock", "dv2", "hpi_vote"]
    M = frame.loc[start:fp.END, pods].dropna()
    spx = data["bench"]["SPXTR"].reindex(M.index)
    worst = spx <= spx.quantile(0.05)
    corr = {"names": pods, "all": M.corr().round(2).values.tolist(), "falls": M[worst].corr().round(2).values.tolist()}
    # Engine profile (stand-alone pods) for the downshock section.
    prof = {}
    for p in ("core5", "btal_qqq", "etf_dv2", "eom_flow", "downshock"):
        r = frame.loc[start:fp.END, p]
        nav = np.r_[1, np.cumprod(1 + r.to_numpy())]
        prof[p] = {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "dd": float((nav / np.maximum.accumulate(nav) - 1).min()),
                   "xs": ev.xsharpe(r.to_numpy(), rf.reindex(r.index).to_numpy()),
                   "crises": {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()},
                   "falls_corr": float(np.corrcoef(r[worst.reindex(r.index).fillna(False)], spx.reindex(r.index)[worst.reindex(r.index).fillna(False)])[0, 1])}
    payload = {"rows": rows, "bench": bench, "alpha": alpha, "corr": corr, "sweep": A["sweep"], "challenges": A["challenges"],
               "notes": A["notes"], "profile": prof, "next_vs_launch": A.get("next_vs_launch")}
    (OUT / "report_a6.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    fp.ledger("report_a6_written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
