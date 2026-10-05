"""Fund products, final pass: products, rungs, challengers, slot tests, dial map, margin and blends (SPEC 3-4).

Usage: PYTHONDONTWRITEBYTECODE=1 python study.py   (after build_sources.py). Writes <study>/report/study.json.
Nothing here chooses a weight: the products are fixed in g_lib.PRODUCTS (SPEC 3); the data can only add BIL.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import g_lib as g
from g_lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Book, Lab, ga, lib

FRAME_KEYS = ("s1_house_cash", "s3_plus_5bps", "s3b_plus_5bps_ex_bil", "s2_proxy_unscaled", "s6_exact", "s9_mr_fair_cash", "s10_mr_cash_0")
CAPSULE_BOOKS = {"capsule TAA 3x": {"taa3x": 1.0}, "capsule TAA 3x 1N": {"taa3x_1n": 1.0}, "capsule MOM": g.MOM, "capsule MR": g.MR,
                 "capsule DEF": g.DEF}


def monthly_nav(r: pd.Series) -> list:
    m = (1.0 + r).cumprod().resample("ME").last()
    return [[d.strftime("%Y-%m"), round(float(v), 4)] for d, v in m.items()]


def monthly_dd(r: pd.Series) -> list:
    nav = (1.0 + r).cumprod()
    dd = (nav / nav.cummax().clip(lower=1.0) - 1.0).resample("ME").min()
    return [[d.strftime("%Y-%m"), round(float(v), 4)] for d, v in dd.items()]


def book_block(lab: Lab, n: str) -> dict:
    """Everything reported for one book (MAIN frame unless stated)."""
    c = lab.cands[n]
    r = lab.r(n)
    data = lab.data
    fm = lib.full_metrics(r, data, "long")
    st = ga.window_stats(r.to_frame(), data["index"])
    yrs = (1.0 + r).groupby(r.index.year).prod() - 1.0
    net = ga.net_return_series(r)
    net_y = (1.0 + net).groupby(net.index.year).prod() - 1.0
    blocks = {k: g.stats(lib.window(r, lo, hi), lab.rf) for k, (lo, hi) in BLOCK_DICT.items()}
    cdd = {}
    for k, (lo, hi) in lib.CRISIS_DICT.items():
        win = r[(r.index > pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))]
        w = np.r_[1.0, np.cumprod(1.0 + win.to_numpy())]
        cdd[k] = float((w / np.maximum.accumulate(w) - 1.0).min())
    cof = [{"start": str(lo.date()), "end": str(hi.date()), "sixty_forty": ret,
            "book": float(lib.common.window_return_float(r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))}
           for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, END)]
    frames_out = {f: lab.frame_stats(n, f) for f in FRAME_KEYS}
    frames_out["plus_10bps"] = g.stats(lab.plus10(n), lab.rf)
    keep = ("vol", "sharpe", "sharpe_excess", "maxdd", "dd_peak", "dd_trough", "dd_recovery", "underwater_days", "calmar", "cvar5_daily",
            "cvar5_21d", "worst_month", "worst_year", "worst_12m", "pos_months", "beta", "corr_spx", "crisis_corr")
    return {"weights": c["w"], "kind": c.get("kind"), "q": c["q"],
            "long": {k: (fm[f"long_{k}"] if isinstance(fm[f"long_{k}"], (str, type(None))) else float(fm[f"long_{k}"])) for k in keep},
            "net": {"cagr": float(np.prod(1.0 + net.to_numpy()) ** (252 / len(net)) - 1), "dd": float(st["net_dd"][0]),
                    "sharpe0": float(st["net_sharpe"][0]), "xs": g.xsharpe(net.to_numpy(), lab.rf.reindex(net.index).to_numpy())},
            "years": {int(y): float(v) for y, v in yrs.items()}, "years_net": {int(y): float(v) for y, v in net_y.items()},
            "blocks": blocks, "halves_xs": list(lab.halves(n)), "crises_dd": cdd, "cofalls": cof, "frames": frames_out,
            "tails": lab.tails(n), "tails_plus5": lab.tails(n, "s3_plus_5bps"),
            "rungs": {rung: lab.rung_detail(n, rung) for rung in g.RUNG_ORDER}, "strictest_rung": lab.strictest_rung(n),
            "nav": monthly_nav(r), "ddm": monthly_dd(r), "nav_net": monthly_nav(net)}


def register_s7(lab: Lab) -> list:
    """S7: walk-forward inverse volatility over the three capsules (SPEC 4)."""
    caps = {"TAA": {"taa3x": 1.0}, "MOM": g.MOM, "MR": g.MR}

    def capsule_frame(frame_key: str) -> pd.DataFrame:
        fr, s = lab.frames[frame_key]
        return pd.DataFrame({k: g.book_returns(fr, g.blend((1.0, w)), s) for k, w in caps.items()})

    log: list = []

    def s7(frame_key: str) -> pd.Series:
        fr, s = lab.frames[frame_key]
        # *** CRITICAL*** weights always come from the MAIN frame's capsule history, strictly before each reset
        # (lib.book_returns drops the period's first session). On the EXACT window the MAIN frame's EXACT part is
        # the source; a period with fewer than 60 sessions of history falls back to equal weights (2008, 2012).
        src = capsule_frame("main" if s == LONG_START else "s6_exact")
        return lib.book_returns(capsule_frame(frame_key), Book("S7", ("TAA", "MOM", "MR"), "IV"), s, weight_source=src,
                                weight_log=log if frame_key == "main" else None)

    lab.add_series(g.S7, s7, kind="challenger")
    return log


def main() -> int:
    g.OUT.mkdir(parents=True, exist_ok=True)
    lab = Lab()
    data, frame, rf = lab.data, lab.frame, lab.rf
    out: dict = {"meta": {"end": str(END.date()), "long_start": str(LONG_START.date()), "exact_start": str(EXACT_START.date()),
                          "cut": str(g.CUT.date()), "rungs": g.RUNGS, "limits": list(g.LIMITS), "sessions": lab.n}, "checks": {}}

    # ── start-up checks: the book model equals lib's, and the old study's incumbent reproduces ──
    w = g.INCUMBENT
    diff = float((lab.ret(w) - lib.book_returns(frame, Book("chk", tuple(w), "EQ", w), lab.start)).abs().max())
    assert diff < 1e-14, diff
    lab.add(g.S13_OLD, g.INCUMBENT, kind="challenger")
    old = json.loads((g.WT_REPO / "results/research/portfolio/fund_products_20260930/report/a6.json").read_text(encoding="utf-8"))["growth"]["launch"]
    now_t, now_q = lab.tails(g.S13_OLD), lab.cands[g.S13_OLD]["q"]
    out["checks"] = {"book_model_max_abs_diff": diff, "incumbent_p20": [old["tails"]["p20"], now_t["p20"]],
                     "incumbent_cagr": [old["q"]["cagr"], now_q["cagr"]], "incumbent_dd": [old["q"]["dd"], now_q["dd"]]}
    assert abs(old["tails"]["p20"] - now_t["p20"]) < 1e-12 and abs(old["q"]["cagr"] - now_q["cagr"]) < 1e-12, out["checks"]
    print("checks OK", out["checks"], flush=True)
    g.ledger("study_started", checks=out["checks"])

    # ── register books ──
    for n, w in g.PRODUCTS.items():
        lab.add(n, w, kind="product")
    for n, w in g.CHALLENGERS.items():
        lab.add(n, w, kind="challenger")
    s7_log = register_s7(lab)
    out["s7_weights"] = [{"reset": str(d.date()), **{k: float(v) for k, v in ww.items()}} for d, _, ww in s7_log]
    for n, w in g.SLOT_TESTS.items():
        lab.add(n, w, kind="slot")
    for n, w in g.STAND_INS.items():
        lab.add(n, w, kind="standin")
    for n, w in CAPSULE_BOOKS.items():
        lab.add(n, g.blend((1.0, w)), kind="capsule")
    lab.add("old growth plus", g.MONTHLY_PLUS, kind="reference")     # the key now holds the NEW Monthly Plus (g_lib, O3)
    lab.add(g.OLD_PLUS_KEY, g.OLD_PLUS, kind="reference")
    dial = {}
    for taa in ("taa3x", "taa3x_1n"):
        for t in g.DIAL_SHARES:
            dial[lab.add(f"dial {taa} {int(round(t * 100))}", g.three(taa, t, (1 - t) / 2, (1 - t) / 2), kind="dial")] = {"taa": taa, "share": t}
    for fk in ("main", "s3_plus_5bps"):
        lab.run_tails(list(lab.cands), fk)

    # ── products: target rung; BIL in 5% steps up to 30% if it fails (SPEC 3) ──
    out["products"] = {}
    for n in g.PRODUCTS:
        rung, final, cash = g.TARGET_RUNG[n], n, 0.0
        while not lab.rung_ok(final, rung) and cash < g.MAX_BIL - 1e-9:
            cash = round(cash + 0.05, 2)
            final = lab.add(f"{n} + {int(round(cash * 100))}% BIL", g.with_cash(g.PRODUCTS[n], cash), kind="product_scaled")
        fits = lab.rung_ok(final, rung)
        out["products"][n] = {"label": g.PRODUCT_LABEL[n], "target_rung": rung, "final": final, "cash_added": cash, "fits_rung": fits,
                              "raw": lab.rung_detail(n, rung), "final_detail": lab.rung_detail(final, rung),
                              "strictest_rung": lab.strictest_rung(final)}
        q = lab.cands[final]["q"]
        print("PRODUCT", n, rung, "->", final, "fits", fits, f"cagr {q['cagr']:.4f} xs {q['xs']:.3f} dd {q['dd']:.4f}", lab.tails(final), flush=True)
    # Menu rule: a higher product is offered only if its MAIN CAGR is >= 1.5 pp above the offered product below it.
    below = None
    for n in g.PRODUCTS:
        p = out["products"][n]
        cagr = lab.cands[p["final"]]["q"]["cagr"]
        step = None if below is None else cagr - lab.cands[out["products"][below]["final"]]["q"]["cagr"]
        p["step_over_product_below"] = step
        p["offered"] = bool(p["fits_rung"] and (below is None or step >= 0.015))
        if p["offered"]:
            below = n

    # ── challenge tests against GR1 (SPEC 4), both directions ──
    out["challenges"], out["challenges_reverse"] = [], []
    for ch in [n for n in lab.cands if lab.cands[n].get("kind") == "challenger"]:
        res = lab.challenge(ch, "GR1")
        out["challenges"].append(res)
        out["challenges_reverse"].append(lab.challenge("GR1", ch))
        print("CHALLENGE", ch, "share", round(res["share_xs"], 3), "passed", res["passed"], res["checks"], flush=True)
    out["gr2_vs_old_plus"] = lab.challenge("GR2", "old growth plus", breach_key="p25", rung="GROWTH PLUS")
    out["old_plus_vs_gr2"] = lab.challenge("old growth plus", "GR2", breach_key="p25", rung=None)

    # ── slot tests on GR1 (SPEC 4) ──
    out["slots"] = {}
    series = {"main": lambda n: lab.r(n), "plus5": lambda n: lab.r(n, "s3_plus_5bps"), "plus10": lambda n: lab.plus10(n)}
    for t in g.SLOT_TESTS:
        row = {"q": lab.cands[t]["q"], "tails": lab.tails(t), "frames": {}}
        for fk, fn in series.items():
            a, b = fn("GR1"), fn(t)
            row["frames"][fk] = {"gr1": g.stats(a, rf), "slot": g.stats(b, rf), **lab.paired(a, b)}
        row["blocks"] = {k: lab.paired(lib.window(lab.r("GR1"), lo, hi), lib.window(lab.r(t), lo, hi), seeds=range(3))
                         for k, (lo, hi) in BLOCK_DICT.items() if k in ("A", "B", "C")}
        row["adds_return_not_sharpe"] = bool(1.0 - row["frames"]["main"]["share_xs"] >= 0.80)
        out["slots"][t] = row
        print("SLOT", t, {fk: (round(v["share_xs"], 3), round(v["share_cagr"], 3)) for fk, v in row["frames"].items()}, flush=True)
    # The MR gate: extra cost per side at which GR1 and GR1-L (MR third in BIL) are equal (linear in cost).
    gate = {}
    for metric in ("xs", "cagr"):
        d0 = out["slots"][g.GR1_L]["frames"]["main"]["gr1"][metric] - out["slots"][g.GR1_L]["frames"]["main"]["slot"][metric]
        d10 = out["slots"][g.GR1_L]["frames"]["plus10"]["gr1"][metric] - out["slots"][g.GR1_L]["frames"]["plus10"]["slot"][metric]
        gate[metric] = {"gap_at_0": d0, "gap_at_10bps": d10, "breakeven_bps_per_side": None if d0 <= d10 else float(10.0 * d0 / (d0 - d10))}
    out["mr_gate_breakeven"] = gate

    # ── stand-ins: GROWTH rung and the test against S9 ──
    out["standins"] = {n: {"rung": lab.rung_detail(n, "GROWTH"), "vs_s9": lab.challenge(n, g.S9),
                           "corr_with_gr1": float(lab.r(n).corr(lab.r("GR1")))} for n in g.STAND_INS}

    # ── margin alternatives (SPEC 3): debt as a negative-weight pod, annual reset; vol-matched and CAGR-matched ──
    out["margin"] = {}

    def lev_row(base: str, L: float, tgt: str) -> dict:
        wl = g.levered(g.PRODUCTS[base], L)
        name = lab.add(f"{base} x{L:.2f}", wl, kind="margin")
        hard = f"p{int(round(-g.RUNGS[g.TARGET_RUNG[tgt]][1] * 100))}"
        wp: list = []
        g.book_returns(frame, wl, lab.start, weight_path=wp)
        _, cols, pw = wp[0]
        gross = 1.0 - pw[:, cols.index(g.DEBT)]                       # pod exposure per unit of equity, prior close
        r0 = lab.r(base)
        daily = L * r0 - (L - 1.0) * frame[g.DEBT].reindex(r0.index)  # constant daily leverage (sensitivity)
        row = {"name": name, "L": L, "q": lab.cands[name]["q"], "tails": lab.tails(name), "breach_key": hard,
               "plus5": lab.frame_stats(name, "s3_plus_5bps"), "exact": lab.frame_stats(name, "s6_exact"),
               "spread_050": g.stats(lab.ret(g.levered(g.PRODUCTS[base], L, f"{g.DEBT}_050")), rf),
               "spread_250": g.stats(lab.ret(g.levered(g.PRODUCTS[base], L, f"{g.DEBT}_250")), rf),
               "daily_constant": g.stats(daily, rf), "peak_leverage": float(gross.max()), "mean_leverage": float(gross.mean()),
               "reg_t": lab.reg_t(wl), "needs_portfolio_margin": bool(lab.reg_t(wl) > 0.90)}
        return row

    for base, tgt in (("GR1", "GR2"), ("GR1", "GR3"), ("GR2", "GR3")):
        qb, qt = lab.cands[base]["q"], lab.cands[tgt]["q"]
        L_vol = round(qt["vol"] / qb["vol"], 2)
        L_cagr = None
        for L in np.arange(1.0, 3.001, 0.01):
            if g.stats(lab.ret(g.levered(g.PRODUCTS[base], float(L))), rf)["cagr"] >= qt["cagr"]:
                L_cagr = round(float(L), 2)
                break
        hard = f"p{int(round(-g.RUNGS[g.TARGET_RUNG[tgt]][1] * 100))}"
        out["margin"][f"{base} -> {tgt}"] = {
            "target": {"q": qt, "tails": lab.tails(tgt), "plus5": lab.frame_stats(tgt, "s3_plus_5bps"), "exact": lab.frame_stats(tgt, "s6_exact"),
                       "reg_t": lab.reg_t(g.PRODUCTS[tgt]), "breach_key": hard},
            "vol_matched": lev_row(base, L_vol, tgt), "cagr_matched": None if L_cagr is None else lev_row(base, L_cagr, tgt)}
        print("MARGIN", base, "->", tgt, "L_vol", L_vol, "L_cagr", L_cagr, flush=True)
    # Owner request 2026-10-05 (O8): Growth at a fixed leverage of 1.40, shown as an alternative to Aggressive.
    out["margin"]["GR1 -> GR3"]["fixed_140"] = lev_row("GR1", 1.40, "GR3")

    # ── client blends with the defensive launch (descriptive) ──
    d_launch = g.with_cash(g.DEF, 0.10)
    lab.add("defensive launch", d_launch, kind="blend")
    out["blends"] = {}
    vol_g1 = lab.cands["GR1"]["q"]["vol"]
    for share in (0.20, 0.40, 0.60, 0.80):
        w = g.blend((share, g.PRODUCTS["GR1"]), (1 - share, d_launch))
        nm = lab.add(f"GR1 {int(share * 100)} / defensive {int((1 - share) * 100)}", w, kind="blend")
        c = max(0.0, round(1.0 - lab.cands[nm]["q"]["vol"] / vol_g1, 2))
        dil = lab.add(f"GR1 + {int(round(c * 100))}% BIL (same vol)", g.with_cash(g.PRODUCTS["GR1"], c), kind="blend")
        out["blends"][nm] = {"q": lab.cands[nm]["q"], "tails": lab.tails(nm), "defense_first_capital": float(w.get("taa3x", 0) + w.get("btal_qqq", 0)),
                             "diluted": {"name": dil, "cash": c, "q": lab.cands[dil]["q"], "tails": lab.tails(dil)}}
    out["blends"]["defensive launch"] = {"q": lab.cands["defensive launch"]["q"], "tails": lab.tails("defensive launch")}
    out["corr_gr1_defensive"] = {"all": float(lab.r("GR1").corr(lab.r("defensive launch"))),
                                 "spx_worst5": lib.crisis_corr(lab.r("GR1"), data["bench"]["SPXTR"]) if False else None}
    spx = data["bench"]["SPXTR"].reindex(lab.r("GR1").index)
    mask = spx <= spx.quantile(0.05)
    out["corr_gr1_defensive"]["spx_worst5"] = float(lab.r("GR1")[mask].corr(lab.r("defensive launch")[mask]))

    # ── dial map and construction table ──
    out["dial"] = dial
    table = ["GR1", "S6 equal pods", "S11 cluster parity", "S12 inverse vol (fixed)", "S4 core + satellites", g.S7]
    rows = {n: {"cagr": lab.cands[n]["q"]["cagr"], "xs": lab.cands[n]["q"]["xs"], "dd": lab.cands[n]["q"]["dd"], "p20": lab.tails(n)["p20"]} for n in table}
    rank = {m: sorted(table, key=lambda n: rows[n][m] * (1 if m == "p20" else -1)).index("GR1") + 1 for m in ("cagr", "xs", "dd", "p20")}
    out["construction"] = {"rows": rows, "gr1_rank": rank, "n": len(table)}

    # ── full blocks for every reported book ──
    finals = [v["final"] for v in out["products"].values()]
    names = list(dict.fromkeys(list(g.PRODUCTS) + finals + [n for n in lab.cands if lab.cands[n].get("kind") in
                                                            ("challenger", "slot", "capsule", "standin", "dial", "margin", "reference", "blend")]))
    out["books"] = {n: book_block(lab, n) for n in names}
    out["bench"] = {k: g.stats(data["bench"][k].loc[LONG_START:END].dropna(), rf) for k in ("SPXTR", "SIXTY_FORTY", "QQQ")}
    out["bench_nav"] = {k: monthly_nav(data["bench"][k].loc[LONG_START:END].dropna()) for k in ("SPXTR", "QQQ")}
    out["inputs"] = {}
    for alias in g.NEW_ALIAS_LIST:
        m = data["meta"][alias]
        out["inputs"][alias] = {k: m.get(k) for k in ("strategy_import_str", "tier_str", "first_invested_date_str", "end_date_str", "transaction_count_int",
                                                      "mean_cash_nav_weight_float", "minimum_cash_nav_weight_float", "negative_cash_day_count_int",
                                                      "positive_cash_rate_policy_str", "negative_cash_financing_policy_str")}
    (g.OUT / "study.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("study_finished", products={k: (v["final"], v["fits_rung"], v["offered"]) for k, v in out["products"].items()},
             challengers_passed=[c["challenger"] for c in out["challenges"] if c["passed"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
