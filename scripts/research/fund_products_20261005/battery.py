"""Fund products, final pass: the robustness battery (SPEC_FROZEN.md section 5, items 3-10, 13, 14).

Usage: PYTHONDONTWRITEBYTECODE=1 python battery.py   (after study.py). Writes <study>/report/battery.json.
Reads study.json only for the final (BIL-scaled) product names; every number is recomputed from the frames.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import g_lib as g
from g_lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Lab, ga, lib
import study as st

from alpha.stats.psr_dsr import deflated_sharpe_ratio  # noqa: E402  (worktree copy, before any main-checkout path)

sys.path.insert(1, str(lib.FUND_MENU_DIR))
import allocator_first_look as afl  # noqa: E402

ALL_CAPS = ("TAA", "MOM", "MR", "DEF")
RESET_POLICIES = ["annual", "none", "quarterly", "monthly"] + [f"annual-{m}" for m in range(2, 13)]
RESET_COST = 0.00025


def decay_frame(frame: pd.DataFrame, k_of: dict, shift_of: dict | None = None) -> pd.DataFrame:
    """SPEC 5.3: r_p' = r_p - (1 - k) x mu_p, mu_p = the pod's mean daily excess over BIL on LONG in this frame.
    shift_of gives an explicit per-pod daily shift instead (the common-factor shock)."""
    out = frame.copy()
    win = frame.loc[LONG_START:END]
    for alias, cap in g.CAPSULE_OF.items():
        if alias not in out.columns:
            continue
        if shift_of is not None:
            out[alias] = frame[alias] - shift_of.get(alias, 0.0)
        elif cap in k_of:
            out[alias] = frame[alias] - (1.0 - k_of[cap]) * float((win[alias] - win[TBILL]).mean())
    return out


def neighbours(t: float, m: float, r: float) -> list[tuple[float, float, float]]:
    """SPEC 5.4: offsets (a, b, c) in {-10, -5, 0, +5, +10} pp with a + b + c = 0 (19 books incl. the product)."""
    out = []
    for da in (-0.10, -0.05, 0.0, 0.05, 0.10):
        for db in (-0.10, -0.05, 0.0, 0.05, 0.10):
            dc = -(da + db)
            if abs(dc) <= 0.10 + 1e-9 and min(t + da, m + db, r + dc) >= -1e-9:
                out.append((round(t + da, 6), round(m + db, 6), round(r + dc, 6)))
    return out


def rolling(r: pd.Series, rf: pd.Series, n: int) -> tuple[dict, pd.Series]:
    x = r - rf.reindex(r.index)
    xs = (x.rolling(n).mean() / x.rolling(n).std(ddof=1) * np.sqrt(252)).dropna()
    cg = np.expm1(np.log1p(r).rolling(n).sum() * 252 / n).dropna()
    return {"xs_min": float(xs.min()), "xs_p10": float(xs.quantile(0.10)), "xs_median": float(xs.median()),
            "cagr_min": float(cg.min()), "cagr_p10": float(cg.quantile(0.10)), "cagr_median": float(cg.median()),
            "share_cagr_below_0": float((cg < 0).mean())}, xs


def variance_ratio(r: pd.Series, q: int) -> float:
    lr = np.log1p(r.to_numpy())
    agg = np.convolve(lr, np.ones(q), mode="valid")
    return float(agg.var(ddof=1) / (q * lr.var(ddof=1)))


def enb(corr: np.ndarray) -> float:
    """Effective number of bets: (sum of eigenvalues)^2 / sum of squared eigenvalues of the correlation matrix."""
    lam = np.linalg.eigvalsh(corr)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def path_stats(lab: Lab, R: np.ndarray, rf: np.ndarray, seeds=range(10), chunk: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Excess Sharpe and CAGR of every column of R on every bootstrap path (seeds pooled): arrays (paths, books)."""
    xs_all, cg_all = [], []
    n = len(R)
    for s in seeds:
        idx_t = lab.idx(s, g.BOOT_BLOCK, n)
        for a in range(0, idx_t.shape[1], chunk):
            ii = idx_t[:, a:a + chunk]                      # (days, paths)
            smp = R[ii]                                     # (days, paths, books)
            x = smp - rf[ii][:, :, None]
            xs_all.append(x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(252))
            cg_all.append(np.expm1(np.log1p(smp).sum(axis=0) * 252 / n))
    return np.vstack(xs_all), np.vstack(cg_all)


def main() -> int:
    lab = Lab()
    data, frame, rf = lab.data, lab.frame, lab.rf
    study = json.loads((g.OUT / "study.json").read_text(encoding="utf-8"))
    finals = {n: study["products"][n]["final"] for n in g.PRODUCTS}
    for n, w in g.PRODUCTS.items():
        lab.add(n, w, kind="product")
        if finals[n] != n:
            lab.add(finals[n], g.with_cash(w, study["products"][n]["cash_added"]), kind="product_scaled")
    for n, w in {**g.CHALLENGERS, **g.SLOT_TESTS, **{k: g.blend((1.0, v)) for k, v in st.CAPSULE_BOOKS.items()}}.items():
        lab.add(n, w)
    for key, row in study["margin"].items():
        for kind in ("vol_matched", "cagr_matched", "fixed_140"):
            if row.get(kind):
                lab.add(row[kind]["name"], g.levered(g.PRODUCTS[key.split(" -> ")[0]], row[kind]["L"]))
    for name in study["dial"]:
        lab.add(name, g.blend((1.0, study["books"][name]["weights"])))      # stored weights are rounded: renormalise
    st.register_s7(lab)
    products = list(dict.fromkeys([finals[n] for n in g.PRODUCTS] + list(g.PRODUCTS)))
    hard_of = {n: g.RUNGS[g.TARGET_RUNG[n.split(" + ")[0]]][1] for n in products}
    core = products + ["S1 no momentum", "S4 core + satellites", "S5 MR tilt", g.S9, g.S10, "S11 cluster parity"] + list(g.SLOT_TESTS)
    lab.add("old growth plus", g.MONTHLY_PLUS, kind="reference")      # the new Monthly Plus (g_lib, O3)
    fixed_books = [n for n in lab.cands if "series_fn" not in lab.cands[n]]
    out: dict = {"finals": finals}

    # ── 5.3 edge decay ──
    beta = {}
    qqq_x = afl.weekly_ser(frame[g.QQQ].loc[LONG_START:END]) - afl.weekly_ser(rf.loc[LONG_START:END])
    for alias in g.CAPSULE_OF:
        y = (afl.weekly_ser(frame[alias].loc[LONG_START:END]) - afl.weekly_ser(rf.loc[LONG_START:END])).iloc[1:-1]
        beta[alias] = float(np.cov(y, qqq_x.reindex(y.index))[0, 1] / qqq_x.reindex(y.index).var())
    qqq_mu = float((frame[g.QQQ] - rf).loc[LONG_START:END].mean())
    scen = {"all at 0.75": dict.fromkeys(ALL_CAPS, 0.75), "all at 0.5": dict.fromkeys(ALL_CAPS, 0.5)}
    for cap in ("TAA", "MOM", "MR"):
        scen[f"{cap} dead"] = {cap: 0.0}
        scen[f"{cap} dead, others 0.75"] = {**dict.fromkeys(ALL_CAPS, 0.75), cap: 0.0}
    decay: dict = {"beta_qqq": beta, "scenarios": {}}
    frames_dec = {name: decay_frame(frame, k_of) for name, k_of in scen.items()}
    frames_dec["common shock (half the QQQ premium)"] = decay_frame(frame, {}, {a: beta[a] * 0.5 * qqq_mu for a in beta})
    win_ = frame.loc[LONG_START:END]
    frames_dec["Defense First dead (TAA and BTAL_QQQ)"] = decay_frame(
        frame, {}, {a: float((win_[a] - win_[TBILL]).mean()) for a in ("taa3x", "taa3x_1n", "btal_qqq")})
    nb_names = {}
    for (a_, b_, c_) in neighbours(*g.PRODUCT_SHAPE["GR1"][1:]):
        nb_names[lab.add(f"nb GR1 {a_:.4f}/{b_:.4f}/{c_:.4f}", g.three("taa3x", a_, b_, c_) if a_ > 1e-9 else g.blend((b_, g.MOM), (c_, g.MR)))] = (a_, b_, c_)
    decay_books = [n for n in fixed_books if not n.startswith("capsule")] + list(nb_names)
    decay_books = list(dict.fromkeys(decay_books))
    for name, fr in frames_dec.items():
        R = np.column_stack([lab.ret(lab.cands[n]["w"], frame=fr).to_numpy() for n in decay_books])
        tm = lab.tail_matrix(R, limits=(-0.20, -0.25, -0.30))
        decay["scenarios"][name] = {}
        for j, n in enumerate(decay_books):
            r_ann = pd.Series(R[:, j], index=lab.r("GR1").index)
            row = g.stats(r_ann, rf)
            row.update({f"p{int(-L * 100)}": float(tm[L][:, j].mean()) for L in tm})
            row["no_reset"] = g.stats(lab.ret(lab.cands[n]["w"], frame=fr, policy="none"), rf)
            decay["scenarios"][name][n] = row
        print("decay", name, {n: round(decay["scenarios"][name][n]["xs"], 3) for n in products}, flush=True)
    dead = ("TAA dead", "MOM dead", "MR dead")
    decay["worst_case"] = {}
    for n in decay_books:
        w_ann = min(dead, key=lambda s: decay["scenarios"][s][n]["xs"])
        w_non = min(dead, key=lambda s: decay["scenarios"][s][n]["no_reset"]["xs"])
        decay["worst_case"][n] = {"scenario": w_ann, "xs": decay["scenarios"][w_ann][n]["xs"], "cagr": decay["scenarios"][w_ann][n]["cagr"],
                                  "no_reset_scenario": w_non, "no_reset_xs": decay["scenarios"][w_non][n]["no_reset"]["xs"]}
    rivals = [n for n in decay_books if n in nb_names or n in g.CHALLENGERS]
    best = max(rivals, key=lambda n: decay["worst_case"][n]["xs"])
    decay["minimax"] = {"gr1_worst_xs": decay["worst_case"]["GR1"]["xs"], "best_rival": best, "best_rival_worst_xs": decay["worst_case"][best]["xs"],
                        "gr1_is_minimax_within_0.02": bool(decay["worst_case"]["GR1"]["xs"] >= decay["worst_case"][best]["xs"] - 0.02)}
    # planning (k 0.75) and floor (k 0.5) in the +5 bps frame: the headline columns of SPEC 5.14 (d)
    f5 = lab.frames["s3_plus_5bps"][0]
    head_books = [n for n in fixed_books if not n.startswith("nb ")]
    decay["headline"] = {n: {"backtest": {**lab.cands[n]["q"], **{k: v for k, v in lab.tails(n).items() if not k.endswith("_max")}}} for n in head_books}
    for label, k in (("planning", 0.75), ("floor", 0.5)):
        fr = decay_frame(f5, dict.fromkeys(ALL_CAPS, k))
        R = np.column_stack([lab.ret(lab.cands[n]["w"], frame_key="s3_plus_5bps", frame=fr).to_numpy() for n in head_books])
        tm = lab.tail_matrix(R, limits=(-0.20, -0.25, -0.30))
        for j, n in enumerate(head_books):
            row = g.stats(pd.Series(R[:, j], index=lab.r("GR1").index), rf)
            row.update({f"p{int(-L * 100)}": float(tm[L][:, j].mean()) for L in tm})
            decay["headline"][n][label] = row
    # edge margin: the lowest k (0.05 grid) at which the product still passes its target rung's breach cap (MAIN)
    decay["edge_margin"] = {}
    em_books = products + [g.S9, "old growth plus"]        # the monthly books too (review v5.2): GROWTH and GROWTH PLUS caps
    hard_em = {**hard_of, g.S9: -0.25, "old growth plus": -0.25}      # Monthly 60 / 40 is read on GROWTH PLUS (O8)
    curve = {n: {} for n in em_books}
    curve_all: dict = {}
    for k in np.round(np.arange(1.0, -0.001, -0.05), 2):
        fr = decay_frame(frame, dict.fromkeys(ALL_CAPS, float(k)))
        R = np.column_stack([lab.ret(lab.cands[n]["w"], frame=fr).to_numpy() for n in em_books])
        tm = lab.tail_matrix(R, limits=(-0.20, -0.25, -0.30))
        for j, n in enumerate(em_books):
            curve[n][float(k)] = float(tm[hard_em[n]][:, j].mean())
            for L_ in (-0.20, -0.25, -0.30):
                curve_all.setdefault(n, {}).setdefault(f"p{int(-L_ * 100)}", {})[float(k)] = float(tm[L_][:, j].mean())
    for n in em_books:
        ok = [k for k, p in curve[n].items() if p <= 0.15]
        decay["edge_margin"][n] = {"curve": curve[n], "lowest_passing_k": min(ok) if ok else None,
                                   "passes_only_at_full_edge": bool(ok and min(ok) > 0.95),
                                   "by_limit": {key: {"curve": c, "lowest_passing_k": (min(k for k, p in c.items() if p <= 0.15) if any(p <= 0.15 for p in c.values()) else None)}
                                                for key, c in curve_all[n].items()}}
    out["edge_decay"] = decay

    # ── 5.4 weight plateau ──
    out["plateau"] = {}
    for n, (taa, t, m, r_) in g.PRODUCT_SHAPE.items():
        nbs = neighbours(t, m, r_)
        ws = [g.three(taa, a_, b_, c_) if a_ > 1e-9 else g.blend((b_, g.MOM), (c_, g.MR)) for (a_, b_, c_) in nbs]
        R = np.column_stack([lab.ret(w).to_numpy() for w in ws])
        hard = g.RUNGS[g.TARGET_RUNG[n]][1]
        build = g.RUNGS[g.TARGET_RUNG[n]][0]
        tm = lab.tail_matrix(R, limits=(hard,))
        rows = [{"t": a_, "m": b_, "r": c_, **g.stats(pd.Series(R[:, j], index=lab.r("GR1").index), rf), "breach": float(tm[hard][:, j].mean())}
                for j, (a_, b_, c_) in enumerate(nbs)]
        me = next(x for x in rows if abs(x["t"] - t) < 1e-6 and abs(x["m"] - m) < 1e-6)
        others = [x for x in rows if x is not me]
        summ = {}
        for key, better_high in (("cagr", True), ("xs", True), ("dd", True), ("breach", False)):
            v = np.array([x[key] for x in rows])
            better = (v > me[key] + 1e-12).sum() if better_high else (v < me[key] - 1e-12).sum()
            summ[key] = {"min": float(v.min()), "p25": float(np.percentile(v, 25)), "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                         "max": float(v.max()), "product": me[key], "rank_from_best": int(better) + 1}
        med_dd, med_br = float(np.median([x["dd"] for x in others])), float(np.median([x["breach"] for x in others]))
        flags = []
        for key in ("dd", "xs"):
            if summ[key]["rank_from_best"] <= 3:
                flags.append(f"local best on {key}, expect worse")
            if summ[key]["rank_from_best"] >= len(rows) - 2:
                flags.append(f"the principle costs on {key}")
        out["plateau"][n] = {"n": len(rows), "rows": rows, "summary": summ, "neighbour_median_dd": med_dd, "neighbour_median_breach": med_br,
                             "rung_robust": bool(med_dd >= build and med_br <= 0.15), "flags": flags}

    # ── 5.5 reset policy ──
    out["reset"] = {}
    for n in g.PRODUCTS:
        w = g.PRODUCTS[n]
        rows = {}
        for p in RESET_POLICIES:
            wp: list = []
            r0 = g.book_returns(frame, w, lab.start, policy=p, weight_path=wp)
            rows[p] = {"free": g.stats(r0, rf), "charged": g.stats(g.book_returns(frame, w, lab.start, policy=p, reset_cost=RESET_COST), rf)}
            if p == "annual":
                _, cols, pw = wp[0]
                cap_w = pd.DataFrame(pw, columns=cols).T.groupby(lambda a: g.CAPSULE_OF[a]).sum().T
                rows[p]["max_capsule_share"] = {c: float(cap_w[c].max()) for c in cap_w}
                rows[p]["min_capsule_share"] = {c: float(cap_w[c].min()) for c in cap_w}
        months = ["annual"] + [f"annual-{m}" for m in range(2, 13)]
        plan = {}
        for key in ("cagr", "xs"):
            v = sorted(rows[p]["free"][key] for p in months)
            jan = rows["annual"]["free"][key]
            inside = v[3] <= jan <= v[8]
            plan[key] = {"january": jan, "median_of_12": float(np.median(v)), "min": v[0], "max": v[-1], "january_in_middle_half": bool(inside),
                         "planning_figure": jan if inside else float(np.median(v))}
        out["reset"][n] = {"policies": rows, "start_month_spread": plan}

    # ── 5.6 start dates and rolling windows; 5.7 leave one year out ──
    out["start_years"], out["rolling"], out["rolling_share_above"], out["loyo"] = {}, {}, {}, {}
    cap_names = ["capsule TAA 3x", "capsule TAA 3x 1N", "capsule MOM", "capsule MR", g.S9]
    roll_xs, roll_all = {}, {}
    for n in core + cap_names[:-1]:
        r = lab.r(n)
        out["start_years"][n] = {int(y): g.stats(r.loc[f"{y}-01-01":], rf) for y in range(2008, 2022)}
        r3, roll_xs[n] = rolling(r, rf, 756)
        r5, xs5 = rolling(r, rf, 1260)
        out["rolling"][n] = {"3y": r3, "5y": r5}
        roll_all[n] = {"xs_3y": roll_xs[n], "xs_5y": xs5, "cagr_3y": np.expm1(np.log1p(r).rolling(756).sum() * 252 / 756).dropna(),
                       "cagr_5y": np.expm1(np.log1p(r).rolling(1260).sum() * 252 / 1260).dropna()}
        rows = {int(y): g.stats(r[r.index.year != y], rf) for y in sorted(set(r.index.year))}
        out["loyo"][n] = {"xs_min": min(v["xs"] for v in rows.values()), "xs_max": max(v["xs"] for v in rows.values()),
                          "cagr_min": min(v["cagr"] for v in rows.values()), "cagr_max": max(v["cagr"] for v in rows.values()),
                          "year_whose_removal_hurts_most": min(rows, key=lambda y: rows[y]["xs"])}
    for n in products:
        out["rolling_share_above"][n] = {o: float((roll_xs[n] > roll_xs[o].reindex(roll_xs[n].index)).mean()) for o in cap_names}
        out.setdefault("rolling_share_above_all", {})[n] = {
            key: {o: float((roll_all[n][key] > roll_all[o][key].reindex(roll_all[n][key].index)).mean()) for o in cap_names}
            for key in ("xs_3y", "xs_5y", "cagr_3y", "cagr_5y")}
    out["rolling_series"] = {n: [[d.strftime("%Y-%m"), round(float(v), 3)] for d, v in roll_xs[n].resample("ME").last().items()]
                             for n in products + [g.S9, "capsule TAA 3x", "capsule MOM", "capsule MR"]}

    # ── 5.8 bootstrap: the rung-reading items and CAGR percentiles ──
    boot: dict = {"blocks": {}, "horizons": {}, "frames": {}, "variance_ratio": {}, "cagr": {}}
    tb = products + [g.S9, "old growth plus"]
    limits3 = (-0.20, -0.25, -0.30)
    R = np.column_stack([lab.r(n).to_numpy() for n in tb])
    for block in (1.0, 21.0, 63.0, 126.0, 252.0):
        tm = lab.tail_matrix(R, limits=limits3, block=block)
        boot["blocks"][str(int(block))] = {n: {f"p{int(-L * 100)}": float(tm[L][:, j].mean()) for L in limits3} for j, n in enumerate(tb)}
    for h in (756, 1260):
        tm = lab.tail_matrix(R, limits=(-0.10, -0.15, -0.20, -0.25, -0.30), horizon=h)
        boot["horizons"][str(h)] = {n: {f"p{int(-L * 100)}": float(tm[L][:, j].mean()) for L in tm} for j, n in enumerate(tb)}
    for fk in ("s2_proxy_unscaled", "s6_exact"):
        Rf = np.column_stack([lab.r(n, fk).to_numpy() for n in tb])
        tm = lab.tail_matrix(Rf, limits=limits3)
        boot["frames"][fk] = {n: {f"p{int(-L * 100)}": float(tm[L][:, j].mean()) for L in limits3} for j, n in enumerate(tb)}
    for n in tb:
        boot["variance_ratio"][n] = {"21": variance_ratio(lab.r(n), 21), "63": variance_ratio(lab.r(n), 63)}
    bp = ga.bootstrap_paths(R, np.ascontiguousarray(lab.idx(0).T))
    for j, n in enumerate(tb):
        boot["cagr"][n] = {f"{basis}_p{q}": float(np.percentile(bp[f"{basis}_cagr"][:, j], q)) for basis in ("gross", "net") for q in (5, 25, 50, 75)}
        boot["cagr"][n].update({"gross_dd_p50": float(np.percentile(bp["gross_dd"][:, j], 50)), "gross_dd_p10": float(np.percentile(bp["gross_dd"][:, j], 10)),
                                "net_dd_p50": float(np.percentile(bp["net_dd"][:, j], 50))})
    out["bootstrap"] = boot

    # ── 5.14 how much to believe ──
    believe = {}
    rf_arr = rf.reindex(lab.r("GR1").index).to_numpy()
    for label, k in (("k1", 1.0), ("k075", 0.75)):
        fr = frame if k == 1.0 else decay_frame(frame, dict.fromkeys(ALL_CAPS, k))
        Rk = np.column_stack([lab.ret(lab.cands[n]["w"], frame=fr).to_numpy() for n in tb])
        xs, cg = path_stats(lab, Rk, rf_arr)
        for j, n in enumerate(tb):
            believe.setdefault(n, {})[label] = {"xs_p5_50_95": [float(np.percentile(xs[:, j], q)) for q in (5, 50, 95)],
                                                "cagr_p5_50_95": [float(np.percentile(cg[:, j], q)) for q in (5, 50, 95)],
                                                "p_xs_below_1.0": float((xs[:, j] < 1.0).mean()), "p_xs_below_0.75": float((xs[:, j] < 0.75).mean()),
                                                "p_xs_below_0.5": float((xs[:, j] < 0.5).mean())}
    for n in tb:
        x = (lab.r(n) - rf.reindex(lab.r(n).index)).to_numpy()
        for N in (100, 1000):
            res = deflated_sharpe_ratio(x, None, float(N))
            believe[n][f"dsr_N{N}"] = {"dsr": float(res.deflated_sharpe_float), "benchmark_sharpe_annual": float(res.benchmark_sharpe_float * np.sqrt(252))}
    out["believe"] = believe

    # ── 5.9 dependence ──
    cf = pd.DataFrame({k: lab.r(f"capsule {k}") for k in ("TAA 3x", "TAA 3x 1N", "MOM", "MR", "DEF")})
    spx = data["bench"]["SPXTR"].reindex(cf.index)
    qqq = frame[g.QQQ].reindex(cf.index)
    wk = (1.0 + cf).resample("W-FRI").prod() - 1.0
    d21 = np.expm1(np.log1p(cf).rolling(21).sum()).dropna()
    import norgatedata  # noqa: PLC0415
    vix = norgatedata.price_timeseries("$VIX", start_date="1990-01-01", end_date=END.strftime("%Y-%m-%d"), timeseriesformat="pandas-dataframe")["Close"]
    vix.index = pd.to_datetime(vix.index).normalize()
    gate_prev = g.vix_gate.stress_gate_open_ser(vix).shift(1).reindex(cf.index)   # *** CRITICAL*** the state after close t-1 governs session t
    dep = {"full": cf.corr().round(4).to_dict(), "weekly": wk.corr().round(4).to_dict(), "d21": d21.corr().round(4).to_dict(),
           "blocks": {k: lib.window(cf, lo, hi).corr().round(4).to_dict() for k, (lo, hi) in BLOCK_DICT.items()},
           "h1": cf.loc[:g.CUT].corr().round(4).to_dict(), "h2": cf.loc[g.CUT + pd.Timedelta(days=1):].corr().round(4).to_dict(),
           "spx_worst5": cf[spx <= spx.quantile(0.05)].corr().round(4).to_dict(), "qqq_worst5": cf[qqq <= qqq.quantile(0.05)].corr().round(4).to_dict(),
           "crises": {k: cf.loc[lo:hi].corr().round(4).to_dict() for k, (lo, hi) in lib.CRISIS_DICT.items()},
           "gate_open": cf[gate_prev == True].corr().round(4).to_dict(), "gate_closed": cf[gate_prev == False].corr().round(4).to_dict(),  # noqa: E712
           "gate_open_share": float((gate_prev == True).mean()),  # noqa: E712
           "corr_with_spx": {k: float(cf[k].corr(spx)) for k in cf}, "corr_with_qqq": {k: float(cf[k].corr(qqq)) for k in cf},
           "vol": (cf.std() * np.sqrt(252)).round(4).to_dict()}
    pairs = [("TAA 3x", "MOM"), ("TAA 3x", "MR"), ("MOM", "MR"), ("TAA 3x 1N", "MOM"), ("TAA 3x 1N", "MR")]
    dep["rolling_252"] = {}
    for a_, b_ in pairs:
        rc = cf[a_].rolling(252).corr(cf[b_]).dropna()
        dep["rolling_252"][f"{a_} | {b_}"] = {"min": float(rc.min()), "p10": float(rc.quantile(0.1)), "median": float(rc.median()), "p90": float(rc.quantile(0.9)),
                                             "max": float(rc.max()), "series": [[d.strftime("%Y-%m"), round(float(v), 3)] for d, v in rc.resample("ME").last().items()]}
    dep["tail_dependence"] = {}
    for label, df in (("daily", cf), ("21d", d21)):
        dep["tail_dependence"][label] = {f"{a_} | {b_}": float(((df[a_] <= df[a_].quantile(0.05)) & (df[b_] <= df[b_].quantile(0.05))).sum()
                                                                / (df[a_] <= df[a_].quantile(0.05)).sum()) for a_, b_ in pairs}
    mon = (1.0 + cf).groupby([cf.index.year, cf.index.month]).prod() - 1.0
    dep["products"] = {}
    h1m, h2m = cf.index <= g.CUT, cf.index > g.CUT
    for n, (taa, wt, wm, wr) in g.PRODUCT_SHAPE.items():
        tcol = "TAA 3x" if taa == "taa3x" else "TAA 3x 1N"
        cols, wv = [tcol, "MOM", "MR"], np.array([wt, wm, wr])
        book = lab.r(n)
        wp: list = []
        g.book_returns(frame, g.PRODUCTS[n], lab.start, weight_path=wp)
        _, pcols, pw = wp[0]
        contrib = pd.DataFrame(pw * frame.loc[lab.start:END, pcols].to_numpy(), index=book.index, columns=pcols).T.groupby(lambda a: g.CAPSULE_OF[a]).sum().T
        xcontrib = pd.DataFrame(pw * (frame.loc[lab.start:END, pcols].sub(rf.loc[lab.start:END], axis=0)).to_numpy(), index=book.index,
                                columns=pcols).T.groupby(lambda a: g.CAPSULE_OF[a]).sum().T
        risk = {c: float(contrib[c].cov(book) / book.var()) for c in contrib}
        ret_share = {c: float(xcontrib[c].mean() / (book - rf.reindex(book.index)).mean()) for c in xcontrib}

        def div_ratio(df: pd.DataFrame) -> float:
            return float((df[cols].std().to_numpy() @ wv) / (df[cols].to_numpy() @ wv).std(ddof=1))

        def pred_vol(corr_src: pd.DataFrame, vol_src: pd.DataFrame) -> float:
            sd = vol_src[cols].std().to_numpy()
            cov = corr_src[cols].corr().to_numpy() * np.outer(sd, sd)
            return float(np.sqrt(wv @ cov @ wv) * np.sqrt(252))

        real_h1, real_h2 = float((cf.loc[h1m, cols].to_numpy() @ wv).std(ddof=1) * np.sqrt(252)), float((cf.loc[h2m, cols].to_numpy() @ wv).std(ddof=1) * np.sqrt(252))
        p_h2, p_h1 = pred_vol(cf.loc[h1m], cf.loc[h2m]), pred_vol(cf.loc[h2m], cf.loc[h1m])
        c1, c2 = cf.loc[h1m, cols].corr(), cf.loc[h2m, cols].corr()
        max_move = float((c1 - c2).abs().to_numpy().max())
        hot = {}
        for label, cm in ({**{f"block {k}": lib.window(cf, lo, hi) for k, (lo, hi) in BLOCK_DICT.items()},
                           **{f"crisis {k}": cf.loc[lo:hi] for k, (lo, hi) in lib.CRISIS_DICT.items()},
                           "spx worst 5%": cf[spx <= spx.quantile(0.05)], "qqq worst 5%": cf[qqq <= qqq.quantile(0.05)]}).items():
            cc = cm[cols].corr().to_numpy()
            off = cc[np.triu_indices(3, 1)]
            if off.max() > 0.70:
                hot[label] = float(off.max())
        allneg = mon[(mon[cols] < 0).all(axis=1)]
        bm = (1.0 + book).groupby([book.index.year, book.index.month]).prod() - 1.0
        b21 = np.expm1(np.log1p(book).rolling(21).sum()).dropna()
        worst, used = [], []
        for end_ts in b21.sort_values().index:
            pos = book.index.get_loc(end_ts)
            if any(abs(pos - u) < 21 for u in used):
                continue
            used.append(pos)
            lo = book.index[pos - 20]
            worst.append({"start": str(lo.date()), "end": str(end_ts.date()), "book": float(b21.loc[end_ts]),
                          **{c: float((1.0 + cf.loc[lo:end_ts, c]).prod() - 1.0) for c in cols}})
            if len(worst) == 10:
                break
        nav = (1.0 + book).cumprod()
        trough = (nav / nav.cummax() - 1.0).idxmin()
        peak = nav.loc[:trough].idxmax()
        dep["products"][n] = {
            "risk_share": risk, "cluster_risk_share": {"Nasdaq pair": risk.get("TAA", 0) + risk.get("MOM", 0), "MR": risk.get("MR", 0)},
            "excess_return_share": ret_share,
            "diversification_ratio": {"full": div_ratio(cf), "h1": div_ratio(cf.loc[h1m]), "h2": div_ratio(cf.loc[h2m]),
                                      **{k: div_ratio(lib.window(cf, lo, hi)) for k, (lo, hi) in BLOCK_DICT.items() if k in ("A", "B", "C")}},
            "vol_predicted_vs_realised": {"h2_from_h1_corr": [p_h2, real_h2], "h1_from_h2_corr": [p_h1, real_h1]},
            "flags": {"realised_above_predicted_15pct": bool(real_h2 > 1.15 * p_h2 or real_h1 > 1.15 * p_h1),
                      "corr_moved_more_than_0.20": bool(max_move > 0.20), "max_corr_move": max_move, "corr_above_0.70": hot},
            "enb": {"full": enb(cf[cols].corr().to_numpy()), "h1": enb(c1.to_numpy()), "h2": enb(c2.to_numpy()),
                    "spx_worst5": enb(cf.loc[spx <= spx.quantile(0.05), cols].corr().to_numpy())},
            "months_all_three_lost": {"count": int(len(allneg)), "of": int(len(mon)), "book_mean": float(bm.reindex(allneg.index).mean()) if len(allneg) else None,
                                      "book_worst": float(bm.reindex(allneg.index).min()) if len(allneg) else None},
            "ten_worst_21d": worst,
            "at_max_dd": {"peak": str(peak.date()), "trough": str(trough.date()), "book": float(nav.loc[trough] / nav.loc[peak] - 1.0),
                          "capsules": {c: float((1.0 + cf.loc[peak:trough, c].iloc[1:]).prod() - 1.0) for c in cols}}}
    out["dependence"] = dep

    # ── 5.10 factor alpha, gross and net ──
    index_all = data["index"]
    closes = pd.concat([lib.common.load_total_return_close_ser(s, "2005-01-01", END.strftime("%Y-%m-%d")) for s in afl.ETF_LIST], axis=1)
    closes.columns = afl.ETF_LIST
    closes = closes.reindex(index_all).loc["2005-01-01":]
    naive = afl.naive_rule_return_df(closes, data["sleeve"][TBILL])
    etf_r = closes.pct_change(fill_method=None)
    tb_w = afl.weekly_ser(data["sleeve"][TBILL])
    fac_w = pd.DataFrame({s: afl.weekly_ser(etf_r[s]) for s in afl.FACTOR_M1_LIST}).sub(tb_w, axis=0)
    trend_w = afl.weekly_ser(naive["NAIVE_QQQ_TREND200"]) - tb_w
    windows = {"long": (LONG_START, END), "h1": (LONG_START, g.CUT), "h2": (g.CUT + pd.Timedelta(days=1), END), "exact": (EXACT_START, END)}
    alpha = []
    for n in products + [g.S9, "capsule TAA 3x", "capsule TAA 3x 1N", "capsule MOM", "capsule MR"]:
        r = lab.r(n)
        for basis, ser in (("gross", r), ("net", ga.net_return_series(r))):
            yw = (afl.weekly_ser(ser) - tb_w).dropna().iloc[1:-1]
            for wn, (lo, hi) in windows.items():
                y = yw.loc[lo:hi]
                x1 = fac_w.reindex(y.index)
                for model, x in (("M0", x1[["QQQ"]]), ("M1", x1), ("M2", x1.assign(TREND200=trend_w.reindex(y.index)))):
                    alpha.append({"book": n, "basis": basis, "window": wn, "model": model, **afl.newey_west_ols(y, x, afl.NW_LAG_INT)})
    out["alpha"] = alpha
    out["naive"] = {k: g.stats(naive[k].loc[LONG_START:END].dropna(), rf) for k in ("NAIVE_QQQ", "NAIVE_60_40", "NAIVE_QQQ_TREND200")}

    # ── 5.13 after the window (not out of sample): 2026-08-20 .. latest, house cash ──
    rerun = g.STUDY / "audit" / "rerun_taa_def"
    late = {}
    for alias in ("taa3x", "taa3x_1n", "core5", "btal_qqq", "ndx_vxn"):
        late[alias] = lib.nav_to_returns(lib.read_path(rerun, alias))
    for alias in g.NEW_ALIAS_LIST:
        late[alias] = data["full"][alias]
    bil = lib.common.load_total_return_close_ser("BIL", "2025-06-01", "2026-10-02")
    late[TBILL] = bil.pct_change(fill_method=None)
    lf = pd.DataFrame(late).loc["2026-01-02":"2026-10-02"]
    after = {"window": ["2026-08-20", str(lf.index[-1].date())], "books": {}, "sleeves": {}}
    for n in products + [g.S9, "old growth plus"]:
        w = lab.cands[n]["w"]
        r = g.book_returns(lf, w, lf.index[0], end=lf.index[-1])
        a = r.loc["2026-08-20":]
        nav = np.r_[1.0, np.cumprod(1.0 + a.to_numpy())]
        after["books"][n] = {"after_window": float(nav[-1] - 1.0), "after_window_dd": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
                             "ytd_2026": float((1.0 + r).prod() - 1.0), "sessions": int(len(a))}
    for alias in lf.columns:
        after["sleeves"][alias] = float((1.0 + lf.loc["2026-08-20":, alias]).prod() - 1.0)
    out["after_window"] = after

    (g.OUT / "battery.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("battery_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
