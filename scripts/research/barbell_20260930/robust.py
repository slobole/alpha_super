"""SPEC 11-12 (reported, never selects): robustness battery for the champions and the picks of the main frame.

Reset-date phases, forward split, year-drop jackknife, PBO/CSCV and fixed-book ranks, White reality check, factor
alpha (gross and net), crises and co-falls, crisis correlation of the sleeves, rolling beta to QQQ, horizon and
first-year risk, rolling entries (also when NDX-VXN is under water at entry), rebalancing policies, drift cohorts,
fee structures and manager income, the interim (NOW) cost of delay, years and NAV for the report.

Usage: python robust.py   (after run_select.py main and main --tag _mainline)
"""

from __future__ import annotations

from itertools import combinations
import json

import numpy as np
import pandas as pd

import bb_lib as bb
from bb_lib import EXACT_START, LONG_START, TBILL, ga, lib
import allocator_first_look as afl  # noqa: E402  (fund menu folder is on sys.path via lib)
import part_b

OUT = bb.STUDY / "robust"
KEYS_LT = ["LT-FUNDED|GROSS|RM", "LT-FUNDED|NET|RM"] + [f"LT-FUNDED|{m}|B{b}" for m in ("GROSS", "NET") for b in (100, 125, 150, 175, 200)]


def key_books() -> dict[str, tuple[str, str]]:
    sel = json.loads((bb.STUDY / "main" / "selection.json").read_text(encoding="utf-8"))["results"]
    main = json.loads((bb.STUDY / "main_mainline" / "selection.json").read_text(encoding="utf-8"))["results"]
    books = {"CH": bb.CH, "CH-G": bb.CH_G}
    for k in KEYS_LT + ["TARGET|GROSS|RM", "NOW|GROSS|RM", "LT-FUNDED-UNCAPPED|GROSS|RM"]:
        e = sel.get(k, {})
        for role in ("pick", "product"):
            if e.get(role):
                r, d = e[role].split(" || ")
                books.setdefault(f"{k}|{role}", (r, d))
    for k in ("MAIN-FUNDED|GROSS|RM", "MAIN-FUNDED|NET|RM"):
        e = main.get(k, {})
        if e.get("pick"):
            r, d = e["pick"].split(" || ")
            books.setdefault(f"{k}|pick", (r, d))
    return books


def patch_period_ids() -> None:
    """Allow Book.policy 'annual_mK': pod resets after the last close of calendar month K (reset-date phase test)."""
    base = lib.period_ids

    def period_ids(index, policy):
        if isinstance(policy, str) and policy.startswith("annual_m"):
            return bb.period_labels(index, "annual", int(policy[8:]))
        return base(index, policy)

    lib.period_ids = period_ids


def reality_check(R: np.ndarray, ch: np.ndarray, reps: int = 1000, seed: int = 20260929) -> dict:
    """White (2000) reality check: H0 no book beats CH in mean daily log return; stationary bootstrap, block 63."""
    d = np.log1p(R) - np.log1p(ch)[:, None]
    n = len(d)
    mean = d.mean(axis=0)
    stat = np.sqrt(n) * mean.max()
    idx = lib.evaluation.stationary_bootstrap_index_mat(n, reps, 63.0, seed)
    centred = d - mean
    boot = np.array([np.sqrt(n) * centred[i].mean(axis=0).max() for i in idx])
    return {"stat": float(stat), "p_value": float(np.mean(boot >= stat)), "books": int(R.shape[1]),
            "best_mean_ann": float(mean.max() * 252)}


def cscv(logr: np.ndarray, fixed: dict[str, int], blocks: int = 16) -> dict:
    n = logr.shape[0] - logr.shape[0] % blocks
    parts = np.array_split(np.arange(n), blocks)
    bs = np.array([logr[p].sum(axis=0) for p in parts])
    bl = np.array([len(p) for p in parts])
    logits, fr = [], {k: [] for k in fixed}
    for combo in combinations(range(blocks), blocks // 2):
        m = np.zeros(blocks, dtype=bool)
        m[list(combo)] = True
        is_v, oos = bs[m].sum(0) / bl[m].sum(), bs[~m].sum(0) / bl[~m].sum()
        best = int(np.argmax(is_v))
        rank = (np.sum(oos < oos[best]) + 0.5 * (np.sum(oos == oos[best]) - 1) + 1) / (len(oos) + 1)
        logits.append(np.log(rank / (1 - rank)))
        for k, j in fixed.items():
            fr[k].append((np.sum(oos < oos[j]) + 0.5 * (np.sum(oos == oos[j]) - 1) + 1) / (len(oos) + 1))
    out = {"pbo_argmax_rule": float(np.mean(np.array(logits) <= 0)), "books": int(logr.shape[1])}
    for k, v in fr.items():
        out[k] = {"median_oos_rank": float(np.median(v)), "share_top_half": float(np.mean(np.array(v) > 0.5))}
    return out


def fees_paid(r: np.ndarray, index: pd.DatetimeIndex, mgmt: float, perf: float) -> float:
    """Fees actually paid (management + crystallised performance fees) as a share of average net NAV per year."""
    years = index.year.to_numpy()
    pre, hwm, paid_total, navs = 1.0, 1.0, 0.0, []
    for t, g in enumerate(r):
        pre *= 1.0 + g
        fee = pre * mgmt / 252.0
        pre -= fee
        paid_total += fee
        acc = perf * max(pre - hwm, 0.0)
        navs.append(pre - acc)
        if t == len(r) - 1 or years[t + 1] != years[t]:
            pre -= acc
            paid_total += acc
            if acc > 0:
                hwm = pre
    return paid_total / float(np.mean(navs)) / (len(r) / 252.0)


def dd_of(r: np.ndarray) -> float:
    v = np.r_[1.0, np.cumprod(1.0 + r)]
    return float((v / np.maximum.accumulate(v) - 1.0).min())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    patch_period_ids()
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    index_all = data["index"]
    sl = {s.name: s for s in bb.risky_sleeves()}
    books = key_books()
    uniq = sorted(set(books.values()))
    RA = pd.DataFrame({r: lib.book_returns(frame, sl[r].book(), start) for r in sorted({u[0] for u in uniq})})
    RD = pd.DataFrame({d: lib.book_returns(frame, bb.def_book(d), start) for d in sorted({u[1] for u in uniq})})
    index = RA.index
    periods = bb.period_labels(index, "annual")
    A = {u: bb.account_returns(RA[u[0]].to_numpy(), RD[u[1]].to_numpy(), bb.S_CLIENT, periods)[:, 0] for u in uniq}
    name = {u: bb.client_name(*u) for u in uniq}
    res: dict = {"books": {k: name[v] for k, v in books.items()}}

    # (a) reset-date phases, pod and sleeve level.
    phase_rows = []
    for u in uniq:
        for ph in range(12):
            pol = "annual" if ph == 0 else f"annual_m{ph}"
            rb = sl[u[0]].book()
            rb = bb.Book(rb.name, rb.pods, rb.rule, rb.weights, pol, "R")
            db = bb.def_book(u[1])
            db = bb.Book(db.name, db.pods, db.rule, db.weights, pol, "D")
            ra = lib.book_returns(frame, rb, start).to_numpy()
            rd = lib.book_returns(frame, db, start).to_numpy()
            acc = bb.account_returns(ra, rd, bb.S_CLIENT, bb.period_labels(index, "annual", ph))[:, 0]
            st = ga.window_stats(pd.DataFrame(acc, index=index), index_all)
            phase_rows.append({"book": name[u], "phase_month": ph if ph else 12, "gross_cagr": float(st["gross_cagr"][0]),
                               "net_cagr": float(st["net_cagr"][0]), "gross_dd": float(st["gross_dd"][0])})
    P = pd.DataFrame(phase_rows)
    P.to_csv(OUT / "phases.csv", index=False, float_format="%.6g")
    ch_ph = P[P["book"] == name[bb.CH]].set_index("phase_month")
    res["phases"] = {name[u]: {"gross_cagr_min": float(P[P.book == name[u]].gross_cagr.min()),
                               "gross_cagr_max": float(P[P.book == name[u]].gross_cagr.max()),
                               "gross_dd_min": float(P[P.book == name[u]].gross_dd.min()),
                               "beats_ch_phases": int((P[P.book == name[u]].set_index("phase_month").gross_cagr > ch_ph.gross_cagr).sum())}
                     for u in uniq}

    # (b)-(e) need the passers of the primary rung (all client books of the main frame).
    T = pd.read_csv(bb.STUDY / "main" / "books.csv", index_col=0)
    AR = pd.read_csv(bb.STUDY / "main" / "account_returns.csv.gz", index_col=0, parse_dates=True)
    sel = json.loads((bb.STUDY / "main" / "selection.json").read_text(encoding="utf-8"))["results"]
    ch_n = name[bb.CH]
    for key in ("LT-FUNDED|GROSS|RM", "LT-FUNDED|GROSS|B150"):
        e = sel[key]
        # Rebuild the passer set with the same rules as run_select (G2-G4 + rung, gross).
        F = T[T.index.str.split(" \\|\\| ").str[1].isin(bb.FUNDED) & (T["risky_line"] == "LT")]
        ok = F["g2_gross"].astype(bool) & F["g3_gross"].astype(bool) & F["g4"].astype(bool)
        ch = T.loc[ch_n]
        if key.endswith("RM"):
            ok &= (F["gross_dd"] >= ch["gross_dd"] - 1e-12) & (F["gross_dd_exact"] >= ch["gross_dd_exact"] - 1e-12) \
                & (F["ddar10_gross"] >= ch["ddar10_gross"] - 1e-12)
        else:
            ok &= (F["gross_dd"] >= -0.15 + 0.03) & (F["p_B150_gross"] <= bb.MAX_BREACH_ABS)
        passers = list(F.index[ok])
        if not passers:
            continue
        cols = passers + [c for c in (ch_n, e.get("pick")) if c and c not in passers]
        fixed = {k: passers.index(v) for k, v in (("CH", ch_n), ("pick", e.get("pick"))) if v in passers}
        res[f"pbo|{key}"] = cscv(np.log1p(AR[passers].to_numpy()), fixed) if len(passers) > 1 else {}
        res[f"reality_check|{key}"] = reality_check(AR[passers].to_numpy(), AR[ch_n].to_numpy())
        # Forward split: select by historical CAGR on H1 among books with H1 hist DD >= CH's H1 DD; evaluate on H2.
        cut = pd.Timestamp("2017-06-30")
        h1, h2 = AR.loc[:cut, cols], AR.loc[cut + pd.Timedelta(days=1):, cols]
        g = lambda df: (1 + df).prod() ** (252 / len(df)) - 1  # noqa: E731
        ddh = lambda df: df.apply(lambda c: dd_of(c.to_numpy()))  # noqa: E731
        fw = {}
        for a_, b_, lab in ((h1, h2, "select_H1_eval_H2"), (h2, h1, "select_H2_eval_H1")):
            ga_, gb_ = g(a_), g(b_)
            elig = ddh(a_) >= dd_of(a_[ch_n].to_numpy()) - 1e-12
            best = ga_[elig].idxmax()
            fw[lab] = {"best_in_sample": best, "its_oos_cagr": float(gb_[best]), "ch_oos_cagr": float(gb_[ch_n]),
                       "best_oos_rank_pct": float((gb_ < gb_[best]).mean()),
                       "pick_oos_cagr": float(gb_[e["pick"]]) if e.get("pick") else None}
        res[f"forward|{key}"] = fw
        # Year-drop jackknife: does the pick still beat CH with each calendar year removed?
        if e.get("pick") and e["pick"] != ch_n:
            yrs = sorted(set(AR.index.year))
            wins = 0
            for y in yrs:
                m = AR.index.year != y
                pk, cc = AR.loc[m, e["pick"]], AR.loc[m, ch_n]
                wins += int(np.prod(1 + pk) > np.prod(1 + cc))
            res[f"jackknife|{key}"] = {"years": len(yrs), "pick_beats_ch_years": wins}

    # (f) factor alpha, gross and net.
    closes = pd.concat([lib.common.load_total_return_close_ser(s, "2005-01-01", bb.END.strftime("%Y-%m-%d")) for s in afl.ETF_LIST], axis=1)
    closes.columns = afl.ETF_LIST
    closes = closes.reindex(index_all).loc["2005-01-01":]
    tbill = data["sleeve"][TBILL]
    naive = afl.naive_rule_return_df(closes, tbill)
    etf_r = closes.pct_change(fill_method=None)
    tb_w = afl.weekly_ser(tbill)
    fac_w = pd.DataFrame({s: afl.weekly_ser(etf_r[s]) for s in afl.FACTOR_M1_LIST}).sub(tb_w, axis=0)
    trend_w = afl.weekly_ser(naive["NAIVE_QQQ_TREND200"]) - tb_w
    alpha_rows = []
    for u in uniq:
        r = pd.Series(A[u], index=index)
        for basis, ser in (("gross", r), ("net", ga.net_return_series(r))):
            ex = ser.loc[EXACT_START:]
            halves = np.array_split(ex.index, 2)
            for window, s_ in (("long", ser), ("exact", ex), ("exact_h1", ex.loc[halves[0]]), ("exact_h2", ex.loc[halves[1]])):
                y = (afl.weekly_ser(s_) - tb_w).dropna().iloc[1:-1]
                x1 = fac_w.reindex(y.index)
                for model, x in (("M0", x1[["QQQ"]]), ("M1", x1), ("M2", x1.assign(TREND200=trend_w.reindex(y.index)))):
                    alpha_rows.append({"book": name[u], "basis": basis, "window": window, "model": model,
                                       **afl.newey_west_ols(y, x, afl.NW_LAG_INT)})
    pd.DataFrame(alpha_rows).to_csv(OUT / "alpha.csv", index=False, float_format="%.6g")

    # (g)-(h) crises, co-falls, crisis correlation of the sleeves, rolling beta to QQQ.
    spx = data["bench"]["SPXTR"].reindex(index)
    worst = spx <= spx.quantile(0.05)
    qqq = etf_r["QQQ"].reindex(index)
    crisis, beta = {}, {}
    for u in uniq:
        r = pd.Series(A[u], index=index)
        c = {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()}
        for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, bb.END):
            c[f"cofall|{lo.date()}|{hi.date()}|{ret:.4f}"] = lib.common.window_return_float(r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
        for ep in lib.common.equity_drawdown_episode_list(data["bench"]["SPXTR"].loc[LONG_START:bb.END], -0.10):
            c[f"spx10|{ep['peak_date_str']}|{ep['trough_date_str']}|{ep['spx_drawdown_float']:.4f}"] = \
                lib.common.window_return_float(r, ep["peak_date_str"], ep["trough_date_str"])
        c["corr_R_D_worst5pct_spx_days"] = float(RA[u[0]][worst].corr(RD[u[1]][worst]))
        c["corr_R_D_all_days"] = float(RA[u[0]].corr(RD[u[1]]))
        crisis[name[u]] = c
        cov = r.rolling(63).cov(qqq)
        b63 = cov / qqq.rolling(63).var()
        beta[name[u]] = {"beta63_max": float(b63.max()), "beta63_p99": float(b63.quantile(0.99)), "beta63_median": float(b63.median())}
    res["crises"], res["beta_qqq"] = crisis, beta

    # (i) horizon breach and first-year risk (joint bootstrap, 10 seeds, truncated paths).
    RAv, RDv = RA.to_numpy(), RD.to_numpy()
    iA = np.array([list(RA.columns).index(u[0]) for u in uniq])
    jD = np.array([list(RD.columns).index(u[1]) for u in uniq])
    hz = {}
    for h, lab in ((252, "1y"), (756, "3y"), (1260, "5y")):
        acc = {}
        for k, seed in enumerate(bb.SEEDS):
            idx = bb.boot_index(len(index), seed)[:, :h]
            b = bb.boot_accounts(RAv, RDv, iA, jD, bb.S_CLIENT, idx, fy=h, dtype=np.float64 if k == 0 else np.float32)
            for B in (-0.05, -0.10, -0.15):
                acc.setdefault(f"p_dd_below_{int(-B * 100)}", []).append((b["gross_dd"] < B).mean(axis=0))
            acc.setdefault("p_loss", []).append((b["gross_cagr"] < 0).mean(axis=0))
            acc.setdefault("cagr_p10", []).append(np.percentile(b["gross_cagr"], 10, axis=0))
        hz[lab] = {name[u]: {k: float(np.mean([x[i] for x in v])) for k, v in acc.items()} for i, u in enumerate(uniq)}
    res["horizons"] = hz

    # (j) rolling month-end entries (P0 from the entry date), also when NDX-VXN is under water by > 10% at entry.
    me = index[np.r_[index.month[1:] != index.month[:-1], True]]
    vxn = frame["ndx_vxn"].reindex(index)
    vxn_nav = (1 + vxn).cumprod()
    vxn_dd = vxn_nav / vxn_nav.cummax() - 1
    ent = {}
    for u in uniq:
        rows = []
        for e0 in me:
            pos = index.get_loc(e0) + 1
            for h, lab in ((252, "1y"), (756, "3y"), (1260, "5y")):
                if pos + h > len(index):
                    continue
                sub = index[pos:pos + h]
                acc = bb.account_returns(RA[u[0]].to_numpy()[pos:pos + h], RD[u[1]].to_numpy()[pos:pos + h], bb.S_CLIENT,
                                         bb.period_labels(sub, "annual"))[:, 0]
                rows.append({"entry": e0, "h": lab, "ret_ann": float(np.prod(1 + acc) ** (252 / h) - 1), "dd": dd_of(acc),
                             "vxn_under": bool(vxn_dd.loc[e0] < -0.10)})
        E = pd.DataFrame(rows)
        ent[name[u]] = {lab: {"n": int((E.h == lab).sum()), "ret_p10": float(E[E.h == lab].ret_ann.quantile(0.1)),
                              "ret_med": float(E[E.h == lab].ret_ann.median()), "p_loss": float((E[E.h == lab].ret_ann < 0).mean()),
                              "dd_p10": float(E[E.h == lab].dd.quantile(0.1)),
                              "n_vxn_under": int(((E.h == lab) & E.vxn_under).sum()),
                              "ret_med_vxn_under": float(E[(E.h == lab) & E.vxn_under].ret_ann.median()) if ((E.h == lab) & E.vxn_under).any() else None}
                         for lab in ("1y", "3y", "5y")}
    res["entries"] = ent

    # (k) rebalancing policies and drift cohorts.
    pol = {}
    for u in uniq:
        out = {}
        for lab, pl in (("P0 annual", bb.period_labels(index, "annual")), ("P3 quarterly", bb.period_labels(index, "quarterly"))):
            acc = bb.account_returns(RA[u[0]].to_numpy(), RD[u[1]].to_numpy(), bb.S_CLIENT, pl)[:, 0]
            out[lab] = {"gross_cagr": float(np.prod(1 + acc) ** (252 / len(acc)) - 1), "gross_dd": dd_of(acc)}
        for band in (0.05, 0.10):
            vR, vD, prev, rr = bb.S_CLIENT, 1 - bb.S_CLIENT, 1.0, []
            for t in range(len(index)):
                vR *= 1 + RA[u[0]].iat[t]
                vD *= 1 + RD[u[1]].iat[t]
                v = vR + vD
                rr.append(v / prev - 1)
                prev = v
                if index[t] in me and abs(vR / v - bb.S_CLIENT) > band:
                    moved = abs(vR - bb.S_CLIENT * v)
                    va = v - 2 * bb.TRANSFER_BPS * moved
                    vR, vD = bb.S_CLIENT * va, (1 - bb.S_CLIENT) * va
            rr = np.array(rr)
            out[f"P2 band {int(band * 100)}pp"] = {"gross_cagr": float(np.prod(1 + rr) ** (252 / len(rr)) - 1), "gross_dd": dd_of(rr)}
        shares = []
        for e0 in me:
            pos = index.get_loc(e0) + 1
            for h, lab in ((756, "3y"), (1260, "5y")):
                if pos + h > len(index):
                    continue
                gA = np.prod(1 + RA[u[0]].to_numpy()[pos:pos + h])
                gD = np.prod(1 + RD[u[1]].to_numpy()[pos:pos + h])
                shares.append({"h": lab, "share_end": bb.S_CLIENT * gA / (bb.S_CLIENT * gA + (1 - bb.S_CLIENT) * gD)})
        S = pd.DataFrame(shares)
        out["P1 drift share at horizon end"] = {lab: {"median": float(S[S.h == lab].share_end.median()),
                                                      "p90": float(S[S.h == lab].share_end.quantile(0.9))} for lab in ("3y", "5y")}
        pol[name[u]] = out
    res["policies"] = pol

    # (l) fees and manager income on USD 1.5M.
    ye = ga.year_end_flags(index)
    fees = {}
    for u in uniq:
        r = A[u]
        days = (index[-1] - lib.base_date(index_all, pd.Series(r, index=index))).days
        out = {}
        for lab, mg, pf in (("none", 0.0, 0.0), ("F1 2/20 total", 0.02, 0.20), ("F3 1/10 total", 0.01, 0.10)):
            nav = part_b.fee_nav_params(r[:, None], ye, mg, pf)[:, 0]
            out[lab] = {"net_cagr": float(nav[-1] ** (365.25 / days) - 1),
                        "fee_pct_aum_yr": fees_paid(r, index, mg, pf) if mg else 0.0}
        rA_net = ga.net_return_series(RA[u[0]]).to_numpy()
        rD_net = ga.net_return_series(RD[u[1]]).to_numpy()
        f2 = bb.account_returns(rA_net, rD_net, bb.S_CLIENT, periods)[:, 0]
        out["F2 2/20 per sleeve"] = {"net_cagr": float(np.prod(1 + f2) ** (365.25 / days) - 1)}
        out["netting_cost_F1_minus_F2"] = out["F1 2/20 total"]["net_cagr"] - out["F2 2/20 per sleeve"]["net_cagr"]
        out["income_usd_1p5m_F1"] = out["F1 2/20 total"]["fee_pct_aum_yr"] * 1.5e6
        out["income_usd_1p5m_F3"] = out["F3 1/10 total"]["fee_pct_aum_yr"] * 1.5e6
        fees[name[u]] = out
    res["fees"] = fees

    # (m) years, NAV and summary stats for the report.
    rep = {}
    for u in uniq:
        r = pd.Series(A[u], index=index)
        fb = ga.fee_breakdown(r)
        nav_g, nav_n = (1 + r).cumprod(), (1 + ga.net_return_series(r)).cumprod()
        rep[name[u]] = {"years": [{"year": int(x.year), "gross": float(x.gross), "net": float(x.net)} for x in fb.itertuples()],
                        "nav_gross": [[d.strftime("%Y-%m"), float(v)] for d, v in nav_g.resample("ME").last().items()],
                        "nav_net": [[d.strftime("%Y-%m"), float(v)] for d, v in nav_n.resample("ME").last().items()],
                        "full": {k: (float(v) if not isinstance(v, str) else v) for k, v in lib.full_metrics(r, data, "long").items()}}
    for lab, col in (("S&P 500 TR", "SPXTR"), ("60/40", "SIXTY_FORTY")):
        r = data["bench"][col].loc[LONG_START:bb.END]
        g = (1 + r).groupby(r.index.year).prod() - 1
        rep[lab] = {"years": [{"year": int(y), "gross": float(v)} for y, v in g.items()],
                    "nav_gross": [[d.strftime("%Y-%m"), float(v)] for d, v in (1 + r).cumprod().resample("ME").last().items()],
                    "full": {k: (float(v) if not isinstance(v, str) else v) for k, v in lib.full_metrics(r, data, "long").items()}}
    res["report"] = rep
    (OUT / "robust.json").write_text(json.dumps(res, default=str), encoding="utf-8")
    bb.ledger("robust_finished", books=list(res["books"].values()))
    print(json.dumps({k: v for k, v in res.items() if k not in ("report", "crises", "entries", "policies", "horizons")}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
