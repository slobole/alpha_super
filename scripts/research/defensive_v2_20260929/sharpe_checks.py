"""Amendment A2: robustness checks for the A1 (LONG Sharpe) selection of defensive shelf v2.

Reported beside the A1 picks; nothing here re-selects the recommendation automatically.
  1. House cash (idle cash 0%): the full rule on the house-cash versions of all five frames, both lines.
  2. Excess Sharpe (Sharpe of book - BIL) as the objective, under fair and house cash.
  3. Sharpe picks under each fair sensitivity frame; CSCV (PBO and fixed-book OOS rank); weight plateau;
     sub-periods, calendar years, crises and stock-bond co-falls; tie-band composition; ease and capacity.

Usage: python sharpe_checks.py
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research" / "shelf_rebuild_20260929"))
sys.path.insert(0, str(HERE))

import lib  # noqa: E402
from lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Book  # noqa: E402
import defensive_v2 as dv  # noqa: E402

OUT = dv.OUT / "sharpe_checks"
TIE = ["pods", "not_live_share", "p_breach10", "trade_days_per_year"]
CHAMPION = dv.CHAMPION
A1_PICK = "CORE5 + EOM + DISP [IV]"
A1_TOP = "CORE5 + EOM + DV2-IND [EQ] + G3 10%"
A1_NOEOM = "BTAL_QQQ + DV2-IND [EQ]"
DPRIME = "CORE5 + BTAL_QQQ [IV]"
BENCH = "CORE5 [EQ]"
RUNNER_UP = "CORE5 + BTAL_QQQ + EOM [IV]"  # A3: the review's runner-up under the SPEC-literal tie-break reading
KEY_BOOKS = [A1_PICK, A1_TOP, A1_NOEOM, CHAMPION, DPRIME, BENCH, RUNNER_UP]
SLEEVES = ["core5", "btal_qqq", "eom_flow", "disp", "etf_dv2", "downshock", "tactical_fi", "trinity"]
WINDOWS = {"LONG": (LONG_START, END), **BLOCK_DICT, "EXACT": (EXACT_START, END)}
ANN = np.sqrt(252.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── objectives ──────────────────────────────────────────────────────────────


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * ANN)


def xs_sharpe(r: pd.Series, tb: pd.Series) -> float:
    x = r - tb.reindex(r.index)
    return float(x.mean() / x.std() * ANN)


def boot_sharpe_dd(R: np.ndarray, tb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per bootstrap path (same paths as the study): zero-rate Sharpe, excess Sharpe and max drawdown per column."""
    idx = lib.boot_index(R.shape[0])
    X = R - tb[:, None]
    sh = np.empty((idx.shape[0], R.shape[1]))
    xs = np.empty_like(sh)
    dd = np.empty_like(sh)
    for k in range(idx.shape[0]):
        s, x = R[idx[k]], X[idx[k]]
        sh[k] = s.mean(axis=0) / s.std(axis=0, ddof=1) * ANN
        xs[k] = x.mean(axis=0) / x.std(axis=0, ddof=1) * ANN
        dd[k] = lib.path_stats(s)[1]
    return sh, xs, dd


def select(t: pd.DataFrame, obj: str, names: list[str], boot: np.ndarray, pool: list[str]) -> dict:
    """SPEC tie band + tie-break + champion rule, with `obj` (a column of t) and its bootstrap matrix."""
    if not pool:
        return {"pool": 0}
    top = t.loc[pool, obj].idxmax()
    jt = names.index(top)
    share = {b: float(np.mean(boot[:, jt] > boot[:, names.index(b)])) for b in pool}
    share[top] = 0.0
    band = t.loc[[b for b in pool if share[b] < lib.TIE_SHARE]].copy()
    band = band.sort_values(TIE + [obj], ascending=[True, True, True, True, False])
    pick = band.index[0]
    beats = float(np.mean(boot[:, names.index(pick)] > boot[:, names.index(CHAMPION)]))
    holds = not (beats >= 0.80 and t.at[pick, "p_breach10"] <= t.at[CHAMPION, "p_breach10"] + 0.02)
    return {"pool": len(pool), "top": top, "pick": pick, "band": list(band.index), "pick_beats_champion": beats,
            "champion_holds": holds, "recommendation": CHAMPION if holds else pick,
            "pick_value": float(t.at[pick, obj]), "champion_value": float(t.at[CHAMPION, obj])}


def pools(t: pd.DataFrame) -> dict[str, list[str]]:
    passers = t.index[t["gates_pass"].astype(bool)]
    return {"all": list(passers), "no_eom": [b for b in passers if "eom_flow" not in str(t.at[b, "pods_list"])]}


# ─── check 1: house cash, full rule ──────────────────────────────────────────


def house_frames(data: dict) -> dict[str, pd.DataFrame]:
    swap_tfi = data["long"].copy()
    swap_tfi["tactical_fi"] = data["tfi_frozen"]
    return {"main": data["long"], "proxy_unscaled": data["long_unscaled"], "plus_5bps": data["stressed_long"],
            "tfi_frozen": swap_tfi, "etf_idle_2008_09": data["long_etf_cash"]}


def house_rule(line: str, data: dict, fair_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    idx = data["index"]
    frames = house_frames(data)
    books = dv.family(line)
    rows, series = [], {}
    for i, b in enumerate(books):
        g = dv.gate_row(frames["main"], b, idx)
        series[b.name] = g.pop("_r")
        row = {"book": b.name, "pods": b.pod_count, "pods_list": "+".join(b.pods), **g}
        for label, frame in frames.items():
            if label == "main":
                continue
            h = dv.gate_row(frame, b, idx)
            h.pop("_r")
            row[f"r5_{label}"] = (h["maxdd"] >= dv.DD_HIST and all(h[f"xs_{k}"] > 0 for k in BLOCK_DICT)
                                  and h["slot_pass"])
        rows.append(row)
        if i % 100 == 0:
            log(f"house {line}: {i}/{len(books)}")
    t = pd.DataFrame(rows).set_index("book")
    t["r1"] = t["maxdd"] >= dv.DD_HIST
    t["r3"] = (t[[f"xs_{k}" for k in BLOCK_DICT]] > 0).all(axis=1)
    t["r4"] = t["slot_pass"]
    t["r5"] = t[[c for c in t.columns if c.startswith("r5_")]].all(axis=1)
    t["not_live_share"] = fair_table["not_live_share"].reindex(t.index)
    t["trade_days_per_year"] = fair_table["trade_days_per_year"].reindex(t.index)
    R = pd.DataFrame(series)
    tb = frames["main"][TBILL].reindex(R.index)
    t["sharpe"] = [sharpe(R[b]) for b in t.index]
    t["xs_sharpe"] = [xs_sharpe(R[b], tb) for b in t.index]
    # R2 needs the bootstrap; run it only for books that pass every other gate (+ the champion and key books).
    cand = list(t.index[t[["r1", "r3", "r4", "r5"]].all(axis=1)])
    cols = list(dict.fromkeys(cand + [b for b in KEY_BOOKS if b in t.index]))
    sh, xs, dd = boot_sharpe_dd(R[cols].to_numpy(), tb.to_numpy())
    t["p_breach10"] = pd.Series((dd < dv.DD_LIMIT).mean(axis=0), index=cols)
    t["r2"] = t["p_breach10"] <= dv.BREACH_MAX
    t["gates_pass"] = t[["r1", "r2", "r3", "r4", "r5"]].fillna(False).all(axis=1)
    # A3 (review L5): R2 is counted only among the bootstrapped books; the rest never reached R2.
    fails = {g: int((~t[g].fillna(False).astype(bool)).sum()) for g in ["r1", "r3", "r4", "r5"]}
    fails["r2_among_bootstrapped"] = int((t.loc[cols, "p_breach10"] > dv.BREACH_MAX).sum())
    fails["bootstrapped"] = len(cols)
    out = {"gate_passers": int(t["gates_pass"].sum()), "fail_counts": fails}
    if CHAMPION in cols:
        for obj, boot in (("sharpe", sh), ("xs_sharpe", xs)):
            out[obj] = {k: select(t, obj, cols, boot, v) for k, v in pools(t).items()}
    return t, R, out


# ─── check 3b: CSCV on Sharpe ────────────────────────────────────────────────


def cscv_sharpe(X: np.ndarray, picks: dict[str, int], blocks: int = 16) -> dict:
    n = X.shape[0] - X.shape[0] % blocks
    X = X[:n]
    parts = np.array_split(np.arange(n), blocks)
    s1 = np.array([X[p].sum(axis=0) for p in parts])
    s2 = np.array([(X[p] ** 2).sum(axis=0) for p in parts])
    cnt = np.array([len(p) for p in parts], dtype=float)

    def sh(sel: list[int]) -> np.ndarray:
        m = cnt[sel].sum()
        mu = s1[sel].sum(axis=0) / m
        var = (s2[sel].sum(axis=0) - m * mu * mu) / (m - 1)
        return mu / np.sqrt(var)

    def omega(v: np.ndarray, j: int) -> float:
        below = np.sum(v < v[j]) + 0.5 * (np.sum(v == v[j]) - 1)
        return (below + 1.0) / (X.shape[1] + 1)

    logits, om = [], {k: [] for k in picks}
    for combo in combinations(range(blocks), blocks // 2):
        ins = list(combo)
        oos = [i for i in range(blocks) if i not in combo]
        v_in, v_out = sh(ins), sh(oos)
        w = omega(v_out, int(np.argmax(v_in)))
        logits.append(np.log(w / (1 - w)))
        for k, j in picks.items():
            om[k].append(omega(v_out, j))
    return {"pbo_argmax": float(np.mean(np.array(logits) <= 0)), "splits": len(logits), "books": int(X.shape[1]),
            "picks": {k: {"median_oos_rank": float(np.median(v)), "share_below_median": float(np.mean(np.array(v) <= 0.5))}
                      for k, v in om.items()}}


# ─── check 3c: weight plateau ────────────────────────────────────────────────


def plateau_books(base: dv.DefBook, weights: dict[str, float]) -> list[dv.DefBook]:
    out = [dv.DefBook(f"{base.name} | fixed avg", base.pods, "FIXED", dict(weights), base.slice_)]
    for p in base.pods:
        for d in (-0.10, 0.10):
            new = min(max(weights[p] + d, 0.0), 1.0)
            rest = 1.0 - weights[p]
            w = {q: (new if q == p else weights[q] * (1.0 - new) / rest) for q in base.pods}
            out.append(dv.DefBook(f"{base.name} | {dv.LABEL[p]} {d * 100:+.0f}", base.pods, "FIXED", w, base.slice_))
    return out


# ─── metrics ─────────────────────────────────────────────────────────────────


def window_metrics(r: pd.Series, tb: pd.Series, idx) -> dict:
    out = {}
    for w, (lo, hi) in WINDOWS.items():
        x = lib.window(r, lo, hi)
        t = tb.reindex(x.index)
        out[w] = {"cagr": lib.cagr(x, lib.base_date(idx, x)), "xs_cagr": lib.excess_cagr(x, tb, idx),
                  "sharpe": sharpe(x), "xs_sharpe": xs_sharpe(x, t), "maxdd": lib.maxdd(x),
                  "vol": float(x.std() * ANN)}
    return out


def calendar_excess(r: pd.Series, tb: pd.Series) -> dict:
    y = (1 + r).groupby(r.index.year).prod() - 1
    b = (1 + tb.reindex(r.index)).groupby(r.index.year).prod() - 1
    return {int(k): float(v) for k, v in (y - b).items()}


def crisis_rows(r: pd.Series, data: dict, cofalls: list) -> dict:
    out = {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()}
    for i, (lo, hi, _) in enumerate(cofalls):
        x = r.loc[lo:hi].iloc[1:]
        out[f"cofall_{i + 1}_{hi.date()}"] = float((1 + x).prod() - 1)
    out["crisis_corr_spx"] = lib.crisis_corr(r, data["bench"]["SPXTR"])
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    out["worst_month"] = float(m.min())
    return out


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("loading inputs")
    data = lib.load_inputs()
    idx = data["index"]
    fair_main = data["cash_long"]
    tb = fair_main[TBILL]
    result: dict = {}

    # fair tables and returns from the study's own run (A1 tables carry the Sharpe column)
    fair_t, fair_R = {}, {}
    for line in ("low", "main"):
        fair_t[line] = pd.read_csv(dv.OUT / f"{line}_books_sharpe.csv", index_col=0)
        fair_R[line] = pd.read_csv(dv.OUT / f"{line}_long_returns.csv.gz", index_col=0, parse_dates=True)
    t, R = fair_t["main"], fair_R["main"]
    names = list(R.columns)
    t["xs_sharpe"] = [xs_sharpe(R[b], tb) for b in t.index]
    books = {b.name: b for b in dv.family("MAIN")}

    # ── check 2 (fair): excess Sharpe as the objective; A1 reproduced on the same code path ──
    log("fair bootstrap (Sharpe, excess Sharpe)")
    sh_f, xs_f, dd_f = boot_sharpe_dd(R.to_numpy(), tb.reindex(R.index).to_numpy())
    chk = pd.Series((dd_f < dv.DD_LIMIT).mean(axis=0), index=names)
    result["fair_breach_reproduced_max_abs_diff"] = float((chk - t["p_breach10"].reindex(names)).abs().max())
    result["fair"] = {"sharpe": {k: select(t, "sharpe", names, sh_f, v) for k, v in pools(t).items()},
                      "xs_sharpe": {k: select(t, "xs_sharpe", names, xs_f, v) for k, v in pools(t).items()}}

    # ── check 3a: Sharpe picks under each fair sensitivity frame (tie band from the main bootstrap) ──
    log("sensitivity frames")
    swap_tfi = data["long"].copy()
    swap_tfi["tactical_fi"] = data["tfi_frozen"]
    sens = {"proxy_unscaled": dv.fair(data["long_unscaled"], data), "plus_5bps": dv.fair(data["stressed_long"], data),
            "tfi_frozen": dv.fair(swap_tfi, data), "etf_idle_2008_09": dv.fair(data["long_etf_cash"], data)}
    passers = list(t.index[t["gates_pass"].astype(bool)])
    sens_out: dict = {}
    for label, frame in sens.items():
        ts = t.copy()
        for b in passers:
            r = dv.book_series(frame, books[b])
            ts.at[b, "sharpe"] = sharpe(r)
            ts.at[b, "xs_sharpe"] = xs_sharpe(r, frame[TBILL])
        sens_out[label] = {obj: {k: {kk: vv for kk, vv in select(ts, obj, names, boot, v).items()
                                     if kk in ("top", "pick", "recommendation")}
                                 for k, v in pools(ts).items()}
                           for obj, boot in (("sharpe", sh_f), ("xs_sharpe", xs_f))}
    result["fair_sensitivity"] = sens_out

    # ── check 3b: CSCV ──
    log("CSCV")
    pick_names = list(dict.fromkeys(KEY_BOOKS + [result["fair"][o][k]["pick"] for o in ("sharpe", "xs_sharpe")
                                                 for k in ("all", "no_eom")]))
    Xf = R.to_numpy()
    Xx = Xf - tb.reindex(R.index).to_numpy()[:, None]
    jp = [names.index(b) for b in passers]
    result["cscv"] = {
        "sharpe_all_books": cscv_sharpe(Xf, {b: names.index(b) for b in pick_names}),
        "xs_sharpe_all_books": cscv_sharpe(Xx, {b: names.index(b) for b in pick_names}),
        "sharpe_gate_passers": cscv_sharpe(Xf[:, jp], {b: passers.index(b) for b in pick_names if b in passers}),
        "xs_sharpe_gate_passers": cscv_sharpe(Xx[:, jp], {b: passers.index(b) for b in pick_names if b in passers}),
    }

    # ── check 1: house cash, full rule (both lines) ──
    house = {}
    for line in ("LOW", "MAIN"):
        log(f"house rule {line}")
        ht, hR, hout = house_rule(line, data, fair_t[line.lower()])
        ht.drop(columns=[c for c in ht.columns if c.startswith("_")], errors="ignore").to_csv(
            OUT / f"house_{line.lower()}_books.csv", float_format="%.6g")
        house[line] = hout
        if line == "MAIN":
            house_R = hR
    result["house"] = house

    # ── check 3c: weight plateau (fair and house) ──
    log("plateau")
    plateau_rows = []
    bases = {A1_PICK: json.loads(t.at[A1_PICK, "avg_weights"]), A1_NOEOM: {"btal_qqq": 0.5, "etf_dv2": 0.5},
             A1_TOP: {"core5": 1 / 3, "eom_flow": 1 / 3, "etf_dv2": 1 / 3}}
    pb_all = []
    for base_name, w in bases.items():
        base = books[base_name]
        tot = sum(w.values())
        w = {p: v / tot for p, v in w.items()}  # defensive-part weights (avg_weights are scaled by 1 - slice)
        pb_all += plateau_books(base, w)
    pb_all.append(dv.DefBook(CHAMPION, ("core5", "btal_qqq"), "FIXED", {"core5": 0.6, "btal_qqq": 0.4}, 0.0))
    pr = {"fair": {}, "house": {}}
    for cash, frame in (("fair", fair_main), ("house", data["long"])):
        for b in pb_all:
            pr[cash][b.name] = dv.book_series(frame, b)
    for cash in ("fair", "house"):
        P = pd.DataFrame(pr[cash])
        sh_p, xs_p, dd_p = boot_sharpe_dd(P.to_numpy(), tb.reindex(P.index).to_numpy())
        for j, b in enumerate(P.columns):
            r = P[b]
            plateau_rows.append({"book": b, "cash": cash, "sharpe": sharpe(r), "xs_sharpe": xs_sharpe(r, tb),
                                 "maxdd": lib.maxdd(r), "xs_cagr": lib.excess_cagr(r, tb, idx),
                                 "min_block_xs": min(lib.excess_cagr(lib.window(r, lo, hi), tb, idx)
                                                     for lo, hi in BLOCK_DICT.values()),
                                 "p_breach10": float((dd_p[:, j] < dv.DD_LIMIT).mean()),
                                 "sharpe_beats_champion": float(np.mean(sh_p[:, j] > sh_p[:, list(P.columns).index(CHAMPION)])),
                                 "xs_sharpe_beats_champion": float(np.mean(xs_p[:, j] > xs_p[:, list(P.columns).index(CHAMPION)]))})
    pd.DataFrame(plateau_rows).to_csv(OUT / "plateau.csv", index=False, float_format="%.6g")

    # ── report set: key books + new picks, sleeves ──
    log("periods, years, crises")
    new_picks = set()
    for scope in (result["fair"], result["house"]["MAIN"]):
        for obj in ("sharpe", "xs_sharpe"):
            for k in ("all", "no_eom"):
                if scope.get(obj, {}).get(k, {}).get("pick"):
                    new_picks.add(scope[obj][k]["pick"])
                    new_picks.add(scope[obj][k]["top"])
    report_books = list(dict.fromkeys(KEY_BOOKS + sorted(new_picks)))
    cofalls = lib.cofall_windows(data["bench"], LONG_START, END)
    periods, years, crises = [], {}, {}
    for cash, frame in (("fair", fair_main), ("house", data["long"])):
        series = {b: (R[b] if cash == "fair" else house_R[b]) for b in report_books}
        series.update({f"sleeve:{s}": frame[s].loc[LONG_START:END] for s in SLEEVES})
        series["sleeve:tbill"] = frame[TBILL].loc[LONG_START:END]
        for name, r in series.items():
            for w, m in window_metrics(r, tb, idx).items():
                periods.append({"book": name, "cash": cash, "window": w, **m})
            years[f"{cash}|{name}"] = calendar_excess(r, tb)
            if cash == "fair":
                crises[name] = crisis_rows(r, data, cofalls)
    pd.DataFrame(periods).to_csv(OUT / "periods.csv", index=False, float_format="%.6g")
    pd.DataFrame(years).to_csv(OUT / "calendar_excess.csv", float_format="%.6g")
    pd.DataFrame(crises).T.to_csv(OUT / "crises.csv", float_format="%.6g")
    result["cofall_windows"] = [[str(a.date()), str(b.date()), c] for a, b, c in cofalls]
    result["report_books"] = report_books

    # ── tie-band composition ──
    comp = {}
    for scope_name, scope in (("fair", result["fair"]), ("house", result["house"]["MAIN"])):
        for obj in ("sharpe", "xs_sharpe"):
            for k in ("all", "no_eom"):
                band = scope.get(obj, {}).get(k, {}).get("band", [])
                if band:
                    comp[f"{scope_name}|{obj}|{k}"] = {dv.LABEL[p]: sum(p in b_pods for b_pods in
                                                                          [books[b].pods for b in band]) / len(band)
                                                         for p in dv.POOL["MAIN"]} | {"G3": float(np.mean(
                                                             [books[b].slice_ > 0 for b in band])), "size": len(band)}
    result["band_composition"] = comp

    (OUT / "checks.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    # ── ease and capacity ──
    log("ease and capacity")
    import part_m  # noqa: PLC0415
    products, ease = {}, {}
    for b in report_books:
        if b not in t.index:
            continue
        w = json.loads(t.at[b, "avg_weights"])
        s = books[b].slice_ if b in books else 0.0
        if s:
            w = w | {"taa3x": s / 2, "ndx_vxn": s / 2}
        tot = sum(w.values())
        w = {p: v / tot for p, v in w.items()}  # avg_weights are rounded to 4 decimals
        products[b] = w
        ease[b] = lib.ops_fields(Book(b, tuple(w), "EQ", w), data, w) | {
            "not_live_share": float(t.at[b, "not_live_share"]), "pods_weights": w,
            "tiers": {p: data["meta"][p]["tier_str"] for p in w}}
    excess = {}
    for b, w in products.items():
        r = lib.book_returns(data["cash_exact"], Book(b, tuple(w), "EQ", w), EXACT_START)
        excess[b] = lib.excess_cagr(r, data["cash_exact"][TBILL], idx)
    cap = part_m.capacity(products, data, excess)
    cap.to_csv(OUT / "capacity.csv", float_format="%.6g")
    result["ease"] = ease
    (OUT / "checks.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
