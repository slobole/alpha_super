"""SPEC 4 (final products), 5 (champions), 6 (robustness) and A0 (menus): evaluate every option of the main frame.

Usage: python evaluate.py   (after stage1/stage2 for main and the sensitivity frames). Writes <study>/report/eval.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import yaml

import fp_lib as fp
from fp_lib import EXACT_START, LONG_START, TBILL, Book, ga, lib

_spec = importlib.util.spec_from_file_location("ga_report_data", fp.GA_DIR / "report_data.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)
import allocator_first_look as afl  # noqa: E402

OUT = fp.STUDY / "report"
FRAMES = ["s1_house_cash", "s2_proxy_unscaled", "s3_plus_5bps", "s5_hpi_live_gap", "s6_exact", "s7_block126", "s8_block21"]
MENU = {"DEFENSIVE": [("D-LAUNCH", "DEFENSIVE|LOW-TOUCH"), ("D-MAIN", "DEFENSIVE|MAIN"), ("D-TARGET", "DEFENSIVE|TARGET"),
                      ("D-CALMER", "D-CALMER|LOW-TOUCH"), ("D-RICHER", "D-RICHER|LOW-TOUCH"), ("D-SIMPLEST", "champ:D0")],
        "GROWTH": [("G-LAUNCH", "GROWTH|LOW-TOUCH"), ("G-PLUS", "GROWTH PLUS|LOW-TOUCH"), ("G-MAIN", "GROWTH|MAIN"),
                   ("G-PLUS-MAIN", "GROWTH PLUS|MAIN"), ("G-TARGET", "GROWTH|TARGET"), ("G-SHARPE", "G-SHARPE|LOW-TOUCH"),
                   ("G-SIMPLEST", "champ:G3"),
                   # A2 (post-result, labelled): TAA block = the two 3x BTAL variants, or TAA3x-1N alone.
                   ("G-LAUNCH-3X", "GROWTH|LOW-TOUCH@_TAA3X"), ("G-PLUS-3X", "GROWTH PLUS|LOW-TOUCH@_TAA3X"),
                   ("G-MAIN-3X", "GROWTH|MAIN@_TAA3X"), ("G-LAUNCH-1N", "GROWTH|LOW-TOUCH@_TAA1N"),
                   ("G-PLUS-1N", "GROWTH PLUS|LOW-TOUCH@_TAA1N"), ("G-MAIN-1N", "GROWTH|MAIN@_TAA1N")]}
MENU["DEFENSIVE"] += [("D-LAUNCH-3X", "DEFENSIVE|LOW-TOUCH@_TAA3X"), ("D-MAIN-3X", "DEFENSIVE|MAIN@_TAA3X")]
CHAMPS = {"champ:G3": {"taa3x": 0.5, "ndx_vxn": 0.5},
          "champ:GROWTH_VERDICT": {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18},
          "champ:AGGR_VERDICT": {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09},
          "champ:D0": {"core5": 0.6, "btal_qqq": 0.4},
          "champ:D5": {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}}
RULES = {"DEFENSIVE": ("xsharpe", -0.07, -0.10, 0.10), "D-CALMER": ("xsharpe", -0.05, -0.07, 0.10), "D-RICHER": ("cagr", -0.07, -0.10, 0.10),
         "GROWTH": ("cagr", -0.17, -0.20, 0.15), "GROWTH PLUS": ("cagr", -0.22, -0.25, 0.15), "G-SHARPE": ("xsharpe", -0.17, -0.20, 0.15)}


def round_weights(w: dict, step: float = 0.01) -> dict:
    keys = [k for k, v in w.items() if v > 1e-6]
    raw = np.array([w[k] for k in keys]) / sum(w[k] for k in keys) / step
    base = np.floor(raw)
    left = int(round(1 / step - base.sum()))
    base[np.argsort(-(raw - base))[:left]] += 1
    return {k: round(b * step, 4) for k, b in zip(keys, base) if b > 0}


def pods_from_blocks(bw: dict, cols: dict, members: dict) -> dict:
    out: dict = {}
    for blk, w in bw.items():
        if w <= 1e-9:
            continue
        col = cols[blk]
        mem = [TBILL] if col == "CASH" else members[col]
        for p in mem:
            out[p] = out.get(p, 0.0) + w / len(mem)
    return out


def xsharpe(r: np.ndarray, rf: np.ndarray) -> float:
    x = r - rf
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def seeds_tail(r: np.ndarray, limits=(-0.07, -0.10, -0.15, -0.20, -0.25)) -> dict:
    acc: dict = {}
    for k in range(10):
        b = ga.bootstrap_paths(r[:, None], lib.evaluation.stationary_bootstrap_index_mat(len(r), 2000, 63.0, fp.SEED0 + k))
        acc.setdefault("ddar10", []).append(float(np.percentile(b["gross_dd"][:, 0], 10)))
        acc.setdefault("net_ddar10", []).append(float(np.percentile(b["net_dd"][:, 0], 10)))
        for L in limits:
            acc.setdefault(f"p{int(round(-L * 100))}", []).append(float((b["gross_dd"][:, 0] < L).mean()))
    return {k: float(np.mean(v)) for k, v in acc.items()} | {f"{k}_max": float(np.max(v)) for k, v in acc.items()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    frames = ga.frames(data)
    frame, start = frames["main"]
    fex, sex = frames["s6_exact"]
    index_all = data["index"]
    s1 = json.loads((fp.STUDY / "main" / "stage1.json").read_text(encoding="utf-8"))
    s2 = json.loads((fp.STUDY / "main" / "stage2.json").read_text(encoding="utf-8"))
    s2_alt = {tag: json.loads((fp.STUDY / "main" / f"stage2{tag}.json").read_text(encoding="utf-8")) for tag in ("_TAA3X", "_TAA1N")}
    members = {}
    for fam, f in s1["families"].items():
        for kind in ("target", "deployable", "monthly"):
            if f.get(f"split_{kind}"):
                members[f"{fam}|{kind}"] = f[f"split_{kind}"]["members"]
    members["TAA|TAA3X"], members["TAA|TAA1N"] = ["taa3x", "taa3x_1n"], ["taa3x_1n"]
    books: dict[str, dict] = {}
    for prod, items in MENU.items():
        for label, key in items:
            if key.startswith("champ:"):
                books[label] = {"pods": CHAMPS[key], "key": key, "product": prod}
                continue
            key0, _, tag = key.partition("@")
            src = s2_alt[tag] if tag else s2
            opt, line = key0.split("|")
            rw = src["options"][key0]["robust_weights"]
            books[label] = {"pods": round_weights(pods_from_blocks(rw, src["lines"][line]["block_cols"], members)),
                            "block_weights": {k: round(v, 3) for k, v in rw.items()}, "key": key, "product": prod, "rule": opt,
                            "line": line, "alt": tag or None, "inclusion": src["options"][key0]["inclusion_share"],
                            "hist_opt": src["options"][key0]["hist_optimum"]}
    for key in ("champ:GROWTH_VERDICT", "champ:AGGR_VERDICT", "champ:D5"):
        books[key.replace("champ:", "REF-")] = {"pods": CHAMPS[key], "key": key, "product": "REF"}
    # References: the fund menu YAMLs and ladder_4.
    rs = __import__("run_sleeves")
    alias_by_module = {v[0].split(":")[0]: k for k, v in rs.SLEEVE_DICT.items() if k not in rs.SENSITIVITY_ALIAS_SET}
    for path in sorted((fp.ga.MAIN_REPO / "portfolios").glob("fund_menu_*.yaml")) + [fp.ga.MAIN_REPO / "portfolios" / "ladder_4_growth.yaml"]:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        w = {}
        for pod in spec["pods"]:
            a = alias_by_module.get(pod["strategy_import_str"].split(":")[0])
            if a is None:
                w = None
                break
            w[a] = w.get(a, 0) + float(pod["weight_float"])
        if w:
            tot = sum(w.values())
            books[f"REF-{path.stem}"] = {"pods": {k: v / tot for k, v in w.items()}, "key": path.name, "product": "REF"}

    rf_long = frame[TBILL]

    def passes(w: dict, rule: tuple) -> tuple[bool, dict]:
        r_ = lib.book_returns(frame, Book("chk", tuple(w), "EQ", w), start).to_numpy()
        t_ = seeds_tail(r_)
        nav = np.r_[1, np.cumprod(1 + r_)]
        dd_ = float((nav / np.maximum.accumulate(nav) - 1).min())
        return bool(dd_ >= rule[1] and t_[f"p{int(round(-rule[2] * 100))}"] <= rule[3]), t_

    # SPEC 4: robust weights must pass the product's own rule (hist build limit + 10-seed breach); otherwise scale toward
    # CASH (growth products) or the DEF block (defensive products) in 5% steps.
    def_block = {p: 1.0 / len(members["DEF|monthly"]) for p in members["DEF|monthly"]}
    for label, b in books.items():
        rule = RULES.get(b.get("rule"))
        if not rule:
            continue
        w0 = dict(b["pods"])
        target = {TBILL: 1.0} if rule[0] == "cagr" else def_block
        for step in range(0, 21):
            a = step * 0.05
            w = {k: (1 - a) * w0.get(k, 0) + a * target.get(k, 0) for k in set(w0) | set(target)}
            w = round_weights(w)
            ok, _ = passes(w, rule)
            if ok:
                break
        b["pods"], b["scaled_toward_safe"] = w, a
        print("rule check", label, "scaled", a, flush=True)
    results = {}
    for label, b in books.items():
        w = b["pods"]
        bk = Book(label, tuple(w), "EQ", w)
        try:
            r = lib.book_returns(frame, bk, start)
        except ValueError as exc:
            results[label] = {**b, "error": str(exc)}
            continue
        rex = lib.book_returns(fex, bk, sex)
        fm = lib.full_metrics(r, data, "long")
        st = ga.window_stats(r.to_frame(), index_all)
        down = r[r < 0]
        yrs = (1 + r).groupby(r.index.year).prod() - 1
        tail = seeds_tail(r.to_numpy())
        res = {**b, "long": {k.replace("long_", ""): (v if isinstance(v, (str, type(None))) else float(v)) for k, v in fm.items()},
               "net_cagr": float(st["net_cagr"][0]), "net_dd": float(st["net_dd"][0]), "net_sharpe": float(st["net_sharpe"][0]),
               "xsharpe": xsharpe(r.to_numpy(), rf_long.reindex(r.index).to_numpy()), "sortino": float(r.mean() / down.std() * np.sqrt(252)),
               "exact": {k.replace("exact_", ""): (v if isinstance(v, (str, type(None))) else float(v)) for k, v in lib.full_metrics(rex, data, "exact").items()},
               "exact_xsharpe": xsharpe(rex.to_numpy(), fex[TBILL].reindex(rex.index).to_numpy()),
               "recent_cagr": float(ga.window_stats(r.loc["2023-08-21":].to_frame(), index_all)["gross_cagr"][0]),
               "tail": tail, "years": {int(y): float(v) for y, v in yrs.items()},
               "crises": {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()},
               "nav": [[d.strftime("%Y-%m"), round(float(v), 4)] for d, v in (1 + r).cumprod().resample("ME").last().items()],
               "ops": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in lib.ops_fields(bk, data).items()
                       if not isinstance(v, str)},
               "needs_wiring": sorted(p for p in w if p != TBILL and (data["meta"][p]["tier_str"] != "wired" or p in fp.NOT_LIVE_TRADABLE)),
               "daily_pods": sorted(p for p in w if p in fp.DAILY)}
        for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, fp.END):
            res["crises"][f"cofall|{lo.date()}|{hi.date()}|{ret:.3f}"] = lib.common.window_return_float(r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
        # Rule check (SPEC 4): historical build limit and 10-seed breach; scale toward CASH/DEF if it fails.
        rule = RULES.get(b.get("rule"))
        if rule:
            obj, build, hard, mb = rule
            res["rule_pass"] = bool(fm["long_maxdd"] >= build and tail[f"p{int(round(-hard * 100))}"] <= mb)
        results[label] = res
        results[label]["_r"] = r
        print("done", label, round(fm["long_cagr"], 4), round(res["xsharpe"], 2), round(fm["long_maxdd"], 4), res.get("rule_pass"), flush=True)

    # Champion tests (SPEC 5): paired bootstrap on the frozen seed.
    def paired(a: pd.Series, c: pd.Series, obj: str) -> float:
        A = pd.concat([a, c, rf_long], axis=1).dropna().to_numpy()
        idx = lib.evaluation.stationary_bootstrap_index_mat(len(A), 2000, 63.0, fp.SEED0)
        wins = 0
        for k in range(2000):
            s = A[idx[k]]
            if obj == "cagr":
                va, vc = np.prod(1 + s[:, 0]), np.prod(1 + s[:, 1])
            else:
                va, vc = xsharpe(s[:, 0], s[:, 2]), xsharpe(s[:, 1], s[:, 2])
            wins += va > vc
        return wins / 2000
    champ_of = {"D-LAUNCH": ["D-SIMPLEST"], "D-MAIN": ["D-SIMPLEST"], "D-TARGET": ["D-SIMPLEST", "REF-D5"], "D-CALMER": ["D-SIMPLEST"],
                "D-RICHER": ["D-SIMPLEST"], "G-LAUNCH": ["G-SIMPLEST", "REF-GROWTH_VERDICT"], "G-MAIN": ["G-SIMPLEST", "REF-GROWTH_VERDICT"],
                "G-TARGET": ["G-SIMPLEST", "REF-GROWTH_VERDICT"], "G-PLUS": ["REF-AGGR_VERDICT"], "G-PLUS-MAIN": ["REF-AGGR_VERDICT"],
                "G-SHARPE": ["G-SIMPLEST", "REF-GROWTH_VERDICT"],
                "G-LAUNCH-3X": ["G-SIMPLEST", "REF-GROWTH_VERDICT"], "G-MAIN-3X": ["G-SIMPLEST", "REF-GROWTH_VERDICT"],
                "G-LAUNCH-1N": ["G-SIMPLEST", "REF-GROWTH_VERDICT"], "G-MAIN-1N": ["G-SIMPLEST", "REF-GROWTH_VERDICT"],
                "G-PLUS-3X": ["REF-AGGR_VERDICT"], "G-PLUS-1N": ["REF-AGGR_VERDICT"],
                "D-LAUNCH-3X": ["D-SIMPLEST"], "D-MAIN-3X": ["D-SIMPLEST"]}
    tests = {}
    for lab, champs in champ_of.items():
        if lab not in results or "error" in results[lab]:
            continue
        obj = RULES[results[lab]["rule"]][0]
        for ch in champs:
            a, c = results[lab], results[ch]
            share = paired(a["_r"], c["_r"], obj)
            hard = f"p{int(round(-RULES[a['rule']][2] * 100))}"
            ex_a = a["exact"]["cagr"] if obj == "cagr" else a["exact_xsharpe"]
            ex_c = c["exact"]["cagr"] if obj == "cagr" else c["exact_xsharpe"]
            tests[f"{lab} vs {ch}"] = {"objective": obj, "beats_share": share, "breach_new": a["tail"][hard], "breach_champ": c["tail"][hard],
                                       "exact_new": ex_a, "exact_champ": ex_c,
                                       "replaces": bool(share >= 0.80 and a["tail"][hard] <= c["tail"][hard] + 1e-12 and ex_a > ex_c)}
    # Sensitivity frames: the main products' fixed weights in each frame, and each frame's own robust block weights.
    sens = {}
    for f in FRAMES:
        fr, st_ = frames[f]
        row = {}
        for lab in [l for items in MENU.values() for l, _ in items]:
            if lab not in results or "error" in results[lab]:
                continue
            w = results[lab]["pods"]
            try:
                r = lib.book_returns(fr, Book(lab, tuple(w), "EQ", w), st_)
            except ValueError:
                continue
            s_ = ga.window_stats(r.to_frame(), index_all)
            row[lab] = {"cagr": float(s_["gross_cagr"][0]), "dd": float(s_["gross_dd"][0]),
                        "xsharpe": xsharpe(r.to_numpy(), fr[TBILL].reindex(r.index).to_numpy())}
        p2 = fp.STUDY / f / "stage2.json"
        own = json.loads(p2.read_text(encoding="utf-8"))["options"] if p2.exists() else {}
        sens[f] = {"fixed": row, "own_weights": {k: v["robust_weights"] for k, v in own.items()}}
    # Capacity and income for the menu books.
    cap_books = {lab: {k: v for k, v in results[lab]["pods"].items()} for lab in results
                 if "error" not in results[lab] and results[lab]["product"] != "REF" or lab.startswith("REF-GROWTH") or lab.startswith("REF-AGGR")}
    excess = {lab: float(results[lab]["exact"]["cagr"] - results[lab]["exact"]["tbill_cagr"]) for lab in cap_books}
    cap = rd.capacity(cap_books, data, excess)
    # Factor alpha (gross and net), menu books only.
    closes = pd.concat([lib.common.load_total_return_close_ser(s, "2005-01-01", fp.END.strftime("%Y-%m-%d")) for s in afl.ETF_LIST], axis=1)
    closes.columns = afl.ETF_LIST
    closes = closes.reindex(index_all).loc["2005-01-01":]
    naive = afl.naive_rule_return_df(closes, data["sleeve"][TBILL])
    etf_r = closes.pct_change(fill_method=None)
    tb_w = afl.weekly_ser(data["sleeve"][TBILL])
    fac_w = pd.DataFrame({s: afl.weekly_ser(etf_r[s]) for s in afl.FACTOR_M1_LIST}).sub(tb_w, axis=0)
    trend_w = afl.weekly_ser(naive["NAIVE_QQQ_TREND200"]) - tb_w
    alpha = []
    for lab in [l for items in MENU.values() for l, _ in items]:
        if lab not in results or "error" in results[lab]:
            continue
        r = results[lab]["_r"]
        for basis, ser in (("gross", r), ("net", ga.net_return_series(r))):
            y = (afl.weekly_ser(ser) - tb_w).dropna().iloc[1:-1]
            x1 = fac_w.reindex(y.index)
            for model, x in (("M0", x1[["QQQ"]]), ("M1", x1), ("M2", x1.assign(TREND200=trend_w.reindex(y.index)))):
                alpha.append({"book": lab, "basis": basis, "model": model, **afl.newey_west_ols(y, x, afl.NW_LAG_INT)})
    for v in results.values():
        v.pop("_r", None)
    payload = {"books": results, "tests": tests, "sens": sens, "capacity": cap, "alpha": alpha,
               "stage1": s1, "stage2": s2}
    (OUT / "eval.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    fp.ledger("evaluate_finished")
    for k, t in tests.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in t.items()})
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(fp.ga.MAIN_REPO / "scripts" / "research" / "shelf_rebuild_20260929"))
    raise SystemExit(main())
