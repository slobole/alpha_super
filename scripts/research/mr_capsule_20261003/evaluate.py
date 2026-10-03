"""MR capsule evaluation (SPEC_FROZEN.md). Usage: python evaluate.py"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mr_gate_selfcal_20261003"))
import components as cp  # noqa: E402
import final_page_data as fp  # noqa: E402

rs, npc, tbc = cp.rs, cp.npc, cp.tbc
OUT = cp.OUT
C0, MOM0, BOOK1 = "2004-01-05", "2015-11-02", "2026-08-19"
BLK = ("G-P1", "G-P2", "G-P3")
ALLB = BLK + ("G-FULL", "G-LONG")


def capsule(parts: dict, weights: dict, start=C0, end=None) -> pd.Series:
    frame = pd.DataFrame(parts).loc[start:end].dropna()
    return tbc.book_return_ser(frame, weights, "annual")[0]


def capsule_invvol(parts: dict, start=C0, end=None) -> pd.Series:
    """Pod model with year-end reset to inverse-volatility weights from the trailing 252 sessions (no look-ahead)."""
    frame = pd.DataFrame(parts).loc[start:end].dropna()
    hist = pd.DataFrame(parts).dropna()
    names = list(frame.columns)
    pod = None
    out = []
    for i, d in enumerate(frame.index):
        if pod is None or (i > 0 and frame.index[i - 1].year != d.year):
            trail = hist.loc[:frame.index[i - 1] if i > 0 else d].iloc[-252:]
            iv = 1.0 / trail.std().replace(0, np.nan)
            w = (iv / iv.sum()).fillna(1.0 / len(names)).to_numpy()
            total = 1.0 if pod is None else pod.sum()
            pod = w * total
        before = pod.sum()
        pod = pod * (1.0 + frame.loc[d].to_numpy())
        out.append(pod.sum() / before - 1.0)
    return pd.Series(out, index=frame.index)


def bootstrap_p(a: pd.Series, b: pd.Series, n=2000, block=20, seed=0) -> float:
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    T, k = len(x), len(x) // block
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        idx = (rng.integers(0, T - block, size=k)[:, None] + np.arange(block)).ravel()
        s = x[idx]
        wins += (s[:, 0].mean() / s[:, 0].std()) > (s[:, 1].mean() / s[:, 1].std())
    return wins / n


def overlap(hold: pd.DataFrame, a: str, b: str, dates: pd.DatetimeIndex) -> dict:
    pos = {d: i for i, d in enumerate(dates)}
    sets = {v: [set() for _ in dates] for v in (a, b)}
    for v in (a, b):
        h = hold[hold["variant"] == v]
        for asset, e0, e1 in h[["asset", "entry_date", "exit_date"]].itertuples(index=False):
            e0 = pd.Timestamp(e0)
            e1 = pd.Timestamp(e1) if pd.notna(e1) else dates[-1]
            i0 = dates.searchsorted(e0)
            i1 = dates.searchsorted(e1)
            for i in range(i0, min(i1, len(dates))):
                sets[v][i].add(asset)
    common = np.array([len(sets[a][i] & sets[b][i]) for i in range(len(dates))])
    na = np.array([len(s) for s in sets[a]])
    nb = np.array([len(s) for s in sets[b]])
    both = (na > 0) & (nb > 0)
    return {"mean_common_names": float(common.mean()), "mean_names_a": float(na.mean()), "mean_names_b": float(nb.mean()),
            "share_days_with_common": float((common > 0)[both].mean()) if both.any() else float("nan"),
            "common_share_of_positions": float(common.sum() / max((na + nb).sum(), 1) * 2)}


def main():
    comp = pd.read_parquet(OUT / "components.parquet")
    rate = rs.cash_rate(pd.DatetimeIndex(comp.index))
    sl = pd.read_csv(cp.SLEEVES, index_col=0, parse_dates=True)
    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(comp.index).fillna(0.0)
    rv = spmo.rolling(20).std().shift(1) * np.sqrt(252)
    w = (0.08 / rv).clip(upper=1).fillna(0.0)
    parks = {"tbills": rate, "core5": sl["core5"].reindex(comp.index), "spmo": w * spmo + (1 - w) * rate}

    def part(name, cost, park="tbills"):
        return comp[f"{name}|{cost}|base"] + comp[f"{name}|{cost}|cw"] * parks[park].reindex(comp.index)

    taa, L, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()
    LX = {"engine": L, "stress": Ls}

    def book(x, cost):
        return npc.candidate_book_blocks(taa, LX[cost], x)

    def book_window(x, cost, a, b):
        return tbc.metric_dict(tbc.book_window_return_ser({"taa": taa, "L": LX[cost], "X": x}, npc.CANDIDATE_WEIGHT_DICT, a, b))

    X = {}
    for cost in ("engine", "stress"):
        X[("main", cost)] = capsule({"DV2": part("DV2-G", cost), "HPI": part("HPI-G", cost)}, {"DV2": 0.5, "HPI": 0.5})
        X[("ungated", cost)] = capsule({"DV2": part("DV2-U", cost), "HPI": part("HPI-U", cost)}, {"DV2": 0.5, "HPI": 0.5})
        X[("DV2-G alone", cost)] = part("DV2-G", cost).loc[C0:]
        X[("HPI-G alone", cost)] = part("HPI-G", cost).loc[C0:]
        for sh in (0.25, 0.75):
            X[(f"DV2 share {sh}", cost)] = capsule({"DV2": part("DV2-G", cost), "HPI": part("HPI-G", cost)}, {"DV2": sh, "HPI": 1 - sh})
        X[("inverse vol", cost)] = capsule_invvol({"DV2": part("DV2-G", cost), "HPI": part("HPI-G", cost)})
        X[("park CORE5", cost)] = capsule({"DV2": part("DV2-G", cost, "core5"), "HPI": part("HPI-G", cost, "core5")}, {"DV2": 0.5, "HPI": 0.5}, start="2008-03-04")
        X[("park SPMO", cost)] = capsule({"DV2": part("DV2-G", cost, "spmo"), "HPI": part("HPI-G", cost, "spmo")}, {"DV2": 0.5, "HPI": 0.5})
        X[("ETF capsule3", cost)] = capsule({"DV2": part("DV2-G", cost), "HPI": part("HPI-G", cost), "ETF": part("ETF-G", cost)}, {"DV2": 1 / 3, "HPI": 1 / 3, "ETF": 1 / 3})
        X[("ETF capsule3 ungated", cost)] = capsule({"DV2": part("DV2-U", cost), "HPI": part("HPI-U", cost), "ETF": part("ETF-U", cost)}, {"DV2": 1 / 3, "HPI": 1 / 3, "ETF": 1 / 3})
    fin = rate + 0.015 / 252
    for cost in ("engine", "stress"):
        k = float(X[("ungated", cost)].std() / X[("main", cost)].std())
        X[("levered T-bills", cost)] = k * X[("main", cost)] - (k - 1) * fin.reindex(X[("main", cost)].index).fillna(0.0)
    rep = {"leverage_engine": float(X[("ungated", "engine")].std() / X[("main", "engine")].std())}
    # ---- books
    B = {key: book(x, key[1]) for key, x in X.items() if key[0] != "park SPMO"}
    B[("T-bills slot", "engine")] = book(bil, "engine")
    B[("T-bills slot", "stress")] = book(bil, "stress")
    rep["book"] = {f"{n}|{c}": {b: B[(n, c)][b] for b in ALLB} for (n, c) in B}
    rep["book_since_2015"] = {f"{n}|{c}": book_window(x, c, MOM0, BOOK1) for (n, c), x in X.items()}
    rep["book_since_2015"].update({f"T-bills slot|{c}": book_window(bil, c, MOM0, BOOK1) for c in ("engine", "stress")})
    # ---- standalone
    rep["standalone"] = {f"{n}|{c}": fp.stats(x.loc[C0:]) for (n, c), x in X.items() if n not in ("park CORE5", "park SPMO")}
    rep["standalone"]["park SPMO|engine (2015-11+)"] = fp.stats(X[("park SPMO", "engine")].loc[MOM0:])
    rep["standalone"]["main|engine (2015-11+)"] = fp.stats(X[("main", "engine")].loc[MOM0:])
    spy = npc.load_spy_tr_ret_ser().reindex(comp.index).fillna(0.0)
    rep["crises"] = []
    for name, a, b in fp.CRISES:
        if a < "2007-01-01":
            continue
        row = {"name": name, "start": a, "end": b}
        for n in ("main", "ungated", "DV2-G alone", "HPI-G alone", "ETF capsule3"):
            nav = (1 + X[(n, "engine")].loc[a:b]).cumprod()
            row[n] = {"ret": float(nav.iloc[-1] - 1), "dd": float((nav / nav.cummax() - 1).min())}
        nav = (1 + spy.loc[a:b]).cumprod()
        row["spy"] = {"ret": float(nav.iloc[-1] - 1), "dd": float((nav / nav.cummax() - 1).min())}
        rep["crises"].append(row)
    # ---- diversification
    dvg, hpg, dvu, hpu = (part(n, "engine").loc[C0:] for n in ("DV2-G", "HPI-G", "DV2-U", "HPI-U"))
    dd = lambda r: (1 + r).cumprod() / (1 + r).cumprod().cummax() - 1
    rep["diversification"] = {"corr_gated": float(dvg.corr(hpg)), "corr_ungated": float(dvu.corr(hpu)),
                              "corr_drawdowns_gated": float(dd(dvg).corr(dd(hpg))),
                              "corr_gated_vs_etf": float(dvg.corr(part("ETF-G", "engine").loc[C0:])),
                              "corr_hpi_gated_vs_etf": float(hpg.corr(part("ETF-G", "engine").loc[C0:]))}
    hold = pd.read_parquet(OUT / "holdings.parquet")
    dts = pd.DatetimeIndex(comp.loc[C0:].index)
    rep["overlap_gated"] = overlap(hold, "DV2-G", "HPI-G", dts)
    rep["overlap_ungated"] = overlap(hold, "DV2-U", "HPI-U", dts)
    # ---- bootstrap (G-FULL book): main vs each single component
    def gfull_ser(x, cost):
        a, b = npc.BOOK_BLOCK_DICT["G-FULL"]
        return tbc.book_window_return_ser({"taa": taa, "L": LX[cost], "X": x}, npc.CANDIDATE_WEIGHT_DICT, a, b)
    rep["bootstrap"] = {f"main vs {n}|{c}": bootstrap_p(gfull_ser(X[("main", c)], c), gfull_ser(X[(n, c)], c))
                        for n in ("DV2-G alone", "HPI-G alone", "ungated") for c in ("engine", "stress")}
    # ---- pre-2004 proxy (DV2 + QPI as HPI stand-in), engine costs, T-bills parking
    px = pd.read_parquet(OUT / "proxy_1995_2003.parquet")
    pr = rs.cash_rate(pd.DatetimeIndex(px.index))
    pp = lambda n: px[f"{n}|engine|base"] + px[f"{n}|engine|cw"] * pr
    rep["proxy_1995_2003"] = {"capsule gated": fp.stats(capsule({"DV2": pp("DV2-G"), "Q": pp("QPI-G")}, {"DV2": 0.5, "Q": 0.5}, start="1995-01-03")),
                              "capsule ungated": fp.stats(capsule({"DV2": pp("DV2-U"), "Q": pp("QPI-U")}, {"DV2": 0.5, "Q": 0.5}, start="1995-01-03")),
                              "DV2-G": fp.stats(pp("DV2-G")), "QPI-G": fp.stats(pp("QPI-G")), "DV2-U": fp.stats(pp("DV2-U")), "QPI-U": fp.stats(pp("QPI-U"))}
    # ---- decision rule
    g = lambda n, c, b: rep["book"][f"{n}|{c}"][b]["sharpe"]
    r1 = all(g("main", c, b) > g("T-bills slot", c, b) for c in ("engine", "stress") for b in BLK)
    r2 = all(g("main", c, b) >= max(g("DV2-G alone", c, b), g("HPI-G alone", c, b)) - 0.02 for c in ("engine", "stress") for b in ("G-FULL", "G-LONG"))
    r3 = all(g("main", c, b) > g("ungated", c, b) for c in ("engine", "stress") for b in ("G-FULL", "G-LONG"))
    r4 = all(abs(g(f"DV2 share {sh}", "engine", "G-FULL") - g("main", "engine", "G-FULL")) <= 0.03 for sh in (0.25, 0.75))
    r5 = rep["book"]["main|engine"]["G-LONG"]["max_dd"] >= rep["book"]["T-bills slot|engine"]["G-LONG"]["max_dd"] - 0.02
    rep["decision"] = {"R1_beats_tbills_each_block_both_costs": r1, "R2_noninferior_to_best_component": r2, "R3_beats_ungated_capsule": r3,
                       "R4_weight_plateau": r4, "R5_dd_vs_tbills": r5, "pass": bool(r1 and r2 and r3 and r4 and r5)}

    def park_ok(n):
        return all(g(n, c, b) >= g("main", c, b) + 0.02 for c in ("engine", "stress") for b in ("G-FULL", "G-LONG")) and \
            all(g(n, c, b) >= g("main", c, b) - 0.03 for c in ("engine", "stress") for b in BLK)
    rep["decision"]["parking_CORE5_replaces_tbills"] = bool(park_ok("park CORE5"))
    s15 = rep["book_since_2015"]
    rep["decision"]["parking_SPMO_since_2015_plus_0.02"] = bool(all(s15[f"park SPMO|{c}"]["sharpe"] >= s15[f"main|{c}"]["sharpe"] + 0.02 for c in ("engine", "stress")))
    rep["decision"]["optional_A_ETF_capsule3_adopt"] = bool(all(g("ETF capsule3", c, b) >= g("main", c, b) for c in ("engine", "stress") for b in ("G-FULL", "G-LONG"))
                                                            and all(g("ETF capsule3", c, b) >= g("main", c, b) - 0.03 for c in ("engine", "stress") for b in BLK))
    (OUT / "evaluation.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    # ---- print
    order = ["T-bills slot", "DV2-G alone", "HPI-G alone", "ungated", "main", "DV2 share 0.25", "DV2 share 0.75", "inverse vol",
             "park CORE5", "levered T-bills", "ETF capsule3", "ETF capsule3 ungated"]
    for c in ("engine", "stress"):
        print(f"\nBOOK Sharpe ({c})")
        for n in order:
            if f"{n}|{c}" in rep["book"]:
                b = rep["book"][f"{n}|{c}"]
                print(f"  {n:<22}", "  ".join(f"{k}:{b[k]['sharpe']:.3f}" for k in ALLB), f" DD-LONG {b['G-LONG']['max_dd']:.3f}  CAGR-LONG {b['G-LONG']['cagr']:.3f}")
    print("\nSINCE 2015-11 book Sharpe", {k: round(v["sharpe"], 3) for k, v in s15.items() if k.split("|")[0] in ("T-bills slot", "main", "park SPMO", "park CORE5", "levered T-bills", "ungated", "DV2-G alone", "HPI-G alone")})
    print("\nSTANDALONE 2004-26", {k: (round(v["cagr"], 3), round(v["sharpe"], 3), round(v["max_dd"], 3), round(v["worst_year"], 3)) for k, v in rep["standalone"].items() if "|engine" in k})
    print("\nDIVERSIFICATION", {k: round(v, 3) for k, v in rep["diversification"].items()}, "\nOVERLAP gated", {k: round(v, 3) for k, v in rep["overlap_gated"].items()}, "\nOVERLAP ungated", {k: round(v, 3) for k, v in rep["overlap_ungated"].items()})
    print("\nBOOTSTRAP P(main better)", {k: round(v, 3) for k, v in rep["bootstrap"].items()})
    print("\nPROXY 1995-2003", {k: (round(v["cagr"], 3), round(v["sharpe"], 3), round(v["max_dd"], 3)) for k, v in rep["proxy_1995_2003"].items()})
    print("\nDECISION", rep["decision"], "leverage", round(rep["leverage_engine"], 3))


if __name__ == "__main__":
    main()
