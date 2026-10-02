"""QPI same-day close test (SPEC_FROZEN.md). Usage: python run_test.py validate|final|exact"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q  # noqa: E402

A0, A1 = "2016-01-04", "2026-09-24"
FULL0 = "2004-01-02"
ARCH = q.REPO / "results/_archive_07-05-2026"
MODES = {
    "M0_open_open": q.Mode("M0", "open", "open"),
    "M1_closeFinal_open": q.Mode("M1", "close", "open", "final"),
    "M1r_closeFinal_prevRank_open": q.Mode("M1r", "close", "open", "final", rank="prev"),
    "M2_close1545_open": q.Mode("M2", "close", "open", "1545", rank="prev"),
    "M3_closeFinal_closeFinal": q.Mode("M3", "close", "close", "final", "final"),
    "M4_close1545_close1545": q.Mode("M4", "close", "close", "1545", "1545", rank="prev"),
}


def held_mask(p, res):
    h = np.zeros(p.C.shape, dtype=bool)
    col = {s: i for i, s in enumerate(p.symbols)}
    pos = {d: k for k, d in enumerate(p.dates)}
    for a, e, x in res.trades[["asset", "entry_date", "exit_date"]].itertuples(index=False):
        h[pos[e]:pos[x] + 1, col[a]] = True
    return h


def validate(p, F):
    out = {}
    for name, arch, mode in (("M0", "strategy_mr_qpi_ibs_rsi_exit/2026-05-04_232528", MODES["M0_open_open"]),
                             ("M1", "strategy_mr_qpi_ibs_rsi_exit_moc_paper/2026-04-20_160024", MODES["M1_closeFinal_open"])):
        tx = pd.read_csv(ARCH / arch / "transactions.csv", parse_dates=["bar"])
        end = str(tx["bar"].max().date())
        res = q.run(p, mode, F, FULL0, end, capital=100_000.0)
        eng = set(zip(tx.loc[tx["amount"] > 0, "bar"], tx.loc[tx["amount"] > 0, "asset"]))
        rep = set(zip(res.trades["entry_date"], res.trades["asset"]))
        out[name] = {"engine_entries": len(eng), "replica_entries": len(rep), "common": len(eng & rep),
                     "share_of_engine": len(eng & rep) / len(eng), "replica": q.summarize(res)}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != "replica"}, flush=True)
    return out


def exact_features(p):
    import moc_layers23 as ml  # dv2_deep_20260925
    frames = [pd.read_csv(f) for d in (ml.ALP, q.OUT / "alpaca" / "sessions") for f in sorted(d.glob("*.csv.gz"))]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["p1545"])
    df = df.drop_duplicates(["date", "norgate_symbol"], keep="last")
    col = {s: i for i, s in enumerate(p.symbols)}
    df["col"] = df["norgate_symbol"].map(col)
    df["row"] = p.dates.get_indexer(pd.to_datetime(df["date"]))
    df = df.dropna(subset=["col"])
    df = df[df["row"] >= 0]
    df["col"] = df["col"].astype(int)
    raw = np.asarray(p.RAW)[df["row"].to_numpy(), df["col"].to_numpy()]
    oc = df["close_official"].to_numpy()
    denom = np.where(np.isfinite(oc) & (oc > 0), oc, raw)
    df["rP"], df["rH"], df["rL"] = df["p1545"] / denom, df["h1545"] / denom, df["l1545"] / denom
    P, Hd, Ld = ml.exact_state(p, df)
    have = np.zeros(p.C.shape, dtype=bool)
    have[df["row"].to_numpy(), df["col"].to_numpy()] = True
    return q.Features(p, P, Hd, Ld), have


def attribution(fin_res, ex_res, p):
    """Entries common to both runs vs only in one, with gross trade return and 1-day market-adjusted return."""
    kf = set(zip(fin_res.trades["entry_date"], fin_res.trades["asset"]))
    ke = set(zip(ex_res.trades["entry_date"], ex_res.trades["asset"]))
    out = {}
    for name, res, other in (("close_final", fin_res, ke), ("close_1545", ex_res, kf)):
        t = res.trades.assign(common=[k in other for k in zip(res.trades["entry_date"], res.trades["asset"])])
        out[name] = {("common" if c else "only"): {"n": int(len(g)), "mean_ret": float(g["ret"].mean())} for c, g in t.groupby("common")}
    return out


def main(phase):
    q.OUT.mkdir(parents=True, exist_ok=True)
    p = q.rp.Panel("sp500")
    F = {"final": q.Features(p)}
    if phase == "validate":
        rep = validate(p, F)
        (q.OUT / "validate.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
        return
    if phase == "final":
        rep = {}
        for name in ("M0_open_open", "M1_closeFinal_open", "M1r_closeFinal_prevRank_open", "M3_closeFinal_closeFinal"):
            m = MODES[name]
            rf = q.run(p, m, F, FULL0, A1)
            ra = q.run(p, m, F, A0, A1)
            np.save(q.OUT / f"held_{name}.npy", held_mask(p, ra))
            rep[name] = {"full_2004": q.summarize(rf), "alpaca_window": q.summarize(ra)}
            print(name, round(rep[name]["full_2004"]["sharpe"], 3), round(rep[name]["alpaca_window"]["sharpe"], 3), flush=True)
        (q.OUT / "final_state.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
        return
    # exact 15:45
    F["1545"], have = exact_features(p)
    rep = {"coverage": {}}
    res = {}
    for name, m in MODES.items():
        for bps in (0.0, 5.0):
            mm = q.Mode(m.name, m.entry, m.exit, m.entry_state, m.exit_state, m.rank, bps)
            r = q.run(p, mm, F, A0, A1)
            res[(name, bps)] = r
            rep.setdefault(name, {})["base" if bps == 0 else "stress_5bps"] = q.summarize(r)
        np.save(q.OUT / f"held_{name}.npy", held_mask(p, res[(name, 0.0)]))
        if "1545" in name:
            e = res[(name, 0.0)].trades
            pos = {d: k for k, d in enumerate(p.dates)}
            col = {s: i for i, s in enumerate(p.symbols)}
            rep["coverage"][name] = float(np.mean([have[pos[d], col[a]] for d, a in zip(e["entry_date"], e["asset"])]))
        print(name, {k: round(v["sharpe"], 3) for k, v in rep[name].items()}, flush=True)
    rep["attribution_M1_vs_M2"] = attribution(res[("M1_closeFinal_open", 0.0)], res[("M2_close1545_open", 0.0)], p)
    rep["attribution_M3_vs_M4"] = attribution(res[("M3_closeFinal_closeFinal", 0.0)], res[("M4_close1545_close1545", 0.0)], p)
    m0, m2 = rep["M0_open_open"]["base"], rep["M2_close1545_open"]["base"]
    rep["gate"] = {"M2_minus_M0_sharpe": m2["sharpe"] - m0["sharpe"],
                   "halves": {k: m2[k] - m0[k] for k in ("sharpe_2016_20", "sharpe_2021_26")},
                   "pass": bool(m2["sharpe"] >= m0["sharpe"] + 0.10 and m2["sharpe_2016_20"] >= m0["sharpe_2016_20"]
                                and m2["sharpe_2021_26"] >= m0["sharpe_2021_26"])}
    navs = pd.DataFrame({f"{n}|{int(b)}": pd.Series(r.nav, index=r.dates) for (n, b), r in res.items()})
    navs.to_parquet(q.OUT / "navs_exact.parquet")
    (q.OUT / "exact.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("coverage", "attribution_M1_vs_M2", "attribution_M3_vs_M4", "gate")}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
