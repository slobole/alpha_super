"""LOC / limit-below-open study (SPEC_LOC.md). Usage: python run_loc.py validate|main"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q  # noqa: E402
import loc_lib as ll  # noqa: E402
import run_test as rt  # noqa: E402

rp = q.rp
A0, A1, FULL0 = rt.A0, rt.A1, rt.FULL0
WIRED = rp.Rule()


def state_1545(p):
    import moc_layers23 as ml
    frames = [pd.read_csv(f) for d in (ml.ALP, q.OUT / "alpaca" / "sessions") for f in sorted(d.glob("*.csv.gz"))]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["p1545"]).drop_duplicates(["date", "norgate_symbol"], keep="last")
    col = {s: i for i, s in enumerate(p.symbols)}
    df["col"] = df["norgate_symbol"].map(col)
    df["row"] = p.dates.get_indexer(pd.to_datetime(df["date"]))
    df = df.dropna(subset=["col"])
    df = df[df["row"] >= 0]
    df["col"] = df["col"].astype(int)
    raw = np.asarray(p.RAW)[df["row"].to_numpy(), df["col"].to_numpy()]
    oc = df["close_official"].to_numpy()
    den = np.where(np.isfinite(oc) & (oc > 0), oc, raw)
    df["rP"], df["rH"], df["rL"] = df["p1545"] / den, df["h1545"] / den, df["l1545"] / den
    P, Hd, Ld = ml.exact_state(p, df)
    have = np.zeros(p.C.shape, dtype=bool)
    have[df["row"].to_numpy(), df["col"].to_numpy()] = True
    return {"P": P, "H": Hd, "L": Ld, "have": have}


def qpi_specs(p, F, S, Q):
    fin = F["final"]
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    specs = {"Q_base_open": ll.Spec("Q_base_open", fin.entry, turn, fin.exit),
             "Q_S1_prevrank_open": ll.Spec("Q_S1_prevrank_open", fin.entry, q.lag(turn), fin.exit)}
    for pol in ("A", "B"):
        for xs in ("xopen", "xloc"):
            for b in (0.0, 5.0):
                n = f"Q_LOC{pol}_{xs}" + ("_cons" if b else "")
                specs[n] = ll.Spec(n, None, None, fin.exit, loc_buy=Q["buy"], loc_P=S["P"], loc_policy=pol,
                                   loc_sell=Q["sell"] if xs == "xloc" else None, buffer_bps=b, extra={"lb": Q["lb"]})
    for anchor in ("open", "prevclose"):
        for k in (0.0025, 0.005, 0.01):
            n = f"Q_S2_{anchor}_{k * 100:g}pct"
            specs[n] = ll.Spec(n, fin.entry, turn, fin.exit, open_limit_k=k, open_anchor=anchor)
    return specs


def dv2_specs(p, S, D):
    entry, score, ex = ll.dv2_masks(p, WIRED)
    specs = {"D_base_open": ll.Spec("D_base_open", entry, score, ex)}
    for pol in ("A", "B"):
        for b in (0.0, 5.0):
            n = f"D_LOC{pol}_xopen" + ("_cons" if b else "")
            specs[n] = ll.Spec(n, None, None, ex, loc_buy=D["buy"], loc_P=S["P"], loc_policy=pol, buffer_bps=b, extra={"lb": D["lb"]})
    for anchor in ("open", "prevclose"):
        for k in (0.0025, 0.005, 0.01):
            n = f"D_S2_{anchor}_{k * 100:g}pct"
            specs[n] = ll.Spec(n, entry, score, ex, open_limit_k=k, open_anchor=anchor)
    return specs


def validate(p, F, S, Q, D):
    out = {}
    m = (p.dates >= A0)[:, None]
    # QPI threshold: rule holds at the limit (where the QPI bound binds) and fails 5 bps above it
    C = np.asarray(p.C)
    rstar_px = Q["buy"]
    Pq = np.where(np.isfinite(rstar_px) & m, rstar_px, C)
    Fq = q.Features(p, Pq, np.fmax(S["H"], Pq), np.fmin(S["L"], Pq))
    cells = np.isfinite(rstar_px) & m
    out["qpi_at_limit_qpi_lt30_share"] = float(np.mean(Fq.qpi[cells] < 30))
    Pq2 = np.where(cells, rstar_px * 1.0005, C)
    Fq2 = q.Features(p, Pq2, np.fmax(S["H"], Pq2), np.fmin(S["L"], Pq2))
    with np.errstate(invalid="ignore"):
        ibs_lim = S["L"] + 0.10 * (S["H"] - S["L"])
        qpi_binds = cells & (rstar_px < ibs_lim * (1 - 1e-6))
    out["qpi_5bps_above_fails_share_where_qpi_binds"] = float(np.mean((Fq2.qpi[qpi_binds] >= 30) | (Fq2.r3[qpi_binds] >= 0)))
    # DV2 threshold
    dcells = np.isfinite(D["buy"]) & m
    Pd = np.where(dcells, D["buy"], C)
    dec = rp.PerfectDecision(p, Pd, np.fmax(S["H"], Pd), np.fmin(S["L"], Pd))
    x = dec.dvpct(2, 126)
    out["dv2_at_limit_lt10_share"] = float(np.mean(x[dcells] < 10))
    Pd2 = np.where(dcells, D["buy"] * 1.0005, C)
    x2 = rp.PerfectDecision(p, Pd2, np.fmax(S["H"], Pd2), np.fmin(S["L"], Pd2)).dvpct(2, 126)
    out["dv2_5bps_above_ge10_share"] = float(np.mean(x2[dcells] >= 10))
    # runner parity
    r_q = ll.run(p, ll.Spec("x", F["final"].entry, p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V)), F["final"].exit), A0, A1)
    r_q0 = q.run(p, rt.MODES["M0_open_open"], F, A0, A1)
    out["qpi_runner_vs_qpi_lib_sharpe"] = [q.summarize(r_q)["sharpe"], q.summarize(r_q0)["sharpe"]]
    entry, score, ex = ll.dv2_masks(p, WIRED)
    r_d = ll.run(p, ll.Spec("d", entry, score, ex), A0, A1)
    r_d0 = rp.run(p, WIRED, start=A0, end=A1)
    out["dv2_runner_vs_replica_sharpe"] = [q.summarize(r_d)["sharpe"], rp.summarize(r_d0)["sharpe"]]
    print(json.dumps(out, indent=1, default=float))
    return out


def explore(p, F, S, Q, D):
    """EXPLORATORY (post-result, labelled): where do the LOC arms lose, and do LOC exits alone help QPI?"""
    fin = F["final"]
    Sf = {"P": np.asarray(p.C), "H": np.asarray(p.H), "L": np.asarray(p.L)}
    Qf, Df = ll.qpi_loc(p, Sf), ll.dv2_loc(p, Sf, WIRED)
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    entry, score, ex = ll.dv2_masks(p, WIRED)
    specs = {
        "X_Q_LOCA_xloc_finalstate": ll.Spec("x", None, None, fin.exit, loc_buy=Qf["buy"], loc_P=Sf["P"], loc_policy="A", loc_sell=Qf["sell"]),
        "X_Q_LOCA_xopen_finalstate": ll.Spec("x", None, None, fin.exit, loc_buy=Qf["buy"], loc_P=Sf["P"], loc_policy="A"),
        "X_D_LOCA_xopen_finalstate": ll.Spec("x", None, None, ex, loc_buy=Df["buy"], loc_P=Sf["P"], loc_policy="A"),
        "X_D_LOCB_xopen_finalstate": ll.Spec("x", None, None, ex, loc_buy=Df["buy"], loc_P=Sf["P"], loc_policy="B"),
        "X_Q_openEntry_xloc": ll.Spec("x", fin.entry, turn, fin.exit, loc_sell=Q["sell"]),
        "X_Q_openEntry_xloc_cons": ll.Spec("x", fin.entry, turn, fin.exit, loc_sell=Q["sell"], buffer_bps=5.0),
        "X_Q_openEntry_xloc_finalstate": ll.Spec("x", fin.entry, turn, fin.exit, loc_sell=Qf["sell"]),
    }
    rep = {}
    for n, sp in specs.items():
        rep[n] = {"alpaca_window": ll.summarize(ll.run(p, sp, A0, A1))}
        if "openEntry" in n and "finalstate" not in n:
            rep[n]["full_2004_fallback_final_state"] = ll.summarize(ll.run(p, sp, FULL0, A1))
        print(n, round(rep[n]["alpaca_window"]["sharpe"], 3), flush=True)
    return rep


def main(phase):
    p = rp.Panel("sp500")
    F = {"final": q.Features(p)}
    S = state_1545(p)
    Q = ll.qpi_loc(p, S)
    D = ll.dv2_loc(p, S, WIRED)
    if phase == "explore":
        (q.OUT / "loc_explore.json").write_text(json.dumps(explore(p, F, S, Q, D), indent=2, default=float), encoding="utf-8")
        return
    if phase == "validate":
        (q.OUT / "loc_validate.json").write_text(json.dumps(validate(p, F, S, Q, D), indent=2, default=float), encoding="utf-8")
        return
    rep, navs = {}, {}
    for specs in (qpi_specs(p, F, S, Q), dv2_specs(p, S, D)):
        for n, sp in specs.items():
            r = ll.run(p, sp, A0, A1)
            rep[n] = {"alpaca_window": ll.summarize(r)}
            navs[n] = pd.Series(r.nav, index=r.dates)
            if "_S2_" in n or "base" in n or "S1" in n:
                rep[n]["full_2004"] = ll.summarize(ll.run(p, sp, FULL0, A1))
            if sp.loc_buy is not None:
                e = r.trades
                pos = {d: k for k, d in enumerate(p.dates)}
                col = {s: i for i, s in enumerate(p.symbols)}
                rep[n]["entry_coverage_1545"] = float(np.mean([S["have"][pos[d], col[a]] for d, a in zip(e["entry_date"], e["asset"])])) if len(e) else np.nan
            print(n, round(rep[n]["alpaca_window"]["sharpe"], 3), flush=True)
    for fam, base in (("Q_", "Q_base_open"), ("D_", "D_base_open")):
        b = rep[base]["alpaca_window"]
        for n in [k for k in rep if k.startswith(fam) and k != base and not k.endswith("_cons")]:
            a = rep[n]["alpaca_window"]
            cons = rep.get(n + "_cons", {}).get("alpaca_window", a)
            rep[n]["gate"] = {"d_sharpe": a["sharpe"] - b["sharpe"],
                              "d_2016_20": a["sharpe_2016_20"] - b["sharpe_2016_20"],
                              "d_2021_26": a["sharpe_2021_26"] - b["sharpe_2021_26"],
                              "cons_minus_base": cons["sharpe"] - b["sharpe"],
                              "pass": bool(a["sharpe"] >= b["sharpe"] + 0.10 and a["sharpe_2016_20"] >= b["sharpe_2016_20"]
                                           and a["sharpe_2021_26"] >= b["sharpe_2021_26"] and cons["sharpe"] >= b["sharpe"])}
    pd.DataFrame(navs).to_parquet(q.OUT / "navs_loc.parquet")
    (q.OUT / "loc.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
