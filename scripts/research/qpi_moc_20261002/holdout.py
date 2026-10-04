"""Amendment L2: 1995-2003 holdout + 2004-2026 paired bootstrap for the QPI near-misses (research-only)."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q  # noqa: E402
import loc_lib as ll  # noqa: E402

H0, H1, B0, B1 = "1995-01-03", "2003-12-31", "2004-01-02", "2026-09-24"
DTB3 = q.REPO.parent / "1_data" / "DTB3.csv"


def tbill(idx):
    d = pd.read_csv(DTB3, parse_dates=["observation_date"], na_values=["."]).set_index("observation_date")["DTB3"].dropna()
    return (d.reindex(d.index.union(idx)).ffill().shift(1).reindex(idx) / 100.0).fillna(0.0)


def rets(res, cash=False):
    r = pd.Series(res.nav, index=res.dates).pct_change().fillna(0.0)
    if cash:
        g = pd.Series(res.diag["gross_ser"], index=res.dates).shift(1).fillna(0.0)
        r = r + (1.0 - g).clip(lower=0) * tbill(res.dates) / 252.0
    return r


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(252))


def stats(r):
    nav = (1 + r).cumprod()
    yrs = len(r) / 252.0
    return {"cagr": float(nav.iloc[-1] ** (1 / yrs) - 1), "sharpe": sharpe(r), "maxdd": float((nav / nav.cummax() - 1).min())}


def boot(ra, rb, n=2000, block=20, seed=0):
    a, b = ra.to_numpy(), rb.to_numpy()
    T = len(a)
    rng = np.random.default_rng(seed)
    k = T // block
    out = np.empty(n)
    for j in range(n):
        st = rng.integers(0, T - block, size=k)
        idx = (st[:, None] + np.arange(block)).ravel()
        x, y = a[idx], b[idx]
        out[j] = x.mean() / x.std() - y.mean() / y.std()
    return float(np.mean(out > 0)), float(np.percentile(out * np.sqrt(252), 5)), float(np.percentile(out * np.sqrt(252), 95))


def main():
    p = q.rp.Panel("sp500")
    fin = q.Features(p)
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    specs = {"base": ll.Spec("base", fin.entry, turn, fin.exit),
             "prevrank": ll.Spec("prevrank", fin.entry, q.lag(turn), fin.exit),
             "lim_prevclose_0.5pct": ll.Spec("l05", fin.entry, turn, fin.exit, open_limit_k=0.005, open_anchor="prevclose"),
             "lim_prevclose_1pct": ll.Spec("l1", fin.entry, turn, fin.exit, open_limit_k=0.01, open_anchor="prevclose")}
    R = {n: {"hold": ll.run(p, s, H0, H1), "main": ll.run(p, s, B0, B1)} for n, s in specs.items()}
    rep = {}
    for n in specs:
        rep[n] = {}
        for cash in (False, True):
            tag = "cash_tbill" if cash else "cash_zero"
            h, m = rets(R[n]["hold"], cash), rets(R[n]["main"], cash)
            hb, mb = rets(R["base"]["hold"], cash), rets(R["base"]["main"], cash)
            d = {"holdout": stats(h), "main_2004_26": stats(m), "gross_mean_main": float(np.mean(R[n]["main"].diag["gross_ser"]))}
            if n != "base":
                pr, lo, hi = boot(m, mb)
                d["holdout_d_sharpe"] = d["holdout"]["sharpe"] - stats(hb)["sharpe"]
                d["boot_p_better"], d["boot_d_sharpe_5_95"] = pr, [lo, hi]
                d["pass"] = bool(d["holdout_d_sharpe"] >= 0.05 and pr >= 0.90)
            rep[n][tag] = d
        print(n, json.dumps({t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k not in ("holdout", "main_2004_26")} | {"hold_sh": round(x["holdout"]["sharpe"], 3), "main_sh": round(x["main_2004_26"]["sharpe"], 3)} for t, x in rep[n].items()}), flush=True)
    (q.OUT / "holdout_L2.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
