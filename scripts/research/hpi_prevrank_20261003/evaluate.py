"""Evaluate HPI current vs previous-session turnover rank (SPEC_FROZEN.md)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[3] / "results/research/hpi_prevrank_20261003"


def load(arm):
    s = pd.read_csv(OUT / f"nav_{arm}.csv", index_col=0, parse_dates=True)["total_value"].astype(float)
    return s.pct_change().dropna()


def stats(r):
    nav = (1 + r).cumprod()
    return {"cagr": float(nav.iloc[-1] ** (252 / len(r)) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "max_dd": float((nav / nav.cummax() - 1).min())}


def boot(a, b, n=2000, block=20, seed=0):
    a, b = a.to_numpy(), b.to_numpy()
    T, k = len(a), len(a) // block
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for j in range(n):
        idx = (rng.integers(0, T - block, size=k)[:, None] + np.arange(block)).ravel()
        d[j] = (a[idx].mean() / a[idx].std() - b[idx].mean() / b[idx].std()) * np.sqrt(252)
    return float((d > 0).mean()), [float(np.percentile(d, 5)), float(np.percentile(d, 95))]


cur, prev = load("current"), load("prev")
idx = cur.index.intersection(prev.index)
cur, prev = cur.loc[idx], prev.loc[idx]
tx = {a: pd.read_csv(OUT / f"transactions_{a}.csv", parse_dates=["bar"]) for a in ("current", "prev")}
ent = {a: set(zip(t.loc[t.amount > 0, "bar"], t.loc[t.amount > 0, "asset"])) for a, t in tx.items()}
rep = {"window": [str(idx[0].date()), str(idx[-1].date())]}
for name, (a, b) in {"full": (None, None), "2004_14": (None, "2014-12-31"), "2015_26": ("2015-01-01", None)}.items():
    rep[name] = {"current": stats(cur.loc[a:b]), "prev": stats(prev.loc[a:b])}
    rep[name]["d_sharpe"] = rep[name]["prev"]["sharpe"] - rep[name]["current"]["sharpe"]
rep["boot_p_prev_better"], rep["boot_d_sharpe_5_95"] = boot(prev, cur)
rep["entries"] = {"current": len(ent["current"]), "prev": len(ent["prev"]), "common": len(ent["current"] & ent["prev"])}
rep["corr_daily"] = float(cur.corr(prev))
rep["pass"] = bool(rep["full"]["d_sharpe"] > 0 and rep["2004_14"]["d_sharpe"] > 0 and rep["2015_26"]["d_sharpe"] > 0 and rep["boot_p_prev_better"] >= 0.90)
(OUT / "evaluation.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=1))
