"""Phase 2: where does DV2's money come from? (research-only, descriptive; no rule is selected here)

Daily decomposition, with e_{t-1} = position value at close t-1 / NAV_{t-1} and r_m = S&P 500 total return:
    market part_t   = e_{t-1} * r_m,t          (being long the index whenever DV2 is invested = "timing")
    residual_t      = r_DV2,t - market part_t   (stock selection / reversal beyond the index)
Trade level: return = exit fill / entry fill - 1 (fills include slippage); market-adjusted = minus the index
total return from Close_{entry-1} to Close_{exit-1} (the fill opens bracket that span; approximation noted).
Overnight vs daytime for held positions: overnight_i,t = Open_t / Close_{t-1} - 1, daytime_i,t = Close_t / Open_t - 1,
value-weighted by the position at Close_{t-1}.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp  # noqa: E402
import batch  # noqa: E402

OUT = batch.OUT


def positions_matrix(p: rp.Panel, res: rp.Result) -> np.ndarray:
    """shares held at the close of each date (dates x symbols)."""
    t0 = p.dates.get_loc(res.dates[0])
    sh = np.zeros((len(res.dates), p.C.shape[1]))
    col = {s: i for i, s in enumerate(p.symbols)}
    tr = res.trades.copy()
    tr["row"] = p.dates.get_indexer(tr["date"]) - t0
    tr["col"] = tr["asset"].map(col)
    delta = np.zeros_like(sh)
    np.add.at(delta, (tr["row"].to_numpy(), tr["col"].to_numpy()), tr["amount"].to_numpy())
    return np.cumsum(delta, axis=0)


def trade_table(p: rp.Panel, res: rp.Result) -> pd.DataFrame:
    tr = res.trades
    ent = tr[tr.kind == "entry"].set_index("trade_id")
    ex = tr[tr.kind != "entry"].drop_duplicates("trade_id").set_index("trade_id")
    t = ent.join(ex[["date", "price"]], rsuffix="_exit", how="inner")
    spx = pd.Series(p.spx, index=p.dates)
    prev = lambda d: spx.index[spx.index.get_indexer(pd.DatetimeIndex(d)) - 1]
    t["ret"] = t["price_exit"] / t["price"] - 1
    t["mkt"] = spx.reindex(prev(t["date_exit"])).to_numpy() / spx.reindex(prev(t["date"])).to_numpy() - 1
    t["adj"] = t["ret"] - t["mkt"]
    t["days"] = (p.dates.get_indexer(pd.DatetimeIndex(t["date_exit"])) - p.dates.get_indexer(pd.DatetimeIndex(t["date"])))
    n_by_day = t.groupby("date").size()
    t["crowd"] = t["date"].map(n_by_day)
    # signal-day volume spike (news proxy): Volume_p / mean Volume over the 20 sessions before p
    V = pd.DataFrame(p.V)
    vratio = (V / V.shift(1).rolling(20).mean()).to_numpy()
    col = {s: i for i, s in enumerate(p.symbols)}
    rows = p.dates.get_indexer(pd.DatetimeIndex(t["date"])) - 1
    t["vol_spike"] = vratio[rows, t["asset"].map(col).to_numpy()]
    t["gap"] = p.O[rows + 1, t["asset"].map(col).to_numpy()] / p.C[rows, t["asset"].map(col).to_numpy()] - 1
    return t


def annual(x: pd.Series) -> float:
    return float(x.mean() * 252)


def analyze(name: str, p: rp.Panel, res: rp.Result) -> dict:
    nav = pd.Series(res.nav, index=res.dates)
    r = nav.pct_change().fillna(0)
    sh = positions_matrix(p, res)
    t0 = p.dates.get_loc(res.dates[0])
    C = np.nan_to_num(p.C[t0:t0 + len(res.dates)])
    O = p.O[t0:t0 + len(res.dates)]
    posval = (sh * C).sum(axis=1)
    expo = pd.Series(posval / nav.to_numpy(), index=res.dates)
    rm = pd.Series(p.spx, index=p.dates).pct_change().reindex(res.dates)
    mkt = expo.shift(1).fillna(0) * rm
    resid = r - mkt
    beta = float(np.cov(r[1:], rm[1:])[0, 1] / np.var(rm[1:], ddof=1))
    # overnight / daytime for positions held at the previous close
    w_prev = np.vstack([np.zeros((1, sh.shape[1])), sh[:-1] * C[:-1]])
    prevC = np.vstack([np.full((1, C.shape[1]), np.nan), C[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        on = np.nan_to_num(O / prevC - 1)
        day = np.nan_to_num(np.where(O > 0, C / O, 1) - 1)
    navp = np.concatenate([[np.nan], nav.to_numpy()[:-1]])
    overnight = pd.Series((w_prev * on).sum(1) / navp, index=res.dates)
    daytime = pd.Series((w_prev * (1 + on) * day).sum(1) / navp, index=res.dates)
    # regimes
    spx = pd.Series(p.spx, index=p.dates)
    above = (spx > spx.rolling(200).mean()).shift(1).reindex(res.dates)
    rv = spx.pct_change().rolling(20).std().shift(1).reindex(res.dates)
    terc = pd.qcut(rv, 3, labels=["lowvol", "midvol", "highvol"])
    trades = trade_table(p, res)
    yearly = pd.DataFrame({"dv2": r, "mkt_part": mkt, "resid": resid}).groupby(r.index.year).apply(lambda d: (1 + d).prod() - 1)
    out = {
        "name": name, "beta_daily": beta, "avg_exposure": float(expo.mean()),
        "total_ann": annual(r), "market_part_ann": annual(mkt), "residual_ann": annual(resid),
        "residual_sharpe": float(resid.mean() / resid.std() * np.sqrt(252)), "market_part_sharpe": float(mkt.mean() / mkt.std() * np.sqrt(252)),
        "overnight_ann": annual(overnight.fillna(0)), "daytime_ann": annual(daytime.fillna(0)),
        "regime_ann": {f"spx_above200={k}": {"dv2": annual(r[above == k]), "resid": annual(resid[above == k]), "share_days": float((above == k).mean())} for k in (True, False)},
        "vol_ann": {str(k): {"dv2": annual(r[terc == k]), "resid": annual(resid[terc == k])} for k in ["lowvol", "midvol", "highvol"]},
        "trades": {"n": int(len(trades)), "mean_ret": float(trades.ret.mean()), "mean_mkt_adj": float(trades.adj.mean()),
                   "median_days": float(trades.days.median()), "win": float((trades.ret > 0).mean()),
                   "worst": float(trades.ret.min()), "cvar5": float(trades.ret[trades.ret <= trades.ret.quantile(0.05)].mean()),
                   "share_entries_crowded5": float((trades.crowd >= 5).mean()),
                   "adj_crowded5": float(trades.adj[trades.crowd >= 5].mean()), "adj_single": float(trades.adj[trades.crowd <= 2].mean()),
                   "adj_volspike2": float(trades.adj[trades.vol_spike >= 2].mean()), "adj_novolspike": float(trades.adj[trades.vol_spike < 1.5].mean()),
                   "share_volspike2": float((trades.vol_spike >= 2).mean())},
    }
    yearly.to_csv(OUT / f"phase2_yearly_{name}.csv")
    trades.to_csv(OUT / f"phase2_trades_{name}.csv.gz", index=True)
    return out


def main():
    p = rp.Panel("sp500")
    report = {}
    for name, rule in [("wired", rp.Rule()), ("floor", rp.Rule(floor=True))]:
        res = rp.run(p, rule)
        report[name] = analyze(name, p, res)
        report[name]["stats"] = rp.summarize(res)
    # same-close with the final close known (optimistic MOC bound), entries and exits at Close_t
    for name, rule in [("wired_close_perfect", rp.Rule(timing="moc")), ("floor_close_perfect", rp.Rule(floor=True, timing="moc"))]:
        res = rp.run(p, rule)
        report[name] = {"stats": rp.summarize(res)}
    (OUT / "phase2_mechanism.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
