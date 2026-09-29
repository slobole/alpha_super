"""Exploratory (post-result, labelled): where does the 15:45 penalty come from - entries or exits?

Mode "exit_close":  entries decided after Close_{t-1}, filled at Open_t (as today);
                    exits decided at 15:45 of t (P_t > High_{t-1}), filled at Close_t.
Mode "entry_close": exits decided after Close_{t-1} (Close_{t-1} > High_{t-2}), filled at Open_t (as today);
                    entries decided at 15:45 of t from the 15:45 state, filled at Close_t.
Same costs, sizing (NAV_{t-1} / slots), dividend and missing-data rules as replica.run.
Also: trade-level attribution of the full 15:45 run vs the final-close run (common / 15:45-only / close-only entries).
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
import moc_layers23 as ml  # noqa: E402
import phase2_mechanism as p2  # noqa: E402


def features(p, r, dec, moc):
    N = p.C.shape[1]
    sig, need_sig = rp.oversold_mask_and_feats(p, r, dec if moc else None)
    trend, need_trend = rp.trend_mask(p, r, dec if moc else None)
    score = rp.rank_score(p, r)
    if moc and (r.rank.startswith("natr") or r.rank == "adv"):
        score = np.vstack([np.full((1, N), np.nan), score[:-1]])
    complete = p.valid.copy()
    for a in need_sig + need_trend + [rp.natr(p, 14)]:
        complete &= np.isfinite(a)
    ok = complete & p.member
    if r.floor:
        adv = rp.adv63(p) if not moc else np.vstack([np.full((1, N), np.nan), rp.adv63(p)[:-1]])
        raw = p.RAW if not moc else dec.raw_price()
        comp = ok & np.isfinite(adv)
        med = np.array([np.median(adv[t][comp[t]]) if comp[t].any() else np.nan for t in range(len(p.dates))])
        ok = comp & (raw > 5.0) & (adv > med[:, None])
    return ok & sig & trend & np.isfinite(score), score, rp.exit_mask(p, r, dec if moc else None)


def run_hybrid(p, r, dec, mode, start, end, capital=1_000_000.0):
    dates = p.dates
    t0, t1 = int(dates.searchsorted(pd.Timestamp(start))), int(dates.searchsorted(pd.Timestamp(end), side="right"))
    N = p.C.shape[1]
    e_close, s_close, x_close = features(p, r, dec, moc=False)
    e_dec, s_dec, x_dec = features(p, r, dec, moc=True)
    cash, shares = capital, np.zeros(N)
    nav = np.full(len(dates), np.nan)
    prev_total = capital
    slip = r.slippage
    comm = lambda q: max(r.commission_min, r.commission_per_share * abs(q))
    trades = []
    pending_entries, pending_exits = [], []
    for t in range(t0, t1):
        held = np.nonzero(shares)[0]
        if held.size:
            d = np.where(np.isfinite(p.DIV[t - 1, held]), p.DIV[t - 1, held], 0.0)
            g = shares[held] * d
            cash += float(np.sum(g - np.maximum(g, 0.0) * rp.WITHHOLDING_RATE_FLOAT))
        for i in np.nonzero(shares)[0]:
            if not (np.isfinite(p.O[t, i]) and np.isfinite(p.C[t, i])):
                h = p.C[:t, i]
                last = h[np.isfinite(h)][-1]
                cash -= -shares[i] * last + comm(shares[i])
                shares[i] = 0.0
        cap = prev_total / r.slots
        if mode == "exit_close":
            for i in pending_entries:  # decided at close t-1, filled at the open
                if shares[i] != 0 or not np.isfinite(p.O[t, i]):
                    continue
                q = float(int(cap / p.C[t - 1, i]))
                if q == 0:
                    continue
                px = p.O[t, i] * (1 + slip)
                cash -= q * px + comm(q)
                shares[i] = q
                trades.append((dates[t], p.symbols[i], q, px, "entry"))
            for i in np.nonzero(shares)[0]:  # 15:45 exit decision, filled at the close
                if bool(x_dec[t, i]) and np.isfinite(p.C[t, i]):
                    q = -shares[i]
                    px = p.C[t, i] * (1 - slip)
                    cash -= q * px + comm(q)
                    shares[i] = 0.0
                    trades.append((dates[t], p.symbols[i], q, px, "exit"))
            slots = r.slots - np.count_nonzero(shares)
            cand = np.nonzero(e_close[t])[0]
            order = cand[np.argsort(-s_close[t, cand], kind="stable")]
            pending_entries = [i for i in order if shares[i] == 0][:max(slots, 0)]
        else:  # entry_close
            for i in pending_exits:
                if shares[i] != 0 and np.isfinite(p.O[t, i]):
                    q = -shares[i]
                    px = p.O[t, i] * (1 - slip)
                    cash -= q * px + comm(q)
                    shares[i] = 0.0
                    trades.append((dates[t], p.symbols[i], q, px, "exit"))
            slots = r.slots - np.count_nonzero(shares)
            cand = np.nonzero(e_dec[t])[0]
            order = cand[np.argsort(-s_dec[t, cand], kind="stable")]
            for i in [i for i in order if shares[i] == 0][:max(slots, 0)]:
                if not np.isfinite(p.C[t, i]):
                    continue
                q = float(int(cap / dec.P[t, i]))
                if q == 0:
                    continue
                px = p.C[t, i] * (1 + slip)
                cash -= q * px + comm(q)
                shares[i] = q
                trades.append((dates[t], p.symbols[i], q, px, "entry"))
            pending_exits = [i for i in np.nonzero(shares)[0] if bool(x_close[t, i])]
        held = np.nonzero(shares)[0]
        total = cash + (float(np.sum(shares[held] * p.C[t, held])) if held.size else 0.0)
        nav[t] = total
        prev_total = total
    sl = slice(t0, t1)
    tr = pd.DataFrame(trades, columns=["date", "asset", "amount", "price", "kind"]).assign(commission=0.0)
    return rp.Result(r, dates[sl], nav[sl], np.ones(t1 - t0, dtype=bool), tr)


def attribution(p, r, dec, start, end):
    perf = rp.run(p, r.with_(timing="moc"), start=start, end=end)
    ex = rp.run(p, r.with_(timing="moc"), start=start, end=end, dec=dec)
    tp, te = p2.trade_table(p, perf), p2.trade_table(p, ex)
    kp, ke = set(zip(tp["date"], tp["asset"])), set(zip(te["date"], te["asset"]))
    lab_e = [("common" if k in kp else "1545_only") for k in zip(te["date"], te["asset"])]
    lab_p = [("common" if k in ke else "close_only") for k in zip(tp["date"], tp["asset"])]
    out = {}
    for name, df, lab in (("exact_1545", te, lab_e), ("final_close", tp, lab_p)):
        g = df.assign(lab=lab).groupby("lab")
        out[name] = {k: {"n": int(len(v)), "mean_ret": float(v["ret"].mean()), "mean_mkt_adj": float(v["adj"].mean())} for k, v in g}
    return out


def main():
    p = rp.Panel("sp500")
    df = ml.load_alpaca(p)
    a0, a1 = "2016-01-04", str(p.dates[int(df["row"].max())].date())
    dec = rp.PerfectDecision(p, *ml.exact_state(p, df))
    rep = {}
    for name in ("F0_floor", "F1_floor_adv", "wired"):
        r = ml.RULES[name]
        rep[name] = {}
        for mode in ("exit_close", "entry_close"):
            rep[name][mode] = ml.sm(run_hybrid(p, r, dec, mode, a0, a1))
            rep[name][mode + "_perfect"] = ml.sm(run_hybrid(p, r, rp.PerfectDecision(p), mode, a0, a1))
        rep[name]["attribution"] = attribution(p, r, dec, a0, a1)
        print(name, {m: round(s["sharpe"], 3) for m, s in rep[name].items() if m != "attribution"}, flush=True)
    (batch.OUT / "moc_hybrid.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v["attribution"] for k, v in rep.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
