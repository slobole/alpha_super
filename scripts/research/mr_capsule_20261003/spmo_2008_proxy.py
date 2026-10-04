"""What would SPMO parking have done in 2008? (owner question 2026-10-03; exploratory, after the parking decision)

SPMO starts 2015-10, so 2008 needs a proxy. Two, both validated against real SPMO on 2015-11 -> 2026:
  PDP      Invesco DWA Momentum ETF (real fund, from 2007-03; total return)
  SYNTH    S&P 500 Momentum rebuilt on the PIT S&P 500 panel: at the last session of February and August, rank members by
           risk-adjusted momentum (price change from t-252 to t-21 divided by the daily-return std over the same window),
           keep the top 100, weight by score x ADV63 (a market-cap stand-in; no shares-outstanding data here), cap 9%,
           hold to the next rebalance; daily total return with dividends
Each proxy gets the same 8% volatility target as SPMO parking (weight = min(1, 8% / 20-day realised vol), rest T-bills).
Spliced series: proxy before 2015-11-02, real SPMO from then on.
Combinations: T-bills both; DV2 -> momentum, HPI -> T-bills (the chosen one); momentum in both pods.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

cp, fp, rs, npc, tbc, rp = ev.cp, ev.fp, ev.rs, ev.npc, ev.tbc, ev.cp.rp
OUT = cp.OUT
SPLICE = "2015-11-02"
WINDOWS = {"GFC (2007-10-09 → 2009-03-09)": ("2007-10-09", "2009-03-09"), "Calendar 2008": ("2008-01-01", "2008-12-31"),
           "Momentum crash (2009-03-09 → 2009-06-01)": ("2009-03-09", "2009-06-01"), "2007–2009": ("2007-01-01", "2009-12-31")}


def synth_momentum(p) -> pd.Series:
    C, D, M = np.asarray(p.C), np.asarray(p.DIV), np.asarray(p.member)
    dates = p.dates
    ret = pd.DataFrame(C).pct_change().to_numpy() + np.nan_to_num(D) / np.vstack([np.full((1, C.shape[1]), np.nan), C[:-1]])
    adv = np.asarray(rp.adv63(p))
    px = pd.DataFrame(C)
    mom = (px.shift(21) / px.shift(252) - 1.0).to_numpy()
    vol = pd.DataFrame(ret).rolling(231, min_periods=200).std().shift(21).to_numpy() * np.sqrt(252)
    score = mom / vol
    month = dates.month
    last_of_month = np.r_[month[1:] != month[:-1], True]
    rebal = last_of_month & np.isin(month, (2, 8))
    w = np.zeros(C.shape[1])
    out = np.full(len(dates), np.nan)
    for t in range(1, len(dates)):
        r = ret[t]
        if w.sum() > 0:
            rr = np.where(np.isfinite(r), r, 0.0)
            out[t] = float(np.sum(w * rr))
            w = w * (1 + rr)
            w = w / w.sum()
        if rebal[t]:
            ok = M[t] & np.isfinite(score[t]) & np.isfinite(adv[t]) & (adv[t] > 0)
            idx = np.nonzero(ok)[0]
            if len(idx) >= 100:
                top = idx[np.argsort(-score[t, idx])[:100]]
                z = score[t, top] - score[t, top].min() + 1e-6
                raw = z * adv[t, top]
                ww = raw / raw.sum()
                for _ in range(10):
                    over = ww > 0.09
                    if not over.any():
                        break
                    excess = (ww[over] - 0.09).sum()
                    ww[over] = 0.09
                    ww[~over] += excess * ww[~over] / ww[~over].sum()
                w = np.zeros(C.shape[1])
                w[top] = ww
    return pd.Series(out, index=dates).fillna(0.0)


def voltarget(r: pd.Series, rate: pd.Series) -> pd.Series:
    rv = r.rolling(20).std().shift(1) * np.sqrt(252)
    w = (0.08 / rv).clip(upper=1).fillna(0.0)
    return w * r + (1 - w) * rate.reindex(r.index).fillna(0.0), w


def main():
    comp = pd.read_parquet(OUT / "components.parquet")
    idx = pd.DatetimeIndex(comp.index)
    rate = rs.cash_rate(idx)
    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(idx)
    pdp = npc.load_total_return_ret_ser("PDP", "PDP").reindex(idx)
    p = rp.Panel("sp500")
    syn = synth_momentum(p).reindex(idx)
    rep = {"validation_vs_spmo_2015_11_on": {}}
    v0 = SPLICE
    for name, s in (("PDP", pdp), ("SYNTH", syn)):
        both = pd.concat([s, spmo], axis=1, keys=["proxy", "spmo"]).loc[v0:"2026-09-24"].dropna()
        rep["validation_vs_spmo_2015_11_on"][name] = {"corr_daily": float(both.corr().iloc[0, 1]),
                                                      "cagr_proxy": fp.stats(both["proxy"])["cagr"], "cagr_spmo": fp.stats(both["spmo"])["cagr"],
                                                      "vol_proxy": float(both["proxy"].std() * np.sqrt(252)), "vol_spmo": float(both["spmo"].std() * np.sqrt(252))}
    # momentum itself in 2008 (no vol target), for context
    spx = npc.load_spy_tr_ret_ser().reindex(idx)
    rep["raw_returns"] = {w: {"PDP": float((1 + pdp.loc[a:b]).prod() - 1), "SYNTH": float((1 + syn.loc[a:b]).prod() - 1), "SPY": float((1 + spx.loc[a:b]).prod() - 1)}
                          for w, (a, b) in WINDOWS.items()}

    def parked(n, pk):
        return comp[f"{n}|engine|base"] + comp[f"{n}|engine|cw"] * pk

    taa, L = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    gate = pd.Series(cp.gate_on(idx), index=idx)
    rep["context_2008"] = {"gate_open_share_2008": float(gate.loc["2008"].mean()),
                           "DV2_idle_cash_2008": float(comp["DV2-G|engine|cw"].loc["2008"].mean()), "HPI_idle_cash_2008": float(comp["HPI-G|engine|cw"].loc["2008"].mean())}
    rep["proxies"] = {}
    for name, proxy in (("PDP", pdp), ("SYNTH", syn)):
        spliced = proxy.where(idx < pd.Timestamp(SPLICE), spmo).fillna(0.0)
        mom, wv = voltarget(spliced, rate)
        rep["context_2008"][f"{name}_voltarget_weight_2008"] = float(wv.loc["2008"].mean())
        start = "2007-04-02" if name == "PDP" else "2004-01-05"
        combos = {"T-bills / T-bills": ("tbills", "tbills"), "DV2 momentum / HPI T-bills (chosen)": ("mom", "tbills"), "momentum / momentum": ("mom", "mom")}
        pk = {"tbills": rate, "mom": mom}
        res = {}
        for lab, (a, b) in combos.items():
            cap = ev.capsule({"DV2": parked("DV2-G", pk[a]), "HPI": parked("HPI-G", pk[b])}, {"DV2": 0.5, "HPI": 0.5}, start=start, end="2026-09-24")
            bk = tbc.book_window_return_ser({"taa": taa, "L": L, "X": cap}, npc.CANDIDATE_WEIGHT_DICT, "2008-03-04", "2026-08-19")
            d = {"capsule_full": fp.stats(cap), "capsule_money_full": float(100_000 * (1 + cap).prod()),
                 "book_full": tbc.metric_dict(bk), "book_money_full": float(100_000 * (1 + bk).prod())}
            for w, (a0, b0) in WINDOWS.items():
                seg = cap.loc[a0:b0]
                if len(seg) == 0:
                    continue
                nav = (1 + seg).cumprod()
                d[w] = {"capsule_ret": float(nav.iloc[-1] - 1), "capsule_dd": float((nav / nav.cummax() - 1).min())}
                bseg = bk.loc[a0:b0]
                if len(bseg):
                    bn = (1 + bseg).cumprod()
                    d[w].update({"book_ret": float(bn.iloc[-1] - 1), "book_dd": float((bn / bn.cummax() - 1).min())})
            res[lab] = d
        rep["proxies"][name] = {"start": start, "results": res}
    (OUT / "spmo_2008_proxy.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print("VALIDATION vs real SPMO (2015-11 → 2026):", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rep["validation_vs_spmo_2015_11_on"].items()}))
    print("RAW momentum returns (no vol target):", json.dumps({w: {k: round(v, 3) for k, v in d.items()} for w, d in rep["raw_returns"].items()}))
    print("CONTEXT 2008:", {k: round(v, 3) for k, v in rep["context_2008"].items()})
    for name, blk in rep["proxies"].items():
        print(f"\n== proxy {name} (capsule from {blk['start']})")
        for lab, d in blk["results"].items():
            print(f"  {lab:<38} capsule ${d['capsule_money_full']:>10,.0f} Sh {d['capsule_full']['sharpe']:.3f} DD {d['capsule_full']['max_dd']:.3f} | book 2008-26 ${d['book_money_full']:>10,.0f} Sh {d['book_full']['sharpe']:.3f} DD {d['book_full']['max_dd']:.3f}")
            for w in WINDOWS:
                if w in d:
                    x = d[w]
                    print(f"      {w:<44} capsule {x['capsule_ret']*100:6.1f}% (DD {x['capsule_dd']*100:5.1f}%)" + (f" | book {x['book_ret']*100:6.1f}% (DD {x['book_dd']*100:5.1f}%)" if 'book_ret' in x else ""))


if __name__ == "__main__":
    main()
