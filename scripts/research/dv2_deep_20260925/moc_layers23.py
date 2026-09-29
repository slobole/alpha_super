"""MOC layers 2 and 3 (research-only).

Layer 2 (exact, 2016-01-04 -> last Alpaca session): decision state of day t at 15:45 from Alpaca SIP bars,
mapped onto Norgate adjusted prices by ratio to the official close (adjustments cancel):
    P_t  = Close_t * p1545 / close_official
    Hd_t = min(High_t, Close_t * h1545 / close_official),  Ld_t = max(Low_t, Close_t * l1545 / close_official)
Symbols without Alpaca data that day keep the final-close state (counted).
Fill: official close (Norgate Close_t) with the same 2.5 bps slippage and commissions.

Layer 3 (model): every stock-day's 15:45 state is drawn from the layer-2 sample of stock-days in the same
bin = (20-day volatility tercile) x (final-bar close-location quintile), as volatility-scaled moves:
    zP = (P/C - 1)/sigma,  zH = (Hd/H - 1)/sigma,  zL = (Ld/L - 1)/sigma
and re-applied with the target's own sigma. Validation: layer 3 on 2016+ must match layer 2
(CAGR within 1pp, daily return correlation >= 0.9) before its 2000-2015 result is used.
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
import phase5_finalists as f5  # noqa: E402

OUT = batch.OUT
ALP = OUT / "alpaca" / "sessions"
RULES = {k: v for k, v in f5.FINALISTS.items() if k in __import__("os").environ.get("MOC_RULES", "F0_floor,F1_floor_adv,F3_E_vote,F4_w252,wired").split(",")}


def load_alpaca(p: rp.Panel):
    frames = [pd.read_csv(f) for f in sorted(ALP.glob("*.csv.gz"))]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["p1545"])
    col = {s: i for i, s in enumerate(p.symbols)}
    df["col"] = df["norgate_symbol"].map(col)
    df["row"] = p.dates.get_indexer(pd.to_datetime(df["date"]))
    df = df.dropna(subset=["col"])
    df = df[df["row"] >= 0]
    df["col"] = df["col"].astype(int)
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    raw = np.asarray(p.RAW)[r, c]
    oc = df["close_official"].to_numpy()
    df["official_vs_norgate"] = oc / raw - 1
    denom = np.where(np.isfinite(oc) & (oc > 0), oc, raw)
    df["rP"] = df["p1545"].to_numpy() / denom
    df["rH"] = df["h1545"].to_numpy() / denom
    df["rL"] = df["l1545"].to_numpy() / denom
    return df


def exact_state(p, df):
    C, H, L = np.asarray(p.C), np.asarray(p.H), np.asarray(p.L)
    P, Hd, Ld = C.copy(), H.copy(), L.copy()
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    P[r, c] = C[r, c] * df["rP"].to_numpy()
    Hd[r, c] = np.fmin(H[r, c], C[r, c] * df["rH"].to_numpy())
    Ld[r, c] = np.fmax(L[r, c], C[r, c] * df["rL"].to_numpy())
    Hd = np.fmax(Hd, P)
    Ld = np.fmin(Ld, P)
    return P, Hd, Ld


def bins(p):
    C, H, L = np.asarray(p.C), np.asarray(p.H), np.asarray(p.L)
    sig = pd.DataFrame(C).pct_change().rolling(20).std().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ibs = np.where(H > L, (C - L) / (H - L), 0.5)
    vt = np.full(sig.shape, -1)
    q = np.nanquantile(sig[np.isfinite(sig)], [1 / 3, 2 / 3])
    vt[np.isfinite(sig)] = np.digitize(sig[np.isfinite(sig)], q)
    it = np.clip((ibs * 5).astype(int), 0, 4)
    return sig, vt * 5 + it


def model_state(p, df, sig, b, seed, rows_mask):
    """Resample 15:45 states for every stock-day in rows_mask from the empirical sample, bin by bin."""
    rng = np.random.default_rng(seed)
    C, H, L = np.asarray(p.C), np.asarray(p.H), np.asarray(p.L)
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    s = sig[r, c]
    ok = np.isfinite(s) & (s > 0)
    zP = (df["rP"].to_numpy() - 1)[ok] / s[ok]
    zH = (np.fmin(1.0, df["rH"].to_numpy() * C[r, c] / H[r, c]) - 1)[ok] / s[ok]
    zL = (np.fmax(1.0, df["rL"].to_numpy() * C[r, c] / L[r, c]) - 1)[ok] / s[ok]
    sb = b[r, c][ok]
    P, Hd, Ld = C.copy(), H.copy(), L.copy()
    target = rows_mask[:, None] & np.isfinite(sig) & np.isfinite(C) & (b >= 0)
    for k in np.unique(sb):
        src = np.nonzero(sb == k)[0]
        tr, tc = np.nonzero(target & (b == k))
        if not len(src) or not len(tr):
            continue
        pick = src[rng.integers(0, len(src), size=len(tr))]
        sg = sig[tr, tc]
        P[tr, tc] = C[tr, tc] * (1 + zP[pick] * sg)
        Hd[tr, tc] = H[tr, tc] * (1 + zH[pick] * sg)
        Ld[tr, tc] = L[tr, tc] * (1 + zL[pick] * sg)
    Hd = np.fmax(Hd, P)
    Ld = np.fmin(Ld, P)
    return P, Hd, Ld


def sm(res):
    s = rp.summarize(res)
    return {k: s.get(k) for k in ("cagr", "sharpe", "maxdd", "turnover_x", "P1_sharpe", "P2_sharpe", "P3_sharpe")}


def main():
    p = rp.Panel("sp500")
    df = load_alpaca(p)
    last = p.dates[int(df["row"].max())]
    a0, a1 = "2016-01-04", str(last.date())
    needed = np.load(rp.CACHE_DIR_PATH / "moc_needed.npy")
    have = np.zeros_like(needed)
    have[df["row"].to_numpy(), df["col"].to_numpy()] = True
    m = (p.dates >= a0) & (p.dates <= last)
    rep = {"window": [a0, a1], "sessions": int(df["date"].nunique()), "rows": int(len(df)),
           "needed_coverage": float(have[m][needed[m]].mean()),
           "official_vs_norgate_abs_median_bps": float(np.nanmedian(np.abs(df["official_vs_norgate"])) * 1e4),
           "official_vs_norgate_share_within_10bps": float(np.nanmean(np.abs(df["official_vs_norgate"]) < 0.001)),
           "coverage_by_year": {str(y): float(have[m & (p.dates.year == y)][needed[m & (p.dates.year == y)]].mean()) for y in range(2016, last.year + 1)}}
    P, Hd, Ld = exact_state(p, df)
    exact_dec = rp.PerfectDecision(p, P, Hd, Ld)
    sig, b = bins(p)
    rep["layer2"], rep["layer3_validation"], rep["layer3_2000_2015"] = {}, {}, {}
    rets = {}
    for name, rule in RULES.items():
        nxt = rp.run(p, rule, start=a0, end=a1)
        perf = rp.run(p, rule.with_(timing="moc"), start=a0, end=a1)
        ex = rp.run(p, rule.with_(timing="moc"), start=a0, end=a1, dec=exact_dec)
        rep["layer2"][name] = {"next_open": sm(nxt), "close_perfect": sm(perf), "close_1545_exact": sm(ex),
                               "entries_overlap_exact_vs_perfect": float(len(set(zip(ex.trades.date, ex.trades.asset)) & set(zip(perf.trades.date, perf.trades.asset)))
                                                                          / max(len(perf.trades), 1))}
        rets[name] = rp.daily_returns(ex)
        print(name, "L2", {k: round(v["sharpe"], 3) for k, v in rep["layer2"][name].items() if isinstance(v, dict)}, flush=True)
    # layer 3: validation on 2016+ and the 2000-2015 model, 10 seeds each, for F0 and F1 (and wired)
    for name in ("F0_floor", "F1_floor_adv", "wired"):
        rule = RULES[name].with_(timing="moc")
        val, old = [], []
        for seed in range(int(__import__("os").environ.get("MOC_SEEDS", "10"))):
            mask = np.ones(len(p.dates), dtype=bool)
            Pm, Hm, Lm = model_state(p, df, sig, b, seed, mask)
            dec = rp.PerfectDecision(p, Pm, Hm, Lm)
            rv = rp.run(p, rule, start=a0, end=a1, dec=dec)
            ro = rp.run(p, rule, start="2000-01-03", end="2015-12-31", dec=dec)
            sv = sm(rv)
            sv["corr_with_exact"] = float(pd.concat([rp.daily_returns(rv), rets[name]], axis=1).dropna().corr().iloc[0, 1])
            val.append(sv)
            old.append(sm(ro))
        base_old_next = sm(rp.run(p, RULES[name], start="2000-01-03", end="2015-12-31"))
        base_old_perf = sm(rp.run(p, rule, start="2000-01-03", end="2015-12-31"))
        rep["layer3_validation"][name] = {"model_mean": pd.DataFrame(val).mean().to_dict(), "model_min_max_cagr": [min(v["cagr"] for v in val), max(v["cagr"] for v in val)],
                                          "exact": rep["layer2"][name]["close_1545_exact"]}
        rep["layer3_2000_2015"][name] = {"next_open": base_old_next, "close_perfect": base_old_perf, "close_1545_model_mean": pd.DataFrame(old).mean().to_dict(),
                                         "model_min_max_sharpe": [min(v["sharpe"] for v in old), max(v["sharpe"] for v in old)]}
        print(name, "L3", round(rep["layer3_validation"][name]["model_mean"]["cagr"], 4), round(rep["layer2"][name]["close_1545_exact"]["cagr"], 4), flush=True)
    (OUT / "moc_layers23.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
