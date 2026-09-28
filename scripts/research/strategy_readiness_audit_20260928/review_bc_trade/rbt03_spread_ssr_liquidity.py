"""Review BC-trade RBT03: spread proxy, short-sale-restriction (Rule 201) exposure and liquidity trend.

1. Effective spread proxy from daily bars (Abdi & Ranaldo 2017, "CHL"): s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})],
   eta = (ln H + ln L)/2, c = ln Close; averaged per calendar month, s = sqrt(max(avg, 0)). Median monthly s over the
   last 3 years, in bp (full round-trip spread; one side ~ s/2). It is a noisy daily-bar proxy of the QUOTED/effective
   spread, not the opening-auction cost.
2. Rule 201 (SSR): a short sale that opens or increases a short and fills at Open_b is exposed when the security's
   Low_(b-1) <= 0.9 * Close_(b-2) (the restriction then covers the rest of b-1 and all of b). Counted for every CTC
   short-increasing fill and every EOM TLT short entry.
3. Liquidity trend: median native Turnover over the last 3y and the last 1y (USD/day).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402

RES = REPO / "results/research/strategy_readiness_audit_20260928"
OUT = RES / "review_bc_trade"
END = "2026-09-25"
L3Y = pd.Timestamp("2023-09-25")
L1Y = pd.Timestamp("2025-09-25")
SYMS = ["VIXM", "FXF", "FXE", "FXY", "BTAL", "KIE", "IHI", "IBB", "IGV", "SOXX", "XLC", "VOX", "IYR", "XLE", "XLU",
        "XLP", "XLK", "IEF", "TLT", "SPY", "QQQ", "UUP", "DBC", "USO", "SLV", "HYG", "LQD", "SHY", "BIL", "GLD", "VTI",
        "SMH", "ITA", "IYT", "XSD", "XPH", "KBE", "XME", "QLD", "SSO", "EEM", "EFA", "IWM", "TQQQ"]


def load(sym: str, start: str = "2001-01-01") -> pd.DataFrame:
    return load_price_timeseries(sym, start_date_str=start, end_date_str=END)


def chl_bp(df: pd.DataFrame) -> float:
    d = df[(df.index >= L3Y) & (df["Volume"] > 0)]
    c, h, l = np.log(d["Close"]), np.log(d["High"]), np.log(d["Low"])
    eta = (h + l) / 2.0
    x = 4.0 * (c - eta) * (c - eta.shift(-1))
    m = x.groupby(x.index.to_period("M")).mean()
    s = np.sqrt(m.clip(lower=0.0))
    return round(1e4 * float(s.median()), 1)


def ssr_flag(df: pd.DataFrame, bar: pd.Timestamp) -> bool | None:
    idx = df.index
    if bar not in idx:
        return None
    i = idx.get_loc(bar)
    if i < 2:
        return None
    return bool(df["Low"].iloc[i - 1] <= 0.9 * df["Close"].iloc[i - 2])


def main() -> None:
    out: dict = {"spread_chl_bp_median_monthly_last3y": {}, "turnover_median_usd": {}}
    cache: dict[str, pd.DataFrame] = {}
    for s in SYMS:
        try:
            df = load(s)
        except Exception as e:  # noqa: BLE001
            out["spread_chl_bp_median_monthly_last3y"][s] = f"error {e}"
            continue
        cache[s] = df
        out["spread_chl_bp_median_monthly_last3y"][s] = chl_bp(df)
        t = df["Turnover"] if "Turnover" in df else df["Close"] * df["Volume"]
        out["turnover_median_usd"][s] = {"last3y": round(float(t[t.index >= L3Y].median()), 0),
                                         "last1y": round(float(t[t.index >= L1Y].median()), 0)}
        print(s, out["spread_chl_bp_median_monthly_last3y"][s], out["turnover_median_usd"][s], flush=True)

    # CTC short-increasing fills
    ctc = pickle.load(open(RES / "tierc_hedge/_cache/ctc_baseline.pkl", "rb"))["tx"].copy()
    ctc["bar"] = pd.to_datetime(ctc["bar"])
    ctc = ctc.sort_values(["bar", "trade_id"])
    ctc["pos_after"] = ctc.groupby("asset")["amount"].cumsum()
    ctc["pos_before"] = ctc["pos_after"] - ctc["amount"]
    short_inc = ctc[(ctc["amount"] < 0) & (ctc["pos_after"] < -1e-9) & (ctc["pos_after"] < ctc["pos_before"])].copy()
    flags = []
    for a, b in zip(short_inc["asset"], short_inc["bar"]):
        df = cache.get(a) if a in cache else load(a)
        cache[a] = df
        flags.append(ssr_flag(df, b))
    short_inc["ssr"] = flags
    ex = short_inc[short_inc["ssr"] == True]  # noqa: E712
    out["ctc_short_increasing_fills"] = {
        "n": int(len(short_inc)), "n_last3y": int((short_inc["bar"] >= L3Y).sum()),
        "n_ssr_exposed": int(len(ex)),
        "ssr_exposed_rows": [f"{b.date()} {a}" for a, b in zip(ex["asset"], ex["bar"])],
        "by_asset": short_inc["asset"].value_counts().to_dict(),
    }
    # EOM TLT short entries: from the fills parquet (negative amount making a negative position)
    eom = pd.read_parquet(RES / "tierb_etf_mr/fills_eom.parquet")
    out["eom_fills_columns"] = list(map(str, eom.columns))
    bar_col = "bar" if "bar" in eom.columns else eom.columns[0]
    eom[bar_col] = pd.to_datetime(eom[bar_col])
    eom = eom.sort_values(bar_col)
    if {"asset", "amount"} <= set(eom.columns):
        eom["pos_after"] = eom.groupby("asset")["amount"].cumsum()
        tlt_short = eom[(eom["asset"] == "TLT") & (eom["amount"] < 0) & (eom["pos_after"] < 0)]
        tdf = cache.get("TLT")
        tflags = [ssr_flag(tdf, b) for b in tlt_short[bar_col]]
        out["eom_tlt_short_entries"] = {"n": int(len(tlt_short)), "n_ssr_exposed": int(sum(bool(f) for f in tflags if f is not None))}
        tlt = tdf
        out["tlt_days_low_le_90pct_prev_close_since_2002"] = int((tlt["Low"] <= 0.9 * tlt["Close"].shift(1)).sum())
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("ctc", "eom", "tlt"))}, indent=1, default=str))
    (OUT / "rbt03_spread_ssr_liquidity.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
