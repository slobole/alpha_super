"""Tradability (protocol C1-C4) for Industry-ETF DV2 and the TAA 2x / linearity variants. Research-only.

Inputs: the ledgers written by ``etf_bc.py runs`` and ``taa2x_bc.py runs`` (production defaults, USD 100K).

C1  order_frac_i = |amount_i x price_i| / NAV_(t-1); ADV20_{s,t} = median native Norgate Turnover over t-20..t-1;
    participation(C) = order_frac_i x C / ADV20. Last 3 years GOVERN (amendment AM-01); full history is a note.
    Also: share of orders above the house MOO guardrails 0.05% / 0.10% of ADV (alpha/engine/capacity_analysis.py:91-92),
    and the pod size at which the last-3y p99 reaches 5% of ADV and the median reaches 1%.
C2  Whole shares at USD 30K with NOMINAL (Unadjusted) decision closes: shares = floor(w x C / P); error = w - shares P/C.
Per-instrument: inception, current ADV (median daily Turnover, last 1y and last 3y), nominal price today.

Usage: uv run python trade.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402

RES = REPO / "results/research/strategy_readiness_audit_20260928/tierbc_dv2etf_taa2x"
END = pd.Timestamp("2026-09-25")
L3Y = pd.Timestamp("2023-09-25")
CAPS = (30_000.0, 1_000_000.0, 10_000_000.0)
_PX: dict[str, pd.DataFrame] = {}


def px(sym: str) -> pd.DataFrame:
    if sym not in _PX:
        _PX[sym] = load_price_timeseries(sym, start_date_str="1998-01-01", end_date_str=END.date().isoformat())
    return _PX[sym]


def participation(tx: pd.DataFrame, nav: pd.Series) -> pd.DataFrame:
    tx = tx.copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    prev_nav = nav.shift(1)
    rows = []
    adv_cache = {}
    for _, r in tx.iterrows():
        a = str(r["asset"])
        if a not in adv_cache:
            turn = px(a)["Turnover"].astype(float)
            adv_cache[a] = turn.rolling(20, min_periods=10).median().shift(1)  # *** CRITICAL*** t-20..t-1 only
        adv = float(adv_cache[a].get(r["bar"], np.nan))
        frac = abs(float(r["amount"]) * float(r["price"])) / float(prev_nav.get(r["bar"], np.nan))
        rows.append({"bar": r["bar"], "asset": a, "frac": frac, "adv20": adv,
                     "side": "buy" if float(r["amount"]) > 0 else "sell"})
    return pd.DataFrame(rows)


def summarize(p: pd.DataFrame) -> dict:
    out = {}
    for wname, w in (("last3y", p[p["bar"] >= L3Y]), ("full", p)):
        q1 = w["frac"] / w["adv20"]  # participation per USD 1 of pod NAV
        res = {"orders": int(len(w))}
        for c in CAPS:
            q = q1 * c
            res[f"C{int(c)}"] = {"median_pct_adv": round(float(q.median() * 100), 4),
                                 "p99_pct_adv": round(float(q.quantile(0.99) * 100), 4),
                                 "max_pct_adv": round(float(q.max() * 100), 4),
                                 "share_over_moo_soft_0.05pct": round(float((q > 0.0005).mean()), 4),
                                 "share_over_moo_hard_0.10pct": round(float((q > 0.0010).mean()), 4),
                                 "worst_asset": str(w.loc[q.idxmax(), "asset"]) if len(w) else ""}
        res["pod_usd_where_p99_hits_5pct_adv"] = round(float(0.05 / q1.quantile(0.99)), 0)
        res["pod_usd_where_median_hits_1pct_adv"] = round(float(0.01 / q1.median()), 0)
        res["pod_usd_where_p99_hits_moo_hard_0.10pct"] = round(float(0.001 / q1.quantile(0.99)), 0)
        by_asset = (q1 * 1e6 * 100).groupby(w["asset"]).quantile(0.99).sort_values(ascending=False)
        res["p99_pct_adv_at_1m_by_asset_top5"] = {k: round(float(v), 4) for k, v in by_asset.head(5).items()}
        out[wname] = res
    return out


def instrument_table(symbols) -> dict:
    out = {}
    for s in symbols:
        df = px(s)
        turn = df["Turnover"].astype(float)
        out[s] = {"first_bar": df.index[0].date().isoformat(),
                  "median_daily_turnover_musd_last1y": round(float(turn.loc["2025-09-25":].median() / 1e6), 2),
                  "median_daily_turnover_musd_last3y": round(float(turn.loc[L3Y:].median() / 1e6), 2),
                  "median_daily_turnover_musd_full": round(float(turn[turn > 0].median() / 1e6), 2),
                  "nominal_close_2026_09_25": round(float(df["Unadjusted Close"].iloc[-1]), 2),
                  "zero_turnover_sessions_last3y": int((turn.loc[L3Y:] <= 0).sum())}
    return out


def unadj_on(sym: str, d: pd.Timestamp) -> float:
    s = px(sym)["Unadjusted Close"].astype(float).dropna().loc[:d]
    return float(s.iloc[-1]) if len(s) else np.nan


def whole_share_taa(key: str) -> dict:
    """Target weights from the traded rebalance table, decision = session before the rebalance, since 2016."""
    sys.path.insert(0, str(HERE))
    import taa2x_common as tc

    tc.use_cached_loader(True)
    _, rb, _, _ = tc.weight_frames(key, tc.config(key))
    rows = []
    for reb, w in rb.iterrows():
        if reb < pd.Timestamp("2016-01-01"):
            continue
        d = px("SPY").index[px("SPY").index < reb][-1]
        for a, wt in w.items():
            if wt > 1e-12:
                p = unadj_on(a, d)
                for c in (15_000.0, 30_000.0):
                    sh = np.floor(wt * c / p)
                    rows.append({"decision": d, "asset": a, "C": c, "w": wt, "err": wt - sh * p / c, "zero": sh == 0})
    df = pd.DataFrame(rows)
    out = {}
    for c, g in df.groupby("C"):
        cash = g.groupby("decision")["err"].sum()
        out[f"C{int(c)}"] = {"max_name_err_pct_nav": round(float(g["err"].max() * 100), 3),
                             "mean_rounding_cash_pct_nav": round(float(cash.mean() * 100), 3),
                             "max_rounding_cash_pct_nav": round(float(cash.max() * 100), 3),
                             "zero_share_name_decisions": int(g["zero"].sum())}
    return out


def whole_share_etf() -> dict:
    tx = pd.read_csv(RES / "etf" / "baseline_transactions.csv", parse_dates=["bar"])
    buys = tx[tx["amount"] > 0]
    idx = px("SPY").index
    rows = []
    for _, r in buys.iterrows():
        d = idx[idx < r["bar"]][-1]
        p = unadj_on(r["asset"], d)
        for c in (15_000.0, 30_000.0):
            sh = np.floor(0.1 * c / p)
            rows.append({"bar": r["bar"], "asset": r["asset"], "C": c, "err": 0.1 - sh * p / c, "zero": sh == 0, "p": p})
    df = pd.DataFrame(rows)
    out = {}
    for c, g in df.groupby("C"):
        out[f"C{int(c)}"] = {"max_name_err_pct_nav": round(float(g["err"].max() * 100), 3),
                             "mean_name_err_pct_nav": round(float(g["err"].mean() * 100), 3),
                             "last3y_max_name_err_pct_nav": round(float(g.loc[g["bar"] >= L3Y, "err"].max() * 100), 3),
                             "zero_share_entries": int(g["zero"].sum()), "max_nominal_price": round(float(g["p"].max()), 2)}
    return out


def main() -> None:
    out = {}
    # Industry-ETF DV2
    tx = pd.read_csv(RES / "etf" / "baseline_transactions.csv")
    nav = pd.read_csv(RES / "etf" / "baseline_nav.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    p = participation(tx, nav)
    p.to_csv(RES / "etf" / "participation_by_fill.csv", index=False)
    from strategies.dv2.strategy_mr_dv2_industry_etf import INDUSTRY_ETF_SYMBOL_TUPLE

    out["etf_dv2"] = {"participation": summarize(p), "instruments": instrument_table(INDUSTRY_ETF_SYMBOL_TUPLE),
                      "whole_share": whole_share_etf()}
    print("etf_dv2", json.dumps(out["etf_dv2"]["participation"]["last3y"]), flush=True)
    # TAA 2x / linearity
    for key in ("qld_1n", "sso_1n", "btal_qld_1n", "lin_qqq"):
        f = RES / "taa2x" / f"transactions_{key}_100k.csv"
        if not f.exists():
            print("missing", f, flush=True)
            continue
        tx = pd.read_csv(f)
        nav = pd.read_csv(RES / "taa2x" / f"nav_{key}_100k.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
        p = participation(tx, nav)
        p.to_csv(RES / "taa2x" / f"participation_by_fill_{key}.csv", index=False)
        syms = sorted(set(p["asset"]))
        out[key] = {"participation": summarize(p), "instruments": instrument_table(syms), "whole_share": whole_share_taa(key)}
        print(key, json.dumps(out[key]["participation"]["last3y"]), flush=True)
    (RES / "tradability.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
