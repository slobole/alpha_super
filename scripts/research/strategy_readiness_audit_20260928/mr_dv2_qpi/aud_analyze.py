"""Summarise the full-run arms: metrics, A10 determinism, A9 capital scaling, A6 trim, A8 share units (research-only).

Usage: uv run python aud_analyze.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import aud_common as ac

FR = ac.OUT / "full_runs"


def load(arm):
    path = FR / arm
    if not (path / "summary.json").exists():
        return None
    summary = json.loads((path / "summary.json").read_text())
    daily = pd.read_csv(path / "daily.csv.gz", index_col=0, parse_dates=True)
    tx = pd.read_csv(path / "transactions.csv.gz", parse_dates=["bar"])
    return {"summary": summary, "daily": daily, "tx": tx, "tv": np.load(path / "total_value.npy")}


def keyset(tx):
    t = tx[tx["order_id"].astype(int) != -1]
    return set(zip(t["bar"].dt.date.astype(str), t["asset"].astype(str), np.sign(t["amount"].astype(float)).astype(int)))


def main():
    out = {"metrics": []}
    arms = {}
    for fam in ("dv2", "qpi"):
        for suffix in ("base", "base_rep", "untrimmed", "10m", "hsu_100k", "hsu_30k"):
            a = load(f"{fam}_{suffix}")
            if a is None:
                continue
            arms[f"{fam}_{suffix}"] = a
            s = a["summary"]
            out["metrics"].append({"arm": f"{fam}_{suffix}", "cagr": s["full"]["cagr"], "sharpe": s["full"]["sharpe"],
                                   "max_dd": s["full"]["max_dd"], "cagr_last3y": s["last_3y"]["cagr"],
                                   "sharpe_last3y": s["last_3y"]["sharpe"], "commission": s["total_commission"],
                                   "commission_bp_notional": 1e4 * s["total_commission"] / s["gross_notional"],
                                   "n_tx": s["n_transactions"], "synthetic_liq": s["n_synthetic_liquidations"],
                                   "missing_open_cancels": s["missing_open_cancels"],
                                   "min_cash_over_nav": s["min_cash_over_nav"],
                                   "n_days_negative_cash": s["n_days_negative_cash"],
                                   "dividend_net": s["accounting_policy"].get("dividend_cash_net_total_float")})
    for fam in ("dv2", "qpi"):
        b = arms.get(f"{fam}_base")
        if b is None:
            continue
        res = {}
        r = arms.get(f"{fam}_base_rep")
        if r is not None:
            res["A10_determinism_bit_identical"] = bool(np.array_equal(b["tv"], r["tv"]))
            res["A10_recorder_inert_tx_identical"] = bool(b["tx"].reset_index(drop=True).equals(r["tx"].reset_index(drop=True)))
        m = arms.get(f"{fam}_10m")
        if m is not None:
            rb = b["daily"]["total_value"].pct_change().dropna()
            rm = m["daily"]["total_value"].pct_change().dropna()
            d = (rb - rm).abs()
            ks_b, ks_m = keyset(b["tx"]), keyset(m["tx"])
            res["A9_capital_100k_vs_10m"] = {
                "daily_return_corr": float(rb.corr(rm)), "max_abs_daily_return_diff": float(d.max()),
                "mean_abs_daily_return_diff_bp": float(1e4 * d.mean()),
                "cagr_100k": b["summary"]["full"]["cagr"], "cagr_10m": m["summary"]["full"]["cagr"],
                "trade_keys_only_100k": len(ks_b - ks_m), "trade_keys_only_10m": len(ks_m - ks_b),
                "first_divergence": min([k[0] for k in (ks_b ^ ks_m)], default=None)}
        u = arms.get(f"{fam}_untrimmed")
        if u is not None:
            ks_b, ks_u = keyset(b["tx"]), keyset(u["tx"])
            res["A6_trim_untrimmed_minus_base"] = {
                "d_cagr_pp": 100 * (u["summary"]["full"]["cagr"] - b["summary"]["full"]["cagr"]),
                "d_sharpe": u["summary"]["full"]["sharpe"] - b["summary"]["full"]["sharpe"],
                "d_maxdd_pp": 100 * (u["summary"]["full"]["max_dd"] - b["summary"]["full"]["max_dd"]),
                "d_cagr_last3y_pp": 100 * (u["summary"]["last_3y"]["cagr"] - b["summary"]["last_3y"]["cagr"]),
                "trade_keys_only_untrimmed": len(ks_u - ks_b), "trade_keys_only_base": len(ks_b - ks_u),
                "first_divergence": min([k[0] for k in (ks_b ^ ks_u)], default=None)}
        h = arms.get(f"{fam}_hsu_100k")
        if h is not None:
            res["A8_hsu_minus_production"] = {
                "d_cagr_pp": 100 * (h["summary"]["full"]["cagr"] - b["summary"]["full"]["cagr"]),
                "d_sharpe": h["summary"]["full"]["sharpe"] - b["summary"]["full"]["sharpe"],
                "d_maxdd_pp": 100 * (h["summary"]["full"]["max_dd"] - b["summary"]["full"]["max_dd"]),
                "commission_prod": b["summary"]["total_commission"], "commission_hsu": h["summary"]["total_commission"]}
        h30 = arms.get(f"{fam}_hsu_30k")
        if h30 is not None and h is not None:
            res["C2_hsu30k_minus_hsu100k"] = {
                "d_cagr_pp": 100 * (h30["summary"]["full"]["cagr"] - h["summary"]["full"]["cagr"]),
                "d_sharpe": h30["summary"]["full"]["sharpe"] - h["summary"]["full"]["sharpe"],
                "d_cagr_last3y_pp": 100 * (h30["summary"]["last_3y"]["cagr"] - h["summary"]["last_3y"]["cagr"]),
                "d_sharpe_last3y": h30["summary"]["last_3y"]["sharpe"] - h["summary"]["last_3y"]["sharpe"]}
        out[fam] = res
    pd.DataFrame(out["metrics"]).to_csv(ac.OUT / "full_run_metrics.csv", index=False)
    (ac.OUT / "full_run_analysis.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(out["metrics"]).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=2, default=str))


if __name__ == "__main__":
    main()
