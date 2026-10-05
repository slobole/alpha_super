"""Audit scratch (fund products 2026-10-05, task rerun_e2): supplementary facts on the E2 book.

- VXN exposure scale at the monthly decision dates and the realised invested fraction after each rebalance.
- The live NDX VXN rule alone: turnover and idle cash on the same definitions as e2_book_daily.csv.
- Whole-share / minimum-commission effect: the same book started 2021-01-01 at $1M and at $100M.
Reads only audit outputs written by rerun_e2_extract.py and the scratch PM runs. Writes e2_supp.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = REPO / "results/research/portfolio/fund_products_20261005/audit/rerun_e2"

import rerun_e2_extract as ex  # noqa: E402


def vxn_scale_block(daily: pd.DataFrame) -> dict:
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
        compute_vxn_scale_signal_df,
        get_asof_vxn_scale_float,
        load_vxn_close_ser,
    )

    vxn_close = load_vxn_close_ser("$VXN", "1999-01-01", None)
    scale_df = compute_vxn_scale_signal_df(vxn_close)
    idx = daily.index
    rebalance_days = idx[daily["is_book_rebalance_day"].to_numpy()]
    rows = []
    for execution_day in rebalance_days:
        decision_day = idx[idx.get_loc(execution_day) - 1]
        rows.append({
            "execution_date": execution_day, "decision_date": decision_day,
            "vxn_scale": get_asof_vxn_scale_float(scale_df, decision_day),
            "cash_frac_close_of_execution_day": float(daily.loc[execution_day, "cash_frac"]),
            "turnover_frac": float(daily.loc[execution_day, "turnover_frac"]),
        })
    month_df = pd.DataFrame(rows).set_index("execution_date")
    month_df["invested_month"] = month_df["cash_frac_close_of_execution_day"] < 0.99
    month_df.to_csv(OUT / "e2_monthly_rebalance.csv", float_format="%.6g")
    invested = month_df[month_df["invested_month"]]
    gap = invested["cash_frac_close_of_execution_day"] - (1.0 - invested["vxn_scale"])
    return {
        "months_total": int(len(month_df)), "months_regime_off_all_cash": int((~month_df["invested_month"]).sum()),
        "share_months_regime_off": float((~month_df["invested_month"]).mean()),
        "vxn_scale_mean_invested_months": float(invested["vxn_scale"].mean()),
        "vxn_scale_median_invested_months": float(invested["vxn_scale"].median()),
        "share_invested_months_scale_eq_1": float((invested["vxn_scale"] >= 1.0 - 1e-12).mean()),
        "share_invested_months_scale_lt_0p75": float((invested["vxn_scale"] < 0.75).mean()),
        "vxn_scale_min_invested_months": float(invested["vxn_scale"].min()),
        "cash_minus_target_cash_mean": float(gap.mean()), "cash_minus_target_cash_median": float(gap.median()),
        "cash_minus_target_cash_p05_p95": [float(gap.quantile(0.05)), float(gap.quantile(0.95))],
        "latest_decision": {"date": month_df["decision_date"].iloc[-1].date().isoformat(), "vxn_scale": float(month_df["vxn_scale"].iloc[-1])},
    }


def live_rule_block() -> dict:
    path = pd.read_csv(OUT / "ndx_vxn_live_daily.csv", index_col="date", parse_dates=True)
    tx = pd.read_csv(OUT / "ndx_vxn_live_transactions.csv", parse_dates=["bar"])
    nav = path["total_value"].astype(float)
    traded = (tx["amount"] * tx["price"]).abs().groupby(tx["bar"]).sum()
    turnover = (traded.reindex(nav.index).fillna(0.0) / nav.shift(1)).fillna(0.0)
    commission = (tx.groupby("bar")["commission"].sum().reindex(nav.index).fillna(0.0) / nav.shift(1)).fillna(0.0)
    cash_frac = path["cash"].astype(float) / nav
    years = len(nav) / 252.0
    return {
        "start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat(), "final_nav": float(nav.iloc[-1]),
        "annual_turnover_two_sided_x_nav": float(turnover.sum() / years),
        "annual_commission_frac_nav": float(commission.sum() / years),
        "trading_days_per_year": float((turnover > 0).sum() / years),
        "cash_frac_mean": float(cash_frac.mean()), "cash_frac_median": float(cash_frac.median()),
        "fills_total": int(len(tx)),
    }


def late_start_block() -> dict:
    root = OUT / "pm_from2021/research/portfolio"
    frames = {}
    for tag in ("1m", "100m"):
        name = f"ndx_e2_sector_cap_5050_from2021_{tag}"
        pickles = sorted((root / name / "vanilla_backtest").glob(f"*/{name}.pkl"))
        if not pickles:
            return {"status": f"missing run {name}"}
        frames[tag] = ex.book_frames(ex.load_port(pickles[-1]))["daily"]
    small, big = frames["1m"], frames["100m"]
    first_fill = small.index[small["turnover_frac"] > 0][0]
    out = {"start": small.index[0].date().isoformat(), "first_fill": first_fill.date().isoformat(), "end": small.index[-1].date().isoformat()}
    for tag, frame in frames.items():
        part = frame.loc[first_fill:]
        years = len(part) / 252.0
        out[tag] = {**ex.stats(part["ret"]), "final_nav_multiple": float(frame["nav"].iloc[-1] / frame["nav"].iloc[0]),
                    "cash_frac_mean": float(part["cash_frac"].mean()),
                    "annual_commission_frac_nav": float(part["commission_frac"].sum() / years),
                    "annual_turnover_two_sided_x_nav": float(part["turnover_frac"].sum() / years)}
    invested = (small["cash_frac"] < 0.99) & (big["cash_frac"] < 0.99)
    out["cash_frac_gap_1m_minus_100m_mean_when_invested"] = float((small["cash_frac"] - big["cash_frac"])[invested].mean())
    out["cagr_gap_1m_minus_100m"] = out["1m"]["cagr"] - out["100m"]["cagr"]
    out["max_abs_daily_ret_diff"] = float((small["ret"] - big["ret"]).abs().max())
    out["nav_1m_range_usd"] = [float(small["nav"].min()), float(small["nav"].max())]
    return out


def main() -> None:
    daily = pd.read_csv(OUT / "e2_book_daily.csv", index_col="date", parse_dates=True)
    result = {"vxn_scale": vxn_scale_block(daily), "live_rule": live_rule_block(), "late_start_2021": late_start_block()}
    new_port = ex.load_port(ex.newest_pickle(ex.NEW_ROOT, "ndx_e2_sector_cap_5050"))
    result["engine_pod_summary"] = {
        ex.POD_SHORT[s.name]: {k: (float(s.summary.loc[k].iloc[0]) if k in s.summary.index else None)
                               for k in ("Turnover (Ann.) [%]", "Cost Drag (Ann.) [%]", "Total Commissions [$]", "Estimated Slippage [$]",
                                         "Exposure Time [%]")}
        for s in new_port.strategies}
    (OUT / "e2_supp.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
