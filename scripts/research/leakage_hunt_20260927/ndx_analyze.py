"""(d)/(e) Analyse the four engine runs: reproduce the refresh sleeves / Codex audit arms, quantify the membership
tail-trim (trimmed vs untrimmed universe), and list synthetic missing-price liquidations (G-014).

Usage: uv run python scripts/research/leakage_hunt_20260927/ndx_analyze.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ndx_common as nc

REFRESH = nc.REPO / "results" / "research" / "portfolio" / "portfolio_refresh_20260927"


def run_returns(tag: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    run = nc.load_run(tag)
    res = run["results"]
    nav = res["total_value"].astype(float)
    ret = nc.nav_returns(nav, res["portfolio_value"].astype(float).abs() > 1e-9)
    return ret, nav, run["transactions"]


def windows(ret: pd.Series) -> dict:
    return {"full": nc.window_metrics(ret, None), "exact": nc.window_metrics(ret, nc.EXACT_START),
            "2000_2011": nc.window_metrics(ret, None, pd.Timestamp("2012-10-01"))}


def synthetic_liquidations(tx: pd.DataFrame, pricing: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    liq = tx[tx["order_id"].astype(int) == -1]
    rows = []
    for _, r in liq.iterrows():
        trade = tx[(tx["trade_id"] == r["trade_id"]) & (tx["asset"] == r["asset"])]
        pnl = float(-(trade["amount"].astype(float) * trade["price"].astype(float)).sum() - trade["commission"].astype(float).sum())
        cost = float((trade.loc[trade["amount"].astype(float) > 0, "amount"].astype(float)
                      * trade.loc[trade["amount"].astype(float) > 0, "price"].astype(float)).sum())
        close = pricing[(r["asset"], "Close")].loc[:r["bar"]].dropna()
        uni = universe[r["asset"]] if r["asset"] in universe.columns else pd.Series(dtype=int)
        last_member = uni[uni == 1].index.max() if len(uni) else pd.NaT
        rows.append({"asset": r["asset"], "liquidation_bar": r["bar"].date().isoformat(),
                     "last_close_date": close.index[-1].date().isoformat() if len(close) else None,
                     "exit_price": float(r["price"]), "exit_value": float(-r["amount"] * r["price"]),
                     "trade_pnl": pnl, "trade_cost": cost, "trade_return": pnl / cost if cost else np.nan,
                     "last_member_date_in_universe": None if pd.isna(last_member) else last_member.date().isoformat()})
    return pd.DataFrame(rows)


def decision_diffs(data_t: dict, data_u: dict, key: str) -> dict:
    strategy_t, signals_t = nc.signals(key, data_t["pricing"], data_t["universe"], data_t["vxn"])
    strategy_u, signals_u = nc.signals(key, data_u["pricing"], data_u["universe"], data_u["vxn"])
    month_ends = nc.month_end_decision_dates(data_t["pricing"])
    month_ends = month_ends[(month_ends >= "2000-01-31") & (month_ends <= nc.STUDY_END)]
    changed = []
    for T in month_ends:
        a = nc.decision_at(strategy_t, signals_t, T)
        b = nc.decision_at(strategy_u, signals_u, T)
        if a["selected"] != b["selected"]:
            changed.append({"T": str(T.date()), "trimmed_only": sorted(set(a["selected"]) - set(b["selected"])),
                            "untrimmed_only": sorted(set(b["selected"]) - set(a["selected"]))})
    return {"n_month_ends": int(len(month_ends)), "n_changed": len(changed),
            "n_changed_exact_window": sum(1 for c in changed if c["T"] >= "2012-09-28"), "changed": changed}


def main() -> None:
    data_t, data_u = nc.load_data("trimmed"), nc.load_data("untrimmed")
    sleeves = pd.read_csv(REFRESH / "momentum_series_full.csv.gz", index_col=0, parse_dates=True)
    sleeve_csv = pd.read_csv(REFRESH / "sleeves.csv")
    report, metric_rows = {}, []
    for key, meta in nc.MODULES.items():
        ret_t, nav_t, tx_t = run_returns(f"{key}_trimmed")
        ret_u, nav_u, tx_u = run_returns(f"{key}_untrimmed")
        w_t, w_u = windows(ret_t), windows(ret_u)
        for variant, w in (("trimmed_as_committed", w_t), ("untrimmed_universe", w_u)):
            for window, m in w.items():
                metric_rows.append({"module": key, "variant": variant, "window": window, **m})
        # (d) reproduction of the refresh sleeve and of the Codex audit arm
        alias = meta["sleeve_alias"]
        sleeve = sleeves[alias].dropna()
        common = sleeve.index.intersection(ret_t.index)
        sleeve_diff = float((sleeve.loc[common] - ret_t.loc[common]).abs().max())
        sleeve_exact = nc.window_metrics(sleeve, nc.EXACT_START)
        csv_row = sleeve_csv[(sleeve_csv["alias"] == alias) & (sleeve_csv["window"] == "exact")].iloc[0].to_dict()
        audit = pd.read_csv(nc.AUDIT_RUNS / meta["audit_arm"][0] / meta["audit_arm"][1] / "daily_results.csv",
                            index_col=0, parse_dates=True)["total_value"].astype(float)
        common_nav = audit.index.intersection(nav_t.index)
        audit_rel = float(((nav_t.loc[common_nav] / audit.loc[common_nav]) - 1.0).abs().max())
        audit_tx = pd.read_csv(nc.AUDIT_RUNS / meta["audit_arm"][0] / meta["audit_arm"][1] / "transactions.csv")
        # (e) membership trim
        diffs = decision_diffs(data_t, data_u, key)
        liq_t = synthetic_liquidations(tx_t, data_t["pricing"], data_t["universe"])
        liq_u = synthetic_liquidations(tx_u, data_u["pricing"], data_u["universe"])
        liq_t.to_csv(nc.OUT / f"synthetic_liquidations_{key}_trimmed.csv", index=False)
        liq_u.to_csv(nc.OUT / f"synthetic_liquidations_{key}_untrimmed.csv", index=False)
        pd.DataFrame(diffs["changed"]).to_csv(nc.OUT / f"trim_decision_changes_{key}.csv", index=False)
        report[key] = {
            "reproduction": {
                "sleeve_alias": alias, "max_abs_daily_return_diff_vs_refresh_sleeve": sleeve_diff,
                "n_common_days": int(len(common)),
                "refresh_sleeves_csv_exact": {k: csv_row[k] for k in ("cagr", "sharpe", "maxdd", "vol")},
                "this_run_exact": w_t["exact"], "refresh_series_recomputed_exact": sleeve_exact,
                "audit_arm": "/".join(meta["audit_arm"]), "max_rel_nav_diff_vs_audit_arm": audit_rel,
                "n_tx_this_run": int(len(tx_t)), "n_tx_audit_arm_to_2026_09_25": int(len(audit_tx)),
            },
            "membership_trim": {
                "metrics_trimmed": w_t, "metrics_untrimmed": w_u,
                "delta_untrimmed_minus_trimmed": {
                    window: {m: w_u[window][m] - w_t[window][m] for m in ("cagr", "sharpe", "maxdd", "vol")}
                    for window in w_t},
                "decision_changes": {k: v for k, v in diffs.items() if k != "changed"},
                "decision_change_examples": diffs["changed"][:10],
            },
            "synthetic_liquidations": {
                "trimmed": {"n": int(len(liq_t)), "sum_trade_pnl": float(liq_t["trade_pnl"].sum()) if len(liq_t) else 0.0,
                            "assets": liq_t["asset"].tolist() if len(liq_t) else []},
                "untrimmed": {"n": int(len(liq_u)), "sum_trade_pnl": float(liq_u["trade_pnl"].sum()) if len(liq_u) else 0.0,
                              "assets": liq_u["asset"].tolist() if len(liq_u) else []},
            },
        }
        print(key, json.dumps(report[key]["reproduction"], indent=1, default=str))
        print(key, json.dumps(report[key]["membership_trim"]["delta_untrimmed_minus_trimmed"], indent=1))
        print(key, report[key]["membership_trim"]["decision_changes"], report[key]["synthetic_liquidations"])
    pd.DataFrame(metric_rows).to_csv(nc.OUT / "backtest_metrics.csv", index=False, float_format="%.6g")
    (nc.OUT / "backtest_analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
