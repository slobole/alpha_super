"""Analyse the mr_full_runs arms (research-only). Writes mr/full_run_metrics.csv and mr/full_run_analysis.json.

- metrics table for every finished arm (full window and 2012-10-02..2026-08-19),
- membership trim (dv2_base vs dv2_untrimmed, and the $1M/2000 pair if present): entries that differ, P&L of the
  trades that exist only in the untrimmed run, and how many of those entries fall inside a trimmed-away window,
- E-02: commissions and metrics, engine vs historical_share_units,
- E-03: HPI same-open slot reuses (and DV2 for reference) and hpi_base vs hpi_liveslot,
- E-04: synthetic liquidations (order_id == -1) with trade P&L and a -30% stress bound.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mr_common as mc
import mr_data

RUNS = mc.OUT / "full_runs"


def load_arm(arm: str):
    folder = RUNS / arm
    if not (folder / "daily.csv.gz").exists():
        return None
    daily = pd.read_csv(folder / "daily.csv.gz", index_col=0, parse_dates=True)
    tx = pd.read_csv(folder / "transactions.csv.gz", parse_dates=["bar"])
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    return daily, tx, summary


def trade_table(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per trade_id: asset, entry date, exit date, cost basis, P&L net of commissions, synthetic flag."""
    tx = tx.copy()
    tx["cash"] = -tx["amount"] * tx["price"] - tx["commission"]
    grouped = tx.groupby("trade_id")
    out = pd.DataFrame({
        "asset": grouped["asset"].first(),
        "entry": grouped["bar"].min(),
        "exit": grouped["bar"].max(),
        "basis": grouped.apply(lambda g: float((g.loc[g["amount"] > 0, "amount"] * g.loc[g["amount"] > 0, "price"]).sum())),
        "pnl": grouped["cash"].sum(),
        "open_qty": grouped["amount"].sum(),
        "synthetic_exit": grouped["order_id"].apply(lambda s: bool((s == -1).any())),
    })
    out["ret"] = out["pnl"] / out["basis"]
    return out[np.isclose(out["open_qty"], 0.0, atol=1e-6)]


def nav_at(daily: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    tv = daily["total_value"].astype(float)
    prev = tv.shift(1)
    return prev.reindex(pd.DatetimeIndex(dates)).to_numpy()


def slot_reuse(tx: pd.DataFrame, max_slots: int = 10) -> dict:
    """Buys at an open that needed a slot freed by an exit filled at that same open."""
    tx = tx.sort_values(["bar"]).copy()
    held = set()
    reuse_total = 0
    reuse_days = 0
    for bar, day in tx.groupby("bar", sort=True):
        held_before = len(held)
        sells = set(day.loc[day["amount"] < 0, "asset"])
        buys = set(day.loc[day["amount"] > 0, "asset"])
        reuse = max(0, len(buys) - (max_slots - held_before))
        reuse_total += reuse
        reuse_days += int(reuse > 0)
        held = (held - sells) | buys
    return {"same_open_slot_reuse_entries": int(reuse_total), "days_with_reuse": int(reuse_days),
            "total_entries": int((tx["amount"] > 0).sum())}


def main() -> None:
    arms = ["dv2_base", "dv2_untrimmed", "dv2_hsu", "dv2_trimmed_1m", "dv2_untrimmed_1m", "hpi_base", "hpi_hsu",
            "hpi_liveslot", "etf_base", "etf_hsu"]
    loaded = {arm: load_arm(arm) for arm in arms}
    rows = []
    for arm, item in loaded.items():
        if item is None:
            continue
        summary = item[2]
        if "error" in summary:
            rows.append({"arm": arm, "error": summary["error"]})
            continue
        rows.append({"arm": arm, "window": f"{summary['full']['start']}..{summary['full']['end']}",
                     "cagr": summary["full"]["cagr"], "sharpe": summary["full"]["sharpe"],
                     "max_dd": summary["full"]["max_dd"], "cagr_2012_10": summary["post_2012_10_02"]["cagr"],
                     "sharpe_2012_10": summary["post_2012_10_02"]["sharpe"],
                     "max_dd_2012_10": summary["post_2012_10_02"]["max_dd"],
                     "commission": summary["total_commission"],
                     "commission_bps_of_notional": 1e4 * summary["total_commission"] / summary["gross_notional"],
                     "n_tx": summary["n_transactions"], "n_synthetic_liq": summary["n_synthetic_liquidations"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(mc.OUT / "full_run_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(metrics.to_string(index=False))
    analysis = {}

    # ── membership trim ────────────────────────────────────────────────────
    dv2 = mr_data.load("dv2")
    removal = dv2["removal_df"]
    untrimmed_u = dv2["universe_untrimmed"]
    trimmed_u = dv2["universe_trimmed"]
    trimmed_away = (untrimmed_u.reindex(columns=trimmed_u.columns.union(untrimmed_u.columns)).fillna(0)
                    - trimmed_u.reindex(index=untrimmed_u.index, columns=trimmed_u.columns.union(untrimmed_u.columns)).fillna(0))
    for base_arm, alt_arm in (("dv2_base", "dv2_untrimmed"), ("dv2_trimmed_1m", "dv2_untrimmed_1m")):
        if loaded.get(base_arm) is None or loaded.get(alt_arm) is None:
            continue
        (d0, t0, _), (d1, t1, _) = loaded[base_arm], loaded[alt_arm]
        tr0, tr1 = trade_table(t0), trade_table(t1)
        key0 = set(zip(tr0["entry"], tr0["asset"]))
        key1 = set(zip(tr1["entry"], tr1["asset"]))
        only1 = tr1[[k not in key0 for k in zip(tr1["entry"], tr1["asset"])]].copy()
        only0 = tr0[[k not in key1 for k in zip(tr0["entry"], tr0["asset"])]].copy()
        # entries whose decision date (previous session) sat inside the trimmed-away window
        def in_trim(row) -> bool:
            prev = untrimmed_u.index[untrimmed_u.index.searchsorted(row["entry"]) - 1]
            return bool(row["asset"] in trimmed_away.columns and trimmed_away.at[prev, row["asset"]] == 1) \
                if prev in trimmed_away.index else False
        only1["decision_in_trimmed_window"] = only1.apply(in_trim, axis=1)
        for frame, daily in ((only1, d1), (only0, d0)):
            frame["pnl_over_nav_bp"] = 1e4 * frame["pnl"].to_numpy() / nav_at(daily, frame["entry"])
        direct = only1[only1["decision_in_trimmed_window"]]
        years = (d1.index[-1] - d1.index[0]).days / 365.25
        analysis[f"trim_{base_arm}_vs_{alt_arm}"] = {
            "n_trades_base": int(len(tr0)), "n_trades_untrimmed": int(len(tr1)),
            "n_only_untrimmed": int(len(only1)), "n_only_base": int(len(only0)),
            "n_only_untrimmed_decided_inside_trimmed_window": int(len(direct)),
            "direct_trim_trades_mean_ret": float(direct["ret"].mean()) if len(direct) else None,
            "direct_trim_trades_median_ret": float(direct["ret"].median()) if len(direct) else None,
            "direct_trim_trades_sum_pnl_over_nav_bp": float(direct["pnl_over_nav_bp"].sum()) if len(direct) else 0.0,
            "direct_trim_trades_per_year": float(len(direct) / years),
            "direct_trim_trades_bp_per_year": float(direct["pnl_over_nav_bp"].sum() / years) if len(direct) else 0.0,
            "only_untrimmed_mean_ret": float(only1["ret"].mean()) if len(only1) else None,
            "only_base_mean_ret": float(only0["ret"].mean()) if len(only0) else None,
            "all_trades_mean_ret_base": float(tr0["ret"].mean()),
            "direct_trades": direct.assign(entry=direct["entry"].dt.date.astype(str), exit=direct["exit"].dt.date.astype(str))[
                ["asset", "entry", "exit", "ret", "pnl_over_nav_bp", "synthetic_exit"]].to_dict("records"),
        }
        only1.to_csv(mc.OUT / f"trim_only_untrimmed_trades__{alt_arm}.csv", index=True)
    analysis["n_past_members_trimmed"] = int(len(removal))

    # ── E-03 slot reuse / E-04 synthetic liquidations ──────────────────────
    for arm in ("dv2_base", "hpi_base", "hpi_liveslot", "etf_base"):
        if loaded.get(arm) is None:
            continue
        daily, tx, _ = loaded[arm]
        analysis[f"slot_reuse_{arm}"] = slot_reuse(tx)
        trades = trade_table(tx)
        synth = trades[trades["synthetic_exit"]].copy()
        if len(synth):
            liq = tx[tx["order_id"] == -1].copy()
            liq["notional"] = -liq["amount"] * liq["price"]
            liq["nav_prev"] = nav_at(daily, liq["bar"])
            liq["stress_30pct_bp"] = -1e4 * 0.30 * liq["notional"] / liq["nav_prev"]
            years = (daily.index[-1] - daily.index[0]).days / 365.25
            synth["pnl_over_nav_bp"] = 1e4 * synth["pnl"].to_numpy() / nav_at(daily, synth["exit"])
            analysis[f"synthetic_liquidations_{arm}"] = {
                "n": int(len(liq)), "per_year": float(len(liq) / years),
                "trade_mean_ret": float(synth["ret"].mean()),
                "stress_minus30pct_total_bp": float(liq["stress_30pct_bp"].sum()),
                "stress_minus30pct_bp_per_year": float(liq["stress_30pct_bp"].sum() / years),
                "events": liq.assign(bar=liq["bar"].dt.date.astype(str))[["bar", "asset", "price", "notional",
                                                                            "stress_30pct_bp"]].to_dict("records"),
            }
        else:
            analysis[f"synthetic_liquidations_{arm}"] = {"n": 0}

    # ── E-02 zero-share exclusions: names whose adjusted price exceeds a slot because of later reverse splits ─────
    pricing = dv2["pricing_df"]
    ratio = (pricing.xs("Unadjusted Close", axis=1, level=1) / pricing.xs("Close", axis=1, level=1)).loc["2000":]
    heavy = sorted(ratio.columns[(ratio < 0.2).any()].astype(str))
    analysis["names_with_raw_over_adjusted_below_0.2"] = heavy
    for base_arm, hsu_arm in (("dv2_base", "dv2_hsu"), ("hpi_base", "hpi_hsu")):
        if loaded.get(base_arm) is None or loaded.get(hsu_arm) is None or "error" in loaded[hsu_arm][2]:
            continue
        (d0, t0, s0), (d1, t1, s1) = loaded[base_arm], loaded[hsu_arm]
        tr0, tr1 = trade_table(t0), trade_table(t1)
        years = (d1.index[-1] - d1.index[0]).days / 365.25
        h1 = tr1[tr1["asset"].isin(heavy)].copy()
        h0 = tr0[tr0["asset"].isin(heavy)].copy()
        h1["pnl_over_nav_bp"] = 1e4 * h1["pnl"].to_numpy() / nav_at(d1, h1["entry"])
        key0 = set(zip(tr0["entry"], tr0["asset"]))
        key1 = set(zip(tr1["entry"], tr1["asset"]))
        analysis[f"e02_{base_arm}_vs_{hsu_arm}"] = {
            "cagr_diff_pp": 100 * (s0["full"]["cagr"] - s1["full"]["cagr"]),
            "sharpe_diff": s0["full"]["sharpe"] - s1["full"]["sharpe"],
            "commission_engine": s0["total_commission"], "commission_hsu": s1["total_commission"],
            "n_trades_engine": int(len(tr0)), "n_trades_hsu": int(len(tr1)),
            "n_trade_keys_only_hsu": len(key1 - key0), "n_trade_keys_only_engine": len(key0 - key1),
            "heavy_name_trades_engine": int(len(h0)), "heavy_name_trades_hsu": int(len(h1)),
            "heavy_name_trades_hsu_mean_ret": float(h1["ret"].mean()) if len(h1) else None,
            "heavy_name_trades_hsu_bp_per_year": float(h1["pnl_over_nav_bp"].sum() / years) if len(h1) else 0.0,
            "heavy_name_trades_hsu": h1.assign(entry=h1["entry"].dt.date.astype(str), exit=h1["exit"].dt.date.astype(str))[
                ["asset", "entry", "exit", "ret", "pnl_over_nav_bp", "synthetic_exit"]].to_dict("records"),
        }

    (mc.OUT / "full_run_analysis.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                               if kk not in ("direct_trades", "events",
                                                                             "heavy_name_trades_hsu")})
                      for k, v in analysis.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
