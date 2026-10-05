"""Audit task rerun_mr (fund products 2026-10-05): MR capsule (BIL parking) reproduced at HEAD, tidy inputs.

Reads (never writes) MAIN results; writes only under WT/results/research/portfolio/fund_products_20261005/audit/rerun_mr/.

1. New engine runs (WT/results/research/mr_capsule_build_20261004/{dv2,hpi}_{bil,cash}_*) vs MAIN's stored copies.
2. Capsule = pod model, 50/50, reset after each year-end close (ev.capsule -> tbc.book_return_ser(..., "annual")),
   start 2004-01-05; compared with the PortfolioManager book mr_capsule_bil.
3. Tidy daily CSV: pod returns, capsule returns, BIL weight, idle cash, stock / BIL turnover, gate state.
4. Stats by window, correlations, turnover, positions, negative cash, extra-cost estimates.

*** CRITICAL*** turnover_t = sum(|amount * price|) of fills dated t / NAV_(t-1) (prior close), so
ret_t - cost_bps * turnover_t is an exact first-order cost deduction on the daily return ret_t = NAV_t / NAV_(t-1) - 1.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:/Users/User/Documents/workspace/alpha_super")
sys.path.insert(0, str(WT))
NEW = WT / "results/research/mr_capsule_build_20261004"
OLD = MAIN / "results/research/mr_capsule_build_20261004"
PM_RUN = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751"
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/rerun_mr"
OUT.mkdir(parents=True, exist_ok=True)
PARK = ("SPMO", "BIL")
FULL_START = "2004-01-05"
RESEARCH_END = "2026-09-24"  # compare.END, the end date book_weights.py passes to ev.capsule
WINDOWS = {
    "full 2004-01-05..end": (FULL_START, None),
    "2008-03-04..2026-08-19": ("2008-03-04", "2026-08-19"),
    "2012-10-02..2026-08-19": ("2012-10-02", "2026-08-19"),
    "after 2026-08-19": ("2026-08-20", None),
}


def load_run(folder: Path, pod: str, mode: str):
    nav = pd.read_csv(folder / f"{pod}_{mode}_nav.csv", index_col=0, parse_dates=True).astype(float)
    tx = pd.read_csv(folder / f"{pod}_{mode}_transactions.csv", parse_dates=["bar"])
    diag = json.loads((folder / f"{pod}_{mode}_diagnostics.json").read_text(encoding="utf-8"))
    return nav, tx, diag


def book_return_ser(frame: pd.DataFrame, weights: dict) -> pd.Series:
    """Exact copy of the pod model in fund_menu_20260923/common.py book_return_ser(..., "annual")."""
    names = list(weights)
    w = np.array([weights[n] for n in names], dtype=float)
    mat = frame[names].to_numpy(dtype=float)
    idx = frame.index
    pod = w.copy()
    out = []
    for i in range(len(idx)):
        before = pod.sum()
        pod = pod * (1.0 + mat[i])
        out.append(pod.sum() / before - 1.0)
        if i + 1 < len(idx) and idx[i + 1].year != idx[i].year:
            pod = w * pod.sum()
    return pd.Series(out, index=idx)


def capsule(parts: dict, start=FULL_START, end=None) -> pd.Series:
    frame = pd.DataFrame(parts).loc[start:end].dropna()
    return book_return_ser(frame, {"DV2": 0.5, "HPI": 0.5})


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    nav = (1 + r).cumprod()
    return {
        "sessions": int(len(r)), "first": str(r.index[0].date()), "last": str(r.index[-1].date()),
        "total_return": float(nav.iloc[-1] - 1),
        "cagr": float(nav.iloc[-1] ** (252.0 / len(r)) - 1),
        "vol": float(r.std() * np.sqrt(252)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "max_dd": float((nav / np.maximum.accumulate(np.r_[1.0, nav.to_numpy()])[1:] - 1).min()),
    }


def nav_compare(new_nav: pd.DataFrame, old_nav: pd.DataFrame, new_tx: pd.DataFrame, old_tx: pd.DataFrame) -> dict:
    common = new_nav.index.intersection(old_nav.index)
    rn = new_nav["total_value"].reindex(common).pct_change().fillna(0.0)
    ro = old_nav["total_value"].reindex(common).pct_change().fillna(0.0)
    diff = (rn - ro).abs()
    rel = (new_nav["total_value"].reindex(common) / old_nav["total_value"].reindex(common) - 1)
    diverged = diff[diff > 1e-12]
    nav_div = rel[rel.abs() > 1e-12]
    out = {
        "new_first": str(new_nav.index[0].date()), "new_last": str(new_nav.index[-1].date()), "new_sessions": int(len(new_nav)),
        "old_first": str(old_nav.index[0].date()), "old_last": str(old_nav.index[-1].date()), "old_sessions": int(len(old_nav)),
        "common_sessions": int(len(common)), "index_identical": bool(new_nav.index.equals(old_nav.index)),
        "max_abs_daily_ret_diff": float(diff.max()), "max_abs_daily_ret_diff_date": str(diff.idxmax().date()),
        "sessions_with_ret_diff_gt_1e-12": int(len(diverged)),
        "sessions_with_ret_diff_gt_1bp": int((diff > 1e-4).sum()),
        "first_ret_diverging_date": str(diverged.index[0].date()) if len(diverged) else None,
        "first_nav_diverging_date": str(nav_div.index[0].date()) if len(nav_div) else None,
        "daily_ret_corr": float(rn.corr(ro)),
        "nav_ratio_last_common": float(1 + rel.iloc[-1]), "nav_ratio_last_common_date": str(common[-1].date()),
        "max_abs_nav_rel_diff": float(rel.abs().max()),
        "new_final_nav": float(new_nav["total_value"].iloc[-1]), "old_final_nav": float(old_nav["total_value"].iloc[-1]),
        "tx_count_new": int(len(new_tx)), "tx_count_old": int(len(old_tx)),
        "tx_count_new_stock": int((~new_tx["asset"].isin(PARK)).sum()), "tx_count_old_stock": int((~old_tx["asset"].isin(PARK)).sum()),
        "tx_count_new_bil": int((new_tx["asset"] == "BIL").sum()), "tx_count_old_bil": int((old_tx["asset"] == "BIL").sum()),
    }
    # first differing transaction (by bar, asset, amount, price)
    key = ["bar", "asset", "amount", "price"]
    a = new_tx[key].copy(); b = old_tx[key].copy()
    a["price"] = a["price"].round(6); b["price"] = b["price"].round(6)
    m = a.merge(b, on=key, how="outer", indicator=True)
    only = m[m["_merge"] != "both"].sort_values("bar")
    out["tx_only_new"] = int((m["_merge"] == "left_only").sum())
    out["tx_only_old"] = int((m["_merge"] == "right_only").sum())
    out["first_tx_diff"] = only.head(12).assign(bar=lambda d: d["bar"].dt.strftime("%Y-%m-%d")).to_dict("records") if len(only) else []
    stock_only = only[~only["asset"].isin(PARK)]
    out["stock_tx_only_new"] = int((stock_only["_merge"] == "left_only").sum())
    out["stock_tx_only_old"] = int((stock_only["_merge"] == "right_only").sum())
    out["first_stock_tx_diff"] = stock_only.head(8).assign(bar=lambda d: d["bar"].dt.strftime("%Y-%m-%d")).to_dict("records") if len(stock_only) else []
    return out


def pod_daily(nav: pd.DataFrame, tx: pd.DataFrame, bil_close: pd.Series) -> pd.DataFrame:
    idx = nav.index
    prev_nav = nav["total_value"].shift(1)
    notional = (tx["amount"] * tx["price"]).abs()
    is_park = tx["asset"].isin(PARK)
    stock_turn = notional[~is_park].groupby(tx.loc[~is_park, "bar"]).sum().reindex(idx).fillna(0.0)
    bil_turn = notional[tx["asset"] == "BIL"].groupby(tx.loc[tx["asset"] == "BIL", "bar"]).sum().reindex(idx).fillna(0.0)
    bil_shares = tx.loc[tx["asset"] == "BIL"].groupby("bar")["amount"].sum().reindex(idx).fillna(0.0).cumsum()
    bil_value = bil_shares * bil_close.reindex(idx).ffill().fillna(0.0)
    pos = tx.loc[~is_park].pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").reindex(idx).fillna(0.0).cumsum()
    n_stock = (pos.abs() > 1e-9).sum(axis=1)
    out = pd.DataFrame({
        "ret": nav["total_value"].pct_change(),
        "nav": nav["total_value"],
        "cash_frac": nav["cash"] / nav["total_value"],
        "bil_weight": bil_value / nav["total_value"],
        "stock_weight": (nav["portfolio_value"] - bil_value) / nav["total_value"],
        "stock_turnover": stock_turn / prev_nav,
        "bil_turnover": bil_turn / prev_nav,
        "n_stock_positions": n_stock,
        "bil_shares": bil_shares,
    }, index=idx)
    out["commission"] = tx.groupby("bar")["commission"].sum().reindex(idx).fillna(0.0) / prev_nav
    return out


def main() -> None:
    rep: dict = {}
    import os
    dry_bool = os.environ.get("RERUN_MR_DRY") == "1"  # script debugging only: cash runs read from MAIN while they still run
    rep["dry_run_cash_from_main_bool"] = dry_bool
    new = {(p, m): load_run(OLD if (dry_bool and m == "cash") else NEW, p, m) for p in ("dv2", "hpi") for m in ("bil", "cash")}
    old = {(p, m): load_run(OLD, p, m) for p in ("dv2", "hpi") for m in ("bil", "cash")}
    # ---- 1. new vs MAIN stored
    rep["new_vs_main"] = {f"{p}_{m}": nav_compare(new[(p, m)][0], old[(p, m)][0], new[(p, m)][1], old[(p, m)][1]) for (p, m) in new}
    rep["diagnostics"] = {f"{p}_{m}": {"new": new[(p, m)][2], "main": old[(p, m)][2]} for (p, m) in new}

    # ---- BIL marks (CAPITALSPECIAL close, the engine's mark) and the gate (pure function of VIX)
    from data.norgate_loader import load_raw_prices
    from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser, stress_gate_open_ser, vix_threshold_ser

    px = load_raw_prices(["BIL"], ["$SPX"], start_date="2004-01-01")
    bil_close = px[("BIL", "Close")].astype(float).dropna()
    vix = load_vix_close_ser()
    gate = stress_gate_open_ser(vix)
    thr = vix_threshold_ser(vix)

    idx = new[("dv2", "bil")][0].index
    daily = {(p, m): pod_daily(new[(p, m)][0], new[(p, m)][1], bil_close) for (p, m) in new}
    for (p, m), d in daily.items():
        if not d.index.equals(idx):
            raise RuntimeError(f"calendar mismatch {p} {m}")
    # sanity: when a pod holds no stock, portfolio_value must equal the BIL value (validates the BIL marks)
    rep["bil_mark_check"] = {}
    for p in ("dv2", "hpi"):
        d = daily[(p, "bil")]
        flat = d[d["n_stock_positions"] == 0]
        rep["bil_mark_check"][p] = {"sessions_without_stock": int(len(flat)),
                                    "max_abs_stock_weight_when_no_stock": float(flat["stock_weight"].abs().max()) if len(flat) else None,
                                    "min_stock_weight_all": float(d["stock_weight"].min()),
                                    "max_n_stock_positions": int(d["n_stock_positions"].max()),
                                    "final_n_stock_positions": int(d["n_stock_positions"].iloc[-1])}

    # ---- 2. capsule series (research rule) and the PM book
    parts = {m: {"DV2": daily[("dv2", m)]["ret"], "HPI": daily[("hpi", m)]["ret"]} for m in ("bil", "cash")}
    cap = {m: capsule(parts[m]) for m in ("bil", "cash")}
    cap_old = {m: capsule({"DV2": old[("dv2", m)][0]["total_value"].pct_change(), "HPI": old[("hpi", m)][0]["total_value"].pct_change()}) for m in ("bil", "cash")}
    try:  # the research function itself, if its imports work in this tree
        sys.path.insert(0, str(WT / "scripts/research/mr_capsule_build_20261004"))
        import compare as cmp  # noqa: E402
        ev_cap = cmp.ev.capsule(parts["bil"], {"DV2": .5, "HPI": .5}, start=FULL_START, end=None)
        rep["ev_capsule_vs_inline_max_abs_diff"] = float((ev_cap - cap["bil"]).abs().max())
    except Exception as exc:  # noqa: BLE001
        rep["ev_capsule_vs_inline_max_abs_diff"] = f"import failed: {type(exc).__name__}: {exc}"

    with open(PM_RUN / "mr_capsule_bil.pkl", "rb") as fh:
        pf = pickle.load(fh)
    pm_tv = pf.results["total_value"].astype(float)
    pm_tv.index = pd.to_datetime(pm_tv.index)
    pm_ret = pm_tv.pct_change()
    pm_pod = {}
    for s in pf.strategies:
        tv = s.results["total_value"].astype(float)
        tv.index = pd.to_datetime(tv.index)
        pm_pod["DV2" if "dv2" in s.name else "HPI"] = tv
    pm_pod_ret = {k: v.pct_change() for k, v in pm_pod.items()}
    common = cap["bil"].index.intersection(pm_ret.dropna().index)
    d_new = (cap["bil"].reindex(common) - pm_ret.reindex(common))
    d_old = (cap_old["bil"].reindex(common) - pm_ret.reindex(common)).dropna()
    cap_pm_pods = capsule(pm_pod_ret)  # research reset rule applied to the PM pods' own ($500K) returns
    d_rule = (cap_pm_pods.reindex(common) - pm_ret.reindex(common))
    rep["pm_book"] = {
        "run": str(PM_RUN), "pm_first": str(pm_tv.index[0].date()), "pm_last": str(pm_tv.index[-1].date()),
        "pm_capital": float(pm_tv.iloc[0]), "pm_final": float(pm_tv.iloc[-1]),
        "pm_pod_capital": {k: float(v.iloc[0]) for k, v in pm_pod.items()},
        "pm_pod_final": {k: float(v.iloc[-1]) for k, v in pm_pod.items()},
        "common_sessions": int(len(common)),
        "capsule_new_vs_pm": {"max_abs_daily_diff": float(d_new.abs().max()), "date": str(d_new.abs().idxmax().date()),
                              "corr": float(cap["bil"].reindex(common).corr(pm_ret.reindex(common))),
                              "tracking_error_ann": float(d_new.std() * np.sqrt(252)),
                              "wealth_ratio_capsule_over_pm": float((1 + cap["bil"].reindex(common)).prod() / (1 + pm_ret.reindex(common)).prod()),
                              "sessions_diff_gt_1bp": int((d_new.abs() > 1e-4).sum())},
        "capsule_main_stored_vs_pm": {"max_abs_daily_diff": float(d_old.abs().max()), "tracking_error_ann": float(d_old.std() * np.sqrt(252))},
        "research_rule_on_pm_pod_returns_vs_pm_book": {"max_abs_daily_diff": float(d_rule.abs().max()),
                                                       "date": str(d_rule.abs().idxmax().date())},
        "pod_100k_vs_pm_pod_500k": {},
        "pm_stats_full": stats(pm_ret.loc[FULL_START:]),
    }
    for k, p in (("DV2", "dv2"), ("HPI", "hpi")):
        a = daily[(p, "bil")]["ret"].reindex(common); b = pm_pod_ret[k].reindex(common)
        dd = (a - b)
        nz = dd[dd.abs() > 1e-12]
        tx_pm = pd.read_csv(PM_RUN / "pods" / ("pod_mr_dv2_gated_bil" if p == "dv2" else "pod_mr_hpi_vote_gated_bil") / "transactions.csv", parse_dates=["bar"])
        tx_new = new[(p, "bil")][1]
        ev_new = set(zip(tx_new["asset"], tx_new["bar"], np.sign(tx_new["amount"])))
        ev_pm = set(zip(tx_pm["asset"], tx_pm["bar"], np.sign(tx_pm["amount"])))
        st_new = {e for e in ev_new if e[0] not in PARK}; st_pm = {e for e in ev_pm if e[0] not in PARK}
        rep["pm_book"]["pod_100k_vs_pm_pod_500k"][k] = {
            "max_abs_daily_diff": float(dd.abs().max()), "date": str(dd.abs().idxmax().date()),
            "first_diverging_date": str(nz.index[0].date()) if len(nz) else None,
            "corr": float(a.corr(b)), "tracking_error_ann": float(dd.std() * np.sqrt(252)),
            "wealth_ratio_100k_over_500k": float((1 + a.fillna(0)).prod() / (1 + b.fillna(0)).prod()),
            "cagr_100k": stats(a)["cagr"], "cagr_500k": stats(b)["cagr"],
            "tx_count_100k": int(len(tx_new)), "tx_count_500k": int(len(tx_pm)),
            "stock_events_100k": len(st_new), "stock_events_500k": len(st_pm), "stock_events_common": len(st_new & st_pm),
            "stock_events_only_100k": sorted((a_, str(b_.date()), int(c_)) for a_, b_, c_ in st_new - st_pm)[:6],
            "stock_events_only_500k": sorted((a_, str(b_.date()), int(c_)) for a_, b_, c_ in st_pm - st_new)[:6],
            "bil_orders_100k": int((tx_new["asset"] == "BIL").sum()), "bil_orders_500k": int((tx_pm["asset"] == "BIL").sum()),
            "commission_pp_per_year_100k": float(daily[(p, "bil")]["commission"].sum() / (len(idx) / 252) * 100),
        }

    # ---- 3. tidy CSV
    gate_close = gate.reindex(idx, method="ffill").fillna(False).astype(bool)
    tidy = pd.DataFrame({
        "dv2_bil_ret": daily[("dv2", "bil")]["ret"], "hpi_bil_ret": daily[("hpi", "bil")]["ret"],
        "dv2_cash_ret": daily[("dv2", "cash")]["ret"], "hpi_cash_ret": daily[("hpi", "cash")]["ret"],
        "capsule_bil_ret": cap["bil"].reindex(idx), "capsule_cash_ret": cap["cash"].reindex(idx),
        "dv2_bil_weight": daily[("dv2", "bil")]["bil_weight"], "hpi_bil_weight": daily[("hpi", "bil")]["bil_weight"],
        "dv2_bilrun_cash_frac": daily[("dv2", "bil")]["cash_frac"], "hpi_bilrun_cash_frac": daily[("hpi", "bil")]["cash_frac"],
        "dv2_bilrun_stock_weight": daily[("dv2", "bil")]["stock_weight"], "hpi_bilrun_stock_weight": daily[("hpi", "bil")]["stock_weight"],
        "dv2_cashrun_idle_cash_frac": daily[("dv2", "cash")]["cash_frac"], "hpi_cashrun_idle_cash_frac": daily[("hpi", "cash")]["cash_frac"],
        "dv2_stock_turnover": daily[("dv2", "bil")]["stock_turnover"], "hpi_stock_turnover": daily[("hpi", "bil")]["stock_turnover"],
        "dv2_bil_turnover": daily[("dv2", "bil")]["bil_turnover"], "hpi_bil_turnover": daily[("hpi", "bil")]["bil_turnover"],
        "dv2_cashrun_stock_turnover": daily[("dv2", "cash")]["stock_turnover"], "hpi_cashrun_stock_turnover": daily[("hpi", "cash")]["stock_turnover"],
        "dv2_n_stock_positions": daily[("dv2", "bil")]["n_stock_positions"], "hpi_n_stock_positions": daily[("hpi", "bil")]["n_stock_positions"],
        "gate_open_at_close": gate_close.astype(int),
        "gate_open_at_prev_close": gate_close.shift(1).fillna(False).astype(int),
        "vix_close": vix.reindex(idx), "vix_threshold": thr.reindex(idx),
        "pm_book_ret": pm_ret.reindex(idx),
    }, index=idx)
    tidy.index.name = "date"
    tidy.to_csv(OUT / "mr_capsule_daily.csv", float_format="%.10g")
    rep["tidy_csv"] = {"path": str(OUT / "mr_capsule_daily.csv"), "rows": int(len(tidy)), "first": str(idx[0].date()), "last": str(idx[-1].date()),
                       "vix_missing_sessions": int(tidy["vix_close"].isna().sum()),
                       "vix_missing_dates": [str(d.date()) for d in tidy.index[tidy["vix_close"].isna()]][:20]}

    # ---- 4. stats
    series = {"capsule_bil": cap["bil"], "capsule_cash": cap["cash"], "dv2_bil": tidy["dv2_bil_ret"], "hpi_bil": tidy["hpi_bil_ret"],
              "dv2_cash": tidy["dv2_cash_ret"], "hpi_cash": tidy["hpi_cash_ret"], "pm_book_bil": pm_ret, "capsule_bil_main_stored": cap_old["bil"]}
    rep["stats"] = {w: {n: stats(s.loc[a:b]) for n, s in series.items()} for w, (a, b) in WINDOWS.items()}
    rep["stats"]["research window 2004-01-05..2026-09-24"] = {n: stats(s.loc[FULL_START:RESEARCH_END]) for n, s in series.items()}
    # capsule restarted at 50/50 at the window start (what book_window_return_ser / ev.capsule(start=...) do)
    rep["stats_capsule_restarted_at_window_start"] = {
        w: {m: stats(capsule(parts[m], start=a, end=b)) for m in ("bil", "cash")} for w, (a, b) in WINDOWS.items() if w != "after 2026-08-19"}
    full = tidy.loc[FULL_START:]
    rep["corr"] = {
        "dv2_bil_vs_hpi_bil_full": float(full["dv2_bil_ret"].corr(full["hpi_bil_ret"])),
        "dv2_cash_vs_hpi_cash_full": float(full["dv2_cash_ret"].corr(full["hpi_cash_ret"])),
        "dv2_bil_vs_hpi_bil_2008-03-04..2026-08-19": float(tidy.loc["2008-03-04":"2026-08-19", "dv2_bil_ret"].corr(tidy.loc["2008-03-04":"2026-08-19", "hpi_bil_ret"])),
        "dv2_bil_vs_hpi_bil_2012-10-02..2026-08-19": float(tidy.loc["2012-10-02":"2026-08-19", "dv2_bil_ret"].corr(tidy.loc["2012-10-02":"2026-08-19", "hpi_bil_ret"])),
        "dv2_bil_vs_hpi_bil_gate_open_sessions": float(full.loc[full["gate_open_at_prev_close"] == 1, "dv2_bil_ret"].corr(full.loc[full["gate_open_at_prev_close"] == 1, "hpi_bil_ret"])),
        "monthly_dv2_bil_vs_hpi_bil": float(((1 + full["dv2_bil_ret"]).resample("ME").prod() - 1).corr((1 + full["hpi_bil_ret"]).resample("ME").prod() - 1)),
    }
    years = len(full) / 252.0
    bil_live = tidy.loc["2007-06-04":]
    rep["exposure"] = {}
    for p in ("dv2", "hpi"):
        rep["exposure"][p] = {
            "mean_bil_weight_full": float(full[f"{p}_bil_weight"].mean()),
            "mean_bil_weight_since_2007-06-04": float(bil_live[f"{p}_bil_weight"].mean()),
            "mean_bil_weight_2008-03-04..2026-08-19": float(tidy.loc["2008-03-04":"2026-08-19", f"{p}_bil_weight"].mean()),
            "mean_stock_weight_full": float(full[f"{p}_bilrun_stock_weight"].mean()),
            "mean_cash_frac_bil_run": float(full[f"{p}_bilrun_cash_frac"].mean()),
            "mean_cash_frac_bil_run_since_2007-06-04": float(bil_live[f"{p}_bilrun_cash_frac"].mean()),
            "median_cash_frac_bil_run_since_2007-06-04": float(bil_live[f"{p}_bilrun_cash_frac"].median()),
            "mean_stock_weight_since_2007-06-04": float(bil_live[f"{p}_bilrun_stock_weight"].mean()),
            "mean_stock_weight_gate_open_sessions": float(full.loc[full["gate_open_at_prev_close"] == 1, f"{p}_bilrun_stock_weight"].mean()),
            "mean_stock_weight_gate_closed_sessions": float(full.loc[full["gate_open_at_prev_close"] == 0, f"{p}_bilrun_stock_weight"].mean()),
            "max_stock_weight": float(full[f"{p}_bilrun_stock_weight"].max()),
            "stock_turnover_per_year_2012-10-02..2026-08-19": float(tidy.loc["2012-10-02":"2026-08-19", f"{p}_stock_turnover"].sum() / (len(tidy.loc["2012-10-02":"2026-08-19"]) / 252.0)),
            "engine_slippage_2.5bps_cost_pp_per_year_stock": float(full[f"{p}_stock_turnover"].sum() / years * 0.00025 * 100),
            "engine_slippage_2.5bps_cost_pp_per_year_bil": float(full[f"{p}_bil_turnover"].sum() / years * 0.00025 * 100),
            "mean_idle_cash_frac_cash_run": float(full[f"{p}_cashrun_idle_cash_frac"].mean()),
            "stock_turnover_per_year_xNAV_buys_plus_sells": float(full[f"{p}_stock_turnover"].sum() / years),
            "stock_turnover_per_year_xNAV_cash_run": float(full[f"{p}_cashrun_stock_turnover"].sum() / years),
            "bil_turnover_per_year_xNAV_buys_plus_sells": float(full[f"{p}_bil_turnover"].sum() / years),
            "bil_turnover_per_year_since_2007-06-04": float(bil_live[f"{p}_bil_turnover"].sum() / (len(bil_live) / 252.0)),
            "avg_n_stock_positions_all_sessions": float(full[f"{p}_n_stock_positions"].mean()),
            "avg_n_stock_positions_when_invested": float(full.loc[full[f"{p}_n_stock_positions"] > 0, f"{p}_n_stock_positions"].mean()),
            "share_sessions_with_stock": float((full[f"{p}_n_stock_positions"] > 0).mean()),
            "share_sessions_negative_cash_bil_run": float((new[(p, "bil")][0]["cash"].loc[FULL_START:] < 0).mean()),
            "share_sessions_negative_cash_cash_run": float((new[(p, "cash")][0]["cash"].loc[FULL_START:] < 0).mean()),
            "min_cash_frac_bil_run": float(full[f"{p}_bilrun_cash_frac"].min()),
            "min_cash_frac_cash_run": float(full[f"{p}_cashrun_idle_cash_frac"].min()),
            "engine_commission_pp_per_year": float(daily[(p, "bil")]["commission"].loc[FULL_START:].sum() / years * 100),
        }
    rep["exposure"]["capsule"] = {
        "mean_bil_weight_full_5050": float((0.5 * full["dv2_bil_weight"] + 0.5 * full["hpi_bil_weight"]).mean()),
        "mean_bil_weight_since_2007-06-04_5050": float((0.5 * bil_live["dv2_bil_weight"] + 0.5 * bil_live["hpi_bil_weight"]).mean()),
        "stock_turnover_per_year_xNAV_5050": float((0.5 * full["dv2_stock_turnover"] + 0.5 * full["hpi_stock_turnover"]).sum() / years),
        "bil_turnover_per_year_xNAV_5050": float((0.5 * full["dv2_bil_turnover"] + 0.5 * full["hpi_bil_turnover"]).sum() / years),
        "gate_open_share_of_sessions": float(full["gate_open_at_prev_close"].mean()),
        "share_sessions_any_pod_negative_cash": float(((new[("dv2", "bil")][0]["cash"].loc[FULL_START:] < 0) | (new[("hpi", "bil")][0]["cash"].loc[FULL_START:] < 0)).mean()),
    }
    # ---- 5. extra-cost estimates (+5 bps per side on top of the engine's own 2.5 bps slippage and commissions)
    bps = 0.0005
    cost = {}
    for label, with_bil in (("plus5bps_stock_only", False), ("plus5bps_all_trades_incl_BIL", True)):
        adj = {}
        for k, p in (("DV2", "dv2"), ("HPI", "hpi")):
            t = tidy[f"{p}_stock_turnover"] + (tidy[f"{p}_bil_turnover"] if with_bil else 0.0)
            adj[k] = tidy[f"{p}_bil_ret"] - bps * t.fillna(0.0)
        c = capsule(adj)
        cost[label] = {w: {"capsule_bil": stats(c.loc[a:b]), "dv2_bil": stats(adj["DV2"].loc[a:b]), "hpi_bil": stats(adj["HPI"].loc[a:b])}
                       for w, (a, b) in WINDOWS.items() if w != "after 2026-08-19"}
        tidy[f"capsule_bil_ret_{label}"] = c.reindex(idx)
    rep["extra_cost"] = cost
    tidy.to_csv(OUT / "mr_capsule_daily.csv", float_format="%.10g")
    # yearly returns for reference
    rep["calendar_years"] = {n: {int(y): float(v) for y, v in ((1 + s.loc[FULL_START:].dropna()).groupby(s.loc[FULL_START:].dropna().index.year).prod() - 1).items()}
                             for n, s in (("capsule_bil", cap["bil"]), ("capsule_cash", cap["cash"]), ("dv2_bil", tidy["dv2_bil_ret"]), ("hpi_bil", tidy["hpi_bil_ret"]))}
    (OUT / "rerun_mr_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("diagnostics", "calendar_years")}, indent=1, default=str))


if __name__ == "__main__":
    main()
