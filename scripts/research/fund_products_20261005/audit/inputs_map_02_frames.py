"""inputs_map audit, step 2: the shelf-rebuild frames, the TAA csv comparison and the two book models.

Read-only on MAIN (inputs via lib.load_inputs of shelf_rebuild_20260929). Writes
results/research/portfolio/fund_products_20261005/audit/inputs_map/02_frames.json and prints tables.

Usage (from the worktree root): PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/Scripts/python.exe <this file>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
import data.norgate_loader  # noqa: E402,F401  (worktree copy first, so MAIN's edited data package is never imported)

sys.path.insert(0, str(WT / "scripts" / "research"))
sys.path.insert(0, str(WT / "scripts" / "research" / "growth_aggressive_20260930"))
import ga_lib as ga  # noqa: E402
from trend_breakout_20260927 import common as tbc  # noqa: E402

lib = ga.lib
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/inputs_map"
MAIN = ga.MAIN_REPO
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def cagr252(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    data = lib.load_inputs()
    index = data["index"]
    meta = data["meta"]
    print("data.norgate_loader from:", data_norgate_path())
    print("index", index[0].date(), index[-1].date(), len(index))
    rep["index"] = [str(index[0].date()), str(index[-1].date()), len(index)]

    # ---- 1. columns of every frame
    frames = ["sleeve", "long", "long_unscaled", "long_etf_cash", "stressed_long", "stressed_exact", "cash_long", "cash_exact"]
    rows = []
    for alias in data["cash_long"].columns:
        row = {"alias": alias, "tier": lib.tier_of(alias, meta)}
        if alias in meta:
            m = meta[alias]
            row.update(path_first=m["engine_first_date_str"], first_invested=m["first_invested_date_str"], end=m["end_date_str"],
                       capital=m["reference_capital_float"], mean_cash_w=m["mean_cash_nav_weight_float"],
                       min_cash_w=m["minimum_cash_nav_weight_float"], module=m["strategy_import_str"])
        for f in ("sleeve", "long", "cash_long", "stressed_long"):
            s = data[f][alias]
            fv = s.first_valid_index()
            row[f"{f}_first"] = None if fv is None else str(fv.date())
            row[f"{f}_last"] = str(s.last_valid_index().date())
            row[f"{f}_nan_inside"] = int(s.loc[fv:].isna().sum())
        lo = max(lib.LONG_START, data["long"][alias].first_valid_index())
        add = (data["cash_long"][alias] - data["long"][alias]).loc[lo:]
        drag = (data["long"][alias] - data["stressed_long"][alias]).loc[lo:]
        row["cash_add_pp_yr_long"] = float(add.mean() * 252 * 100)
        row["cash_add_pp_yr_recent"] = float(add.loc["2023-08-21":].mean() * 252 * 100)
        row["cash_add_min_daily_bp"] = float(add.min() * 1e4)
        row["stress_drag_pp_yr_long"] = float(drag.mean() * 252 * 100)
        row["stress_drag_pp_yr_exact"] = float(drag.loc[lib.EXACT_START:].mean() * 252 * 100)
        row["unscaled_diff_max_abs"] = float((data["long_unscaled"][alias] - data["long"][alias]).abs().max())
        row["etf_cash_diff_max_abs"] = float((data["long_etf_cash"][alias] - data["long"][alias]).abs().max())
        row["long_vs_sleeve_pre2012_differs"] = bool((data["long"][alias].loc[:"2012-10-01"].fillna(-9) != data["sleeve"][alias].loc[:"2012-10-01"].fillna(-9)).any())
        row["long_vs_sleeve_post2012_max_abs"] = float((data["long"][alias] - data["sleeve"][alias]).loc[lib.EXACT_START:].abs().max())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("alias")
    table.to_csv(OUT / "02_columns.csv")
    print(table[["tier", "path_first", "first_invested", "end", "sleeve_first", "long_first", "long_last", "long_nan_inside",
                 "long_vs_sleeve_pre2012_differs", "long_vs_sleeve_post2012_max_abs"]].to_string())
    print(table[["mean_cash_w", "min_cash_w", "cash_add_pp_yr_long", "cash_add_pp_yr_recent", "cash_add_min_daily_bp",
                 "stress_drag_pp_yr_long", "stress_drag_pp_yr_exact", "unscaled_diff_max_abs", "etf_cash_diff_max_abs"]].round(4).to_string())
    for f in frames:
        fr = data[f]
        print(f"frame {f}: shape {fr.shape}, first {fr.index[0].date()}, last {fr.index[-1].date()}, cols {len(fr.columns)}")
    rep["frames"] = {f: {"shape": list(data[f].shape), "first": str(data[f].index[0].date()), "last": str(data[f].index[-1].date())} for f in frames}
    rep["bench_cols"] = list(data["bench"].columns)
    tb = data["sleeve"][lib.TBILL]
    rep["tbill"] = {"first_valid": str(tb.first_valid_index().date()), "cagr_long": cagr252(tb.loc[lib.LONG_START:]),
                    "cagr_recent": cagr252(tb.loc["2023-08-21":]),
                    "dtb3_accrual_cagr_long": cagr252(data["bench"]["DTB3"].loc[lib.LONG_START:]),
                    "dtb3_accrual_cagr_recent": cagr252(data["bench"]["DTB3"].loc["2023-08-21":])}
    print("tbill", rep["tbill"])
    rate = lib.dtb3_annual_rate(index)
    rep["dtb3_rate"] = {"last_value": float(rate.iloc[-1]), "mean_long": float(rate.loc[lib.LONG_START:].mean()),
                        "mean_recent": float(rate.loc["2023-08-21":].mean()), "nan": int(rate.isna().sum())}
    print("dtb3 annual rate", rep["dtb3_rate"])

    # proxy splice facts
    proxy = {}
    for alias in lib.PROXY_ALIAS_LIST:
        p = lib.read_path(lib.PROXY / "splice_scaled", alias)
        real = lib.read_path(lib.SOURCE, alias)
        pr = lib.nav_to_returns(p)
        rr = lib.nav_to_returns(real)
        both = pd.concat([pr, rr], axis=1, keys=["proxy", "real"]).dropna()
        proxy[alias] = {"proxy_path_first": str(p.index[0].date()), "proxy_first_return": str(pr.index[0].date()),
                        "proxy_path_last": str(p.index[-1].date()), "real_path_first": str(real.index[0].date()),
                        "real_first_return": str(rr.index[0].date()),
                        "overlap_from": str(both.index[0].date()), "overlap_max_abs_diff": float((both["proxy"] - both["real"]).abs().max()),
                        "overlap_corr": float(both.corr().iloc[0, 1]),
                        "long_equals_proxy_pre_cut": float((data["long"][alias] - pr.reindex(index)).loc[:"2012-10-01"].abs().max()),
                        "long_equals_real_post_cut": float((data["long"][alias] - rr.reindex(index)).loc[lib.EXACT_START:].abs().max()),
                        "real_capital": float(real["total_value_float"].iloc[0]), "proxy_nav_first": float(p["total_value_float"].iloc[0])}
    rep["proxy"] = proxy
    print(pd.DataFrame(proxy).T.to_string())

    # ---- 4(iii). TAA csv vs the shelf-rebuild frames
    csv = pd.read_csv(tbc.SLEEVE_SERIES_PATH, index_col=0, parse_dates=True)
    rep["csv"] = {"path": str(tbc.SLEEVE_SERIES_PATH), "first": str(csv.index[0].date()), "last": str(csv.index[-1].date()),
                  "rows": len(csv), "columns": list(csv.columns)}
    cmp_rows = []
    pairs = {"taa_btal_tqqq": "taa3x", "taa_btal_1n_tqqq": "taa3x_1n", "taa_btal_lin_qqq": "btal_qqq", "ndx_vxn": "ndx_vxn",
             "ndx_natr20": "ndx_natr20", "ndx_atrfix": "ndx_vxn", "dv2": "dv2", "hpi_vote": "hpi_vote", "core5": "core5"}
    for col, alias in pairs.items():
        for f in ("long", "cash_long", "long_unscaled", "stressed_long"):
            for wname, (a, b) in {"all": ("2008-03-04", "2026-08-19"), "pre_cut": ("2008-03-04", "2012-10-01"), "post_cut": ("2012-10-02", "2026-08-19")}.items():
                both = pd.concat([csv[col], data[f][alias]], axis=1, keys=["csv", "frame"]).loc[a:b].dropna()
                d = both["csv"] - both["frame"]
                cmp_rows.append({"csv_col": col, "alias": alias, "frame": f, "window": wname, "n": len(both),
                                 "first": str(both.index[0].date()), "last": str(both.index[-1].date()),
                                 "max_abs_diff": float(d.abs().max()), "mean_abs_diff": float(d.abs().mean()),
                                 "corr": float(both.corr().iloc[0, 1]), "cagr_csv": cagr252(both["csv"]), "cagr_frame": cagr252(both["frame"]),
                                 "days_abs_diff_gt_1bp": int((d.abs() > 1e-4).sum())})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT / "02_taa_csv_vs_frames.csv", index=False)
    print(cmp[cmp["csv_col"].isin(["taa_btal_tqqq", "taa_btal_1n_tqqq"])].round(6).to_string())
    print(cmp[~cmp["csv_col"].isin(["taa_btal_tqqq", "taa_btal_1n_tqqq"]) & (cmp["frame"] == "long")].round(6).to_string())
    # is the csv's older source reproducible? inventory (fund_product_menu_20260923) + growth_shelf_v2 proxies
    inv_path = MAIN / "results/research/portfolio/fund_product_menu_20260923/inventory/sleeve_returns.csv.gz"
    if inv_path.exists():
        inv = pd.read_csv(inv_path, index_col=0, parse_dates=True)
        for col in ("taa_btal_tqqq", "taa_btal_1n_tqqq"):
            if col in inv:
                both = pd.concat([csv[col], inv[col]], axis=1).loc["2012-10-02":"2026-08-19"].dropna()
                rep.setdefault("csv_vs_menu_inventory_post_cut", {})[col] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
        v2 = MAIN / "results/research/portfolio/growth_shelf_v2_20260926/proxy_runs/splice_scaled"
        for col in ("taa_btal_tqqq", "taa_btal_1n_tqqq"):
            f = v2 / f"{col}__path.csv.gz"
            if f.exists():
                nav = pd.read_csv(f, index_col="date", parse_dates=True)["total_value_float"]
                both = pd.concat([csv[col], nav.pct_change()], axis=1).loc["2008-03-04":"2012-10-01"].dropna()
                rep.setdefault("csv_vs_growth_v2_proxy_pre_cut", {})[col] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    print("csv provenance checks", rep.get("csv_vs_menu_inventory_post_cut"), rep.get("csv_vs_growth_v2_proxy_pre_cut"))

    # ---- 4(end). book model: tbc.book_window_return_ser vs lib.book_returns vs common.book_return_ser
    fr = data["long"]
    w3 = {"taa3x": 0.5, "ndx_vxn": 0.25, "core5": 0.25}
    a = lib.book_returns(fr, lib.Book("x", tuple(w3), "EQ", w3), lib.LONG_START)
    b = tbc.book_window_return_ser({k: fr[k] for k in w3}, w3, "2008-03-04", "2026-08-19")
    c = lib.common.book_return_ser(fr.loc[lib.LONG_START:lib.END, list(w3)], w3, "annual")[0]
    d_daily = tbc.book_window_return_ser({k: fr[k] for k in w3}, w3, "2008-03-04", "2026-08-19", daily_bool=True)
    rep["book_model"] = {"weights": w3, "n": len(a), "same_index": bool(a.index.equals(b.index)),
                         "max_abs_diff_lib_vs_tbc": float((a - b).abs().max()), "max_abs_diff_lib_vs_common": float((a - c).abs().max()),
                         "max_abs_diff_lib_vs_daily_rebalanced": float((a - d_daily).abs().max()),
                         "cagr_lib": cagr252(a), "cagr_tbc": cagr252(b), "cagr_daily_rebalanced": cagr252(d_daily)}
    # reset-day evidence: the pod weights before the first session of 2009 must equal the targets
    _, prior_w = lib.common.book_return_ser(fr.loc[lib.LONG_START:lib.END, list(w3)], w3, "annual")
    first_2009 = prior_w.loc["2009"].iloc[0]
    last_2008 = prior_w.loc["2008"].iloc[-1]
    rep["book_model"]["prior_weights_first_session_2009"] = {k: float(v) for k, v in first_2009.items()}
    rep["book_model"]["prior_weights_last_session_2008"] = {k: float(v) for k, v in last_2008.items()}
    rep["book_model"]["first_session_2009"] = str(prior_w.loc["2009"].index[0].date())
    # a mid-window start: each window is its own run from target weights (tbc) vs slicing a longer run
    b_mid = tbc.book_window_return_ser({k: fr[k] for k in w3}, w3, "2017-11-01", "2026-08-19")
    a_mid = lib.book_returns(fr, lib.Book("x", tuple(w3), "EQ", w3), pd.Timestamp("2017-11-01"))
    rep["book_model"]["mid_start_max_abs_diff_lib_vs_tbc"] = float((a_mid - b_mid).abs().max())
    rep["book_model"]["mid_start_vs_sliced_long_run_max_abs_diff"] = float((a.loc["2017-11-01":] - b_mid).abs().max())
    print("book model", json.dumps(rep["book_model"], indent=1))

    # ---- frames of ga.frames
    gf = ga.frames(data)
    rep["ga_frames"] = {}
    for k, (f, s) in gf.items():
        rep["ga_frames"][k] = {"start": str(s.date()), "shape": list(f.shape)}
    chk = (gf["s3_plus_5bps"][0] - (data["stressed_long"] + (data["cash_long"] - data["long"]).fillna(0.0))).abs().max().max()
    rep["ga_frames"]["s3_identity_max_abs"] = float(chk)
    print("ga frames", rep["ga_frames"])
    (OUT / "02_frames.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    # save the four TAA columns for later steps
    data["long"].to_pickle(OUT / "frame_long.pkl")
    data["cash_long"].to_pickle(OUT / "frame_cash_long.pkl")
    data["stressed_long"].to_pickle(OUT / "frame_stressed_long.pkl")
    data["bench"].to_pickle(OUT / "frame_bench.pkl")
    rate.to_pickle(OUT / "dtb3_rate.pkl")
    print("written", OUT / "02_frames.json")


def data_norgate_path() -> str:
    import data.norgate_loader as nl
    return str(Path(nl.__file__).resolve())


if __name__ == "__main__":
    main()
