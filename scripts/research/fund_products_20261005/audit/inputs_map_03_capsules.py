"""inputs_map audit, step 3: the MR capsule pods and the E2 momentum capsule against the shelf-rebuild conventions.

Read-only on MAIN. Needs step 2's saved frames. Writes .../audit/inputs_map/03_capsules.json and prints tables.

What it measures:
- MR capsule engine runs (cash / bil / parked): NAV columns, capital, cash and BIL weights, house / fair-cash / BIL returns,
  +5 bps drag from the fills, PM-run pods ($500K) vs research runs ($100K), ev.capsule vs two pod columns in lib.book_returns;
- E2: the PM pickle vs a monthly-reset replica from its two pods, pod cash weights, fair-cash add, +5 bps drag;
- reproduction of one book_weights.json row with these loaders (the capsule-era code path).
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT))
import data.norgate_loader as nl  # noqa: E402  (worktree copy first)

sys.path.insert(0, str(WT / "scripts" / "research"))
sys.path.insert(0, str(WT / "scripts" / "research" / "growth_aggressive_20260930"))
import ga_lib as ga  # noqa: E402
from trend_breakout_20260927 import common as tbc  # noqa: E402

lib = ga.lib
MAIN = ga.MAIN_REPO
# *** CRITICAL*** lib.py puts the MAIN checkout at sys.path[0]; MAIN holds another session's uncommitted edits
# (strategies/hpi/stateful_long.py, data/norgate_snapshot_store.py, ...). Unpickling the pods imports `strategies.*`,
# so the MAIN root is removed here and the worktree (= main HEAD) is searched first.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != MAIN.resolve()]
sys.path.insert(0, str(WT))
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/inputs_map"
CAP = MAIN / "results/research/mr_capsule_build_20261004"
PM_CAP = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751"
PM_E2 = MAIN / "results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest/2026-10-04_095907"
WIN = {"LONG": ("2008-03-04", "2026-08-19"), "EXACT": ("2012-10-02", "2026-08-19"), "RECENT": ("2023-08-21", "2026-08-19")}
PARK = ("BIL", "SPMO")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def m(r: pd.Series) -> dict:
    r = r.dropna()
    v = np.r_[1.0, np.cumprod(1 + r.to_numpy())]
    return {"cagr": float(v[-1] ** (252 / len(r)) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "maxdd": float((v / np.maximum.accumulate(v) - 1).min()), "n": int(len(r))}


def wins(r: pd.Series) -> dict:
    return {k: m(r.loc[a:b]) for k, (a, b) in WIN.items()}


def path_from_nav(nav: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"total_value_float": nav["total_value"].astype(float), "portfolio_value_float": nav["portfolio_value"].astype(float),
                         "cash_float": nav["cash"].astype(float)})


def tx_std(tx: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(tx["bar"]).dt.normalize(), "asset_str": tx["asset"].astype(str),
                         "signed_notional_float": tx["amount"].astype(float) * tx["price"].astype(float)})


def reset_book(ret: pd.DataFrame, w: dict, freq: str) -> pd.Series:
    """Pod model with a reset to targets after the last close of each period (year or month)."""
    idx = ret.index
    lab = idx.year.to_numpy() if freq == "annual" else (idx.year * 100 + idx.month).to_numpy()
    wa = np.array([w[c] for c in ret.columns])
    pod = wa.copy()
    out = np.empty(len(idx))
    x = ret.to_numpy()
    for t in range(len(idx)):
        before = pod.sum()
        pod = pod * (1 + x[t])
        out[t] = pod.sum() / before - 1
        if t + 1 < len(idx) and lab[t + 1] != lab[t]:
            pod = wa * pod.sum()
    return pd.Series(out, index=idx)


def main() -> None:
    rep: dict = {}
    long = pd.read_pickle(OUT / "frame_long.pkl")
    cash_long = pd.read_pickle(OUT / "frame_cash_long.pkl")
    stressed_long = pd.read_pickle(OUT / "frame_stressed_long.pkl")
    index = long.index
    bil_px = nl.load_price_timeseries("BIL", adjustment_str=nl.CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str="2007-01-01", end_date_str=None)["Close"].astype(float)
    bil_px.index = pd.to_datetime(bil_px.index).normalize()
    bil_tr = lib.common.load_total_return_close_ser("BIL", "2007-01-01", "2026-12-31").pct_change(fill_method=None)
    rep["bil"] = {"first_bar": str(bil_px.index[0].date()), "last_bar": str(bil_px.index[-1].date())}

    # ------------------------------------------------------------------ MR capsule engine runs
    pods: dict = {}
    series: dict = {}
    for pod in ("dv2", "hpi"):
        for mode in ("cash", "bil", "parked"):
            nav = pd.read_csv(CAP / f"{pod}_{mode}_nav.csv", index_col=0, parse_dates=True).astype(float)
            tx = pd.read_csv(CAP / f"{pod}_{mode}_transactions.csv", parse_dates=["bar"])
            diag = json.loads((CAP / f"{pod}_{mode}_diagnostics.json").read_text(encoding="utf-8"))
            path = path_from_nav(nav)
            rate = lib.dtb3_annual_rate(path.index)
            r = nav["total_value"].pct_change()
            add = lib.cash_realism_add(path, rate)
            t = tx_std(tx)
            drag_all = lib.evaluation.extra_slippage_cost_ser(t, nav["total_value"], 0.0005)
            drag_stock = lib.evaluation.extra_slippage_cost_ser(t[~t["asset_str"].isin(PARK)], nav["total_value"], 0.0005)
            cw = (nav["cash"] / nav["total_value"])
            invested = nav["portfolio_value"].abs() > 1e-9
            shares = tx[tx["asset"] == "BIL"].groupby("bar")["amount"].sum().reindex(nav.index).fillna(0.0).cumsum()
            bil_w = (shares * bil_px.reindex(nav.index)).fillna(0.0) / nav["total_value"]
            key = f"{pod}_{mode}"
            series[key] = {"r": r, "add": add, "drag_all": drag_all, "drag_stock": drag_stock, "cw": cw, "nav": nav, "tx": t}
            lo, hi = WIN["LONG"]
            years_long = len(r.loc[lo:hi]) / 252
            pods[key] = {
                "nav_columns": list(nav.columns), "tx_columns": list(tx.columns), "first": str(nav.index[0].date()), "last": str(nav.index[-1].date()),
                "capital": float(nav["total_value"].iloc[0]), "first_invested": str(invested[invested].index[0].date()),
                "first_bil_trade": str(tx.loc[tx["asset"] == "BIL", "bar"].min().date()) if (tx["asset"] == "BIL").any() else None,
                "bil_orders": int((tx["asset"] == "BIL").sum()), "spmo_orders": int((tx["asset"] == "SPMO").sum()),
                "stock_orders": int((~tx["asset"].isin(PARK)).sum()),
                "mean_cash_w_long": float(cw.loc[lo:hi].mean()), "min_cash_w_long": float(cw.loc[lo:hi].min()),
                "mean_cash_w_recent": float(cw.loc["2023-08-21":hi].mean()),
                "mean_bil_w_long": float(bil_w.loc[lo:hi].mean()), "mean_bil_w_recent": float(bil_w.loc["2023-08-21":hi].mean()),
                "negative_cash_days_long": int((nav["cash"].loc[lo:hi] < 0).sum()),
                "fair_add_pp_yr_long": float(add.loc[lo:hi].mean() * 252 * 100), "fair_add_pp_yr_recent": float(add.loc["2023-08-21":hi].mean() * 252 * 100),
                "drag_all_pp_yr_long": float(drag_all.loc[lo:hi].mean() * 252 * 100), "drag_stock_pp_yr_long": float(drag_stock.loc[lo:hi].mean() * 252 * 100),
                "one_way_turnover_x_per_yr_long_all": float(drag_all.loc[lo:hi].sum() / 0.0005 / years_long),
                "one_way_turnover_x_per_yr_long_stock": float(drag_stock.loc[lo:hi].sum() / 0.0005 / years_long),
                "house": wins(r), "fair": wins(r + add), "plus5_all_fair": wins(r + add - drag_all), "plus5_stock_fair": wins(r + add - drag_stock),
                "parking_note": diag.get("capsule_parking_str"), "withholding_note": diag.get("parking_dividend_withholding_note_str"),
            }
        # research sweep convention on the cash-only run: r + cash weight(t-1) x BIL TR
        s = series[f"{pod}_cash"]
        swept = s["r"] + s["cw"].clip(lower=0).shift(1).fillna(0.0) * bil_tr.reindex(s["r"].index).fillna(0.0)
        pods[f"{pod}_cash"]["swept_bil_tr"] = wins(swept)
        series[f"{pod}_cash"]["swept"] = swept
    rep["mr_pods"] = pods
    show = pd.DataFrame({k: {"first": v["first"], "last": v["last"], "capital": v["capital"], "first_inv": v["first_invested"], "first_bil": v["first_bil_trade"],
                             "cash_w": round(v["mean_cash_w_long"], 4), "bil_w": round(v["mean_bil_w_long"], 4), "neg_cash_d": v["negative_cash_days_long"],
                             "fair_add_pp": round(v["fair_add_pp_yr_long"], 3), "fair_add_recent_pp": round(v["fair_add_pp_yr_recent"], 3),
                             "drag_all_pp": round(v["drag_all_pp_yr_long"], 3), "drag_stock_pp": round(v["drag_stock_pp_yr_long"], 3),
                             "turn_all_x": round(v["one_way_turnover_x_per_yr_long_all"], 1), "turn_stock_x": round(v["one_way_turnover_x_per_yr_long_stock"], 1),
                             "house_cagr_long": round(v["house"]["LONG"]["cagr"], 4), "fair_cagr_long": round(v["fair"]["LONG"]["cagr"], 4),
                             "house_cagr_recent": round(v["house"]["RECENT"]["cagr"], 4), "fair_cagr_recent": round(v["fair"]["RECENT"]["cagr"], 4)} for k, v in pods.items()}).T
    print(show.to_string())
    print("swept (cash run + cw x BIL TR) LONG/RECENT:", {p: (round(pods[f"{p}_cash"]["swept_bil_tr"]["LONG"]["cagr"], 4), round(pods[f"{p}_cash"]["swept_bil_tr"]["RECENT"]["cagr"], 4)) for p in ("dv2", "hpi")})

    # capsule composites (ev.capsule = tbc.book_return_ser(frame.loc[start:end].dropna(), w, "annual"))
    def capsule(parts: dict, start="2004-01-05", end="2026-09-24") -> pd.Series:
        frame = pd.DataFrame(parts).loc[start:end].dropna()
        return tbc.book_return_ser(frame, {"DV2": 0.5, "HPI": 0.5}, "annual")[0]

    cap = {
        "house (cash runs, 0%)": capsule({"DV2": series["dv2_cash"]["r"], "HPI": series["hpi_cash"]["r"]}),
        "fair (cash runs + DTB3-0.5%)": capsule({"DV2": series["dv2_cash"]["r"] + series["dv2_cash"]["add"], "HPI": series["hpi_cash"]["r"] + series["hpi_cash"]["add"]}),
        "swept (cash runs + cw x BIL TR)": capsule({"DV2": series["dv2_cash"]["swept"], "HPI": series["hpi_cash"]["swept"]}),
        "BIL held (bil runs, as book_weights.py)": capsule({"DV2": series["dv2_bil"]["r"], "HPI": series["hpi_bil"]["r"]}),
        "BIL held + fair add on residual cash": capsule({"DV2": series["dv2_bil"]["r"] + series["dv2_bil"]["add"], "HPI": series["hpi_bil"]["r"] + series["hpi_bil"]["add"]}),
        "fair, +5 bps stock fills": capsule({p.upper(): series[f"{p}_cash"]["r"] + series[f"{p}_cash"]["add"] - series[f"{p}_cash"]["drag_stock"] for p in ("dv2", "hpi")}),
        "BIL held, +5 bps all fills": capsule({p.upper(): series[f"{p}_bil"]["r"] - series[f"{p}_bil"]["drag_all"] for p in ("dv2", "hpi")}),
        "BIL held, +5 bps stock fills only": capsule({p.upper(): series[f"{p}_bil"]["r"] - series[f"{p}_bil"]["drag_stock"] for p in ("dv2", "hpi")}),
        "SPMO parked (parked runs)": capsule({"DV2": series["dv2_parked"]["r"], "HPI": series["hpi_parked"]["r"]}),
    }
    rep["mr_capsule"] = {k: wins(v) for k, v in cap.items()}
    rep["mr_capsule"]["_first_last"] = [str(cap["BIL held (bil runs, as book_weights.py)"].index[0].date()), str(cap["BIL held (bil runs, as book_weights.py)"].index[-1].date())]
    print(pd.DataFrame({k: {f"{w}_{q}": round(v[w][q], 4) for w in WIN for q in ("cagr", "sharpe", "maxdd")} for k, v in rep["mr_capsule"].items() if not k.startswith("_")}).T.to_string())
    tb = {w: m(bil_tr.loc[a:b])["cagr"] for w, (a, b) in WIN.items()}
    rate_all = lib.dtb3_annual_rate(series["dv2_cash"]["r"].index)
    days = pd.Series(rate_all.index, index=rate_all.index).diff().dt.days
    fair_rate = ((rate_all - 0.005).clip(lower=0) * days / 360)
    rep["cash_rates"] = {"bil_tr_cagr": tb, "fair_cash_cagr(DTB3-0.5%)": {w: m(fair_rate.loc[a:b])["cagr"] for w, (a, b) in WIN.items()}}
    print("cash rates", rep["cash_rates"])

    # PM-run pods ($500K) vs research runs ($100K), and the PM capsule book vs ev.capsule
    pm = {}
    with open(PM_CAP / "mr_capsule_bil.pkl", "rb") as fh:
        pm_cap = pickle.load(fh)
    pm_book_r = pm_cap.results["total_value"].astype(float).pct_change().dropna()
    pm["book_first_last"] = [str(pm_cap.results.index[0].date()), str(pm_cap.results.index[-1].date())]
    pm["rebalance"] = [pm_cap._rebalance, pm_cap._rebalance_policy]
    pm_pod_r = {}
    for s_obj, pod in zip(pm_cap.strategies, ("dv2", "hpi")):
        res = s_obj.results
        r_pm = res["total_value"].astype(float).pct_change()
        r_rs = series[f"{pod}_bil"]["r"]
        both = pd.concat([r_pm, r_rs], axis=1, keys=["pm", "research"]).dropna()
        pm_pod_r[pod.upper()] = r_pm
        pm[pod] = {"name": s_obj.name, "capital": float(res["total_value"].iloc[0]), "result_columns": [c for c in res.columns][:12],
                   "max_abs_daily_diff_vs_100k": float((both["pm"] - both["research"]).abs().max()), "corr": float(both.corr().iloc[0, 1]),
                   "cagr_pm_long": m(both["pm"].loc[WIN["LONG"][0]:WIN["LONG"][1]])["cagr"], "cagr_100k_long": m(both["research"].loc[WIN["LONG"][0]:WIN["LONG"][1]])["cagr"],
                   "cagr_pm_full": m(both["pm"])["cagr"], "cagr_100k_full": m(both["research"])["cagr"]}
    rep_pm_cap = capsule(pm_pod_r, start=str(pm_cap.results.index[0].date()), end="2026-10-02")
    both = pd.concat([pm_book_r, rep_pm_cap], axis=1).dropna()
    pm["pm_book_vs_annual_reset_of_its_pods_max_abs"] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    both = pd.concat([pm_book_r, cap["BIL held (bil runs, as book_weights.py)"]], axis=1).dropna()
    pm["pm_book_vs_ev_capsule_100k_max_abs"] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    pm["pm_book_long"] = wins(pm_book_r)
    rep["mr_pm"] = pm
    print("PM capsule", json.dumps(pm, indent=1))
    del pm_cap

    # ------------------------------------------------------------------ E2
    with open(PM_E2 / "ndx_e2_sector_cap_5050.pkl", "rb") as fh:
        e2 = pickle.load(fh)
    e2_r = e2.results["total_value"].astype(float).pct_change().dropna()
    e2_rep: dict = {"first": str(e2.results.index[0].date()), "last": str(e2.results.index[-1].date()), "rebalance": [e2._rebalance, e2._rebalance_policy],
                    "capital": float(e2._capital_base), "weights": list(map(float, e2.weights))}
    pod_r, pod_fair, pod_p5, pod_p5_fair = {}, {}, {}, {}
    names = []
    for s_obj, folder in zip(e2.strategies, ("pod_ndx_atr_vxn_sector_cap", "pod_ndx_natr20_vxn_sector_cap")):
        res = s_obj.results
        names.append(s_obj.name)
        nav = res[["total_value", "cash", "portfolio_value"]].astype(float)
        path = path_from_nav(nav)
        rate = lib.dtb3_annual_rate(path.index)
        r = nav["total_value"].pct_change().fillna(0.0)          # PM anchors the first common row at 0
        add = lib.cash_realism_add(path, rate)
        tx = pd.read_csv(PM_E2 / "pods" / folder / "transactions.csv", parse_dates=["bar"])
        drag = lib.evaluation.extra_slippage_cost_ser(tx_std(tx), nav["total_value"], 0.0005)
        cw = nav["cash"] / nav["total_value"]
        invested = nav["portfolio_value"].abs() > 1e-9
        lo, hi = WIN["LONG"]
        e2_rep[s_obj.name] = {"capital": float(nav["total_value"].iloc[0]), "first": str(nav.index[0].date()), "last": str(nav.index[-1].date()),
                              "first_invested": str(invested[invested].index[0].date()), "result_columns": list(res.columns)[:12], "tx_columns": list(tx.columns),
                              "mean_cash_w_long": float(cw.loc[lo:hi].mean()), "mean_cash_w_recent": float(cw.loc["2023-08-21":hi].mean()),
                              "min_cash_w_long": float(cw.loc[lo:hi].min()), "negative_cash_days_long": int((nav["cash"].loc[lo:hi] < 0).sum()),
                              "fair_add_pp_yr_long": float(add.loc[lo:hi].mean() * 252 * 100), "fair_add_pp_yr_recent": float(add.loc["2023-08-21":hi].mean() * 252 * 100),
                              "drag_pp_yr_long": float(drag.loc[lo:hi].mean() * 252 * 100), "house": wins(r), "fair": wins(r + add)}
        pod_r[s_obj.name], pod_fair[s_obj.name], pod_p5[s_obj.name], pod_p5_fair[s_obj.name] = r, r + add, r - drag, r + add - drag
    w = dict(zip(names, e2.weights))
    replica = reset_book(pd.DataFrame(pod_r), w, "monthly").iloc[1:]
    both = pd.concat([e2_r, replica], axis=1).dropna()
    e2_rep["pm_vs_monthly_reset_replica_max_abs"] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    annual = reset_book(pd.DataFrame(pod_r), w, "annual").iloc[1:]
    e2_rep["monthly_vs_annual_internal_reset_max_abs"] = float((replica - annual).abs().max())
    e2_series = {"house (PM pickle)": e2_r, "house (annual internal reset)": annual,
                 "fair cash": reset_book(pd.DataFrame(pod_fair), w, "monthly").iloc[1:],
                 "+5 bps (house)": reset_book(pd.DataFrame(pod_p5), w, "monthly").iloc[1:],
                 "+5 bps (fair)": reset_book(pd.DataFrame(pod_p5_fair), w, "monthly").iloc[1:]}
    e2_rep["series"] = {k: wins(v) for k, v in e2_series.items()}
    e2_rep["corr_with_frame_long"] = {a: float(pd.concat([e2_r, long[a]], axis=1).loc[WIN["LONG"][0]:WIN["LONG"][1]].dropna().corr().iloc[0, 1]) for a in ("ndx_vxn", "ndx_atr", "ndx_natr20")}
    rep["e2"] = e2_rep
    print("E2", json.dumps({k: v for k, v in e2_rep.items() if k != "series"}, indent=1))
    print(pd.DataFrame({k: {f"{wn}_{q}": round(v[wn][q], 4) for wn in WIN for q in ("cagr", "sharpe", "maxdd")} for k, v in e2_rep["series"].items()}).T.to_string())
    del e2

    # ------------------------------------------------------------------ reproduce the capsule-era book (book_weights.py, TAA 3x base row)
    csv = pd.read_csv(tbc.SLEEVE_SERIES_PATH, index_col=0, parse_dates=True)
    mr_bil = cap["BIL held (bil runs, as book_weights.py)"]
    ref = json.loads((CAP / "book_weights.json").read_text(encoding="utf-8"))
    chk = {}
    for taa_name, col in (("TAA 3x", "taa_btal_tqqq"), ("TAA 3x 1N", "taa_btal_1n_tqqq")):
        book = tbc.book_window_return_ser({"taa": csv[col].astype(float), "mom": e2_r, "mr": mr_bil}, {"taa": 0.5, "mom": 0.25, "mr": 0.25}, "2008-03-04", "2026-08-19")
        mine = m(book)
        theirs = ref[taa_name]["Base: TAA 0.5 / Mom 0.25 / MR 0.25"]["2008-26"]
        chk[taa_name] = {"mine": mine, "book_weights_json": {k: theirs[k] for k in ("cagr", "sharpe", "max_dd")}}
    rep["capsule_era_book_reproduction"] = chk
    print("reproduction", json.dumps(chk, indent=1))

    # ------------------------------------------------------------------ same book inside the shelf-rebuild frame: composite vs pod columns
    fr = long.copy()
    fr["e2"] = e2_r.reindex(index)
    fr["mr_capsule"] = mr_bil.reindex(index)
    fr["dv2_g"] = series["dv2_bil"]["r"].reindex(index)
    fr["hpi_g"] = series["hpi_bil"]["r"].reindex(index)
    a = lib.book_returns(fr, lib.Book("A", ("taa3x", "e2", "mr_capsule"), "EQ", {"taa3x": 0.5, "e2": 0.25, "mr_capsule": 0.25}), lib.LONG_START)
    b = lib.book_returns(fr, lib.Book("B", ("taa3x", "e2", "dv2_g", "hpi_g"), "EQ", {"taa3x": 0.5, "e2": 0.25, "dv2_g": 0.125, "hpi_g": 0.125}), lib.LONG_START)
    c = tbc.book_window_return_ser({"taa": csv["taa_btal_tqqq"].astype(float), "mom": e2_r, "mr": mr_bil}, {"taa": 0.5, "mom": 0.25, "mr": 0.25}, "2008-03-04", "2026-08-19")
    rep["composite_vs_pod_columns"] = {"max_abs_daily_diff": float((a - b).abs().max()), "max_abs_diff_after_2008": float((a - b).loc["2009-01-02":].abs().max()),
                                       "cagr_composite": m(a)["cagr"], "cagr_pod_columns": m(b)["cagr"], "maxdd_composite": m(a)["maxdd"], "maxdd_pod_columns": m(b)["maxdd"],
                                       "lib_frame_vs_capsule_era_code_max_abs": float((a - c).abs().max())}
    print("composite vs pod columns", rep["composite_vs_pod_columns"])
    rep["frame_first_valid"] = {k: str(fr[k].first_valid_index().date()) for k in ("e2", "mr_capsule", "dv2_g", "hpi_g")}
    rep["frame_last_valid"] = {k: str(fr[k].last_valid_index().date()) for k in ("e2", "mr_capsule", "dv2_g", "hpi_g")}
    # the three cash conventions for one growth-style book
    conv = {}
    variants = {"house": (long, e2_series["house (PM pickle)"], cap["house (cash runs, 0%)"]),
                "fair": (cash_long, e2_series["fair cash"], cap["fair (cash runs + DTB3-0.5%)"]),
                "fair + BIL-held capsule": (cash_long, e2_series["fair cash"], cap["BIL held + fair add on residual cash"]),
                "house TAA/E2 + BIL-held capsule (capsule-era)": (long, e2_series["house (PM pickle)"], cap["BIL held (bil runs, as book_weights.py)"]),
                "+5 bps (fair)": (stressed_long + (cash_long - long).fillna(0.0), e2_series["+5 bps (fair)"], cap["fair, +5 bps stock fills"])}
    for name, (frame, e2s, mrs) in variants.items():
        f2 = frame[["taa3x", "taa3x_1n"]].copy()
        f2["e2"], f2["mr"] = e2s.reindex(index), mrs.reindex(index)
        for taa in ("taa3x", "taa3x_1n"):
            r = lib.book_returns(f2, lib.Book("x", (taa, "e2", "mr"), "EQ", {taa: 0.5, "e2": 0.25, "mr": 0.25}), lib.LONG_START)
            conv[f"{name} | {taa}"] = m(r)
    rep["book_by_convention_TAA50_E2_25_MR25_LONG"] = conv
    print(pd.DataFrame(conv).T.round(4).to_string())
    (OUT / "03_capsules.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("written", OUT / "03_capsules.json")


if __name__ == "__main__":
    main()
