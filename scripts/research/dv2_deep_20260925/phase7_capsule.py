"""The 12/12/12 growth book in detail (A) and the marginal value of every other MR pod (B). 2026-09-26.

Book: TAA 3x 32 / NDX VXN 32 / DV2 live 12 / HPI vote 12 / industry-ETF DV2 12, annual reset (pod model).
Industry-ETF DV2 = strategies/dv2/strategy_mr_dv2_industry_etf.py run through the real engine from 2012-01-03;
for the 2008 proxy only, the same rules from the research run fill the years before 2012.
Stress: +5 bps per side on every traded dollar of every pod. Capacity: last three years of orders, MOO and MOC
routes at the house auction limits (growth-shelf model); ETF orders are worked within the day.
B: each other MR pod X enters as a fourth leg (DV2 9 / HPI 9 / ETF 9 / X 9), and HPI-like pods also replace HPI;
compared with 12/12/12 on the window where X exists.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/research/growth_shelf_20260924"))
sys.path.insert(0, str(REPO / "scripts/research/fund_menu_20260923"))
import common  # noqa: E402
import evaluation  # noqa: E402
import growth_capacity as gc  # noqa: E402
import growth_dossier as gd  # noqa: E402
from growth_dossier import load_price_timeseries  # noqa: E402

OUT = REPO / "results/research/dv2_deep_20260925"
END = pd.Timestamp("2026-08-19")
CUT = pd.Timestamp("2012-10-02")
STOCK_MR = ["dv2", "hpi_vote", "qpi", "hpi_ibs_rsi"]
ETF_MR = ["etf_ind", "sector_vox_iyr", "disp_kie_ihi_sma", "disp_kie_ihi_xlc", "disp_kie_ihi_xlc_sma"]
CANDIDATES = ["qpi", "hpi_ibs_rsi", "sector_vox_iyr", "disp_kie_ihi_sma", "disp_kie_ihi_xlc", "disp_kie_ihi_xlc_sma"]
BASE = {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.12, "hpi_vote": 0.12, "etf_ind": 0.12}


def load():
    sleeve = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END]
    bench = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END]
    paths = common.load_sleeve_path_dict()
    tx, navs = {}, {}
    for a in ["taa_btal_tqqq", "ndx_vxn"] + STOCK_MR + ETF_MR[1:]:
        tx[a] = pd.read_csv(common.SOURCE_DIR_PATH / f"{a}__transactions.csv.gz", parse_dates=["date"])
        navs[a] = paths[a]["total_value_float"].loc[:END]
    etf_nav = pd.read_csv(OUT / "wired_check" / "etf__path.csv", index_col=0, parse_dates=True)["total_value"]
    et = pd.read_csv(OUT / "wired_check" / "etf__transactions.csv", parse_dates=["bar"])
    tx["etf_ind"] = pd.DataFrame({"date": et["bar"], "asset_str": et["asset"], "signed_notional_float": et["amount"] * et["price"]})
    navs["etf_ind"] = etf_nav
    sleeve["etf_ind"] = etf_nav.pct_change().reindex(sleeve.index)
    stressed = sleeve.copy()
    for a, t in tx.items():
        drag = evaluation.extra_slippage_cost_ser(t, navs[a], 0.0005)
        live = sleeve[a].notna()
        stressed.loc[live, a] = sleeve.loc[live, a] - drag.reindex(sleeve.index).fillna(0.0)[live]
    long_df = sleeve.copy()
    long_df.loc[long_df.index < CUT, "taa_btal_tqqq"] = sleeve.loc[sleeve.index < CUT, "taa_1n_qld"]
    etf_long = pd.read_csv(OUT / "sources" / "etf_ind_adv50__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"].pct_change()
    pre = long_df.index < pd.Timestamp("2012-01-03")
    long_df.loc[pre, "etf_ind"] = etf_long.reindex(long_df.index)[pre]
    return sleeve, stressed, long_df, bench, tx, navs


def stats(x: pd.Series, spx: pd.Series) -> dict:
    x = x.dropna()
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd_ser = nav / nav.cummax() - 1
    dd = dd_ser.min()
    under = dd_ser < 0
    run = under.groupby((~under).cumsum()).cumsum()
    mon = (1 + x).resample("ME").prod() - 1
    roll12 = nav.pct_change(252).dropna()
    r3 = x.rolling(756).apply(lambda v: v.mean() / v.std() * np.sqrt(252), raw=True).dropna()
    m = spx.reindex(x.index).fillna(0.0)
    half = len(x) // 2
    return {"CAGR": cagr, "vol": x.std() * np.sqrt(252), "Sharpe": x.mean() / x.std() * np.sqrt(252),
            "Sortino": x.mean() * 252 / (x[x < 0].std() * np.sqrt(252)), "maxDD": dd, "Calmar": cagr / abs(dd),
            "longest_underwater_days": int(run.max()), "Sharpe_H1": x.iloc[:half].mean() / x.iloc[:half].std() * np.sqrt(252),
            "Sharpe_H2": x.iloc[half:].mean() / x.iloc[half:].std() * np.sqrt(252), "worst_month": mon.min(),
            "pct_pos_months": (mon > 0).mean(), "worst_12m": roll12.min(), "min_rolling_3y_sharpe": r3.min() if len(r3) else np.nan,
            "beta_spx": np.cov(x, m)[0, 1] / m.var(), "corr_spx": x.corr(m)}


def book(sleeve, w, start=CUT):
    df = sleeve.loc[start:, list(w)]
    return common.book_return_ser(df.fillna(0.0), w, "annual")


def capacity(orders, weights_df, w, excess, etf_set, nasdaq_set, liq, urgent):
    bo = orders[orders.alias_str.isin(w)].copy()
    wa = weights_df.reindex(bo["date"]).to_numpy()
    ci = {a: i for i, a in enumerate(weights_df.columns)}
    bo["book_fraction_float"] = bo["fraction_float"].to_numpy() * np.array([wa[i, ci[a]] for i, a in enumerate(bo["alias_str"])])
    bo["is_urgent"] = bo["alias_str"].isin(urgent)
    daily = bo.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
    daily["is_etf"] = daily["asset_str"].isin(etf_set)
    daily["is_nasdaq"] = daily["asset_str"].isin(nasdaq_set)
    for col, fr in liq.items():
        daily[col] = [fr.at[d, t] if (t in fr.columns and d in fr.index) else np.nan for d, t in zip(daily["date"], daily["asset_str"])]
    daily = daily.dropna(subset=list(liq))
    yrs = (END - pd.Timestamp("2023-08-21")).days / 365.25
    out = {}
    for route in ("MOO", "MOC"):
        rec, fail = None, ""
        for aum in (1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8):
            cost, gates = gc.route_cost_and_gates(daily, route, aum)
            cpy = cost / aum / yrs
            ok = all(v for k, v in gates.items() if k.endswith("_ok")) and cpy <= 0.25 * excess
            if aum in (1e7, 2.5e7):
                out[f"{route}_cost_{int(aum / 1e6)}m"] = cpy
            if ok and not fail:
                rec = aum
            elif not fail:
                bad = [k.replace("_ok", "") + f" ({gates[k.replace('_ok', '_worst')]})" for k, v in gates.items() if k.endswith("_ok") and not v]
                fail = f"${aum / 1e6:g}M: " + ("; ".join(bad) if bad else f"cost {cpy:.2%}")
        out[f"{route}_recommended"], out[f"{route}_first_fail"] = rec, fail
    return out


def main():
    sleeve, stressed, long_df, bench, tx, navs = load()
    spx, tb = bench["SPXTR"], bench["TBILL"]
    # ---------------- A: the book in detail
    books_a = {"G3": {"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5},
               "G3 + DV2 18 + HPI 18": {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.18, "hpi_vote": 0.18},
               "12/12/12": BASE}
    rep = {"A": {}, "B": {}}
    rets = {}
    for name, w in books_a.items():
        x, pw = book(sleeve, w)
        xs = book(stressed, w)[0]
        xl = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(w)].fillna(0.0), w, "annual")[0]
        nl = (1 + xl).cumprod()
        s = stats(x, spx)
        s["maxDD_incl_2008_proxy"] = min(s["maxDD"], (nl / nl.cummax() - 1).min())
        ss = stats(xs, spx)
        s["stress_CAGR"], s["stress_Sharpe"], s["stress_Calmar"] = ss["CAGR"], ss["Sharpe"], ss["CAGR"] / abs(min(ss["maxDD"], s["maxDD_incl_2008_proxy"]))
        s["excess_over_tbill"] = s["CAGR"] - ((1 + tb.loc[CUT:]).prod() ** (365.25 / (END - CUT).days) - 1)
        rep["A"][name] = {"stats": s}
        rets[name] = (x, pw, w)
    yr = pd.DataFrame({n: (1 + r[0]).resample("YE").prod() - 1 for n, r in rets.items()})
    yr.index = yr.index.year
    yr["S&P 500 TR"] = ((1 + spx.loc[CUT:]).resample("YE").prod() - 1).values[: len(yr)]
    # contribution by pod (12/12/12): average annual return contribution = sum_t w_{i,t-1} r_{i,t}
    x, pw, w = rets["12/12/12"]
    contrib = (pw * sleeve.loc[CUT:, list(w)].fillna(0.0).reindex(pw.index)).groupby(pw.index.year).sum()
    rep["A"]["contribution_by_year"] = contrib.to_dict()
    rep["A"]["contribution_avg_per_year"] = contrib.mean().to_dict()
    crises = {"2015-08": ("2015-08-10", "2015-08-31"), "2018-Q4": ("2018-10-01", "2018-12-24"), "2020 crash": ("2020-02-19", "2020-03-23"),
              "2020 rebound": ("2020-03-24", "2020-06-08"), "2022 bear": ("2022-01-03", "2022-10-12"), "2025 April": ("2025-02-19", "2025-04-08")}
    rep["A"]["crises"] = {n: {k: float((1 + r[0].loc[a:b]).prod() - 1) for k, (a, b) in crises.items()} for n, r in rets.items()}
    rep["A"]["crises"]["S&P 500 TR"] = {k: float((1 + spx.loc[a:b]).prod() - 1) for k, (a, b) in crises.items()}
    rep["A"]["yearly"] = yr.to_dict()
    rep["A"]["sleeve_corr"] = sleeve.loc[CUT:, list(BASE)].corr().to_dict()
    # capacity
    frames = []
    for a, t in tx.items():
        wdf = t[(t["date"] >= pd.Timestamp("2023-08-21")) & (t["date"] <= END)]
        prior = navs[a].shift(1)
        frames.append(wdf.assign(alias_str=a, fraction_float=wdf["signed_notional_float"].abs().values / prior.reindex(wdf["date"]).values)[["date", "alias_str", "asset_str", "fraction_float"]])
    orders = pd.concat(frames, ignore_index=True)
    etf_set = set(orders.loc[orders.alias_str.isin(["taa_btal_tqqq"] + ETF_MR), "asset_str"])
    nasdaq_set = set(orders.loc[orders.alias_str == "ndx_vxn", "asset_str"])
    liq = {"adv20": {}, "adv60": {}, "sigma": {}}
    for tk in sorted(orders.asset_str.unique()):
        try:
            pr = load_price_timeseries(tk, start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            continue
        pr.index = pd.to_datetime(pr.index).normalize()
        dol = (pr["Close"] * pr["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): liquidity known before the trade
        liq["adv20"][tk] = dol.rolling(20, min_periods=10).median().shift(1)
        liq["adv60"][tk] = dol.rolling(60, min_periods=20).median().shift(1)
        liq["sigma"][tk] = pr["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liq = {k: pd.DataFrame(v) for k, v in liq.items()}
    for name, (x, pw, w) in rets.items():
        rep["A"][name]["capacity"] = capacity(orders, pw, w, rep["A"][name]["stats"]["excess_over_tbill"], etf_set, nasdaq_set, liq, STOCK_MR)
    # ---------------- B: marginal value of other MR pods
    for X in CANDIDATES:
        start = max(CUT, sleeve[X].first_valid_index() + pd.Timedelta(days=5))
        variants = {"12/12/12": BASE, f"+{X} (9/9/9/9)": {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.09, "hpi_vote": 0.09, "etf_ind": 0.09, X: 0.09}}
        if X in ("qpi", "hpi_ibs_rsi"):
            variants[f"{X} instead of HPI"] = {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.12, X: 0.12, "etf_ind": 0.12}
        if X.startswith("disp") or X == "sector_vox_iyr":
            variants[f"{X} instead of ETF DV2"] = {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.12, "hpi_vote": 0.12, X: 0.12}
        res = {}
        cap_x = (0.5 * sleeve["dv2"] + 0.5 * sleeve["hpi_vote"]).loc[start:]
        for vn, w in variants.items():
            x = book(sleeve, w, start)[0]
            xs = book(stressed, w, start)[0]
            s = stats(x, spx)
            s["stress_Sharpe"] = stats(xs, spx)["Sharpe"]
            s["P3_Sharpe"] = stats(x.loc["2021-01-01":], spx)["Sharpe"]
            s["P3_maxDD"] = stats(x.loc["2021-01-01":], spx)["maxDD"]
            res[vn] = {k: s[k] for k in ("CAGR", "Sharpe", "maxDD", "Calmar", "stress_Sharpe", "P3_Sharpe", "P3_maxDD")}
        res["window_start"] = str(start.date())
        res["corr_X_with_dv2_hpi"] = float(sleeve[X].loc[start:].corr(cap_x))
        res["corr_X_with_etf_ind"] = float(sleeve[X].loc[start:].corr(sleeve["etf_ind"].loc[start:]))
        rep["B"][X] = res
    (OUT / "phase7_capsule.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(pd.DataFrame({n: v["stats"] for n, v in rep["A"].items() if isinstance(v, dict) and "stats" in v}).round(3).to_string())
    print(pd.DataFrame({n: v["capacity"] for n, v in rep["A"].items() if isinstance(v, dict) and "capacity" in v}).to_string())
    print(yr.round(3).to_string())
    print(pd.DataFrame(rep["A"]["crises"]).round(3).to_string())
    print(contrib.round(3).to_string())
    print(pd.DataFrame(rep["A"]["sleeve_corr"]).round(2).to_string())
    for X, res in rep["B"].items():
        print("\n==", X, res["window_start"], "corr capsule", round(res["corr_X_with_dv2_hpi"], 2), "corr etf", round(res["corr_X_with_etf_ind"], 2))
        print(pd.DataFrame({k: v for k, v in res.items() if isinstance(v, dict)}).round(3).to_string())


if __name__ == "__main__":
    main()
