"""Phase 5b: finalists inside the growth book (same machinery and gates as dv2_liquidity_eval.py, 2026-09-25).

Books: G3 = TAA 3x 50 / NDX 50 (annual reset); G3+MR(x) = TAA 32 / NDX 32 / DV2 variant x 18 / HPI 18.
Extra: G3+MR(F1 9 + ETF-industries DV2 9) - half of the stock DV2 weight moved to the ETF DV2 pod.
Growth gates (owner): Sharpe >= 1.35 on 2012-10-02 -> 2026-08-19, both halves >= 1.20, worst DD incl. the 2008
proxy >= -20%, Sharpe >= 1.25 under +5 bps/side. Capacity: last three years of orders, MOO and MOC routes at the
house auction limits; ETF orders are worked within the day (house model).
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
import replica as rp  # noqa: E402
import batch  # noqa: E402
import phase4_universes as p4  # noqa: E402
import growth_capacity as gc  # noqa: E402
import growth_dossier as gd  # noqa: E402
import growth_study as gs  # noqa: E402
from growth_dossier import common, load_price_timeseries  # noqa: E402
sys.path.insert(0, str(REPO / "scripts/research/fund_menu_20260923"))
import evaluation  # noqa: E402

SRC = batch.OUT / "sources"
VARIANTS = ["wired", "F0_floor", "F1_floor_adv", "F2_floor_adv_s15", "F3_E_vote", "F4_w252", "etf_industries"]
MR_BASE = {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "hpi_vote": 0.18}
G3 = {"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5}
CAP_START = pd.Timestamp("2023-08-21")
AUM_GRID = (1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)


def ensure_etf_source():
    if (SRC / "etf_industries__path.csv.gz").exists():
        return
    import phase5_finalists as f5
    p = p4.etf_panel(p4.GROUPS["industries"])
    f5.write_source("etf_industries", rp.run(p, rp.Rule()))


def variant_ret(alias, index):
    path = pd.read_csv(SRC / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
    return path.pct_change(fill_method=None).reindex(index), path


def stats(ser):
    ser = ser.dropna()
    nav = (1 + ser).cumprod()
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    return float(nav.iloc[-1] ** (1 / yrs) - 1), float(ser.mean() / ser.std() * np.sqrt(252)), float((nav / nav.cummax() - 1).min())


def main():
    ensure_etf_source()
    sleeve = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    paths = common.load_sleeve_path_dict()
    tx, navs = {}, {}
    for a in list(MR_BASE):
        tx[a] = pd.read_csv(common.SOURCE_DIR_PATH / f"{a}__transactions.csv.gz", parse_dates=["date"])
        navs[a] = paths[a]["total_value_float"].loc[:gd.END_TS]
    for a in VARIANTS:
        sleeve[a], navs[a] = variant_ret(a, sleeve.index)
        tx[a] = pd.read_csv(SRC / f"{a}__transactions.csv.gz", parse_dates=["date"])
    stressed = sleeve.copy()
    for a, t in tx.items():
        drag = evaluation.extra_slippage_cost_ser(t, navs[a], gs.EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live = sleeve[a].notna()
        stressed.loc[live, a] = sleeve.loc[live, a] - drag.reindex(sleeve.index).fillna(0.0)[live]
    long_df = sleeve.copy()
    long_df.loc[long_df.index < gd.CUT_TS, "taa_btal_tqqq"] = sleeve.loc[sleeve.index < gd.CUT_TS, "taa_1n_qld"]
    exact = sleeve.loc[gd.CUT_TS:].index
    h1, h2 = exact[: len(exact) // 2], exact[len(exact) // 2:]
    books = {"G3": G3}
    for a in VARIANTS[:-1]:
        books[f"G3+MR({a})"] = {**MR_BASE, a: 0.18}
    books["G3+MR(F1 9 + ETFind 9)"] = {**MR_BASE, "F1_floor_adv": 0.09, "etf_industries": 0.09}
    books["G3+MR(F0 9 + ETFind 9)"] = {**MR_BASE, "F0_floor": 0.09, "etf_industries": 0.09}
    rows, prior_w = [], {}
    for b, w in books.items():
        ex, pw = common.book_return_ser(sleeve.loc[gd.CUT_TS:, list(w)], w, "annual")
        lg = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(w)], w, "annual")[0]
        st = common.book_return_ser(stressed.loc[gd.CUT_TS:, list(w)], w, "annual")[0]
        prior_w[b] = pw
        c, s, d = stats(ex)
        ld = stats(lg)[2]
        sc, ss, sd = stats(st)
        worst = min(d, ld)
        rows.append({"book": b, "cagr": c, "sharpe": s, "sharpe_h1": stats(ex.loc[h1])[1], "sharpe_h2": stats(ex.loc[h2])[1],
                     "maxdd": d, "maxdd_incl_2008": worst, "calmar": c / abs(worst), "cagr_stress": sc, "sharpe_stress": ss,
                     "calmar_stress": sc / abs(min(sd, ld)),
                     "growth_rules": bool(s >= 1.35 and min(stats(ex.loc[h1])[1], stats(ex.loc[h2])[1]) >= 1.20 and worst >= -0.20 and ss >= 1.25)})
    book_df = pd.DataFrame(rows).set_index("book")
    # capacity (last three years of orders)
    frames = []
    for a, t in tx.items():
        wdf = t[(t["date"] >= CAP_START) & (t["date"] <= gd.END_TS)]
        prior_nav = navs[a].shift(1)
        frames.append(wdf.assign(alias_str=a, fraction_float=wdf["signed_notional_float"].abs().values / prior_nav.reindex(wdf["date"]).values)[["date", "alias_str", "asset_str", "fraction_float"]])
    orders = pd.concat(frames, ignore_index=True)
    etf_set = set(orders.loc[orders.alias_str.isin(["taa_btal_tqqq", "etf_industries"]), "asset_str"])
    nasdaq_set = set(orders.loc[orders.alias_str == "ndx_vxn", "asset_str"])
    liq = {"adv20": {}, "adv60": {}, "sigma": {}}
    for tk in sorted(orders.asset_str.unique()):
        try:
            pr = load_price_timeseries(tk, start_date_str="2023-03-01", end_date_str=gd.END_TS.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            continue
        pr.index = pd.to_datetime(pr.index).normalize()
        dol = (pr["Close"] * pr["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): liquidity known before the trade
        liq["adv20"][tk] = dol.rolling(20, min_periods=10).median().shift(1)
        liq["adv60"][tk] = dol.rolling(60, min_periods=20).median().shift(1)
        liq["sigma"][tk] = pr["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liq = {k: pd.DataFrame(v) for k, v in liq.items()}
    yrs = (gd.END_TS - CAP_START).days / 365.25
    cap_rows = []
    urgent_aliases = set(VARIANTS) | {"hpi_vote"}
    for b, w in books.items():
        bo = orders[orders.alias_str.isin(w)].copy()
        wa = prior_w[b].reindex(bo["date"]).to_numpy()
        ci = {a: i for i, a in enumerate(prior_w[b].columns)}
        bo["book_fraction_float"] = bo["fraction_float"].to_numpy() * np.array([wa[i, ci[a]] for i, a in enumerate(bo["alias_str"])])
        bo["is_urgent"] = bo["alias_str"].isin(urgent_aliases) & ~bo["alias_str"].eq("etf_industries")
        daily = bo.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily["is_etf"] = daily["asset_str"].isin(etf_set)
        daily["is_nasdaq"] = daily["asset_str"].isin(nasdaq_set)
        for col, fr in liq.items():
            daily[col] = [fr.at[d, t] if (t in fr.columns and d in fr.index) else np.nan for d, t in zip(daily["date"], daily["asset_str"])]
        daily = daily.dropna(subset=list(liq))
        excess = book_df.at[b, "cagr"] - 0.02
        row = {"book": b}
        for route in ("MOO", "MOC"):
            rec, fail = None, ""
            for aum in AUM_GRID:
                cost, gates = gc.route_cost_and_gates(daily, route, aum)
                cpy = cost / aum / yrs
                ok = all(v for k, v in gates.items() if k.endswith("_ok")) and cpy <= 0.25 * excess
                if aum in (1e7, 2.5e7):
                    row[f"{route}_cost_{int(aum / 1e6)}m"] = cpy
                if ok and not fail:
                    rec = aum
                elif not fail:
                    bad = [k.replace("_ok", "") + f" ({gates[k.replace('_ok', '_worst')]})" for k, v in gates.items() if k.endswith("_ok") and not v]
                    fail = f"${aum / 1e6:g}M: " + ("; ".join(bad) if bad else f"cost {cpy:.2%}")
            row[f"{route}_recommended"] = rec
            row[f"{route}_first_fail"] = fail
        cap_rows.append(row)
    cap_df = pd.DataFrame(cap_rows).set_index("book")
    out = book_df.join(cap_df)
    out.to_csv(batch.OUT / "phase5_book.csv", float_format="%.5g")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 60)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
