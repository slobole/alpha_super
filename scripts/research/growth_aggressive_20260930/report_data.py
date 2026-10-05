"""Collect everything the Hebrew report needs into report/report_data.json (after select_books.py and pbo.py).

Descriptive parts (SPEC 9): capacity at $10M / $25M / $50M, manager fee income, margin "what if", breach frontier,
fee mapping. Nothing here selects a book.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import ga_lib as ga
from ga_lib import END, EXACT_START, LONG_START, TBILL, Book, lib

OUT = ga.STUDY / "report"
AUM_LEVELS = (1e7, 2.5e7, 5e7)
FRAMES = ["main", "s1_house_cash", "s2_proxy_unscaled", "s3_plus_5bps", "s4_etf_idle_pre2010", "s5_hpi_live_gap",
          "s6_exact", "s7_block126", "s8_block21"]
LADDER4 = Book("ladder_4_growth (yaml)", ("dv2", "hpi_vote", "ndx_vxn", "taa3x"), "EQ",
               {"dv2": 0.17, "hpi_vote": 0.19, "ndx_vxn": 0.27, "taa3x": 0.37}, "annual", "ref")


def r4(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), 5)


def book_by_name(books: list[Book]) -> dict[str, Book]:
    return {b.name: b for b in books}


def monthly_nav(r: pd.Series) -> list:
    nav = (1.0 + r).cumprod()
    m = nav.resample("ME").last()
    return [[d.strftime("%Y-%m"), round(float(v), 5)] for d, v in m.items()]


def year_table(r: pd.Series, with_net: bool) -> list[dict]:
    rows = []
    if with_net:
        fb = ga.fee_breakdown(r)
        for _, x in fb.iterrows():
            rows.append({"year": int(x["year"]), "gross": r4(x["gross"]), "net": r4(x["net"]),
                         "fee": r4((x["mgmt_fee"] + x["perf_fee"]) / x["start_nav"])})
    else:
        g = (1.0 + r).groupby(r.index.year).prod() - 1.0
        rows = [{"year": int(y), "gross": r4(v)} for y, v in g.items()]
    return rows


def crises(r: pd.Series, data: dict) -> dict:
    out = {k: r4(lib.common.window_return_float(r, lo, hi)) for k, (lo, hi) in lib.CRISIS_DICT.items()}
    spx = data["bench"]["SPXTR"].loc[LONG_START:END]
    for ep in lib.common.equity_drawdown_episode_list(spx, -0.10):
        out[f"spx10|{ep['peak_date_str']}|{ep['trough_date_str']}|{ep['spx_drawdown_float']:.4f}"] = \
            r4(lib.common.window_return_float(r, ep["peak_date_str"], ep["trough_date_str"]))
    for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, END):
        out[f"cofall|{lo.date()}|{hi.date()}|{ret:.4f}"] = r4(lib.common.window_return_float(
            r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")))
    return out


def leverage(r: pd.Series, L: float, rate: pd.Series) -> pd.Series:
    days = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1.0)
    borrow = (rate.reindex(r.index).ffill() + 0.015) * days / 360.0
    return L * r - (L - 1.0) * borrow


def drift_haircut(r: pd.Series, share: float = 0.30) -> pd.Series:
    """A2: live returns 30% below backtest as an even daily drag of 30% of the gross CAGR (vol and DD unchanged)."""
    growth = float((1.0 + r).prod())
    gross = growth ** (252.0 / len(r)) - 1.0
    return r - share * gross / 252.0


def fee_income(r: pd.Series) -> dict:
    """Management and performance fees as % of average investor NAV per year."""
    fb = ga.fee_breakdown(r)
    years = (fb["sessions"].sum()) / 252.0
    avg = float((fb["avg_nav"] * fb["sessions"]).sum() / fb["sessions"].sum())
    mg, pf = float(fb["mgmt_fee"].sum()), float(fb["perf_fee"].sum())
    return {"mgmt_pct": mg / avg / years, "perf_pct": pf / avg / years, "total_pct": (mg + pf) / avg / years,
            "years_with_perf_fee_share": float((fb["perf_fee"] > 0).mean()),
            "years_without_perf_fee": [int(y) for y in fb.loc[fb["perf_fee"] <= 0, "year"]]}


def capacity(products: dict[str, dict], data: dict, excess: dict[str, float]) -> dict:
    sys.path.insert(0, str(ga.MAIN_REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
    import shelf_books as sb  # noqa: PLC0415

    # Superset of every sleeve's route (the fund-products study added pods; earlier results used none of the additions).
    sb.ETF_POD_SET = {"core5", "btal_qqq", "taa3x", "taa3x_1n", "taa2x_1n", "taa_1n_qld", "taa_1n_sso", "compass_qqq", "compass",
                      "etf_dv2", "downshock", "disp", "eom_flow", "trinity", "tactical_fi"}
    sb.NASDAQ_SET = {"ndx_vxn", "ndx_atr", "ndx_natr20"}
    sb.URGENT_SET = {"dv2", "dv2_adv", "dv2_floor", "hpi_vote", "hpi_ibs_rsi", "etf_dv2", "downshock", "disp"}
    sb.END = END
    years = (END - sb.CAPACITY_START).days / 365.25
    out = {}
    alias_all = sorted({a for w in products.values() for a in w if a != TBILL})
    frames = []
    for alias in alias_all:
        t = data["tx"][alias]
        t = t[(t["date"] >= sb.CAPACITY_START) & (t["date"] <= END)]
        prior_nav = data["nav"][alias].shift(1)
        frames.append(t.assign(alias_str=alias, fraction_float=t["signed_notional_float"].abs().to_numpy()
                               / prior_nav.reindex(t["date"]).to_numpy())[["date", "alias_str", "asset_str", "fraction_float"]])
    orders = pd.concat(frames, ignore_index=True)
    etf_tickers = set(orders.loc[orders["alias_str"].isin(sb.ETF_POD_SET), "asset_str"])
    nasdaq_tickers = set(orders.loc[orders["alias_str"].isin(sb.NASDAQ_SET), "asset_str"])
    liquidity = {"adv20": {}, "adv60": {}, "sigma": {}}
    for ticker in sorted(orders["asset_str"].unique()):
        try:
            px = sb.load_price_timeseries(ticker, start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            continue
        px.index = pd.to_datetime(px.index).normalize()
        dollar = (px["Close"] * px["Volume"]).replace(0.0, np.nan)
        liquidity["adv20"][ticker] = dollar.rolling(20, min_periods=10).median().shift(1)
        liquidity["adv60"][ticker] = dollar.rolling(60, min_periods=20).median().shift(1)
        liquidity["sigma"][ticker] = px["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liquidity = {k: pd.DataFrame(v) for k, v in liquidity.items()}
    start = sb.CAPACITY_START - pd.Timedelta(days=10)
    grid = sorted(set(sb.AUM_GRID) | set(AUM_LEVELS))
    for name, weights in products.items():
        pw = lib.common.book_return_ser(data["sleeve"].loc[start:END, list(weights)], weights, "annual")[1]
        book_orders = orders[orders["alias_str"].isin(weights)].copy()
        warr = pw.reindex(book_orders["date"]).to_numpy()
        col = {a: i for i, a in enumerate(pw.columns)}
        book_orders["book_fraction_float"] = book_orders["fraction_float"].to_numpy() * np.array(
            [warr[i, col[a]] for i, a in enumerate(book_orders["alias_str"])])
        book_orders["is_urgent"] = book_orders["alias_str"].isin(sb.URGENT_SET)
        daily = book_orders.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily["is_etf"] = daily["asset_str"].isin(etf_tickers)
        daily["is_nasdaq"] = daily["asset_str"].isin(nasdaq_tickers)
        for column, frame in liquidity.items():
            daily[column] = [frame.at[d, t] if (t in frame.columns and d in frame.index) else np.nan
                             for d, t in zip(daily["date"], daily["asset_str"])]
        covered = daily.dropna(subset=list(liquidity))
        entry = {"orders_3y": int(len(daily)), "orders_uncovered": int(len(daily) - len(covered)), "routes": {}}
        for route in ("MOO", "MOC", "worked+blocks"):
            rec, fail, levels = None, "", {}
            for aum in grid:
                cost_dollar, gates = sb.route_cost_and_gates_v2(covered, route, aum)
                cost = cost_dollar / aum / years
                gates_ok = all(v for k, v in gates.items() if k.endswith("_ok"))
                ok = gates_ok and cost <= 0.25 * excess[name]
                if aum in AUM_LEVELS:
                    failed = [k.replace("_ok", "") + f" ({gates.get(k.replace('_ok', '_worst'), '')})"
                              for k, v in gates.items() if k.endswith("_ok") and not v]
                    levels[f"{aum / 1e6:g}M"] = {"cost": cost, "gates_ok": gates_ok, "failed": failed}
                if ok and not fail:
                    rec = aum
                elif not fail:
                    failed = [k.replace("_ok", "") + f" ({gates.get(k.replace('_ok', '_worst'), '')})"
                              for k, v in gates.items() if k.endswith("_ok") and not v]
                    fail = f"${aum / 1e6:g}M: " + ("; ".join(failed) if failed else f"cost {cost:.2%}")
            entry["routes"][route] = {"recommended": rec, "first_fail": fail, "levels": levels}
        out[name] = entry
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    books = book_by_name(ga.family_books())
    books[LADDER4.name] = LADDER4
    table = pd.read_csv(ga.STUDY / "main" / "books.csv", index_col=0)
    sel = json.loads((ga.STUDY / "main" / "selection.json").read_text(encoding="utf-8"))
    pbo = json.loads((ga.STUDY / "main" / "pbo.json").read_text(encoding="utf-8"))
    fr = ga.frames(data)
    main_frame, exact_frame = fr["main"][0], fr["s6_exact"][0]
    index_all = data["index"]
    rate = lib.dtb3_annual_rate(index_all)

    # Which books the report shows (roles are informative; a book can hold several).
    roles: dict[str, list[str]] = {}

    def add(name, role):
        roles.setdefault(name, []).append(role)

    add(ga.G3_NAME, "champion")
    for rung in ga.RUNGS:
        for line in ("MAIN", "LOW-TOUCH"):
            e = sel["rungs"][rung][line]
            for k in ("pick", "top", "product"):
                if e.get(k):
                    add(e[k], f"{rung}|{line}|{k}")
    for extra in ("TAA3x + NDX-VXN | pair_dv2@36", "TAA3x + NDX-VXN | capsule_dv2@36", "TAA3x-1N + NDX-VXN"):
        add(extra, "earlier study reference")
    add(LADDER4.name, "benchmark (yaml)")
    g3p0 = float(table.at[ga.G3_NAME, "p_gross_GROWTH"])
    for n in table.index[table["pass_GROWTH"].astype(bool) & (table["beats_g3_share"] >= ga.CHAMP_SHARE)
                         & (table["p_gross_GROWTH"] <= g3p0)]:
        add(n, "A1 champion-test qualifier (post-result)")

    rows = {}
    series_long = {}
    for name in roles:
        b = books[name]
        r_long = lib.book_returns(main_frame, b, LONG_START)
        r_exact = lib.book_returns(exact_frame, b, EXACT_START)
        series_long[name] = r_long
        st_long = ga.window_stats(r_long.to_frame(), index_all)
        st_exact = ga.window_stats(r_exact.to_frame(), index_all)
        lo, hi = ga.BLOCK_DICT["RECENT"]
        st_recent = ga.window_stats(r_long.loc[lo:hi].to_frame(), index_all)
        row = {"roles": roles[name], "weights": {k: round(v, 4) for k, v in b.targets().items()},
               "gross_long": {k: r4(v) for k, v in lib.full_metrics(r_long, data, "long").items() if not isinstance(v, str)},
               "dd_dates": {k: v for k, v in lib.full_metrics(r_long, data, "long").items() if isinstance(v, str)},
               "gross_exact": {k: r4(v) for k, v in lib.full_metrics(r_exact, data, "exact").items() if not isinstance(v, str)},
               "net_long": {k: r4(v[0]) for k, v in st_long.items()},
               "net_exact": {k: r4(v[0]) for k, v in st_exact.items()},
               "net_recent": {k: r4(v[0]) for k, v in st_recent.items()},
               "years": year_table(r_long, True), "crises": crises(r_long, data),
               "crises_net": crises(ga.net_return_series(r_long), data),
               "nav_gross": monthly_nav(r_long), "nav_net": monthly_nav(ga.net_return_series(r_long)),
               "ops": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else r4(v)) for k, v in lib.ops_fields(b, data).items()},
               "fees_long": fee_income(r_long), "fees_exact": fee_income(r_exact),
               "fees_recent": fee_income(r_long.loc[lo:hi]), "fees_haircut30": fee_income(drift_haircut(r_long)),
               "haircut30_net_cagr": r4(ga.window_stats(drift_haircut(r_long).to_frame(), index_all)["net_cagr"][0]),
               "haircut30_gross_cagr": r4(ga.window_stats(drift_haircut(r_long).to_frame(), index_all)["gross_cagr"][0])}
        if name in table.index:
            t = table.loc[name]
            row["table"] = {k: (r4(t[k]) if isinstance(t[k], (float, np.floating, int, np.integer)) and not isinstance(t[k], bool) else
                                (bool(t[k]) if isinstance(t[k], (bool, np.bool_)) else str(t[k]))) for k in table.columns
                            if not k.startswith("band_")}
        lev = {}
        for L in (1.25, 1.5):
            rl = leverage(r_long, L, rate)
            s = ga.window_stats(rl.to_frame(), index_all)
            lev[str(L)] = {k: r4(v[0]) for k, v in s.items()}
            idx = ga.boot_index(len(rl))
            bp = ga.bootstrap_paths(rl.to_numpy()[:, None], idx)
            lev[str(L)]["p_gross_GROWTH"] = r4(float((bp["gross_dd"][:, 0] < -0.20).mean()))
            lev[str(L)]["p_gross_AGGRESSIVE"] = r4(float((bp["gross_dd"][:, 0] < -0.25).mean()))
        row["leverage"] = lev
        rows[name] = row
        print("done", name, flush=True)

    bench = {}
    for label, col in (("S&P 500 TR", "SPXTR"), ("60/40", "SIXTY_FORTY"), ("T-bills (BIL)", "BIL")):
        r_long = data["bench"][col].loc[LONG_START:END]
        r_exact = data["bench"][col].loc[EXACT_START:END]
        bench[label] = {"gross_long": {k: r4(v) for k, v in lib.full_metrics(r_long, data, "long").items() if not isinstance(v, str)},
                        "gross_exact": {k: r4(v) for k, v in lib.full_metrics(r_exact, data, "exact").items() if not isinstance(v, str)},
                        "years": year_table(r_long, False), "crises": crises(r_long, data), "nav_gross": monthly_nav(r_long)}
        if col != "BIL":
            idx = ga.boot_index(len(r_long))
            bp = ga.bootstrap_paths(r_long.to_numpy()[:, None], idx)
            bench[label]["p_gross_GROWTH"] = r4(float((bp["gross_dd"][:, 0] < -0.20).mean()))
            bench[label]["p_gross_AGGRESSIVE"] = r4(float((bp["gross_dd"][:, 0] < -0.25).mean()))

    # Capacity for the shown fund books (not benchmarks).
    cap_books = {n: books[n].targets() for n in rows}
    excess = {n: float(rows[n]["gross_exact"]["exact_cagr"] - rows[n]["gross_exact"]["exact_tbill_cagr"]) for n in rows}
    cap = capacity(cap_books, data, excess)
    # Manager income after the route's trading cost at each AUM (cheapest route whose gates pass, else worked+blocks).
    income = {}
    for n in rows:
        income[n] = {}
        for aum in AUM_LEVELS:
            key = f"{aum / 1e6:g}M"
            chosen, cost = None, None
            for route in ("MOO", "MOC", "worked+blocks"):
                lv = cap[n]["routes"][route]["levels"][key]
                if lv["gates_ok"]:
                    chosen, cost = route, lv["cost"]
                    break
            if chosen is None:
                chosen, cost = "worked+blocks (gates fail)", cap[n]["routes"]["worked+blocks"]["levels"][key]["cost"]
            r_cost = series_long[n] - cost / 252.0
            fi = fee_income(r_cost)
            st = ga.window_stats(r_cost.to_frame(), index_all)
            fi_h = fee_income(drift_haircut(r_cost))
            income[n][key] = {"route": chosen, "cost": cost, "net_cagr": float(st["net_cagr"][0]),
                              "gross_cagr": float(st["gross_cagr"][0]), "fee_pct": fi["total_pct"],
                              "income_usd": fi["total_pct"] * aum, "mgmt_usd": fi["mgmt_pct"] * aum,
                              "perf_usd": fi["perf_pct"] * aum, "income_usd_haircut30": fi_h["total_pct"] * aum}
        # Year-by-year manager income at $25M (lumpiness of the performance fee), no trading cost.
        fb = ga.fee_breakdown(series_long[n])
        rows[n]["income_years_25m"] = [{"year": int(x["year"]), "mgmt": r4(x["mgmt_fee"] / x["start_nav"] * 2.5e7),
                                        "perf": r4(x["perf_fee"] / x["start_nav"] * 2.5e7)} for _, x in fb.iterrows()]

    # Breach frontier and fee mapping over the whole family.
    frontier = []
    for n, t in table.iterrows():
        frontier.append({"name": n, "net": r4(t["net_cagr"]), "gross": r4(t["gross_cagr"]), "dd": r4(t["gross_dd"]),
                         "net_dd": r4(t["net_dd"]), "pG": r4(t["p_gross_GROWTH"]), "pA": r4(t["p_gross_AGGRESSIVE"]),
                         "sat": t["satellite"], "share": r4(t["share"]), "lt": bool(t["low_touch"]),
                         "compass": bool(t["compass"]), "passG": bool(t["pass_GROWTH"]), "passA": bool(t["pass_AGGRESSIVE"]),
                         "beatsG3": r4(t["beats_g3_share"]), "pods": int(t["pods"])})
    fit = np.polyfit(table["gross_cagr"], table["net_cagr"], 1)
    mapping = {"slope": float(fit[0]), "intercept": float(fit[1]),
               "owner_points": {str(g): float(np.polyval(fit, g)) for g in (0.176, 0.20, 0.23)}}

    # Sensitivities: each frame's complete selection, plus G3 and the main products in that frame.
    sens = {}
    shown = [ga.G3_NAME] + sorted({sel["rungs"][r][l].get(k) for r in ga.RUNGS for l in ("MAIN", "LOW-TOUCH")
                                   for k in ("pick", "top", "product") if sel["rungs"][r][l].get(k)})
    for f in FRAMES:
        p = ga.STUDY / f / "selection.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        t = pd.read_csv(ga.STUDY / f / "books.csv", index_col=0)
        sens[f] = {"selection": s, "books": {n: {"net_cagr": r4(t.at[n, "net_cagr"]), "gross_cagr": r4(t.at[n, "gross_cagr"]),
                                                  "gross_dd": r4(t.at[n, "gross_dd"]), "net_dd": r4(t.at[n, "net_dd"]),
                                                  "pG": r4(t.at[n, "p_gross_GROWTH"]), "pA": r4(t.at[n, "p_gross_AGGRESSIVE"]),
                                                  "passG": bool(t.at[n, "pass_GROWTH"]), "passA": bool(t.at[n, "pass_AGGRESSIVE"]),
                                                  "beatsG3": r4(t.at[n, "beats_g3_share"])} for n in shown if n in t.index}}

    # Gate funnel per rung-line (main frame).
    funnel = {}
    for rung in ga.RUNGS:
        for line in ("MAIN", "LOW-TOUCH"):
            fam = table if line == "MAIN" else table[table["low_touch"].astype(bool)]
            m = pd.Series(True, index=fam.index)
            steps = [("family", int(len(fam)))]
            for g in (f"gate_r1_{rung}", f"gate_r2_{rung}", "gate_r3", "gate_r4", "gate_r5"):
                m &= fam[g].astype(bool)
                steps.append((g, int(m.sum())))
            funnel[f"{rung}|{line}"] = steps

    # A1 (post-result, descriptive): gate passers meeting both champion conditions; G3's margin to each gate.
    qualifiers = {}
    g3p = float(table.at[ga.G3_NAME, "p_gross_GROWTH"])
    for rung in ga.RUNGS:
        for line in ("MAIN", "LOW-TOUCH"):
            fam = table if line == "MAIN" else table[table["low_touch"].astype(bool)]
            growth_product = sel["rungs"]["GROWTH"][line]["product"]
            if rung == "GROWTH":
                q = fam[fam["pass_GROWTH"].astype(bool) & (fam["beats_g3_share"] >= ga.CHAMP_SHARE)
                        & (fam["p_gross_GROWTH"] <= g3p)]
            else:
                q = fam[fam["pass_AGGRESSIVE"].astype(bool)]
                q = q[q["beats_g3_share"] >= ga.CHAMP_SHARE] if growth_product == ga.G3_NAME else q
            q = q.sort_values("net_cagr", ascending=False)
            qualifiers[f"{rung}|{line}"] = {"count": int(len(q)), "rows": [
                {"name": n, "net": r4(x["net_cagr"]), "gross": r4(x["gross_cagr"]), "dd": r4(x["gross_dd"]),
                 "net_dd": r4(x["net_dd"]), "p": r4(x[f"p_gross_{rung}"]), "beatsG3": r4(x["beats_g3_share"]),
                 "shadow": r4(x["shadow_share"]), "pods": int(x["pods"])} for n, x in q.head(12).iterrows()]}
    g3row = table.loc[ga.G3_NAME]
    g3_margins = {"gross_dd": r4(g3row["gross_dd"]), "net_dd": r4(g3row["net_dd"]),
                  "p_gross_20": r4(g3row["p_gross_GROWTH"]), "p_net_20": r4(g3row["p_net_GROWTH"]),
                  "net_xs_B": r4(g3row["net_xs_B"]), "net_xs_C": r4(g3row["net_xs_C"]), "net_xs_RECENT": r4(g3row["net_xs_RECENT"])}

    corr = main_frame.loc[LONG_START:END, ["taa3x", "taa3x_1n", "taa2x_1n", "ndx_vxn", "ndx_atr", "ndx_natr20",
                                           "compass_qqq", "dv2", "dv2_adv", "dv2_floor", "hpi_vote", "etf_dv2",
                                           "core5", "btal_qqq"]].corr()
    sleeves = {}
    for a in corr.columns:
        s = ga.window_stats(main_frame.loc[LONG_START:END, [a]], index_all)
        sleeves[a] = {"tier": data["meta"][a]["tier_str"], "gross_cagr": r4(s["gross_cagr"][0]),
                      "gross_dd": r4(s["gross_dd"][0]), "sharpe": r4(s["gross_sharpe"][0]), **lib.OPS_DICT.get(a, {})}

    payload = {"meta": {"end": str(END.date()), "long_start": str(LONG_START.date()), "exact_start": str(EXACT_START.date()),
                        "family": int(len(table)), "low_touch_family": int(table["low_touch"].astype(bool).sum()),
                        "ledger_tail": (ga.STUDY / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]},
               "selection": sel, "pbo": pbo, "books": rows, "bench": bench, "capacity": cap, "income": income,
               "frontier": frontier, "mapping": mapping,
               "multiseed": json.loads((OUT / "multiseed.json").read_text(encoding="utf-8")),
               "ladder4_boot": json.loads((OUT / "ladder4_boot.json").read_text(encoding="utf-8")), "qualifiers": qualifiers, "g3_margins": g3_margins, "sens": sens, "funnel": funnel,
               "corr": {"names": list(corr.columns), "values": corr.round(3).values.tolist()}, "sleeves": sleeves}
    (OUT / "report_data.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    print("written", (OUT / "report_data.json").stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
