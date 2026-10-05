"""Fund products, final pass: capacity by route, participation screen, ease and fee income (SPEC 5.11).

Route model = growth_aggressive_20260930/report_data.py::capacity (growth-shelf v2 route arithmetic), with the
classifications of the frozen plan: the two capped NDX pods in the Nasdaq set, the MR capsule pods in the urgent set,
BIL an ETF order in any pod. Orders 2023-08-21 -> END. All figures are pre-TCA.

Usage: PYTHONDONTWRITEBYTECODE=1 python capacity.py   (after study.py). Writes <study>/report/capacity.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

import g_lib as g
from g_lib import END, TBILL, Book, Lab, ga, lib

AUM_LEVELS = (1e7, 2.5e7, 5e7)
FINE_GRID = tuple(float(x) for x in np.round(10 ** np.arange(5.5, 9.001, 0.1), -3))
ETF_PODS = {"core5", "btal_qqq", "taa3x", "taa3x_1n"}
NASDAQ_PODS = {"ndx_vxn", "ndx_atr_cap", "ndx_natr_cap"}
URGENT_PODS = {"dv2", "hpi_vote", "dv2_g", "hpi_g"}
BTAL_NET_ASSETS = 317e6          # audit/capacity (source file of 2026-09-24, not re-verified)
MIN_POD = {"TAA": 15e3, "MOM": 100e3, "MR": 200e3, "DEF": 30e3}   # clean pod sizes (momentum decision; capsule = 2 x $100K)


def main() -> int:
    lab = Lab()
    data = lab.data
    lib.OPS_DICT.update(g.NEW_OPS)
    study = json.loads((g.OUT / "study.json").read_text(encoding="utf-8"))
    sys.path.insert(1, str(ga.MAIN_REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
    import shelf_books as sb  # noqa: PLC0415  (route arithmetic and the price loader)
    spec = importlib.util.spec_from_file_location("ga_report_data", g.GA_DIR / "report_data.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    books: dict[str, dict] = {}
    for n in g.PRODUCTS:
        books[n] = g.PRODUCTS[n]
        final = study["products"][n]["final"]
        if final != n:
            books[final] = g.with_cash(g.PRODUCTS[n], study["products"][n]["cash_added"])
    books[g.S9] = g.MONTHLY
    books["old growth plus"] = g.MONTHLY_PLUS
    books[g.S13_OLD] = g.INCUMBENT
    books[g.GR1_L] = g.SLOT_TESTS[g.GR1_L]
    for name in study["dial"]:
        books[name] = g.blend((1.0, study["books"][name]["weights"]))      # the stored weights are rounded: renormalise
    for name in g.STAND_INS:
        books[name] = g.STAND_INS[name]
    for name, w in (("leg TAA 3x", {"taa3x": 1.0}), ("leg TAA 3x 1N", {"taa3x_1n": 1.0}), ("leg MOM", g.MOM), ("leg MR", g.MR),
                    ("leg NDX-VXN", {"ndx_vxn": 1.0}), ("leg CORE5", {"core5": 1.0}), ("leg BTAL_QQQ", {"btal_qqq": 1.0})):
        books[name] = w

    # ── orders of the last three years, as fractions of each pod's prior-close NAV ──
    start = sb.CAPACITY_START
    years = (END - start).days / 365.25
    alias_all = sorted({a for w in books.values() for a in w if a not in (TBILL, g.QQQ)})
    frames = []
    for alias in alias_all:
        t = data["tx"][alias]
        t = t[(t["date"] >= start) & (t["date"] <= END)]
        prior_nav = data["nav"][alias].shift(1)
        frames.append(t.assign(alias_str=alias, fraction_float=t["signed_notional_float"].abs().to_numpy()
                               / prior_nav.reindex(t["date"]).to_numpy())[["date", "alias_str", "asset_str", "fraction_float"]])
    orders = pd.concat(frames, ignore_index=True)
    etf_tickers = set(orders.loc[orders["alias_str"].isin(ETF_PODS), "asset_str"]) | {g.PARKING_SYMBOL}
    nasdaq_tickers = set(orders.loc[orders["alias_str"].isin(NASDAQ_PODS), "asset_str"])
    liquidity = {"adv20": {}, "adv60": {}, "sigma": {}}
    for ticker in sorted(orders["asset_str"].unique()):
        try:
            px = sb.load_price_timeseries(ticker, start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001 - uncovered tickers drop out of the gates and are counted below
            continue
        px.index = pd.to_datetime(px.index).normalize()
        dollar = (px["Close"] * px["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before the trade.
        liquidity["adv20"][ticker] = dollar.rolling(20, min_periods=10).median().shift(1)
        liquidity["adv60"][ticker] = dollar.rolling(60, min_periods=20).median().shift(1)
        liquidity["sigma"][ticker] = px["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liquidity = {k: pd.DataFrame(v) for k, v in liquidity.items()}

    out: dict = {"meta": {"orders_from": str(start.date()), "orders_to": str(END.date()), "pre_tca": True,
                          "btal_net_assets": BTAL_NET_ASSETS, "cost_cap": "25% of EXACT-window excess CAGR over BIL (MAIN frame)"}, "books": {}}
    rf = lab.rf
    grid = sorted(set(sb.AUM_GRID) | set(AUM_LEVELS))
    for name, weights in books.items():
        w_pods = {a: v for a, v in weights.items() if a not in (TBILL, g.QQQ)}
        # *** CRITICAL*** prior-close pod weights of the running book: the annual reset is the first session of each
        # calendar year, so the path starts at the first session of the window's first year (2023) and carries the
        # drift into the window (review finding: restarting at target weights ten days before the window understated
        # the TAA pod's share in 2023).
        pw = lib.common.book_return_ser(data["sleeve"].loc["2023-01-01":END, list(weights)], weights, "annual")[1]
        bo = orders[orders["alias_str"].isin(w_pods)].copy()
        warr = pw.reindex(bo["date"]).to_numpy()
        col = {a: i for i, a in enumerate(pw.columns)}
        bo["book_fraction_float"] = bo["fraction_float"].to_numpy() * np.array([warr[i, col[a]] for i, a in enumerate(bo["alias_str"])])
        bo["is_urgent"] = bo["alias_str"].isin(URGENT_PODS)
        daily = bo.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily["is_etf"] = daily["asset_str"].isin(etf_tickers)
        daily["is_nasdaq"] = daily["asset_str"].isin(nasdaq_tickers)
        for column, frame in liquidity.items():
            daily[column] = [frame.at[d, t] if (t in frame.columns and d in frame.index) else np.nan for d, t in zip(daily["date"], daily["asset_str"])]
        covered = daily.dropna(subset=list(liquidity))
        # Amendment C1: the MR pods' BIL parking rows are left out of the gates and of the cost. They are an ETF order
        # in a market of about $0.8 billion a day, but about half of a capsule product's ETF order rows, so they
        # pulled the 95th percentile of the one-day ETF gate down and let the BTAL orders pass at twice the size.
        parking = (covered["asset_str"] == g.PARKING_SYMBOL) & covered["is_urgent"]
        entry_parking_rows = int(parking.sum())
        covered = covered[~parking]
        r_ex = g.book_returns(lab.frames["s6_exact"][0], weights, lab.frames["s6_exact"][1])
        rf_ex = rf.reindex(r_ex.index).to_numpy()
        excess = g.stats(r_ex, rf)["cagr"] - float(np.prod(1.0 + rf_ex) ** (252 / len(rf_ex)) - 1.0)
        stock_rows = bo[~bo["asset_str"].isin(etf_tickers)]
        entry = {"orders_3y": int(len(daily)), "orders_uncovered": int(daily[list(liquidity)].isna().any(axis=1).sum()), "orders_per_year": float(len(bo) / years),
                 "parking_rows_excluded": entry_parking_rows, "turnover_x_nav": float(bo["book_fraction_float"].sum() / years),
                 "stock_turnover_x_nav": float(stock_rows["book_fraction_float"].sum() / years), "exact_excess_cagr": excess, "routes": {}}
        for route in ("MOO", "MOC", "worked+blocks"):
            rec, fail, levels, gate_only = None, "", {}, None
            for aum in grid:
                cost_dollar, gates = sb.route_cost_and_gates_v2(covered, route, aum)
                cost = cost_dollar / aum / years
                gates_ok = all(v for k, v in gates.items() if k.endswith("_ok"))
                failed = [k.replace("_ok", "") + f" ({gates.get(k.replace('_ok', '_worst'), '')})" for k, v in gates.items() if k.endswith("_ok") and not v]
                if aum in AUM_LEVELS:
                    levels[f"{aum / 1e6:g}M"] = {"cost": cost, "gates_ok": gates_ok, "failed": failed}
                if gates_ok and not fail:
                    gate_only = aum
                if gates_ok and cost <= 0.25 * excess and not fail:
                    rec = aum
                elif not fail:
                    fail = f"${aum / 1e6:g}M: " + ("; ".join(failed) if failed else f"cost {cost:.2%} of NAV a year")
            entry["routes"][route] = {"recommended": rec, "gates_hold_to": gate_only, "first_fail": fail, "levels": levels,
                                      "at_grid_top": bool(rec is not None and rec >= max(grid))}
        # participation screen on the book's own orders: AUM at which the P90 / P99 order is 5% of a median day
        k = (covered["book_fraction_float"] / covered["adv60"]).to_numpy()
        worst = covered.iloc[int(np.argmax(k))] if len(covered) else None
        entry["participation"] = {"aum_p90_at_5pct": float(0.05 / np.percentile(k, 90)), "aum_p99_at_5pct": float(0.05 / np.percentile(k, 99)),
                                  "aum_max_at_5pct": float(0.05 / k.max()), "binding_symbol": None if worst is None else str(worst["asset_str"])}
        top = covered.assign(k=k).nlargest(max(1, len(covered) // 10), "k")["asset_str"].value_counts(normalize=True).head(4)
        entry["participation"]["top_decile_symbols"] = {s: float(v) for s, v in top.items()}
        btal_w = weights.get("taa3x", 0.0) * 0.336 + weights.get("taa3x_1n", 0.0) * 0.224 + weights.get("btal_qqq", 0.0) * 0.217
        entry["btal_wall"] = None if btal_w <= 0 else float(0.10 * BTAL_NET_ASSETS / btal_w)
        # ease
        bk = Book(name, tuple(weights), "EQ", weights)
        try:
            ops = {k_: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k_, v in lib.ops_fields(bk, data).items() if not isinstance(v, str)}
        except KeyError:
            ops = {}
        caps = {}
        for a, v in weights.items():
            if a in g.CAPSULE_OF:
                caps[g.CAPSULE_OF[a]] = caps.get(g.CAPSULE_OF[a], 0.0) + v
        entry["ease"] = {**ops, "research_pods": len(w_pods), "live_pods": len(w_pods) - (1 if {"ndx_atr_cap", "ndx_natr_cap"} <= set(w_pods) else 0),
                         "margin_accounts": sum(1 for a in w_pods if a in ("dv2_g", "hpi_g", "dv2", "hpi_vote")),
                         "not_wired": sorted(a for a in w_pods if data["meta"][a]["tier_str"] != "wired"),
                         "min_clean_size": float(max(MIN_POD[c] / v for c, v in caps.items())) if caps else None}
        entry["capacity_limited"] = bool((entry["routes"]["worked+blocks"]["recommended"] or 0.0) < 25e6)
        r_main = lab.ret(weights)
        entry["fee_income"] = rd.fee_income(r_main)
        out["books"][name] = entry
        print(name, {r: (v["recommended"], v["gates_hold_to"], v["first_fail"][:40]) for r, v in entry["routes"].items()},
              "P99@5%:", round(entry["participation"]["aum_p99_at_5pct"] / 1e6, 1), "M", flush=True)
    # SPEC 5.11 participation screen per leg: product AUM = the smallest of (leg AUM / leg weight), at P90 and P99.
    leg_of = {"taa3x": "leg TAA 3x", "taa3x_1n": "leg TAA 3x 1N", "ndx_atr_cap": "leg MOM", "ndx_natr_cap": "leg MOM", "dv2_g": "leg MR", "hpi_g": "leg MR",
              "ndx_vxn": "leg NDX-VXN", "core5": "leg CORE5", "btal_qqq": "leg BTAL_QQQ"}
    for name, weights in books.items():
        if name.startswith("leg "):
            continue
        legw: dict = {}
        for a, v in weights.items():
            if a in leg_of:
                legw[leg_of[a]] = legw.get(leg_of[a], 0.0) + v
        if not legw or any(a not in leg_of for a in weights if a not in (TBILL, g.QQQ)):
            continue
        row = {}
        for key in ("aum_p90_at_5pct", "aum_p99_at_5pct", "aum_max_at_5pct"):
            cand = {leg: out["books"][leg]["participation"][key] / w for leg, w in legw.items()}
            bind = min(cand, key=cand.get)
            row[key] = {"aum": float(cand[bind]), "binding_leg": bind, "binding_symbol": out["books"][bind]["participation"]["binding_symbol"]}
        out["books"][name]["participation_by_leg"] = row
    (g.OUT / "capacity.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("capacity_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
