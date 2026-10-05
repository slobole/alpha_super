"""Extension E: collect the report block (report/ext_data.json). Descriptive; nothing here selects.

Usage: python ext_report_data.py (after select_ext.py for every frame and seeds_ext.py)
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

import ga_lib as ga
from ga_lib import END, EXACT_START, LONG_START, lib
import select_ext as se

_spec = importlib.util.spec_from_file_location("ga_report_data", ga.HERE / "report_data.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)  # this study's report_data (the shelf-rebuild folder has a module of the same name)

OUT = ga.STUDY / "report"


def main() -> int:
    data = lib.load_inputs()
    data["meta"]["ndx_rm"] = {"tier_str": "shadow"}
    books = {b.name: b for b in se.family()}
    frames = se.ext_frames(data)
    main_frame, exact_frame = frames["main"][0], frames["s6_exact"][0]
    T = pd.read_csv(se.EXT / "main" / "books.csv", index_col=0)
    sel = json.loads((se.EXT / "main" / "selection.json").read_text(encoding="utf-8"))
    seeds = json.loads((se.EXT / "seeds.json").read_text(encoding="utf-8"))["books"]
    index_all = data["index"]
    rate = lib.dtb3_annual_rate(index_all)

    roles: dict[str, list[str]] = {}
    for mode, md in sel["modes"].items():
        for rung in se.RUNGS:
            for line in se.LINES:
                e = md[rung][line]
                for k in ("pick", "top", "product"):
                    if e.get(k):
                        roles.setdefault(e[k], []).append(f"{mode}|{rung}|{line}|{k}")
    for extra in (ga.G3_NAME, "TAA3x-1N + NDX-VXN | def2@36", "TAA3x-1N + NDX-VXN | def2@18",
                  "TAA3x + NDX-VXN/RM", "TAA3x + NDX-VXN @70:30", "TAA3x + NDX-VXN @60:40",
                  "TAA3x-1N + NDX-VXN @60:40 | def2@36", "TAA3x-1N + NDX-VXN @70:30 | def2@18"):
        roles.setdefault(extra, []).append("reference")

    rows, series = {}, {}
    for n in roles:
        b = books[n]
        r_long = lib.book_returns(main_frame, b, LONG_START)
        r_exact = lib.book_returns(exact_frame, b, EXACT_START)
        series[n] = r_long
        lo, hi = ga.BLOCK_DICT["RECENT"]
        s_l, s_e, s_r = (ga.window_stats(x.to_frame(), index_all) for x in (r_long, r_exact, r_long.loc[lo:hi]))
        t = T.loc[n]
        nav = (1 + r_long).cumprod()
        row = {"roles": roles[n], "weights": {k: round(v, 4) for k, v in b.targets().items()},
               "long": {k: rd.r4(v[0]) for k, v in s_l.items()}, "exact": {k: rd.r4(v[0]) for k, v in s_e.items()},
               "recent": {k: rd.r4(v[0]) for k, v in s_r.items()},
               "gross_full": {k: rd.r4(v) for k, v in lib.full_metrics(r_long, data, "long").items() if not isinstance(v, str)},
               "dd_trough": str((nav / nav.cummax() - 1).idxmin().date()),
               "p": {r: rd.r4(t[f"p_gross_{r}"]) for r in se.RUNGS}, "p_net": {r: rd.r4(t[f"p_net_{r}"]) for r in se.RUNGS},
               "beats_g3": {"gross": rd.r4(t["beats_g3_gross"]), "net": rd.r4(t["beats_g3_net"])},
               "xs": {k: rd.r4(t[k]) for k in ("gross_xs_A", "gross_xs_B", "gross_xs_C", "gross_xs_RECENT")},
               "rm_twin": rd.r4(t["rm_beats_twin_gross"]) if isinstance(t["rm_beats_twin_gross"], float) and not np.isnan(t["rm_beats_twin_gross"]) else None,
               "seeds": seeds.get(n), "years": rd.year_table(r_long, True), "crises": rd.crises(r_long, data),
               "nav_gross": rd.monthly_nav(r_long), "nav_net": rd.monthly_nav(ga.net_return_series(r_long)),
               "fees": rd.fee_income(r_long), "pods": int(t["pods"]), "shadow_share": rd.r4(t["shadow_share"]),
               "daily": any(lib.OPS_DICT.get(p, {}).get("daily", False) for p in b.pods)}
        lev = {}
        for L in (1.25,):
            rl = rd.leverage(r_long, L, rate)
            s = ga.window_stats(rl.to_frame(), index_all)
            lev[str(L)] = {k: rd.r4(v[0]) for k, v in s.items()}
        row["leverage"] = lev
        rows[n] = row
        print("done", n, flush=True)

    # Capacity: house route model for house sleeves; ndx_rm has no house fills, so books holding it are measured
    # without it and flagged (its own record: ~2% of ADV per order at $100M).
    cap_books = {}
    for n in rows:
        w = {k: v for k, v in books[n].targets().items() if k != "ndx_rm"}
        tot = sum(w.values())
        cap_books[n] = {k: v / tot for k, v in w.items()}
    excess = {n: float(rows[n]["exact"]["gross_cagr"] - lib.cagr(data["sleeve"][ga.TBILL].loc[EXACT_START:END],
                                                                   lib.base_date(index_all, data["sleeve"][ga.TBILL].loc[EXACT_START:END])))
              for n in rows}
    cap = rd.capacity(cap_books, data, excess)
    income = {}
    for n in rows:
        income[n] = {}
        for aum in rd.AUM_LEVELS:
            key = f"{aum / 1e6:g}M"
            route, cost = None, None
            for rt in ("MOO", "MOC", "worked+blocks"):
                lv = cap[n]["routes"][rt]["levels"][key]
                if lv["gates_ok"]:
                    route, cost = rt, lv["cost"]
                    break
            if route is None:
                route, cost = "worked+blocks (gates fail)", cap[n]["routes"]["worked+blocks"]["levels"][key]["cost"]
            r_cost = series[n] - cost / 252.0
            fi = rd.fee_income(r_cost)
            st = ga.window_stats(r_cost.to_frame(), index_all)
            income[n][key] = {"route": route, "cost": cost, "gross_cagr": float(st["gross_cagr"][0]),
                              "net_cagr": float(st["net_cagr"][0]), "income_usd": fi["total_pct"] * aum}

    sens = {}
    for f in se.FRAMES:
        p = se.EXT / f / "selection.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            tf = pd.read_csv(se.EXT / f / "books.csv", index_col=0)
            sens[f] = {"modes": {m: {r: {l: {k: md[r][l].get(k) for k in ("pick", "product", "product_how", "separate_product",
                                                                            "pick_obj", "beats_prev_product", "pick_beats_g3", "passers")}
                                             for l in se.LINES} | {"g3_passes": md[r]["g3_passes"]} for r in se.RUNGS}
                                 for m, md in s["modes"].items()},
                       "books": {n: {"gross": rd.r4(tf.at[n, "gross_cagr"]), "net": rd.r4(tf.at[n, "net_cagr"]),
                                     "pass": {r: bool(tf.at[n, f"pass_GROSS_{r}"]) for r in se.RUNGS}}
                                 for n in rows if n in tf.index}}
    frontier = [{"name": n, "g": rd.r4(t["gross_cagr"]), "n": rd.r4(t["net_cagr"]), "dd": rd.r4(t["gross_dd"]),
                 "pG": rd.r4(t["p_gross_GROWTH"]), "pA": rd.r4(t["p_gross_AGGRESSIVE"]), "pM": rd.r4(t["p_gross_MAX"]),
                 "ratio": float(t["ratio"]), "rm": bool(t["rm"]), "lt": bool(t["low_touch"]),
                 "passG": bool(t["pass_GROSS_GROWTH"]), "passA": bool(t["pass_GROSS_AGGRESSIVE"]), "passM": bool(t["pass_GROSS_MAX"])}
                for n, t in T.iterrows()]
    by_ratio = {str(r): {rung: int(T[(T["ratio"] == r) & T[f"pass_GROSS_{rung}"]].shape[0]) for rung in se.RUNGS}
                for r in se.RATIOS}
    rm_period = {}
    x = main_frame[["ndx_rm", "ndx_vxn"]]
    for lo, hi, lab in (("2008-03-04", "2012-10-01", "2008-2012"), ("2012-10-02", "2019-12-31", "2012-2019"),
                        ("2020-01-01", "2020-12-31", "2020"), ("2021-01-01", "2025-12-31", "2021-2025"),
                        ("2026-01-01", "2026-08-19", "2026")):
        w = x.loc[lo:hi]
        rm_period[lab] = {c: float((1 + w[c]).prod() - 1) for c in w}
    sleeves = {}
    for c in ("taa3x", "taa3x_1n", "taa2x_1n", "ndx_vxn", "ndx_atr", "ndx_natr20", "ndx_rm"):
        r = main_frame.loc[LONG_START:END, c]
        nav = (1 + r).cumprod()
        s = ga.window_stats(r.to_frame(), index_all)
        sleeves[c] = {"cagr": rd.r4(s["gross_cagr"][0]), "dd": rd.r4(s["gross_dd"][0]), "sharpe": rd.r4(s["gross_sharpe"][0]),
                      "trough": str((nav / nav.cummax() - 1).idxmin().date())}
    payload = {"selection": sel, "books": rows, "capacity": cap, "income": income, "sens": sens, "frontier": frontier,
               "by_ratio": by_ratio, "rm_period": rm_period, "sleeves": sleeves}
    (OUT / "ext_data.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    ga.ledger("ext_report_data_written")
    print("written", (OUT / "ext_data.json").stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
