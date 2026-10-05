"""SPEC 8: the portfolio map, the rung products and their checks; reference books; sleeve before/after table.

Components: D* (part_d, mechanical), D' (amendment A3: the same rules without eom_flow; post-result), G* (part_g),
G' (amendment A4: the gate-passer ranked first at the -16% design point; post-result), the simple growth alternative G3 (taa3x 50 / ndx_vxn 50) and T-bills. Rungs by LONG drawdown budget: DEF-7 -7%,
DEF-10 -10%, BAL -15%, GRO -20%. For each rung and each (defensive, growth) pair, (d, g, t) on a 0.05 grid
(d + g + t = 1) maximising LONG CAGR with LONG max drawdown >= budget (ties: more d, then more t). Mixing the three
component books with an annual reset is the pod-level annual reset of the combined pods, because every component
resets on the same year-end closes.

Usage: python part_m.py   (after part_d.py and part_g.py)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

import lib
import part_d
import part_g
from lib import END, EXACT_START, LONG_START, TBILL, Book

OUT = lib.STUDY / "part_m"
RUNG_DICT = {"DEF-7": -0.07, "DEF-10": -0.10, "BAL": -0.15, "GRO": -0.20}
STEP = 0.05
GRID = [(round(d * STEP, 2), round(g * STEP, 2), round(1.0 - (d + g) * STEP, 2))
        for d in range(21) for g in range(21 - d)]
G3_BOOK = Book("G3 (TAA3x + NDX-VXN)", ("taa3x", "ndx_vxn"), "EQ", {"taa3x": 0.5, "ndx_vxn": 0.5}, "annual", "ref")
LEGACY_ALIAS = {"taa_btal_tqqq": "taa3x", "taa_btal_1n_tqqq": "taa3x_1n", "taa_btal_1n_qld": "taa2x_1n",
                "taa_btal_lin_qqq": "btal_qqq", "sector_vox_iyr": "downshock", "disp_kie_ihi_sma": "disp",
                "disp_kie_ihi_xlc": "disp_xlc", "disp_kie_ihi_xlc_sma": "disp_xlc_sma", "infl_compass": "compass"}


# ─── components ──────────────────────────────────────────────────────────────


def find_book(name: str, books: list[Book]) -> Book:
    return next(b for b in books if b.name == name)


DLABEL = {"D": "D*", "Dp": "D'"}
GLABEL = {"G": "G*", "Gp": "G'", "G3": "G3"}
PAIR_LIST = [("D", "G"), ("D", "Gp"), ("D", "G3"), ("Dp", "G"), ("Dp", "Gp"), ("Dp", "G3")]


def components() -> dict[str, Book]:
    d_sel = json.loads((lib.STUDY / "part_d" / "d_selection.json").read_text(encoding="utf-8"))
    dp_sel = json.loads((lib.STUDY / "part_d" / "d_prime_selection.json").read_text(encoding="utf-8"))
    g_sel = json.loads((lib.STUDY / "part_g" / "g_selection.json").read_text(encoding="utf-8"))
    return {"D": find_book(d_sel["d_star"], part_d.family_books()),
            "Dp": find_book(dp_sel["d_prime"], part_d.family_books()),
            "G": find_book(g_sel["g_star"], part_g.family_books()),
            "Gp": find_book(g_prime_name(), part_g.family_books()), "G3": G3_BOOK}


def g_prime_name() -> str:
    """Amendment A4: the gate-passing Part G book ranked first by the -16% design-point objective."""
    table = pd.read_csv(lib.STUDY / "part_g" / "g_books.csv", index_col=0)
    return str(table[table["gates_pass"]]["objective_design_16"].idxmax())


def product_name(rung: str, d_key: str, g_key: str) -> str:
    return f"{rung} | {DLABEL[d_key]} + {GLABEL[g_key]}"


def component_frame(frame: pd.DataFrame, comp: dict[str, Book], d_key: str, g_key: str,
                    start: pd.Timestamp) -> pd.DataFrame:
    """Daily returns of the defensive core, the growth component and T-bills on [start, END] from one frame."""
    return pd.DataFrame({"D": lib.book_returns(frame, comp[d_key], start),
                         "X": lib.book_returns(frame, comp[g_key], start),
                         TBILL: frame.loc[start:END, TBILL]})


def mix(cf: pd.DataFrame, w: tuple[float, float, float], start: pd.Timestamp, policy: str = "annual") -> pd.Series:
    d, g, t = w
    book = Book("mix", ("D", "X", TBILL), "EQ", {"D": d, "X": g, TBILL: t}, policy)
    return lib.book_returns(cf, book, start)


def pod_weights(comp: dict[str, Book], growth_key: str, w: tuple[float, float, float], d_avg: dict) -> dict[str, float]:
    """Average pod weights of a mix; the defensive core enters with its (IV-averaged) weights d_avg."""
    d, g, t = w
    out: dict[str, float] = {}
    for pod, value in d_avg.items():
        out[pod] = out.get(pod, 0.0) + d * value
    for pod, value in comp[growth_key].targets().items():
        out[pod] = out.get(pod, 0.0) + g * value
    if t > 0:
        out[TBILL] = out.get(TBILL, 0.0) + t
    return {k: v for k, v in out.items() if v > 1e-12}


def map_grid(cf_long: pd.DataFrame, index_all: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for w in GRID:
        r = mix(cf_long, w, LONG_START)
        rows.append({"d": w[0], "g": w[1], "t": w[2], "long_cagr": lib.cagr(r, lib.base_date(index_all, r)),
                     "long_maxdd": lib.maxdd(r), "long_sharpe": float(r.mean() / r.std() * np.sqrt(252))})
    return pd.DataFrame(rows)


def pick_rung(grid: pd.DataFrame, budget: float) -> tuple[float, float, float]:
    ok = grid[grid["long_maxdd"] >= budget]
    best = ok.sort_values(["long_cagr", "d", "t"], ascending=[False, False, False]).iloc[0]
    return float(best["d"]), float(best["g"]), float(best["t"])


# ─── commission on real share counts (SPEC 9, rung products only) ────────────


def commission_add_back(aliases: list[str], data: dict) -> pd.DataFrame:
    sys.path.insert(0, str(lib.REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
    import commission_fix as cf  # noqa: PLC0415

    ratio = cf.SplitRatio()
    index = data["index"]
    out = {}
    for alias in aliases:
        if alias == TBILL:
            continue
        tx = cf.normalise(data["tx"][alias])
        add = cf.add_back_ser(tx, data["nav"][alias], ratio.at(tx["asset_str"], tx["date"])).reindex(index).fillna(0.0)
        if alias in lib.PROXY_ALIAS_LIST:
            # Before 2012-10-02 the LONG frame holds the proxy run's returns, so its fills set the add-back there.
            ptx = cf.normalise(lib.read_tx(lib.PROXY / "splice_scaled", alias))
            pnav = lib.read_path(lib.PROXY / "splice_scaled", alias)["total_value_float"]
            padd = cf.add_back_ser(ptx, pnav, ratio.at(ptx["asset_str"], ptx["date"])).reindex(index).fillna(0.0)
            add = add.where(index >= EXACT_START, padd)
        out[alias] = add
    return pd.DataFrame(out, index=index)


# ─── references ──────────────────────────────────────────────────────────────


def module_alias_map() -> dict[str, str]:
    import run_sleeves  # noqa: PLC0415

    out = {}
    for alias, (import_str, _) in run_sleeves.SLEEVE_DICT.items():
        if alias in run_sleeves.SENSITIVITY_ALIAS_SET:
            continue
        out[import_str.split(":")[0]] = alias
    return out


def yaml_books() -> tuple[list[Book], list[dict]]:
    alias_by_module = module_alias_map()
    books, skipped = [], []
    for path in sorted((lib.REPO / "portfolios").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        pods = spec.get("pods", [])
        weights, missing = {}, []
        for pod in pods:
            module = pod["strategy_import_str"].split(":")[0]
            if module not in alias_by_module:
                missing.append(module)
                continue
            alias = alias_by_module[module]
            weights[alias] = weights.get(alias, 0.0) + float(pod.get("weight_float") or 0.0)
        if missing:
            skipped.append({"yaml": path.name, "not_in_inventory": missing})
            continue
        rebalance = spec.get("rebalance") or {}
        policy = "annual" if rebalance.get("frequency_str") == "annually" else "none"
        if rebalance.get("policy_str") == "equal" or not any(weights.values()):
            weights = {a: 1.0 / len(weights) for a in weights}
        total = sum(weights.values())
        weights = {a: v / total for a, v in weights.items()}
        books.append(Book(f"yaml:{path.stem}", tuple(weights), "EQ", weights, policy, "yaml"))
    return books, skipped


def previous_pick_books() -> list[Book]:
    return [Book("prev: CORE5 alone", ("core5",), "EQ", {"core5": 1.0}, "annual", "prev"),
            Book("prev: CORE5 + BTAL_QQQ 50/50", ("core5", "btal_qqq"), "EQ", {"core5": 0.5, "btal_qqq": 0.5},
                 "annual", "prev"),
            G3_BOOK]


def reference_rows(books: list[Book], data: dict) -> pd.DataFrame:
    index_all = data["index"]
    rows = []
    for book in books:
        first_long = max(data["long"][p].first_valid_index() for p in book.pods)
        start = LONG_START if first_long <= LONG_START else None
        # A book with a pod that starts after 2012-10-02 (XLC dispersion) is measured from its last pod's start.
        exact_start = max([EXACT_START] + [data["sleeve"][p].first_valid_index() for p in book.pods])
        row = {"book": book.name, "policy": book.policy, "exact_start": exact_start.date().isoformat(),
               "weights": json.dumps({k: round(v, 4) for k, v in book.targets().items()})}
        r_exact = lib.book_returns(data["sleeve"], book, exact_start)
        row.update(lib.full_metrics(r_exact, data, "exact"))
        tb = data["long"][TBILL]
        row["exact_cagr_at_10"] = lib.cagr_at_budget(r_exact, tb, index_all, -0.10)[0]
        row["exact_cagr_at_20"] = lib.cagr_at_budget(r_exact, tb, index_all, -0.20)[0]
        row["xs_RECENT"] = lib.excess_cagr(lib.window(r_exact, *lib.BLOCK_DICT["RECENT"]), tb, index_all)
        if start is not None:
            r_long = lib.book_returns(data["long"], book, start)
            row.update(lib.full_metrics(r_long, data, "long"))
            row["long_cagr_at_10"] = lib.cagr_at_budget(r_long, tb, index_all, -0.10)[0]
            row["long_cagr_at_20"] = lib.cagr_at_budget(r_long, tb, index_all, -0.20)[0]
            row["crisis_gfc"] = lib.common.window_return_float(r_long, *lib.CRISIS_DICT["gfc"])
        row.update(lib.ops_fields(book, data))
        rows.append(row)
    return pd.DataFrame(rows).set_index("book")


def benchmark_rows(data: dict) -> pd.DataFrame:
    rows = []
    for name, column in (("S&P 500 TR", "SPXTR"), ("60/40 SPY/AGG", "SIXTY_FORTY"), ("T-bills (BIL)", "BIL")):
        row = {"book": name}
        for prefix, start in (("long", LONG_START), ("exact", EXACT_START)):
            r = data["bench"][column].loc[start:END]
            row.update(lib.full_metrics(r, data, prefix))
        rows.append(row)
    return pd.DataFrame(rows).set_index("book")


# ─── sleeves before / after the fixes ────────────────────────────────────────


def sleeve_table(data: dict) -> pd.DataFrame:
    legacy = pd.read_csv(lib.LEGACY_INVENTORY / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    legacy = legacy.rename(columns=LEGACY_ALIAS)
    index_all = data["index"]
    tb = data["long"][TBILL]
    rows = []
    for alias in sorted(data["meta"]):
        meta = data["meta"][alias]
        row = {"alias": alias, "tier": meta["tier_str"], "first_invested": meta["first_invested_date_str"],
               "negative_cash_days": meta["negative_cash_day_count_int"],
               "min_cash_weight": meta["minimum_cash_nav_weight_float"],
               "mean_cash_weight": meta["mean_cash_nav_weight_float"], **lib.OPS_DICT.get(alias, {})}
        r_exact = data["sleeve"][alias].loc[EXACT_START:END].dropna()
        if len(r_exact) and r_exact.index[0] <= EXACT_START + pd.Timedelta(days=7):
            row.update(lib.full_metrics(r_exact, data, "exact"))
            row["xs_RECENT"] = lib.excess_cagr(lib.window(r_exact, *lib.BLOCK_DICT["RECENT"]), tb, index_all)
            row["trade_days_per_year"] = len(lib.trade_dates(data, alias)) / ((END - EXACT_START).days / 365.25)
            if alias in legacy.columns:
                old = legacy[alias].loc[EXACT_START:END].dropna()
                if len(old) and old.index[0] <= EXACT_START + pd.Timedelta(days=7):
                    om = lib.full_metrics(old, data, "legacy")
                    row.update({k: om[k] for k in ("legacy_cagr", "legacy_sharpe", "legacy_maxdd")})
        r_long = data["long"][alias].loc[LONG_START:END]
        if r_long.notna().all():
            row.update({k: v for k, v in lib.full_metrics(r_long, data, "long").items()
                        if k in ("long_cagr", "long_sharpe", "long_maxdd", "long_calmar", "long_crisis_corr")})
            row["long_gfc"] = lib.common.window_return_float(r_long, *lib.CRISIS_DICT["gfc"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("alias")


# ─── capacity (descriptive) ──────────────────────────────────────────────────


def capacity(products: dict[str, dict[str, float]], data: dict, excess: dict[str, float]) -> pd.DataFrame:
    sys.path.insert(0, str(lib.REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
    import shelf_books as sb  # noqa: PLC0415

    sb.ETF_POD_SET = {"core5", "btal_qqq", "tactical_fi", "trinity", "eom_flow", "downshock", "disp", "taa3x",
                      "taa3x_1n", "taa2x_1n", "compass_qqq", "etf_dv2"}
    sb.NASDAQ_SET = {"ndx_vxn", "ndx_atr", "ndx_natr20"}
    sb.URGENT_SET = {"dv2", "dv2_adv", "hpi_vote", "etf_dv2", "downshock", "disp"}
    sb.END = END
    tx = {}
    for alias in data["meta"]:
        t = data["tx"][alias]
        tx[alias] = t[["date", "asset_str", "signed_notional_float"]]
    alias_set = {a for w in products.values() for a in w if a != TBILL}
    cap_data = {"tx": tx, "nav": data["nav"], "alias_set": alias_set}
    start = sb.CAPACITY_START - pd.Timedelta(days=10)
    prior, books = {}, {}
    for name, weights in products.items():
        pods = [p for p in weights]
        frame = data["sleeve"].loc[start:END, pods]
        prior[name] = lib.common.book_return_ser(frame, weights, "annual")[1]
        books[name] = (weights, "annual")
    table = pd.DataFrame({"excess_over_tbill": excess})
    return sb.capacity_rows(books, prior, table, cap_data)


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.load_inputs()
    lib.ledger("part_m_started")
    index_all = data["index"]
    comp = components()
    d_avg_by_key = {}
    for d_key in ("D", "Dp"):
        d_log: list = []
        lib.book_returns(data["long"], comp[d_key], LONG_START, weight_log=d_log)
        d_avg_by_key[d_key] = lib.average_weights(d_log) if comp[d_key].rule == "IV" else comp[d_key].targets()

    products, product_rows, grids = {}, [], {}
    series_long, series_exact = {}, {}
    for d_key, growth_key in PAIR_LIST:
        cf_long = component_frame(data["long"], comp, d_key, growth_key, LONG_START)
        # The EXACT series re-derives IV weights from real history only; the first period falls back to equal weights
        # when a pod has no history before 2012-10-02 (SPEC 4).
        cf_exact = component_frame(data["sleeve"], comp, d_key, growth_key, EXACT_START)
        grid = map_grid(cf_long, index_all)
        grid.to_csv(OUT / f"map_grid_{d_key}_{growth_key}.csv", index=False, float_format="%.6g")
        grids[(d_key, growth_key)] = grid
        for rung, budget in RUNG_DICT.items():
            w = pick_rung(grid, budget)
            name = product_name(rung, d_key, growth_key)
            weights = pod_weights(comp, growth_key, w, d_avg_by_key[d_key])
            products[name] = weights
            r_long = mix(cf_long, w, LONG_START)
            r_exact = mix(cf_exact, w, EXACT_START)
            series_long[name], series_exact[name] = r_long, r_exact
            row = {"product": name, "rung": rung, "budget": budget, "defensive": d_key, "growth": growth_key,
                   "d": w[0], "g": w[1], "t": w[2],
                   "pod_weights": json.dumps({k: round(v, 4) for k, v in weights.items()})}
            row.update(lib.full_metrics(r_long, data, "long"))
            row.update(lib.full_metrics(r_exact, data, "exact"))
            for block, (lo, hi) in lib.BLOCK_DICT.items():
                row[f"xs_{block}"] = lib.excess_cagr(lib.window(r_long, lo, hi), data["long"][TBILL], index_all)
            for crisis, (lo, hi) in lib.CRISIS_DICT.items():
                row[f"crisis_{crisis}"] = lib.common.window_return_float(r_long, lo, hi)
            for k, (lo, hi, _) in enumerate(lib.cofall_windows(data["bench"], LONG_START, END)):
                row[f"cofall_{k}_{lo.date()}"] = lib.common.window_return_float(r_long, lo.strftime("%Y-%m-%d"),
                                                                                hi.strftime("%Y-%m-%d"))
            pods = tuple(weights)
            row.update(lib.ops_fields(Book(name, pods, "EQ", weights), data, weights))
            # Neighbours on the grid (plateau check): the best CAGR within +-0.05 of each weight.
            near = grid[(grid["d"].sub(w[0]).abs() <= 0.051) & (grid["g"].sub(w[1]).abs() <= 0.051)]
            row["neighbour_cagr_range"] = f"{near['long_cagr'].min():.4f}..{near['long_cagr'].max():.4f}"
            row["neighbour_dd_range"] = f"{near['long_maxdd'].min():.4f}..{near['long_maxdd'].max():.4f}"
            # Bootstrap: how often would the budget have been broken on resampled histories?
            idx = lib.boot_index(len(r_long))
            arr = r_long.to_numpy()
            dd = np.array([lib.maxdd(arr[i]) for i in idx])
            row["boot_share_dd_worse_than_budget"] = float(np.mean(dd < budget))
            row["boot_dd_p05"] = float(np.percentile(dd, 5))
            row["boot_dd_p50"] = float(np.percentile(dd, 50))
            product_rows.append(row)

    table = pd.DataFrame(product_rows).set_index("product")

    # Sensitivities (SPEC 9) for every product.
    swap_tfi = data["long"].copy()
    if data["tfi_frozen"] is not None:
        swap_tfi["tactical_fi"] = data["tfi_frozen"]
    component_pods = set().union(*(comp[k].pods for k in ("D", "Dp", "G", "Gp", "G3")))
    add_back = commission_add_back(sorted({p for w in products.values() for p in w} | component_pods), data)
    fixed_long = data["long"].copy()
    for alias in add_back.columns:
        live = fixed_long[alias].notna()
        fixed_long.loc[live, alias] = fixed_long.loc[live, alias] + add_back[alias][live]
    sens_frames = {"proxy_unscaled": data["long_unscaled"], "plus_5bps": data["stressed_long"],
                   "cash_realism": data["cash_long"], "tfi_frozen_mode": swap_tfi, "commission_real_shares": fixed_long,
                   "etf_dv2_idle_before_2010": data["long_etf_cash"]}
    sens_rows = []
    for label, frame in sens_frames.items():
        for d_key, growth_key in PAIR_LIST:
            cf = component_frame(frame, comp, d_key, growth_key, LONG_START)
            for rung in RUNG_DICT:
                name = product_name(rung, d_key, growth_key)
                row = table.loc[name]
                r = mix(cf, (row["d"], row["g"], row["t"]), LONG_START)
                sens_rows.append({"product": name, "sensitivity": label, "long_cagr": lib.cagr(r, lib.base_date(index_all, r)),
                                  "long_maxdd": lib.maxdd(r), "long_sharpe": float(r.mean() / r.std() * np.sqrt(252))})
    for name, weights in products.items():
        drift = lib.book_returns(data["long"], Book(name, tuple(weights), "EQ", weights, "none"), LONG_START)
        sens_rows.append({"product": name, "sensitivity": "drift_no_reset",
                          "long_cagr": lib.cagr(drift, lib.base_date(index_all, drift)), "long_maxdd": lib.maxdd(drift),
                          "long_sharpe": float(drift.mean() / drift.std() * np.sqrt(252))})
    sens = pd.DataFrame(sens_rows)

    excess = {name: float(table.at[name, "exact_cagr"] - table.at[name, "exact_tbill_cagr"]) for name in products}
    cap = capacity(products, data, excess)

    refs, skipped = yaml_books()
    ref_table = reference_rows(previous_pick_books() + refs, data)
    bench_table = benchmark_rows(data)
    sleeves = sleeve_table(data)
    corr_exact = data["sleeve"].loc[EXACT_START:END, sorted(data["meta"]) + [TBILL]].corr()
    corr_long = data["long"].loc[LONG_START:END].dropna(axis=1).corr()

    table.to_csv(OUT / "products.csv", float_format="%.6g")
    sens.to_csv(OUT / "products_sensitivity.csv", index=False, float_format="%.6g")
    cap.to_csv(OUT / "products_capacity.csv", float_format="%.6g")
    ref_table.to_csv(OUT / "references.csv", float_format="%.6g")
    bench_table.to_csv(OUT / "benchmarks.csv", float_format="%.6g")
    sleeves.to_csv(OUT / "sleeves.csv", float_format="%.6g")
    corr_exact.to_csv(OUT / "corr_exact.csv", float_format="%.4f")
    corr_long.to_csv(OUT / "corr_long.csv", float_format="%.4f")
    pd.DataFrame(series_long).to_csv(OUT / "products_long_returns.csv.gz", float_format="%.10g", compression="gzip")
    pd.DataFrame(series_exact).to_csv(OUT / "products_exact_returns.csv.gz", float_format="%.10g", compression="gzip")
    (OUT / "references_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    (OUT / "components.json").write_text(json.dumps({
        "D": comp["D"].name, "D_avg_weights": d_avg_by_key["D"], "Dp": comp["Dp"].name,
        "Dp_avg_weights": d_avg_by_key["Dp"], "G": comp["G"].name, "G_weights": comp["G"].targets(), "Gp": comp["Gp"].name, "Gp_weights": comp["Gp"].targets(),
        "G3": G3_BOOK.targets()}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["d", "g", "t", "long_cagr", "long_maxdd", "long_sharpe", "exact_cagr", "exact_maxdd", "xs_RECENT",
            "pods", "trade_days_per_year", "boot_share_dd_worse_than_budget"]
    print(table[show].round(4).to_string())
    print(sens.pivot(index="product", columns="sensitivity", values="long_cagr").round(4).to_string())
    lib.ledger("part_m_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
