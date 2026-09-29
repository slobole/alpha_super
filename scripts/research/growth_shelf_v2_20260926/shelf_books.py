"""Parts B and C of SPEC_FROZEN.md: defensive status check and the growth shelf v2 table.

Inputs
- fund-menu inventory: every sleeve's real daily returns and fills, $1M reference, to 2026-08-19;
- industry-ETF DV2: the engine run from 2012-01-03 (DV2 deep study, wired_check) and, only before that date, the
  same rules' research run (replica equal to the engine);
- proxy_runs.py: the BTAL TAA sleeves re-run with synthetic TQQQ / BTAL (splice_scaled = main, splice_unscaled =
  sensitivity). They fill only the dates before 2012-10-02; the real sleeves are used from that date on.

Book model: pods compound independently; annual reset to target weights at the last session of each year
(common.book_return_ser). Costs as in the engine runs; +5 bps per side is a sensitivity. Capacity: last three years
of fills, growth-shelf route model with the declared fix for urgent ETF orders (worked within one day).

Usage: python shelf_books.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, REPO / "scripts" / "research" / "fund_menu_20260923", REPO / "scripts" / "research" / "growth_shelf_20260924"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common  # noqa: E402
import evaluation  # noqa: E402
import growth_capacity as gc  # noqa: E402
import growth_dossier as gd  # noqa: E402
from growth_dossier import load_price_timeseries  # noqa: E402

OUT = REPO / "results" / "research" / "portfolio" / "growth_shelf_v2_20260926"
RUN_DIR = OUT / "proxy_runs"
FIX_DIR = OUT / "commission_fix"
DV2_OUT = REPO / "results" / "research" / "dv2_deep_20260925"
END, CUT, LONG = gd.END_TS, gd.CUT_TS, gd.LONG_TS
ETF_ENGINE_START = pd.Timestamp("2012-01-03")
H1_END, H2_START = pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01")
PROXY_ALIAS_LIST = ["taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_lin_qqq"]
OLD_STAND_IN = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld", "taa_btal_lin_qqq": "taa_lin_qqq"}
WIRED_SET = {"taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_lin_qqq", "ndx_vxn", "dv2", "hpi_vote"}
DAILY_MR_SET = {"dv2", "hpi_vote", "etf_ind"}
ETF_POD_SET = {"taa_btal_tqqq", "taa_btal_1n_tqqq", "etf_ind"}
URGENT_SET = {"dv2", "hpi_vote", "etf_ind"}
NASDAQ_SET = {"ndx_vxn"}
NEEDS_DICT = {"mosaic": "wire MOSAIC (PM_READY)", "etf_ind": "wire industry-ETF DV2 (research tier)",
              "core5": "wire CORE5 (PM_READY)", "tactical_fi": "re-freeze + wire Tactical FI (PM_READY)",
              "eom_flow": "live MOC + TLT short (gap G-032)", "sector_vox_iyr": "wire sector ETF dips (PM_READY)"}
CRISIS_DICT = {"gfc": ("2008-05-19", "2009-03-09"), "q4_2018": ("2018-09-20", "2018-12-24"),
               "covid": ("2020-02-19", "2020-03-23"), "bear_2022": ("2022-01-03", "2022-10-12"),
               "tariffs_2025": ("2025-02-19", "2025-04-08")}
AUM_GRID = (5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)
CAPACITY_START = pd.Timestamp("2023-08-21")
T = 1.0 / 3.0


def scaled(core: dict, mr: dict, core_share: float) -> dict:
    out = {a: w * core_share for a, w in core.items()}
    for a, w in mr.items():
        out[a] = out.get(a, 0.0) + w
    return out


CORE_DICT = {"G3": {"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5},
             "G3-1N": {"taa_btal_1n_tqqq": 0.5, "ndx_vxn": 0.5},
             "G4": {"taa_btal_1n_tqqq": T, "ndx_vxn": T, "mosaic": T}}
MR_DICT = {"none": {}, "stock pair": {"dv2": 0.18, "hpi_vote": 0.18}, "capsule": {"dv2": 0.12, "hpi_vote": 0.12, "etf_ind": 0.12}}
LADDER_4 = {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}
LADDER_4_1N = {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_1n_tqqq": 0.34}


def growth_books() -> dict:
    books = {}
    for core_name, core in CORE_DICT.items():
        for mr_name, mr in MR_DICT.items():
            label = core_name if mr_name == "none" else f"{core_name} + {mr_name}"
            books[label] = (scaled(core, mr, 1.0 if mr_name == "none" else 0.64), "annual")
    books["ladder_4 (annual)"] = (LADDER_4, "annual")
    books["ladder_4 (drift)"] = (LADDER_4, "none")
    books["ladder_4_1n (annual)"] = (LADDER_4_1N, "annual")
    books["ladder_4_1n (drift)"] = (LADDER_4_1N, "none")
    return books


def defensive_books() -> dict:
    defensive = {"core5": 0.33, "tactical_fi": 0.17, "eom_flow": 0.17, "hpi_vote": 0.09, "sector_vox_iyr": 0.08,
                 "ndx_vxn": 0.08, "taa_btal_tqqq": 0.08}
    ex_eom = {a: w / 0.83 for a, w in defensive.items() if a != "eom_flow"}
    return {"CORE5 alone": ({"core5": 1.0}, "annual"),
            "CORE5 + BTAL_QQQ": ({"core5": 0.5, "taa_btal_lin_qqq": 0.5}, "annual"),
            "LT_DEF (A1)": ({"core5": 0.55, "tactical_fi": 0.27, "ndx_vxn": 0.06, "mosaic": 0.06, "taa_btal_tqqq": 0.06}, "annual"),
            "DEF main line": (defensive, "annual"),
            "DEF main line without EOM": (ex_eom, "annual")}


# ─── inputs ──────────────────────────────────────────────────────────────────


def run_returns(mode_str: str, alias_str: str) -> pd.Series:
    path = pd.read_csv(RUN_DIR / mode_str / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)
    invested = path["portfolio_value_float"].abs() > 1e-9
    first = invested[invested].index[0]
    nav = path["total_value_float"]
    position = nav.index.get_loc(first)
    if position == 0:
        # The engine fills at the first bar's open, so the NAV base is the starting capital before that bar.
        out = nav.pct_change()
        out.iloc[0] = nav.iloc[0] / 1_000_000.0 - 1.0
        return out
    return nav.iloc[position - 1:].pct_change().iloc[1:]


def fill_before(target: pd.Series, fill: pd.Series, cut: pd.Timestamp) -> pd.Series:
    """*** CRITICAL*** the proxy fills only the dates before the real sleeve exists; real returns stay untouched."""
    out = target.copy()
    early = out.index < cut
    out[early] = fill.reindex(out.index)[early]
    return out


def add_back(alias: str, source: str, index: pd.DatetimeIndex) -> pd.Series:
    """Commission add-back from commission_fix.py (review H1), aligned to the study calendar."""
    ser = pd.read_csv(FIX_DIR / f"{alias}__{source}.csv.gz", index_col=0, parse_dates=True)["add_back"]
    return ser.reindex(index).fillna(0.0)


def load_inputs(fixed: bool = False) -> dict:
    sleeve = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END]
    bench = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END]
    paths = common.load_sleeve_path_dict()
    etf_nav = pd.read_csv(DV2_OUT / "wired_check" / "etf__path.csv", index_col=0, parse_dates=True)["total_value"]
    etf_tx = pd.read_csv(DV2_OUT / "wired_check" / "etf__transactions.csv", parse_dates=["bar"])
    etf_research = pd.read_csv(DV2_OUT / "sources" / "etf_ind_adv50__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"].pct_change()
    etf_ser = etf_nav.pct_change().reindex(sleeve.index)
    # The research run (same rules, replica equal to the engine) fills the dates before the engine run's first
    # return, including 2012-01-03 itself, whose engine return has no prior NAV.
    pre = sleeve.index <= ETF_ENGINE_START
    etf_ser[pre] = etf_research.reindex(sleeve.index)[pre]
    sleeve["etf_ind"] = etf_ser
    alias_set = {a for w, _ in list(growth_books().values()) + list(defensive_books().values()) for a in w}
    fix_by_alias = {}
    if fixed:
        for alias in sorted(alias_set - {"etf_ind"}):
            fix_by_alias[alias] = add_back(alias, "inventory", sleeve.index)
        fix_by_alias["etf_ind"] = add_back("etf_ind", "engine", sleeve.index).where(
            sleeve.index > ETF_ENGINE_START, add_back("etf_ind", "research", sleeve.index))
        for alias, extra in fix_by_alias.items():
            live = sleeve[alias].notna()
            sleeve.loc[live, alias] = sleeve.loc[live, alias] + extra[live]
    tx = {"etf_ind": pd.DataFrame({"date": etf_tx["bar"], "asset_str": etf_tx["asset"], "signed_notional_float": etf_tx["amount"] * etf_tx["price"]})}
    nav = {"etf_ind": etf_nav.loc[:END]}
    for alias in sorted(alias_set - {"etf_ind"}):
        tx[alias] = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias}__transactions.csv.gz", parse_dates=["date"])
        nav[alias] = paths[alias]["total_value_float"].loc[:END]
    stressed = sleeve.copy()
    for alias in alias_set:
        drag = evaluation.extra_slippage_cost_ser(tx[alias], nav[alias], 0.0005)
        live = sleeve[alias].notna()
        stressed.loc[live, alias] = sleeve.loc[live, alias] - drag.reindex(sleeve.index).fillna(0.0)[live]
    long_by_proxy = {}
    for label, source in (("new", "splice_scaled"), ("new_unscaled", "splice_unscaled"), ("old", None)):
        frame = sleeve.copy()
        for alias in PROXY_ALIAS_LIST:
            fill = sleeve[OLD_STAND_IN[alias]] if source is None else run_returns(source, alias).reindex(sleeve.index)
            if fixed and source is not None:
                fill = fill + add_back(alias, source, sleeve.index)
            frame[alias] = fill_before(frame[alias], fill, CUT)
        long_by_proxy[label] = frame
    return {"sleeve": sleeve, "bench": bench, "stressed": stressed, "long": long_by_proxy, "tx": tx, "nav": nav, "alias_set": alias_set}


# ─── metrics ─────────────────────────────────────────────────────────────────


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252))


def maxdd(r: pd.Series) -> float:
    v = (1 + r).cumprod()
    return float((v / v.cummax() - 1).min())


def book_rows(books: dict, data: dict) -> tuple[pd.DataFrame, dict]:
    sleeve, bench = data["sleeve"], data["bench"]
    exact_index = sleeve.loc[CUT:END].index
    years = len(exact_index) / 252.0
    turnover = {}
    for alias in data["alias_set"]:
        t = data["tx"][alias]
        t = t[(t["date"] >= CUT) & (t["date"] <= END)]
        turnover[alias] = float(t["signed_notional_float"].abs().sum() / data["nav"][alias].loc[CUT:].mean() / years)
    rows, prior_weights = [], {}
    for name, (weights, policy) in books.items():
        cols = list(weights)
        exact, prior_w = common.book_return_ser(sleeve.loc[CUT:END, cols], weights, policy)
        prior_weights[name] = prior_w
        base = sleeve.index[sleeve.index.get_loc(exact.index[0]) - 1]
        m = common.metric_dict(exact, bench["SPXTR"], bench["TBILL"], base)
        stress = common.book_return_ser(data["stressed"].loc[CUT:END, cols], weights, policy)[0]
        sm = common.metric_dict(stress, bench["SPXTR"], bench["TBILL"], base)
        row = {"book": name, "pods": len(weights), "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"],
               "maxdd": m["max_drawdown_float"], "calmar": m["cagr_float"] / abs(m["max_drawdown_float"]),
               "worst_12m": float(((1 + exact).cumprod().pct_change(252)).min()), "worst_year": m["worst_year_float"],
               "pos_months": m["positive_month_share_float"], "beta": m["beta_spx_float"],
               "sharpe_2012_21": sharpe(exact.loc[:H1_END]), "sharpe_2022_26": sharpe(exact.loc[H2_START:]),
               "cagr_plus5": sm["cagr_float"], "sharpe_plus5": sm["sharpe_rf0_float"],
               "calmar_plus5": sm["cagr_float"] / abs(sm["max_drawdown_float"]),
               "excess_over_tbill": m["cagr_float"] - m["tbill_cagr_float"]}
        for label, frame in data["long"].items():
            long_ser = common.book_return_ser(frame.loc[LONG:END, cols], weights, policy)[0]
            suffix = "" if label == "new" else f"_{label}"
            row[f"long_maxdd{suffix}"] = maxdd(long_ser)
            row[f"gfc{suffix}"] = common.window_return_float(long_ser, *CRISIS_DICT["gfc"])
            if label == "new":
                lb = sleeve.index[sleeve.index.get_loc(long_ser.index[0]) - 1]
                lm = common.metric_dict(long_ser, bench["SPXTR"], bench["TBILL"], lb)
                row.update({"long_cagr": lm["cagr_float"], "long_sharpe": lm["sharpe_rf0_float"],
                            "long_calmar": lm["cagr_float"] / abs(lm["max_drawdown_float"])})
                row.update({k: common.window_return_float(long_ser, s, e) for k, (s, e) in CRISIS_DICT.items() if k != "gfc"})
                year_ser = (1 + long_ser).groupby(long_ser.index.year).prod() - 1
                row["year_2008"] = float(year_ser.loc[2008])
        trade_dates = set()
        for alias in weights:
            d = data["tx"][alias]["date"]
            trade_dates |= set(d[(d >= CUT) & (d <= END)])
        row.update({"trade_days_per_year": len(trade_dates) / years,
                    "turnover_x_nav": float(sum(w * turnover[a] for a, w in weights.items())),
                    "wired_share": float(sum(w for a, w in weights.items() if a in WIRED_SET)),
                    "daily_mr": any(a in DAILY_MR_SET for a in weights),
                    "needs": "; ".join(NEEDS_DICT[a] for a in weights if a in NEEDS_DICT) or "nothing"})
        row["gates_pass"] = bool(row["sharpe"] >= 1.35 and row["sharpe_2012_21"] >= 1.20 and row["sharpe_2022_26"] >= 1.20
                                 and row["long_maxdd"] >= -0.20)
        rows.append(row)
    return pd.DataFrame(rows).set_index("book"), prior_weights


# ─── capacity ────────────────────────────────────────────────────────────────


def route_cost_and_gates_v2(order_df: pd.DataFrame, route: str, aum: float) -> tuple[float, dict]:
    """growth_capacity.route_cost_and_gates with the declared fix: urgent ETF orders (industry-ETF DV2) are worked
    within one day in the worked routes instead of being left uncosted."""
    if route in ("MOO", "MOC"):
        return gc.route_cost_and_gates(order_df, route, aum)
    urgent_etf = order_df["is_etf"].to_numpy() & order_df["is_urgent"].to_numpy()
    cost, gates = gc.route_cost_and_gates(order_df[~urgent_etf], route, aum)
    if urgent_etf.any():
        sub = order_df[urgent_etf]
        dollars = sub["book_fraction_float"].to_numpy() * aum
        participation = dollars / sub["adv60"].to_numpy()
        cost += float((dollars * sub["sigma"].to_numpy() * np.sqrt(participation)).sum())
        gates["urgent_etf_one_day_p95"] = float(np.percentile(participation, 95))
        gates["urgent_etf_one_day_max"] = float(participation.max())
        gates["urgent_etf_one_day_ok"] = gates["urgent_etf_one_day_p95"] <= 0.05 and gates["urgent_etf_one_day_max"] <= 0.20
        gates["urgent_etf_one_day_worst"] = sub["asset_str"].to_numpy()[int(np.argmax(participation))]
    return cost, gates


def capacity_rows(books: dict, prior_weights: dict, table: pd.DataFrame, data: dict) -> pd.DataFrame:
    frames = []
    for alias in data["alias_set"]:
        if alias not in {a for w, _ in books.values() for a in w}:
            continue
        t = data["tx"][alias]
        t = t[(t["date"] >= CAPACITY_START) & (t["date"] <= END)]
        prior_nav = data["nav"][alias].shift(1)
        frames.append(t.assign(alias_str=alias, fraction_float=t["signed_notional_float"].abs().to_numpy()
                               / prior_nav.reindex(t["date"]).to_numpy())[["date", "alias_str", "asset_str", "fraction_float"]])
    orders = pd.concat(frames, ignore_index=True)
    etf_tickers = set(orders.loc[orders["alias_str"].isin(ETF_POD_SET), "asset_str"])
    nasdaq_tickers = set(orders.loc[orders["alias_str"].isin(NASDAQ_SET), "asset_str"])
    liquidity = {"adv20": {}, "adv60": {}, "sigma": {}}
    for ticker in sorted(orders["asset_str"].unique()):
        try:
            px = load_price_timeseries(ticker, start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001 - uncovered tickers drop out of the gates and are counted below
            continue
        px.index = pd.to_datetime(px.index).normalize()
        dollar = (px["Close"] * px["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before the trade.
        liquidity["adv20"][ticker] = dollar.rolling(20, min_periods=10).median().shift(1)
        liquidity["adv60"][ticker] = dollar.rolling(60, min_periods=20).median().shift(1)
        liquidity["sigma"][ticker] = px["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liquidity = {k: pd.DataFrame(v) for k, v in liquidity.items()}
    years = (END - CAPACITY_START).days / 365.25
    rows = []
    for name, (weights, _) in books.items():
        pw = prior_weights[name]
        book_orders = orders[orders["alias_str"].isin(weights)].copy()
        warr = pw.reindex(book_orders["date"]).to_numpy()
        col = {a: i for i, a in enumerate(pw.columns)}
        book_orders["book_fraction_float"] = book_orders["fraction_float"].to_numpy() * np.array(
            [warr[i, col[a]] for i, a in enumerate(book_orders["alias_str"])])
        book_orders["is_urgent"] = book_orders["alias_str"].isin(URGENT_SET)
        attribution = {}
        for pod in sorted(set(weights) & (URGENT_SET - ETF_POD_SET)):
            # Review M3: which short-hold stock pod breaks the close-auction limit at $25M on its own.
            pod_orders = book_orders[book_orders["alias_str"] == pod]
            adv = np.array([liquidity["adv20"].at[d, t] if (t in liquidity["adv20"].columns and d in liquidity["adv20"].index) else np.nan
                            for d, t in zip(pod_orders["date"], pod_orders["asset_str"])])
            share = pod_orders["book_fraction_float"].to_numpy() * 2.5e7 / adv
            share = share[np.isfinite(share)]
            attribution[f"moc25_{pod}_p95"] = float(np.percentile(share, 95))
            attribution[f"moc25_{pod}_p99"] = float(np.percentile(share, 99))
            attribution[f"moc25_{pod}_orders_above_hard"] = int((share > gc.MOC_DICT["hard"]).sum())
        daily = book_orders.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily["is_etf"] = daily["asset_str"].isin(etf_tickers)
        daily["is_nasdaq"] = daily["asset_str"].isin(nasdaq_tickers)
        for column, frame in liquidity.items():
            daily[column] = [frame.at[d, t] if (t in frame.columns and d in frame.index) else np.nan for d, t in zip(daily["date"], daily["asset_str"])]
        covered = daily.dropna(subset=list(liquidity))
        row = {"book": name, "orders_3y": int(len(daily)), "orders_uncovered": int(len(daily) - len(covered)), **attribution}
        excess = float(table.at[name, "excess_over_tbill"])
        for route in ("MOO", "MOC", "worked+blocks"):
            recommended, fail = None, ""
            for aum in AUM_GRID:
                cost_dollar, gates = route_cost_and_gates_v2(covered, route, aum)
                cost = cost_dollar / aum / years
                ok = all(v for k, v in gates.items() if k.endswith("_ok")) and cost <= 0.25 * excess
                if aum == 2.5e7:
                    row[f"{route}_cost_at_25m"] = cost
                if ok and not fail:
                    recommended = aum
                elif not fail:
                    failed = [k.replace("_ok", "") + f" ({gates[k.replace('_ok', '_worst')]})" for k, v in gates.items() if k.endswith("_ok") and not v]
                    fail = f"${aum / 1e6:g}M: " + ("; ".join(failed) if failed else f"cost {cost:.2%}")
            row.update({f"{route}_recommended": recommended, f"{route}_first_fail": fail})
        rows.append(row)
    return pd.DataFrame(rows).set_index("book")


def main() -> int:
    growth, defensive = growth_books(), defensive_books()
    capacity = None
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    show = ["cagr", "sharpe", "maxdd", "calmar", "long_cagr", "long_maxdd", "long_maxdd_old", "long_calmar", "gfc", "gfc_old",
            "sharpe_2012_21", "sharpe_2022_26", "calmar_plus5", "worst_year", "trade_days_per_year", "wired_share", "gates_pass"]
    for fixed in (False, True):
        data = load_inputs(fixed)
        growth_table, prior_weights = book_rows(growth, data)
        if capacity is None:
            # Capacity depends on the orders only, which the commission fix does not change.
            capacity = capacity_rows(growth, prior_weights, growth_table, data)
        growth_table = growth_table.join(capacity)
        defensive_table, _ = book_rows(defensive, data)
        suffix = "_commission_fixed" if fixed else ""
        growth_table.to_csv(OUT / f"growth_shelf_v2{suffix}.csv", float_format="%.6g")
        defensive_table.to_csv(OUT / f"defensive_status{suffix}.csv", float_format="%.6g")
        print(f"===== {'commission-fixed' if fixed else 'engine costs'}")
        print(growth_table[show].round(3).sort_values("long_calmar", ascending=False).to_string())
        print(defensive_table[["cagr", "sharpe", "maxdd", "long_maxdd", "long_maxdd_old", "gfc", "gfc_old", "long_calmar", "worst_year"]].round(3).to_string())
    print(capacity.filter(like="moc25_").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
