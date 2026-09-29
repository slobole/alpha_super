"""Growth frontier: every book scaled to the SAME drawdown budget, then compared on growth (2026-09-24).

Owner question: "so how do we maximise growth?" A book's growth under a drawdown cap is its return per unit of
drawdown times the risk actually taken. The search compared books at whatever risk they happened to carry; this
scales each one (margin leverage above 1x, cash below 1x) until its worst drawdown incl. the 2008 proxy hits the
budget, so books are compared like for like.

PRE-DECLARED:
  books      growth candidates G1-G5 (growth_dossier.py), the seven menu products, ladder_4 (drift), CORE5 alone,
             CORE5 + BTAL_QQQ 50/50 (the defensive shelf pick)
  budgets    drawdown -20% (the owner's rule) and -15% (keeps a quarter of the budget for a worse-than-history crash)
  financing  above 1x: T-bill + 1.5% (IBKR retail tier) and T-bill + 0.5% (fund-size margin or futures); below 1x the
             unused share earns T-bills
  leverage   held constant daily (an approximation; live would rebalance less often); solved by bisection on the
             long window (2008-03-04 on; 2x no-BTAL TAA stand-ins before 2012-10-02 only)
  pass       Sharpe >= 1.35 on 2012-10-02 -> 2026-08-19 after financing (the owner's growth rule)
Margin leverage is NOT supported by the engine or LIVE today; this measures what it would be worth.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_dossier as gd  # noqa: E402
from growth_dossier import common  # noqa: E402

sys.path.insert(0, str(gd.REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import evaluation  # noqa: E402

STAND_IN_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld", "taa_btal_lin_qqq": "taa_lin_qqq"}
BUDGET_TUPLE = (-0.20, -0.15)
SPREAD_TUPLE = (0.015, 0.005)
CRISIS_DICT = {"gfc": ("2008-05-19", "2009-03-09"), "covid": ("2020-02-19", "2020-03-23"), "y2022": ("2022-01-03", "2022-10-12")}


def book_dict() -> dict[str, tuple[dict, str]]:
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    out_dict = {n: (w, p) for n, (w, p, _) in gd.candidate_dict().items() if n.startswith("G")}
    for product_str, group_df in weight_df.groupby("product_id_str"):
        out_dict[f"menu {product_str}"] = (dict(zip(group_df["alias_str"], group_df["weight_float"].astype(float))), "annual")
    out_dict["ladder_4 (drift)"] = gd.candidate_dict()["ladder_4 (drift)"][:2]
    out_dict["CORE5 alone"] = ({"core5": 1.0}, "annual")
    out_dict["CORE5 + BTAL_QQQ"] = ({"core5": 0.5, "taa_btal_lin_qqq": 0.5}, "annual")
    # Added after the first run, from the owner's bench screenshot (portfolios/ladder_*.yaml weights), post-hoc.
    ladder_4_1n_dict = {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_1n_tqqq": 0.34}
    out_dict["ladder_4_1n (drift)"] = (ladder_4_1n_dict, "none")
    out_dict["ladder_4_1n (annual)"] = (ladder_4_1n_dict, "annual")
    out_dict["ladder_4 (annual)"] = (gd.candidate_dict()["ladder_4 (drift)"][0], "annual")
    out_dict["ladder_3 (drift)"] = ({"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.18, "hpi_vote": 0.18}, "none")
    return out_dict


def levered_ser(base_ser: pd.Series, lever_float: float, tbill_daily_ser: pd.Series, spread_daily_ser: pd.Series) -> pd.Series:
    if lever_float >= 1.0:
        return lever_float * base_ser - (lever_float - 1.0) * (tbill_daily_ser + spread_daily_ser)
    return lever_float * base_ser + (1.0 - lever_float) * tbill_daily_ser


def max_dd_float(ser: pd.Series) -> float:
    nav_ser = (1.0 + ser).cumprod()
    return float((nav_ser / nav_ser.cummax() - 1.0).min())


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < gd.CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < gd.CUT_TS, stand_in_str]
    rate_ser = evaluation.lagged_tbill_annual_rate_ser(sleeve_df.index)
    day_ser = pd.Series(sleeve_df.index, index=sleeve_df.index).diff().dt.days.fillna(0.0)
    tbill_daily_ser = rate_ser * day_ser / 360.0

    row_list = []
    for name_str, (weight_dict, policy_str) in book_dict().items():
        exact_ser = common.book_return_ser(sleeve_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        long_ser = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(exact_ser.index[0]) - 1]
        native_metric = common.metric_dict(exact_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
        native_long_maxdd_float = max_dd_float(long_ser)
        row_list.append({"book": name_str, "budget": 0.0, "spread": 0.0, "leverage": 1.0, "cagr": native_metric["cagr_float"],
                         "vol": native_metric["volatility_float"], "sharpe": native_metric["sharpe_rf0_float"],
                         "maxdd_exact": native_metric["max_drawdown_float"], "maxdd_long": native_long_maxdd_float,
                         "calmar_exact": native_metric["cagr_float"] / abs(native_metric["max_drawdown_float"]),
                         "calmar_incl_2008": native_metric["cagr_float"] / abs(min(native_metric["max_drawdown_float"], native_long_maxdd_float)),
                         "passes_sharpe": native_metric["sharpe_rf0_float"] >= 1.35})
        for budget_float in BUDGET_TUPLE:
            for spread_float in SPREAD_TUPLE:
                spread_daily_ser = pd.Series(spread_float, index=sleeve_df.index) * day_ser / 360.0
                low_float, high_float = 0.2, 8.0
                for _ in range(60):
                    mid_float = 0.5 * (low_float + high_float)
                    if max_dd_float(levered_ser(long_ser, mid_float, tbill_daily_ser.reindex(long_ser.index), spread_daily_ser.reindex(long_ser.index))) > budget_float:
                        low_float = mid_float
                    else:
                        high_float = mid_float
                lever_float = low_float
                lev_exact_ser = levered_ser(exact_ser, lever_float, tbill_daily_ser.reindex(exact_ser.index), spread_daily_ser.reindex(exact_ser.index))
                lev_long_ser = levered_ser(long_ser, lever_float, tbill_daily_ser.reindex(long_ser.index), spread_daily_ser.reindex(long_ser.index))
                base_ts = sleeve_df.index[sleeve_df.index.get_loc(exact_ser.index[0]) - 1]
                metric = common.metric_dict(lev_exact_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
                long_base_ts = sleeve_df.index[sleeve_df.index.get_loc(long_ser.index[0]) - 1]
                long_metric = common.metric_dict(lev_long_ser, bench_df["SPXTR"], bench_df["TBILL"], long_base_ts)
                nav_ser = (1.0 + lev_exact_ser).cumprod()
                row = {"book": name_str, "budget": budget_float, "spread": spread_float, "leverage": lever_float,
                       "cagr": metric["cagr_float"], "vol": metric["volatility_float"], "sharpe": metric["sharpe_rf0_float"],
                       "sharpe_excess": metric["sharpe_excess_float"], "maxdd_exact": metric["max_drawdown_float"],
                       "cagr_long": long_metric["cagr_float"], "maxdd_long": long_metric["max_drawdown_float"],
                       "worst_12m": float((nav_ser / nav_ser.shift(252) - 1.0).dropna().min()),
                       "passes_sharpe": metric["sharpe_rf0_float"] >= 1.35}
                row.update({k: common.window_return_float(lev_long_ser, s, e) for k, (s, e) in CRISIS_DICT.items()})
                row_list.append(row)
    frontier_df = pd.DataFrame(row_list)
    frontier_df.to_csv(gd.OUT_DIR_PATH / "growth_frontier.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 260)
    native_df = frontier_df[frontier_df["budget"] == 0.0].sort_values("calmar_incl_2008", ascending=False)
    print("== native (1x), sorted by CAGR / worst drawdown incl. 2008")
    print(native_df[["book", "cagr", "vol", "sharpe", "maxdd_exact", "maxdd_long", "calmar_exact", "calmar_incl_2008"]].round(3).to_string(index=False))
    for budget_float in BUDGET_TUPLE:
        for spread_float in SPREAD_TUPLE:
            view_df = frontier_df[(frontier_df["budget"] == budget_float) & (frontier_df["spread"] == spread_float)].sort_values("cagr", ascending=False)
            print(f"== drawdown budget {budget_float:.0%}, financing T-bill + {spread_float:.1%}")
            print(view_df.drop(columns=["budget", "spread"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
