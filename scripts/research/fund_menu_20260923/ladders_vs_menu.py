"""The fund menu against the ladder books that exist today in portfolios/ (owner question, 2026-09-24).

Descriptive comparison of fixed books - nothing is selected or tuned here.
  Menu products: weights from books/product_weights.csv, annual reset (the product policy).
  Ladders: pods and weights exactly as in portfolios/ladder_*.yaml; rebalance as the yaml says
           (null -> drift, 'annually' -> annual reset). The other policy is reported as a sensitivity.
Exact window 2012-10-02 -> 2026-08-19 (every sleeve live). Long window 2008-03-04 onward with each
BTAL-based TAA replaced by its no-BTAL sibling ONLY before 2012-10-02 (the owner's own proxy choice for
the linearity TAA, the frozen spec's choice for the 3x TAA). Stress: +5 bps per side on every traded
dollar, from each sleeve's own fills.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import evaluation  # noqa: E402
import run_sources  # noqa: E402

REPO_PATH = Path(__file__).resolve().parents[3]
END_TS, CUT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
EXTRA_SLIPPAGE_PER_SIDE_FLOAT = 0.0005
LOW_TOUCH_MAX_TRADE_DAYS_FLOAT = 36.0
CLOSING_AUCTION_ALIAS_SET = {"eom_flow"}
STAND_IN_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld", "taa_btal_1n_qld": "taa_1n_qld",
                 "taa_btal_lin_qqq": "taa_lin_qqq"}
LADDER_NAME_LIST = ["ladder_1_defensive", "ladder_2_balanced", "ladder_3_growth", "ladder_3b_growth_2x", "ladder_3c_growth_2x_btal",
                    "ladder_4_growth", "ladder_4_growth_rebalance", "ladder_4_growth_1n", "ladder_4_growth_1n_rebalance",
                    "ladder_4_growth_inflation_compass_05_rebalance", "ladder_4_growth_inflation_compass_10_rebalance"]
EPISODE_LIST = [("q4_2018", "2018-09-20", "2018-12-24"), ("covid", "2020-02-19", "2020-03-23"), ("y2022", "2022-01-03", "2022-10-12"),
                ("tariff_2025", "2025-02-19", "2025-04-08")]


def ladder_book(name_str: str) -> tuple[dict[str, float], str]:
    config_dict = yaml.safe_load((REPO_PATH / "portfolios" / f"{name_str}.yaml").read_text(encoding="utf-8"))
    weight_dict = {}
    for pod_dict in config_dict["pods"]:
        alias_str = run_sources.SLEEVE_ALIAS_BY_IMPORT_DICT[pod_dict["strategy_import_str"]]
        weight_dict[alias_str] = weight_dict.get(alias_str, 0.0) + float(pod_dict["weight_float"])
    rebalance_dict = config_dict.get("rebalance") or {}
    policy_str = "annual" if rebalance_dict.get("frequency_str") == "annually" else "none"
    return weight_dict, policy_str


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    activity_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_activity.csv", index_col="alias_str")
    metadata_dict = common.load_sleeve_metadata_dict()
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]

    book_list = []  # (name, family, weight_dict, policy)
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    for product_str, group_df in weight_df.groupby("product_id_str", sort=False):
        book_list.append((product_str, "menu", dict(zip(group_df["alias_str"], group_df["weight_float"])), "annual"))
    for ladder_str in LADDER_NAME_LIST:
        weight_dict, policy_str = ladder_book(ladder_str)
        book_list.append((ladder_str, "ladder", weight_dict, policy_str))

    used_alias_set = {a for _, _, d, _ in book_list for a in d}
    path_by_alias_dict = common.load_sleeve_path_dict()
    stressed_df = sleeve_df.copy()
    for alias_str in sorted(used_alias_set):
        path_df = path_by_alias_dict[alias_str].loc[:END_TS]
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_df, path_df["total_value_float"], EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]

    def book(frame_df: pd.DataFrame, weight_dict: dict, start_ts: pd.Timestamp, policy_str: str) -> pd.Series:
        return common.book_return_ser(frame_df.loc[start_ts:, list(weight_dict)], weight_dict, policy_str)[0]

    def metrics(ser: pd.Series) -> dict:
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        return common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)

    row_list = []
    for name_str, family_str, weight_dict, policy_str in book_list:
        other_policy_str = "none" if policy_str == "annual" else "annual"
        exact_ser = book(sleeve_df, weight_dict, CUT_TS, policy_str)
        m, ms = metrics(exact_ser), metrics(book(stressed_df, weight_dict, CUT_TS, policy_str))
        mo = metrics(book(sleeve_df, weight_dict, CUT_TS, other_policy_str))
        long_ser = book(long_df, weight_dict, LONG_TS, policy_str)
        ml = metrics(long_ser)
        nav_ser = common.nav_from_return_ser(exact_ser)
        out = {"book": name_str, "family": family_str, "policy": policy_str, "pods": len(weight_dict),
               "wired_share": sum(w for a, w in weight_dict.items() if metadata_dict[a]["tier_str"] == "wired"),
               "low_touch_share": sum(w for a, w in weight_dict.items()
                                      if activity_df.loc[a, "trade_days_per_year_float"] <= LOW_TOUCH_MAX_TRADE_DAYS_FLOAT and a not in CLOSING_AUCTION_ALIAS_SET),
               "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"], "maxdd": m["max_drawdown_float"],
               "worst12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()), "worst_year": m["worst_year_float"],
               "beta": m["beta_spx_float"], "cagr_stress": ms["cagr_float"], "sharpe_stress": ms["sharpe_rf0_float"],
               "cagr_other_policy": mo["cagr_float"], "sharpe_other_policy": mo["sharpe_rf0_float"], "maxdd_other_policy": mo["max_drawdown_float"],
               "cagr_long": ml["cagr_float"], "sharpe_long": ml["sharpe_rf0_float"], "maxdd_long": ml["max_drawdown_float"],
               "maxdd_long_trough": ml["max_dd_trough_date_str"], "maxdd_exact_trough": m["max_dd_trough_date_str"],
               "gfc": common.window_return_float(long_ser, "2008-05-19", "2009-03-09")}
        for ep_str, start_str, end_str in EPISODE_LIST:
            out[ep_str] = common.window_return_float(exact_ser, start_str, end_str)
        row_list.append(out)
    result_df = pd.DataFrame(row_list).set_index("book")
    result_df.to_csv(common.STUDY_DIR_PATH / "books" / "ladders_vs_menu.csv", float_format="%.6g")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(result_df[["policy", "pods", "wired_share", "low_touch_share", "cagr", "vol", "sharpe", "maxdd", "worst12m", "worst_year", "beta",
                     "cagr_stress", "sharpe_stress"]].round(3).to_string())
    print()
    print(result_df[["cagr_other_policy", "sharpe_other_policy", "maxdd_other_policy", "cagr_long", "sharpe_long", "maxdd_long", "gfc",
                     "q4_2018", "covid", "y2022", "tariff_2025"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
