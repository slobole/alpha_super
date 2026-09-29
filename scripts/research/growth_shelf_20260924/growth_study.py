"""Growth shelf: the highest-return book of at most four easy pods with a sound Sharpe (owner request, 2026-09-24).

PRE-DECLARED with the owner's answers (Sharpe 1.35+, drawdown within 20% including 2008, all return engines, up to
four pods), before any of these books was built:
  Candidates (WIRED + PM_READY return engines, CORE5 allowed as ballast), at most one per group:
    Nasdaq TAA group : taa_btal_tqqq (3x rank), taa_btal_1n_tqqq (3x equal slots), taa_btal_1n_qld (2x), taa_btal_lin_qqq (1x)
    NDX momentum group: ndx_vxn, ndx_atr
    singles          : mosaic, infl_compass, dv2, hpi_vote, sector_vox_iyr, disp_kie_ihi_sma, eom_flow, core5
    (QPI and HPI IBS-RSI are near-duplicates of HPI vote, correlation >= 0.94, and are left out.)
  Books: every combination of 1-4 pods, equal capital, annual reset (1,016 books).
  Qualifies when: Sharpe >= 1.35 on 2012-10 -> 2026-08; Sharpe >= 1.20 in each half; max drawdown within -20% on
    that window AND from March 2008 (BTAL TAA sleeves filled by the 2x no-BTAL taa_1n_qld ONLY before 2012-10-02);
    Sharpe >= 1.25 after +5 bps a side on every traded dollar.
  Pick: highest CAGR on 2012-10 -> 2026-08; ties within 0.1 point -> fewer pods -> higher Sharpe.
  Selection check: the same rules applied to the first half only pick a book; its second-half result and rank show
    how much of the pick is hindsight.
  Informational only (not used to pick): does a 5th or 6th pod added to the winner help?
Sharpe = mean / volatility of daily returns, annualised, no risk-free subtraction (house convention).
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import common  # noqa: E402
import evaluation  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "growth_shelf_20260924"
END_TS, CUT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
TAA_GROUP_LIST = ["taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_1n_qld", "taa_btal_lin_qqq"]
NDX_GROUP_LIST = ["ndx_vxn", "ndx_atr"]
SINGLE_LIST = ["mosaic", "infl_compass", "dv2", "hpi_vote", "sector_vox_iyr", "disp_kie_ihi_sma", "eom_flow", "core5"]
STAND_IN_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld", "taa_btal_1n_qld": "taa_1n_qld", "taa_btal_lin_qqq": "taa_lin_qqq"}
LABEL_DICT = {"taa_btal_tqqq": "TAA 3x", "taa_btal_1n_tqqq": "TAA 3x 1/N", "taa_btal_1n_qld": "TAA 2x", "taa_btal_lin_qqq": "TAA 1x (BTAL_QQQ)",
              "ndx_vxn": "NDX", "ndx_atr": "NDX ATR", "mosaic": "MOSAIC", "infl_compass": "Inflation Compass", "dv2": "DV2", "hpi_vote": "HPI",
              "sector_vox_iyr": "Sector downshock", "disp_kie_ihi_sma": "Dispersion", "eom_flow": "Month-end flow", "core5": "CORE5"}
WIRED_SET = {"taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_lin_qqq", "ndx_vxn", "ndx_atr", "dv2", "hpi_vote"}
MOC_SET = {"eom_flow"}
EXTRA_SLIPPAGE_PER_SIDE_FLOAT = 0.0005
RULE_DICT = {"sharpe_min": 1.35, "sharpe_half_min": 1.20, "maxdd_floor": -0.20, "sharpe_stress_min": 1.25}
MAX_PODS_INT = 4


def candidate_book_list(max_pods_int: int) -> list[tuple[str, ...]]:
    book_list = []
    for taa_obj, ndx_obj in product([None] + TAA_GROUP_LIST, [None] + NDX_GROUP_LIST):
        fixed_list = [a for a in (taa_obj, ndx_obj) if a]
        for size_int in range(0, max_pods_int - len(fixed_list) + 1):
            for single_tuple in combinations(SINGLE_LIST, size_int):
                pods_tuple = tuple(fixed_list) + single_tuple
                if pods_tuple:
                    book_list.append(pods_tuple)
    return book_list


def main() -> int:
    OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]
    floor_dict = yaml.safe_load((REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923" / "report_text_static.yaml").read_text(encoding="utf-8"))["pod_floor_usd"]

    alias_list = TAA_GROUP_LIST + NDX_GROUP_LIST + SINGLE_LIST
    path_by_alias_dict = common.load_sleeve_path_dict()
    stressed_df = sleeve_df.copy()
    trade_date_dict = {}
    for alias_str in alias_list:
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        trade_date_dict[alias_str] = set(transaction_df.loc[(transaction_df["date"] >= CUT_TS) & (transaction_df["date"] <= END_TS), "date"])
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_df, path_by_alias_dict[alias_str]["total_value_float"].loc[:END_TS], EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]
    exact_index = sleeve_df.loc[CUT_TS:].index
    half_int = len(exact_index) // 2
    h1_index, h2_index = exact_index[:half_int], exact_index[half_int:]
    exact_years_float = len(exact_index) / 252.0

    def stats(ser: pd.Series) -> tuple[float, float, float]:
        nav_ser = common.nav_from_return_ser(ser)
        years_float = (ser.index[-1] - ser.index[0]).days / 365.25
        return (float(nav_ser.iloc[-1] ** (1.0 / years_float) - 1.0), float(ser.mean() / ser.std() * np.sqrt(252.0)),
                float((nav_ser / nav_ser.cummax().clip(lower=1.0) - 1.0).min()))

    def evaluate(pods_tuple: tuple[str, ...]) -> dict:
        weight_dict = {a: 1.0 / len(pods_tuple) for a in pods_tuple}
        exact_ser = common.book_return_ser(sleeve_df.loc[CUT_TS:, list(pods_tuple)], weight_dict, "annual")[0]
        long_ser = common.book_return_ser(long_df.loc[LONG_TS:, list(pods_tuple)], weight_dict, "annual")[0]
        stress_ser = common.book_return_ser(stressed_df.loc[CUT_TS:, list(pods_tuple)], weight_dict, "annual")[0]
        cagr_float, sharpe_float, maxdd_float = stats(exact_ser)
        h1_cagr, h1_sharpe, h1_dd = stats(exact_ser.loc[h1_index])
        h2_cagr, h2_sharpe, h2_dd = stats(exact_ser.loc[h2_index])
        long_cagr, long_sharpe, long_dd = stats(long_ser)
        stress_cagr, stress_sharpe, _ = stats(stress_ser)
        # First-half stress and 2008 drawdown, for the selection check.
        h1_stress_sharpe = stats(stress_ser.loc[h1_index])[1]
        long_to_h1_dd = stats(long_ser.loc[:h1_index[-1]])[2]
        return {"book": " + ".join(LABEL_DICT[a] for a in pods_tuple), "pods": len(pods_tuple), "aliases": "|".join(pods_tuple),
                "cagr": cagr_float, "vol": float(exact_ser.std() * np.sqrt(252.0)), "sharpe": sharpe_float, "maxdd": maxdd_float,
                "sharpe_h1": h1_sharpe, "sharpe_h2": h2_sharpe, "cagr_h1": h1_cagr, "cagr_h2": h2_cagr, "maxdd_h1": h1_dd, "maxdd_h2": h2_dd,
                "cagr_long": long_cagr, "sharpe_long": long_sharpe, "maxdd_long": long_dd, "cagr_stress": stress_cagr, "sharpe_stress": stress_sharpe,
                "sharpe_stress_h1": h1_stress_sharpe, "maxdd_long_to_h1": long_to_h1_dd,
                "trade_days_per_year": len(set().union(*(trade_date_dict[a] for a in pods_tuple))) / exact_years_float,
                "min_account": max(float(floor_dict.get(a, floor_dict["default"])) / w for a, w in weight_dict.items()),
                "wired_share": sum(w for a, w in weight_dict.items() if a in WIRED_SET), "needs_moc": any(a in MOC_SET for a in pods_tuple)}

    book_list = candidate_book_list(MAX_PODS_INT)
    print(f"{len(book_list)} books")
    result_df = pd.DataFrame([evaluate(b) for b in book_list])
    result_df["qualifies"] = ((result_df["sharpe"] >= RULE_DICT["sharpe_min"]) & (result_df["sharpe_h1"] >= RULE_DICT["sharpe_half_min"])
                              & (result_df["sharpe_h2"] >= RULE_DICT["sharpe_half_min"]) & (result_df["maxdd"] >= RULE_DICT["maxdd_floor"])
                              & (result_df["maxdd_long"] >= RULE_DICT["maxdd_floor"]) & (result_df["sharpe_stress"] >= RULE_DICT["sharpe_stress_min"]))
    qualified_df = result_df[result_df["qualifies"]].copy()
    qualified_df["cagr_bucket"] = (qualified_df["cagr"] * 1000).round()  # ties within 0.1 point
    ranked_df = qualified_df.sort_values(["cagr_bucket", "pods", "sharpe"], ascending=[False, True, False])
    winner_row = ranked_df.iloc[0]
    frontier_df = qualified_df.sort_values("cagr", ascending=False).groupby("pods").head(3).sort_values(["pods", "cagr"], ascending=[True, False])

    # Selection check: the same rules on the first half only.
    h1_mask = ((result_df["sharpe_h1"] >= RULE_DICT["sharpe_min"]) & (result_df["maxdd_h1"] >= RULE_DICT["maxdd_floor"])
               & (result_df["maxdd_long_to_h1"] >= RULE_DICT["maxdd_floor"]) & (result_df["sharpe_stress_h1"] >= RULE_DICT["sharpe_stress_min"]))
    h1_pick_row = result_df[h1_mask].sort_values(["cagr_h1", "pods"], ascending=[False, True]).iloc[0]
    h2_rank_int = int((result_df["cagr_h2"] > h1_pick_row["cagr_h2"]).sum()) + 1
    h1_qualified_h2_median_float = float(result_df.loc[h1_mask, "cagr_h2"].median())

    # Informational: a 5th / 6th pod on top of the winner.
    winner_tuple = tuple(winner_row["aliases"].split("|"))
    used_group_set = {g for g, members in (("taa", TAA_GROUP_LIST), ("ndx", NDX_GROUP_LIST)) if any(a in members for a in winner_tuple)}
    addable_list = [a for a in SINGLE_LIST if a not in winner_tuple]
    if "taa" not in used_group_set:
        addable_list += TAA_GROUP_LIST
    if "ndx" not in used_group_set:
        addable_list += NDX_GROUP_LIST
    superset_row_list = []
    for extra_int in (1, 2):
        for extra_tuple in combinations(addable_list, extra_int):
            if sum(a in TAA_GROUP_LIST for a in extra_tuple) > 1 or sum(a in NDX_GROUP_LIST for a in extra_tuple) > 1:
                continue
            superset_row_list.append(evaluate(winner_tuple + extra_tuple))
    superset_df = pd.DataFrame(superset_row_list)

    reference_dict = {"TAA 3x alone": {"taa_btal_tqqq": 1.0}, "TAA 3x 1/N alone": {"taa_btal_1n_tqqq": 1.0},
                      "ladder_4 (as defined, drift)": {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}}
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    for product_str in ("AGG", "GRO", "LT_GRO"):
        group_df = weight_df[weight_df["product_id_str"] == product_str]
        reference_dict[f"menu {product_str}"] = dict(zip(group_df["alias_str"], group_df["weight_float"].astype(float)))
    reference_row_list = []
    for name_str, weight_dict in reference_dict.items():
        policy_str = "none" if name_str.startswith("ladder") else "annual"
        exact_ser = common.book_return_ser(sleeve_df.loc[CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        long_ser = common.book_return_ser(long_df.loc[LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        stress_ser = common.book_return_ser(stressed_df.loc[CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        c, s, d = stats(exact_ser)
        reference_row_list.append({"book": name_str, "pods": len(weight_dict), "cagr": c, "sharpe": s, "maxdd": d,
                                   "sharpe_h1": stats(exact_ser.loc[h1_index])[1], "sharpe_h2": stats(exact_ser.loc[h2_index])[1],
                                   "maxdd_long": stats(long_ser)[2], "sharpe_stress": stats(stress_ser)[1]})
    reference_df = pd.DataFrame(reference_row_list)

    result_df.to_csv(OUT_DIR_PATH / "books.csv", index=False, float_format="%.6g")
    frontier_df.to_csv(OUT_DIR_PATH / "frontier_top3_by_pods.csv", index=False, float_format="%.6g")
    superset_df.to_csv(OUT_DIR_PATH / "winner_plus_pods.csv", index=False, float_format="%.6g")
    reference_df.to_csv(OUT_DIR_PATH / "references.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show_list = ["book", "pods", "cagr", "vol", "sharpe", "sharpe_h1", "sharpe_h2", "maxdd", "maxdd_long", "sharpe_stress", "cagr_long",
                 "trade_days_per_year", "min_account", "wired_share", "needs_moc"]
    print(f"qualifying books: {len(qualified_df)} of {len(result_df)}")
    print("\nWINNER:")
    print(winner_row[show_list].to_string())
    print("\nTop 3 qualifiers by pod count:")
    print(frontier_df[show_list].round(3).to_string(index=False))
    print("\nReferences:")
    print(reference_df.round(3).to_string(index=False))
    print(f"\nSelection check: first-half pick = {h1_pick_row['book']} (H1 CAGR {h1_pick_row['cagr_h1']:.3f}); "
          f"second half CAGR {h1_pick_row['cagr_h2']:.3f}, Sharpe {h1_pick_row['sharpe_h2']:.2f}, rank {h2_rank_int} of {len(result_df)}; "
          f"median H2 CAGR of first-half qualifiers {h1_qualified_h2_median_float:.3f}; full-window winner H2 CAGR {winner_row['cagr_h2']:.3f}")
    print("\nWinner plus a 5th / 6th pod (best 5 by CAGR among those passing the rules):")
    superset_df["qualifies"] = ((superset_df["sharpe"] >= RULE_DICT["sharpe_min"]) & (superset_df["sharpe_h1"] >= RULE_DICT["sharpe_half_min"])
                                & (superset_df["sharpe_h2"] >= RULE_DICT["sharpe_half_min"]) & (superset_df["maxdd"] >= RULE_DICT["maxdd_floor"])
                                & (superset_df["maxdd_long"] >= RULE_DICT["maxdd_floor"]) & (superset_df["sharpe_stress"] >= RULE_DICT["sharpe_stress_min"]))
    print(superset_df[superset_df["qualifies"]].sort_values("cagr", ascending=False).head(5)[show_list].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
