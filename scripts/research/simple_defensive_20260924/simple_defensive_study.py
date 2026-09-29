"""Simple defensive book: can 2-4 easy pods clearly beat CORE5 alone? (owner question, 2026-09-24)

PRE-DECLARED before any of these books was built (only standalone sleeve facts had been seen):
  Anchor / benchmark: CORE5 alone.
  Candidates (owner's list): BTAL_QQQ = taa_btal_lin_qqq (WIRED) or BTAL_SPY = btal_lin_spy (RESEARCH, never both),
    DOWNSHOCK = sector_vox_iyr, FI = tactical_fi, DISP = disp_kie_ihi_sma (the only dispersion sleeve with 2008 history).
  Books: CORE5 + every non-empty subset of {BTAL (QQQ or SPY), DOWNSHOCK, FI, DISP} -> 23 books.
    Primary weighting: equal capital per pod. Sensitivity (not used for selection): CORE5 50% + the rest equal.
    Annual reset, the menu's product policy.
  References: CORE5 alone, LT_DEF, DEF (menu weights), ladder_1 as defined (drift) and with annual reset.
  Windows: exact 2012-10-02 -> 2026-08-19; long 2008-03-04 onward with each BTAL sleeve replaced by its no-BTAL
    twin ONLY before 2012-10-02 (taa_lin_qqq; nobtal_lin_spy, verified builder in run_extra_sleeves.py).
  Qualifies (equal-capital books) when, against CORE5 alone:
    (1) Sharpe higher on the exact window AND in each half of it;
    (2) CAGR higher on the exact window;
    (3) long-window max drawdown no deeper than -10% (the owner's defensive band);
    (4) Sharpe after +5 bps per side on every trade still higher than CORE5's after the same stress.
  Pick: fewest pods among qualifiers; ties -> higher long-window Sharpe, then exact-window Sharpe.
Sharpe here = mean / volatility of daily returns, annualised (no risk-free subtraction), as in the menu study.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import common  # noqa: E402
import evaluation  # noqa: E402

END_TS, CUT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "simple_defensive_20260924"
EXTRA_SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
LABEL_BY_ALIAS_DICT = {"core5": "CORE5", "taa_btal_lin_qqq": "BTAL_QQQ", "btal_lin_spy": "BTAL_SPY", "sector_vox_iyr": "DOWNSHOCK",
                       "tactical_fi": "FI", "disp_kie_ihi_sma": "DISP"}
TIER_BY_ALIAS_DICT = {"core5": "pm-ready", "taa_btal_lin_qqq": "wired", "btal_lin_spy": "research", "sector_vox_iyr": "pm-ready",
                      "tactical_fi": "pm-ready", "disp_kie_ihi_sma": "pm-ready"}
STAND_IN_DICT = {"taa_btal_lin_qqq": "taa_lin_qqq", "btal_lin_spy": "nobtal_lin_spy"}
EXTRA_SLIPPAGE_PER_SIDE_FLOAT = 0.0005
MAX_DD_BAND_FLOAT = -0.10
EPISODE_LIST = [("gfc", "2008-05-19", "2009-03-09"), ("aug_2011", "2011-04-29", "2011-10-03"), ("covid", "2020-02-19", "2020-03-23"),
                ("y2022", "2022-01-03", "2022-10-12"), ("tariff_2025", "2025-02-19", "2025-04-08")]


def extra_return_ser(alias_str: str, index: pd.DatetimeIndex) -> pd.Series:
    path_df = pd.read_csv(EXTRA_SOURCE_DIR_PATH / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)
    nav_ser = path_df["total_value_float"].astype(float)
    invested_ser = path_df["portfolio_value_float"].abs() > 1e-9
    first_position_int = nav_ser.index.get_loc(invested_ser[invested_ser].index[0])
    # One cash day kept in front of the first invested day as the NAV base, as common.sleeve_nav_df does.
    return nav_ser.iloc[max(first_position_int - 1, 0):].pct_change(fill_method=None).reindex(index)


def transaction_df_for(alias_str: str) -> pd.DataFrame:
    source_path = EXTRA_SOURCE_DIR_PATH if (EXTRA_SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz").exists() else common.SOURCE_DIR_PATH
    return pd.read_csv(source_path / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])


def nav_for(alias_str: str) -> pd.Series:
    source_path = EXTRA_SOURCE_DIR_PATH if (EXTRA_SOURCE_DIR_PATH / f"{alias_str}__path.csv.gz").exists() else common.SOURCE_DIR_PATH
    return pd.read_csv(source_path / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]


def book_list_build() -> list[tuple[str, dict, str, str]]:
    """(name, weights, policy, kind) for every pre-declared book."""
    book_list = [("CORE5 alone", {"core5": 1.0}, "annual", "benchmark")]
    other_list = ["sector_vox_iyr", "tactical_fi", "disp_kie_ihi_sma"]
    for btal_str in (None, "taa_btal_lin_qqq", "btal_lin_spy"):
        for size_int in range(0, len(other_list) + 1):
            for subset_tuple in combinations(other_list, size_int):
                member_list = ([btal_str] if btal_str else []) + list(subset_tuple)
                if not member_list:
                    continue
                pod_list = ["core5"] + member_list
                name_str = " + ".join(LABEL_BY_ALIAS_DICT[a] for a in pod_list)
                book_list.append((name_str, {a: 1.0 / len(pod_list) for a in pod_list}, "annual", "equal"))
                anchor_dict = {"core5": 0.5, **{a: 0.5 / len(member_list) for a in member_list}}
                book_list.append((name_str + " [CORE5 50%]", anchor_dict, "annual", "anchor50"))
    return book_list


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    for alias_str in ("btal_lin_spy", "nobtal_lin_spy"):
        sleeve_df[alias_str] = extra_return_ser(alias_str, sleeve_df.index)
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]

    book_list = book_list_build()
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    for product_str in ("LT_DEF", "DEF"):
        group_df = weight_df[weight_df["product_id_str"] == product_str]
        book_list.append((f"menu {product_str}", dict(zip(group_df["alias_str"], group_df["weight_float"])), "annual", "reference"))
    ladder_1_dict = {"taa_btal_lin_qqq": 0.55, "sector_vox_iyr": 0.45}
    book_list.append(("ladder_1 (drift, as defined)", ladder_1_dict, "none", "reference"))
    book_list.append(("ladder_1 (annual reset)", ladder_1_dict, "annual", "reference"))
    long_stand_in_all_dict = {"taa_btal_tqqq": "taa_1n_qld", **STAND_IN_DICT}
    for alias_str, stand_in_str in long_stand_in_all_dict.items():
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]

    used_alias_set = {a for _, d, _, _ in book_list for a in d}
    stressed_df = sleeve_df.copy()
    trade_date_dict = {}
    for alias_str in sorted(used_alias_set):
        transaction_df = transaction_df_for(alias_str)
        trade_date_dict[alias_str] = set(transaction_df.loc[(transaction_df["date"] >= CUT_TS) & (transaction_df["date"] <= END_TS), "date"])
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_df, nav_for(alias_str).loc[:END_TS], EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]
    exact_years_float = len(sleeve_df.loc[CUT_TS:]) / 252.0

    def book(frame_df: pd.DataFrame, weight_dict: dict, start_ts: pd.Timestamp, policy_str: str) -> pd.Series:
        return common.book_return_ser(frame_df.loc[start_ts:, list(weight_dict)], weight_dict, policy_str)[0]

    def metrics(ser: pd.Series) -> dict:
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        return common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)

    def sharpe_float(ser: pd.Series) -> float:
        return float(ser.mean() / ser.std() * np.sqrt(252.0))

    row_list = []
    for name_str, weight_dict, policy_str, kind_str in book_list:
        exact_ser = book(sleeve_df, weight_dict, CUT_TS, policy_str)
        long_ser = book(long_df, weight_dict, LONG_TS, policy_str)
        stress_ser = book(stressed_df, weight_dict, CUT_TS, policy_str)
        m, ml, ms = metrics(exact_ser), metrics(long_ser), metrics(stress_ser)
        halves = np.array_split(exact_ser.index, 2)
        nav_ser = common.nav_from_return_ser(exact_ser)
        out = {"book": name_str, "kind": kind_str, "pods": len(weight_dict),
               "trade_days_per_year": len(set().union(*(trade_date_dict[a] for a in weight_dict))) / exact_years_float,
               "wired_share": sum(w for a, w in weight_dict.items() if TIER_BY_ALIAS_DICT.get(a) == "wired"),
               "research_share": sum(w for a, w in weight_dict.items() if TIER_BY_ALIAS_DICT.get(a) == "research"),
               "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"], "sharpe_excess": m["sharpe_excess_float"],
               "sharpe_excess_long": ml["sharpe_excess_float"], "maxdd": m["max_drawdown_float"],
               "worst_12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()), "worst_year": m["worst_year_float"],
               "sharpe_h1": sharpe_float(exact_ser.loc[halves[0]]), "sharpe_h2": sharpe_float(exact_ser.loc[halves[1]]),
               "sharpe_stress": ms["sharpe_rf0_float"], "cagr_stress": ms["cagr_float"],
               "cagr_long": ml["cagr_float"], "sharpe_long": ml["sharpe_rf0_float"], "maxdd_long": ml["max_drawdown_float"],
               "maxdd_long_trough": ml["max_dd_trough_date_str"]}
        for episode_str, start_str, end_str in EPISODE_LIST:
            out[episode_str] = common.window_return_float(long_ser, start_str, end_str)
        row_list.append(out)
    result_df = pd.DataFrame(row_list).set_index("book")

    core5_row = result_df.loc["CORE5 alone"]
    equal_df = result_df[result_df["kind"] == "equal"].copy()
    equal_df["q1_sharpe_all_windows"] = (equal_df["sharpe"] > core5_row["sharpe"]) & (equal_df["sharpe_h1"] > core5_row["sharpe_h1"]) & (
        equal_df["sharpe_h2"] > core5_row["sharpe_h2"])
    equal_df["q2_cagr"] = equal_df["cagr"] > core5_row["cagr"]
    equal_df["q3_dd_2008_band"] = equal_df["maxdd_long"] >= MAX_DD_BAND_FLOAT
    equal_df["q4_stress_sharpe"] = equal_df["sharpe_stress"] > core5_row["sharpe_stress"]
    equal_df["qualifies"] = equal_df[["q1_sharpe_all_windows", "q2_cagr", "q3_dd_2008_band", "q4_stress_sharpe"]].all(axis=1)
    pick_df = equal_df[equal_df["qualifies"]].sort_values(["pods", "sharpe_long", "sharpe"], ascending=[True, False, False])

    corr_alias_list = ["core5", "taa_btal_lin_qqq", "btal_lin_spy", "sector_vox_iyr", "tactical_fi", "disp_kie_ihi_sma"]
    corr_df = sleeve_df.loc[CUT_TS:, corr_alias_list].corr().rename(index=LABEL_BY_ALIAS_DICT, columns=LABEL_BY_ALIAS_DICT)
    long_corr_df = long_df.loc[LONG_TS:, corr_alias_list].corr().rename(index=LABEL_BY_ALIAS_DICT, columns=LABEL_BY_ALIAS_DICT)
    sleeve_row_list = []
    for alias_str in corr_alias_list:
        m, ml = metrics(sleeve_df[alias_str].loc[CUT_TS:].dropna()), metrics(long_df[alias_str].loc[LONG_TS:].dropna())
        sleeve_row_list.append({"sleeve": LABEL_BY_ALIAS_DICT[alias_str], "tier": TIER_BY_ALIAS_DICT[alias_str], "cagr": m["cagr_float"],
                                "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"], "maxdd": m["max_drawdown_float"],
                                "cagr_long": ml["cagr_float"], "sharpe_long": ml["sharpe_rf0_float"], "maxdd_long": ml["max_drawdown_float"],
                                "gfc": common.window_return_float(long_df[alias_str], "2008-05-19", "2009-03-09"),
                                "trade_days_per_year": len(trade_date_dict[alias_str]) / exact_years_float})
    sleeve_stats_df = pd.DataFrame(sleeve_row_list).set_index("sleeve")

    STUDY_DIR_PATH.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(STUDY_DIR_PATH / "books.csv", float_format="%.6g")
    equal_df.to_csv(STUDY_DIR_PATH / "books_equal_qualification.csv", float_format="%.6g")
    sleeve_stats_df.to_csv(STUDY_DIR_PATH / "sleeves.csv", float_format="%.6g")
    corr_df.to_csv(STUDY_DIR_PATH / "corr_exact.csv", float_format="%.4g")
    long_corr_df.to_csv(STUDY_DIR_PATH / "corr_long.csv", float_format="%.4g")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)
    print(sleeve_stats_df.round(3).to_string())
    print()
    print(corr_df.round(2).to_string())
    print()
    show_list = ["pods", "trade_days_per_year", "cagr", "vol", "sharpe", "sharpe_h1", "sharpe_h2", "maxdd", "worst_year", "sharpe_stress",
                 "cagr_long", "sharpe_long", "maxdd_long", "gfc", "covid", "y2022"]
    print(result_df[result_df["kind"].isin(["benchmark", "reference", "equal"])][show_list].round(3).to_string())
    print()
    print(result_df[result_df["kind"] == "anchor50"][show_list].round(3).to_string())
    print()
    print(equal_df[["pods", "q1_sharpe_all_windows", "q2_cagr", "q3_dd_2008_band", "q4_stress_sharpe", "qualifies"]].to_string())
    print()
    print("PICK ORDER:")
    print(pick_df[["pods", "cagr", "sharpe", "sharpe_long", "maxdd_long"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
