"""Does BTAL earn its place, and why 6% of a 3x TAA rather than a 1x one? (owner questions, 2026-09-24)

PRE-DECLARED before any of these comparisons was computed:
A. BTAL isolation: the same Defense First TAA with and without BTAL in the defensive basket, same weighting rule
   and same fallback leverage, on the common window 2012-10-02 -> 2026-08-19 (BTAL-based sleeves start there):
     1x linearity  taa_btal_lin_qqq  vs taa_lin_qqq        1x rank  btal_rank_qqq    vs nobtal_rank_qqq
     2x equal slot taa_btal_1n_qld   vs taa_1n_qld         3x equal slot taa_btal_1n_tqqq vs nobtal_1n_tqqq
     3x rank       taa_btal_tqqq     vs nobtal_rank_tqqq
   Metrics: CAGR, volatility, Sharpe, Sharpe over T-bills, max drawdown, worst 12 months, 21-day CVaR 5%,
   Sharpe in each half, and five equity shocks (Feb 2018, Q4 2018, COVID, 2022, 2025 tariffs).
B. Books, annual reset:
   CORE5 + BTAL_QQQ vs CORE5 + the same TAA without BTAL (50/50).
   Low-touch Defensive (amended A1) with its 6% TAA slot filled by:
     L0 3x rank with BTAL (as built)          L1 3x rank without BTAL (6%)
     L2 1x rank with BTAL at the same 6%       L3 1x rank with BTAL at equal risk
     L4 1x linearity with BTAL at equal risk    L5 1x linearity without BTAL at equal risk
   Equal risk = 6% x (volatility of the 3x sleeve / volatility of the 1x sleeve) on the exact window; the other pods
   are scaled pro rata to make room. Long window from 2008-03-04 fills each BTAL or 3x sleeve with its closest
   no-BTAL / 2x sibling ONLY before that sleeve exists.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import common  # noqa: E402

END_TS, CUT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "btal_leverage_20260924"
EXTRA_ALIAS_LIST = ["nobtal_rank_tqqq", "nobtal_1n_tqqq", "btal_rank_qqq", "nobtal_rank_qqq"]
PAIR_LIST = [("1x linearity", "taa_btal_lin_qqq", "taa_lin_qqq"), ("1x rank", "btal_rank_qqq", "nobtal_rank_qqq"),
             ("2x equal slots", "taa_btal_1n_qld", "taa_1n_qld"), ("3x equal slots", "taa_btal_1n_tqqq", "nobtal_1n_tqqq"),
             ("3x rank (the menu's TAA)", "taa_btal_tqqq", "nobtal_rank_tqqq")]
SHOCK_LIST = [("feb_2018", "2018-01-26", "2018-02-08"), ("q4_2018", "2018-09-20", "2018-12-24"), ("covid", "2020-02-19", "2020-03-23"),
              ("y2022", "2022-01-03", "2022-10-12"), ("tariff_2025", "2025-02-19", "2025-04-08")]
# Long-window fill for sleeves without 2008 history: the closest sibling that has it.
LONG_FILL_DICT = {"taa_btal_tqqq": "taa_1n_qld", "nobtal_rank_tqqq": "taa_1n_qld", "btal_rank_qqq": "nobtal_rank_qqq",
                  "taa_btal_lin_qqq": "taa_lin_qqq"}


def extra_return_ser(alias_str: str, index: pd.DatetimeIndex) -> pd.Series:
    path_df = pd.read_csv(STUDY_DIR_PATH / "sources" / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)
    nav_ser = path_df["total_value_float"].astype(float)
    invested_ser = path_df["portfolio_value_float"].abs() > 1e-9
    first_position_int = nav_ser.index.get_loc(invested_ser[invested_ser].index[0])
    return nav_ser.iloc[max(first_position_int - 1, 0):].pct_change(fill_method=None).reindex(index)


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    for alias_str in EXTRA_ALIAS_LIST:
        sleeve_df[alias_str] = extra_return_ser(alias_str, sleeve_df.index)
    long_df = sleeve_df.copy()
    for alias_str, fill_str in LONG_FILL_DICT.items():
        # *** CRITICAL*** the sibling fills only the dates before the sleeve itself has a return.
        missing_mask = sleeve_df[alias_str].isna() & (sleeve_df.index >= LONG_TS)
        long_df.loc[missing_mask, alias_str] = sleeve_df.loc[missing_mask, fill_str]

    def metrics(ser: pd.Series) -> dict:
        ser = ser.dropna()
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        m = common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
        nav_ser = common.nav_from_return_ser(ser)
        halves = np.array_split(ser.index, 2)
        out = {"cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"], "sharpe_excess": m["sharpe_excess_float"],
               "maxdd": m["max_drawdown_float"], "worst_12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()),
               "cvar5_21d": m["es95_21d_float"],
               "sharpe_h1": float(ser.loc[halves[0]].mean() / ser.loc[halves[0]].std() * np.sqrt(252)),
               "sharpe_h2": float(ser.loc[halves[1]].mean() / ser.loc[halves[1]].std() * np.sqrt(252))}
        out.update({k: common.window_return_float(ser, s, e) for k, s, e in SHOCK_LIST})
        return out

    pair_row_list = []
    for label_str, with_str, without_str in PAIR_LIST:
        for variant_str, alias_str in (("with BTAL", with_str), ("without BTAL", without_str)):
            pair_row_list.append({"pair": label_str, "variant": variant_str, "alias": alias_str, **metrics(sleeve_df[alias_str].loc[CUT_TS:])})
    pair_df = pd.DataFrame(pair_row_list)
    delta_row_list = []
    for label_str, group_df in pair_df.groupby("pair", sort=False):
        with_row, without_row = group_df.iloc[0], group_df.iloc[1]
        delta_row_list.append({"pair": label_str, **{k: with_row[k] - without_row[k] for k in ["cagr", "vol", "sharpe", "sharpe_excess", "maxdd",
                                                                                                  "cvar5_21d", "covid", "y2022", "tariff_2025"]}})
    delta_df = pd.DataFrame(delta_row_list).set_index("pair")

    def book(frame_df: pd.DataFrame, weight_dict: dict, start_ts: pd.Timestamp) -> pd.Series:
        return common.book_return_ser(frame_df.loc[start_ts:, list(weight_dict)], weight_dict, "annual")[0]

    vol_ser = sleeve_df.loc[CUT_TS:].std() * np.sqrt(252)
    base_dict = {"core5": 0.55, "tactical_fi": 0.27, "ndx_vxn": 0.06, "mosaic": 0.06}

    def with_slot(alias_str: str, slot_float: float) -> dict:
        scale_float = (1.0 - slot_float) / sum(base_dict.values())
        return {**{a: w * scale_float for a, w in base_dict.items()}, alias_str: slot_float}

    equal_risk_dict = {a: 0.06 * vol_ser["taa_btal_tqqq"] / vol_ser[a] for a in ("btal_rank_qqq", "taa_btal_lin_qqq", "taa_lin_qqq")}
    book_list = [
        ("CORE5 + BTAL_QQQ", {"core5": 0.5, "taa_btal_lin_qqq": 0.5}),
        ("CORE5 + same TAA without BTAL", {"core5": 0.5, "taa_lin_qqq": 0.5}),
        ("L0 LT_DEF as built: 6% 3x rank, BTAL", with_slot("taa_btal_tqqq", 0.06)),
        ("L1 6% 3x rank, no BTAL", with_slot("nobtal_rank_tqqq", 0.06)),
        ("L2 6% 1x rank, BTAL (same capital)", with_slot("btal_rank_qqq", 0.06)),
        (f"L3 {equal_risk_dict['btal_rank_qqq']:.1%} 1x rank, BTAL (equal risk)", with_slot("btal_rank_qqq", equal_risk_dict["btal_rank_qqq"])),
        (f"L4 {equal_risk_dict['taa_btal_lin_qqq']:.1%} 1x linearity, BTAL (equal risk)", with_slot("taa_btal_lin_qqq", equal_risk_dict["taa_btal_lin_qqq"])),
        (f"L5 {equal_risk_dict['taa_lin_qqq']:.1%} 1x linearity, no BTAL (equal risk)", with_slot("taa_lin_qqq", equal_risk_dict["taa_lin_qqq"])),
    ]
    book_row_list = []
    for name_str, weight_dict in book_list:
        exact_dict = metrics(book(sleeve_df, weight_dict, CUT_TS))
        long_dict = metrics(book(long_df, weight_dict, LONG_TS))
        book_row_list.append({"book": name_str, **exact_dict, "cagr_long": long_dict["cagr"], "sharpe_long": long_dict["sharpe"],
                              "maxdd_long": long_dict["maxdd"], "gfc": common.window_return_float(book(long_df, weight_dict, LONG_TS), "2008-05-19", "2009-03-09")})
    book_df = pd.DataFrame(book_row_list).set_index("book")

    STUDY_DIR_PATH.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(STUDY_DIR_PATH / "btal_pairs.csv", index=False, float_format="%.6g")
    delta_df.to_csv(STUDY_DIR_PATH / "btal_pair_deltas.csv", float_format="%.6g")
    book_df.to_csv(STUDY_DIR_PATH / "books.csv", float_format="%.6g")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(pair_df.drop(columns="alias").round(3).to_string(index=False))
    print()
    print((delta_df * 100).round(2).to_string())
    print()
    print("volatility, exact window:", vol_ser[["taa_btal_tqqq", "nobtal_rank_tqqq", "btal_rank_qqq", "taa_btal_lin_qqq", "taa_lin_qqq"]].round(3).to_dict())
    print(book_df.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
