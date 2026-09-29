"""Supplementary checks for the simple-defensive study (2026-09-24), descriptive only.

  1. Is FI worth a pod? Replace FI by plain cash earning T-bills minus 0.5% (roughly what IBKR pays on idle
     USD), same weights, same annual reset.
  2. When did each sleeve's worst 2008+ drawdown happen (the 2008 proxy question)?
  3. Calendar years 2008-2026 (long window, stand-ins only before 2012-10-02) for the leading simple books,
     CORE5 alone, LT_DEF and ladder_1.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fund_menu_20260923"))
import common  # noqa: E402
import simple_defensive_study as study  # noqa: E402

CASH_HAIRCUT_FLOAT = 0.005


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:study.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:study.END_TS]
    for alias_str in ("btal_lin_spy", "nobtal_lin_spy"):
        sleeve_df[alias_str] = study.extra_return_ser(alias_str, sleeve_df.index)
    rate_ser = common.load_tbill_return_ser(sleeve_df.index)
    days_ser = pd.Series(sleeve_df.index, index=sleeve_df.index).diff().dt.days.fillna(0.0)
    # *** CRITICAL*** lagged T-bill accrual (common.load_tbill_return_ser) minus the broker spread, floored at zero.
    sleeve_df["cash"] = (rate_ser - CASH_HAIRCUT_FLOAT * days_ser / 360.0).clip(lower=0.0)
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in {"taa_btal_tqqq": "taa_1n_qld", **study.STAND_IN_DICT}.items():
        long_df.loc[long_df.index < study.CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < study.CUT_TS, stand_in_str]

    def metrics(ser: pd.Series) -> dict:
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        return common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)

    third_float = 1.0 / 3.0
    book_dict = {
        "CORE5 alone": {"core5": 1.0},
        "CORE5 + cash (50/50)": {"core5": 0.5, "cash": 0.5},
        "CORE5 + FI (50/50)": {"core5": 0.5, "tactical_fi": 0.5},
        "CORE5 + BTAL_QQQ": {"core5": 0.5, "taa_btal_lin_qqq": 0.5},
        "CORE5 + BTAL_QQQ + cash": {"core5": third_float, "taa_btal_lin_qqq": third_float, "cash": third_float},
        "CORE5 + BTAL_QQQ + FI": {"core5": third_float, "taa_btal_lin_qqq": third_float, "tactical_fi": third_float},
        "CORE5 + BTAL_QQQ + DISP": {"core5": third_float, "taa_btal_lin_qqq": third_float, "disp_kie_ihi_sma": third_float},
        "CORE5 + BTAL_QQQ + DOWNSHOCK": {"core5": third_float, "taa_btal_lin_qqq": third_float, "sector_vox_iyr": third_float},
    }
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    group_df = weight_df[weight_df["product_id_str"] == "LT_DEF"]
    book_dict["menu LT_DEF"] = dict(zip(group_df["alias_str"], group_df["weight_float"]))
    ladder_1_dict = {"taa_btal_lin_qqq": 0.55, "sector_vox_iyr": 0.45}

    row_list, year_dict = [], {}
    for name_str, weight_dict in list(book_dict.items()) + [("ladder_1 (drift)", ladder_1_dict)]:
        policy_str = "none" if name_str.startswith("ladder_1") else "annual"
        exact_ser = common.book_return_ser(sleeve_df.loc[study.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        long_ser = common.book_return_ser(long_df.loc[study.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        m, ml = metrics(exact_ser), metrics(long_ser)
        row_list.append({"book": name_str, "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"],
                         "sharpe_excess": m["sharpe_excess_float"], "maxdd": m["max_drawdown_float"], "cagr_long": ml["cagr_float"],
                         "sharpe_long": ml["sharpe_rf0_float"], "maxdd_long": ml["max_drawdown_float"],
                         "maxdd_long_trough": ml["max_dd_trough_date_str"]})
        year_dict[name_str] = (1.0 + long_ser).groupby(long_ser.index.year).prod() - 1.0
    result_df = pd.DataFrame(row_list).set_index("book")
    year_df = pd.DataFrame(year_dict)
    year_df.index = [str(y) if y != study.END_TS.year else f"{y} to Aug-19" for y in year_df.index]

    trough_row_list = []
    for alias_str in ["core5", "taa_btal_lin_qqq", "btal_lin_spy", "sector_vox_iyr", "tactical_fi", "disp_kie_ihi_sma"]:
        ser = long_df[alias_str].loc[study.LONG_TS:].dropna()
        detail_dict = common.max_drawdown_detail_dict(common.nav_from_return_ser(ser))
        trough_row_list.append({"sleeve": study.LABEL_BY_ALIAS_DICT[alias_str], **detail_dict})
    trough_df = pd.DataFrame(trough_row_list).set_index("sleeve")

    result_df.to_csv(study.STUDY_DIR_PATH / "supplementary_books.csv", float_format="%.6g")
    year_df.to_csv(study.STUDY_DIR_PATH / "supplementary_calendar_years_long.csv", float_format="%.6g")
    trough_df.to_csv(study.STUDY_DIR_PATH / "supplementary_sleeve_drawdowns_long.csv", float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(result_df.round(3).to_string())
    print()
    print(trough_df.to_string())
    print()
    print((year_df * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
