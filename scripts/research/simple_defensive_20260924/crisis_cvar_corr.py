"""Crisis, CVaR-5% and correlation dossier for the simple defensive candidates and LT_DEF variants (2026-09-24).

PRE-DECLARED book list (weights fixed before this run; nothing is tuned):
  Simple: CORE5 alone; CORE5 + BTAL_QQQ (50/50); CORE5 + BTAL_QQQ + {DISP | FI | DOWNSHOCK} (1/3 each);
          CORE5 + FI + DISP (1/3 each).
  LT_DEF family (owner question on Inflation Compass):
    LT_DEF as frozen; LT_DEF without InflC - frozen-spec template (its slot goes to the 3x TAA, share re-solved
    to the 5% volatility target: the menu study's 'without_flagged' weights); LT_DEF without InflC - pro rata;
    LT_DEF with InflC's 6% moved to BTAL_QQQ.
  References: DEF, ladder_1 (drift, as defined).
Windows: long 2008-03-04 -> 2026-08-19 with BTAL sleeves replaced by their no-BTAL twins ONLY before 2012-10-02
  (taa_btal_lin_qqq -> taa_lin_qqq, taa_btal_tqqq -> taa_1n_qld); exact 2012-10-02 -> 2026-08-19 for reference.
Crises: every S&P 500 TR peak-to-trough fall of 10% or more in the long window (objective list), plus four
  pre-declared rate shocks that hurt bond and gold holders: taper tantrum 2013-05-02 -> 2013-09-05, reflation
  2016-07-08 -> 2016-12-15, 2022 rate shock 2022-01-03 -> 2022-10-24, long-bond rout 2023-07-31 -> 2023-10-19.
CVaR 5%: mean of the worst 5% of daily returns, and of the worst 5% of rolling 21-day returns.
Correlation: book vs S&P 500 on all days, monthly, on the S&P's worst 10% of days, and on crisis days; and the
  pods against each other on all days vs the S&P's worst 10% of days.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fund_menu_20260923"))
import common  # noqa: E402
import simple_defensive_study as study  # noqa: E402

LONG_STAND_IN_DICT = {"taa_btal_lin_qqq": "taa_lin_qqq", "taa_btal_tqqq": "taa_1n_qld"}
RATE_SHOCK_LIST = [("rates: taper tantrum 2013", "2013-05-02", "2013-09-05"), ("rates: reflation 2016", "2016-07-08", "2016-12-15"),
                   ("rates: 2022 shock", "2022-01-03", "2022-10-24"), ("rates: bond rout 2023", "2023-07-31", "2023-10-19")]
POD_LIST = ["core5", "taa_btal_lin_qqq", "disp_kie_ihi_sma", "tactical_fi", "sector_vox_iyr", "infl_compass"]
POD_LABEL_DICT = {**study.LABEL_BY_ALIAS_DICT, "infl_compass": "INFL_C"}


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:study.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:study.END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in LONG_STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < study.CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < study.CUT_TS, stand_in_str]

    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    menu_dict = {p: dict(zip(g["alias_str"], g["weight_float"])) for p, g in weight_df.groupby("product_id_str")}
    lt_def_dict = menu_dict["LT_DEF"]
    third_float = 1.0 / 3.0
    pro_rata_dict = {a: w / (1.0 - lt_def_dict["infl_compass"]) for a, w in lt_def_dict.items() if a != "infl_compass"}
    swap_dict = {("taa_btal_lin_qqq" if a == "infl_compass" else a): w for a, w in lt_def_dict.items()}
    book_list = [
        ("CORE5 alone", {"core5": 1.0}, "annual"),
        ("CORE5 + BTAL_QQQ", {"core5": 0.5, "taa_btal_lin_qqq": 0.5}, "annual"),
        ("CORE5 + BTAL_QQQ + DISP", {"core5": third_float, "taa_btal_lin_qqq": third_float, "disp_kie_ihi_sma": third_float}, "annual"),
        ("CORE5 + BTAL_QQQ + FI", {"core5": third_float, "taa_btal_lin_qqq": third_float, "tactical_fi": third_float}, "annual"),
        ("CORE5 + BTAL_QQQ + DOWNSHOCK", {"core5": third_float, "taa_btal_lin_qqq": third_float, "sector_vox_iyr": third_float}, "annual"),
        ("CORE5 + FI + DISP", {"core5": third_float, "tactical_fi": third_float, "disp_kie_ihi_sma": third_float}, "annual"),
        ("LT_DEF (frozen)", lt_def_dict, "annual"),
        ("LT_DEF - InflC (template)", {"core5": 0.513, "tactical_fi": 0.256, "taa_btal_tqqq": 0.116, "ndx_vxn": 0.058, "mosaic": 0.057}, "annual"),
        ("LT_DEF - InflC (pro rata)", pro_rata_dict, "annual"),
        ("LT_DEF InflC -> BTAL_QQQ", swap_dict, "annual"),
        ("DEF (reference)", menu_dict["DEF"], "annual"),
        ("ladder_1 (reference, drift)", {"taa_btal_lin_qqq": 0.55, "sector_vox_iyr": 0.45}, "none"),
    ]

    spx_long_ser = bench_df["SPXTR"].loc[study.LONG_TS:]
    episode_list = [(f"equity: {e['peak_date_str'][:7]} ({e['spx_drawdown_float']:.0%})", e["peak_date_str"], e["trough_date_str"])
                    for e in common.equity_drawdown_episode_list(spx_long_ser)]
    crisis_mask_ser = pd.Series(False, index=spx_long_ser.index)
    for _, start_str, end_str in episode_list:
        crisis_mask_ser.loc[start_str:end_str] = True
    stress_mask_ser = spx_long_ser <= spx_long_ser.quantile(0.10)

    def metrics(ser: pd.Series) -> dict:
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        return common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)

    summary_row_list, crisis_row_list = [], []
    for name_str, weight_dict, policy_str in book_list:
        long_ser = common.book_return_ser(long_df.loc[study.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        exact_ser = common.book_return_ser(sleeve_df.loc[study.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        ml, me = metrics(long_ser), metrics(exact_ser)
        nav_ser = common.nav_from_return_ser(long_ser)
        monthly_df = pd.concat([long_ser, spx_long_ser], axis=1, keys=["book", "spx"]).add(1).resample("ME").prod().sub(1)
        summary_row_list.append({
            "book": name_str, "pods": len(weight_dict),
            "cagr_long": ml["cagr_float"], "vol_long": ml["volatility_float"], "sharpe_long": ml["sharpe_rf0_float"],
            "sharpe_excess_long": ml["sharpe_excess_float"], "maxdd_long": ml["max_drawdown_float"],
            "underwater_days_long": ml["longest_underwater_days_int"], "cvar5_daily": ml["es95_daily_float"],
            "cvar5_21d": ml["es95_21d_float"], "worst_month": ml["worst_month_float"],
            "worst_12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()),
            "corr_spx_daily": float(long_ser.corr(spx_long_ser)), "corr_spx_monthly": float(monthly_df["book"].corr(monthly_df["spx"])),
            "beta_spx": ml["beta_spx_float"], "corr_spx_worst10pct_days": float(long_ser[stress_mask_ser].corr(spx_long_ser[stress_mask_ser])),
            "corr_spx_crisis_days": float(long_ser[crisis_mask_ser].corr(spx_long_ser[crisis_mask_ser])),
            "mean_return_worst10pct_spx_days": float(long_ser[stress_mask_ser].mean()),
            "cagr_exact": me["cagr_float"], "sharpe_exact": me["sharpe_rf0_float"], "maxdd_exact": me["max_drawdown_float"],
            "cvar5_21d_exact": me["es95_21d_float"]})
        crisis_row = {"book": name_str}
        for label_str, start_str, end_str in episode_list + RATE_SHOCK_LIST:
            crisis_row[label_str] = common.window_return_float(long_ser, start_str, end_str)
        crisis_row_list.append(crisis_row)
    summary_df = pd.DataFrame(summary_row_list).set_index("book")
    crisis_df = pd.DataFrame(crisis_row_list).set_index("book")
    spx_crisis_row = {label_str: common.window_return_float(spx_long_ser, s, e) for label_str, s, e in episode_list + RATE_SHOCK_LIST}
    crisis_df.loc["S&P 500 TR"] = pd.Series(spx_crisis_row)
    crisis_df["worst_crisis"] = crisis_df.min(axis=1)

    pod_df = long_df.loc[study.LONG_TS:, POD_LIST].rename(columns=POD_LABEL_DICT)
    corr_all_df = pod_df.corr()
    corr_stress_df = pod_df[stress_mask_ser.reindex(pod_df.index).fillna(False)].corr()
    corr_crisis_df = pod_df[crisis_mask_ser.reindex(pod_df.index).fillna(False)].corr()

    out_path = study.STUDY_DIR_PATH
    summary_df.to_csv(out_path / "dossier_summary.csv", float_format="%.6g")
    crisis_df.to_csv(out_path / "dossier_crises.csv", float_format="%.6g")
    corr_all_df.to_csv(out_path / "dossier_pod_corr_all.csv", float_format="%.4g")
    corr_stress_df.to_csv(out_path / "dossier_pod_corr_spx_worst10pct.csv", float_format="%.4g")
    corr_crisis_df.to_csv(out_path / "dossier_pod_corr_crisis_days.csv", float_format="%.4g")
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 60)
    print(summary_df.round(3).to_string())
    print()
    print((crisis_df.T * 100).round(1).to_string())
    print()
    for label_str, frame_df in (("all days", corr_all_df), ("S&P worst 10% days", corr_stress_df), ("S&P >=10% crisis days", corr_crisis_df)):
        print(f"pod correlation, {label_str}:")
        print(frame_df.round(2).to_string())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
