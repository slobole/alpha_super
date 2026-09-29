"""Can a book from the catalog reach 24%+ CAGR with a good Sharpe? (owner question, 2026-09-23)

Pre-declared, short candidate list - no search:
  AGG_1N   : the menu's Aggressive with the 1/N TAA in place of the rank-weighted TAA (25% each)
  AGG_MAX  : the most-exposed implementation of each return engine, 25% each:
             TAA 1/N, NDX momentum unscaled, DV2, Inflation Compass
  RET_ALL_1N: all eight main-line return sleeves, equal capital per engine, TAA 1/N
  LEV_BAL / LEV_GRO: the menu's Balanced / Growth run with margin leverage, financed daily at
             T-bills + 1.5% (IBKR-like), leverage 1.75x / 1.5x - NOT supported by the engine or LIVE today
References: menu Aggressive, Growth, Balanced, TAA rank alone, TAA 1/N alone, current ladder 4.
Sleeve returns are the $1M fresh runs of the fund-menu study; annual reset; window 2012-10-02 -> 2026-08-19,
plus 2008-03-04 onward with the 2x no-BTAL TAA standing in ONLY before 2012-10-02.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import evaluation  # noqa: E402

END_TS, CUT_TS, EXACT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
STAND_IN_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld"}
BOOK_DICT = {
    "AGG_1N": {"taa_btal_1n_tqqq": .25, "ndx_vxn": .25, "mosaic": .25, "infl_compass": .25},
    "AGG_MAX": {"taa_btal_1n_tqqq": .25, "ndx_atr": .25, "dv2": .25, "infl_compass": .25},
    "RET_ALL_1N": {"hpi_vote": .125, "dv2": .125, "sector_vox_iyr": .125, "disp_kie_ihi_sma": .125,
                   "ndx_vxn": .125, "mosaic": .125, "taa_btal_1n_tqqq": .125, "infl_compass": .125},
    "MENU_AGG": {"ndx_vxn": .25, "mosaic": .25, "taa_btal_tqqq": .25, "infl_compass": .25},
    "MENU_GRO": {"hpi_vote": .12, "dv2": .12, "disp_kie_ihi_sma": .12, "sector_vox_iyr": .11, "ndx_vxn": .11, "mosaic": .11,
                 "taa_btal_tqqq": .11, "infl_compass": .11, "core5": .09},
    "MENU_BAL": {"core5": .18, "tactical_fi": .09, "eom_flow": .09, "sector_vox_iyr": .08, "dv2": .08, "hpi_vote": .08,
                 "disp_kie_ihi_sma": .08, "taa_btal_tqqq": .08, "mosaic": .08, "ndx_vxn": .08, "infl_compass": .08},
    "TAA_RANK": {"taa_btal_tqqq": 1.0},
    "TAA_1N": {"taa_btal_1n_tqqq": 1.0},
    "LADDER_4": {"dv2": .16, "hpi_vote": .17, "ndx_vxn": .25, "mosaic": .08, "taa_btal_tqqq": .34},
}
LEVER_DICT = {"LEV_BAL_1.75x": ("MENU_BAL", 1.75), "LEV_GRO_1.5x": ("MENU_GRO", 1.5)}


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]
    rate_ser = evaluation.lagged_tbill_annual_rate_ser(sleeve_df.index)
    day_ser = pd.Series(sleeve_df.index, index=sleeve_df.index).diff().dt.days.fillna(0.0)

    def book(frame_df: pd.DataFrame, weight_dict: dict, start_ts: pd.Timestamp) -> pd.Series:
        return common.book_return_ser(frame_df.loc[start_ts:, list(weight_dict)], weight_dict, "annual")[0]

    def levered(ser: pd.Series, lever_float: float) -> pd.Series:
        # r_L = L * r - (L - 1) * (T-bill + 1.5%) * days / 360 ; leverage held constant daily (an approximation).
        financing_ser = (rate_ser.reindex(ser.index) + 0.015) * day_ser.reindex(ser.index) / 360.0
        return lever_float * ser - (lever_float - 1.0) * financing_ser

    rows = []
    series_dict = {}
    for name_str, weight_dict in BOOK_DICT.items():
        series_dict[name_str] = (book(sleeve_df, weight_dict, EXACT_TS), book(long_df, weight_dict, LONG_TS))
    for name_str, (base_str, lever_float) in LEVER_DICT.items():
        exact_ser, long_ser = series_dict[base_str]
        series_dict[name_str] = (levered(exact_ser, lever_float), levered(long_ser, lever_float))
    for name_str, (exact_ser, long_ser) in series_dict.items():
        out = {"book": name_str}
        for label_str, ser in (("exact", exact_ser), ("long", long_ser)):
            base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
            m = common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
            nav_ser = common.nav_from_return_ser(ser)
            out.update({f"cagr_{label_str}": m["cagr_float"], f"vol_{label_str}": m["volatility_float"], f"sharpe_{label_str}": m["sharpe_rf0_float"],
                        f"maxdd_{label_str}": m["max_drawdown_float"], f"worst12m_{label_str}": float((nav_ser / nav_ser.shift(252) - 1).dropna().min())})
            if label_str == "exact":
                thirds = np.array_split(ser.index, 3)
                for i, part in enumerate(thirds, 1):
                    part_nav = common.nav_from_return_ser(ser.loc[part])
                    out[f"cagr_third{i}"] = float(part_nav.iloc[-1] ** (365.25 / (part[-1] - part[0]).days) - 1)
                out["beta"] = m["beta_spx_float"]
        for ep_str, start_str, end_str in (("covid", "2020-02-19", "2020-03-23"), ("y2022", "2022-01-03", "2022-10-12"), ("gfc", "2008-05-19", "2009-03-09")):
            out[ep_str] = common.window_return_float(long_ser, start_str, end_str)
        rows.append(out)
    result_df = pd.DataFrame(rows).set_index("book")
    result_df.to_csv(common.STUDY_DIR_PATH / "books" / "aggressive_24_check.csv", float_format="%.6g")
    pd.set_option("display.width", 260)
    print(result_df[["cagr_exact", "vol_exact", "sharpe_exact", "maxdd_exact", "worst12m_exact", "beta", "cagr_third1", "cagr_third2", "cagr_third3"]].round(3).to_string())
    print(result_df[["cagr_long", "vol_long", "sharpe_long", "maxdd_long", "worst12m_long", "gfc", "covid", "y2022"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
