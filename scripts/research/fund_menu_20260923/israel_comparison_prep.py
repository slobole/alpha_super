"""Inputs for comparing the menu with Israeli hedge funds (owner question, 2026-09-24).

Descriptive, nothing tuned.
  1. In-house shrinkage: Sharpe of every sleeve in the first half of the exact window vs the second half.
     The slope of H2 on H1 across sleeves and the rank correlation say how much of a good first-half
     Sharpe carried into the second half (selection / regression-to-the-mean evidence from our own data).
     Caveat: the catalog was chosen with the full window in view, so the second half is not truly
     out-of-sample and this understates selection bias.
  2. Calendar-year returns of the products the way an Israeli fund reports them: net of a typical hedge-fund
     fee (1.5% a year management accrued daily, 20% performance fee over a high-water mark crystallised at
     each year end), in USD and in ILS unhedged (USDILS from Norgate).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

END_TS, START_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02")
PRODUCT_LIST = ["DEF", "LT_DEF", "LT_BAL", "LT_GRO", "GRO", "AGG"]
LADDER_DICT = {"ladder_4_growth": {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}}
MANAGEMENT_FEE_FLOAT, PERFORMANCE_FEE_FLOAT = 0.015, 0.20


def sharpe_float(ser: pd.Series) -> float:
    return float(ser.mean() / ser.std() * np.sqrt(252.0))


def net_of_fees_ser(gross_ser: pd.Series) -> pd.Series:
    """Daily net returns after management accrual and a year-end performance fee over a high-water mark."""
    nav_float, hwm_float, net_list = 1.0, 1.0, []
    year_arr = gross_ser.index.year
    for position_int, gross_float in enumerate(gross_ser.to_numpy()):
        prior_nav_float = nav_float
        nav_float *= (1.0 + gross_float) * (1.0 - MANAGEMENT_FEE_FLOAT / 252.0)
        is_year_end_bool = position_int == len(gross_ser) - 1 or year_arr[position_int + 1] != year_arr[position_int]
        if is_year_end_bool and nav_float > hwm_float:
            nav_float -= PERFORMANCE_FEE_FLOAT * (nav_float - hwm_float)
            hwm_float = nav_float
        net_list.append(nav_float / prior_nav_float - 1.0)
    return pd.Series(net_list, index=gross_ser.index)


def calendar_year_ser(ret_ser: pd.Series) -> pd.Series:
    return (1.0 + ret_ser).groupby(ret_ser.index.year).prod() - 1.0


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[START_TS:END_TS]
    sleeve_df = sleeve_df.iloc[1:]
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    book_ser_dict = {}
    for product_str, group_df in weight_df.groupby("product_id_str"):
        weight_dict = dict(zip(group_df["alias_str"], group_df["weight_float"]))
        book_ser_dict[product_str] = common.book_return_ser(sleeve_df[list(weight_dict)], weight_dict, "annual")[0]
    for name_str, weight_dict in LADDER_DICT.items():
        book_ser_dict[name_str] = common.book_return_ser(sleeve_df[list(weight_dict)], weight_dict, "none")[0]

    # 1. Shrinkage across sleeves (and the books for reference).
    mid_ts = sleeve_df.index[len(sleeve_df) // 2]
    shrink_row_list = []
    for name_str, ser in list(sleeve_df.items()) + list(book_ser_dict.items()):
        ser = ser.dropna()
        if ser.index[0] > START_TS + pd.Timedelta(days=10):
            continue
        shrink_row_list.append({"series": name_str, "kind": "book" if name_str in book_ser_dict else "sleeve",
                                "sharpe_h1": sharpe_float(ser.loc[:mid_ts]), "sharpe_h2": sharpe_float(ser.loc[mid_ts:])})
    shrink_df = pd.DataFrame(shrink_row_list).set_index("series")
    shrink_df["ratio_h2_h1"] = shrink_df["sharpe_h2"] / shrink_df["sharpe_h1"]
    sleeve_part_df = shrink_df[shrink_df["kind"] == "sleeve"]
    slope_float, intercept_float = np.polyfit(sleeve_part_df["sharpe_h1"], sleeve_part_df["sharpe_h2"], 1)
    rank_corr_float = float(sleeve_part_df["sharpe_h1"].rank().corr(sleeve_part_df["sharpe_h2"].rank()))
    top_df = sleeve_part_df[sleeve_part_df["sharpe_h1"] >= sleeve_part_df["sharpe_h1"].median()]

    # 2. Calendar-year returns, net of a typical hedge-fund fee, USD and ILS unhedged.
    usdils_ser = common.load_total_return_close_ser("USDILS", "2012-01-01", END_TS.strftime("%Y-%m-%d")).reindex(
        pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).index).ffill()
    fx_ser = usdils_ser.pct_change(fill_method=None).reindex(sleeve_df.index).fillna(0.0)
    year_frame_dict = {}
    for name_str in PRODUCT_LIST + list(LADDER_DICT):
        gross_usd_ser = book_ser_dict[name_str]
        gross_ils_ser = (1.0 + gross_usd_ser) * (1.0 + fx_ser.reindex(gross_usd_ser.index)) - 1.0
        for label_str, ser in (("usd_gross", gross_usd_ser), ("usd_net", net_of_fees_ser(gross_usd_ser)),
                               ("ils_gross", gross_ils_ser), ("ils_net", net_of_fees_ser(gross_ils_ser))):
            year_frame_dict[f"{name_str}:{label_str}"] = calendar_year_ser(ser)
    year_df = pd.DataFrame(year_frame_dict)
    year_df.index = [str(y) if y != END_TS.year else f"{y} YTD" for y in year_df.index]
    year_df = year_df.drop(index=str(START_TS.year))  # partial first year

    output_dir_path = common.STUDY_DIR_PATH / "books"
    shrink_df.to_csv(output_dir_path / "israel_prep_shrinkage.csv", float_format="%.6g")
    year_df.to_csv(output_dir_path / "israel_prep_calendar_years.csv", float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(f"split at {mid_ts.date()}")
    print(shrink_df.sort_values(["kind", "sharpe_h1"], ascending=[False, False]).round(2).to_string())
    print(f"\nsleeves: slope H2 on H1 = {slope_float:.2f}, intercept = {intercept_float:.2f}, rank corr = {rank_corr_float:.2f}, "
          f"median ratio = {sleeve_part_df['ratio_h2_h1'].median():.2f}; top-half-by-H1 mean H1 = {top_df['sharpe_h1'].mean():.2f} -> H2 = {top_df['sharpe_h2'].mean():.2f}")
    cols = [c for c in year_df.columns if c.split(":")[0] in ("DEF", "LT_GRO", "AGG", "ladder_4_growth")]
    print()
    print((year_df[cols] * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
