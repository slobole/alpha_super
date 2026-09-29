"""The menu products against the Israeli hedge fund industry, month by month (owner question, 2026-09-24).

Descriptive, nothing tuned.
  Industry: TGI (Gilboa Funds / Tzur index of Israeli hedge funds; equal-weighted, net of fees, self-reported,
  backfilled to 2007) and its quant sub-index, monthly, copied to israel/tgi_monthly_raw.csv with the source
  stamp in israel/tgi_monthly_source.json.
  Products: the menu books as an Israeli investor would see a local fund: in shekels (USD returns converted
  with Norgate USDILS, unhedged), after the standard Israeli fee of 2% a year management (accrued daily)
  plus 20% performance fee over a high-water mark (crystallised each year end). Backtest as is, and with
  a 30% cut of the excess return over T-bills (constant daily drag, applied before FX and fees).
  Windows: 2014-01 -> 2026-08 (quant sub-index available), 2019-2025 (seven full years), and the five
  years to Sep 2025 (the window of Gilboa's own published statistics).
  Caveat: an index of ~200 funds is smoother than any single fund, so its volatility and drawdown are a
  low bar for one product; self-reported indices also carry survivorship bias in their favour.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

END_TS, START_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02")
ISRAEL_DIR_PATH = common.STUDY_DIR_PATH / "israel"
PRODUCT_LIST = ["DEF", "LT_BAL", "LT_GRO", "AGG"]
LADDER_DICT = {"ladder_4_growth": {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}}
MANAGEMENT_FEE_FLOAT, PERFORMANCE_FEE_FLOAT = 0.02, 0.20
HAIRCUT_LIST = [0.0, 0.3]
WINDOW_LIST = [("2014-01_2026-08", "2014-01", "2026-08"), ("2019_2025", "2019-01", "2025-12"), ("5y_to_2025-09", "2020-10", "2025-09")]
MONTH_COLUMN_LIST = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def load_tgi_monthly_df() -> pd.DataFrame:
    raw_df = pd.read_csv(ISRAEL_DIR_PATH / "tgi_monthly_raw.csv")
    series_dict = {}
    for index_str, group_df in raw_df.groupby("index"):
        value_dict = {}
        for _, row in group_df.iterrows():
            for month_int, column_str in enumerate(MONTH_COLUMN_LIST, 1):
                if pd.notna(row[column_str]):
                    value_dict[pd.Timestamp(int(row["year"]), month_int, 1) + pd.offsets.MonthEnd(0)] = float(row[column_str]) / 100.0
        series_dict[index_str] = pd.Series(value_dict).sort_index()
    return pd.DataFrame(series_dict)


def net_of_fees_ser(gross_ser: pd.Series) -> pd.Series:
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


def monthly_stats(month_ser: pd.Series) -> dict:
    month_ser = month_ser.dropna()
    nav_ser = (1.0 + month_ser).cumprod()
    year_ser = (1.0 + month_ser).groupby(month_ser.index.year).prod() - 1.0
    full_year_ser = year_ser[[month_ser.index[month_ser.index.year == y].size == 12 for y in year_ser.index]]
    years_float = len(month_ser) / 12.0
    vol_float = float(month_ser.std() * np.sqrt(12.0))
    cagr_float = float(nav_ser.iloc[-1] ** (1.0 / years_float) - 1.0)
    return {"months": len(month_ser), "cagr": cagr_float, "vol": vol_float, "return_over_vol": float(month_ser.mean() * 12.0 / vol_float),
            "maxdd": float((nav_ser / nav_ser.cummax().clip(lower=1.0) - 1.0).min()), "worst_month": float(month_ser.min()),
            "worst_year": float(full_year_ser.min()) if len(full_year_ser) else np.nan, "negative_years": int((full_year_ser < 0).sum()),
            "full_years": int(len(full_year_ser))}


def main() -> int:
    tgi_df = load_tgi_monthly_df()
    all_sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    sleeve_df = all_sleeve_df.loc[START_TS:END_TS].iloc[1:]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True)
    tbill_ser = bench_df["TBILL"].reindex(sleeve_df.index).fillna(0.0)
    usdils_ser = common.load_total_return_close_ser("USDILS", "2012-01-01", END_TS.strftime("%Y-%m-%d")).reindex(all_sleeve_df.index).ffill()
    fx_ser = usdils_ser.pct_change(fill_method=None).reindex(sleeve_df.index).fillna(0.0)

    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    book_dict = {p: (dict(zip(g["alias_str"], g["weight_float"])), "annual") for p, g in weight_df.groupby("product_id_str") if p in PRODUCT_LIST}
    book_dict.update({k: (v, "none") for k, v in LADDER_DICT.items()})
    month_series_dict = {"TGI (all Israeli hedge funds)": tgi_df["TGI"], "TGI quant sub-index": tgi_df["TGI_QUANT"]}
    for name_str in PRODUCT_LIST + list(LADDER_DICT):
        weight_dict, policy_str = book_dict[name_str]
        usd_ser = common.book_return_ser(sleeve_df[list(weight_dict)], weight_dict, policy_str)[0]
        mean_excess_float = float((usd_ser - tbill_ser).mean())
        for haircut_float in HAIRCUT_LIST:
            # *** CRITICAL*** haircut on the USD backtest first, then shekel conversion, then fees - the order a real fund lives in.
            cut_ser = usd_ser - haircut_float * mean_excess_float
            ils_net_ser = net_of_fees_ser((1.0 + cut_ser) * (1.0 + fx_ser) - 1.0)
            label_str = f"{name_str} ILS net" + ("" if haircut_float == 0 else f" -{haircut_float:.0%}")
            month_series_dict[label_str] = (1.0 + ils_net_ser).resample("ME").prod() - 1.0
            if haircut_float == 0:
                # A shekel-hedged share class earns roughly the USD return plus the ILS-USD rate gap (not modelled here).
                month_series_dict[f"{name_str} USD net (~ILS hedged)"] = (1.0 + net_of_fees_ser(cut_ser)).resample("ME").prod() - 1.0
    month_df = pd.DataFrame(month_series_dict)

    row_list = []
    for window_str, start_str, end_str in WINDOW_LIST:
        window_df = month_df.loc[start_str:end_str]
        for name_str in month_df.columns:
            ser = window_df[name_str].dropna()
            if ser.empty:
                continue
            row_list.append({"window": window_str, "series": name_str, **monthly_stats(ser),
                             "corr_tgi": float(ser.corr(window_df["TGI (all Israeli hedge funds)"]))})
    stats_df = pd.DataFrame(row_list)
    year_df = (1.0 + month_df.loc["2013-01":]).groupby(month_df.loc["2013-01":].index.year).prod(min_count=1) - 1.0
    year_df.index = [str(y) if y != END_TS.year else f"{y} Jan-Aug" for y in year_df.index]

    stats_df.to_csv(ISRAEL_DIR_PATH / "tgi_comparison_stats.csv", index=False, float_format="%.6g")
    year_df.to_csv(ISRAEL_DIR_PATH / "tgi_comparison_calendar_years.csv", float_format="%.6g")
    month_df.to_csv(ISRAEL_DIR_PATH / "tgi_comparison_monthly.csv", float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for window_str, _, _ in WINDOW_LIST:
        print(f"== {window_str}")
        print(stats_df[stats_df.window == window_str].drop(columns="window").set_index("series").round(3).to_string())
        print()
    print((year_df * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
