"""The menu products against individual Israeli hedge funds (owner question, 2026-09-24).

Descriptive, nothing tuned. Inputs (compiled 2026-09-24 from the Israeli press and TASE Maya, see
israel/funds_source.json):
  israel/funds_private_annual.csv  - annual net returns of private funds as reported to the press (2019-2025)
  israel/funds_mutual_monthly.txt  - monthly net returns of 16 mutual hedge funds (in trust), Apr-2023 on
Products come from israel_hedge_fund_comparison.py outputs (shekels, net of 2% + 20% over a high-water mark;
backtest and a 30% cut of the excess return; USD net as a rough stand-in for a shekel-hedged class).
  A. Private funds with all six years 2020-2025: compound return, worst year, losing years, spread of years.
  B. Mutual funds, common window May-2023 -> Jul-2026, monthly: return, volatility, drawdown, correlations.
  C. Persistence: do the best funds of one year stay on top the next year (rank correlation across funds)?
Caveats: press tables are unaudited, lean to large surviving funds, and a few 2025 figures are Jan-Nov.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ISRAEL_DIR_PATH = common.STUDY_DIR_PATH / "israel"
YEAR_LIST = [2020, 2021, 2022, 2023, 2024, 2025]
PRODUCT_COLUMN_DICT = {"AGG": "AGG ILS net", "AGG -30%": "AGG ILS net -30%", "AGG hedged~": "AGG USD net (~ILS hedged)",
                       "LT_GRO": "LT_GRO ILS net", "LT_GRO -30%": "LT_GRO ILS net -30%", "LT_GRO hedged~": "LT_GRO USD net (~ILS hedged)",
                       "DEF": "DEF ILS net", "DEF -30%": "DEF ILS net -30%", "DEF hedged~": "DEF USD net (~ILS hedged)"}
MUTUAL_WINDOW_TUPLE = ("2023-05", "2026-07")


def parse_mutual_monthly_df() -> pd.DataFrame:
    series_dict = {}
    for line_str in (ISRAEL_DIR_PATH / "funds_mutual_monthly.txt").read_text(encoding="utf-8").splitlines():
        if not line_str.strip():
            continue
        fund_str, body_str = line_str.split("|", 1)
        value_dict = {}
        for segment_str in body_str.split("|"):
            label_str, numbers_str = segment_str.split(":", 1)
            year_int = 2000 + int(label_str.strip()[:2])
            start_month_int = 5 if "May" in label_str else (4 if "Apr" in label_str else 1)
            for offset_int, token_str in enumerate(numbers_str.split()):
                value_dict[pd.Timestamp(year_int, start_month_int + offset_int, 1) + pd.offsets.MonthEnd(0)] = float(token_str) / 100.0
        series_dict[fund_str.strip()] = pd.Series(value_dict).sort_index()
    return pd.DataFrame(series_dict)


def annual_stats(year_ser: pd.Series) -> dict:
    year_ser = year_ser.dropna()
    return {"years": len(year_ser), "cagr": float(np.prod(1.0 + year_ser) ** (1.0 / len(year_ser)) - 1.0),
            "worst_year": float(year_ser.min()), "losing_years": int((year_ser < 0).sum()), "stdev_years": float(year_ser.std()),
            "y2022": float(year_ser.get(2022, np.nan)), "y2025": float(year_ser.get(2025, np.nan))}


def monthly_stats(month_ser: pd.Series) -> dict:
    nav_ser = (1.0 + month_ser).cumprod()
    vol_float = float(month_ser.std() * np.sqrt(12.0))
    return {"cagr": float(nav_ser.iloc[-1] ** (12.0 / len(month_ser)) - 1.0), "vol": vol_float,
            "return_over_vol": float(month_ser.mean() * 12.0 / vol_float),
            "maxdd": float((nav_ser / nav_ser.cummax().clip(lower=1.0) - 1.0).min()), "worst_month": float(month_ser.min())}


def main() -> int:
    private_df = pd.read_csv(ISRAEL_DIR_PATH / "funds_private_annual.csv")
    product_year_df = pd.read_csv(ISRAEL_DIR_PATH / "tgi_comparison_calendar_years.csv", index_col=0)
    product_month_df = pd.read_csv(ISRAEL_DIR_PATH / "tgi_comparison_monthly.csv", index_col=0, parse_dates=True)

    # A. Private funds, 2020-2025.
    year_column_list = [f"y{y}" for y in YEAR_LIST]
    row_list = []
    for _, row in private_df.iterrows():
        year_ser = pd.Series({y: row[f"y{y}"] for y in YEAR_LIST}, dtype=float) / 100.0
        if year_ser.notna().all():
            kind_str = "benchmark" if row["strategy"] == "benchmark" else "fund"
            row_list.append({"series": row["fund"], "kind": kind_str, "note": row["note"] if pd.notna(row["note"]) else "", **annual_stats(year_ser)})
    for label_str, column_str in PRODUCT_COLUMN_DICT.items():
        year_ser = pd.Series({y: product_year_df.loc[str(y), column_str] for y in YEAR_LIST}, dtype=float)
        row_list.append({"series": label_str, "kind": "product", "note": "backtest", **annual_stats(year_ser)})
    private_stats_df = pd.DataFrame(row_list).set_index("series").sort_values("cagr", ascending=False)

    # B. Mutual funds, monthly, common window.
    mutual_df = parse_mutual_monthly_df().loc[MUTUAL_WINDOW_TUPLE[0]:MUTUAL_WINDOW_TUPLE[1]]
    product_window_df = product_month_df.loc[MUTUAL_WINDOW_TUPLE[0]:MUTUAL_WINDOW_TUPLE[1]]
    tgi_ser = product_window_df["TGI (all Israeli hedge funds)"]
    agg_ser = product_window_df["AGG ILS net"]
    mutual_row_list = []
    for name_str, ser in list(mutual_df.items()) + [(k, product_window_df[v]) for k, v in PRODUCT_COLUMN_DICT.items()] + [("TGI index", tgi_ser)]:
        ser = ser.dropna()
        mutual_row_list.append({"series": name_str, "kind": "product" if name_str in PRODUCT_COLUMN_DICT else ("index" if name_str == "TGI index" else "fund"),
                                "months": len(ser), **monthly_stats(ser), "corr_tgi": float(ser.corr(tgi_ser)), "corr_agg": float(ser.corr(agg_ser))})
    mutual_stats_df = pd.DataFrame(mutual_row_list).set_index("series").sort_values("cagr", ascending=False)

    # C. Persistence across private funds: rank correlation of consecutive years.
    fund_only_df = private_df[private_df["strategy"] != "benchmark"].set_index("fund")[year_column_list] / 100.0
    persistence_row_list = []
    for year_int in YEAR_LIST[:-1]:
        pair_df = fund_only_df[[f"y{year_int}", f"y{year_int + 1}"]].dropna()
        top_list = pair_df[f"y{year_int}"].nlargest(5).index
        persistence_row_list.append({"pair": f"{year_int}->{year_int + 1}", "funds": len(pair_df),
                                     "rank_corr": float(pair_df.iloc[:, 0].rank().corr(pair_df.iloc[:, 1].rank())),
                                     "top5_next_year_mean": float(pair_df.loc[top_list].iloc[:, 1].mean()),
                                     "all_next_year_mean": float(pair_df.iloc[:, 1].mean())})
    persistence_df = pd.DataFrame(persistence_row_list)
    fund_mean_ser = fund_only_df.mean()
    ta125_ser = private_df.set_index("fund").loc["TA-125 index", year_column_list] / 100.0

    private_stats_df.to_csv(ISRAEL_DIR_PATH / "fund_level_private_2020_2025.csv", float_format="%.6g")
    mutual_stats_df.to_csv(ISRAEL_DIR_PATH / "fund_level_mutual_monthly.csv", float_format="%.6g")
    persistence_df.to_csv(ISRAEL_DIR_PATH / "fund_level_persistence.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print(private_stats_df.round(3).to_string())
    print()
    print(mutual_stats_df.round(3).to_string())
    print()
    print(persistence_df.round(3).to_string(index=False))
    print()
    print("fund average by year:", (fund_mean_ser * 100).round(1).to_dict())
    print("TA-125 by year:", (ta125_ser.astype(float) * 100).round(1).to_dict())
    print("corr(fund average, TA-125) across years:", round(float(np.corrcoef(fund_mean_ser.astype(float), ta125_ser.astype(float))[0, 1]), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
