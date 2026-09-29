"""Growth shelf: 2/20 year by year, and fee income if live returns come in 30% below the backtest (2026-09-24).

Fees as in growth_dossier.fee_path (2% a year daily; 20% above the high-water mark, accrued daily, paid each 31
December; no hurdle). The haircut case scales every daily gross return by 0.7 (the central live-vs-backtest haircut
used in the fund-seriousness assessment), which cuts CAGR by a bit more than 30% and leaves drawdown shape unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_dossier as gd  # noqa: E402
from growth_dossier import common  # noqa: E402

HAIRCUT_FLOAT = 0.70


def main() -> int:
    books_dict = gd.candidate_dict()
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    year_frame_list, haircut_row_list = [], []
    for name_str, (weight_dict, policy_str, _) in books_dict.items():
        exact_ser = common.book_return_ser(sleeve_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        _, year_df = gd.fee_path(exact_ser, bench_df["TBILL"], False)
        year_frame_list.append(year_df.assign(book=name_str))
        for label_str, scale_float in (("backtest", 1.0), ("haircut_30pct", HAIRCUT_FLOAT)):
            scaled_ser = exact_ser * scale_float
            fee_dict = gd.fee_stats(scaled_ser, bench_df["TBILL"], False)
            haircut_row_list.append({"book": name_str, "case": label_str,
                                     "gross_cagr": float((1 + scaled_ser).prod() ** (252 / len(scaled_ser)) - 1), **fee_dict})
    year_df = pd.concat(year_frame_list, ignore_index=True)
    haircut_df = pd.DataFrame(haircut_row_list)
    year_df.to_csv(gd.OUT_DIR_PATH / "fee_years.csv", index=False, float_format="%.6g")
    haircut_df.to_csv(gd.OUT_DIR_PATH / "fee_haircut.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    show_df = year_df.assign(fee_pct_of_start=(year_df["management_fee"] + year_df["performance_fee"]) / year_df["start_nav"])
    print((show_df.pivot_table(index="year", columns="book", values="return_after_cost") * 100).round(1).to_string())
    print((show_df.pivot_table(index="year", columns="book", values="investor_return") * 100).round(1).to_string())
    print((show_df.pivot_table(index="year", columns="book", values="fee_pct_of_start") * 100).round(2).to_string())
    print(haircut_df[["book", "case", "gross_cagr", "investor_net_cagr", "manager_fee_pct_of_aum_per_year", "years_without_performance_fee",
                      "investor_net_maxdd"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
