"""How big do the defensive products' market-on-open orders really get? (owner challenge, 2026-09-24)

The auction columns first published in the report applied the house capacity model's order caps (at most 0.05% of an
instrument's daily volume for 95% of orders, 0.10% for 99%). Those caps were calibrated on single-stock opening
auctions. An ETF's opening price is held near its fair value by market makers who can hedge in the underlying basket,
so the caps turn a $5K BTAL order into a "breach" and a product into a "$50K capacity" - an artifact, not economics.

This script replaces that with plain facts about order size, on the last three years of trades and today's volume:
for each product, per $1M of product size, the size of each day's order in each ETF relative to that ETF's typical
daily dollar volume (20-session median before the trade), split into the three thin ETFs (DBC, UUP, BTAL) and
everything else; then the product size at which the typical (median), the 95th-percentile and the largest order
reach 1%, 5% and 10% of daily volume. Orders scale linearly with product size.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_study as cs  # noqa: E402
from capacity_study import common, load_price_timeseries  # noqa: E402

WINDOW_START_TS = pd.Timestamp("2023-08-21")
ADV_LOOKBACK_INT = 20
THRESHOLD_TUPLE = (0.01, 0.05, 0.10)


def main() -> int:
    products_dict = cs.product_weight_dict()
    alias_set = {a for w in products_dict.values() for a in w}
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:cs.END_TS]
    path_by_alias_dict = common.load_sleeve_path_dict()
    order_frame_list = []
    for alias_str in sorted(alias_set):
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        transaction_df = transaction_df[(transaction_df["date"] >= WINDOW_START_TS) & (transaction_df["date"] <= cs.END_TS)]
        prior_nav_ser = path_by_alias_dict[alias_str]["total_value_float"].shift(1)
        transaction_df = transaction_df.assign(alias_str=alias_str, fraction_float=transaction_df["signed_notional_float"].abs().values
                                               / prior_nav_ser.reindex(transaction_df["date"]).values)
        order_frame_list.append(transaction_df[["date", "alias_str", "asset_str", "fraction_float"]])
    order_df = pd.concat(order_frame_list, ignore_index=True)
    adv_dict = {}
    for ticker_str in sorted(order_df["asset_str"].unique()):
        price_df = load_price_timeseries(ticker_str, start_date_str="2023-05-01", end_date_str=cs.END_TS.strftime("%Y-%m-%d"))
        price_df.index = pd.to_datetime(price_df.index).normalize()
        # *** CRITICAL*** shift(1): only volume known before the open.
        adv_dict[ticker_str] = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan).rolling(ADV_LOOKBACK_INT, min_periods=10).median().shift(1)
    adv_df = pd.DataFrame(adv_dict)

    row_list = []
    for product_str, weight_dict in products_dict.items():
        _, prior_weight_df = common.book_return_ser(sleeve_df.loc["2012-10-02":, list(weight_dict)], weight_dict, "annual")
        product_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        pod_weight_arr = prior_weight_df.reindex(product_order_df["date"]).to_numpy()
        column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
        product_order_df["book_fraction_float"] = product_order_df["fraction_float"].to_numpy() * np.array(
            [pod_weight_arr[i, column_index_dict[a]] for i, a in enumerate(product_order_df["alias_str"])])
        daily_df = product_order_df.groupby(["date", "asset_str"], as_index=False)["book_fraction_float"].sum()
        daily_df["adv_float"] = [adv_df.at[d, t] if d in adv_df.index else np.nan for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        daily_df = daily_df.dropna(subset=["adv_float"])
        daily_df["order_per_1m_usd"] = daily_df["book_fraction_float"] * 1e6
        daily_df["participation_per_1m"] = daily_df["order_per_1m_usd"] / daily_df["adv_float"]
        for group_str, group_df in (("thin three (DBC, UUP, BTAL)", daily_df[daily_df["asset_str"].isin(cs.FUTURES_WRAPPER_SET)]),
                                    ("everything else", daily_df[~daily_df["asset_str"].isin(cs.FUTURES_WRAPPER_SET)])):
            if group_df.empty:
                continue
            stat_dict = {"median": float(group_df["participation_per_1m"].median()), "p95": float(group_df["participation_per_1m"].quantile(0.95)),
                         "max": float(group_df["participation_per_1m"].max())}
            row = {"product": product_str, "group": group_str, "orders": len(group_df),
                   "largest_order_per_1m_usd": float(group_df["order_per_1m_usd"].max()),
                   "largest_order_ticker": group_df.loc[group_df["participation_per_1m"].idxmax(), "asset_str"]}
            for stat_str, per_1m_float in stat_dict.items():
                row[f"{stat_str}_pct_of_volume_at_1m"] = per_1m_float
                for threshold_float in THRESHOLD_TUPLE:
                    # Orders scale linearly with product size: size at which this order statistic reaches the threshold.
                    row[f"aum_{stat_str}_hits_{threshold_float:.0%}"] = threshold_float / per_1m_float * 1e6
            row_list.append(row)
    result_df = pd.DataFrame(row_list)
    result_df.to_csv(cs.OUT_DIR_PATH / "moo_order_sizes.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    show_df = result_df.copy()
    for column_str in [c for c in show_df.columns if c.startswith("aum_")]:
        show_df[column_str] = (show_df[column_str] / 1e6).round(1)
    print(show_df[["product", "group", "orders", "largest_order_per_1m_usd", "largest_order_ticker", "median_pct_of_volume_at_1m",
                   "p95_pct_of_volume_at_1m", "max_pct_of_volume_at_1m"]].round(5).to_string(index=False))
    print()
    print(show_df[["product", "group"] + [c for c in show_df.columns if c.startswith("aum_")]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
