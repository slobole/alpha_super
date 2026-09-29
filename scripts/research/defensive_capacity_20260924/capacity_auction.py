"""Capacity of the defensive products when every order goes into an auction: MOO today, MOC planned (2026-09-24).

The owner trades market-on-open today and plans to move to market-on-close. This applies the house capacity model's
own auction assumptions (alpha/engine/capacity_analysis.py, capacity_v2_1) at the PRODUCT level and on TODAY's
liquidity, so the 2011-era BTAL volume never enters:
  order / daily volume limits      MOO soft 0.05%, hard 0.10%     MOC soft 0.25%, hard 0.50%
  impact at 1% of daily volume     MOO 40 bps central, 66.4 stress (the house ETF / large-cap proxy)
                                   MOC 8.2 bps central, 17.8 stress; cost = lambda x sqrt(order / volume / 1%)
  daily volume                     20-session median dollar volume before the trade (house lookback)
Trades, pods, window (last three years) and aggregation exactly as in capacity_study.py.
Recommended = P95 of order/volume under the soft limit, P99 under the hard limit, central cost <= 25% of the product's
return over T-bills. Outer = under 5% of orders above the hard limit and stress cost <= 50% of that return.
These limits were set for single-stock auctions; for ETFs, where market makers can hedge the basket, they are strict.
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
AUCTION_DICT = {"MOO": {"soft": 0.0005, "hard": 0.0010, "central": 40.0, "stress": 66.4},
                "MOC": {"soft": 0.0025, "hard": 0.0050, "central": 8.2, "stress": 17.8}}
AUM_GRID_TUPLE = (2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)


def main() -> int:
    products_dict = cs.product_weight_dict()
    alias_set = {a for w in products_dict.values() for a in w}
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:cs.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:cs.END_TS]
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
        # *** CRITICAL*** shift(1): only volume known before the auction.
        adv_dict[ticker_str] = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan).rolling(ADV_LOOKBACK_INT, min_periods=10).median().shift(1)
    adv_df = pd.DataFrame(adv_dict)
    years_float = (cs.END_TS - WINDOW_START_TS).days / 365.25

    row_list, summary_row_list = [], []
    for product_str, weight_dict in products_dict.items():
        book_ser, prior_weight_df = common.book_return_ser(sleeve_df.loc["2012-10-02":, list(weight_dict)], weight_dict, "annual")
        exact_metric = common.metric_dict(book_ser, bench_df["SPXTR"], bench_df["TBILL"], sleeve_df.index[sleeve_df.index.get_loc(book_ser.index[0]) - 1])
        excess_return_float = exact_metric["cagr_float"] - exact_metric["tbill_cagr_float"]
        product_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        pod_weight_arr = prior_weight_df.reindex(product_order_df["date"]).to_numpy()
        column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
        product_order_df["book_fraction_float"] = product_order_df["fraction_float"].to_numpy() * np.array(
            [pod_weight_arr[i, column_index_dict[a]] for i, a in enumerate(product_order_df["alias_str"])])
        daily_df = product_order_df.groupby(["date", "asset_str"], as_index=False)["book_fraction_float"].sum()
        daily_df["adv_float"] = [adv_df.at[d, t] if d in adv_df.index else np.nan for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        daily_df = daily_df.dropna(subset=["adv_float"])
        summary_row = {"product": product_str, "excess_return_2012_26": excess_return_float}
        for (auction_str, param_dict), exclude_thin_bool in [(item, flag) for flag in (False, True) for item in AUCTION_DICT.items()]:
            # Hybrid: the three thin wrappers leave the auction (worked or block orders, see capacity_study.py).
            mode_df = daily_df[~daily_df["asset_str"].isin(cs.FUTURES_WRAPPER_SET)] if exclude_thin_bool else daily_df
            auction_str = f"{auction_str}_ex_thin" if exclude_thin_bool else auction_str
            recommended_float, outer_float = None, None
            for aum_float in AUM_GRID_TUPLE + ((5e8, 1e9) if exclude_thin_bool else ()):
                order_dollar_arr = mode_df["book_fraction_float"].to_numpy() * aum_float
                participation_arr = order_dollar_arr / mode_df["adv_float"].to_numpy()
                central_drag_float = float(np.sum(order_dollar_arr * param_dict["central"] / 1e4 * np.sqrt(participation_arr / 0.01))) / aum_float / years_float
                stress_drag_float = central_drag_float * param_dict["stress"] / param_dict["central"]
                p95_float, p99_float = float(np.percentile(participation_arr, 95)), float(np.percentile(participation_arr, 99))
                hard_share_float = float((participation_arr > param_dict["hard"]).mean())
                worst_ticker_str = mode_df["asset_str"].iloc[int(np.argmax(participation_arr))]
                row_list.append({"product": product_str, "auction": auction_str, "aum": aum_float, "p95": p95_float, "p99": p99_float,
                                 "hard_breach_share": hard_share_float, "central_drag": central_drag_float, "stress_drag": stress_drag_float,
                                 "worst_ticker": worst_ticker_str})
                if p95_float <= param_dict["soft"] and p99_float <= param_dict["hard"] and central_drag_float <= 0.25 * excess_return_float:
                    recommended_float = aum_float
                if hard_share_float <= 0.05 and stress_drag_float <= 0.50 * excess_return_float:
                    outer_float = aum_float
            summary_row.update({f"{auction_str}_recommended": recommended_float, f"{auction_str}_outer": outer_float})
            # Tickers that breach the hard limit at $1M, most often first.
            at_1m_ser = mode_df.assign(p=mode_df["book_fraction_float"] * 1e6 / mode_df["adv_float"])
            breach_ser = at_1m_ser[at_1m_ser["p"] > param_dict["hard"]]["asset_str"].value_counts()
            summary_row[f"{auction_str}_hard_breaches_at_1m"] = ", ".join(f"{t} {n}" for t, n in breach_ser.head(5).items())
        summary_row_list.append(summary_row)
        print(summary_row)
    curve_df = pd.DataFrame(row_list)
    summary_df = pd.DataFrame(summary_row_list)
    curve_df.to_csv(cs.OUT_DIR_PATH / "auction_capacity_curve.csv", index=False, float_format="%.6g")
    summary_df.to_csv(cs.OUT_DIR_PATH / "auction_capacity_summary.csv", index=False)
    pd.set_option("display.width", 250)
    show_df = curve_df[curve_df["aum"].isin([1e5, 1e6, 5e6, 2.5e7])].copy()
    print(show_df.round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
