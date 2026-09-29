"""Growth shelf: capacity by execution route, and what size does to returns and to fee income (2026-09-24).

PRE-DECLARED before running. Same candidates as growth_dossier.py plus ladder_4 and menu Aggressive for reference; same
fills window (last three years), cross-pod aggregation (orders in one ticker on one day added) and liquidity lookbacks as
the defensive capacity chapter (defensive_capacity_20260924).

New for growth: single stocks. A stock's auction has no market maker pricing it off a basket, so the house single-stock
auction limits (alpha/engine/capacity_analysis.py, capacity_v2_1) apply as written:
  MOO (today)   stock order <= 0.05% (P95) and 0.10% (P99) of 20-day median dollar volume;
                cost = lambda x sqrt(order / volume / 1%), lambda 66.4 bps for Nasdaq-100 names, 40 bps for the rest
  MOC (planned) 0.25% / 0.50%; lambda 8.2 bps
  worked        monthly pods (NDX, MOSAIC, all ETF orders) worked over up to 5 days at <= 10% of volume a day,
                cost = sigma x sqrt(daily participation) (Y = 1), gate P95 <= 5% and max <= 20% of a day's volume;
                the short-hold mean-reversion pods (DV2, HPI) cannot wait and stay in the close auction (MOC limits)
  worked+blocks as worked, with BTAL / UUP / DBC done as blocks against their underlying at 15 bps
ETF orders in the MOO and MOC routes: one day at sigma x sqrt, same 5% / 20% gate (the defensive treatment; ETF market
makers anchor the auction to fair value, so the stock auction limits do not apply to them).
Recommended AUM = largest grid size at which every gate holds and the extra cost <= 25% of the book's return over T-bills.
Returns at size: that yearly cost is taken off the 2012-2026 daily returns evenly, then 2/20 fees (HWM, accrued daily,
paid yearly) are run, giving the investor's net CAGR and the manager's income at that size.
Wall: owning 10% of BTAL / UUP / DBC (from growth_dossier.py).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_dossier as gd  # noqa: E402
from growth_dossier import common, load_price_timeseries  # noqa: E402

WINDOW_START_TS = pd.Timestamp("2023-08-21")
ETF_POD_SET = {"taa_btal_1n_tqqq", "taa_btal_tqqq", "infl_compass"}
URGENT_STOCK_POD_SET = {"dv2", "hpi_vote"}
NASDAQ_POD_SET = {"ndx_vxn", "ndx_atr"}
MOO_DICT = {"soft": 0.0005, "hard": 0.0010, "lambda_nasdaq_bps": 66.4, "lambda_other_bps": 40.0}
MOC_DICT = {"soft": 0.0025, "hard": 0.0050, "lambda_nasdaq_bps": 8.2, "lambda_other_bps": 8.2}
ROUTE_LIST = ["MOO", "MOC", "worked", "worked+blocks"]
AUM_GRID_TUPLE = (2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8)
REVENUE_AUM_TUPLE = (1e7, 2.5e7, 5e7, 1e8, 2.5e8)


def route_cost_and_gates(order_df: pd.DataFrame, route_str: str, aum_float: float) -> tuple[float, dict]:
    """Yearly-sum cost in dollars (before dividing by AUM and years) and gate results for one route at one size."""
    order_dollar_arr = order_df["book_fraction_float"].to_numpy() * aum_float
    is_etf_arr = order_df["is_etf"].to_numpy()
    is_urgent_arr = order_df["is_urgent"].to_numpy()
    is_thin_arr = order_df["asset_str"].isin(gd.THIN_SET).to_numpy()
    lambda_nasdaq_arr = order_df["is_nasdaq"].to_numpy()
    p20_arr = order_dollar_arr / order_df["adv20"].to_numpy()
    p60_arr = order_dollar_arr / order_df["adv60"].to_numpy()
    sigma_arr = order_df["sigma"].to_numpy()
    cost_arr = np.zeros_like(order_dollar_arr)
    gate_dict = {}

    def auction(mask_arr: np.ndarray, param_dict: dict, label_str: str) -> None:
        if not mask_arr.any():
            return
        lambda_arr = np.where(lambda_nasdaq_arr[mask_arr], param_dict["lambda_nasdaq_bps"], param_dict["lambda_other_bps"]) / 1e4
        cost_arr[mask_arr] = order_dollar_arr[mask_arr] * lambda_arr * np.sqrt(p20_arr[mask_arr] / 0.01)
        gate_dict[f"{label_str}_p95"] = float(np.percentile(p20_arr[mask_arr], 95))
        gate_dict[f"{label_str}_p99"] = float(np.percentile(p20_arr[mask_arr], 99))
        gate_dict[f"{label_str}_ok"] = gate_dict[f"{label_str}_p95"] <= param_dict["soft"] and gate_dict[f"{label_str}_p99"] <= param_dict["hard"]
        gate_dict[f"{label_str}_worst"] = order_df["asset_str"].to_numpy()[mask_arr][int(np.argmax(p20_arr[mask_arr]))]

    def worked(mask_arr: np.ndarray, max_day_float: float, label_str: str) -> None:
        if not mask_arr.any():
            return
        day_arr = np.clip(np.ceil(p60_arr[mask_arr] / 0.10), 1.0, max_day_float)
        daily_participation_arr = p60_arr[mask_arr] / day_arr
        cost_arr[mask_arr] = order_dollar_arr[mask_arr] * sigma_arr[mask_arr] * np.sqrt(daily_participation_arr)
        gate_dict[f"{label_str}_p95"] = float(np.percentile(daily_participation_arr, 95))
        gate_dict[f"{label_str}_max"] = float(daily_participation_arr.max())
        gate_dict[f"{label_str}_ok"] = gate_dict[f"{label_str}_p95"] <= 0.05 and gate_dict[f"{label_str}_max"] <= 0.20
        gate_dict[f"{label_str}_worst"] = order_df["asset_str"].to_numpy()[mask_arr][int(np.argmax(daily_participation_arr))]

    if route_str in ("MOO", "MOC"):
        auction(~is_etf_arr, MOO_DICT if route_str == "MOO" else MOC_DICT, "stock_auction")
        worked(is_etf_arr, 1.0, "etf_one_day")
    else:
        auction(~is_etf_arr & is_urgent_arr, MOC_DICT, "urgent_stock_moc")
        work_mask_arr = ~is_urgent_arr if route_str == "worked" else (~is_urgent_arr & ~is_thin_arr)
        worked(work_mask_arr, 5.0, "worked")
        if route_str == "worked+blocks":
            block_mask_arr = is_thin_arr
            cost_arr[block_mask_arr] = order_dollar_arr[block_mask_arr] * gd.BLOCK_COST_FLOAT
    return float(cost_arr.sum()), gate_dict


def main() -> int:
    books_dict = gd.candidate_dict()
    book_name_list = [n for n, v in books_dict.items() if v[2] == "candidate"] + ["ladder_4 (drift)", "menu Aggressive"]
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    path_by_alias_dict = common.load_sleeve_path_dict()
    alias_set = {a for n in book_name_list for a in books_dict[n][0]}
    order_frame_list = []
    for alias_str in sorted(alias_set):
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        transaction_df = transaction_df[(transaction_df["date"] >= WINDOW_START_TS) & (transaction_df["date"] <= gd.END_TS)]
        prior_nav_ser = path_by_alias_dict[alias_str]["total_value_float"].shift(1)
        order_frame_list.append(transaction_df.assign(alias_str=alias_str, fraction_float=transaction_df["signed_notional_float"].abs().values
                                                      / prior_nav_ser.reindex(transaction_df["date"]).values)[["date", "alias_str", "asset_str", "fraction_float"]])
    order_df = pd.concat(order_frame_list, ignore_index=True)
    etf_ticker_set = set(order_df.loc[order_df["alias_str"].isin(ETF_POD_SET), "asset_str"])
    stock_ticker_set = set(order_df.loc[~order_df["alias_str"].isin(ETF_POD_SET), "asset_str"])
    assert not etf_ticker_set & stock_ticker_set, etf_ticker_set & stock_ticker_set
    nasdaq_ticker_set = set(order_df.loc[order_df["alias_str"].isin(NASDAQ_POD_SET), "asset_str"])
    ticker_list = sorted(order_df["asset_str"].unique())
    adv20_dict, adv60_dict, sigma_dict, missing_list = {}, {}, {}, []
    for ticker_str in ticker_list:
        try:
            price_df = load_price_timeseries(ticker_str, start_date_str="2023-03-01", end_date_str=gd.END_TS.strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001 - a missing ticker is reported as missing coverage below
            missing_list.append(ticker_str)
            continue
        price_df.index = pd.to_datetime(price_df.index).normalize()
        dollar_ser = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before the trade.
        adv20_dict[ticker_str] = dollar_ser.rolling(20, min_periods=10).median().shift(1)
        adv60_dict[ticker_str] = dollar_ser.rolling(60, min_periods=20).median().shift(1)
        sigma_dict[ticker_str] = price_df["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    adv20_df, adv60_df, sigma_df = pd.DataFrame(adv20_dict), pd.DataFrame(adv60_dict), pd.DataFrame(sigma_dict)
    years_float = (gd.END_TS - WINDOW_START_TS).days / 365.25
    print(f"{len(order_df)} orders, {len(ticker_list)} tickers, missing prices: {missing_list}")

    summary_row_list, curve_row_list, pod_row_list = [], [], []
    for name_str in book_name_list:
        weight_dict, policy_str, _ = books_dict[name_str]
        exact_ser, prior_weight_df = common.book_return_ser(sleeve_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, policy_str)
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(exact_ser.index[0]) - 1]
        metric = common.metric_dict(exact_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
        excess_return_float = metric["cagr_float"] - metric["tbill_cagr_float"]
        product_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        pod_weight_arr = prior_weight_df.reindex(product_order_df["date"]).to_numpy()
        column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
        product_order_df["book_fraction_float"] = product_order_df["fraction_float"].to_numpy() * np.array(
            [pod_weight_arr[i, column_index_dict[a]] for i, a in enumerate(product_order_df["alias_str"])])
        product_order_df["is_urgent"] = product_order_df["alias_str"].isin(URGENT_STOCK_POD_SET)
        # Urgent and patient orders in one ticker on one day are routed separately in the worked routes, so keep the flag.
        daily_df = product_order_df.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily_df["is_etf"] = daily_df["asset_str"].isin(etf_ticker_set)
        daily_df["is_nasdaq"] = daily_df["asset_str"].isin(nasdaq_ticker_set)
        for column_str, frame_df in (("adv20", adv20_df), ("adv60", adv60_df), ("sigma", sigma_df)):
            daily_df[column_str] = [frame_df.at[d, t] if (t in frame_df.columns and d in frame_df.index) else np.nan
                                    for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        coverage_float = float(daily_df[["adv20", "adv60", "sigma"]].notna().all(axis=1).mean())
        daily_df = daily_df.dropna(subset=["adv20", "adv60", "sigma"])
        summary_row = {"book": name_str, "excess_return": excess_return_float, "order_coverage": coverage_float,
                       "gross_cagr": metric["cagr_float"]}
        for route_str in ROUTE_LIST:
            recommended_float, first_fail_str = None, ""
            for aum_float in AUM_GRID_TUPLE:
                cost_dollar_float, gate_dict = route_cost_and_gates(daily_df, route_str, aum_float)
                cost_per_year_float = cost_dollar_float / aum_float / years_float
                gates_ok_bool = all(v for k, v in gate_dict.items() if k.endswith("_ok"))
                cost_ok_bool = cost_per_year_float <= 0.25 * excess_return_float
                curve_row_list.append({"book": name_str, "route": route_str, "aum": aum_float, "cost_per_year": cost_per_year_float,
                                       "gates_ok": gates_ok_bool, "cost_ok": cost_ok_bool, **gate_dict})
                if gates_ok_bool and cost_ok_bool and not first_fail_str:
                    recommended_float = aum_float
                elif not first_fail_str:
                    failed_list = [k.replace("_ok", "") + f" ({gate_dict[k.replace('_ok', '_worst')]})" for k, v in gate_dict.items() if k.endswith("_ok") and not v]
                    first_fail_str = f"${aum_float / 1e6:g}M: " + ("; ".join(failed_list) if failed_list else f"cost {cost_per_year_float:.1%}")
            summary_row.update({f"{route_str}_recommended": recommended_float, f"{route_str}_first_fail": first_fail_str})
        # Returns and fee income at size, per route.
        for route_str in ROUTE_LIST:
            for aum_float in REVENUE_AUM_TUPLE:
                cost_dollar_float, _ = route_cost_and_gates(daily_df, route_str, aum_float)
                cost_per_year_float = cost_dollar_float / aum_float / years_float
                fee_dict = gd.fee_stats(exact_ser, bench_df["TBILL"], False, cost_per_year_float)
                curve_row_list.append({"book": name_str, "route": route_str, "aum": aum_float, "cost_per_year": cost_per_year_float,
                                       "revenue_row": True, "gross_cagr_after_cost": float((1 + exact_ser - cost_per_year_float / 252).prod()
                                                                                         ** (252 / len(exact_ser)) - 1),
                                       "investor_net_cagr": fee_dict["investor_net_cagr"],
                                       "manager_income_per_year": fee_dict["manager_fee_pct_of_aum_per_year"] * aum_float})
        summary_row_list.append(summary_row)
        # Which pod carries the cost at $25M in the MOC route (diagnostic).
        for alias_str in weight_dict:
            pod_order_df = product_order_df[product_order_df["alias_str"] == alias_str].groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
            pod_order_df = pod_order_df.merge(daily_df.drop(columns=["book_fraction_float"]).drop_duplicates(["date", "asset_str", "is_urgent"]),
                                              on=["date", "asset_str", "is_urgent"], how="inner")
            for route_str in ("MOO", "MOC", "worked+blocks"):
                cost_dollar_float, _ = route_cost_and_gates(pod_order_df, route_str, 2.5e7)
                pod_row_list.append({"book": name_str, "pod": alias_str, "route": route_str, "cost_per_year_at_25m": cost_dollar_float / 2.5e7 / years_float})
        print(summary_row)

    summary_df = pd.DataFrame(summary_row_list).set_index("book")
    curve_df = pd.DataFrame(curve_row_list)
    pod_df = pd.DataFrame(pod_row_list)
    summary_df.to_csv(gd.OUT_DIR_PATH / "capacity_routes_summary.csv")
    curve_df.to_csv(gd.OUT_DIR_PATH / "capacity_routes_curve.csv", index=False, float_format="%.6g")
    pod_df.to_csv(gd.OUT_DIR_PATH / "capacity_pod_cost_at_25m.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 120)
    print(summary_df.T.to_string())
    revenue_df = curve_df[curve_df.get("revenue_row", pd.Series(False, index=curve_df.index)).fillna(False).astype(bool)]
    print(revenue_df.pivot_table(index=["route", "book"], columns="aum", values="cost_per_year").mul(100).round(2).to_string())
    print(revenue_df.pivot_table(index=["route", "book"], columns="aum", values="investor_net_cagr").mul(100).round(1).to_string())
    print(revenue_df.pivot_table(index=["route", "book"], columns="aum", values="manager_income_per_year").div(1e6).round(2).to_string())
    print(pod_df.pivot_table(index=["book", "pod"], columns="route", values="cost_per_year_at_25m").mul(100).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
