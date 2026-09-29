"""DV2 liquidity variants: the evaluation rule, written BEFORE any variant result was read (2026-09-25).

Book tested ("G3 + MR"): TAA 3x rank 32% / NDX (VXN) 32% / DV2 variant 18% / HPI 18%, annual reset, i.e. ladder_3's
weights with G3's annual reset. Reference: G3 = TAA 3x rank 50% / NDX 50%, annual reset.

A DV2 variant PASSES when its book:
  1. meets the owner's growth rules: Sharpe >= 1.35 on 2012-10-02 -> 2026-08-19, each half >= 1.20, worst drawdown
     incl. the 2008 proxy no deeper than -20%, and Sharpe >= 1.25 with +5 bps per side on every traded dollar;
  2. beats G3 on Calmar (CAGR / worst drawdown incl. 2008) with the same +5 bps cost stress applied to both books;
  3. has recommended capacity >= $25M in the route where mean-reversion orders go to the same-day close auction at the
     house MOC limits and the monthly pods are worked, BTAL in blocks (growth_capacity.py 'worked+blocks').
If both variants pass, the higher stressed Calmar is preferred. If none passes, the growth product stays G3.
Reported but outside the rule: the book with the WIRED DV2, DV2 alone, and today's all-MOO route.
Caveat carried from the timing study: the close-auction route assumes a same-day MOC version of the MR pods keeps
their edge (house timing analyzer: roughly yes, biased label) - not tested here; these runs fill at the next open.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_capacity as gc  # noqa: E402
import growth_dossier as gd  # noqa: E402
import growth_study as gs  # noqa: E402
from growth_dossier import common, load_price_timeseries  # noqa: E402
from dv2_liquidity_variants import OUT_DIR_PATH as VARIANT_DIR_PATH  # noqa: E402

sys.path.insert(0, str(gd.REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import evaluation  # noqa: E402

VARIANT_LIST = ["dv2_check", "dv2_turnover_rank", "dv2_liq_floor"]
MR_WEIGHT_DICT = {"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "hpi_vote": 0.18}
G3_WEIGHT_DICT = {"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5}
CAPACITY_WINDOW_START_TS = pd.Timestamp("2023-08-21")
AUM_GRID_TUPLE = (1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)


def variant_return_ser(alias_str: str, index: pd.DatetimeIndex) -> pd.Series:
    path_df = pd.read_csv(VARIANT_DIR_PATH / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)
    nav_ser = path_df["total_value_float"].astype(float)
    invested_ser = path_df["portfolio_value_float"].abs() > 1e-9
    base_position_int = max(nav_ser.index.get_loc(invested_ser[invested_ser].index[0]) - 1, 0)
    return nav_ser.iloc[base_position_int:].pct_change(fill_method=None).reindex(index)


def stats(ser: pd.Series) -> tuple[float, float, float]:
    ser = ser.dropna()
    nav_ser = (1.0 + ser).cumprod()
    years_float = (ser.index[-1] - ser.index[0]).days / 365.25
    return (float(nav_ser.iloc[-1] ** (1.0 / years_float) - 1.0), float(ser.mean() / ser.std() * np.sqrt(252.0)),
            float((nav_ser / nav_ser.cummax() - 1.0).min()))


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    path_by_alias_dict = common.load_sleeve_path_dict()
    transaction_by_alias_dict, nav_by_alias_dict = {}, {}
    for alias_str in list(MR_WEIGHT_DICT) + ["dv2"]:
        transaction_by_alias_dict[alias_str] = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        nav_by_alias_dict[alias_str] = path_by_alias_dict[alias_str]["total_value_float"].loc[:gd.END_TS]
    for alias_str in VARIANT_LIST:
        sleeve_df[alias_str] = variant_return_ser(alias_str, sleeve_df.index)
        transaction_by_alias_dict[alias_str] = pd.read_csv(VARIANT_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        nav_by_alias_dict[alias_str] = pd.read_csv(VARIANT_DIR_PATH / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]

    # Reproduction check: the unchanged strategy through this runner vs the fund-menu dv2 source.
    diff_ser = (sleeve_df["dv2_check"] - sleeve_df["dv2"]).abs()
    print(f"dv2_check vs fund-menu dv2: max |daily return diff| = {diff_ser.max():.2e}; "
          f"transactions {len(transaction_by_alias_dict['dv2_check'])} vs {len(transaction_by_alias_dict['dv2'])}")

    stressed_df = sleeve_df.copy()
    for alias_str, transaction_df in transaction_by_alias_dict.items():
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_df, nav_by_alias_dict[alias_str], gs.EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]
    long_df = sleeve_df.copy()
    long_df.loc[long_df.index < gd.CUT_TS, "taa_btal_tqqq"] = sleeve_df.loc[sleeve_df.index < gd.CUT_TS, "taa_1n_qld"]
    exact_index = sleeve_df.loc[gd.CUT_TS:].index
    h1_index, h2_index = exact_index[: len(exact_index) // 2], exact_index[len(exact_index) // 2:]

    # DV2 alone.
    alone_row_list = []
    for alias_str in ["dv2"] + VARIANT_LIST:
        full_tuple, exact_tuple = stats(sleeve_df[alias_str]), stats(sleeve_df.loc[gd.CUT_TS:, alias_str])
        stress_tuple = stats(stressed_df.loc[gd.CUT_TS:, alias_str])
        buys_df = transaction_by_alias_dict[alias_str]
        buys_df = buys_df[(buys_df["date"] >= gd.CUT_TS) & (buys_df["amount_float"] > 0)]
        alone_row_list.append({"dv2": alias_str, "cagr_2000_26": full_tuple[0], "sharpe_2000_26": full_tuple[1], "maxdd_2000_26": full_tuple[2],
                               "cagr_2012_26": exact_tuple[0], "sharpe_2012_26": exact_tuple[1], "maxdd_2012_26": exact_tuple[2],
                               "cagr_2012_26_cost_stress": stress_tuple[0], "sharpe_2012_26_cost_stress": stress_tuple[1],
                               "entries_per_year": len(buys_df) / (len(exact_index) / 252.0)})
    alone_df = pd.DataFrame(alone_row_list).set_index("dv2")

    # Books.
    book_def_dict = {"G3": G3_WEIGHT_DICT}
    for alias_str in ["dv2"] + VARIANT_LIST:
        book_def_dict[f"G3 + MR ({alias_str})"] = {**MR_WEIGHT_DICT, alias_str: 0.18}
    book_row_list, prior_weight_by_book_dict, excess_by_book_dict = [], {}, {}
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    for book_str, weight_dict in book_def_dict.items():
        exact_ser, prior_weight_df = common.book_return_ser(sleeve_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, "annual")
        long_ser = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(weight_dict)], weight_dict, "annual")[0]
        stress_ser = common.book_return_ser(stressed_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, "annual")[0]
        prior_weight_by_book_dict[book_str] = prior_weight_df
        cagr_float, sharpe_float, dd_float = stats(exact_ser)
        long_dd_float = stats(long_ser)[2]
        stress_cagr_float, stress_sharpe_float, stress_dd_float = stats(stress_ser)
        worst_dd_float = min(dd_float, long_dd_float)
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(exact_ser.index[0]) - 1]
        excess_by_book_dict[book_str] = cagr_float - common.metric_dict(exact_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)["tbill_cagr_float"]
        book_row_list.append({"book": book_str, "cagr": cagr_float, "sharpe": sharpe_float, "sharpe_h1": stats(exact_ser.loc[h1_index])[1],
                              "sharpe_h2": stats(exact_ser.loc[h2_index])[1], "maxdd": dd_float, "maxdd_incl_2008": worst_dd_float,
                              "calmar": cagr_float / abs(worst_dd_float), "cagr_stress": stress_cagr_float, "sharpe_stress": stress_sharpe_float,
                              "calmar_stress": stress_cagr_float / abs(min(stress_dd_float, long_dd_float))})
    book_df = pd.DataFrame(book_row_list).set_index("book")

    # Capacity, last three years of fills.
    order_frame_list = []
    for alias_str, transaction_df in transaction_by_alias_dict.items():
        window_df = transaction_df[(transaction_df["date"] >= CAPACITY_WINDOW_START_TS) & (transaction_df["date"] <= gd.END_TS)]
        prior_nav_ser = nav_by_alias_dict[alias_str].shift(1)
        order_frame_list.append(window_df.assign(alias_str=alias_str, fraction_float=window_df["signed_notional_float"].abs().values
                                                 / prior_nav_ser.reindex(window_df["date"]).values)[["date", "alias_str", "asset_str", "fraction_float"]])
    order_df = pd.concat(order_frame_list, ignore_index=True)
    etf_ticker_set = set(order_df.loc[order_df["alias_str"] == "taa_btal_tqqq", "asset_str"])
    nasdaq_ticker_set = set(order_df.loc[order_df["alias_str"] == "ndx_vxn", "asset_str"])
    adv20_dict, adv60_dict, sigma_dict = {}, {}, {}
    for ticker_str in sorted(order_df["asset_str"].unique()):
        price_df = load_price_timeseries(ticker_str, start_date_str="2023-03-01", end_date_str=gd.END_TS.strftime("%Y-%m-%d"))
        price_df.index = pd.to_datetime(price_df.index).normalize()
        dollar_ser = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before the trade.
        adv20_dict[ticker_str] = dollar_ser.rolling(20, min_periods=10).median().shift(1)
        adv60_dict[ticker_str] = dollar_ser.rolling(60, min_periods=20).median().shift(1)
        sigma_dict[ticker_str] = price_df["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liquidity_dict = {"adv20": pd.DataFrame(adv20_dict), "adv60": pd.DataFrame(adv60_dict), "sigma": pd.DataFrame(sigma_dict)}
    years_float = (gd.END_TS - CAPACITY_WINDOW_START_TS).days / 365.25

    def capacity_row(label_str: str, weight_dict: dict, prior_weight_df: pd.DataFrame | None, excess_float: float) -> dict:
        book_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        if prior_weight_df is None:
            book_order_df["book_fraction_float"] = book_order_df["fraction_float"]
        else:
            weight_arr = prior_weight_df.reindex(book_order_df["date"]).to_numpy()
            column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
            book_order_df["book_fraction_float"] = book_order_df["fraction_float"].to_numpy() * np.array(
                [weight_arr[i, column_index_dict[a]] for i, a in enumerate(book_order_df["alias_str"])])
        book_order_df["is_urgent"] = book_order_df["alias_str"].str.startswith("dv2") | (book_order_df["alias_str"] == "hpi_vote")
        daily_df = book_order_df.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily_df["is_etf"] = daily_df["asset_str"].isin(etf_ticker_set)
        daily_df["is_nasdaq"] = daily_df["asset_str"].isin(nasdaq_ticker_set)
        for column_str, frame_df in liquidity_dict.items():
            daily_df[column_str] = [frame_df.at[d, t] if (t in frame_df.columns and d in frame_df.index) else np.nan
                                    for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        daily_df = daily_df.dropna(subset=list(liquidity_dict))
        row = {"book": label_str}
        for route_str in ("MOO", "MOC", "worked+blocks"):
            recommended_float, fail_str = None, ""
            for aum_float in AUM_GRID_TUPLE:
                cost_dollar_float, gate_dict = gc.route_cost_and_gates(daily_df, route_str, aum_float)
                cost_per_year_float = cost_dollar_float / aum_float / years_float
                ok_bool = all(v for k, v in gate_dict.items() if k.endswith("_ok")) and cost_per_year_float <= 0.25 * excess_float
                if aum_float == 2.5e7:
                    row[f"{route_str}_cost_at_25m"] = cost_per_year_float
                if ok_bool and not fail_str:
                    recommended_float = aum_float
                elif not fail_str:
                    failed_list = [k.replace("_ok", "") + f" ({gate_dict[k.replace('_ok', '_worst')]})" for k, v in gate_dict.items() if k.endswith("_ok") and not v]
                    fail_str = f"${aum_float / 1e6:g}M: " + ("; ".join(failed_list) if failed_list else f"cost {cost_per_year_float:.1%}")
            row.update({f"{route_str}_recommended": recommended_float, f"{route_str}_first_fail": fail_str})
        return row

    capacity_row_list = [capacity_row(b, w, prior_weight_by_book_dict[b], excess_by_book_dict[b]) for b, w in book_def_dict.items()]
    for alias_str in ["dv2"] + VARIANT_LIST:
        capacity_row_list.append(capacity_row(f"{alias_str} alone", {alias_str: 1.0}, None, float(alone_df.at[alias_str, "cagr_2012_26"])))
    capacity_df = pd.DataFrame(capacity_row_list).set_index("book")

    # The pre-declared rule.
    g3_calmar_stress_float = float(book_df.at["G3", "calmar_stress"])
    verdict_row_list = []
    for alias_str in ["dv2"] + VARIANT_LIST:
        book_str = f"G3 + MR ({alias_str})"
        row_ser = book_df.loc[book_str]
        rules_bool = (row_ser["sharpe"] >= 1.35 and min(row_ser["sharpe_h1"], row_ser["sharpe_h2"]) >= 1.20
                      and row_ser["maxdd_incl_2008"] >= -0.20 and row_ser["sharpe_stress"] >= 1.25)
        calmar_bool = row_ser["calmar_stress"] > g3_calmar_stress_float
        capacity_float = capacity_df.at[book_str, "worked+blocks_recommended"]
        capacity_bool = capacity_float is not None and pd.notna(capacity_float) and capacity_float >= 2.5e7
        verdict_row_list.append({"book": book_str, "growth_rules": rules_bool, "beats_G3_stressed_calmar": calmar_bool,
                                 "capacity_ge_25m": capacity_bool, "PASS": rules_bool and calmar_bool and capacity_bool})
    verdict_df = pd.DataFrame(verdict_row_list).set_index("book")

    for frame_df, file_str in ((alone_df, "dv2_liquidity_alone.csv"), (book_df, "dv2_liquidity_books.csv"),
                               (capacity_df, "dv2_liquidity_capacity.csv"), (verdict_df, "dv2_liquidity_verdict.csv")):
        frame_df.to_csv(gd.OUT_DIR_PATH / file_str, float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    print(alone_df.round(3).to_string())
    print(book_df.round(3).to_string())
    print(capacity_df.to_string())
    print(verdict_df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
