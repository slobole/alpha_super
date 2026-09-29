"""Growth menu: every growth book worth keeping, on one page, with the 2008 proxy (owner request, 2026-09-25).

Books (fixed before running; weights from growth_dossier.py and portfolios/ladder_*.yaml):
  G3 TAA 3x + NDX 50/50 annual (live pair) | G4 TAA 3x 1/N + NDX + MOSAIC | G2 TAA 3x 1/N + MOSAIC
  ladder_4 and ladder_4_1n (drift, as in the YAMLs), each also with the liquidity-floor DV2 (dv2_liq_floor)
  ladder_3 (drift) | G3 + MR with the floor DV2 (TAA 32 / NDX 32 / DV2-floor 18 / HPI 18, annual)
  G1 TAA 3x 1/N + NDX ATR + DV2 (the search's rule winner) | G5 TAA 3x 1/N + NDX + InflC (InflC data caveat)
Windows: 2012-10-02 -> 2026-08-19 as measured; 2008-03-04 on with the 2008 proxy (the 2x no-BTAL TAA stands in for the
BTAL TAA sleeves before 2012-10-02 only; every other sleeve has its own history). Costs as in the engine runs
(commissions + slippage); the +5 bps per side column is a sensitivity only. Capacity: last three years of fills, house
limits, as in growth_capacity.py; plus, for the mean-reversion orders sent to the close auction, the share of dollars
above the house hard limit (0.5% of daily volume) at a given size - the part that would need trimming or splitting.
Turnover: traded dollars per year / average NAV, per pod; 1 bp per side costs turnover x 0.01% a year.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import growth_capacity as gc  # noqa: E402
import growth_dossier as gd  # noqa: E402
import growth_study as gs  # noqa: E402
from growth_dossier import common, load_price_timeseries  # noqa: E402
from dv2_liquidity_eval import variant_return_ser  # noqa: E402
from dv2_liquidity_variants import OUT_DIR_PATH as VARIANT_DIR_PATH  # noqa: E402

sys.path.insert(0, str(gd.REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import evaluation  # noqa: E402

T = 1.0 / 3.0
LADDER_4_DICT = {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}
LADDER_4_1N_DICT = {"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_1n_tqqq": 0.34}


def swap_floor(weight_dict: dict) -> dict:
    return {("dv2_liq_floor" if a == "dv2" else a): w for a, w in weight_dict.items()}


BOOK_DICT = {
    "G3 TAA 3x + NDX (live)": ({"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5}, "annual"),
    "G4 TAA 3x 1/N + NDX + MOSAIC": ({"taa_btal_1n_tqqq": T, "ndx_vxn": T, "mosaic": T}, "annual"),
    "G2 TAA 3x 1/N + MOSAIC": ({"taa_btal_1n_tqqq": 0.5, "mosaic": 0.5}, "annual"),
    "ladder_4": (LADDER_4_DICT, "none"),
    "ladder_4 + DV2 floor": (swap_floor(LADDER_4_DICT), "none"),
    "ladder_4_1n": (LADDER_4_1N_DICT, "none"),
    "ladder_4_1n + DV2 floor": (swap_floor(LADDER_4_1N_DICT), "none"),
    "ladder_3": ({"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2": 0.18, "hpi_vote": 0.18}, "none"),
    "G3 + MR (DV2 floor + HPI)": ({"taa_btal_tqqq": 0.32, "ndx_vxn": 0.32, "dv2_liq_floor": 0.18, "hpi_vote": 0.18}, "annual"),
    "G1 TAA 3x 1/N + NDX ATR + DV2": ({"taa_btal_1n_tqqq": T, "ndx_atr": T, "dv2": T}, "annual"),
    "G5 TAA 3x 1/N + NDX + InflC": ({"taa_btal_1n_tqqq": T, "ndx_vxn": T, "infl_compass": T}, "annual"),
}
WIRED_SET = {"taa_btal_tqqq", "taa_btal_1n_tqqq", "ndx_vxn", "ndx_atr", "dv2", "hpi_vote"}
ETF_POD_SET = {"taa_btal_tqqq", "taa_btal_1n_tqqq", "infl_compass"}
URGENT_POD_SET = {"dv2", "dv2_liq_floor", "hpi_vote"}
NASDAQ_POD_SET = {"ndx_vxn", "ndx_atr"}
STAND_IN_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld"}
CRISIS_KEY_DICT = {"gfc_2008": "May 2008", "covid_2020": "Feb 2020", "bear_2022": "Jan 2022", "tariffs_2025": "Feb 2025"}
CAPACITY_START_TS = pd.Timestamp("2023-08-21")
AUM_GRID_TUPLE = (1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)


def main() -> int:
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:gd.END_TS]
    sleeve_df["dv2_liq_floor"] = variant_return_ser("dv2_liq_floor", sleeve_df.index)
    alias_set = {a for w, _ in BOOK_DICT.values() for a in w}
    path_by_alias_dict = common.load_sleeve_path_dict()
    transaction_by_alias_dict, nav_by_alias_dict = {}, {}
    for alias_str in sorted(alias_set):
        source_path = VARIANT_DIR_PATH if alias_str == "dv2_liq_floor" else common.SOURCE_DIR_PATH
        transaction_by_alias_dict[alias_str] = pd.read_csv(source_path / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        nav_by_alias_dict[alias_str] = (pd.read_csv(VARIANT_DIR_PATH / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
                                        if alias_str == "dv2_liq_floor" else path_by_alias_dict[alias_str]["total_value_float"]).loc[:gd.END_TS]
    stressed_df, long_df = sleeve_df.copy(), sleeve_df.copy()
    for alias_str in sorted(alias_set):
        drag_ser = evaluation.extra_slippage_cost_ser(transaction_by_alias_dict[alias_str], nav_by_alias_dict[alias_str], gs.EXTRA_SLIPPAGE_PER_SIDE_FLOAT)
        live_mask = sleeve_df[alias_str].notna()
        stressed_df.loc[live_mask, alias_str] = sleeve_df.loc[live_mask, alias_str] - drag_ser.reindex(sleeve_df.index).fillna(0.0)[live_mask]
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < gd.CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < gd.CUT_TS, stand_in_str]
    episode_list = json.loads((gd.OUT_DIR_PATH / "dossier_episodes.json").read_text(encoding="utf-8"))
    crisis_window_dict = {k: next((e["start"], e["end"]) for e in episode_list if v in e["label"]) for k, v in CRISIS_KEY_DICT.items()}
    exact_index = sleeve_df.loc[gd.CUT_TS:].index
    years_float = len(exact_index) / 252.0

    # Turnover per pod: traded dollars / average NAV per year, measured window.
    turnover_dict = {}
    for alias_str in sorted(alias_set):
        window_df = transaction_by_alias_dict[alias_str]
        window_df = window_df[(window_df["date"] >= gd.CUT_TS) & (window_df["date"] <= gd.END_TS)]
        turnover_dict[alias_str] = float(window_df["signed_notional_float"].abs().sum() / nav_by_alias_dict[alias_str].loc[gd.CUT_TS:].mean() / years_float)

    row_list, prior_weight_by_book_dict = [], {}
    for book_str, (weight_dict, policy_str) in BOOK_DICT.items():
        exact_ser, prior_weight_df = common.book_return_ser(sleeve_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, policy_str)
        long_ser = common.book_return_ser(long_df.loc[gd.LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        stress_ser = common.book_return_ser(stressed_df.loc[gd.CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        prior_weight_by_book_dict[book_str] = prior_weight_df
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(exact_ser.index[0]) - 1]
        metric = common.metric_dict(exact_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
        long_base_ts = sleeve_df.index[sleeve_df.index.get_loc(long_ser.index[0]) - 1]
        long_metric = common.metric_dict(long_ser, bench_df["SPXTR"], bench_df["TBILL"], long_base_ts)
        stress_metric = common.metric_dict(stress_ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)
        worst_dd_float = min(metric["max_drawdown_float"], long_metric["max_drawdown_float"])
        trade_date_set = set()
        for alias_str in weight_dict:
            date_ser = transaction_by_alias_dict[alias_str]["date"]
            trade_date_set |= set(date_ser[(date_ser >= gd.CUT_TS) & (date_ser <= gd.END_TS)])
        half_int = len(exact_index) // 2
        row = {"book": book_str, "pods": len(weight_dict), "cagr": metric["cagr_float"], "vol": metric["volatility_float"],
               "sharpe": metric["sharpe_rf0_float"],
               "sharpe_h1": float(exact_ser.iloc[:half_int].mean() / exact_ser.iloc[:half_int].std() * np.sqrt(252)),
               "sharpe_h2": float(exact_ser.iloc[half_int:].mean() / exact_ser.iloc[half_int:].std() * np.sqrt(252)),
               "maxdd": metric["max_drawdown_float"], "maxdd_incl_2008": worst_dd_float,
               "calmar_incl_2008": metric["cagr_float"] / abs(worst_dd_float),
               "cagr_2008_26": long_metric["cagr_float"], "sharpe_2008_26": long_metric["sharpe_rf0_float"],
               "worst_year": metric["worst_year_float"],
               "cagr_cost_plus5": stress_metric["cagr_float"], "sharpe_cost_plus5": stress_metric["sharpe_rf0_float"],
               "trade_days_per_year": len(trade_date_set) / years_float,
               "wired_share": float(sum(w for a, w in weight_dict.items() if a in WIRED_SET)),
               "turnover_x_nav": float(sum(w * turnover_dict[a] for a, w in weight_dict.items())),
               "excess_return": metric["cagr_float"] - metric["tbill_cagr_float"]}
        row["cagr_per_1bp_side"] = row["turnover_x_nav"] * 0.0001
        row.update({k: common.window_return_float(long_ser, s, e) for k, (s, e) in crisis_window_dict.items()})
        row_list.append(row)
    menu_df = pd.DataFrame(row_list).set_index("book")

    # Capacity.
    order_frame_list = []
    for alias_str in sorted(alias_set):
        window_df = transaction_by_alias_dict[alias_str]
        window_df = window_df[(window_df["date"] >= CAPACITY_START_TS) & (window_df["date"] <= gd.END_TS)]
        prior_nav_ser = nav_by_alias_dict[alias_str].shift(1)
        order_frame_list.append(window_df.assign(alias_str=alias_str, fraction_float=window_df["signed_notional_float"].abs().values
                                                 / prior_nav_ser.reindex(window_df["date"]).values)[["date", "alias_str", "asset_str", "fraction_float"]])
    order_df = pd.concat(order_frame_list, ignore_index=True)
    etf_ticker_set = set(order_df.loc[order_df["alias_str"].isin(ETF_POD_SET), "asset_str"])
    nasdaq_ticker_set = set(order_df.loc[order_df["alias_str"].isin(NASDAQ_POD_SET), "asset_str"])
    liquidity_dict = {"adv20": {}, "adv60": {}, "sigma": {}}
    for ticker_str in sorted(order_df["asset_str"].unique()):
        price_df = load_price_timeseries(ticker_str, start_date_str="2023-03-01", end_date_str=gd.END_TS.strftime("%Y-%m-%d"))
        price_df.index = pd.to_datetime(price_df.index).normalize()
        dollar_ser = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before the trade.
        liquidity_dict["adv20"][ticker_str] = dollar_ser.rolling(20, min_periods=10).median().shift(1)
        liquidity_dict["adv60"][ticker_str] = dollar_ser.rolling(60, min_periods=20).median().shift(1)
        liquidity_dict["sigma"][ticker_str] = price_df["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
    liquidity_df_dict = {k: pd.DataFrame(v) for k, v in liquidity_dict.items()}
    capacity_years_float = (gd.END_TS - CAPACITY_START_TS).days / 365.25
    capacity_row_list = []
    for book_str, (weight_dict, _) in BOOK_DICT.items():
        prior_weight_df = prior_weight_by_book_dict[book_str]
        book_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        weight_arr = prior_weight_df.reindex(book_order_df["date"]).to_numpy()
        column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
        book_order_df["book_fraction_float"] = book_order_df["fraction_float"].to_numpy() * np.array(
            [weight_arr[i, column_index_dict[a]] for i, a in enumerate(book_order_df["alias_str"])])
        book_order_df["is_urgent"] = book_order_df["alias_str"].isin(URGENT_POD_SET)
        daily_df = book_order_df.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
        daily_df["is_etf"] = daily_df["asset_str"].isin(etf_ticker_set)
        daily_df["is_nasdaq"] = daily_df["asset_str"].isin(nasdaq_ticker_set)
        for column_str, frame_df in liquidity_df_dict.items():
            daily_df[column_str] = [frame_df.at[d, t] if (t in frame_df.columns and d in frame_df.index) else np.nan
                                    for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        daily_df = daily_df.dropna(subset=list(liquidity_df_dict))
        excess_float = float(menu_df.at[book_str, "excess_return"])
        row = {"book": book_str}
        for route_str in ("MOO", "worked+blocks"):
            recommended_float, fail_str = None, ""
            for aum_float in AUM_GRID_TUPLE:
                cost_dollar_float, gate_dict = gc.route_cost_and_gates(daily_df, route_str, aum_float)
                cost_float = cost_dollar_float / aum_float / capacity_years_float
                ok_bool = all(v for k, v in gate_dict.items() if k.endswith("_ok")) and cost_float <= 0.25 * excess_float
                if aum_float in (2.5e7, 1e8):
                    row[f"{route_str}_cost_at_{aum_float / 1e6:g}m"] = cost_float
                if ok_bool and not fail_str:
                    recommended_float = aum_float
                elif not fail_str:
                    failed_list = [k.replace("_ok", "") + f" ({gate_dict[k.replace('_ok', '_worst')]})" for k, v in gate_dict.items() if k.endswith("_ok") and not v]
                    fail_str = f"${aum_float / 1e6:g}M: " + ("; ".join(failed_list) if failed_list else f"cost {cost_float:.1%}")
            row.update({f"{route_str}_recommended": recommended_float, f"{route_str}_first_fail": fail_str})
        urgent_df = daily_df[daily_df["is_urgent"] & ~daily_df["is_etf"]]
        if len(urgent_df):
            for aum_float in (2.5e7, 5e7, 1e8):
                dollar_arr = urgent_df["book_fraction_float"].to_numpy() * aum_float
                above_arr = dollar_arr / urgent_df["adv20"].to_numpy() > gc.MOC_DICT["hard"]
                row[f"mr_dollars_above_moc_limit_at_{aum_float / 1e6:g}m"] = float(dollar_arr[above_arr].sum() / dollar_arr.sum())
                row[f"mr_orders_above_moc_limit_at_{aum_float / 1e6:g}m"] = int(above_arr.sum())
            row["mr_orders_total_3y"] = int(len(urgent_df))
        capacity_row_list.append(row)
    capacity_df = pd.DataFrame(capacity_row_list).set_index("book")

    menu_df.to_csv(gd.OUT_DIR_PATH / "growth_menu.csv", float_format="%.6g")
    capacity_df.to_csv(gd.OUT_DIR_PATH / "growth_menu_capacity.csv", float_format="%.6g")
    pd.Series(turnover_dict).to_csv(gd.OUT_DIR_PATH / "growth_menu_pod_turnover.csv", header=["turnover_x_nav_per_year"])
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    print(pd.Series(turnover_dict).round(1).to_string())
    print(menu_df.drop(columns=["excess_return"]).round(3).T.to_string())
    print(capacity_df.T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
