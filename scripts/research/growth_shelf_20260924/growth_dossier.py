"""Growth shelf dossier: crises, tails, correlation, 2008, capacity and 2/20 fee economics (2026-09-24).

PRE-DECLARED candidate set (fixed after the search, before any of the numbers below was computed):
  G1 rule winner          TAA 3x 1/N + NDX ATR + DV2 (equal capital)
  G2 best two pods        TAA 3x 1/N + MOSAIC
  G3 your live pair       TAA 3x + NDX
  G4 low-touch three      TAA 3x 1/N + NDX + MOSAIC
  references              ladder_4 (drift, as defined), menu Aggressive, menu Low-touch Growth
Windows: 2012-10-02 -> 2026-08-19, and from 2008-03-04 with BTAL TAA sleeves filled by the 2x no-BTAL taa_1n_qld only
before 2012-10-02. Crises: every S&P 500 TR fall of 10% or more since March 2008 plus four rate shocks (as in the
defensive dossier). CVaR 5% = mean of the worst 5% of rolling 21-day returns.
Fees: 2% a year management accrued daily + 20% of gains above the high-water mark, crystallised each 31 December
(the Israeli standard without a hurdle); a T-bill hurdle is the sensitivity.
Capacity: exactly the defensive method (defensive_capacity_20260924): last three years of fills, orders of all pods in
the same ticker and day added; market-on-open order sizes against 20-session median dollar volume; worked orders over
up to 5 days against 60-session volume with square-root impact (Y = 1); thin wrappers (BTAL, UUP, DBC) as blocks at
15 bps; wall = owning 10% of BTAL ($317M), UUP ($573M) or DBC (about $2.0B).
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import common  # noqa: E402
from data.norgate_loader import load_price_timeseries  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "growth_shelf_20260924"
END_TS, CUT_TS, LONG_TS = pd.Timestamp("2026-08-19"), pd.Timestamp("2012-10-02"), pd.Timestamp("2008-03-04")
CAPACITY_START_TS = pd.Timestamp("2023-08-21")
STAND_IN_DICT = {"taa_btal_tqqq": "taa_1n_qld", "taa_btal_1n_tqqq": "taa_1n_qld"}
RATE_SHOCK_LIST = [("Taper tantrum (bonds)", "2013-05-02", "2013-09-05"), ("Reflation sell-off (bonds)", "2016-07-08", "2016-12-15"),
                   ("2022 rate shock", "2022-01-03", "2022-10-24"), ("Long-bond rout", "2023-07-31", "2023-10-19")]
THIN_SET = {"BTAL", "UUP", "DBC"}
ETF_ASSET_DICT = {"BTAL": 317e6, "UUP": 572.8e6, "DBC": 2.0e9}
BLOCK_COST_FLOAT = 0.0015
AUM_GRID_TUPLE = (1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9)
THIRD_FLOAT = 1.0 / 3.0


def candidate_dict() -> dict[str, tuple[dict, str, str]]:
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    menu_dict = {p: dict(zip(g["alias_str"], g["weight_float"].astype(float))) for p, g in weight_df.groupby("product_id_str")}
    return {
        "G1 TAA 3x 1/N + NDX ATR + DV2": ({"taa_btal_1n_tqqq": THIRD_FLOAT, "ndx_atr": THIRD_FLOAT, "dv2": THIRD_FLOAT}, "annual", "candidate"),
        "G2 TAA 3x 1/N + MOSAIC": ({"taa_btal_1n_tqqq": 0.5, "mosaic": 0.5}, "annual", "candidate"),
        "G3 TAA 3x + NDX (live pair)": ({"taa_btal_tqqq": 0.5, "ndx_vxn": 0.5}, "annual", "candidate"),
        "G4 TAA 3x 1/N + NDX + MOSAIC": ({"taa_btal_1n_tqqq": THIRD_FLOAT, "ndx_vxn": THIRD_FLOAT, "mosaic": THIRD_FLOAT}, "annual", "candidate"),
        # Added after the first dossier run, for completeness: the best qualifier once the auction-bound DV2/HPI books
        # are set aside (growth_capacity.py). Labelled post-hoc; it does not change the selection rule.
        "G5 TAA 3x 1/N + NDX + InflC": ({"taa_btal_1n_tqqq": THIRD_FLOAT, "ndx_vxn": THIRD_FLOAT, "infl_compass": THIRD_FLOAT}, "annual", "candidate"),
        "ladder_4 (drift)": ({"dv2": 0.16, "hpi_vote": 0.17, "ndx_vxn": 0.25, "mosaic": 0.08, "taa_btal_tqqq": 0.34}, "none", "reference"),
        "menu Aggressive": (menu_dict["AGG"], "annual", "reference"),
        "menu Low-touch Growth": (menu_dict["LT_GRO"], "annual", "reference"),
    }


def fee_path(gross_ser: pd.Series, tbill_ser: pd.Series, hurdle_bool: bool, cost_per_year_float: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """Investor NAV under 2/20: 2% a year taken daily; 20% of gains above the high-water mark (and above T-bills for
    the year if hurdle) ACCRUED daily in the NAV, as a fund administrator does, and paid each 31 December.
    cost_per_year_float is an extra trading cost (market impact at size) taken off evenly each day."""
    pre_fee_nav_float, hwm_float, year_start_nav_float, hurdle_growth_float = 1.0, 1.0, 1.0, 1.0
    year_row_list, nav_list = [], []
    management_year_float, gross_year_growth_float = 0.0, 1.0
    year_arr = gross_ser.index.year
    tbill_arr = tbill_ser.reindex(gross_ser.index).fillna(0.0).to_numpy()
    for position_int, gross_float in enumerate(gross_ser.to_numpy()):
        net_of_cost_float = gross_float - cost_per_year_float / 252.0
        pre_fee_nav_float *= 1.0 + net_of_cost_float
        gross_year_growth_float *= 1.0 + net_of_cost_float
        management_fee_float = pre_fee_nav_float * 0.02 / 252.0
        pre_fee_nav_float -= management_fee_float
        management_year_float += management_fee_float
        hurdle_growth_float *= 1.0 + tbill_arr[position_int]
        base_float = max(hwm_float, year_start_nav_float * hurdle_growth_float) if hurdle_bool else hwm_float
        accrued_float = 0.20 * max(pre_fee_nav_float - base_float, 0.0)
        nav_list.append(pre_fee_nav_float - accrued_float)
        if position_int == len(gross_ser) - 1 or year_arr[position_int + 1] != year_arr[position_int]:
            year_row_list.append({"year": int(year_arr[position_int]), "return_after_cost": gross_year_growth_float - 1.0,
                                  "investor_return": (pre_fee_nav_float - accrued_float) / year_start_nav_float - 1.0,
                                  "management_fee": management_year_float, "performance_fee": accrued_float,
                                  "start_nav": year_start_nav_float, "performance_fee_paid": accrued_float > 0.0})
            pre_fee_nav_float -= accrued_float
            if accrued_float > 0.0:
                hwm_float = pre_fee_nav_float
            year_start_nav_float, hurdle_growth_float, management_year_float, gross_year_growth_float = pre_fee_nav_float, 1.0, 0.0, 1.0
    return pd.Series(nav_list, index=gross_ser.index), pd.DataFrame(year_row_list)


def fee_stats(gross_ser: pd.Series, tbill_ser: pd.Series, hurdle_bool: bool, cost_per_year_float: float = 0.0) -> dict:
    net_nav_ser, year_df = fee_path(gross_ser, tbill_ser, hurdle_bool, cost_per_year_float)
    years_float = (gross_ser.index[-1] - gross_ser.index[0]).days / 365.25
    average_nav_float = float(net_nav_ser.mean())
    # Longest run of year-ends without a performance fee.
    longest_dry_int, current_int = 0, 0
    for paid_bool in year_df["performance_fee_paid"]:
        current_int = 0 if paid_bool else current_int + 1
        longest_dry_int = max(longest_dry_int, current_int)
    fee_total_float = float(year_df["management_fee"].sum() + year_df["performance_fee"].sum())
    return {"investor_net_cagr": float(net_nav_ser.iloc[-1] ** (1.0 / years_float) - 1.0),
            "manager_fee_pct_of_aum_per_year": fee_total_float / average_nav_float / years_float,
            "performance_fee_pct_of_aum_per_year": float(year_df["performance_fee"].sum()) / average_nav_float / years_float,
            "years_with_performance_fee_share": float(year_df["performance_fee_paid"].mean()),
            "longest_run_of_years_without_performance_fee": longest_dry_int,
            "years_without_performance_fee": ", ".join(str(y) for y in year_df.loc[~year_df["performance_fee_paid"], "year"]),
            "investor_net_maxdd": float((net_nav_ser / net_nav_ser.cummax() - 1.0).min())}


def main() -> int:
    OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    books_dict = candidate_dict()
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    long_df = sleeve_df.copy()
    for alias_str, stand_in_str in STAND_IN_DICT.items():
        # *** CRITICAL*** the stand-in fills only the dates before the real sleeve exists.
        long_df.loc[long_df.index < CUT_TS, alias_str] = sleeve_df.loc[sleeve_df.index < CUT_TS, stand_in_str]
    spx_long_ser = bench_df["SPXTR"].loc[LONG_TS:]
    episode_list = [{"label": f"S&P 500 −{abs(e['spx_drawdown_float']) * 100:.0f}%: {pd.Timestamp(e['peak_date_str']):%b %Y} to {pd.Timestamp(e['trough_date_str']):%b %Y}",
                     "start": e["peak_date_str"], "end": e["trough_date_str"], "kind": "equity"} for e in common.equity_drawdown_episode_list(spx_long_ser)]
    episode_list += [{"label": l, "start": s, "end": e, "kind": "rates"} for l, s, e in RATE_SHOCK_LIST]
    episode_list.sort(key=lambda d: d["start"])
    crisis_mask_ser = pd.Series(False, index=spx_long_ser.index)
    for e in episode_list:
        if e["kind"] == "equity":
            crisis_mask_ser.loc[e["start"]:e["end"]] = True

    def metrics(ser: pd.Series) -> dict:
        base_ts = sleeve_df.index[sleeve_df.index.get_loc(ser.index[0]) - 1]
        return common.metric_dict(ser, bench_df["SPXTR"], bench_df["TBILL"], base_ts)

    book_row_list, crisis_row_list, series_dict = [], [], {}
    for name_str, (weight_dict, policy_str, kind_str) in books_dict.items():
        exact_ser = common.book_return_ser(sleeve_df.loc[CUT_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        long_ser = common.book_return_ser(long_df.loc[LONG_TS:, list(weight_dict)], weight_dict, policy_str)[0]
        series_dict[name_str] = (exact_ser, long_ser)
        me, ml = metrics(exact_ser), metrics(long_ser)
        nav_ser = common.nav_from_return_ser(exact_ser)
        row = {"book": name_str, "kind": kind_str, "pods": len(weight_dict), "cagr": me["cagr_float"], "vol": me["volatility_float"],
               "sharpe": me["sharpe_rf0_float"], "sharpe_excess": me["sharpe_excess_float"], "maxdd": me["max_drawdown_float"],
               "worst_12m": float((nav_ser / nav_ser.shift(252) - 1).dropna().min()), "worst_year": me["worst_year_float"],
               "cvar5_21d": me["es95_21d_float"], "beta": me["beta_spx_float"], "cagr_long": ml["cagr_float"], "sharpe_long": ml["sharpe_rf0_float"],
               "maxdd_long": ml["max_drawdown_float"], "maxdd_long_trough": ml["max_dd_trough_date_str"], "underwater_days_long": ml["longest_underwater_days_int"],
               "cvar5_21d_long": ml["es95_21d_float"], "corr_spx_crisis": float(long_ser[crisis_mask_ser].corr(spx_long_ser[crisis_mask_ser]))}
        for hurdle_bool in (False, True):
            prefix_str = "fee_hurdle_" if hurdle_bool else "fee_"
            row.update({prefix_str + k: v for k, v in fee_stats(exact_ser, bench_df["TBILL"], hurdle_bool).items()})
            row.update({prefix_str + "long_" + k: v for k, v in fee_stats(long_ser, bench_df["TBILL"], hurdle_bool).items()})
        book_row_list.append(row)
        crisis_row_list.append({"book": name_str, **{e["label"]: common.window_return_float(long_ser, e["start"], e["end"]) for e in episode_list}})
    book_df = pd.DataFrame(book_row_list).set_index("book")
    crisis_df = pd.DataFrame(crisis_row_list).set_index("book")
    crisis_df.loc["S&P 500 TR"] = pd.Series({e["label"]: common.window_return_float(spx_long_ser, e["start"], e["end"]) for e in episode_list})

    pod_list = ["taa_btal_1n_tqqq", "taa_btal_tqqq", "ndx_vxn", "ndx_atr", "mosaic", "dv2"]
    stress_mask_ser = spx_long_ser <= spx_long_ser.quantile(0.10)
    pod_df = long_df.loc[LONG_TS:, pod_list]
    corr_all_df = pod_df.corr()
    corr_stress_df = pod_df[stress_mask_ser.reindex(pod_df.index).fillna(False)].corr()

    # Capacity, candidates only.
    path_by_alias_dict = common.load_sleeve_path_dict()
    candidate_names = [n for n, v in books_dict.items() if v[2] == "candidate"]
    alias_set = {a for n in candidate_names for a in books_dict[n][0]}
    order_frame_list = []
    for alias_str in sorted(alias_set):
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        transaction_df = transaction_df[(transaction_df["date"] >= CAPACITY_START_TS) & (transaction_df["date"] <= END_TS)]
        prior_nav_ser = path_by_alias_dict[alias_str]["total_value_float"].shift(1)
        order_frame_list.append(transaction_df.assign(alias_str=alias_str, fraction_float=transaction_df["signed_notional_float"].abs().values
                                                      / prior_nav_ser.reindex(transaction_df["date"]).values)[["date", "alias_str", "asset_str", "fraction_float", "amount_float"]])
    order_df = pd.concat(order_frame_list, ignore_index=True)
    ticker_list = sorted(order_df["asset_str"].unique())
    print(f"capacity: {len(order_df)} orders on {len(ticker_list)} tickers")
    adv20_dict, adv60_dict, sigma_dict, close_dict = {}, {}, {}, {}
    for ticker_str in ticker_list:
        price_df = load_price_timeseries(ticker_str, start_date_str="2023-03-01", end_date_str=END_TS.strftime("%Y-%m-%d"))
        price_df.index = pd.to_datetime(price_df.index).normalize()
        dollar_ser = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before trading.
        adv20_dict[ticker_str] = dollar_ser.rolling(20, min_periods=10).median().shift(1)
        adv60_dict[ticker_str] = dollar_ser.rolling(60, min_periods=20).median().shift(1)
        sigma_dict[ticker_str] = price_df["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
        close_dict[ticker_str] = price_df["Close"]
    adv20_df, adv60_df, sigma_df, close_df = pd.DataFrame(adv20_dict), pd.DataFrame(adv60_dict), pd.DataFrame(sigma_dict), pd.DataFrame(close_dict)
    years_float = (END_TS - CAPACITY_START_TS).days / 365.25

    capacity_row_list = []
    for name_str in candidate_names:
        weight_dict, policy_str, _ = books_dict[name_str]
        exact_ser, _ = series_dict[name_str]
        _, prior_weight_df = common.book_return_ser(sleeve_df.loc[CUT_TS:, list(weight_dict)], weight_dict, policy_str)
        excess_return_float = float(book_df.at[name_str, "cagr"] - metrics(exact_ser)["tbill_cagr_float"])
        product_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        pod_weight_arr = prior_weight_df.reindex(product_order_df["date"]).to_numpy()
        column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
        product_order_df["book_fraction_float"] = product_order_df["fraction_float"].to_numpy() * np.array(
            [pod_weight_arr[i, column_index_dict[a]] for i, a in enumerate(product_order_df["alias_str"])])
        daily_df = product_order_df.groupby(["date", "asset_str"], as_index=False)["book_fraction_float"].sum()
        for column_str, frame_df in (("adv20", adv20_df), ("adv60", adv60_df), ("sigma", sigma_df)):
            daily_df[column_str] = [frame_df.at[d, t] if d in frame_df.index else np.nan for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        daily_df = daily_df.dropna(subset=["adv20", "adv60", "sigma"])
        thin_mask = daily_df["asset_str"].isin(THIN_SET).to_numpy()
        row = {"book": name_str}
        per_1m_arr = daily_df["book_fraction_float"].to_numpy() * 1e6 / daily_df["adv20"].to_numpy()
        for group_str, mask_arr in (("thin", thin_mask), ("other", ~thin_mask)):
            if mask_arr.any():
                row[f"moo_{group_str}_largest_hits_5pct"] = 0.05 / float(per_1m_arr[mask_arr].max()) * 1e6
                row[f"moo_{group_str}_largest_hits_1pct"] = 0.01 / float(per_1m_arr[mask_arr].max()) * 1e6
                row[f"moo_{group_str}_typical_hits_1pct"] = 0.01 / float(np.median(per_1m_arr[mask_arr])) * 1e6
                row[f"moo_{group_str}_largest_ticker"] = daily_df["asset_str"].to_numpy()[mask_arr][int(np.argmax(per_1m_arr[mask_arr]))]
        for mode_str in ("worked_5_days", "blocks"):
            recommended_float, outer_float, drag_at_100m_float = None, None, np.nan
            for aum_float in AUM_GRID_TUPLE:
                order_dollar_arr = daily_df["book_fraction_float"].to_numpy() * aum_float
                participation_arr = order_dollar_arr / daily_df["adv60"].to_numpy()
                day_arr = np.clip(np.ceil(participation_arr / 0.10), 1.0, 5.0)
                daily_participation_arr = participation_arr / day_arr
                impact_arr = order_dollar_arr * daily_df["sigma"].to_numpy() * np.sqrt(daily_participation_arr)
                if mode_str == "blocks":
                    impact_arr = np.where(thin_mask, order_dollar_arr * BLOCK_COST_FLOAT, impact_arr)
                    daily_participation_arr = np.where(thin_mask, 0.0, daily_participation_arr)
                drag_float = float(impact_arr.sum()) / aum_float / years_float
                p95_float, max_float = float(np.percentile(daily_participation_arr, 95)), float(daily_participation_arr.max())
                if aum_float == 1e8:
                    drag_at_100m_float = drag_float
                if p95_float <= 0.05 and max_float <= 0.20 and drag_float <= 0.25 * excess_return_float:
                    recommended_float = aum_float
                if p95_float <= 0.10 and max_float <= 0.30 and drag_float <= 0.50 * excess_return_float:
                    outer_float = aum_float
            row.update({f"{mode_str}_recommended": recommended_float, f"{mode_str}_outer": outer_float, f"{mode_str}_impact_at_100m": drag_at_100m_float})
        # Wall: largest book weight in each thin ETF (positions rebuilt from fills), versus 10% of the ETF's assets.
        wall_dict = {}
        for ticker_str in THIN_SET:
            book_weight_ser = None
            for alias_str, weight_float in weight_dict.items():
                transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
                transaction_df = transaction_df[transaction_df["asset_str"] == ticker_str]
                if transaction_df.empty or ticker_str not in close_df.columns:
                    continue
                share_ser = transaction_df.groupby("date")["amount_float"].sum().reindex(path_by_alias_dict[alias_str].index).fillna(0.0).cumsum()
                sleeve_weight_ser = (share_ser * close_df[ticker_str].reindex(share_ser.index)).abs() / path_by_alias_dict[alias_str]["total_value_float"]
                part_ser = sleeve_weight_ser.loc[CAPACITY_START_TS:END_TS] * prior_weight_df[alias_str].reindex(sleeve_weight_ser.loc[CAPACITY_START_TS:END_TS].index)
                book_weight_ser = part_ser if book_weight_ser is None else book_weight_ser.add(part_ser, fill_value=0.0)
            if book_weight_ser is not None and book_weight_ser.max() > 0:
                wall_dict[ticker_str] = 0.10 * ETF_ASSET_DICT[ticker_str] / float(book_weight_ser.max())
        if wall_dict:
            row["wall_aum"], row["wall_ticker"] = min(wall_dict.values()), min(wall_dict, key=wall_dict.get)
        capacity_row_list.append(row)
    capacity_df = pd.DataFrame(capacity_row_list).set_index("book")

    book_df.to_csv(OUT_DIR_PATH / "dossier_books.csv", float_format="%.6g")
    crisis_df.to_csv(OUT_DIR_PATH / "dossier_crises.csv", float_format="%.6g")
    corr_all_df.to_csv(OUT_DIR_PATH / "dossier_pod_corr_all.csv", float_format="%.4g")
    corr_stress_df.to_csv(OUT_DIR_PATH / "dossier_pod_corr_stress.csv", float_format="%.4g")
    capacity_df.to_csv(OUT_DIR_PATH / "dossier_capacity.csv", float_format="%.6g")
    (OUT_DIR_PATH / "dossier_episodes.json").write_text(json.dumps(episode_list, indent=1), encoding="utf-8")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    print(book_df[["pods", "cagr", "vol", "sharpe", "maxdd", "worst_12m", "cvar5_21d", "cagr_long", "sharpe_long", "maxdd_long", "maxdd_long_trough",
                   "underwater_days_long", "corr_spx_crisis"]].round(3).to_string())
    print()
    print(book_df[[c for c in book_df.columns if c.startswith("fee_") and not c.startswith("fee_hurdle")]].round(4).to_string())
    print()
    print(book_df[[c for c in book_df.columns if c.startswith("fee_hurdle_")]].round(4).to_string())
    print()
    print((crisis_df.T * 100).round(1).to_string())
    print()
    print(corr_all_df.round(2).to_string())
    print(corr_stress_df.round(2).to_string())
    print()
    print(capacity_df.T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
