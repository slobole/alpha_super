"""How much money can the defensive products take? (owner question, 2026-09-24)

Why not the house capacity tool as is: capacity_v2 routes every order through the OPENING AUCTION with limits of
0.05% / 0.10% of daily volume and a stock-derived impact proxy for ETFs, and its full-history window measures
liquidity in years when an ETF was new (BTAL 2011-2013). For monthly ETF books that says "below $50K" for CORE5,
which describes the tool's assumptions, not the product. This study instead asks what a fund would actually do.

PRE-DECLARED (before any number below was computed):
  Products: CORE5 alone; CORE5 + BTAL_QQQ (50/50); Low-touch Defensive (amended A1); main-line Defensive.
  Trades: each sleeve's own fills from its $1M reference run, as a fraction of that sleeve's NAV at the prior close,
    scaled by the pod's weight in the book on that day (annual reset) and by the product AUM. Same ticker on the
    same day across pods is ADDED (no internal netting, conservative).
  Window: the last three years of trading, 2023-08-21 -> 2026-08-19, i.e. today's market; the last five years is a
    sensitivity.
  Liquidity: each ticker's median daily dollar volume over the 60 sessions before the trade (causal, contemporaneous).
  Execution: monthly-style orders worked over the session (VWAP), not only in the opening auction; one day per order.
  Impact: square-root law on top of the backtest's own 2.5 bps: cost = Y x daily volatility x sqrt(order / daily volume),
    Y = 1 (conservative central), volatility = 60-session std of the ticker's daily returns before the trade.
  Capacity, per product:
    Recommended = largest AUM with order/volume P95 <= 5%, max <= 20%, and impact <= 25% of the product's return over
      T-bills (exact window, 2012-2026);
    Outer = P95 <= 10%, max <= 30%, impact <= 50% of that return.
  Also reported: the binding ticker, and BTAL's volume by year, to show why early-era liquidity misleads.
ADDED AFTER THE FIRST RUN (it showed a handful of single-day DBC / UUP / BTAL orders binding): a second execution
  mode that works each order over as many days as keep it at or under 10% of daily volume, at most 5 days, with the
  square-root cost applied to each day's slice. Monthly books can afford a few days of implementation.
  And a third: DBC, UUP and BTAL (thin wrappers around deep futures / stock baskets) dealt as blocks with a liquidity
  provider or through creation and redemption, at an assumed 15 bps per traded dollar and no screen-volume limit;
  everything else as in the second mode. This shows what binds next; it is an assumption, not a quote.
Research only: pre-TCA estimates, no broker data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "fund_menu_20260923"))
import common  # noqa: E402
from data.norgate_loader import load_price_timeseries  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "defensive_capacity_20260924"
END_TS = pd.Timestamp("2026-08-19")
WINDOW_DICT = {"recent_3y": pd.Timestamp("2023-08-21"), "recent_5y": pd.Timestamp("2021-08-19")}
LIQUIDITY_LOOKBACK_INT = 60
IMPACT_Y_FLOAT = 1.0
AUM_GRID_TUPLE = (1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 2.5e9, 5e9)
RULE_DICT = {"recommended": (0.05, 0.20, 0.25), "outer": (0.10, 0.30, 0.50)}
MAX_WORK_DAYS_INT = 5
# Thin wrappers whose real liquidity is their underlying: DBC and UUP hold futures, BTAL a long/short US stock basket.
FUTURES_WRAPPER_SET = {"DBC", "UUP", "BTAL"}
BLOCK_COST_FLOAT = 0.0015  # 15 bps per traded dollar for a block against the basket / futures, assumed, conservative


def product_weight_dict() -> dict[str, dict[str, float]]:
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    menu_dict = {p: dict(zip(g["alias_str"], g["weight_float"].astype(float))) for p, g in weight_df.groupby("product_id_str")}
    return {"CORE5 alone": {"core5": 1.0}, "CORE5 + BTAL_QQQ": {"core5": 0.5, "taa_btal_lin_qqq": 0.5},
            "Low-touch Defensive (A1)": menu_dict["LT_DEF"], "Defensive (main line)": menu_dict["DEF"]}


def load_liquidity(ticker_list: list[str], start_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Causal 60-session median dollar volume and return volatility per ticker (both end the day before)."""
    dollar_dict, vol_dict, raw_dollar_dict = {}, {}, {}
    for ticker_str in ticker_list:
        price_df = load_price_timeseries(ticker_str, start_date_str=start_str, end_date_str=END_TS.strftime("%Y-%m-%d"))
        price_df.index = pd.to_datetime(price_df.index).normalize()
        dollar_ser = (price_df["Close"] * price_df["Volume"]).replace(0.0, np.nan)
        raw_dollar_dict[ticker_str] = dollar_ser
        # *** CRITICAL*** shift(1): the liquidity a trader could know before trading on day t.
        dollar_dict[ticker_str] = dollar_ser.rolling(LIQUIDITY_LOOKBACK_INT, min_periods=20).median().shift(1)
        vol_dict[ticker_str] = price_df["Close"].pct_change(fill_method=None).rolling(LIQUIDITY_LOOKBACK_INT, min_periods=20).std().shift(1)
    return pd.DataFrame(dollar_dict), pd.DataFrame(vol_dict), pd.DataFrame(raw_dollar_dict)


def main() -> int:
    OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    products_dict = product_weight_dict()
    alias_set = {a for w in products_dict.values() for a in w}
    sleeve_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    bench_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True).loc[:END_TS]
    path_by_alias_dict = common.load_sleeve_path_dict()

    # Order size as a fraction of the sleeve's own NAV at the prior close.
    order_frame_list = []
    first_window_ts = min(WINDOW_DICT.values())
    for alias_str in sorted(alias_set):
        transaction_df = pd.read_csv(common.SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", parse_dates=["date"])
        transaction_df = transaction_df[(transaction_df["date"] >= first_window_ts) & (transaction_df["date"] <= END_TS)]
        prior_nav_ser = path_by_alias_dict[alias_str]["total_value_float"].shift(1)
        transaction_df = transaction_df.assign(alias_str=alias_str,
                                               fraction_float=transaction_df["signed_notional_float"].abs().values
                                               / prior_nav_ser.reindex(transaction_df["date"]).values)
        order_frame_list.append(transaction_df[["date", "alias_str", "asset_str", "fraction_float"]])
    order_df = pd.concat(order_frame_list, ignore_index=True)
    ticker_list = sorted(order_df["asset_str"].unique())
    print(f"{len(order_df)} orders on {len(ticker_list)} tickers since {first_window_ts.date()}")
    liquidity_start_str = (first_window_ts - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    adv_df, sigma_df, _ = load_liquidity(ticker_list, liquidity_start_str)

    row_list, binding_row_list, capacity_row_list = [], [], []
    for product_str, weight_dict in products_dict.items():
        book_ser, prior_weight_df = common.book_return_ser(sleeve_df.loc["2012-10-02":, list(weight_dict)], weight_dict, "annual")
        exact_metric = common.metric_dict(book_ser, bench_df["SPXTR"], bench_df["TBILL"], sleeve_df.index[sleeve_df.index.get_loc(book_ser.index[0]) - 1])
        excess_return_float = exact_metric["cagr_float"] - exact_metric["tbill_cagr_float"]
        product_order_df = order_df[order_df["alias_str"].isin(weight_dict)].copy()
        pod_weight_arr = prior_weight_df.reindex(product_order_df["date"]).to_numpy()
        column_index_dict = {a: i for i, a in enumerate(prior_weight_df.columns)}
        product_order_df["pod_weight_float"] = [pod_weight_arr[i, column_index_dict[a]] for i, a in enumerate(product_order_df["alias_str"])]
        # Same ticker, same day, all pods: added, never netted.
        product_order_df["book_fraction_float"] = product_order_df["fraction_float"] * product_order_df["pod_weight_float"]
        daily_df = product_order_df.groupby(["date", "asset_str"], as_index=False)["book_fraction_float"].sum()
        daily_df["adv_float"] = [adv_df.at[d, t] if d in adv_df.index else np.nan for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        daily_df["sigma_float"] = [sigma_df.at[d, t] if d in sigma_df.index else np.nan for d, t in zip(daily_df["date"], daily_df["asset_str"])]
        for window_str, window_start_ts in WINDOW_DICT.items():
            window_df = daily_df[daily_df["date"] >= window_start_ts].dropna(subset=["adv_float", "sigma_float"])
            years_float = (END_TS - window_start_ts).days / 365.25
            passing_dict = {(rule_str, mode_str): None for rule_str in RULE_DICT
                            for mode_str in ("one_day", "up_to_5_days", "5_days_wrappers_as_blocks")}
            for aum_float in AUM_GRID_TUPLE:
                order_dollar_arr = window_df["book_fraction_float"].to_numpy() * aum_float
                participation_arr = order_dollar_arr / window_df["adv_float"].to_numpy()
                spread_day_arr = np.clip(np.ceil(participation_arr / 0.10), 1.0, float(MAX_WORK_DAYS_INT))
                wrapper_mask_arr = window_df["asset_str"].isin(FUTURES_WRAPPER_SET).to_numpy()
                for mode_str, day_arr in (("one_day", np.ones_like(participation_arr)),
                                          # Worked over as many days as keep each day at or under 10% of volume, at most 5.
                                          ("up_to_5_days", spread_day_arr),
                                          # Thin futures / basket wrappers dealt as blocks against their underlying (see docstring).
                                          ("5_days_wrappers_as_blocks", spread_day_arr)):
                    daily_participation_arr = participation_arr / day_arr
                    impact_arr = order_dollar_arr * IMPACT_Y_FLOAT * window_df["sigma_float"].to_numpy() * np.sqrt(daily_participation_arr)
                    if mode_str == "5_days_wrappers_as_blocks":
                        impact_arr = np.where(wrapper_mask_arr, order_dollar_arr * BLOCK_COST_FLOAT, impact_arr)
                        daily_participation_arr = np.where(wrapper_mask_arr, 0.0, daily_participation_arr)
                    impact_dollar_float = float(np.sum(impact_arr))
                    impact_drag_float = impact_dollar_float / aum_float / years_float
                    p95_float, max_float = float(np.percentile(daily_participation_arr, 95)), float(daily_participation_arr.max())
                    worst_idx = int(np.argmax(daily_participation_arr))
                    row_list.append({"product": product_str, "window": window_str, "execution": mode_str, "aum": aum_float, "orders": len(window_df),
                                     "participation_p95": p95_float, "participation_max": max_float,
                                     "max_ticker": window_df["asset_str"].iloc[worst_idx], "impact_drag_per_year": impact_drag_float,
                                     "excess_return_2012_26": excess_return_float, "drag_share_of_excess": impact_drag_float / excess_return_float})
                    for rule_str, (p95_limit_float, max_limit_float, drag_limit_float) in RULE_DICT.items():
                        if p95_float <= p95_limit_float and max_float <= max_limit_float and impact_drag_float <= drag_limit_float * excess_return_float:
                            passing_dict[(rule_str, mode_str)] = aum_float
            if window_str == "recent_3y":
                # Which tickers bind: the largest order/volume at $100M, per ticker.
                at_100m_df = window_df.assign(participation_float=window_df["book_fraction_float"] * 1e8 / window_df["adv_float"])
                top_df = at_100m_df.groupby("asset_str")["participation_float"].max().sort_values(ascending=False).head(5)
                binding_row_list.append({"product": product_str, **{f"top{i + 1}": f"{t} {v:.1%}" for i, (t, v) in enumerate(top_df.items())}})
            print(f"{product_str:<28} {window_str}: " + ", ".join(f"{r}/{m} <= {v}" for (r, m), v in passing_dict.items()))
            capacity_row_list.append({"product": product_str, "window": window_str, **{f"{r}__{m}": v for (r, m), v in passing_dict.items()}})
    curve_df = pd.DataFrame(row_list)
    binding_df = pd.DataFrame(binding_row_list).set_index("product")
    pd.DataFrame(capacity_row_list).to_csv(OUT_DIR_PATH / "capacity_summary.csv", index=False)

    # Why early liquidity misleads: BTAL and the other small ETFs, median dollar volume by year.
    _, _, raw_dollar_df = load_liquidity(["BTAL", "UUP", "DBC", "GLD", "TLT", "IEF", "LQD", "QQQ", "TQQQ", "BIL"], "2011-09-13")
    yearly_df = raw_dollar_df.groupby(raw_dollar_df.index.year).median() / 1e6
    today_adv_ser = raw_dollar_df.loc["2025-08-20":].median() / 1e6

    OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(OUT_DIR_PATH / "capacity_curve.csv", index=False, float_format="%.6g")
    binding_df.to_csv(OUT_DIR_PATH / "binding_tickers_at_100m.csv")
    yearly_df.to_csv(OUT_DIR_PATH / "etf_median_dollar_volume_by_year_musd.csv", float_format="%.3f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show_df = curve_df[(curve_df["window"] == "recent_3y") & curve_df["aum"].isin([1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8])].copy()
    show_df["aum_m"] = show_df["aum"] / 1e6
    print(show_df[["product", "execution", "aum_m", "participation_p95", "participation_max", "max_ticker", "impact_drag_per_year",
                   "drag_share_of_excess"]].round(4).to_string(index=False))
    print()
    print(binding_df.to_string())
    print()
    print("median daily dollar volume, $M, by year:")
    print(yearly_df.round(1).to_string())
    print("last 12 months:", today_adv_ser.round(1).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
