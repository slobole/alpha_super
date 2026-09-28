"""Shared tradability measures for the readiness audit (protocol checks C1/C2).

C1 participation. For every backtest fill i on session t:

    order_frac_i = |amount_i * price_i| / NAV_(t-1)
    ADV20_{s,t}   = median(Turnover_{s, t-20..t-1})           (native Norgate Turnover, USD)
    part_i(C)     = order_frac_i * C / ADV20_{s,t}

so participation is evaluated at a constant pod size C, independent of how large the backtest
NAV grew. Turnover is Norgate's native daily dollar turnover (never raw Close x adjusted Volume).

C2 whole shares. For a target weight w on a name with nominal (unadjusted) decision close P:

    shares = floor(w * C / P)
    weight_error = w - shares * P / C

reported per name, with names where one share costs more than the target (shares = 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.norgate_loader import load_price_timeseries

CAPITAL_LIST = (30_000.0, 1_000_000.0, 10_000_000.0)


def load_turnover_ser(symbol_str: str, start_date_str: str = "1998-01-01", end_date_str: str | None = None) -> pd.Series:
    price_df = load_price_timeseries(symbol_str, start_date_str=start_date_str, end_date_str=end_date_str)
    if "Turnover" not in price_df.columns:
        raise RuntimeError(f"{symbol_str} has no native Turnover field.")
    turnover_ser = price_df["Turnover"].astype(float)
    # Padded (no-trade) sessions carry zero turnover; keep them, they are real zero-liquidity days.
    return turnover_ser.sort_index()


def adv20_before_ser(turnover_ser: pd.Series) -> pd.Series:
    # *** CRITICAL*** ADV for an order filled on session t uses sessions t-20..t-1 only.
    return turnover_ser.rolling(20, min_periods=10).median().shift(1)


def participation_table_df(strategy_obj, turnover_by_symbol_dict: dict[str, pd.Series]) -> pd.DataFrame:
    transaction_df = strategy_obj.get_transactions().copy()
    transaction_df = transaction_df[transaction_df["order_id"] != -1] if "order_id" in transaction_df.columns else transaction_df
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"])
    total_value_ser = strategy_obj.total_value_series.astype(float)
    total_value_ser.index = pd.to_datetime(total_value_ser.index)
    prev_nav_ser = total_value_ser.shift(1)
    row_list = []
    for _, tx_row in transaction_df.iterrows():
        asset_str = str(tx_row["asset"])
        bar_ts = pd.Timestamp(tx_row["bar"])
        nav_prev_float = float(prev_nav_ser.get(bar_ts, np.nan))
        if not np.isfinite(nav_prev_float) or nav_prev_float <= 0.0:
            continue
        order_frac_float = abs(float(tx_row["amount"]) * float(tx_row["price"])) / nav_prev_float
        turnover_ser = turnover_by_symbol_dict.get(asset_str)
        adv_float = np.nan
        if turnover_ser is not None:
            adv_ser = adv20_before_ser(turnover_ser)
            if bar_ts in adv_ser.index:
                adv_float = float(adv_ser.loc[bar_ts])
        row_dict = {"bar": bar_ts, "asset": asset_str, "order_frac_of_nav": order_frac_float, "adv20_usd": adv_float}
        for capital_float in CAPITAL_LIST:
            row_dict[f"part_{int(capital_float)}"] = (
                order_frac_float * capital_float / adv_float if np.isfinite(adv_float) and adv_float > 0 else np.inf
            )
        row_list.append(row_dict)
    return pd.DataFrame(row_list)


def summarize_participation_df(participation_df: pd.DataFrame, recent_start_str: str = "2023-09-25") -> pd.DataFrame:
    row_list = []
    for window_str, window_df in (
        ("full", participation_df),
        ("last3y", participation_df[participation_df["bar"] >= pd.Timestamp(recent_start_str)]),
    ):
        for capital_float in CAPITAL_LIST:
            column_str = f"part_{int(capital_float)}"
            value_ser = window_df[column_str].replace([np.inf], np.nan)
            row_list.append(
                {
                    "window": window_str,
                    "capital_usd": int(capital_float),
                    "orders": int(len(window_df)),
                    "median_pct_adv": float(value_ser.median() * 100.0),
                    "p99_pct_adv": float(value_ser.quantile(0.99) * 100.0),
                    "max_pct_adv": float(value_ser.max() * 100.0),
                    "share_orders_over_1pct": float((value_ser > 0.01).mean()),
                    "share_orders_over_5pct": float((value_ser > 0.05).mean()),
                    "orders_without_adv": int(window_df[column_str].isin([np.inf]).sum()),
                    "worst_asset": str(window_df.loc[value_ser.idxmax(), "asset"]) if value_ser.notna().any() else "",
                }
            )
    return pd.DataFrame(row_list)


def whole_share_table_df(
    target_weight_by_decision_dict: dict[str, dict[str, float]],
    unadjusted_close_lookup_fn,
    capital_list: tuple[float, ...] = (10_000.0, 15_000.0, 30_000.0),
) -> pd.DataFrame:
    row_list = []
    for decision_date_str, weight_dict in target_weight_by_decision_dict.items():
        for asset_str, weight_float in weight_dict.items():
            price_float = float(unadjusted_close_lookup_fn(asset_str, decision_date_str))
            for capital_float in capital_list:
                share_int = int(np.floor(weight_float * capital_float / price_float)) if price_float > 0 else 0
                row_list.append(
                    {
                        "decision_date": decision_date_str,
                        "asset": asset_str,
                        "capital_usd": capital_float,
                        "target_weight": weight_float,
                        "nominal_price": price_float,
                        "shares": share_int,
                        "held_weight": share_int * price_float / capital_float,
                        "weight_error": weight_float - share_int * price_float / capital_float,
                        "zero_share_bool": share_int == 0,
                    }
                )
    return pd.DataFrame(row_list)
