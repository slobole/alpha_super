"""Scout weights engine: full-target rebalancing with the real engine's execution semantics (parity mode).

Mapped from alpha/engine/strategy.py and backtester.py on 2026-09-30 (P3) for the full-target monthly pods
(TAA 3x, NDX VXN); both independent replicas built from this contract matched the engine to machine precision.

Per session t (the decision for a rebalance at t was taken after the close of t−1 = T):

1. Dividends, every session, before the open:
       gross_i = position_i(close T) × Dividend_i(T)
       cash   += gross_i − 0.25 × max(gross_i, 0)                (25% withholding on long dividends)
   A NaN dividend on a held asset raises.
2. Missing-price liquidation: a held asset with NaN Open(t) or NaN Close(t) is sold in full at its last finite
   Close ≤ T, with no slippage and the usual fee.
3. Rebalance (only on dates in `rebalance_weight_df`), sized from V = total value at the close of T (before the
   dividend credit of step 1):
       adjusted units    target_i = trunc(V × w_i / Close_i(T))
       historical units  target_i = trunc(V × w_i / UnadjClose_i(T)) × k_i(T),   k = UnadjClose / Close
   An asset with w_i = 0, or whose target rounds to 0, goes to 0. Every held or targeted asset is traded to its
   target (monthly drift is corrected); a zero delta is no trade. An order whose Open(t) is NaN is cancelled.
       fill price  = Open_i(t) × (1 + sign(Δ) × slippage)
       fee         = max(min_fee, fee_per_share × |Δ|)           adjusted units
                     max(min_fee, fee_per_share × |Δ| / k_i(t))   historical units (raw-share-equivalent)
   All orders settle together: there is no cash check, cash may go negative and earns nothing.
4. Mark: total(t) = cash + Σ position_i × Close_i(t).

Shorts (opt-in, `allow_short_bool`; added 2026-10-01 for CORE5's DBC short, strategy_taa_adaptive_macro_core5.py):
- a negative weight sizes target_i = trunc(V × w_i / Close_i(T)) (truncation toward zero, as the engine's `int()`);
  short proceeds stay in cash and earn nothing; a short pays the full dividend (step 1, no withholding credit);
- `split_sign_flip_bool`: a long-to-short (or short-to-long) change is two orders at the same open, the close of the
  old leg and the opening of the new one, each with its own fee (CORE5 `_submit_target_orders`);
- `borrow_model`: after the close mark of t, a held short pays
      fee = |shares| × ceil(multiplier × Close_i(t)) × annual_rate × calendar days(t → next session) / day_count
  debited from cash and total(t); no fee on the last session (CORE5 `apply_post_mark_accounting`).
Without `allow_short_bool` a negative weight raises: a long-only spec must never silently drop a short.

*** CRITICAL*** Sizing reads prices of T only; fills use Open(t). The engine's order of operations is part of the
model (QUANT_PHILOSOPHY.md "Engine Order Is Part Of The Model").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SHARE_UNIT_MODE_TUPLE = ("adjusted", "historical")


@dataclass(frozen=True)
class CostModel:
    slippage_float: float = 0.00025
    fee_per_share_float: float = 0.005
    min_fee_float: float = 1.0
    dividend_withholding_float: float = 0.25


@dataclass(frozen=True)
class BorrowModel:
    """Short borrow fee accrued after the close mark (CORE5's fixed DBC research baseline)."""

    annual_rate_float: float = 0.01
    collateral_multiplier_float: float = 1.02
    day_count_int: int = 360


@dataclass
class WeightsResult:
    total_value_ser: pd.Series
    daily_return_ser: pd.Series
    position_after_rebalance_df: pd.DataFrame  # ledger units, one row per rebalance date
    trade_df: pd.DataFrame
    daily_position_df: pd.DataFrame  # ledger units held at each close
    borrow_fee_df: pd.DataFrame | None = None  # short borrow fees (borrow_model only)


def simulate(
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    start_date,
    capital_float: float = 100_000.0,
    share_unit_mode_str: str = "adjusted",
    unadjusted_close_df: pd.DataFrame | None = None,
    cost_model: CostModel = CostModel(),
    allow_short_bool: bool = False,
    split_sign_flip_bool: bool = False,
    borrow_model: BorrowModel | None = None,
) -> WeightsResult:
    if share_unit_mode_str not in SHARE_UNIT_MODE_TUPLE:
        raise ValueError(f"share_unit_mode_str must be one of {SHARE_UNIT_MODE_TUPLE}.")
    if share_unit_mode_str == "historical" and unadjusted_close_df is None:
        raise ValueError("Historical share units need unadjusted_close_df.")

    date_index = open_df.index
    asset_list = list(open_df.columns)
    open_mat = open_df.to_numpy(dtype=float)
    close_mat = close_df.reindex(index=date_index, columns=asset_list).to_numpy(dtype=float)
    dividend_mat = dividend_df.reindex(index=date_index, columns=asset_list).to_numpy(dtype=float)
    if share_unit_mode_str == "historical":
        unadjusted_mat = unadjusted_close_df.reindex(index=date_index, columns=asset_list).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor_mat = unadjusted_mat / close_mat  # k = UnadjClose / Close
    else:
        factor_mat = np.ones_like(close_mat)
    rebalance_lookup_dict = {
        pd.Timestamp(date): row.reindex(asset_list).fillna(0.0).to_numpy(dtype=float)
        for date, row in rebalance_weight_df.iterrows()
    }
    if not allow_short_bool and any((weight_vec < 0.0).any() for weight_vec in rebalance_lookup_dict.values()):
        raise ValueError("Negative target weights need allow_short_bool=True.")

    start_idx_int = int(date_index.searchsorted(pd.Timestamp(start_date)))
    if start_idx_int == 0:
        raise ValueError("The simulation needs one session before start_date (the first decision's T).")
    position_vec = np.zeros(len(asset_list))
    cash_float = float(capital_float)
    previous_total_float = float(capital_float)
    total_list, trade_list, position_row_dict, daily_position_list, borrow_fee_list = [], [], {}, [], []

    def _fee(delta_float: float, factor_float: float) -> float:
        return max(cost_model.min_fee_float, cost_model.fee_per_share_float * abs(delta_float) / factor_float)

    for t_idx_int in range(start_idx_int, len(date_index)):
        t_date = date_index[t_idx_int]
        previous_idx_int = t_idx_int - 1

        # 1. Dividends of T, credited before the open of t.
        held_mask = position_vec != 0.0
        if held_mask.any():
            dividend_vec = dividend_mat[previous_idx_int, held_mask]
            if np.isnan(dividend_vec).any():
                raise ValueError(f"NaN dividend on a held asset at {date_index[previous_idx_int].date()}.")
            gross_vec = position_vec[held_mask] * dividend_vec
            cash_float += float(np.sum(gross_vec - cost_model.dividend_withholding_float * np.maximum(gross_vec, 0.0)))

        # 2. Missing-price liquidation at the last finite close <= T.
        for asset_idx_int in np.flatnonzero(position_vec != 0.0):
            if np.isfinite(open_mat[t_idx_int, asset_idx_int]) and np.isfinite(close_mat[t_idx_int, asset_idx_int]):
                continue
            finite_idx_arr = np.flatnonzero(np.isfinite(close_mat[:t_idx_int, asset_idx_int]))
            last_idx_int = int(finite_idx_arr[-1])
            delta_float = -position_vec[asset_idx_int]
            fee_float = _fee(delta_float, factor_mat[last_idx_int, asset_idx_int])
            cash_float -= delta_float * close_mat[last_idx_int, asset_idx_int] + fee_float
            position_vec[asset_idx_int] = 0.0
            trade_list.append((t_date, asset_list[asset_idx_int], delta_float, close_mat[last_idx_int, asset_idx_int], fee_float, "liquidation"))

        # 3. Rebalance to targets sized on T.
        weight_vec = rebalance_lookup_dict.get(t_date)
        if weight_vec is not None:
            target_vec = np.zeros(len(asset_list))
            for asset_idx_int in np.flatnonzero(weight_vec != 0.0):
                if share_unit_mode_str == "adjusted":
                    price_float = close_mat[previous_idx_int, asset_idx_int]
                    target_vec[asset_idx_int] = float(int(previous_total_float * weight_vec[asset_idx_int] / price_float))
                else:
                    raw_price_float = unadjusted_mat[previous_idx_int, asset_idx_int]
                    raw_share_int = int(previous_total_float * weight_vec[asset_idx_int] / raw_price_float)
                    target_vec[asset_idx_int] = float(raw_share_int * factor_mat[previous_idx_int, asset_idx_int])
            for asset_idx_int in np.flatnonzero((target_vec != position_vec)):
                delta_float = target_vec[asset_idx_int] - position_vec[asset_idx_int]
                open_float = open_mat[t_idx_int, asset_idx_int]
                if not np.isfinite(open_float):
                    continue  # cancelled: no bar to fill on
                current_float, target_float = position_vec[asset_idx_int], target_vec[asset_idx_int]
                if split_sign_flip_bool and current_float * target_float < 0.0:
                    leg_delta_tuple = (-current_float, target_float)  # close the old leg, then open the new one
                else:
                    leg_delta_tuple = (delta_float,)
                for leg_delta_float in leg_delta_tuple:
                    fill_float = open_float * (1.0 + np.sign(leg_delta_float) * cost_model.slippage_float)
                    fee_float = _fee(leg_delta_float, factor_mat[t_idx_int, asset_idx_int])
                    cash_float -= leg_delta_float * fill_float + fee_float
                    position_vec[asset_idx_int] += leg_delta_float
                    trade_list.append((t_date, asset_list[asset_idx_int], leg_delta_float, fill_float, fee_float, "rebalance"))
            position_row_dict[t_date] = position_vec.copy()

        # 4. Mark to market at the close of t.
        held_idx_arr = np.flatnonzero(position_vec != 0.0)
        total_float = cash_float + float(np.sum(position_vec[held_idx_arr] * close_mat[t_idx_int, held_idx_arr]))
        # 5. Short borrow fee after the mark, accrued to the next session of the run (none on the last session).
        if borrow_model is not None and borrow_model.annual_rate_float != 0.0 and t_idx_int + 1 < len(date_index):
            day_count_int = int((date_index[t_idx_int + 1].normalize() - t_date.normalize()).days)
            for asset_idx_int in np.flatnonzero(position_vec < 0.0):
                collateral_price_float = float(np.ceil(borrow_model.collateral_multiplier_float * close_mat[t_idx_int, asset_idx_int]))
                collateral_value_float = abs(position_vec[asset_idx_int]) * collateral_price_float
                fee_float = float(collateral_value_float * borrow_model.annual_rate_float * day_count_int / borrow_model.day_count_int)
                cash_float -= fee_float
                total_float -= fee_float
                borrow_fee_list.append((t_date, asset_list[asset_idx_int], fee_float))
        total_list.append(total_float)
        daily_position_list.append(position_vec.copy())
        previous_total_float = total_float

    total_value_ser = pd.Series(total_list, index=date_index[start_idx_int:], name="total_value")
    daily_return_ser = total_value_ser / total_value_ser.shift(1).fillna(capital_float) - 1.0
    return WeightsResult(
        total_value_ser=total_value_ser,
        daily_return_ser=daily_return_ser,
        position_after_rebalance_df=pd.DataFrame(position_row_dict, index=asset_list).T,
        trade_df=pd.DataFrame(trade_list, columns=["date", "asset", "delta_float", "price_float", "fee_float", "kind_str"]),
        daily_position_df=pd.DataFrame(daily_position_list, index=date_index[start_idx_int:], columns=asset_list),
        borrow_fee_df=pd.DataFrame(borrow_fee_list, columns=["date", "asset", "fee_float"]),
    )
