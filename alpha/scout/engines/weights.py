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

Same-session close execution (opt-in `fill_at_close_bool`, MOC; added 2026-10-01 for the month-end rebalancing flow,
strategy_taa_month_end_rebalancing_flow.py `process_orders`, which copies the price frame, writes Close(t) into Open(t)
for its traded assets and then runs the engine's ordinary `process_orders` on that copy):
- the order of operations is unchanged: (0) decision hook, (1) dividends of T on the pre-fill position, (2)
  missing-price liquidation, (3) orders sized on V and Close of T, (4) mark at Close(t), (5) borrow;
- every execution price of session t is Close(t): fill = Close_i(t) × (1 + sign(Δ) × slippage), the same fee; an
  order whose Close(t) is NaN is cancelled; a held asset is liquidated (step 2) only when Close(t) is NaN, because
  the substituted Open(t) is Close(t);
- the new position is marked at the same Close(t) it filled at (the fill session's return is only the slippage and
  fee), and the dividend entitlement of t (credited before t+1) is on the post-fill position.
Close-and-reopen orders (opt-in `close_and_reopen_bool`; the same strategy's `iterate`): every rebalance first closes
each held position in full (one order per asset), then opens each non-zero target as a new order, even when the
share count is unchanged; each order pays its own slippage and fee (a +100% -> -100% reversal trades 200% of NAV).
It implies `split_sign_flip_bool` and replaces the netted delta.

Path-dependent rules (added 2026-10-01 for Trinity's no-trade band): `decision_fn(t_idx, position_vec, total_T)` is
called at the top of session t, before step 1, with the ledger state at the close of T (the engine's `iterate()`
also runs before `process_orders()`), and returns a target weight vector for a rebalance at t or None. Its decisions
are returned as `WeightsResult.decided_weight_df`; re-running `simulate` with that frame as `rebalance_weight_df`
reproduces the same path exactly.
"""

from __future__ import annotations

from collections.abc import Callable
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


DEFAULT_COST_MODEL = CostModel()


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
    decided_weight_df: pd.DataFrame | None = None  # rebalance rows chosen by `decision_fn` (None without one)


def simulate(
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    start_date,
    capital_float: float = 100_000.0,
    share_unit_mode_str: str = "adjusted",
    unadjusted_close_df: pd.DataFrame | None = None,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    allow_short_bool: bool = False,
    split_sign_flip_bool: bool = False,
    borrow_model: BorrowModel | None = None,
    decision_fn: Callable[[int, np.ndarray, float], np.ndarray | None] | None = None,
    fill_at_close_bool: bool = False,
    close_and_reopen_bool: bool = False,
) -> WeightsResult:
    if share_unit_mode_str not in SHARE_UNIT_MODE_TUPLE:
        raise ValueError(f"share_unit_mode_str must be one of {SHARE_UNIT_MODE_TUPLE}.")
    if share_unit_mode_str == "historical" and unadjusted_close_df is None:
        raise ValueError("Historical share units need unadjusted_close_df.")

    date_index = open_df.index
    asset_list = list(open_df.columns)
    close_mat = close_df.reindex(index=date_index, columns=asset_list).to_numpy(dtype=float)
    # *** CRITICAL*** MOC: Close(t) becomes the session's execution price (the strategy's Open := Close copy); sizing
    # below still reads Close(T) only.
    open_mat = close_mat if fill_at_close_bool else open_df.to_numpy(dtype=float)
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
    decided_row_dict: dict = {}

    def _fee(delta_float: float, factor_float: float) -> float:
        return max(cost_model.min_fee_float, cost_model.fee_per_share_float * abs(delta_float) / factor_float)

    for t_idx_int in range(start_idx_int, len(date_index)):
        t_date = date_index[t_idx_int]
        previous_idx_int = t_idx_int - 1

        # 0. A path-dependent rule decides after the close of T, on the ledger as it stood then.
        if decision_fn is not None:
            decided_vec = decision_fn(t_idx_int, position_vec.copy(), previous_total_float)
            if decided_vec is not None:
                if t_date in rebalance_lookup_dict:
                    raise ValueError(f"Both rebalance_weight_df and decision_fn set a target for {t_date.date()}.")
                rebalance_lookup_dict[t_date] = np.asarray(decided_vec, dtype=float)
                decided_row_dict[t_date] = rebalance_lookup_dict[t_date]

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
            if close_and_reopen_bool:
                # Every held position is closed, then every non-zero target opened: (asset, leg) in the engine's order.
                leg_list = [(a, -position_vec[a]) for a in np.flatnonzero(position_vec != 0.0)]
                leg_list += [(a, target_vec[a]) for a in np.flatnonzero(target_vec != 0.0)]
            else:
                leg_list = []
                for asset_idx_int in np.flatnonzero(target_vec != position_vec):
                    current_float, target_float = position_vec[asset_idx_int], target_vec[asset_idx_int]
                    if split_sign_flip_bool and current_float * target_float < 0.0:
                        # close the old leg, then open the new one
                        leg_list += [(asset_idx_int, -current_float), (asset_idx_int, target_float)]
                    else:
                        leg_list.append((asset_idx_int, target_float - current_float))
            for asset_idx_int, leg_delta_float in leg_list:
                open_float = open_mat[t_idx_int, asset_idx_int]
                if not np.isfinite(open_float):
                    continue  # cancelled: no bar to fill on
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
        decided_weight_df=(
            pd.DataFrame(decided_row_dict, index=asset_list).T if decision_fn is not None else None
        ),
    )
