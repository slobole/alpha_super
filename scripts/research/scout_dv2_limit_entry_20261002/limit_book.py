"""The frozen DV2 rule with limit-order execution on daily bars (registration dv2_limit_entry_20261002).

The signal, ranking, slot logic, sizing, delisting handling and fees are those of `alpha.scout.universes.costed_book`
(itself the dollar version of `alpha.scout.specs.dv2._dv2_book_daily`); only the order type of entries and exits changes.
With market-on-open entries and exits it reproduces `costed_book` exactly (tested).

Entry orders (decision after Close_T, session t = T + 1)
    "moo"      buy at Open_t, pay the per-side slippage of row t (max(2.5 bp, half-spread of row T)).
    limit k    day limit buy at L_T = Close_T x (1 - k x NATR14_T / 100), the replica's NATR is in percent, so
               NATR / 100 is the fraction of price; L is rounded DOWN to the nominal tick ($0.01, or $0.0001 below $1)
               of the nominal price Unadjusted Close_T, then converted back to adjusted units.
               Fill on t:  Open_t <= L_T                      -> at Open_t, marketable in the auction: slippage charged
                           else Low_t <= L_T x (1 - m_T)      -> at L_T, passive: commission only
                           else                               -> no fill; the order expires, the slot stays empty for t
               m_T = max(tick / Unadjusted Close_T, 0.1 x half-spread_T): the price must TRADE THROUGH the limit by a
               tick (or a tenth of the half-spread), so a touch never fills.
Exit orders
    "moo"      Close_T > High_(T-1) -> sell at Open_t, slippage charged (today's rule).
    "limit"    once Close_T > High_(T-1) triggers, the position is committed to exit. On each following session s it
               rests a day sell limit at Close_(s-1) rounded UP to the tick:
                           Open_s >= limit                    -> at Open_s, slippage charged
                           else High_s >= limit x (1 + m_(s-1)) -> at the limit, commission only
                           else                               -> carried to s + 1 with the new limit Close_s
               After MAX_EXIT_ATTEMPT_INT unfilled sessions it sells market-on-open on the next session (slippage
               charged). The exit condition is not re-evaluated once committed.
Slots      at the decision of T: n - held + exits CERTAIN to leave at Open_t. With "moo" exits every signalled exit is
           certain (today's rule); with "limit" exits only a forced market-on-open is: a resting sell keeps its slot
           until it fills (daily bars cannot tell whether it will). Orders go to the first `slots` flat candidates of
           row T's NATR ranking (held names, including those with a resting sell, are skipped without a slot).
           Unfilled entry orders are never backfilled intraday.
Sizing     V_T / n / Close_T adjusted shares for every entry type (the replica's sizing; the dollar value at a limit
           fill is shares x L).
Costs      fee max(min_fee, fee_per_share x nominal shares), capped at a fraction of the trade value; every marketable
           fill pays value x slip_mat[t]; passive fills pay the fee only.

*** CRITICAL*** the entry limit, the exit limit and the trade-through margin of an order for session t read rows <= T
(t - 1) only; fills read the bar of t only; the slippage of a fill on row t is the half-spread estimate of row t - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from alpha.scout.universes import _fee_float

MAX_EXIT_ATTEMPT_INT = 5
TRADE_THROUGH_SPREAD_FRACTION_FLOAT = 0.1
FILL_NONE_INT, FILL_OPEN_INT, FILL_PASSIVE_INT, FILL_FORCED_INT = 0, 1, 2, 3
KIND_ENTRY_INT, KIND_UNFILLED_INT, KIND_CANCELLED_INT, KIND_EXIT_INT, KIND_DELIST_INT = 1, 0, 9, -1, -2


# ---------------------------------------------------------------- order prices (rows <= T only)
def nominal_tick_mat(unadjusted_close_mat: np.ndarray) -> np.ndarray:
    """Reg NMS minimum tick at the nominal price of T: $0.01 at or above $1, $0.0001 below."""
    price_mat = np.asarray(unadjusted_close_mat, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(price_mat >= 1.0, 0.01, 0.0001)


def clean_nominal_mat(unadjusted_close_mat: np.ndarray) -> np.ndarray:
    """The nominal close rounded to 4 decimals: Norgate's float32 prices carry ~1e-7 relative noise, which would push a
    price that sits on the tick (10.01 stored as 10.0099998) to the wrong side of a floor or ceiling."""
    return np.round(np.asarray(unadjusted_close_mat, dtype=float), 4)


def entry_limit_mat(close_mat: np.ndarray, unadjusted_close_mat: np.ndarray, natr_percent_mat: np.ndarray, k_float: float) -> np.ndarray:
    """Buy limit for session T+1, on row T, in adjusted units: Close_T x (1 - k x NATR_T / 100), rounded DOWN to the
    nominal tick. NaN where it cannot be computed or is not positive (no order).

    *** CRITICAL*** row T reads Close_T, Unadjusted Close_T and NATR14_T only (all known after the close of T)."""
    close_mat = np.asarray(close_mat, dtype=float)
    nominal_mat = clean_nominal_mat(unadjusted_close_mat)
    tick_mat = nominal_tick_mat(nominal_mat)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_nominal_mat = nominal_mat * (1.0 - k_float * np.asarray(natr_percent_mat, dtype=float) / 100.0)
        # 1e-6 tick tolerance: a limit already on the tick (k = 0) stays there despite float32 prices
        rounded_nominal_mat = np.floor(raw_nominal_mat / tick_mat + 1e-6) * tick_mat
        out_mat = rounded_nominal_mat * (close_mat / nominal_mat)
    out_mat[~np.isfinite(out_mat) | (rounded_nominal_mat <= 0)] = np.nan
    return out_mat


def exit_limit_mat(close_mat: np.ndarray, unadjusted_close_mat: np.ndarray) -> np.ndarray:
    """Sell limit for session T+1, on row T, in adjusted units: Close_T rounded UP to the nominal tick.

    *** CRITICAL*** row T reads Close_T and Unadjusted Close_T only."""
    close_mat = np.asarray(close_mat, dtype=float)
    nominal_mat = clean_nominal_mat(unadjusted_close_mat)
    tick_mat = nominal_tick_mat(nominal_mat)
    with np.errstate(invalid="ignore", divide="ignore"):
        rounded_nominal_mat = np.ceil(nominal_mat / tick_mat - 1e-6) * tick_mat
        out_mat = rounded_nominal_mat * (close_mat / nominal_mat)
    out_mat[~np.isfinite(out_mat) | (out_mat <= 0)] = np.nan
    return out_mat


def trade_through_margin_mat(unadjusted_close_mat: np.ndarray, half_spread_mat_list: list[np.ndarray],
                             spread_fraction_float: float = TRADE_THROUGH_SPREAD_FRACTION_FLOAT) -> np.ndarray:
    """m_T = max(one nominal tick / Unadjusted Close_T, fraction x the LARGEST half-spread estimate at T) as a fraction
    of price. Using the largest estimate keeps the fills identical under every cost model (and conservative).

    *** CRITICAL*** row T reads Unadjusted Close_T and half-spreads at T (estimators of bars <= T)."""
    nominal_mat = np.asarray(unadjusted_close_mat, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tick_fraction_mat = nominal_tick_mat(nominal_mat) / nominal_mat
    spread_mat = np.zeros(nominal_mat.shape)
    for half_spread in half_spread_mat_list:
        spread_mat = np.fmax(spread_mat, np.nan_to_num(np.asarray(half_spread, dtype=float), nan=0.0))
    out_mat = np.fmax(tick_fraction_mat, spread_fraction_float * spread_mat)
    out_mat[~np.isfinite(out_mat)] = np.nan
    return out_mat


# ---------------------------------------------------------------- fill rules on one bar
@njit(cache=False)
def buy_limit_fill(open_float, low_float, limit_float, margin_float):
    """(fill code, price) of a day buy limit on one bar. Open at or below the limit: filled at the open (marketable).
    Else the low must trade THROUGH the limit by the margin: filled at the limit (passive). Else no fill."""
    if not (np.isfinite(limit_float) and np.isfinite(open_float)):
        return 0, np.nan
    if open_float <= limit_float:
        return 1, open_float
    if np.isfinite(low_float) and np.isfinite(margin_float) and low_float <= limit_float * (1.0 - margin_float):
        return 2, limit_float
    return 0, np.nan


@njit(cache=False)
def sell_limit_fill(open_float, high_float, limit_float, margin_float):
    """(fill code, price) of a day sell limit on one bar (mirror of buy_limit_fill)."""
    if not (np.isfinite(limit_float) and np.isfinite(open_float)):
        return 0, np.nan
    if open_float >= limit_float:
        return 1, open_float
    if np.isfinite(high_float) and np.isfinite(margin_float) and high_float >= limit_float * (1.0 + margin_float):
        return 2, limit_float
    return 0, np.nan


# ---------------------------------------------------------------- the book
@njit(cache=False)
def _limit_book(open_mat, high_mat, low_mat, close_mat, pointer_vec, candidate_vec, exit_signal_mat, max_positions_int, start_row_int,
                entry_limit_mat_, entry_mode_int, exit_limit_mat_, exit_mode_int, max_exit_attempt_int, margin_mat, slip_mat,
                share_scale_mat, fee_per_share_float, min_fee_float, max_fee_fraction_float, capital_float, ruin_fraction_float):
    """entry_mode_int 0 = market-on-open, 1 = limit (entry_limit_mat_ row T); exit_mode_int 0 = market-on-open, 1 = limit.
    Log: row, asset, kind (+1 entry, 0 unfilled order, 9 cancelled order (no bar), -1 exit, -2 delisting sale), fill code
    (1 open, 2 passive, 3 forced market-on-open), pre-cost dollar value, spread dollars, fee dollars."""
    row_count_int, asset_count_int = close_mat.shape
    share_vec = np.zeros(asset_count_int)
    exit_state_vec = np.full(asset_count_int, -1, dtype=np.int64)  # -1 not committed; else failed exit sessions so far
    held_vec = np.zeros(max_positions_int, dtype=np.int64)
    entry_vec = np.zeros(max_positions_int, dtype=np.int64)
    size_int = row_count_int * max_positions_int * 2 + 1
    log_row = np.zeros(size_int, dtype=np.int64)
    log_asset = np.zeros(size_int, dtype=np.int64)
    log_kind = np.zeros(size_int, dtype=np.int64)
    log_code = np.zeros(size_int, dtype=np.int64)
    log_value = np.zeros(size_int)
    log_spread = np.zeros(size_int)
    log_fee = np.zeros(size_int)
    log_int, held_int = 0, 0
    entry_weight_float = 1.0 / max_positions_int
    cash_float, previous_total_float = capital_float, capital_float
    daily_vec = np.zeros(row_count_int)
    total_vec = np.full(row_count_int, capital_float)
    for t_int in range(1, row_count_int):
        p_int = t_int - 1
        # ---- decision at the close of T = p: certain exits, then slots
        certain_int = 0
        for k_int in range(held_int):
            a_int = held_vec[k_int]
            if exit_mode_int == 0:
                if exit_signal_mat[p_int, a_int]:
                    certain_int += 1
            else:
                if exit_state_vec[a_int] < 0 and exit_signal_mat[p_int, a_int]:
                    exit_state_vec[a_int] = 0  # committed at the close of T; first resting sell on t
                if exit_state_vec[a_int] >= max_exit_attempt_int:
                    certain_int += 1  # forced market-on-open on t
        slot_int = max_positions_int - held_int + certain_int
        entry_int = 0
        if slot_int > 0 and t_int >= start_row_int:
            for j_int in range(pointer_vec[p_int], pointer_vec[p_int + 1]):
                a_int = candidate_vec[j_int]
                if share_vec[a_int] > 0.0:
                    continue  # held at T (with or without a resting sell): skipped without a slot
                entry_vec[entry_int] = a_int
                entry_int += 1
                if entry_int == slot_int:
                    break
        # ---- session t: held positions
        kept_int = 0
        for k_int in range(held_int):
            a_int = held_vec[k_int]
            code_int = 0
            price_float = np.nan
            if not (np.isfinite(open_mat[t_int, a_int]) and np.isfinite(close_mat[t_int, a_int])):
                q_int = p_int
                while not np.isfinite(close_mat[q_int, a_int]):
                    q_int -= 1
                value_float = share_vec[a_int] * close_mat[q_int, a_int]
                fee_float = _fee_float(share_vec[a_int] * share_scale_mat[q_int, a_int], value_float, fee_per_share_float, min_fee_float, max_fee_fraction_float)
                cash_float += value_float - fee_float
                log_row[log_int], log_asset[log_int], log_kind[log_int], log_code[log_int] = t_int, a_int, -2, 0
                log_value[log_int], log_spread[log_int], log_fee[log_int] = value_float, 0.0, fee_float
                log_int += 1
                share_vec[a_int] = 0.0
                exit_state_vec[a_int] = -1
                continue
            if exit_mode_int == 0:
                if exit_signal_mat[p_int, a_int]:
                    code_int, price_float = 1, open_mat[t_int, a_int]
            elif exit_state_vec[a_int] >= 0:
                if exit_state_vec[a_int] >= max_exit_attempt_int:
                    code_int, price_float = 3, open_mat[t_int, a_int]
                else:
                    code_int, price_float = sell_limit_fill(open_mat[t_int, a_int], high_mat[t_int, a_int], exit_limit_mat_[p_int, a_int],
                                                            margin_mat[p_int, a_int])
                    if code_int == 0:
                        exit_state_vec[a_int] += 1
            if code_int == 0:
                held_vec[kept_int] = a_int
                kept_int += 1
                continue
            value_float = share_vec[a_int] * price_float
            spread_float = value_float * slip_mat[t_int, a_int] if code_int != 2 else 0.0
            fee_float = _fee_float(share_vec[a_int] * share_scale_mat[p_int, a_int], value_float, fee_per_share_float, min_fee_float, max_fee_fraction_float)
            cash_float += value_float - spread_float - fee_float
            log_row[log_int], log_asset[log_int], log_kind[log_int], log_code[log_int] = t_int, a_int, -1, code_int
            log_value[log_int], log_spread[log_int], log_fee[log_int] = value_float, spread_float, fee_float
            log_int += 1
            share_vec[a_int] = 0.0
            exit_state_vec[a_int] = -1
        held_int = kept_int
        # ---- session t: entry orders
        for k_int in range(entry_int):
            a_int = entry_vec[k_int]
            if not (np.isfinite(open_mat[t_int, a_int]) and np.isfinite(close_mat[t_int, a_int])):
                log_row[log_int], log_asset[log_int], log_kind[log_int], log_code[log_int] = t_int, a_int, 9, 0
                log_value[log_int], log_spread[log_int], log_fee[log_int] = 0.0, 0.0, 0.0
                log_int += 1
                continue  # cancelled: no bar to fill on
            if entry_mode_int == 0:
                code_int, price_float = 1, open_mat[t_int, a_int]
            else:
                code_int, price_float = buy_limit_fill(open_mat[t_int, a_int], low_mat[t_int, a_int], entry_limit_mat_[p_int, a_int], margin_mat[p_int, a_int])
            if code_int == 0:
                log_row[log_int], log_asset[log_int], log_kind[log_int], log_code[log_int] = t_int, a_int, 0, 0
                log_value[log_int], log_spread[log_int], log_fee[log_int] = 0.0, 0.0, 0.0
                log_int += 1
                continue  # expired: the slot stays empty this session
            share_vec[a_int] = previous_total_float * entry_weight_float / close_mat[p_int, a_int]
            value_float = share_vec[a_int] * price_float
            spread_float = value_float * slip_mat[t_int, a_int] if code_int != 2 else 0.0
            fee_float = _fee_float(share_vec[a_int] * share_scale_mat[p_int, a_int], value_float, fee_per_share_float, min_fee_float, max_fee_fraction_float)
            cash_float -= value_float + spread_float + fee_float
            log_row[log_int], log_asset[log_int], log_kind[log_int], log_code[log_int] = t_int, a_int, 1, code_int
            log_value[log_int], log_spread[log_int], log_fee[log_int] = value_float, spread_float, fee_float
            log_int += 1
            exit_state_vec[a_int] = -1
            held_vec[held_int] = a_int
            held_int += 1
        total_float = cash_float
        for k_int in range(held_int):
            total_float += share_vec[held_vec[k_int]] * close_mat[t_int, held_vec[k_int]]
        daily_vec[t_int] = total_float / previous_total_float - 1.0
        total_vec[t_int] = total_float
        previous_total_float = total_float
        if total_float <= ruin_fraction_float * capital_float:
            daily_vec[t_int] = max(daily_vec[t_int], -1.0)
            total_vec[t_int:] = max(total_float, 0.0)
            return (daily_vec, total_vec, log_row[:log_int], log_asset[:log_int], log_kind[:log_int], log_code[:log_int], log_value[:log_int],
                    log_spread[:log_int], log_fee[:log_int], t_int)
    return (daily_vec, total_vec, log_row[:log_int], log_asset[:log_int], log_kind[:log_int], log_code[:log_int], log_value[:log_int],
            log_spread[:log_int], log_fee[:log_int], -1)


@dataclass
class LimitResult:
    daily_ser: pd.Series
    total_value_ser: pd.Series
    log_df: pd.DataFrame  # date, asset, kind_int, code_int, value_float, spread_float, fee_float
    ruin_date: pd.Timestamp | None = None


def limit_book(date_index: pd.DatetimeIndex, symbol_list: list, mats: dict, high_mat: np.ndarray, low_mat: np.ndarray, max_positions_int: int,
               start_date_str: str, entry_limit, exit_str: str, margin_mat: np.ndarray, exit_limit: np.ndarray | None, slippage,
               share_scale_mat: np.ndarray, fee_per_share_float: float, min_fee_float: float, max_fee_fraction_float: float,
               capital_float: float = 100_000.0, ruin_fraction_float: float = 0.01, max_exit_attempt_int: int = MAX_EXIT_ATTEMPT_INT) -> LimitResult:
    """Run the book. `mats` = alpha.scout.universes.rule_mats output (open, close, pointer_vec, candidate_vec, exit_signal).
    `entry_limit`: None (market-on-open) or the entry_limit_mat of row T. `exit_str`: "moo" or "limit" (needs exit_limit).
    `slippage`: a per-side float or a (date x symbol) matrix of fill rows."""
    shape_tuple = mats["close"].shape
    slip_mat = np.full(shape_tuple, float(slippage)) if np.isscalar(slippage) else np.asarray(slippage, dtype=float)
    if exit_str not in ("moo", "limit"):
        raise ValueError("exit_str must be 'moo' or 'limit'.")
    if exit_str == "limit" and exit_limit is None:
        raise ValueError("a limit exit needs exit_limit.")
    nan_mat = np.full(shape_tuple, np.nan)
    start_row_int = int(np.searchsorted(date_index.to_numpy(), np.datetime64(pd.Timestamp(start_date_str))))
    out_tuple = _limit_book(
        mats["open"], np.asarray(high_mat, dtype=float), np.asarray(low_mat, dtype=float), mats["close"], mats["pointer_vec"], mats["candidate_vec"],
        mats["exit_signal"], int(max_positions_int), start_row_int,
        nan_mat if entry_limit is None else np.asarray(entry_limit, dtype=float), 0 if entry_limit is None else 1,
        nan_mat if exit_limit is None else np.asarray(exit_limit, dtype=float), 0 if exit_str == "moo" else 1, int(max_exit_attempt_int),
        np.asarray(margin_mat, dtype=float), slip_mat, np.asarray(share_scale_mat, dtype=float), float(fee_per_share_float), float(min_fee_float),
        float(max_fee_fraction_float), float(capital_float), float(ruin_fraction_float))
    daily_vec, total_vec, row_vec, asset_vec, kind_vec, code_vec, value_vec, spread_vec, fee_vec, ruin_row_int = out_tuple
    symbol_arr = np.array(symbol_list, dtype=object)
    log_df = pd.DataFrame({"date": date_index[row_vec], "asset": symbol_arr[asset_vec], "kind_int": kind_vec, "code_int": code_vec,
                           "value_float": value_vec, "spread_float": spread_vec, "fee_float": fee_vec})
    return LimitResult(daily_ser=pd.Series(daily_vec, index=date_index), total_value_ser=pd.Series(total_vec, index=date_index), log_df=log_df,
                       ruin_date=date_index[ruin_row_int] if ruin_row_int >= 0 else None)


# ---------------------------------------------------------------- event-level fills (adverse-selection table)
def event_fill_mats(open_mat: np.ndarray, low_mat: np.ndarray, entry_limit: np.ndarray, margin_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fill code, fill price) on row T for an order decided at T and worked on T+1 (vectorised buy_limit_fill).

    *** CRITICAL*** row T of the output reads the limit/margin of row T and the bar of T+1 (the session the order
    works); it is a label for the event study, never a feature."""
    next_open_mat = np.full(open_mat.shape, np.nan)
    next_low_mat = np.full(low_mat.shape, np.nan)
    next_open_mat[:-1] = open_mat[1:]
    next_low_mat[:-1] = low_mat[1:]
    with np.errstate(invalid="ignore"):
        open_fill_mat = np.isfinite(entry_limit) & np.isfinite(next_open_mat) & (next_open_mat <= entry_limit)
        passive_fill_mat = (np.isfinite(entry_limit) & np.isfinite(next_open_mat) & ~open_fill_mat & np.isfinite(next_low_mat)
                            & np.isfinite(margin_mat) & (next_low_mat <= entry_limit * (1.0 - margin_mat)))
    code_mat = np.where(open_fill_mat, 1, np.where(passive_fill_mat, 2, 0)).astype(np.int8)
    price_mat = np.where(open_fill_mat, next_open_mat, np.where(passive_fill_mat, entry_limit, np.nan))
    return code_mat, price_mat
