"""Daily replica of the rebuilt IPO / split all-time-high rule (PREREG section 5).

Timeline of session s (bar s of the market calendar):

    open s     : market BUYs decided at the close of s-1 fill at Open_s x (1 + slip)
                 E2 only: close-decided exits of s-1 fill at Open_s x (1 - slip)
    during s   : E1 only: day orders placed at the close of s-1 for positions entered on or before s-1
                   stop   S = H_{s-1} x (1 - L)  fills if Low_s <= S at min(Open_s, S)
                   target G = F x (1 + P)         fills if High_s >= G at max(Open_s, G)
                   both touched -> the stop is assumed first
    close s    : dividend entitlement 0.75 x Div_s x q (held at the close; cash posted before the next open),
                 marks at Close_s,
                 H_s = max(H_{s-1}, Close_s), new decisions (entries for s+1; E2 exit flags)

Pod value V_s = cash_s + sum_i q_i x Close_{i,s} (last available close for a halted name).
Budget of a new position b = min(V_s / N, max(cash_s, 0) / free_slots), shares = floor(b / Unadjusted Close_s).
Positions are held in adjusted-share units q = nominal shares x k_s, k = Unadjusted Close / Close.
Commission = max($1, $0.005 x nominal shares on the fill date).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COMMISSION_PER_SHARE_FLOAT = 0.005
COMMISSION_MINIMUM_FLOAT = 1.0
DIVIDEND_NET_RATE_FLOAT = 0.75


@dataclass(frozen=True)
class SimConfig:
    slot_count_int: int = 20
    profit_target_float: float = 0.20
    trailing_stop_float: float = 0.10
    exit_mode_str: str = "E1"  # E1 intraday day orders; E2 close-decided next-open market exits
    entry_mode_str: str = "open"  # "open" (next open) or "close" (label: fill at the decision close)
    slippage_float: float = 0.00025
    capital_float: float = 100_000.0
    terminal_haircut_float: float = 1.0


@dataclass
class Position:
    symbol_str: str
    q_float: float  # adjusted shares
    entry_cal_pos_int: int
    fill_ref_float: float  # entry fill before slippage (adjusted units)
    high_ref_float: float  # H: max(F, closes since entry)
    last_close_float: float
    order_value_float: float
    entry_adv_float: float
    exit_flag_bool: bool = False
    entry_cost_float: float = 0.0


class BarStore:
    """Per-symbol bars keyed by calendar position.

    bar(symbol, s) returns (Open, High, Low, Close, Unadjusted Close, Dividend) for session s:
      - the symbol's own bar when it traded on s;
      - a padded bar (O = H = L = C = last close, same Unadjusted Close, Dividend 0) when s lies between its first and
        last bar but it did not trade (a halt) - the engine's ALLMARKETDAYS padding;
      - None before the first bar or after the last bar (not listed / delisted).
    """

    def __init__(self, bars_by_symbol_dict: dict):
        self.array_dict = {}
        self.cal_pos_dict = {}
        self.first_pos_dict = {}
        self.last_pos_dict = {}
        for symbol_str, bar_df in bars_by_symbol_dict.items():
            arr_dict = {name_str: bar_df[name_str].to_numpy(dtype=float) for name_str in
                        ["Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"]}
            arr_dict["Dividend"] = np.nan_to_num(arr_dict["Dividend"], nan=0.0)
            cal_pos_arr = bar_df["cal_pos"].to_numpy().astype(np.int64)
            self.array_dict[symbol_str] = arr_dict
            self.cal_pos_dict[symbol_str] = cal_pos_arr
            self.first_pos_dict[symbol_str] = int(cal_pos_arr[0])
            self.last_pos_dict[symbol_str] = int(cal_pos_arr[-1])

    def bar(self, symbol_str: str, cal_pos_int: int):
        if cal_pos_int < self.first_pos_dict[symbol_str] or cal_pos_int > self.last_pos_dict[symbol_str]:
            return None
        cal_pos_arr = self.cal_pos_dict[symbol_str]
        row_int = int(np.searchsorted(cal_pos_arr, cal_pos_int, side="right")) - 1
        arr_dict = self.array_dict[symbol_str]
        if cal_pos_arr[row_int] == cal_pos_int:
            return (arr_dict["Open"][row_int], arr_dict["High"][row_int], arr_dict["Low"][row_int],
                    arr_dict["Close"][row_int], arr_dict["Unadjusted Close"][row_int], arr_dict["Dividend"][row_int])
        close_float = arr_dict["Close"][row_int]
        return (close_float, close_float, close_float, close_float, arr_dict["Unadjusted Close"][row_int], 0.0)


def _commission_float(q_float: float, k_float: float) -> float:
    nominal_shares_float = abs(q_float) / k_float
    return max(COMMISSION_MINIMUM_FLOAT, COMMISSION_PER_SHARE_FLOAT * nominal_shares_float)


def run_pod(candidates_by_pos_dict: dict, store: BarStore, calendar_idx: pd.DatetimeIndex, start_pos_int: int,
            end_pos_int: int, config: SimConfig, intent_log_list: list | None = None) -> dict:
    """candidates_by_pos_dict[cal_pos] = list of (symbol, adv) already ranked (adv desc, symbol asc).

    intent_log_list (optional, engine parity): receives (decision cal_pos, kind, symbol, a, b) with kind "exit"
    (a = stop, b = target, E1 day orders for the next session) or "entry" (a = budget in dollars).
    """
    cash_float = config.capital_float
    position_dict: dict[str, Position] = {}
    pending_entry_list: list[tuple[str, float, float, float]] = []  # (symbol, q, adv, budget)
    value_list, cash_weight_list, held_count_list, trade_list = [], [], [], []
    slip_float = config.slippage_float
    commission_total_float = 0.0
    slippage_total_float = 0.0

    def close_position(pos: Position, price_ref_float: float, cal_pos_int: int, reason_str: str, k_float: float,
                       slip_apply_float: float) -> None:
        nonlocal cash_float, commission_total_float, slippage_total_float
        fill_float = price_ref_float * (1.0 - slip_apply_float)
        commission_float = _commission_float(pos.q_float, k_float)
        proceeds_float = pos.q_float * fill_float
        cash_float += proceeds_float - commission_float
        commission_total_float += commission_float
        slippage_total_float += pos.q_float * price_ref_float * slip_apply_float
        trade_list.append({
            "symbol": pos.symbol_str, "entry_pos": pos.entry_cal_pos_int, "exit_pos": cal_pos_int,
            "reason": reason_str, "entry_cost": pos.entry_cost_float, "exit_proceeds": proceeds_float - commission_float,
            "ret": (proceeds_float - commission_float) / pos.entry_cost_float - 1.0,
            "order_value": pos.order_value_float, "entry_adv": pos.entry_adv_float,
        })
        del position_dict[pos.symbol_str]

    pending_dividend_float = 0.0
    for cal_pos_int in range(start_pos_int, end_pos_int + 1):
        # ---- before the open: the previous session's dividend entitlement is posted (engine convention) ---------
        cash_float += pending_dividend_float
        pending_dividend_float = 0.0
        # ---- open: entries decided at the previous close -------------------------------------------------------
        for symbol_str, q_float, adv_float, budget_float in pending_entry_list:
            bar_tuple = store.bar(symbol_str, cal_pos_int)
            if bar_tuple is None:
                continue  # no tradable open (delisted): the order is cancelled, as in the engine
            open_float = bar_tuple[0]
            k_float = bar_tuple[4] / bar_tuple[3]
            fill_float = open_float * (1.0 + slip_float)
            commission_float = _commission_float(q_float, k_float)
            cost_float = q_float * fill_float + commission_float
            cash_float -= cost_float
            commission_total_float += commission_float
            slippage_total_float += q_float * open_float * slip_float
            position_dict[symbol_str] = Position(symbol_str, q_float, cal_pos_int, open_float, open_float, open_float,
                                                 q_float * open_float, adv_float, entry_cost_float=cost_float)
        pending_entry_list = []

        # ---- exits ----------------------------------------------------------------------------------------------
        for symbol_str in list(position_dict):
            pos = position_dict[symbol_str]
            bar_tuple = store.bar(symbol_str, cal_pos_int)
            if bar_tuple is None:
                # *** CRITICAL*** series ended: terminal liquidation at the last close, no slippage, commission at
                # the last bar's scale (engine _liquidate_missing_price_positions); label: x terminal haircut
                close_position(pos, pos.last_close_float * config.terminal_haircut_float, cal_pos_int, "terminal",
                               _last_k_float(store, symbol_str), 0.0)
                continue
            if pos.entry_cal_pos_int >= cal_pos_int:
                continue  # exit orders apply only from the session after the entry session
            open_float, high_float, low_float = bar_tuple[0], bar_tuple[1], bar_tuple[2]
            k_float = bar_tuple[4] / bar_tuple[3]
            if config.exit_mode_str == "E2":
                if pos.exit_flag_bool:
                    close_position(pos, open_float, cal_pos_int, "e2", k_float, slip_float)
                continue
            # *** CRITICAL*** stop and target levels come from information through the previous close only.
            stop_float = pos.high_ref_float * (1.0 - config.trailing_stop_float)
            target_float = pos.fill_ref_float * (1.0 + config.profit_target_float)
            if low_float <= stop_float:
                close_position(pos, min(open_float, stop_float), cal_pos_int, "stop", k_float, slip_float)
            elif high_float >= target_float:
                close_position(pos, max(open_float, target_float), cal_pos_int, "target", k_float, slip_float)

        # ---- close: dividends, marks, trailing reference ---------------------------------------------------------
        held_value_float = 0.0
        for symbol_str, pos in position_dict.items():
            bar_tuple = store.bar(symbol_str, cal_pos_int)
            if bar_tuple is not None:
                dividend_float = bar_tuple[5]
                if dividend_float > 0:
                    # *** CRITICAL*** entitlement = shares held at this close x Dividend of this bar; the cash is
                    # posted before the next open, exactly as alpha/engine/strategy.py::_credit_dividend_cash_before_open
                    pending_dividend_float += DIVIDEND_NET_RATE_FLOAT * dividend_float * pos.q_float
                close_float = bar_tuple[3]
                pos.last_close_float = close_float
                pos.high_ref_float = max(pos.high_ref_float, close_float)
                if config.exit_mode_str == "E2":
                    stop_float = pos.high_ref_float * (1.0 - config.trailing_stop_float)
                    target_float = pos.fill_ref_float * (1.0 + config.profit_target_float)
                    pos.exit_flag_bool = close_float <= stop_float or close_float >= target_float
            held_value_float += pos.q_float * pos.last_close_float
        value_float = cash_float + held_value_float

        if intent_log_list is not None and config.exit_mode_str == "E1" and cal_pos_int < end_pos_int:
            for symbol_str, pos in position_dict.items():
                intent_log_list.append((cal_pos_int, "exit", symbol_str, pos.high_ref_float * (1.0 - config.trailing_stop_float),
                                        pos.fill_ref_float * (1.0 + config.profit_target_float)))

        # ---- close: entry decisions for the next session -----------------------------------------------------------
        free_int = config.slot_count_int - len(position_dict)
        if free_int > 0 and cal_pos_int < end_pos_int:
            for symbol_str, adv_float in candidates_by_pos_dict.get(cal_pos_int, []):
                if free_int <= 0:
                    break
                if symbol_str in position_dict:
                    continue
                bar_tuple = store.bar(symbol_str, cal_pos_int)
                if bar_tuple is None:
                    continue
                budget_float = min(value_float / config.slot_count_int, max(cash_float, 0.0) / free_int)
                unadjusted_close_float = bar_tuple[4]
                close_float = bar_tuple[3]
                nominal_shares_int = int(np.floor(budget_float / unadjusted_close_float))
                if nominal_shares_int < 1:
                    continue
                q_float = nominal_shares_int * unadjusted_close_float / close_float
                if config.entry_mode_str == "close":
                    # label only: fill at the decision close (not tradable without a pre-close order)
                    k_float = unadjusted_close_float / close_float
                    commission_float = _commission_float(q_float, k_float)
                    cost_float = q_float * close_float * (1.0 + slip_float) + commission_float
                    cash_float -= cost_float
                    commission_total_float += commission_float
                    slippage_total_float += q_float * close_float * slip_float
                    position_dict[symbol_str] = Position(symbol_str, q_float, cal_pos_int, close_float, close_float,
                                                         close_float, q_float * close_float, adv_float, entry_cost_float=cost_float)
                    value_float -= q_float * close_float * slip_float + commission_float
                else:
                    pending_entry_list.append((symbol_str, q_float, adv_float, budget_float))
                    if intent_log_list is not None:
                        intent_log_list.append((cal_pos_int, "entry", symbol_str, budget_float, 0.0))
                free_int -= 1

        value_list.append(value_float)
        cash_weight_list.append(max(cash_float, 0.0) / value_float if value_float > 0 else 0.0)
        held_count_list.append(len(position_dict))

    index = calendar_idx[start_pos_int:end_pos_int + 1]
    value_ser = pd.Series(value_list, index=index)
    return_ser = value_ser.pct_change()
    return_ser.iloc[0] = value_ser.iloc[0] / config.capital_float - 1.0
    return {
        "return_ser": return_ser,
        "cash_weight_ser": pd.Series(cash_weight_list, index=index),
        "held_count_ser": pd.Series(held_count_list, index=index),
        "trade_df": pd.DataFrame(trade_list),
        "commission_total_float": commission_total_float,
        "slippage_total_float": slippage_total_float,
    }


def _last_k_float(store: BarStore, symbol_str: str) -> float:
    arr_dict = store.array_dict[symbol_str]
    return float(arr_dict["Unadjusted Close"][-1] / arr_dict["Close"][-1])
