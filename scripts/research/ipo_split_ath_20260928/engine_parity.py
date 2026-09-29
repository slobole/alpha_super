"""Engine parity (PREREG section 7): the real repo engine replays the replica's IPO-A0 intents, 2015-2019.

The replica runs from a fresh $100k on the window and logs its intents. ReplayStrategy (engine Strategy, historical
share units on) places, at the bar whose previous_bar is the decision close:
    - for each held position: StopOrder sell-to-0 at S, then LimitOrder sell-to-0 at G (day orders; the engine keeps
      untriggered stops, so every pending order is cleared before new ones are placed);
    - for each entry: order_value(symbol, budget) - a market order sized from the decision close.
Everything else (fills, slippage, commission, dividends, missing-price liquidation) is the engine's own code.
Pricing = the replica's bars with ALLMARKETDAYS-style padding (halts: O=H=L=C=last close, Dividend 0).

Usage: python engine_parity.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import events  # noqa: E402
import simulate  # noqa: E402
from alpha.engine.backtest import run_daily  # noqa: E402
from alpha.engine.strategy import Strategy  # noqa: E402

WINDOW_START_TS = pd.Timestamp("2015-01-02")
WINDOW_END_TS = pd.Timestamp("2019-12-31")
CORR_THRESHOLD_FLOAT = 0.9999
CAGR_TOLERANCE_FLOAT = 0.0005
FIELD_LIST = ["Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"]


class ReplayStrategy(Strategy):
    enable_signal_audit = False

    def __init__(self, intents_by_decision_dict: dict, capital_float: float, slippage_float: float):
        super().__init__(name="ipo_ath_parity", benchmarks=[], capital_base=capital_float, slippage=slippage_float,
                         commission_per_share=simulate.COMMISSION_PER_SHARE_FLOAT,
                         commission_minimum=simulate.COMMISSION_MINIMUM_FLOAT)
        self.historical_share_units_bool = True
        self.intents_by_decision_dict = intents_by_decision_dict
        self.trade_id_int = 0
        self.trade_id_by_symbol_dict: dict = {}

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        self.clear_orders()
        # *** CRITICAL*** intents are keyed by the decision close = previous_bar; fills happen on this bar.
        for kind_str, symbol_str, a_float, b_float in self.intents_by_decision_dict.get(pd.Timestamp(self.previous_bar), []):
            if kind_str == "exit":
                trade_id = self.trade_id_by_symbol_dict.get(symbol_str)
                self.order_target(symbol_str, 0, stop_price=float(a_float), trade_id=trade_id)
                self.order_target(symbol_str, 0, limit_price=float(b_float), trade_id=trade_id)
            elif kind_str == "entry":
                self.trade_id_int += 1
                self.trade_id_by_symbol_dict[symbol_str] = self.trade_id_int
                self.order_value(symbol_str, float(a_float), trade_id=self.trade_id_int)
            else:
                raise ValueError(kind_str)


def padded_pricing_df(store: simulate.BarStore, symbol_list: list[str], calendar_idx: pd.DatetimeIndex,
                      start_pos_int: int, end_pos_int: int) -> pd.DataFrame:
    frame_dict = {}
    for symbol_str in symbol_list:
        value_arr = np.full((end_pos_int - start_pos_int + 1, len(FIELD_LIST)), np.nan)
        for offset_int, cal_pos_int in enumerate(range(start_pos_int, end_pos_int + 1)):
            bar_tuple = store.bar(symbol_str, cal_pos_int)
            if bar_tuple is not None:
                value_arr[offset_int] = bar_tuple
        for field_pos_int, field_str in enumerate(FIELD_LIST):
            frame_dict[(symbol_str, field_str)] = value_arr[:, field_pos_int]
    pricing_df = pd.DataFrame(frame_dict, index=calendar_idx[start_pos_int:end_pos_int + 1])
    pricing_df.columns = pd.MultiIndex.from_tuples(pricing_df.columns)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {s: "CAPITALSPECIAL" for s in symbol_list}
    return pricing_df


def main() -> None:
    started_float = time.time()
    rows_df = events.load_rows()
    calendar_idx = events.load_calendar_idx()
    start_pos_int = int(calendar_idx.searchsorted(WINDOW_START_TS))
    end_pos_int = int(calendar_idx.searchsorted(WINDOW_END_TS, side="right")) - 1
    candidates_dict = events.candidates_by_pos(rows_df, events.population_mask_dict(rows_df, 1000)["IPO_ATH"])
    symbol_set = {s for pos_int, lst in candidates_dict.items() if start_pos_int <= pos_int <= end_pos_int for s, _ in lst}
    store = simulate.BarStore(events.load_bar_store_dict(symbol_set))
    config = simulate.SimConfig(20, 0.20, 0.10, slippage_float=0.00025)
    intent_list: list = []
    replica_dict = simulate.run_pod(candidates_dict, store, calendar_idx, start_pos_int, end_pos_int, config, intent_list)
    traded_symbol_list = sorted({row[2] for row in intent_list if row[1] == "entry"})
    intents_by_decision_dict: dict = defaultdict(list)
    for cal_pos_int, kind_str, symbol_str, a_float, b_float in intent_list:
        intents_by_decision_dict[calendar_idx[cal_pos_int]].append((kind_str, symbol_str, a_float, b_float))
    pricing_df = padded_pricing_df(store, traded_symbol_list, calendar_idx, start_pos_int, end_pos_int)
    strategy_obj = ReplayStrategy(dict(intents_by_decision_dict), config.capital_float, config.slippage_float)
    run_daily(strategy_obj, pricing_df, calendar=pricing_df.index, show_progress=False, show_signal_progress_bool=False,
              audit_override_bool=False)

    engine_nav_ser = strategy_obj.results["total_value"].astype(float)
    engine_nav_ser.index = pd.to_datetime(engine_nav_ser.index)
    replica_nav_ser = config.capital_float * (1.0 + replica_dict["return_ser"]).cumprod()
    common_idx = engine_nav_ser.index.intersection(replica_nav_ser.index)
    engine_ret_vec = engine_nav_ser.reindex(common_idx).pct_change().iloc[1:].to_numpy()
    replica_ret_vec = replica_nav_ser.reindex(common_idx).pct_change().iloc[1:].to_numpy()
    years_float = len(engine_ret_vec) / 252.0

    transaction_df = strategy_obj.get_transactions().copy()
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"])
    engine_trade_set = {(str(a), pd.Timestamp(b), int(np.sign(q))) for a, b, q in zip(transaction_df["asset"], transaction_df["bar"], transaction_df["amount"])}
    replica_trade_df = replica_dict["trade_df"]
    # entries from the intent log (includes positions still open at the window end), exits from closed trades
    replica_trade_set = {(r.symbol, calendar_idx[r.entry_pos], 1) for r in replica_trade_df.itertuples()}
    replica_trade_set |= {(sym_str, calendar_idx[pos_int + 1], 1) for pos_int, kind_str, sym_str, _a, _b in intent_list
                          if kind_str == "entry" and store.bar(sym_str, pos_int + 1) is not None}
    replica_trade_set |= {(r.symbol, calendar_idx[r.exit_pos], -1) for r in replica_trade_df.itertuples() if r.exit_pos <= end_pos_int}
    report_dict = {
        "window": [str(WINDOW_START_TS.date()), str(WINDOW_END_TS.date())],
        "entries_int": int(sum(1 for r in intent_list if r[1] == "entry")),
        "engine_transactions_int": int(len(transaction_df)),
        "daily_return_corr": float(np.corrcoef(engine_ret_vec, replica_ret_vec)[0, 1]),
        "max_abs_daily_diff": float(np.max(np.abs(engine_ret_vec - replica_ret_vec))),
        "cagr_engine": float(np.prod(1 + engine_ret_vec) ** (1 / years_float) - 1),
        "cagr_replica": float(np.prod(1 + replica_ret_vec) ** (1 / years_float) - 1),
        "final_nav_engine": float(engine_nav_ser.iloc[-1]), "final_nav_replica": float(replica_nav_ser.iloc[-1]),
        "trades_only_in_engine": sorted([f"{a}|{b.date()}|{s}" for a, b, s in engine_trade_set - replica_trade_set])[:20],
        "trades_only_in_replica": sorted([f"{a}|{b.date()}|{s}" for a, b, s in replica_trade_set - engine_trade_set])[:20],
        "trade_set_equal_bool": engine_trade_set == replica_trade_set,
        "runtime_seconds": round(time.time() - started_float, 1),
    }
    report_dict["cagr_gap_pp"] = abs(report_dict["cagr_engine"] - report_dict["cagr_replica"]) * 100
    report_dict["passed_bool"] = bool(report_dict["daily_return_corr"] >= CORR_THRESHOLD_FLOAT
                                      and report_dict["cagr_gap_pp"] <= CAGR_TOLERANCE_FLOAT * 100
                                      and report_dict["trade_set_equal_bool"])
    common.write_json("engine_parity.json", report_dict)
    common.log_progress(f"engine_parity: passed={report_dict['passed_bool']} corr={report_dict['daily_return_corr']:.6f} "
                        f"cagr gap {report_dict['cagr_gap_pp']:.4f} pp, trade sets equal {report_dict['trade_set_equal_bool']}")


if __name__ == "__main__":
    main()
