"""POST-HOC diagnostic (amendment A3, added after the frozen verdict was computed; cannot change it).

Question: does a size filter that uses each company's LATEST market cap - as Sharadar's TICKERS.scalemarketcap does
("based on the most recent market cap") - reproduce the talk's results? Such a filter is look-ahead: it keeps the
IPOs that later became big.

    last_cap(i) = last known shares outstanding(i) x Unadjusted Close(i) on the last bar on or before that date

Populations (IPO window, ATH, SPAC listings allowed as in the talk):
    LA_BIG      last_cap >= $2B  (Sharadar mid/large/mega), no price or liquidity filter   <- the talk's universe, if built so
    LA_SMALL    last_cap <  $2B
    PIT         our frozen point-in-time universe (IPO_ATH_SP)

Usage: python posthoc_lookahead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import data_build  # noqa: E402
import events  # noqa: E402
import simulate  # noqa: E402
import stage_a  # noqa: E402
from new_pod_search_20260927 import common as npc  # noqa: E402
from trend_breakout_20260927 import common as trend_common  # noqa: E402

BIG_CAP_FLOAT = 2e9
TALK_WINDOW_TUPLE = ("2001-01-01", "2023-12-31")


def last_cap_frame(symbol_list: list[str]) -> pd.DataFrame:
    out_path = common.CACHE_DIR_PATH / "last_cap.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    import norgatedata as nd

    record_list = []
    for symbol_str in symbol_list:
        shares_tuple = nd.sharesoutstanding(symbol_str)
        shares_float, shares_date_str = (shares_tuple if shares_tuple else (np.nan, None))
        bars_df = data_build.load_symbol_bars_df(symbol_str)
        price_float = np.nan
        if bars_df is not None and len(bars_df):
            unadjusted_ser = bars_df["Unadjusted Close"].dropna()
            if shares_date_str:
                unadjusted_ser = unadjusted_ser.loc[:pd.Timestamp(shares_date_str)] if len(unadjusted_ser.loc[:pd.Timestamp(shares_date_str)]) else unadjusted_ser
            price_float = float(unadjusted_ser.iloc[-1]) if len(unadjusted_ser) else np.nan
        record_list.append({"symbol": symbol_str, "shares": shares_float, "shares_date": shares_date_str, "price": price_float,
                            "last_cap": shares_float * price_float if shares_float and np.isfinite(price_float) else np.nan})
    frame_df = pd.DataFrame(record_list)
    frame_df.to_parquet(out_path)
    return frame_df


def main() -> None:
    rows_df = stage_a.add_market_excess(events.load_rows())
    ipo_sym_list = sorted(rows_df.loc[rows_df["in_ipo_window"], "symbol"].unique())
    cap_df = last_cap_frame(ipo_sym_list)
    rows_df = rows_df.merge(cap_df[["symbol", "last_cap"]], on="symbol", how="left")
    ipo_ath_sp_arr = rows_df["in_ipo_window"].to_numpy() & rows_df["ath"].to_numpy()
    big_arr = rows_df["last_cap"].to_numpy() >= BIG_CAP_FLOAT
    small_arr = rows_df["last_cap"].to_numpy() < BIG_CAP_FLOAT
    pit_arr = events.population_mask_dict(rows_df, 1000)["IPO_ATH_SP"]
    pop_dict = {"LA_BIG": ipo_ath_sp_arr & big_arr, "LA_SMALL": ipo_ath_sp_arr & small_arr, "PIT": pit_arr,
                "PIT_and_LA_BIG": pit_arr & big_arr}
    block_dict = dict(stage_a.BLOCK_DICT)
    block_dict["TALK_2001_2023"] = TALK_WINDOW_TUPLE
    result_dict = {"cap_coverage": {"ipo_symbols": len(ipo_sym_list), "with_last_cap": int(cap_df["last_cap"].notna().sum()),
                                    "big_share": float((cap_df["last_cap"] >= BIG_CAP_FLOAT).mean())},
                   "stage_a": {}, "pods": {}}
    for pop_str, mask_arr in pop_dict.items():
        pop_df = rows_df.loc[mask_arr]
        result_dict["stage_a"][pop_str] = {b: stage_a.describe(pop_df[(pop_df["date"] >= s) & (pop_df["date"] <= e)]) for b, (s, e) in block_dict.items()}

    calendar_idx = events.load_calendar_idx()
    start_pos_int = int(calendar_idx.searchsorted(pd.Timestamp("1993-01-04")))
    end_pos_int = int(calendar_idx.searchsorted(pd.Timestamp("2026-08-19"), side="right")) - 1
    bil_ser = npc.load_bil_ret_ser()
    series_dict = {}
    for pop_str in ["LA_BIG", "PIT"]:
        candidates_dict = events.candidates_by_pos(rows_df, pop_dict[pop_str])
        symbol_set = {s for lst in candidates_dict.values() for s, _ in lst}
        store = simulate.BarStore(events.load_bar_store_dict(symbol_set))
        missing_set = symbol_set - set(store.array_dict)
        if missing_set:
            extra_dict = {}
            for symbol_str in missing_set:
                bars_df, _ = __import__("features").clean_bars_df(data_build.load_symbol_bars_df(symbol_str), calendar_idx)
                frame_df = bars_df[data_build.BAR_FIELD_LIST].copy()
                frame_df.insert(0, "cal_pos", calendar_idx.get_indexer(bars_df.index))
                extra_dict[symbol_str] = frame_df.reset_index(drop=True)
            store = simulate.BarStore({**events.load_bar_store_dict(symbol_set), **extra_dict})
        for exit_str in ["E1", "E2"]:
            out_dict = simulate.run_pod(candidates_dict, store, calendar_idx, start_pos_int, end_pos_int,
                                        simulate.SimConfig(20, 0.20, 0.10, exit_mode_str=exit_str))
            sweep_ser = npc.sweep_return_ser(out_dict["return_ser"], out_dict["cash_weight_ser"], bil_ser)
            series_dict[f"{pop_str}|{exit_str}"] = sweep_ser
            result_dict["pods"][f"{pop_str}|{exit_str}"] = {
                "sweep": trend_common.window_metrics(sweep_ser, block_dict),
                "cash_utilization": float(1 - out_dict["cash_weight_ser"].mean()),
                "trades": int(len(out_dict["trade_df"])),
                "mean_trade_ret": float(out_dict["trade_df"]["ret"].mean()),
                "win_rate": float((out_dict["trade_df"]["ret"] > 0).mean()),
            }
    pd.DataFrame(series_dict).to_parquet(common.RESULTS_DIR_PATH / "posthoc_lookahead_returns_sweep.parquet")
    common.write_json("posthoc_lookahead.json", result_dict)
    common.log_progress("posthoc_lookahead: done")


if __name__ == "__main__":
    main()
