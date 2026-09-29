"""Streaming detection over the Russell 3000 and the confirmed-symbol panel (research only; PREREG section 3).

Pass 1 (all Russell 3000 Current & Past symbols, one at a time through norgatedata): membership = Russell 1000 or
Russell 2000 constituent at d (as-of, forward-filled, never back-filled, with the repo convention of
build_index_constituent_matrix: only member rows are kept and a symbol whose membership ends before the calendar's last
session loses its last five member rows); ADV20_{d-1} of every symbol is kept in one float32 date x symbol array
(12,401 x 6,975 x 4 bytes = 346 MB) together with the int8 membership array, so the daily 25th percentile among members
is EXACT (numpy quantile over the members' ADV values of each day). Symbols with at least one jump + volume + membership
candidate at the loosest jump (10%) are listed.
Pass 2 (candidate symbols only): the liquidity test and the pin tests at the loosest grid settings (theta 0.8%, W in
{3, 5, 10}, both J); symbols with at least one confirmation are kept and their Open / Close / Unadjusted Close /
Dividend / Turnover series (float64) form the simulation panel, with the R3000, R1000 and R2000-only membership panels
and the daily q25 vector. Detection statistics (events and confirmations by year for every grid setting) are recorded.
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from merger_arb_v2_20260927 import cells as cells_module
from merger_arb_v2_20260927 import common
from merger_arb_v2_20260927.features import adv20_prev_arr, candidate_event_arr, confirmation_dict, pin_feature_dict

R3000_WATCHLIST_STR = "Russell 3000 Current & Past"
INDEX_NAME_TUPLE = ("Russell 1000", "Russell 2000")
FIELD_KEY_DICT = {"Open": "open_arr", "Close": "close_arr", "Unadjusted Close": "unadjusted_close_arr", "Dividend": "dividend_arr", "Turnover": "turnover_arr"}
STALE_MEMBERSHIP_DROP_INT = 5
LOOSEST_JUMP_FLOAT = min(cells_module.P_JUMP_TUPLE)
LOOSEST_THETA_FLOAT = max(cells_module.P_THETA_TUPLE)


def panel_path():
    return common.CACHE_DIR_PATH / "panel_R3000_confirmed.pkl"


def calendar_index() -> pd.DatetimeIndex:
    import norgatedata as nd

    spx_df = nd.price_timeseries("$SPX", padding_setting=nd.PaddingType.ALLMARKETDAYS, start_date=common.HISTORY_START_STR, end_date=None, timeseriesformat="pandas-dataframe")
    return pd.DatetimeIndex(spx_df.index)


def load_symbol_frame(symbol_str: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    import norgatedata as nd

    price_df = nd.price_timeseries(symbol_str, stock_price_adjustment_setting=nd.StockPriceAdjustmentType.CAPITALSPECIAL, padding_setting=nd.PaddingType.ALLMARKETDAYS,
                                   start_date=common.HISTORY_START_STR, end_date=None, timeseriesformat="pandas-dataframe")
    if price_df is None or len(price_df) == 0:
        return None
    return price_df.reindex(date_index)


def member_vec_from_series(constituent_ser: pd.Series, date_index: pd.DatetimeIndex, last_trading_day_ts: pd.Timestamp) -> np.ndarray:
    """Repo convention (data/norgate_loader.py::build_index_constituent_matrix): keep the rows where the flag is 1; if
    the symbol's last such row is not the calendar's last session, drop its last five rows; 0 elsewhere; as-of on the
    calendar (the flags are daily, so no forward fill beyond the series' own dates is needed)."""
    member_ser = constituent_ser[constituent_ser == 1]
    if len(member_ser) == 0:
        return np.zeros(len(date_index), dtype=np.int8)
    if pd.Timestamp(member_ser.index[-1]) != last_trading_day_ts:
        member_ser = member_ser.iloc[:-STALE_MEMBERSHIP_DROP_INT]
    # *** CRITICAL *** as-of alignment: a flag dated d applies to d; nothing is back-filled.
    return member_ser.reindex(date_index).fillna(0).to_numpy(dtype=np.int8)


def load_membership(symbol_str: str, date_index: pd.DatetimeIndex, last_trading_day_ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    import norgatedata as nd

    vec_list = []
    for index_name_str in INDEX_NAME_TUPLE:
        constituent_df = nd.index_constituent_timeseries(symbol_str, index_name_str, timeseriesformat="pandas-dataframe")
        constituent_ser = constituent_df["Index Constituent"] if constituent_df is not None and len(constituent_df) else pd.Series(dtype=float)
        vec_list.append(member_vec_from_series(constituent_ser, date_index, last_trading_day_ts))
    return vec_list[0], vec_list[1]


def union_member_vec(r1000_vec: np.ndarray, r2000_vec: np.ndarray) -> np.ndarray:
    return ((r1000_vec == 1) | (r2000_vec == 1)).astype(np.int8)


def run_pass1() -> dict:
    import norgatedata as nd

    start_float = time.perf_counter()
    date_index = calendar_index()
    last_ts = pd.Timestamp(date_index[-1])
    symbol_list = [str(s) for s in nd.watchlist_symbols(R3000_WATCHLIST_STR)]
    adv_arr = np.full((len(date_index), len(symbol_list)), np.nan, dtype=np.float32)
    member_arr = np.zeros((len(date_index), len(symbol_list)), dtype=np.int8)
    candidate_list: list[str] = []
    no_price_list: list[str] = []
    for idx, symbol_str in enumerate(symbol_list):
        frame_df = load_symbol_frame(symbol_str, date_index)
        if frame_df is None:
            no_price_list.append(symbol_str)
            continue
        r1000_vec, r2000_vec = load_membership(symbol_str, date_index, last_ts)
        member_vec = union_member_vec(r1000_vec, r2000_vec)
        member_arr[:, idx] = member_vec
        turnover_vec = frame_df["Turnover"].to_numpy(dtype=np.float64)
        close_vec = frame_df["Close"].to_numpy(dtype=np.float64)
        adv_arr[:, idx] = adv20_prev_arr(turnover_vec).astype(np.float32)
        if candidate_event_arr(close_vec, turnover_vec, member_vec, LOOSEST_JUMP_FLOAT).any():
            candidate_list.append(symbol_str)
        if idx % 1000 == 0:
            print(f"  pass 1: {idx}/{len(symbol_list)} symbols, {len(candidate_list)} candidates, {time.perf_counter() - start_float:.0f}s", flush=True)
    # exact daily 25th percentile of ADV20_{d-1} among the members at d
    q25_vec = np.full(len(date_index), np.nan)
    member_count_vec = np.zeros(len(date_index), dtype=np.int64)
    for pos_int in range(len(date_index)):
        pool_vec = adv_arr[pos_int][(member_arr[pos_int] == 1) & np.isfinite(adv_arr[pos_int])].astype(np.float64)
        member_count_vec[pos_int] = int((member_arr[pos_int] == 1).sum())
        if len(pool_vec):
            q25_vec[pos_int] = float(np.quantile(pool_vec, 0.25))
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    np.save(common.CACHE_DIR_PATH / "q25_adv20_prev_vec.npy", q25_vec)
    pass1_dict = {"symbol_count_int": len(symbol_list), "no_price_symbols_int": len(no_price_list), "candidate_symbols_int": len(candidate_list),
                  "members_per_day": {"min": int(member_count_vec[member_count_vec > 0].min()), "median": float(np.median(member_count_vec)), "max": int(member_count_vec.max()),
                                      "first_day": int(member_count_vec[0]), "last_day": int(member_count_vec[-1])},
                  "q25_method": "exact numpy quantile per day over a float32 date x symbol ADV20_{d-1} array of all 12,401 symbols (346 MB) and an int8 membership array",
                  "q25_examples_usd": {str(date_index[p].date()): float(q25_vec[p]) for p in (300, 2500, 5000, len(date_index) - 40)},
                  "seconds": time.perf_counter() - start_float, "candidate_symbols": candidate_list}
    (common.CACHE_DIR_PATH / "pass1.json").write_text(json.dumps(pass1_dict, indent=1))
    # data check: my Russell 1000 membership against the trend study's Russell 1000 cache (same repo convention)
    try:
        from new_pod_search_20260927 import data as pod_data

        r1000_cache = pod_data.load_base_universe("R1000")
        cache_idx = {s: i for i, s in enumerate(r1000_cache["symbol_list"])}
        mismatch_int, compared_int = 0, 0
        for symbol_str in list(cache_idx)[:400]:
            if symbol_str not in symbol_list:
                continue
            r1000_vec, _ = load_membership(symbol_str, date_index, last_ts)
            mismatch_int += int((r1000_vec != r1000_cache["member_arr"][:, cache_idx[symbol_str]]).sum())
            compared_int += 1
        pass1_dict["membership_check_vs_r1000_cache"] = {"symbols_compared_int": compared_int, "day_mismatches_int": mismatch_int}
        del r1000_cache
    except Exception as exc:  # noqa: BLE001
        pass1_dict["membership_check_vs_r1000_cache"] = {"error": str(exc)}
    (common.CACHE_DIR_PATH / "pass1.json").write_text(json.dumps(pass1_dict, indent=1))
    common.log_progress(f"pass 1 done: {len(symbol_list)} symbols, {len(candidate_list)} candidates, members/day median {pass1_dict['members_per_day']['median']:.0f}, "
                        f"membership check {pass1_dict['membership_check_vs_r1000_cache']}, {time.perf_counter() - start_float:.0f}s")
    return pass1_dict


def run_pass2() -> dict:
    start_float = time.perf_counter()
    pass1_dict = json.loads((common.CACHE_DIR_PATH / "pass1.json").read_text())
    q25_vec = np.load(common.CACHE_DIR_PATH / "q25_adv20_prev_vec.npy")
    date_index = calendar_index()
    last_ts = pd.Timestamp(date_index[-1])
    year_vec = date_index.year.to_numpy()
    kept: dict[str, dict] = {}
    event_by_year: dict[str, dict] = {}
    conf_by_year: dict[str, dict] = {}
    stage_settings = sorted({(c.jump_float, c.theta_float, c.window_int) for c in cells_module.grid_cells()})
    for idx, symbol_str in enumerate(pass1_dict["candidate_symbols"]):
        frame_df = load_symbol_frame(symbol_str, date_index)
        if frame_df is None:
            continue
        r1000_vec, r2000_vec = load_membership(symbol_str, date_index, last_ts)
        member_vec = union_member_vec(r1000_vec, r2000_vec)
        close_vec = frame_df["Close"].to_numpy(dtype=np.float64)
        turnover_vec = frame_df["Turnover"].to_numpy(dtype=np.float64)
        adv_vec = adv20_prev_arr(turnover_vec)
        with np.errstate(invalid="ignore"):
            rel25_vec = np.isfinite(adv_vec) & np.isfinite(q25_vec) & (adv_vec >= q25_vec)
        event_dict = {j: candidate_event_arr(close_vec, turnover_vec, member_vec, j) & rel25_vec for j in cells_module.P_JUMP_TUPLE}
        pin_dict = {w: pin_feature_dict(close_vec, turnover_vec, w) for w in cells_module.S_WINDOW_TUPLE}
        keep_bool = False
        for j_float, event_vec in event_dict.items():
            if event_vec.any():
                for y, n in zip(*np.unique(year_vec[event_vec], return_counts=True)):
                    event_by_year.setdefault(f"J{j_float * 100:g}", {}).setdefault(str(int(y)), 0)
                    event_by_year[f"J{j_float * 100:g}"][str(int(y))] += int(n)
        for j_float, theta_float, w_int in stage_settings:
            conf = confirmation_dict(event_dict[j_float], pin_dict[w_int], theta_float, w_int)
            if conf["conf"].any():
                key_str = f"J{j_float * 100:g}|th{theta_float * 100:g}|W{w_int}"
                for y, n in zip(*np.unique(year_vec[conf["conf"]], return_counts=True)):
                    conf_by_year.setdefault(key_str, {}).setdefault(str(int(y)), 0)
                    conf_by_year[key_str][str(int(y))] += int(n)
                keep_bool = True
        if keep_bool:
            kept[symbol_str] = {FIELD_KEY_DICT[f]: frame_df[f].to_numpy(dtype=np.float64) for f in FIELD_KEY_DICT}
            kept[symbol_str]["member_vec"] = member_vec
            kept[symbol_str]["r1000_vec"] = r1000_vec
            kept[symbol_str]["r2000_only_vec"] = ((r2000_vec == 1) & (r1000_vec == 0)).astype(np.int8)
        if idx % 500 == 0:
            print(f"  pass 2: {idx}/{len(pass1_dict['candidate_symbols'])} candidates, {len(kept)} kept, {time.perf_counter() - start_float:.0f}s", flush=True)
    symbol_list = sorted(kept)
    panel_dict = {"universe_str": "R3000_confirmed", "date_index": date_index, "symbol_list": symbol_list, "panel_dtype_str": "float64"}
    for key_str in FIELD_KEY_DICT.values():
        panel_dict[key_str] = np.column_stack([kept[s][key_str] for s in symbol_list])
    panel_dict["member_arr"] = np.column_stack([kept[s]["member_vec"] for s in symbol_list])
    panel_dict["r1000_member_arr"] = np.column_stack([kept[s]["r1000_vec"] for s in symbol_list])
    panel_dict["r2000_only_member_arr"] = np.column_stack([kept[s]["r2000_only_vec"] for s in symbol_list])
    panel_dict["q25_adv20_prev_vec"] = q25_vec
    import norgatedata as nd

    spy_df = nd.price_timeseries("SPY", stock_price_adjustment_setting=nd.StockPriceAdjustmentType.CAPITALSPECIAL, padding_setting=nd.PaddingType.ALLMARKETDAYS,
                                 start_date=common.HISTORY_START_STR, end_date=None, timeseriesformat="pandas-dataframe").reindex(date_index)
    panel_dict["spy_close_ser"] = spy_df["Close"].astype(float)
    panel_dict["month_end_index"] = pd.DatetimeIndex(pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max().to_numpy())
    with open(panel_path(), "wb") as file_obj:
        pickle.dump(panel_dict, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    stats_dict = {"candidate_symbols_int": len(pass1_dict["candidate_symbols"]), "kept_symbols_int": len(symbol_list), "events_by_year_rel25": event_by_year,
                  "confirmations_by_year": conf_by_year, "panel_bytes": int(sum(panel_dict[k].nbytes for k in FIELD_KEY_DICT.values())), "seconds": time.perf_counter() - start_float}
    (common.CACHE_DIR_PATH / "detection_stats.json").write_text(json.dumps(stats_dict, indent=1))
    common.log_progress(f"pass 2 done: {len(symbol_list)} symbols kept of {len(pass1_dict['candidate_symbols'])} candidates, panel {stats_dict['panel_bytes'] / 1e6:.0f} MB, {time.perf_counter() - start_float:.0f}s")
    return stats_dict


def load_panel() -> dict:
    with open(panel_path(), "rb") as file_obj:
        return pickle.load(file_obj)
