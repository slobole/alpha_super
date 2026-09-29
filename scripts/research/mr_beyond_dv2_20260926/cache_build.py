"""Immutable numpy caches for the MR-beyond-DV2 study (research-only).

Stock universes follow the DV2 deep study's cache contract exactly (same fields, engine membership semantics,
1998-start reload for the Norgate dividend quirk) so results are comparable; the S&P 500 / 400 / 600 caches of that
study are reused read-only. New here: Russell 1000, Nasdaq-100 and a cross-asset ETF panel with $VIX and $SPX.

    member[t, i] = universe_row(max{u <= t})[i]      (engine get_asof_universe_symbol_list semantics)

Usage: python cache_build.py r1000 ndx etfx
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
DV2_DEEP_DIR_PATH = REPO_ROOT_PATH / "scripts" / "research" / "dv2_deep_20260925"
for path in (REPO_ROOT_PATH, DV2_DEEP_DIR_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.norgate_loader import build_index_constituent_matrix, load_price_timeseries, load_raw_prices  # noqa: E402
from data_cache import RAW_FIELD_LIST, START_DATE_STR, asof_member_arr  # noqa: E402  (DV2 deep study)

STUDY_OUT_PATH = REPO_ROOT_PATH / "results" / "research" / "mr_beyond_dv2_20260926"
CACHE_DIR_PATH = STUDY_OUT_PATH / "cache"
DV2_CACHE_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "dv2_deep_20260925" / "cache"
INDEX_NAME_BY_LABEL_DICT = {"r1000": "Russell 1000", "ndx": "Nasdaq 100"}
REUSED_LABEL_SET = {"sp500", "sp400", "sp600", "etf"}

# Frozen lists (SPEC_FROZEN.md): hedges, family C classes, family D pairs.
SECTOR_SPDR_LIST = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
CLASS_ETF_DICT = {
    "bonds": ["TLT", "IEF", "LQD", "HYG", "TIP", "EMB", "AGG"],
    "commodities": ["GLD", "SLV", "USO", "DBC", "DBA", "UNG"],
    "currencies": ["UUP", "FXE", "FXY", "FXA", "FXC", "FXB", "FXF"],
    "real_estate": ["VNQ", "IYR"],
    "equity": ["SPY", "QQQ", "IWM", "MDY", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ", "FXI",
               "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
}
PAIR_GROUP_DICT = {
    "tech": (["SMH", "SOXX", "IGV", "XSD"], "XLK"),
    "health": (["XBI", "IBB", "IHI", "XPH"], "XLV"),
    "financials": (["KRE", "KBE"], "XLF"),
    "energy": (["XOP", "OIH"], "XLE"),
    "materials": (["XME"], "XLB"),
    "consumer": (["XHB", "ITB", "XRT"], "XLY"),
    "industrials": (["ITA", "IYT"], "XLI"),
    "developed": (["EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWP", "EWQ", "EWI", "EWL", "EWN"], "EFA"),
    "emerging": (["EWZ", "EWT", "EWY", "EWW", "FXI", "INDA", "EZA"], "EEM"),
    "size": (["IWM", "MDY"], "SPY"),
}
SIZE_ETF_LIST = ["MDY", "IWM", "IJR", "DIA"]
INDEX_SYMBOL_LIST = ["$VIX", "$SPX"]


def etfx_symbol_list() -> list[str]:
    symbol_list = ["SPY"] + SECTOR_SPDR_LIST + SIZE_ETF_LIST
    for etf_list in CLASS_ETF_DICT.values():
        symbol_list += etf_list
    for narrow_list, broad_str in PAIR_GROUP_DICT.values():
        symbol_list += narrow_list + [broad_str]
    return list(dict.fromkeys(symbol_list))


def _save_arrays(label_str: str, date_idx: pd.DatetimeIndex, symbol_list: list[str], array_dict: dict, meta_dict: dict) -> None:
    out_dir_path = CACHE_DIR_PATH / label_str
    out_dir_path.mkdir(parents=True, exist_ok=True)
    np.save(out_dir_path / "dates.npy", date_idx.values.astype("datetime64[ns]"))
    np.save(out_dir_path / "symbols.npy", np.array(symbol_list))
    for key_str, value_arr in array_dict.items():
        np.save(out_dir_path / f"{key_str}.npy", value_arr)
    (out_dir_path / "_done").write_text("ok")
    (out_dir_path / "meta.json").write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
    print(json.dumps(meta_dict))


def _field_arrays(pricing_df: pd.DataFrame, symbol_list: list[str]) -> dict:
    array_dict = {}
    for field_str in RAW_FIELD_LIST:
        array_dict[field_str] = pricing_df.xs(field_str, axis=1, level=1).reindex(columns=symbol_list).to_numpy(dtype=np.float64)
    return array_dict


def _reload_from_1998(array_dict: dict, request_list: list[str], loaded_symbol_list: list[str], date_idx: pd.DatetimeIndex,
                      benchmark_list: list[str]) -> None:
    # *** CRITICAL*** Same Norgate dividend-quirk handling as the DV2 deep cache: from 1998 on every field comes from
    # a 1998-start load (the engine's start), so the two studies see identical data.
    engine_start_df = load_raw_prices(request_list, benchmark_list, start_date="1998-01-01", end_date=None)
    later_mask = date_idx >= engine_start_df.index[0]
    for field_str in RAW_FIELD_LIST:
        array_dict[field_str][later_mask] = engine_start_df.xs(field_str, axis=1, level=1).reindex(
            index=date_idx[later_mask], columns=loaded_symbol_list).to_numpy(dtype=np.float64)


def build_stock_cache(label_str: str) -> None:
    started_float = time.time()
    symbol_list, universe_df = build_index_constituent_matrix(indexname=INDEX_NAME_BY_LABEL_DICT[label_str])
    pricing_df = load_raw_prices(list(symbol_list), ["$SPX"], start_date=START_DATE_STR, end_date=None)
    loaded_symbol_list = [s for s in pricing_df.columns.get_level_values(0).unique() if not str(s).startswith("$")]
    date_idx = pd.DatetimeIndex(pricing_df.index)
    array_dict = _field_arrays(pricing_df, loaded_symbol_list)
    field_set_by_symbol = {s: set(pricing_df[s].columns) for s in loaded_symbol_list}
    _reload_from_1998(array_dict, list(symbol_list), loaded_symbol_list, date_idx, ["$SPX"])
    valid_arr = np.ones((len(date_idx), len(loaded_symbol_list)), dtype=bool)
    for field_str in RAW_FIELD_LIST:
        present_arr = np.array([field_str in field_set_by_symbol[s] for s in loaded_symbol_list])
        valid_arr &= np.isfinite(array_dict[field_str]) | ~present_arr[None, :]
    array_dict["all_fields_valid"] = valid_arr
    array_dict["spx_close"] = pricing_df[("$SPX", "Close")].to_numpy(dtype=np.float64)
    array_dict["member"] = asof_member_arr(universe_df, date_idx, loaded_symbol_list)
    meta_dict = {"label_str": label_str, "index_name_str": INDEX_NAME_BY_LABEL_DICT[label_str],
                 "symbol_count_int": len(loaded_symbol_list), "requested_symbol_count_int": len(symbol_list),
                 "first_date_str": str(date_idx[0].date()), "last_date_str": str(date_idx[-1].date()),
                 "first_membership_row_str": str(universe_df.index.min().date()), "field_list": RAW_FIELD_LIST,
                 "built_at_str": pd.Timestamp.now().isoformat(), "runtime_seconds_float": round(time.time() - started_float, 1)}
    _save_arrays(label_str, date_idx, loaded_symbol_list, array_dict, meta_dict)


def build_etfx_cache() -> None:
    started_float = time.time()
    symbol_list = etfx_symbol_list()
    pricing_df = load_raw_prices(symbol_list, [], start_date=START_DATE_STR, end_date=None)
    loaded_symbol_list = [s for s in pricing_df.columns.get_level_values(0).unique()]
    # The ETF frame starts at the first ETF (SPY, 1993); the stock caches start 1989. Use the $SPX market calendar
    # so every stock date has a row (ETF fields are NaN before each fund exists).
    date_idx = pd.DatetimeIndex(load_price_timeseries("$SPX", start_date_str=START_DATE_STR).index)
    pricing_df = pricing_df.reindex(date_idx)
    array_dict = _field_arrays(pricing_df, loaded_symbol_list)
    _reload_from_1998(array_dict, symbol_list, loaded_symbol_list, date_idx, [])
    array_dict["all_fields_valid"] = np.ones((len(date_idx), len(loaded_symbol_list)), dtype=bool)
    for field_str in RAW_FIELD_LIST:
        array_dict["all_fields_valid"] &= np.isfinite(array_dict[field_str])
    array_dict["member"] = array_dict["all_fields_valid"].copy()
    # $VIX and $SPX index levels read directly (not through the benchmark path, which maps $SPX to $SPXTR):
    # $SPX here is the PRICE index, used only as the pre-1993 SPY hedge proxy; $VIX only for regime labels.
    for symbol_str in INDEX_SYMBOL_LIST:
        key_str = symbol_str.strip("$").lower()
        index_df = load_price_timeseries(symbol_str, start_date_str=START_DATE_STR)
        array_dict[f"{key_str}_close"] = index_df["Close"].reindex(date_idx).to_numpy(dtype=np.float64)
        array_dict[f"{key_str}_open"] = index_df["Open"].reindex(date_idx).to_numpy(dtype=np.float64)
    first_valid_dict = {}
    for i, s in enumerate(loaded_symbol_list):
        ok_arr = np.nonzero(array_dict["all_fields_valid"][:, i])[0]
        first_valid_dict[s] = str(date_idx[ok_arr[0]].date()) if ok_arr.size else None
    meta_dict = {"label_str": "etfx", "symbol_count_int": len(loaded_symbol_list), "requested_symbol_list": symbol_list,
                 "missing_symbol_list": sorted(set(symbol_list) - set(loaded_symbol_list)),
                 "first_valid_date_by_symbol": first_valid_dict, "first_date_str": str(date_idx[0].date()),
                 "last_date_str": str(date_idx[-1].date()), "field_list": RAW_FIELD_LIST,
                 "built_at_str": pd.Timestamp.now().isoformat(), "runtime_seconds_float": round(time.time() - started_float, 1)}
    _save_arrays("etfx", date_idx, loaded_symbol_list, array_dict, meta_dict)


if __name__ == "__main__":
    for label_str in sys.argv[1:]:
        if label_str == "etfx":
            build_etfx_cache()
        else:
            build_stock_cache(label_str)
