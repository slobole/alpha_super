"""Build immutable numpy caches for the DV2 deep study (research-only).

One cache per universe: dates x symbols arrays of raw Norgate fields, the point-in-time membership mapped to
every trading date exactly as the engine does (latest universe row on or before the date), and, for the S&P 500,
the engine's own DV2 features from `DVO2Strategy.compute_signals` so the replica can prove feature parity.

    member[t, i] = universe_row(max{u <= t})[i]      (get_asof_universe_symbol_list semantics)
    all_fields_valid[t, i] = every raw field of symbol i is non-NaN on date t
                             (the engine's `close.unstack().dropna()` drops a symbol with ANY NaN field)

Usage: python data_cache.py sp500 | sp400 | sp600 | etf
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.norgate_loader import build_index_constituent_matrix, load_raw_prices  # noqa: E402

CACHE_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "dv2_deep_20260925" / "cache"
START_DATE_STR = "1989-01-01"
RAW_FIELD_LIST = ["Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend"]
INDEX_NAME_BY_LABEL_DICT = {"sp500": "S&P 500", "sp400": "S&P MidCap 400", "sp600": "S&P SmallCap 600"}
ETF_LIST = (
    "XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC "
    "XBI IBB SMH SOXX KRE KBE XHB ITB XRT XOP OIH XME GDX IYT IGV ITA IHI XSD XPH "
    "EWJ EWG EWU EWC EWA EWZ EWH EWT EWY EWW EWS EWP EWQ EWI EWL EWN FXI INDA EZA EEM EFA "
    "SPY QQQ IWM DIA MDY IWD IWF IWN IWO"
).split()


def asof_member_arr(universe_df: pd.DataFrame, date_idx: pd.DatetimeIndex, symbol_list: list[str]) -> np.ndarray:
    universe_df = universe_df.sort_index().reindex(columns=symbol_list).fillna(0).astype(np.int8)
    # *** CRITICAL*** Membership on decision date t = latest universe row dated <= t, never a later row.
    row_pos_arr = universe_df.index.searchsorted(date_idx, side="right") - 1
    member_arr = np.zeros((len(date_idx), len(symbol_list)), dtype=bool)
    ok_mask = row_pos_arr >= 0
    member_arr[ok_mask] = universe_df.to_numpy()[row_pos_arr[ok_mask]] == 1
    return member_arr


def build_cache(label_str: str) -> Path:
    started_float = time.time()
    if label_str == "etf":
        symbol_list, universe_df = list(ETF_LIST), None
    else:
        symbol_list, universe_df = build_index_constituent_matrix(indexname=INDEX_NAME_BY_LABEL_DICT[label_str])
    pricing_df = load_raw_prices(list(symbol_list), ["$SPX"], start_date=START_DATE_STR, end_date=None)
    loaded_symbol_list = [s for s in pricing_df.columns.get_level_values(0).unique() if not str(s).startswith("$")]
    date_idx = pd.DatetimeIndex(pricing_df.index)
    array_dict: dict[str, np.ndarray] = {}
    for field_str in RAW_FIELD_LIST:
        frame_df = pricing_df.xs(field_str, axis=1, level=1).reindex(columns=loaded_symbol_list) if field_str in pricing_df.columns.get_level_values(1) else None
        if frame_df is None:
            raise RuntimeError(f"Missing field {field_str}")
        array_dict[field_str] = frame_df.to_numpy(dtype=np.float64)
    field_set_by_symbol = {s: set(pricing_df[s].columns) for s in loaded_symbol_list}
    # *** CRITICAL*** Norgate quirk found 2026-09-25: for some symbols (e.g. VLO) the Dividend field is all zero
    # when the request starts in 1989 but populated when it starts in 1998 (the engine's start). From 1998 on,
    # every field is taken from a 1998-start load so the cache matches the engine; the 1989-1997 rows keep the
    # 1989 load and carry this dividend caveat in the holdout.
    engine_start_df = load_raw_prices(list(symbol_list), ["$SPX"], start_date="1998-01-01", end_date=None)
    later_mask = date_idx >= engine_start_df.index[0]
    changed_dict = {}
    for field_str in RAW_FIELD_LIST:
        later_arr = engine_start_df.xs(field_str, axis=1, level=1).reindex(index=date_idx[later_mask], columns=loaded_symbol_list).to_numpy(dtype=np.float64)
        old_arr = array_dict[field_str][later_mask]
        diff_cols = ~np.all((old_arr == later_arr) | (np.isnan(old_arr) & np.isnan(later_arr)), axis=0)
        changed_dict[field_str] = [loaded_symbol_list[i] for i in np.nonzero(diff_cols)[0]]
        array_dict[field_str][later_mask] = later_arr
    del engine_start_df
    array_dict["all_fields_valid"] = np.ones((len(date_idx), len(loaded_symbol_list)), dtype=bool)
    for field_str in RAW_FIELD_LIST:
        present_arr = np.array([field_str in field_set_by_symbol[s] for s in loaded_symbol_list])
        array_dict["all_fields_valid"] &= np.isfinite(array_dict[field_str]) | ~present_arr[None, :]
    array_dict["spx_close"] = pricing_df[("$SPX", "Close")].to_numpy(dtype=np.float64)
    if universe_df is not None:
        array_dict["member"] = asof_member_arr(universe_df, date_idx, loaded_symbol_list)
    else:
        array_dict["member"] = np.ones((len(date_idx), len(loaded_symbol_list)), dtype=bool)
    if label_str == "sp500":
        # Engine feature parity: the WIRED strategy's own compute_signals on the same frame.
        from strategies.dv2.strategy_mr_dv2 import DVO2Strategy
        strategy_obj = DVO2Strategy(name="cache", benchmarks=["$SPX"], capital_base=1_000_000.0)
        signal_df = strategy_obj.compute_signals(pricing_df)
        for feature_str in ["p126d_return", "natr", "dv2", "sma_200"]:
            array_dict[f"eng_{feature_str}"] = signal_df.xs(feature_str, axis=1, level=1).reindex(columns=loaded_symbol_list).to_numpy(dtype=np.float64)
        del signal_df
    CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR_PATH / f"{label_str}.npz"
    np.savez(out_path, dates=date_idx.values.astype("datetime64[ns]"), symbols=np.array(loaded_symbol_list), **array_dict)
    meta_dict = {"label_str": label_str, "symbol_count_int": len(loaded_symbol_list), "first_date_str": str(date_idx[0].date()),
                 "last_date_str": str(date_idx[-1].date()), "field_list": RAW_FIELD_LIST, "built_at_str": pd.Timestamp.now().isoformat(),
                 "runtime_seconds_float": round(time.time() - started_float, 1), "requested_symbol_count_int": len(symbol_list),
                 "fields_changed_by_1998_start_load": {k: v for k, v in changed_dict.items() if v}}
    (CACHE_DIR_PATH / f"{label_str}_meta.json").write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
    print(json.dumps(meta_dict))
    return out_path


if __name__ == "__main__":
    for label_str in sys.argv[1:]:
        build_cache(label_str)
