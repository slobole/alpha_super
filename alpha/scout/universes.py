"""Multi-index point-in-time universes on one price panel (the DV2 size ladder, 2026-10-02).

Building one Scout panel per index (alpha.scout.panel.build_panel) would load the same stocks many times: the Russell
2000 alone has 11k "Current & Past" symbols. Here one SUPERSET price panel holds every symbol that was a member of any
of the requested indexes inside the window, and each index keeps its own exact membership matrix on the same
(date x symbol) axes. `universe_panel` hands research code an ordinary `alpha.scout.panel.Panel` for one index (its
ever-members, its membership), so every Scout consumer (`dv2.fast_daily_list_panel`, `stations.s3_edge.run_s3`,
`null.permuted_panel`) works unchanged.

Data contract (as build_panel):
    prices      data.norgate_loader.load_price_timeseries per symbol, CAPITALSPECIAL, Norgate's ALLMARKETDAYS padding,
                fields FIELD_TUPLE in the loader's float32; the date axis is the union of every loaded symbol's dates
    membership  norgatedata.index_constituent_timeseries(symbol, index): 1 exactly on the sessions Norgate flags the
                symbol a constituent, 0 elsewhere (no forward fill across gaps, no tail trim; build_index_constituent_matrix
                semantics). Only symbols in the index's "Current & Past" watchlist are queried.
    window      start_date_str .. end_date_str; the end is the last in-sample session (the vault seal, 2022-12-30), so the
                cache never holds a vault bar. `universe_panel` also cuts at the seal.

Causal features for the ladder (each value at row T uses bars dated <= T only, unless stated):
    adv63_df            ADV63_T = mean of native Norgate Turnover over [T-62, T], as
                        strategies/dv2/strategy_mr_dv2_liquidity_floor.py: a non-finite or non-positive Turnover inside the
                        window makes ADV63 NaN.
    half_spread_df      causal Abdi-Ranaldo (2017) half-spread from daily High, Low, Close. With c = log Close and
                        eta = (log High + log Low) / 2, the two-day product ending at k is
                            g_k = (c_(k-1) - eta_(k-1)) x (c_(k-1) - eta_k)        (bars k-1 and k only)
                        and the spread estimate at T is S_T = sqrt(max(0, 4 x median(g_(T-20) .. g_T))) (21 products, at
                        least 15 finite); half-spread = S_T / 2. The published monthly estimator averages the products;
                        the median is used here (robust to gap days). `fill_slippage_mat` applies the value of row t-1 to a
                        fill on row t (lagged one day: the estimate never reads the fill bar).
    adv_tercile_df      per date, terciles of ADV63 among the universe's members with a finite ADV63 that date (1 = lowest).

Costed ledger (`costed_book`): the gross replica `alpha.scout.specs.dv2._dv2_book_daily` with fills priced at
Open x (1 +- slippage of the fill row) and a fee max(min_fee, fee_per_share x shares) per fill, in dollars on a
compounding book. `share_scale_mat` converts the replica's adjusted share count to the fee's share count: ones = the
engine's adjusted share units (ENGINE parity), Close / Unadjusted Close at T = nominal shares (what a broker charges).
With zero costs it equals the gross replica exactly (tested).

*** CRITICAL*** membership is the flag of the session itself; ADV63 and the spread at T read no bar after T; a fill on
row t is charged the spread estimate of row t-1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.panel import FIELD_TUPLE, VAULT_SEAL_STR, Panel

SUPERSET_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "dv2_size_ladder" / "superset"
SEAL_END_STR = "2022-12-30"
ADV_WINDOW_INT = 63
SPREAD_WINDOW_INT, SPREAD_MIN_COUNT_INT = 21, 15


# ---------------------------------------------------------------- the superset panel
@dataclass(frozen=True)
class SupersetPanel:
    field_dict: dict  # field -> float32 ndarray (date x symbol)
    member_dict: dict  # index name -> int8 ndarray (date x symbol)
    date_index: pd.DatetimeIndex
    symbol_list: list
    snapshot_id_str: str

    def ever_member_symbol_list(self, index_name_str: str, start_str: str | None = None, end_str: str = SEAL_END_STR) -> list[str]:
        row_mask = self._row_mask(start_str, end_str)
        column_vec = self.member_dict[index_name_str][row_mask].any(axis=0)
        return [s for s, keep_bool in zip(self.symbol_list, column_vec) if keep_bool]

    def _row_mask(self, start_str: str | None, end_str: str) -> np.ndarray:
        row_mask = np.asarray(self.date_index <= pd.Timestamp(end_str))
        if start_str is not None:
            row_mask &= np.asarray(self.date_index >= pd.Timestamp(start_str))
        return row_mask


def membership_matrix(flag_ser_dict: dict, date_index: pd.DatetimeIndex, symbol_list: list) -> np.ndarray:
    """int8 (date x symbol): 1 exactly on the dates a symbol's Norgate flag is 1 (a date missing from its series is 0).

    *** CRITICAL*** no forward fill: a flag dated after T never reaches row T, and a gap in the flag stays a gap."""
    out_mat = np.zeros((len(date_index), len(symbol_list)), dtype=np.int8)
    position_dict = {s: i for i, s in enumerate(symbol_list)}
    for symbol_str, flag_ser in flag_ser_dict.items():
        if symbol_str not in position_dict:
            continue
        member_index = pd.DatetimeIndex(flag_ser.index[np.asarray(flag_ser) == 1])
        row_vec = date_index.get_indexer(member_index)
        out_mat[row_vec[row_vec >= 0], position_dict[symbol_str]] = 1
    return out_mat


def _norgate_module():
    from data.norgate_loader import _load_direct_norgate_module
    from data.norgate_snapshot_store import is_snapshot_mode_enabled_bool

    if is_snapshot_mode_enabled_bool():
        raise RuntimeError("The superset panel reads Norgate directly (Russell memberships are not in the snapshot store).")
    return _load_direct_norgate_module()


def build_superset_panel(index_name_list: list[str], name_str: str, start_date_str: str = "2001-01-01",
                         end_date_str: str = SEAL_END_STR, log_fn=print) -> Path:
    """Build and cache the superset panel. Returns its folder (results/scout/dv2_size_ladder/superset/<name>/<id>)."""
    from data.norgate_loader import load_price_timeseries

    if pd.Timestamp(end_date_str) >= pd.Timestamp(VAULT_SEAL_STR):
        raise ValueError("The superset panel ends before the vault seal.")
    nd = _norgate_module()
    start_ts, end_ts = pd.Timestamp(start_date_str), pd.Timestamp(end_date_str)
    flag_dict_dict, member_symbol_set = {}, set()
    for index_name_str in index_name_list:
        symbol_list = nd.watchlist_symbols(f"{index_name_str} Current & Past")
        flag_dict = {}
        for symbol_str in symbol_list:
            frame = nd.index_constituent_timeseries(symbol_str, index_name_str, timeseriesformat="pandas-dataframe")
            flag_ser = frame["Index Constituent"]
            flag_ser = flag_ser[(flag_ser.index >= start_ts) & (flag_ser.index <= end_ts)]
            if (flag_ser == 1).any():
                flag_dict[symbol_str] = flag_ser
        flag_dict_dict[index_name_str] = flag_dict
        member_symbol_set |= set(flag_dict)
        log_fn(f"membership {index_name_str}: {len(flag_dict)} of {len(symbol_list)} symbols are members in the window")

    price_dict = {}
    for i_int, symbol_str in enumerate(sorted(member_symbol_set)):
        frame = load_price_timeseries(symbol_str, start_date_str=start_date_str, end_date_str=end_date_str)
        if len(frame):
            price_dict[symbol_str] = frame
        if i_int % 2000 == 0:
            log_fn(f"prices {i_int} / {len(member_symbol_set)}")
    symbol_list = sorted(price_dict)
    date_index = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in price_dict.values()))))
    date_index = date_index[(date_index >= start_ts) & (date_index <= end_ts)]
    field_dict = {f: np.full((len(date_index), len(symbol_list)), np.nan, dtype=np.float32) for f in FIELD_TUPLE}
    for column_int, symbol_str in enumerate(symbol_list):
        frame = price_dict[symbol_str]
        row_vec = date_index.get_indexer(frame.index)
        keep_vec = row_vec >= 0
        for field_str in FIELD_TUPLE:
            if field_str in frame.columns:
                field_dict[field_str][row_vec[keep_vec], column_int] = frame[field_str].to_numpy(dtype=np.float32)[keep_vec]
    del price_dict
    member_dict = {name: membership_matrix(flag_dict_dict[name], date_index, symbol_list) for name in index_name_list}

    digest_obj = hashlib.sha256("\n".join(FIELD_TUPLE + ("|",) + tuple(symbol_list) + ("|",) + tuple(index_name_list)).encode("utf-8"))
    digest_obj.update(date_index.asi8.tobytes())
    for field_str in FIELD_TUPLE:
        digest_obj.update(np.nan_to_num(field_dict[field_str], nan=-1.0).tobytes())
    for index_name_str in index_name_list:
        digest_obj.update(member_dict[index_name_str].tobytes())
    snapshot_id_str = digest_obj.hexdigest()[:16]

    folder_path = SUPERSET_ROOT_PATH / name_str / snapshot_id_str
    folder_path.mkdir(parents=True, exist_ok=True)
    for field_str in FIELD_TUPLE:
        np.save(folder_path / f"field_{field_str.replace(' ', '_')}.npy", field_dict[field_str])
    for k_int, index_name_str in enumerate(index_name_list):
        np.save(folder_path / f"member_{k_int:02d}.npy", member_dict[index_name_str])
    first_member_dict = {name: (str(date_index[np.flatnonzero(member_dict[name].any(axis=1))[0]].date())
                                if member_dict[name].any() else None) for name in index_name_list}
    meta_dict = {
        "name_str": name_str, "index_name_list": index_name_list, "start_date_str": start_date_str, "end_date_str": end_date_str,
        "first_date_str": str(date_index[0].date()), "last_date_str": str(date_index[-1].date()),
        "symbol_count_int": len(symbol_list), "snapshot_id_str": snapshot_id_str, "first_member_date_dict": first_member_dict,
        "built_utc_str": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    (folder_path / "meta.json").write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
    (folder_path / "symbols.json").write_text(json.dumps(symbol_list), encoding="utf-8")
    np.save(folder_path / "dates.npy", date_index.asi8)
    (SUPERSET_ROOT_PATH / name_str / "latest.json").write_text(json.dumps({"snapshot_id_str": snapshot_id_str}), encoding="utf-8")
    return folder_path


def load_superset_panel(name_str: str, snapshot_id_str: str | None = None, index_name_list: list[str] | None = None,
                        mmap_bool: bool = True) -> SupersetPanel:
    """Load a cached superset (memory-mapped by default); `index_name_list` limits the membership matrices loaded."""
    base_path = SUPERSET_ROOT_PATH / name_str
    if snapshot_id_str is None:
        snapshot_id_str = json.loads((base_path / "latest.json").read_text(encoding="utf-8"))["snapshot_id_str"]
    folder_path = base_path / snapshot_id_str
    meta_dict = json.loads((folder_path / "meta.json").read_text(encoding="utf-8"))
    mode_str = "r" if mmap_bool else None
    field_dict = {f: np.load(folder_path / f"field_{f.replace(' ', '_')}.npy", mmap_mode=mode_str) for f in FIELD_TUPLE}
    wanted_list = meta_dict["index_name_list"] if index_name_list is None else index_name_list
    member_dict = {name: np.load(folder_path / f"member_{meta_dict['index_name_list'].index(name):02d}.npy", mmap_mode=mode_str)
                   for name in wanted_list}
    return SupersetPanel(field_dict=field_dict, member_dict=member_dict, date_index=pd.DatetimeIndex(np.load(folder_path / "dates.npy")),
                         symbol_list=json.loads((folder_path / "symbols.json").read_text(encoding="utf-8")),
                         snapshot_id_str=meta_dict["snapshot_id_str"])


def universe_panel(superset: SupersetPanel, index_name_str: str, member_from_str: str | None = None,
                   symbol_list: list[str] | None = None, field_tuple: tuple = FIELD_TUPLE) -> Panel:
    """An ordinary sealed Scout Panel of one index: its ever-members (inside the window), its exact membership.

    `member_from_str` zeroes membership before that date (the backtest start: no entry and no baseline member before
    it; prices before it stay for indicator warm-up). `symbol_list` overrides the column set (still the index's flag)."""
    columns = symbol_list if symbol_list is not None else superset.ever_member_symbol_list(index_name_str)
    position_vec = np.array([superset.symbol_list.index(s) for s in columns]) if len(columns) < 50 else \
        pd.Index(superset.symbol_list).get_indexer(columns)
    if (position_vec < 0).any():
        raise KeyError("universe_panel: a requested symbol is not in the superset.")
    row_mask = np.asarray(superset.date_index < pd.Timestamp(VAULT_SEAL_STR))
    date_index = superset.date_index[row_mask]
    member_mat = np.asarray(superset.member_dict[index_name_str][row_mask][:, position_vec]).copy()
    if member_from_str is not None:
        member_mat[np.asarray(date_index < pd.Timestamp(member_from_str))] = 0
    field_dict = {f: pd.DataFrame(np.asarray(superset.field_dict[f][row_mask][:, position_vec]), index=date_index, columns=list(columns))
                  for f in field_tuple}
    member_df = pd.DataFrame(member_mat, index=date_index, columns=list(columns))
    return Panel(name_str=f"{index_name_str} (superset {superset.snapshot_id_str})", field_dict=field_dict, member_df=member_df,
                 snapshot_id_str=f"{superset.snapshot_id_str}:{index_name_str}", sealed_bool=True)


# ---------------------------------------------------------------- causal liquidity features
def adv63_mat(turnover_mat: np.ndarray, window_int: int = ADV_WINDOW_INT) -> np.ndarray:
    """ADV63 at T = mean Turnover over [T-62, T]; any non-finite or non-positive Turnover in the window -> NaN."""
    value_df = pd.DataFrame(np.asarray(turnover_mat, dtype=float))
    valid_df = value_df.where(np.isfinite(value_df) & (value_df > 0))
    return valid_df.rolling(window_int, min_periods=window_int).mean().to_numpy()


def half_spread_mat(high_mat: np.ndarray, low_mat: np.ndarray, close_mat: np.ndarray, window_int: int = SPREAD_WINDOW_INT,
                    min_count_int: int = SPREAD_MIN_COUNT_INT) -> np.ndarray:
    """Causal Abdi-Ranaldo half-spread at T (bars <= T): 0.5 x sqrt(max(0, 4 x rolling median of the two-day products))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c_mat = np.log(np.asarray(close_mat, dtype=float))
        eta_mat = 0.5 * (np.log(np.asarray(high_mat, dtype=float)) + np.log(np.asarray(low_mat, dtype=float)))
    product_mat = np.full(c_mat.shape, np.nan)
    product_mat[1:] = (c_mat[:-1] - eta_mat[:-1]) * (c_mat[:-1] - eta_mat[1:])
    product_mat[~np.isfinite(product_mat)] = np.nan
    median_mat = pd.DataFrame(product_mat).rolling(window_int, min_periods=min_count_int).median().to_numpy()
    return 0.5 * np.sqrt(np.maximum(4.0 * median_mat, 0.0))


def fill_slippage_mat(half_spread: np.ndarray, floor_float: float = 0.00025) -> np.ndarray:
    """Per-side slippage of a fill on row t = max(floor, half-spread of row t-1); an unknown estimate -> the floor."""
    lagged_mat = np.full(half_spread.shape, np.nan)
    lagged_mat[1:] = half_spread[:-1]
    return np.fmax(np.nan_to_num(lagged_mat, nan=floor_float), floor_float)


def adv_tercile_mat(adv_mat: np.ndarray, member_mat: np.ndarray) -> np.ndarray:
    """Per date, ADV63 terciles (1 low .. 3 high) among members with a finite ADV63; 0 where not ranked."""
    ranked_df = pd.DataFrame(np.where(member_mat.astype(bool) & np.isfinite(adv_mat), adv_mat, np.nan)).rank(axis=1, pct=True)
    return np.nan_to_num(np.ceil(ranked_df.to_numpy() * 3.0), nan=0.0).clip(0, 3).astype(np.int8)


# ---------------------------------------------------------------- costed replica of the DV2 rule
@njit(cache=False)
def _costed_book(open_mat, close_mat, pointer_vec, candidate_vec, exit_signal_mat, max_positions_int, start_row_int,
                 slip_mat, share_scale_mat, fee_per_share_float, min_fee_float, capital_float):
    """`alpha.scout.specs.dv2._dv2_book_daily` in dollars with costs. Same decisions, same order (exits before entries,
    a held stock skipped without a slot, a stock without Open(t)/Close(t) sold at its last close <= T with the fee).
    Entries before `start_row_int` are not taken. Fill log: (row, asset, +1 entry / -1 exit / -2 delisting, dollar value)."""
    row_count_int, asset_count_int = close_mat.shape
    share_vec = np.zeros(asset_count_int)
    held_vec = np.zeros(max_positions_int, dtype=np.int64)
    entry_vec = np.zeros(max_positions_int, dtype=np.int64)
    log_row = np.zeros(row_count_int * max_positions_int * 2 + 1, dtype=np.int64)
    log_asset = np.zeros(row_count_int * max_positions_int * 2 + 1, dtype=np.int64)
    log_kind = np.zeros(row_count_int * max_positions_int * 2 + 1, dtype=np.int64)
    log_value = np.zeros(row_count_int * max_positions_int * 2 + 1)
    log_int, held_int = 0, 0
    entry_weight_float = 1.0 / max_positions_int
    cash_float, previous_total_float = capital_float, capital_float
    daily_vec = np.zeros(row_count_int)
    total_vec = np.full(row_count_int, capital_float)
    for t_int in range(1, row_count_int):
        p_int = t_int - 1
        exit_int = 0
        for k_int in range(held_int):
            if exit_signal_mat[p_int, held_vec[k_int]]:
                exit_int += 1
        slot_int = max_positions_int - held_int + exit_int
        entry_int = 0
        if slot_int > 0 and t_int >= start_row_int:
            for j_int in range(pointer_vec[p_int], pointer_vec[p_int + 1]):
                a_int = candidate_vec[j_int]
                if share_vec[a_int] > 0.0:
                    continue
                entry_vec[entry_int] = a_int
                entry_int += 1
                if entry_int == slot_int:
                    break
        kept_int = 0
        for k_int in range(held_int):
            a_int = held_vec[k_int]
            if not (np.isfinite(open_mat[t_int, a_int]) and np.isfinite(close_mat[t_int, a_int])):
                q_int = p_int
                while not np.isfinite(close_mat[q_int, a_int]):
                    q_int -= 1
                value_float = share_vec[a_int] * close_mat[q_int, a_int]
                fee_float = max(min_fee_float, fee_per_share_float * share_vec[a_int] * share_scale_mat[q_int, a_int]) if fee_per_share_float > 0 else 0.0
                cash_float += value_float - fee_float
                kind_int = -2
            elif exit_signal_mat[p_int, a_int]:
                value_float = share_vec[a_int] * open_mat[t_int, a_int]
                fee_float = max(min_fee_float, fee_per_share_float * share_vec[a_int] * share_scale_mat[p_int, a_int]) if fee_per_share_float > 0 else 0.0
                cash_float += value_float * (1.0 - slip_mat[t_int, a_int]) - fee_float
                kind_int = -1
            else:
                held_vec[kept_int] = a_int
                kept_int += 1
                continue
            log_row[log_int], log_asset[log_int], log_kind[log_int], log_value[log_int] = t_int, a_int, kind_int, value_float
            log_int += 1
            share_vec[a_int] = 0.0
        held_int = kept_int
        for k_int in range(entry_int):
            a_int = entry_vec[k_int]
            if not (np.isfinite(open_mat[t_int, a_int]) and np.isfinite(close_mat[t_int, a_int])):
                continue
            share_vec[a_int] = previous_total_float * entry_weight_float / close_mat[p_int, a_int]
            value_float = share_vec[a_int] * open_mat[t_int, a_int]
            fee_float = max(min_fee_float, fee_per_share_float * share_vec[a_int] * share_scale_mat[p_int, a_int]) if fee_per_share_float > 0 else 0.0
            cash_float -= value_float * (1.0 + slip_mat[t_int, a_int]) + fee_float
            log_row[log_int], log_asset[log_int], log_kind[log_int], log_value[log_int] = t_int, a_int, 1, value_float
            log_int += 1
            held_vec[held_int] = a_int
            held_int += 1
        total_float = cash_float
        for k_int in range(held_int):
            total_float += share_vec[held_vec[k_int]] * close_mat[t_int, held_vec[k_int]]
        daily_vec[t_int] = total_float / previous_total_float - 1.0
        total_vec[t_int] = total_float
        previous_total_float = total_float
    return daily_vec, total_vec, log_row[:log_int], log_asset[:log_int], log_kind[:log_int], log_value[:log_int]


@dataclass
class CostedResult:
    daily_ser: pd.Series
    total_value_ser: pd.Series
    fill_df: pd.DataFrame  # date, asset, kind_int (+1 entry, -1 exit, -2 delisting sale), value_float (pre-cost dollars)


def rule_mats(panel: Panel, config) -> dict:
    """The DV2 rule's candidate CSR and exit signal on a panel, exactly as dv2.fast_daily_list_panel builds them."""
    from alpha.scout.specs import dv2

    mats = dv2._panel_mats(panel)
    close_df = panel.field("Close")
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum_mat = dv2._momentum_df(close_df, config.momentum_lookback_int).to_numpy(dtype=float)
        sma_mat = close_df.rolling(config.trend_sma_int).mean().to_numpy(dtype=float)
        natr = dv2.natr_mat(mats["high"], mats["low"], mats["close"], config.natr_length_int)
        base_mat = (mats["complete"] & mats["member"] & ~np.isnan(natr) & ~np.isnan(momentum_mat) & (mats["close"] > sma_mat)
                    & (momentum_mat > config.momentum_min_float))
    dv2_values = dv2.dv2_mat(mats["close"], mats["high"], mats["low"], config.dv2_length_int, base_mat)
    with np.errstate(invalid="ignore"):
        pointer_vec, candidate_vec = dv2.ranked_candidate_csr(base_mat & (dv2_values < config.entry_dv2_max_float), natr)
    return {"open": mats["open"], "close": mats["close"], "member": mats["member"], "pointer_vec": pointer_vec,
            "candidate_vec": candidate_vec, "exit_signal": dv2.exit_mat(mats["close"], mats["high"], config.exit_rule_str)}


def costed_book(panel: Panel, config, slippage, fee_per_share_float: float, min_fee_float: float, share_unit_str: str = "nominal",
                start_date_str: str = "2004-01-01", capital_float: float = 100_000.0, mats: dict | None = None) -> CostedResult:
    """The frozen DV2 rule on a panel with costs. `slippage`: a per-side float or a (date x symbol) matrix of fill rows.
    `share_unit_str`: "adjusted" (the engine's share units for the fee) or "nominal" (adjusted shares x Close / Unadjusted Close at T)."""
    mats = mats if mats is not None else rule_mats(panel, config)
    shape_tuple = mats["close"].shape
    slip_mat = np.full(shape_tuple, float(slippage)) if np.isscalar(slippage) else np.asarray(slippage, dtype=float)
    if share_unit_str == "adjusted":
        scale_mat = np.ones(shape_tuple)
    elif share_unit_str == "nominal":
        with np.errstate(divide="ignore", invalid="ignore"):
            # nominal shares = value / nominal price = adjusted shares x (adjusted Close / Unadjusted Close)
            scale_mat = np.nan_to_num(mats["close"] / panel.field("Unadjusted Close").to_numpy(dtype=float), nan=1.0, posinf=1.0)
    else:
        raise ValueError("share_unit_str must be 'adjusted' or 'nominal'.")
    # The first decision is the close of the last session before the start (the engine's calendar).
    start_row_int = int(np.searchsorted(panel.date_index.to_numpy(), np.datetime64(pd.Timestamp(start_date_str))))
    daily_vec, total_vec, row_vec, asset_vec, kind_vec, value_vec = _costed_book(
        mats["open"], mats["close"], mats["pointer_vec"], mats["candidate_vec"], mats["exit_signal"], config.max_positions_int,
        start_row_int, slip_mat, scale_mat, float(fee_per_share_float), float(min_fee_float), float(capital_float))
    date_index = panel.date_index
    symbol_arr = np.array(panel.symbol_list, dtype=object)
    fill_df = pd.DataFrame({"date": date_index[row_vec], "asset": symbol_arr[asset_vec], "kind_int": kind_vec, "value_float": value_vec})
    return CostedResult(daily_ser=pd.Series(daily_vec, index=date_index), total_value_ser=pd.Series(total_vec, index=date_index), fill_df=fill_df)
