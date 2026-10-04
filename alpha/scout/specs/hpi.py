"""Scout spec of the HPI mean-reversion pods on point-in-time S&P 500 members: the 2/3/5 vote (WIRED, live account;
`strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote`) and the single-horizon rule with the IBS/RSI exits (PM_READY;
`strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit`). Both share the exits, ranking, slots and sizing.

An independent re-implementation; execution is the shared weights engine with the path-dependent `decision_fn` hook,
`hold_nan_bool` (a held name is never resized) and the opt-in `missing_open_hold_df` (a halted index member stays held).
Mapped on 2026-10-01 against main 8fd3c37 (the same-open slot refill of 8b21a2e, owner decision 2026-09-28):
strategies/hpi/stateful_long.py (lines "H:") and alpha/engine (strategy.py, order.py, backtester.py).

Data (H: load_exact_hpi_inputs :161-272, direct Norgate mode; snapshot mode goes through data.norgate_loader as H does)
    universe   every "S&P 500 Current & Past" symbol whose index_constituent_timeseries from 1998-01-01 has a 1; the
               columns are concatenated on the union of their dates and NaN -> 0 (not a member)
    prices     CAPITALSPECIAL, padding NONE, from 1998-01-01, plus $SPXTR (TOTALRETURN; it only widens the session index)
    sessions   the union of every loaded date; Dividend NaN -> 0 only on a symbol's synthetic (no OHLC) rows;
               Close forward-filled (valuation only); Open/High/Low stay NaN, so nothing fills on a missing bar
    membership as of T = the latest universe row on or before T (H: get_asof_universe_symbol_set :136-158)

Features per symbol on its own rows with Close (filled), High and Low all present (H: compute_signals :388-491)
    Return_w,T   Close_T / Close_(T-w) - 1 over those rows, w in {2, 3, 5} (vote) or {3}
    HPI_w,T      the prior 1,260 Return_w values P (T excluded):
                 Return <= 0: 100 x #(P <= Return) / #(P <= 0);  Return > 0: 100 x #(P > Return) / #(P > 0)
                 (H: compute_strict_hpi :83-133; NaN until 1,261 returns exist or with an empty side)
    IBS_T        (Close - Low) / (High - Low), a zero range NaN;  SMA200_T  rolling 200 mean of Close
    RSI2_T       TA-Lib RSI(2) (Wilder), first value on the 3rd row
Entry candidates at the close of T (H: get_opportunity_list :696-806)
    member at T; Close, Turnover, SMA200, IBS and every Return_w / HPI_w of the mode present
    vote:     #{w in 2,3,5 : Return_w < 0 and HPI_w < 30} >= 2        single: HPI_3 < 30 and Return_3 < 0
    IBS_T < 0.10 and Close_T > SMA200_T; ranked by Norgate Turnover_T descending, ties by symbol ascending
Decision after the close of T (H: iterate :523-602)
    pending      held names (shares > 0) with IBS_T > 0.90, RSI2_T > 90 or not a member at T join a sticky set
    exits        pending names whose Open_(T+1) is finite -> target 0 shares; each frees its slot NOW
                 (*** live-slot alignment, 8b21a2e: the replacement is bought in the same MOO basket)
    slots        10 - #held + #exits; the first `slots` candidates with zero shares enter (a held name is skipped
                 without using a slot; a candidate whose open is missing still uses one and is cancelled)
    sizing       order_value(V_T / 10): shares = trunc(V_T / 10 / Close_T) (order.py amount_in_shares; split-adjusted
                 whole shares); held names are never resized
    fills        Open_(T+1) x (1 +- 2.5 bp), fee max(1, 0.005 x shares), all orders netted, no cash check; dividends of
                 T net of 25% withholding before the open (engine Strategy defaults; the pod passes no cost arguments)
    missing open a held name with no Open(t) is liquidated at its last close only if it is not a member at t
                 (H: _liquidate_missing_price_positions :604-694); a member is held and marked at the filled close
Calendar: the first session on or after 2004-01-01; its decision reads the previous session (backtester.py).

*** CRITICAL*** Every feature at T reads bars up to T (HPI excludes T from its reference set); fills at Open(T+1).
The exit rule reads whether Open(T+1) is finite before freeing the slot: a tradability look-ahead the live host does
not have (it assumes the open prints). Registered as deviation `hpi_open_known_slot_refill`; `slot_rule_str="live"`
is truth mode.

Family parameters (`HpiConfig`; each variant's default is its engine configuration): hpi_threshold_float,
entry_ibs_max_float, exit_ibs_min_float, exit_rsi_min_float, max_positions_int, the vote horizons and minimum;
decision_offset_int must be 0 (a daily event rule has no rebalance schedule: no luck band).

MCPT (`fast_daily_list_panel`): the same features recomputed from an alpha.scout.panel.Panel alone (its padded bars,
exact membership, native Turnover) and the slot rule on a gross fractional-share ledger (no costs, no dividends).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from numba import njit, prange

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate

INDEX_NAME_STR = "S&P 500"
BENCHMARK_STR = "$SPXTR"
HISTORY_START_STR = "1998-01-01"
BACKTEST_START_STR = "2004-01-01"
ENGINE_COST_MODEL = CostModel()  # engine Strategy defaults: 2.5 bp, $0.005 a share, $1 minimum, 25% withholding
SEAL_END_STR = "2022-12-30"
ENTRY_MODE_TUPLE = ("vote", "single")


@dataclass(frozen=True)
class HpiConfig:
    entry_mode_str: str = "vote"
    vote_horizon_tuple: tuple = (2, 3, 5)  # vote mode: the Return/HPI horizons
    vote_min_int: int = 2
    single_horizon_int: int = 3  # single mode
    hpi_lookback_int: int = 1_260
    hpi_threshold_float: float = 30.0
    entry_ibs_max_float: float = 0.10
    sma_window_int: int = 200
    exit_ibs_min_float: float = 0.90
    rsi_window_int: int = 2
    exit_rsi_min_float: float = 90.0
    max_positions_int: int = 10
    decision_offset_int: int = 0

    def __post_init__(self):
        if self.entry_mode_str not in ENTRY_MODE_TUPLE:
            raise ValueError(f"HpiConfig: entry_mode_str must be one of {ENTRY_MODE_TUPLE}.")
        if not 0.0 <= self.entry_ibs_max_float < self.exit_ibs_min_float <= 1.0 or not 0.0 < self.hpi_threshold_float <= 100.0:
            raise ValueError("HpiConfig: 0 <= entry IBS < exit IBS <= 1 and 0 < HPI threshold <= 100.")
        if self.hpi_lookback_int < 2 or self.max_positions_int < 1 or not 1 <= self.vote_min_int <= len(self.vote_horizon_tuple):
            raise ValueError("HpiConfig: HPI lookback >= 2, at least one slot, 1 <= vote_min_int <= #horizons.")
        if self.decision_offset_int != 0:
            raise ValueError("HpiConfig: a daily event rule has no rebalance offset (decision_offset_int = 0).")

    @property
    def horizon_tuple(self) -> tuple:
        return tuple(self.vote_horizon_tuple) if self.entry_mode_str == "vote" else (self.single_horizon_int,)


@dataclass(frozen=True)
class HpiVariant:
    strategy_import_str: str
    config: HpiConfig = field(default_factory=HpiConfig)


VARIANT_DICT = {
    "hpi_vote": HpiVariant("strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote", HpiConfig(entry_mode_str="vote")),
    "hpi_ibs_rsi": HpiVariant("strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit", HpiConfig(entry_mode_str="single")),
}


@dataclass(frozen=True)
class HpiInputs:
    """Point-in-time S&P 500 bars on the engine's session index (the union of every loaded series), columns sorted."""

    open_df: pd.DataFrame
    high_df: pd.DataFrame
    low_df: pd.DataFrame
    close_df: pd.DataFrame  # forward-filled (valuation only)
    turnover_df: pd.DataFrame
    dividend_df: pd.DataFrame
    member_df: pd.DataFrame  # bool: member as of the latest universe row on or before the session
    backtest_start_str: str = BACKTEST_START_STR


# ---------------------------------------------------------------- data
def _direct_universe_and_frames(end_date_str: str | None) -> tuple:
    import norgatedata

    membership_ser_list = []
    for symbol_str in norgatedata.watchlist_symbols(f"{INDEX_NAME_STR} Current & Past"):
        membership_df = norgatedata.index_constituent_timeseries(
            symbol_str, INDEX_NAME_STR, start_date=HISTORY_START_STR, end_date=end_date_str, timeseriesformat="pandas-dataframe")
        if membership_df is None or membership_df.empty or not membership_df["Index Constituent"].eq(1).any():
            continue
        membership_ser_list.append(membership_df["Index Constituent"].rename(symbol_str))
    universe_df = pd.concat(membership_ser_list, axis=1).fillna(0).astype(int).sort_index()

    frame_dict = {}
    for symbol_str in list(universe_df.columns) + [BENCHMARK_STR]:
        adjustment_obj = (norgatedata.StockPriceAdjustmentType.TOTALRETURN if symbol_str == BENCHMARK_STR
                          else norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL)
        price_df = norgatedata.price_timeseries(
            symbol_str, stock_price_adjustment_setting=adjustment_obj, padding_setting=norgatedata.PaddingType.NONE,
            start_date=HISTORY_START_STR, end_date=end_date_str, timeseriesformat="pandas-dataframe")
        if price_df is not None and not price_df.empty:
            frame_dict[symbol_str] = price_df
    session_index = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in frame_dict.values()))))  # pd.concat's union
    return universe_df, frame_dict, session_index


def _snapshot_universe_and_frames(end_date_str: str | None) -> tuple:
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
    from data.norgate_snapshot_store import (
        HPI_SP500_PROFILE_STR,
        get_active_data_profile_str,
    )

    if get_active_data_profile_str() != HPI_SP500_PROFILE_STR:  # H: :169-175
        raise RuntimeError(f"HPI snapshot mode requires data profile {HPI_SP500_PROFILE_STR!r}.")
    symbol_list, universe_df = build_index_constituent_matrix(indexname=INDEX_NAME_STR)
    pricing_df = load_raw_prices(list(symbol_list), [BENCHMARK_STR], start_date=HISTORY_START_STR, end_date=end_date_str).sort_index()
    frame_dict = {s: pricing_df[s] for s in pricing_df.columns.get_level_values(0).unique()}
    return universe_df.sort_index(), frame_dict, pd.DatetimeIndex(pricing_df.index)


def load_inputs(end_date_str: str | None = None) -> HpiInputs:
    """The engine's HPI inputs (H: load_exact_hpi_inputs). Snapshot mode follows the engine's snapshot branch."""
    from data.norgate_snapshot_store import is_snapshot_mode_enabled_bool

    loader_fn = _snapshot_universe_and_frames if is_snapshot_mode_enabled_bool() else _direct_universe_and_frames
    universe_df, frame_dict, session_index = loader_fn(end_date_str)
    symbol_list = sorted(str(s) for s in frame_dict if s != BENCHMARK_STR)  # ties in the ranking break by symbol ascending

    def field_df(field_str: str) -> pd.DataFrame:
        return pd.DataFrame({s: pd.to_numeric(frame_dict[s][field_str], errors="coerce") if field_str in frame_dict[s].columns
                             else pd.Series(np.nan, index=frame_dict[s].index) for s in symbol_list}).reindex(session_index)

    open_df, high_df, low_df, close_df = (field_df(f) for f in ("Open", "High", "Low", "Close"))
    # *** CRITICAL*** a row with no OHLC observation is synthetic: it carries no dividend (H: :245-264).
    observed_df = open_df.notna() | high_df.notna() | low_df.notna() | close_df.notna()
    dividend_df = field_df("Dividend")
    dividend_df = dividend_df.mask(~observed_df & dividend_df.isna(), 0.0)
    # *** CRITICAL*** membership as of T: the latest universe row on or before T, never a later one.
    member_df = (universe_df.reindex(columns=symbol_list).fillna(0).astype(int).reindex(session_index, method="ffill")
                 .fillna(0).astype(int).eq(1))
    return HpiInputs(open_df=open_df, high_df=high_df, low_df=low_df, close_df=close_df.ffill(), turnover_df=field_df("Turnover"),
                     dividend_df=dividend_df, member_df=member_df)


# ---------------------------------------------------------------- features
@njit(cache=False)
def strict_hpi_vec(return_vec, lookback_int):
    """HPI of each return against the prior `lookback_int` returns (window of lookback + 1 finite values, T removed),
    exact counts with a Fenwick tree over the ranks of the column's values."""
    row_count_int = return_vec.size
    out_vec = np.full(row_count_int, np.nan)
    finite_mask = np.isfinite(return_vec)
    if finite_mask.sum() <= lookback_int:
        return out_vec
    unique_vec = np.unique(return_vec[finite_mask])
    rank_vec = np.zeros(row_count_int, dtype=np.int64)
    for i in range(row_count_int):
        if finite_mask[i]:
            rank_vec[i] = np.searchsorted(unique_vec, return_vec[i]) + 1  # 1-based
    size_int = unique_vec.size
    tree_vec = np.zeros(size_int + 1, dtype=np.int64)
    window_int = lookback_int + 1
    finite_count_int, nonpositive_count_int, positive_count_int = 0, 0, 0
    for t in range(row_count_int):
        if finite_mask[t]:
            k = rank_vec[t]
            while k <= size_int:
                tree_vec[k] += 1
                k += k & (-k)
            finite_count_int += 1
            if return_vec[t] <= 0.0:
                nonpositive_count_int += 1
            else:
                positive_count_int += 1
        old_int = t - window_int
        if old_int >= 0 and finite_mask[old_int]:
            k = rank_vec[old_int]
            while k <= size_int:
                tree_vec[k] -= 1
                k += k & (-k)
            finite_count_int -= 1
            if return_vec[old_int] <= 0.0:
                nonpositive_count_int -= 1
            else:
                positive_count_int -= 1
        if finite_count_int < window_int or not finite_mask[t]:
            continue
        leq_int = 0  # values <= Return_T in the window, T included (pandas rolling rank "max")
        k = rank_vec[t]
        while k > 0:
            leq_int += tree_vec[k]
            k -= k & (-k)
        value_float = return_vec[t]
        if value_float <= 0.0:
            prior_int = nonpositive_count_int - 1
            if prior_int > 0:
                out_vec[t] = min(max(100.0 * float(leq_int - 1) / float(prior_int), 0.0), 100.0)
        else:
            prior_int = positive_count_int - 1
            if prior_int > 0:
                out_vec[t] = min(max(100.0 * float(window_int - leq_int) / float(prior_int), 0.0), 100.0)
    return out_vec


@njit(cache=False)
def talib_rsi_vec(close_vec, period_int):
    """TA-Lib RSI (classic Wilder, unstable period 0) on a column whose finite values form a prefix."""
    row_count_int = close_vec.size
    out_vec = np.full(row_count_int, np.nan)
    valid_int = 0
    while valid_int < row_count_int and np.isfinite(close_vec[valid_int]):
        valid_int += 1
    if valid_int <= period_int:
        return out_vec
    gain_float, loss_float = 0.0, 0.0
    for i in range(1, period_int + 1):
        change_float = close_vec[i] - close_vec[i - 1]
        if change_float < 0.0:
            loss_float -= change_float
        else:
            gain_float += change_float
    loss_float /= period_int
    gain_float /= period_int
    total_float = gain_float + loss_float
    out_vec[period_int] = 100.0 * (gain_float / total_float) if not (-1e-8 < total_float < 1e-8) else 0.0
    for i in range(period_int + 1, valid_int):
        change_float = close_vec[i] - close_vec[i - 1]
        loss_float *= period_int - 1
        gain_float *= period_int - 1
        if change_float < 0.0:
            loss_float -= change_float
        else:
            gain_float += change_float
        loss_float /= period_int
        gain_float /= period_int
        total_float = gain_float + loss_float
        out_vec[i] = 100.0 * (gain_float / total_float) if not (-1e-8 < total_float < 1e-8) else 0.0
    return out_vec


@njit(cache=False, parallel=True)
def _column_apply_hpi(compressed_mat, lookback_int):
    out_mat = np.full(compressed_mat.shape, np.nan)
    for j in prange(compressed_mat.shape[1]):
        out_mat[:, j] = strict_hpi_vec(compressed_mat[:, j], lookback_int)
    return out_mat


@njit(cache=False, parallel=True)
def _column_apply_rsi(compressed_mat, period_int):
    out_mat = np.full(compressed_mat.shape, np.nan)
    for j in prange(compressed_mat.shape[1]):
        out_mat[:, j] = talib_rsi_vec(compressed_mat[:, j], period_int)
    return out_mat


def _compressor(valid_mat: np.ndarray):
    """Each column's valid rows packed to the top (NaN below), and the map back: the engine computes every feature
    on a symbol's own observed rows, so a gap is skipped, not filled."""
    row_vec, col_vec = np.nonzero(valid_mat)
    pos_vec = (np.cumsum(valid_mat, axis=0) - 1)[row_vec, col_vec]

    def pack(value_mat: np.ndarray) -> np.ndarray:
        packed_mat = np.full(valid_mat.shape, np.nan)
        packed_mat[pos_vec, col_vec] = value_mat[row_vec, col_vec]
        return packed_mat

    def unpack(packed_mat: np.ndarray) -> np.ndarray:
        value_mat = np.full(valid_mat.shape, np.nan)
        value_mat[row_vec, col_vec] = packed_mat[pos_vec, col_vec]
        return value_mat

    return pack, unpack


def feature_mat_dict(high_mat: np.ndarray, low_mat: np.ndarray, close_mat: np.ndarray, horizon_tuple: tuple, config: HpiConfig) -> dict:
    """Features on each symbol's rows with Close, High and Low present (close forward-filled as the engine's)."""
    valid_mat = np.isfinite(close_mat) & np.isfinite(high_mat) & np.isfinite(low_mat)
    pack, unpack = _compressor(valid_mat)
    packed_close_mat = pack(close_mat)
    with np.errstate(divide="ignore", invalid="ignore"):
        range_mat = high_mat - low_mat
        ibs_mat = np.where(valid_mat & (range_mat != 0.0), (close_mat - low_mat) / np.where(range_mat != 0.0, range_mat, 1.0), np.nan)
        out_dict = {
            "ibs": ibs_mat,
            "sma": unpack(pd.DataFrame(packed_close_mat).rolling(config.sma_window_int, min_periods=config.sma_window_int).mean().to_numpy()),
            "rsi": unpack(_column_apply_rsi(packed_close_mat, config.rsi_window_int)),
        }
        for horizon_int in horizon_tuple:
            packed_return_mat = np.full(packed_close_mat.shape, np.nan)
            packed_return_mat[horizon_int:] = packed_close_mat[horizon_int:] / packed_close_mat[:-horizon_int] - 1.0
            out_dict[f"return_{horizon_int}"] = unpack(packed_return_mat)
            out_dict[f"hpi_{horizon_int}"] = unpack(_column_apply_hpi(packed_return_mat, config.hpi_lookback_int))
    return out_dict


def feature_dict(inputs: HpiInputs, config: HpiConfig) -> dict:
    return feature_mat_dict(inputs.high_df.to_numpy(float), inputs.low_df.to_numpy(float), inputs.close_df.to_numpy(float),
                            config.horizon_tuple, config)


def regime_mat(features: dict, close_mat: np.ndarray, turnover_mat: np.ndarray, member_mat: np.ndarray, config: HpiConfig) -> np.ndarray:
    """Members at T with every field the mode requires present (H: dropna(subset=required_field_list)) and
    Close_T > SMA200_T: the candidates before the HPI and IBS conditions."""
    present_mat = np.isfinite(close_mat) & np.isfinite(turnover_mat) & np.isfinite(features["sma"]) & np.isfinite(features["ibs"])
    for horizon_int in config.horizon_tuple:
        present_mat &= np.isfinite(features[f"return_{horizon_int}"]) & np.isfinite(features[f"hpi_{horizon_int}"])
    with np.errstate(invalid="ignore"):
        return member_mat & present_mat & (close_mat > features["sma"])


def signal_mats(features: dict, close_mat: np.ndarray, turnover_mat: np.ndarray, member_mat: np.ndarray, config: HpiConfig):
    """(entry, exit, rank): entry = every candidate condition at T (membership included); exit = IBS or RSI2 above
    its bar (membership is checked in the decision); rank = Turnover_T. NaN comparisons are False."""
    with np.errstate(invalid="ignore"):
        vote_mat = sum(((features[f"return_{h}"] < 0.0) & (features[f"hpi_{h}"] < config.hpi_threshold_float)).astype(int)
                       for h in config.horizon_tuple)
        hpi_ok_mat = vote_mat >= (config.vote_min_int if config.entry_mode_str == "vote" else 1)
        entry_mat = regime_mat(features, close_mat, turnover_mat, member_mat, config) & hpi_ok_mat & (features["ibs"] < config.entry_ibs_max_float)
        exit_mat = (features["ibs"] > config.exit_ibs_min_float) | (features["rsi"] > config.exit_rsi_min_float)
    return entry_mat, exit_mat, turnover_mat


# ---------------------------------------------------------------- the event rule as a weights-engine hook
def hpi_decision_fn(entry_mat, exit_mat, rank_mat, member_mat, open_mat, max_positions_int: int, slot_rule_str: str = "engine"):
    """H: iterate. `slot_rule_str` "engine": an exit is placed (and frees its slot) only when Open(T+1) is finite;
    "live": always (the live host's tradability marker; an exit with no open is cancelled and stays pending)."""
    asset_count_int = entry_mat.shape[1]
    entry_weight_float = 1.0 / float(max_positions_int)  # order_value(V_T / max_positions)
    pending_vec = np.zeros(asset_count_int, dtype=bool)

    def decide(t_idx_int: int, position_vec: np.ndarray, total_float: float):
        p_int = t_idx_int - 1  # T
        held_vec = position_vec > 0.0
        pending_vec[:] &= held_vec  # H: pending_exit_symbol_set.intersection_update(long_symbol_set)
        pending_vec[:] |= held_vec & (exit_mat[p_int] | ~member_mat[p_int])
        exit_vec = pending_vec & np.isfinite(open_mat[t_idx_int]) if slot_rule_str == "engine" else pending_vec.copy()
        slot_int = max_positions_int - int(held_vec.sum()) + int(exit_vec.sum())
        candidate_vec = np.flatnonzero(entry_mat[p_int] & (position_vec == 0.0))
        if candidate_vec.size > 1:  # Turnover descending; ties keep the sorted-symbol column order
            candidate_vec = candidate_vec[np.argsort(-rank_mat[p_int, candidate_vec], kind="stable")]
        entry_vec = candidate_vec[:max(slot_int, 0)]
        if not exit_vec.any() and entry_vec.size == 0:
            return None
        weight_vec = np.full(asset_count_int, np.nan)
        weight_vec[exit_vec] = 0.0
        weight_vec[entry_vec] = entry_weight_float
        return weight_vec

    return decide


def simulate_config(inputs: HpiInputs, config: HpiConfig, cost_model: CostModel = ENGINE_COST_MODEL, capital_float: float = 100_000.0,
                    slot_rule_str: str = "engine", features: dict | None = None) -> WeightsResult:
    features = features if features is not None else feature_dict(inputs, config)
    close_mat, open_mat, member_mat = inputs.close_df.to_numpy(float), inputs.open_df.to_numpy(float), inputs.member_df.to_numpy(bool)
    entry_mat, exit_mat, rank_mat = signal_mats(features, close_mat, inputs.turnover_df.to_numpy(float), member_mat, config)
    session_index = inputs.close_df.index
    start_ts = session_index[session_index >= pd.Timestamp(inputs.backtest_start_str)][0]
    empty_df = pd.DataFrame(columns=list(inputs.close_df.columns), dtype=float)
    return simulate(
        inputs.open_df, inputs.close_df, inputs.dividend_df, empty_df, start_date=start_ts, capital_float=capital_float,
        share_unit_mode_str="adjusted", cost_model=cost_model, hold_nan_bool=True, missing_open_hold_df=inputs.member_df,
        decision_fn=hpi_decision_fn(entry_mat, exit_mat, rank_mat, member_mat, open_mat, config.max_positions_int, slot_rule_str),
    )


# ---------------------------------------------------------------- per-asset MCPT replica (S5) on a Scout panel
@njit(cache=False)
def _hpi_book_daily(open_mat, close_mat, entry_mat, exit_mat, member_mat, rank_mat, max_positions_int):
    """Gross fractional-share ledger of the slot rule (no costs, no dividends): exits at Open(t) when it prints, a held
    non-member with no open sold at its last close, entries V_T / K / Close_T shares at Open(t)."""
    row_count_int, asset_count_int = close_mat.shape
    share_vec = np.zeros(asset_count_int)
    pending_vec = np.zeros(asset_count_int, dtype=np.bool_)
    candidate_vec = np.zeros(asset_count_int, dtype=np.int64)
    score_vec = np.zeros(asset_count_int)
    cash_float, previous_total_float = 1.0, 1.0
    daily_vec = np.zeros(row_count_int)
    for t in range(1, row_count_int):
        p = t - 1
        held_int, exit_int, candidate_int = 0, 0, 0
        for a in range(asset_count_int):
            if share_vec[a] > 0.0:
                held_int += 1
                if exit_mat[p, a] or not member_mat[p, a]:
                    pending_vec[a] = True
            else:
                pending_vec[a] = False
            if share_vec[a] == 0.0 and entry_mat[p, a]:
                candidate_vec[candidate_int] = a
                score_vec[candidate_int] = -rank_mat[p, a]
                candidate_int += 1
        for a in range(asset_count_int):
            if share_vec[a] > 0.0 and not np.isfinite(open_mat[t, a]):
                if not member_mat[t, a] or not np.isfinite(close_mat[t, a]):
                    cash_float += share_vec[a] * close_mat[p, a]  # liquidated at the last close
                    share_vec[a] = 0.0
                    pending_vec[a] = False
            elif share_vec[a] > 0.0 and pending_vec[a]:
                cash_float += share_vec[a] * open_mat[t, a]
                share_vec[a] = 0.0
                pending_vec[a] = False
                exit_int += 1
        slot_int = max_positions_int - held_int + exit_int
        if candidate_int > 0 and slot_int > 0:
            order_vec = np.argsort(score_vec[:candidate_int], kind="mergesort")
            for k in range(min(slot_int, candidate_int)):
                a = candidate_vec[order_vec[k]]
                if np.isfinite(open_mat[t, a]):
                    share_vec[a] = previous_total_float / max_positions_int / close_mat[p, a]
                    cash_float -= share_vec[a] * open_mat[t, a]
        total_float = cash_float
        for a in range(asset_count_int):
            if share_vec[a] > 0.0:
                total_float += share_vec[a] * close_mat[t, a]
        daily_vec[t] = total_float / previous_total_float - 1.0
        previous_total_float = total_float
    return daily_vec


def fast_daily_list_panel(panel, config_list: list[dict], base_config: HpiConfig) -> tuple[list[np.ndarray], np.ndarray]:
    """Gross daily returns of each configuration (`config_list` holds overrides of `base_config`) from the panel alone,
    and the equal-weight member baseline (members at T with a close, held from the close of T+1, as
    alpha.scout.searches._hold_daily). Valid on `alpha.scout.null.permuted_panel` draws: every feature is recomputed
    from the (moved) bars; membership and Turnover stay on their real dates."""
    raw_close_mat = panel.field("Close").to_numpy(float)
    close_mat = pd.DataFrame(raw_close_mat).ffill().to_numpy()
    high_mat, low_mat = panel.field("High").to_numpy(float), panel.field("Low").to_numpy(float)
    open_mat, turnover_mat = panel.field("Open").to_numpy(float), panel.field("Turnover").to_numpy(float)
    listed_mat = np.isfinite(raw_close_mat) & (raw_close_mat > 0.0)
    high_mat, low_mat, open_mat = (np.where(listed_mat, m, np.nan) for m in (high_mat, low_mat, open_mat))
    member_mat = (panel.member_df == 1).to_numpy()
    config_obj_list = [replace(base_config, **config_dict) for config_dict in config_list]
    horizon_tuple = tuple(sorted({h for c in config_obj_list for h in c.horizon_tuple}))
    feature_cache_dict, rule_cache_dict, daily_list = {}, {}, []
    for config in config_obj_list:
        key_tuple = (config.hpi_lookback_int, config.sma_window_int, config.rsi_window_int)
        if key_tuple not in feature_cache_dict:
            feature_cache_dict[key_tuple] = feature_mat_dict(high_mat, low_mat, np.where(listed_mat, raw_close_mat, np.nan), horizon_tuple, config)
        features = feature_cache_dict[key_tuple]
        rule_key_tuple = key_tuple + (config.entry_mode_str, config.horizon_tuple, config.vote_min_int)
        if rule_key_tuple not in rule_cache_dict:
            # The threshold-free parts once per panel: the regime, and the oversold score whose value < threshold is
            # exactly the vote (`oversold_score_mat`); only the threshold comparisons run per configuration.
            rule_cache_dict[rule_key_tuple] = (regime_mat(features, close_mat, turnover_mat, member_mat, config), oversold_score_mat(features, config))
        trend_mat, score_mat = rule_cache_dict[rule_key_tuple]
        with np.errstate(invalid="ignore"):
            entry_mat = trend_mat & (score_mat < config.hpi_threshold_float) & (features["ibs"] < config.entry_ibs_max_float)
            exit_mat = (features["ibs"] > config.exit_ibs_min_float) | (features["rsi"] > config.exit_rsi_min_float)
        daily_list.append(_hpi_book_daily(open_mat, close_mat, entry_mat, exit_mat, member_mat, turnover_mat, config.max_positions_int))

    with np.errstate(divide="ignore", invalid="ignore"):
        return_mat = np.vstack([np.zeros((1, raw_close_mat.shape[1])), raw_close_mat[1:] / raw_close_mat[:-1] - 1.0])
    decided_mat = member_mat & listed_mat
    weight_mat = decided_mat / np.maximum(decided_mat.sum(axis=1, keepdims=True), 1)
    baseline_vec = np.zeros(raw_close_mat.shape[0])
    baseline_vec[2:] = (weight_mat[:-2] * np.nan_to_num(return_mat[2:], nan=0.0, posinf=0.0, neginf=0.0)).sum(axis=1)
    return daily_list, baseline_vec


# ---------------------------------------------------------------- S3 (class E: the entry event inside the trend regime)
S3_HORIZON_TUPLE = (1, 2, 3, 5, 10, 20)


def median_holding_sessions_int(result: WeightsResult) -> int:
    """The median entry-to-exit holding period (sessions) of a run (each position is one entry then one exit)."""
    session_index = result.total_value_ser.index
    hold_list = []
    for _asset_str, frame in result.trade_df.groupby("asset"):
        entry_list = frame.loc[frame["delta_float"] > 0, "date"].tolist()
        exit_list = frame.loc[frame["delta_float"] < 0, "date"].tolist()
        hold_list.extend(session_index.get_loc(x) - session_index.get_loc(e) for e, x in zip(entry_list, exit_list))
    return int(np.median(hold_list))


def oversold_score_mat(features: dict, config: HpiConfig) -> np.ndarray:
    """Per horizon HPI_w where Return_w < 0, else 100; the vote passes exactly when the median of the three scores is
    below the threshold (the single rule: its one score). Lower = more oversold."""
    score_list = [np.where(features[f"return_{h}"] < 0.0, features[f"hpi_{h}"], np.where(np.isfinite(features[f"return_{h}"]), 100.0, np.nan))
                  for h in config.horizon_tuple]
    stacked_mat = np.stack(score_list)
    if config.entry_mode_str == "vote":
        return np.sort(stacked_mat, axis=0)[len(score_list) - config.vote_min_int]
    return stacked_mat[0]


def s3_inputs(variant_name_str: str, panel=None, inputs: HpiInputs | None = None, end_date_str: str = SEAL_END_STR) -> dict:
    """Inputs for alpha.scout.stations.s3_edge.run_s3 on the sealed S&P 500 panel (features from the panel's own bars):
    regime = members with all the variant's features present and Close > SMA200 (the rule's trend gate), event = the
    variant's HPI condition and IBS < 0.10 inside it (the full entry rule, before the slot cap and the ranking),
    indicator = minus the oversold score (higher = more oversold), liquidity = the 63-session Turnover rank, horizon =
    the S3 horizon nearest the engine run's median holding period (entry open to exit open)."""
    from alpha.scout import features as feature_lib
    from alpha.scout.panel import load_panel

    config = VARIANT_DICT[variant_name_str].config
    panel = panel if panel is not None else load_panel(INDEX_NAME_STR)
    panel = panel.truncated(end_date_str)
    raw_close_mat = panel.field("Close").to_numpy(float)
    listed_mat = np.isfinite(raw_close_mat) & (raw_close_mat > 0.0)
    high_mat, low_mat = (np.where(listed_mat, panel.field(n).to_numpy(float), np.nan) for n in ("High", "Low"))
    features = feature_mat_dict(high_mat, low_mat, np.where(listed_mat, raw_close_mat, np.nan), config.horizon_tuple, config)
    close_mat = pd.DataFrame(raw_close_mat).ffill().to_numpy()
    member_mat = (panel.member_df == 1).to_numpy()
    turnover_mat = panel.field("Turnover").to_numpy(float)
    trend_mat = regime_mat(features, close_mat, turnover_mat, member_mat, config)
    event_mat, _, _ = signal_mats(features, close_mat, turnover_mat, member_mat, config)
    frame = lambda mat: pd.DataFrame(mat, index=panel.date_index, columns=panel.symbol_list)
    if inputs is None:
        inputs = load_inputs()
    hold_int = median_holding_sessions_int(simulate_config(inputs, config))
    horizon_int = min(S3_HORIZON_TUPLE, key=lambda h: (abs(h - hold_int), h))
    return {"name_str": f"HPI ({variant_name_str})", "panel": panel, "regime_mask_df": frame(trend_mat), "event_mask_df": frame(event_mat),
            "horizon_int": horizon_int, "indicator_df": -frame(oversold_score_mat(features, config)),
            "liquidity_rank_df": feature_lib.turnover_rank(63).compute_fn(panel), "expected_sign_int": 1}


def s3_result(input_dict: dict) -> dict:
    """Run S3 (class E) and shape it for the card, as alpha.scout.specs.sector_ibs.s3_result."""
    from alpha.scout.specs.sector_ibs import s3_result as sector_s3_result

    return sector_s3_result(input_dict)
