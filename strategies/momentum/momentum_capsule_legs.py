"""
The three legs of the momentum capsule: one set of rules for the backtest and for live decisions.

Closed design and evidence: docs/research/RESIDUAL_MOMENTUM_20261031.md, "The closed design: three legs". Each leg is
one strategy in one account. No parameter was searched for this build; the values are the closed design's.

    Leg 1  Nasdaq-100, 10 names, ROC12 / NATR20
    Leg 2  Nasdaq-100, 10 names, market and sector residual, mean / std
    Leg 3  Russell 1000, 20 names, market and industry-peer residual, cumulative / volatility

Two books inside each leg
-------------------------
A leg decides on two days of every month: T0, the last session of the month, and T10, the session 10 sessions before
it. Each day keeps its own book of names, and that book's rank buffer reads only what that book held. At each swap the
account goes to the average of the two latest books:

    target_{i,T} = 0.5 * w_{i}(latest T0 book) + 0.5 * w_{i}(latest T10 book)

Research ran the two books as separate accounts with an annual reset. This one-account form trades the other book's
names back to their weights at each swap. The gap is measured in
scripts/research/residual_momentum_20261031/r34_one_account_form.py.

The path of decisions depends only on prices, membership and labels, not on the account: a rank buffer reads the
book's own earlier picks, never holdings. So a live decision replays the same path as the backtest and takes its last
row, and no strategy memory is stored anywhere.

Book rules at decision session T, all read at T
-----------------------------------------------
    Legs 1, 2
        gate_T      = SPY_T > SMA200(SPY)_T
        eligible    = PIT member at T and Close_T > SMA100_T and finite score
        picks       = held names within rank 18 first, then by rank; at most 4 per GICS sector; 10 names
        w_i         = clip(22 / VXN_asof(T), 0.25, 1) / 10; all cash when the gate is shut

    Leg 1 score     = (Close_T / Close_(same book, 12 decisions earlier) - 1) / (ATR20_T / Close_T)
    Leg 2 score     = mean(u) / std(u) over T-125..T-10, where u is the residual of r_i on $NDX and the stock's GICS
                      sector index, OLS with intercept over the 252 sessions ending at T; sector dropped when unusable

    Leg 3
        gate_T      = $RUI_T > SMA200($RUI)_T
        eligible    = PIT member at T and raw Close_T > $10 and 20-day dollar volume > $10M and finite score
        score       = sum(u) / (sqrt(126) * sd(u)) over T-130..T-5, where u is the residual of r_i on $RUI and the
                      leave-one-out equal-weight return of its PIT GICS industry group, 5 peers at least on every
                      session, else the sector index; OLS over the same window
        picks       = held names within rank 35 first, then by rank; at most 4 per GICS sector; 20 names
        w_i         = 0.985 / 20 per slot, shared by inverse 63-day volatility, at most 1.5 slots; a volatility at or
                      under 8% a year, or missing, reads the median of the others

Execution mapping: decision at Close_T, fill at the next session's open. Unfilled slots and VXN-reduced exposure stay
in cash; weights are not renormalized. A name of the other book that stopped trading since its decision carries no
weight at T: its share stays in cash. Whole shares, $0.005 a share with a $1 minimum, 2.5 bps a side, 25% dividend
withholding. Cash earns 0% in the engine.

*** REALISM GAPS *** GICS labels are Norgate's current labels carried back in time (G-025). Research costs were higher:
5 bps a side plus a causal half-spread. Leg 1 has no pinned-price guard, unlike legs 2 and 3; research measured that at
0.01 to 0.035 of Sharpe in its favour.

*** SELECTION *** History through 2026 has been examined over 21 rounds and about 3,000 series; this is not an untouched
holdout. Three legs fell short of the frozen rule twice and were kept by the owner's decision. No deflated Sharpe exists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import exchange_calendars
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy
from data.momentum_legs_snapshot_contract import (
    MOMENTUM_LEGS_MEMBER_START_DATE_STR,
    MOMENTUM_LEGS_PRICE_HISTORY_START_DATE_STR,
    SECTOR_INDEX_BY_LABEL_DICT,
    UNKNOWN_LABEL_STR,
    read_stable_gics_label_str,
)
from data.norgate_loader import build_index_constituent_matrix, is_snapshot_mode_enabled_bool, load_price_timeseries, load_raw_prices
from data.norgate_snapshot_store import load_momentum_legs_label_map
from strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled import (
    append_total_return_benchmark_data_df,
    audit_pit_universe_df,
    configure_total_return_benchmark_provenance,
    default_trade_id_int,
)

BOOK_OFFSET_TUPLE: tuple[int, int] = (0, 10)
BOOK_SHARE_FLOAT = 0.5
SCORE_NATR_STR = "natr"
SCORE_SECTOR_RESIDUAL_STR = "sector_residual"
SCORE_INDUSTRY_RESIDUAL_STR = "industry_residual"
MAX_ZERO_RETURN_FRACTION_FLOAT = 0.20
MIN_DAILY_RESIDUAL_VOL_FLOAT = 0.0025
MIN_INDUSTRY_PEER_COUNT_INT = 5
COLLINEARITY_TOLERANCE_FLOAT = 1e-16
INVERSE_VOL_FLOOR_FLOAT = 0.08
INVERSE_VOL_MAX_SLOTS_FLOAT = 1.5
REAL_ESTATE_SECTOR_STR = "60000000"
FINANCIALS_SECTOR_STR = "40000000"


@dataclass(frozen=True)
class MomentumCapsuleLegConfig:
    score_kind_str: str
    indexname_str: str
    gate_symbol_str: str
    market_symbol_str: str | None
    vxn_symbol_str: str | None
    max_positions_int: int
    sector_cap_int: int
    hold_rank_int: int
    gross_float: float
    stock_sma_int: int
    min_raw_close_float: float
    min_dollar_volume_float: float
    inverse_vol_bool: bool
    history_start_date_str: str = MOMENTUM_LEGS_PRICE_HISTORY_START_DATE_STR
    member_start_date_str: str = MOMENTUM_LEGS_MEMBER_START_DATE_STR
    # The first execution of the replayed path. A run with a later backtest_start_date_str reads the same path.
    path_start_date_str: str = "2000-09-01"
    backtest_start_date_str: str = "2000-09-01"
    end_date_str: str | None = None
    performance_benchmark_symbol_str: str = "$SPX"
    performance_benchmark_data_symbol_str: str = "$SPXTR"
    capital_base_float: float = 1_000_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0
    gate_sma_int: int = 200
    target_vxn_float: float = 22.0
    min_vxn_scale_float: float = 0.25
    roc_decision_count_int: int = 12
    natr_window_int: int = 20
    residual_window_int: int = 252
    residual_look_int: int = 126
    residual_skip_int: int = 10
    peer_window_int: int = 126
    peer_skip_int: int = 5
    dollar_volume_window_int: int = 20
    inverse_vol_window_int: int = 63


LEG1_CONFIG = MomentumCapsuleLegConfig(
    score_kind_str=SCORE_NATR_STR, indexname_str="Nasdaq 100", gate_symbol_str="SPY", market_symbol_str=None,
    vxn_symbol_str="$VXN", max_positions_int=10, sector_cap_int=4, hold_rank_int=18, gross_float=1.0,
    stock_sma_int=100, min_raw_close_float=0.0, min_dollar_volume_float=0.0, inverse_vol_bool=False,
)
LEG2_CONFIG = replace(LEG1_CONFIG, score_kind_str=SCORE_SECTOR_RESIDUAL_STR, market_symbol_str="$NDX")
LEG3_CONFIG = MomentumCapsuleLegConfig(
    score_kind_str=SCORE_INDUSTRY_RESIDUAL_STR, indexname_str="Russell 1000", gate_symbol_str="$RUI",
    market_symbol_str="$RUI", vxn_symbol_str=None, max_positions_int=20, sector_cap_int=4, hold_rank_int=35,
    gross_float=0.985, stock_sma_int=0, min_raw_close_float=10.0, min_dollar_volume_float=10_000_000.0,
    inverse_vol_bool=True,
)


# ---------------------------------------------------------------- the calendar of the two books
def is_last_session_of_month_index(session_idx: pd.DatetimeIndex) -> np.ndarray:
    month_vec = session_idx.to_period("M").asi8
    return np.r_[month_vec[1:] != month_vec[:-1], True]


def decision_date_by_book_dict(session_idx: pd.DatetimeIndex) -> dict[int, pd.DatetimeIndex]:
    """Decision sessions of each book. *** CRITICAL*** a month's last session comes from the exchange schedule, known in advance, so the T10 book of an unfinished month is dated correctly; its T0 decision does not exist until that session is in the data."""
    session_idx = pd.DatetimeIndex(session_idx)
    last_ts = session_idx[-1]
    month_start_ts, month_end_ts = last_ts.to_period("M").start_time.normalize(), last_ts.to_period("M").end_time.normalize()
    calendar_obj = exchange_calendars.get_calendar("XNYS", start=f"{last_ts.year - 1}-12-01", end=f"{last_ts.year + 1}-01-31")
    month_session_idx = pd.DatetimeIndex(calendar_obj.sessions_in_range(month_start_ts, month_end_ts)).tz_localize(None)
    last_row_vec = np.flatnonzero(is_last_session_of_month_index(session_idx))
    finished_bool = last_ts >= month_session_idx[-1]
    if not finished_bool:
        last_row_vec = last_row_vec[:-1]
    out_dict: dict[int, pd.DatetimeIndex] = {}
    for offset_int in BOOK_OFFSET_TUPLE:
        row_vec = last_row_vec - offset_int
        date_idx = session_idx[row_vec[row_vec >= 0]]
        if not finished_bool and offset_int > 0:
            open_month_date_ts = month_session_idx[-1 - offset_int]
            if open_month_date_ts <= last_ts:
                date_idx = date_idx.append(pd.DatetimeIndex([open_month_date_ts]))
        out_dict[offset_int] = date_idx
    return out_dict


def build_decision_df(session_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per decision, in date order: decision date, its book, and the next session where the data has it."""
    session_idx = pd.DatetimeIndex(session_idx)
    frame_list = []
    for offset_int, date_idx in decision_date_by_book_dict(session_idx).items():
        row_vec = session_idx.get_indexer(date_idx)
        if (row_vec < 0).any():
            raise RuntimeError(f"Book {offset_int} has decision dates missing from the price calendar.")
        frame_list.append(pd.DataFrame({"decision_date_ts": date_idx, "book_offset_int": offset_int, "decision_row_int": row_vec}))
    decision_df = pd.concat(frame_list, ignore_index=True).sort_values("decision_date_ts", kind="mergesort").reset_index(drop=True)
    if decision_df["decision_date_ts"].duplicated().any():
        raise RuntimeError("Two books decide on the same session.")
    execution_row_vec = decision_df["decision_row_int"].to_numpy() + 1
    decision_df["execution_date_ts"] = [session_idx[r] if r < len(session_idx) else pd.NaT for r in execution_row_vec]
    return decision_df


def build_rebalance_schedule_df(session_idx: pd.DatetimeIndex, start_date_str: str) -> pd.DataFrame:
    """Execution date to decision date, as the engine strategies keep it."""
    decision_df = build_decision_df(session_idx)
    usable_df = decision_df.loc[decision_df["execution_date_ts"].notna() & (decision_df["execution_date_ts"] >= pd.Timestamp(start_date_str))]
    schedule_df = pd.DataFrame({"decision_date_ts": usable_df["decision_date_ts"].to_numpy(), "book_offset_int": usable_df["book_offset_int"].to_numpy()}, index=pd.DatetimeIndex(usable_df["execution_date_ts"].to_numpy(), name="execution_date_ts"))
    if len(schedule_df) == 0:
        raise RuntimeError("No decision falls inside the backtest window.")
    return schedule_df


# ---------------------------------------------------------------- data
@dataclass
class LegPanel:
    """One leg's inputs as arrays on one calendar. Columns are sorted by symbol, so a stable sort on -score breaks ties by symbol."""

    date_index: pd.DatetimeIndex
    symbol_vec: np.ndarray
    close_mat: np.ndarray
    high_mat: np.ndarray
    low_mat: np.ndarray
    raw_close_mat: np.ndarray
    dollar_volume_mat: np.ndarray
    member_mat: np.ndarray
    sector_label_vec: np.ndarray
    group_label_vec: np.ndarray | None
    gate_close_vec: np.ndarray
    market_ret_vec: np.ndarray | None
    sector_ret_df: pd.DataFrame | None
    vxn_vec: np.ndarray | None


def current_gics_label_dict(symbol_list: Sequence[str], level_int: int) -> dict[str, str]:
    """Norgate's current GICS id at a level, or UNKNOWN when it has none. Current labels: a mild look-ahead, G-025. A snapshot run reads the labels captured with the snapshot and never calls the provider."""
    if is_snapshot_mode_enabled_bool():
        return load_momentum_legs_label_map(symbol_list, level_int)
    import norgatedata

    return {str(symbol_str): read_stable_gics_label_str(norgatedata, str(symbol_str), level_int) for symbol_str in symbol_list}


def load_leg_universe_df(config_obj: MomentumCapsuleLegConfig) -> tuple[list[str], pd.DataFrame]:
    """Every symbol with a member session in the window. *** CRITICAL*** the set is never read at a decision; each decision reads membership at its own T."""
    _, raw_universe_df = build_index_constituent_matrix(indexname=config_obj.indexname_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= pd.Timestamp(config_obj.history_start_date_str)]
    if config_obj.end_date_str is not None:
        filtered_universe_df = filtered_universe_df.loc[filtered_universe_df.index <= pd.Timestamp(config_obj.end_date_str)]
    window_df = filtered_universe_df.loc[filtered_universe_df.index >= pd.Timestamp(config_obj.member_start_date_str)]
    return sorted(str(symbol_str) for symbol_str in window_df.columns[window_df.sum(axis=0) > 0]), filtered_universe_df


def load_leg_pricing_data(config_obj: MomentumCapsuleLegConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stock prices plus the reporting benchmark, and audited membership aligned to the price calendar."""
    symbol_list, filtered_universe_df = load_leg_universe_df(config_obj)
    if len(symbol_list) == 0:
        raise RuntimeError(f"No {config_obj.indexname_str} members were found in the window.")
    pricing_data_df = load_raw_prices(symbols=symbol_list, benchmarks=[], start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str)
    loaded_symbol_list = [symbol_str for symbol_str in symbol_list if symbol_str in pricing_data_df.columns.get_level_values(0)]
    universe_df = audit_pit_universe_df(universe_df=filtered_universe_df, execution_idx=pricing_data_df.index, tradeable_symbol_list=loaded_symbol_list)
    pricing_data_df = append_total_return_benchmark_data_df(pricing_data_df=pricing_data_df, config_obj=config_obj)
    return pricing_data_df, universe_df


def helper_symbol_list(config_obj: MomentumCapsuleLegConfig) -> list[str]:
    symbol_list = [config_obj.gate_symbol_str]
    if config_obj.market_symbol_str is not None:
        symbol_list.append(config_obj.market_symbol_str)
    if config_obj.vxn_symbol_str is not None:
        symbol_list.append(config_obj.vxn_symbol_str)
    if config_obj.score_kind_str != SCORE_NATR_STR:
        symbol_list.extend(SECTOR_INDEX_BY_LABEL_DICT.values())
    return list(dict.fromkeys(symbol_list))


def load_helper_close_dict(config_obj: MomentumCapsuleLegConfig) -> dict[str, pd.Series]:
    if is_snapshot_mode_enabled_bool():
        # One read of the frozen long table, not one per helper. dropna only removes rows the join added.
        helper_df = load_raw_prices(symbols=helper_symbol_list(config_obj), benchmarks=[], start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str)
        return {symbol_str: helper_df[(symbol_str, "Close")].dropna().astype(float) for symbol_str in helper_symbol_list(config_obj)}
    return {
        symbol_str: load_price_timeseries(symbol_str, start_date_str=config_obj.history_start_date_str, end_date_str=config_obj.end_date_str)["Close"].astype(float)
        for symbol_str in helper_symbol_list(config_obj)
    }


def build_sector_return_df(helper_close_dict: Mapping[str, pd.Series], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily GICS sector index returns. *** CRITICAL*** a missing bar stays NaN; Real Estate before $SPXR is spliced from Financials in return space, date by date, so no later level enters."""
    return_dict: dict[str, pd.Series] = {}
    for label_str, symbol_str in SECTOR_INDEX_BY_LABEL_DICT.items():
        return_dict[label_str] = helper_close_dict[symbol_str].reindex(date_index).astype(float).pct_change(fill_method=None)
    return_dict[REAL_ESTATE_SECTOR_STR] = return_dict[REAL_ESTATE_SECTOR_STR].combine_first(return_dict[FINANCIALS_SECTOR_STR])
    return pd.DataFrame(return_dict, index=date_index).astype(float)


def build_leg_panel(
    config_obj: MomentumCapsuleLegConfig,
    pricing_data_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    helper_close_dict: Mapping[str, pd.Series],
    sector_label_dict: Mapping[str, str],
    group_label_dict: Mapping[str, str] | None,
) -> LegPanel:
    date_index = pd.DatetimeIndex(pricing_data_df.index)
    symbol_list = sorted(str(symbol_str) for symbol_str in universe_df.columns)

    def field_mat(field_str: str) -> np.ndarray:
        frame_df = pricing_data_df.xs(field_str, axis=1, level=1)
        return frame_df.reindex(columns=symbol_list).to_numpy(dtype=float)

    member_mat = universe_df.reindex(index=date_index, columns=symbol_list).fillna(0).to_numpy(dtype=bool)
    vxn_vec = None
    if config_obj.vxn_symbol_str is not None:
        vxn_ser = helper_close_dict[config_obj.vxn_symbol_str].astype(float)
        vxn_ser = vxn_ser[np.isfinite(vxn_ser) & (vxn_ser > 0.0)]
        # *** CRITICAL*** VXN at a session is the last valid close dated on or before it.
        vxn_vec = vxn_ser.reindex(vxn_ser.index.union(date_index)).ffill().reindex(date_index).to_numpy(dtype=float)
    market_ret_vec = None
    if config_obj.market_symbol_str is not None:
        market_ret_vec = helper_close_dict[config_obj.market_symbol_str].reindex(date_index).astype(float).pct_change(fill_method=None).to_numpy(dtype=float)
    sector_ret_df = None if config_obj.score_kind_str == SCORE_NATR_STR else build_sector_return_df(helper_close_dict, date_index)
    return LegPanel(
        date_index=date_index,
        symbol_vec=np.array(symbol_list, dtype=object),
        close_mat=field_mat("Close"),
        high_mat=field_mat("High"),
        low_mat=field_mat("Low"),
        raw_close_mat=field_mat("Unadjusted Close"),
        dollar_volume_mat=field_mat("Turnover"),
        member_mat=member_mat,
        sector_label_vec=np.array([sector_label_dict.get(s, UNKNOWN_LABEL_STR) for s in symbol_list], dtype=object),
        group_label_vec=None if group_label_dict is None else np.array([group_label_dict.get(s, UNKNOWN_LABEL_STR) for s in symbol_list], dtype=object),
        gate_close_vec=helper_close_dict[config_obj.gate_symbol_str].reindex(date_index).to_numpy(dtype=float),
        market_ret_vec=market_ret_vec,
        sector_ret_df=sector_ret_df,
        vxn_vec=vxn_vec,
    )


# ---------------------------------------------------------------- scores
def daily_return_mat(close_mat: np.ndarray) -> np.ndarray:
    ret_mat = np.full(close_mat.shape, np.nan)
    # *** CRITICAL*** row t is the return of session t, known after its close; a missing bar stays NaN.
    ret_mat[1:] = close_mat[1:] / close_mat[:-1] - 1.0
    return ret_mat


def _one_factor_residual_mat(y_mat: np.ndarray, market_vec: np.ndarray) -> np.ndarray:
    """u = y - m b, OLS with intercept on centered data; the intercept stays in u."""
    market_center_vec = market_vec - market_vec.mean()
    beta_vec = (market_center_vec @ (y_mat - y_mat.mean(axis=0))) / float(market_center_vec @ market_center_vec)
    return y_mat - np.outer(market_vec, beta_vec)


def _two_factor_residual_mat(y_mat: np.ndarray, market_vec: np.ndarray, second_mat: np.ndarray) -> np.ndarray:
    """u = y - b_m m - b_s s, where s differs by column. A degenerate second factor gives NaN and the caller keeps the market-only residual."""
    market_center_vec = market_vec - market_vec.mean()
    y_center_mat = y_mat - y_mat.mean(axis=0)
    second_center_mat = second_mat - second_mat.mean(axis=0)
    s_mm_float = float(market_center_vec @ market_center_vec)
    s_my_vec = market_center_vec @ y_center_mat
    s_ms_vec = market_center_vec @ second_center_mat
    s_ss_vec = np.einsum("ij,ij->j", second_center_mat, second_center_mat)
    s_sy_vec = np.einsum("ij,ij->j", second_center_mat, y_center_mat)
    det_vec = s_mm_float * s_ss_vec - s_ms_vec * s_ms_vec
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_m_vec = np.where(det_vec > COLLINEARITY_TOLERANCE_FLOAT * s_mm_float * s_ss_vec, (s_ss_vec * s_my_vec - s_ms_vec * s_sy_vec) / det_vec, np.nan)
        beta_s_vec = np.where(det_vec > COLLINEARITY_TOLERANCE_FLOAT * s_mm_float * s_ss_vec, (s_mm_float * s_sy_vec - s_ms_vec * s_my_vec) / det_vec, np.nan)
    return y_mat - np.outer(market_vec, beta_m_vec) - second_mat * beta_s_vec


def natr_score_dict(config_obj: MomentumCapsuleLegConfig, panel: LegPanel, row_vec: np.ndarray) -> dict[int, np.ndarray]:
    """ROC over this book's own earlier decisions, over NATR. *** CRITICAL*** all read at T; both price units cancel, so a later split cannot change the ratio."""
    close_mat = panel.close_mat
    previous_mat = np.vstack([np.full((1, close_mat.shape[1]), np.nan), close_mat[:-1]])
    true_range_mat = np.maximum(panel.high_mat - panel.low_mat, np.maximum(np.abs(panel.high_mat - previous_mat), np.abs(panel.low_mat - previous_mat)))
    atr_mat = pd.DataFrame(true_range_mat).rolling(config_obj.natr_window_int, min_periods=config_obj.natr_window_int).mean().to_numpy()
    out_dict: dict[int, np.ndarray] = {}
    for position_int, row_int in enumerate(row_vec):
        if position_int < config_obj.roc_decision_count_int:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            score_vec = close_mat[row_int] / close_mat[row_vec[position_int - config_obj.roc_decision_count_int]] - 1.0
            score_vec = score_vec / (atr_mat[row_int] / close_mat[row_int])
        out_dict[int(row_int)] = np.where(np.isfinite(score_vec), score_vec, np.nan)
    return out_dict


def sector_residual_score_dict(config_obj: MomentumCapsuleLegConfig, panel: LegPanel, row_vec: np.ndarray) -> dict[int, np.ndarray]:
    """mean(u) / std(u) over T-L+1..T-S. *** CRITICAL*** the regression ends at T; the momentum window ends S sessions before T."""
    window_int, look_int, skip_int = config_obj.residual_window_int, config_obj.residual_look_int, config_obj.residual_skip_int
    ret_mat = daily_return_mat(panel.close_mat)
    sector_ret_mat = panel.sector_ret_df.to_numpy(dtype=float)
    sector_col_dict = {label_str: k for k, label_str in enumerate(panel.sector_ret_df.columns)}
    sector_col_vec = np.array([sector_col_dict.get(label_str, -1) for label_str in panel.sector_label_vec], dtype=int)
    out_dict: dict[int, np.ndarray] = {}
    for row_int in row_vec:
        row_int = int(row_int)
        score_vec = np.full(ret_mat.shape[1], np.nan)
        out_dict[row_int] = score_vec
        first_int = row_int - window_int + 1
        if first_int < 1:
            continue
        window_mat = ret_mat[first_int:row_int + 1]
        usable_vec = np.isfinite(window_mat).all(axis=0) & ((window_mat == 0.0).mean(axis=0) <= MAX_ZERO_RETURN_FRACTION_FLOAT) & panel.member_mat[row_int]
        column_vec = np.flatnonzero(usable_vec)
        market_vec = panel.market_ret_vec[first_int:row_int + 1]
        if column_vec.size == 0 or not np.isfinite(market_vec).all():
            continue
        y_mat = window_mat[:, column_vec]
        u_mat = _one_factor_residual_mat(y_mat, market_vec)
        stock_sector_vec = sector_col_vec[column_vec]
        sector_window_mat = sector_ret_mat[first_int:row_int + 1]
        second_mat = sector_window_mat[:, np.maximum(stock_sector_vec, 0)]
        sector_usable_vec = np.flatnonzero((stock_sector_vec >= 0) & np.isfinite(second_mat).all(axis=0))
        if sector_usable_vec.size:
            two_factor_mat = _two_factor_residual_mat(y_mat[:, sector_usable_vec], market_vec, second_mat[:, sector_usable_vec])
            solved_vec = np.isfinite(two_factor_mat).all(axis=0)  # a degenerate second factor keeps the market-only residual
            u_mat[:, sector_usable_vec[solved_vec]] = two_factor_mat[:, solved_vec]
        momentum_mat = u_mat[window_int - look_int:window_int - skip_int]
        std_vec = momentum_mat.std(axis=0, ddof=1)
        pass_vec = np.isfinite(std_vec) & (std_vec >= MIN_DAILY_RESIDUAL_VOL_FLOAT)
        score_vec[column_vec[pass_vec]] = momentum_mat[:, pass_vec].mean(axis=0) / std_vec[pass_vec]
    return out_dict


def industry_residual_score_dict(config_obj: MomentumCapsuleLegConfig, panel: LegPanel, row_vec: np.ndarray) -> dict[int, np.ndarray]:
    """The outside study's score: cumulative residual over its volatility, on a market and leave-one-out peer regression.
    *** CRITICAL*** the whole window ends at T - skip. A peer is a member on that return date, never by later membership, and a stock's own return leaves its group's sum on every day it was in it."""
    window_int, skip_int = config_obj.peer_window_int, config_obj.peer_skip_int
    ret_mat = daily_return_mat(panel.close_mat)
    symbol_count_int = ret_mat.shape[1]
    sector_ret_mat = panel.sector_ret_df.to_numpy(dtype=float)
    sector_col_dict = {label_str: k for k, label_str in enumerate(panel.sector_ret_df.columns)}
    stock_sector_vec = np.array([sector_col_dict.get(label_str, -1) for label_str in panel.sector_label_vec], dtype=int)
    contribution_mat = panel.member_mat & np.isfinite(ret_mat)
    safe_ret_mat = np.where(contribution_mat, ret_mat, 0.0)
    group_list = sorted({str(label_str) for label_str in panel.group_label_vec if str(label_str) != UNKNOWN_LABEL_STR})
    group_col_dict = {label_str: k for k, label_str in enumerate(group_list)}
    stock_group_vec = np.array([group_col_dict.get(str(label_str), -1) for label_str in panel.group_label_vec], dtype=int)
    group_sum_mat = np.zeros((ret_mat.shape[0], max(len(group_list), 1)))
    group_count_mat = np.zeros((ret_mat.shape[0], max(len(group_list), 1)))
    for group_int in range(len(group_list)):
        mask_vec = stock_group_vec == group_int
        group_sum_mat[:, group_int] = safe_ret_mat[:, mask_vec].sum(axis=1)
        group_count_mat[:, group_int] = contribution_mat[:, mask_vec].sum(axis=1)
    sqrt_window_float = float(np.sqrt(float(window_int)))
    out_dict: dict[int, np.ndarray] = {}
    for row_int in row_vec:
        row_int = int(row_int)
        score_vec = np.full(symbol_count_int, np.nan)
        out_dict[row_int] = score_vec
        last_int = row_int - skip_int
        first_int = last_int - window_int + 1
        if first_int < 0:
            continue
        window_slice = slice(first_int, last_int + 1)
        market_vec = panel.market_ret_vec[window_slice]
        if not np.isfinite(market_vec).all():
            continue
        market_center_vec = market_vec - float(np.mean(market_vec))
        s_mm_float = float(market_center_vec @ market_center_vec)
        if s_mm_float <= COLLINEARITY_TOLERANCE_FLOAT:
            continue
        y_window_mat = ret_mat[window_slice, :]
        valid_vec = np.isfinite(y_window_mat).all(axis=0) & (np.mean(y_window_mat == 0.0, axis=0) <= MAX_ZERO_RETURN_FRACTION_FLOAT)
        column_vec = np.flatnonzero(valid_vec)
        if column_vec.size == 0:
            continue
        valid_count_int = column_vec.size
        y_mat = y_window_mat[:, column_vec]
        y_center_mat = y_mat - y_mat.mean(axis=0)
        s_my_vec = market_center_vec @ y_center_mat
        beta_m_vec = s_my_vec / s_mm_float
        beta_s_vec = np.zeros(valid_count_int)
        second_mat = np.zeros((window_int, valid_count_int))
        has_second_vec = np.zeros(valid_count_int, dtype=bool)
        # Baseline second factor: the stock's GICS level-1 sector index.
        sector_window_mat = sector_ret_mat[window_slice, :]
        sector_valid_vec = np.isfinite(sector_window_mat).all(axis=0)
        column_sector_vec = stock_sector_vec[column_vec]
        sector_position_vec = np.flatnonzero((column_sector_vec >= 0) & sector_valid_vec[np.maximum(column_sector_vec, 0)])
        if sector_position_vec.size:
            second_mat[:, sector_position_vec] = sector_window_mat[:, column_sector_vec[sector_position_vec]]
            has_second_vec[sector_position_vec] = True
        # Override with the leave-one-out peer return when the group has the peers on every session.
        column_group_vec = stock_group_vec[column_vec]
        group_position_vec = np.flatnonzero(column_group_vec >= 0)
        if group_position_vec.size:
            self_mat = contribution_mat[window_slice, :][:, column_vec[group_position_vec]].astype(float)
            peer_sum_mat = group_sum_mat[window_slice, :][:, column_group_vec[group_position_vec]] - self_mat * y_mat[:, group_position_vec]
            peer_count_mat = group_count_mat[window_slice, :][:, column_group_vec[group_position_vec]] - self_mat
            enough_vec = np.flatnonzero(peer_count_mat.min(axis=0) >= float(MIN_INDUSTRY_PEER_COUNT_INT))
            if enough_vec.size:
                peer_mat = peer_sum_mat[:, enough_vec] / peer_count_mat[:, enough_vec]
                finite_vec = np.isfinite(peer_mat).all(axis=0)
                final_position_vec = group_position_vec[enough_vec][finite_vec]
                second_mat[:, final_position_vec] = peer_mat[:, finite_vec]
                has_second_vec[final_position_vec] = True
        second_position_vec = np.flatnonzero(has_second_vec)
        if second_position_vec.size:
            second_center_mat = second_mat[:, second_position_vec] - second_mat[:, second_position_vec].mean(axis=0)
            s_ms_vec = market_center_vec @ second_center_mat
            s_ss_vec = np.einsum("ij,ij->j", second_center_mat, second_center_mat)
            det_vec = s_mm_float * s_ss_vec - np.square(s_ms_vec)
            solved_vec = np.flatnonzero((s_ss_vec > COLLINEARITY_TOLERANCE_FLOAT) & (det_vec > COLLINEARITY_TOLERANCE_FLOAT * s_mm_float * s_ss_vec))
            if solved_vec.size:
                solved_position_vec = second_position_vec[solved_vec]
                s_sy_vec = np.einsum("ij,ij->j", second_center_mat[:, solved_vec], y_center_mat[:, solved_position_vec])
                beta_m_vec[solved_position_vec] = (s_ss_vec[solved_vec] * s_my_vec[solved_position_vec] - s_ms_vec[solved_vec] * s_sy_vec) / det_vec[solved_vec]
                beta_s_vec[solved_position_vec] = (s_mm_float * s_sy_vec - s_ms_vec[solved_vec] * s_my_vec[solved_position_vec]) / det_vec[solved_vec]
        u_mat = y_mat - np.outer(market_vec, beta_m_vec) - second_mat * beta_s_vec[np.newaxis, :]
        mean_vec = u_mat.mean(axis=0)
        total_vec = u_mat.sum(axis=0)
        centered_mat = u_mat - mean_vec[np.newaxis, :]
        square_vec = np.einsum("ij,ij->j", centered_mat, centered_mat)
        freedom_vec = np.where(beta_s_vec != 0.0, float(window_int - 3), float(window_int - 2))
        vol_vec = np.sqrt(np.maximum(square_vec, 0.0) / freedom_vec)
        pass_vec = np.isfinite(vol_vec) & (vol_vec >= MIN_DAILY_RESIDUAL_VOL_FLOAT)
        score_vec[column_vec[pass_vec]] = total_vec[pass_vec] / (sqrt_window_float * vol_vec[pass_vec])
    return out_dict


SCORE_FUNCTION_DICT = {
    SCORE_NATR_STR: natr_score_dict,
    SCORE_SECTOR_RESIDUAL_STR: sector_residual_score_dict,
    SCORE_INDUSTRY_RESIDUAL_STR: industry_residual_score_dict,
}


# ---------------------------------------------------------------- one book at one decision
def pick_with_rank_buffer(ranked_vec: np.ndarray, held_set: set[str], symbol_vec: np.ndarray, label_vec: np.ndarray, max_positions_int: int, sector_cap_int: int, hold_rank_int: int) -> np.ndarray:
    """Held names still inside the rank limit stay first, in rank order, then the slots fill by rank; the sector cap binds throughout."""
    picked_list: list[int] = []
    picked_set: set[int] = set()
    count_dict: dict[str, int] = {}
    for column_int in ranked_vec[:hold_rank_int]:
        if len(picked_list) >= max_positions_int:
            break
        if str(symbol_vec[column_int]) in held_set and count_dict.get(label_vec[column_int], 0) < sector_cap_int:
            picked_list.append(int(column_int))
            picked_set.add(int(column_int))
            count_dict[label_vec[column_int]] = count_dict.get(label_vec[column_int], 0) + 1
    for column_int in ranked_vec:
        if len(picked_list) >= max_positions_int:
            break
        if int(column_int) not in picked_set and count_dict.get(label_vec[column_int], 0) < sector_cap_int:
            picked_list.append(int(column_int))
            picked_set.add(int(column_int))
            count_dict[label_vec[column_int]] = count_dict.get(label_vec[column_int], 0) + 1
    return np.array(picked_list, dtype=int)


def inverse_vol_weight_vec(vol_vec: np.ndarray, gross_float: float, max_positions_int: int) -> np.ndarray:
    """Each filled slot brings gross / N, shared by inverse volatility, capped at 1.5 slots by water-filling. Unfilled slots stay in cash."""
    slot_float = float(gross_float) / float(max_positions_int)
    safe_vec = np.where(np.isfinite(vol_vec) & (vol_vec > INVERSE_VOL_FLOOR_FLOAT), vol_vec, np.nan)
    if not np.isfinite(safe_vec).all():
        median_float = float(np.nanmedian(safe_vec)) if np.isfinite(safe_vec).any() else 0.25
        safe_vec = np.where(np.isfinite(safe_vec), safe_vec, median_float)
    inverse_vec = 1.0 / safe_vec
    cap_float = INVERSE_VOL_MAX_SLOTS_FLOAT * slot_float
    weight_vec = slot_float * float(inverse_vec.size) * inverse_vec / float(inverse_vec.sum())
    for _ in range(10):
        over_vec = weight_vec > cap_float + 1e-12
        if not over_vec.any():
            break
        excess_float = float((weight_vec[over_vec] - cap_float).sum())
        weight_vec[over_vec] = cap_float
        under_vec = ~over_vec
        if not under_vec.any() or float(weight_vec[under_vec].sum()) <= 0.0:
            break
        weight_vec[under_vec] += excess_float * weight_vec[under_vec] / float(weight_vec[under_vec].sum())
    return np.minimum(weight_vec, cap_float)


def rolling_mean_mat(value_mat: np.ndarray, window_int: int) -> np.ndarray:
    return pd.DataFrame(value_mat).rolling(window_int, min_periods=window_int).mean().to_numpy()


@dataclass
class TargetPath:
    """target_by_decision_dict holds what the account targets after each decision; book_by_decision_dict that decision's own book; evidence_by_decision_dict what the decision read, for a live decision to refuse unavailable data."""

    target_by_decision_dict: dict[pd.Timestamp, pd.Series]
    book_by_decision_dict: dict[pd.Timestamp, pd.Series]
    book_offset_by_decision_dict: dict[pd.Timestamp, int]
    evidence_by_decision_dict: dict[pd.Timestamp, dict[str, object]]


def build_target_path(config_obj: MomentumCapsuleLegConfig, panel: LegPanel) -> TargetPath:
    """Replay the leg's decisions from the design's first one, whatever the run's window, so the rank buffers have one path. *** CRITICAL*** at each decision only rows through T are read: scores, filters, gate and scale are read at T, the buffer reads this book's earlier picks. The result for T does not change when later rows are added; tests/test_strategy_mo_capsule_legs.py holds that on a cut panel."""
    decision_df = build_decision_df(panel.date_index)
    score_fn = SCORE_FUNCTION_DICT[config_obj.score_kind_str]
    score_dict: dict[int, np.ndarray] = {}
    for offset_int in BOOK_OFFSET_TUPLE:
        row_vec = decision_df.loc[decision_df["book_offset_int"] == offset_int, "decision_row_int"].to_numpy()
        score_dict.update(score_fn(config_obj, panel, row_vec))
    gate_ser = pd.Series(panel.gate_close_vec)
    gate_level_ser = gate_ser.rolling(config_obj.gate_sma_int, min_periods=config_obj.gate_sma_int).mean()
    gate_vec = (gate_ser > gate_level_ser).where(gate_level_ser.notna(), False).astype(bool).to_numpy()
    stock_sma_mat = rolling_mean_mat(panel.close_mat, config_obj.stock_sma_int) if config_obj.stock_sma_int else None
    dollar_volume_mat = rolling_mean_mat(panel.dollar_volume_mat, config_obj.dollar_volume_window_int) if config_obj.min_dollar_volume_float > 0.0 else None
    vol_mat = None
    if config_obj.inverse_vol_bool:
        # *** CRITICAL*** 63-day volatility ends at T.
        vol_mat = pd.DataFrame(daily_return_mat(panel.close_mat)).rolling(config_obj.inverse_vol_window_int, min_periods=config_obj.inverse_vol_window_int).std(ddof=1).to_numpy() * np.sqrt(252.0)
    start_ts = min(pd.Timestamp(config_obj.path_start_date_str), pd.Timestamp(config_obj.backtest_start_date_str))
    column_by_symbol_dict = {str(symbol_str): column_int for column_int, symbol_str in enumerate(panel.symbol_vec)}
    held_dict: dict[int, set[str]] = {offset_int: set() for offset_int in BOOK_OFFSET_TUPLE}
    latest_book_dict: dict[int, pd.Series] = {offset_int: pd.Series(dtype=float) for offset_int in BOOK_OFFSET_TUPLE}
    target_by_decision_dict: dict[pd.Timestamp, pd.Series] = {}
    book_by_decision_dict: dict[pd.Timestamp, pd.Series] = {}
    book_offset_by_decision_dict: dict[pd.Timestamp, int] = {}
    evidence_by_decision_dict: dict[pd.Timestamp, dict[str, object]] = {}
    for decision_row in decision_df.itertuples(index=False):
        execution_ts = decision_row.execution_date_ts
        if pd.notna(execution_ts) and execution_ts < start_ts:
            continue
        row_int, offset_int = int(decision_row.decision_row_int), int(decision_row.book_offset_int)
        decision_ts = pd.Timestamp(decision_row.decision_date_ts)
        score_vec = score_dict.get(row_int)
        book_ser = pd.Series(dtype=float)
        member_vec = panel.member_mat[row_int]
        evidence_by_decision_dict[decision_ts] = {
            "gate_open_bool": bool(gate_vec[row_int]),
            "gate_close_float": float(panel.gate_close_vec[row_int]),
            "gate_level_float": float(gate_level_ser.iloc[row_int]),
            "member_count_int": int(member_vec.sum()),
            "finite_score_member_count_int": 0 if score_vec is None else int(np.isfinite(score_vec[member_vec]).sum()),
            "vxn_close_float": None if panel.vxn_vec is None else float(panel.vxn_vec[row_int]),
        }
        if gate_vec[row_int] and score_vec is not None:
            close_vec = panel.close_mat[row_int]
            with np.errstate(invalid="ignore"):
                mask_vec = panel.member_mat[row_int] & np.isfinite(close_vec) & np.isfinite(score_vec)
                if stock_sma_mat is not None:
                    mask_vec = mask_vec & (close_vec > stock_sma_mat[row_int])
                if config_obj.min_raw_close_float > 0.0:
                    mask_vec = mask_vec & (close_vec > 0.0) & (panel.raw_close_mat[row_int] > config_obj.min_raw_close_float)
                if dollar_volume_mat is not None:
                    mask_vec = mask_vec & (dollar_volume_mat[row_int] > config_obj.min_dollar_volume_float)
            column_vec = np.flatnonzero(mask_vec)
            if column_vec.size:
                # Ties: symbol ascending. Columns are sorted by symbol and this sort is stable.
                ranked_vec = column_vec[np.argsort(-score_vec[column_vec], kind="mergesort")]
                picked_vec = pick_with_rank_buffer(ranked_vec, held_dict[offset_int], panel.symbol_vec, panel.sector_label_vec, config_obj.max_positions_int, config_obj.sector_cap_int, config_obj.hold_rank_int)
                if picked_vec.size:
                    if vol_mat is not None:
                        weight_vec = inverse_vol_weight_vec(vol_mat[row_int, picked_vec], config_obj.gross_float, config_obj.max_positions_int)
                    else:
                        scale_float = 1.0 if panel.vxn_vec is None else float(np.clip(config_obj.target_vxn_float / panel.vxn_vec[row_int], config_obj.min_vxn_scale_float, 1.0))
                        weight_vec = np.full(picked_vec.size, config_obj.gross_float * scale_float / config_obj.max_positions_int)
                    book_ser = pd.Series(weight_vec, index=panel.symbol_vec[picked_vec].astype(str), dtype=float)
        held_dict[offset_int] = set(book_ser.index)
        latest_book_dict[offset_int] = book_ser
        # *** CRITICAL*** the account's target reads only the two books decided so far.
        target_ser = BOOK_SHARE_FLOAT * latest_book_dict[BOOK_OFFSET_TUPLE[0]].add(latest_book_dict[BOOK_OFFSET_TUPLE[1]], fill_value=0.0)
        # A name of the other book without a close at T stopped trading after that book's decision: its share stays in cash, as in the research form.
        priced_vec = np.array([np.isfinite(panel.close_mat[row_int, column_by_symbol_dict[symbol_str]]) for symbol_str in target_ser.index], dtype=bool)
        target_by_decision_dict[decision_ts] = target_ser.loc[priced_vec].sort_index()
        book_by_decision_dict[decision_ts] = book_ser.sort_index()
        book_offset_by_decision_dict[decision_ts] = offset_int
    return TargetPath(target_by_decision_dict, book_by_decision_dict, book_offset_by_decision_dict, evidence_by_decision_dict)


# ---------------------------------------------------------------- the engine strategy
class MomentumCapsuleLegStrategy(Strategy):
    """A long-only leg. Each swap day the account goes to the path's target for that decision, on the prior close's value and prices."""

    enable_signal_audit = False
    leg_config_obj: MomentumCapsuleLegConfig

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        capital_base: float,
        slippage: float,
        commission_per_share: float,
        commission_minimum: float,
    ):
        super().__init__(name=name, benchmarks=list(benchmarks), capital_base=capital_base, slippage=slippage, commission_per_share=commission_per_share, commission_minimum=commission_minimum)
        if len(rebalance_schedule_df) == 0 or "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must hold decision_date_ts.")
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.schedule_date_set = set(pd.DatetimeIndex(self.rebalance_schedule_df.index))
        # Research execution rounds and charges fees in historical nominal share units.
        self.historical_share_units_bool = True
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.target_path_obj: TargetPath | None = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        # The path was built from the whole universe before the run; this frame holds only what it trades.
        self._data_adjustment_policy_dict.update({
            "ranking_formula_str": self.leg_config_obj.score_kind_str,
            "book_form_str": "one_account_average_of_two_books",
            "research_history_status_str": "previously_seen_2000_2026_not_untouched_holdout",
        })
        return pricing_data_df

    def restrict_data(self, full_data: pd.DataFrame) -> tuple:
        # Only a swap day reads prices. *** CRITICAL*** the row is the previous bar's.
        if self.previous_bar is None or self.current_bar not in self.schedule_date_set:
            return None, None, None
        return None, full_data.loc[self.previous_bar], None

    def get_target_share_map(self, target_weight_ser: pd.Series, close_row_ser: pd.Series) -> dict[str, float]:
        budget_value_float = float(self.previous_total_value)
        share_map: dict[str, float] = {}
        for symbol_str, weight_float in target_weight_ser.items():
            close_float = float(close_row_ser[(symbol_str, "Close")])
            if not np.isfinite(close_float) or close_float <= 0.0:
                raise RuntimeError(f"Invalid prior close for target asset {symbol_str} on {self.previous_bar}.")
            # *** CRITICAL*** target shares are fixed from the previous_bar close, so a rebalance cannot adapt to the current open.
            share_float = self.historical_share_amount_float(budget_value_float * float(weight_float), close_float, float(close_row_ser[(symbol_str, "Unadjusted Close")]))
            if share_float > 0:
                share_map[str(symbol_str)] = share_float
        return share_map

    def iterate(self, data, close, open_prices):
        if close is None or self.current_bar not in self.schedule_date_set:
            return
        decision_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** the scheduled decision close must equal previous_bar, otherwise signals and next-open execution drift.
        if pd.Timestamp(self.previous_bar) != decision_ts:
            raise RuntimeError(f"Schedule misalignment on {self.current_bar}: decision_date_ts={decision_ts}, previous_bar={self.previous_bar}.")
        target_weight_ser = self.target_path_obj.target_by_decision_dict[decision_ts]
        target_share_map = self.get_target_share_map(target_weight_ser, close)
        position_ser = self.get_positions()
        for symbol_str in position_ser[position_ser > 0].index:
            if symbol_str not in target_share_map:
                self.order_target_value(symbol_str, 0.0, trade_id=self.current_trade_map[symbol_str])
        for symbol_str, share_float in target_share_map.items():
            current_share_float = float(position_ser.get(symbol_str, 0.0))
            if current_share_float == share_float:
                continue
            if current_share_float == 0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target_percent(symbol_str, float(target_weight_ser.loc[symbol_str]), trade_id=self.current_trade_map[symbol_str])


def configured_leg_config(default_config_obj: MomentumCapsuleLegConfig, backtest_start_date_str: str | None, capital_base_float: float | None, end_date_str: str | None) -> MomentumCapsuleLegConfig:
    if backtest_start_date_str is None and capital_base_float is None and end_date_str is None:
        return default_config_obj
    return replace(
        default_config_obj,
        backtest_start_date_str=default_config_obj.backtest_start_date_str if backtest_start_date_str is None else backtest_start_date_str,
        capital_base_float=default_config_obj.capital_base_float if capital_base_float is None else float(capital_base_float),
        end_date_str=end_date_str,
    )


def build_leg_strategy(strategy_cls: type, name_str: str, config_obj: MomentumCapsuleLegConfig, rebalance_schedule_df: pd.DataFrame, target_path_obj: TargetPath):
    strategy_obj = strategy_cls(
        name=name_str,
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
    )
    strategy_obj.leg_config_obj = config_obj
    strategy_obj.target_path_obj = target_path_obj
    configure_total_return_benchmark_provenance(strategy_obj=strategy_obj, config_obj=config_obj)
    return strategy_obj


def load_leg_inputs(config_obj: MomentumCapsuleLegConfig) -> dict[str, object]:
    pricing_data_df, universe_df = load_leg_pricing_data(config_obj)
    symbol_list = [str(symbol_str) for symbol_str in universe_df.columns]
    return {
        "pricing_data_df": pricing_data_df,
        "universe_df": universe_df,
        "helper_close_dict": load_helper_close_dict(config_obj),
        "sector_label_dict": current_gics_label_dict(symbol_list, 1),
        "group_label_dict": current_gics_label_dict(symbol_list, 2) if config_obj.score_kind_str == SCORE_INDUSTRY_RESIDUAL_STR else None,
    }


class SharedProvenanceDict(dict):
    """Pandas deep-copies a frame's attrs on every .loc. This read-only map needs no copy: with thousands of names the copies were most of a run's time."""

    def __deepcopy__(self, memo):
        return self


def traded_pricing_df(pricing_data_df: pd.DataFrame, universe_df: pd.DataFrame, target_path_obj: TargetPath) -> pd.DataFrame:
    """The names the path ever holds plus the reporting benchmark. Nothing the engine does reads a name the account never holds."""
    universe_set = {str(symbol_str) for symbol_str in universe_df.columns}
    traded_set = {str(symbol_str) for target_ser in target_path_obj.target_by_decision_dict.values() for symbol_str in target_ser.index}
    symbol_level_ser = pricing_data_df.columns.get_level_values(0)
    traded_df = pricing_data_df.loc[:, symbol_level_ser.isin(traded_set) | ~symbol_level_ser.isin(universe_set)]
    traded_df.attrs = {key_str: SharedProvenanceDict(value_obj) if isinstance(value_obj, dict) else value_obj for key_str, value_obj in pricing_data_df.attrs.items()}
    return traded_df


def prepare_leg_strategy(strategy_cls: type, name_str: str, config_obj: MomentumCapsuleLegConfig):
    inputs_dict = load_leg_inputs(config_obj)
    pricing_data_df = inputs_dict.pop("pricing_data_df")
    target_path_obj = build_target_path(config_obj, build_leg_panel(config_obj, pricing_data_df, **inputs_dict))
    schedule_df = build_rebalance_schedule_df(pricing_data_df.index, config_obj.backtest_start_date_str)
    return build_leg_strategy(strategy_cls, name_str, config_obj, schedule_df, target_path_obj), traded_pricing_df(pricing_data_df, inputs_dict["universe_df"], target_path_obj)


def run_leg_strategy(strategy_obj, pricing_data_df: pd.DataFrame, config_obj: MomentumCapsuleLegConfig, show_display_bool: bool) -> None:
    # *** CRITICAL*** history before the start stays in the frame for the features; the executable calendar starts at the first deployment fill session.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool, audit_override_bool=None)


def leg_timing_inputs(strategy_cls: type, name_str: str, config_obj: MomentumCapsuleLegConfig) -> dict[str, object]:
    """Execution timing analysis. Every cell gets the same decisions, so the path is built once."""
    from strategies.momentum.strategy_mo_atr_normalized_ndx import _map_rebalance_schedule_to_decision_close_schedule_df

    inputs_dict = load_leg_inputs(config_obj)
    pricing_data_df = inputs_dict.pop("pricing_data_df")
    schedule_df = build_rebalance_schedule_df(pricing_data_df.index, config_obj.backtest_start_date_str)
    decision_schedule_df = _map_rebalance_schedule_to_decision_close_schedule_df(rebalance_schedule_df=schedule_df)
    target_path_obj = build_target_path(config_obj, build_leg_panel(config_obj, pricing_data_df, **inputs_dict))
    traded_df = traded_pricing_df(pricing_data_df, inputs_dict["universe_df"], target_path_obj)

    def strategy_factory_fn():
        strategy_obj = build_leg_strategy(strategy_cls, name_str, config_obj, decision_schedule_df, target_path_obj)
        strategy_obj.trade_id_int = 0
        strategy_obj.current_trade_map = defaultdict(default_trade_id_int)
        return strategy_obj

    calendar_idx = traded_df.index[traded_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": traded_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }
