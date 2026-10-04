"""Scout spec of the US sector ETF IBS Downshock pod, VOX/IYR basket (PM_READY;
`strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr`), and the shared machinery of the sector
ETF IBS event rules (the dispersion family in `sector_dispersion_ibs.py` reuses it).

An independent re-implementation of the signal; execution is the shared weights engine with the path-dependent
`decision_fn` hook and two opt-in switches (`fractional_shares_bool`, `hold_nan_bool`): the engine sizes a new entry in
fractional shares and never touches a held position until its exit. Mapped on 2026-10-02 against main b3855fc:
strategy_mr_us_sector_etf_ibs_downshock.py (lines "D:"), strategy_mr_sector_dispersion_ibs.py (lines "S:") and
alpha/engine (strategy.py process_orders, backtester.py).

Data (D: get_us_sector_etf_ibs_downshock_data :453-477 -> data.norgate_loader.load_raw_prices; VOX/IYR DEFAULT_CONFIG)
    CAPITALSPECIAL Open/High/Low/Close/Dividend of XLB XLE XLF XLI XLK XLP XLU XLV XLY VOX IYR plus the $SPX benchmark
    (read as its TR data symbol; it only widens the date index), from 1998-01-01 to today.

Signal at the close of T, per ETF (D: compute_us_sector_etf_ibs_downshock_signal_df :162-288)
    IBS_T          (Close_T - Low_T) / (High_T - Low_T); a zero range is NaN (alpha.indicators.ibs_indicator)
    Range_T        ln(High_T / Low_T) where High > 0, Low > 0, High > Low, else NaN
    RangeRatio_T   Range_T / median(Range_(T-21) .. Range_(T-1)) (rolling 21 with min_periods 21, shifted one session;
                   a zero median is NaN)
    ATR14          Wilder ATR (TA-Lib: first value = mean of the first 14 true ranges, then (ATR x 13 + TR) / 14) on the
                   ETF's own rows with finite High, Low and Close, from the start of the history
    DownShock_T    (Close_T / Close_prev - 1) / (ATR14_prev / Close_prev); "prev" = the ETF's previous valid row
    NATR_prev      100 x ATR14_prev / Close_prev (the ranking score)
    entry_T        IBS_T < 0.05 and DownShock_T < -0.5        (NaN -> False)
    exit_T         IBS_T > 0.90 and RangeRatio_T > 1          (NaN -> False)

Decision after the close of T (D: iterate :382-432, get_entry_candidate_list :346-380; the engine's iterate runs before
process_orders)
    held           ETFs with shares > 0 at the close of T
    exits          held ETFs with exit_T -> target 0 shares (all of them, in basket order)
    slots          5 - |held| + |exits|
    candidates     ETFs with zero shares at the close of T (an ETF exiting today cannot re-enter today), entry_T and a
                   finite NATR_prev, sorted by NATR_prev descending (stable: ties keep the basket order); the first
                   `slots` enter
    sizing         shares = V_T x (1.5 / 11) / Close_T, fractional (order_target in SHARES); V_T = total value at the
                   close of T; held ETFs are never resized (`hold_nan_bool`: NaN target weight = untouched)
    fills          Open_(T+1) x (1 +- 2.5 bp), fee max(1, 0.005 x shares) (engine Strategy defaults: the pod passes no
                   cost arguments), dividends net of 25% withholding credited before the next open, no cash check
Calendar (D: resolve_us_sector_etf_execution_calendar_idx :489-509; S: resolve_full_basket_calendar_idx :252-388)
    ready_T        each ETF has 22 = max(21 + 1, 14 + 2) consecutive rows with valid OHLC and High > Low ending at T
    first fill     the session after the first all-ready T on or after 1999-01-01 (2004-11-26: VOX lists 2004-09-29)
    The engine raises if any ETF's OHLC is invalid after that; Scout raises too.

*** CRITICAL*** Every feature at T uses bars up to T only (the range median and ATR are shifted); fills at Open(T+1).

Family parameters (`SectorIbsConfig`; the default is the engine's configuration and the identity gate runs on it):
    entry_ibs_max_float, downshock_atr_max_float, exit_ibs_min_float, atr_lookback_int, range_median_lookback_int,
    max_positions_int, sizing_multiplier_float / sizing_universe_count_int (the per-entry weight)
    decision_offset_int   must be 0: a daily event rule has no rebalance schedule, so there is no luck band (the family
                          runs with offset_count_int = 1)

MCPT (`mcpt_matrix`, `fast_daily_list`): columns are the TR daily returns of the traded ETFs (the SD baseline), then per
ETF the bar in logs relative to the previous close: gap ln(Open / Close_prev), ln(High / Open), ln(Low / Open),
ln(Close / Open). A date-row shuffle moves whole bars, across ETFs together. The replica rebuilds OHLC from them,
recomputes every feature, and runs the same slot rule on a gross fractional-share ledger (no costs, no dividends).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate

STRATEGY_IMPORT_STR = "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"
TRADED_TUPLE = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "VOX", "IYR")
BENCHMARK_STR = "$SPX"
HISTORY_START_STR = "1998-01-01"
BACKTEST_START_STR = "1999-01-01"
ENGINE_COST_MODEL = CostModel()  # engine Strategy defaults: 2.5 bp, $0.005 a share, $1 minimum, 25% withholding
SEAL_END_STR = "2022-12-30"


@dataclass(frozen=True)
class SectorIbsConfig:
    entry_ibs_max_float: float = 0.05
    downshock_atr_max_float: float = -0.5
    exit_ibs_min_float: float = 0.90
    atr_lookback_int: int = 14
    range_median_lookback_int: int = 21
    max_positions_int: int = 5
    sizing_multiplier_float: float = 1.5
    sizing_universe_count_int: int = 11
    decision_offset_int: int = 0

    def __post_init__(self):
        if not 0.0 <= self.entry_ibs_max_float < self.exit_ibs_min_float <= 1.0 or self.downshock_atr_max_float >= 0.0:
            raise ValueError("SectorIbsConfig: 0 <= entry IBS < exit IBS <= 1 and a negative downshock threshold.")
        if self.atr_lookback_int < 2 or self.range_median_lookback_int < 2 or not 1 <= self.max_positions_int <= len(TRADED_TUPLE):
            raise ValueError("SectorIbsConfig: lookbacks >= 2 and 1 <= max_positions_int <= the basket size.")
        if self.decision_offset_int != 0:
            raise ValueError("SectorIbsConfig: a daily event rule has no rebalance offset (decision_offset_int = 0).")

    @property
    def entry_weight_float(self) -> float:
        return float(self.sizing_multiplier_float) / float(self.sizing_universe_count_int)  # D: __init__ :331-334

    @property
    def required_history_int(self) -> int:
        return max(self.range_median_lookback_int + 1, self.atr_lookback_int + 2)  # D: :480-486


LIVE_CONFIG = SectorIbsConfig()  # the PM_READY engine configuration


@dataclass(frozen=True)
class SectorEtfInputs:
    """CAPITALSPECIAL bars of a fixed ETF basket on the engine's session index (the union of every loaded series)."""

    open_df: pd.DataFrame
    high_df: pd.DataFrame
    low_df: pd.DataFrame
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    total_return_close_df: pd.DataFrame  # TOTALRETURN closes (MCPT baseline, S3 labels are CAPITALSPECIAL)
    backtest_start_str: str = BACKTEST_START_STR


def load_etf_inputs(symbol_tuple: tuple, history_start_str: str, backtest_start_str: str, end_date_str: str | None = None) -> SectorEtfInputs:
    from data.norgate_loader import load_price_timeseries, load_raw_prices

    pricing_df = load_raw_prices(symbols=list(symbol_tuple), benchmarks=[BENCHMARK_STR], start_date=history_start_str, end_date=end_date_str)
    pricing_df = pricing_df.sort_index()
    session_index = pd.DatetimeIndex(pricing_df.index)

    def field_df(field_str: str) -> pd.DataFrame:
        return pd.DataFrame({s: pd.to_numeric(pricing_df[(s, field_str)], errors="coerce") for s in symbol_tuple}, index=session_index, dtype=float)

    total_return_df = pd.DataFrame({
        s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=history_start_str, end_date_str=end_date_str)["Close"]
        for s in symbol_tuple
    }).reindex(session_index)
    return SectorEtfInputs(open_df=field_df("Open"), high_df=field_df("High"), low_df=field_df("Low"), close_df=field_df("Close"),
                           dividend_df=field_df("Dividend").fillna(0.0), total_return_close_df=total_return_df,
                           backtest_start_str=backtest_start_str)


def load_inputs(end_date_str: str | None = None) -> SectorEtfInputs:
    return load_etf_inputs(TRADED_TUPLE, HISTORY_START_STR, BACKTEST_START_STR, end_date_str)


# ---------------------------------------------------------------- features
@njit(cache=False)
def wilder_atr_vec(high_vec, low_vec, close_vec, period_int):
    """TA-Lib ATR on contiguous valid rows: TR from the second row, first ATR = mean of TR[1..p], then Wilder."""
    row_count_int = high_vec.size
    atr_vec = np.full(row_count_int, np.nan)
    if row_count_int <= period_int:
        return atr_vec
    sum_float = 0.0
    for i_int in range(1, period_int + 1):
        sum_float += max(high_vec[i_int] - low_vec[i_int], abs(high_vec[i_int] - close_vec[i_int - 1]), abs(low_vec[i_int] - close_vec[i_int - 1]))
    atr_float = sum_float / period_int
    atr_vec[period_int] = atr_float
    for i_int in range(period_int + 1, row_count_int):
        tr_float = max(high_vec[i_int] - low_vec[i_int], abs(high_vec[i_int] - close_vec[i_int - 1]), abs(low_vec[i_int] - close_vec[i_int - 1]))
        atr_float = (atr_float * (period_int - 1) + tr_float) / period_int
        atr_vec[i_int] = atr_float
    return atr_vec


def ibs_df(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    return (close_df - low_df) / (high_df - low_df).replace(0.0, np.nan)


def log_range_df(high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = high_df.gt(0.0) & low_df.gt(0.0) & high_df.gt(low_df)
    return np.log(high_df / low_df).where(valid_df)


def downshock_feature_dict(high_df: pd.DataFrame, low_df: pd.DataFrame, close_df: pd.DataFrame, atr_lookback_int: int,
                           range_median_lookback_int: int) -> dict:
    """IBS, RangeRatio, DownShock and prior NATR (D: :162-288). *** CRITICAL*** the range median and ATR are shifted."""
    range_df = log_range_df(high_df, low_df)
    median_df = range_df.rolling(range_median_lookback_int, min_periods=range_median_lookback_int).median().shift(1)
    downshock_dict, prior_natr_dict = {}, {}
    for symbol_str in close_df.columns:
        frame = pd.DataFrame({"High": high_df[symbol_str], "Low": low_df[symbol_str], "Close": close_df[symbol_str]}).dropna()
        atr_ser = pd.Series(wilder_atr_vec(frame["High"].to_numpy(float), frame["Low"].to_numpy(float), frame["Close"].to_numpy(float),
                                           atr_lookback_int), index=frame.index)
        close_ser = frame["Close"]
        prior_close_ser = close_ser.shift(1)
        natr_ser = 100.0 * atr_ser / close_ser.replace(0.0, np.nan)
        prior_fraction_ser = atr_ser.shift(1) / prior_close_ser.replace(0.0, np.nan)
        downshock_dict[symbol_str] = ((close_ser / prior_close_ser - 1.0) / prior_fraction_ser.replace(0.0, np.nan)).reindex(close_df.index)
        prior_natr_dict[symbol_str] = natr_ser.shift(1).reindex(close_df.index)
    return {
        "ibs_df": ibs_df(close_df, high_df, low_df),
        "range_ratio_df": range_df / median_df.replace(0.0, np.nan),
        "downshock_df": pd.DataFrame(downshock_dict, index=close_df.index),
        "prior_natr_df": pd.DataFrame(prior_natr_dict, index=close_df.index),
    }


def signal_mats(feature_dict: dict, config: SectorIbsConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(entry, exit, rank) matrices on the feature index; NaN comparisons are False."""
    ibs_mat = feature_dict["ibs_df"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        entry_mat = (ibs_mat < config.entry_ibs_max_float) & (feature_dict["downshock_df"].to_numpy(float) < config.downshock_atr_max_float)
        exit_mat = (ibs_mat > config.exit_ibs_min_float) & (feature_dict["range_ratio_df"].to_numpy(float) > 1.0)
    return entry_mat, exit_mat, feature_dict["prior_natr_df"].to_numpy(float)


# ---------------------------------------------------------------- calendar (shared with the dispersion family)
def ready_mask_df(inputs: SectorEtfInputs, required_history_int: int, required_close_history_int: int | None = None) -> pd.DataFrame:
    """Per ETF: `required_history_int` consecutive valid-OHLC rows with High > Low ending at T (and, for the SMA
    variants, that many consecutive valid closes) (S: :273-338)."""
    h_df, l_df, c_df = inputs.high_df, inputs.low_df, inputs.close_df
    tradable_df = tradable_mask_df(inputs)
    ready_df = (tradable_df & h_df.gt(l_df)).astype(int).rolling(required_history_int, min_periods=required_history_int).sum().eq(required_history_int)
    if required_close_history_int is not None:
        close_ok_df = (np.isfinite(c_df) & c_df.gt(0.0)).astype(int)
        ready_df &= close_ok_df.rolling(required_close_history_int, min_periods=required_close_history_int).sum().eq(required_close_history_int)
    return ready_df


def tradable_mask_df(inputs: SectorEtfInputs) -> pd.DataFrame:
    o_df, h_df, l_df, c_df = inputs.open_df, inputs.high_df, inputs.low_df, inputs.close_df
    return (np.isfinite(o_df) & np.isfinite(h_df) & np.isfinite(l_df) & np.isfinite(c_df) & o_df.gt(0.0) & h_df.gt(0.0) & l_df.gt(0.0)
            & c_df.gt(0.0) & h_df.ge(o_df) & h_df.ge(c_df) & h_df.ge(l_df) & l_df.le(o_df) & l_df.le(c_df))


def execution_start_ts(inputs: SectorEtfInputs, required_history_int: int, required_close_history_int: int | None = None,
                       skip_ready_bar_bool: bool = True) -> pd.Timestamp:
    """The engine's first calendar session: the first all-ready T on or after the configured start, or (downshock,
    `skip_ready_bar_bool`) the session after it, since readiness is known only at that close (D: :506-509)."""
    session_index = inputs.close_df.index
    ready_vec = ready_mask_df(inputs, required_history_int, required_close_history_int).all(axis=1).to_numpy()
    eligible_vec = ready_vec & (session_index >= pd.Timestamp(inputs.backtest_start_str))
    if not eligible_vec.any():
        raise ValueError("No full-basket-ready session on or after the configured start.")
    start_idx_int = int(np.flatnonzero(eligible_vec)[0]) + (1 if skip_ready_bar_bool else 0)
    if start_idx_int >= len(session_index):
        raise ValueError("No execution session after the first ready close.")
    if not tradable_mask_df(inputs).iloc[int(np.flatnonzero(eligible_vec)[0]):].to_numpy().all():
        raise ValueError("A basket ETF has invalid OHLC after the start (the engine raises too).")
    return pd.Timestamp(session_index[start_idx_int])


# ---------------------------------------------------------------- the event rule as a weights-engine hook
def event_decision_fn(entry_mat: np.ndarray, exit_mat: np.ndarray, rank_mat: np.ndarray | None, max_positions_int: int,
                      entry_weight_float: float):
    """Exits before entries, entries only from flat ETFs, at most `max_positions_int` held, ranked by `rank_mat`
    descending (None = basket order). Returns NaN (untouched) for every ETF that is neither exited nor entered."""
    asset_count_int = entry_mat.shape[1]

    def decide(t_idx_int: int, position_vec: np.ndarray, total_float: float):
        p_int = t_idx_int - 1  # T
        held_vec = position_vec > 0.0
        exit_vec = held_vec & exit_mat[p_int]
        candidate_mask = (position_vec == 0.0) & entry_mat[p_int]
        if rank_mat is not None:
            candidate_mask &= np.isfinite(rank_mat[p_int])
        candidate_vec = np.flatnonzero(candidate_mask)
        if rank_mat is not None and candidate_vec.size > 1:
            candidate_vec = candidate_vec[np.argsort(-rank_mat[p_int, candidate_vec], kind="stable")]
        slot_int = max_positions_int - int(held_vec.sum()) + int(exit_vec.sum())
        entry_vec = candidate_vec[:max(slot_int, 0)]
        if not exit_vec.any() and entry_vec.size == 0:
            return None
        weight_vec = np.full(asset_count_int, np.nan)
        weight_vec[exit_vec] = 0.0
        weight_vec[entry_vec] = entry_weight_float
        return weight_vec

    return decide


def simulate_event_rule(inputs: SectorEtfInputs, entry_mat, exit_mat, rank_mat, max_positions_int: int, entry_weight_float: float,
                        start_ts: pd.Timestamp, cost_model: CostModel, capital_float: float) -> WeightsResult:
    empty_df = pd.DataFrame(columns=list(inputs.close_df.columns), dtype=float)
    return simulate(
        inputs.open_df, inputs.close_df, inputs.dividend_df, empty_df, start_date=start_ts, capital_float=capital_float,
        share_unit_mode_str="adjusted", cost_model=cost_model, fractional_shares_bool=True, hold_nan_bool=True,
        decision_fn=event_decision_fn(entry_mat, exit_mat, rank_mat, max_positions_int, entry_weight_float),
    )


def feature_dict(inputs: SectorEtfInputs, config: SectorIbsConfig = LIVE_CONFIG) -> dict:
    return downshock_feature_dict(inputs.high_df, inputs.low_df, inputs.close_df, config.atr_lookback_int, config.range_median_lookback_int)


def simulate_config(inputs: SectorEtfInputs, config: SectorIbsConfig = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    entry_mat, exit_mat, rank_mat = signal_mats(feature_dict(inputs, config), config)
    start_ts = execution_start_ts(inputs, config.required_history_int, skip_ready_bar_bool=True)
    return simulate_event_rule(inputs, entry_mat, exit_mat, rank_mat, config.max_positions_int, config.entry_weight_float, start_ts,
                               cost_model, capital_float)


# ---------------------------------------------------------------- MCPT replica (S5)
def bar_matrix(inputs: SectorEtfInputs, end_date_str: str = SEAL_END_STR, first_ts: pd.Timestamp | None = None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix): TR daily returns of the ETFs, then four blocks of log bars (gap, high, low, close) per ETF.
    Rows start the session after the last first bar, so nothing is zero-filled at a listing; an invalid bar after that
    is a flat bar (0)."""
    symbol_list = list(inputs.close_df.columns)
    session_index = inputs.close_df.index
    if first_ts is None:
        first_ts = max(inputs.close_df[s].first_valid_index() for s in symbol_list) + pd.Timedelta(days=1)
    date_index = session_index[(session_index >= first_ts) & (session_index <= pd.Timestamp(end_date_str))]
    total_return_df = inputs.total_return_close_df[symbol_list].ffill().pct_change(fill_method=None).reindex(date_index).fillna(0.0)
    previous_close_df = inputs.close_df.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        block_list = [
            np.log(inputs.open_df / previous_close_df), np.log(inputs.high_df / inputs.open_df),
            np.log(inputs.low_df / inputs.open_df), np.log(inputs.close_df / inputs.open_df),
        ]
    valid_df = tradable_mask_df(inputs) & tradable_mask_df(inputs).shift(1, fill_value=False)
    block_mat_list = [frame.where(valid_df).reindex(date_index).fillna(0.0).to_numpy(float) for frame in block_list]
    return date_index, np.column_stack([total_return_df.to_numpy(float), *block_mat_list])


def mcpt_matrix(inputs: SectorEtfInputs, end_date_str: str = SEAL_END_STR) -> tuple[pd.DatetimeIndex, np.ndarray]:
    return bar_matrix(inputs, end_date_str)


def rebuilt_ohlc_dict(matrix: np.ndarray, asset_count_int: int, date_index: pd.DatetimeIndex, column_list: list) -> dict:
    """Prices from the (possibly shuffled) log bars: Close_t = Close_(t-1) x exp(gap + close), Open = Close_prev x exp(gap)."""
    gap_mat, high_mat, low_mat, close_mat = (matrix[:, asset_count_int * (k + 1): asset_count_int * (k + 2)] for k in range(4))
    log_close_mat = np.cumsum(gap_mat + close_mat, axis=0)
    log_open_mat = log_close_mat - close_mat
    frame = lambda mat: pd.DataFrame(np.exp(mat), index=date_index, columns=column_list)
    return {"Open": frame(log_open_mat), "High": frame(log_open_mat + high_mat), "Low": frame(log_open_mat + low_mat), "Close": frame(log_close_mat)}


@njit(cache=False)
def _event_book_daily(open_mat, close_mat, entry_mat, exit_mat, rank_mat, use_rank_bool, max_positions_int, entry_weight_float):
    """Gross fractional-share ledger of the event rule (the engine's arithmetic without costs or dividends)."""
    row_count_int, asset_count_int = close_mat.shape
    share_vec = np.zeros(asset_count_int)
    exiting_vec = np.zeros(asset_count_int, dtype=np.bool_)
    candidate_vec = np.zeros(asset_count_int, dtype=np.int64)
    score_vec = np.zeros(asset_count_int)
    cash_float, previous_total_float = 1.0, 1.0
    daily_vec = np.zeros(row_count_int)
    for t_int in range(1, row_count_int):
        p_int = t_int - 1
        held_int, exit_int, candidate_int = 0, 0, 0
        for a_int in range(asset_count_int):
            exiting_vec[a_int] = False
            if share_vec[a_int] > 0.0:
                held_int += 1
                if exit_mat[p_int, a_int]:
                    cash_float += share_vec[a_int] * open_mat[t_int, a_int]
                    exiting_vec[a_int] = True  # still held at the close of T: cannot re-enter today
                    exit_int += 1
        for a_int in range(asset_count_int):
            if share_vec[a_int] == 0.0 and entry_mat[p_int, a_int] and (not use_rank_bool or np.isfinite(rank_mat[p_int, a_int])):
                candidate_vec[candidate_int] = a_int
                score_vec[candidate_int] = -rank_mat[p_int, a_int] if use_rank_bool else 0.0
                candidate_int += 1
        for a_int in range(asset_count_int):
            if exiting_vec[a_int]:
                share_vec[a_int] = 0.0
        slot_int = max_positions_int - held_int + exit_int
        if candidate_int > 0 and slot_int > 0:
            order_vec = np.argsort(score_vec[:candidate_int], kind="mergesort")  # stable: ties keep the basket order
            for k_int in range(min(slot_int, candidate_int)):
                a_int = candidate_vec[order_vec[k_int]]
                share_vec[a_int] = previous_total_float * entry_weight_float / close_mat[p_int, a_int]
                cash_float -= share_vec[a_int] * open_mat[t_int, a_int]
        total_float = cash_float
        for a_int in range(asset_count_int):
            total_float += share_vec[a_int] * close_mat[t_int, a_int]
        daily_vec[t_int] = total_float / previous_total_float - 1.0
        previous_total_float = total_float
    return daily_vec


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list[dict]) -> list[np.ndarray]:
    """Gross daily returns per configuration from the matrix alone: OHLC rebuilt from the log bars, every feature
    recomputed (the replica's ATR starts at the matrix's first row), the slot rule on a fractional-share ledger. A
    date-row shuffle of `matrix` is a valid null: whole bars move with their dates, the cross-section stays together."""
    asset_count_int = len(TRADED_TUPLE)
    ohlc_dict = rebuilt_ohlc_dict(matrix, asset_count_int, date_index, list(TRADED_TUPLE))
    open_mat, close_mat = ohlc_dict["Open"].to_numpy(), ohlc_dict["Close"].to_numpy()
    feature_cache_dict, daily_list = {}, []
    for config_dict in config_list:
        config = SectorIbsConfig(**config_dict)
        key_tuple = (config.atr_lookback_int, config.range_median_lookback_int)
        if key_tuple not in feature_cache_dict:
            feature_cache_dict[key_tuple] = downshock_feature_dict(ohlc_dict["High"], ohlc_dict["Low"], ohlc_dict["Close"], *key_tuple)
        entry_mat, exit_mat, rank_mat = signal_mats(feature_cache_dict[key_tuple], config)
        daily_list.append(_event_book_daily(open_mat, close_mat, entry_mat, exit_mat, rank_mat, True,
                                            config.max_positions_int, config.entry_weight_float))
    return daily_list


# ---------------------------------------------------------------- S3 (class E: the entry event against the basket)
def s3_panel(inputs: SectorEtfInputs, end_date_str: str = SEAL_END_STR):
    """The ETF basket as an S3 panel: CAPITALSPECIAL bars, member = the ETF has a valid bar, cut at the seal."""
    from alpha.scout.panel import Panel

    cut = lambda frame: frame.loc[:end_date_str]
    member_df = cut(tradable_mask_df(inputs)).astype(int)
    field_dict = {"Open": cut(inputs.open_df), "High": cut(inputs.high_df), "Low": cut(inputs.low_df), "Close": cut(inputs.close_df)}
    return Panel(name_str="sector ETF basket " + "-".join(inputs.close_df.columns), field_dict=field_dict, member_df=member_df,
                 snapshot_id_str="etf_basket_" + str(pd.Timestamp(member_df.index[-1]).date()), sealed_bool=True)


def median_holding_sessions_int(result: WeightsResult) -> int:
    """The median entry-to-exit holding period (sessions) of a run, for the S3 horizon."""
    trade_df = result.trade_df
    session_index = result.total_value_ser.index
    hold_list = []
    for asset_str, frame in trade_df.groupby("asset"):
        entry_list = frame.loc[frame["delta_float"] > 0, "date"].tolist()
        exit_list = frame.loc[frame["delta_float"] < 0, "date"].tolist()
        hold_list.extend(session_index.get_loc(exit_ts) - session_index.get_loc(entry_ts) for entry_ts, exit_ts in zip(entry_list, exit_list))
    return int(np.median(hold_list))


S3_HORIZON_TUPLE = (1, 2, 3, 5, 10, 20)


def s3_inputs(inputs: SectorEtfInputs | None = None, end_date_str: str = SEAL_END_STR) -> dict:
    """Inputs for alpha.scout.stations.s3_edge.run_s3: event = the raw entry signal at T (before the slot cap), regime
    = every ETF with a valid bar that day (excess is measured against the same-date basket mean, so a market bounce
    does not count), indicator = -IBS (higher = more oversold), horizon = the S3 horizon nearest the live run's median
    holding period (sessions, entry open to exit open)."""
    inputs = inputs or load_inputs()
    panel = s3_panel(inputs, end_date_str)
    features = feature_dict(inputs)
    entry_mat, _, _ = signal_mats(features, LIVE_CONFIG)
    event_df = pd.DataFrame(entry_mat, index=inputs.close_df.index, columns=inputs.close_df.columns).loc[:end_date_str]
    hold_int = median_holding_sessions_int(simulate_config(inputs))
    horizon_int = min(S3_HORIZON_TUPLE, key=lambda h: (abs(h - hold_int), h))
    return {"name_str": "sector ETF IBS downshock (VOX/IYR)", "panel": panel, "regime_mask_df": panel.member_df == 1,
            "event_mask_df": event_df, "horizon_int": horizon_int, "indicator_df": -features["ibs_df"].loc[:end_date_str],
            "expected_sign_int": 1}


def s3_result(input_dict: dict) -> dict:
    """Run S3 (class E) on `s3_inputs` and shape it for `PodPlan.s3_fn` and the card: check rows (name, PASS / WARN for a
    soft miss / FAIL for a hard miss, detail), plus the EdgeReport itself."""
    from alpha.scout.stations.s3_edge import run_s3

    report = run_s3(**input_dict)
    headline = report.headline_dict
    detail_str = (f"h {report.horizon_int}: {headline['events_int']} events on {headline['event_dates_int']} dates, date-mean excess "
                  f"{headline['date_mean_excess_float'] * 1e4:+.1f} bp, NW t {headline['nw_t_float']:.2f}, cost coverage "
                  f"{headline['cost_coverage_float']:.2f}, placebo p {headline['placebo_p_float']:.3f}")
    check_list = [(name_str, "PASS" if ok_bool else ("FAIL" if level_str == "hard" else "WARN"), detail_str)
                  for name_str, ok_bool, level_str in report.check_list]
    return {"check_list": check_list, "verdict_str": report.verdict_str, "edge_report": report}
