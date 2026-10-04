"""Scout spec of CORE5 adaptive macro (`strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5`, PM_READY).

An independent re-implementation of the signal; execution is the shared weights engine with its opt-in short
support (allow_short_bool, split_sign_flip_bool, BorrowModel). Semantics mapped on 2026-10-01, with the engine lines
(strategy_taa_adaptive_macro_core5.py) they mirror:

Data (get_adaptive_macro_core5_data :395-442; data/norgate_loader.py load_raw_prices)
    history from 1990-01-01, Norgate ALLMARKETDAYS padding, data profile norgate_eod_core5.
    signal closes   TOTALRETURN Close of SPY IEF GLD DBC UUP (each from its own inception)
    execution       CAPITALSPECIAL Open / Close / Dividend of SPY IEF GLD DBC UUP BIL; share units are split-adjusted
                    (engine default; deviation `split_adjusted_share_units`)
    calendar        the union of every loaded series' dates, including the $SPX benchmark (read as $SPXTR)
    No macro or exogenous series: despite the name, CORE5 reads only the five ETFs' own total-return prices (no FRED,
    VIX or rates), so there is no publication lag to respect.

Per-asset signal at Close_T (compute_adaptive_asset_signal_df :187-334), on the asset's observed closes only:
    high_T          expanding max of the TR close (never reset at the backtest start)
    severity_T      -(close_T / high_T - 1)
    percentile_T    inclusive trailing mid-rank of severity_T in the last 126 observations:
                    (N_less + (N_equal + 1) / 2) / 126, N_equal counting T itself (:145-184)
    alpha_T         w x 2/(50+1) + (1 - w) x 2/(200+1), w = percentile_T ^ 2
    AMA_T           alpha_T x close_T + (1 - alpha_T) x AMA_(T-1); the first finite alpha starts AMA at close_T
    SMA10_T         mean of the 10 closes ending at T
    long state      SMA10_T > AMA_T (strict); DBC short state SMA10_T < AMA_T (strict); NaN until both exist
    DBC vol_T       sample std (ddof 1) of the 63 TR daily returns ending at T, x sqrt(252)

Target (build_target_weight_ser :337-392), only on a rebalance decision:
    risk sleeve i   0.20 x long_i;  BIL = 1 - sum of the risk sleeves
    DBC short       when DBC's short state is on: DBC = -min(0.10, 0.025 / vol_T); proceeds stay in cash (0%)
    rebalance at T  first decision of the run, or any of the five LONG states differs from T-1 (diff over the full
                    calendar; the DBC short state alone never triggers), or T is the last XNYS session of its month
                    (exchange_calendars, extended to the end of the terminal month: _month_end_rebalance_ser :445-463)
    between rebalances nothing trades (weights drift)
Execution (iterate :725-773; _submit_target_orders :615-695; alpha/engine/strategy.py process_orders):
    shares = trunc(V_T x w / Close_T) with V_T the total value at the close of T; fills at Open_(T+1) with 2.5 bp
    slippage, fee max(1, 0.005 x shares); an unchanged share count is no order; a DBC long<->short flip is two orders
    (close the old leg, open the new one), each with its own fee.
Borrow (apply_post_mark_accounting :534-604): after the close mark of t, a held DBC short pays
    |shares| x ceil(1.02 x Close_t) x 1% x calendar days to the next session / 360 (none on the run's last session).
Start (build_execution_calendar_idx :466-517): the first T with all five long states and DBC vol defined whose next
    session has finite positive opens for all six ETFs, and not before 2007-09-01 (UUP and BIL list in 2007).

*** CRITICAL*** Every feature at T uses closes up to T only; orders fill at the T+1 open. TOTALRETURN back-adjustment
multiplies all closes up to T by one factor, to which every feature here (ratios, rank of ratios, a comparison of two
linear averages, return volatility) is invariant: later dividends cannot change a decision.

Family parameters (`Core5Config`; the default is the engine's configuration and the identity gate runs on it):
    percentile_lookback_int, percentile_power_float, fast_lookback_int, slow_lookback_int, price_filter_lookback_int,
    commodity_vol_lookback_int, commodity_short_vol_target_float, commodity_short_cap_float
    decision_offset_int   luck band: the calendar (drift-correcting) rebalance moves to k sessions before the
                          month's last XNYS session; state-change rebalances are unaffected (0 = the engine rule)
Ablation switches (robustness diagnostics, 2026-10-02; the defaults are the engine rule and the identity gate runs on
them): adaptive_speed_bool=False fixes the speed weight w at 1 / (1 + power), the mean of U^power for a uniform
percentile U (the live rule's average speed, without the drawdown adaptation);
trend_rule_bool=False holds every sleeve long and never shorts DBC (static 20% sleeves, month-end rebalancing);
price_filter_lookback_int may be 1 (the close itself, no smoothing); commodity_short_cap_float = 0 removes the short.

MCPT (`mcpt_matrix`, `fast_daily_list`): columns are the TR daily returns of SPY IEF GLD DBC UUP BIL; there are no
exogenous columns. The replica rebuilds prices from the shuffled returns, starts when every ETF has a price (2007),
uses the last session of each month in the matrix as the month end, holds weights constant from the close of T+1
(`_hold_daily`) and is gross (no costs, no borrow).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5"
RISK_ASSET_TUPLE = ("SPY", "IEF", "GLD", "DBC", "UUP")
RESERVE_ASSET_STR = "BIL"
COMMODITY_ASSET_STR = "DBC"
TRADED_TUPLE = RISK_ASSET_TUPLE + (RESERVE_ASSET_STR,)
HISTORY_START_DATE_STR = "1990-01-01"
BACKTEST_START_DATE_STR = "2007-09-01"
CALENDAR_SYMBOL_STR = "$SPXTR"  # the engine's $SPX benchmark is read as $SPXTR; it only widens the date index
DAYS_INT = 252


@dataclass(frozen=True)
class Core5Config:
    percentile_lookback_int: int = 126
    percentile_power_float: float = 2.0
    fast_lookback_int: int = 50
    slow_lookback_int: int = 200
    price_filter_lookback_int: int = 10
    commodity_vol_lookback_int: int = 63
    commodity_short_vol_target_float: float = 0.025
    commodity_short_cap_float: float = 0.10
    decision_offset_int: int = 0
    # Structural (never grid axes).
    sleeve_weight_float: float = 0.20
    annual_borrow_rate_float: float = 0.01
    backtest_start_date_str: str = BACKTEST_START_DATE_STR
    # Ablation switches (robustness diagnostics; the defaults are the engine rule).
    adaptive_speed_bool: bool = True
    trend_rule_bool: bool = True

    def __post_init__(self):
        if self.slow_lookback_int <= self.fast_lookback_int:
            raise ValueError("slow_lookback_int must exceed fast_lookback_int.")
        if min(self.percentile_lookback_int, self.fast_lookback_int, self.commodity_vol_lookback_int) <= 1:
            raise ValueError("Every lookback must be greater than one.")
        if self.price_filter_lookback_int < 1:
            raise ValueError("price_filter_lookback_int must be at least one (one = the close itself).")
        if self.decision_offset_int < 0:
            raise ValueError("decision_offset_int must be non-negative.")


LIVE_CONFIG = Core5Config()  # the engine's PM_READY configuration


@dataclass(frozen=True)
class Core5Inputs:
    signal_close_df: pd.DataFrame  # TOTALRETURN closes of the five risk ETFs, calendar index
    reserve_total_return_close_ser: pd.Series  # TOTALRETURN close of BIL (MCPT and S3 only; not a signal input)
    open_df: pd.DataFrame  # CAPITALSPECIAL, six traded ETFs, calendar index
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    cache_dict: dict = field(default_factory=dict, repr=False, compare=False)


def load_inputs() -> Core5Inputs:
    from data.norgate_loader import load_price_timeseries
    from data.norgate_snapshot_store import CORE5_PROFILE_STR, use_norgate_data_profile

    with use_norgate_data_profile(CORE5_PROFILE_STR):
        capital_frame_dict = {
            s: load_price_timeseries(s, adjustment_str="CAPITALSPECIAL", start_date_str=HISTORY_START_DATE_STR) for s in TRADED_TUPLE
        }
        total_return_dict = {
            s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=HISTORY_START_DATE_STR)["Close"] for s in TRADED_TUPLE
        }
        calendar_index = load_price_timeseries(CALENDAR_SYMBOL_STR, adjustment_str="TOTALRETURN", start_date_str=HISTORY_START_DATE_STR).index
    date_index = pd.DatetimeIndex(sorted(set(calendar_index).union(*[frame.index for frame in capital_frame_dict.values()])))
    traded_list = list(TRADED_TUPLE)
    return Core5Inputs(
        signal_close_df=pd.DataFrame({s: total_return_dict[s] for s in RISK_ASSET_TUPLE}).reindex(date_index),
        reserve_total_return_close_ser=total_return_dict[RESERVE_ASSET_STR].reindex(date_index),
        open_df=pd.DataFrame({s: capital_frame_dict[s]["Open"] for s in traded_list}).reindex(date_index),
        close_df=pd.DataFrame({s: capital_frame_dict[s]["Close"] for s in traded_list}).reindex(date_index),
        dividend_df=pd.DataFrame({s: capital_frame_dict[s]["Dividend"] for s in traded_list}).reindex(date_index).fillna(0.0),
    )


# ---------------------------------------------------------------- per-asset signal
def _midrank_percentile_vec(severity_vec: np.ndarray, lookback_int: int) -> np.ndarray:
    """Inclusive trailing mid-rank percentile; NaN until a full window of finite values."""
    percentile_vec = np.full(severity_vec.size, np.nan)
    if severity_vec.size < lookback_int:
        return percentile_vec
    # *** CRITICAL*** trailing windows only: window r covers rows r .. r + L - 1 and is stored at row r + L - 1.
    window_mat = np.lib.stride_tricks.sliding_window_view(severity_vec, lookback_int)
    current_vec = window_mat[:, -1]
    less_vec = (window_mat < current_vec[:, None]).sum(axis=1)
    equal_vec = (window_mat == current_vec[:, None]).sum(axis=1)
    value_vec = (less_vec + (equal_vec + 1.0) / 2.0) / float(lookback_int)
    value_vec[~np.isfinite(window_mat).all(axis=1)] = np.nan
    percentile_vec[lookback_int - 1:] = value_vec
    return percentile_vec


def _ama_python(price_vec: np.ndarray, alpha_vec: np.ndarray) -> np.ndarray:
    ama_vec = np.full(price_vec.size, np.nan)
    prior_float = np.nan
    for idx_int in range(price_vec.size):
        alpha_float = alpha_vec[idx_int]
        if not np.isfinite(alpha_float):
            continue
        # *** CRITICAL*** recursive boundary: AMA_T uses close_T, alpha_T and AMA_(T-1) only.
        prior_float = price_vec[idx_int] if not np.isfinite(prior_float) else alpha_float * price_vec[idx_int] + (1.0 - alpha_float) * prior_float
        ama_vec[idx_int] = prior_float
    return ama_vec


try:
    from numba import njit

    _ama = njit(cache=False)(_ama_python)
except ImportError:  # pragma: no cover - numba is a project dependency
    _ama = _ama_python


def adaptive_moving_average_vec(price_vec: np.ndarray, config: Core5Config) -> np.ndarray:
    """AMA of one asset's observed (finite) closes."""
    high_vec = np.maximum.accumulate(price_vec)
    severity_vec = -(price_vec / high_vec - 1.0)
    weight_vec = np.power(_midrank_percentile_vec(severity_vec, config.percentile_lookback_int), config.percentile_power_float)
    if not config.adaptive_speed_bool:
        # ablation: same warm-up, a fixed speed at the live rule's average weight E[U^p] = 1 / (1 + p)
        weight_vec = np.where(np.isfinite(weight_vec), 1.0 / (1.0 + config.percentile_power_float), np.nan)
    fast_float, slow_float = 2.0 / float(config.fast_lookback_int + 1), 2.0 / float(config.slow_lookback_int + 1)
    alpha_vec = weight_vec * fast_float + (1.0 - weight_vec) * slow_float
    return _ama(np.ascontiguousarray(price_vec, dtype=float), np.ascontiguousarray(alpha_vec, dtype=float))


def asset_signal_df(close_ser: pd.Series, config: Core5Config) -> pd.DataFrame:
    """long_state, short_state (1.0 / 0.0, NaN until defined) and annualised volatility, on the full calendar."""
    observed_ser = close_ser.dropna().astype(float)
    ama_vec = adaptive_moving_average_vec(observed_ser.to_numpy(), config)
    # *** CRITICAL*** rolling windows end at Close_T: SMA of the last n closes, volatility of the last 63 returns.
    sma_vec = observed_ser.rolling(config.price_filter_lookback_int, min_periods=config.price_filter_lookback_int).mean().to_numpy()
    volatility_vec = (
        observed_ser.pct_change(fill_method=None).rolling(config.commodity_vol_lookback_int, min_periods=config.commodity_vol_lookback_int).std(ddof=1)
        * np.sqrt(252.0)
    ).to_numpy()
    valid_vec = np.isfinite(sma_vec) & np.isfinite(ama_vec)
    long_vec = np.where(valid_vec, (sma_vec > ama_vec).astype(float), np.nan)
    short_vec = np.where(valid_vec, (sma_vec < ama_vec).astype(float), np.nan)
    frame = pd.DataFrame({"long": long_vec, "short": short_vec, "volatility": volatility_vec}, index=observed_ser.index)
    return frame.reindex(close_ser.index)


def _signal_frame_dict(inputs: Core5Inputs, config: Core5Config) -> dict[str, pd.DataFrame]:
    key_tuple = (config.percentile_lookback_int, config.percentile_power_float, config.fast_lookback_int, config.slow_lookback_int,
                 config.price_filter_lookback_int, config.commodity_vol_lookback_int, config.adaptive_speed_bool)
    if key_tuple not in inputs.cache_dict:
        inputs.cache_dict[key_tuple] = {s: asset_signal_df(inputs.signal_close_df[s], config) for s in RISK_ASSET_TUPLE}
    return inputs.cache_dict[key_tuple]


# ---------------------------------------------------------------- calendar
def month_end_flag_vec(date_index: pd.DatetimeIndex, decision_offset_int: int = 0) -> np.ndarray:
    """True on the last XNYS session of each month (or `decision_offset_int` sessions before it), known in advance
    from the exchange calendar, extended to the end of the terminal month (never inferred from the next data row)."""
    import exchange_calendars

    first_ts = date_index.min().to_period("M").start_time.normalize()
    last_ts = date_index.max().to_period("M").end_time.normalize()
    session_index = exchange_calendars.get_calendar("XNYS", start=first_ts, end=last_ts).sessions.tz_localize(None)
    position_ser = pd.Series(np.arange(len(session_index)), index=session_index)
    month_position_ser = position_ser.groupby(session_index.to_period("M")).max() - decision_offset_int
    month_first_ser = position_ser.groupby(session_index.to_period("M")).min()
    decision_session_index = session_index[month_position_ser[month_position_ser >= month_first_ser].to_numpy()]
    return np.asarray(date_index.isin(decision_session_index))


def _cached_month_end_flag_vec(inputs: Core5Inputs, decision_offset_int: int) -> np.ndarray:
    key_tuple = ("month_end", decision_offset_int)
    if key_tuple not in inputs.cache_dict:
        inputs.cache_dict[key_tuple] = month_end_flag_vec(inputs.open_df.index, decision_offset_int)
    return inputs.cache_dict[key_tuple]


# ---------------------------------------------------------------- targets
def target_weight_vec(long_vec: np.ndarray, short_bool: bool, volatility_float: float, config: Core5Config) -> np.ndarray:
    """[SPY, IEF, GLD, DBC, UUP, BIL]: 0.2 per long sleeve, BIL the rest, plus the DBC volatility-scaled short."""
    risk_vec = long_vec * config.sleeve_weight_float
    reserve_float = float(1.0 - np.sum(risk_vec))
    if short_bool:
        if not (np.isfinite(volatility_float) and volatility_float > 0.0):
            raise ValueError("DBC short sizing needs a positive finite volatility.")
        risk_vec = risk_vec.copy()
        risk_vec[RISK_ASSET_TUPLE.index(COMMODITY_ASSET_STR)] = -min(
            config.commodity_short_cap_float, config.commodity_short_vol_target_float / volatility_float
        )
    return np.append(risk_vec, reserve_float)


def rebalance_weight_df(inputs: Core5Inputs, config: Core5Config = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights indexed by execution date (the session after each rebalance decision)."""
    date_index = inputs.open_df.index
    signal_dict = _signal_frame_dict(inputs, config)
    long_mat = np.column_stack([signal_dict[s]["long"].to_numpy() for s in RISK_ASSET_TUPLE])
    commodity_df = signal_dict[COMMODITY_ASSET_STR]
    short_vec, volatility_vec = commodity_df["short"].to_numpy(), commodity_df["volatility"].to_numpy()
    if not config.trend_rule_bool:  # ablation: static long sleeves once the signals are defined (same start), no short
        long_mat = np.where(np.isfinite(long_mat), 1.0, np.nan)
        short_vec = np.zeros_like(short_vec)

    # *** CRITICAL*** the state change at T compares Close_T with Close_(T-1) only (a NaN neighbour is no change).
    changed_vec = np.zeros(len(date_index), dtype=bool)
    changed_vec[1:] = np.nan_to_num(np.abs(np.diff(long_mat, axis=0)), nan=0.0).max(axis=1) > 0.0
    month_end_vec = _cached_month_end_flag_vec(inputs, config.decision_offset_int)

    # Start: the first decision with every signal defined whose next session has finite positive opens.
    valid_decision_vec = np.isfinite(long_mat).all(axis=1) & np.isfinite(volatility_vec)
    open_mat = inputs.open_df[list(TRADED_TUPLE)].to_numpy(dtype=float)
    valid_open_vec = (np.isfinite(open_mat) & (open_mat > 0.0)).all(axis=1)
    candidate_vec = np.flatnonzero(valid_decision_vec[:-1] & valid_open_vec[1:]) + 1
    if candidate_vec.size == 0:
        raise RuntimeError("No actionable CORE5 execution date.")
    start_ts = max(date_index[candidate_vec[0]], pd.Timestamp(config.backtest_start_date_str))
    start_position_int = int(date_index.searchsorted(start_ts))

    close_mat = inputs.close_df[list(TRADED_TUPLE)].to_numpy(dtype=float)
    row_dict = {}
    for decision_int in range(start_position_int - 1, len(date_index) - 1):
        if not (decision_int == start_position_int - 1 or changed_vec[decision_int] or month_end_vec[decision_int]):
            continue
        if not np.isfinite(long_mat[decision_int]).all() or not (np.isfinite(close_mat[decision_int]) & (close_mat[decision_int] > 0)).all():
            raise RuntimeError(f"Incomplete CORE5 decision snapshot at {date_index[decision_int].date()}.")
        row_dict[date_index[decision_int + 1]] = target_weight_vec(
            long_mat[decision_int], short_vec[decision_int] == 1.0, volatility_vec[decision_int], config
        )
    return pd.DataFrame.from_dict(row_dict, orient="index", columns=list(TRADED_TUPLE))


def simulate_config(inputs: Core5Inputs, config: Core5Config = LIVE_CONFIG, cost_model=None, capital_float: float = 100_000.0):
    """The parity weights engine run of one configuration (what the gate and the family execute)."""
    from alpha.scout.engines.weights import BorrowModel, CostModel, simulate

    weight_df = rebalance_weight_df(inputs, config)
    traded_list = list(TRADED_TUPLE)
    return simulate(
        inputs.open_df[traded_list], inputs.close_df[traded_list], inputs.dividend_df[traded_list], weight_df,
        start_date=weight_df.index[0], capital_float=capital_float, share_unit_mode_str="adjusted",
        cost_model=cost_model or CostModel(), allow_short_bool=True, split_sign_flip_bool=True,
        borrow_model=BorrowModel(annual_rate_float=config.annual_borrow_rate_float),
    )


# ---------------------------------------------------------------- MCPT (fast replica)
def mcpt_matrix(inputs: Core5Inputs, end_date_str: str | None = "2022-12-30") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix): TR daily returns of SPY IEF GLD DBC UUP BIL, from the session after every ETF has a price
    to `end_date_str` (default the vault seal; None = all data). No exogenous columns: CORE5 reads none."""
    close_df = inputs.signal_close_df.assign(**{RESERVE_ASSET_STR: inputs.reserve_total_return_close_ser})[list(TRADED_TUPLE)]
    first_ts = max(close_df[s].first_valid_index() for s in TRADED_TUPLE)
    date_index = close_df.index[close_df.index > first_ts]
    if end_date_str is not None:
        date_index = date_index[date_index <= pd.Timestamp(end_date_str)]
    # *** CRITICAL*** return of day t = close_t / close_(t-1) - 1 (ALLMARKETDAYS padding: a padded day returns 0).
    return_df = close_df.loc[first_ts:].ffill().pct_change(fill_method=None).reindex(date_index).fillna(0.0)
    return date_index, return_df.to_numpy(dtype=float)


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list) -> list[np.ndarray]:
    """Gross daily return of each configuration (dicts of Core5Config fields, or Core5Config) from the matrix alone:
    signals from prices rebuilt by compounding the returns, decision at the close of T on the first defined session,
    a long-state change or the matrix's last session of a month, constant weights from the close of T+1."""
    from alpha.scout.searches import _hold_daily, refuse_ablation_switches

    refuse_ablation_switches(config_list)
    risk_count_int = len(RISK_ASSET_TUPLE)
    commodity_int = RISK_ASSET_TUPLE.index(COMMODITY_ASSET_STR)
    price_mat = np.cumprod(1.0 + matrix[:, :risk_count_int], axis=0)
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    month_end_vec = np.zeros(len(date_index), dtype=bool)
    month_end_vec[position_ser.groupby(date_index.to_period("M")).max().to_numpy()] = True
    ama_cache_dict, sma_cache_dict, volatility_cache_dict = {}, {}, {}
    daily_list = []
    for config_obj in config_list:
        config = config_obj if isinstance(config_obj, Core5Config) else Core5Config(**config_obj)
        ama_key = (config.percentile_lookback_int, config.percentile_power_float, config.fast_lookback_int, config.slow_lookback_int)
        if ama_key not in ama_cache_dict:
            ama_cache_dict[ama_key] = np.column_stack([adaptive_moving_average_vec(price_mat[:, i], config) for i in range(risk_count_int)])
        if config.price_filter_lookback_int not in sma_cache_dict:
            sma_cache_dict[config.price_filter_lookback_int] = (
                pd.DataFrame(price_mat).rolling(config.price_filter_lookback_int, min_periods=config.price_filter_lookback_int).mean().to_numpy()
            )
        if config.commodity_vol_lookback_int not in volatility_cache_dict:
            volatility_cache_dict[config.commodity_vol_lookback_int] = (
                pd.Series(matrix[:, commodity_int]).rolling(config.commodity_vol_lookback_int, min_periods=config.commodity_vol_lookback_int).std(ddof=1)
                * np.sqrt(252.0)
            ).to_numpy()
        ama_mat, sma_mat = ama_cache_dict[ama_key], sma_cache_dict[config.price_filter_lookback_int]
        volatility_vec = volatility_cache_dict[config.commodity_vol_lookback_int]
        valid_vec = np.isfinite(ama_mat).all(axis=1) & np.isfinite(sma_mat).all(axis=1) & np.isfinite(volatility_vec)
        if not valid_vec.any():
            daily_list.append(np.zeros(len(date_index)))
            continue
        first_int = int(np.argmax(valid_vec))
        long_mat = (sma_mat > ama_mat).astype(float)
        short_vec = sma_mat[:, commodity_int] < ama_mat[:, commodity_int]
        changed_vec = np.zeros(len(date_index), dtype=bool)
        changed_vec[1:] = (long_mat[1:] != long_mat[:-1]).any(axis=1)
        decision_mask_vec = (changed_vec | month_end_vec) & (np.arange(len(date_index)) > first_int)
        decision_mask_vec[first_int] = True
        decision_row_vec = np.flatnonzero(decision_mask_vec)
        weight_mat = np.array([target_weight_vec(long_mat[r], bool(short_vec[r]), volatility_vec[r], config) for r in decision_row_vec])
        daily_list.append(_hold_daily(weight_mat, decision_row_vec, matrix))
    return daily_list


# ---------------------------------------------------------------- S3 (class W, time-series signal)
def s3_inputs(inputs: Core5Inputs | None = None, config: Core5Config = LIVE_CONFIG, end_date_str: str = "2022-12-30"):
    """(score_df, next_return_df, threshold_ser) for `stations.s3_allocation.predictive_tests`.

    CORE5 does not rank: each sleeve is an absolute (time-series) trend signal. The score is SMA_n / AMA - 1 at each
    month's last session (> 0 is the long state), the threshold is 0, and next_return_df is the NEXT month's TR return
    of each risk ETF in excess of BIL's (the sleeve's alternative). The on/off spread and the per-asset slopes are the
    relevant tests; the cross-sectional (Fama-MacBeth) slope is not this strategy's hypothesis. The engine's daily
    state-change rebalances are not represented (month-end sampling only)."""
    inputs = inputs or load_inputs()
    score_dict = {}
    for asset_str in RISK_ASSET_TUPLE:
        observed_ser = inputs.signal_close_df[asset_str].dropna()
        ama_vec = adaptive_moving_average_vec(observed_ser.to_numpy(), config)
        sma_ser = observed_ser.rolling(config.price_filter_lookback_int, min_periods=config.price_filter_lookback_int).mean()
        score_dict[asset_str] = sma_ser / pd.Series(ama_vec, index=observed_ser.index) - 1.0
    # *** CRITICAL*** score sampled at month end m (closes up to m); the next-month return is a label only.
    score_df = pd.DataFrame(score_dict).resample("ME").last().loc[:end_date_str]
    month_close_df = inputs.signal_close_df.resample("ME").last()
    reserve_return_ser = inputs.reserve_total_return_close_ser.resample("ME").last().pct_change(fill_method=None).shift(-1)
    next_return_df = month_close_df.pct_change(fill_method=None).shift(-1).sub(reserve_return_ser, axis=0).loc[:end_date_str]
    return score_df, next_return_df, pd.Series(0.0, index=score_df.index)
