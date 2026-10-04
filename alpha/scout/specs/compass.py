"""Scout spec of the Inflation Compass family (macro regime allocation); PM_READY, research only.

Default = `strategies.taa_df.strategy_taa_inflation_compass` (XLK in the growth-up / inflation-off cell); the variant
switch `goldilocks_str="QQQ"` is `strategy_taa_inflation_compass_qqq` (`VARIANT_DICT`). An independent
re-implementation of the signal; execution is the shared weights engine. Semantics mapped on 2026-10-01 against the
engine after the T5YIE publication-lag fix (b7a0018, 99ff200, cb29d4f); line numbers are strategy_taa_inflation_compass.py.

Data (get_inflation_compass_data :463-493; strategy_taa_df.py load_signal_close_df :193, load_execution_price_df :225)
    signal closes   TOTALRETURN Close of SPY XLE XLI XLF XLB XLU XLV XLP from 2002-01-01; the session index is the
                    union of their dates (the signal index), sorted
    execution       CAPITALSPECIAL Open / Close / Dividend of XLE, XLK (or QQQ), XLU, XLP, IEF; the execution index is
                    the union of those dates and the $SPXTR benchmark's (the engine's calendar)
    T5YIE           FRED daily 5-year breakeven, percent (the engine downloads it on every run and writes the cache)

Macro series and publication lags (align_fred_to_session_ser :204-254, compute_month_end_signal_and_weight_df :327-341)
    published_T     last T5YIE observation dated STRICTLY before session T, within 7 calendar days (else NaN):
                    FRED publishes the value dated T on T+1, so at the T decision the newest is normally T-1's
    dated_T         last observation dated ON OR BEFORE T (7-day tolerance); read only through prior_T below
    prior_T         dated_(T - L) with L = breakeven_lookback_int sessions of the signal index (shift(+L), :377);
                    the value dated T-L was published at T-L+1, before the T close

Regime logic (:343-380)
    growth_on_T     SPY_T > SMA_200(SPY)_T (rolling mean, min_periods = window; NaN SMA -> off)
    basket returns  positive = 0.50 XLE + XLI/6 + XLF/6 + XLB/6, negative = (XLU + XLV + XLP)/3 of the daily TR returns
                    (pct_change without fill; NaN if any member is NaN, min_count); wealth = cumprod(1 + r) (NaN
                    rows skipped); asset_ratio = positive wealth / negative wealth
    asset slope     OLS slope of the trailing asset_slope_lookback_int ratios (x = 0..L-1; NaN if any value in the
                    window is not finite) (compute_rolling_ols_slope_ser :257-278)
    inflation_on_T  published_T > 2.0 AND (published_T > prior_T OR slope_T > 0), all strict (NaN -> False)
    regime map      growth on + inflation on -> 100% XLE; growth on + off -> 100% XLK (QQQ); growth off + on ->
                    100% XLU; growth off + off -> 50% XLP + 50% IEF (_regime_target_weight_ser :290-311); no ranking,
                    so no ties (an exact equality, e.g. T5YIE 2.29 vs 2.29 on 2023-04-28, is "not greater")

Decision and execution dates (:401-437; strategy_taa_df.py map_month_end_weights_to_rebalance_open_df :337-363)
    decision        the last session of each month in the signal index, after the close; a month-end with any of
                    SMA, published T5YIE, prior T5YIE, ratio or slope missing is dropped before the first complete
                    one and is an ERROR after it (the engine refuses to hold the prior sleeve silently)
    execution       the first session of the next month in the execution index, at the Open (a decision without a
                    next-month session is not executed); sizing from the close of the previous session

*** CRITICAL*** The decision uses closes up to T and T5YIE published by the T evening; fills at the next Open.

Weights and units: full 100% targets in the order XLE, goldilocks, XLU, XLP, IEF; split-adjusted (CAPITALSPECIAL)
share units, truncated whole shares, sized from V(T) and Close(T) (DefenseFirstStrategy.iterate, strategy_taa_df.py
:465-506). Costs (:116-121): 5 bp slippage per side, no per-share commission, no minimum (`ENGINE_COST_MODEL`).

T5YIE is read from the MAIN checkout's cache (workspace/1_data/T5YIE.csv, where an engine run writes it) without
refreshing it. The spec refuses a cache whose last observation is older than the session before the last executed
decision (the newest value that decision may read).

Family parameters (`CompassConfig`; the default is the engine's configuration):
    growth_sma_int             SPY trend window (sessions)
    breakeven_lookback_int     sessions between published_T and the prior T5YIE anchor
    asset_slope_lookback_int   sessions in the basket-ratio slope
    inflation_threshold_float  T5YIE level threshold (percent)
    decision_offset_int        luck band: decide k sessions before the month's last session, execute on the next
                               session (0 = the engine's month-end rule)
    t5yie_extra_lag_int        truth-mode probe only (never a grid axis): read T5YIE k sessions later than the engine
Structural: goldilocks_str ("XLK" or "QQQ"), start_date_str.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel

STRATEGY_IMPORT_STR = "strategies.taa_df.strategy_taa_inflation_compass"
SIGNAL_TUPLE = ("SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP")
POSITIVE_WEIGHT_DICT = {"XLE": 0.50, "XLI": 1.0 / 6.0, "XLF": 1.0 / 6.0, "XLB": 1.0 / 6.0}
NEGATIVE_WEIGHT_DICT = {"XLU": 1.0 / 3.0, "XLV": 1.0 / 3.0, "XLP": 1.0 / 3.0}
BENCHMARK_DATA_STR = "$SPXTR"  # the engine's calendar includes the $SPX benchmark's TR data series
FRED_TOLERANCE_DAY_INT = 7
ENGINE_COST_MODEL = CostModel(slippage_float=0.0005, fee_per_share_float=0.0, min_fee_float=0.0)


def default_t5yie_csv_path() -> Path:
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    return MAIN_CHECKOUT_ROOT_PATH.parent / "1_data" / "T5YIE.csv"


@dataclass(frozen=True)
class CompassConfig:
    growth_sma_int: int = 200
    breakeven_lookback_int: int = 60
    asset_slope_lookback_int: int = 60
    inflation_threshold_float: float = 2.0
    decision_offset_int: int = 0
    t5yie_extra_lag_int: int = 0
    # Structural fields: which variant this is.
    goldilocks_str: str = "XLK"
    start_date_str: str = "2002-01-01"

    def __post_init__(self):
        if self.goldilocks_str not in ("XLK", "QQQ"):
            raise ValueError(f"Unknown goldilocks_str {self.goldilocks_str!r}.")
        if self.growth_sma_int < 2 or self.asset_slope_lookback_int < 2 or self.breakeven_lookback_int < 1:
            raise ValueError("Lookbacks must be at least 2 sessions (1 for the breakeven anchor).")

    @property
    def traded_tuple(self) -> tuple:
        return ("XLE", self.goldilocks_str, "XLU", "XLP", "IEF")


LIVE_CONFIG = CompassConfig()  # the engine's configuration (literal XLK rule)


@dataclass(frozen=True)
class CompassVariant:
    config: CompassConfig
    strategy_import_str: str
    status_str: str  # the strategy registry status when the spec was gated


VARIANT_DICT = {
    "compass": CompassVariant(LIVE_CONFIG, STRATEGY_IMPORT_STR, "PM_READY"),
    "compass_qqq": CompassVariant(CompassConfig(goldilocks_str="QQQ"), "strategies.taa_df.strategy_taa_inflation_compass_qqq", "PM_READY"),
}


def config_from_dict(base_config: CompassConfig, config_dict: dict) -> CompassConfig:
    """A family grid row as a config: `trend_lookback_int` sets both inflation-trend windows (the source rule ties them)."""
    value_dict = dict(config_dict)
    if "trend_lookback_int" in value_dict:
        lookback_int = int(value_dict.pop("trend_lookback_int"))
        value_dict.update(breakeven_lookback_int=lookback_int, asset_slope_lookback_int=lookback_int)
    return replace(base_config, **value_dict)


@dataclass(frozen=True)
class CompassInputs:
    signal_close_df: pd.DataFrame  # TOTALRETURN closes, signal index
    open_df: pd.DataFrame  # CAPITALSPECIAL, traded symbols, execution index
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    t5yie_ser: pd.Series  # percent, observation-dated
    cache_dict: dict = field(default_factory=dict, repr=False, compare=False)


def read_t5yie_csv(csv_path: Path) -> pd.Series:
    """The FRED cache as the engine parses it (date column, numeric values, non-numeric rows dropped)."""
    frame = pd.read_csv(csv_path)
    value_ser = pd.Series(pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(frame.iloc[:, 0]))
    return value_ser.dropna().sort_index().rename("T5YIE")


def load_inputs(t5yie_csv_path: Path | None = None, config: CompassConfig = LIVE_CONFIG) -> CompassInputs:
    from data.norgate_loader import load_price_timeseries

    start_str = config.start_date_str
    signal_close_df = pd.DataFrame(
        {s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=start_str)["Close"] for s in SIGNAL_TUPLE}
    ).sort_index()
    frame_dict = {s: load_price_timeseries(s, adjustment_str="CAPITALSPECIAL", start_date_str=start_str) for s in config.traded_tuple}
    benchmark_index = load_price_timeseries(BENCHMARK_DATA_STR, adjustment_str="TOTALRETURN", start_date_str=start_str).index
    execution_index = pd.DatetimeIndex(sorted(set(benchmark_index).union(*[frame.index for frame in frame_dict.values()])))
    traded_list = list(config.traded_tuple)
    return CompassInputs(
        signal_close_df=signal_close_df,
        open_df=pd.DataFrame({s: frame_dict[s]["Open"] for s in traded_list}).reindex(execution_index),
        close_df=pd.DataFrame({s: frame_dict[s]["Close"] for s in traded_list}).reindex(execution_index),
        dividend_df=pd.DataFrame({s: frame_dict[s]["Dividend"] for s in traded_list}).reindex(execution_index).fillna(0.0),
        t5yie_ser=read_t5yie_csv(t5yie_csv_path or default_t5yie_csv_path()),
    )


# ---------------------------------------------------------------- features
def asof_value_ser(value_ser: pd.Series, session_index: pd.DatetimeIndex, include_same_date_bool: bool) -> pd.Series:
    """Backward as-of value per session: the last observation dated before T (strictly, or on T with the flag),
    NaN when that observation is more than 7 calendar days old."""
    observation_index = pd.DatetimeIndex(value_ser.index).normalize()
    # *** CRITICAL*** side "left" excludes an observation dated T (published on T+1); "right" includes it.
    position_vec = observation_index.searchsorted(session_index, side="right" if include_same_date_bool else "left") - 1
    found_vec = position_vec >= 0
    safe_vec = np.maximum(position_vec, 0)
    age_vec = (session_index - observation_index[safe_vec]).days.to_numpy()
    keep_vec = found_vec & (age_vec <= FRED_TOLERANCE_DAY_INT)
    return pd.Series(np.where(keep_vec, value_ser.to_numpy(dtype=float)[safe_vec], np.nan), index=session_index)


def rolling_slope_vec(value_vec: np.ndarray, lookback_int: int) -> np.ndarray:
    """OLS slope of the trailing `lookback_int` values (x = 0..L-1), stored at the window's last row; NaN if any
    value in the window is not finite."""
    slope_vec = np.full(len(value_vec), np.nan)
    if len(value_vec) < lookback_int:
        return slope_vec
    x_centered_vec = np.arange(lookback_int, dtype=float) - (lookback_int - 1) / 2.0
    # *** CRITICAL*** trailing windows only: window r covers rows r .. r + L - 1 and is stored at row r + L - 1.
    window_mat = np.lib.stride_tricks.sliding_window_view(np.asarray(value_vec, dtype=float), lookback_int)
    with np.errstate(invalid="ignore"):
        window_slope_vec = (window_mat - window_mat.mean(axis=1, keepdims=True)) @ x_centered_vec / float(x_centered_vec @ x_centered_vec)
    window_slope_vec[~np.isfinite(window_mat).all(axis=1)] = np.nan
    slope_vec[lookback_int - 1:] = window_slope_vec
    return slope_vec


def _basket_ratio_ser(return_df: pd.DataFrame) -> pd.Series:
    positive_ser = return_df[list(POSITIVE_WEIGHT_DICT)].mul(pd.Series(POSITIVE_WEIGHT_DICT)).sum(axis=1, min_count=len(POSITIVE_WEIGHT_DICT))
    negative_ser = return_df[list(NEGATIVE_WEIGHT_DICT)].mul(pd.Series(NEGATIVE_WEIGHT_DICT)).sum(axis=1, min_count=len(NEGATIVE_WEIGHT_DICT))
    return positive_ser.add(1.0).cumprod() / negative_ser.add(1.0).cumprod()


def regime_weight_mat(growth_on_vec: np.ndarray, inflation_on_vec: np.ndarray) -> np.ndarray:
    """Rows of (XLE, goldilocks, XLU, XLP, IEF) weights for each (growth, inflation) state."""
    weight_mat = np.zeros((len(growth_on_vec), 5))
    weight_mat[growth_on_vec & inflation_on_vec, 0] = 1.0
    weight_mat[growth_on_vec & ~inflation_on_vec, 1] = 1.0
    weight_mat[~growth_on_vec & inflation_on_vec, 2] = 1.0
    weight_mat[~growth_on_vec & ~inflation_on_vec, 3:5] = 0.5
    return weight_mat


def daily_feature_df(inputs: CompassInputs, config: CompassConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Daily features on the signal index, as known after each session's close."""
    close_df = inputs.signal_close_df[list(SIGNAL_TUPLE)].astype(float)
    session_index = pd.DatetimeIndex(close_df.index)
    published_ser = asof_value_ser(inputs.t5yie_ser, session_index, include_same_date_bool=False)
    if config.t5yie_extra_lag_int:
        published_ser = published_ser.shift(config.t5yie_extra_lag_int)  # truth-mode probe: the value known k sessions earlier
    dated_ser = asof_value_ser(inputs.t5yie_ser, session_index, include_same_date_bool=True)
    # *** CRITICAL*** the observation-dated series is read only through a positive shift (dated T-L, published T-L+1).
    prior_ser = dated_ser.shift(config.breakeven_lookback_int)
    ratio_ser = _basket_ratio_ser(close_df.pct_change(fill_method=None))
    slope_ser = pd.Series(rolling_slope_vec(ratio_ser.to_numpy(), config.asset_slope_lookback_int), index=session_index)
    # *** CRITICAL*** the SMA window ends at the T close (decision after the close, fill at the next open).
    sma_ser = close_df["SPY"].rolling(config.growth_sma_int, min_periods=config.growth_sma_int).mean()
    inflation_on_ser = published_ser.gt(config.inflation_threshold_float) & (published_ser.gt(prior_ser) | slope_ser.gt(0.0))
    return pd.DataFrame({
        "growth_sma_float": sma_ser, "growth_on_bool": close_df["SPY"].gt(sma_ser),
        "t5yie_float": published_ser, "t5yie_prior_float": prior_ser, "asset_ratio_float": ratio_ser,
        "asset_slope_float": slope_ser, "inflation_on_bool": inflation_on_ser,
    }, index=session_index)


def decision_index(session_index: pd.DatetimeIndex, decision_offset_int: int) -> pd.DatetimeIndex:
    """Per calendar month, the session `decision_offset_int` sessions before the month's last session."""
    from alpha.scout.specs.taa_3x import offset_decision_index

    return offset_decision_index(session_index, decision_offset_int)


REQUIRED_FEATURE_LIST = ["growth_sma_float", "t5yie_float", "t5yie_prior_float", "asset_ratio_float", "asset_slope_float"]


def decision_feature_df(inputs: CompassInputs, config: CompassConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Features on the decision sessions with complete inputs; an incomplete decision after the first complete one
    raises, as the engine does."""
    feature_df = daily_feature_df(inputs, config)
    sampled_df = feature_df.reindex(decision_index(feature_df.index, config.decision_offset_int))
    complete_ser = sampled_df[REQUIRED_FEATURE_LIST].notna().all(axis=1)
    if not complete_ser.any():
        raise RuntimeError("No complete Inflation Compass decision.")
    after_ser = complete_ser.loc[complete_ser.idxmax():]
    if not after_ser.all():
        raise RuntimeError(f"Incomplete Inflation Compass decision after warmup: {list(after_ser[~after_ser].index[:5].date)}.")
    return sampled_df.loc[complete_ser]


def decision_weight_df(inputs: CompassInputs, config: CompassConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights per decision session (before mapping to execution dates)."""
    sampled_df = decision_feature_df(inputs, config)
    weight_mat = regime_weight_mat(sampled_df["growth_on_bool"].to_numpy(dtype=bool), sampled_df["inflation_on_bool"].to_numpy(dtype=bool))
    return pd.DataFrame(weight_mat, index=sampled_df.index, columns=list(config.traded_tuple))


def rebalance_weight_df(inputs: CompassInputs, config: CompassConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights indexed by execution date: the first session of the next month (offset 0, the engine rule), or
    the execution session after the offset decision session."""
    execution_index = inputs.open_df.index
    first_session_ser = pd.Series(execution_index, index=execution_index.to_period("M")).groupby(level=0).min()
    row_dict, last_decision_ts = {}, None
    for decision_ts, weight_ser in decision_weight_df(inputs, config).iterrows():
        if config.decision_offset_int == 0:
            next_period = (decision_ts + pd.offsets.MonthBegin(1)).to_period("M")
            if next_period not in first_session_ser.index:
                continue
            execution_ts = first_session_ser[next_period]
        else:
            position_int = int(execution_index.searchsorted(decision_ts, side="right"))
            if position_int >= len(execution_index):
                continue
            execution_ts = execution_index[position_int]
        row_dict[execution_ts] = weight_ser
        last_decision_ts = decision_ts
    if not row_dict:
        raise RuntimeError("No Inflation Compass rebalance dates.")
    # A stale cache would silently read an older T5YIE for the last executed decision: the newest observation that
    # decision may read is dated on the session before it.
    session_index = inputs.signal_close_df.index
    needed_ts = session_index[session_index.get_loc(last_decision_ts) - 1 - config.t5yie_extra_lag_int]
    if inputs.t5yie_ser.index[-1] < needed_ts:
        raise ValueError(
            f"T5YIE cache ends {inputs.t5yie_ser.index[-1].date()}, before {needed_ts.date()} which the decision of "
            f"{last_decision_ts.date()} reads; refresh it (an engine run does) before running the spec."
        )
    return pd.DataFrame(row_dict).T.sort_index()


# ---------------------------------------------------------------- MCPT (fast replica, S5)
def mcpt_column_list(config: CompassConfig = LIVE_CONFIG) -> list[str]:
    """Matrix columns: TR daily returns of the traded ETFs, TR daily returns of the other signal ETFs, then the
    T5YIE levels as published at each close and as observation-dated (the latter read only 60 rows back)."""
    other_list = [s for s in SIGNAL_TUPLE if s not in config.traded_tuple]
    return list(config.traded_tuple) + other_list + ["T5YIE_published", "T5YIE_dated"]


def mcpt_matrix(inputs: CompassInputs, config: CompassConfig = LIVE_CONFIG, end_date_str: str | None = None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix) for a plain date-row shuffle: every row is complete (no NaN), so permuted rows never
    carry warm-up gaps. Starts once every ETF has a return and both T5YIE columns have a value."""
    from data.norgate_loader import load_price_timeseries

    column_list = mcpt_column_list(config)
    asset_list = column_list[:-2]
    close_dict = {s: inputs.signal_close_df[s] for s in asset_list if s in inputs.signal_close_df}
    for symbol_str in asset_list:
        if symbol_str not in close_dict:  # traded but not a signal asset: its TR closes
            close_dict[symbol_str] = load_price_timeseries(symbol_str, adjustment_str="TOTALRETURN", start_date_str=config.start_date_str)["Close"]
    session_index = pd.DatetimeIndex(inputs.signal_close_df.index)
    return_df = pd.DataFrame(close_dict)[asset_list].reindex(session_index).pct_change(fill_method=None)
    level_df = pd.DataFrame({
        "T5YIE_published": asof_value_ser(inputs.t5yie_ser, session_index, include_same_date_bool=False),
        "T5YIE_dated": asof_value_ser(inputs.t5yie_ser, session_index, include_same_date_bool=True),
    })
    frame = pd.concat([return_df, level_df], axis=1)
    first_ts = frame.notna().all(axis=1).idxmax()
    frame = frame.loc[first_ts:end_date_str]
    if frame.isna().any().any():
        raise ValueError("The Compass MCPT matrix has a gap after its start.")
    return pd.DatetimeIndex(frame.index), frame.to_numpy(dtype=float)


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list[dict], base_config: CompassConfig = LIVE_CONFIG) -> list[np.ndarray]:
    """Gross daily returns of each configuration from the matrix alone: the regime decided at the close of T, its
    constant weights held from the close of T+1 (`searches._hold_daily`); cash (zero) before the warm-up ends."""
    from alpha.scout.searches import _hold_daily

    column_list = mcpt_column_list(base_config)
    column_dict = {name_str: i for i, name_str in enumerate(column_list)}
    return_mat = matrix[:, :5]
    return_df = pd.DataFrame(matrix[:, : len(column_list) - 2], columns=column_list[:-2])
    published_vec, dated_vec = matrix[:, column_dict["T5YIE_published"]], matrix[:, column_dict["T5YIE_dated"]]
    ratio_vec = _basket_ratio_ser(return_df).to_numpy()
    spy_price_ser = pd.Series(np.cumprod(1.0 + matrix[:, column_dict["SPY"]]))
    row_count_int = len(date_index)
    cache_dict: dict = {}
    daily_list = []
    for config_dict in config_list:
        config = config_from_dict(base_config, config_dict)
        sma_key, slope_key = ("sma", config.growth_sma_int), ("slope", config.asset_slope_lookback_int)
        if sma_key not in cache_dict:
            cache_dict[sma_key] = spy_price_ser.rolling(config.growth_sma_int, min_periods=config.growth_sma_int).mean().to_numpy()
        if slope_key not in cache_dict:
            cache_dict[slope_key] = rolling_slope_vec(ratio_vec, config.asset_slope_lookback_int)
        sma_vec, slope_vec = cache_dict[sma_key], cache_dict[slope_key]
        prior_vec = np.full(row_count_int, np.nan)
        prior_vec[config.breakeven_lookback_int:] = dated_vec[: row_count_int - config.breakeven_lookback_int]
        level_vec = published_vec
        if config.t5yie_extra_lag_int:
            level_vec = np.concatenate([np.full(config.t5yie_extra_lag_int, np.nan), published_vec[: -config.t5yie_extra_lag_int]])
        decision_ts_index = decision_index(date_index, config.decision_offset_int)
        decision_row_vec = date_index.get_indexer(decision_ts_index)
        decision_row_vec = decision_row_vec[decision_row_vec + 1 < row_count_int]  # a decision needs a next session
        with np.errstate(invalid="ignore"):
            growth_vec = spy_price_ser.to_numpy()[decision_row_vec] > sma_vec[decision_row_vec]
            level_row_vec = level_vec[decision_row_vec]
            inflation_vec = (level_row_vec > config.inflation_threshold_float) & (
                (level_row_vec > prior_vec[decision_row_vec]) | (slope_vec[decision_row_vec] > 0.0))
        complete_vec = np.isfinite(sma_vec[decision_row_vec]) & np.isfinite(prior_vec[decision_row_vec]) & np.isfinite(slope_vec[decision_row_vec]) & np.isfinite(level_row_vec)
        weight_mat = regime_weight_mat(growth_vec, inflation_vec) * complete_vec[:, None]
        daily_list.append(_hold_daily(weight_mat, decision_row_vec, return_mat))
    return daily_list


# ---------------------------------------------------------------- S3 (class W)
SEAL_END_STR = "2022-12-30"


def s3_inputs(config: CompassConfig = LIVE_CONFIG, inputs: CompassInputs | None = None) -> dict:
    """Inputs for alpha.scout.stations.s3_allocation, to the vault seal. Compass ranks nothing; it gates by two binary
    signals, so S3 reads:
    - "regime_pick" -> predictive_tests(**): the four regime sleeves (XLE, goldilocks, XLU, 50/50 XLP+IEF); score 1 for
      the sleeve the regime picks and 0 otherwise, hurdle 0.5; next-month TR returns of each sleeve (labels only).
      The on/off spread is "picked sleeve minus the other three"; the cross-sectional slope measures the same thing.
    - "growth_gate" -> gate_split(**): next-month SPY TR return when SPY > SMA200 vs not (a risk gate).
    The inflation switch has no separate test: in growth-up months it is XLE vs the goldilocks sleeve, inside regime_pick."""
    from data.norgate_loader import load_price_timeseries

    inputs = inputs or load_inputs(config=config)
    sampled_df = decision_feature_df(inputs, config).loc[:SEAL_END_STR]
    sleeve_close_df = pd.DataFrame({s: inputs.signal_close_df[s] for s in ("SPY", "XLE", "XLU", "XLP")})
    for symbol_str in (config.goldilocks_str, "IEF"):
        sleeve_close_df[symbol_str] = load_price_timeseries(symbol_str, adjustment_str="TOTALRETURN", start_date_str=config.start_date_str)["Close"]
    decision_close_df = sleeve_close_df.reindex(decision_index(inputs.signal_close_df.index, config.decision_offset_int))
    # *** CRITICAL*** next-period return (decision to next decision): a label only.
    next_return_df = decision_close_df.pct_change(fill_method=None).shift(-1).reindex(sampled_df.index)
    sleeve_return_df = pd.DataFrame({
        "XLE": next_return_df["XLE"], config.goldilocks_str: next_return_df[config.goldilocks_str], "XLU": next_return_df["XLU"],
        "XLP+IEF": 0.5 * (next_return_df["XLP"] + next_return_df["IEF"]),
    })
    weight_mat = regime_weight_mat(sampled_df["growth_on_bool"].to_numpy(dtype=bool), sampled_df["inflation_on_bool"].to_numpy(dtype=bool))
    score_df = pd.DataFrame(np.column_stack([weight_mat[:, :3], weight_mat[:, 3] > 0]).astype(float), index=sampled_df.index, columns=sleeve_return_df.columns)
    return {
        "regime_pick": {"score_df": score_df, "next_return_df": sleeve_return_df, "hurdle_ser": pd.Series(0.5, index=score_df.index)},
        "growth_gate": {"next_return_ser": next_return_df["SPY"], "gate_on_ser": sampled_df["growth_on_bool"]},
    }
