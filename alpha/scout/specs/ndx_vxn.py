"""Scout spec of the LIVE pod NDX VXN (`strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled`).

An independent re-implementation of the signal; execution is the shared weights engine in historical-share mode.
Semantics mapped on 2026-09-30 (P3) against the engine after readiness-audit fixes fb81e86 (decision-day dollar ATR,
raw-share units) and 9afc293 (exact index membership).

Universe: Nasdaq-100 point-in-time membership (exact), symbols with any membership on or after 2000-01-01, plus SPY.
Decision date T: the last price-index session of each calendar month (the last month only if it ended on the XNYS
month-end); execution: the next session.

For each stock i at T (prices CAPITALSPECIAL unless marked raw):
    ROC12    = Close(T) / Close(T−12 month-ends) − 1
    TR(d)    = max(High − Low, |High − Close(d−1)|, |Low − Close(d−1)|)
    ATR20$   = mean(TR over the last 20 sessions) × RawClose(T) / Close(T)       (dollars of day T)
    score    = ROC12 / ATR20$                                                  (±inf -> NaN)
    eligible = member(T) and Close(T) > SMA100(T) and score finite
    regime   = SPY Close(T) > SPY SMA200(T); off -> empty target (all cash)
    top 10 by score (ties: symbol ascending), weight = 0.1 × clip(22 / VXN(T), 0.25, 1.0)

*** CRITICAL*** Every input is read at T or earlier; VXN is the last close dated <= T. Fills happen at Open(T+1).

Family parameters (P5, `NdxConfig`; the default is the LIVE pod and the identity gate runs on it): the ROC length in
month-ends, the number of stocks held, the stock trend average, the ATR window, the SPY regime average, the VXN
reference level, and decision_offset_int (luck band: decide k sessions before the month's last session).

Engine siblings (mapped 2026-10-01; `NDX_VARIANT_DICT` holds one gated config per engine strategy). They share the
NDX VXN data loader, decision calendar, membership, filters, ranking (ties: symbol ascending), historical-share sizing
and execution line for line; only two switches move:

- `vxn_scaled_bool=False` -> weight = 1 / top_count, no VXN read.
    strategy_mo_atr_normalized_ndx.AtrNormalizedNdxStrategy.get_target_weight_ser: `1.0 / float(max_positions_int)`.
    The VXN pod subclasses it and only multiplies that series by the as-of scale, so 0.1 × 1.0 is bit-identical.
- `atr_unit_str="percent"` (NATR20) -> score = (ROC12 / ATR20$) × RawClose(T) = ROC12 / (ATR20 / Close), ±inf -> NaN.
    strategy_mo_natr20_ndx.Natr20NdxStrategy.compute_signals: `risk_adj_score_ser *= raw_close_ser`, applied to the
    base score `(roc / atr_dollar).replace(±inf, NaN)`;
    strategy_mo_natr20_ndx_vxn_scaled.compute_natr20_signal_tables: `((roc / atr_dollar) * raw_close).replace(±inf,
    NaN)`. Both orders give the same floats: RawClose(T) is checked finite and positive wherever Close(T) exists,
    and ±inf × RawClose stays ±inf. Only the ranking changes; the weights stay 1 / top_count (× the VXN scale).
  The NATR20 VXN module is a stand-alone copy (no inheritance) of the same rules; its one extra guard (VXN closes
  must be finite and positive, else raise) is the filter `load_inputs` already applies.

| Gate name | Engine strategy | vxn_scaled_bool | atr_unit_str |
|---|---|---|---|
| ndx_vxn | strategy_mo_atr_normalized_ndx_vxn_scaled (LIVE) | True | dollar |
| ndx_atr | strategy_mo_atr_normalized_ndx (WIRED) | False | dollar |
| ndx_natr20 | strategy_mo_natr20_ndx (research) | False | percent |
| ndx_natr20_vxn | strategy_mo_natr20_ndx_vxn_scaled (research) | True | percent |

Ablation switches (robustness diagnostics, 2026-10-02; no engine strategy, the defaults are the engine rule and the
identity gate runs on them): regime_filter_bool=False treats the SPY regime as always on (a decision still waits for
the regime average to be warm, so the decision months are unchanged); stock_trend_filter_bool=False drops the
Close > SMA filter; atr_unit_str="none" ranks on ROC alone (no volatility normalisation).

Trend-filter variants (research, 2026-10-02 owner question; the defaults are the engine rule Close > SMA(stock_sma_int),
evaluated by the original comparison so the identity gate is untouched):
    trend_fast_sma_int > 0      the filter compares SMA(fast) with SMA(stock_sma_int) instead of the close
    trend_threshold_float != 0  the filter requires (Close or SMA(fast)) / SMA(stock_sma_int) - 1 > threshold
    trend_filter_str = "cmma"   Masters' Close Minus Moving Average (Statistically Sound Indicators, ch. 2) with
                                L = stock_sma_int and A = cmma_atr_int, the filter requires CMMA(T) > cmma_threshold_float:
        x(T)    = [ln C(T) - mean(ln C(T-L), ..., ln C(T-1))] / [ATR_ln(A, T) * sqrt(L + 1)]
        ATR_ln  = mean over the last A sessions of max(ln H - ln L, |ln H - ln C(t-1)|, |ln L - ln C(t-1)|)
        CMMA(T) = 100 * Phi(x(T)) - 50   (bounded in (-50, 50); 0 = the close at its log average)
      Every term ends at T and is a log difference, so a back-adjustment factor (a constant per date range) cancels.

Linear-trend ranking (research, 2026-10-03; Masters, Statistically Sound Indicators, "linear trend"):
atr_unit_str = "linear_trend" ranks on
    LT(T) = b(T) * (n - 1) / ATR_ln(A, T) * R2(T)        (R2 left out when lt_rsq_bool is False)
    b, R2 = the OLS slope and R-squared of ln C on the session index over the n sessions ending at T
  (n = lt_lookback_int, A = lt_atr_int). Masters compresses with 100 * Phi(0.6 * x) - 50; a monotone map with a
  common constant cannot change a cross-sectional ranking, so the ranking uses LT itself (`linear_trend_frame` also
  returns a compressed form, scaled by 1 / sqrt(n) so that it does not saturate, for the indicator battery). roc_month_int keeps its role in the decision calendar
  (the first decision waits roc_month_int month-ends), so the start date matches the ROC variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STRATEGY_IMPORT_STR = "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled"
INDEX_NAME_STR = "Nasdaq 100"
HISTORY_START_STR = "1999-01-01"
TRADING_START_STR = "2000-01-01"
REGIME_SYMBOL_STR = "SPY"
TOP_COUNT_INT = 10
ROC_MONTH_INT = 12
ATR_WINDOW_INT = 20
STOCK_SMA_INT = 100
REGIME_SMA_INT = 200
VXN_REFERENCE_FLOAT = 22.0
VXN_FLOOR_FLOAT = 0.25
ATR_UNIT_TUPLE = ("dollar", "percent", "none", "linear_trend")  # "none": ablation only (ROC alone); "linear_trend": research


@dataclass(frozen=True)
class NdxConfig:
    roc_month_int: int = ROC_MONTH_INT
    top_count_int: int = TOP_COUNT_INT
    stock_sma_int: int = STOCK_SMA_INT
    atr_window_int: int = ATR_WINDOW_INT
    regime_sma_int: int = REGIME_SMA_INT
    vxn_reference_float: float = VXN_REFERENCE_FLOAT
    decision_offset_int: int = 0
    vxn_scaled_bool: bool = True  # False: plain 1 / top_count slots (strategy_mo_atr_normalized_ndx, NATR20 plain)
    atr_unit_str: str = "dollar"  # "percent": NATR20 ranking, score = ROC / (ATR20 / Close)
    # Ablation switches (robustness diagnostics; the defaults are the engine rule).
    regime_filter_bool: bool = True
    stock_trend_filter_bool: bool = True
    # Trend-filter variants (research; the defaults are the engine rule).
    trend_fast_sma_int: int = 0  # 0 = the close
    trend_threshold_float: float = 0.0
    trend_filter_str: str = "sma"  # "sma" (the engine rule and its variants) or "cmma"
    lt_lookback_int: int = 252  # linear-trend ranking (atr_unit_str = "linear_trend")
    lt_atr_int: int = 20
    lt_rsq_bool: bool = True
    cmma_atr_int: int = 252
    cmma_threshold_float: float = 0.0

    def __post_init__(self) -> None:
        if self.atr_unit_str not in ATR_UNIT_TUPLE:
            raise ValueError(f"atr_unit_str must be one of {ATR_UNIT_TUPLE}.")


LIVE_CONFIG = NdxConfig()


@dataclass(frozen=True)
class NdxVariant:
    strategy_module_str: str  # the engine module whose run_variant() the identity gate runs
    config: NdxConfig = field(default_factory=NdxConfig)


NDX_VARIANT_DICT = {
    "ndx_vxn": NdxVariant(STRATEGY_IMPORT_STR),
    "ndx_atr": NdxVariant("strategies.momentum.strategy_mo_atr_normalized_ndx", NdxConfig(vxn_scaled_bool=False)),
    "ndx_natr20": NdxVariant("strategies.momentum.strategy_mo_natr20_ndx", NdxConfig(vxn_scaled_bool=False, atr_unit_str="percent")),
    "ndx_natr20_vxn": NdxVariant("strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled", NdxConfig(atr_unit_str="percent")),
}


@dataclass(frozen=True)
class NdxInputs:
    open_df: pd.DataFrame
    high_df: pd.DataFrame
    low_df: pd.DataFrame
    close_df: pd.DataFrame
    raw_close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    member_df: pd.DataFrame  # 1/0 on the price index, causal forward fill of the membership rows
    vxn_close_ser: pd.Series


def load_inputs() -> NdxInputs:
    from data.norgate_loader import (
        build_index_constituent_matrix,
        load_price_timeseries,
        load_raw_prices,
    )

    _, universe_df = build_index_constituent_matrix(INDEX_NAME_STR)
    universe_df = universe_df.loc[universe_df.index >= pd.Timestamp(HISTORY_START_STR)]
    active_symbol_list = [s for s in universe_df.columns if universe_df.loc[universe_df.index >= TRADING_START_STR, s].any()]
    price_df = load_raw_prices(active_symbol_list + [REGIME_SYMBOL_STR], [], start_date=HISTORY_START_STR)
    loaded_symbol_list = sorted({symbol for symbol, _ in price_df.columns})

    def field_df(field_str: str) -> pd.DataFrame:
        return pd.DataFrame({s: price_df[(s, field_str)] for s in loaded_symbol_list}, index=price_df.index)

    # *** CRITICAL*** causal forward fill: membership on T is the last membership row dated <= T.
    member_df = (
        universe_df.reindex(columns=[s for s in loaded_symbol_list if s != REGIME_SYMBOL_STR])
        .reindex(universe_df.index.union(price_df.index)).ffill().reindex(price_df.index).fillna(0).astype(int)
    )
    vxn_close_ser = load_price_timeseries("$VXN", adjustment_str="CAPITALSPECIAL", start_date_str=HISTORY_START_STR)["Close"]
    # The engine drops non-finite or non-positive VXN closes and uses the previous valid one.
    vxn_close_ser = vxn_close_ser[np.isfinite(vxn_close_ser) & (vxn_close_ser > 0)]
    return NdxInputs(
        open_df=field_df("Open"),
        high_df=field_df("High"),
        low_df=field_df("Low"),
        close_df=field_df("Close"),
        raw_close_df=field_df("Unadjusted Close"),
        dividend_df=field_df("Dividend"),
        member_df=member_df,
        vxn_close_ser=vxn_close_ser,
    )


def decision_dates(date_index: pd.DatetimeIndex, decision_offset_int: int = 0) -> pd.DatetimeIndex:
    """The last session of each calendar month (or `decision_offset_int` sessions before it); the running month counts
    only if it ended on the XNYS month-end."""
    import exchange_calendars

    last_session_ser = pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max()
    last_available_ts = date_index[-1]
    month_period = last_available_ts.to_period("M")
    session_index = exchange_calendars.get_calendar("XNYS").sessions_in_range(
        month_period.start_time.normalize(), month_period.end_time.normalize()
    )
    expected_month_end_ts = pd.Timestamp(session_index[-1]).tz_localize(None) if session_index.tz is not None else pd.Timestamp(session_index[-1])
    if expected_month_end_ts.normalize() != last_available_ts.normalize():
        last_session_ser = last_session_ser.iloc[:-1]
    month_end_index = pd.DatetimeIndex(last_session_ser.to_numpy())
    if decision_offset_int == 0:
        return month_end_index
    position_vec = date_index.get_indexer(month_end_index) - decision_offset_int
    return date_index[position_vec[position_vec >= 0]]


def cmma_frame(inputs: NdxInputs, lookback_int: int, atr_int: int) -> pd.DataFrame:
    """Masters' CMMA on every column (see the module docstring); NaN until both windows are full."""
    from scipy.special import ndtr

    log_close_df = np.log(inputs.close_df.where(inputs.close_df > 0))
    log_high_df, log_low_df = np.log(inputs.high_df.where(inputs.high_df > 0)), np.log(inputs.low_df.where(inputs.low_df > 0))
    previous_log_close_df = log_close_df.shift(1)
    # *** CRITICAL*** the average covers the L closes BEFORE T; the ATR window ends at T.
    mean_df = previous_log_close_df.rolling(lookback_int, min_periods=lookback_int).mean()
    true_range_df = np.maximum(log_high_df - log_low_df,
                               np.maximum((log_high_df - previous_log_close_df).abs(), (log_low_df - previous_log_close_df).abs()))
    atr_df = true_range_df.rolling(atr_int, min_periods=atr_int).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        x_df = (log_close_df - mean_df) / (atr_df * np.sqrt(lookback_int + 1.0))
    x_df = x_df.where(atr_df > 0)
    return pd.DataFrame(100.0 * ndtr(x_df.to_numpy()) - 50.0, index=x_df.index, columns=x_df.columns).where(x_df.notna())


def linear_trend_frame(inputs: NdxInputs, lookback_int: int, atr_int: int, rsq_bool: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(LT, 100 * Phi(0.6 * LT / sqrt(n)) - 50) on every column (see the module docstring); NaN until both windows are
    full or when a window holds a NaN. The compressed form divides by sqrt(n) (random-walk scaling, as CMMA) so that
    it does not saturate; it is used only by the indicator battery, never by the ranking."""
    from scipy.special import ndtr

    log_close_df = np.log(inputs.close_df.where(inputs.close_df > 0))
    log_high_df, log_low_df = np.log(inputs.high_df.where(inputs.high_df > 0)), np.log(inputs.low_df.where(inputs.low_df > 0))
    previous_log_close_df = log_close_df.shift(1)
    x_centered_vec = np.arange(lookback_int, dtype=float) - (lookback_int - 1) / 2.0
    sxx_float = float(x_centered_vec @ x_centered_vec)
    slope_mat = np.full(log_close_df.shape, np.nan)
    rsq_mat = np.full(log_close_df.shape, np.nan)
    for column_int in range(log_close_df.shape[1]):
        y_vec = log_close_df.iloc[:, column_int].to_numpy(dtype=float)
        if len(y_vec) < lookback_int:
            continue
        # *** CRITICAL*** trailing windows only: window r covers rows r .. r + n - 1 and is stored at row r + n - 1.
        window_mat = np.lib.stride_tricks.sliding_window_view(y_vec, lookback_int)
        centered_mat = window_mat - window_mat.mean(axis=1, keepdims=True)
        sxy_vec = centered_mat @ x_centered_vec
        syy_vec = np.einsum("ij,ij->i", centered_mat, centered_mat)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsq_vec = np.where(syy_vec > 0.0, sxy_vec * sxy_vec / (sxx_float * syy_vec), 0.0)
        slope_mat[lookback_int - 1:, column_int] = sxy_vec / sxx_float  # NaN windows stay NaN (NaN propagates)
        rsq_mat[lookback_int - 1:, column_int] = np.clip(rsq_vec, 0.0, 1.0)
    true_range_df = np.maximum(log_high_df - log_low_df,
                               np.maximum((log_high_df - previous_log_close_df).abs(), (log_low_df - previous_log_close_df).abs()))
    atr_mat = true_range_df.rolling(atr_int, min_periods=atr_int).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lt_mat = np.where(atr_mat > 0, slope_mat * (lookback_int - 1.0) / atr_mat, np.nan)
    if rsq_bool:
        lt_mat = lt_mat * rsq_mat
    lt_df = pd.DataFrame(lt_mat, index=log_close_df.index, columns=log_close_df.columns)
    compressed_df = pd.DataFrame(100.0 * ndtr(0.6 * lt_mat / np.sqrt(lookback_int)) - 50.0, index=lt_df.index, columns=lt_df.columns).where(lt_df.notna())
    return lt_df, compressed_df


def rebalance_weight_df(inputs: NdxInputs, config: NdxConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights indexed by execution date (the session after each decision date)."""
    close_df, date_index = inputs.close_df, inputs.close_df.index
    stock_list = [s for s in close_df.columns if s != REGIME_SYMBOL_STR]
    previous_close_df = close_df.shift(1)
    true_range_df = np.maximum(
        inputs.high_df - inputs.low_df,
        np.maximum((inputs.high_df - previous_close_df).abs(), (inputs.low_df - previous_close_df).abs()),
    )
    # NaN anywhere in the three terms propagates (np.maximum keeps NaN), as in the engine.
    atr_adjusted_df = true_range_df.rolling(config.atr_window_int, min_periods=config.atr_window_int).mean()
    stock_sma_df = close_df.rolling(config.stock_sma_int, min_periods=config.stock_sma_int).mean()
    fast_sma_df = (close_df.rolling(config.trend_fast_sma_int, min_periods=config.trend_fast_sma_int).mean()
                   if config.trend_fast_sma_int > 0 else None)
    cmma_df = cmma_frame(inputs, config.stock_sma_int, config.cmma_atr_int) if config.trend_filter_str == "cmma" else None
    lt_df = (linear_trend_frame(inputs, config.lt_lookback_int, config.lt_atr_int, config.lt_rsq_bool)[0]
             if config.atr_unit_str == "linear_trend" else None)
    regime_close_ser = close_df[REGIME_SYMBOL_STR]
    regime_sma_ser = regime_close_ser.rolling(config.regime_sma_int, min_periods=config.regime_sma_int).mean()

    decision_index = decision_dates(date_index, config.decision_offset_int)
    vxn_index = inputs.vxn_close_ser.index
    row_dict = {}
    for decision_pos_int in range(config.roc_month_int, len(decision_index)):
        decision_ts = decision_index[decision_pos_int]
        execution_pos_int = int(date_index.get_loc(decision_ts)) + 1
        if execution_pos_int >= len(date_index):
            continue
        execution_ts = date_index[execution_pos_int]
        if execution_ts < pd.Timestamp(TRADING_START_STR):
            continue
        weight_ser = pd.Series(0.0, index=stock_list)
        if not np.isfinite(regime_sma_ser.loc[decision_ts]):
            continue  # the engine skips a decision whose regime average is not warm yet
        regime_on_bool = bool(regime_close_ser.loc[decision_ts] > regime_sma_ser.loc[decision_ts]) or not config.regime_filter_bool
        if regime_on_bool:
            lookback_ts = decision_index[decision_pos_int - config.roc_month_int]
            close_now_ser = close_df.loc[decision_ts, stock_list]
            roc_ser = close_now_ser / close_df.loc[lookback_ts, stock_list] - 1.0
            anchor_ser = inputs.raw_close_df.loc[decision_ts, stock_list] / close_now_ser
            has_close_mask = close_now_ser.notna()
            bad_anchor_mask = has_close_mask & ~(np.isfinite(anchor_ser) & (anchor_ser > 0))
            if bad_anchor_mask.any():
                raise ValueError(f"Invalid raw/adjusted anchor at {decision_ts.date()}: {list(anchor_ser[bad_anchor_mask].index)}")
            atr_dollar_ser = atr_adjusted_df.loc[decision_ts, stock_list] * anchor_ser
            score_ser = roc_ser / atr_dollar_ser
            if config.atr_unit_str == "none":
                score_ser = roc_ser.copy()  # ablation: ROC alone
            elif config.atr_unit_str == "linear_trend":
                score_ser = lt_df.loc[decision_ts, stock_list].copy()  # research: Masters' linear trend
            elif config.atr_unit_str == "percent":
                # NATR20: ROC / (ATR$ / RawClose(T)), computed as the engine does: (ROC / ATR$) × RawClose(T).
                score_ser = score_ser * inputs.raw_close_df.loc[decision_ts, stock_list]
            score_ser = score_ser.replace([np.inf, -np.inf], np.nan)
            if cmma_df is not None:
                trend_mask = (cmma_df.loc[decision_ts, stock_list] > config.cmma_threshold_float).fillna(False)
            elif config.trend_fast_sma_int == 0 and config.trend_threshold_float == 0.0:
                trend_mask = (close_now_ser > stock_sma_df.loc[decision_ts, stock_list]).fillna(False)
            else:
                # *** CRITICAL*** both averages end at T (rolling windows over closes <= T).
                fast_ser = fast_sma_df.loc[decision_ts, stock_list] if fast_sma_df is not None else close_now_ser
                distance_ser = fast_ser / stock_sma_df.loc[decision_ts, stock_list] - 1.0
                trend_mask = (distance_ser > config.trend_threshold_float).fillna(False)
            if not config.stock_trend_filter_bool:
                trend_mask = close_now_ser.notna()  # ablation: no stock trend filter
            member_mask = inputs.member_df.loc[decision_ts, stock_list] == 1
            eligible_ser = score_ser[member_mask & trend_mask & score_ser.notna()]
            ranked_frame = pd.DataFrame({"score": eligible_ser, "symbol": eligible_ser.index})
            ranked_frame = ranked_frame.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
            selected_list = list(ranked_frame["symbol"].iloc[: config.top_count_int])
            if selected_list:
                scale_float = 1.0
                if config.vxn_scaled_bool:
                    # *** CRITICAL*** last VXN close dated <= T; no later information.
                    vxn_float = float(inputs.vxn_close_ser.iloc[int(vxn_index.searchsorted(decision_ts, side="right")) - 1])
                    scale_float = min(max(config.vxn_reference_float / vxn_float, VXN_FLOOR_FLOAT), 1.0)
                weight_ser.loc[selected_list] = (1.0 / config.top_count_int) * scale_float
        row_dict[execution_ts] = weight_ser
    return pd.DataFrame(row_dict).T.sort_index()
