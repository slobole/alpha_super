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
"""

from __future__ import annotations

from dataclasses import dataclass

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
    from data.norgate_loader import build_index_constituent_matrix, load_price_timeseries, load_raw_prices

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


def decision_dates(date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last session of each calendar month; the running month counts only if it ended on the XNYS month-end."""
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
    return pd.DatetimeIndex(last_session_ser.to_numpy())


def rebalance_weight_df(inputs: NdxInputs) -> pd.DataFrame:
    """Target weights indexed by execution date (the session after each decision date)."""
    close_df, date_index = inputs.close_df, inputs.close_df.index
    stock_list = [s for s in close_df.columns if s != REGIME_SYMBOL_STR]
    previous_close_df = close_df.shift(1)
    true_range_df = np.maximum(
        inputs.high_df - inputs.low_df,
        np.maximum((inputs.high_df - previous_close_df).abs(), (inputs.low_df - previous_close_df).abs()),
    )
    # NaN anywhere in the three terms propagates (np.maximum keeps NaN), as in the engine.
    atr_adjusted_df = true_range_df.rolling(ATR_WINDOW_INT, min_periods=ATR_WINDOW_INT).mean()
    stock_sma_df = close_df.rolling(STOCK_SMA_INT, min_periods=STOCK_SMA_INT).mean()
    regime_close_ser = close_df[REGIME_SYMBOL_STR]
    regime_sma_ser = regime_close_ser.rolling(REGIME_SMA_INT, min_periods=REGIME_SMA_INT).mean()

    decision_index = decision_dates(date_index)
    vxn_index = inputs.vxn_close_ser.index
    row_dict = {}
    for decision_pos_int in range(ROC_MONTH_INT, len(decision_index)):
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
        regime_on_bool = bool(regime_close_ser.loc[decision_ts] > regime_sma_ser.loc[decision_ts])
        if regime_on_bool:
            lookback_ts = decision_index[decision_pos_int - ROC_MONTH_INT]
            close_now_ser = close_df.loc[decision_ts, stock_list]
            roc_ser = close_now_ser / close_df.loc[lookback_ts, stock_list] - 1.0
            anchor_ser = inputs.raw_close_df.loc[decision_ts, stock_list] / close_now_ser
            has_close_mask = close_now_ser.notna()
            bad_anchor_mask = has_close_mask & ~(np.isfinite(anchor_ser) & (anchor_ser > 0))
            if bad_anchor_mask.any():
                raise ValueError(f"Invalid raw/adjusted anchor at {decision_ts.date()}: {list(anchor_ser[bad_anchor_mask].index)}")
            atr_dollar_ser = atr_adjusted_df.loc[decision_ts, stock_list] * anchor_ser
            score_ser = (roc_ser / atr_dollar_ser).replace([np.inf, -np.inf], np.nan)
            trend_mask = (close_now_ser > stock_sma_df.loc[decision_ts, stock_list]).fillna(False)
            member_mask = inputs.member_df.loc[decision_ts, stock_list] == 1
            eligible_ser = score_ser[member_mask & trend_mask & score_ser.notna()]
            ranked_frame = pd.DataFrame({"score": eligible_ser, "symbol": eligible_ser.index})
            ranked_frame = ranked_frame.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
            selected_list = list(ranked_frame["symbol"].iloc[:TOP_COUNT_INT])
            if selected_list:
                # *** CRITICAL*** last VXN close dated <= T; no later information.
                vxn_float = float(inputs.vxn_close_ser.iloc[int(vxn_index.searchsorted(decision_ts, side="right")) - 1])
                scale_float = min(max(VXN_REFERENCE_FLOAT / vxn_float, VXN_FLOOR_FLOAT), 1.0)
                weight_ser.loc[selected_list] = (1.0 / TOP_COUNT_INT) * scale_float
        row_dict[execution_ts] = weight_ser
    return pd.DataFrame(row_dict).T.sort_index()
