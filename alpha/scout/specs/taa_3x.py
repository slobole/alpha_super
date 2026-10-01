"""Scout spec of the LIVE pod TAA 3x (`strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash`).

An independent re-implementation of the signal; execution is the shared weights engine. Semantics, with the engine
lines they mirror, were mapped on 2026-09-30 (P3):

Signal (month-end ME, strategy_taa_df.py:293-325; vix_cash_variant_utils.py:111-229)
    monthly close   mc = TOTALRETURN Close of GLD, UUP, TLT, DBC, BTAL, last value per calendar month
    cash hurdle     cash_m = (1 + DTB3/100)^(1/12) − 1, last DTB3 observation of the month
    momentum score  mean over k in {1, 3, 6, 12} of mc.pct_change(k)
    ranking         descending score; slot i keeps weight RW_i = (5, 4, 3, 2, 1)/15 if score > cash_m (strict),
                    otherwise that slot's weight goes to TQQQ
    VIX cash gate   rv20 = population std of the last 20 SPY (CAPITALSPECIAL) daily returns x sqrt(252) x 100;
                    gate on when rv20 < $VIX (strict), last value of the month; gate off -> TQQQ weight to cash
    execution date  the first session of the next month (a decision with no next-month bar yet is not executed)

*** CRITICAL*** Decision uses closes up to the last session of month m; fills happen at the next session's Open.

Data: Norgate via `data.norgate_loader.load_price_timeseries` (the engine's own loader, ALLMARKETDAYS padding).
DTB3 is read from the MAIN checkout's cache file (workspace/1_data/DTB3.csv, where the engine run writes it)
without refreshing it: the engine downloads FRED on every run and Scout must not have that side effect. The spec
refuses a cache whose last observation is older than the last executed decision month.

Known shared deviation (both engine and spec): the DTB3 value dated T is used at the T close although FRED
publishes it on T+1 (0 of 168 decisions change; registered in alpha/scout/gate/deviations.py).

Family parameters (P5, `TaaConfig`; the default is the LIVE pod and the identity gate runs on it):
    momentum_month_tuple      the k-month returns averaged into the score
    realized_vol_window_int   the SPY realised-volatility window of the VIX cash gate
    decision_offset_int       luck band: decide k sessions before the month's last session and execute on the
                              next session (0 = the live month-end rule)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_IMPORT_STR = "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"
DEFENSIVE_TUPLE = ("GLD", "UUP", "TLT", "DBC", "BTAL")
FALLBACK_STR = "TQQQ"
TRADED_TUPLE = DEFENSIVE_TUPLE + (FALLBACK_STR,)
RANK_WEIGHT_VEC = np.array([5, 4, 3, 2, 1]) / 15.0
MOMENTUM_MONTH_TUPLE = (1, 3, 6, 12)
START_DATE_STR = "2011-09-13"
REALIZED_VOL_WINDOW_INT = 20


def default_dtb3_csv_path() -> Path:
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    return MAIN_CHECKOUT_ROOT_PATH.parent / "1_data" / "DTB3.csv"


@dataclass(frozen=True)
class TaaConfig:
    momentum_month_tuple: tuple = MOMENTUM_MONTH_TUPLE
    realized_vol_window_int: int = REALIZED_VOL_WINDOW_INT
    decision_offset_int: int = 0


@dataclass(frozen=True)
class TaaInputs:
    total_return_close_df: pd.DataFrame  # signal closes, defensive ETFs
    open_df: pd.DataFrame  # CAPITALSPECIAL, traded symbols, execution index
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    spy_close_ser: pd.Series
    vix_close_ser: pd.Series
    dtb3_ser: pd.Series  # percent


def load_inputs(dtb3_csv_path: Path | None = None) -> TaaInputs:
    from data.norgate_loader import load_price_timeseries

    total_return_close_df = pd.DataFrame(
        {s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=START_DATE_STR)["Close"] for s in DEFENSIVE_TUPLE}
    ).sort_index()
    capital_frame_dict = {
        s: load_price_timeseries(s, adjustment_str="CAPITALSPECIAL", start_date_str=START_DATE_STR) for s in TRADED_TUPLE
    }
    execution_index = pd.DatetimeIndex(sorted(set().union(*[frame.index for frame in capital_frame_dict.values()])))
    dtb3_frame = pd.read_csv(dtb3_csv_path or default_dtb3_csv_path())
    dtb3_ser = pd.Series(
        pd.to_numeric(dtb3_frame.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(dtb3_frame.iloc[:, 0])
    ).dropna().sort_index()
    return TaaInputs(
        total_return_close_df=total_return_close_df,
        open_df=pd.DataFrame({s: capital_frame_dict[s]["Open"] for s in TRADED_TUPLE}).reindex(execution_index),
        close_df=pd.DataFrame({s: capital_frame_dict[s]["Close"] for s in TRADED_TUPLE}).reindex(execution_index),
        dividend_df=pd.DataFrame({s: capital_frame_dict[s]["Dividend"] for s in TRADED_TUPLE}).reindex(execution_index).fillna(0.0),
        spy_close_ser=load_price_timeseries("SPY", adjustment_str="CAPITALSPECIAL", start_date_str=START_DATE_STR)["Close"],
        vix_close_ser=load_price_timeseries("$VIX", adjustment_str="CAPITALSPECIAL", start_date_str=START_DATE_STR)["Close"],
        dtb3_ser=dtb3_ser,
    )


def offset_decision_index(date_index: pd.DatetimeIndex, decision_offset_int: int) -> pd.DatetimeIndex:
    """Per calendar month, the session `decision_offset_int` sessions before the month's last session."""
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    last_position_ser = position_ser.groupby(date_index.to_period("M")).max() - decision_offset_int
    first_position_ser = position_ser.groupby(date_index.to_period("M")).min()
    return date_index[last_position_ser[last_position_ser >= first_position_ser].to_numpy()]


def _monthly_last(frame, decision_offset_int: int, reference_index: pd.DatetimeIndex):
    """Month-end values (offset 0, the live rule) or the values on each month's offset decision session."""
    if decision_offset_int == 0:
        return frame.resample("ME").last()
    decision_index = offset_decision_index(reference_index, decision_offset_int)
    sampled = frame.reindex(frame.index.union(decision_index)).ffill().reindex(decision_index)
    sampled.index = decision_index.to_period("M").to_timestamp("M")
    return sampled


def month_end_weight_df(inputs: TaaInputs, config: TaaConfig = TaaConfig()) -> pd.DataFrame:
    """Target weights per calendar month-end label (before mapping to execution dates)."""
    reference_index = inputs.open_df.index
    monthly_close_df = _monthly_last(inputs.total_return_close_df, config.decision_offset_int, reference_index)
    cash_hurdle_ser = _monthly_last((1.0 + inputs.dtb3_ser / 100.0) ** (1.0 / 12.0) - 1.0, config.decision_offset_int, reference_index)
    # *** CRITICAL*** k-month returns over month-end closes; the score for month m uses closes up to m only.
    momentum_df = sum(monthly_close_df.pct_change(k, fill_method=None) for k in config.momentum_month_tuple) / len(config.momentum_month_tuple)
    combined_df = pd.concat([momentum_df, cash_hurdle_ser.rename("cash_hurdle")], axis=1).dropna()

    helper_df = pd.concat([inputs.spy_close_ser, inputs.vix_close_ser], axis=1, join="inner").dropna()
    helper_df.columns = ["spy", "vix"]
    spy_return_ser = helper_df["spy"] / helper_df["spy"].shift(1) - 1.0
    # *** CRITICAL*** trailing window of the last 20 daily returns, population std (ddof = 0), as the engine.
    realized_vol_ser = spy_return_ser.rolling(config.realized_vol_window_int).std(ddof=0) * np.sqrt(252.0) * 100.0
    gate_df = pd.DataFrame({"rv": realized_vol_ser, "vix": helper_df["vix"]}).dropna()
    gate_month_ser = _monthly_last((gate_df["rv"] < gate_df["vix"]).astype(float), config.decision_offset_int, reference_index).dropna().astype(bool)

    weight_row_dict = {}
    for month_end, row in combined_df.iterrows():
        if month_end not in gate_month_ser.index:
            continue
        score_ser = row[list(DEFENSIVE_TUPLE)].astype(float)
        weight_ser = pd.Series(0.0, index=list(TRADED_TUPLE))
        for slot_int, asset_str in enumerate(score_ser.sort_values(ascending=False).index):
            if score_ser[asset_str] > row["cash_hurdle"]:
                weight_ser[asset_str] = RANK_WEIGHT_VEC[slot_int]
            else:
                weight_ser[FALLBACK_STR] += RANK_WEIGHT_VEC[slot_int]
        if not bool(gate_month_ser.loc[month_end]):
            weight_ser[FALLBACK_STR] = 0.0
        weight_row_dict[month_end] = weight_ser
    return pd.DataFrame(weight_row_dict).T


def rebalance_weight_df(inputs: TaaInputs, config: TaaConfig = TaaConfig()) -> pd.DataFrame:
    """Target weights indexed by execution date: the first session of the month after each decision (offset 0), or
    the session after the offset decision session."""
    execution_index = inputs.open_df.index
    first_session_ser = pd.Series(execution_index, index=execution_index.to_period("M")).groupby(level=0).min()
    decision_ser = pd.Series(offset_decision_index(execution_index, config.decision_offset_int))
    decision_ser.index = pd.DatetimeIndex(decision_ser).to_period("M")
    row_dict, last_executed_month_end = {}, None
    for month_end, weight_ser in month_end_weight_df(inputs, config).iterrows():
        if config.decision_offset_int == 0:
            next_month_period = (month_end + pd.offsets.MonthBegin(1)).to_period("M")
            if next_month_period not in first_session_ser.index:
                continue
            execution_ts = first_session_ser[next_month_period]
        else:
            decision_position_int = int(execution_index.get_loc(decision_ser[month_end.to_period("M")]))
            if decision_position_int + 1 >= len(execution_index):
                continue
            execution_ts = execution_index[decision_position_int + 1]
        row_dict[execution_ts] = weight_ser
        last_executed_month_end = month_end
    # A stale DTB3 cache would silently take an earlier month's hurdle for the last executed decision.
    if last_executed_month_end is not None and inputs.dtb3_ser.index[-1] < last_executed_month_end - pd.Timedelta(days=7):
        raise ValueError(
            f"DTB3 cache ends {inputs.dtb3_ser.index[-1].date()}, before the last executed decision month "
            f"{last_executed_month_end.date()}; refresh it (an engine run does) before running the spec."
        )
    return pd.DataFrame(row_dict).T.sort_index()
