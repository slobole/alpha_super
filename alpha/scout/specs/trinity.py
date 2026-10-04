"""Scout spec of Trinity volatility control 8% with BIL (PM_READY;
`strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil`).

An independent re-implementation of the signal; execution is the shared weights engine (adjusted share units) with a
path-dependent decision hook, because the no-trade band reads the ledger. Mapped on 2026-10-01 against main 970bb5c:
strategy_taa_trinity_vol_control_8_bil.py (lines "T:") and strategy_taa_beyond_6040.py (lines "B:").

Data (B: get_beyond_6040_data :120-135 -> data.norgate_loader.load_raw_prices): VTI GLD TLT BIL CAPITALSPECIAL with
ALLMARKETDAYS padding plus the $SPX benchmark (TR data symbol), from 1995-01-01 to today. The session index is the
union of all of them (pre-inception rows are NaN). Signals use CAPITALSPECIAL closes, not total return.

Monthly base weights (B: compute_month_end_inverse_vol_weight_df :146-183)
    r_i      = Close_i.pct_change(fill_method=None)            (VTI, GLD, TLT)
    sigma_i  = r_i.rolling(63).std(ddof=1) x sqrt(252)
    month m  = sigma resampled "ME" with last (last non-NaN of the month); <= 0 -> NaN; 1/sigma normalised to sum 1;
               months with any NaN dropped
    timing   month m's weights apply from the close of the last session before the first session of month m+1 (the
             month's last index session; a month with no next month is never used) and are carried forward
             (B: build_signal_base_weight_df :186-224, map_month_end_weights_to_rebalance_open_df :227-251;
             T: build_monthly_rebalance_signal_ser :167-190 flags that same close as the "monthly" decision)

Daily decision after the close of T (T: iterate :422-471; the engine's iterate runs before process_orders)
    b        = the base weights in force at T; none yet -> no decision
    r_p      = the last 63 rows of r (through T) weighted by today's b (T: compute_base_portfolio_return_ser :93-137;
               every value must be finite, else the engine raises)
    sigma_p  = r_p.std(ddof=1) x sqrt(252) (pandas, as the engine)
    m        = 1 if sigma_p <= 0.085 + 1e-12 (or sigma_p not finite / <= 0), else min(1, 0.08 / sigma_p)
               (B: compute_gross_exposure_float :277-302)
    current  = sum over VTI GLD TLT of shares x Close(T) / total value at the close of T; every Close(T) of the four
               ETFs must be finite and > 0 (T: _current_close_weight_ser :346-367)
    trade if monthly, or current > 1 + 1e-12, or |m - current| + 1e-12 >= 0.05 (T: should_rebalance_exposure_bool :77-90)
    targets  VTI/GLD/TLT = b x m, BIL = 1 - m (T: build_target_weight_ser :140-164)
    sizing   trunc(total(T) x w / Close(T)), fill at Open(T+1); 1 bp slippage, $0.005 a share, $1 minimum, dividends
             net of 25% withholding (T: _submit_target_orders :369-420 + engine process_orders; `ENGINE_COST_MODEL`)
    calendar starts at the first monthly fill where all four ETFs have an Open (T: get_first_actionable_trinity_
             rebalance_ts :193-229): 2007-06-01, after BIL's first bar.

*** CRITICAL*** Every input is read at the close of T or earlier; fills happen at Open(T+1).

Family parameters (`TrinityConfig`; the default is the engine's configuration):
    asset_vol_lookback_int      the asset volatility window of the inverse-volatility weights
    portfolio_vol_lookback_int  the base-portfolio volatility window of the exposure overlay
    vol_target_tuple            (target, trigger): scale to target / sigma_p once sigma_p exceeds the trigger
    exposure_band_float         the no-trade band on risky exposure (fixed in the family grid)
    decision_offset_int         luck band: the monthly base-weight decision k sessions before the month's last session
                                (the daily overlay is unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate
from alpha.scout.specs.taa_3x import offset_decision_index

STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil"
RISK_TUPLE = ("VTI", "GLD", "TLT")
CASH_VEHICLE_STR = "BIL"
TRADED_TUPLE = RISK_TUPLE + (CASH_VEHICLE_STR,)
BENCHMARK_STR = "$SPX"
HISTORY_START_STR = "1995-01-01"
MAX_RISK_EXPOSURE_FLOAT = 1.0
ENGINE_COST_MODEL = CostModel(slippage_float=0.0001, fee_per_share_float=0.005, min_fee_float=1.0, dividend_withholding_float=0.25)


@dataclass(frozen=True)
class TrinityConfig:
    asset_vol_lookback_int: int = 63
    portfolio_vol_lookback_int: int = 63
    vol_target_tuple: tuple = (0.08, 0.085)  # (target, trigger)
    exposure_band_float: float = 0.05
    decision_offset_int: int = 0

    def __post_init__(self):
        if self.asset_vol_lookback_int < 2 or self.portfolio_vol_lookback_int < 2 or self.decision_offset_int < 0:
            raise ValueError("TrinityConfig: lookbacks >= 2 and decision_offset_int >= 0.")
        if len(self.vol_target_tuple) != 2 or min(self.vol_target_tuple) <= 0.0 or not 0.0 <= self.exposure_band_float <= 1.0:
            raise ValueError("TrinityConfig: vol_target_tuple = (target, trigger) > 0 and a band in [0, 1].")


LIVE_CONFIG = TrinityConfig()  # the PM_READY engine configuration


@dataclass(frozen=True)
class TrinityInputs:
    open_df: pd.DataFrame  # CAPITALSPECIAL, TRADED_TUPLE, on the engine's session index (from 1995)
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    total_return_close_df: pd.DataFrame  # TOTALRETURN closes of TRADED_TUPLE (MCPT matrix, S3 labels)


def load_inputs(end_date_str: str | None = None) -> TrinityInputs:
    from data.norgate_loader import load_price_timeseries, load_raw_prices

    pricing_df = load_raw_prices(symbols=list(TRADED_TUPLE), benchmarks=[BENCHMARK_STR], start_date=HISTORY_START_STR, end_date=end_date_str)
    session_index = pd.DatetimeIndex(pricing_df.index)

    def field_df(field_str: str) -> pd.DataFrame:
        return pd.DataFrame({s: pricing_df[(s, field_str)].astype(float) for s in TRADED_TUPLE}, index=session_index)

    total_return_df = pd.DataFrame({
        s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=HISTORY_START_STR, end_date_str=end_date_str)["Close"]
        for s in TRADED_TUPLE
    }).reindex(session_index)
    return TrinityInputs(open_df=field_df("Open"), close_df=field_df("Close"), dividend_df=field_df("Dividend").fillna(0.0),
                         total_return_close_df=total_return_df)


def risk_return_df(inputs: TrinityInputs) -> pd.DataFrame:
    return inputs.close_df[list(RISK_TUPLE)].pct_change(fill_method=None)


def base_weight_decision_df(inputs: TrinityInputs, config: TrinityConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Inverse-volatility base weights indexed by their decision session (the close they are first used at)."""
    session_index = inputs.close_df.index
    # *** CRITICAL*** trailing windows only.
    vol_df = risk_return_df(inputs).rolling(config.asset_vol_lookback_int).std(ddof=1) * np.sqrt(252.0)
    position_ser = pd.Series(np.arange(len(session_index)), index=session_index)
    if config.decision_offset_int == 0:
        month_vol_df = vol_df.resample("ME").last()
        last_position_ser = position_ser.groupby(session_index.to_period("M")).max()
        # Month m's weights are first used at the close before the first session of month m+1 (its last session).
        decision_list = [session_index[last_position_ser[p]] if p in last_position_ser.index and p + 1 in last_position_ser.index else pd.NaT
                         for p in month_vol_df.index.to_period("M")]
        month_vol_df.index = pd.DatetimeIndex(decision_list)
        month_vol_df = month_vol_df[month_vol_df.index.notna()]
    else:
        decision_index = offset_decision_index(session_index, config.decision_offset_int)
        decision_index = decision_index[position_ser.reindex(decision_index).to_numpy() + 1 < len(session_index)]
        # Luck band only: an offset decision before every traded asset has a price (BIL lists in late May 2007) is
        # not actionable; the live month-end path never meets one.
        close_mat = inputs.close_df[list(TRADED_TUPLE)].reindex(decision_index).to_numpy(dtype=float)
        decision_index = decision_index[(np.isfinite(close_mat) & (close_mat > 0.0)).all(axis=1)]
        month_vol_df = vol_df.reindex(decision_index)
    inverse_df = 1.0 / month_vol_df.where(month_vol_df > 0.0)
    weight_df = inverse_df.div(inverse_df.sum(axis=1), axis=0)
    return weight_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def first_execution_ts(inputs: TrinityInputs, decision_df: pd.DataFrame) -> pd.Timestamp:
    session_index = inputs.open_df.index
    execution_index = session_index[session_index.get_indexer(decision_df.index) + 1]
    open_df = inputs.open_df.reindex(execution_index)
    valid_vec = (np.isfinite(open_df.to_numpy()) & (open_df.to_numpy() > 0.0)).all(axis=1)
    if not valid_vec.any():
        raise RuntimeError("No actionable Trinity rebalance date.")
    return pd.Timestamp(execution_index[valid_vec][0])


def _decision_fn(inputs: TrinityInputs, config: TrinityConfig, decision_df: pd.DataFrame):
    """The engine's daily rule as a weights-engine decision hook (alpha/scout/engines/weights.py)."""
    session_index = inputs.close_df.index
    return_mat = risk_return_df(inputs).to_numpy(dtype=float)
    close_mat = inputs.close_df[list(TRADED_TUPLE)].to_numpy(dtype=float)
    decision_position_vec = session_index.get_indexer(decision_df.index)
    base_weight_mat = decision_df[list(RISK_TUPLE)].to_numpy(dtype=float)
    monthly_set = set(decision_position_vec.tolist())
    lookback_int = config.portfolio_vol_lookback_int
    target_float, trigger_float = config.vol_target_tuple
    risk_count_int = len(RISK_TUPLE)

    def decide(t_idx_int: int, position_vec: np.ndarray, total_float: float):
        p_int = t_idx_int - 1  # T
        slot_int = int(np.searchsorted(decision_position_vec, p_int, side="right")) - 1
        if slot_int < 0:
            return None
        base_vec = base_weight_mat[slot_int]
        window_mat = return_mat[p_int - lookback_int + 1: p_int + 1]
        if p_int + 1 < lookback_int or not np.isfinite(window_mat).all():
            raise RuntimeError(f"Incomplete base-portfolio return window at {session_index[p_int].date()}.")
        base_return_ser = pd.DataFrame(window_mat).mul(base_vec, axis=1).sum(axis=1)
        sigma_float = float(base_return_ser.std(ddof=1) * np.sqrt(252.0))
        if not np.isfinite(sigma_float) or sigma_float <= 0.0 or sigma_float <= trigger_float + 1e-12:
            exposure_float = 1.0
        else:
            exposure_float = float(min(1.0, target_float / sigma_float))
        close_vec = close_mat[p_int]
        if not (np.isfinite(close_vec).all() and (close_vec > 0.0).all()):
            raise RuntimeError(f"Invalid close for a Trinity asset at {session_index[p_int].date()}.")
        current_float = 0.0
        for asset_int in range(risk_count_int):
            current_float += float(position_vec[asset_int] * close_vec[asset_int] / total_float)
        trade_bool = (
            p_int in monthly_set
            or current_float > MAX_RISK_EXPOSURE_FLOAT + 1e-12
            or abs(exposure_float - current_float) + 1e-12 >= config.exposure_band_float
        )
        if not trade_bool:
            return None
        return np.append(base_vec * exposure_float, 1.0 - exposure_float)

    return decide


def simulate_config(inputs: TrinityInputs, config: TrinityConfig = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    decision_df = base_weight_decision_df(inputs, config)
    empty_df = pd.DataFrame(columns=list(TRADED_TUPLE), dtype=float)
    return simulate(
        inputs.open_df, inputs.close_df, inputs.dividend_df, empty_df, start_date=first_execution_ts(inputs, decision_df),
        capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=cost_model,
        decision_fn=_decision_fn(inputs, config, decision_df),
    )


def rebalance_weight_df(inputs: TrinityInputs, config: TrinityConfig = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                        capital_float: float = 100_000.0) -> pd.DataFrame:
    """Target weights by execution date. Path-dependent: the band reads the ledger, so the rows depend on capital and
    costs; `simulate` on these rows (same capital and costs) reproduces `simulate_config` exactly."""
    return simulate_config(inputs, config, cost_model, capital_float).decided_weight_df


# ---------------------------------------------------------------- MCPT replica (S5)
def mcpt_matrix(inputs: TrinityInputs, end_date_str: str = "2022-12-30") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix). Columns: TR daily returns of VTI GLD TLT BIL, then the CAPITALSPECIAL daily returns of
    VTI GLD TLT that the signal reads. Rows start the session after the last first bar (BIL), so nothing is
    zero-filled; the replica warms its windows up inside the matrix."""
    session_index = inputs.close_df.index
    first_ts = max(inputs.total_return_close_df[s].first_valid_index() for s in TRADED_TUPLE) + pd.Timedelta(days=1)
    date_index = session_index[(session_index >= first_ts) & (session_index <= end_date_str)]
    total_return_df = inputs.total_return_close_df[list(TRADED_TUPLE)].ffill().pct_change().reindex(date_index).fillna(0.0)
    price_return_df = inputs.close_df[list(RISK_TUPLE)].ffill().pct_change().reindex(date_index).fillna(0.0)
    return date_index, np.column_stack([total_return_df.to_numpy(), price_return_df.to_numpy()])


@njit(cache=False)
def _band_rows(base_day_mat, exposure_vec, monthly_vec, return_mat, band_float):
    """Decision rows and target weights: drifted exposure against the band, as the rule trades (gross, no costs).
    The first position is opened by a monthly decision, as in the engine."""
    row_count_int, asset_count_int = return_mat.shape
    row_vec = np.zeros(row_count_int, dtype=np.int64)
    weight_mat = np.zeros((row_count_int, asset_count_int))
    held_vec, pending_vec = np.zeros(asset_count_int), np.zeros(asset_count_int)
    holding_bool, pending_bool, decision_count_int = False, False, 0
    for t_int in range(row_count_int):
        if holding_bool:
            total_float = 0.0
            for a_int in range(asset_count_int):
                held_vec[a_int] *= 1.0 + return_mat[t_int, a_int]
                total_float += held_vec[a_int]
            for a_int in range(asset_count_int):
                held_vec[a_int] /= total_float
        if pending_bool:  # filled during row t: held from its close
            held_vec[:] = pending_vec
            holding_bool, pending_bool = True, False
        exposure_float = exposure_vec[t_int]
        if not np.isfinite(exposure_float) or not np.isfinite(base_day_mat[t_int, 0]):
            continue
        current_float = held_vec[0] + held_vec[1] + held_vec[2] if holding_bool else 0.0
        if not holding_bool:
            trade_bool = monthly_vec[t_int]
        else:
            trade_bool = monthly_vec[t_int] or current_float > 1.0 + 1e-12 or abs(exposure_float - current_float) + 1e-12 >= band_float
        if trade_bool:
            for a_int in range(3):
                pending_vec[a_int] = base_day_mat[t_int, a_int] * exposure_float
            pending_vec[3] = 1.0 - exposure_float
            row_vec[decision_count_int] = t_int
            weight_mat[decision_count_int] = pending_vec
            decision_count_int += 1
            pending_bool = True
    return row_vec[:decision_count_int], weight_mat[:decision_count_int]


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list[dict]) -> list[np.ndarray]:
    """Gross daily returns per configuration from the matrix alone.

    Monthly inverse-volatility weights at each month's last row; every row, the trailing base-portfolio volatility
    sets the exposure and the rule trades when the drifted exposure leaves the band (or at the monthly decision);
    weights decided at the close of T are held from the close of T+1 (`_hold_daily`, constant between trades). A
    date-row shuffle of `matrix` is a valid null: signal and traded returns move together with their dates."""
    from alpha.scout.searches import _hold_daily

    total_return_mat, price_return_mat = matrix[:, :4], matrix[:, 4:7]
    row_count_int = len(date_index)
    position_ser = pd.Series(np.arange(row_count_int), index=date_index)
    month_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]
    monthly_vec = np.zeros(row_count_int, dtype=bool)
    monthly_vec[month_row_vec] = True
    month_slot_vec = np.searchsorted(month_row_vec, np.arange(row_count_int), side="right") - 1
    price_df = pd.DataFrame(price_return_mat)
    window_cache_dict, daily_list = {}, []
    for config_dict in config_list:
        config = TrinityConfig(**config_dict)
        window_key = (config.asset_vol_lookback_int, config.portfolio_vol_lookback_int)
        if window_key not in window_cache_dict:
            asset_vol_mat = price_df.rolling(window_key[0]).std(ddof=1).to_numpy()[month_row_vec] * np.sqrt(252.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                inverse_mat = np.where(asset_vol_mat > 0.0, 1.0 / asset_vol_mat, np.nan)
                month_weight_mat = inverse_mat / inverse_mat.sum(axis=1, keepdims=True)
            # Base-portfolio returns under every month's weights; each row reads its trailing volatility under its own
            # month's weights (the engine applies today's weights to the whole window).
            base_return_mat = price_return_mat @ np.nan_to_num(month_weight_mat).T
            base_vol_mat = pd.DataFrame(base_return_mat).rolling(window_key[1]).std(ddof=1).to_numpy() * np.sqrt(252.0)
            valid_vec = (month_slot_vec >= 0) & (np.arange(row_count_int) + 1 >= window_key[1])
            sigma_vec = np.full(row_count_int, np.nan)
            sigma_vec[valid_vec] = base_vol_mat[np.flatnonzero(valid_vec), month_slot_vec[valid_vec]]
            base_day_mat = np.full((row_count_int, 3), np.nan)
            base_day_mat[month_slot_vec >= 0] = month_weight_mat[month_slot_vec[month_slot_vec >= 0]]
            window_cache_dict[window_key] = (sigma_vec, base_day_mat)
        sigma_vec, base_day_mat = window_cache_dict[window_key]
        target_float, trigger_float = config.vol_target_tuple
        with np.errstate(divide="ignore", invalid="ignore"):
            exposure_vec = np.where(sigma_vec <= trigger_float + 1e-12, 1.0, np.minimum(1.0, target_float / sigma_vec))
        exposure_vec[~np.isfinite(sigma_vec)] = np.nan
        row_vec, weight_mat = _band_rows(base_day_mat, exposure_vec, monthly_vec, total_return_mat, config.exposure_band_float)
        if row_vec.size == 0:
            daily_list.append(np.zeros(row_count_int))
            continue
        daily_list.append(_hold_daily(weight_mat, row_vec, total_return_mat))
    return daily_list


# ---------------------------------------------------------------- S3 (class W, the exposure overlay as a risk gate)
def s3_inputs(inputs: TrinityInputs | None = None, end_date_str: str = "2022-12-30") -> dict:
    """Inputs for alpha.scout.stations.s3_allocation.gate_split: at each monthly decision, gate on = the trailing
    63-day base-portfolio volatility is at or below the trigger (full exposure); label = the next month's total return
    of the inverse-volatility base portfolio. The overlay's premise is volatility clustering, so the test is the
    volatility ratio off / on (Levene). The inverse-volatility weights themselves are a risk model, not a score."""
    inputs = inputs or load_inputs()
    config = LIVE_CONFIG
    decision_df = base_weight_decision_df(inputs, config)
    decision_df = decision_df[decision_df.index <= end_date_str]
    return_df = risk_return_df(inputs)
    gate_dict, next_dict = {}, {}
    total_return_df = inputs.total_return_close_df[list(RISK_TUPLE)]
    for slot_int, decision_ts in enumerate(decision_df.index):
        position_int = int(return_df.index.get_loc(decision_ts))
        window_df = return_df.iloc[position_int - config.portfolio_vol_lookback_int + 1: position_int + 1]
        if position_int + 1 < config.portfolio_vol_lookback_int or not np.isfinite(window_df.to_numpy()).all():
            continue
        sigma_float = float(window_df.mul(decision_df.iloc[slot_int].to_numpy(), axis=1).sum(axis=1).std(ddof=1) * np.sqrt(252.0))
        gate_dict[decision_ts] = sigma_float <= config.vol_target_tuple[1] + 1e-12
        if slot_int + 1 < len(decision_df):
            # *** CRITICAL*** label only: the base portfolio's return to the next monthly decision.
            growth_ser = total_return_df.loc[decision_df.index[slot_int + 1]] / total_return_df.loc[decision_ts] - 1.0
            next_dict[decision_ts] = float((growth_ser * decision_df.iloc[slot_int]).sum())
    return {"gate_split": {"next_return_ser": pd.Series(next_dict, dtype=float), "gate_on_ser": pd.Series(gate_dict, dtype=bool)}}
