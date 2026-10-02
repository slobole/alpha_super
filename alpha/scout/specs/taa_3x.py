"""Scout spec of the TAA (Defense First) family; the default is the LIVE pod TAA 3x
(`strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash`).

An independent re-implementation of the signal; execution is the shared weights engine. Semantics, with the engine
lines they mirror, were mapped on 2026-09-30 (P3) for TAA 3x and extended on 2026-10-01 to the 1/N, linearity and
2x variants (`VARIANT_DICT`):

Signal (month-end ME, strategy_taa_df.py:280-334; vix_cash_variant_utils.py:111-229)
    monthly close   mc = TOTALRETURN Close of the defensive ETFs, last value per calendar month (strategy_taa_df.py:295)
    cash hurdle     cash_m = (1 + DTB3/100)^(1/12) - 1, last DTB3 observation of the month (strategy_taa_df.py:299)
    momentum score  mean over k in {1, 3, 6, 12} of mc.pct_change(k); months with any NaN score or no hurdle dropped
    ranking         descending score (pandas sort_values); slot i keeps weight SW_i if score > hurdle (strict),
                    otherwise that slot's weight is added to the fallback (strategy_taa_df.py:316-325)
    slot weights    "rank": (5, 4, 3, 2, 1)/15 (strategy_taa_df_btal.py:63-69); "equal": 1/N each, 0.2 with BTAL
                    (strategy_taa_df_btal_1n.py:78), 0.25 without (strategy_taa_df_1n_fallback_tqqq_vix_cash.py:68).
                    With equal slots the rank order (and so a score tie) cannot change any weight.
    VIX cash gate   rv20 = population std of the last 20 SPY (CAPITALSPECIAL) daily returns x sqrt(252) x 100;
                    gate on when rv20 < $VIX (strict), last value of the month (vix_cash_variant_utils.py:111-166);
                    gate off -> fallback weight to cash; months without a gate value are dropped (:184)
    execution date  the first session of the next month (a decision with no next-month bar yet is not executed)
                    (strategy_taa_df.py:337-363)

Linearity score (score_str = "linearity"; strategy_taa_df_btal_linearity.py:102-170, _linearity_1n.py:118-158)
    per asset and lookback L in {21, 63, 126, 252} sessions, on y = log(TOTALRETURN Close), trailing window:
        slope = Sxy / Sxx with x = 0..L-1; corr = Sxy / sqrt(Sxx Syy); R2adj = 1 - (1 - corr^2)(L - 1)/(L - 2)
        score_L = R2adj x slope; a window with a NaN gives NaN; a flat window (Syy <= 0) gives 0
    daily score     mean of the four score_L (NaN if any is NaN)
    month-end       daily score resampled "ME" with last (last non-NaN per asset), then months with any NaN
                    dropped (_linearity_1n.py:134)
    qualification   score > 0.0 (strict, no DTB3 hurdle); equal slots in defensive-list order; failed slots to
                    the fallback (_linearity_1n.py:141-149)

*** CRITICAL*** Decision uses closes up to the last session of month m; fills happen at the next session's Open.

Start dates (fallback_variant_utils.py:47-87; vix_cash_variant_utils.py:68-86): start = max(base start, fallback
inception, VIX inception 1990-01-02). With BTAL the base start is BTAL's inception 2011-09-13; without BTAL it is
2000-01-01, so QLD/SSO give 2006-06-21. Every series (signal, execution, SPY, VIX) is loaded from that start, so the
first momentum decision needs 12 month ends after it.

Data: Norgate via `data.norgate_loader.load_price_timeseries` (the engine's own loader, ALLMARKETDAYS padding).
Execution and marks are CAPITALSPECIAL; share units are split-adjusted (engine default, see deviations.py).
DTB3 is read from the MAIN checkout's cache file (workspace/1_data/DTB3.csv, where the engine run writes it)
without refreshing it: the engine downloads FRED on every run and Scout must not have that side effect. The spec
refuses a cache whose last observation is older than the last executed decision month (momentum score only; the
linearity score does not use DTB3, which is still loaded for the stations' T-bill series).

Known shared deviation (both engine and spec): the DTB3 value dated T is used at the T close although FRED
publishes it on T+1 (0 of 168 decisions change; registered in alpha/scout/gate/deviations.py).

Family parameters (P5, `TaaConfig`; the default is the LIVE pod and the identity gate runs on it):
    momentum_month_tuple      the k-month returns averaged into the score
    realized_vol_window_int   the SPY realised-volatility window of the VIX cash gate
    decision_offset_int       luck band: decide k sessions before the month's last session and execute on the
                              next session (0 = the live month-end rule)
    linearity_day_tuple       the regression lookbacks (sessions) averaged into the linearity score
    linearity_threshold_float the linearity qualification threshold (daily log slope x adjusted R2)
The remaining fields are structural (fixed per variant, never grid axes): defensive_tuple, fallback_str,
slot_weight_str, score_str, start_date_str.

Ablation switches (robustness diagnostics, 2026-10-02; never grid axes, the defaults are the engine rule and the
identity gate runs on them): cash_hurdle_bool=False qualifies a momentum score against 0 instead of the DTB3 hurdle;
vix_gate_bool=False never sends the fallback to cash; defensive_hold_str="cash" keeps a qualifying slot's weight in cash
instead of the asset; fallback_hold_str="cash" keeps a failed slot's weight in cash instead of the fallback;
cash_asset_tuple holds only these assets' qualifying slots in cash (the ranking and the other slots unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_IMPORT_STR = "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"
DEFENSIVE_TUPLE = ("GLD", "UUP", "TLT", "DBC", "BTAL")
NO_BTAL_DEFENSIVE_TUPLE = ("GLD", "UUP", "TLT", "DBC")
FALLBACK_STR = "TQQQ"
TRADED_TUPLE = DEFENSIVE_TUPLE + (FALLBACK_STR,)
RANK_WEIGHT_VEC = np.array([5, 4, 3, 2, 1]) / 15.0
MOMENTUM_MONTH_TUPLE = (1, 3, 6, 12)
LINEARITY_DAY_TUPLE = (21, 63, 126, 252)
START_DATE_STR = "2011-09-13"  # BTAL inception
NO_BTAL_2X_START_DATE_STR = "2006-06-21"  # QLD and SSO inception (the no-BTAL base start is 2000-01-01)
REALIZED_VOL_WINDOW_INT = 20


def default_dtb3_csv_path() -> Path:
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    return MAIN_CHECKOUT_ROOT_PATH.parent / "1_data" / "DTB3.csv"


@dataclass(frozen=True)
class TaaConfig:
    momentum_month_tuple: tuple = MOMENTUM_MONTH_TUPLE
    realized_vol_window_int: int = REALIZED_VOL_WINDOW_INT
    decision_offset_int: int = 0
    linearity_day_tuple: tuple = LINEARITY_DAY_TUPLE
    linearity_threshold_float: float = 0.0
    # Structural fields: which variant of the family this is.
    defensive_tuple: tuple = DEFENSIVE_TUPLE
    fallback_str: str = FALLBACK_STR
    slot_weight_str: str = "rank"  # "rank" (RANK_WEIGHT_VEC) or "equal" (1/N)
    score_str: str = "momentum"  # "momentum" (vs the DTB3 hurdle) or "linearity" (vs linearity_threshold_float)
    start_date_str: str = START_DATE_STR
    # Ablation switches (robustness diagnostics; the defaults are the engine rule).
    cash_hurdle_bool: bool = True
    vix_gate_bool: bool = True
    defensive_hold_str: str = "assets"  # "assets" or "cash"
    fallback_hold_str: str = "asset"  # "asset" or "cash"
    cash_asset_tuple: tuple = ()  # assets whose qualifying slot is held in cash

    def __post_init__(self):
        if self.slot_weight_str not in ("rank", "equal") or self.score_str not in ("momentum", "linearity"):
            raise ValueError(f"Unknown slot_weight_str {self.slot_weight_str!r} or score_str {self.score_str!r}.")
        if self.defensive_hold_str not in ("assets", "cash") or self.fallback_hold_str not in ("asset", "cash"):
            raise ValueError(f"Unknown defensive_hold_str {self.defensive_hold_str!r} or fallback_hold_str {self.fallback_hold_str!r}.")
        if self.slot_weight_str == "rank" and len(self.defensive_tuple) != len(RANK_WEIGHT_VEC):
            raise ValueError("Rank slot weights are defined for five defensive assets only.")

    @property
    def traded_tuple(self) -> tuple:
        return tuple(self.defensive_tuple) + (self.fallback_str,)


LIVE_CONFIG = TaaConfig()  # the LIVE pod TAA 3x


@dataclass(frozen=True)
class TaaVariant:
    config: TaaConfig
    strategy_import_str: str
    status_str: str  # the strategy registry status when the spec was gated


VARIANT_DICT = {
    "taa_3x": TaaVariant(LIVE_CONFIG, STRATEGY_IMPORT_STR, "LIVE"),
    "taa_3x_1n": TaaVariant(
        TaaConfig(slot_weight_str="equal"), "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "WIRED"
    ),
    "taa_lin_1n_qqq": TaaVariant(
        TaaConfig(slot_weight_str="equal", score_str="linearity", fallback_str="QQQ"),
        "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "WIRED",
    ),
    "taa_2x_1n_qld": TaaVariant(
        TaaConfig(slot_weight_str="equal", fallback_str="QLD"),
        "strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash", "PM_READY",
    ),
    "taa_nobtal_2x_1n_qld": TaaVariant(
        TaaConfig(slot_weight_str="equal", fallback_str="QLD", defensive_tuple=NO_BTAL_DEFENSIVE_TUPLE, start_date_str=NO_BTAL_2X_START_DATE_STR),
        "strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash", "PM_READY",
    ),
    "taa_nobtal_2x_1n_sso": TaaVariant(
        TaaConfig(slot_weight_str="equal", fallback_str="SSO", defensive_tuple=NO_BTAL_DEFENSIVE_TUPLE, start_date_str=NO_BTAL_2X_START_DATE_STR),
        "strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash", "PM_READY",
    ),
}


@dataclass(frozen=True)
class TaaInputs:
    total_return_close_df: pd.DataFrame  # signal closes, defensive ETFs
    open_df: pd.DataFrame  # CAPITALSPECIAL, traded symbols, execution index
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    spy_close_ser: pd.Series
    vix_close_ser: pd.Series
    dtb3_ser: pd.Series  # percent
    linearity_cache_dict: dict = field(default_factory=dict, repr=False, compare=False)  # (assets, lookback) -> scores


def load_inputs(dtb3_csv_path: Path | None = None, config: TaaConfig = LIVE_CONFIG) -> TaaInputs:
    from data.norgate_loader import load_price_timeseries

    start_str, defensive_tuple, traded_tuple = config.start_date_str, tuple(config.defensive_tuple), config.traded_tuple
    total_return_close_df = pd.DataFrame(
        {s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str=start_str)["Close"] for s in defensive_tuple}
    ).sort_index()
    capital_frame_dict = {
        s: load_price_timeseries(s, adjustment_str="CAPITALSPECIAL", start_date_str=start_str) for s in traded_tuple
    }
    execution_index = pd.DatetimeIndex(sorted(set().union(*[frame.index for frame in capital_frame_dict.values()])))
    dtb3_frame = pd.read_csv(dtb3_csv_path or default_dtb3_csv_path())
    dtb3_ser = pd.Series(
        pd.to_numeric(dtb3_frame.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(dtb3_frame.iloc[:, 0])
    ).dropna().sort_index()
    return TaaInputs(
        total_return_close_df=total_return_close_df,
        open_df=pd.DataFrame({s: capital_frame_dict[s]["Open"] for s in traded_tuple}).reindex(execution_index),
        close_df=pd.DataFrame({s: capital_frame_dict[s]["Close"] for s in traded_tuple}).reindex(execution_index),
        dividend_df=pd.DataFrame({s: capital_frame_dict[s]["Dividend"] for s in traded_tuple}).reindex(execution_index).fillna(0.0),
        spy_close_ser=load_price_timeseries("SPY", adjustment_str="CAPITALSPECIAL", start_date_str=start_str)["Close"],
        vix_close_ser=load_price_timeseries("$VIX", adjustment_str="CAPITALSPECIAL", start_date_str=start_str)["Close"],
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


def _linearity_lookback_df(log_close_df: pd.DataFrame, lookback_int: int) -> pd.DataFrame:
    """R2adj x OLS slope of the trailing `lookback_int` log closes, per asset (vectorised; NaN window -> NaN)."""
    x_centered_vec = np.arange(lookback_int, dtype=float) - (lookback_int - 1) / 2.0
    sxx_float = float(x_centered_vec @ x_centered_vec)
    score_mat = np.full(log_close_df.shape, np.nan)
    for column_int, asset_str in enumerate(log_close_df.columns):
        y_vec = log_close_df[asset_str].to_numpy(dtype=float)
        if len(y_vec) < lookback_int:
            continue
        # *** CRITICAL*** trailing windows only: window r covers rows r .. r + L - 1 and is stored at row r + L - 1.
        window_mat = np.lib.stride_tricks.sliding_window_view(y_vec, lookback_int)
        centered_mat = window_mat - window_mat.mean(axis=1, keepdims=True)
        syy_vec = np.einsum("ij,ij->i", centered_mat, centered_mat)
        sxy_vec = centered_mat @ x_centered_vec
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_vec = sxy_vec / np.sqrt(sxx_float * syy_vec)
            adjusted_r2_vec = 1.0 - (1.0 - corr_vec * corr_vec) * (lookback_int - 1.0) / (lookback_int - 2.0)
            window_score_vec = np.where(syy_vec > 0.0, adjusted_r2_vec * (sxy_vec / sxx_float), 0.0)
        window_score_vec[np.isnan(window_mat).any(axis=1)] = np.nan
        score_mat[lookback_int - 1:, column_int] = window_score_vec
    return pd.DataFrame(score_mat, index=log_close_df.index, columns=log_close_df.columns)


def daily_linearity_score_df(inputs: TaaInputs, config: TaaConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Mean over `linearity_day_tuple` of the per-lookback scores (NaN if any lookback is NaN)."""
    log_close_df = np.log(inputs.total_return_close_df[list(config.defensive_tuple)].astype(float))
    component_list = []
    for lookback_int in config.linearity_day_tuple:
        cache_key = (tuple(config.defensive_tuple), int(lookback_int))
        if cache_key not in inputs.linearity_cache_dict:
            inputs.linearity_cache_dict[cache_key] = _linearity_lookback_df(log_close_df, int(lookback_int))
        component_list.append(inputs.linearity_cache_dict[cache_key])
    return sum(component_list) / float(len(component_list))


def _slot_weight_vec(config: TaaConfig) -> np.ndarray:
    if config.slot_weight_str == "rank":
        return RANK_WEIGHT_VEC  # read at call time: the gate tests plant a broken module value
    return np.full(len(config.defensive_tuple), 1.0 / float(len(config.defensive_tuple)))


def month_end_weight_df(inputs: TaaInputs, config: TaaConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights per calendar month-end label (before mapping to execution dates)."""
    reference_index = inputs.open_df.index
    defensive_list, traded_list, fallback_str = list(config.defensive_tuple), list(config.traded_tuple), config.fallback_str
    if config.score_str == "momentum":
        monthly_close_df = _monthly_last(inputs.total_return_close_df[defensive_list], config.decision_offset_int, reference_index)
        cash_hurdle_ser = _monthly_last((1.0 + inputs.dtb3_ser / 100.0) ** (1.0 / 12.0) - 1.0, config.decision_offset_int, reference_index)
        # *** CRITICAL*** k-month returns over month-end closes; the score for month m uses closes up to m only.
        momentum_df = sum(monthly_close_df.pct_change(k, fill_method=None) for k in config.momentum_month_tuple) / len(config.momentum_month_tuple)
        combined_df = pd.concat([momentum_df, cash_hurdle_ser.rename("hurdle")], axis=1).dropna()
        if not config.cash_hurdle_bool:
            combined_df["hurdle"] = 0.0  # ablation: the months kept are unchanged, only the bar moves
    else:
        # *** CRITICAL*** trailing daily scores sampled at the month's last session; a fixed threshold, no DTB3.
        month_score_df = _monthly_last(daily_linearity_score_df(inputs, config), config.decision_offset_int, reference_index)
        combined_df = month_score_df.dropna(how="any").assign(hurdle=float(config.linearity_threshold_float))

    helper_df = pd.concat([inputs.spy_close_ser, inputs.vix_close_ser], axis=1, join="inner").dropna()
    helper_df.columns = ["spy", "vix"]
    spy_return_ser = helper_df["spy"] / helper_df["spy"].shift(1) - 1.0
    # *** CRITICAL*** trailing window of the last 20 daily returns, population std (ddof = 0), as the engine.
    realized_vol_ser = spy_return_ser.rolling(config.realized_vol_window_int).std(ddof=0) * np.sqrt(252.0) * 100.0
    gate_df = pd.DataFrame({"rv": realized_vol_ser, "vix": helper_df["vix"]}).dropna()
    gate_month_ser = _monthly_last((gate_df["rv"] < gate_df["vix"]).astype(float), config.decision_offset_int, reference_index).dropna().astype(bool)

    slot_weight_vec = _slot_weight_vec(config)
    weight_row_dict = {}
    for month_end, row in combined_df.iterrows():
        if month_end not in gate_month_ser.index:
            continue
        score_ser = row[defensive_list].astype(float)
        weight_ser = pd.Series(0.0, index=traded_list)
        for slot_int, asset_str in enumerate(score_ser.sort_values(ascending=False).index):
            if score_ser[asset_str] > row["hurdle"]:
                if config.defensive_hold_str == "assets" and asset_str not in config.cash_asset_tuple:
                    weight_ser[asset_str] = slot_weight_vec[slot_int]
            elif config.fallback_hold_str == "asset":
                weight_ser[fallback_str] += slot_weight_vec[slot_int]
        if config.vix_gate_bool and not bool(gate_month_ser.loc[month_end]):
            weight_ser[fallback_str] = 0.0
        weight_row_dict[month_end] = weight_ser
    return pd.DataFrame(weight_row_dict).T


def rebalance_weight_df(inputs: TaaInputs, config: TaaConfig = LIVE_CONFIG) -> pd.DataFrame:
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
    stale_bool = last_executed_month_end is not None and inputs.dtb3_ser.index[-1] < last_executed_month_end - pd.Timedelta(days=7)
    if config.score_str == "momentum" and stale_bool:
        raise ValueError(
            f"DTB3 cache ends {inputs.dtb3_ser.index[-1].date()}, before the last executed decision month "
            f"{last_executed_month_end.date()}; refresh it (an engine run does) before running the spec."
        )
    return pd.DataFrame(row_dict).T.sort_index()
