"""Scout spec of Tactical Fixed Income L14 (PM_READY;
`strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd`, "TFI").

An independent re-implementation of the signal; execution is the shared weights engine (adjusted share units). The
default config is the engine's frozen contract (frozen current-vintage FRED files, BIL cash sleeve, 25% withholding,
block-and-hold stale rule). Mapped on 2026-10-01 against the module at main 970bb5c (line numbers below).

Data
    prices      IEF, LQD, BIL CAPITALSPECIAL (Norgate, ALLMARKETDAYS padding) from 2002-07-26 to the frozen end
                2026-08-19 (TacticalYieldConfig, :254-256 and :304). Sessions = dates where IEF and LQD both have a
                Close (get_tactical_yield_data :993-1000); BIL is reindexed onto them, NaN before its first bar
                2007-05-30 (_attach_cash_vehicle_price_df :947-981).
    yields      DGS10, DGS3MO, DAAA, DBAA from the hash-locked current-vintage FRED files committed in
                data/research/tactical_yield_tbill_spread/ (load_frozen_fred_snapshot :350-387). They are local: Scout
                downloads nothing. They are NOT point in time (later backfills, e.g. the 2016-10..2017-03 Moody's
                outage): see deviations.py "tfi_current_vintage_fred". The spec refuses files whose SHA-256 changed.

Signal (month-end decision T = the last session of each complete month <= 2026-07; complete_month_end_index :405-418)
    observation  the last date <= session T-1 where all four series exist (select_publication_safe_observation_date
                 :459-489: every candidate <= T-1 is released by T 17:15 ET under the modelled 17:00 / 12:00 releases,
                 so the newest common row <= T-1 always wins)
    spreads      Term = DGS10 - DGS3MO;  Credit = 0.5 x (DAAA + DBAA) - DGS3MO (spread_value_float :492-500)
    prehistory   per proxy, the months before the first decision month on that proxy's own common dates, one value
                 per month from the penultimate common row (the last if only one) (historical_monthly_spread_records
                 :503-536)
    threshold    median of prehistory + every decision spread up to and including T (np.median, current appended
                 first: causal_expanding_median_state_ser :539-560)
    state        spread > threshold (strict; equality stays in cash)
    weights      IEF = 0.5 x term state; LQD = 0.5 x credit state; Cash = 1 - IEF - LQD (:653-667)
    cash sleeve  BIL weight = Cash weight if BIL Close(T) is finite and > 0, else 0 (0% cash before BIL;
                 _execution_target_weight_ser :1191-1213)
    stale rule   a decision whose observation is more than 2 sessions older than T-1 places no order and keeps the
                 positions (apply_stale_input_rule :693-739, block_and_hold); the frozen files have none
    execution    the next session after T (next_session :431-438); the calendar starts at max(first fill,
                 2002-08-01) and ends at the last session (_execution_calendar_index :1456-1468, run_variant :1593)

*** CRITICAL*** Decision after the close of T on yields observed at T-1 or earlier; fills at Open(T+1).

Execution (TacticalYieldStrategy.iterate :1215-1275 + engine process_orders): budget = total value at the close of T
(the DGS3MO accrual is 0 with the BIL vehicle), target shares trunc(budget x w / Close(T)), skip when equal, zero
weight sells all; 5 bps slippage per side, NO commission (fee 0, minimum 0), dividends net of 25% withholding.
`ENGINE_COST_MODEL` holds these costs. The skip test is `np.isclose(target, current)` (rtol 1e-5), equal to an exact
integer test while positions stay below 100,000 shares.

Family parameters (`TfiConfig`; the default is the engine's contract):
    threshold_quantile_float  the quantile of the spread history the spread must exceed (0.5 = np.median, the rule)
    history_month_int         0 = expanding history (the rule); N > 0 = only the last N monthly values
    decision_offset_int       luck band: decide k sessions before the month's last session, fill on the next session
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate
from alpha.scout.specs.taa_3x import offset_decision_index

STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd"
TRADED_TUPLE = ("IEF", "LQD", "BIL")
CASH_VEHICLE_STR = "BIL"
FRED_SERIES_TUPLE = ("DGS10", "DGS3MO", "DAAA", "DBAA")
PRICE_START_STR = "2002-07-26"
BACKTEST_START_STR = "2002-08-01"
FROZEN_END_STR = "2026-08-19"
LAST_COMPLETE_MONTH_STR = "2026-07"
MAX_OBSERVATION_AGE_SESSIONS_INT = 2
SLEEVE_WEIGHT_FLOAT = 0.5
FRED_DIR_PATH = Path(__file__).resolve().parents[3] / "data" / "research" / "tactical_yield_tbill_spread"
FRED_SHA256_DICT = {  # the governed snapshot (same files and hashes as the engine module)
    "DGS10": "afdf06b65f5727d4a7b570cc45addd919d6a53dc5adbea62e238fa3cd278e8a7",
    "DGS3MO": "691c4ba53291a43bdc2c79a81360f1b7494d33d1027ca30212fd47501750d168",
    "DAAA": "9ae61838428e18131f65aeb0e95ebba551380a7278334fdb64313cfbe1ff933f",
    "DBAA": "4db2dbede8a8b574af3444a65c31dff8190890fec1f7084d0bed520145c880ae",
}
TERM_SERIES_TUPLE = ("DGS10", "DGS3MO")
CREDIT_SERIES_TUPLE = ("DAAA", "DBAA", "DGS3MO")
ENGINE_COST_MODEL = CostModel(slippage_float=0.0005, fee_per_share_float=0.0, min_fee_float=0.0, dividend_withholding_float=0.25)


@dataclass(frozen=True)
class TfiConfig:
    threshold_quantile_float: float = 0.5
    history_month_int: int = 0  # 0 = expanding
    decision_offset_int: int = 0

    def __post_init__(self):
        if not 0.0 < self.threshold_quantile_float < 1.0 or self.history_month_int < 0 or self.decision_offset_int < 0:
            raise ValueError("TfiConfig: quantile in (0, 1), history_month_int >= 0, decision_offset_int >= 0.")


LIVE_CONFIG = TfiConfig()  # the PM_READY engine contract


@dataclass(frozen=True)
class TfiInputs:
    open_df: pd.DataFrame  # CAPITALSPECIAL, TRADED_TUPLE, on the IEF/LQD session index
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame
    total_return_close_df: pd.DataFrame  # TOTALRETURN closes of TRADED_TUPLE (MCPT matrix, S3 labels)
    yield_df: pd.DataFrame  # FRED_SERIES_TUPLE in percent, by observation date (not aligned to sessions)


def _read_fred_ser(series_str: str, fred_dir_path: Path) -> pd.Series:
    path = fred_dir_path / f"fred_{series_str.lower()}.csv"
    digest_str = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest_str != FRED_SHA256_DICT[series_str]:
        raise RuntimeError(f"{path.name} changed (sha256 {digest_str}); TFI is frozen on the governed FRED snapshot.")
    frame = pd.read_csv(path)
    value_ser = pd.to_numeric(frame.set_index("observation_date")[series_str], errors="coerce").dropna()
    value_ser.index = pd.to_datetime(value_ser.index).normalize()
    return value_ser.sort_index().astype(float).rename(series_str)


def load_inputs(fred_dir_path: Path = FRED_DIR_PATH) -> TfiInputs:
    from data.norgate_loader import load_price_timeseries

    def frame(symbol_str: str, adjustment_str: str) -> pd.DataFrame:
        return load_price_timeseries(symbol_str, adjustment_str=adjustment_str, start_date_str=PRICE_START_STR, end_date_str=FROZEN_END_STR)

    capital_dict = {s: frame(s, "CAPITALSPECIAL") for s in TRADED_TUPLE}
    session_index = capital_dict["IEF"].index.union(capital_dict["LQD"].index)
    both_bool = capital_dict["IEF"]["Close"].reindex(session_index).notna() & capital_dict["LQD"]["Close"].reindex(session_index).notna()
    session_index = pd.DatetimeIndex(session_index[both_bool.to_numpy()])
    return TfiInputs(
        open_df=pd.DataFrame({s: capital_dict[s]["Open"] for s in TRADED_TUPLE}).reindex(session_index),
        close_df=pd.DataFrame({s: capital_dict[s]["Close"] for s in TRADED_TUPLE}).reindex(session_index),
        dividend_df=pd.DataFrame({s: capital_dict[s]["Dividend"] for s in TRADED_TUPLE}).reindex(session_index).fillna(0.0),
        total_return_close_df=pd.DataFrame({s: frame(s, "TOTALRETURN")["Close"] for s in TRADED_TUPLE}).reindex(session_index),
        yield_df=pd.concat([_read_fred_ser(s, fred_dir_path) for s in FRED_SERIES_TUPLE], axis=1).sort_index(),
    )


def term_spread(yield_row) -> float:
    return float(yield_row["DGS10"] - yield_row["DGS3MO"])


def credit_spread(yield_row) -> float:
    return 0.5 * float(yield_row["DAAA"] + yield_row["DBAA"]) - float(yield_row["DGS3MO"])


def decision_index(session_index: pd.DatetimeIndex, decision_offset_int: int) -> pd.DatetimeIndex:
    """Per complete month (<= LAST_COMPLETE_MONTH_STR), the session `decision_offset_int` before its last session."""
    index = offset_decision_index(session_index, decision_offset_int)
    return index[index.to_period("M") <= pd.Period(LAST_COMPLETE_MONTH_STR, freq="M")]


def _prehistory_list(yield_df: pd.DataFrame, series_tuple: tuple, spread_fn, first_decision_ts: pd.Timestamp) -> list[float]:
    common_df = yield_df.loc[:, list(series_tuple)].dropna()
    # *** CRITICAL*** prehistory ends before the first decision month; that month enters once, as a decision.
    common_df = common_df[common_df.index < first_decision_ts.to_period("M").start_time]
    value_list = []
    for _period, month_df in common_df.groupby(common_df.index.to_period("M")):
        value_list.append(spread_fn(month_df.iloc[-2 if len(month_df) >= 2 else -1]))
    return value_list


def _threshold(history_list: list[float], config: TfiConfig) -> float:
    value_vec = np.asarray(history_list if config.history_month_int == 0 else history_list[-config.history_month_int:], dtype=float)
    if config.threshold_quantile_float == 0.5:
        return float(np.median(value_vec))  # bit-identical to the engine's np.median
    return float(np.quantile(value_vec, config.threshold_quantile_float))


def signal_df(inputs: TfiInputs, config: TfiConfig = LIVE_CONFIG) -> pd.DataFrame:
    """One row per decision T: observation date, age, spreads, thresholds and states (blocked rows included)."""
    session_index = inputs.open_df.index
    common_yield_df = inputs.yield_df.loc[:, list(FRED_SERIES_TUPLE)].dropna()
    row_list = []
    for decision_ts in decision_index(session_index, config.decision_offset_int):
        position_int = int(session_index.get_loc(decision_ts))
        if position_int == 0:
            if config.decision_offset_int:
                continue  # luck band only: an offset decision on the data's first session has no T-1 to read
            raise ValueError(f"No session before the decision {decision_ts.date()}.")
        previous_ts = session_index[position_int - 1]
        # *** CRITICAL*** publication-safe: the newest common observation dated T-1 or earlier, never T.
        candidate_index = common_yield_df.index[common_yield_df.index <= previous_ts]
        if len(candidate_index) == 0:
            raise ValueError(f"No common yield observation before {decision_ts.date()}.")
        observation_ts = candidate_index[-1]
        yield_row = common_yield_df.loc[observation_ts]
        age_int = int(((session_index > observation_ts) & (session_index <= previous_ts)).sum())
        row_list.append({"decision_date": decision_ts, "observation_date": observation_ts, "age_int": age_int,
                         "term_float": term_spread(yield_row), "credit_float": credit_spread(yield_row)})
    frame = pd.DataFrame(row_list).set_index("decision_date")
    first_ts = frame.index[0]
    for name_str, series_tuple, spread_fn in (("term", TERM_SERIES_TUPLE, term_spread), ("credit", CREDIT_SERIES_TUPLE, credit_spread)):
        history_list = _prehistory_list(inputs.yield_df, series_tuple, spread_fn, first_ts)
        threshold_list = []
        for spread_float in frame[f"{name_str}_float"]:
            history_list.append(float(spread_float))  # *** CRITICAL*** the current spread is in its own median
            threshold_list.append(_threshold(history_list, config))
        frame[f"{name_str}_threshold_float"] = threshold_list
        frame[f"{name_str}_state_float"] = (frame[f"{name_str}_float"] > frame[f"{name_str}_threshold_float"]).astype(float)
    frame["blocked_bool"] = frame["age_int"] > MAX_OBSERVATION_AGE_SESSIONS_INT
    return frame


def rebalance_weight_df(inputs: TfiInputs, config: TfiConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Target weights (IEF, LQD, BIL) indexed by execution date, the session after each non-blocked decision."""
    session_index = inputs.open_df.index
    frame = signal_df(inputs, config)
    row_dict = {}
    for decision_ts, row in frame[~frame["blocked_bool"]].iterrows():
        position_int = int(session_index.get_loc(decision_ts))
        if position_int + 1 >= len(session_index):
            continue
        ief_float, lqd_float = SLEEVE_WEIGHT_FLOAT * row["term_state_float"], SLEEVE_WEIGHT_FLOAT * row["credit_state_float"]
        bil_close_float = float(inputs.close_df.at[decision_ts, CASH_VEHICLE_STR])
        bil_float = 1.0 - ief_float - lqd_float if np.isfinite(bil_close_float) and bil_close_float > 0.0 else 0.0
        row_dict[session_index[position_int + 1]] = pd.Series({"IEF": ief_float, "LQD": lqd_float, "BIL": bil_float})
    return pd.DataFrame(row_dict).T.sort_index()


def simulate_config(inputs: TfiInputs, config: TfiConfig = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    weight_df = rebalance_weight_df(inputs, config)
    start_ts = max(weight_df.index[0], pd.Timestamp(BACKTEST_START_STR))
    return simulate(inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=start_ts,
                    capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=cost_model)


# ---------------------------------------------------------------- MCPT replica (S5)
def mcpt_matrix(inputs: TfiInputs, end_date_str: str = "2022-12-30") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix). Columns: TR daily returns of IEF LQD BIL, then DGS10 DGS3MO DAAA DBAA as known for a
    decision at the close of each date (the newest common observation dated on or before the previous session).
    Rows start the session after BIL's first bar, so no traded column is zero-filled."""
    session_index = inputs.open_df.index
    first_ts = inputs.total_return_close_df[CASH_VEHICLE_STR].first_valid_index() + pd.Timedelta(days=1)
    date_index = session_index[(session_index >= first_ts) & (session_index <= end_date_str)]
    return_df = inputs.total_return_close_df.reindex(session_index).ffill().pct_change().reindex(date_index).fillna(0.0)
    common_yield_df = inputs.yield_df.loc[:, list(FRED_SERIES_TUPLE)].dropna()
    previous_index = session_index[session_index.get_indexer(date_index) - 1]
    known_df = common_yield_df.reindex(common_yield_df.index.union(previous_index)).ffill().reindex(previous_index)
    return date_index, np.column_stack([return_df[list(TRADED_TUPLE)].to_numpy(), known_df.to_numpy()])


def mcpt_prehistory_dict(inputs: TfiInputs, date_index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The spread history before the matrix's first decision (the rule's prehistory plus the engine decisions before
    date_index[0]): fixed context for `fast_daily_list`, never permuted."""
    frame = signal_df(inputs, LIVE_CONFIG)
    before_df = frame[frame.index < date_index[0]]
    result_dict = {}
    for name_str, series_tuple, spread_fn in (("term", TERM_SERIES_TUPLE, term_spread), ("credit", CREDIT_SERIES_TUPLE, credit_spread)):
        prehistory_list = _prehistory_list(inputs.yield_df, series_tuple, spread_fn, frame.index[0])
        result_dict[name_str] = np.asarray(prehistory_list + before_df[f"{name_str}_float"].tolist(), dtype=float)
    return result_dict


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list[dict],
                    prehistory_dict: dict[str, np.ndarray] | None = None) -> list[np.ndarray]:
    """Gross daily returns per configuration from the matrix alone (plus the optional fixed pre-sample history).

    Decision at the last row of each month (the last month has no fill), weights held from the close of the next
    row (`_hold_daily`). A date-row shuffle of `matrix` is a valid null: the yields move with their dates, and the
    in-sample spread history (the threshold's input) is rebuilt in permuted order."""
    from alpha.scout.searches import _hold_daily

    return_mat = matrix[:, :3]
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    decision_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]
    yield_mat = matrix[decision_row_vec, 3:]
    spread_dict = {
        "term": yield_mat[:, 0] - yield_mat[:, 1],
        "credit": 0.5 * (yield_mat[:, 2] + yield_mat[:, 3]) - yield_mat[:, 1],
    }
    prehistory_dict = prehistory_dict or {"term": np.array([]), "credit": np.array([])}
    daily_list = []
    for config_dict in config_list:
        config = TfiConfig(**{k: v for k, v in config_dict.items() if k != "decision_offset_int"})
        state_dict = {}
        for name_str, spread_vec in spread_dict.items():
            history_list = list(prehistory_dict[name_str])
            state_vec = np.zeros(spread_vec.size)
            for m_int, spread_float in enumerate(spread_vec):
                history_list.append(float(spread_float))
                state_vec[m_int] = float(spread_float > _threshold(history_list, config))
            state_dict[name_str] = state_vec
        weight_mat = np.column_stack([
            SLEEVE_WEIGHT_FLOAT * state_dict["term"], SLEEVE_WEIGHT_FLOAT * state_dict["credit"],
            1.0 - SLEEVE_WEIGHT_FLOAT * (state_dict["term"] + state_dict["credit"]),
        ])
        daily_list.append(_hold_daily(weight_mat, decision_row_vec, return_mat))
    return daily_list


# ---------------------------------------------------------------- S3 (class W)
def s3_inputs(inputs: TfiInputs | None = None, end_date_str: str = "2022-12-30") -> dict:
    """Inputs for alpha.scout.stations.s3_allocation, in sample, at the rule's month-end decisions.

    Scores: spread minus its threshold (IEF: term, LQD: credit), as the rule sees them at T. Labels: next-month total
    return over the decision-to-decision span minus the DGS3MO month (the cash the sleeve otherwise holds).
    `predictive_tests` (per-asset time-series slopes are the meaningful part with two assets) and one `gate_split`
    per sleeve (its mean on / off is the return question; its Levene test is the risk question)."""
    inputs = inputs or load_inputs()
    frame = signal_df(inputs, LIVE_CONFIG)
    frame = frame[(~frame["blocked_bool"]) & (frame.index <= end_date_str)]
    close_df = inputs.total_return_close_df[["IEF", "LQD"]].reindex(frame.index)
    cash_ser = (1.0 + inputs.yield_df["DGS3MO"].reindex(frame["observation_date"]).to_numpy() / 100.0) ** (1.0 / 12.0) - 1.0
    # *** CRITICAL*** labels only: the return from decision T to the next decision.
    next_return_df = (close_df.shift(-1) / close_df - 1.0).sub(pd.Series(cash_ser, index=frame.index), axis=0)
    score_df = pd.DataFrame({"IEF": frame["term_float"] - frame["term_threshold_float"],
                             "LQD": frame["credit_float"] - frame["credit_threshold_float"]})
    return {
        "predictive_tests": {"score_df": score_df, "next_return_df": next_return_df, "hurdle_ser": pd.Series(0.0, index=frame.index), "min_asset_int": 2},
        "gate_split": {
            "IEF": {"next_return_ser": next_return_df["IEF"], "gate_on_ser": frame["term_state_float"].astype(bool)},
            "LQD": {"next_return_ser": next_return_df["LQD"], "gate_on_ser": frame["credit_state_float"].astype(bool)},
        },
    }
