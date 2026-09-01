"""Frozen L14 tactical fixed-income implementation for IEF, LQD, and cash.

The rule is the publication-safe modern proxy studied in Pakal:

    Term_T = DGS10_T - DGS3MO_T
    Credit_T = 0.5 * (DAAA_T + DBAA_T) - DGS3MO_T

Each spread is compared with its own expanding median, including the current
observation. A sleeve is active only when the spread is strictly above that
median; equality stays in cash:

    w_IEF,T = 0.5 * 1[Term_T > expanding_median(Term)_T]
    w_LQD,T = 0.5 * 1[Credit_T > expanding_median(Credit)_T]
    w_Cash,T = 1 - w_IEF,T - w_LQD,T

The month-end decision is modeled at 17:15 ET. Treasury observations become
available on the next Norgate session at 17:00 ET and Moody's observations on
the next session at 12:00 ET. Orders are submitted after Close_T and filled at
Open_(T+1). The implementation is deliberately frozen through 2026-08-19 and
uses the exact current-vintage FRED snapshots hashed by the Pakal study.

The legacy Pakal implementation accidentally admitted two observations from
July 2002 into the monthly expanding history. This module applies the literal
one-observation-per-month formula, changing only the 2007-12-31 and 2016-12-30
target decisions relative to those legacy artifacts.

Alpha Super translates the research path into the house execution contract:

1. IEF/LQD fills and marks use Norgate CAPITALSPECIAL prices.
2. Gross dividends are credited explicitly with zero withholding.
3. Positive residual cash earns causal DGS3MO ACT/365 interest.
4. Five basis points of slippage are charged on each executed ETF side.
5. Target shares are sized from Close_T and filled at Open_(T+1).

This is PM_READY research plumbing, not proof of edge and not PAPER/LIVE
approval. The Pakal verdict remains diagnostic/inconclusive because the frozen
stability gate failed and no untouched historical confirmation exists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.taa_df.strategy_taa_df import (
    DefenseFirstStrategy,
    load_execution_price_df,
)


STRATEGY_NAME_STR = "strategy_taa_tactical_fixed_income_ief_lqd"
TRADEABLE_ASSET_TUPLE = ("IEF", "LQD")
BENCHMARK_TUPLE = ("$SPX",)
FRED_SERIES_ID_TUPLE = ("DGS10", "DGS3MO", "DAAA", "DBAA")
TREASURY_SERIES_ID_SET = {"DGS10", "DGS3MO"}
CORPORATE_SERIES_ID_SET = {"DAAA", "DBAA"}

TERM_PROXY_STR = "DGS10-DGS3MO"
CREDIT_PROXY_STR = "mean_DAAA_DBAA-DGS3MO"
THRESHOLD_PERCENTILE_FLOAT = 0.50
DECISION_CUTOFF_MINUTE_INT = 17 * 60 + 15
TREASURY_RELEASE_MINUTE_INT = 17 * 60
CORPORATE_RELEASE_MINUTE_INT = 12 * 60

SLIPPAGE_PER_SIDE_FLOAT = 0.0005
COMMISSION_PER_SHARE_FLOAT = 0.0
COMMISSION_MINIMUM_FLOAT = 0.0

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DEFAULT_FRED_DATA_DIR_PATH = (
    REPO_ROOT_PATH / "data" / "research" / "tactical_yield_tbill_spread"
)
FROZEN_FRED_SHA256_BY_SERIES_DICT = {
    "DGS10": "afdf06b65f5727d4a7b570cc45addd919d6a53dc5adbea62e238fa3cd278e8a7",
    "DGS3MO": "691c4ba53291a43bdc2c79a81360f1b7494d33d1027ca30212fd47501750d168",
    "DAAA": "9ae61838428e18131f65aeb0e95ebba551380a7278334fdb64313cfbe1ff933f",
    "DBAA": "4db2dbede8a8b574af3444a65c31dff8190890fec1f7084d0bed520145c880ae",
}
FRED_FILENAME_BY_SERIES_DICT = {
    "DGS10": "fred_dgs10.csv",
    "DGS3MO": "fred_dgs3mo.csv",
    "DAAA": "fred_daaa.csv",
    "DBAA": "fred_dbaa.csv",
}
FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT = {
    "IEF": "f3e7d846d21bbba6f082016266e0399e6890419db5e543a082ff9f303219d637",
    "LQD": "4ac9a859374d5fc520eb96b64214c4b08c7954772cf8cc5567255cbf28e23413",
    "$SPXTR": "540d94b585c8db800fe7fe2bc991e7c5731ea36c395141df44d6055fe86d0ce8",
}
FROZEN_SIGNAL_CONTRACT_SHA256_STR = (
    "85f16e7376977f7ab762fd907c4d3edf3d863760edbe09a2a887b6e13e56a3b6"
)


def canonical_dataframe_sha256_str(data_df: pd.DataFrame) -> str:
    canonical_df = data_df.copy()
    canonical_df.index = pd.Index(
        [
            value_obj.strftime("%Y-%m-%d")
            if isinstance(value_obj, pd.Timestamp)
            else value_obj
            for value_obj in canonical_df.index
        ],
        name=canonical_df.index.name,
    )
    for column_str in canonical_df.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical_df[column_str]):
            canonical_df[column_str] = canonical_df[column_str].dt.strftime(
                "%Y-%m-%d"
            )
    canonical_csv_str = canonical_df.to_csv(
        index=True,
        na_rep="NA",
        float_format="%.9g",
        lineterminator="\n",
    )
    return hashlib.sha256(canonical_csv_str.encode("utf-8")).hexdigest()


def build_canonical_signal_contract_df(
    signal_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
) -> pd.DataFrame:
    signal_contract_df = signal_df.loc[
        :,
        [
            "observation_date",
            "term_spread_float",
            "credit_spread_float",
            "term_threshold_float",
            "credit_threshold_float",
            "term_state_float",
            "credit_state_float",
        ],
    ].copy()
    signal_contract_df.index.name = "decision_date"
    weight_contract_df = rebalance_weight_df.loc[
        :,
        ["decision_date", "IEF", "LQD", "Cash"],
    ].copy()
    weight_contract_df.index.name = "fill_date"
    weight_contract_df = weight_contract_df.reset_index().set_index("decision_date")
    return signal_contract_df.join(weight_contract_df, how="inner", validate="one_to_one")


@dataclass(frozen=True)
class TacticalYieldConfig:
    tradeable_asset_tuple: tuple[str, ...] = TRADEABLE_ASSET_TUPLE
    benchmark_tuple: tuple[str, ...] = BENCHMARK_TUPLE
    price_start_date_str: str = "2002-07-26"
    end_date_str: str = "2026-08-19"
    last_complete_signal_month_str: str = "2026-07"
    fred_data_dir_path_str: str = str(DEFAULT_FRED_DATA_DIR_PATH)
    capital_base_float: float = 100_000.0
    slippage_per_side_float: float = SLIPPAGE_PER_SIDE_FLOAT
    commission_per_share_float: float = COMMISSION_PER_SHARE_FLOAT
    commission_minimum_float: float = COMMISSION_MINIMUM_FLOAT

    def __post_init__(self) -> None:
        if tuple(self.tradeable_asset_tuple) != TRADEABLE_ASSET_TUPLE:
            raise ValueError("The frozen L14 tradeable assets must be exactly IEF and LQD.")
        if tuple(self.benchmark_tuple) != BENCHMARK_TUPLE:
            raise ValueError("The PM reporting benchmark must remain $SPX.")
        if self.price_start_date_str != "2002-07-26":
            raise ValueError("The frozen execution-price start date must remain 2002-07-26.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_per_side_float != SLIPPAGE_PER_SIDE_FLOAT:
            raise ValueError("The frozen implementation requires 5 bps slippage per side.")
        if self.commission_per_share_float != COMMISSION_PER_SHARE_FLOAT:
            raise ValueError("The frozen implementation requires zero per-share commission.")
        if self.commission_minimum_float != COMMISSION_MINIMUM_FLOAT:
            raise ValueError("The frozen implementation requires zero minimum commission.")
        if self.end_date_str != "2026-08-19":
            raise ValueError(
                "The PM_READY L14 implementation is frozen through 2026-08-19. "
                "A later end date requires a separately governed forward-shadow update."
            )
        if self.last_complete_signal_month_str != "2026-07":
            raise ValueError("The frozen last complete decision month must remain 2026-07.")


DEFAULT_CONFIG = TacticalYieldConfig()


@dataclass(frozen=True)
class FrozenFredSnapshot:
    series_id_str: str
    value_ser: pd.Series
    source_path_str: str
    sha256_str: str
    latest_observation_date_ts: pd.Timestamp
    vintage_policy_str: str = "current_vintage_frozen_not_alfred"


TacticalYieldDataTuple = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    tuple[FrozenFredSnapshot, ...],
]


def _sha256_file_str(file_path: Path) -> str:
    digest_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def load_frozen_fred_snapshot(
    series_id_str: str,
    data_dir_path: Path,
) -> FrozenFredSnapshot:
    if series_id_str not in FRED_FILENAME_BY_SERIES_DICT:
        raise ValueError(f"Unsupported frozen FRED series: {series_id_str}.")
    source_path = data_dir_path / FRED_FILENAME_BY_SERIES_DICT[series_id_str]
    actual_sha256_str = _sha256_file_str(source_path)
    expected_sha256_str = FROZEN_FRED_SHA256_BY_SERIES_DICT[series_id_str]
    if actual_sha256_str != expected_sha256_str:
        raise RuntimeError(
            f"Frozen FRED hash mismatch for {series_id_str}: "
            f"expected {expected_sha256_str}, found {actual_sha256_str}."
        )

    source_df = pd.read_csv(source_path)
    required_column_set = {"observation_date", series_id_str}
    if not required_column_set.issubset(source_df.columns):
        raise ValueError(
            f"Frozen FRED file for {series_id_str} must contain "
            f"{sorted(required_column_set)}."
        )
    value_ser = pd.to_numeric(
        source_df.set_index("observation_date")[series_id_str],
        errors="coerce",
    ).dropna()
    value_ser.index = pd.to_datetime(value_ser.index).normalize()
    value_ser = value_ser.sort_index().astype(float)
    value_ser.name = series_id_str
    if value_ser.empty:
        raise ValueError(f"Frozen FRED file for {series_id_str} has no values.")
    return FrozenFredSnapshot(
        series_id_str=series_id_str,
        value_ser=value_ser,
        source_path_str=str(source_path),
        sha256_str=actual_sha256_str,
        latest_observation_date_ts=pd.Timestamp(value_ser.index[-1]),
    )


def load_frozen_yield_panel(
    config_obj: TacticalYieldConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, tuple[FrozenFredSnapshot, ...]]:
    data_dir_path = Path(config_obj.fred_data_dir_path_str)
    snapshot_tuple = tuple(
        load_frozen_fred_snapshot(series_id_str, data_dir_path)
        for series_id_str in FRED_SERIES_ID_TUPLE
    )
    yield_df = pd.concat(
        [snapshot_obj.value_ser for snapshot_obj in snapshot_tuple],
        axis=1,
    ).sort_index()
    return yield_df, snapshot_tuple


def complete_month_end_index(
    session_index: pd.DatetimeIndex,
    last_complete_signal_month_str: str,
) -> pd.DatetimeIndex:
    normalized_session_index = pd.DatetimeIndex(session_index).tz_localize(None).normalize()
    session_ser = pd.Series(normalized_session_index, index=normalized_session_index)
    # *** CRITICAL*** The last actual Norgate session in each complete month is
    # the decision Close_T. The partial August 2026 month is excluded.
    month_end_ser = session_ser.groupby(normalized_session_index.to_period("M")).last()
    complete_bool_ser = month_end_ser.index <= pd.Period(
        last_complete_signal_month_str,
        freq="M",
    )
    return pd.DatetimeIndex(month_end_ser.loc[complete_bool_ser].to_numpy())


def previous_session(
    reference_date_ts: pd.Timestamp,
    session_index: pd.DatetimeIndex,
) -> pd.Timestamp:
    position_int = int(session_index.searchsorted(reference_date_ts, side="left")) - 1
    if position_int < 0:
        raise ValueError(f"No previous session before {reference_date_ts}.")
    return pd.Timestamp(session_index[position_int])


def next_session(
    reference_date_ts: pd.Timestamp,
    session_index: pd.DatetimeIndex,
) -> pd.Timestamp:
    position_int = int(session_index.searchsorted(reference_date_ts, side="right"))
    if position_int >= len(session_index):
        raise ValueError(f"No next session after {reference_date_ts}.")
    return pd.Timestamp(session_index[position_int])


def modeled_release_session(
    observation_date_ts: pd.Timestamp,
    series_id_str: str,
    session_index: pd.DatetimeIndex,
) -> tuple[pd.Timestamp, int]:
    if series_id_str in TREASURY_SERIES_ID_SET:
        release_minute_int = TREASURY_RELEASE_MINUTE_INT
    elif series_id_str in CORPORATE_SERIES_ID_SET:
        release_minute_int = CORPORATE_RELEASE_MINUTE_INT
    else:
        raise ValueError(f"Unknown publication model for {series_id_str}.")

    first_session_ts = pd.Timestamp(session_index[0])
    if observation_date_ts < first_session_ts:
        return first_session_ts, release_minute_int
    return next_session(observation_date_ts, session_index), release_minute_int


def select_publication_safe_observation_date(
    decision_date_ts: pd.Timestamp,
    yield_df: pd.DataFrame,
    session_index: pd.DatetimeIndex,
) -> pd.Timestamp:
    valid_yield_df = yield_df.loc[:, list(FRED_SERIES_ID_TUPLE)].dropna()
    prior_session_ts = previous_session(decision_date_ts, session_index)
    candidate_index = valid_yield_df.index[valid_yield_df.index <= prior_session_ts]
    if len(candidate_index) == 0:
        raise ValueError(f"No common yield observation before {decision_date_ts}.")

    for observation_date_ts in reversed(candidate_index):
        observation_date_ts = pd.Timestamp(observation_date_ts)
        availability_bool = True
        for series_id_str in FRED_SERIES_ID_TUPLE:
            release_session_ts, release_minute_int = modeled_release_session(
                observation_date_ts,
                series_id_str,
                session_index,
            )
            release_before_cutoff_bool = (
                release_session_ts < decision_date_ts
                or (
                    release_session_ts == decision_date_ts
                    and release_minute_int <= DECISION_CUTOFF_MINUTE_INT
                )
            )
            availability_bool = availability_bool and release_before_cutoff_bool
        if availability_bool:
            return observation_date_ts
    raise ValueError(f"No publication-safe observation before {decision_date_ts}.")


def spread_value_float(yield_row_ser: pd.Series, proxy_str: str) -> float:
    if proxy_str == TERM_PROXY_STR:
        return float(yield_row_ser["DGS10"] - yield_row_ser["DGS3MO"])
    if proxy_str == CREDIT_PROXY_STR:
        corporate_yield_float = 0.5 * float(
            yield_row_ser["DAAA"] + yield_row_ser["DBAA"]
        )
        return corporate_yield_float - float(yield_row_ser["DGS3MO"])
    raise ValueError(f"Unsupported frozen proxy: {proxy_str}.")


def historical_monthly_spread_records(
    proxy_str: str,
    yield_df: pd.DataFrame,
    before_date_ts: pd.Timestamp,
) -> list[tuple[pd.Timestamp, float]]:
    series_id_list = (
        ["DGS10", "DGS3MO"]
        if proxy_str == TERM_PROXY_STR
        else ["DAAA", "DBAA", "DGS3MO"]
    )
    common_yield_df = yield_df.loc[:, series_id_list].dropna()
    first_decision_month_start_ts = pd.Timestamp(before_date_ts).to_period("M").start_time
    # *** CRITICAL*** expanding-window boundary: prehistory must end before
    # the first decision month. The current month's publication-safe spread is
    # appended exactly once below; admitting earlier dates from that same month
    # would double-weight the first decision month in every later median.
    common_yield_df = common_yield_df[
        common_yield_df.index < first_decision_month_start_ts
    ]
    record_list: list[tuple[pd.Timestamp, float]] = []
    for _month_period, month_yield_df in common_yield_df.groupby(
        common_yield_df.index.to_period("M")
    ):
        # The monthly prehistory sampling rule uses the penultimate common
        # daily observation as a conservative one-session release lag.
        observation_position_int = -2 if len(month_yield_df) >= 2 else -1
        observation_date_ts = pd.Timestamp(month_yield_df.index[observation_position_int])
        record_list.append(
            (
                observation_date_ts,
                spread_value_float(month_yield_df.iloc[observation_position_int], proxy_str),
            )
        )
    return record_list


def causal_expanding_median_state_ser(
    observation_date_ser: pd.Series,
    spread_ser: pd.Series,
    prehistory_record_list: list[tuple[pd.Timestamp, float]],
) -> tuple[pd.Series, pd.Series]:
    history_value_list = [record_tuple[1] for record_tuple in prehistory_record_list]
    state_value_list: list[float] = []
    threshold_value_list: list[float] = []
    for decision_date_ts, spread_float in spread_ser.items():
        observation_date_ts = pd.Timestamp(observation_date_ser.loc[decision_date_ts])
        del observation_date_ts
        # *** CRITICAL*** lookahead-sensitive: the current spread is appended
        # before the median is computed, exactly matching the frozen inclusive-
        # current definition. No future spread is present in this list.
        history_value_list.append(float(spread_float))
        threshold_float = float(np.median(np.asarray(history_value_list, dtype=float)))
        threshold_value_list.append(threshold_float)
        # Strictly greater only. Equality deliberately remains in cash.
        state_value_list.append(float(float(spread_float) > threshold_float))
    state_ser = pd.Series(state_value_list, index=spread_ser.index, dtype=float)
    threshold_ser = pd.Series(threshold_value_list, index=spread_ser.index, dtype=float)
    return state_ser, threshold_ser


def build_month_end_signal_and_weight_df(
    yield_df: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    last_complete_signal_month_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_date_index = complete_month_end_index(
        session_index,
        last_complete_signal_month_str,
    )
    signal_row_dict_list: list[dict[str, object]] = []
    for decision_date_ts in decision_date_index:
        decision_date_ts = pd.Timestamp(decision_date_ts)
        observation_date_ts = select_publication_safe_observation_date(
            decision_date_ts,
            yield_df,
            session_index,
        )
        yield_row_ser = yield_df.loc[observation_date_ts]
        release_detail_dict: dict[str, str] = {}
        maximum_release_session_ts = pd.Timestamp.min
        maximum_release_minute_int = -1
        for series_id_str in FRED_SERIES_ID_TUPLE:
            release_session_ts, release_minute_int = modeled_release_session(
                observation_date_ts,
                series_id_str,
                session_index,
            )
            if (
                release_session_ts > maximum_release_session_ts
                or (
                    release_session_ts == maximum_release_session_ts
                    and release_minute_int > maximum_release_minute_int
                )
            ):
                maximum_release_session_ts = release_session_ts
                maximum_release_minute_int = release_minute_int
            release_detail_dict[series_id_str] = (
                f"{release_session_ts.date()}T"
                f"{release_minute_int // 60:02d}:{release_minute_int % 60:02d}:00 ET"
            )

        prior_session_ts = previous_session(decision_date_ts, session_index)
        if observation_date_ts > prior_session_ts:
            raise AssertionError("Publication-safe signal used a same-day observation.")
        if maximum_release_session_ts > decision_date_ts or (
            maximum_release_session_ts == decision_date_ts
            and maximum_release_minute_int > DECISION_CUTOFF_MINUTE_INT
        ):
            raise AssertionError("Publication-safe signal used an unreleased observation.")

        signal_row_dict_list.append(
            {
                "decision_date": decision_date_ts,
                "observation_date": observation_date_ts,
                "term_spread_float": spread_value_float(yield_row_ser, TERM_PROXY_STR),
                "credit_spread_float": spread_value_float(yield_row_ser, CREDIT_PROXY_STR),
                "maximum_modeled_release_session": maximum_release_session_ts,
                "maximum_modeled_release_minute_int": maximum_release_minute_int,
                "publication_available_by_cutoff_bool": True,
                "release_detail_json_str": json.dumps(release_detail_dict, sort_keys=True),
            }
        )

    signal_df = pd.DataFrame(signal_row_dict_list).set_index("decision_date")
    first_decision_ts = pd.Timestamp(decision_date_index[0])
    term_prehistory_record_list = historical_monthly_spread_records(
        TERM_PROXY_STR,
        yield_df,
        first_decision_ts,
    )
    credit_prehistory_record_list = historical_monthly_spread_records(
        CREDIT_PROXY_STR,
        yield_df,
        first_decision_ts,
    )
    term_state_ser, term_threshold_ser = causal_expanding_median_state_ser(
        signal_df["observation_date"],
        signal_df["term_spread_float"],
        term_prehistory_record_list,
    )
    credit_state_ser, credit_threshold_ser = causal_expanding_median_state_ser(
        signal_df["observation_date"],
        signal_df["credit_spread_float"],
        credit_prehistory_record_list,
    )
    signal_df["term_threshold_float"] = term_threshold_ser
    signal_df["credit_threshold_float"] = credit_threshold_ser
    signal_df["term_state_float"] = term_state_ser
    signal_df["credit_state_float"] = credit_state_ser

    target_row_dict_list: list[dict[str, object]] = []
    for decision_date_ts, signal_row_ser in signal_df.iterrows():
        ief_weight_float = 0.5 * float(signal_row_ser["term_state_float"])
        lqd_weight_float = 0.5 * float(signal_row_ser["credit_state_float"])
        cash_weight_float = 1.0 - ief_weight_float - lqd_weight_float
        target_row_dict_list.append(
            {
                "rebalance_date": next_session(pd.Timestamp(decision_date_ts), session_index),
                "decision_date": pd.Timestamp(decision_date_ts),
                "observation_date": pd.Timestamp(signal_row_ser["observation_date"]),
                "IEF": ief_weight_float,
                "LQD": lqd_weight_float,
                "Cash": cash_weight_float,
            }
        )
    month_end_weight_df = pd.DataFrame(target_row_dict_list).set_index("rebalance_date")
    target_weight_sum_ser = month_end_weight_df.loc[:, ["IEF", "LQD", "Cash"]].sum(axis=1)
    if not np.allclose(target_weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise AssertionError("Frozen L14 target weights must sum to one.")
    if (month_end_weight_df.loc[:, ["IEF", "LQD", "Cash"]].to_numpy() < -1e-12).any():
        raise AssertionError("Frozen L14 target weights must remain long-only.")
    return signal_df, month_end_weight_df


def build_causal_cash_return_ser(
    session_index: pd.DatetimeIndex,
    dgs3mo_value_ser: pd.Series,
) -> pd.Series:
    cash_return_value_list: list[float] = []
    for session_position_int, session_date_ts in enumerate(session_index):
        session_date_ts = pd.Timestamp(session_date_ts)
        if session_position_int == 0:
            cash_return_value_list.append(0.0)
            continue
        prior_session_ts = pd.Timestamp(session_index[session_position_int - 1])
        observation_cutoff_ts = (
            pd.Timestamp(session_index[session_position_int - 2])
            if session_position_int >= 2
            else prior_session_ts - pd.offsets.BDay(1)
        )
        # *** CRITICAL*** publication-sensitive: the rate for Close_(T-1) to
        # Close_T comes from an observation no later than session T-2, because
        # the frozen Treasury release model has a one-session lag.
        eligible_yield_ser = dgs3mo_value_ser[
            dgs3mo_value_ser.index <= observation_cutoff_ts
        ]
        if eligible_yield_ser.empty:
            raise ValueError(f"No causal DGS3MO yield before {session_date_ts}.")
        annual_yield_float = float(eligible_yield_ser.iloc[-1]) / 100.0
        calendar_day_count_int = int((session_date_ts - prior_session_ts).days)
        cash_return_value_list.append(
            annual_yield_float * calendar_day_count_int / 365.0
        )
    cash_return_ser = pd.Series(
        cash_return_value_list,
        index=session_index,
        dtype=float,
        name="causal_cash_return_float",
    )
    return cash_return_ser


def get_tactical_yield_data(
    config_obj: TacticalYieldConfig = DEFAULT_CONFIG,
) -> TacticalYieldDataTuple:
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config_obj.tradeable_asset_tuple,
        benchmark_list=config_obj.benchmark_tuple,
        start_date_str=config_obj.price_start_date_str,
        end_date_str=config_obj.end_date_str,
    )
    common_tradeable_bool_ser = execution_price_df.loc[
        :,
        [(asset_str, "Close") for asset_str in config_obj.tradeable_asset_tuple],
    ].notna().all(axis=1)
    common_session_index = pd.DatetimeIndex(
        execution_price_df.index[common_tradeable_bool_ser]
    )
    execution_price_df = execution_price_df.loc[common_session_index].copy()
    actual_norgate_sha256_by_symbol_dict = {
        symbol_str: canonical_dataframe_sha256_str(execution_price_df[symbol_str])
        for symbol_str in FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT
    }
    if actual_norgate_sha256_by_symbol_dict != FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT:
        raise RuntimeError(
            "Frozen Norgate price fingerprint mismatch. Review vendor revisions "
            "before changing the governed snapshot."
        )
    yield_df, fred_snapshot_tuple = load_frozen_yield_panel(config_obj)
    signal_df, rebalance_weight_df = build_month_end_signal_and_weight_df(
        yield_df=yield_df,
        session_index=common_session_index,
        last_complete_signal_month_str=config_obj.last_complete_signal_month_str,
    )
    signal_contract_sha256_str = canonical_dataframe_sha256_str(
        build_canonical_signal_contract_df(signal_df, rebalance_weight_df)
    )
    if signal_contract_sha256_str != FROZEN_SIGNAL_CONTRACT_SHA256_STR:
        raise RuntimeError(
            "Frozen 289-row signal/target contract fingerprint mismatch."
        )
    dgs3mo_snapshot_obj = next(
        snapshot_obj
        for snapshot_obj in fred_snapshot_tuple
        if snapshot_obj.series_id_str == "DGS3MO"
    )
    cash_return_ser = build_causal_cash_return_ser(
        common_session_index,
        dgs3mo_snapshot_obj.value_ser,
    )
    return (
        execution_price_df,
        yield_df,
        signal_df,
        rebalance_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    )


class TacticalYieldStrategy(DefenseFirstStrategy):
    """Monthly L14 allocator plus causal positive-cash accrual."""

    def __init__(
        self,
        *,
        name: str,
        benchmarks: Sequence[str],
        rebalance_weight_df: pd.DataFrame,
        cash_return_ser: pd.Series,
        tradeable_asset_list: Sequence[str],
        capital_base: float,
        slippage: float,
        commission_per_share: float,
        commission_minimum: float,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_weight_df=rebalance_weight_df,
            tradeable_asset_list=tradeable_asset_list,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.cash_return_ser = cash_return_ser.astype(float).copy()
        self.cash_interest_processed_date_set: set[pd.Timestamp] = set()
        self.cash_interest_ledger_row_dict_list: list[dict[str, object]] = []
        self.cash_interest_total_float = 0.0
        self.configure_dividend_cash_ledger(
            enabled_bool=True,
            withholding_rate_float=0.0,
        )
        self._accounting_policy_dict.update(
            {
                "positive_cash_rate_policy_str": "causal_DGS3MO_ACT_365",
                "negative_cash_financing_policy_str": "not_modeled",
                "cash_rate_publication_lag_str": "one_Norgate_session",
                "dividend_withholding_rate_float": 0.0,
                "research_status_str": "diagnostic_inconclusive",
                "paper_live_authorized_bool": False,
            }
        )
        self._data_adjustment_policy_dict["return_space_signal_adjustment_str"] = (
            "not_applicable_FRED_yield_signal"
        )

    def _accrue_positive_cash_interest_float(self) -> float:
        current_bar_ts = pd.Timestamp(self.current_bar)
        if current_bar_ts in self.cash_interest_processed_date_set:
            return 0.0
        if current_bar_ts not in self.cash_return_ser.index:
            raise RuntimeError(f"Missing causal cash return for {current_bar_ts.date()}.")
        cash_return_float = float(self.cash_return_ser.loc[current_bar_ts])
        if not np.isfinite(cash_return_float):
            raise RuntimeError(f"Invalid causal cash return for {current_bar_ts.date()}.")
        positive_cash_base_float = max(float(self.cash), 0.0)
        cash_interest_float = positive_cash_base_float * cash_return_float
        self.cash += cash_interest_float
        self.cash_interest_total_float += cash_interest_float
        self.cash_interest_processed_date_set.add(current_bar_ts)
        self.cash_interest_ledger_row_dict_list.append(
            {
                "date": current_bar_ts,
                "positive_cash_base_float": positive_cash_base_float,
                "cash_return_float": cash_return_float,
                "cash_interest_float": cash_interest_float,
            }
        )
        self._accounting_policy_dict["cash_interest_total_float"] = float(
            self.cash_interest_total_float
        )
        return cash_interest_float

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        del data_df, open_price_ser
        if close_row_ser is None:
            return

        cash_interest_float = self._accrue_positive_cash_interest_float()
        if self.current_bar not in self.rebalance_weight_df.index:
            return

        target_weight_ser = self.rebalance_weight_df.loc[self.current_bar].fillna(0.0)
        # The causal cash accrual belongs to the just-finished close-to-close
        # interval and is available before the current rebalance order budget.
        budget_value_float = float(self.previous_total_value) + cash_interest_float
        current_position_ser = self.get_positions().reindex(
            self.tradeable_asset_list,
            fill_value=0.0,
        )

        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            current_share_float = float(current_position_ser.loc[asset_str])
            if target_weight_float != 0.0 or np.isclose(current_share_float, 0.0):
                continue
            self.order_target_value(
                asset_str,
                0.0,
                trade_id=self.current_trade_map[asset_str],
            )

        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            if target_weight_float <= 0.0:
                continue
            close_price_float = float(close_row_ser[(asset_str, "Close")])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid Close_T sizing price for {asset_str} on {self.previous_bar}."
                )
            current_share_float = float(current_position_ser.loc[asset_str])
            target_value_float = budget_value_float * target_weight_float
            target_share_int = int(target_value_float / close_price_float)
            if np.isclose(target_share_int, current_share_float):
                continue
            if np.isclose(current_share_float, 0.0):
                self.trade_id_int += 1
                self.current_trade_map[asset_str] = self.trade_id_int
            # *** CRITICAL*** The dollar target is frozen from Close_T-known
            # NAV and price; only the actual fill price comes from Open_(T+1).
            self.order_target_value(
                asset_str,
                target_value_float,
                trade_id=self.current_trade_map[asset_str],
            )


class TacticalYieldTimingStrategy(TacticalYieldStrategy):
    """Timing adapter that preserves Vanilla dividend/cash ordering."""

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        current_bar_ts = pd.Timestamp(self.current_bar)
        current_dividend_cash_float = float(
            sum(
                float(dividend_row_dict["net_dividend_cash_float"])
                for dividend_row_dict in self._dividend_ledger_row_dict_list
                if pd.Timestamp(dividend_row_dict["ex_date"]) == current_bar_ts
            )
        )
        # *** CRITICAL*** ExecutionTimingAnalysis credits Dividend_T before it
        # calls iterate(), while Vanilla credits it in process_orders() after
        # iterate(). Temporarily remove that cash so cash interest and sizing
        # remain identical to Vanilla, then restore it after order creation.
        self.cash -= current_dividend_cash_float
        self.total_value -= current_dividend_cash_float
        if self._total_value_history_list:
            self._total_value_history_list = [
                float(self._total_value_history_list[-1]) - current_dividend_cash_float
            ]
        try:
            super().iterate(data_df, close_row_ser, open_price_ser)
        finally:
            self.cash += current_dividend_cash_float
            self.total_value += current_dividend_cash_float
            if self._total_value_history_list:
                self._total_value_history_list = [
                    float(self._total_value_history_list[-1]) + current_dividend_cash_float
                ]


def _attach_fred_provenance(
    strategy_obj: TacticalYieldStrategy,
    fred_snapshot_tuple: tuple[FrozenFredSnapshot, ...],
) -> None:
    strategy_obj.fred_snapshot_tuple = fred_snapshot_tuple
    strategy_obj._data_adjustment_policy_dict["fred_series_provenance_list"] = [
        {
            "series_id_str": snapshot_obj.series_id_str,
            "source_path_str": snapshot_obj.source_path_str,
            "sha256_str": snapshot_obj.sha256_str,
            "latest_observation_date_str": snapshot_obj.latest_observation_date_ts.date().isoformat(),
            "vintage_policy_str": snapshot_obj.vintage_policy_str,
        }
        for snapshot_obj in fred_snapshot_tuple
    ]
    strategy_obj._data_adjustment_policy_dict.update(
        {
            "signal_timing_str": "month_end_Close_T_after_17_15_ET_cutoff",
            "fill_timing_str": "Open_T_plus_1",
            "source_replication_outcome_str": "directionally_replicated",
            "pakal_verdict_str": "diagnostic_inconclusive",
            "pakal_frozen_variant_str": "L14",
            "pakal_legacy_duplicate_first_month_corrected_bool": True,
            "pakal_legacy_inference_canonical_bool": False,
            "economic_benchmark_str": "monthly_rebalanced_50_50_IEF_LQD_matched_costs",
            "pm_reporting_benchmark_str": "$SPX_total_return",
            "norgate_price_sha256_by_symbol_dict": dict(
                FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT
            ),
            "signal_contract_sha256_str": FROZEN_SIGNAL_CONTRACT_SHA256_STR,
        }
    )


def _build_strategy_obj(
    config_obj: TacticalYieldConfig,
    rebalance_weight_df: pd.DataFrame,
    cash_return_ser: pd.Series,
    strategy_class_obj: type[TacticalYieldStrategy] = TacticalYieldStrategy,
) -> TacticalYieldStrategy:
    return strategy_class_obj(
        name=STRATEGY_NAME_STR,
        benchmarks=config_obj.benchmark_tuple,
        rebalance_weight_df=rebalance_weight_df,
        cash_return_ser=cash_return_ser,
        tradeable_asset_list=config_obj.tradeable_asset_tuple,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_per_side_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
    )


def _execution_calendar_index(
    execution_price_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    backtest_start_date_str: str | None,
    end_date_str: str | None = None,
) -> pd.DatetimeIndex:
    calendar_start_ts = pd.Timestamp(rebalance_weight_df.index[0])
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))
    calendar_mask_arr = execution_price_df.index >= calendar_start_ts
    if end_date_str is not None:
        calendar_mask_arr &= execution_price_df.index <= pd.Timestamp(end_date_str)
    return pd.DatetimeIndex(execution_price_df.index[calendar_mask_arr])


def _run_strategy(
    *,
    config_obj: TacticalYieldConfig,
    execution_price_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    cash_return_ser: pd.Series,
    fred_snapshot_tuple: tuple[FrozenFredSnapshot, ...],
    backtest_start_date_str: str | None,
    end_date_str: str | None,
    show_progress_bool: bool,
) -> TacticalYieldStrategy:
    effective_end_ts = pd.Timestamp(end_date_str or config_obj.end_date_str)
    effective_execution_price_df = execution_price_df.loc[
        execution_price_df.index <= effective_end_ts
    ].copy()
    effective_signal_df = signal_df.loc[signal_df.index <= effective_end_ts].copy()
    effective_rebalance_weight_df = rebalance_weight_df.loc[
        rebalance_weight_df.index <= effective_end_ts
    ].copy()
    effective_cash_return_ser = cash_return_ser.loc[
        cash_return_ser.index <= effective_end_ts
    ].copy()
    if effective_rebalance_weight_df.empty:
        raise ValueError(
            "The requested PM window ends before the first executable L14 target."
        )
    strategy_obj = _build_strategy_obj(
        config_obj,
        effective_rebalance_weight_df,
        effective_cash_return_ser,
    )
    strategy_obj.show_taa_weights_report = True
    strategy_obj.month_end_signal_df = effective_signal_df
    strategy_obj.month_end_weight_df = effective_rebalance_weight_df
    _attach_fred_provenance(strategy_obj, fred_snapshot_tuple)
    # *** CRITICAL*** Forward fill is report-only. Execution reads only the
    # discrete rebalance rows inside iterate().
    strategy_obj.daily_target_weights = (
        effective_rebalance_weight_df.loc[:, ["IEF", "LQD", "Cash"]]
        .reindex(effective_execution_price_df.index)
        .ffill()
        .dropna()
    )
    calendar_index = _execution_calendar_index(
        effective_execution_price_df,
        effective_rebalance_weight_df,
        backtest_start_date_str,
        end_date_str,
    )
    run_daily(
        strategy_obj,
        effective_execution_price_df,
        calendar=calendar_index,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=None,
    )
    daily_return_ser = strategy_obj.results["daily_returns"].astype(float)
    causal_cash_return_ser = strategy_obj.cash_return_ser.reindex(
        strategy_obj.results.index
    ).astype(float)
    excess_return_ser = daily_return_ser - causal_cash_return_ser
    excess_return_std_float = float(excess_return_ser.std(ddof=1))
    causal_cash_excess_sharpe_float = (
        float(excess_return_ser.mean() / excess_return_std_float * np.sqrt(252.0))
        if excess_return_std_float > 0.0
        else math.nan
    )
    strategy_obj.research_metric_basis_dict = {
        "alpha_headline_sharpe_basis_str": "zero_risk_free_rate_all_days",
        "pakal_sharpe_basis_str": "daily_strategy_return_minus_causal_DGS3MO_all_days",
        "pakal_basis_sharpe_float": causal_cash_excess_sharpe_float,
        "alpha_headline_sharpe_float": float(
            strategy_obj.summary.loc["Sharpe Ratio", "Strategy"]
        ),
        "average_target_cash_weight_float": float(
            strategy_obj.daily_target_weights["Cash"].mean()
        ),
        "negative_cash_day_count_int": int(
            (strategy_obj.results["cash"].astype(float) < 0.0).sum()
        ),
        "minimum_cash_weight_float": float(
            (
                strategy_obj.results["cash"].astype(float)
                / strategy_obj.results["total_value"].astype(float)
            ).min()
        ),
    }
    return strategy_obj


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = "2002-08-01",
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    config_obj: TacticalYieldConfig = DEFAULT_CONFIG,
) -> TacticalYieldStrategy:
    if end_date_str is not None and pd.Timestamp(end_date_str) > pd.Timestamp(
        config_obj.end_date_str
    ):
        raise ValueError(
            "The frozen L14 PM_READY module cannot run beyond 2026-08-19."
        )
    config_obj = replace(config_obj, capital_base_float=capital_base_float)
    (
        execution_price_df,
        _yield_df,
        signal_df,
        rebalance_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    ) = get_tactical_yield_data(config_obj)
    strategy_obj = _run_strategy(
        config_obj=config_obj,
        execution_price_df=execution_price_df,
        signal_df=signal_df,
        rebalance_weight_df=rebalance_weight_df,
        cash_return_ser=cash_return_ser,
        fred_snapshot_tuple=fred_snapshot_tuple,
        backtest_start_date_str=backtest_start_date_str,
        end_date_str=end_date_str,
        show_progress_bool=show_display_bool,
    )
    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.month_end_signal_df.tail())
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = "2002-08-01",
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = replace(DEFAULT_CONFIG, capital_base_float=capital_base_float)
    if end_date_str is not None and pd.Timestamp(end_date_str) > pd.Timestamp(
        config_obj.end_date_str
    ):
        raise ValueError("Capacity analysis cannot run beyond the frozen L14 end date.")
    (
        execution_price_df,
        _yield_df,
        signal_df,
        rebalance_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    ) = get_tactical_yield_data(config_obj)
    strategy_obj = _run_strategy(
        config_obj=config_obj,
        execution_price_df=execution_price_df,
        signal_df=signal_df,
        rebalance_weight_df=rebalance_weight_df,
        cash_return_ser=cash_return_ser,
        fred_snapshot_tuple=fred_snapshot_tuple,
        backtest_start_date_str=backtest_start_date_str,
        end_date_str=end_date_str,
        show_progress_bool=show_display_bool,
    )
    strategy_obj._performance_benchmark_symbol_str = str(config_obj.benchmark_tuple[0])
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": execution_price_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    (
        execution_price_df,
        _yield_df,
        signal_df,
        rebalance_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    ) = get_tactical_yield_data(config_obj)
    calendar_index = _execution_calendar_index(
        execution_price_df,
        rebalance_weight_df,
        "2002-08-01",
    )

    def strategy_factory_fn() -> TacticalYieldTimingStrategy:
        strategy_obj = _build_strategy_obj(
            config_obj,
            rebalance_weight_df,
            cash_return_ser,
            strategy_class_obj=TacticalYieldTimingStrategy,
        )
        strategy_obj.month_end_signal_df = signal_df.copy()
        strategy_obj.month_end_weight_df = rebalance_weight_df.copy()
        _attach_fred_provenance(strategy_obj, fred_snapshot_tuple)
        return strategy_obj

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": execution_price_df,
        "calendar_idx": calendar_index,
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "exit_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    (
        execution_price_df,
        _yield_df,
        signal_df,
        rebalance_weight_df,
        cash_return_ser,
        fred_snapshot_tuple,
    ) = get_tactical_yield_data(config_obj)
    return {
        "strategy_name_str": STRATEGY_NAME_STR,
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": execution_price_df,
        "calendar_idx": _execution_calendar_index(
            execution_price_df,
            rebalance_weight_df,
            "2002-08-01",
        ),
        "signal_df": signal_df,
        "rebalance_weight_df": rebalance_weight_df,
        "cash_return_ser": cash_return_ser,
        "fred_snapshot_tuple": fred_snapshot_tuple,
    }


def build_stress_test_strategy_obj(
    context_dict: dict[str, object],
) -> TacticalYieldStrategy:
    strategy_obj = _build_strategy_obj(
        context_dict["config_obj"],
        context_dict["rebalance_weight_df"],
        context_dict["cash_return_ser"],
    )
    strategy_obj.month_end_signal_df = context_dict["signal_df"].copy()
    strategy_obj.month_end_weight_df = context_dict["rebalance_weight_df"].copy()
    _attach_fred_provenance(
        strategy_obj,
        context_dict["fred_snapshot_tuple"],
    )
    return strategy_obj


if __name__ == "__main__":
    run_variant()
