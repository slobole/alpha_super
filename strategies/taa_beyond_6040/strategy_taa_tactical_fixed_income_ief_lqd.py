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

FRED data modes (``TacticalYieldConfig.fred_data_mode_str``):

- ``frozen_current_vintage`` (default): the hash-locked current-vintage files
  above. They contain later backfills, e.g. Moody's DAAA/DBAA for the
  2016-10..2017-03 FRED outage, so they are not point in time.
- ``alfred_point_in_time``: every decision from 2014-04-30 is recomputed from
  the ALFRED vintage published by its vintage date (T, or session T-1 with
  ``alfred_vintage_policy_str="previous_session"``). Earlier decisions have no
  Moody's vintage and keep their frozen rows, labelled as unverifiable. Cash
  accrual uses the frozen DGS3MO file in both modes.

Stale-input rule (both modes): a decision whose common FRED observation is
more than MAX_OBSERVATION_AGE_SESSIONS_INT sessions older than session T-1 is
stale and produces no target. ``stale_input_policy_str="raise"`` stops the run;
``"block_and_hold"`` places no order and keeps the existing positions. The
frozen files have no stale decision, so the frozen contract is unchanged.

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

from alpha.data.alfred_snapshot import (
    AlfredVintageSnapshot,
    load_alfred_snapshot_manifest,
    load_alfred_vintage_snapshot,
)
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
    # Checked 2026-09-24 after Norgate benchmark revision; ETF hashes and NAV match.
    "$SPXTR": "ecf9e3f8aa3fa6e1718f99b6fcd36f808a48fe5efa7238e9c3817c3b0b1f5dee",
}
FROZEN_SIGNAL_CONTRACT_SHA256_STR = (
    "85f16e7376977f7ab762fd907c4d3edf3d863760edbe09a2a887b6e13e56a3b6"
)

# FRED data modes. The frozen current-vintage files remain the governed default.
# The ALFRED mode replays each decision from 2014-04-30 onward with the values
# FRED had published by the vintage date; see
# docs/research/TACTICAL_FI_ALFRED_POINT_IN_TIME_HANDOFF.md.
FRED_DATA_MODE_FROZEN_STR = "frozen_current_vintage"
FRED_DATA_MODE_ALFRED_PIT_STR = "alfred_point_in_time"
SUPPORTED_FRED_DATA_MODE_TUPLE = (FRED_DATA_MODE_FROZEN_STR, FRED_DATA_MODE_ALFRED_PIT_STR)
ALFRED_VINTAGE_DECISION_DATE_STR = "decision_date"
ALFRED_VINTAGE_PREVIOUS_SESSION_STR = "previous_session"
SUPPORTED_ALFRED_VINTAGE_POLICY_TUPLE = (
    ALFRED_VINTAGE_DECISION_DATE_STR,
    ALFRED_VINTAGE_PREVIOUS_SESSION_STR,
)
STALE_INPUT_POLICY_BLOCK_AND_HOLD_STR = "block_and_hold"
STALE_INPUT_POLICY_RAISE_STR = "raise"
SUPPORTED_STALE_INPUT_POLICY_TUPLE = (
    STALE_INPUT_POLICY_BLOCK_AND_HOLD_STR,
    STALE_INPUT_POLICY_RAISE_STR,
)
# Sessions between the common observation used and the prior session T-1.
# Normal publication gives 0 (observation T-1); a bond-market holiday or a
# one-day FRED delay gives 1. The 2014-2026 ALFRED vintages show 0 or 1 on every
# usable decision and 15-96 during the 2016-17 Moody's outage. The value 2 also
# matches the forward-shadow research gate (tactical_fi_forward_snapshot.py,
# not yet committed). Under the previous-session vintage the normal age is
# already 1, so the effective tolerance there is one session.
MAX_OBSERVATION_AGE_SESSIONS_INT = 2
# First archived ALFRED vintage of DAAA/DBAA is 2014-04-02; this is the first
# month-end decision whose own and previous-session vintages both exist.
FIRST_ALFRED_DECISION_DATE_STR = "2014-04-30"
DEFAULT_ALFRED_SNAPSHOT_DIR_PATH = DEFAULT_FRED_DATA_DIR_PATH / "alfred_pit_20260928"
FROZEN_ALFRED_MANIFEST_SHA256_STR = (
    "eb29d05e06d953c9177090635b97530d54a359f3e75af46443f7c9a65be436db"
)
FROZEN_ALFRED_PIT_SIGNAL_CONTRACT_SHA256_BY_VINTAGE_POLICY_DICT = {
    ALFRED_VINTAGE_DECISION_DATE_STR: (
        "4f74d479bd5f0d2d3ed09fb3cc73ab7d0c44c2023f28a974510aedbafe1de040"
    ),
    ALFRED_VINTAGE_PREVIOUS_SESSION_STR: (
        "e98b104bff149e62c2ec5c32405188ba6e3480e0bd5d02cf660f2986e488867d"
    ),
}


class StaleMacroInputError(RuntimeError):
    """A decision would have used a macro observation older than the stale limit."""


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
    fred_data_mode_str: str = FRED_DATA_MODE_FROZEN_STR
    alfred_vintage_policy_str: str = ALFRED_VINTAGE_DECISION_DATE_STR
    stale_input_policy_str: str = STALE_INPUT_POLICY_BLOCK_AND_HOLD_STR
    alfred_snapshot_dir_path_str: str = str(DEFAULT_ALFRED_SNAPSHOT_DIR_PATH)

    def __post_init__(self) -> None:
        if self.fred_data_mode_str not in SUPPORTED_FRED_DATA_MODE_TUPLE:
            raise ValueError(
                f"fred_data_mode_str must be one of {SUPPORTED_FRED_DATA_MODE_TUPLE}."
            )
        if self.alfred_vintage_policy_str not in SUPPORTED_ALFRED_VINTAGE_POLICY_TUPLE:
            raise ValueError(
                "alfred_vintage_policy_str must be one of "
                f"{SUPPORTED_ALFRED_VINTAGE_POLICY_TUPLE}."
            )
        if self.stale_input_policy_str not in SUPPORTED_STALE_INPUT_POLICY_TUPLE:
            raise ValueError(
                "stale_input_policy_str must be one of "
                f"{SUPPORTED_STALE_INPUT_POLICY_TUPLE}."
            )
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


def observation_age_sessions_int(
    observation_date_ts: pd.Timestamp,
    decision_date_ts: pd.Timestamp,
    session_index: pd.DatetimeIndex,
) -> int:
    """Sessions after the observation date, up to and including session T-1.

    Observation T-1 gives 0. Each missing publication day adds one.
    """
    prior_session_ts = previous_session(decision_date_ts, session_index)
    newer_session_bool_arr = (session_index > pd.Timestamp(observation_date_ts)) & (
        session_index <= prior_session_ts
    )
    return int(newer_session_bool_arr.sum())


def apply_stale_input_rule(
    signal_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    stale_input_policy_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fail closed when the common FRED row behind a decision is stale.

    A decision is stale when its observation is more than
    MAX_OBSERVATION_AGE_SESSIONS_INT sessions older than session T-1. A stale
    decision never produces a target:

    - ``raise`` stops the run with StaleMacroInputError (PAPER/LIVE semantics);
    - ``block_and_hold`` records the block and emits no rebalance row, so no
      order is placed and the pod keeps its existing positions until the next
      non-stale decision (historical replay semantics).
    """
    if stale_input_policy_str not in SUPPORTED_STALE_INPUT_POLICY_TUPLE:
        raise ValueError(f"Unsupported stale-input policy: {stale_input_policy_str}.")
    checked_signal_df = signal_df.copy()
    age_value_list = [
        observation_age_sessions_int(
            pd.Timestamp(checked_signal_df.loc[decision_date_ts, "observation_date"]),
            pd.Timestamp(decision_date_ts),
            session_index,
        )
        for decision_date_ts in checked_signal_df.index
    ]
    checked_signal_df["observation_age_sessions_int"] = age_value_list
    checked_signal_df["stale_input_blocked_bool"] = (
        checked_signal_df["observation_age_sessions_int"] > MAX_OBSERVATION_AGE_SESSIONS_INT
    )
    stale_signal_df = checked_signal_df.loc[checked_signal_df["stale_input_blocked_bool"]]
    if not stale_signal_df.empty and stale_input_policy_str == STALE_INPUT_POLICY_RAISE_STR:
        first_decision_ts = pd.Timestamp(stale_signal_df.index[0])
        raise StaleMacroInputError(
            f"Stale FRED input for decision {first_decision_ts.date()}: common "
            f"observation {pd.Timestamp(stale_signal_df.iloc[0]['observation_date']).date()} "
            f"is {int(stale_signal_df.iloc[0]['observation_age_sessions_int'])} sessions "
            f"older than session T-1 (limit {MAX_OBSERVATION_AGE_SESSIONS_INT}). "
            f"{len(stale_signal_df)} stale decision(s) in total; no target was produced."
        )
    blocked_decision_index = pd.DatetimeIndex(stale_signal_df.index)
    checked_weight_df = rebalance_weight_df.loc[
        ~pd.DatetimeIndex(rebalance_weight_df["decision_date"]).isin(blocked_decision_index)
    ].copy()
    return checked_signal_df, checked_weight_df


def alfred_yield_panel_as_of(
    alfred_snapshot_by_series_dict: dict[str, AlfredVintageSnapshot],
    vintage_date_ts: pd.Timestamp,
) -> pd.DataFrame:
    """All four FRED series exactly as published on one sampled vintage date."""
    yield_df = pd.concat(
        [
            alfred_snapshot_by_series_dict[series_id_str].value_ser_as_of(vintage_date_ts)
            for series_id_str in FRED_SERIES_ID_TUPLE
        ],
        axis=1,
    ).sort_index()
    # *** CRITICAL*** publication boundary: a vintage dated v contains nothing
    # dated after v, and v <= decision date T by construction of the caller.
    if len(yield_df.index) and pd.Timestamp(yield_df.index[-1]) > pd.Timestamp(vintage_date_ts):
        raise AssertionError("ALFRED panel contains an observation after its vintage date.")
    return yield_df


def build_point_in_time_signal_and_weight_df(
    frozen_signal_df: pd.DataFrame,
    frozen_weight_df: pd.DataFrame,
    alfred_snapshot_by_series_dict: dict[str, AlfredVintageSnapshot],
    session_index: pd.DatetimeIndex,
    alfred_vintage_policy_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute each decision from 2014-04-30 onward from its ALFRED vintage.

    For decision T the vintage date v is T itself (``decision_date``) or the
    session before T (``previous_session``, conservative about time of day).
    The unchanged frozen rule is re-run from scratch on the panel published by
    v, and only row T is kept. Its spread, its expanding-median threshold and
    all earlier monthly spreads in that median therefore use only values FRED
    had published by v <= T. Earlier decisions have no Moody's vintage and keep
    their frozen current-vintage rows, labelled as such.

    The stale-input rule is applied afterwards by apply_stale_input_rule().
    """
    if alfred_vintage_policy_str not in SUPPORTED_ALFRED_VINTAGE_POLICY_TUPLE:
        raise ValueError(f"Unsupported ALFRED vintage policy: {alfred_vintage_policy_str}.")
    first_alfred_decision_ts = pd.Timestamp(FIRST_ALFRED_DECISION_DATE_STR)
    signal_row_list: list[pd.Series] = []
    weight_row_list: list[pd.Series] = []
    frozen_weight_by_decision_df = frozen_weight_df.reset_index().set_index("decision_date")
    for decision_date_ts in frozen_signal_df.index:
        decision_date_ts = pd.Timestamp(decision_date_ts)
        if decision_date_ts < first_alfred_decision_ts:
            signal_row_ser = frozen_signal_df.loc[decision_date_ts].copy()
            signal_row_ser["fred_data_source_str"] = "frozen_current_vintage_before_alfred_archive"
            signal_row_ser["vintage_date"] = pd.NaT
            signal_row_list.append(signal_row_ser)
            weight_row_list.append(frozen_weight_by_decision_df.loc[decision_date_ts].copy())
            continue

        if alfred_vintage_policy_str == ALFRED_VINTAGE_DECISION_DATE_STR:
            vintage_date_ts = decision_date_ts
        else:
            vintage_date_ts = previous_session(decision_date_ts, session_index)
        vintage_yield_df = alfred_yield_panel_as_of(
            alfred_snapshot_by_series_dict,
            vintage_date_ts,
        )
        # *** CRITICAL*** point-in-time recompute: the whole rule, including
        # every earlier monthly spread inside the expanding median, is rebuilt
        # from the vintage published by v <= T. Only row T is kept.
        vintage_signal_df, vintage_weight_df = build_month_end_signal_and_weight_df(
            yield_df=vintage_yield_df,
            session_index=session_index,
            last_complete_signal_month_str=str(decision_date_ts.to_period("M")),
        )
        if pd.Timestamp(vintage_signal_df.index[-1]) != decision_date_ts:
            raise AssertionError(
                f"ALFRED recompute for {decision_date_ts.date()} ended on "
                f"{vintage_signal_df.index[-1].date()}."
            )
        history_age_ser = pd.Series(
            [
                observation_age_sessions_int(
                    pd.Timestamp(history_row_ser["observation_date"]),
                    pd.Timestamp(history_decision_ts),
                    session_index,
                )
                for history_decision_ts, history_row_ser in vintage_signal_df.iterrows()
            ],
            index=vintage_signal_df.index,
        )
        # A usable decision whose median still contains a month that is stale
        # in this vintage would need an unapproved rule (drop or keep that
        # month). It does not occur in the 2014-2026 snapshot; fail loud if a
        # new snapshot produces it.
        if history_age_ser.iloc[-1] <= MAX_OBSERVATION_AGE_SESSIONS_INT and bool(
            (history_age_ser.iloc[:-1] > MAX_OBSERVATION_AGE_SESSIONS_INT).any()
        ):
            raise AssertionError(
                f"ALFRED median history for {decision_date_ts.date()} contains a stale month; "
                "the median-composition rule for this case is not approved."
            )
        signal_row_ser = vintage_signal_df.loc[decision_date_ts].copy()
        signal_row_ser["fred_data_source_str"] = f"alfred_vintage_{alfred_vintage_policy_str}"
        signal_row_ser["vintage_date"] = vintage_date_ts
        signal_row_list.append(signal_row_ser)
        weight_row_ser = (
            vintage_weight_df.reset_index().set_index("decision_date").loc[decision_date_ts].copy()
        )
        weight_row_list.append(weight_row_ser)

    signal_df = pd.DataFrame(signal_row_list)
    signal_df.index = pd.DatetimeIndex(signal_df.index, name="decision_date")
    signal_df["observation_date"] = pd.to_datetime(signal_df["observation_date"])
    signal_df["vintage_date"] = pd.to_datetime(signal_df["vintage_date"])
    weight_df = pd.DataFrame(weight_row_list)
    weight_df.index = pd.DatetimeIndex(weight_df.index, name="decision_date")
    weight_df = weight_df.reset_index().set_index("rebalance_date")
    weight_df.index = pd.DatetimeIndex(weight_df.index, name="rebalance_date")
    weight_df = weight_df.loc[:, list(frozen_weight_df.columns)]
    for column_str in ("IEF", "LQD", "Cash"):
        weight_df[column_str] = weight_df[column_str].astype(float)
    return signal_df, weight_df


def build_point_in_time_contract_df(
    signal_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
) -> pd.DataFrame:
    """Every decision row, including blocked ones, with its executable target."""
    contract_df = signal_df.loc[
        :,
        [
            "fred_data_source_str",
            "vintage_date",
            "observation_date",
            "observation_age_sessions_int",
            "stale_input_blocked_bool",
            "term_spread_float",
            "credit_spread_float",
            "term_threshold_float",
            "credit_threshold_float",
            "term_state_float",
            "credit_state_float",
        ],
    ].copy()
    contract_df.index.name = "decision_date"
    target_df = rebalance_weight_df.loc[:, ["decision_date", "IEF", "LQD", "Cash"]].copy()
    target_df.index.name = "fill_date"
    target_df = target_df.reset_index().set_index("decision_date")
    return contract_df.join(target_df, how="left", validate="one_to_one")


def load_alfred_point_in_time_snapshots(
    config_obj: TacticalYieldConfig,
) -> tuple[dict[str, object], dict[str, AlfredVintageSnapshot]]:
    snapshot_dir_path = Path(config_obj.alfred_snapshot_dir_path_str)
    manifest_dict = load_alfred_snapshot_manifest(
        snapshot_dir_path,
        FROZEN_ALFRED_MANIFEST_SHA256_STR,
    )
    snapshot_by_series_dict = {
        series_id_str: load_alfred_vintage_snapshot(
            snapshot_dir_path,
            series_id_str,
            manifest_dict,
        )
        for series_id_str in FRED_SERIES_ID_TUPLE
    }
    return manifest_dict, snapshot_by_series_dict


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
    if config_obj.fred_data_mode_str == FRED_DATA_MODE_ALFRED_PIT_STR:
        _manifest_dict, alfred_snapshot_by_series_dict = load_alfred_point_in_time_snapshots(
            config_obj
        )
        signal_df, rebalance_weight_df = build_point_in_time_signal_and_weight_df(
            frozen_signal_df=signal_df,
            frozen_weight_df=rebalance_weight_df,
            alfred_snapshot_by_series_dict=alfred_snapshot_by_series_dict,
            session_index=common_session_index,
            alfred_vintage_policy_str=config_obj.alfred_vintage_policy_str,
        )
    signal_df, rebalance_weight_df = apply_stale_input_rule(
        signal_df,
        rebalance_weight_df,
        common_session_index,
        config_obj.stale_input_policy_str,
    )
    if config_obj.fred_data_mode_str == FRED_DATA_MODE_ALFRED_PIT_STR:
        point_in_time_contract_sha256_str = canonical_dataframe_sha256_str(
            build_point_in_time_contract_df(signal_df, rebalance_weight_df)
        )
        expected_point_in_time_sha256_str = (
            FROZEN_ALFRED_PIT_SIGNAL_CONTRACT_SHA256_BY_VINTAGE_POLICY_DICT[
                config_obj.alfred_vintage_policy_str
            ]
        )
        if point_in_time_contract_sha256_str != expected_point_in_time_sha256_str:
            raise RuntimeError(
                "ALFRED point-in-time signal/target contract fingerprint mismatch: "
                f"expected {expected_point_in_time_sha256_str}, "
                f"found {point_in_time_contract_sha256_str}."
            )
    elif bool(signal_df["stale_input_blocked_bool"].any()):
        # The frozen files are hash-locked and measured stale-free (all 289
        # observations are T-1). A block here would mean the contract changed.
        raise AssertionError("The frozen current-vintage contract must have no stale decision.")
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


def strategy_name_for_config_str(config_obj: TacticalYieldConfig) -> str:
    """Point-in-time runs save under their own results folder."""
    if config_obj.fred_data_mode_str == FRED_DATA_MODE_FROZEN_STR:
        return STRATEGY_NAME_STR
    return f"{STRATEGY_NAME_STR}__alfred_pit_{config_obj.alfred_vintage_policy_str}"


def fred_data_mode_record_dict(
    config_obj: TacticalYieldConfig,
    signal_df: pd.DataFrame,
) -> dict[str, object]:
    """What FRED data a run used, for metadata.json and run_info.json."""
    blocked_date_list = [
        pd.Timestamp(decision_date_ts).date().isoformat()
        for decision_date_ts in signal_df.index[
            signal_df.get("stale_input_blocked_bool", pd.Series(False, index=signal_df.index))
            .astype(bool)
            .to_numpy()
        ]
    ]
    record_dict: dict[str, object] = {
        "fred_data_mode_str": config_obj.fred_data_mode_str,
        "stale_input_policy_str": config_obj.stale_input_policy_str,
        "max_observation_age_sessions_int": MAX_OBSERVATION_AGE_SESSIONS_INT,
        "stale_input_blocked_decision_count_int": len(blocked_date_list),
        "stale_input_blocked_decision_date_list": blocked_date_list,
        "cash_rate_vintage_policy_str": "frozen_current_vintage_DGS3MO_in_every_fred_data_mode",
    }
    if config_obj.fred_data_mode_str == FRED_DATA_MODE_ALFRED_PIT_STR:
        record_dict.update(
            {
                "alfred_vintage_policy_str": config_obj.alfred_vintage_policy_str,
                "alfred_first_decision_date_str": FIRST_ALFRED_DECISION_DATE_STR,
                "alfred_snapshot_dir_str": config_obj.alfred_snapshot_dir_path_str,
                "alfred_manifest_sha256_str": FROZEN_ALFRED_MANIFEST_SHA256_STR,
                "alfred_point_in_time_contract_sha256_str": (
                    FROZEN_ALFRED_PIT_SIGNAL_CONTRACT_SHA256_BY_VINTAGE_POLICY_DICT[
                        config_obj.alfred_vintage_policy_str
                    ]
                ),
                "pre_alfred_decisions_str": "frozen_current_vintage_unverifiable",
            }
        )
    return record_dict


def _record_fred_data_mode_in_run_info(
    output_path: Path,
    record_dict: dict[str, object],
) -> None:
    """Add the FRED data mode to run_info.json parameters so Bench shows it."""
    run_info_path = Path(output_path) / "run_info.json"
    run_info_dict = json.loads(run_info_path.read_text(encoding="utf-8"))
    parameter_dict = dict(run_info_dict.get("parameters") or {})
    parameter_dict.update(
        {
            key_str: value_obj
            for key_str, value_obj in record_dict.items()
            if key_str != "stale_input_blocked_decision_date_list"
        }
    )
    run_info_dict["parameters"] = parameter_dict
    run_info_path.write_text(
        json.dumps(run_info_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _attach_fred_provenance(
    strategy_obj: TacticalYieldStrategy,
    fred_snapshot_tuple: tuple[FrozenFredSnapshot, ...],
    config_obj: TacticalYieldConfig = DEFAULT_CONFIG,
) -> None:
    strategy_obj.fred_snapshot_tuple = fred_snapshot_tuple
    signal_df = getattr(strategy_obj, "month_end_signal_df", None)
    strategy_obj.fred_data_mode_record_dict = fred_data_mode_record_dict(
        config_obj,
        signal_df if isinstance(signal_df, pd.DataFrame) else pd.DataFrame(),
    )
    strategy_obj._data_adjustment_policy_dict.update(strategy_obj.fred_data_mode_record_dict)
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
        name=strategy_name_for_config_str(config_obj),
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
    _attach_fred_provenance(strategy_obj, fred_snapshot_tuple, config_obj)
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
    fred_data_mode_str: str | None = None,
    alfred_vintage_policy_str: str | None = None,
    stale_input_policy_str: str | None = None,
) -> TacticalYieldStrategy:
    """Run the frozen L14 backtest.

    ``fred_data_mode_str`` selects the FRED inputs: ``frozen_current_vintage``
    (governed default, reproduces the 289-row contract) or
    ``alfred_point_in_time`` (decisions from 2014-04-30 use only values FRED had
    published by the ALFRED vintage date). ``alfred_vintage_policy_str`` picks
    that vintage date: ``decision_date`` or the conservative
    ``previous_session``. ``stale_input_policy_str`` is ``block_and_hold`` (a
    stale decision places no order) or ``raise`` (a stale decision stops the
    run). A keyword left at None keeps the value in ``config_obj`` (defaults:
    ``frozen_current_vintage``, ``decision_date``, ``block_and_hold``). The
    mode is written to metadata.json and run_info.json.
    """
    if end_date_str is not None and pd.Timestamp(end_date_str) > pd.Timestamp(
        config_obj.end_date_str
    ):
        raise ValueError(
            "The frozen L14 PM_READY module cannot run beyond 2026-08-19."
        )
    config_obj = replace(
        config_obj,
        capital_base_float=capital_base_float,
        fred_data_mode_str=fred_data_mode_str or config_obj.fred_data_mode_str,
        alfred_vintage_policy_str=(
            alfred_vintage_policy_str or config_obj.alfred_vintage_policy_str
        ),
        stale_input_policy_str=stale_input_policy_str or config_obj.stale_input_policy_str,
    )
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
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        _record_fred_data_mode_in_run_info(
            output_path,
            strategy_obj.fred_data_mode_record_dict,
        )
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
