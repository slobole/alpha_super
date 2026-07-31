"""Compare strict-HPI data semantics with the current S&P 500 snapshot contract.

The frozen comparison crosses two independent data choices:

1. Price observation clock:
   - padded: the snapshot's Norgate ``ALLMARKETDAYS`` rows;
   - unpadded: confirmed provider-padding rows are removed from the feature and
     execution clock while Close remains available for valuation.
2. Point-in-time membership:
   - trimmed: the current exporter's five-row removal for past members;
   - exact: those five membership rows are restored.

Both requested HPI variants keep the same Close-T decision / Open-(T+1)
execution contract in every arm. This is an event-parity audit; it does not
recompute P&L or cost drag.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

DEFAULT_SNAPSHOT_ROOT_PATH = Path(r"C:\alpha\norgate_snapshots")
DEFAULT_PROFILE_STR = "norgate_eod_sp500_pit"
DEFAULT_SNAPSHOT_DATE_STR = "2026-05-18"
DEFAULT_FEATURE_START_DATE_STR = "1998-01-01"
DEFAULT_BACKTEST_START_DATE_STR = "2004-01-01"
BENCHMARK_SYMBOL_STR = "$SPX"
CAPITAL_BASE_FLOAT = 100_000.0


def _log(message_str: str) -> None:
    print(message_str, flush=True)


def _json_ready(value_obj):
    if isinstance(value_obj, dict):
        return {
            str(key_obj): _json_ready(item_obj)
            for key_obj, item_obj in value_obj.items()
        }
    if isinstance(value_obj, (list, tuple)):
        return [_json_ready(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value_obj).isoformat()
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        return None if np.isnan(value_obj) else float(value_obj)
    if isinstance(value_obj, float) and math.isnan(value_obj):
        return None
    return value_obj


def _load_manifest_dict(snapshot_dir_path_obj: Path) -> dict:
    manifest_path_obj = snapshot_dir_path_obj / "manifest.json"
    manifest_bytes = manifest_path_obj.read_bytes()
    manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
    manifest_dict["manifest_sha256_str"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    return manifest_dict


def _load_trimmed_universe_df(snapshot_dir_path_obj: Path) -> pd.DataFrame:
    universe_df = pd.read_parquet(snapshot_dir_path_obj / "universe.parquet")
    if "date" in universe_df.columns:
        universe_df = universe_df.set_index("date")
    universe_df.index = pd.to_datetime(universe_df.index).normalize()
    return universe_df.sort_index().fillna(0).astype(np.int8)


def _reconstruct_untrimmed_universe_df(
    trimmed_universe_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Invert the exporter's exact ``constituent_df.iloc[:-5]`` operation."""

    exact_universe_df = trimmed_universe_df.copy()
    restored_row_count_int = 0
    restored_symbol_count_int = 0
    short_tail_symbol_list: list[str] = []
    row_count_int = len(exact_universe_df.index)

    for symbol_str in exact_universe_df.columns.astype(str):
        membership_arr = exact_universe_df[symbol_str].to_numpy(dtype=np.int8)
        active_position_arr = np.flatnonzero(membership_arr == 1)
        if len(active_position_arr) == 0:
            continue
        last_active_position_int = int(active_position_arr[-1])
        if last_active_position_int == row_count_int - 1:
            continue

        restore_end_position_int = min(
            last_active_position_int + 5,
            row_count_int - 1,
        )
        restore_position_arr = np.arange(
            last_active_position_int + 1,
            restore_end_position_int + 1,
            dtype=int,
        )
        if len(restore_position_arr) < 5:
            short_tail_symbol_list.append(symbol_str)
        if len(restore_position_arr) == 0:
            continue

        exact_universe_df.iloc[
            restore_position_arr,
            exact_universe_df.columns.get_loc(symbol_str),
        ] = 1
        restored_row_count_int += int(len(restore_position_arr))
        restored_symbol_count_int += 1

    return exact_universe_df, {
        "restored_membership_row_count_int": restored_row_count_int,
        "restored_past_symbol_count_int": restored_symbol_count_int,
        "short_tail_symbol_list": short_tail_symbol_list,
    }


def _load_direct_exact_universe_df(
    *,
    trimmed_universe_df: pd.DataFrame,
    reconstructed_universe_df: pd.DataFrame,
    feature_start_date_str: str,
    snapshot_date_str: str,
) -> tuple[pd.DataFrame, dict]:
    """Load HPI's raw PIT membership, retaining a disclosed alias fallback."""

    import norgatedata

    exact_universe_df = reconstructed_universe_df.copy()
    direct_symbol_count_int = 0
    fallback_symbol_error_dict: dict[str, str] = {}
    pre_start_symbol_count_int = 0
    reconstruction_mismatch_symbol_list: list[str] = []
    reconstruction_mismatch_cell_count_int = 0
    analysis_mask_ser = (
        exact_universe_df.index >= pd.Timestamp(feature_start_date_str)
    )

    for symbol_index_int, symbol_str in enumerate(
        trimmed_universe_df.columns.astype(str),
        start=1,
    ):
        reconstructed_analysis_ser = exact_universe_df.loc[
            analysis_mask_ser,
            symbol_str,
        ]
        try:
            membership_df = norgatedata.index_constituent_timeseries(
                symbol_str,
                "S&P 500",
                start_date=feature_start_date_str,
                end_date=snapshot_date_str,
                timeseriesformat="pandas-dataframe",
            )
            if membership_df is None or membership_df.empty:
                if reconstructed_analysis_ser.eq(1).any():
                    raise RuntimeError("empty membership series")
                pre_start_symbol_count_int += 1
                continue
            membership_ser = (
                membership_df["Index Constituent"]
                .reindex(exact_universe_df.index)
                .fillna(0)
                .astype(np.int8)
            )
            if not membership_ser.eq(1).any():
                if reconstructed_analysis_ser.eq(1).any():
                    raise RuntimeError(
                        "membership series has no active rows"
                    )
                pre_start_symbol_count_int += 1
                continue

            reconstructed_ser = exact_universe_df.loc[
                analysis_mask_ser,
                symbol_str,
            ]
            membership_analysis_ser = membership_ser.loc[
                analysis_mask_ser
            ]
            mismatch_count_int = int(
                (
                    membership_analysis_ser.to_numpy(dtype=np.int8)
                    != reconstructed_ser.to_numpy(dtype=np.int8)
                ).sum()
            )
            if mismatch_count_int:
                reconstruction_mismatch_symbol_list.append(symbol_str)
                reconstruction_mismatch_cell_count_int += (
                    mismatch_count_int
                )
            exact_universe_df[symbol_str] = membership_ser
            direct_symbol_count_int += 1
        except Exception as exception_obj:
            fallback_symbol_error_dict[symbol_str] = repr(exception_obj)

        if symbol_index_int % 250 == 0:
            _log(
                "membership audit "
                f"{symbol_index_int}/"
                f"{len(trimmed_universe_df.columns)} symbols"
            )

    return exact_universe_df, {
        "direct_exact_membership_symbol_count_int": direct_symbol_count_int,
        "pre_feature_start_symbol_count_int": pre_start_symbol_count_int,
        "reconstruction_fallback_symbol_error_dict": (
            fallback_symbol_error_dict
        ),
        "reconstruction_mismatch_symbol_list": (
            reconstruction_mismatch_symbol_list
        ),
        "reconstruction_mismatch_cell_count_int": (
            reconstruction_mismatch_cell_count_int
        ),
    }


def _load_snapshot_price_long_df(
    snapshot_dir_path_obj: Path,
    feature_start_date_str: str,
) -> tuple[pd.DataFrame, dict[str, set[pd.Timestamp]]]:
    price_long_df = pd.read_parquet(
        snapshot_dir_path_obj / "prices.parquet",
        columns=[
            "date",
            "symbol_str",
            "adjustment_str",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Turnover",
        ],
        filters=[("date", ">=", pd.Timestamp(feature_start_date_str))],
    )
    price_long_df["date"] = pd.to_datetime(
        price_long_df["date"]
    ).dt.normalize()
    price_long_df["symbol_str"] = price_long_df["symbol_str"].astype(str)
    price_long_df["adjustment_str"] = (
        price_long_df["adjustment_str"].astype(str).str.upper()
    )

    adjustment_mask_ser = (
        price_long_df["symbol_str"].eq(BENCHMARK_SYMBOL_STR)
        & price_long_df["adjustment_str"].eq("TOTALRETURN")
    ) | (
        ~price_long_df["symbol_str"].eq(BENCHMARK_SYMBOL_STR)
        & price_long_df["adjustment_str"].eq("CAPITALSPECIAL")
    )
    price_long_df = price_long_df.loc[adjustment_mask_ser].copy()
    duplicate_count_int = int(
        price_long_df.duplicated(["date", "symbol_str"]).sum()
    )
    if duplicate_count_int:
        raise RuntimeError(
            f"Snapshot has {duplicate_count_int} duplicate date/symbol rows."
        )

    zero_volume_df = price_long_df.loc[
        ~price_long_df["symbol_str"].eq(BENCHMARK_SYMBOL_STR)
        & pd.to_numeric(
            price_long_df["Volume"],
            errors="coerce",
        ).fillna(-1.0).eq(0.0),
        ["date", "symbol_str"],
    ]
    zero_date_by_symbol_dict = {
        str(symbol_str): set(
            pd.to_datetime(symbol_df["date"]).dt.normalize()
        )
        for symbol_str, symbol_df in zero_volume_df.groupby(
            "symbol_str",
            sort=True,
        )
    }
    return (
        price_long_df.drop(columns=["Volume", "adjustment_str"]),
        zero_date_by_symbol_dict,
    )


def _audit_zero_volume_padding_dates(
    zero_date_by_symbol_dict: dict[str, set[pd.Timestamp]],
    feature_start_date_str: str,
    snapshot_date_str: str,
) -> tuple[dict[str, set[pd.Timestamp]], dict]:
    """Prove which zero-volume snapshot rows disappear under ``PaddingType.NONE``."""

    import norgatedata

    synthetic_date_by_symbol_dict: dict[str, set[pd.Timestamp]] = {}
    observed_zero_date_by_symbol_dict: dict[str, set[pd.Timestamp]] = {}
    failed_symbol_error_dict: dict[str, str] = {}

    for symbol_index_int, (
        symbol_str,
        zero_date_set,
    ) in enumerate(sorted(zero_date_by_symbol_dict.items()), start=1):
        try:
            none_price_df = norgatedata.price_timeseries(
                symbol_str,
                stock_price_adjustment_setting=(
                    norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL
                ),
                padding_setting=norgatedata.PaddingType.NONE,
                start_date=feature_start_date_str,
                end_date=snapshot_date_str,
                timeseriesformat="pandas-dataframe",
            )
            observed_date_set = (
                set(pd.to_datetime(none_price_df.index).normalize())
                if none_price_df is not None and not none_price_df.empty
                else set()
            )
            synthetic_date_by_symbol_dict[symbol_str] = (
                zero_date_set.difference(observed_date_set)
            )
            observed_zero_date_set = zero_date_set.intersection(
                observed_date_set
            )
            if observed_zero_date_set:
                observed_zero_date_by_symbol_dict[symbol_str] = (
                    observed_zero_date_set
                )
        except Exception as exception_obj:
            failed_symbol_error_dict[symbol_str] = repr(exception_obj)

        if symbol_index_int % 25 == 0:
            _log(
                "padding audit "
                f"{symbol_index_int}/{len(zero_date_by_symbol_dict)} symbols"
            )

    audit_dict = {
        "zero_volume_symbol_count_int": len(zero_date_by_symbol_dict),
        "zero_volume_row_count_int": int(
            sum(len(date_set) for date_set in zero_date_by_symbol_dict.values())
        ),
        "confirmed_padding_symbol_count_int": int(
            sum(
                bool(date_set)
                for date_set in synthetic_date_by_symbol_dict.values()
            )
        ),
        "confirmed_padding_row_count_int": int(
            sum(
                len(date_set)
                for date_set in synthetic_date_by_symbol_dict.values()
            )
        ),
        "observed_zero_volume_symbol_count_int": len(
            observed_zero_date_by_symbol_dict
        ),
        "observed_zero_volume_row_count_int": int(
            sum(
                len(date_set)
                for date_set in observed_zero_date_by_symbol_dict.values()
            )
        ),
        "failed_symbol_error_dict": failed_symbol_error_dict,
        "failed_symbol_zero_volume_row_count_int": int(
            sum(
                len(zero_date_by_symbol_dict[symbol_str])
                for symbol_str in failed_symbol_error_dict
            )
        ),
    }
    return synthetic_date_by_symbol_dict, audit_dict


def _build_pricing_data_df(price_long_df: pd.DataFrame) -> pd.DataFrame:
    price_wide_df = price_long_df.set_index(
        ["date", "symbol_str"]
    )[["Open", "High", "Low", "Close", "Turnover"]].unstack("symbol_str")
    price_wide_df.columns = price_wide_df.columns.swaplevel(0, 1)
    price_wide_df = price_wide_df.sort_index(axis=1).sort_index()
    price_wide_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        str(symbol_str): (
            "TOTALRETURN"
            if str(symbol_str) == BENCHMARK_SYMBOL_STR
            else "CAPITALSPECIAL"
        )
        for symbol_str in price_wide_df.columns.get_level_values(0).unique()
    }
    return price_wide_df


def _apply_unpadding_in_place(
    pricing_data_df: pd.DataFrame,
    synthetic_date_by_symbol_dict: dict[str, set[pd.Timestamp]],
) -> dict:
    """Match HPI's no-padding union calendar without changing valuation Close."""

    applied_row_count_int = 0
    missing_pair_count_int = 0
    available_symbol_set = set(
        pricing_data_df.columns.get_level_values(0).astype(str)
    )

    for symbol_str, synthetic_date_set in synthetic_date_by_symbol_dict.items():
        if not synthetic_date_set or symbol_str not in available_symbol_set:
            continue
        available_date_idx = pricing_data_df.index.intersection(
            pd.DatetimeIndex(sorted(synthetic_date_set))
        )
        missing_pair_count_int += int(
            len(synthetic_date_set) - len(available_date_idx)
        )
        if len(available_date_idx) == 0:
            continue

        # *** CRITICAL*** These rows were proven absent under PaddingType.NONE.
        # HPI's union calendar retains a valuation-only forward-filled Close,
        # while Open/High/Low remain NaN. Clearing Turnover matches the absent
        # source observation and prevents ranking a synthetic session.
        for field_str in ("Open", "High", "Low", "Turnover"):
            column_tuple = (symbol_str, field_str)
            if column_tuple in pricing_data_df.columns:
                pricing_data_df.loc[available_date_idx, column_tuple] = np.nan
        applied_row_count_int += int(len(available_date_idx))

    return {
        "unpadding_applied_row_count_int": applied_row_count_int,
        "unpadding_missing_pair_count_int": missing_pair_count_int,
        "close_policy_str": (
            "retain padded Close as valuation-only forward fill; "
            "clear Open/High/Low/Turnover"
        ),
    }


def _load_hpi_runtime():
    from strategies.hpi.stateful_long import (
        ENTRY_BASELINE_STR,
        ENTRY_HORIZON_VOTE_STR,
        EXIT_IBS_THRESHOLD_FLOAT,
        EXIT_RSI2_THRESHOLD_FLOAT,
        HPI_THRESHOLD_FLOAT,
        HPIStatefulLongStrategy,
        MAX_ENTRY_IBS_FLOAT,
        MAX_POSITIONS_INT,
        TURNOVER_FIELD_STR,
    )

    return {
        "entry_baseline_str": ENTRY_BASELINE_STR,
        "entry_vote_str": ENTRY_HORIZON_VOTE_STR,
        "exit_ibs_threshold_float": EXIT_IBS_THRESHOLD_FLOAT,
        "exit_rsi2_threshold_float": EXIT_RSI2_THRESHOLD_FLOAT,
        "hpi_threshold_float": HPI_THRESHOLD_FLOAT,
        "max_entry_ibs_float": MAX_ENTRY_IBS_FLOAT,
        "max_positions_int": MAX_POSITIONS_INT,
        "strategy_class": HPIStatefulLongStrategy,
        "turnover_field_str": TURNOVER_FIELD_STR,
    }


def _precompute_vote_signal_df(
    pricing_data_df: pd.DataFrame,
    label_str: str,
    runtime_dict: dict,
    backtest_start_date_str: str,
) -> pd.DataFrame:
    _log(f"{label_str}: signal precompute start")
    feature_strategy_obj = runtime_dict["strategy_class"](
        name=f"{label_str}_feature_builder",
        benchmarks=[BENCHMARK_SYMBOL_STR],
        ranking_field_str=runtime_dict["turnover_field_str"],
        capital_base=CAPITAL_BASE_FLOAT,
        entry_mode_str=runtime_dict["entry_vote_str"],
        backtest_start_date_str=backtest_start_date_str,
    )
    signal_df = feature_strategy_obj.compute_signals(pricing_data_df)
    _log(
        f"{label_str}: signal precompute done "
        f"rows={len(signal_df)} cols={len(signal_df.columns)}"
    )
    return signal_df


def _extract_affected_feature_df(
    signal_df: pd.DataFrame,
    affected_symbol_set: set[str],
) -> pd.DataFrame:
    feature_field_set = {
        "return_2d_ser",
        "return_3d_ser",
        "return_5d_ser",
        "hpi_2d_ser",
        "hpi_value_ser",
        "hpi_5d_ser",
        "ibs_value_ser",
        "rsi2_value_ser",
        "sma_200_price_ser",
    }
    column_list = [
        column_tuple
        for column_tuple in signal_df.columns
        if str(column_tuple[0]) in affected_symbol_set
        and str(column_tuple[1]) in feature_field_set
    ]
    return signal_df[column_list].copy()


def _field_wide_df(
    signal_df: pd.DataFrame,
    symbol_list: list[str],
    field_str: str,
) -> pd.DataFrame:
    column_list = [
        (symbol_str, field_str)
        for symbol_str in symbol_list
        if (symbol_str, field_str) in signal_df.columns
    ]
    field_df = signal_df[column_list].copy()
    field_df.columns = [
        str(column_tuple[0])
        for column_tuple in field_df.columns
    ]
    return field_df.reindex(columns=symbol_list)


def _build_candidate_map_dict(
    *,
    signal_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    decision_date_idx: pd.DatetimeIndex,
    symbol_list: list[str],
    entry_mode_str: str,
    runtime_dict: dict,
) -> dict[str, list[str]]:
    close_df = _field_wide_df(signal_df, symbol_list, "Close")
    turnover_df = _field_wide_df(
        signal_df,
        symbol_list,
        runtime_dict["turnover_field_str"],
    )
    sma_df = _field_wide_df(
        signal_df,
        symbol_list,
        "sma_200_price_ser",
    )
    ibs_df = _field_wide_df(
        signal_df,
        symbol_list,
        "ibs_value_ser",
    )
    member_bool_df = (
        universe_df.reindex(index=signal_df.index, columns=symbol_list)
        .fillna(0)
        .eq(1)
    )
    eligible_bool_df = (
        member_bool_df
        & ibs_df.lt(runtime_dict["max_entry_ibs_float"])
        & close_df.gt(sma_df)
        & turnover_df.notna()
    )

    if entry_mode_str == runtime_dict["entry_vote_str"]:
        vote_count_df = pd.DataFrame(
            0,
            index=signal_df.index,
            columns=symbol_list,
            dtype=np.int8,
        )
        for return_field_str, hpi_field_str in (
            ("return_2d_ser", "hpi_2d_ser"),
            ("return_3d_ser", "hpi_value_ser"),
            ("return_5d_ser", "hpi_5d_ser"),
        ):
            return_df = _field_wide_df(
                signal_df,
                symbol_list,
                return_field_str,
            )
            hpi_df = _field_wide_df(
                signal_df,
                symbol_list,
                hpi_field_str,
            )
            vote_count_df += (
                return_df.lt(0.0)
                & hpi_df.lt(runtime_dict["hpi_threshold_float"])
            ).astype(np.int8)
        eligible_bool_df &= vote_count_df.ge(2)
    else:
        return_3d_df = _field_wide_df(
            signal_df,
            symbol_list,
            "return_3d_ser",
        )
        hpi_df = _field_wide_df(
            signal_df,
            symbol_list,
            "hpi_value_ser",
        )
        eligible_bool_df &= (
            return_3d_df.lt(0.0)
            & hpi_df.lt(runtime_dict["hpi_threshold_float"])
        )

    # *** CRITICAL*** Candidate rows are restricted to actual decision closes.
    # Every resulting list is consumed only for the following session's open.
    decision_eligible_bool_df = eligible_bool_df.reindex(decision_date_idx)
    decision_turnover_df = turnover_df.reindex(decision_date_idx)
    candidate_turnover_ser = decision_turnover_df.where(
        decision_eligible_bool_df
    ).stack(future_stack=True).dropna()
    if candidate_turnover_ser.empty:
        return {}

    candidate_df = candidate_turnover_ser.rename(
        "turnover_float"
    ).reset_index()
    candidate_df.columns = [
        "decision_date_ts",
        "symbol_str",
        "turnover_float",
    ]
    candidate_df["symbol_str"] = candidate_df["symbol_str"].astype(str)
    candidate_df = candidate_df.sort_values(
        by=["decision_date_ts", "turnover_float", "symbol_str"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return {
        pd.Timestamp(decision_date_ts).date().isoformat(): (
            decision_candidate_df["symbol_str"].astype(str).tolist()
        )
        for decision_date_ts, decision_candidate_df in candidate_df.groupby(
            "decision_date_ts",
            sort=True,
        )
    }


def _run_arm_variant(
    *,
    arm_str: str,
    variant_str: str,
    entry_mode_str: str,
    pricing_data_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    runtime_dict: dict,
    backtest_start_date_str: str,
    snapshot_date_str: str,
) -> dict:
    calendar_idx = pricing_data_df.index[
        (pricing_data_df.index >= pd.Timestamp(backtest_start_date_str))
        & (pricing_data_df.index <= pd.Timestamp(snapshot_date_str))
    ]
    first_calendar_position_int = int(
        pricing_data_df.index.get_loc(calendar_idx[0])
    )
    if first_calendar_position_int == 0:
        raise RuntimeError(
            "Event replay requires one pre-start decision session."
        )
    decision_date_idx = pricing_data_df.index[
        first_calendar_position_int - 1:
        first_calendar_position_int - 1 + len(calendar_idx)
    ]
    symbol_list = [
        str(symbol_str)
        for symbol_str in universe_df.columns
        if not str(symbol_str).startswith("$")
        and (str(symbol_str), "Open") in pricing_data_df.columns
    ]
    _log(
        f"event replay {arm_str}/{variant_str} start "
        f"sessions={len(calendar_idx)}"
    )
    candidate_map_dict = _build_candidate_map_dict(
        signal_df=signal_df,
        universe_df=universe_df,
        decision_date_idx=decision_date_idx,
        symbol_list=symbol_list,
        entry_mode_str=entry_mode_str,
        runtime_dict=runtime_dict,
    )
    open_df = _field_wide_df(
        pricing_data_df,
        symbol_list,
        "Open",
    )
    close_df = _field_wide_df(
        pricing_data_df,
        symbol_list,
        "Close",
    )
    ibs_df = _field_wide_df(
        signal_df,
        symbol_list,
        "ibs_value_ser",
    )
    rsi_df = _field_wide_df(
        signal_df,
        symbol_list,
        "rsi2_value_ser",
    )
    aligned_universe_df = (
        universe_df.reindex(index=signal_df.index, columns=symbol_list)
        .fillna(0)
        .astype(np.int8)
    )

    holding_symbol_set: set[str] = set()
    pending_exit_symbol_set: set[str] = set()
    transaction_row_list = []
    for decision_date_ts, execution_date_ts in zip(
        decision_date_idx,
        calendar_idx,
        strict=True,
    ):
        decision_date_str = pd.Timestamp(
            decision_date_ts
        ).date().isoformat()
        execution_date_str = pd.Timestamp(
            execution_date_ts
        ).date().isoformat()

        pending_exit_symbol_set.intersection_update(holding_symbol_set)
        normal_exit_symbol_set: set[str] = set()
        for symbol_str in sorted(holding_symbol_set):
            ibs_value_float = ibs_df.at[decision_date_ts, symbol_str]
            rsi_value_float = rsi_df.at[decision_date_ts, symbol_str]
            member_bool = bool(
                aligned_universe_df.at[decision_date_ts, symbol_str] == 1
            )
            exit_bool = (
                (
                    pd.notna(ibs_value_float)
                    and float(ibs_value_float)
                    > runtime_dict["exit_ibs_threshold_float"]
                )
                or (
                    pd.notna(rsi_value_float)
                    and float(rsi_value_float)
                    > runtime_dict["exit_rsi2_threshold_float"]
                )
                or not member_bool
            )
            execution_open_float = open_df.at[
                execution_date_ts,
                symbol_str,
            ]
            if exit_bool:
                pending_exit_symbol_set.add(symbol_str)
            if (
                symbol_str in pending_exit_symbol_set
                and np.isfinite(execution_open_float)
            ):
                normal_exit_symbol_set.add(symbol_str)

        available_slot_count_int = (
            runtime_dict["max_positions_int"]
            - len(holding_symbol_set)
        )
        selected_entry_symbol_list: list[str] = []
        for symbol_str in candidate_map_dict.get(
            decision_date_str,
            [],
        ):
            if available_slot_count_int == 0:
                break
            if symbol_str in holding_symbol_set:
                continue
            selected_entry_symbol_list.append(symbol_str)
            available_slot_count_int -= 1

        # The engine performs this fallback after iterate() allocates entry
        # slots but before it executes those orders. A forced liquidation
        # therefore cannot fund another entry on the same open.
        # *** CRITICAL*** The fallback price is capped at decision_date_ts;
        # no execution-session close or future close may enter this replay.
        execution_member_ser = aligned_universe_df.loc[execution_date_ts]
        forced_exit_symbol_set: set[str] = set()
        forced_exit_price_dict: dict[str, float] = {}
        for symbol_str in sorted(
            holding_symbol_set - normal_exit_symbol_set
        ):
            execution_open_float = open_df.at[
                execution_date_ts,
                symbol_str,
            ]
            execution_member_bool = bool(
                execution_member_ser.at[symbol_str] == 1
            )
            if execution_member_bool or np.isfinite(execution_open_float):
                continue
            prior_close_ser = close_df.loc[
                :decision_date_ts,
                symbol_str,
            ].dropna()
            if prior_close_ser.empty:
                raise RuntimeError(
                    f"No prior close available for forced liquidation of "
                    f"{symbol_str} on {execution_date_str}."
                )
            forced_exit_symbol_set.add(symbol_str)
            forced_exit_price_dict[symbol_str] = float(
                prior_close_ser.iloc[-1]
            )

        for symbol_str in sorted(normal_exit_symbol_set):
            execution_open_float = open_df.at[
                execution_date_ts,
                symbol_str,
            ]
            holding_symbol_set.remove(symbol_str)
            pending_exit_symbol_set.discard(symbol_str)
            transaction_row_list.append(
                {
                    "bar_date_str": execution_date_str,
                    "asset_str": symbol_str,
                    "direction_str": "sell",
                    "amount_float": -1.0,
                    "price_float": float(execution_open_float),
                    "commission_float": None,
                }
            )

        for symbol_str in sorted(forced_exit_symbol_set):
            holding_symbol_set.remove(symbol_str)
            pending_exit_symbol_set.discard(symbol_str)
            transaction_row_list.append(
                {
                    "bar_date_str": execution_date_str,
                    "asset_str": symbol_str,
                    "direction_str": "sell",
                    "amount_float": -1.0,
                    "price_float": forced_exit_price_dict[symbol_str],
                    "commission_float": None,
                }
            )

        for symbol_str in selected_entry_symbol_list:
            execution_open_float = open_df.at[
                execution_date_ts,
                symbol_str,
            ]
            if not np.isfinite(execution_open_float):
                continue
            holding_symbol_set.add(symbol_str)
            transaction_row_list.append(
                {
                    "bar_date_str": execution_date_str,
                    "asset_str": symbol_str,
                    "direction_str": "buy",
                    "amount_float": 1.0,
                    "price_float": float(execution_open_float),
                    "commission_float": None,
                }
            )

    session_count_int = int(len(calendar_idx))
    result_dict = {
        "arm_str": arm_str,
        "variant_str": variant_str,
        "entry_mode_str": entry_mode_str,
        "start_date_str": pd.Timestamp(calendar_idx[0]).date().isoformat(),
        "end_date_str": pd.Timestamp(calendar_idx[-1]).date().isoformat(),
        "session_count_int": session_count_int,
        "final_total_value_float": None,
        "cagr_float": None,
        "annualized_volatility_float": None,
        "sharpe_float": None,
        "max_drawdown_float": None,
        "transaction_count_int": int(len(transaction_row_list)),
        "candidate_date_count_int": int(len(candidate_map_dict)),
        "candidate_symbol_count_int": int(
            sum(
                len(symbol_list)
                for symbol_list in candidate_map_dict.values()
            )
        ),
        "transaction_row_list": transaction_row_list,
        "candidate_map_dict": candidate_map_dict,
        "total_value_by_date_dict": {},
    }
    _log(
        f"event replay {arm_str}/{variant_str} done "
        f"events={len(transaction_row_list)}"
    )
    return result_dict


def _compare_candidate_maps(
    left_result_dict: dict,
    right_result_dict: dict,
) -> dict:
    left_map_dict = left_result_dict["candidate_map_dict"]
    right_map_dict = right_result_dict["candidate_map_dict"]
    date_list = sorted(set(left_map_dict).union(right_map_dict))
    mismatch_date_list = [
        date_str
        for date_str in date_list
        if left_map_dict.get(date_str, [])
        != right_map_dict.get(date_str, [])
    ]
    mismatch_example_list = []
    for date_str in mismatch_date_list[:10]:
        left_list = left_map_dict.get(date_str, [])
        right_list = right_map_dict.get(date_str, [])
        mismatch_example_list.append(
            {
                "date_str": date_str,
                "left_candidate_list": left_list[:20],
                "right_candidate_list": right_list[:20],
                "left_only_symbol_list": sorted(
                    set(left_list).difference(right_list)
                )[:20],
                "right_only_symbol_list": sorted(
                    set(right_list).difference(left_list)
                )[:20],
            }
        )
    return {
        "candidate_mismatch_date_count_int": len(mismatch_date_list),
        "first_candidate_mismatch_date_str": (
            mismatch_date_list[0] if mismatch_date_list else None
        ),
        "candidate_mismatch_examples": mismatch_example_list,
    }


def _transaction_event_counter(result_dict: dict) -> Counter:
    return Counter(
        (
            row_dict["bar_date_str"],
            row_dict["asset_str"],
            row_dict["direction_str"],
        )
        for row_dict in result_dict["transaction_row_list"]
    )


def _compare_transactions(
    left_result_dict: dict,
    right_result_dict: dict,
) -> dict:
    left_counter = _transaction_event_counter(left_result_dict)
    right_counter = _transaction_event_counter(right_result_dict)
    left_only_counter = left_counter - right_counter
    right_only_counter = right_counter - left_counter
    differing_event_list = sorted(
        set(left_only_counter).union(right_only_counter)
    )
    mismatch_example_list = [
        {
            "bar_date_str": event_tuple[0],
            "asset_str": event_tuple[1],
            "direction_str": event_tuple[2],
            "left_count_int": int(left_counter[event_tuple]),
            "right_count_int": int(right_counter[event_tuple]),
        }
        for event_tuple in differing_event_list[:20]
    ]

    return {
        "directional_event_symmetric_difference_count_int": int(
            sum(left_only_counter.values())
            + sum(right_only_counter.values())
        ),
        "left_only_event_count_int": int(sum(left_only_counter.values())),
        "right_only_event_count_int": int(sum(right_only_counter.values())),
        "first_transaction_mismatch_date_str": (
            differing_event_list[0][0] if differing_event_list else None
        ),
        "transaction_mismatch_examples": mismatch_example_list,
        "final_total_value_difference_float": None,
        "max_absolute_equity_relative_difference_float": None,
    }


def _compare_result_pair(
    left_result_dict: dict,
    right_result_dict: dict,
    comparison_str: str,
) -> dict:
    comparison_dict = {
        "comparison_str": comparison_str,
        "left_arm_str": left_result_dict["arm_str"],
        "right_arm_str": right_result_dict["arm_str"],
        "variant_str": left_result_dict["variant_str"],
    }
    comparison_dict.update(
        _compare_candidate_maps(left_result_dict, right_result_dict)
    )
    comparison_dict.update(
        _compare_transactions(left_result_dict, right_result_dict)
    )
    return comparison_dict


def _compare_feature_frames(
    padded_feature_df: pd.DataFrame,
    unpadded_feature_df: pd.DataFrame,
    exact_universe_df: pd.DataFrame,
) -> list[dict]:
    row_dict_list: list[dict] = []
    common_column_idx = padded_feature_df.columns.intersection(
        unpadded_feature_df.columns
    )
    for column_tuple in common_column_idx:
        symbol_str = str(column_tuple[0])
        field_str = str(column_tuple[1])
        padded_ser = pd.to_numeric(
            padded_feature_df[column_tuple],
            errors="coerce",
        )
        unpadded_ser = pd.to_numeric(
            unpadded_feature_df[column_tuple],
            errors="coerce",
        )
        padded_finite_ser = np.isfinite(padded_ser)
        unpadded_finite_ser = np.isfinite(unpadded_ser)
        common_finite_ser = padded_finite_ser & unpadded_finite_ser
        finite_status_mismatch_ser = (
            padded_finite_ser != unpadded_finite_ser
        )
        value_mismatch_ser = pd.Series(False, index=padded_ser.index)
        value_mismatch_ser.loc[common_finite_ser] = ~np.isclose(
            padded_ser.loc[common_finite_ser].to_numpy(dtype=float),
            unpadded_ser.loc[common_finite_ser].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        any_mismatch_ser = (
            finite_status_mismatch_ser | value_mismatch_ser
        )
        member_ser = (
            exact_universe_df[symbol_str]
            .reindex(padded_ser.index)
            .fillna(0)
            .eq(1)
            if symbol_str in exact_universe_df.columns
            else pd.Series(False, index=padded_ser.index)
        )
        absolute_difference_ser = (
            padded_ser - unpadded_ser
        ).abs().where(common_finite_ser)
        row_dict_list.append(
            {
                "symbol_str": symbol_str,
                "field_str": field_str,
                "finite_status_mismatch_count_int": int(
                    finite_status_mismatch_ser.sum()
                ),
                "value_mismatch_count_int": int(
                    value_mismatch_ser.sum()
                ),
                "member_date_mismatch_count_int": int(
                    (any_mismatch_ser & member_ser).sum()
                ),
                "max_absolute_difference_float": (
                    float(absolute_difference_ser.max())
                    if absolute_difference_ser.notna().any()
                    else 0.0
                ),
            }
        )
    return row_dict_list


def _write_artifacts(
    output_dir_path_obj: Path,
    full_result_dict: dict,
) -> None:
    output_dir_path_obj.mkdir(parents=True, exist_ok=True)
    arm_metric_row_dict_list = full_result_dict[
        "arm_metric_row_dict_list"
    ]
    comparison_row_dict_list = full_result_dict[
        "comparison_row_dict_list"
    ]
    feature_diff_row_dict_list = full_result_dict[
        "feature_diff_row_dict_list"
    ]

    (output_dir_path_obj / "parity_summary.json").write_text(
        json.dumps(
            _json_ready(full_result_dict),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(arm_metric_row_dict_list).to_csv(
        output_dir_path_obj / "arm_metrics.csv",
        index=False,
    )
    pd.DataFrame(comparison_row_dict_list).drop(
        columns=[
            "candidate_mismatch_examples",
            "transaction_mismatch_examples",
        ]
    ).to_csv(
        output_dir_path_obj / "comparison_summary.csv",
        index=False,
    )
    pd.DataFrame(feature_diff_row_dict_list).to_csv(
        output_dir_path_obj / "feature_diff.csv",
        index=False,
    )

    specification_dict = full_result_dict["specification_dict"]
    padding_audit_dict = full_result_dict["padding_audit_dict"]
    membership_audit_dict = full_result_dict["membership_audit_dict"]
    feature_summary_dict = full_result_dict["feature_summary_dict"]
    report_line_list = [
        "# HPI snapshot parity test",
        "",
        (
            f"Snapshot: `{specification_dict['snapshot_profile_str']}` "
            f"on `{specification_dict['snapshot_date_str']}`."
        ),
        (
            f"Backtest: `{specification_dict['backtest_start_date_str']}` "
            f"through `{specification_dict['snapshot_date_str']}`; "
            f"capital `${CAPITAL_BASE_FLOAT:,.0f}`."
        ),
        "Signal rules and Close-T/Open-T+1 timing were unchanged.",
        (
            "This is an event-parity replay: it reproduces candidate ordering, "
            "persistent pending exits, PIT-removal liquidations, ten-slot "
            "state, and next-open buy/sell events. It does not recompute cash, "
            "commission, performance, or cost drag."
        ),
        "",
        "## Data audit",
        "",
        (
            "- Confirmed padded rows: "
            f"{padding_audit_dict['confirmed_padding_row_count_int']:,} "
            "across "
            f"{padding_audit_dict['confirmed_padding_symbol_count_int']:,} "
            "symbols."
        ),
        (
            "- Restored PIT membership rows: "
            f"{membership_audit_dict['trimmed_vs_exact_membership_cell_difference_count_int']:,} "
            "across "
            f"{membership_audit_dict['restored_past_symbol_count_int']:,} "
            "past members."
        ),
        (
            "- Member-date feature mismatches: "
            f"{feature_summary_dict['total_member_date_feature_mismatch_count_int']:,} "
            "across "
            f"{feature_summary_dict['symbols_with_member_date_feature_mismatch_int']:,} "
            "symbols."
        ),
        "",
        "## Comparison summary",
        "",
        (
            "| Comparison | Variant | Candidate mismatch dates | "
            "Directional event symmetric diff | First order mismatch | "
            "Final value difference |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row_dict in comparison_row_dict_list:
        first_mismatch_str = (
            row_dict["first_transaction_mismatch_date_str"] or "none"
        )
        report_line_list.append(
            f"| {row_dict['comparison_str']} "
            f"| {row_dict['variant_str']} "
            f"| {row_dict['candidate_mismatch_date_count_int']} "
            f"| {row_dict['directional_event_symmetric_difference_count_int']} "
            f"| {first_mismatch_str} "
            "| n/a |"
        )
    (output_dir_path_obj / "report.md").write_text(
        "\n".join(report_line_list) + "\n",
        encoding="utf-8",
    )


def run_parity_test(
    *,
    snapshot_root_path_obj: Path,
    profile_str: str,
    snapshot_date_str: str,
    feature_start_date_str: str,
    backtest_start_date_str: str,
    output_dir_path_obj: Path,
) -> dict:
    snapshot_dir_path_obj = (
        snapshot_root_path_obj / profile_str / snapshot_date_str
    )
    runtime_dict = _load_hpi_runtime()
    manifest_dict = _load_manifest_dict(snapshot_dir_path_obj)

    _log("load snapshot universe")
    trimmed_universe_full_df = _load_trimmed_universe_df(
        snapshot_dir_path_obj
    )
    (
        reconstructed_universe_full_df,
        membership_reconstruction_audit_dict,
    ) = _reconstruct_untrimmed_universe_df(trimmed_universe_full_df)
    (
        exact_universe_full_df,
        direct_membership_audit_dict,
    ) = _load_direct_exact_universe_df(
        trimmed_universe_df=trimmed_universe_full_df,
        reconstructed_universe_df=reconstructed_universe_full_df,
        feature_start_date_str=feature_start_date_str,
        snapshot_date_str=snapshot_date_str,
    )
    analysis_index_mask_ser = (
        trimmed_universe_full_df.index
        >= pd.Timestamp(feature_start_date_str)
    )
    trimmed_universe_df = trimmed_universe_full_df.loc[
        analysis_index_mask_ser
    ].copy()
    exact_universe_df = exact_universe_full_df.loc[
        analysis_index_mask_ser
    ].copy()
    membership_difference_bool_df = (
        trimmed_universe_df != exact_universe_df
    )
    membership_difference_count_int = int(
        membership_difference_bool_df.to_numpy().sum()
    )
    membership_difference_date_count_int = int(
        membership_difference_bool_df.any(axis=1).sum()
    )
    _log(
        "membership restoration "
        f"rows={membership_difference_count_int} "
        f"dates={membership_difference_date_count_int}"
    )

    _log("load snapshot price rows")
    (
        price_long_df,
        zero_date_by_symbol_dict,
    ) = _load_snapshot_price_long_df(
        snapshot_dir_path_obj,
        feature_start_date_str,
    )
    _log(
        f"price rows={len(price_long_df)} "
        f"zero-volume symbols={len(zero_date_by_symbol_dict)}"
    )
    (
        synthetic_date_by_symbol_dict,
        padding_audit_dict,
    ) = _audit_zero_volume_padding_dates(
        zero_date_by_symbol_dict,
        feature_start_date_str,
        snapshot_date_str,
    )
    unresolved_future_member_symbol_list: list[str] = []
    for symbol_str in padding_audit_dict["failed_symbol_error_dict"]:
        if symbol_str not in exact_universe_df.columns:
            continue
        unresolved_date_set = zero_date_by_symbol_dict[symbol_str]
        if not unresolved_date_set:
            continue
        first_unresolved_date_ts = min(unresolved_date_set)
        future_membership_ser = exact_universe_df.loc[
            exact_universe_df.index >= first_unresolved_date_ts,
            symbol_str,
        ]
        if future_membership_ser.eq(1).any():
            unresolved_future_member_symbol_list.append(symbol_str)
    padding_audit_dict["failed_symbol_future_member_symbol_list"] = (
        unresolved_future_member_symbol_list
    )
    if unresolved_future_member_symbol_list:
        raise RuntimeError(
            "Unresolved zero-volume rows could affect a future PIT-member "
            f"feature clock: {unresolved_future_member_symbol_list}"
        )
    _log(
        "confirmed padded rows="
        f"{padding_audit_dict['confirmed_padding_row_count_int']}"
    )

    _log("pivot snapshot prices")
    pricing_data_df = _build_pricing_data_df(price_long_df)
    del price_long_df
    gc.collect()
    common_calendar_idx = pricing_data_df.index
    trimmed_universe_df = (
        trimmed_universe_df.reindex(common_calendar_idx)
        .fillna(0)
        .astype(np.int8)
    )
    exact_universe_df = (
        exact_universe_df.reindex(common_calendar_idx)
        .fillna(0)
        .astype(np.int8)
    )
    _log(
        f"pricing rows={len(pricing_data_df)} "
        f"columns={len(pricing_data_df.columns)}"
    )

    padded_signal_df = _precompute_vote_signal_df(
        pricing_data_df,
        "padded",
        runtime_dict,
        backtest_start_date_str,
    )
    affected_symbol_set = {
        symbol_str
        for symbol_str, date_set in synthetic_date_by_symbol_dict.items()
        if date_set
    }
    padded_affected_feature_df = _extract_affected_feature_df(
        padded_signal_df,
        affected_symbol_set,
    )

    variant_entry_mode_dict = {
        "ibs_rsi_exit": runtime_dict["entry_baseline_str"],
        "hpi_2_3_5_vote": runtime_dict["entry_vote_str"],
    }
    arm_result_dict: dict[str, dict[str, dict]] = {}
    for arm_str, universe_df in {
        "padded_trimmed": trimmed_universe_df,
        "padded_exact": exact_universe_df,
    }.items():
        arm_result_dict[arm_str] = {}
        for variant_str, entry_mode_str in (
            variant_entry_mode_dict.items()
        ):
            arm_result_dict[arm_str][variant_str] = _run_arm_variant(
                arm_str=arm_str,
                variant_str=variant_str,
                entry_mode_str=entry_mode_str,
                pricing_data_df=pricing_data_df,
                signal_df=padded_signal_df,
                universe_df=universe_df,
                runtime_dict=runtime_dict,
                backtest_start_date_str=backtest_start_date_str,
                snapshot_date_str=snapshot_date_str,
            )
    del padded_signal_df
    gc.collect()

    unpadding_audit_dict = _apply_unpadding_in_place(
        pricing_data_df,
        synthetic_date_by_symbol_dict,
    )
    _log(
        "unpadding rows applied="
        f"{unpadding_audit_dict['unpadding_applied_row_count_int']}"
    )
    unpadded_signal_df = _precompute_vote_signal_df(
        pricing_data_df,
        "unpadded",
        runtime_dict,
        backtest_start_date_str,
    )
    unpadded_affected_feature_df = _extract_affected_feature_df(
        unpadded_signal_df,
        affected_symbol_set,
    )
    feature_diff_row_dict_list = _compare_feature_frames(
        padded_affected_feature_df,
        unpadded_affected_feature_df,
        exact_universe_df,
    )
    del padded_affected_feature_df, unpadded_affected_feature_df
    gc.collect()

    for arm_str, universe_df in {
        "unpadded_trimmed": trimmed_universe_df,
        "unpadded_exact": exact_universe_df,
    }.items():
        arm_result_dict[arm_str] = {}
        for variant_str, entry_mode_str in (
            variant_entry_mode_dict.items()
        ):
            arm_result_dict[arm_str][variant_str] = _run_arm_variant(
                arm_str=arm_str,
                variant_str=variant_str,
                entry_mode_str=entry_mode_str,
                pricing_data_df=pricing_data_df,
                signal_df=unpadded_signal_df,
                universe_df=universe_df,
                runtime_dict=runtime_dict,
                backtest_start_date_str=backtest_start_date_str,
                snapshot_date_str=snapshot_date_str,
            )

    comparison_row_dict_list: list[dict] = []
    for variant_str in variant_entry_mode_dict:
        strict_result_dict = arm_result_dict[
            "unpadded_exact"
        ][variant_str]
        comparison_row_dict_list.extend(
            [
                _compare_result_pair(
                    arm_result_dict["padded_trimmed"][variant_str],
                    strict_result_dict,
                    "combined_current_snapshot_vs_strict_hpi",
                ),
                _compare_result_pair(
                    arm_result_dict["padded_exact"][variant_str],
                    strict_result_dict,
                    "padding_only",
                ),
                _compare_result_pair(
                    arm_result_dict["unpadded_trimmed"][variant_str],
                    strict_result_dict,
                    "membership_trim_only",
                ),
            ]
        )

    arm_metric_row_dict_list = []
    for variant_result_dict in arm_result_dict.values():
        for result_dict in variant_result_dict.values():
            arm_metric_row_dict_list.append(
                {
                    key_str: value_obj
                    for key_str, value_obj in result_dict.items()
                    if key_str
                    not in {
                        "transaction_row_list",
                        "candidate_map_dict",
                        "total_value_by_date_dict",
                    }
                }
            )

    feature_diff_df = pd.DataFrame(feature_diff_row_dict_list)
    feature_summary_dict = {
        "affected_symbol_count_int": len(affected_symbol_set),
        "feature_column_count_int": int(len(feature_diff_df.index)),
        "feature_columns_with_any_mismatch_int": int(
            (
                feature_diff_df[
                    "finite_status_mismatch_count_int"
                ].gt(0)
                | feature_diff_df["value_mismatch_count_int"].gt(0)
            ).sum()
        )
        if not feature_diff_df.empty
        else 0,
        "total_member_date_feature_mismatch_count_int": int(
            feature_diff_df["member_date_mismatch_count_int"].sum()
        )
        if not feature_diff_df.empty
        else 0,
        "symbols_with_member_date_feature_mismatch_int": int(
            feature_diff_df.loc[
                feature_diff_df[
                    "member_date_mismatch_count_int"
                ].gt(0),
                "symbol_str",
            ].nunique()
        )
        if not feature_diff_df.empty
        else 0,
    }
    full_result_dict = {
        "specification_dict": {
            "snapshot_profile_str": profile_str,
            "snapshot_date_str": snapshot_date_str,
            "feature_start_date_str": feature_start_date_str,
            "backtest_start_date_str": backtest_start_date_str,
            "capital_base_float": CAPITAL_BASE_FLOAT,
            "slippage_float": 0.00025,
            "commission_per_share_float": 0.005,
            "commission_minimum_float": 1.0,
            "decision_timing_str": "after Close_T",
            "execution_timing_str": "Open_T_plus_1",
            "search_space_count_int": 1,
            "comparison_arms_dict": {
                "padded_trimmed": "current frozen snapshot contract",
                "padded_exact": (
                    "padding-only control with restored five PIT rows"
                ),
                "unpadded_trimmed": (
                    "membership-trim-only control with confirmed padded "
                    "rows removed from feature clock"
                ),
                "unpadded_exact": "strict HPI target contract",
            },
        },
        "manifest_dict": manifest_dict,
        "membership_audit_dict": {
            **membership_reconstruction_audit_dict,
            **direct_membership_audit_dict,
            "trimmed_vs_exact_membership_cell_difference_count_int": (
                membership_difference_count_int
            ),
            "trimmed_vs_exact_membership_date_count_int": (
                membership_difference_date_count_int
            ),
        },
        "padding_audit_dict": padding_audit_dict,
        "unpadding_audit_dict": unpadding_audit_dict,
        "feature_summary_dict": feature_summary_dict,
        "arm_metric_row_dict_list": arm_metric_row_dict_list,
        "comparison_row_dict_list": comparison_row_dict_list,
        "arm_result_dict": arm_result_dict,
        "feature_diff_row_dict_list": feature_diff_row_dict_list,
    }
    _write_artifacts(output_dir_path_obj, full_result_dict)
    _log(f"DONE output={output_dir_path_obj}")
    return full_result_dict


def main() -> int:
    parser_obj = argparse.ArgumentParser(
        description=(
            "Compare strict HPI with the padded/trimmed S&P 500 snapshot."
        )
    )
    parser_obj.add_argument(
        "--snapshot-root",
        default=str(DEFAULT_SNAPSHOT_ROOT_PATH),
    )
    parser_obj.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_STR,
    )
    parser_obj.add_argument(
        "--snapshot-date",
        default=DEFAULT_SNAPSHOT_DATE_STR,
    )
    parser_obj.add_argument(
        "--feature-start-date",
        default=DEFAULT_FEATURE_START_DATE_STR,
    )
    parser_obj.add_argument(
        "--backtest-start-date",
        default=DEFAULT_BACKTEST_START_DATE_STR,
    )
    parser_obj.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT_PATH
            / "results"
            / "research"
            / "hpi_snapshot_parity"
            / f"snapshot_{DEFAULT_SNAPSHOT_DATE_STR}"
        ),
    )
    parser_obj.add_argument(
        "--norgate-root",
        default=str(REPO_ROOT_PATH / ".tmp_norgatedata"),
    )
    args_obj = parser_obj.parse_args()

    os.environ["NORGATEDATA_ROOT"] = str(args_obj.norgate_root)
    os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "false"
    run_parity_test(
        snapshot_root_path_obj=Path(args_obj.snapshot_root),
        profile_str=str(args_obj.profile),
        snapshot_date_str=str(args_obj.snapshot_date),
        feature_start_date_str=str(args_obj.feature_start_date),
        backtest_start_date_str=str(args_obj.backtest_start_date),
        output_dir_path_obj=Path(args_obj.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
