"""Capture hash-checked FRED inputs and audit current-vintage drift against frozen TFI.

This is a forward research data collector, not a live DecisionPlan adapter.
Snapshots prove what this process received by its capture time; they do not
prove the source's original publication timestamp or reconstruct past vintages.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, time
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.data.fred_loader import build_fred_csv_url  # noqa: E402
from strategies.taa_beyond_6040 import (  # noqa: E402
    strategy_taa_tactical_fixed_income_ief_lqd as tactical_module,
)


DEFAULT_SNAPSHOT_ROOT_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "strategy"
    / tactical_module.STRATEGY_NAME_STR
    / "forward_fred_snapshots"
)
DECISION_RECORD_DIRNAME_STR = "decision_records"
NEW_YORK_ZONE_OBJ = ZoneInfo("America/New_York")
MAX_COMMON_OBSERVATION_AGE_SESSIONS_INT = 2


def parse_fred_series(series_id_str: str, csv_bytes: bytes) -> pd.Series:
    source_df = pd.read_csv(BytesIO(csv_bytes), keep_default_na=False)
    date_column_str = "observation_date" if "observation_date" in source_df else "DATE"
    if date_column_str not in source_df or series_id_str not in source_df:
        raise ValueError(f"FRED {series_id_str} CSV has missing required columns.")
    date_ser = pd.to_datetime(source_df[date_column_str], errors="raise").dt.normalize()
    if date_ser.duplicated().any() or not date_ser.is_monotonic_increasing:
        raise ValueError(f"FRED {series_id_str} dates must be unique and ascending.")
    raw_value_ser = source_df[series_id_str].astype("string").str.strip()
    value_ser = pd.to_numeric(raw_value_ser, errors="coerce")
    invalid_value_ser = value_ser.isna() & ~raw_value_ser.isin(["", "."])
    if invalid_value_ser.fillna(False).any() or not np.isfinite(value_ser.dropna()).all():
        raise ValueError(f"FRED {series_id_str} contains an invalid numeric value.")
    value_ser.index = pd.DatetimeIndex(date_ser)
    value_ser = value_ser.dropna().astype(float)
    if value_ser.empty:
        raise ValueError(f"FRED {series_id_str} has no numeric observations.")
    value_ser.name = series_id_str
    return value_ser


def capture_snapshot(snapshot_root_path: Path = DEFAULT_SNAPSHOT_ROOT_PATH) -> Path:
    series_bytes_by_id_dict: dict[str, bytes] = {}
    series_metadata_by_id_dict: dict[str, dict[str, str]] = {}
    for series_id_str in tactical_module.FRED_SERIES_ID_TUPLE:
        source_url_str = build_fred_csv_url(series_id_str)
        with urlopen(source_url_str, timeout=30) as response_obj:
            csv_bytes = response_obj.read()
        captured_at_dt = datetime.now(tz=UTC)
        value_ser = parse_fred_series(series_id_str, csv_bytes)
        if value_ser.index[-1].date() > captured_at_dt.date():
            raise ValueError(f"FRED {series_id_str} contains a future-dated observation.")
        series_bytes_by_id_dict[series_id_str] = csv_bytes
        series_metadata_by_id_dict[series_id_str] = {
            "source_url_str": source_url_str,
            "captured_at_utc_str": captured_at_dt.isoformat(),
            "sha256_str": hashlib.sha256(csv_bytes).hexdigest(),
            "latest_observation_date_str": value_ser.index[-1].date().isoformat(),
            "row_count_str": str(len(value_ser)),
        }

    snapshot_id_str = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_path = snapshot_root_path / snapshot_id_str
    snapshot_path.mkdir(parents=True, exist_ok=False)
    for series_id_str, csv_bytes in series_bytes_by_id_dict.items():
        (snapshot_path / f"{series_id_str}.csv").write_bytes(csv_bytes)
    manifest_dict = {
        "schema_version_int": 1,
        "snapshot_id_str": snapshot_id_str,
        "source_policy_str": "fresh_FRED_CSV_no_cache_capture_time_not_publication_time",
        "series_by_id_dict": series_metadata_by_id_dict,
    }
    (snapshot_path / "manifest.json").write_text(
        json.dumps(manifest_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def load_snapshot(snapshot_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest_dict = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    if manifest_dict.get("schema_version_int") != 1:
        raise ValueError("Unsupported forward FRED snapshot manifest version.")
    if manifest_dict.get("snapshot_id_str") != snapshot_path.name:
        raise ValueError("Forward FRED snapshot directory does not match its manifest.")
    series_metadata_by_id_dict = manifest_dict.get("series_by_id_dict", {})
    if set(series_metadata_by_id_dict) != set(tactical_module.FRED_SERIES_ID_TUPLE):
        raise ValueError("Forward FRED snapshot must contain exactly the four TFI series.")
    value_ser_list: list[pd.Series] = []
    for series_id_str in tactical_module.FRED_SERIES_ID_TUPLE:
        csv_bytes = (snapshot_path / f"{series_id_str}.csv").read_bytes()
        actual_sha256_str = hashlib.sha256(csv_bytes).hexdigest()
        if actual_sha256_str != series_metadata_by_id_dict[series_id_str]["sha256_str"]:
            raise ValueError(f"Forward FRED snapshot hash mismatch for {series_id_str}.")
        value_ser = parse_fred_series(series_id_str, csv_bytes)
        if value_ser.index[-1].date().isoformat() != series_metadata_by_id_dict[series_id_str]["latest_observation_date_str"]:
            raise ValueError(f"Forward FRED snapshot endpoint mismatch for {series_id_str}.")
        value_ser_list.append(value_ser)
    yield_df = pd.concat(value_ser_list, axis=1).sort_index()
    return yield_df, manifest_dict


def assess_capture_timing(manifest_dict: dict) -> dict[str, object]:
    capture_dt_list = [
        datetime.fromisoformat(series_metadata_dict["captured_at_utc_str"]).astimezone(UTC)
        for series_metadata_dict in manifest_dict["series_by_id_dict"].values()
    ]
    market_date_set = {
        captured_dt.astimezone(NEW_YORK_ZONE_OBJ).date()
        for captured_dt in capture_dt_list
    }
    reason_code_str = "eligible"
    if len(market_date_set) != 1:
        reason_code_str = "capture_crossed_market_dates"
    else:
        market_date_obj = next(iter(market_date_set))
        session_ts = pd.Timestamp(market_date_obj)
        calendar_obj = xcals.get_calendar(
            "XNYS",
            start=market_date_obj - pd.Timedelta(days=7),
            end=market_date_obj + pd.Timedelta(days=7),
        )
        if not calendar_obj.is_session(session_ts):
            reason_code_str = "not_xnys_session"
        elif calendar_obj.next_session(session_ts).month == session_ts.month:
            reason_code_str = "not_month_end_session"
        elif min(capture_dt_list) < calendar_obj.session_close(session_ts).to_pydatetime():
            reason_code_str = "capture_before_market_close"
        else:
            cutoff_dt = datetime.combine(
                market_date_obj, time(17, 15), tzinfo=NEW_YORK_ZONE_OBJ
            ).astimezone(UTC)
            if max(capture_dt_list) > cutoff_dt:
                reason_code_str = "capture_after_1715_et_cutoff"
    return {
        "source_received_in_decision_window_bool": reason_code_str == "eligible",
        "reason_code_str": reason_code_str,
        "market_date_str": (
            next(iter(market_date_set)).isoformat() if len(market_date_set) == 1 else None
        ),
    }


def build_forward_target(
    yield_row_ser: pd.Series,
    term_history_list: list[float],
    credit_history_list: list[float],
) -> dict[str, float]:
    term_spread_float = tactical_module.spread_value_float(
        yield_row_ser, tactical_module.TERM_PROXY_STR
    )
    credit_spread_float = tactical_module.spread_value_float(
        yield_row_ser, tactical_module.CREDIT_PROXY_STR
    )
    # *** CRITICAL *** The new observation enters its median exactly once;
    # future snapshots and revised historical values cannot enter this history.
    term_history_list.append(term_spread_float)
    credit_history_list.append(credit_spread_float)
    term_median_float = float(np.median(term_history_list))
    credit_median_float = float(np.median(credit_history_list))
    ief_weight_float = 0.5 * float(term_spread_float > term_median_float)
    lqd_weight_float = 0.5 * float(credit_spread_float > credit_median_float)
    return {
        "term_spread_float": term_spread_float,
        "credit_spread_float": credit_spread_float,
        "term_median_float": term_median_float,
        "credit_median_float": credit_median_float,
        "IEF_weight_float": ief_weight_float,
        "LQD_weight_float": lqd_weight_float,
        "Cash_weight_float": 1.0 - ief_weight_float - lqd_weight_float,
    }


def frozen_spread_history_lists(
    frozen_yield_df: pd.DataFrame,
    frozen_signal_df: pd.DataFrame,
) -> tuple[list[float], list[float]]:
    first_decision_ts = pd.Timestamp(frozen_signal_df.index[0])
    term_prehistory_record_list = tactical_module.historical_monthly_spread_records(
        tactical_module.TERM_PROXY_STR, frozen_yield_df, first_decision_ts
    )
    credit_prehistory_record_list = tactical_module.historical_monthly_spread_records(
        tactical_module.CREDIT_PROXY_STR, frozen_yield_df, first_decision_ts
    )
    term_history_list = [
        spread_float for _observation_ts, spread_float in term_prehistory_record_list
    ] + frozen_signal_df["term_spread_float"].astype(float).tolist()
    credit_history_list = [
        spread_float for _observation_ts, spread_float in credit_prehistory_record_list
    ] + frozen_signal_df["credit_spread_float"].astype(float).tolist()
    return term_history_list, credit_history_list


def load_governed_frozen_signal_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen_yield_df, _snapshot_tuple = tactical_module.load_frozen_yield_panel(
        tactical_module.DEFAULT_CONFIG
    )
    calendar_obj = xcals.get_calendar(
        "XNYS",
        start=tactical_module.DEFAULT_CONFIG.price_start_date_str,
        end=tactical_module.DEFAULT_CONFIG.end_date_str,
    )
    session_index = pd.DatetimeIndex(
        calendar_obj.sessions_in_range(
            tactical_module.DEFAULT_CONFIG.price_start_date_str,
            tactical_module.DEFAULT_CONFIG.end_date_str,
        )
    ).tz_localize(None)
    frozen_signal_df, frozen_weight_df = tactical_module.build_month_end_signal_and_weight_df(
        yield_df=frozen_yield_df,
        session_index=session_index,
        last_complete_signal_month_str=tactical_module.DEFAULT_CONFIG.last_complete_signal_month_str,
    )
    signal_hash_str = tactical_module.canonical_dataframe_sha256_str(
        tactical_module.build_canonical_signal_contract_df(
            frozen_signal_df, frozen_weight_df
        )
    )
    if signal_hash_str != tactical_module.FROZEN_SIGNAL_CONTRACT_SHA256_STR:
        raise ValueError("The frozen FRED/XNYS signal history no longer matches its governed hash.")
    return frozen_yield_df, frozen_signal_df


def build_forward_decision_report(
    snapshot_root_path: Path = DEFAULT_SNAPSHOT_ROOT_PATH,
) -> dict[str, object]:
    latest_snapshot_by_month_dict: dict[pd.Period, tuple[Path, pd.DataFrame, dict]] = {}
    eligible_snapshot_by_id_dict: dict[str, tuple[Path, pd.DataFrame, dict]] = {}
    decision_record_by_month_dict: dict[pd.Period, dict] = {}
    capture_status_by_snapshot_dict: dict[str, dict] = {}
    for manifest_path in sorted(snapshot_root_path.glob("*/manifest.json")):
        snapshot_path = manifest_path.parent
        yield_df, manifest_dict = load_snapshot(snapshot_path)
        timing_dict = assess_capture_timing(manifest_dict)
        capture_status_by_snapshot_dict[snapshot_path.name] = timing_dict
        if not timing_dict["source_received_in_decision_window_bool"]:
            continue
        eligible_snapshot_by_id_dict[snapshot_path.name] = (
            snapshot_path, yield_df, manifest_dict
        )
        month_period = pd.Period(timing_dict["market_date_str"], freq="M")
        frozen_month_period = pd.Period(
            tactical_module.DEFAULT_CONFIG.last_complete_signal_month_str, freq="M"
        )
        if month_period <= frozen_month_period:
            continue
        previous_record_tuple = latest_snapshot_by_month_dict.get(month_period)
        if previous_record_tuple is None or snapshot_path.name > previous_record_tuple[0].name:
            latest_snapshot_by_month_dict[month_period] = (
                snapshot_path, yield_df, manifest_dict
            )

    decision_record_root_path = snapshot_root_path / DECISION_RECORD_DIRNAME_STR
    for decision_record_path in sorted(decision_record_root_path.glob("*.json")):
        decision_record_dict = json.loads(decision_record_path.read_text(encoding="utf-8"))
        if decision_record_dict.get("schema_version_int") != 1:
            raise ValueError("Unsupported TFI forward decision record version.")
        month_period = pd.Period(decision_record_dict["month_str"], freq="M")
        if decision_record_path.stem != str(month_period):
            raise ValueError("TFI decision record filename and month differ.")
        snapshot_id_str = decision_record_dict["snapshot_id_str"]
        if snapshot_id_str not in eligible_snapshot_by_id_dict:
            raise ValueError("TFI decision record references an ineligible snapshot.")
        snapshot_tuple = eligible_snapshot_by_id_dict[snapshot_id_str]
        snapshot_path = snapshot_tuple[0]
        if pd.Period(assess_capture_timing(snapshot_tuple[2])["market_date_str"], freq="M") != month_period:
            raise ValueError("TFI decision record snapshot belongs to another month.")
        manifest_sha256_str = hashlib.sha256(
            (snapshot_path / "manifest.json").read_bytes()
        ).hexdigest()
        if manifest_sha256_str != decision_record_dict["manifest_sha256_str"]:
            raise ValueError("TFI decision record manifest hash mismatch.")
        decision_record_by_month_dict[month_period] = decision_record_dict
        latest_snapshot_by_month_dict[month_period] = snapshot_tuple

    report_dict: dict[str, object] = {
        "scope_str": "forward_shadow_target_only_no_orders_or_live_authority",
        "capture_status_by_snapshot_dict": capture_status_by_snapshot_dict,
        "decision_row_list": [],
    }
    if not latest_snapshot_by_month_dict:
        report_dict["status_str"] = "no_eligible_month_end_snapshot"
        return report_dict

    frozen_month_period = pd.Period(
        tactical_module.DEFAULT_CONFIG.last_complete_signal_month_str, freq="M"
    )
    expected_month_index = pd.period_range(
        start=frozen_month_period + 1,
        end=max(latest_snapshot_by_month_dict),
        freq="M",
    )
    missing_month_list = [
        str(month_period)
        for month_period in expected_month_index
        if month_period not in latest_snapshot_by_month_dict
    ]
    if missing_month_list:
        report_dict["status_str"] = "missing_contiguous_month_end_snapshot"
        report_dict["missing_month_list"] = missing_month_list
        return report_dict

    # Frozen prehistory plus the governed 289 decisions anchor each median.
    # Only rows captured within a new decision window may extend it.
    frozen_yield_df, frozen_signal_df = load_governed_frozen_signal_history()
    term_history_list, credit_history_list = frozen_spread_history_lists(
        frozen_yield_df, frozen_signal_df
    )
    latest_decision_ts = pd.Timestamp(
        assess_capture_timing(latest_snapshot_by_month_dict[max(expected_month_index)][2])[
            "market_date_str"
        ]
    )
    calendar_obj = xcals.get_calendar(
        "XNYS",
        start="2026-08-01",
        end=latest_decision_ts + pd.Timedelta(days=7),
    )
    decision_row_list: list[dict[str, object]] = []
    for month_period in expected_month_index:
        snapshot_path, yield_df, manifest_dict = latest_snapshot_by_month_dict[month_period]
        decision_date_ts = pd.Timestamp(
            assess_capture_timing(manifest_dict)["market_date_str"]
        )
        prior_session_ts = calendar_obj.previous_session(decision_date_ts)
        common_yield_df = yield_df.loc[:, list(tactical_module.FRED_SERIES_ID_TUPLE)].dropna()
        eligible_yield_df = common_yield_df.loc[
            common_yield_df.index <= prior_session_ts
        ]
        if eligible_yield_df.empty:
            raise ValueError(f"No common published-by-capture FRED row for {month_period}.")
        observation_date_ts = pd.Timestamp(eligible_yield_df.index[-1])
        age_sessions_int = len(
            calendar_obj.sessions_in_range(observation_date_ts, prior_session_ts)
        ) - 1
        if age_sessions_int > MAX_COMMON_OBSERVATION_AGE_SESSIONS_INT:
            raise ValueError(
                f"Common FRED row for {month_period} is {age_sessions_int} XNYS sessions stale."
            )
        target_dict = build_forward_target(
            eligible_yield_df.iloc[-1], term_history_list, credit_history_list
        )
        decision_record_dict = decision_record_by_month_dict.get(month_period)
        if decision_record_dict is not None:
            recorded_at_dt = datetime.fromisoformat(
                decision_record_dict["decision_row_dict"]["computed_at_utc_str"]
            )
            if recorded_at_dt.tzinfo is None:
                raise ValueError("TFI recorded decision time must have a timezone.")
            computed_at_dt = recorded_at_dt.astimezone(UTC)
        else:
            computed_at_dt = datetime.now(tz=UTC)
        decision_cutoff_dt = datetime.combine(
            decision_date_ts.date(), time(17, 15), tzinfo=NEW_YORK_ZONE_OBJ
        ).astimezone(UTC)
        capture_dt_list = [
            datetime.fromisoformat(metadata_dict["captured_at_utc_str"]).astimezone(UTC)
            for metadata_dict in manifest_dict["series_by_id_dict"].values()
        ]
        decision_row_dict = {
            "decision_date_str": decision_date_ts.date().isoformat(),
            "observation_date_str": observation_date_ts.date().isoformat(),
            "observation_age_sessions_int": age_sessions_int,
            "next_open_date_str": calendar_obj.next_session(decision_date_ts).date().isoformat(),
            "snapshot_id_str": snapshot_path.name,
            "computed_at_utc_str": computed_at_dt.isoformat(),
            "computed_by_1715_et_bool": max(capture_dt_list) <= computed_at_dt <= decision_cutoff_dt,
            **target_dict,
        }
        if decision_record_dict is not None and decision_row_dict != decision_record_dict["decision_row_dict"]:
            raise ValueError("TFI recorded decision differs from its recomputed inputs.")
        decision_row_list.append(decision_row_dict)
    report_dict["status_str"] = (
        "shadow_target_computed_by_cutoff"
        if all(
            decision_row_dict["computed_by_1715_et_bool"]
            for decision_row_dict in decision_row_list
        )
        else "shadow_target_reconstructed_after_cutoff"
    )
    report_dict["decision_row_list"] = decision_row_list
    return report_dict


def persist_on_time_decision(
    report_dict: dict[str, object],
    snapshot_root_path: Path,
) -> None:
    if report_dict["status_str"] != "shadow_target_computed_by_cutoff":
        raise ValueError("Only on-time TFI shadow decisions can be recorded.")
    decision_row_dict = report_dict["decision_row_list"][-1]
    month_str = str(pd.Period(decision_row_dict["decision_date_str"], freq="M"))
    decision_record_root_path = snapshot_root_path / DECISION_RECORD_DIRNAME_STR
    decision_record_path = decision_record_root_path / f"{month_str}.json"
    snapshot_path = snapshot_root_path / decision_row_dict["snapshot_id_str"]
    decision_record_dict = {
        "schema_version_int": 1,
        "month_str": month_str,
        "snapshot_id_str": snapshot_path.name,
        "manifest_sha256_str": hashlib.sha256(
            (snapshot_path / "manifest.json").read_bytes()
        ).hexdigest(),
        "decision_row_dict": decision_row_dict,
    }
    if decision_record_path.exists():
        existing_record_dict = json.loads(decision_record_path.read_text(encoding="utf-8"))
        if existing_record_dict != decision_record_dict:
            raise ValueError("An existing TFI decision record differs from this decision.")
        return
    decision_record_root_path.mkdir(parents=True, exist_ok=True)
    with decision_record_path.open("x", encoding="utf-8") as decision_file_obj:
        json.dump(decision_record_dict, decision_file_obj, indent=2, sort_keys=True)
        decision_file_obj.write("\n")


def compare_snapshot(snapshot_path: Path, *, include_nav_bool: bool = True) -> dict:
    current_yield_df, manifest_dict = load_snapshot(snapshot_path)
    (
        execution_price_df,
        frozen_yield_df,
        frozen_signal_df,
        frozen_weight_df,
        frozen_cash_return_ser,
        frozen_fred_snapshot_tuple,
    ) = tactical_module.get_tactical_yield_data(tactical_module.DEFAULT_CONFIG)
    frozen_end_ts = pd.Timestamp(tactical_module.DEFAULT_CONFIG.end_date_str)
    current_yield_df = current_yield_df.loc[current_yield_df.index <= frozen_end_ts]
    series_delta_by_id_dict: dict[str, dict] = {}
    for series_id_str in tactical_module.FRED_SERIES_ID_TUPLE:
        frozen_ser = frozen_yield_df[series_id_str].dropna()
        current_ser = current_yield_df[series_id_str].dropna()
        common_index = frozen_ser.index.intersection(current_ser.index)
        value_gap_ser = (frozen_ser.loc[common_index] - current_ser.loc[common_index]).abs()
        series_delta_by_id_dict[series_id_str] = {
            "common_observation_count_int": len(common_index),
            "changed_observation_count_int": int((value_gap_ser > 1e-12).sum()),
            "frozen_only_observation_count_int": len(frozen_ser.index.difference(current_ser.index)),
            "current_only_observation_count_int": len(current_ser.index.difference(frozen_ser.index)),
            "maximum_yield_point_gap_float": float(value_gap_ser.max()) if len(value_gap_ser) else None,
        }

    # *** CRITICAL *** This rerun uses a snapshot captured now to inspect
    # historical revisions. It is a retroactive diagnostic, never a causal
    # historical backtest or evidence of a valid past decision-time vintage.
    current_signal_df, current_weight_df = tactical_module.build_month_end_signal_and_weight_df(
        yield_df=current_yield_df,
        session_index=pd.DatetimeIndex(execution_price_df.index),
        last_complete_signal_month_str=tactical_module.DEFAULT_CONFIG.last_complete_signal_month_str,
    )
    if not current_signal_df.index.equals(frozen_signal_df.index):
        raise ValueError("Current FRED snapshot changed the historical decision calendar.")
    frozen_target_df = frozen_weight_df.set_index("decision_date")[["IEF", "LQD", "Cash"]]
    current_target_df = current_weight_df.set_index("decision_date")[["IEF", "LQD", "Cash"]]
    changed_target_bool_ser = frozen_target_df.ne(current_target_df).any(axis=1)
    signal_column_list = [
        "observation_date", "term_spread_float", "credit_spread_float",
        "term_threshold_float", "credit_threshold_float", "term_state_float", "credit_state_float",
    ]
    changed_signal_bool_ser = frozen_signal_df[signal_column_list].ne(
        current_signal_df[signal_column_list]
    ).any(axis=1)
    report_dict = {
        "snapshot_path_str": str(snapshot_path.resolve()),
        "comparison_type_str": "retroactive_current_vintage_diagnostic_not_decision_time_replay",
        "frozen_baseline_modified_bool": False,
        "capture_timing_dict": assess_capture_timing(manifest_dict),
        "series_delta_by_id_dict": series_delta_by_id_dict,
        "historical_signal_row_count_int": len(frozen_signal_df),
        "changed_historical_signal_row_count_int": int(changed_signal_bool_ser.sum()),
        "changed_historical_target_row_count_int": int(changed_target_bool_ser.sum()),
        "changed_target_decision_date_list": [
            decision_ts.date().isoformat()
            for decision_ts in changed_target_bool_ser.index[changed_target_bool_ser]
        ],
        "snapshot_capture_time_by_series_dict": {
            series_id_str: series_metadata_dict["captured_at_utc_str"]
            for series_id_str, series_metadata_dict in manifest_dict["series_by_id_dict"].items()
        },
    }
    if include_nav_bool:
        current_cash_return_ser = tactical_module.build_causal_cash_return_ser(
            pd.DatetimeIndex(execution_price_df.index), current_yield_df["DGS3MO"].dropna()
        )
        current_fred_snapshot_tuple = tuple(
            tactical_module.FrozenFredSnapshot(
                series_id_str=series_id_str,
                value_ser=current_yield_df[series_id_str].dropna(),
                source_path_str=str(snapshot_path / f"{series_id_str}.csv"),
                sha256_str=manifest_dict["series_by_id_dict"][series_id_str]["sha256_str"],
                latest_observation_date_ts=current_yield_df[series_id_str].dropna().index[-1],
                vintage_policy_str="current_vintage_capture_diagnostic_not_alfred",
            )
            for series_id_str in tactical_module.FRED_SERIES_ID_TUPLE
        )
        run_keyword_dict = {
            "config_obj": tactical_module.DEFAULT_CONFIG,
            "execution_price_df": execution_price_df,
            "backtest_start_date_str": "2002-08-01",
            "end_date_str": tactical_module.DEFAULT_CONFIG.end_date_str,
            "show_progress_bool": False,
        }
        frozen_strategy_obj = tactical_module._run_strategy(
            **run_keyword_dict,
            signal_df=frozen_signal_df,
            rebalance_weight_df=frozen_weight_df,
            cash_return_ser=frozen_cash_return_ser,
            fred_snapshot_tuple=frozen_fred_snapshot_tuple,
        )
        current_strategy_obj = tactical_module._run_strategy(
            **run_keyword_dict,
            signal_df=current_signal_df,
            rebalance_weight_df=current_weight_df,
            cash_return_ser=current_cash_return_ser,
            fred_snapshot_tuple=current_fred_snapshot_tuple,
        )
        frozen_nav_ser = frozen_strategy_obj.results["total_value"].astype(float)
        current_nav_ser = current_strategy_obj.results["total_value"].astype(float)
        if not frozen_nav_ser.index.equals(current_nav_ser.index):
            raise ValueError("Current-vintage diagnostic changed the NAV comparison dates.")
        report_dict["nav_delta_dict"] = {
            "frozen_terminal_nav_float": float(frozen_nav_ser.iloc[-1]),
            "current_vintage_terminal_nav_float": float(current_nav_ser.iloc[-1]),
            "terminal_nav_gap_float": float(current_nav_ser.iloc[-1] - frozen_nav_ser.iloc[-1]),
            "maximum_absolute_daily_nav_gap_float": float((current_nav_ser - frozen_nav_ser).abs().max()),
        }
    return report_dict


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    command_subparser_obj = parser_obj.add_subparsers(dest="command_str", required=True)
    capture_parser_obj = command_subparser_obj.add_parser("capture")
    capture_parser_obj.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT_PATH)
    compare_parser_obj = command_subparser_obj.add_parser("compare")
    compare_parser_obj.add_argument("snapshot_path", type=Path)
    compare_parser_obj.add_argument("--no-nav", action="store_true")
    forward_parser_obj = command_subparser_obj.add_parser("forward")
    forward_parser_obj.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT_PATH)
    argument_obj = parser_obj.parse_args()
    if argument_obj.command_str == "capture":
        snapshot_path = capture_snapshot(argument_obj.snapshot_root)
        print(snapshot_path.resolve())
    elif argument_obj.command_str == "compare":
        report_dict = compare_snapshot(
            argument_obj.snapshot_path, include_nav_bool=not argument_obj.no_nav
        )
        print(json.dumps(report_dict, indent=2, sort_keys=True))
    else:
        report_dict = build_forward_decision_report(argument_obj.snapshot_root)
        print(json.dumps(report_dict, indent=2, sort_keys=True))
        decision_row_list = report_dict["decision_row_list"]
        if (
            report_dict["status_str"] != "shadow_target_computed_by_cutoff"
            or not decision_row_list
            or not all(
                decision_row_dict["computed_by_1715_et_bool"]
                for decision_row_dict in decision_row_list
            )
        ):
            return 2
        latest_decision_date_ts = pd.Timestamp(decision_row_list[-1]["decision_date_str"])
        calendar_obj = xcals.get_calendar(
            "XNYS",
            start=latest_decision_date_ts - pd.Timedelta(days=7),
            end=latest_decision_date_ts + pd.Timedelta(days=7),
        )
        decision_close_dt = calendar_obj.session_close(latest_decision_date_ts).to_pydatetime()
        decision_cutoff_dt = datetime.combine(
            latest_decision_date_ts.date(), time(17, 15), tzinfo=NEW_YORK_ZONE_OBJ
        ).astimezone(UTC)
        current_dt = datetime.now(tz=UTC)
        if not decision_close_dt <= current_dt <= decision_cutoff_dt:
            return 2
        persist_on_time_decision(report_dict, argument_obj.snapshot_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
