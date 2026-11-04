"""Frozen snapshot contract of the momentum capsule's three legs; used only by their two profiles.

A live leg decision replays the whole path from the design's first decision, so a snapshot carries the full history a
backtest reads. Labels travel in the manifest: a client never calls the provider.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import time

import exchange_calendars as exchange_calendar_module
import numpy as np
import pandas as pd

MOMENTUM_LEGS_NDX_PROFILE_STR = "norgate_eod_ndx_momentum_legs_pit"
MOMENTUM_LEGS_R1000_PROFILE_STR = "norgate_eod_r1000_momentum_legs_pit"
MOMENTUM_LEGS_PROFILE_TUPLE = (MOMENTUM_LEGS_NDX_PROFILE_STR, MOMENTUM_LEGS_R1000_PROFILE_STR)
MOMENTUM_LEGS_PRICE_HISTORY_START_DATE_STR = "1998-01-01"
MOMENTUM_LEGS_MEMBER_START_DATE_STR = "1999-06-01"
SECTOR_INDEX_BY_LABEL_DICT: dict[str, str] = {
    "10000000": "$SPXE",
    "15000000": "$SPXM",
    "20000000": "$SPXI",
    "25000000": "$SPXD",
    "30000000": "$SPXS",
    "35000000": "$SPXA",
    "40000000": "$SPXF",
    "45000000": "$SPXT",
    "50000000": "$SPXL",
    "55000000": "$SPXU",
    "60000000": "$SPXR",
}
SECTOR_INDEX_SYMBOL_TUPLE = tuple(SECTOR_INDEX_BY_LABEL_DICT.values())
UNKNOWN_LABEL_STR = "UNKNOWN"
SECTOR_LABEL_SET = {*SECTOR_INDEX_BY_LABEL_DICT, UNKNOWN_LABEL_STR}
BENCHMARK_SYMBOL_STR = "$SPXTR"
VXN_SYMBOL_STR = "$VXN"
# A member without an unpadded close at T is a halt, not a stale feed. A larger share than this means the stocks have
# not updated yet. Those members keep the carried close the backtest also reads, and are listed in the manifest.
MAX_UNOBSERVED_MEMBER_FRACTION_FLOAT = 0.02
# A stock with no bar at T also reads as a non-member at T, so a partial update shrinks the list instead of filling
# unobserved_member_list. Worst real one-day drops: Nasdaq-100 1.9%, Russell 1000 0.4%.
MIN_MEMBER_COUNT_RATIO_FLOAT = 0.98
# Real: 1 of 1,019 Russell 1000 members. Many more means the labels did not arrive.
MAX_UNKNOWN_MEMBER_FRACTION_FLOAT = 0.05
LABEL_READ_COUNT_INT = 3
LABEL_READ_PAUSE_SECONDS_FLOAT = 0.05
MOMENTUM_LEGS_DATA_CONTRACT_DICT = {
    "price_history_start_date_str": MOMENTUM_LEGS_PRICE_HISTORY_START_DATE_STR,
    "member_start_date_str": MOMENTUM_LEGS_MEMBER_START_DATE_STR,
    "price_padding_setting_str": "ALLMARKETDAYS",
    "past_member_tail_policy_str": "exact",
    "label_classification_source_str": "Norgate",
    "label_classification_scheme_str": "GICS",
    "label_classification_basis_str": "current_at_export",
}
MOMENTUM_LEGS_SPEC_BY_PROFILE_DICT: dict[str, dict] = {
    MOMENTUM_LEGS_NDX_PROFILE_STR: {
        "indexname_str": "Nasdaq 100",
        "capital_symbol_tuple": ("SPY",),
        "helper_symbol_tuple": ("$NDX", VXN_SYMBOL_STR, *SECTOR_INDEX_SYMBOL_TUPLE),
        "group_label_bool": False,
    },
    MOMENTUM_LEGS_R1000_PROFILE_STR: {
        "indexname_str": "Russell 1000",
        "capital_symbol_tuple": (),
        "helper_symbol_tuple": ("$RUI", *SECTOR_INDEX_SYMBOL_TUPLE),
        "group_label_bool": True,
    },
}


def read_stable_gics_label_str(provider_obj, symbol_str: str, level_int: int) -> str:
    """*** CRITICAL *** Norgate can answer None for a classified name while another process reads it (DIS, once in about 6,000 reads, 2026-11-03). A false UNKNOWN moves a sector cap or a peer group, and through the rank buffers the path behind it. A name reads UNKNOWN only after every read says so. A provider exception is not an observed answer and is raised."""
    for read_int in range(LABEL_READ_COUNT_INT):
        label_obj = provider_obj.classification_at_level(symbol_str, "GICS", "ClassificationId", int(level_int))
        if label_obj is not None and str(label_obj).strip():
            return str(label_obj).strip()
        if read_int + 1 < LABEL_READ_COUNT_INT:
            time.sleep(LABEL_READ_PAUSE_SECONDS_FLOAT)
    return UNKNOWN_LABEL_STR


def momentum_legs_label_hash_str(sector_by_symbol_dict: dict[str, str], group_by_symbol_dict: dict[str, str] | None) -> str:
    return hashlib.sha256(json.dumps(
        {"sector": sector_by_symbol_dict, "group": group_by_symbol_dict}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def strict_endpoint_symbol_list(profile_str: str) -> list[str]:
    """Series whose close at T must be an unpadded observation. VXN keeps the as-of rule the strategy reads."""
    spec_dict = MOMENTUM_LEGS_SPEC_BY_PROFILE_DICT[profile_str]
    helper_list = [symbol_str for symbol_str in spec_dict["helper_symbol_tuple"] if symbol_str != VXN_SYMBOL_STR]
    return sorted({*spec_dict["capital_symbol_tuple"], *helper_list, BENCHMARK_SYMBOL_STR})


def _validate_labels(contract_dict: dict, universe_symbol_set: set[str], profile_str: str, snapshot_date_ts: pd.Timestamp) -> None:
    if contract_dict.get("label_snapshot_date_str") != snapshot_date_ts.date().isoformat():
        raise ValueError("Momentum legs labels belong to another snapshot date.")
    try:
        if datetime.fromisoformat(contract_dict["label_observed_timestamp_str"]).tzinfo is None:
            raise ValueError("Timezone missing")
    except (KeyError, TypeError, ValueError) as error_obj:
        raise ValueError("Momentum legs label observation timestamp is invalid.") from error_obj
    sector_by_symbol_dict = contract_dict.get("sector_by_symbol_dict")
    if not isinstance(sector_by_symbol_dict, dict) or set(sector_by_symbol_dict) != universe_symbol_set or any(
            label_str not in SECTOR_LABEL_SET for label_str in sector_by_symbol_dict.values()):
        raise ValueError("Momentum legs require a GICS sector or UNKNOWN for exactly the PIT universe.")
    group_by_symbol_dict = contract_dict.get("group_by_symbol_dict")
    if MOMENTUM_LEGS_SPEC_BY_PROFILE_DICT[profile_str]["group_label_bool"]:
        if not isinstance(group_by_symbol_dict, dict) or set(group_by_symbol_dict) != universe_symbol_set or any(
                not isinstance(label_str, str) or not label_str.strip() for label_str in group_by_symbol_dict.values()):
            raise ValueError("The Russell 1000 leg requires a GICS industry group or UNKNOWN for exactly the PIT universe.")
        if any((sector_by_symbol_dict[symbol_str] == UNKNOWN_LABEL_STR) != (group_str == UNKNOWN_LABEL_STR)
                or (group_str != UNKNOWN_LABEL_STR and group_str[:2] != sector_by_symbol_dict[symbol_str][:2])
                for symbol_str, group_str in group_by_symbol_dict.items()):
            raise ValueError("A name's GICS sector and industry group disagree; one read was false.")
    elif group_by_symbol_dict is not None:
        raise ValueError("The Nasdaq-100 legs carry no industry-group map.")
    if contract_dict.get("label_map_hash_str") != momentum_legs_label_hash_str(sector_by_symbol_dict, group_by_symbol_dict):
        raise ValueError("Momentum legs label map hash mismatch.")


def validate_momentum_legs_snapshot_content(
    price_df: pd.DataFrame, universe_df: pd.DataFrame, manifest_dict: dict, snapshot_date_ts: pd.Timestamp, profile_str: str,
) -> None:
    """Reject missing lineage, altered units, holes inside a history, stale endpoints and unavailable evidence."""
    if manifest_dict.get("schema_version") != 2:
        raise ValueError("Momentum legs require schema v2.")
    contract_dict = manifest_dict.get("data_contract", {})
    if not isinstance(contract_dict, dict) or any(contract_dict.get(key_str) != value_obj for key_str, value_obj in MOMENTUM_LEGS_DATA_CONTRACT_DICT.items()):
        raise ValueError("Momentum legs data contract mismatch.")
    spec_dict = MOMENTUM_LEGS_SPEC_BY_PROFILE_DICT[profile_str]
    universe_symbol_set = {str(symbol_str) for symbol_str in universe_df.columns}
    _validate_labels(contract_dict, universe_symbol_set, profile_str, snapshot_date_ts)
    # *** CRITICAL *** Membership at T must exist explicitly; no later row may certify the signal session.
    universe_date_idx = pd.DatetimeIndex(pd.to_datetime(universe_df.index)).normalize()
    if universe_date_idx.has_duplicates or snapshot_date_ts not in universe_date_idx or universe_date_idx.max() != snapshot_date_ts:
        raise ValueError("Momentum legs PIT universe needs one row at the snapshot session and none later.")
    membership_df = universe_df.apply(pd.to_numeric, errors="coerce")
    if not membership_df.isin([0, 1]).all().all():
        raise ValueError("Momentum legs PIT membership must contain only explicit zero/one values.")
    capital_symbol_set = set(spec_dict["capital_symbol_tuple"])
    helper_symbol_set = set(spec_dict["helper_symbol_tuple"])
    if set(manifest_dict.get("required_helpers", [])) != helper_symbol_set:
        raise ValueError("Momentum legs manifest helpers differ from the profile.")
    if set(manifest_dict.get("required_symbols", [])) != universe_symbol_set | capital_symbol_set | {BENCHMARK_SYMBOL_STR}:
        raise ValueError("Momentum legs manifest symbols differ from the PIT universe and benchmark.")
    if not {"date", "symbol_str", "adjustment_str", "Open", "High", "Low", "Close", "Turnover", "Unadjusted Close"}.issubset(price_df.columns):
        raise ValueError("Momentum legs require OHLC, Turnover and nominal closing-price anchors.")
    date_ser = pd.to_datetime(price_df["date"]).dt.normalize()
    if date_ser.isna().any() or date_ser.gt(snapshot_date_ts).any() or date_ser.lt(pd.Timestamp(MOMENTUM_LEGS_PRICE_HISTORY_START_DATE_STR)).any():
        raise ValueError("Momentum legs price rows lie outside the declared history.")
    if price_df.duplicated(["date", "symbol_str", "adjustment_str"]).any():
        raise ValueError("Momentum legs price rows are duplicated.")
    capital_mask_ser = price_df["adjustment_str"].eq("CAPITALSPECIAL")
    stock_mask_ser = capital_mask_ser & price_df["symbol_str"].isin(universe_symbol_set | capital_symbol_set)
    observed_mask_ser = stock_mask_ser & price_df["Close"].notna()
    anchor_mat = price_df.loc[observed_mask_ser, ["Open", "High", "Low", "Close", "Unadjusted Close"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(anchor_mat).all() or (anchor_mat <= 0).any():
        raise ValueError("Momentum legs observed stock OHLC and nominal anchors must be finite and positive.")
    current_member_set = set(universe_df.columns[membership_df.loc[universe_date_idx == snapshot_date_ts].iloc[0] == 1].astype(str))
    if not current_member_set:
        raise ValueError("Momentum legs need current PIT members at the snapshot session.")
    prior_row_list = sorted(date_ts for date_ts in universe_date_idx if date_ts < snapshot_date_ts)
    if prior_row_list and len(current_member_set) < MIN_MEMBER_COUNT_RATIO_FLOAT * float(membership_df.loc[universe_date_idx == prior_row_list[-1]].iloc[0].sum()):
        raise ValueError("Momentum legs have too few members at the snapshot session; the stocks have not all updated.")
    unknown_member_int = sum(contract_dict["sector_by_symbol_dict"][symbol_str] == UNKNOWN_LABEL_STR for symbol_str in current_member_set)
    if unknown_member_int > MAX_UNKNOWN_MEMBER_FRACTION_FLOAT * len(current_member_set):
        raise ValueError("Too many current members have no GICS sector.")
    unobserved_member_set = set(contract_dict.get("unobserved_member_list", ["?"]))
    if not unobserved_member_set.issubset(current_member_set) or len(unobserved_member_set) > MAX_UNOBSERVED_MEMBER_FRACTION_FLOAT * len(current_member_set):
        raise ValueError("Too many Momentum legs members have no unpadded close at the snapshot session.")
    expected_endpoint_dict = {symbol_str: snapshot_date_ts.date().isoformat() for symbol_str in (current_member_set - unobserved_member_set) | set(strict_endpoint_symbol_list(profile_str))}
    if contract_dict.get("observed_endpoint_date_by_symbol_dict") != expected_endpoint_dict:
        raise ValueError("Momentum legs require unpadded source observations through the snapshot session.")
    at_t_set = set(price_df.loc[observed_mask_ser & date_ser.eq(snapshot_date_ts), "symbol_str"])
    if not (current_member_set | capital_symbol_set).issubset(at_t_set):
        raise ValueError("Momentum legs current members and capital symbols require snapshot-session prices.")
    turnover_at_t_vec = pd.to_numeric(price_df.loc[stock_mask_ser & date_ser.eq(snapshot_date_ts) & price_df["symbol_str"].isin(current_member_set), "Turnover"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(turnover_at_t_vec).all() or (turnover_at_t_vec < 0).any():
        raise ValueError("Momentum legs current members require a finite Turnover at the snapshot session.")
    # *** CRITICAL *** Rolling windows count sessions, not surviving observations. A hole between a series' first and
    # last row would shorten its windows; pre-listing and post-delisting absence stays legitimate.
    first_date_ts = pd.Timestamp(date_ser.min())
    calendar_obj = exchange_calendar_module.get_calendar("XNYS", start=str((first_date_ts - pd.Timedelta(days=31)).date()), end=str((snapshot_date_ts + pd.Timedelta(days=31)).date()))
    session_idx = pd.DatetimeIndex(calendar_obj.sessions_in_range(first_date_ts, snapshot_date_ts)).tz_localize(None)
    session_position_ser = pd.Series(np.arange(len(session_idx)), index=session_idx)
    position_ser = date_ser.loc[stock_mask_ser].map(session_position_ser)
    if position_ser.isna().any():
        raise ValueError("Momentum legs stock history holds a date outside the XNYS sessions.")
    for symbol_str, symbol_position_ser in position_ser.groupby(price_df.loc[stock_mask_ser, "symbol_str"]):
        position_vec = np.sort(symbol_position_ser.to_numpy(dtype=int))
        if len(position_vec) != position_vec[-1] - position_vec[0] + 1:
            raise ValueError(f"Momentum legs stock history has a missing market session: {symbol_str}.")
    for symbol_str in strict_endpoint_symbol_list(profile_str):
        adjustment_str = "TOTALRETURN" if symbol_str == BENCHMARK_SYMBOL_STR else "CAPITALSPECIAL"
        row_mask_ser = price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str)
        series_date_idx = pd.DatetimeIndex(date_ser.loc[row_mask_ser]).sort_values()
        close_vec = pd.to_numeric(price_df.loc[row_mask_ser, "Close"], errors="coerce").to_numpy(dtype=float)
        if series_date_idx.empty or series_date_idx[-1] != snapshot_date_ts or not np.isfinite(close_vec).all() or (close_vec <= 0).any():
            raise ValueError(f"Momentum legs require a finite positive series through T: {symbol_str}.")
        if not series_date_idx.equals(session_idx[session_idx >= series_date_idx[0]]):
            raise ValueError(f"Momentum legs series has a missing market session: {symbol_str}.")
    if VXN_SYMBOL_STR in helper_symbol_set:
        vxn_vec = pd.to_numeric(price_df.loc[capital_mask_ser & price_df["symbol_str"].eq(VXN_SYMBOL_STR), "Close"], errors="coerce").to_numpy(dtype=float)
        if vxn_vec.size == 0 or not np.isfinite(vxn_vec).all() or (vxn_vec <= 0).any():
            raise ValueError("Momentum legs require finite positive as-of VXN observations.")
