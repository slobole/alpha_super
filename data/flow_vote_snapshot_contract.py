"""Frozen Month-End Flow Vote snapshot contract; only used by the Flow Vote profile.

The research loader reads SPY and TLT CAPITALSPECIAL, SPY, IEF and TLT TOTALRETURN and $SPXTR from Norgate with
PaddingType.NONE from 2002-07-26. The snapshot keeps exactly that: native fields, native dtypes, observed rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import exchange_calendars as exchange_calendar_module

FLOW_VOTE_PROFILE_STR = "norgate_eod_flow_vote"
FLOW_VOTE_HISTORY_START_STR = "2002-07-26"
FLOW_VOTE_CAPITAL_SYMBOL_TUPLE = ("SPY", "TLT")
FLOW_VOTE_TOTAL_RETURN_SYMBOL_TUPLE = ("SPY", "IEF", "TLT", "$SPXTR")
FLOW_VOTE_DATA_CONTRACT_DICT = {
    "history_start_date_str": FLOW_VOTE_HISTORY_START_STR,
    "price_padding_setting_str": "NONE",
    "past_member_tail_policy_str": "exact",
}


def flow_vote_required_pair_set() -> set[tuple[str, str]]:
    required_pair_set = {(symbol_str, "CAPITALSPECIAL") for symbol_str in FLOW_VOTE_CAPITAL_SYMBOL_TUPLE}
    required_pair_set.update((symbol_str, "TOTALRETURN") for symbol_str in FLOW_VOTE_TOTAL_RETURN_SYMBOL_TUPLE)
    return required_pair_set


def flow_vote_price_coverage_dict(price_df: pd.DataFrame) -> dict:
    coverage_dict = {}
    for (symbol_str, adjustment_str), symbol_price_df in price_df.groupby(["symbol_str", "adjustment_str"]):
        date_idx = pd.DatetimeIndex(pd.to_datetime(symbol_price_df["date"])).sort_values()
        coverage_dict[f"{symbol_str}|{adjustment_str}"] = {
            "first_date_str": date_idx[0].date().isoformat(),
            "last_date_str": date_idx[-1].date().isoformat(),
            "row_count_int": len(date_idx),
        }
    return coverage_dict


def validate_flow_vote_snapshot(price_df: pd.DataFrame, manifest_dict: dict, snapshot_date_ts: pd.Timestamp) -> None:
    """Reject missing lineage, altered units, a missing pair, a gap or an endpoint other than an observed Close_T."""
    if manifest_dict.get("schema_version") != 2:
        raise ValueError("Flow Vote requires schema v2.")
    contract_dict = manifest_dict.get("data_contract", {})
    if any(contract_dict.get(key_str) != value_obj for key_str, value_obj in FLOW_VOTE_DATA_CONTRACT_DICT.items()):
        raise ValueError("Flow Vote snapshot contract mismatch.")
    required_pair_set = flow_vote_required_pair_set()
    if set(price_df[["symbol_str", "adjustment_str"]].itertuples(index=False, name=None)) != required_pair_set:
        raise ValueError("Flow Vote requires the exact symbol/adjustment pairs.")
    price_date_ser = pd.to_datetime(price_df["date"])
    if (price_df.duplicated(["date", "symbol_str", "adjustment_str"]).any()
            or price_date_ser.isna().any() or price_date_ser.gt(snapshot_date_ts).any()
            or price_date_ser.lt(pd.Timestamp(FLOW_VOTE_HISTORY_START_STR)).any()):
        raise ValueError("Flow Vote has duplicate, future, or invalid history dates.")
    if contract_dict.get("series_coverage_dict") != flow_vote_price_coverage_dict(price_df):
        raise ValueError("Flow Vote history differs from declared source coverage.")
    calendar_obj = exchange_calendar_module.get_calendar("XNYS", start=FLOW_VOTE_HISTORY_START_STR, end=snapshot_date_ts)
    for symbol_str, adjustment_str in sorted(required_pair_set):
        pair_str = f"{symbol_str}|{adjustment_str}"
        source_field_list = contract_dict.get("source_field_by_pair_dict", {}).get(pair_str, [])
        source_dtype_dict = contract_dict.get("source_dtype_by_pair_dict", {}).get(pair_str, {})
        required_field_set = {"Open", "High", "Low", "Close"}
        if symbol_str != "$SPXTR":
            required_field_set.add("Dividend")
        if (not required_field_set.issubset(source_field_list)
                or not set(source_field_list).issubset(price_df.columns)
                or set(source_dtype_dict) != set(source_field_list)):
            raise ValueError(f"Flow Vote native field/dtype provenance missing for {pair_str}.")
        if any(contract_dict.get("source_request_by_pair_dict", {}).get(pair_str, {}).get(key_str) != value_obj for key_str, value_obj in {
                "requested_start_date_str": FLOW_VOTE_HISTORY_START_STR,
                "effective_start_date_str": FLOW_VOTE_HISTORY_START_STR,
                "end_date_str": snapshot_date_ts.date().isoformat(),
                "padding_type_name_str": "NONE",
                "empty_primary_request_bool": False,
        }.items()):
            raise ValueError(f"Flow Vote source request mismatch for {pair_str}.")
        symbol_price_df = price_df.loc[price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str)].copy()
        symbol_price_df["date"] = pd.to_datetime(symbol_price_df["date"])
        symbol_price_df = symbol_price_df.sort_values("date")
        source_date_idx = pd.DatetimeIndex(symbol_price_df["date"])
        # *** CRITICAL *** Unpadded rows: every exchange session from the first row to T must be an observed bar, and the
        # last one must be T itself. A carried or missing close can neither become a signal endpoint nor size a fill.
        if not source_date_idx.equals(calendar_obj.sessions_in_range(source_date_idx[0], snapshot_date_ts)):
            raise ValueError(f"Flow Vote has a missing session or a stale endpoint for {pair_str}.")
        for field_str in source_field_list:
            source_value_ser = symbol_price_df[field_str]
            try:
                restored_value_ser = source_value_ser.astype(source_dtype_dict[field_str])
                equal_value_ser = source_value_ser.eq(restored_value_ser) | (source_value_ser.isna() & restored_value_ser.isna())
                if not equal_value_ser.all():
                    raise ValueError("lossy native dtype")
            except (ValueError, TypeError, OverflowError) as error_obj:
                raise ValueError(f"Flow Vote invalid native dtype for {pair_str}/{field_str}.") from error_obj
        endpoint_ser = symbol_price_df.iloc[-1]
        if any(not np.isfinite(float(endpoint_ser[field_str])) or float(endpoint_ser[field_str]) <= 0 for field_str in ("Open", "High", "Low", "Close")):
            raise ValueError(f"Flow Vote invalid source endpoint for {pair_str}.")
