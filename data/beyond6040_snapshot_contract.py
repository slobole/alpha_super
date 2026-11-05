"""Frozen Beyond 60/40 trend-tilt (T1) snapshot contract; only used by the Beyond 60/40 profile.

The research loader (`get_trend_tilt_data`) reads VTI, TLT, GLD and BIL CAPITALSPECIAL, VTI, TLT and GLD TOTALRETURN and
the `$SPX`/`$SPXTR` benchmark from Norgate from 2001-05-31 (VTI inception). The snapshot keeps exactly that: native fields,
native dtypes, observed rows only (PaddingType.NONE), so the live frame equals the research frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import exchange_calendars as exchange_calendar_module

BEYOND6040_PROFILE_STR = "norgate_eod_beyond6040"
BEYOND6040_HISTORY_START_STR = "2001-05-31"
BEYOND6040_CAPITAL_SYMBOL_TUPLE = ("VTI", "TLT", "GLD", "BIL")
BEYOND6040_TOTAL_RETURN_SYMBOL_TUPLE = ("VTI", "TLT", "GLD", "$SPX", "$SPXTR")
BEYOND6040_DATA_CONTRACT_DICT = {
    "history_start_date_str": BEYOND6040_HISTORY_START_STR,
    "price_padding_setting_str": "NONE",
    "past_member_tail_policy_str": "exact",
}


def beyond6040_required_pair_set() -> set[tuple[str, str]]:
    required_pair_set = {(symbol_str, "CAPITALSPECIAL") for symbol_str in BEYOND6040_CAPITAL_SYMBOL_TUPLE}
    required_pair_set.update((symbol_str, "TOTALRETURN") for symbol_str in BEYOND6040_TOTAL_RETURN_SYMBOL_TUPLE)
    return required_pair_set


def beyond6040_price_coverage_dict(price_df: pd.DataFrame) -> dict:
    coverage_dict = {}
    for (symbol_str, adjustment_str), symbol_price_df in price_df.groupby(["symbol_str", "adjustment_str"]):
        date_idx = pd.DatetimeIndex(pd.to_datetime(symbol_price_df["date"])).sort_values()
        coverage_dict[f"{symbol_str}|{adjustment_str}"] = {
            "first_date_str": date_idx[0].date().isoformat(),
            "last_date_str": date_idx[-1].date().isoformat(),
            "row_count_int": len(date_idx),
        }
    return coverage_dict


def validate_beyond6040_snapshot(price_df: pd.DataFrame, manifest_dict: dict, snapshot_date_ts: pd.Timestamp) -> None:
    """Reject missing lineage, altered units, a missing pair, a gap or an endpoint other than an observed Close_T."""
    if manifest_dict.get("schema_version") != 2:
        raise ValueError("Beyond 60/40 requires schema v2.")
    contract_dict = manifest_dict.get("data_contract", {})
    if any(contract_dict.get(key_str) != value_obj for key_str, value_obj in BEYOND6040_DATA_CONTRACT_DICT.items()):
        raise ValueError("Beyond 60/40 snapshot contract mismatch.")
    required_pair_set = beyond6040_required_pair_set()
    if set(price_df[["symbol_str", "adjustment_str"]].itertuples(index=False, name=None)) != required_pair_set:
        raise ValueError("Beyond 60/40 requires the exact symbol/adjustment pairs.")
    price_date_ser = pd.to_datetime(price_df["date"])
    if (price_df.duplicated(["date", "symbol_str", "adjustment_str"]).any()
            or price_date_ser.isna().any() or price_date_ser.gt(snapshot_date_ts).any()
            or price_date_ser.lt(pd.Timestamp(BEYOND6040_HISTORY_START_STR)).any()):
        raise ValueError("Beyond 60/40 has duplicate, future, or invalid history dates.")
    if contract_dict.get("series_coverage_dict") != beyond6040_price_coverage_dict(price_df):
        raise ValueError("Beyond 60/40 history differs from declared source coverage.")
    calendar_obj = exchange_calendar_module.get_calendar("XNYS", start=BEYOND6040_HISTORY_START_STR, end=snapshot_date_ts)
    for symbol_str, adjustment_str in sorted(required_pair_set):
        pair_str = f"{symbol_str}|{adjustment_str}"
        source_field_list = contract_dict.get("source_field_by_pair_dict", {}).get(pair_str, [])
        source_dtype_dict = contract_dict.get("source_dtype_by_pair_dict", {}).get(pair_str, {})
        required_field_set = {"Open", "High", "Low", "Close"}
        if not symbol_str.startswith("$"):
            required_field_set.add("Dividend")
        if (not required_field_set.issubset(source_field_list)
                or not set(source_field_list).issubset(price_df.columns)
                or set(source_dtype_dict) != set(source_field_list)):
            raise ValueError(f"Beyond 60/40 native field/dtype provenance missing for {pair_str}.")
        if any(contract_dict.get("source_request_by_pair_dict", {}).get(pair_str, {}).get(key_str) != value_obj for key_str, value_obj in {
                "requested_start_date_str": BEYOND6040_HISTORY_START_STR,
                "effective_start_date_str": BEYOND6040_HISTORY_START_STR,
                "end_date_str": snapshot_date_ts.date().isoformat(),
                "padding_type_name_str": "NONE",
                "empty_primary_request_bool": False,
        }.items()):
            raise ValueError(f"Beyond 60/40 source request mismatch for {pair_str}.")
        symbol_price_df = price_df.loc[price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str)].copy()
        symbol_price_df["date"] = pd.to_datetime(symbol_price_df["date"])
        symbol_price_df = symbol_price_df.sort_values("date")
        source_date_idx = pd.DatetimeIndex(symbol_price_df["date"])
        # *** CRITICAL *** Unpadded rows: every exchange session from the first row (each ETF's own inception) to T must be
        # an observed bar, and the last one must be T itself. A carried or missing close can neither become a signal
        # endpoint nor size a fill.
        if not source_date_idx.equals(calendar_obj.sessions_in_range(source_date_idx[0], snapshot_date_ts)):
            raise ValueError(f"Beyond 60/40 has a missing session or a stale endpoint for {pair_str}.")
        for field_str in source_field_list:
            source_value_ser = symbol_price_df[field_str]
            try:
                restored_value_ser = source_value_ser.astype(source_dtype_dict[field_str])
                equal_value_ser = source_value_ser.eq(restored_value_ser) | (source_value_ser.isna() & restored_value_ser.isna())
                if not equal_value_ser.all():
                    raise ValueError("lossy native dtype")
            except (ValueError, TypeError, OverflowError) as error_obj:
                raise ValueError(f"Beyond 60/40 invalid native dtype for {pair_str}/{field_str}.") from error_obj
        endpoint_ser = symbol_price_df.iloc[-1]
        if any(not np.isfinite(float(endpoint_ser[field_str])) or float(endpoint_ser[field_str]) <= 0 for field_str in ("Open", "High", "Low", "Close")):
            raise ValueError(f"Beyond 60/40 invalid source endpoint for {pair_str}.")
