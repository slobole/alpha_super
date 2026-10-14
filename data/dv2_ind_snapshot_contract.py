"""Frozen industry-ETF snapshot contract; only used by the DV2-IND profile."""
from __future__ import annotations

import numpy as np
import pandas as pd
import exchange_calendars as exchange_calendar_module

DV2_IND_PROFILE_STR = "norgate_eod_dv2_industry_etf"
DV2_IND_HISTORY_START_STR = "2009-01-01"
DV2_IND_ETF_TUPLE = (
    "XBI", "IBB", "SMH", "SOXX", "KRE", "KBE", "XHB", "ITB", "XRT", "XOP",
    "OIH", "XME", "GDX", "IYT", "IGV", "ITA", "IHI", "XSD", "XPH",
)
DV2_IND_DATA_CONTRACT_DICT = {
    "history_start_date_str": DV2_IND_HISTORY_START_STR,
    "price_padding_setting_str": "ALLMARKETDAYS",
    "past_member_tail_policy_str": "exact",
    "liquidity_source_str": "native_Turnover",
}


def dv2_ind_price_coverage_dict(price_df: pd.DataFrame) -> dict:
    coverage_dict = {}
    for (symbol_str, adjustment_str), symbol_price_df in price_df.groupby(["symbol_str", "adjustment_str"]):
        date_idx = pd.DatetimeIndex(pd.to_datetime(symbol_price_df["date"])).sort_values()
        coverage_dict[f"{symbol_str}|{adjustment_str}"] = {
            "first_date_str": date_idx[0].date().isoformat(),
            "last_date_str": date_idx[-1].date().isoformat(),
            "row_count_int": len(date_idx),
        }
    return coverage_dict


def validate_dv2_ind_snapshot(price_df: pd.DataFrame, manifest_dict: dict, snapshot_date_ts: pd.Timestamp) -> None:
    """Reject missing lineage, altered units, missing ETFs, and stale endpoints."""
    if manifest_dict.get("schema_version") != 2:
        raise ValueError("DV2-IND requires schema v2.")
    contract_dict = manifest_dict.get("data_contract", {})
    if any(contract_dict.get(key_str) != value_obj for key_str, value_obj in DV2_IND_DATA_CONTRACT_DICT.items()):
        raise ValueError("DV2-IND snapshot contract mismatch.")
    required_pair_set = {(symbol_str, "CAPITALSPECIAL") for symbol_str in DV2_IND_ETF_TUPLE}
    required_pair_set.add(("$SPXTR", "TOTALRETURN"))
    if set(price_df[["symbol_str", "adjustment_str"]].itertuples(index=False, name=None)) != required_pair_set:
        raise ValueError("DV2-IND requires the exact ETF/adjustment pairs and true total-return benchmark.")
    price_date_ser = pd.to_datetime(price_df["date"])
    if (price_df.duplicated(["date", "symbol_str", "adjustment_str"]).any()
            or price_date_ser.isna().any() or price_date_ser.gt(snapshot_date_ts).any()
            or price_date_ser.lt(pd.Timestamp(DV2_IND_HISTORY_START_STR)).any()):
        raise ValueError("DV2-IND has duplicate, future, or invalid history dates.")
    if contract_dict.get("series_coverage_dict") != dv2_ind_price_coverage_dict(price_df):
        raise ValueError("DV2-IND history differs from declared source coverage.")
    calendar_obj = exchange_calendar_module.get_calendar("XNYS", start=DV2_IND_HISTORY_START_STR, end=snapshot_date_ts)
    for symbol_str, adjustment_str in sorted(required_pair_set):
        pair_str = f"{symbol_str}|{adjustment_str}"
        source_field_list = contract_dict.get("source_field_by_pair_dict", {}).get(pair_str, [])
        source_dtype_dict = contract_dict.get("source_dtype_by_pair_dict", {}).get(pair_str, {})
        required_field_set = {"Open", "High", "Low", "Close"}
        if symbol_str in DV2_IND_ETF_TUPLE:
            required_field_set |= {"Turnover", "Unadjusted Close", "Dividend"}
        if (not required_field_set.issubset(source_field_list)
                or not set(source_field_list).issubset(price_df.columns)
                or set(source_dtype_dict) != set(source_field_list)):
            raise ValueError(f"DV2-IND native field/dtype provenance missing for {symbol_str}.")
        symbol_price_df = price_df.loc[price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str)].copy()
        symbol_price_df["date"] = pd.to_datetime(symbol_price_df["date"])
        symbol_price_df = symbol_price_df.sort_values("date")
        source_request_dict = contract_dict.get("source_request_by_pair_dict", {}).get(pair_str, {})
        if any(source_request_dict.get(key_str) != value_obj for key_str, value_obj in {
                "requested_start_date_str": DV2_IND_HISTORY_START_STR,
                "effective_start_date_str": DV2_IND_HISTORY_START_STR,
                "end_date_str": snapshot_date_ts.date().isoformat(),
                "padding_type_name_str": "ALLMARKETDAYS",
                "empty_primary_request_bool": False,
        }.items()):
            raise ValueError(f"DV2-IND source request mismatch for {symbol_str}.")
        source_date_idx = pd.DatetimeIndex(symbol_price_df["date"])
        if not source_date_idx.equals(calendar_obj.sessions_in_range(source_date_idx[0], snapshot_date_ts)):
            raise ValueError(f"DV2-IND has an interior source history gap for {symbol_str}.")
        # *** CRITICAL *** Close_T freshness must come from an unpadded vendor
        # observation, never from a carried price. Features end at T; fill T+1.
        if (symbol_price_df["date"].iloc[-1] != snapshot_date_ts
                or contract_dict.get("observed_endpoint_date_by_symbol_dict", {}).get(symbol_str) != snapshot_date_ts.date().isoformat()):
            raise ValueError(f"DV2-IND stale source endpoint for {symbol_str}.")
        for field_str in source_field_list:
            source_value_ser = symbol_price_df[field_str]
            try:
                restored_value_ser = source_value_ser.astype(source_dtype_dict[field_str])
                equal_value_ser = source_value_ser.eq(restored_value_ser) | (source_value_ser.isna() & restored_value_ser.isna())
                if not equal_value_ser.all():
                    raise ValueError("lossy native dtype")
            except (ValueError, TypeError, OverflowError) as error_obj:
                raise ValueError(f"DV2-IND invalid native dtype for {symbol_str}/{field_str}.") from error_obj
        endpoint_ser = symbol_price_df.iloc[-1]
        endpoint_field_set = {"Open", "High", "Low", "Close"}
        if symbol_str in DV2_IND_ETF_TUPLE:
            endpoint_field_set |= {"Unadjusted Close", "Turnover"}
        if any(not np.isfinite(float(endpoint_ser[field_str])) or float(endpoint_ser[field_str]) <= 0 for field_str in endpoint_field_set):
            raise ValueError(f"DV2-IND invalid source endpoint for {symbol_str}.")
