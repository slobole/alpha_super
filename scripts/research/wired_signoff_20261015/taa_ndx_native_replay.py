"""Research-only actual-host monthly replay; never connects to a broker.

TAA reads native prices once per symbol/adjustment and frozen local FRED bytes.
NDX reads a previously exported, validated native snapshot, then compares the
unchanged host with the full-history research classes. Historical E2 metadata
is reconstructed over ONE current-sector map: this tests mechanics, not PIT
classifications or historical producer attestations. No run_variant is run.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
NEW_YORK_TZ = ZoneInfo("America/New_York")
END_DATE_STR = "2026-10-14"
NDX_SOURCE_SNAPSHOT_DATE_STR = "2026-09-30"
VARIANT_DICT = {
    "taa_rank_tqqq": ("taa", "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "norgate_eod_etf_plus_vix_helper"),
    "taa_equal_tqqq": ("taa", "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "norgate_eod_etf_plus_vix_helper"),
    "taa_linearity_qqq": ("taa", "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "norgate_eod_etf_plus_vix_helper"),
    "ndx_atr": ("ndx", "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy", "norgate_eod_ndx_pit"),
    "ndx_atr_vxn": ("ndx", "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy", "norgate_eod_ndx_pit_plus_vxn_helper"),
    "ndx_e2_atr": ("ndx", "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap:SectorCapVxnScaledAtrNormalizedNdxStrategy", "norgate_eod_ndx_e2_pit_plus_vxn_helper"),
    "ndx_e2_natr20": ("ndx", "strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap:SectorCapNatr20VxnScaledNdxStrategy", "norgate_eod_ndx_e2_pit_plus_vxn_helper"),
}
STRESS_MONTH_SET = {"2008-09", "2008-10", "2009-03", "2011-08", "2015-08", "2018-12", "2020-02", "2020-03", "2020-04", "2022-01", "2022-06"}


def build_release_obj(variant_str: str, capital_float: float):
    from alpha.live.models import LiveRelease
    _, import_str, profile_str = VARIANT_DICT[variant_str]
    return LiveRelease(release_id_str=f"audit.{variant_str}", user_id_str="audit", pod_id_str=variant_str,
        account_route_str="DU_AUDIT_DISABLED", strategy_import_str=import_str, mode_str="paper",
        session_calendar_id_str="XNYS", signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open", data_profile_str=profile_str,
        params_dict={"capital_base_float": capital_float}, risk_profile_str="audit_only",
        enabled_bool=False, source_path_str="isolated_audit", pod_budget_fraction_float=1.0)


def selected_date_list(weight_by_date_dict: dict, minimum_int: int) -> list[pd.Timestamp]:
    informative_date_list = [date_ts for date_ts, weight_dict in weight_by_date_dict.items() if weight_dict]
    cash_date_list = [date_ts for date_ts, weight_dict in weight_by_date_dict.items() if not weight_dict]
    stress_date_list = [date_ts for date_ts in weight_by_date_dict if str(date_ts)[:7] in STRESS_MONTH_SET]
    return sorted(set(informative_date_list[-minimum_int:] + cash_date_list[-8:] + stress_date_list))


def normalized_weight_dict(weight_ser) -> dict[str, float]:
    return {str(symbol_str): float(weight_float) for symbol_str, weight_float in weight_ser.items()
        if abs(float(weight_float)) > 1e-12}


def replay_host(variant_str: str, capital_float: float, weight_by_date_dict: dict,
        execution_by_date_dict: dict, minimum_int: int, metadata_fn=None) -> dict:
    from alpha.live import strategy_host
    release_obj = build_release_obj(variant_str, capital_float)
    row_list = []
    for date_ts in selected_date_list(weight_by_date_dict, minimum_int):
        if metadata_fn is not None:
            metadata_fn(date_ts)
        as_of_ts = datetime(date_ts.year, date_ts.month, date_ts.day, 20, 0, tzinfo=NEW_YORK_TZ)
        expected_dict = weight_by_date_dict[date_ts]
        row_dict = {"decision_date_str": date_ts.date().isoformat(), "expected_weight_dict": expected_dict,
            "informative_bool": bool(expected_dict)}
        try:
            plan_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None)
            actual_dict = dict(plan_obj.full_target_weight_map_dict)
            asset_set = set(expected_dict) | set(actual_dict)
            difference_float = max([abs(expected_dict.get(symbol_str, 0) - actual_dict.get(symbol_str, 0))
                for symbol_str in asset_set] or [0])
            signal_str = pd.Timestamp(plan_obj.signal_timestamp_ts).tz_convert(NEW_YORK_TZ).date().isoformat()
            execution_str = pd.Timestamp(plan_obj.target_execution_timestamp_ts).tz_convert(NEW_YORK_TZ).date().isoformat()
            row_dict.update(status_str="ok", actual_weight_dict=actual_dict, maximum_weight_difference_float=difference_float,
                signal_date_match_bool=signal_str == date_ts.date().isoformat(),
                execution_date_match_bool=execution_str == execution_by_date_dict[date_ts].date().isoformat(),
                helper_date_str=plan_obj.snapshot_metadata_dict.get("vxn_reference_date_str"),
                cash_weight_float=plan_obj.cash_reserve_weight_float)
        except Exception as exception_obj:
            row_dict.update(status_str="error", error_str=f"{type(exception_obj).__name__}: {exception_obj}")
        row_list.append(row_dict)
        print(variant_str, row_dict["decision_date_str"], row_dict["status_str"], row_dict.get("maximum_weight_difference_float"), flush=True)
    success_list = [row_dict for row_dict in row_list if row_dict["status_str"] == "ok"]
    exact_list = [row_dict for row_dict in success_list if row_dict["maximum_weight_difference_float"] <= 1e-9
        and row_dict["signal_date_match_bool"] and row_dict["execution_date_match_bool"]]
    informative_exact_int = sum(row_dict["informative_bool"] for row_dict in exact_list)
    return {"variant_str": variant_str, "strategy_import_str": release_obj.strategy_import_str,
        "model_budget_float": capital_float, "row_list": row_list, "attempted_int": len(row_list),
        "successful_int": len(success_list), "exact_int": len(exact_list),
        "informative_exact_int": informative_exact_int, "minimum_informative_int": minimum_int,
        "coverage_pass_bool": informative_exact_int >= minimum_int,
        "mechanics_pass_bool": len(exact_list) == len(row_list) and informative_exact_int >= minimum_int}


def replay_taa(arguments_obj) -> dict:
    import alpha.data.fred_loader as fred_module
    from strategies.taa_df import strategy_taa_df as base_module
    from strategies.taa_df import strategy_taa_df_btal_linearity_1n as linearity_module
    from strategies.taa_df import strategy_taa_df_fallback_vix_cash_variant_utils as overlay_module
    import data.norgate_loader as loader_module
    from alpha.live import scheduler_utils
    os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "false"

    original_loader_fn = loader_module.load_price_timeseries
    loader_signature_obj = inspect.signature(original_loader_fn)
    price_cache_dict = {}
    cache_path_obj = arguments_obj.output_dir / "DTB3_frozen_audit.csv"
    cache_path_obj.write_bytes(arguments_obj.fred_cache.read_bytes())

    def offline_response_fn(*argument_list, **keyword_dict):
        return io.BytesIO(cache_path_obj.read_bytes())

    def cached_price_loader_fn(*argument_list, **keyword_dict):
        argument_map = loader_signature_obj.bind_partial(*argument_list, **keyword_dict)
        requested_end_obj = argument_map.arguments.get("end_date", argument_map.arguments.get("end_date_str"))
        end_parameter_str = "end_date_str" if "end_date_str" in loader_signature_obj.parameters else "end_date"
        argument_map.arguments[end_parameter_str] = END_DATE_STR
        cache_key_str = repr(argument_map.arguments)
        if cache_key_str not in price_cache_dict:
            price_cache_dict[cache_key_str] = original_loader_fn(*argument_map.args, **argument_map.kwargs)
        full_price_df = price_cache_dict[cache_key_str]
        # *** CRITICAL *** Cached future observations are sliced BEFORE the unchanged
        # strategy loader/feature code sees them: information ends at Close_T.
        return full_price_df.loc[:pd.Timestamp(requested_end_obj or END_DATE_STR)].copy()

    variant_module_list = [import_module(import_str) for family_str, import_str, _ in VARIANT_DICT.values() if family_str == "taa"]
    for module_obj in variant_module_list:
        module_obj.DEFAULT_CONFIG = replace(module_obj.DEFAULT_CONFIG, dtb3_csv_path_str=str(cache_path_obj))
    family_result_dict = {}
    with patch.object(fred_module, "urlopen", offline_response_fn), \
            patch.object(base_module, "load_price_timeseries", cached_price_loader_fn), \
            patch.object(overlay_module, "load_price_timeseries", cached_price_loader_fn):
        for variant_str, (family_str, import_str, _) in VARIANT_DICT.items():
            if family_str != "taa":
                continue
            module_obj = import_module(import_str)
            config_obj = replace(module_obj.DEFAULT_CONFIG, end_date_str=END_DATE_STR)
            if variant_str == "taa_linearity_qqq":
                result_tuple = overlay_module.get_linearity_1n_fallback_vix_cash_data(config_obj, linearity_module.get_defense_first_linearity_1n_data)
                month_weight_df = result_tuple[4]
            else:
                result_tuple = overlay_module.get_standard_fallback_vix_cash_data(config_obj, base_module.get_defense_first_data)
                month_weight_df = result_tuple[3]
            pricing_df = result_tuple[0]
            weight_by_date_dict, execution_by_date_dict = {}, {}
            final_decision_ts = scheduler_utils.get_latest_completed_month_end_session_label_ts(
                datetime.fromisoformat(END_DATE_STR).replace(hour=23, tzinfo=NEW_YORK_TZ), "XNYS")
            for month_ts, weight_ser in month_weight_df.iterrows():
                session_idx = pricing_df.index[pricing_df.index.to_period("M") == month_ts.to_period("M")]
                date_ts = pd.Timestamp(session_idx[-1])
                if date_ts > pd.Timestamp(final_decision_ts):
                    continue
                weight_by_date_dict[date_ts] = normalized_weight_dict(weight_ser)
                execution_by_date_dict[date_ts] = pd.Timestamp(scheduler_utils.next_business_day_timestamp_ts(date_ts, "XNYS").date())
            family_result_dict[variant_str] = replay_host(variant_str, arguments_obj.capital,
                weight_by_date_dict, execution_by_date_dict, arguments_obj.minimum_dates)
    return {"source_str": "native_prices_plus_frozen_current_vintage_FRED", "fred_sha256_str": hashlib.sha256(cache_path_obj.read_bytes()).hexdigest(),
        "variant_result_dict": family_result_dict, "limitation_list": ["Offline FRED transport uses frozen current-vintage bytes, not archived publication vintages.",
            "Native raw reads are cached then causally truncated. Host entrypoint and strategy feature calculations are unchanged."]}


def replay_ndx(arguments_obj) -> dict:
    from alpha.live import strategy_host
    from data import norgate_snapshot_store as snapshot_module
    from strategies.momentum import strategy_mo_atr_normalized_ndx as atr_module
    from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module
    from strategies.momentum import strategy_mo_natr20_ndx_vxn_scaled as natr_module
    from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap as cap_module
    snapshot_root_path = arguments_obj.ndx_snapshot_root
    os.environ[snapshot_module.NORGATE_SNAPSHOT_ROOT_ENV_STR] = str(snapshot_root_path)
    os.environ[snapshot_module.ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR] = "true"
    with snapshot_module.use_norgate_data_profile(snapshot_module.NDX_E2_PROFILE_STR):
        manifest_obj = snapshot_module.load_valid_snapshot_manifest(snapshot_module.NDX_E2_PROFILE_STR, snapshot_date_str=NDX_SOURCE_SNAPSHOT_DATE_STR)
        sector_dict = dict(manifest_obj.manifest_dict["data_contract"]["sector_by_symbol_dict"])
        full_atr_tuple = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(replace(vxn_module.DEFAULT_CONFIG, end_date_str=NDX_SOURCE_SNAPSHOT_DATE_STR))
        full_natr_tuple = natr_module.get_natr20_vxn_scaled_ndx_data(replace(natr_module.DEFAULT_CONFIG, end_date_str=NDX_SOURCE_SNAPSHOT_DATE_STR))
    os.environ[snapshot_module.ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR] = "false"
    source_manifest_hash_str = manifest_obj.manifest_hash_str
    historical_metadata_dict = {}

    def metadata_fn(date_ts):
        historical_metadata_dict.update(norgate_data_source_mode_str="snapshot",
            norgate_snapshot_date_str=date_ts.date().isoformat(), norgate_manifest_hash_str=source_manifest_hash_str,
            norgate_data_profile_str=snapshot_module.NDX_E2_PROFILE_STR)

    def metadata_reader_fn(profile_str):
        return {**historical_metadata_dict, "norgate_data_profile_str": profile_str}

    def frozen_map_fn(symbol_list, **keyword_dict):
        return {symbol_str: sector_dict[symbol_str] for symbol_str in symbol_list}

    family_result_dict = {}
    for variant_str, (family_str, import_str, _) in VARIANT_DICT.items():
        if family_str != "ndx":
            continue
        module_name_str, class_name_str = import_str.split(":")
        module_obj = import_module(module_name_str)
        full_tuple = full_natr_tuple if variant_str == "ndx_e2_natr20" else full_atr_tuple
        full_pricing_df, full_universe_df, source_schedule_df, full_vxn_df = full_tuple
        schedule_df = source_schedule_df.copy()
        # *** CRITICAL *** The last completed month-end is a valid decision even
        # when the frozen source contains no next-month execution bar. Only the
        # exchange calendar supplies T+1 here; no future price is manufactured.
        final_date_ts = pd.Timestamp(full_pricing_df.index[-1])
        final_execution_ts = pd.Timestamp(strategy_host.scheduler_utils.next_business_day_timestamp_ts(final_date_ts, "XNYS").date())
        schedule_df.loc[final_execution_ts, "decision_date_ts"] = final_date_ts
        config_obj = replace(module_obj.DEFAULT_CONFIG, capital_base_float=arguments_obj.capital, end_date_str=END_DATE_STR)
        e2_bool = "e2" in variant_str
        vxn_bool = variant_str != "ndx_atr"
        if e2_bool:
            reference_obj = cap_module.build_strategy(getattr(module_obj, class_name_str), variant_str, config_obj,
                schedule_df, full_vxn_df, full_universe_df, sector_dict)
        else:
            strategy_keyword_dict = {"name": variant_str, "benchmarks": ["SPY"], "rebalance_schedule_df": schedule_df,
                "regime_symbol_str": "SPY", "capital_base": arguments_obj.capital,
                "lookback_month_int": config_obj.lookback_month_int, "index_trend_window_int": config_obj.index_trend_window_int,
                "stock_trend_window_int": config_obj.stock_trend_window_int, "max_positions_int": config_obj.max_positions_int}
            if vxn_bool:
                strategy_keyword_dict["vxn_scale_signal_df"] = full_vxn_df
            reference_obj = getattr(module_obj, class_name_str)(**strategy_keyword_dict)
            reference_obj.universe_df = full_universe_df
        signal_df = reference_obj.compute_signals(full_pricing_df.copy())
        weight_by_date_dict, execution_by_date_dict = {}, {}
        for execution_ts, schedule_ser in schedule_df.iterrows():
            date_ts = pd.Timestamp(schedule_ser["decision_date_ts"])
            reference_obj.previous_bar = date_ts
            reference_obj.current_bar = pd.Timestamp(execution_ts)
            weight_by_date_dict[date_ts] = normalized_weight_dict(reference_obj.get_target_weight_ser(signal_df.loc[date_ts]))
            execution_by_date_dict[date_ts] = pd.Timestamp(execution_ts)

        def frozen_loader_fn(config_obj, **keyword_dict):
            # *** CRITICAL *** Apply both requested history start and Close_T end
            # before actual host features run, including its new 36-month E2 window.
            start_ts = pd.Timestamp(config_obj.history_start_date_str)
            end_ts = pd.Timestamp(config_obj.end_date_str)
            pricing_df = full_pricing_df.loc[start_ts:end_ts].copy()
            universe_df = full_universe_df.loc[start_ts:end_ts].copy()
            result_tuple = (pricing_df, universe_df, schedule_df.loc[:end_ts].copy())
            return (*result_tuple, full_vxn_df.loc[start_ts:end_ts].copy()) if vxn_bool else result_tuple

        loader_name_str = ("get_natr20_vxn_scaled_ndx_data" if variant_str == "ndx_e2_natr20"
            else "get_vxn_scaled_atr_normalized_ndx_data" if vxn_bool else "get_atr_normalized_ndx_data")
        with patch.object(module_obj, loader_name_str, frozen_loader_fn), \
                patch.object(strategy_host, "build_data_source_metadata_dict", metadata_reader_fn), \
                patch.object(snapshot_module, "build_data_source_metadata_dict", metadata_reader_fn), \
                patch.object(snapshot_module, "is_snapshot_mode_enabled_bool", lambda: e2_bool), \
                patch.object(snapshot_module, "load_valid_snapshot_manifest", lambda *_, **__: manifest_obj), \
                patch.object(snapshot_module, "load_ndx_e2_sector_map", frozen_map_fn):
            family_result_dict[variant_str] = replay_host(variant_str, arguments_obj.capital,
                weight_by_date_dict, execution_by_date_dict, arguments_obj.minimum_dates, metadata_fn)
    return {"source_str": "previous_native_E2_snapshot_reconstruction", "source_snapshot_date_str": NDX_SOURCE_SNAPSHOT_DATE_STR,
        "source_manifest_hash_str": source_manifest_hash_str,
        "variant_result_dict": family_result_dict, "limitation_list": [
            "Historical per-T manifests are reconstructed in memory over one previously validated native snapshot.",
            "Current GICS map is reused historically; this is diagnostic mechanics parity, never PIT-sector evidence.",
            "Full-history expected targets are compared to the unchanged actual host, including E2's 36-month decision window.",
            "Native provider/exporter and actual historical snapshot freshness attestations are not replayed."]}


def main() -> None:
    global END_DATE_STR
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--family", choices=("taa", "ndx"), required=True)
    parser_obj.add_argument("--capital", type=float, default=1_000_000)
    parser_obj.add_argument("--minimum-dates", type=int, default=24)
    parser_obj.add_argument("--end-date", default=END_DATE_STR)
    parser_obj.add_argument("--output-dir", type=Path, default=REPO_ROOT_PATH / "results/review/wired_signoff_20261015/taa_ndx_native")
    parser_obj.add_argument("--fred-cache", type=Path, default=REPO_ROOT_PATH / "results/review/wired_signoff_20261015/inputs/DTB3.csv")
    parser_obj.add_argument("--ndx-snapshot-root", type=Path, default=REPO_ROOT_PATH / "results/review/e2_memory_20261014/refresh_snapshot_root")
    arguments_obj = parser_obj.parse_args()
    END_DATE_STR = arguments_obj.end_date
    arguments_obj.output_dir.mkdir(parents=True, exist_ok=True)
    started_float = time.monotonic()
    result_dict = replay_taa(arguments_obj) if arguments_obj.family == "taa" else replay_ndx(arguments_obj)
    result_dict.update(family_str=arguments_obj.family, elapsed_seconds_float=time.monotonic() - started_float,
        script_sha256_str=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output_path_obj = arguments_obj.output_dir / f"{arguments_obj.family}_actual_host_replay.json"
    output_path_obj.write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output_path_str": str(output_path_obj), "elapsed_seconds_float": result_dict["elapsed_seconds_float"],
        "variant_summary_dict": {variant_str: {key_str: value_obj for key_str, value_obj in result_obj.items() if key_str != "row_list"}
            for variant_str, result_obj in result_dict["variant_result_dict"].items()}}, default=str), flush=True)


if __name__ == "__main__":
    main()
