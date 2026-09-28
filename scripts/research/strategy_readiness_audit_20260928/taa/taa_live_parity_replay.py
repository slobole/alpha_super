"""TAA family live-host parity replay (audit protocol checks B1/B2).

For every historical month-end decision session T, call the real live entry point
``alpha.live.strategy_host.build_decision_plan_for_release`` with ``as_of_ts`` = T 20:00 New York
(so every loader truncates its data at T), and compare the live full-target weights with the
backtest's rebalance row for the first session of the next month.

Study-code only. Runtime patches (no repository file is changed):
- the DTB3 cache path of each variant's DEFAULT_CONFIG points into the results folder, so the
  owner's shared ``1_data/DTB3.csv`` cache is never written;
- ``alpha.data.fred_loader.urlopen`` serves one FRED download captured at start-up, so the
  replay makes one network call instead of hundreds. The live code's own ``<= as_of_date``
  filter still truncates the series at every T.

Usage: uv run python scripts/research/strategy_readiness_audit_20260928/taa/taa_live_parity_replay.py [variant ...]
"""

from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from importlib import import_module
from pathlib import Path
from urllib.request import urlopen as real_urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import alpha.data.fred_loader as fred_loader_module  # noqa: E402
from alpha.live import strategy_host  # noqa: E402
from alpha.live.models import LiveRelease  # noqa: E402

OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/taa"
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
DTB3_CACHE_PATH = OUTPUT_DIR_PATH / "DTB3_audit_cache.csv"
END_DATE_STR = "2026-09-25"
FIRST_REBALANCE_STR = "2012-10-01"
NEW_YORK_TZ = ZoneInfo("America/New_York")

VARIANT_DICT = {
    "taa3x": {
        "module_str": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        "data_profile_str": "norgate_eod_etf_plus_vix_helper",
        "kind_str": "standard",
    },
    "taa1n": {
        "module_str": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
        "data_profile_str": "norgate_eod_etf_plus_vix_helper",
        "kind_str": "standard",
    },
    "btal_qqq": {
        "module_str": "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
        "data_profile_str": "norgate_eod_etf_plus_vix_helper",
        "kind_str": "linearity",
    },
}


class _CachedResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


_FRED_CACHE_DICT: dict[str, bytes] = {}


def _cached_urlopen(url_str, timeout=None):
    if url_str not in _FRED_CACHE_DICT:
        with real_urlopen(url_str, timeout=timeout) as response_obj:
            _FRED_CACHE_DICT[url_str] = response_obj.read()
    return _CachedResponse(_FRED_CACHE_DICT[url_str])


def _patch_environment() -> None:
    fred_loader_module.urlopen = _cached_urlopen
    for variant_dict in VARIANT_DICT.values():
        module_obj = import_module(variant_dict["module_str"])
        module_obj.DEFAULT_CONFIG = replace(module_obj.DEFAULT_CONFIG, dtb3_csv_path_str=str(DTB3_CACHE_PATH))


def _build_release(variant_key_str: str) -> LiveRelease:
    variant_dict = VARIANT_DICT[variant_key_str]
    return LiveRelease(
        release_id_str=f"audit.{variant_key_str}",
        user_id_str="audit_user",
        pod_id_str=f"audit_{variant_key_str}",
        account_route_str="U00000000",
        strategy_import_str=variant_dict["module_str"],
        mode_str="live",
        session_calendar_id_str="XNYS",
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
        data_profile_str=variant_dict["data_profile_str"],
        params_dict={"capital_base_float": 100_000.0},
        risk_profile_str="audit",
        enabled_bool=False,
        source_path_str="audit",
    )


def _backtest_rebalance_weight_df(variant_key_str: str) -> pd.DataFrame:
    variant_dict = VARIANT_DICT[variant_key_str]
    module_obj = import_module(variant_dict["module_str"])
    utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
    config_obj = replace(module_obj.DEFAULT_CONFIG, end_date_str=END_DATE_STR)
    if variant_dict["kind_str"] == "standard":
        base_module = import_module("strategies.taa_df.strategy_taa_df")
        result_tuple = utils_module.get_standard_fallback_vix_cash_data(
            config=config_obj, base_data_loader_fn=base_module.get_defense_first_data
        )
        rebalance_weight_df = result_tuple[4]
    else:
        linearity_module = import_module("strategies.taa_df.strategy_taa_df_btal_linearity_1n")
        result_tuple = utils_module.get_linearity_1n_fallback_vix_cash_data(
            config=config_obj, base_data_loader_fn=linearity_module.get_defense_first_linearity_1n_data
        )
        rebalance_weight_df = result_tuple[5]
    return rebalance_weight_df


def _decision_session_for_rebalance(rebalance_ts: pd.Timestamp, session_index: pd.DatetimeIndex) -> pd.Timestamp:
    prior_session_index = session_index[session_index < rebalance_ts]
    return pd.Timestamp(prior_session_index[-1])


def replay_variant(variant_key_str: str) -> dict:
    release_obj = _build_release(variant_key_str)
    rebalance_weight_df = _backtest_rebalance_weight_df(variant_key_str)
    import exchange_calendars

    xnys_obj = exchange_calendars.get_calendar("XNYS", start="2010-01-01", end="2027-12-31")
    session_index = pd.DatetimeIndex(xnys_obj.sessions).tz_localize(None)

    row_list: list[dict] = []
    rebalance_index = rebalance_weight_df.index[rebalance_weight_df.index >= pd.Timestamp(FIRST_REBALANCE_STR)]
    for rebalance_ts in rebalance_index:
        decision_session_ts = _decision_session_for_rebalance(pd.Timestamp(rebalance_ts), session_index)
        as_of_ts = datetime(
            decision_session_ts.year, decision_session_ts.month, decision_session_ts.day, 20, 0, tzinfo=NEW_YORK_TZ
        )
        backtest_weight_ser = rebalance_weight_df.loc[rebalance_ts].astype(float)
        backtest_weight_dict = {k: float(v) for k, v in backtest_weight_ser.items() if abs(float(v)) > 1e-12}
        row_dict = {
            "rebalance_date": pd.Timestamp(rebalance_ts).date().isoformat(),
            "decision_session": decision_session_ts.date().isoformat(),
            "backtest_weights": json.dumps(backtest_weight_dict, sort_keys=True),
        }
        try:
            plan_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None)
            live_weight_dict = {k: float(v) for k, v in plan_obj.full_target_weight_map_dict.items()}
            asset_set = set(live_weight_dict) | set(backtest_weight_dict)
            max_abs_diff_float = max(
                abs(live_weight_dict.get(asset_str, 0.0) - backtest_weight_dict.get(asset_str, 0.0))
                for asset_str in asset_set
            )
            live_exec_date_str = pd.Timestamp(plan_obj.target_execution_timestamp_ts).tz_convert(NEW_YORK_TZ).date().isoformat()
            row_dict.update(
                {
                    "status": "ok",
                    "live_weights": json.dumps(live_weight_dict, sort_keys=True),
                    "live_cash_reserve": float(plan_obj.cash_reserve_weight_float),
                    "max_abs_weight_diff": float(max_abs_diff_float),
                    "live_execution_date": live_exec_date_str,
                    "execution_date_match": live_exec_date_str == row_dict["rebalance_date"],
                    "live_signal_date": pd.Timestamp(plan_obj.signal_timestamp_ts).tz_convert(NEW_YORK_TZ).date().isoformat(),
                    "dtb3_latest_obs": plan_obj.snapshot_metadata_dict.get("dtb3_latest_observation_date_str", ""),
                }
            )
        except Exception as exception_obj:  # noqa: BLE001 - audit records every failure
            row_dict.update({"status": f"error: {type(exception_obj).__name__}: {exception_obj}"[:500]})
        row_list.append(row_dict)
        print(variant_key_str, row_dict["decision_session"], row_dict.get("status"), row_dict.get("max_abs_weight_diff"), flush=True)

    result_df = pd.DataFrame(row_list)
    result_df.to_csv(OUTPUT_DIR_PATH / f"live_parity_{variant_key_str}.csv", index=False)
    ok_df = result_df[result_df["status"] == "ok"]
    summary_dict = {
        "variant": variant_key_str,
        "decisions_total": int(len(result_df)),
        "decisions_ok": int(len(ok_df)),
        "decisions_error": int((result_df["status"] != "ok").sum()),
        "exact_match_1e-9": int((ok_df["max_abs_weight_diff"] <= 1e-9).sum()) if len(ok_df) else 0,
        "max_abs_weight_diff": float(ok_df["max_abs_weight_diff"].max()) if len(ok_df) else None,
        "execution_date_mismatch": int((~ok_df["execution_date_match"].astype(bool)).sum()) if len(ok_df) else 0,
    }
    return summary_dict


def main() -> None:
    _patch_environment()
    variant_key_list = sys.argv[1:] or list(VARIANT_DICT)
    summary_list = []
    for variant_key_str in variant_key_list:
        start_float = time.time()
        summary_dict = replay_variant(variant_key_str)
        summary_dict["elapsed_seconds"] = round(time.time() - start_float, 1)
        summary_list.append(summary_dict)
        print(json.dumps(summary_dict), flush=True)
    (OUTPUT_DIR_PATH / f"live_parity_summary_{'_'.join(variant_key_list)}.json").write_text(
        json.dumps(summary_list, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
