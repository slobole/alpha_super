"""NDX ATR / ATR-VXN live-host parity replay (audit protocol B1/B2) plus engine cross-check.

1. Backtest reference: the real engine run (run_variant, end 2026-09-25). For each monthly
   rebalance we record the target weights the strategy computed from FULL-history signals at
   decision close T (get_target_weight_ser with previous_bar = T), and the positions actually
   held after the engine filled that rebalance.
2. Live: ``build_decision_plan_for_release`` with as_of = T 20:00 New York, so every loader is
   truncated at T. The full-target weights are compared with (1) to 1e-9.

Runtime patch (study code only): ``build_index_constituent_matrix`` inside the NDX module is
served from one cached call, because it is date-independent apart from the membership trim,
which is computed against today's last $SPX date in both paths. The live snapshot exporter
computes the same trim at its own export time; that difference is recorded as not replayable.

Usage: uv run python .../ndx_live_parity_replay.py <variant: vxn|atr> [n_recent_month_ends]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import data.norgate_loader as norgate_loader_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402
from alpha.live import strategy_host  # noqa: E402
from alpha.live.models import LiveRelease  # noqa: E402

OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/ndx"
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
END_DATE_STR = "2026-09-25"
NEW_YORK_TZ = ZoneInfo("America/New_York")

VARIANT_DICT = {
    "vxn": {
        "import_str": "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy",
        "data_profile_str": "norgate_eod_ndx_pit_plus_vxn_helper",
        "module": vxn_module,
    },
    "atr": {
        "import_str": "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
        "data_profile_str": "norgate_eod_ndx_pit",
        "module": atr_module,
    },
}

_UNIVERSE_CACHE: dict[str, tuple] = {}
_real_build_universe_fn = norgate_loader_module.build_index_constituent_matrix


def _cached_build_universe(indexname: str = "S&P 500"):
    if indexname not in _UNIVERSE_CACHE:
        _UNIVERSE_CACHE[indexname] = _real_build_universe_fn(indexname=indexname)
    symbol_list, universe_df = _UNIVERSE_CACHE[indexname]
    return list(symbol_list), universe_df.copy()


def _build_release(variant_key_str: str) -> LiveRelease:
    variant_dict = VARIANT_DICT[variant_key_str]
    params_dict = {
        "max_positions_int": 10,
        "lookback_month_int": 12,
        "index_trend_window_int": 200,
        "stock_trend_window_int": 100,
        "regime_symbol_str": "SPY",
        "slippage_float": 0.00025,
        "commission_per_share_float": 0.005,
        "commission_minimum_float": 1.0,
        "capital_base_float": 100_000.0,
    }
    if variant_key_str == "vxn":
        params_dict.update(
            {
                "vxn_symbol_str": "$VXN",
                "target_vxn_pct_float": 22.0,
                "min_exposure_scale_float": 0.25,
                "max_exposure_scale_float": 1.0,
            }
        )
    return LiveRelease(
        release_id_str=f"audit.ndx.{variant_key_str}",
        user_id_str="audit_user",
        pod_id_str=f"audit_ndx_{variant_key_str}",
        account_route_str="U00000000",
        strategy_import_str=variant_dict["import_str"],
        mode_str="live",
        session_calendar_id_str="XNYS",
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
        data_profile_str=variant_dict["data_profile_str"],
        params_dict=params_dict,
        risk_profile_str="audit",
        enabled_bool=False,
        source_path_str="audit",
    )


def _run_backtest(variant_key_str: str):
    module_obj = VARIANT_DICT[variant_key_str]["module"]
    strategy_obj = module_obj.run_variant(
        show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR
    )
    return strategy_obj


def _full_history_target_weights(variant_key_str: str, strategy_obj) -> tuple[dict, pd.DataFrame]:
    module_obj = VARIANT_DICT[variant_key_str]["module"]
    config_obj = module_obj.DEFAULT_CONFIG.__class__(**{**module_obj.DEFAULT_CONFIG.__dict__, "end_date_str": END_DATE_STR})
    if variant_key_str == "vxn":
        pricing_data_df, universe_df, rebalance_schedule_df, vxn_df = module_obj.get_vxn_scaled_atr_normalized_ndx_data(config_obj)
    else:
        pricing_data_df, universe_df, rebalance_schedule_df = module_obj.get_atr_normalized_ndx_data(config_obj)
    full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    weight_dict: dict[str, dict[str, float]] = {}
    for execution_ts, row_ser in rebalance_schedule_df.iterrows():
        decision_ts = pd.Timestamp(row_ser["decision_date_ts"])
        strategy_obj.previous_bar = decision_ts
        target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=full_signal_df.loc[decision_ts])
        weight_dict[decision_ts.date().isoformat()] = {str(k): float(v) for k, v in target_weight_ser.items()}
    return weight_dict, rebalance_schedule_df


def _held_after_rebalance(strategy_obj, rebalance_schedule_df: pd.DataFrame) -> dict[str, list[str]]:
    transaction_df = strategy_obj.get_transactions().copy()
    held_dict: dict[str, list[str]] = {}
    position_ser = pd.Series(dtype=float)
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"])
    grouped_obj = transaction_df.groupby("bar")
    running_position_dict: dict[str, float] = {}
    schedule_by_exec = {pd.Timestamp(k): pd.Timestamp(v) for k, v in rebalance_schedule_df["decision_date_ts"].items()}
    for bar_ts, bar_df in grouped_obj:
        for _, tx_row in bar_df.iterrows():
            running_position_dict[str(tx_row["asset"])] = running_position_dict.get(str(tx_row["asset"]), 0.0) + float(tx_row["amount"])
        if pd.Timestamp(bar_ts) in schedule_by_exec:
            decision_ts = schedule_by_exec[pd.Timestamp(bar_ts)]
            held_dict[decision_ts.date().isoformat()] = sorted(
                asset_str for asset_str, amount_float in running_position_dict.items() if abs(amount_float) > 1e-9
            )
    return held_dict


def main() -> None:
    variant_key_str = sys.argv[1]
    n_recent_int = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    atr_module.build_index_constituent_matrix = _cached_build_universe

    start_float = time.time()
    backtest_obj = _run_backtest(variant_key_str)
    full_weight_dict, rebalance_schedule_df = _full_history_target_weights(variant_key_str, backtest_obj)
    held_dict = _held_after_rebalance(backtest_obj, rebalance_schedule_df)
    print(f"backtest done in {time.time() - start_float:.0f}s; decisions={len(full_weight_dict)}", flush=True)

    decision_date_list = sorted(full_weight_dict)
    # Recent month-ends plus a spread of older stress dates.
    stress_list = [d for d in decision_date_list if d[:7] in {
        "2000-12", "2001-09", "2002-07", "2008-09", "2008-10", "2009-03", "2011-08", "2015-08",
        "2018-12", "2020-02", "2020-03", "2020-04", "2022-01", "2022-06",
    }]
    replay_date_list = sorted(set(decision_date_list[-n_recent_int:]) | set(stress_list))

    release_obj = _build_release(variant_key_str)
    row_list = []
    for decision_date_str in replay_date_list:
        decision_ts = pd.Timestamp(decision_date_str)
        as_of_ts = datetime(decision_ts.year, decision_ts.month, decision_ts.day, 20, 0, tzinfo=NEW_YORK_TZ)
        backtest_weight_dict = full_weight_dict[decision_date_str]
        row_dict = {
            "decision_date": decision_date_str,
            "backtest_weights": json.dumps(backtest_weight_dict, sort_keys=True),
            "backtest_held_after_fill": json.dumps(held_dict.get(decision_date_str, [])),
        }
        try:
            plan_obj = strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, None)
            live_weight_dict = {k: float(v) for k, v in plan_obj.full_target_weight_map_dict.items()}
            asset_set = set(live_weight_dict) | set(backtest_weight_dict)
            max_abs_diff_float = max(
                [abs(live_weight_dict.get(a, 0.0) - backtest_weight_dict.get(a, 0.0)) for a in asset_set] or [0.0]
            )
            row_dict.update(
                {
                    "status": "ok",
                    "live_signal_date": pd.Timestamp(plan_obj.signal_timestamp_ts).tz_convert(NEW_YORK_TZ).date().isoformat(),
                    "live_execution_date": pd.Timestamp(plan_obj.target_execution_timestamp_ts).tz_convert(NEW_YORK_TZ).date().isoformat(),
                    "live_weights": json.dumps(live_weight_dict, sort_keys=True),
                    "max_abs_weight_diff": float(max_abs_diff_float),
                    "set_match": set(live_weight_dict) == set(backtest_weight_dict),
                    "held_set_match": sorted(live_weight_dict) == held_dict.get(decision_date_str, sorted(live_weight_dict)),
                    "vxn_reference_date": plan_obj.snapshot_metadata_dict.get("vxn_reference_date_str", ""),
                }
            )
        except Exception as exception_obj:  # noqa: BLE001
            row_dict["status"] = f"error: {type(exception_obj).__name__}: {exception_obj}"[:500]
        row_list.append(row_dict)
        print(decision_date_str, row_dict.get("status"), row_dict.get("max_abs_weight_diff"), row_dict.get("set_match"), flush=True)

    result_df = pd.DataFrame(row_list)
    result_df.to_csv(OUTPUT_DIR_PATH / f"live_parity_{variant_key_str}.csv", index=False)
    ok_df = result_df[result_df["status"] == "ok"]
    summary_dict = {
        "variant": variant_key_str,
        "replayed": int(len(result_df)),
        "ok": int(len(ok_df)),
        "errors": int((result_df["status"] != "ok").sum()),
        "exact_1e-9": int((ok_df["max_abs_weight_diff"] <= 1e-9).sum()),
        "set_match": int(ok_df["set_match"].sum()),
        "held_set_match": int(ok_df["held_set_match"].sum()),
        "signal_date_match": int((ok_df["live_signal_date"] == ok_df["decision_date"]).sum()),
        "backtest_final_equity": float(backtest_obj.total_value),
        "elapsed_seconds": round(time.time() - start_float, 1),
    }
    (OUTPUT_DIR_PATH / f"live_parity_summary_{variant_key_str}.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    pd.Series(full_weight_dict).to_json(OUTPUT_DIR_PATH / f"backtest_full_history_weights_{variant_key_str}.json")
    print(json.dumps(summary_dict), flush=True)


if __name__ == "__main__":
    main()
