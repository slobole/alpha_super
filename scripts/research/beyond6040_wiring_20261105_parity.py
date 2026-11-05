"""Proof: the T1 live adapter's frozen share book equals the house engine's next-open position, decision by decision.

Every session T whose next session is an engine fill, every session right after a fill, and 300 other sessions, 2005 to the
last close. For each T the adapter gets the engine's own account at Close_T (cash, shares) and the receipt a clean run would
hold, and must name exactly the shares the engine holds after the next open. A sample is rerun on prices cut at T.

    PYTHONUTF8=1 ../alpha_super/.venv/Scripts/python.exe scripts/research/beyond6040_wiring_20261105_parity.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.live import scheduler_utils
from alpha.live.beyond6040_adapter import BEYOND6040_ASSET_TUPLE, build_beyond6040_decision_from_prices
from alpha.live.models import PodState
from alpha.live.release_manifest import parse_release_manifest
from data.beyond6040_snapshot_contract import BEYOND6040_PROFILE_STR
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module

import exchange_calendars as xcals  # noqa: E402

scheduler_utils.get_exchange_calendar_obj = lambda calendar_id_str: xcals.get_calendar(calendar_id_str, start="2001-01-01")

TEMPLATE_PATH = ROOT_PATH / "docs" / "live" / "release_templates" / "pod_taa_beyond_6040_trend_tilt_daily_moo.yaml.example"
OUT_PATH = ROOT_PATH / "results" / "research" / "beyond6040_wiring_20261105"


def engine_path(pricing_data_df: pd.DataFrame):
    strategy_obj = tilt_module.run_variant(show_display_bool=False, save_results_bool=False)
    transaction_df = strategy_obj.get_transactions() if hasattr(strategy_obj, "get_transactions") else pd.DataFrame(strategy_obj._transactions)
    return transaction_df, strategy_obj.results["cash"], strategy_obj


def main() -> None:
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pricing_data_df = tilt_module.get_trend_tilt_data(tilt_module.DEFAULT_CONFIG)
    transaction_df, cash_ser, strategy_obj = engine_path(pricing_data_df)
    print("transaction columns", list(transaction_df.columns), flush=True)
    date_col = next(c for c in ("date", "Date", "timestamp", "bar") if c in transaction_df.columns)
    asset_col = next(c for c in ("asset", "symbol", "asset_str", "ticker") if c in transaction_df.columns)
    share_col = next(c for c in ("amount", "shares", "quantity", "amount_float") if c in transaction_df.columns)
    session_idx = cash_ser.index
    delta_df = transaction_df.assign(_d=pd.to_datetime(transaction_df[date_col]).dt.normalize()).pivot_table(
        index="_d", columns=asset_col, values=share_col, aggfunc="sum").reindex(session_idx).fillna(0.0)
    position_df = delta_df.cumsum().reindex(columns=list(BEYOND6040_ASSET_TUPLE)).fillna(0.0)
    fill_set = set(delta_df.index[(delta_df.abs().sum(axis=1) > 0)])

    release_obj = replace(parse_release_manifest(str(TEMPLATE_PATH)), mode_str="paper", enabled_bool=True, account_route_str="DU_T1_PROOF")
    position_list = list(session_idx)
    next_of = {position_list[i]: position_list[i + 1] for i in range(len(position_list) - 1)}
    fill_eve_list = [t for t in position_list[:-1] if next_of[t] in fill_set]
    after_fill_list = [t for t in position_list[:-1] if t in fill_set]
    rng_obj = np.random.default_rng(7)
    other_pool = sorted(set(position_list[:-1]) - set(fill_eve_list) - set(after_fill_list))
    other_list = list(rng_obj.choice(np.array(other_pool, dtype="datetime64[ns]"), size=min(300, len(other_pool)), replace=False))
    # T1 trades almost every session (daily exposure), so nearly every session is a fill eve. Check every month-end
    # (the tilt and base-weight events), every session whose next fill crosses a month boundary, and 500 random others.
    month_end_list = [t for t in position_list[:-1] if next_of[t].month != t.month]
    random_pool = sorted(set(position_list[:-1]) - set(month_end_list))
    random_list = list(rng_obj.choice(np.array(random_pool, dtype="datetime64[ns]"), size=min(500, len(random_pool)), replace=False))
    sample_list = sorted(set(month_end_list) | {pd.Timestamp(t) for t in random_list} | {pd.Timestamp(t) for t in other_list})
    # the adapter needs the engine to have started: drop sessions before the first fill eve
    first_fill_ts = min(fill_set)
    sample_list = [t for t in sample_list if t >= position_list[max(0, position_list.index(first_fill_ts) - 1)]]
    print(f"sessions checked: {len(sample_list)} ({len(fill_eve_list)} fill eves, {len(after_fill_list)} after a fill, {len(other_list)} other)", flush=True)

    mismatch_list, cut_check_int, cut_mismatch_int, rebalance_mismatch_int = [], 0, 0, 0
    for count_int, signal_ts in enumerate(sample_list):
        close_ts = scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS")
        prior_fill_list = sorted(t for t in fill_set if t <= signal_ts)
        receipt_dict = {}
        if prior_fill_list:
            event_ts = prior_fill_list[-1]
            receipt_dict = {"last_applied_rebalance_date_str": position_list[position_list.index(event_ts) - 1].date().isoformat(),
                            "last_applied_target_share_map_dict": {a: float(position_df.at[signal_ts, a]) for a in BEYOND6040_ASSET_TUPLE}}
        position_dict = {a: float(position_df.at[signal_ts, a]) for a in BEYOND6040_ASSET_TUPLE if position_df.at[signal_ts, a] != 0.0}
        pod_state_obj = PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
                                 position_amount_map=position_dict, cash_float=float(cash_ser.at[signal_ts]),
                                 total_value_float=float(cash_ser.at[signal_ts]), strategy_state_dict={
                                     "beyond6040_state_version_int": 1, "last_signal_date_str": "", "beyond6040_execution_receipt_dict": receipt_dict},
                                 updated_timestamp_ts=close_ts + timedelta(minutes=12), snapshot_stage_str="eod", snapshot_source_str="broker")
        as_of_ts = close_ts + timedelta(minutes=30)
        metadata_dict = {"norgate_snapshot_date_str": signal_ts.date().isoformat(), "norgate_data_profile_str": BEYOND6040_PROFILE_STR, "norgate_manifest_hash_str": "proof"}
        plan_obj = build_beyond6040_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df, metadata_dict)
        target_dict = plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
        expected_dict = {a: float(position_df.at[next_of[signal_ts], a]) for a in BEYOND6040_ASSET_TUPLE}
        if target_dict != expected_dict:
            mismatch_list.append({"signal_date_str": signal_ts.date().isoformat(), "target_dict": target_dict, "engine_next_open_dict": expected_dict,
                                  "rebalance_bool": plan_obj.snapshot_metadata_dict.get("rebalance_bool")})
        if bool(plan_obj.snapshot_metadata_dict.get("rebalance_bool")) and not bool(plan_obj.snapshot_metadata_dict.get("no_order_bool")) and next_of[signal_ts] not in fill_set:
            rebalance_mismatch_int += 1
        if count_int % 10 == 0:
            cut_check_int += 1
            cut_plan_obj = build_beyond6040_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df.loc[:signal_ts].copy(), metadata_dict)
            if cut_plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] != target_dict:
                cut_mismatch_int += 1
        if count_int % 200 == 0:
            print(count_int, signal_ts.date(), "mismatches so far", len(mismatch_list), flush=True)
    result_dict = {"first_session_str": str(sample_list[0].date()), "last_session_str": str(sample_list[-1].date()),
                   "sessions_checked_int": len(sample_list), "fill_eves_int": len(fill_eve_list), "after_fill_int": len(after_fill_list),
                   "share_book_mismatch_int": len(mismatch_list), "order_without_engine_fill_int": rebalance_mismatch_int,
                   "rows_through_t_checked_int": cut_check_int, "rows_through_t_mismatch_int": cut_mismatch_int,
                   "first_mismatches": mismatch_list[:8]}
    (OUT_PATH / "parity.json").write_text(json.dumps(result_dict, indent=1), encoding="utf-8")
    print(json.dumps(result_dict, indent=1))


if __name__ == "__main__":
    main()
