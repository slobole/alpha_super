"""Proof: the live adapter's frozen share book equals the house engine's next-close position, decision by decision.

Every session T whose next session is a scheduled fill, every session right after a fill, and 300 other sessions, 2003 to the
last close. For each T the adapter gets the engine's own account at Close_T (cash, signed shares) and the receipt a clean
run would hold, and must name exactly the shares the engine holds after the next close. A sample is rerun on prices cut at T.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/flow_vote_wiring_20261104_parity.py
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
from alpha.live.flow_vote_adapter import FLOW_VOTE_ASSET_TUPLE, build_flow_vote_decision_from_prices
from alpha.live.models import PodState
from alpha.live.release_manifest import parse_release_manifest
from data.flow_vote_snapshot_contract import FLOW_VOTE_PROFILE_STR
from strategies.taa_beyond_6040 import strategy_taa_month_end_flow_vote as vote_module

import exchange_calendars as xcals  # noqa: E402

# The live scheduler's calendar reaches 20 years back (xcals default). The proof replays from 2003, so it gets the full range;
# the live logic is unchanged.
scheduler_utils.get_exchange_calendar_obj = lambda calendar_id_str: xcals.get_calendar(calendar_id_str, start="2002-01-01")

TEMPLATE_PATH = ROOT_PATH / "docs" / "live" / "release_templates" / "pod_taa_month_end_flow_vote_daily_moc.yaml.example"
OUT_PATH = ROOT_PATH / "results" / "research" / "flow_vote_wiring_20261104"


def engine_path(pricing_data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, vote_module.MonthEndVoteFlowStrategy]:
    context_dict = vote_module._build_analysis_context_dict(vote_module.DEFAULT_CONFIG, pricing_data_df)
    strategy_obj = vote_module.MonthEndVoteFlowStrategy(vote_module.DEFAULT_CONFIG)
    run_daily(strategy_obj, context_dict["pricing_data_df"], calendar=context_dict["calendar_idx"], show_progress=False, show_signal_progress_bool=False)
    transaction_df = strategy_obj.get_transactions() if hasattr(strategy_obj, "get_transactions") else pd.DataFrame(strategy_obj._transactions)
    return transaction_df, strategy_obj.results["cash"], strategy_obj


def main() -> None:
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pricing_data_df = vote_module.get_month_end_flow_data(vote_module.DEFAULT_CONFIG)
    transaction_df, cash_ser, strategy_obj = engine_path(pricing_data_df)
    print("transaction columns", list(transaction_df.columns), flush=True)
    date_col = next(c for c in ("date", "Date", "timestamp", "bar") if c in transaction_df.columns)
    asset_col = next(c for c in ("asset", "symbol", "asset_str", "ticker") if c in transaction_df.columns)
    share_col = next(c for c in ("amount", "shares", "quantity", "amount_float") if c in transaction_df.columns)
    session_idx = cash_ser.index
    delta_df = transaction_df.assign(_d=pd.to_datetime(transaction_df[date_col]).dt.normalize()).pivot_table(
        index="_d", columns=asset_col, values=share_col, aggfunc="sum").reindex(session_idx).fillna(0.0)
    position_df = delta_df.cumsum().reindex(columns=list(FLOW_VOTE_ASSET_TUPLE)).fillna(0.0)

    release_obj = replace(parse_release_manifest(str(TEMPLATE_PATH)), mode_str="paper", enabled_bool=True, account_route_str="DU_FLOW_PROOF")
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    schedule_df = strategy_obj.order_schedule_df
    fill_set = set(schedule_df.index)
    position_list = list(session_idx)
    next_of = {position_list[i]: position_list[i + 1] for i in range(len(position_list) - 1)}
    fill_eve_list = [t for t in position_list[:-1] if next_of[t] in fill_set]
    after_fill_list = [t for t in position_list[:-1] if t in fill_set]
    rng_obj = np.random.default_rng(7)
    other_pool = sorted(set(position_list[:-1]) - set(fill_eve_list) - set(after_fill_list))
    other_list = list(rng_obj.choice(np.array(other_pool, dtype="datetime64[ns]"), size=300, replace=False))
    sample_list = sorted(set(fill_eve_list) | set(after_fill_list) | {pd.Timestamp(t) for t in other_list})
    print(f"sessions checked: {len(sample_list)} ({len(fill_eve_list)} fill eves, {len(after_fill_list)} after a fill, 300 other)", flush=True)

    mismatch_list, cut_check_int, cut_mismatch_int, rebalance_mismatch_int = [], 0, 0, 0
    for count_int, signal_ts in enumerate(sample_list):
        close_ts = scheduler_utils.get_session_close_timestamp_ts(signal_ts, "XNYS")
        prior_event_df = schedule_df.loc[schedule_df.index <= signal_ts]
        receipt_dict = {}
        if not prior_event_df.empty:
            event_ts = prior_event_df.index[-1]
            receipt_dict = {"last_applied_event_fill_date_str": event_ts.date().isoformat(),
                            "last_applied_target_weight_map_dict": {"SPY": float(prior_event_df["SPY"].iloc[-1]), "TLT": float(prior_event_df["TLT"].iloc[-1]),
                                                                    "Cash": 1.0 - float(prior_event_df["SPY"].iloc[-1]) - float(prior_event_df["TLT"].iloc[-1])},
                            "last_applied_target_share_map_dict": {a: float(position_df.at[signal_ts, a]) for a in FLOW_VOTE_ASSET_TUPLE}}
        position_dict = {a: float(position_df.at[signal_ts, a]) for a in FLOW_VOTE_ASSET_TUPLE if position_df.at[signal_ts, a] != 0.0}
        pod_state_obj = PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
                                 position_amount_map=position_dict, cash_float=float(cash_ser.at[signal_ts]),
                                 total_value_float=float(cash_ser.at[signal_ts]), strategy_state_dict={
                                     "flow_vote_state_version_int": 1, "last_signal_date_str": "", "flow_vote_execution_receipt_dict": receipt_dict},
                                 updated_timestamp_ts=close_ts + timedelta(minutes=12), snapshot_stage_str="eod", snapshot_source_str="broker")
        as_of_ts = close_ts + timedelta(minutes=30)
        metadata_dict = {"norgate_snapshot_date_str": signal_ts.date().isoformat(), "norgate_data_profile_str": FLOW_VOTE_PROFILE_STR, "norgate_manifest_hash_str": "proof"}
        plan_obj = build_flow_vote_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df, metadata_dict)
        target_dict = plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"]
        expected_dict = {a: float(position_df.at[next_of[signal_ts], a]) for a in FLOW_VOTE_ASSET_TUPLE}
        if target_dict != expected_dict:
            mismatch_list.append({"signal_date_str": signal_ts.date().isoformat(), "target_dict": target_dict, "engine_next_close_dict": expected_dict})
        if plan_obj.snapshot_metadata_dict["rebalance_bool"] != (next_of[signal_ts] in fill_set):
            rebalance_mismatch_int += 1
        if count_int % 10 == 0:
            cut_check_int += 1
            cut_plan_obj = build_flow_vote_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df.loc[:signal_ts].copy(), metadata_dict)
            if cut_plan_obj.snapshot_metadata_dict["fixed_target_share_map_dict"] != target_dict:
                cut_mismatch_int += 1
    result_dict = {"first_session_str": str(sample_list[0].date()), "last_session_str": str(sample_list[-1].date()),
                   "sessions_checked_int": len(sample_list), "fill_eves_int": len(fill_eve_list), "after_fill_int": len(after_fill_list),
                   "share_book_mismatch_int": len(mismatch_list), "rebalance_flag_mismatch_int": rebalance_mismatch_int,
                   "rows_through_t_checked_int": cut_check_int, "rows_through_t_mismatch_int": cut_mismatch_int,
                   "first_mismatches": mismatch_list[:5]}
    (OUT_PATH / "parity.json").write_text(json.dumps(result_dict, indent=1), encoding="utf-8")
    print(json.dumps(result_dict, indent=1))


if __name__ == "__main__":
    main()
