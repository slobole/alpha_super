"""Replay the LIVE capsule decision path on real history and compare it with the backtest, order by order.

Question: on a real session T, given exactly the account the backtest held after Close_T, does the live adapter
(alpha/live/mr_capsule_adapter.build_mr_capsule_decision_plan, the code a release runs) produce the same exits, the
same entry dollars in the same priority, and the same BIL / SPMO whole-share targets as the backtest's own orders?

Method (exploratory verification, not a strategy test; no parameters are tuned):
1. Run the Bench backtest once (the pod's run function path: same loaders, same builder) with a recorder that saves,
   at every decision close T, the engine account (whole-share positions, cash, NAV_T) and strategy state before
   iterate, and the orders iterate placed.
2. For selected sessions T, rebuild a broker-style EOD PodState from that record and call the live adapter at
   T 18:00 ET. Data loaders return the same preloaded Norgate frames (the adapter truncates them to <= T itself);
   $VIX is cut at T; the snapshot identity is stubbed because the snapshot reader is qualified separately
   (results/research/mr_capsule_review_20261005/qualification_v3: 121M cells, 0 mismatches).
   compute_signals, the gate, HPI feature readiness, parking and the order classifier all run for real.
3. Compare the live DecisionPlan with the engine orders at T.

Usage: python replay_live_decisions.py dv2|hpi [bil|cash|spmo] [max_dates]
Writes results/research/mr_capsule_build_20261004/replay_live_<pod>_<mode>.json. Norgate loads: run alone.
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/mr_capsule_build_20261004"
END_DATE_STR = "2026-10-02"
NY_TZ = ZoneInfo("America/New_York")
REPLAY_START_TS = pd.Timestamp("2019-01-01")
SEED_INT = 20261005


def _record_backtest(pod_str: str, mode_str: str):
    from alpha.engine.backtest import run_daily
    from alpha.live import strategy_host
    from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated
    from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

    if pod_str == "dv2":
        pricing_data_df, universe_df = dv2_vix_gated.load_pricing_data(END_DATE_STR)
        builder_fn = dv2_vix_gated.build_dv2_capsule_strategy
    else:
        pricing_data_df, universe_df = hpi_vote_vix_gated.load_hpi_capsule_pricing_data(END_DATE_STR)
        builder_fn = hpi_vote_vix_gated.build_hpi_capsule_strategy
    vix_close_ser = load_vix_close_ser(END_DATE_STR)
    strategy_obj = builder_fn(
        strategy_name_str=f"replay_{pod_str}_{mode_str}", parking_enabled_bool=mode_str != "cash",
        spmo_parking_enabled_bool=mode_str == "spmo", universe_df=universe_df, vix_close_ser=vix_close_ser,
    )
    record_dict: dict[pd.Timestamp, dict] = {}
    base_class = type(strategy_obj)

    class _Recorder(base_class):  # same object and configuration; only observes iterate
        def iterate(self, data, close, open_prices):
            decision_ts = pd.Timestamp(self.previous_bar)
            position_ser = self.get_positions()
            state_dict = {
                "position_map": {str(k): float(v) for k, v in position_ser.items() if float(v) != 0.0},
                "cash_float": float(self.cash),
                "nav_float": float(self.previous_total_value),
                "strategy_state_dict": strategy_host._extract_strategy_state_dict(self),
            }
            super().iterate(data, close, open_prices)
            state_dict["order_list"] = [strategy_host._build_order_shape_dict(o) for o in self.get_orders()]
            record_dict[decision_ts] = state_dict

    strategy_obj.__class__ = _Recorder
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp("2004-01-01")]
    run_daily(strategy_obj, pricing_data_df, calendar_idx, show_progress=False, show_signal_progress_bool=False)
    final_nav_float = float(strategy_obj.results["total_value"].iloc[-1])
    return pricing_data_df, universe_df, vix_close_ser, strategy_obj.gate_open_ser, record_dict, final_nav_float


def _select_dates(record_dict: dict, gate_open_ser: pd.Series, max_int: int) -> list[pd.Timestamp]:
    rng = np.random.default_rng(SEED_INT)
    date_list = sorted(d for d in record_dict if d >= REPLAY_START_TS)
    gate_ser = gate_open_ser.reindex(date_list).astype(bool)
    switch_list = [d for i, d in enumerate(date_list[1:], 1) if gate_ser.iloc[i] != gate_ser.iloc[i - 1]]

    def kinds(d):
        orders = record_dict[d]["order_list"]
        stock = [o for o in orders if o["asset_str"] not in ("BIL", "SPMO")]
        return (
            any(o["unit_str"] == "value" and not o["target_bool"] for o in stock),
            any(o["target_bool"] and o["amount_float"] == 0.0 for o in stock),
            any(o["asset_str"] in ("BIL", "SPMO") for o in orders),
            bool(record_dict[d]["strategy_state_dict"].get("pending_exit_symbol_list")),
        )

    entry_list = [d for d in date_list if kinds(d)[0]]
    exit_only_list = [d for d in date_list if kinds(d)[1] and not kinds(d)[0]]
    parking_only_list = [d for d in date_list if kinds(d)[2] and not kinds(d)[0] and not kinds(d)[1]]
    pending_list = [d for d in date_list if kinds(d)[3]]
    quiet_list = [d for d in date_list if not any(kinds(d))]

    def take(pool, n):
        pool = [d for d in pool if d not in chosen]
        return [pd.Timestamp(x) for x in rng.choice(pool, size=min(n, len(pool)), replace=False)] if pool else []

    chosen: list[pd.Timestamp] = []
    chosen += take(switch_list, max(4, max_int // 5))
    chosen += take(entry_list, max(6, max_int * 2 // 5))
    chosen += take(exit_only_list, max(3, max_int // 5))
    chosen += take(parking_only_list, max(2, max_int // 10))
    chosen += take(pending_list, 3)
    chosen += take(quiet_list, max(2, max_int // 10))
    chosen.append(date_list[-1])  # the latest session, as a live run on 2026-10-02 would see it
    return sorted(set(pd.Timestamp(d) for d in chosen))


def _live_decision(pod_str, mode_str, decision_ts, state_dict, pricing_data_df, universe_df, vix_close_ser, monkeypatch_list):
    from alpha.live import mr_capsule_adapter, strategy_host
    from alpha.live.models import PodState
    from alpha.live.release_manifest import parse_release_manifest

    release_obj = parse_release_manifest(
        str(REPO / f"docs/live/release_templates/pod_mr_{'dv2_vix_gated' if pod_str == 'dv2' else 'hpi_vote_vix_gated'}_{mode_str}_daily_moo.yaml.example")
    )
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str,
        position_amount_map=dict(state_dict["position_map"]), cash_float=state_dict["cash_float"],
        total_value_float=state_dict["nav_float"],
        strategy_state_dict={**state_dict["strategy_state_dict"], "mr_capsule_strategy_import_str": release_obj.strategy_import_str},
        updated_timestamp_ts=datetime(decision_ts.year, decision_ts.month, decision_ts.day, 16, 30, tzinfo=NY_TZ),
        snapshot_stage_str="eod", snapshot_source_str="broker",
    )
    metadata_dict = {"norgate_snapshot_date_str": decision_ts.date().isoformat(),
                     "norgate_data_profile_str": release_obj.data_profile_str, "norgate_manifest_hash_str": "replay-direct-norgate"}
    for set_fn in monkeypatch_list:
        set_fn(release_obj, metadata_dict, decision_ts)
    as_of_ts = datetime(decision_ts.year, decision_ts.month, decision_ts.day, 18, 0, tzinfo=NY_TZ)
    return strategy_host.build_decision_plan_for_release(release_obj, as_of_ts, pod_state_obj), mr_capsule_adapter


def _compare(decision_obj, state_dict) -> dict:
    orders = state_dict["order_list"]
    engine_exit_set = {o["asset_str"] for o in orders if o["target_bool"] and o["amount_float"] == 0.0}
    engine_entry_list = [(o["asset_str"], o["amount_float"]) for o in orders if o["unit_str"] == "value" and not o["target_bool"]]
    engine_target_dict = {o["asset_str"]: o["amount_float"] for o in orders
                          if o["target_bool"] and o["unit_str"] == "shares" and o["amount_float"] > 0.0}
    live_nav_float = float(decision_obj.snapshot_metadata_dict["decision_nav_float"])
    live_entry_list = [(a, decision_obj.entry_target_weight_map_dict[a] * live_nav_float) for a in decision_obj.entry_priority_list]
    entry_value_diff_float = max((abs(lv - ev) for (_, lv), (_, ev) in zip(live_entry_list, engine_entry_list)), default=0.0)
    result_dict = {
        "exit_equal_bool": set(decision_obj.exit_asset_set) == engine_exit_set,
        "entry_assets_equal_bool": [a for a, _ in live_entry_list] == [a for a, _ in engine_entry_list],
        "entry_value_max_abs_diff_float": float(entry_value_diff_float),
        "parking_targets_equal_bool": {k: float(v) for k, v in decision_obj.target_share_map_dict.items()} == engine_target_dict,
        "nav_abs_diff_float": abs(live_nav_float - state_dict["nav_float"]),
        "n_engine_orders_int": len(orders),
    }
    result_dict["match_bool"] = bool(
        result_dict["exit_equal_bool"] and result_dict["entry_assets_equal_bool"] and result_dict["parking_targets_equal_bool"]
        and result_dict["entry_value_max_abs_diff_float"] <= 1e-6 * max(1.0, live_nav_float)
        and result_dict["nav_abs_diff_float"] <= 1e-6 * max(1.0, live_nav_float)
    )
    if not result_dict["match_bool"]:
        result_dict["detail_dict"] = {
            "live_exit_list": sorted(decision_obj.exit_asset_set), "engine_exit_list": sorted(engine_exit_set),
            "live_entry_list": live_entry_list, "engine_entry_list": engine_entry_list,
            "live_target_dict": dict(decision_obj.target_share_map_dict), "engine_target_dict": engine_target_dict,
        }
    return result_dict


def main(pod_str: str, mode_str: str = "bil", max_dates_int: int = 40) -> None:
    from alpha.live import mr_capsule_adapter, strategy_host
    from strategies.mr_capsule import dv2_vix_gated, hpi_vote_vix_gated, vix_stress_gate

    t0 = time.time()
    pricing_data_df, universe_df, vix_close_ser, gate_open_ser, record_dict, final_nav_float = _record_backtest(pod_str, mode_str)
    print(f"backtest recorded: {len(record_dict)} decisions, final NAV {final_nav_float:,.2f}, {time.time() - t0:.0f}s", flush=True)
    date_list = _select_dates(record_dict, gate_open_ser, int(max_dates_int))
    print(f"replaying {len(date_list)} sessions", flush=True)

    def install(release_obj, metadata_dict, decision_ts):
        # Same preloaded Norgate frames as the backtest; the adapter itself truncates them to <= T.
        dv2_vix_gated.load_pricing_data = lambda *_a, **_k: (pricing_data_df, universe_df)
        hpi_vote_vix_gated.load_hpi_capsule_pricing_data = lambda *_a, **_k: (pricing_data_df, universe_df)
        vix_stress_gate.load_vix_close_ser = lambda *_a, **_k: vix_close_ser.loc[:decision_ts]
        mr_capsule_adapter.build_data_source_metadata_dict = lambda *_a, **_k: dict(metadata_dict)
        strategy_host.build_data_source_metadata_dict = lambda *_a, **_k: dict(metadata_dict)

    result_list = []
    for decision_ts in date_list:
        t1 = time.time()
        try:
            decision_obj, _ = _live_decision(pod_str, mode_str, decision_ts, record_dict[decision_ts],
                                             pricing_data_df, universe_df, vix_close_ser, [install])
            row_dict = {"date_str": decision_ts.date().isoformat(), **_compare(decision_obj, record_dict[decision_ts])}
        except Exception as exc:  # report, never hide: a live block on a session the backtest traded is a finding
            row_dict = {"date_str": decision_ts.date().isoformat(), "match_bool": False, "error_str": f"{type(exc).__name__}: {exc}",
                        "n_engine_orders_int": len(record_dict[decision_ts]["order_list"])}
        row_dict["seconds_float"] = round(time.time() - t1, 1)
        result_list.append(row_dict)
        print(json.dumps(row_dict, default=str)[:400], flush=True)
    summary_dict = {
        "pod_str": pod_str, "mode_str": mode_str, "end_date_str": END_DATE_STR,
        "backtest_final_nav_float": final_nav_float, "n_replayed_int": len(result_list),
        "n_match_int": int(sum(r["match_bool"] for r in result_list)),
        "n_with_engine_orders_int": int(sum(r["n_engine_orders_int"] > 0 for r in result_list)),
        "n_entry_sessions_int": int(sum(any(o["unit_str"] == "value" and not o["target_bool"]
                                            for o in record_dict[pd.Timestamp(r["date_str"])]["order_list"]) for r in result_list)),
        "runtime_seconds_float": round(time.time() - t0, 1),
        "row_list": result_list,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"replay_live_{pod_str}_{mode_str}.json").write_text(json.dumps(summary_dict, indent=2, default=str), encoding="utf-8")
    print(f"MATCH {summary_dict['n_match_int']}/{summary_dict['n_replayed_int']} "
          f"(entry sessions {summary_dict['n_entry_sessions_int']}, with orders {summary_dict['n_with_engine_orders_int']})", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
