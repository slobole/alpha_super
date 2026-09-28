"""B1/B4 live-host replay for both HPI pods on REAL Norgate data (direct mode; the VPS snapshot profile is not
available on this workstation).

For each selected signal session T, the production entry point
    alpha.live.strategy_host.build_decision_plan_for_release(release, as_of=T 18:00 ET, pod_state)
is called WITHOUT any monkeypatch of compute_signals or of the loader: the host itself calls
load_exact_hpi_inputs(end_date_str=T), computes signals on that truncated load, runs the 400/80% readiness gate
and builds the DecisionPlan. The pod state is the backtest's own state at Close_T (positions, pending exits, trade
ids, previous total value, cash) recorded by the full production-path backtest (hpi_full_runs.py base arm).

Compared exactly: exit set, ordered entry list, entry weights (1e-6 and exact), the pending-exit list the plan
persists, and trade_id_int. The only change to the host is a memoising wrapper around load_exact_hpi_inputs so the
two pods share one real load per date (same arguments, same object returned).

B4 synthetic-state cases (real data, real host): a held name with no bar on T (member halt, membership series is
0 on no-bar days), and a pending exit whose Open_(T+1) does not print. Each is compared with the backtest's own
iterate() on the same state with the real Open_(T+1) row.

Usage: uv run python hpi_live_replay.py [replay|b4]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import hpi_common as hc
from strategies.hpi import stateful_long as hpi_mod

os.environ.pop("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", None)
from alpha.live import strategy_host  # noqa: E402
from alpha.live.models import LiveRelease, PodState  # noqa: E402

ET = ZoneInfo("America/New_York")
_REAL_LOADER = hpi_mod.load_exact_hpi_inputs
_LOAD_CACHE: dict = {}
LOAD_LOG: list = []


def cached_loader(indexname_str, benchmark_symbol_str, start_date_str, end_date_str):
    key = (indexname_str, benchmark_symbol_str, start_date_str, end_date_str)
    if key not in _LOAD_CACHE:
        _LOAD_CACHE.clear()
        t0 = time.time()
        _LOAD_CACHE[key] = _REAL_LOADER(indexname_str=indexname_str, benchmark_symbol_str=benchmark_symbol_str,
                                        start_date_str=start_date_str, end_date_str=end_date_str)
        LOAD_LOG.append({"key": key, "seconds": round(time.time() - t0, 1),
                         "last_row": str(_LOAD_CACHE[key][2].index[-1].date())})
    symbols, universe, pricing = _LOAD_CACHE[key]
    return list(symbols), universe.copy(), pricing.copy()


hpi_mod.load_exact_hpi_inputs = cached_loader


def release(variant: str) -> LiveRelease:
    return LiveRelease(
        release_id_str=f"release::audit_{variant}", user_id_str="user_audit", pod_id_str=f"pod_audit_{variant}",
        account_route_str="DU_AUDIT", strategy_import_str=hc.VARIANTS[variant][2], mode_str="paper",
        session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready", execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_hpi_pit",
        params_dict={"capital_base_float": 100_000.0, "max_positions_int": 10},
        risk_profile_str="standard", enabled_bool=True, source_path_str="audit.yaml")


def pod_state(variant: str, rec: dict, positions=None, pending=None) -> PodState:
    return PodState(
        pod_id_str=f"pod_audit_{variant}", user_id_str="user_audit", account_route_str="DU_AUDIT",
        position_amount_map=dict(rec["positions"] if positions is None else positions),
        cash_float=float(rec["cash_before"]), total_value_float=float(rec["prev_total_value"]),
        strategy_state_dict={"trade_id_int": int(rec["trade_id_int"]),
                             "current_trade_map": dict(rec["current_trade_map"]),
                             "pending_exit_symbol_list": list(rec["pending_before"] if pending is None else pending)},
        updated_timestamp_ts=datetime(2000, 1, 1, tzinfo=ET))


def as_of(date: pd.Timestamp) -> datetime:
    return datetime(date.year, date.month, date.day, 18, 0, tzinfo=ET)


def load_records(variant: str) -> list[dict]:
    with (hc.OUT / "full_runs" / f"{variant}_base" / "records.pkl").open("rb") as handle:
        return pickle.load(handle)


def classify(rec: dict, members_before=None) -> set[str]:
    tags = set()
    n_held, n_exit, n_entry = len(rec["positions"]), len(rec["exits"]), len(rec["entries"])
    if n_exit and n_entry and n_held == 10:
        tags.add("refill_full_book")
    if n_exit >= 2:
        tags.add("multi_exit")
    if n_entry and not n_exit:
        tags.add("entry_only")
    if n_exit and not n_entry:
        tags.add("exit_only")
    if n_entry >= 3:
        tags.add("multi_entry")
    if not n_exit and not n_entry:
        tags.add("no_op")
    if rec["missing_open_held"]:
        tags.add("missing_open_held")
    if set(rec["pending_before"]) - set(rec["exits"]):
        tags.add("pending_not_exited")
    return tags


def select_dates(rec_by_variant: dict) -> list[pd.Timestamp]:
    """Deterministic: per variant and tag, evenly spaced picks over 2006-10..2026-09 + fixed recent dates."""
    chosen: set[pd.Timestamp] = set()
    for variant, recs in rec_by_variant.items():
        recs = [r for r in recs if r["signal_date"] >= pd.Timestamp("2006-10-02")]
        by_tag: dict[str, list] = {}
        for r in recs:
            for t in classify(r):
                by_tag.setdefault(t, []).append(r["signal_date"])
        quota = {"refill_full_book": 5, "multi_exit": 3, "entry_only": 3, "exit_only": 3, "multi_entry": 2,
                 "no_op": 1, "missing_open_held": 2, "pending_not_exited": 2}
        for t, q in quota.items():
            ds = by_tag.get(t, [])
            if not ds:
                continue
            idx = np.unique(np.linspace(0, len(ds) - 1, min(q, len(ds))).round().astype(int))
            chosen.update(ds[i] for i in idx)
    # recent sessions (deployment-relevant) incl. the last signal date with a backtest record
    last = max(r["signal_date"] for r in rec_by_variant["vote"])
    for d in ("2025-03-06", "2025-03-10", "2020-02-21", "2008-09-03"):
        chosen.add(pd.Timestamp(d))
    chosen.add(last)
    return sorted(chosen)


def compare(plan, rec: dict) -> dict:
    live_entries = list(plan.entry_priority_list)
    live_w = [float(plan.entry_target_weight_map_dict[s]) for s in live_entries]
    bt_w = [v / rec["prev_total_value"] for v in rec["entry_values"]]
    out = {
        "exits_equal": set(plan.exit_asset_set) == set(rec["exits"]),
        "entries_equal": live_entries == rec["entries"],
        "weights_equal_1e-6": len(live_w) == len(bt_w) and bool(np.allclose(live_w, bt_w, rtol=0, atol=1e-6)),
        "weights_exact": live_w == bt_w,
        "pending_after_equal": list(plan.strategy_state_dict.get("pending_exit_symbol_list", [])) == rec["pending_after"],
        "trade_id_equal": int(plan.strategy_state_dict.get("trade_id_int")) == rec["trade_id_int"] + len(rec["entries"]),
        "live_exits": sorted(plan.exit_asset_set), "bt_exits": rec["exits"],
        "live_entries": live_entries, "bt_entries": rec["entries"],
        "max_weight_abs_diff": float(np.max(np.abs(np.array(live_w) - np.array(bt_w)))) if live_w and len(live_w) == len(bt_w) else 0.0,
        "reuse_key": plan.snapshot_metadata_dict.get("hpi_exit_slot_reuse_str"),
        "signal_ts": str(plan.signal_timestamp_ts), "target_exec_ts": str(plan.target_execution_timestamp_ts),
    }
    out["match"] = all(out[k] for k in ("exits_equal", "entries_equal", "weights_equal_1e-6",
                                        "pending_after_equal", "trade_id_equal"))
    return out


def part_replay() -> None:
    rec_by_variant = {v: load_records(v) for v in hc.VARIANTS}
    index = {v: {r["signal_date"]: r for r in recs} for v, recs in rec_by_variant.items()}
    dates = select_dates(rec_by_variant)
    print(len(dates), "dates:", [str(d.date()) for d in dates])
    rows = []
    for d in dates:
        for v in hc.VARIANTS:
            rec = index[v].get(d)
            if rec is None:
                continue
            t0 = time.time()
            try:
                plan = strategy_host.build_decision_plan_for_release(release(v), as_of(d), pod_state(v, rec))
                row = {"date": str(d.date()), "variant": v, "tags": sorted(classify(rec)),
                       "n_held": len(rec["positions"]), **compare(plan, rec)}
            except Exception as exc:
                row = {"date": str(d.date()), "variant": v, "match": False,
                       "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-1500:]}
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            print({k: row.get(k) for k in ("date", "variant", "tags", "match", "live_exits", "live_entries",
                                           "bt_exits", "bt_entries", "error")})
            hc.dump_json({"rows": rows, "loads": LOAD_LOG}, "live_replay/replay.json")
    summary = {v: {"n": sum(r["variant"] == v for r in rows), "match": sum(r["variant"] == v and r["match"] for r in rows),
                   "refill_days": sum(r["variant"] == v and "refill_full_book" in r.get("tags", []) for r in rows),
                   "n_exits": sum(len(r.get("bt_exits", [])) for r in rows if r["variant"] == v),
                   "n_entries": sum(len(r.get("bt_entries", [])) for r in rows if r["variant"] == v)}
               for v in hc.VARIANTS}
    hc.dump_json({"summary": summary, "rows": rows, "loads": LOAD_LOG}, "live_replay/replay.json")
    print(summary)


# ------------------------------------------------------------------ B4 synthetic-state cases
B4_CASES = [
    # (label, T, halted/special symbol, kind)
    ("member_halt_on_T_BIIB_2020", "2020-11-06", "BIIB", "no_bar_on_T"),
    ("member_halt_on_T_VRTX_2015", "2015-05-12", "VRTX", "no_bar_on_T"),
    ("member_halt_on_T_BIIB_2023", "2023-06-09", "BIIB", "no_bar_on_T"),
    ("pending_exit_no_open_T1_BIIB_2020", "2020-11-05", "BIIB", "pending_no_open_T1"),
    ("pending_exit_no_open_T1_REGN_2015", "2015-06-08", "REGN", "pending_no_open_T1"),
    ("held_not_pending_no_open_T1_BIIB_2023", "2023-06-08", "BIIB", "held_no_open_T1"),
]


def backtest_iterate_on_state(variant: str, universe, signal_df, pricing, T, positions, pending, rec):
    """Production iterate() on a given state with the REAL Open_(T+1) row (what the backtest would do)."""
    s = hc.make_strategy(variant, universe, cls=hpi_mod.HPIStatefulLongStrategy)
    s._position_amount_map = dict(positions)
    s._total_value_history_list = [float(rec["prev_total_value"])]
    s.pending_exit_symbol_set = set(pending)
    s.trade_id_int = int(rec["trade_id_int"])
    s.current_trade_map.update(rec["current_trade_map"])
    t1 = pricing.index[pricing.index.get_loc(T) + 1]
    s.previous_bar, s.current_bar = T, t1
    open_row = pricing.loc[t1, (slice(None), "Open")]
    open_row.index = open_row.index.get_level_values(0)
    s.iterate(signal_df.loc[:T], signal_df.loc[T], open_row)
    orders = s.get_orders()
    return {"exits": sorted(str(o.asset) for o in orders if o.target),
            "entries": [str(o.asset) for o in orders if not o.target],
            "exec_date": str(t1.date()),
            "open_T1_of_symbol": None}


def part_b4() -> None:
    data = hc.load_full_inputs()
    pricing, universe = data["pricing_df"], data["universe"]
    rec_idx = {r["signal_date"]: r for r in load_records("vote")}
    rows = []
    signal_cache: dict = {}
    for label, T_str, sym, kind in B4_CASES:
        T = pd.Timestamp(T_str)
        rec = rec_idx.get(T)
        if rec is None:
            rows.append({"label": label, "error": "no backtest record"})
            continue
        base_pos = dict(rec["positions"])
        others = [s for s in base_pos if s != sym]
        positions = {s: base_pos[s] for s in others[:9]}
        positions[sym] = 10.0
        pending = [sym] if kind == "pending_no_open_T1" else []
        t1 = pricing.index[pricing.index.get_loc(T) + 1]
        info = {
            "label": label, "T": T_str, "symbol": sym, "kind": kind, "n_held": len(positions),
            "bar_on_T": bool(pd.notna(pricing.loc[T, (sym, "High")])),
            "open_T1": None if pd.isna(pricing.loc[t1, (sym, "Open")]) else float(pricing.loc[t1, (sym, "Open")]),
            "member_on_T_per_universe": int(universe.loc[T, sym]),
            "member_on_T1_per_universe": int(universe.loc[t1, sym]),
        }
        live_plan = strategy_host.build_decision_plan_for_release(
            release("vote"), as_of(T), pod_state("vote", rec, positions=positions, pending=pending))
        info["live_exits"] = sorted(live_plan.exit_asset_set)
        info["live_entries"] = list(live_plan.entry_priority_list)
        if T not in signal_cache:
            signal_cache.clear()
            s0 = hc.make_strategy("vote", universe)
            signal_cache[T] = s0.compute_signals(hc.subset_pricing(pricing, universe, T - pd.Timedelta(days=10), t1,
                                                                   extra=positions.keys()))
        bt = backtest_iterate_on_state("vote", universe, signal_cache[T], pricing, T, positions, pending, rec)
        info["backtest_exits"], info["backtest_entries"] = bt["exits"], bt["entries"]
        info["same_orders"] = (info["live_exits"] == bt["exits"] and info["live_entries"] == bt["entries"])
        rows.append(info)
        print(info)
        hc.dump_json(rows, "live_replay/b4_cases.json")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "replay"
    (part_replay if part == "replay" else part_b4)()
