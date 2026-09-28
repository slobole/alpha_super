"""B1/B2/B4 live-parity replay for DV2 and QPI (research-only, no broker, no state store).

For each selected historical session T the REAL live builder
    alpha.live.strategy_host._build_dv2_decision_plan / _build_qpi_ibs_rsi_exit_decision_plan
is called with
  * data truncated at T: the strategy module's `build_index_constituent_matrix` and `get_prices` (the two functions
    the builder calls, strategy_host.py:436-441 / 509-514) are patched to return the cached Norgate data exactly as
    the direct loader would return it on T: prices sliced to [start_date, T] and the universe the loader would build
    from data ending at T (as-of 5-session trim, aud_data.asof_trimmed_universe).  The VPS snapshot profile is not
    available on this workstation (direct Norgate mode; profile differences audited by code read);
  * a PodState equal to the production backtest's state before its decision at T (positions in engine units, cash,
    previous_total_value, trade_id, current_trade_map) recorded by aud_common.Rec* in full_runs/<fam>_base.
The DecisionPlan (exit_asset_set, entry_priority_list, entry_target_weight_map_dict) is compared to the backtest's
recorded orders at T: exits as sets, entries as ORDERED lists, weights = entry_value / previous_total_value to 1e-12.

Usage: uv run python aud_live_replay.py <dv2|qpi> [n_sessions]
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import aud_common as ac
import aud_data
from alpha.live import strategy_host
from alpha.live.models import LiveRelease, PodState

IMPORT = {"dv2": "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
          "qpi": "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy"}
MODULE = {"dv2": "strategies.dv2.strategy_mr_dv2", "qpi": "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit"}
BUILDER = {"dv2": strategy_host._build_dv2_decision_plan, "qpi": strategy_host._build_qpi_ibs_rsi_exit_decision_plan}
PARAMS = {
    "dv2": {"benchmark_list_str": ["$SPX"], "start_date_str": "1998-01-01", "indexname_str": "S&P 500",
            "max_positions_int": 10, "slippage_float": 0.00025, "commission_per_share_float": 0.005,
            "commission_minimum_float": 1.0},
    "qpi": {"benchmark_list_str": ["$SPX"], "start_date_str": "1998-01-01", "indexname_str": "S&P 500",
            "max_positions_int": 10, "qpi_threshold_float": 30.0, "sma_window_int": 200, "qpi_window_int": 3,
            "qpi_lookback_years_int": 5, "return_lookback_days_int": 3, "max_entry_ibs_float": 0.1,
            "exit_ibs_threshold_float": 0.90, "rsi_window_int": 2, "exit_rsi2_threshold_float": 90.0,
            "slippage_float": 0.00025, "commission_per_share_float": 0.005, "commission_minimum_float": 1.0},
}


def make_release(fam: str) -> LiveRelease:
    # Values from docs/live/release_templates/pod_<fam>_daily_moo.yaml.example
    return LiveRelease(release_id_str=f"audit.pod_{fam}.daily_moo.v1", user_id_str="audit", pod_id_str=f"pod_{fam}_audit",
                       account_route_str="DU0000000", strategy_import_str=IMPORT[fam], mode_str="paper",
                       session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
                       execution_policy_str="next_open_moo", data_profile_str="norgate_eod_sp500_pit",
                       params_dict=dict(PARAMS[fam]), risk_profile_str="standard_equity_mr", enabled_bool=True,
                       source_path_str="<audit>", pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False)


class DataPatch:
    """Patch the strategy module's loader functions to serve cached data as of `asof_ts`."""

    def __init__(self, fam, pricing, untrimmed, universe_override=None):
        import importlib
        self.mod = importlib.import_module(MODULE[fam])
        self.pricing, self.untrimmed, self.override = pricing, untrimmed, universe_override
        self.calls = []

    def __enter__(self):
        self.orig = (self.mod.build_index_constituent_matrix, self.mod.get_prices)
        patch = self

        def build_index_constituent_matrix(indexname="S&P 500"):
            universe = patch.override if patch.override is not None else aud_data.asof_trimmed_universe(
                patch.untrimmed, patch.asof_data_ts)
            return list(universe.columns), universe

        def get_prices(symbols, benchmarks, *args, **kwargs):
            start = kwargs.get("start_date", kwargs.get("start_date_str", args[0] if args else "1998-01-01"))
            end = kwargs.get("end_date", kwargs.get("end_date_str", args[1] if len(args) > 1 else None))
            want = set(map(str, symbols)) | set(map(str, benchmarks))
            cols = [c for c in patch.pricing.columns if str(c[0]) in want]
            frame = patch.pricing.loc[pd.Timestamp(start): pd.Timestamp(end), cols].copy()
            frame.attrs.update(patch.pricing.attrs)
            patch.calls.append({"n_symbols_requested": len(want), "end": str(end), "last_row": str(frame.index[-1].date())})
            return frame

        self.mod.build_index_constituent_matrix = build_index_constituent_matrix
        self.mod.get_prices = get_prices
        return self

    def __exit__(self, *exc):
        self.mod.build_index_constituent_matrix, self.mod.get_prices = self.orig


def pod_state_from_record(fam: str, rec: dict) -> PodState:
    return PodState(pod_id_str=f"pod_{fam}_audit", user_id_str="audit", account_route_str="DU0000000",
                    position_amount_map=dict(rec["positions"]), cash_float=float(rec["cash"]),
                    total_value_float=float(rec["prev_total_value"]),
                    strategy_state_dict={"trade_id_int": int(rec["trade_id"]),
                                         "current_trade_map": dict(rec["current_trade_map"])},
                    updated_timestamp_ts=datetime(2000, 1, 1))


def backtest_intents(rec: dict) -> dict:
    exits = sorted(o["asset"] for o in rec["orders"] if o["target"] and abs(o["amount"]) <= 1e-9)
    entries = [(o["asset"], o["amount"] / rec["prev_total_value"]) for o in rec["orders"]
               if not o["target"] and o["unit"] == "value" and o["amount"] > 0]
    return {"exits": exits, "entry_order": [a for a, _ in entries], "weights": dict(entries)}


def compare(plan, bt: dict) -> dict:
    live_exits = sorted(plan.exit_asset_set)
    live_order = list(plan.entry_priority_list)
    w_live = dict(plan.entry_target_weight_map_dict)
    w_diff = max([abs(w_live.get(a, np.nan) - w) for a, w in bt["weights"].items()] + [0.0])
    match = (live_exits == bt["exits"] and live_order == bt["entry_order"]
             and set(w_live) == set(bt["weights"]) and (w_diff <= 1e-12))
    return {"match": bool(match), "live_exits": live_exits, "bt_exits": bt["exits"], "live_entries": live_order,
            "bt_entries": bt["entry_order"], "max_weight_abs_diff": float(w_diff) if np.isfinite(w_diff) else None,
            "live_weights": w_live}


def select_sessions(log: list[dict], n: int, extra_dates: list) -> list[dict]:
    by_date = {r["decision_date"]: r for r in log}
    eligible = [r for r in log if r["decision_date"] >= pd.Timestamp("2007-01-03")
                and any(o["target"] for o in r["orders"]) and any(not o["target"] for o in r["orders"])]
    idx = np.linspace(0, len(eligible) - 1, n).round().astype(int)
    chosen = {eligible[i]["decision_date"]: eligible[i] for i in idx}
    # full book with exits and entries at the same decision = same-open slot reuse
    full = [r for r in eligible if len(r["positions"]) == 10]
    for r in full[:: max(1, len(full) // 4)][:4]:
        chosen[r["decision_date"]] = r
    for d in extra_dates:
        d = pd.Timestamp(d)
        if d in by_date:
            chosen[d] = by_date[d]
    last = log[-1]
    chosen[last["decision_date"]] = last
    return [chosen[d] for d in sorted(chosen)]


def main(fam: str, n: int = 30) -> None:
    data = aud_data.load()
    pricing, untrimmed, trimmed = data["pricing_df"], data["universe_untrimmed"], data["universe_trimmed"]
    del data
    with (ac.OUT / "full_runs" / f"{fam}_base" / "decision_log.pkl").open("rb") as handle:
        log = pickle.load(handle)
    with (ac.OUT / "full_runs" / f"{fam}_untrimmed" / "decision_log.pkl").open("rb") as handle:
        log_u = pickle.load(handle)
    # Dates where the trimmed (production) and untrimmed (live-universe) backtests make different decisions from
    # the SAME state are only identifiable before the paths diverge; use the first differing decision dates.
    trim_dates = []
    for r_b, r_u in zip(log, log_u):
        if r_b["decision_date"] != r_u["decision_date"]:
            break
        if r_b["orders"] != r_u["orders"]:
            trim_dates.append(r_b["decision_date"])
            if len(trim_dates) >= 1:
                break
    # Decision dates >= 2007 inside a trim window (a name that is still a member at T in the untrimmed / as-of
    # universe is already dropped by the full-history trim) on which the backtest opened new positions.
    trim_window = []
    ut = untrimmed.reindex(columns=trimmed.columns, fill_value=0)
    for r in log:
        T = r["decision_date"]
        if T < pd.Timestamp("2007-01-03") or not any(not o["target"] for o in r["orders"]):
            continue
        if T in ut.index and T in trimmed.index and bool(((ut.loc[T] == 1) & (trimmed.loc[T] == 0)).any()):
            trim_window.append(T)
    trim_pick = [trim_window[i] for i in np.linspace(0, len(trim_window) - 1, 8).round().astype(int)] if trim_window else []
    trim_dates = [d for d in trim_dates if d >= pd.Timestamp("2006-10-02")] + trim_pick
    extra = trim_dates + ["2024-11-27", "2025-12-24", "2008-10-10", "2020-03-16"]
    sessions = select_sessions(log, n, extra)
    release = make_release(fam)
    rows = []
    t0 = time.time()
    for rec in sessions:
        T = rec["decision_date"]
        bt = backtest_intents(rec)
        with DataPatch(fam, pricing, untrimmed) as patch:
            patch.asof_data_ts = T
            plan = BUILDER[fam](release, datetime(T.year, T.month, T.day, 20, 0), pod_state_from_record(fam, rec))
        cmp = compare(plan, bt)
        row = {"decision_date": T.date().isoformat(), "universe": "asof_T", "n_held": len(rec["positions"]),
               "n_exits": len(bt["exits"]), "n_entries": len(bt["entry_order"]),
               "same_open_reuse": bool(len(rec["positions"]) == 10 and len(bt["exits"]) > 0 and len(bt["entry_order"]) > 0),
               "signal_ts": str(plan.signal_timestamp_ts), "target_exec_ts": str(plan.target_execution_timestamp_ts),
               "decision_book_type": plan.decision_book_type_str, **cmp}
        if not cmp["match"]:
            with DataPatch(fam, pricing, untrimmed, universe_override=trimmed.loc[trimmed.index.isin(pricing.index)]) as patch:
                patch.asof_data_ts = T
                plan2 = BUILDER[fam](release, datetime(T.year, T.month, T.day, 20, 0), pod_state_from_record(fam, rec))
            row["rerun_with_backtest_universe"] = compare(plan2, bt)
        rows.append(row)
        print(row["decision_date"], row["match"], row["n_exits"], row["n_entries"], round(time.time() - t0), flush=True)

    # B2 invocation timing: holiday / weekend invocation must reproduce the last session's decision.
    b2 = []
    for as_of, expected in (("2025-12-25", "2025-12-24"), ("2026-09-26", "2026-09-25"), ("2024-11-28", "2024-11-27")):
        exp_ts = pd.Timestamp(expected)
        rec = next((r for r in log if r["decision_date"] == exp_ts), None)
        with DataPatch(fam, pricing, untrimmed) as patch:
            patch.asof_data_ts = exp_ts
            state_rec = rec if rec is not None else log[-1]
            plan = BUILDER[fam](release, datetime(*map(int, as_of.split("-")), 9, 0), pod_state_from_record(fam, state_rec))
        entry = {"as_of": as_of, "signal_ts": str(plan.signal_timestamp_ts),
                 "target_exec_ts": str(plan.target_execution_timestamp_ts),
                 "exits": sorted(plan.exit_asset_set), "entries": list(plan.entry_priority_list)}
        if rec is not None:
            entry["matches_backtest_decision"] = compare(plan, backtest_intents(rec))["match"]
        else:
            entry["note"] = "no backtest decision at the last bar (engine needs Open_(T+1)); plan shown for inspection"
        b2.append(entry)
        print("B2", entry, flush=True)

    out = {"fam": fam, "n_sessions": len(rows), "n_match": int(sum(r["match"] for r in rows)),
           "n_same_open_reuse_sessions": int(sum(r["same_open_reuse"] for r in rows)),
           "trim_divergence_dates_injected": [d.date().isoformat() for d in trim_dates],
           "runtime_s": round(time.time() - t0, 1), "rows": rows, "b2_invocation": b2,
           "note": "direct Norgate mode on the workstation; VPS snapshot profile norgate_eod_sp500_pit not available"}
    path = ac.OUT / "live_replay" / f"{fam}_replay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "b2_invocation")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
