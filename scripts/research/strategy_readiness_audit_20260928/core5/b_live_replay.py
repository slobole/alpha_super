"""B1 live-parity replay for CORE5 on real Norgate data (current vintage).

The historical engine runs normally. On each selected decision date T the live adapter
`alpha.live.core5_adapter.build_core5_decision_from_prices` is called with
  * prices explicitly truncated at T (df.loc[:T]),
  * an EOD pod state equal to the engine's state at Close_T (cash, signed whole shares, NAV),
  * strategy memory built from the engine (T-1 long states, last target weights, last rebalance date).
Compared exactly: rebalance trigger, full target-weight book (1e-6 and bit-level), frozen whole-share targets
vs the shares the engine holds after the Open_(T+1) fills, no-order flag, and the broker order legs produced by
build_vplan -> build_broker_order_request_list_from_vplan (MOO, shares) at quote multipliers 1.0 and 1.2.

Usage: python b_live_replay.py [n_random_non_rebalance_days]
"""
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import core5_common as c
from alpha.engine.backtest import run_daily
from alpha.live import scheduler_utils
from alpha.live.core5_adapter import CORE5_ASSET_TUPLE, CORE5_STRATEGY_IMPORT_STR, build_core5_decision_from_prices
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, LivePriceSnapshot, LiveRelease, PodState

core5 = c.core5
N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 80

df = c.load_pricing()
release_obj = LiveRelease(
    release_id_str="core5.audit_replay.v1", user_id_str="audit", pod_id_str="core5_audit",
    account_route_str="SIM_CORE5_AUDIT", strategy_import_str=CORE5_STRATEGY_IMPORT_STR,
    mode_str="incubation", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
    execution_policy_str="next_open_moo", data_profile_str="norgate_eod_core5", params_dict={},
    risk_profile_str="audit", enabled_bool=False, source_path_str="in_memory_only",
    pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
)

# ---- choose decision dates: all month-ends, all long-state changes, DBC-short periods, random others ----
cfg = core5.DEFAULT_CONFIG
calendar_idx = core5.build_execution_calendar_idx(df, cfg, cfg.backtest_start_date_str)
probe = core5.AdaptiveMacroCore5Strategy()
full_sig = probe.compute_signals(df)
decision_dates = pd.DatetimeIndex(df.index[df.index.get_indexer(calendar_idx) - 1])  # previous_bar of each bar
me = full_sig[(core5.PORTFOLIO_NAMESPACE_STR, core5.MONTH_END_REBALANCE_FIELD_STR)].reindex(decision_dates).astype(bool)
ch = full_sig[(core5.PORTFOLIO_NAMESPACE_STR, core5.LONG_STATE_CHANGED_FIELD_STR)].reindex(decision_dates).astype(bool)
sel = set(decision_dates[me.to_numpy()]) | set(decision_dates[ch.to_numpy()])
rng = np.random.default_rng(20260928)
others = [d for d in decision_dates if d not in sel]
sel |= set(pd.DatetimeIndex(rng.choice(np.array(others, dtype="datetime64[ns]"), size=N_RANDOM, replace=False)))
sel.add(decision_dates[0])
print(f"selected decision dates: {len(sel)} of {len(decision_dates)}", flush=True)

oracle = core5.AdaptiveMacroCore5Strategy()
orig_iterate = oracle.iterate
orig_process = oracle.process_orders
pending = {}
rows = []
fails = []
last_rebalance = {"d": None}


def state_from_engine(T):
    if not oracle.initialized_bool:
        return {}
    prev_ts = scheduler_utils.get_exchange_calendar_obj("XNYS").previous_session(T)
    prev_long = oracle._long_state_ser(full_sig.loc[prev_ts])
    return {
        "core5_state_version_int": 1,
        "initialized_bool": True,
        "last_signal_date_str": prev_ts.date().isoformat(),
        "last_long_state_map_dict": {a: int(v) for a, v in prev_long.items()},
        "last_target_weight_map_dict": {k: float(v) for k, v in oracle.last_target_weight_ser.items()},
        "last_rebalance_date_str": last_rebalance["d"].date().isoformat(),
    }


def my_iterate(signal_df, close_row_ser, open_price_ser):
    T = pd.Timestamp(oracle.previous_bar)
    pending.clear()
    if T in sel:
        as_of_ts = (T.tz_localize("America/New_York") + pd.Timedelta(hours=18)).to_pydatetime()
        pos = {a: float(v) for a, v in oracle.get_positions().to_dict().items() if float(v) != 0.0}
        nav = float(oracle.previous_total_value)
        st = PodState(release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
                      pos, float(oracle.cash), nav, state_from_engine(T), as_of_ts,
                      snapshot_stage_str="eod", snapshot_source_str="virtual_broker")
        try:
            dec = build_core5_decision_from_prices(release_obj, as_of_ts, st, df.loc[:T].copy(), {
                "norgate_snapshot_date_str": T.date().isoformat(), "norgate_data_profile_str": "norgate_eod_core5",
                "norgate_manifest_hash_str": "audit_prefix_reconstruction"})
            pending.update({"T": T, "dec": dec, "pos": pos, "nav": nav, "cash": float(oracle.cash),
                            "close": close_row_ser})
        except Exception as exc:  # recorded, never swallowed silently
            fails.append({"date": T.date().isoformat(), "stage": "adapter", "error": f"{type(exc).__name__}: {exc}"})
    n_before = len(oracle.rebalance_target_weight_row_dict_list)
    orig_iterate(signal_df, close_row_ser, open_price_ser)
    engine_rebalanced = len(oracle.rebalance_target_weight_row_dict_list) > n_before
    if engine_rebalanced:
        last_rebalance["d"] = T
    if pending:
        dec = pending["dec"]
        md = dec.snapshot_metadata_dict
        eng_w = oracle.last_target_weight_ser
        ad_w = {**dec.full_target_weight_map_dict, "Cash": dec.cash_reserve_weight_float}
        wdiff = max(abs(float(ad_w[k]) - float(eng_w.get(k, 0.0))) for k in ad_w)
        # order legs from the engine's queued orders, sized exactly as process_orders will size them
        exp_pos = dict(pending["pos"])
        exp_leg = defaultdict(list)
        for o in oracle.get_orders():
            amt = o.amount_in_shares(float(close_row_ser[(o.asset, "Close")]), pending["nav"], exp_pos.get(o.asset, 0.0))
            exp_pos[o.asset] = exp_pos.get(o.asset, 0.0) + amt
            if amt:
                exp_leg[o.asset].append(float(amt))
        leg_ok = True
        for mult in (1.0, 1.2):
            acct = BrokerSnapshot(release_obj.account_route_str, dec.submission_timestamp_ts, cash_float=pending["cash"],
                                  total_value_float=pending["nav"] * mult, position_amount_map=pending["pos"])
            quote = LivePriceSnapshot(release_obj.account_route_str, dec.submission_timestamp_ts, "audit_quote",
                                      {a: float(close_row_ser[(a, "Close")]) * mult for a in CORE5_ASSET_TUPLE})
            try:
                vplan = build_vplan(release_obj, dec, acct, quote)
                got = defaultdict(list)
                for req in build_broker_order_request_list_from_vplan(vplan):
                    assert req.unit_str == "shares" and req.broker_order_type_str == "MOO"
                    got[req.asset_str].append(float(req.amount_float))
                leg_ok &= dict(got) == dict(exp_leg)
                if dict(got) != dict(exp_leg) and mult == 1.0:
                    fails.append({"date": T.date().isoformat(), "stage": "vplan_legs", "got": dict(got), "exp": dict(exp_leg)})
            except Exception as exc:
                leg_ok = False
                fails.append({"date": T.date().isoformat(), "stage": f"vplan_x{mult}", "error": f"{type(exc).__name__}: {exc}"})
        pending.update({"engine_rebalanced": engine_rebalanced, "wdiff": wdiff, "leg_ok": leg_ok,
                        "exp_leg": dict(exp_leg), "short": float(ad_w["DBC"]) < 0})


def my_process(prices_df):
    orig_process(prices_df)
    if not pending:
        return
    dec = pending["dec"]
    md = dec.snapshot_metadata_dict
    held = {a: float(oracle.get_position(a)) for a in CORE5_ASSET_TUPLE}
    tgt = {a: float(md["fixed_target_share_map_dict"][a]) for a in CORE5_ASSET_TUPLE}
    row = {
        "decision_date": pending["T"].date().isoformat(),
        "adapter_rebalance": bool(md["rebalance_bool"]), "engine_rebalance": bool(pending["engine_rebalanced"]),
        "adapter_month_end": bool(md["month_end_bool"]), "adapter_state_changed": bool(md["long_state_changed_bool"]),
        "max_weight_diff": pending["wdiff"], "target_shares_equal_engine_holdings": tgt == held,
        "no_order_adapter": bool(md["no_order_bool"]), "no_order_engine": not pending["exp_leg"],
        "vplan_legs_equal": bool(pending["leg_ok"]), "dbc_short": bool(pending["short"]),
        "dbc_two_legs": len(pending["exp_leg"].get("DBC", [])) == 2,
        "nav_diff": abs(float(md["sizing_close_nav_float"]) - pending["nav"]),
    }
    row["pass"] = bool(row["adapter_rebalance"] == row["engine_rebalance"] and row["max_weight_diff"] <= 1e-12
                       and row["target_shares_equal_engine_holdings"] and row["no_order_adapter"] == row["no_order_engine"]
                       and row["vplan_legs_equal"] and row["nav_diff"] < 1e-6)
    if not row["pass"]:
        fails.append({"date": row["decision_date"], "stage": "compare", "row": row, "tgt": tgt, "held": held})
    rows.append(row)
    pending.clear()


oracle.iterate = my_iterate
oracle.process_orders = my_process
oracle.configure_run_calendar(calendar_idx)
run_daily(oracle, df, calendar=calendar_idx, show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)

r = pd.DataFrame(rows)
summary = {
    "n_decisions_replayed": int(len(r)),
    "n_pass": int(r["pass"].sum()),
    "n_fail_records": len(fails),
    "n_month_end": int(r["adapter_month_end"].sum()),
    "n_month_end_distinct_months": int(pd.to_datetime(r.loc[r["adapter_month_end"], "decision_date"]).dt.to_period("M").nunique()),
    "n_rebalance": int(r["engine_rebalance"].sum()),
    "n_state_change": int(r["adapter_state_changed"].sum()),
    "n_no_order": int(r["no_order_engine"].sum()),
    "n_with_dbc_short_target": int(r["dbc_short"].sum()),
    "n_dbc_two_leg_flips": int(r["dbc_two_legs"].sum()),
    "max_weight_diff": float(r["max_weight_diff"].max()),
    "max_nav_diff": float(r["nav_diff"].max()),
    "first_date": r["decision_date"].min(), "last_date": r["decision_date"].max(),
    "fails": fails[:20],
}
print(json.dumps(summary, indent=2, default=str))
c.dump(summary, "b1_live_replay_summary.json")
r.to_csv(c.OUT / "b1_live_replay_rows.csv", index=False)
