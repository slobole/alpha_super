"""What a live route for the unwired TAA 2x / linearity variants would decide (LP evidence for 'NOT WIRED' notes).

None of the four variants has a host builder (``alpha/live/strategy_host.py:1298-1363`` raises NotImplementedError).
A live route would copy the TAA 1/N builder pattern (``strategy_host.py:878-915``, linearity ``:918-958``) and call
the shared ``_build_taa_vix_cash_full_target_decision_plan`` (``:961-1098``). This study builds exactly that copy at
runtime (no repository file changed), calls it at 20:00 New York on historical month-end decision sessions with the
REAL Norgate loaders (direct mode, profile ``norgate_eod_etf_plus_vix_helper``), DTB3 in live mode from the audit's
offline cache copy, and compares the live full-target weights with the backtest rebalance row.
A4 control: the same replay with the signal close read at T+1 must show differences.

Usage: uv run python taa2x_host_pattern_replay.py [variant ...]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from importlib import import_module
from zoneinfo import ZoneInfo

import pandas as pd

import taa2x_common as tc
from alpha.live import strategy_host
from alpha.live.models import LiveRelease
from data.norgate_snapshot_store import use_norgate_data_profile

NY = ZoneInfo("America/New_York")


def _release(key: str) -> LiveRelease:
    return LiveRelease(release_id_str=f"audit.{key}", user_id_str="audit_user", pod_id_str=f"audit_{key}",
                       account_route_str="U00000000", strategy_import_str=tc.VARIANT_DICT[key]["module"], mode_str="live",
                       session_calendar_id_str="XNYS", signal_clock_str="month_end_snapshot_ready",
                       execution_policy_str="next_month_first_open", data_profile_str="norgate_eod_etf_plus_vix_helper",
                       params_dict={"capital_base_float": 100_000.0}, risk_profile_str="audit", enabled_bool=False,
                       source_path_str="audit")


CLIP_TO_CALENDAR_BOOL = True  # study workaround, see F-finding: host session loop raises before the XNYS window


def copied_builder(key: str, release_obj: LiveRelease, as_of_ts: datetime):
    """Runtime copy of strategy_host._build_taa_btal_1n_tqqq_vix_cash_decision_plan / _linearity_ with the variant module."""
    variant_module = import_module(tc.VARIANT_DICT[key]["module"])
    base_taa_module = import_module("strategies.taa_df.strategy_taa_df")
    vix = tc.utils_module
    if tc.VARIANT_DICT[key]["kind"] == "standard":
        config_obj = replace(variant_module.DEFAULT_CONFIG, end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
                             dtb3_mode_str="live", dtb3_as_of_timestamp_ts=as_of_ts)
        execution_price_df, _, base_w, _, dtb3_snapshot_obj = base_taa_module.get_defense_first_data_with_snapshot(config_obj)
        extra = strategy_host._build_dtb3_snapshot_metadata_dict(dtb3_snapshot_obj)
    else:
        config_obj = replace(variant_module.DEFAULT_CONFIG, end_date_str=pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"))
        execution_price_df, _, _, base_w, _ = variant_module.get_defense_first_linearity_1n_data(config_obj)
        extra = None
    _, me_vrp = vix._load_vrp_overlay_signal_frames(config_obj)
    w, diag = vix.apply_vrp_cash_gate_to_month_end_weight_df(base_month_end_weight_df=base_w, month_end_vrp_signal_df=me_vrp, config=config_obj)
    if CLIP_TO_CALENDAR_BOOL:
        # *** CRITICAL*** exchange_calendars' default XNYS window starts 20 years before today; the host's
        # resolve_calendar_month_end_label_to_last_tradable_session() checks every available price date against it.
        # The host uses execution_price_df only for dates and Close_T, so clipping old rows is decision-neutral.
        first_session_ts = pd.Timestamp(strategy_host.scheduler_utils.get_exchange_calendar_obj("XNYS").first_session)
        execution_price_df = execution_price_df.loc[execution_price_df.index >= first_session_ts]
    with use_norgate_data_profile(release_obj.data_profile_str):
        return strategy_host._build_taa_vix_cash_full_target_decision_plan(
            release_obj=release_obj, as_of_ts=as_of_ts, pod_state_obj=None, base_taa_module=base_taa_module,
            config_obj=config_obj, execution_price_df=execution_price_df, month_end_weight_df=w,
            month_end_vrp_diagnostic_df=diag, strategy_family_str=f"audit_copy_{key}", snapshot_metadata_extra_dict=extra)


def replay(key: str, rebalance_dates: list[pd.Timestamp], full_rb: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    rel = _release(key)
    rows = []
    for rb in rebalance_dates:
        t = sessions[sessions < rb][-1]
        as_of = datetime(t.year, t.month, t.day, 20, 0, tzinfo=NY)
        bt = {k: float(v) for k, v in full_rb.loc[rb].items() if abs(float(v)) > 1e-12}
        try:
            plan = copied_builder(key, rel, as_of)
            live = {k: float(v) for k, v in plan.full_target_weight_map_dict.items()}
            diff = max(abs(live.get(a, 0.0) - bt.get(a, 0.0)) for a in set(live) | set(bt))
            exe = pd.Timestamp(plan.target_execution_timestamp_ts).tz_convert(NY).date()
            rows.append({"decision": t.date().isoformat(), "rebalance": rb.date().isoformat(), "status": "ok",
                         "max_abs_weight_diff": diff, "exec_date_match": exe == rb.date(),
                         "dtb3_latest_obs": plan.snapshot_metadata_dict.get("dtb3_latest_observation_date_str", "")})
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            rows.append({"decision": t.date().isoformat(), "rebalance": rb.date().isoformat(),
                         "status": f"error: {type(exc).__name__}: {exc}"[:300]})
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    if len(ok) == 0:
        return {"decisions": int(len(df)), "ok": 0, "exact_1e-9": 0, "errors": df["status"].head(3).tolist(), "_rows": rows}
    return {"decisions": int(len(df)), "ok": int(len(ok)), "exact_1e-9": int((ok["max_abs_weight_diff"] <= 1e-9).sum()),
            "max_abs_weight_diff": float(ok["max_abs_weight_diff"].max()) if len(ok) else None,
            "exec_date_mismatch": int((~ok["exec_date_match"].astype(bool)).sum()) if len(ok) else None,
            "non_empty_fallback_or_defensive_rows": int(len(ok)),
            "errors": df.loc[df["status"] != "ok", "status"].head(3).tolist(), "_rows": rows}


def main() -> None:
    global CLIP_TO_CALENDAR_BOOL
    keys = sys.argv[1:] or list(tc.VARIANT_DICT)
    sessions = tc.xnys_sessions()
    out = {}
    for key in keys:
        t0 = time.time()
        CLIP_TO_CALENDAR_BOOL = False
        full_rb0 = tc.weight_frames(key, tc.config(key))[1]
        unclipped = replay(key, list(full_rb0.index[-3:-1]), full_rb0, sessions)
        CLIP_TO_CALENDAR_BOOL = True
        full_rb = tc.weight_frames(key, tc.config(key))[1]
        rb_idx = list(full_rb.index[full_rb.index >= pd.Timestamp("2023-09-01")])  # 37 most recent month-ends
        extra = [d for d in full_rb.index if d.to_period("M") in (pd.Period("2008-10"), pd.Period("2008-11"),
                                                                    pd.Period("2020-04"), pd.Period("2022-07"))]
        dates = sorted(set(extra + rb_idx))
        clean = replay(key, dates, full_rb, sessions)
        real = tc.base_module.compute_month_end_weight_df
        real_lin = tc.linearity_1n_module.compute_daily_linearity_score_df
        tc.base_module.compute_month_end_weight_df = lambda s, c, cfg: real(s.shift(-1), c, cfg)  # planted leak
        tc.linearity_1n_module.compute_daily_linearity_score_df = lambda signal_close_df, lookback_day_vec: real_lin(
            signal_close_df=signal_close_df.shift(-1), lookback_day_vec=lookback_day_vec)
        try:
            leak_rb = tc.weight_frames(key, tc.config(key))[1]
            leak = replay(key, dates[-12:], leak_rb, sessions)
        finally:
            tc.base_module.compute_month_end_weight_df = real
            tc.linearity_1n_module.compute_daily_linearity_score_df = real_lin
        out[key] = {"unclipped_as_a_plain_copy": {"decisions": unclipped["decisions"], "ok": unclipped["ok"],
                                                  "errors": unclipped["errors"]},
                    "clean": {k: v for k, v in clean.items() if k != "_rows"},
                    "planted_leak": {k: v for k, v in leak.items() if k != "_rows"},
                    "planted_leak_caught": leak["exact_1e-9"] < leak["ok"], "elapsed_s": round(time.time() - t0, 1)}
        pd.DataFrame(clean["_rows"]).to_csv(tc.OUT / f"host_pattern_replay_{key}.csv", index=False)
        print(key, json.dumps(out[key]), flush=True)
    (tc.OUT / f"host_pattern_replay_{'_'.join(keys)}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
