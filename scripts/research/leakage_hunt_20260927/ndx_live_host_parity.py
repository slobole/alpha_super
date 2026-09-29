"""(3) Live-host parity: call alpha.live.strategy_host.build_decision_plan_for_release offline (direct Norgate, no
broker, no network) for past as_of dates with a LiveRelease built from the repo release template, then compare
the DecisionPlan target weights with the backtest decision at the same month-end (decision on the full cached
history; the prefix test shows it equals the as-of decision).  Also checks the VPS snapshot export path keeps
'Unadjusted Close', and records that the NATR20 module has no live-host route.

Usage: uv run python scripts/research/leakage_hunt_20260927/ndx_live_host_parity.py
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime

import pandas as pd

import ndx_common as nc

AS_OF = ["2003-07-01", "2013-07-01", "2014-07-01", "2020-09-01", "2022-08-01", "2024-07-01", "2026-09-01",
         "2026-09-15"]


def make_release():
    from alpha.live.models import LiveRelease
    return LiveRelease(
        release_id_str="audit.pod_ndx_atr_normalized_vxn_scaled.monthly_open.v1", user_id_str="audit",
        pod_id_str="pod_ndx_atr_normalized_vxn_scaled_audit", account_route_str="DU0000000",
        strategy_import_str="strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy",
        mode_str="paper", session_calendar_id_str="XNYS", signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open", data_profile_str="norgate_eod_ndx_pit_plus_vxn_helper",
        params_dict={"max_positions_int": 10, "lookback_month_int": 12, "index_trend_window_int": 200,
                     "stock_trend_window_int": 100, "regime_symbol_str": "SPY", "vxn_symbol_str": "$VXN",
                     "target_vxn_pct_float": 22.0, "min_exposure_scale_float": 0.25, "max_exposure_scale_float": 1.0,
                     "slippage_float": 0.00025, "commission_per_share_float": 0.005, "commission_minimum_float": 1.0},
        risk_profile_str="standard_monthly_momentum_vxn_scaled", enabled_bool=False, source_path_str="<audit>",
        auto_submit_enabled_bool=False,
    )


def main() -> None:
    from alpha.live import strategy_host
    data = nc.load_data("trimmed")
    full_strategy, full_signals = nc.signals("atr_vxn", data["pricing"], data["universe"], data["vxn"])
    release = make_release()
    rows = []
    for as_of_str in AS_OF:
        as_of = datetime.fromisoformat(as_of_str + "T08:00:00")
        try:
            plan = strategy_host.build_decision_plan_for_release(release, as_of, None)
        except Exception as exc:  # record the failure mode (e.g. stale target execution for historic as_of)
            rows.append({"as_of": as_of_str, "error": f"{type(exc).__name__}: {exc}"[:400]})
            print(as_of_str, "ERROR", exc)
            continue
        host_w = {k: float(v) for k, v in plan.full_target_weight_map_dict.items()}
        T = pd.Timestamp(plan.signal_timestamp_ts).tz_localize(None).normalize() if pd.Timestamp(plan.signal_timestamp_ts).tzinfo else pd.Timestamp(plan.signal_timestamp_ts).normalize()
        bt = nc.decision_at(full_strategy, full_signals, T)
        keys = set(host_w) | set(bt["weights"])
        max_diff = max([abs(host_w.get(k, 0.0) - bt["weights"].get(k, 0.0)) for k in keys] or [0.0])
        ok = set(host_w) == set(bt["weights"]) and max_diff < 1e-12
        meta = plan.snapshot_metadata_dict
        rows.append({"as_of": as_of_str, "signal_date": str(T.date()), "passed": ok, "max_weight_diff": max_diff,
                     "host_selected": sorted(host_w), "backtest_selected": bt["selected"],
                     "host_exposure": sum(host_w.values()), "backtest_exposure": bt["exposure"],
                     "vxn_reference_date": meta.get("vxn_reference_date_str"),
                     "vxn_scale_meta": meta.get("vxn_exposure_scale_float"), "vxn_close": meta.get("vxn_close_float"),
                     "cash_reserve": plan.cash_reserve_weight_float})
        print(as_of_str, T.date(), "PASS" if ok else "FAIL", round(max_diff, 14), sorted(host_w)[:10])

    # snapshot export keeps every Norgate field, including Unadjusted Close (read-only call, nothing written)
    import importlib
    export = importlib.import_module("scripts.export_norgate_snapshot")
    snap = {}
    for symbol in ("NVDA", "SPY", "$VXN"):
        frame = export._load_price_frame_df(symbol_str=symbol, adjustment_str="CAPITALSPECIAL",
                                            start_date_str="2024-06-01", end_date_str="2024-06-30")
        snap[symbol] = {"columns": [c for c in frame.columns if c not in ("date", "symbol_str", "adjustment_str")],
                        "has_unadjusted_close": "Unadjusted Close" in frame.columns}
    spec = export.PROFILE_EXPORT_SPEC_DICT["norgate_eod_ndx_pit_plus_vxn_helper"]
    host_src = inspect.getsource(strategy_host.build_decision_plan_for_release)
    report = {
        "host_vs_backtest": rows,
        "n_pass": sum(1 for r in rows if r.get("passed")), "n_fail": sum(1 for r in rows if r.get("passed") is False),
        "n_error": sum(1 for r in rows if "error" in r),
        "snapshot_export_fields": snap,
        "snapshot_profile_trim_past_member_tail_bool": spec.trim_past_member_tail_bool,
        "natr20_module_routed_in_live_host": "natr20" in host_src,
        "natr20_module_has_compute_atr_normalized_signal_tables": hasattr(nc.mod("natr_vxn"), "compute_atr_normalized_signal_tables"),
    }
    (nc.OUT / "live_host_parity.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "host_vs_backtest"}, indent=2, default=str))


if __name__ == "__main__":
    main()
