"""Review probe (live-parity lens): does the LIVE data path (snapshot mode) give the same
DecisionPlan as the direct-Norgate path the primary replays used, and is the plan invariant to the
seeded PodState (replays passed pod_state=None; live passes the stored broker-truth PodState)?

For TAA 3x (norgate_eod_etf_plus_vix_helper) and NDX ATR VXN (norgate_eod_ndx_pit_plus_vxn_helper):
1. Export a real snapshot with the production exporter (scripts/export_norgate_snapshot.py) for
   snapshot date = decision date T, into the review_live results folder (never the VPS root).
2. Build the plan with build_decision_plan_for_release at T 20:00 New York:
   (a) direct mode, pod_state None; (b) direct mode, hostile PodState (fractional, unrelated,
   negative positions, small NAV, foreign trade map); (c) snapshot mode, hostile PodState.
3. Compare full target weights (exact), signal/execution timestamps and key metadata.
FRED DTB3 is served from the audit cache file (no network). Study code only.
Usage: uv run python <this file> [T ...]   (default T = 2026-08-31)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_live"
SNAP_ROOT = OUT / "snapshots"
OUT.mkdir(parents=True, exist_ok=True)
DTB3_CACHE = REPO / "results/research/strategy_readiness_audit_20260928/taa/DTB3_audit_cache.csv"
NY = ZoneInfo("America/New_York")

import alpha.data.fred_loader as fred_loader_module  # noqa: E402
import data.norgate_snapshot_store as store  # noqa: E402
from alpha.live import strategy_host  # noqa: E402
from alpha.live.models import LiveRelease, PodState  # noqa: E402
from scripts.export_norgate_snapshot import export_profile_snapshot  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _fake_urlopen(url_str, timeout=None):
    return _Resp(DTB3_CACHE.read_bytes())


fred_loader_module.urlopen = _fake_urlopen

VARIANTS = {
    "taa3x": dict(
        import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        profile="norgate_eod_etf_plus_vix_helper",
        params={"capital_base_float": 100_000.0},
        hostile_positions={"TQQQ": 1234.5, "AAPL": 10.0, "GLD": -3.0, "SGOV": 77.0},
    ),
    "ndx_vxn": dict(
        import_str="strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy",
        profile="norgate_eod_ndx_pit_plus_vxn_helper",
        params={
            "max_positions_int": 10, "lookback_month_int": 12, "index_trend_window_int": 200,
            "stock_trend_window_int": 100, "regime_symbol_str": "SPY", "slippage_float": 0.00025,
            "commission_per_share_float": 0.005, "commission_minimum_float": 1.0,
            "capital_base_float": 100_000.0, "vxn_symbol_str": "$VXN", "target_vxn_pct_float": 22.0,
            "min_exposure_scale_float": 0.25, "max_exposure_scale_float": 1.0,
        },
        hostile_positions={"NVDA": 12.7, "TQQQ": 100.0, "BKNG": 1.0, "XYZNOTREAL": 5.0},
    ),
}


def _release(key):
    v = VARIANTS[key]
    return LiveRelease(
        release_id_str=f"review.{key}", user_id_str="review", pod_id_str=f"review_{key}",
        account_route_str="U00000000", strategy_import_str=v["import_str"], mode_str="live",
        session_calendar_id_str="XNYS", signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open", data_profile_str=v["profile"],
        params_dict=dict(v["params"]), risk_profile_str="review", enabled_bool=False, source_path_str="review",
    )


def _hostile_state(key, as_of):
    return PodState(
        pod_id_str=f"review_{key}", user_id_str="review", account_route_str="U00000000",
        position_amount_map=dict(VARIANTS[key]["hostile_positions"]), cash_float=-512.25,
        total_value_float=29_876.5,
        strategy_state_dict={"trade_id_int": 41, "current_trade_map": {"TQQQ": 40, "NVDA": 41}},
        updated_timestamp_ts=as_of,
    )


def _summ(plan):
    md = plan.snapshot_metadata_dict
    keep = ("norgate_data_source_mode_str", "norgate_snapshot_date_str", "vxn_reference_date_str",
            "vxn_close_float", "vxn_exposure_scale_float", "cash_weight_float",
            "dtb3_latest_observation_date_str", "resolved_signal_session_date_str")
    return {
        "weights": {k: float(v) for k, v in sorted(plan.full_target_weight_map_dict.items())},
        "signal_ts": plan.signal_timestamp_ts.isoformat(),
        "target_exec_ts": plan.target_execution_timestamp_ts.isoformat(),
        "decision_base_position_map": plan.decision_base_position_map,
        "cash_reserve": plan.cash_reserve_weight_float,
        "md": {k: md.get(k) for k in keep},
    }


def _build(key, as_of, pod_state):
    t0 = time.time()
    plan = strategy_host.build_decision_plan_for_release(_release(key), as_of, pod_state)
    return _summ(plan), round(time.time() - t0, 1)


def main():
    date_list = sys.argv[1:] or ["2026-08-31"]
    result = {}
    for T in date_list:
        as_of = datetime.fromisoformat(T).replace(hour=20, tzinfo=NY)
        for key, v in VARIANTS.items():
            row = {}
            os.environ.pop(store.ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, None)
            store.clear_snapshot_manifest_cache()
            row["direct_none"], row["t_direct_none"] = _build(key, as_of, None)
            row["direct_hostile"], row["t_direct_hostile"] = _build(key, as_of, _hostile_state(key, as_of))
            t0 = time.time()
            export_profile_snapshot(snapshot_root_str=str(SNAP_ROOT), profile_str=v["profile"], snapshot_date_str=T,
                                    start_date_str="1990-01-01", end_date_str=T, overwrite_bool=False)
            row["t_export"] = round(time.time() - t0, 1)
            os.environ[store.ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR] = "true"
            os.environ[store.NORGATE_SNAPSHOT_ROOT_ENV_STR] = str(SNAP_ROOT)
            store.clear_snapshot_manifest_cache()
            try:
                row["snapshot_hostile"], row["t_snapshot"] = _build(key, as_of, _hostile_state(key, as_of))
            except Exception as exc:  # noqa: BLE001
                row["snapshot_hostile"] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
            finally:
                os.environ.pop(store.ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, None)
                store.clear_snapshot_manifest_cache()
            w0 = row["direct_none"]["weights"]
            row["weights_equal_direct_none_vs_hostile"] = w0 == row["direct_hostile"]["weights"]
            row["weights_equal_direct_vs_snapshot"] = w0 == row["snapshot_hostile"].get("weights")
            snap_w = row["snapshot_hostile"].get("weights")
            if snap_w is not None:
                ks = set(w0) | set(snap_w)
                row["max_abs_weight_diff_direct_vs_snapshot"] = (
                    max(abs(w0.get(k, 0.0) - snap_w.get(k, 0.0)) for k in ks) if ks else 0.0
                )
            result[f"{key}@{T}"] = row
            brief = {k: row[k] for k in row if k.startswith(("weights_equal", "max_abs", "t_"))}
            print(key, T, json.dumps(brief), flush=True)
    (OUT / "snapshot_vs_direct_host_probe.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
