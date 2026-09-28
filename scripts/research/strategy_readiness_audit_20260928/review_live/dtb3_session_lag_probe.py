"""Review probe (live-parity lens): DTB3 publication lag, re-done with a session-correct lag.

The primary A5 check (taa_bc_checks.check_dtb3_lag) re-dates each DTB3 observation by +1 CALENDAR
day. When a month's last XNYS session is not its last calendar day (weekend or holiday month-end),
the shifted DTB3_T lands on a non-session day of the SAME month and resample("ME").last() still
uses it, so those months were not lagged. Here each observation dated d is re-dated to the next
business day (BDay(1)), so the month-end hurdle can only use observations dated strictly before
the month's last session (live at T 20:00 NY has DTB3_(T-1); DTB3_T is published on T+1).

Also records, per decision, the smallest |defensive score - cash hurdle| margin, so the flip risk
can be bounded. Study code only. Usage: uv run python <this file>
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_live"
OUT.mkdir(parents=True, exist_ok=True)
DTB3_CACHE_STR = str(REPO / "results/research/strategy_readiness_audit_20260928/taa/DTB3_audit_cache.csv")
END_DATE_STR = "2026-09-25"

base_module = import_module("strategies.taa_df.strategy_taa_df")
utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
VARIANT_DICT = {
    "taa3x": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
}


def _config(key_str):
    mod = import_module(VARIANT_DICT[key_str])
    return replace(mod.DEFAULT_CONFIG, end_date_str=END_DATE_STR, dtb3_csv_path_str=DTB3_CACHE_STR)


def _weights(config_obj):
    return utils_module.get_standard_fallback_vix_cash_data(
        config=config_obj, base_data_loader_fn=base_module.get_defense_first_data
    )[3]


def main():
    result = {}
    real_fn = base_module.load_cash_return_ser_and_snapshot
    captured = {}

    def capture_fn(config):
        ser, snap = real_fn(config)
        captured["cash"] = ser
        return ser, snap

    for key_str in VARIANT_DICT:
        cfg = _config(key_str)
        base_module.load_cash_return_ser_and_snapshot = capture_fn
        try:
            ref_df = _weights(cfg)
        finally:
            base_module.load_cash_return_ser_and_snapshot = real_fn

        def lagged_fn(config, mode_str="bday"):
            ser, snap = real_fn(config)
            lag = ser.copy()
            lag.index = lag.index + pd.offsets.BDay(1)
            return lag, snap

        base_module.load_cash_return_ser_and_snapshot = lagged_fn
        try:
            lag_df = _weights(cfg)
        finally:
            base_module.load_cash_return_ser_and_snapshot = real_fn
        common = ref_df.index.intersection(lag_df.index)
        flip = (ref_df.loc[common] - lag_df.loc[common]).abs().max(axis=1) > 1e-12

        # Months that the calendar-day lag did NOT lag: shifted DTB3_T stays inside month T.
        cash = captured["cash"]
        cal_shift = cash.copy(); cal_shift.index = cal_shift.index + pd.Timedelta(days=1)
        unlagged_month_list = []
        for m_ts in common:
            month_obs = cash[cash.index.to_period("M") == m_ts.to_period("M")]
            if len(month_obs) == 0:
                continue
            last_obs_ts = month_obs.index[-1]
            if (last_obs_ts + pd.Timedelta(days=1)).to_period("M") == m_ts.to_period("M"):
                unlagged_month_list.append(m_ts.date().isoformat())

        # Margin: smallest |score - hurdle| among defensive assets, using the backtest hurdle.
        signal_close_df = base_module.load_signal_close_df(cfg.defensive_asset_list, cfg.start_date_str, END_DATE_STR)
        monthly_close_df = signal_close_df.resample("ME").last()
        score_df = sum(monthly_close_df.pct_change(k, fill_method=None) for k in cfg.momentum_lookback_month_vec) / float(len(cfg.momentum_lookback_month_vec))
        hurdle_ser = cash.resample("ME").last()
        hurdle_lag_ser = cash.copy(); hurdle_lag_ser.index = hurdle_lag_ser.index + pd.offsets.BDay(1)
        hurdle_lag_ser = hurdle_lag_ser.resample("ME").last()
        margin_ser = (score_df.loc[common].sub(hurdle_ser.reindex(common), axis=0)).abs().min(axis=1)
        hurdle_move_ser = (hurdle_ser.reindex(common) - hurdle_lag_ser.reindex(common)).abs()
        result[key_str] = {
            "decisions": int(len(common)),
            "flips_session_lag": int(flip.sum()),
            "flip_dates": [d.date().isoformat() for d in common[flip]],
            "months_not_lagged_by_calendar_day_check": len(unlagged_month_list),
            "months_not_lagged_examples": unlagged_month_list[:12],
            "min_score_hurdle_margin": float(margin_ser.min()),
            "min_margin_date": margin_ser.idxmin().date().isoformat(),
            "p01_score_hurdle_margin": float(margin_ser.quantile(0.01)),
            "max_monthly_hurdle_move_from_lag": float(hurdle_move_ser.max()),
            "p99_monthly_hurdle_move_from_lag": float(hurdle_move_ser.quantile(0.99)),
            "months_margin_below_max_move": int((margin_ser < hurdle_move_ser.max()).sum()),
        }
        print(key_str, json.dumps(result[key_str]), flush=True)
    (OUT / "dtb3_session_lag_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
