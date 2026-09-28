"""Quant-lens reviewer checks for the TAA family (read-only on production code).

R-TAA-1  DTB3 publication lag, done per XNYS session (the primary audit re-dated each DTB3 observation by +1
         CALENDAR day, which leaves the last-session observation inside the month whenever the month's last
         calendar day is a weekend or holiday).
R-TAA-2  Blindness of the primary A3 truncation harness to a ONE-session look-ahead (it only compares months
         strictly before the cut-off month).
R-TAA-3  Commission units: the production ledger charges $0.005 per ADJUSTED share; TQQQ's adjusted units are
         many raw shares.
R-TAA-4  Backtest metrics with the session-lagged DTB3 hurdle.

No network: fred_loader.urlopen is patched to fail, so the copied cache in the review folder is used.
Usage: uv run python scripts/research/strategy_readiness_audit_20260928/review_quant/rq_taa_checks.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import alpha.data.fred_loader as fred_loader_module  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/review_quant"
DTB3_CACHE_STR = str(OUT_DIR_PATH / "DTB3_review_cache.csv")
END_DATE_STR = "2026-09-25"


def _no_network(*args, **kwargs):
    raise URLError("review: network disabled, use cache")


fred_loader_module.urlopen = _no_network

base_module = import_module("strategies.taa_df.strategy_taa_df")
utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
VARIANT_DICT = {
    "taa3x": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
}

import exchange_calendars  # noqa: E402

XNYS = exchange_calendars.get_calendar("XNYS", start="1990-01-02", end="2027-12-31")
SESSION_IDX = pd.DatetimeIndex(XNYS.sessions).tz_localize(None)


def _config(variant_key_str: str, end_date_str: str | None = END_DATE_STR):
    module_obj = import_module(VARIANT_DICT[variant_key_str])
    return replace(module_obj.DEFAULT_CONFIG, end_date_str=end_date_str, dtb3_csv_path_str=DTB3_CACHE_STR)


def _month_end_weights(variant_key_str: str, config_obj) -> pd.DataFrame:
    return utils_module.get_standard_fallback_vix_cash_data(
        config=config_obj, base_data_loader_fn=base_module.get_defense_first_data
    )[3]


def _redate_to_session_after(index: pd.DatetimeIndex, n_sessions_int: int) -> pd.DatetimeIndex:
    """Observation dated d becomes usable only from the n-th XNYS session strictly after d."""
    pos_arr = SESSION_IDX.searchsorted(index, side="right") + (n_sessions_int - 1)
    pos_arr = np.clip(pos_arr, 0, len(SESSION_IDX) - 1)
    return pd.DatetimeIndex(SESSION_IDX[pos_arr])


def _patched_cash_loader(mode_str: str):
    real_fn = base_module.load_cash_return_ser_and_snapshot

    def lagged_fn(config):
        cash_return_ser, snapshot_obj = real_fn(config)
        lagged_ser = cash_return_ser.copy()
        if mode_str == "calendar_plus_1d":
            lagged_ser.index = lagged_ser.index + pd.Timedelta(days=1)
        elif mode_str == "session_lag_1":
            lagged_ser.index = _redate_to_session_after(lagged_ser.index, 1)
            lagged_ser = lagged_ser.groupby(level=0).last()
        elif mode_str == "session_lag_2":
            lagged_ser.index = _redate_to_session_after(lagged_ser.index, 2)
            lagged_ser = lagged_ser.groupby(level=0).last()
        return lagged_ser, snapshot_obj

    return real_fn, lagged_fn


def dtb3_lag_flips(variant_key_str: str) -> dict:
    config_obj = _config(variant_key_str)
    reference_df = _month_end_weights(variant_key_str, config_obj)
    out = {"decisions": int(len(reference_df))}
    for mode_str in ("calendar_plus_1d", "session_lag_1", "session_lag_2"):
        real_fn, lagged_fn = _patched_cash_loader(mode_str)
        base_module.load_cash_return_ser_and_snapshot = lagged_fn
        try:
            lagged_df = _month_end_weights(variant_key_str, config_obj)
        finally:
            base_module.load_cash_return_ser_and_snapshot = real_fn
        common_index = reference_df.index.intersection(lagged_df.index)
        diff_ser = (reference_df.loc[common_index] - lagged_df.loc[common_index]).abs().max(axis=1)
        flip_index = common_index[diff_ser > 1e-12]
        flip_list = []
        for label_ts in flip_index:
            month_session_idx = SESSION_IDX[SESSION_IDX.to_period("M") == label_ts.to_period("M")]
            flip_list.append(
                {
                    "month_end_label": label_ts.date().isoformat(),
                    "decision_session": month_session_idx[-1].date().isoformat(),
                    "max_weight_change": float(diff_ser.loc[label_ts]),
                    "base": {k: round(float(v), 4) for k, v in reference_df.loc[label_ts].items() if abs(v) > 1e-12},
                    "lagged": {k: round(float(v), 4) for k, v in lagged_df.loc[label_ts].items() if abs(v) > 1e-12},
                }
            )
        out[mode_str] = {"flips": int(len(flip_index)), "detail": flip_list}
    # How many decision months have a last calendar day that is not the last session (the +1 day shift is a no-op)?
    label_idx = reference_df.index
    noop_count_int = 0
    for label_ts in label_idx:
        month_session_idx = SESSION_IDX[SESSION_IDX.to_period("M") == label_ts.to_period("M")]
        if len(month_session_idx) and (month_session_idx[-1] + pd.Timedelta(days=1)).to_period("M") == label_ts.to_period("M"):
            noop_count_int += 1
    out["months_where_calendar_plus_1d_does_not_lag_last_session"] = noop_count_int
    return out


def one_session_leak_control(variant_key_str: str) -> dict:
    """Plant a ONE-session look-ahead (daily signal close shifted -1) and run the primary A3 comparison rule."""
    real_fn = base_module.compute_month_end_weight_df

    def leaky_fn(signal_close_df, cash_return_ser, config):
        # *** CRITICAL*** deliberate positive control: reads Close_(T+1) at T.
        return real_fn(signal_close_df.shift(-1), cash_return_ser, config)

    cutoff_list = [
        "2013-03-28", "2014-11-28", "2016-07-15", "2018-06-01", "2020-03-18",
        "2021-12-31", "2023-09-29", "2026-08-31", "2026-09-25",
    ]
    base_module.compute_month_end_weight_df = leaky_fn
    try:
        full_df = _month_end_weights(variant_key_str, _config(variant_key_str))
        row_list = []
        for cutoff_str in cutoff_list:
            trunc_df = _month_end_weights(variant_key_str, _config(variant_key_str, end_date_str=cutoff_str))
            cutoff_ts = pd.Timestamp(cutoff_str)
            completed_idx = trunc_df.index[trunc_df.index.to_period("M") < cutoff_ts.to_period("M")].intersection(full_df.index)
            primary_rule_diff = float((trunc_df.loc[completed_idx] - full_df.loc[completed_idx]).abs().max().max()) if len(completed_idx) else 0.0
            own_month_idx = trunc_df.index[trunc_df.index.to_period("M") == cutoff_ts.to_period("M")].intersection(full_df.index)
            own_month_diff = float((trunc_df.loc[own_month_idx] - full_df.loc[own_month_idx]).abs().max().max()) if len(own_month_idx) else None
            row_list.append({"cutoff": cutoff_str, "primary_rule_caught": primary_rule_diff > 1e-12,
                             "cutoff_month_row_caught": (own_month_diff is not None and own_month_diff > 1e-12)})
    finally:
        base_module.compute_month_end_weight_df = real_fn
    return {
        "primary_rule_caught": int(sum(r["primary_rule_caught"] for r in row_list)),
        "cutoff_month_row_caught": int(sum(r["cutoff_month_row_caught"] for r in row_list)),
        "cases": len(row_list),
        "detail": row_list,
    }


def _metrics(total_value_ser: pd.Series) -> dict:
    total_value_ser = total_value_ser.astype(float)
    return_ser = total_value_ser.pct_change().dropna()
    years_float = len(return_ser) / 252.0
    return {
        "cagr": float((total_value_ser.iloc[-1] / total_value_ser.iloc[0]) ** (1.0 / years_float) - 1.0),
        "sharpe": float(return_ser.mean() / return_ser.std() * np.sqrt(252.0)),
        "max_dd": float((total_value_ser / total_value_ser.cummax() - 1.0).min()),
    }


def _run(variant_key_str: str):
    module_obj = import_module(VARIANT_DICT[variant_key_str])
    module_obj.DEFAULT_CONFIG = replace(module_obj.DEFAULT_CONFIG, dtb3_csv_path_str=DTB3_CACHE_STR)
    return module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR)


def run_level(variant_key_str: str) -> dict:
    base_run = _run(variant_key_str)
    base_metrics = _metrics(base_run.results["total_value"])
    real_fn, lagged_fn = _patched_cash_loader("session_lag_1")
    base_module.load_cash_return_ser_and_snapshot = lagged_fn
    try:
        lag_run = _run(variant_key_str)
    finally:
        base_module.load_cash_return_ser_and_snapshot = real_fn
    lag_metrics = _metrics(lag_run.results["total_value"])

    # R-TAA-3 commission units on the baseline run.
    tx_df = base_run.get_transactions().copy()
    tx_df["bar"] = pd.to_datetime(tx_df["bar"])
    load_fn = base_module.load_price_timeseries
    fee_row_list = []
    raw_fee_total_float = 0.0
    model_fee_total_float = 0.0
    for asset_str, asset_tx_df in tx_df.groupby("asset"):
        price_df = load_fn(str(asset_str), adjustment_str="CAPITALSPECIAL", start_date_str="2010-01-01", end_date_str=END_DATE_STR)
        k_ser = (price_df["Unadjusted Close"].astype(float) / price_df["Close"].astype(float))
        # fee-paying raw shares = adjusted shares / k at the fill bar
        k_fill = k_ser.reindex(asset_tx_df["bar"]).to_numpy()
        adj_amount = asset_tx_df["amount"].astype(float).abs().to_numpy()
        raw_amount = adj_amount / k_fill
        model_fee = np.maximum(1.0, 0.005 * adj_amount)
        raw_fee = np.maximum(1.0, 0.005 * np.floor(raw_amount + 1e-9))
        model_fee_total_float += float(np.nansum(model_fee))
        raw_fee_total_float += float(np.nansum(raw_fee))
        fee_row_list.append({"asset": str(asset_str), "fills": int(len(asset_tx_df)),
                             "k_min": float(np.nanmin(k_fill)), "k_max": float(np.nanmax(k_fill)),
                             "fee_model_adjusted_units": float(np.nansum(model_fee)),
                             "fee_raw_units": float(np.nansum(raw_fee))})
    reported_commission_float = float(tx_df["commission"].sum()) if "commission" in tx_df.columns else None
    return {
        "baseline_metrics": base_metrics,
        "session_lag_1_metrics": lag_metrics,
        "delta_cagr_pp": 100.0 * (lag_metrics["cagr"] - base_metrics["cagr"]),
        "delta_sharpe": lag_metrics["sharpe"] - base_metrics["sharpe"],
        "delta_maxdd_pp": 100.0 * (lag_metrics["max_dd"] - base_metrics["max_dd"]),
        "fee_reported_total": reported_commission_float,
        "fee_recomputed_adjusted_units_total": model_fee_total_float,
        "fee_raw_units_total": raw_fee_total_float,
        "fee_by_asset": fee_row_list,
        "transactions_columns": list(map(str, tx_df.columns)),
        "final_value": float(base_run.results["total_value"].iloc[-1]),
    }


def main() -> None:
    result_dict = {}
    for variant_key_str in VARIANT_DICT:
        result_dict[variant_key_str] = {
            "R_TAA_1_dtb3_lag": dtb3_lag_flips(variant_key_str),
            "R_TAA_2_one_session_leak_control": one_session_leak_control(variant_key_str),
            "R_TAA_3_4_run_level": run_level(variant_key_str),
        }
        print(variant_key_str, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "detail"} if isinstance(v, dict) else v
                                           for k, v in result_dict[variant_key_str].items()}, default=str)[:3000], flush=True)
    (OUT_DIR_PATH / "rq_taa_checks.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
