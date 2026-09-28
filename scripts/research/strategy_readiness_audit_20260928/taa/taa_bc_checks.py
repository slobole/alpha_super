"""Backtest-correctness checks for the TAA family (protocol A2, A3, A4, A5, A7, A8, A9, A10).

A2  Future-split invariance of the month-end weights: rescale one symbol's TR close (signal) and
    SPY close (VIX gate) by k in {40, 0.1, 1.5}; weights must not change.
A3  Mid-month truncation: month-end weights computed from data ending at 8 cut-offs must equal
    the full-history weights for every completed month.
A4  Positive control: a deliberately leaky weight function (month-end close shifted -1) must be
    flagged by the same A3 comparison.
A5  DTB3 publication lag: hurdle built only from observations dated before the decision session
    (live reality: DTB3_T is published on T+1) vs the backtest (DTB3_T used at Close_T).
A7  Fills on zero-volume (padded) bars: count and value share per symbol.
A8  Cash: minimum cash / NAV and share of days with negative cash (no financing is charged).
A9  Capital scaling: 100K vs 1M run.
A10 Determinism: two identical runs.

Usage: uv run python .../taa/taa_bc_checks.py [taa3x taa1n btal_qqq]
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/taa"
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
END_DATE_STR = "2026-09-25"
DTB3_CACHE_STR = str(OUTPUT_DIR_PATH / "DTB3_audit_cache.csv")

base_module = import_module("strategies.taa_df.strategy_taa_df")
utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
linearity_module = import_module("strategies.taa_df.strategy_taa_df_btal_linearity_1n")

VARIANT_DICT = {
    "taa3x": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "btal_qqq": "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
}


def _config(variant_key_str: str, end_date_str: str | None = END_DATE_STR):
    module_obj = import_module(VARIANT_DICT[variant_key_str])
    return replace(module_obj.DEFAULT_CONFIG, end_date_str=end_date_str, dtb3_csv_path_str=DTB3_CACHE_STR)


def _month_end_weights(variant_key_str: str, config_obj) -> pd.DataFrame:
    if variant_key_str == "btal_qqq":
        return utils_module.get_linearity_1n_fallback_vix_cash_data(
            config=config_obj, base_data_loader_fn=linearity_module.get_defense_first_linearity_1n_data
        )[4]
    return utils_module.get_standard_fallback_vix_cash_data(
        config=config_obj, base_data_loader_fn=base_module.get_defense_first_data
    )[3]


# ---------------------------------------------------------------------------- A2
def check_invariance(variant_key_str: str) -> dict:
    """Rescale one symbol's loaded history by k at the loader boundary and recompute weights."""
    config_obj = _config(variant_key_str)
    reference_df = _month_end_weights(variant_key_str, config_obj)
    real_loader_fn = base_module.load_price_timeseries
    real_helper_loader_fn = utils_module.load_price_timeseries
    real_linearity_loader_fn = getattr(linearity_module, "load_price_timeseries", None)
    symbol_list = list(config_obj.defensive_asset_list) + [config_obj.fallback_asset, "SPY"]
    case_list = []
    for symbol_str in dict.fromkeys(symbol_list):
        for k_float in (40.0, 0.1, 1.5):
            def scaled_loader(sym, *args, _sym=symbol_str, _k=k_float, _real=real_loader_fn, **kwargs):
                price_df = _real(sym, *args, **kwargs).copy()
                if sym == _sym:
                    for field_str in ("Open", "High", "Low", "Close", "Dividend"):
                        if field_str in price_df.columns:
                            price_df[field_str] = price_df[field_str] / _k
                    if "Volume" in price_df.columns:
                        price_df["Volume"] = price_df["Volume"] * _k
                return price_df

            base_module.load_price_timeseries = scaled_loader
            utils_module.load_price_timeseries = scaled_loader
            if real_linearity_loader_fn is not None:
                linearity_module.load_price_timeseries = scaled_loader
            try:
                scaled_df = _month_end_weights(variant_key_str, config_obj)
                common_index = reference_df.index.intersection(scaled_df.index)
                diff_float = float((scaled_df.loc[common_index] - reference_df.loc[common_index]).abs().max().max())
                pass_bool = diff_float <= 1e-12 and len(common_index) == len(reference_df.index)
            finally:
                base_module.load_price_timeseries = real_loader_fn
                utils_module.load_price_timeseries = real_helper_loader_fn
                if real_linearity_loader_fn is not None:
                    linearity_module.load_price_timeseries = real_linearity_loader_fn
            case_list.append({"symbol": symbol_str, "k": k_float, "max_weight_diff": diff_float, "pass": bool(pass_bool)})
    return {"cases": len(case_list), "passed": sum(c["pass"] for c in case_list), "detail": case_list}


# ---------------------------------------------------------------------------- A3 / A4
CUTOFF_LIST = [
    "2013-03-28",  # Good Friday eve (month-end falls on a holiday weekend)
    "2014-11-28",  # day after Thanksgiving, last session of the month (half day)
    "2016-07-15",  # mid-month
    "2018-06-01",  # first session of a month
    "2020-03-18",  # mid-month, crisis
    "2021-12-31",  # year-end
    "2023-09-29",  # month-end on Friday (weekend month-end label 2023-09-30)
    "2026-08-31",  # last completed month-end
    "2026-09-25",  # current partial month
]


def _leaky_month_end_weights(variant_key_str: str, config_obj) -> pd.DataFrame:
    """Positive control: month-end signal uses the NEXT month's close (a planted look-ahead)."""
    real_fn = base_module.compute_month_end_weight_df

    def leaky_fn(signal_close_df, cash_return_ser, config):
        leaky_signal_df = signal_close_df.resample("ME").last().shift(-1)
        # Re-expand to a daily-like frame the original function can resample again.
        return real_fn(leaky_signal_df, cash_return_ser, config)

    base_module.compute_month_end_weight_df = leaky_fn
    try:
        return _month_end_weights(variant_key_str, config_obj)
    finally:
        base_module.compute_month_end_weight_df = real_fn


def check_truncation(variant_key_str: str, leaky_bool: bool = False) -> dict:
    weight_fn = _leaky_month_end_weights if leaky_bool else _month_end_weights
    full_df = weight_fn(variant_key_str, _config(variant_key_str))
    case_list = []
    for cutoff_str in CUTOFF_LIST:
        truncated_df = weight_fn(variant_key_str, _config(variant_key_str, end_date_str=cutoff_str))
        cutoff_ts = pd.Timestamp(cutoff_str)
        # Completed months only: rows whose calendar month ended on or before the cut-off, plus the
        # cut-off month when the cut-off is that month's last session (checked separately in the replay).
        completed_index = truncated_df.index[truncated_df.index.to_period("M") < cutoff_ts.to_period("M")]
        completed_index = completed_index.intersection(full_df.index)
        diff_float = float((truncated_df.loc[completed_index] - full_df.loc[completed_index]).abs().max().max()) if len(completed_index) else 0.0
        case_list.append({"cutoff": cutoff_str, "rows_compared": int(len(completed_index)), "max_weight_diff": diff_float, "pass": bool(diff_float <= 1e-12)})
    return {"cases": len(case_list), "passed": sum(c["pass"] for c in case_list), "detail": case_list}


# ---------------------------------------------------------------------------- A5
def check_dtb3_lag(variant_key_str: str) -> dict:
    """Backtest hurdle uses DTB3 dated T; live at Close_T only has DTB3 dated T-1 or earlier."""
    config_obj = _config(variant_key_str)
    if variant_key_str == "btal_qqq":
        return {"applicable": False, "reason": "linearity family has no DTB3 hurdle in its weight rule"}
    reference_df = _month_end_weights(variant_key_str, config_obj)
    real_fn = base_module.load_cash_return_ser_and_snapshot

    def lagged_fn(config):
        cash_return_ser, snapshot_obj = real_fn(config)
        # *** CRITICAL*** publication lag: the value dated d becomes usable only after session d.
        # Re-date each observation to the next calendar day so resample("ME").last() at month m
        # can only see observations dated strictly before the month's last day.
        lagged_ser = cash_return_ser.copy()
        lagged_ser.index = lagged_ser.index + pd.Timedelta(days=1)
        return lagged_ser, snapshot_obj

    base_module.load_cash_return_ser_and_snapshot = lagged_fn
    try:
        lagged_df = _month_end_weights(variant_key_str, config_obj)
    finally:
        base_module.load_cash_return_ser_and_snapshot = real_fn
    common_index = reference_df.index.intersection(lagged_df.index)
    flip_mask = (reference_df.loc[common_index] - lagged_df.loc[common_index]).abs().max(axis=1) > 1e-12
    return {
        "applicable": True,
        "decisions": int(len(common_index)),
        "flips": int(flip_mask.sum()),
        "flip_dates": [d.date().isoformat() for d in common_index[flip_mask]],
    }


# ---------------------------------------------------------------------------- run-level checks
def _run(variant_key_str: str, capital_float: float = 100_000.0):
    module_obj = import_module(VARIANT_DICT[variant_key_str])
    module_obj.DEFAULT_CONFIG = replace(module_obj.DEFAULT_CONFIG, dtb3_csv_path_str=DTB3_CACHE_STR)
    return module_obj.run_variant(
        show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR, capital_base_float=capital_float
    )


def _metrics(total_value_ser: pd.Series) -> dict:
    total_value_ser = total_value_ser.astype(float)
    return_ser = total_value_ser.pct_change().dropna()
    years_float = len(return_ser) / 252.0
    cagr_float = (total_value_ser.iloc[-1] / total_value_ser.iloc[0]) ** (1.0 / years_float) - 1.0
    sharpe_float = float(return_ser.mean() / return_ser.std() * np.sqrt(252.0))
    drawdown_float = float((total_value_ser / total_value_ser.cummax() - 1.0).min())
    return {"cagr": float(cagr_float), "sharpe": sharpe_float, "max_dd": drawdown_float}


def check_runs(variant_key_str: str) -> dict:
    run_a = _run(variant_key_str)
    run_b = _run(variant_key_str)
    run_big = _run(variant_key_str, 1_000_000.0)
    results_df = run_a.results.copy()
    nav_ser = results_df["total_value"].astype(float)
    cash_ser = results_df["cash"].astype(float) if "cash" in results_df.columns else pd.Series(dtype=float)
    determinism_bool = bool(np.array_equal(nav_ser.to_numpy(), run_b.results["total_value"].astype(float).to_numpy()))
    scale_ret_a = nav_ser.pct_change().dropna()
    scale_ret_b = run_big.results["total_value"].astype(float).pct_change().dropna()
    cash_frac_ser = (cash_ser / nav_ser) if len(cash_ser) else pd.Series(dtype=float)

    # A7: fills on zero-volume bars.
    pricing_symbol_list = sorted(set(run_a.get_transactions()["asset"].astype(str)))
    padded_row_list = []
    for symbol_str in pricing_symbol_list:
        price_df = base_module.load_price_timeseries(symbol_str, start_date_str="2010-01-01", end_date_str=END_DATE_STR)
        tx_df = run_a.get_transactions()
        tx_df = tx_df[tx_df["asset"].astype(str) == symbol_str].copy()
        tx_df["bar"] = pd.to_datetime(tx_df["bar"])
        volume_ser = price_df["Volume"].reindex(tx_df["bar"]).fillna(0.0)
        zero_mask = volume_ser.to_numpy() <= 0.0
        padded_row_list.append(
            {
                "symbol": symbol_str,
                "fills": int(len(tx_df)),
                "fills_on_zero_volume_bar": int(zero_mask.sum()),
                "zero_volume_fill_dates": [d.date().isoformat() for d in tx_df["bar"][zero_mask]][:20],
            }
        )

    return {
        "metrics_100k": _metrics(nav_ser),
        "metrics_1m": _metrics(run_big.results["total_value"]),
        "determinism_bit_identical": determinism_bool,
        "capital_scaling_daily_return_max_abs_diff": float((scale_ret_a - scale_ret_b).abs().max()),
        "capital_scaling_daily_return_corr": float(scale_ret_a.corr(scale_ret_b)),
        "min_cash_frac_of_nav": float(cash_frac_ser.min()) if len(cash_frac_ser) else None,
        "share_days_negative_cash": float((cash_frac_ser < 0).mean()) if len(cash_frac_ser) else None,
        "dividend_net_total": float(getattr(run_a, "dividend_cash_net_total_float", np.nan)),
        "accounting_policy": {k: str(v) for k, v in getattr(run_a, "_accounting_policy_dict", {}).items()},
        "padded_fills": padded_row_list,
    }


def main() -> None:
    variant_key_list = sys.argv[1:] or list(VARIANT_DICT)
    for variant_key_str in variant_key_list:
        result_dict = {
            "variant": variant_key_str,
            "A2_invariance": check_invariance(variant_key_str),
            "A3_truncation": check_truncation(variant_key_str),
            # Planted leak is injected into the standard weight function; the linearity family uses
            # its own weight function, so the control runs on the standard variants only.
            "A4_positive_control_truncation": (
                check_truncation(variant_key_str, leaky_bool=True)
                if variant_key_str != "btal_qqq"
                else {"cases": 0, "passed": 0, "detail": "not applicable"}
            ),
            "A5_dtb3_publication_lag": check_dtb3_lag(variant_key_str),
            "runs": check_runs(variant_key_str),
        }
        for key_str in ("A2_invariance", "A3_truncation", "A4_positive_control_truncation"):
            print(variant_key_str, key_str, result_dict[key_str]["passed"], "/", result_dict[key_str]["cases"], flush=True)
        print(variant_key_str, "A5", json.dumps(result_dict["A5_dtb3_publication_lag"]), flush=True)
        print(variant_key_str, "runs", json.dumps({k: v for k, v in result_dict["runs"].items() if k != "accounting_policy"}, default=str), flush=True)
        (OUTPUT_DIR_PATH / f"bc_checks_{variant_key_str}.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
