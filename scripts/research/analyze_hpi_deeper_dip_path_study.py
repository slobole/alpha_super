"""Accounting bridge and same-date restart comparisons; no strategy selection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
from scripts.research import analyze_mr_deeper_dip_study as accounting
from scripts.research import run_mr_deeper_dip_study as original
from scripts.research.run_hpi_deeper_dip_path_study import (
    STUDY_PATH, LAYER_TUPLE, YEAR_TUPLE, checked_spec,
)


def attribution(moo_df: pd.DataFrame, limit_df: pd.DataFrame) -> tuple:
    key_list = ["asset", "entry_date"]
    if moo_df.duplicated(key_list).any() or limit_df.duplicated(key_list).any():
        raise AssertionError("Duplicate asset/date trade matching key.")
    # *** CRITICAL*** Ex-post accounting join; never used to select or fill orders.
    bridge_df = moo_df.merge(limit_df, on=key_list, how="outer", suffixes=("_moo", "_limit"),
                             indicator=True, validate="one_to_one")
    bridge_df["group"] = bridge_df["_merge"].map(
        {"both": "shared", "left_only": "moo_only", "right_only": "limit_only"}).astype(str)
    component_list = ["entry_cash", "sale_cash", "terminal_value", "dividend_cash",
                      "commission_cash", "friction_cash"]
    for arm_str in ("moo", "limit"):
        present_ser = bridge_df["_merge"].ne("right_only" if arm_str == "moo" else "left_only")
        quantity_ser = bridge_df[f"shares_{arm_str}"]
        if not (np.isfinite(quantity_ser[present_ser]) & quantity_ser[present_ser].gt(0.)).all():
            raise AssertionError("Present trade must have finite positive shares.")
        # *** CRITICAL*** Zero only ABSENT-arm cash contributions, never missing prices.
        bridge_df[f"entry_cash_{arm_str}"] = (-quantity_ser * bridge_df[f"entry_price_{arm_str}"]).where(present_ser, 0.)
        bridge_df[f"sale_cash_{arm_str}"] = (quantity_ser * bridge_df[f"exit_equivalent_price_{arm_str}"]
            - bridge_df[f"terminal_mark_value_{arm_str}"]).where(present_ser, 0.)
        for target_str, source_str, sign_int in (("terminal_value", "terminal_mark_value", 1),
                ("dividend_cash", "dividends", 1), ("commission_cash", "native_commission", -1),
                ("friction_cash", "research_friction", -1)):
            bridge_df[f"{target_str}_{arm_str}"] = (sign_int * bridge_df[f"{source_str}_{arm_str}"]).where(present_ser, 0.)
        component_df = bridge_df[[f"{name_str}_{arm_str}" for name_str in component_list]]
        if not np.isfinite(component_df.to_numpy()).all():
            raise AssertionError("Nonfinite monetary trade component.")
        expected_ser = bridge_df[f"net_pnl_{arm_str}"].where(present_ser, 0.)
        if not np.allclose(component_df.sum(axis=1), expected_ser, rtol=0, atol=1e-7):
            raise AssertionError("Trade cash components do not sum to PnL.")
        bridge_df[f"pnl_contribution_{arm_str}"] = expected_ser
    for name_str in component_list + ["pnl_contribution"]:
        bridge_df[f"delta_{name_str}"] = bridge_df[f"{name_str}_limit"] - bridge_df[f"{name_str}_moo"]
    group_df = bridge_df.groupby("group", sort=False)[
        [f"delta_{name_str}" for name_str in component_list + ["pnl_contribution"]]].sum()
    group_df["trade_count"] = bridge_df.groupby("group").size()
    shared_df = bridge_df.loc[bridge_df["group"] == "shared"].copy()
    # Exact MOO-quantity-reference identity; order-dependent accounting, not causality:
    # dPnL=qM(eM-eL)+qM(xL-xM)+(qL-qM)(xL-eL)+dD-dC-dF.
    shared_df["entry_price_effect"] = shared_df["shares_moo"] * (shared_df["entry_price_moo"] - shared_df["entry_price_limit"])
    shared_df["exit_price_effect"] = shared_df["shares_moo"] * (shared_df["exit_equivalent_price_limit"] - shared_df["exit_equivalent_price_moo"])
    shared_df["quantity_effect"] = (shared_df["shares_limit"] - shared_df["shares_moo"]) * (
        shared_df["exit_equivalent_price_limit"] - shared_df["entry_price_limit"])
    effect_list = ["entry_price_effect", "exit_price_effect", "quantity_effect",
                   "delta_dividend_cash", "delta_commission_cash", "delta_friction_cash"]
    if not np.allclose(shared_df[effect_list].sum(axis=1), shared_df["delta_pnl_contribution"], rtol=0, atol=1e-7):
        raise AssertionError("Shared trade identity does not reconcile.")
    return bridge_df, group_df, shared_df, shared_df[effect_list].sum()


def continuous_window_returns(nav_ser: pd.Series, window_idx: pd.DatetimeIndex) -> pd.Series:
    # *** CRITICAL*** Compute backward returns BEFORE slicing. E_first/E_previous-1
    # preserves the first window day's inherited-position gain/loss.
    return accounting.nav_returns(nav_ser).loc[window_idx]


def market_metrics(return_ser: pd.Series, benchmark_ser: pd.Series) -> dict:
    if not return_ser.index.equals(benchmark_ser.index) or benchmark_ser.isna().any():
        raise AssertionError("Strategy/benchmark dates must match exactly without filled returns.")
    # End-of-month grouping of realized returns; no feature or execution sampling.
    monthly_return_ser = (1. + return_ser).groupby(return_ser.index.to_period("M")).prod() - 1.
    monthly_benchmark_ser = (1. + benchmark_ser).groupby(benchmark_ser.index.to_period("M")).prod() - 1.
    return {"daily_correlation": float(return_ser.corr(benchmark_ser)),
            "monthly_correlation": float(monthly_return_ser.corr(monthly_benchmark_ser)),
            "beta": float(return_ser.cov(benchmark_ser) / benchmark_ser.var())}


def cell_metrics(daily_df, return_ser, transaction_df, entry_df, benchmark_ser) -> dict:
    result_dict = accounting.performance(return_ser)
    result_dict.update(market_metrics(return_ser, benchmark_ser))
    result_dict.update({
        "exposure": float((daily_df["invested"] / daily_df["nav"]).mean()),
        "negative_cash_days": int((daily_df["cash"] < -1e-7).sum()),
        "minimum_cash": float(daily_df["cash"].min()),
        "minimum_cash_weight": float((daily_df["cash"] / daily_df["nav"]).min()),
        "turnover_annual": float(transaction_df["total_value"].abs().sum() / daily_df["nav"].mean() * 252 / len(daily_df)),
        "fills": int(entry_df["status"].eq("filled").sum()),
        "sessions": len(daily_df),
    })
    return result_dict


def load_daily(cell_path):
    return pd.read_csv(cell_path / "daily.csv", index_col="date", parse_dates=True, float_precision="round_trip")


def validate_receipt_grid(receipt_list: list[dict], spec_dict: dict, spec_hash_str: str) -> None:
    expected_set = {(window_dict["start_year"], layer_str, policy_str)
                    for window_dict in spec_dict["windows"]
                    for layer_str in spec_dict["layers"] for policy_str in spec_dict["policies"]}
    actual_list = [(receipt_dict["start_year"], receipt_dict["layer"], receipt_dict["policy"])
                   for receipt_dict in receipt_list]
    if len(actual_list) != len(expected_set) or set(actual_list) != expected_set:
        raise AssertionError("Receipt grid differs from frozen14x2x2 design.")
    window_map = {window_dict["start_year"]: window_dict for window_dict in spec_dict["windows"]}
    for receipt_dict in receipt_list:
        if receipt_dict["spec_sha256"] != spec_hash_str or receipt_dict["runner_sha256"] != spec_dict["runner_sha256"]:
            raise AssertionError("Receipt source/spec lineage mismatch.")
        window_dict = window_map[receipt_dict["start_year"]]
        if any(receipt_dict[name_str] != window_dict[name_str] for name_str in ("start", "end", "sessions")):
            raise AssertionError("Receipt calendar differs from frozen window.")


def analyze() -> None:
    spec_dict = checked_spec()
    cache_dict = json.loads((STUDY_PATH / "data/cache_manifest.json").read_text())
    native_dict = json.loads((original.STUDY_PATH / "runs/hpi235/native_parity.json").read_text())
    if cache_dict["spec_sha256"] != original.sha256_file(STUDY_PATH / "research_spec_frozen.json") or cache_dict["signal_sha256"] != native_dict["source_signal_sha256"]:
        raise AssertionError("Cached signal lineage differs from frozen specification/original signals.")
    if cache_dict["pickle_sha256"] != original.sha256_file(STUDY_PATH / "data/signals.pkl"):
        raise AssertionError("Cached signal bytes changed.")
    receipt_path_list = sorted((STUDY_PATH / "runs").glob("*/*/complete.json"))
    if len(receipt_path_list) != 56:
        raise AssertionError(f"Expected all56completed cells, found {len(receipt_path_list)}.")
    receipt_list = [json.loads(receipt_path.read_text()) for receipt_path in receipt_path_list]
    validate_receipt_grid(receipt_list, spec_dict, original.sha256_file(STUDY_PATH / "research_spec_frozen.json"))
    for receipt_path in receipt_path_list:
        receipt_dict = json.loads(receipt_path.read_text())
        if receipt_path.parent.parent.name != str(receipt_dict["start_year"]) or receipt_path.parent.name != f"{receipt_dict['layer']}_{receipt_dict['policy']}":
            raise AssertionError("Receipt location does not match semantic key.")
        if set(receipt_dict["files_sha256"]) != {f"{name_str}.csv" for name_str in
                ("daily", "entries", "transactions", "dividends", "friction", "all_in_trades")}:
            raise AssertionError("Receipt does not cover all six cell ledgers.")
        for name_str, digest_str in receipt_dict["files_sha256"].items():
            if original.sha256_file(receipt_path.parent / name_str) != digest_str:
                raise AssertionError("Cell artifact hash mismatch.")
    table_path = STUDY_PATH / "tables"
    table_path.mkdir(parents=True, exist_ok=True)
    pricing_df, universe_df, calendar_idx = original.load_inputs("hpi235")
    benchmark_close_ser = pricing_df[("$SPXTR", "Close")]
    # *** CRITICAL*** Market return uses preceding observed close, before window slice.
    benchmark_ser = benchmark_close_ser / benchmark_close_ser.shift(1) - 1.
    benchmark_ser.loc[calendar_idx].rename("benchmark_return").to_csv(table_path / "benchmark_returns.csv")
    original_map = {}
    attribution_row_list = []
    for layer_str in LAYER_TUPLE:
        trade_map = {}
        for policy_str in ("moo", "limit_0.5pct"):
            cell_path = original.STUDY_PATH / "runs/hpi235" / f"{layer_str}_{policy_str}"
            daily_df = load_daily(cell_path)
            trade_df = pd.read_csv(cell_path / "all_in_trades.csv", parse_dates=["entry_date", "exit_date"], float_precision="round_trip")
            trade_map[policy_str] = trade_df
            if abs(trade_df["net_pnl"].sum() - (daily_df["nav"].iloc[-1] - 100000.)) > 1e-5:
                raise AssertionError("Original trades do not reconcile to terminal NAV.")
            original_map[(layer_str, policy_str)] = daily_df
        bridge_df, group_df, shared_df, effect_ser = attribution(trade_map["moo"], trade_map["limit_0.5pct"])
        nav_delta_float = (original_map[(layer_str, "limit_0.5pct")]["nav"].iloc[-1]
                           - original_map[(layer_str, "moo")]["nav"].iloc[-1])
        if abs(group_df["delta_pnl_contribution"].sum() - nav_delta_float) > 1e-5:
            raise AssertionError("Accounting bridge does not reconcile to terminal NAV difference.")
        bridge_df.to_csv(table_path / f"{layer_str}_trade_bridge.csv", index=False)
        shared_df.to_csv(table_path / f"{layer_str}_shared_trade_effects.csv", index=False)
        effect_ser.rename("dollars").to_csv(table_path / f"{layer_str}_shared_effect_totals.csv")
        group_df["layer"] = layer_str
        attribution_row_list.append(group_df.reset_index())
    pd.concat(attribution_row_list, ignore_index=True).to_csv(table_path / "attribution_groups.csv", index=False)
    metric_row_list = []
    for receipt_path in receipt_path_list:
        receipt_dict = json.loads(receipt_path.read_text())
        cell_path = receipt_path.parent
        restart_df = load_daily(cell_path)
        expected_idx = calendar_idx[(calendar_idx >= receipt_dict["start"]) & (calendar_idx <= receipt_dict["end"])]
        if not restart_df.index.equals(expected_idx):
            raise AssertionError("Saved daily calendar differs from frozen window.")
        reference_df = original_map[(receipt_dict["layer"], receipt_dict["policy"])]
        benchmark_window_ser = benchmark_ser.loc[restart_df.index]
        transaction_df = pd.read_csv(cell_path / "transactions.csv", parse_dates=["bar"], float_precision="round_trip")
        entry_df = pd.read_csv(cell_path / "entries.csv", parse_dates=["date"], float_precision="round_trip")
        parent_cell_path = original.STUDY_PATH / "runs/hpi235" / f"{receipt_dict['layer']}_{receipt_dict['policy']}"
        for mode_str in ("restart", "continuous"):
            if mode_str == "restart":
                daily_df = restart_df
                return_ser = accounting.nav_returns(daily_df["nav"])
                selected_transaction_df, selected_entry_df = transaction_df, entry_df
            else:
                daily_df = reference_df.loc[restart_df.index]
                return_ser = continuous_window_returns(reference_df["nav"], restart_df.index)
                parent_transaction_df = pd.read_csv(parent_cell_path / "transactions.csv", parse_dates=["bar"], float_precision="round_trip")
                parent_entry_df = pd.read_csv(parent_cell_path / "entries.csv", parse_dates=["date"], float_precision="round_trip")
                selected_transaction_df = parent_transaction_df.loc[parent_transaction_df["bar"].isin(restart_df.index)]
                selected_entry_df = parent_entry_df.loc[parent_entry_df["date"].isin(restart_df.index)]
            metric_row_list.append({"start_year": receipt_dict["start_year"], "layer": receipt_dict["layer"],
                "policy": receipt_dict["policy"], "mode": mode_str, "start": receipt_dict["start"],
                "end": receipt_dict["end"], **cell_metrics(daily_df, return_ser, selected_transaction_df,
                                                           selected_entry_df, benchmark_window_ser)})
    metric_df = pd.DataFrame(metric_row_list)
    metric_df.to_csv(table_path / "window_metrics.csv", index=False)
    index_list = ["start_year", "layer", "mode"]
    moo_df = metric_df[metric_df["policy"] == "moo"].set_index(index_list)
    limit_df = metric_df[metric_df["policy"] == "limit_0.5pct"].set_index(index_list)
    delta_df = limit_df.select_dtypes("number") - moo_df.select_dtypes("number")
    delta_df.to_csv(table_path / "window_differences.csv")
    # Difference of paired policy gaps over identical dates. This includes
    # initial holdings, accumulated capital, rounding and minimum-fee effects.
    sensitivity_df = (delta_df.xs("restart", level="mode")[["cagr", "max_drawdown", "exposure"]]
                      - delta_df.xs("continuous", level="mode")[["cagr", "max_drawdown", "exposure"]])
    sensitivity_df["cagr_sign_changed"] = (
        np.sign(delta_df.xs("restart", level="mode")["cagr"])
        != np.sign(delta_df.xs("continuous", level="mode")["cagr"]))
    if not np.allclose(sensitivity_df.loc[2010, ["cagr", "max_drawdown", "exposure"]], 0., rtol=0, atol=1e-10):
        raise AssertionError("2010 restart/reference metric parity failed.")
    sensitivity_df.to_csv(table_path / "start_state_sensitivity.csv")
    delta_df.reset_index().groupby(["layer", "mode"])[["cagr", "max_drawdown", "exposure"]].agg(
        ["median", "min", "max", "mean"]).to_csv(table_path / "descriptive_ranges.csv")
    full_row_list = []
    for (layer_str, policy_str), daily_df in original_map.items():
        return_ser = accounting.nav_returns(daily_df["nav"])
        full_row_list.append({"layer": layer_str, "policy": policy_str,
            **accounting.performance(return_ser), **market_metrics(return_ser, benchmark_ser.loc[daily_df.index]),
            "exposure": float((daily_df["invested"] / daily_df["nav"]).mean())})
        # *** CRITICAL*** Trailing126 observed sessions; diagnostic only, never signal input.
        rolling_ser = return_ser.rolling(126, min_periods=126).corr(benchmark_ser.loc[daily_df.index])
        rolling_ser.to_csv(table_path / f"{layer_str}_{policy_str}_rolling_correlation.csv")
    full_row_list.append({"layer": "benchmark", "policy": "$SPXTR", **accounting.performance(benchmark_ser.loc[calendar_idx]),
                          "daily_correlation": 1., "monthly_correlation": 1., "beta": 1., "exposure": 1.})
    pd.DataFrame(full_row_list).to_csv(table_path / "full_period_metrics.csv", index=False)
    original.write_json(STUDY_PATH / "analysis_complete.json", {
        "cells": len(receipt_path_list), "windows": 14, "descriptive_only": True,
        "analyzer_sha256": original.sha256_file(Path(__file__)),
        "spec_sha256": original.sha256_file(STUDY_PATH / "research_spec_frozen.json")})
    print("PASS all56 cells and accounting attribution analyzed", flush=True)


if __name__ == "__main__":
    analyze()
