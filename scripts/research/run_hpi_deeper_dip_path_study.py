"""Frozen HPI restart sensitivity: 14 starts x 2 fixed policies x 2 cost layers.

Research only. Reuses the original engine, entry adapter, PIT inputs and signals.
First orders use the preceding close; every restart starts flat with $100,000.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
from scripts.research import run_mr_deeper_dip_study as original
from scripts.research.run_mr_deeper_dip_fast import freeze_frame_metadata
from scripts.research import analyze_mr_deeper_dip_study as accounting

STUDY_PATH = REPO_PATH / "results/research/hpi_deeper_dip_path_study"
LAYER_TUPLE = ("central", "stress")
DEPTH_TUPLE = (None, 0.005)
YEAR_TUPLE = tuple(range(2010, 2024))


def window_calendar(calendar_idx: pd.DatetimeIndex, year_int: int):
    # *** CRITICAL*** Predetermined calendar windows, never selected by results.
    # 2010 starts Jan 5 to preserve the original common execution calendar.
    selected_idx = calendar_idx[(calendar_idx >= f"{year_int}-01-01")
                                & (calendar_idx < f"{year_int + 3}-01-01")]
    if len(selected_idx) < 700 or selected_idx[-1].year != year_int + 2:
        raise AssertionError("Incomplete three-calendar-year window.")
    return selected_idx


def verify_dependencies() -> dict:
    manifest_path = original.STUDY_PATH / "source_code_manifest.json"
    dependency_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name_str, hash_str in dependency_dict.items():
        if original.sha256_file(REPO_PATH / name_str) != hash_str:
            raise AssertionError("Original dependency changed: " + name_str)
    return dependency_dict


def freeze_spec() -> None:
    spec_path = STUDY_PATH / "research_spec_frozen.json"
    if spec_path.exists():
        raise FileExistsError("Follow-up spec already frozen.")
    dependency_dict = verify_dependencies()
    calendar_idx = pd.DatetimeIndex(pd.read_csv(
        original.STUDY_PATH / "data/hpi235/calendar.csv")["date"])
    parent_spec_path = original.STUDY_PATH / "research_spec_frozen.json"
    parent_dict = json.loads(parent_spec_path.read_text(encoding="utf-8"))
    source_file_list = [parent_spec_path, original.STUDY_PATH / "run_manifest.json",
                        original.STUDY_PATH / "data/hpi235/input_manifest.json",
                        Path(original.__file__), Path(accounting.__file__),
                        REPO_PATH / "scripts/research/run_mr_deeper_dip_fast.py"]
    source_file_list += list((original.STUDY_PATH / "runs/hpi235").glob("*/*.csv"))
    source_file_list += list((original.STUDY_PATH / "runs/hpi235").glob("*/complete.json"))
    source_file_list += [original.STUDY_PATH / "runs/hpi235/native_parity.json"]
    window_list = []
    for year_int in YEAR_TUPLE:
        selected_idx = window_calendar(calendar_idx, year_int)
        window_list.append({"start_year": year_int, "start": str(selected_idx[0].date()),
                            "end": str(selected_idx[-1].date()), "sessions": len(selected_idx)})
    original.write_json(spec_path, {
        "study_id": "hpi_deeper_dip_path_study", "research_only": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "objective": "Explain HPI portfolio difference and test sensitivity to fresh starting states.",
        "hypothesis": "HPI limit0.5 historical benefit may depend on portfolio path and market period.",
        "authorization": "User approved proposed attribution plus 56 restarts; no intraday study yet.",
        "prior_seen": "All original24cells, HPI0.5chosenafterresults; reviewer computed saved-ledger group dollars before this freeze; restart results unseen; no untouched holdout.",
        "windows": window_list, "policies": ["moo", "limit_0.5pct"],
        "layers": list(LAYER_TUPLE), "economic_runs": 56,
        "initial_capital": 100000, "initial_positions": "empty at every window start",
        "warmup": "all original price history from1998; identical full causal signal matrix",
        "first_decision": "preceding observed price close, including before window start",
        "terminal": "last window session Close mark; no forced final liquidation or exit fees",
        "original_timing": parent_dict["timing"], "original_signal": parent_dict["signal"]["hpi235"],
        "original_portfolio": parent_dict["portfolio"], "original_costs": parent_dict["costs"],
        "data_contract": parent_dict["data"],
        "scope_note": "Only original HPI PIT data used; DV2/sector-specific inherited gaps not applicable.",
        "comparison": "Withinwindow limit-minus-MOO; compare same-date continuous return slices.",
        "continuous_basis": "Return=E_t/E_previous_session-1 computed BEFORE window slicing; 100k only at2010 start.",
        "capital_caveat": "Continuous account capital, whole-share rounding and minimum fees differ; not pure position-history attribution.",
        "attribution": "Outerjoin actual trades on(asset,entry_date); shareddelta, minusMOOonly, pluslimitonly; includes costs/dividends/terminalmarks.",
        "attribution_caveat": "Additive terminal-dollar accounting bridge, not causal CAGR or counterfactual decomposition.",
        "metrics": ["CAGR252", "volatility252", "Sharpe_rf0_all_days", "maxDD_with_initialNAV",
                    "exposure", "turnover", "negativecash", "fills", "benchmarkcorrelation", "beta"],
        "inference": "Descriptive overlapping known-history windows; no IID pvalues, no selected-window aggregation, no new pass-count gate.",
        "decision": "Assess magnitude, sign, regime concentration, start-state sensitivity, drawdown and cost survival jointly; no promotion.",
        "future_fill_check": "Only if historical evidence warrants; intraday activation/fills not part of these56 runs.",
        "source_sha256": {str(path_obj.relative_to(REPO_PATH)): original.sha256_file(path_obj)
                          for path_obj in source_file_list},
        "dependency_sha256": dependency_dict,
        "runner_sha256": original.sha256_file(Path(__file__)),
    })
    print("FROZEN", spec_path, flush=True)


def checked_spec() -> dict:
    spec_dict = json.loads((STUDY_PATH / "research_spec_frozen.json").read_text(encoding="utf-8"))
    if spec_dict["runner_sha256"] != original.sha256_file(Path(__file__)):
        raise AssertionError("Runner differs from frozen source.")
    verify_dependencies()
    for name_str, digest_str in spec_dict["source_sha256"].items():
        if original.sha256_file(REPO_PATH / name_str) != digest_str:
            raise AssertionError("Original study artifact changed: " + name_str)
    return spec_dict


def prepare_signals() -> None:
    checked_spec()
    cache_path = STUDY_PATH / "data/signals.pkl"
    if cache_path.exists():
        raise FileExistsError("Signal cache already exists.")
    pricing_df, universe_df, calendar_idx = original.load_inputs("hpi235")
    metadata_dict = pricing_df.attrs.copy()
    freeze_frame_metadata(pricing_df)
    strategy_obj = original.make_strategy("hpi235", universe_df, native_bool=True)
    signal_df = strategy_obj.compute_signals(pricing_df)
    signal_hash_str = original.frame_digest(signal_df)
    native_dict = json.loads((original.STUDY_PATH / "runs/hpi235/native_parity.json").read_text())
    if signal_hash_str != native_dict["source_signal_sha256"]:
        raise AssertionError("Full signals changed from original study.")
    prefix_list = []
    for cutoff_str in ("2015-12-31", "2020-12-31"):
        # *** CRITICAL*** Recompute all symbols using only prices <= cutoff.
        # Future-created feature columns may align only when all prior values are NaN.
        prefix_df = strategy_obj.compute_signals(pricing_df.loc[:cutoff_str])
        expected_df = signal_df.loc[:cutoff_str]
        missing_idx = expected_df.columns.difference(prefix_df.columns)
        if expected_df.loc[:, missing_idx].notna().any().any():
            raise AssertionError("Future feature column contains past data.")
        pd.testing.assert_frame_equal(prefix_df.reindex(columns=expected_df.columns),
                                      expected_df, check_freq=False, check_dtype=False)
        prefix_list.append({"cutoff": cutoff_str, "pass": True, "columns": len(expected_df.columns)})
        del prefix_df, expected_df
    signal_df.attrs = metadata_dict  # serialization of original plain metadata
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_pickle(cache_path)
    original.write_json(STUDY_PATH / "data/cache_manifest.json", {
        "signal_sha256": signal_hash_str, "pickle_sha256": original.sha256_file(cache_path),
        "prefix_audits": prefix_list, "rows": len(signal_df), "columns": len(signal_df.columns),
        "spec_sha256": original.sha256_file(STUDY_PATH / "research_spec_frozen.json")})
    print("PASS cached original signal parity and all-symbol prefix audits", flush=True)


def assert_original_prefix(cell_path: Path, layer_str: str, policy_str: str) -> None:
    reference_path = original.STUDY_PATH / "runs/hpi235" / f"{layer_str}_{policy_str}"
    daily_df = pd.read_csv(cell_path / "daily.csv", index_col="date", parse_dates=True,
                           float_precision="round_trip")
    expected_df = pd.read_csv(reference_path / "daily.csv", index_col="date", parse_dates=True,
                              float_precision="round_trip").loc[daily_df.index]
    pd.testing.assert_frame_equal(daily_df, expected_df, check_freq=False, atol=1e-8, rtol=0)
    for name_str, date_str in (("transactions.csv", "bar"), ("entries.csv", "date"),
                               ("friction.csv", "date"), ("dividends.csv", "ex_date")):
        actual_df = pd.read_csv(cell_path / name_str, float_precision="round_trip")
        reference_df = pd.read_csv(reference_path / name_str, float_precision="round_trip")
        reference_df = reference_df.loc[pd.to_datetime(reference_df[date_str]) <= daily_df.index[-1]]
        pd.testing.assert_frame_equal(actual_df.drop(columns="order_id", errors="ignore").reset_index(drop=True),
                                      reference_df.drop(columns="order_id", errors="ignore").reset_index(drop=True),
                                      check_dtype=False, atol=1e-8, rtol=0)


def run_years(year_list: list[int]) -> None:
    spec_dict = checked_spec()
    pricing_df, universe_df, calendar_idx = original.load_inputs("hpi235")
    freeze_frame_metadata(pricing_df)
    cache_path = STUDY_PATH / "data/signals.pkl"
    cache_dict = json.loads((STUDY_PATH / "data/cache_manifest.json").read_text())
    if original.sha256_file(cache_path) != cache_dict["pickle_sha256"]:
        raise AssertionError("Signal cache bytes changed.")
    signal_df = freeze_frame_metadata(pd.read_pickle(cache_path))
    for year_int in year_list:
        if year_int not in YEAR_TUPLE:
            raise ValueError("Year outside frozen design.")
        selected_idx = window_calendar(calendar_idx, year_int)
        # *** CRITICAL*** Preserve all warmup, remove every post-window observation.
        window_pricing_df = pricing_df.loc[:selected_idx[-1]]
        window_signal_df = signal_df.loc[:selected_idx[-1]]
        for layer_str in LAYER_TUPLE:
            for depth_float in DEPTH_TUPLE:
                policy_str = original.policy_name(depth_float)
                cell_path = STUDY_PATH / "runs" / str(year_int) / f"{layer_str}_{policy_str}"
                if (cell_path / "complete.json").exists():
                    receipt_dict = json.loads((cell_path / "complete.json").read_text())
                    for file_str, digest_str in receipt_dict["files_sha256"].items():
                        if original.sha256_file(cell_path / file_str) != digest_str:
                            raise AssertionError("Completed cell bytes changed.")
                    if receipt_dict["spec_sha256"] != original.sha256_file(STUDY_PATH / "research_spec_frozen.json"):
                        raise AssertionError("Completed cell belongs to another spec.")
                    print("VERIFIED existing", year_int, layer_str, policy_str, flush=True)
                    continue
                started_float = time.perf_counter()
                strategy_obj = original.make_strategy("hpi235", universe_df)
                strategy_obj.configure_research(depth_float, layer_str == "stress", window_signal_df)
                if strategy_obj.cash != 100000. or len(strategy_obj.get_transactions()):
                    raise AssertionError("Restart must be flat100k.")
                print("RUN", year_int, layer_str, policy_str, flush=True)
                original.run_daily(strategy_obj, window_pricing_df, selected_idx, show_progress=False,
                                   show_signal_progress_bool=False)
                daily_df = pd.DataFrame(strategy_obj.daily_row_list).set_index("date")
                daily_df.index = pd.DatetimeIndex(daily_df.index)
                daily_df.index.name = "date"
                if not daily_df.index.equals(selected_idx):
                    raise AssertionError("Missing first/last execution session.")
                entry_df = pd.DataFrame(strategy_obj.entry_row_list)
                entry_df["date"] = pd.to_datetime(entry_df["date"])
                entry_df["decision_date"] = pd.to_datetime(entry_df["decision_date"])
                # *** CRITICAL*** Every recorded decision uses the preceding observed price session.
                expected_idx = pricing_df.index[pricing_df.index.get_indexer(entry_df["date"]) - 1]
                if not np.array_equal(entry_df["decision_date"].to_numpy(), expected_idx.to_numpy()):
                    raise AssertionError("Decision/execution boundary changed.")
                transaction_df = strategy_obj.get_transactions()
                dividend_df = strategy_obj.get_dividend_ledger()
                friction_df = pd.DataFrame(strategy_obj.friction_row_list)
                friction_df["date"] = pd.to_datetime(friction_df["date"])
                # *** CRITICAL*** Terminal price view ends at THIS window, never2026.
                terminal_pricing_df = window_pricing_df
                trade_df = accounting.make_trade_table(transaction_df, dividend_df, friction_df,
                                                       terminal_pricing_df)
                audit_dict = {
                    "cash_error": accounting.reconcile_cash(daily_df, transaction_df, dividend_df, friction_df),
                    "mark_error": accounting.reconcile_daily_marks(daily_df, transaction_df, pricing_df),
                    "dividend_events": accounting.verify_dividend_entitlements(selected_idx, pricing_df,
                                                                                transaction_df, dividend_df),
                    "trade_pnl_error": abs(float(trade_df["net_pnl"].sum()) - (strategy_obj.total_value - 100000.)),
                }
                if audit_dict["trade_pnl_error"] > 1e-5:
                    raise AssertionError("Trade PnL does not reconcile to window terminal NAV.")
                accounting.verify_execution_contract("hpi235", layer_str, depth_float, entry_df,
                                                       transaction_df, friction_df, pricing_df)
                cell_path.mkdir(parents=True, exist_ok=True)
                for name_str, frame_df in (("daily", daily_df), ("entries", entry_df),
                        ("transactions", transaction_df), ("dividends", dividend_df),
                        ("friction", friction_df), ("all_in_trades", trade_df)):
                    frame_df.to_csv(cell_path / f"{name_str}.csv", index=name_str == "daily")
                if year_int == 2010:
                    assert_original_prefix(cell_path, layer_str, policy_str)
                    audit_dict["original2010_prefix_parity"] = True
                original.write_json(cell_path / "complete.json", {
                    "start_year": year_int, "layer": layer_str, "policy": policy_str,
                    "start": str(selected_idx[0].date()), "end": str(selected_idx[-1].date()),
                    "sessions": len(selected_idx), "terminal_nav": float(strategy_obj.total_value),
                    "audit": audit_dict, "elapsed_seconds": time.perf_counter() - started_float,
                    "spec_sha256": original.sha256_file(STUDY_PATH / "research_spec_frozen.json"),
                    "runner_sha256": spec_dict["runner_sha256"],
                    "files_sha256": {path_obj.name: original.sha256_file(path_obj)
                                     for path_obj in cell_path.glob("*.csv")}})
                print("PASS", year_int, layer_str, policy_str,
                      round(time.perf_counter() - started_float, 1), "seconds", flush=True)
                del strategy_obj, terminal_pricing_df
                gc.collect()


def main() -> None:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("stage", choices=("freeze", "prepare", "run"))
    parser_obj.add_argument("--years", nargs="+", type=int)
    args_obj = parser_obj.parse_args()
    if args_obj.stage == "freeze":
        freeze_spec()
    elif args_obj.stage == "prepare":
        prepare_signals()
    else:
        run_years(args_obj.years or list(YEAR_TUPLE))


if __name__ == "__main__":
    main()
