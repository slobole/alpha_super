"""Stateful, unlevered CORE5 cost/capital replay on one frozen price panel."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import DIVIDEND_LEDGER_COLUMN_TUPLE
from scripts.research.portfolio_family_20260923.benchmarks import (
    EXPECTED_INPUT_SHA256_STR, INPUT_PATH, ROOT_PATH, sha256_str, write_json,
)
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core_module

OUTPUT_PATH = ROOT_PATH / "results/research/portfolio_family_20260923/native_replay"
START_DATE_STR = "2012-10-01"
ANCHOR_DATE_STR = "2012-09-28"
END_DATE_STR = "2026-07-31"
CASE_LIST = [
    {"case_id_str": f"core5_common_{capital_int}", "capital_float": float(capital_int),
     "funding_rate_float": 0.05, "borrow_rate_float": 0.01, "slippage_float": 0.00025}
    for capital_int in (1_000_000, 750_000, 500_000, 250_000)
] + [
    {"case_id_str": "core5_native_zero_1000000", "capital_float": 1_000_000.0,
     "funding_rate_float": 0.0, "borrow_rate_float": 0.01, "slippage_float": 0.00025},
    {"case_id_str": "core5_conservative_1000000", "capital_float": 1_000_000.0,
     "funding_rate_float": 0.08, "borrow_rate_float": 0.05, "slippage_float": 0.00125},
]
SOURCE_PATH_TUPLE = (
    "scripts/research/portfolio_family_20260923/core_replay.py",
    "tests/test_portfolio_family_core_replay.py",
    "scripts/research/portfolio_family_20260923/benchmarks.py",
    "strategies/taa_beyond_6040/strategy_taa_adaptive_macro_core5.py",
    "alpha/engine/backtest.py", "alpha/engine/backtester.py",
    "alpha/engine/strategy.py", "alpha/engine/order.py", "alpha/engine/metrics.py",
)


def source_hash_dict() -> dict:
    return {path_str: sha256_str(ROOT_PATH / path_str) for path_str in SOURCE_PATH_TUPLE}


def freeze_spec() -> dict:
    input_hash_str = sha256_str(INPUT_PATH)
    if input_hash_str != EXPECTED_INPUT_SHA256_STR:
        raise RuntimeError("Archived input hash mismatch.")
    rule_dict = {
        "start_str": START_DATE_STR, "anchor_str": ANCHOR_DATE_STR, "end_str": END_DATE_STR,
        "cases": CASE_LIST, "full_history_signal_warmup_bool": True,
        "rule": "Unchanged current CORE5 signal, fixed sleeves, DBC short, cadence and Close_T/Open_T+1 quantities; no new leverage.",
        "funding": "max(0, sum(abs(short_shares)*ceil(1.02*close))-cash_after_native_borrow)*annual_rate*known_days_to_next_session/360",
        "funding_timing": "After native fills/close mark/borrow, before metrics and future sizing; no terminal interval.",
        "rates_basis": "Fixed5% common and8% stress research assumptions, not historical or current broker rates.",
        "dividends": "25% long withholding; full gross short dividend debit.",
        "native_borrow": "Charged once by unchanged source;1% annual baseline or declared5% stress.",
        "native_config_dict": asdict(core_module.DEFAULT_CONFIG),
        "account_start": "Fresh cash account on2012-09-28; first native execution2012-10-01. No pre-start positions.",
        "parity": "Seventh run is plain current native source at1m, zero funding, same full data/calendar; compare exact NAV/cash/transactions/borrow.",
        "scope": "Six CORE5 scenarios and one parity proof; no optimizer, allocation, whole-family physical rebalance claim or live action.",
        "history": "All historical source and prior-study periods were already seen; retrospective implementation/cost sensitivity only.",
    }
    spec_path = OUTPUT_PATH / "spec.json"
    if spec_path.exists():
        spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec_dict["rule_dict"] != json.loads(json.dumps(rule_dict)) or spec_dict["input_sha256_str"] != input_hash_str:
            raise RuntimeError("Existing frozen scientific contract differs.")
        return spec_dict
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    spec_dict = {
        "frozen_at_utc_str": datetime.now(timezone.utc).isoformat(),
        "status_str": "frozen_before_replay", "input_path_str": str(INPUT_PATH),
        "input_sha256_str": input_hash_str, "source_hash_at_freeze_dict": source_hash_dict(),
        "rule_dict": rule_dict,
    }
    write_json(spec_path, spec_dict)
    return spec_dict


def funding_amount_tuple(cash_float: float, position_ser: pd.Series,
                         close_price_ser: pd.Series) -> tuple[float, float]:
    collateral_float = 0.0
    for asset_str, share_float in position_ser.items():
        if share_float < 0:
            close_float = float(close_price_ser[asset_str])
            if not np.isfinite(close_float) or close_float <= 0:
                raise ValueError("Current close required for held short collateral.")
            collateral_float += abs(float(share_float)) * float(np.ceil(1.02 * close_float))
    return max(0.0, collateral_float - cash_float), collateral_float


class FinancedCore5Strategy(core_module.AdaptiveMacroCore5Strategy):
    def __init__(self, config_obj, funding_rate_float: float):
        super().__init__(config_obj=config_obj)
        self.funding_rate_float = funding_rate_float
        self.financing_row_list: list[dict] = []
        self._accounting_policy_dict.update({
            "negative_cash_financing_policy_str": "stateful_rounded_short_collateral_minus_cash_ACT360",
            "annual_funding_scenario_rate_float": funding_rate_float,
            "funding_broker_truth_bool": False,
        })

    def apply_financing(self) -> None:
        current_position_int = int(self.borrow_calendar_idx.get_loc(self.current_bar))
        next_session_ts = (
            self.borrow_calendar_idx[current_position_int + 1]
            if current_position_int + 1 < len(self.borrow_calendar_idx) else None
        )
        day_count_int = 0 if next_session_ts is None else int((next_session_ts - self.current_bar).days)
        debt_float, collateral_float = funding_amount_tuple(
            float(self.cash), self.get_positions(), self._latest_close_price_ser,
        )
        # *** CRITICAL *** Current fills, dividend cash, close marks and native
        # borrow already happened. Debit funding now; future sizing sees it.
        # fee_T = debt_T * fixed_rate * known_days_to_next_session /360.
        fee_float = debt_float * self.funding_rate_float * day_count_int / 360.0
        cash_before_float, nav_before_float = float(self.cash), float(self.total_value)
        self.cash -= fee_float
        self.total_value -= fee_float
        self.financing_row_list.append({
            "date": self.current_bar, "next_session": next_session_ts,
            "calendar_days_int": day_count_int, "collateral_float": collateral_float,
            "debt_float": debt_float, "funding_rate_float": self.funding_rate_float,
            "funding_fee_float": fee_float, "cash_before_funding_float": cash_before_float,
            "cash_after_funding_float": float(self.cash), "nav_before_funding_float": nav_before_float,
            "nav_after_funding_float": float(self.total_value),
        })

    def process_orders(self, prices_df: pd.DataFrame) -> None:
        super().process_orders(prices_df)  # Native DBC borrow charged exactly once.
        self.apply_financing()


def run_case(price_df: pd.DataFrame, case_dict: dict, *,
             plain_native_bool: bool = False,
             start_str: str = START_DATE_STR, end_str: str = END_DATE_STR):
    config_obj = replace(
        core_module.DEFAULT_CONFIG, capital_base_float=case_dict["capital_float"],
        annual_dbc_borrow_rate_float=case_dict["borrow_rate_float"],
        slippage_float=case_dict["slippage_float"], end_date_str=end_str,
    )
    # *** CRITICAL *** Never clip pre-start prices before compute_signals:
    # cummax,126-day rank, AMA recursion and63-day volatility need full history.
    full_history_df = price_df.loc[:end_str].copy()
    calendar_idx = core_module.build_execution_calendar_idx(
        full_history_df, config_obj=config_obj, backtest_start_date_str=start_str,
    )
    strategy_obj = (core_module.AdaptiveMacroCore5Strategy(config_obj=config_obj)
                    if plain_native_bool else FinancedCore5Strategy(config_obj, case_dict["funding_rate_float"]))
    strategy_obj.configure_run_calendar(calendar_idx)
    strategy_obj.configure_dividend_cash_ledger(enabled_bool=True, withholding_rate_float=0.25)
    with redirect_stdout(io.StringIO()):
        run_daily(strategy_obj, full_history_df, calendar=calendar_idx,
                  show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    strategy_obj.results.index = pd.DatetimeIndex(strategy_obj.results.index, name="date")
    strategy_obj.realized_weight_df.index = pd.DatetimeIndex(strategy_obj.realized_weight_df.index, name="date")
    return strategy_obj


def assert_native_parity(financed_obj, native_obj) -> None:
    assert_frame_equal(financed_obj.results, native_obj.results, check_exact=True)
    # Native Order IDs are process-global counters, not account economics.
    # Compare their within-run sequence and every other transaction field.
    financed_order_ser = financed_obj._transactions['order_id'].astype(int)
    native_order_ser = native_obj._transactions['order_id'].astype(int)
    if len(financed_order_ser):
        np.testing.assert_array_equal(financed_order_ser-financed_order_ser.iloc[0],
                                      native_order_ser-native_order_ser.iloc[0])
    assert_frame_equal(financed_obj._transactions.drop(columns=['order_id']),
                       native_obj._transactions.drop(columns=['order_id']), check_exact=True)
    assert_frame_equal(financed_obj.borrow_fee_df, native_obj.borrow_fee_df, check_exact=True)
    assert_frame_equal(financed_obj.realized_weight_df, native_obj.realized_weight_df, check_exact=True)
    assert_frame_equal(financed_obj.daily_target_weights, native_obj.daily_target_weights, check_exact=True)
    if any(row_dict["funding_fee_float"] != 0 for row_dict in financed_obj.financing_row_list):
        raise ValueError("Zero-rate parity case charged funding.")


def anchored_nav_df(strategy_obj, anchor_str: str = ANCHOR_DATE_STR) -> pd.DataFrame:
    nav_df = strategy_obj.results.loc[:, ["total_value", "portfolio_value", "cash", "daily_returns"]].copy()
    anchor_ts = pd.Timestamp(anchor_str)
    if anchor_ts >= nav_df.index[0]:
        raise ValueError("Cash anchor must precede first native execution.")
    anchor_df = pd.DataFrame({
        "total_value": [strategy_obj._capital_base], "portfolio_value": [0.0],
        "cash": [strategy_obj._capital_base], "daily_returns": [0.0],
    }, index=pd.DatetimeIndex([anchor_ts], name="date"))
    nav_df = pd.concat([anchor_df, nav_df])
    if not np.allclose(nav_df["daily_returns"].iloc[1:].astype(float),
                       nav_df["total_value"].astype(float).pct_change(fill_method=None).iloc[1:],
                       rtol=1e-10, atol=1e-12):
        raise ValueError("Anchored returns lose first-fill account economics.")
    return nav_df


def export_case(strategy_obj, case_dict: dict, hashes_dict: dict) -> dict:
    output_path = OUTPUT_PATH / case_dict["case_id_str"]
    output_path.mkdir(parents=True, exist_ok=True)
    nav_df = anchored_nav_df(strategy_obj)
    financing_df = pd.DataFrame(strategy_obj.financing_row_list)
    dividend_df = pd.DataFrame(strategy_obj._dividend_ledger_row_dict_list, columns=DIVIDEND_LEDGER_COLUMN_TUPLE)
    cash_change_ser = pd.Series(0.0, index=nav_df.index)
    for row_obj in strategy_obj._transactions.itertuples(index=False):
        cash_change_ser.loc[pd.Timestamp(row_obj.bar)] -= float(row_obj.amount)*float(row_obj.price)+float(row_obj.commission)
    for row_obj in dividend_df.itertuples(index=False):
        cash_change_ser.loc[pd.Timestamp(row_obj.ex_date)] += float(row_obj.net_dividend_cash_float)
    for row_obj in strategy_obj.borrow_fee_df.itertuples(index=False):
        cash_change_ser.loc[pd.Timestamp(row_obj.accrual_start_date_ts)] -= float(row_obj.borrow_fee_float)
    for row_obj in financing_df.itertuples(index=False):
        cash_change_ser.loc[pd.Timestamp(row_obj.date)] -= float(row_obj.funding_fee_float)
    if not np.allclose(strategy_obj._capital_base + cash_change_ser.cumsum(),
                       nav_df["cash"].astype(float), rtol=1e-10, atol=1e-6):
        raise ValueError("Cash does not reconcile to fills, dividends, native borrow and funding.")
    if not np.allclose(nav_df["total_value"].astype(float),
                       nav_df["cash"].astype(float)+nav_df["portfolio_value"].astype(float), atol=1e-7):
        raise ValueError("NAV identity failed.")
    weight_df = strategy_obj.realized_weight_df.copy()
    anchor_weight_df = pd.DataFrame({"Cash": [1.0]}, index=nav_df.index[:1])
    weight_df = pd.concat([anchor_weight_df, weight_df])
    frame_dict = {
        "nav": (nav_df, True), "transactions": (strategy_obj._transactions, False),
        "realized_weights": (weight_df, True), "dividends": (dividend_df, False),
        "borrow": (strategy_obj.borrow_fee_df, False), "financing": (financing_df, False),
        "rebalance_targets": (strategy_obj.rebalance_target_weight_df, True),
    }
    file_dict = {}
    for label_str, (frame_df, index_bool) in frame_dict.items():
        file_path = output_path / f"{label_str}.csv.gz"
        frame_df.to_csv(file_path, index=index_bool, index_label="date" if index_bool else None,
                        compression={"method": "gzip", "mtime": 0})
        file_dict[label_str] = {"path_str": str(file_path), "sha256_str": sha256_str(file_path)}
    record_dict = {
        **case_dict, "native_capital_float": strategy_obj._capital_base,
        "actual_start_date_str": ANCHOR_DATE_STR, "first_execution_date_str": START_DATE_STR,
        "actual_end_date_str": END_DATE_STR, "native_row_count_int": len(nav_df),
        "accounting_policy_dict": dict(strategy_obj._accounting_policy_dict),
        "data_adjustment_policy_dict": dict(strategy_obj._data_adjustment_policy_dict),
        "current_source_hash_dict": hashes_dict, "input_sha256_str": EXPECTED_INPUT_SHA256_STR,
        "spec_sha256_str": sha256_str(OUTPUT_PATH / "spec.json"), "output_file_dict": file_dict,
        "cash_identity_checked_bool": True, "terminal_funding_fee_float": float(financing_df.iloc[-1]["funding_fee_float"]),
        "initial_anchor_note_str": "True uninvested capital immediately before fresh-account native first orders; no pre-start holdings.",
        "financing_scope_note_str": "Stateful CORE5 account only; not full-family physical rebalancing or broker margin validation.",
    }
    write_json(output_path / "source_metadata.json", record_dict)
    return record_dict


def main(freeze_only_bool: bool = False) -> None:
    freeze_spec()
    if freeze_only_bool:
        print("Frozen native_replay/spec.json before any replay.")
        return
    run_hash_dict = source_hash_dict()
    input_hash_str = sha256_str(INPUT_PATH)
    price_df = pd.read_parquet(INPUT_PATH)
    if pd.Timestamp(ANCHOR_DATE_STR) != price_df.index[price_df.index.get_loc(pd.Timestamp(START_DATE_STR)) - 1]:
        raise ValueError("Anchor is not actual preceding archived session.")
    record_list = []
    zero_obj = None
    for case_dict in CASE_LIST:
        print("Running " + case_dict["case_id_str"], flush=True)
        strategy_obj = run_case(price_df, case_dict)
        record_list.append(export_case(strategy_obj, case_dict, run_hash_dict))
        if case_dict["funding_rate_float"] == 0:
            zero_obj = strategy_obj
    print("Running unchanged native-source parity proof", flush=True)
    native_obj = run_case(price_df, CASE_LIST[4], plain_native_bool=True)
    assert_native_parity(zero_obj, native_obj)
    if sha256_str(INPUT_PATH) != input_hash_str or source_hash_dict() != run_hash_dict:
        raise RuntimeError("Input or source code changed during replay.")
    write_json(OUTPUT_PATH / "replay_manifest.json", {
        "status_str": "complete", "completed_at_utc_str": datetime.now(timezone.utc).isoformat(),
        "spec_sha256_str": sha256_str(OUTPUT_PATH / "spec.json"),
        "input_sha256_str": input_hash_str, "current_source_hash_dict": run_hash_dict,
        "case_count_int": len(CASE_LIST), "total_run_count_int": 7,
        "native_zero_funding_exact_parity_bool": True,
        "parity_fields": ["full_results", "transaction_economics_and_relative_order_sequence", "borrow", "realized_weights", "daily_target_weights"],
        "case_metadata_list": record_list,
        "no_performance_ranking_bool": True,
    })
    print("Completed six stateful cases and exact native-zero parity.", flush=True)


if __name__ == "__main__":
    argument_parser_obj = argparse.ArgumentParser()
    argument_parser_obj.add_argument("--freeze-only", action="store_true")
    main(argument_parser_obj.parse_args().freeze_only)
