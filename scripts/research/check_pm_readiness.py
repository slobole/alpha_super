"""The expensive half of the promotion gate: does the engine contract hold?

``tests/test_strategy_registry.py`` checks what can be read from source in
milliseconds — a run_variant with the right parameters, a class that exists, a
tier that is not self-contradictory. It cannot check whether those parameters
are *honoured*, and a parameter accepted and then ignored is the failure mode a
signature check is blind to.

So this script runs the strategy and asks three questions no static check can:

  capital     Run at C and at 2C. A strategy that honours capital_base_float
              ends near twice the equity; one that accepts the argument and
              ignores it ends at exactly the same number. That silent case
              would let a book allocate a sleeve the strategy never sized to.

  benchmark   Compare the benchmark series the run stored against the genuine
              total-return index for the same symbol. Norgate's TOTALRETURN
              adjustment does nothing on an index symbol, so a strategy can
              declare total return while having loaded the price index — worth
              ~1.8pp/yr of understated benchmark, which flatters every alpha
              computed against it.

  determinism Two identical runs must agree exactly. Opt-in, since it doubles
              the cost and is the least likely of the three to fail.

Usage:
    uv run python scripts/research/check_pm_readiness.py strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit
    uv run python scripts/research/check_pm_readiness.py --all
    uv run python scripts/research/check_pm_readiness.py --all --check-determinism
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha import strategy_registry  # noqa: E402
from data.norgate_loader import (  # noqa: E402
    INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT,
    TOTALRETURN_ADJUSTMENT_STR,
    load_price_timeseries,
)


BASE_CAPITAL_FLOAT = 100_000.0
# Doubling capital must roughly double terminal equity. The band is wide
# because share rounding and fixed commissions genuinely bend the ratio; it is
# there to separate "scaled" from "ignored" (ratio 1.0), not to grade friction.
CAPITAL_SCALE_LOW_FLOAT = 1.80
CAPITAL_SCALE_HIGH_FLOAT = 2.20
# Above this the two series are telling different stories about the same
# symbol — the price/total-return gap is ~1.8pp a year.
BENCHMARK_CAGR_TOLERANCE_FLOAT = 0.005
OUTPUT_ROOT_PATH = REPO_ROOT_PATH / "results" / "research" / "pm_readiness"


@dataclass
class CheckResult:
    strategy_import_str: str
    capital_status_str: str = "skipped"
    capital_detail_str: str = ""
    benchmark_status_str: str = "skipped"
    benchmark_detail_str: str = ""
    determinism_status_str: str = "skipped"
    determinism_detail_str: str = ""

    @property
    def passed_bool(self) -> bool:
        return all(
            status_str in ("pass", "skipped")
            for status_str in (
                self.capital_status_str,
                self.benchmark_status_str,
                self.determinism_status_str,
            )
        )


def _run_strategy(strategy_import_str: str, capital_base_float: float):
    module_obj = importlib.import_module(
        strategy_registry.module_import_str(strategy_import_str)
    )
    run_variant_fn = getattr(module_obj, "run_variant", None)
    if run_variant_fn is None:
        raise AttributeError(f"{strategy_import_str} exposes no run_variant.")
    return run_variant_fn(
        show_display_bool=False,
        save_results_bool=False,
        output_dir_str="",
        backtest_start_date_str=None,
        capital_base_float=capital_base_float,
        end_date_str=None,
    )


def _final_equity_float(strategy_obj) -> float:
    return float(strategy_obj.results["total_value"].astype(float).iloc[-1])


def _cagr_float(value_ser) -> float | None:
    value_ser = value_ser.dropna()
    if len(value_ser) < 2:
        return None
    year_float = (value_ser.index[-1] - value_ser.index[0]).days / 365.25
    if year_float <= 0:
        return None
    return float((value_ser.iloc[-1] / value_ser.iloc[0]) ** (1.0 / year_float) - 1.0)


def _check_capital(result_obj: CheckResult, base_strategy_obj, double_strategy_obj) -> None:
    base_equity_float = _final_equity_float(base_strategy_obj)
    double_equity_float = _final_equity_float(double_strategy_obj)
    if base_equity_float <= 0:
        result_obj.capital_status_str = "fail"
        result_obj.capital_detail_str = "base run ended at non-positive equity"
        return

    scale_float = double_equity_float / base_equity_float
    result_obj.capital_detail_str = (
        f"{BASE_CAPITAL_FLOAT:,.0f} -> {base_equity_float:,.0f}; "
        f"{BASE_CAPITAL_FLOAT * 2:,.0f} -> {double_equity_float:,.0f}; "
        f"scale={scale_float:.3f}"
    )
    if abs(scale_float - 1.0) < 0.01:
        result_obj.capital_status_str = "fail"
        result_obj.capital_detail_str += (
            " — capital_base_float accepted but IGNORED: doubling it changed nothing."
        )
        return
    result_obj.capital_status_str = (
        "pass"
        if CAPITAL_SCALE_LOW_FLOAT <= scale_float <= CAPITAL_SCALE_HIGH_FLOAT
        else "fail"
    )


def _check_benchmark(result_obj: CheckResult, strategy_obj) -> None:
    benchmark_list = list(getattr(strategy_obj, "_benchmarks", []) or [])
    if not benchmark_list:
        result_obj.benchmark_status_str = "skipped"
        result_obj.benchmark_detail_str = "strategy declares no benchmark"
        return

    label_str = str(benchmark_list[0])
    declared_adjustment_str = str(
        getattr(strategy_obj, "_performance_benchmark_adjustment_str", "not_declared")
    )
    # *** CRITICAL*** A stored series is verified whatever the strategy claims
    # about it. Only auditing strategies that *declare* total return would let
    # the worse case through silently: a promoted strategy that stores a series
    # the report presents as its benchmark while declaring nothing about what it
    # is. The house rule is to declare adjustment role and provenance, so an
    # undeclared benchmark on a promoted strategy is itself the finding.
    if label_str not in strategy_obj.results.columns:
        result_obj.benchmark_status_str = "skipped"
        result_obj.benchmark_detail_str = f"{label_str} not stored in results"
        return

    stored_value_ser = strategy_obj.results[label_str].astype(float).dropna()
    stored_cagr_float = _cagr_float(stored_value_ser)
    if stored_cagr_float is None:
        result_obj.benchmark_status_str = "skipped"
        result_obj.benchmark_detail_str = f"{label_str} series too short to measure"
        return

    truth_symbol_str = INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT.get(label_str, label_str)
    truth_price_df = load_price_timeseries(
        truth_symbol_str,
        adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        start_date_str=str(stored_value_ser.index[0].date()),
        end_date_str=str(stored_value_ser.index[-1].date()),
    )
    truth_cagr_float = _cagr_float(truth_price_df["Close"].astype(float))
    if truth_cagr_float is None:
        result_obj.benchmark_status_str = "skipped"
        result_obj.benchmark_detail_str = f"could not load {truth_symbol_str}"
        return

    gap_float = truth_cagr_float - stored_cagr_float
    matches_total_return_bool = abs(gap_float) <= BENCHMARK_CAGR_TOLERANCE_FLOAT
    result_obj.benchmark_detail_str = (
        f"{label_str} declared '{declared_adjustment_str}'; stored "
        f"{stored_cagr_float:.2%} vs {truth_symbol_str} {truth_cagr_float:.2%} "
        f"(gap {gap_float:+.2%})"
    )

    if not matches_total_return_bool:
        result_obj.benchmark_status_str = "fail"
        result_obj.benchmark_detail_str += (
            " — the stored benchmark is not total return"
            + (
                ", yet the strategy declares it is."
                if declared_adjustment_str == TOTALRETURN_ADJUSTMENT_STR
                else "; the report presents it as the benchmark regardless."
            )
        )
        return

    if declared_adjustment_str != TOTALRETURN_ADJUSTMENT_STR:
        result_obj.benchmark_status_str = "fail"
        result_obj.benchmark_detail_str += (
            " — the series is total return but the strategy declares "
            f"'{declared_adjustment_str}'. Promoted strategies must declare "
            "benchmark provenance."
        )
        return

    result_obj.benchmark_status_str = "pass"


def _check_determinism(result_obj: CheckResult, strategy_import_str: str, base_strategy_obj) -> None:
    repeat_strategy_obj = _run_strategy(strategy_import_str, BASE_CAPITAL_FLOAT)
    first_equity_float = _final_equity_float(base_strategy_obj)
    repeat_equity_float = _final_equity_float(repeat_strategy_obj)
    result_obj.determinism_detail_str = (
        f"{first_equity_float:,.4f} vs {repeat_equity_float:,.4f}"
    )
    result_obj.determinism_status_str = (
        "pass" if first_equity_float == repeat_equity_float else "fail"
    )


def check_strategy(strategy_import_str: str, check_determinism_bool: bool) -> CheckResult:
    result_obj = CheckResult(strategy_import_str=strategy_import_str)
    base_strategy_obj = _run_strategy(strategy_import_str, BASE_CAPITAL_FLOAT)
    double_strategy_obj = _run_strategy(strategy_import_str, BASE_CAPITAL_FLOAT * 2.0)

    _check_capital(result_obj, base_strategy_obj, double_strategy_obj)
    _check_benchmark(result_obj, base_strategy_obj)
    if check_determinism_bool:
        _check_determinism(result_obj, strategy_import_str, base_strategy_obj)
    return result_obj


def main() -> int:
    arg_parser_obj = argparse.ArgumentParser(
        prog="check_pm_readiness",
        description="Run the expensive PM_READY promotion checks for one or more strategies.",
    )
    arg_parser_obj.add_argument("strategy", nargs="?", default=None)
    arg_parser_obj.add_argument(
        "--all",
        action="store_true",
        help="Check every registered pm-ready strategy (slow: several runs each).",
    )
    arg_parser_obj.add_argument("--check-determinism", action="store_true")
    parsed_args_obj = arg_parser_obj.parse_args()

    if parsed_args_obj.all:
        target_list = list(strategy_registry.pm_ready_import_tuple())
    elif parsed_args_obj.strategy:
        target_list = [parsed_args_obj.strategy]
    else:
        arg_parser_obj.error("Pass a strategy import path or --all.")

    result_list: list[CheckResult] = []
    for strategy_import_str in target_list:
        print(f"\n=== {strategy_import_str} ===")
        try:
            result_obj = check_strategy(
                strategy_import_str, bool(parsed_args_obj.check_determinism)
            )
        except Exception as exception_obj:  # noqa: BLE001 — a failed run is a result
            result_obj = CheckResult(
                strategy_import_str=strategy_import_str,
                capital_status_str="fail",
                capital_detail_str=f"run raised: {exception_obj}",
            )
        result_list.append(result_obj)
        for check_name_str, status_str, detail_str in (
            ("capital", result_obj.capital_status_str, result_obj.capital_detail_str),
            ("benchmark", result_obj.benchmark_status_str, result_obj.benchmark_detail_str),
            ("determinism", result_obj.determinism_status_str, result_obj.determinism_detail_str),
        ):
            print(f"  {check_name_str:12s} {status_str.upper():8s} {detail_str}")

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = OUTPUT_ROOT_PATH / timestamp_str
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "pm_readiness.json").write_text(
        json.dumps([asdict(result_obj) for result_obj in result_list], indent=2),
        encoding="utf-8",
    )

    failed_list = [result_obj for result_obj in result_list if not result_obj.passed_bool]
    print(f"\n{len(result_list) - len(failed_list)}/{len(result_list)} passed")
    print(f"Saved to: {output_path}")
    for result_obj in failed_list:
        print(f"  FAILED: {result_obj.strategy_import_str}")
    return 1 if failed_list else 0


if __name__ == "__main__":
    raise SystemExit(main())
