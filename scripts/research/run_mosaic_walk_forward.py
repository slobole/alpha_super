"""
Walk-forward analysis for the MOSAIC correlation-penalty parameter (lambda).

Question answered
-----------------
The research cycle chose lambda = 0.75 by looking at the full 2000-2026
sample. A live trader in, say, 2010 could not have seen that table. Would a
disciplined operator — re-selecting lambda every two years using only the data
available at that moment — have (a) kept choosing a positive lambda, and
(b) earned the improvement out of sample?

Method
------
Each lambda variant is one complete, independently compounded backtest over
the full period (equity curves from the lambda-grid sweep). The walk-forward
then works purely on stored monthly returns:

    1. Anchored training window: 2000-01 .. selection date
       (first selection after min_training_months).
    2. Pick lambda* = argmax of the selection metric (Sharpe by default,
       MAR as a robustness alternative) over the training window ONLY.
    3. Hold that variant's returns for the next oos_step_months months.
    4. Roll forward and repeat. Stitch all OOS segments into one series.

*** CRITICAL*** The stitched series uses only returns from periods strictly
after each selection date, so no selection ever sees its own evaluation data.

Approximation note: stitching monthly returns across variants ignores the
share-rounding path dependence of switching lambda mid-flight in a real
account. For a diagnostic of parameter selection validity this is the standard
and appropriate simplification; it does not change signal or execution
semantics of any underlying run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))


MONTHS_PER_YEAR_INT = 12


def _metric_float(return_ser: pd.Series, metric_str: str) -> float:
    clean_ser = return_ser.dropna()
    if len(clean_ser) < 12 or clean_ser.std() == 0.0:
        return float("-inf")
    if metric_str == "sharpe":
        return float(clean_ser.mean() / clean_ser.std() * np.sqrt(MONTHS_PER_YEAR_INT))
    if metric_str == "mar":
        equity_ser = (1.0 + clean_ser).cumprod()
        drawdown_float = float((equity_ser / equity_ser.cummax() - 1.0).min())
        year_count_float = len(clean_ser) / MONTHS_PER_YEAR_INT
        cagr_float = float(equity_ser.iloc[-1] ** (1.0 / year_count_float) - 1.0)
        if drawdown_float == 0.0:
            return float("inf")
        return cagr_float / abs(drawdown_float)
    raise ValueError(f"Unknown selection metric: {metric_str}")


def _summary_row_dict(return_ser: pd.Series, label_str: str) -> dict[str, object]:
    clean_ser = return_ser.dropna()
    equity_ser = (1.0 + clean_ser).cumprod()
    year_count_float = len(clean_ser) / MONTHS_PER_YEAR_INT
    return {
        "series": label_str,
        "months": int(len(clean_ser)),
        "cagr_pct": (float(equity_ser.iloc[-1] ** (1.0 / year_count_float)) - 1.0) * 100.0,
        "vol_pct": float(clean_ser.std() * np.sqrt(MONTHS_PER_YEAR_INT)) * 100.0,
        "sharpe": float(clean_ser.mean() / clean_ser.std() * np.sqrt(MONTHS_PER_YEAR_INT)),
        "max_dd_pct": float((equity_ser / equity_ser.cummax() - 1.0).min()) * 100.0,
    }


def run_walk_forward(
    sweep_dir_str: str,
    min_training_months_int: int = 96,
    oos_step_months_int: int = 24,
    selection_metric_str: str = "sharpe",
    frozen_variant_str: str = "n20_lam0p75",
    baseline_variant_str: str = "n20_lam0",
) -> pd.DataFrame:
    sweep_path = Path(sweep_dir_str)
    equity_df = pd.read_csv(sweep_path / "equity_curve.csv", index_col=0, parse_dates=True)

    # *** CRITICAL*** Monthly returns per variant from independently
    # compounded daily equity, month-end marks only.
    monthly_return_df = (
        equity_df.astype(float).resample("ME").last().pct_change().dropna(how="all")
    )
    variant_str_list = list(monthly_return_df.columns)
    for required_str in (frozen_variant_str, baseline_variant_str):
        if required_str not in variant_str_list:
            raise RuntimeError(f"equity_curve.csv is missing required variant {required_str}.")

    month_index = monthly_return_df.index
    if len(month_index) <= min_training_months_int + oos_step_months_int:
        raise RuntimeError("Not enough history for the requested training/OOS split.")

    selection_row_list: list[dict[str, object]] = []
    oos_return_list: list[pd.Series] = []
    cursor_int = min_training_months_int
    while cursor_int < len(month_index):
        selection_date_ts = month_index[cursor_int - 1]
        training_return_df = monthly_return_df.iloc[:cursor_int]

        metric_by_variant_ser = pd.Series(
            {
                variant_str: _metric_float(training_return_df[variant_str], selection_metric_str)
                for variant_str in variant_str_list
            }
        )
        # Deterministic tie-break: highest metric, then variant name.
        chosen_variant_str = str(
            metric_by_variant_ser.sort_index().sort_values(ascending=False, kind="mergesort").index[0]
        )

        oos_slice_df = monthly_return_df.iloc[cursor_int : cursor_int + oos_step_months_int]
        oos_return_list.append(oos_slice_df[chosen_variant_str])
        selection_row_list.append(
            {
                "selection_date": selection_date_ts.date().isoformat(),
                "chosen_variant": chosen_variant_str,
                "training_months": int(cursor_int),
                "oos_months": int(len(oos_slice_df)),
                "training_metric": float(metric_by_variant_ser[chosen_variant_str]),
            }
        )
        cursor_int += oos_step_months_int

    wfa_return_ser = pd.concat(oos_return_list)
    oos_start_ts = wfa_return_ser.index[0]

    frozen_oos_ser = monthly_return_df.loc[oos_start_ts:, frozen_variant_str]
    baseline_oos_ser = monthly_return_df.loc[oos_start_ts:, baseline_variant_str]

    summary_df = pd.DataFrame(
        [
            _summary_row_dict(wfa_return_ser, f"walk_forward_{selection_metric_str}"),
            _summary_row_dict(frozen_oos_ser, f"frozen_{frozen_variant_str}"),
            _summary_row_dict(baseline_oos_ser, f"baseline_{baseline_variant_str}"),
        ]
    )
    selection_df = pd.DataFrame(selection_row_list)

    out_path = sweep_path / f"wfa_{selection_metric_str}"
    out_path.mkdir(exist_ok=True)
    summary_df.to_csv(out_path / "wfa_summary.csv", index=False)
    selection_df.to_csv(out_path / "wfa_selections.csv", index=False)
    wfa_return_ser.rename("wfa_return").to_csv(out_path / "wfa_oos_returns.csv")
    (out_path / "wfa_config.json").write_text(
        json.dumps(
            {
                "min_training_months": min_training_months_int,
                "oos_step_months": oos_step_months_int,
                "selection_metric": selection_metric_str,
                "frozen_variant": frozen_variant_str,
                "baseline_variant": baseline_variant_str,
                "oos_start": oos_start_ts.date().isoformat(),
                "variants": variant_str_list,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"OOS period: {oos_start_ts.date()} -> {wfa_return_ser.index[-1].date()}")
    print(selection_df.to_string(index=False))
    print(summary_df.round(3).to_string(index=False))
    print(f"wrote: {out_path}")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--min-training-months", type=int, default=96)
    parser.add_argument("--oos-step-months", type=int, default=24)
    parser.add_argument("--metric", default="sharpe", choices=["sharpe", "mar"])
    parser.add_argument("--frozen-variant", default="n20_lam0p75")
    parser.add_argument("--baseline-variant", default="n20_lam0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_walk_forward(
        sweep_dir_str=args.sweep_dir,
        min_training_months_int=args.min_training_months,
        oos_step_months_int=args.oos_step_months,
        selection_metric_str=args.metric,
        frozen_variant_str=args.frozen_variant,
        baseline_variant_str=args.baseline_variant,
    )
