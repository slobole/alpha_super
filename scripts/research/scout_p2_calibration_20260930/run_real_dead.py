"""Part C: known-dead real cases through the S5 gates, on their in-sample period only (PROTOCOL.md, Part C).

Cases (saved daily return grids from earlier studies, read-only):
- Z8 L2016 (16 variants) and Z9 L2017 / Z9 N14 (24 each): excess over the equal-weight (1/N) benchmark of the same
  list; in sample = before the publication date (Z8 2016-07-01, Z9 2017-10-01).
- Alpha101 Stage A (100 alphas): daily decile long-short return, GROSS; in sample = P1 2000-2011.
Selection = highest in-sample Sharpe (these grids are not all regular, and the original studies ranked by Sharpe).
MCPT is not run: re-running these searches on permuted data needs their full simulators.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH  # noqa: E402
from alpha.scout.trials import effective_trial_count  # noqa: E402
from alpha.stats.pbo import probability_of_backtest_overfitting  # noqa: E402
from alpha.stats.psr_dsr import deflated_sharpe_ratio  # noqa: E402
from alpha.stats.walk_forward import (  # noqa: E402
    REGISTERED_DESIGN,
    annualized_sharpe_float,
    design_sensitivity_df,
    run_walk_forward,
    select_max_sharpe,
)

ZORRO_PATH = Path(r"C:\Users\User\Documents\workspace\Pakal\pakal-research\reports\zorro_zsystems_daily_audit\tables\daily_returns_all_runs.parquet")
ALPHA101_PATH = MAIN_CHECKOUT_ROOT_PATH / "results/research/alpha101_20260928/_cache/stageA_daily_U1.parquet"
OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p2_calibration"


def _cases() -> list[dict]:
    zorro_df = pd.read_parquet(ZORRO_PATH)
    case_list = []
    for name_str, prefix_str, bench_str, publication_str in (
        ("Z8 L2016 vs 1/N", "Z8|L2016|", "BENCH|1/N L2016", "2016-07-01"),
        ("Z9 L2017 vs 1/N", "Z9|L2017|", "BENCH|1/N L2017", "2017-10-01"),
        ("Z9 N14 vs 1/N", "Z9|N14|", "BENCH|1/N N14 (Z9 window)", "2017-10-01"),
    ):
        variant_df = zorro_df[[c for c in zorro_df.columns if c.startswith(prefix_str) and "claim_like" not in c]]
        excess_df = variant_df.sub(zorro_df[bench_str], axis=0)
        case_list.append({"name_str": name_str, "excess_df": excess_df, "split_str": publication_str})
    alpha_df = pd.read_parquet(ALPHA101_PATH)
    ls_df = alpha_df[[f"{i}|ls" for i in range(1, 102) if f"{i}|ls" in alpha_df.columns]]
    case_list.append({"name_str": "Alpha101 long-short (gross)", "excess_df": ls_df, "split_str": "2012-01-01"})
    return case_list


def _evaluate(case_dict: dict) -> dict:
    excess_df = case_dict["excess_df"]
    split_ts = pd.Timestamp(case_dict["split_str"])
    in_df = excess_df.loc[excess_df.index < split_ts]
    out_df = excess_df.loc[excess_df.index >= split_ts]
    # Start where at least 80% of the variants have data; drop variants still under 95% coverage after that
    # (reported); then use the common span.
    share_live_ser = in_df.notna().mean(axis=1)
    in_df = in_df.loc[share_live_ser.index[share_live_ser >= 0.8][0] :]
    coverage_ser = in_df.notna().mean()
    dropped_count_int = int((coverage_ser < 0.95).sum())
    in_df = in_df.loc[:, coverage_ser >= 0.95].dropna(how="any")
    out_df = out_df[in_df.columns]
    out_df = out_df.dropna(how="all")

    sharpe_ser = in_df.apply(annualized_sharpe_float)
    chosen_str = sharpe_ser.idxmax()
    years_float = len(in_df) / 252.0
    per_period_vec = sharpe_ser.to_numpy() / np.sqrt(252.0)
    n_eff_float = effective_trial_count(in_df.loc[:, in_df.std() > 0])
    dsr = deflated_sharpe_ratio(in_df[chosen_str].to_numpy(), float(np.var(per_period_vec, ddof=1)), n_eff_float)
    pbo = probability_of_backtest_overfitting(in_df.fillna(0.0).to_numpy(), 10)

    wf_row_dict = {"wf_runnable_bool": False, "wf_pass_bool": None}
    registered_start = in_df.index[0] + pd.DateOffset(years=REGISTERED_DESIGN.train_years_int)
    if pd.Timestamp(year=registered_start.year + 1, month=1, day=1) <= in_df.index[-1]:
        registered = run_walk_forward(in_df, select_max_sharpe, REGISTERED_DESIGN)
        sensitivity_df = design_sensitivity_df(in_df, select_max_sharpe)
        runnable_df = sensitivity_df.dropna(subset=["oos_sharpe_float"])
        positive_share_float = float(runnable_df["oos_positive_bool"].astype(bool).mean())
        wf_row_dict = {
            "wf_runnable_bool": True,
            "wf_efficiency_float": registered.efficiency_float,
            "wf_oos_sharpe_float": registered.oos_sharpe_float,
            "wf_positive_share_float": positive_share_float,
            "wf_pass_bool": bool(
                np.isfinite(registered.efficiency_float) and registered.efficiency_float >= 0.5
                and registered.oos_sharpe_float > 0 and positive_share_float >= 0.75
            ),
        }
    out_sharpe_ser = out_df.apply(annualized_sharpe_float)
    return {
        "case_str": case_dict["name_str"],
        "in_sample_str": f"{in_df.index[0].date()} to {in_df.index[-1].date()}",
        "variant_count_int": in_df.shape[1],
        "dropped_low_coverage_int": dropped_count_int,
        "chosen_str": chosen_str,
        "in_sample_sharpe_float": float(sharpe_ser[chosen_str]),
        "naive_t_float": float(sharpe_ser[chosen_str] * np.sqrt(years_float)),
        "n_eff_float": n_eff_float,
        "dsr_float": dsr.deflated_sharpe_float,
        "dsr_pass_bool": dsr.deflated_sharpe_float >= 0.95,
        "pbo_float": pbo.pbo_float,
        **wf_row_dict,
        "out_of_sample_sharpe_float": float(out_sharpe_ser[chosen_str]),
        "out_of_sample_positive_share_float": float((out_sharpe_ser > 0).mean()),
    }


def main() -> None:
    row_list = [_evaluate(case_dict) for case_dict in _cases()]
    frame = pd.DataFrame(row_list)
    frame.to_json(OUTPUT_DIR_PATH / "real_dead.json", orient="records", indent=2)
    pd.set_option("display.width", 250)
    print(frame.round(3).T.to_string())


if __name__ == "__main__":
    main()
