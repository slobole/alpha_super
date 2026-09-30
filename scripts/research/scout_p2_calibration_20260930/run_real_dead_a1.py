"""Amendment 1, Part C: the known-dead grids with the correlation-aware DSR benchmark (selection = maximum)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH))
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from alpha.stats.psr_dsr import null_selected_sharpe_benchmark, probabilistic_sharpe_ratio, sharpe_moments  # noqa: E402
from alpha.stats.walk_forward import annualized_sharpe_float  # noqa: E402

from run_real_dead import OUTPUT_DIR_PATH, _cases  # noqa: E402


def main() -> None:
    row_list = []
    for case_dict in _cases():
        excess_df = case_dict["excess_df"]
        split_ts = pd.Timestamp(case_dict["split_str"])
        in_df = excess_df.loc[excess_df.index < split_ts]
        share_live_ser = in_df.notna().mean(axis=1)
        in_df = in_df.loc[share_live_ser.index[share_live_ser >= 0.8][0] :]
        coverage_ser = in_df.notna().mean()
        in_df = in_df.loc[:, coverage_ser >= 0.95].dropna(how="any")
        out_df = excess_df.loc[excess_df.index >= split_ts, in_df.columns]

        sharpe_ser = in_df.apply(annualized_sharpe_float)
        chosen_str = sharpe_ser.idxmax()
        moments = sharpe_moments(in_df[chosen_str].to_numpy())
        correlation_mat = np.corrcoef(in_df.to_numpy(), rowvar=False)
        benchmark_float = null_selected_sharpe_benchmark(correlation_mat, len(in_df), draw_count_int=50_000, random_seed_int=7)
        dsr_corr_float = probabilistic_sharpe_ratio(
            moments.sharpe_float, len(in_df), moments.skewness_float, moments.kurtosis_float, benchmark_float
        )
        off_diagonal_vec = correlation_mat[~np.eye(correlation_mat.shape[0], dtype=bool)]
        row_list.append(
            {
                "case_str": case_dict["name_str"],
                "variant_count_int": in_df.shape[1],
                "mean_pairwise_correlation_float": float(off_diagonal_vec.mean()),
                "chosen_str": chosen_str,
                "in_sample_sharpe_float": float(sharpe_ser[chosen_str]),
                "benchmark_annual_sharpe_float": benchmark_float * np.sqrt(252),
                "dsr_corr_float": dsr_corr_float,
                "dsr_corr_pass_bool": dsr_corr_float >= 0.95,
                "out_of_sample_sharpe_float": float(annualized_sharpe_float(out_df[chosen_str])),
            }
        )
    frame = pd.DataFrame(row_list)
    frame.to_json(OUTPUT_DIR_PATH / "a1_real_dead.json", orient="records", indent=2)
    pd.set_option("display.width", 250)
    print(frame.round(3).T.to_string())


if __name__ == "__main__":
    main()
