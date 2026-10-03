"""Multiplicity-adjusted paired comparison for the DV2 limit-anchor study (added after the quant-pitfalls review).

The per-cell paired bootstrap in run_anchor.py has no multiple-comparison control. Here, per universe and cost model,
every non-baseline limit-exit cell's Sharpe difference against close x natr14 at the same fill target is tested jointly:
one stationary-bootstrap index matrix (the same seed, block and length as run_anchor) is shared by all cells, and a
single-step Romano-Wolf max-t gives each cell an adjusted one-sided p-value for "better than the baseline":
    t_j = d_j / sd_j,   t*_jb = (d*_jb - d_j) / sd_j,   p_j = share of draws b with max_k t*_kb >= t_j.
Two families: the 6 open-anchor downside-excursion cells (the registered hypothesis) and all 21 non-baseline cells.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/multiplicity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.stats.bootstrap import stationary_bootstrap_index_mat
from scripts.research.scout_dv2_limit_anchor_20261003.register import (
    BASELINE_TUPLE,
    FILL_TARGET_TUPLE,
    UNIVERSE_TUPLE,
)

OUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "dv2_limit_anchor"
PATH_INT, BLOCK_FLOAT, SEED_INT = 2000, 20.0, 20261003  # as run_anchor.paired_sharpe_difference
HYPOTHESIS_MEASURE_TUPLE = ("dex21_mean", "dex63_quantile")


def slug(name_str: str) -> str:
    return name_str.replace(" ", "_").replace("&", "and")


def path_sharpe_mat(value_vec: np.ndarray, index_mat: np.ndarray) -> np.ndarray:
    path_mat = value_vec[index_mat]
    return path_mat.mean(axis=1) / path_mat.std(axis=1, ddof=1) * np.sqrt(252.0)


def sharpe(value_vec: np.ndarray) -> float:
    return float(value_vec.mean() / value_vec.std(ddof=1) * np.sqrt(252.0))


def max_t_table(daily_df: pd.DataFrame, case_str: str, label_list: list[str]) -> pd.DataFrame:
    index_mat = stationary_bootstrap_index_mat(len(daily_df), PATH_INT, BLOCK_FLOAT, len(daily_df), SEED_INT)
    observed_list, boot_list = [], []
    for label_str in label_list:
        _, _, target_str, exit_str = label_str.split("|")
        base_str = f"{BASELINE_TUPLE[0]}|{BASELINE_TUPLE[1]}|{target_str}|{exit_str}"
        variant_vec = daily_df[f"{label_str}|{case_str}"].to_numpy()
        base_vec = daily_df[f"{base_str}|{case_str}"].to_numpy()
        observed_list.append(sharpe(variant_vec) - sharpe(base_vec))
        boot_list.append(path_sharpe_mat(variant_vec, index_mat) - path_sharpe_mat(base_vec, index_mat))
    observed_vec = np.array(observed_list)
    boot_mat = np.column_stack(boot_list)  # paths x cells
    sd_vec = boot_mat.std(axis=0, ddof=1)
    t_vec = observed_vec / sd_vec
    max_null_vec = ((boot_mat - observed_vec) / sd_vec).max(axis=1)
    adjusted_vec = np.array([(max_null_vec >= t_float).mean() for t_float in t_vec])
    raw_vec = np.array([((boot_mat[:, j] - observed_vec[j]) / sd_vec[j] >= t_vec[j]).mean() for j in range(len(label_list))])
    return pd.DataFrame({"cell": label_list, "case": case_str, "difference": observed_vec, "t": t_vec, "p_raw_one_sided": raw_vec,
                         "p_romano_wolf": adjusted_vec})


def main() -> None:
    frame_list = []
    for name_str in UNIVERSE_TUPLE:
        path = OUT_PATH / "universes" / f"{slug(name_str)}_daily.csv"
        if not path.exists():
            continue
        daily_df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
        all_list = sorted({c.rsplit("|", 1)[0] for c in daily_df.columns if c.count("|") == 4})
        all_list = [c for c in all_list if not c.startswith(f"{BASELINE_TUPLE[0]}|{BASELINE_TUPLE[1]}|")
                    and any(c.split("|")[2] == f"f{t:.2f}" for t in FILL_TARGET_TUPLE)]
        hypothesis_list = [c for c in all_list if c.split("|")[0] == "open" and c.split("|")[1] in HYPOTHESIS_MEASURE_TUPLE]
        for family_str, label_list in (("open_excursion_6", hypothesis_list), ("all_21", all_list)):
            for case_str in ("ar", "pooled"):
                table = max_t_table(daily_df, case_str, label_list).assign(universe=name_str, family=family_str)
                frame_list.append(table)
    out = pd.concat(frame_list, ignore_index=True)
    (OUT_PATH / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH / "tables" / "multiplicity.csv", index=False)
    pd.set_option("display.width", 220)
    for (universe_str, family_str), group in out.groupby(["universe", "family"], sort=False):
        wide = group.pivot(index="cell", columns="case", values=["difference", "p_raw_one_sided", "p_romano_wolf"])
        print(f"\n## {universe_str}, family {family_str}\n")
        print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
