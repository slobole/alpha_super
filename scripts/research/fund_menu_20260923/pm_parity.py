"""Parity: the study's matrix books vs fresh PortfolioManager runs of the same books.

The study combines sleeves run once at $1M (returns are scale-free), joined
mid-flight at the window start. The PortfolioManager runs every pod at its real
capital and starts each pod in cash. This script runs selected books through
the real PortfolioManager on the same end date and reports how far apart the two
constructions land. Products holding Tactical Fixed Income cannot run fresh (its
frozen-fingerprint guard, see run_sources.py), so parity uses books without it.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_sources import SLEEVE_ALIAS_BY_IMPORT_DICT  # noqa: E402

from alpha.engine.portfolio_manager import PortfolioManager, build_portfolio_manager_config  # noqa: E402

ALIAS_BY_IMPORT_DICT = dict(SLEEVE_ALIAS_BY_IMPORT_DICT)


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", action="append", required=True, help="id=path/to/portfolio.yaml:rebalance(none|annual)")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(arg_list)

    spec_dict = yaml.safe_load((Path(__file__).resolve().parent / "frozen_spec.yaml").read_text(encoding="utf-8"))
    end_date_str = spec_dict["end_date_str"]
    sleeve_return_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    row_list = []
    for book_arg_str in args.book:
        book_id_str, rest_str = book_arg_str.split("=", maxsplit=1)
        yaml_path_str, rebalance_str = rest_str.rsplit(":", maxsplit=1)
        config_dict = yaml.safe_load((common.REPO_ROOT_PATH / yaml_path_str).read_text(encoding="utf-8"))
        run_config_dict = copy.deepcopy(config_dict)
        run_config_dict["name_str"] = f"fund_menu_parity_{book_id_str}"
        run_config_dict["end_date_str"] = end_date_str
        run_config_dict["max_workers_int"] = args.workers
        manager_obj = PortfolioManager(
            config=build_portfolio_manager_config(run_config_dict),
            source_config_path_str=None,
            source_config_dict=run_config_dict,
        )
        run_result_obj = manager_obj.run(output_dir_str=str(common.STUDY_DIR_PATH / "pm_parity_runs"), save_results_bool=False)
        pm_value_ser = run_result_obj.portfolio.results["total_value"].astype(float)
        pm_value_ser.index = pd.to_datetime(pm_value_ser.index).normalize()
        pm_return_ser = pm_value_ser.pct_change().dropna()

        weight_dict = {ALIAS_BY_IMPORT_DICT[p["strategy_import_str"]]: float(p["weight_float"]) for p in config_dict["pods"]}
        start_ts = max(pm_return_ser.index[0], max(sleeve_return_df[a].first_valid_index() for a in weight_dict))
        window_index = sleeve_return_df.loc[start_ts:end_date_str].index.intersection(pm_return_ser.index)
        matrix_ser, _ = common.book_return_ser(sleeve_return_df.loc[window_index], weight_dict, rebalance_str)
        pm_window_ser = pm_return_ser.loc[window_index]
        year_count_float = len(window_index) / 252.0
        row_list.append(
            {
                "book_str": book_id_str,
                "window_str": f"{window_index[0].date()} to {window_index[-1].date()}",
                "pm_cagr": float((1 + pm_window_ser).prod() ** (1 / year_count_float) - 1),
                "matrix_cagr": float((1 + matrix_ser).prod() ** (1 / year_count_float) - 1),
                "pm_vol": float(pm_window_ser.std() * np.sqrt(252)),
                "matrix_vol": float(matrix_ser.std() * np.sqrt(252)),
                "daily_corr": float(pm_window_ser.corr(matrix_ser)),
                "tracking_error": float((pm_window_ser - matrix_ser).std() * np.sqrt(252)),
            }
        )
        print(row_list[-1], flush=True)
    parity_df = pd.DataFrame(row_list)
    parity_dir_path = common.STUDY_DIR_PATH / "parity"
    parity_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = parity_dir_path / "pm_parity.csv"
    if output_path.exists():
        parity_df = pd.concat([pd.read_csv(output_path), parity_df]).drop_duplicates("book_str", keep="last")
    parity_df.to_csv(output_path, index=False, float_format="%.6g")
    print(parity_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
