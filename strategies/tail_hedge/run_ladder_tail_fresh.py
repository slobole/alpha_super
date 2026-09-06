"""Fresh share-level PM confirmation of a preselected hedge candidate.

The unchanged baseline and explicit candidate YAML run at the same capital,
requested start and end. Original YAMLs are never mutated. No live APIs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from alpha.engine.portfolio_manager import PortfolioManager


def run_pair(base_path: Path, candidate_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=False)
    artifact_dict = {}
    candidate_manager_obj = PortfolioManager.from_yaml(candidate_path)
    base_manager_obj = PortfolioManager.from_yaml(base_path)
    if base_manager_obj.config.capital_base_float != candidate_manager_obj.config.capital_base_float:
        raise ValueError("Matched pair must use the same capital.")
    if base_manager_obj.config.backtest_start_date_str != candidate_manager_obj.config.backtest_start_date_str:
        raise ValueError("Matched pair must request the same start.")
    for label_str, manager_obj in (("baseline", base_manager_obj), ("candidate", candidate_manager_obj)):
        manager_obj.config = replace(manager_obj.config,
            end_date_str=candidate_manager_obj.config.end_date_str,
            name_str=manager_obj.config.name_str + ("_tail_control" if label_str == "baseline" else ""))
        result_obj = manager_obj.run(output_dir_str="results", save_results_bool=True,
                                    show_display_bool=False, max_workers_int=3)
        portfolio_obj = result_obj.portfolio
        portfolio_obj.results.to_csv(output_path / f"{label_str}_results.csv")
        portfolio_obj._daily_rets.to_csv(output_path / f"{label_str}_pod_returns.csv")
        portfolio_obj.drift_weight_df.to_csv(output_path / f"{label_str}_drift.csv")
        artifact_dict[label_str] = {"artifact": str(result_obj.portfolio_output_dir_path),
            "capital": manager_obj.config.capital_base_float,
            "config": str(base_path if label_str == "baseline" else candidate_path),
            "end": manager_obj.config.end_date_str}
        (output_path / "manifest.json").write_text(json.dumps(artifact_dict, indent=2), encoding="utf-8")
        print(f"Completed {label_str}: {result_obj.portfolio_output_dir_path}", flush=True)


if __name__ == "__main__":
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--base", type=Path, required=True)
    parser_obj.add_argument("--candidate", type=Path, required=True)
    parser_obj.add_argument("--output", type=Path, required=True)
    argument_obj = parser_obj.parse_args()
    run_pair(argument_obj.base, argument_obj.candidate, argument_obj.output)
