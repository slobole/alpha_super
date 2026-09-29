"""Write each menu product as a PortfolioManager YAML in portfolios/ (house schema).

Weights come from books/product_weights.csv (the frozen-spec build). The files
are research configurations for the Portfolio Manager and Bench; they do not
wire anything to a live account.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_sources import SLEEVE_ALIAS_BY_IMPORT_DICT  # noqa: E402

IMPORT_BY_ALIAS_DICT = {alias_str: import_str for import_str, alias_str in SLEEVE_ALIAS_BY_IMPORT_DICT.items()}
PORTFOLIO_DIR_PATH = common.REPO_ROOT_PATH / "portfolios"


def main() -> int:
    spec_dict = yaml.safe_load((Path(__file__).resolve().parent / "frozen_spec.yaml").read_text(encoding="utf-8"))
    amendment_path = Path(__file__).resolve().parent / "amendments.yaml"
    amendment_by_product_dict = ({a["product_id_str"]: a for a in yaml.safe_load(amendment_path.read_text(encoding="utf-8"))["amendments"]}
                                 if amendment_path.exists() else {})
    weight_df = pd.read_csv(common.STUDY_DIR_PATH / "books" / "product_weights.csv")
    run_summary_dict = json.loads((common.STUDY_DIR_PATH / "books" / "run_summary.json").read_text(encoding="utf-8"))
    exact_start_str = run_summary_dict["exact_window"][0]
    written_list = []
    for product_dict in spec_dict["products"]:
        if not product_dict.get("write_yaml_bool", True):
            continue
        product_id_str = product_dict["product_id_str"]
        rows_df = weight_df[(weight_df["product_id_str"] == product_id_str) & weight_df["weight_float"].notna()]
        rows_df = rows_df.sort_values("weight_float", ascending=False)
        file_name_str = product_dict["yaml_name_str"]
        risk_label_str = (
            f"volatility target {product_dict['target_volatility_float']:.1%}"
            if "target_volatility_float" in product_dict
            else "return engines only (no volatility target)"
        )
        header_str = (
            f"# {product_dict['display_name_str']} — fund product menu 2026-09 "
            f"(results/research/portfolio/fund_product_menu_20260923).\n"
            f"# Two-bucket capital template from the frozen spec (return engines at equal capital,\n"
            f"# CORE5-anchored stabilizers); {risk_label_str}, drawdown budget "
            f"{product_dict['max_drawdown_budget_float']:.0%}. Annual reset to target weights.\n"
            f"# Pods holding Tactical FI cannot run fresh until its frozen Norgate fingerprint is\n"
            f"# re-frozen (a benchmark-only revision; IEF and LQD are unchanged).\n"
            f"# Research configuration only: not an allocation approval and not wired to LIVE.\n"
        )
        amendment_dict = amendment_by_product_dict.get(product_id_str)
        if amendment_dict:
            header_str += (
                f"# Amended {amendment_dict['decided_date_str']} ({amendment_dict['amendment_id_str']}, owner decision after the results):\n"
                f"# removed {', '.join(amendment_dict['drop_alias_list'])}; the other pods scaled up pro rata "
                f"(scripts/research/fund_menu_20260923/amendments.yaml).\n"
            )
        config_dict = {
            "name_str": file_name_str,
            "capital_base_float": 1_000_000.0,
            "backtest_start_date_str": exact_start_str,
            "end_date_str": spec_dict["end_date_str"],
            "allocation_policy_str": "fixed",
            "max_workers_int": None,
            "rebalance": {"frequency_str": "annually", "policy_str": "fixed"},
            "save_pod_artifacts_bool": True,
            "regression_benchmark_symbol_str": "$SPX",
            "pods": [
                {
                    "pod_id_str": f"pod_{row.alias_str}",
                    "strategy_import_str": IMPORT_BY_ALIAS_DICT[row.alias_str],
                    "weight_float": round(float(row.weight_float), 4),
                }
                for row in rows_df.itertuples()
            ],
        }
        output_path = PORTFOLIO_DIR_PATH / f"{file_name_str}.yaml"
        output_path.write_text(header_str + yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
        written_list.append(str(output_path.relative_to(common.REPO_ROOT_PATH)))
    print("\n".join(written_list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
