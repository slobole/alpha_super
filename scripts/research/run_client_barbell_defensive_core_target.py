"""Fresh research-only run of the client defensive target.

The target includes shadow DV2-IND, so it deliberately does not pass the
PortfolioManager PM_READY gate. Initial weights are equal thirds (rounded to
sum to one). At each annual rebalance, w_i = (1 / sigma_i) / sum_j(1 / sigma_j),
where sigma uses the preceding 252 pod returns and excludes the rebalance row.
This runner never changes strategy registration or live wiring.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.portfolio import Portfolio  # noqa: E402
from alpha.engine.report import (  # noqa: E402
    build_research_output_path,
    save_portfolio_results,
    save_results as save_strategy_results,
)
from data.norgate_loader import (  # noqa: E402
    INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT,
    TOTALRETURN_ADJUSTMENT_STR,
    load_price_timeseries,
)

CONFIG_PATH = REPO_ROOT_PATH / "portfolios" / "client_barbell_defensive_core_target.yaml"
EXPECTED_IMPORT_TUPLE = (
    "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5",
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "strategies.dv2.strategy_mr_dv2_industry_etf",
)


def main() -> int:
    config_dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pod_dict_list = config_dict["pods"]
    if config_dict["name_str"] != "client_barbell_defensive_core_target":
        raise ValueError("Unexpected target portfolio name")
    if tuple(pod_dict["strategy_import_str"] for pod_dict in pod_dict_list) != EXPECTED_IMPORT_TUPLE:
        raise ValueError("Unexpected target pod imports")
    weight_list = [float(pod_dict["weight_float"]) for pod_dict in pod_dict_list]
    if abs(sum(weight_list) - 1.0) >= 1e-6:
        raise ValueError("Target pod weights must sum to one")
    if config_dict["rebalance"] != {
        "frequency_str": "annually", "policy_str": "inverse_volatility", "lookback_day_int": 252,
    }:
        raise ValueError("Unexpected target rebalance rule")

    output_path = build_research_output_path(
        REPO_ROOT_PATH / "results", "portfolio", config_dict["name_str"], "research_target",
    )
    strategy_list = []
    pod_info_list = []
    for pod_dict, weight_float in zip(pod_dict_list, weight_list):
        strategy_module_obj = importlib.import_module(pod_dict["strategy_import_str"])
        strategy_obj = strategy_module_obj.run_variant(
            show_display_bool=False,
            save_results_bool=False,
            output_dir_str=str(output_path),
            backtest_start_date_str=config_dict["backtest_start_date_str"],
            end_date_str=config_dict["end_date_str"],
            capital_base_float=float(config_dict["capital_base_float"]) * weight_float,
        )
        pod_output_path = save_strategy_results(
            strategy_obj, output_path=output_path / "pods" / pod_dict["pod_id_str"],
        )
        strategy_list.append(strategy_obj)
        pod_info_list.append({
            "pod_id_str": pod_dict["pod_id_str"],
            "strategy_name": strategy_obj.name,
            "strategy_import_str": pod_dict["strategy_import_str"],
            "source_type_str": "fresh_research_run",
            "pod_artifact_dir": str(pod_output_path),
            "backtest_start_date_str": pd.Timestamp(strategy_obj.results.index[0]).date().isoformat(),
        })
        print(f"Saved {pod_dict['pod_id_str']}: {pod_output_path}", flush=True)

    # *** CRITICAL *** The Portfolio engine excludes the rebalance date and uses
    # only the prior 252 pod returns to set annual inverse-volatility weights.
    benchmark_symbol_str = INDEX_TOTALRETURN_DATA_SYMBOL_MAP_DICT["$SPX"]
    benchmark_price_df = load_price_timeseries(
        benchmark_symbol_str,
        adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        start_date_str=config_dict["backtest_start_date_str"],
        end_date_str=config_dict["end_date_str"],
    )
    portfolio_obj = Portfolio(
        strategies=strategy_list,
        weights=weight_list,
        name=config_dict["name_str"],
        capital_base=float(config_dict["capital_base_float"]),
        rebalance="annually",
        rebalance_policy_str="inverse_volatility",
        rebalance_inverse_volatility_lookback_day_int=252,
        pod_info_list=pod_info_list,
        regression_benchmark_value_ser=benchmark_price_df["Close"].astype(float),
        regression_benchmark_label_str="$SPX · TOTALRETURN",
        regression_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
    )
    portfolio_obj.source_config_path = str(CONFIG_PATH)
    portfolio_obj.source_config_dict = config_dict
    save_portfolio_results(portfolio_obj, output_path=output_path)
    (output_path / "research_target_metadata.json").write_text(json.dumps({
        "status_str": "research_only_shadow_pod",
        "source_config_sha256_str": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "runner_sha256_str": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pm_ready_validation_str": "not_run_shadow_dv2_industry_etf",
        "methodology_str": "Portfolio engine requires 252 prior returns; early annual resets are skipped",
        "limits_list": [
            "Current fixed 19-ETF DV2-IND universe has survivorship bias",
            "Portfolio-level annual rebalancing has no transfer friction or tax",
            "Not a literal replay of the defensive-v2 study or a live-ready portfolio",
        ],
        "requested_start_date_str": config_dict["backtest_start_date_str"],
        "effective_start_date_str": pd.Timestamp(portfolio_obj.results.index[0]).date().isoformat(),
        "effective_end_date_str": pd.Timestamp(portfolio_obj.results.index[-1]).date().isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"Research target saved to: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
