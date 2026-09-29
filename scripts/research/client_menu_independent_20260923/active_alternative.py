"""One disclosed adaptive hypothesis: sector-ETF reversal as an active mandate."""
from __future__ import annotations
from datetime import datetime, timezone
from itertools import combinations
import json
import pandas as pd
from scripts.research.client_menu_independent_20260923.protocol import SOURCE_PATH, STUDY_PATH, digest_str, write_json
from scripts.research.client_menu_independent_20260923.analyze import horizon_metrics, risk_gate
from scripts.research.portfolio_family_20260923 import analyze as ledger


def main() -> None:
    main_spec_dict = json.loads((STUDY_PATH/"research_spec_frozen.json").read_text(encoding="utf-8"))
    spec_path = STUDY_PATH/"adaptive_active_spec.json"
    if not spec_path.exists():
        write_json(spec_path, {"frozen_at": datetime.now(timezone.utc).isoformat(), "post_result": True, "hypothesis_id": "A3", "provenance": "Original all25 diagnostic A0_swap_SECTOR_6 meets the active smoothing/tradeoff idea on2019+; now test all actual available2018+history includingQ42018. Same composition already inspected; no new chosen weights.", "weights": {"MOSAIC": 1/3, "SECTOR_6": 1/3, "DF_BTAL_QQQ_LINEAR": 1/3}, "anchor": "2018-07-19", "end": "2026-07-31", "comparator": "M0 on identical dates", "acceptance": "Original A risk/horizon gates AND DD>=2pp improvement OR ES5loss>=10% improvement versus M0, with CAGR sacrifice<=1.5pp, in BOTH central and conservative scenarios; every6pairwise10pp neighbor must retain risk gates underbothcosts. Passing means shorter-history provisional active candidate, never equivalent14year evidence or independent holdout.", "cells": 20, "original_frozen_sha256": digest_str(STUDY_PATH/"research_spec_frozen.json"), "diagnostic_seen_sha256": digest_str(STUDY_PATH/"tables/results.csv")})
    spec_dict = json.loads(spec_path.read_text())
    if digest_str(STUDY_PATH/"research_spec_frozen.json") != spec_dict["original_frozen_sha256"]:
        raise ValueError("Original protocol changed")
    for input_dict in main_spec_dict["input_files"]:
        ledger.verify_file(input_dict)
    ledger.preflight_source_ids(main_spec_dict, json.loads((SOURCE_PATH/"source_audit_addendum.json").read_text()))
    catalog_list = json.loads((SOURCE_PATH/"catalog_complete.json").read_text(encoding="utf-8"))
    alias_id_dict = {row_dict["alias"]: row_dict["strategy_import"].split(":")[0].split(".")[-1] for row_dict in catalog_list}
    alias_id_dict.update({"BIL": "benchmark_bil", "SPY": "benchmark_spy"})
    component_dict = {alias_str: ledger.source_components(SOURCE_PATH/("benchmarks" if alias_str in {"BIL", "SPY"} else "data")/alias_id_dict[alias_str])[0] for alias_str in [*spec_dict["weights"], "SPY", "BIL"]}
    configuration_dict = {"A3": {"weights": spec_dict["weights"], "scenarios": ["native", "common_account", "conservative"]}, "M0_matched": {"weights": main_spec_dict["portfolio"]["candidates"]["M0"]["weights"], "scenarios": ["native", "common_account", "conservative"]}, "BIL_matched": {"weights": {"BIL": 1.}, "scenarios": ["common_account"]}, "SPY_matched": {"weights": {"SPY": 1.}, "scenarios": ["common_account"]}}
    for pair_int, (left_str, right_str) in enumerate(combinations(spec_dict["weights"], 2)):
        for sign_int in [-1, 1]:
            weight_dict = dict(spec_dict["weights"])
            weight_dict[left_str] += sign_int*.1
            weight_dict[right_str] -= sign_int*.1
            configuration_dict[f"A3_pair{pair_int}_{sign_int:+d}"] = {"weights": weight_dict, "scenarios": ["common_account", "conservative"]}
    result_list, return_dict = [], {}
    for candidate_str, configuration_info_dict in configuration_dict.items():
        for scenario_str in configuration_info_dict["scenarios"]:
            panel_df = ledger.strict_return_panel(component_dict, list(component_dict), spec_dict["anchor"], spec_dict["end"], scenario_str)
            nav_series, weight_df, contribution_df, cost_series = ledger.simulate_book(panel_df, configuration_info_dict["weights"], pd.Timestamp(spec_dict["anchor"]), "annual_fixed", .001)
            return_series = nav_series.pct_change(fill_method=None).iloc[1:]
            metric_dict = {**ledger.metrics_dict(return_series, panel_df.SPY, panel_df.BIL), **horizon_metrics(return_series)}
            passed_bool, failure_list = risk_gate(metric_dict, main_spec_dict["promotion_rule"]["A"])
            result_list.append({"candidate": candidate_str, "scenario": scenario_str, "risk_pass": passed_bool, "failures": ";".join(failure_list), **metric_dict})
            if candidate_str in {"A3", "M0_matched"} and scenario_str == "common_account":
                return_dict[candidate_str] = return_series
    result_df = pd.DataFrame(result_list)
    verdict_list = []
    for scenario_str in ["common_account", "conservative"]:
        active_series = result_df.loc[(result_df.candidate == "A3")&(result_df.scenario == scenario_str)].iloc[0]
        monthly_series = result_df.loc[(result_df.candidate == "M0_matched")&(result_df.scenario == scenario_str)].iloc[0]
        benefit_bool = bool(active_series.max_drawdown-monthly_series.max_drawdown >= .02 or active_series.es5_loss <= .9*monthly_series.es5_loss)
        carry_bool = bool(active_series.cagr >= monthly_series.cagr-.015)
        neighbor_bool = bool(result_df.loc[result_df.candidate.str.startswith("A3_pair")&(result_df.scenario == scenario_str), "risk_pass"].all())
        verdict_list.append({"scenario": scenario_str, "risk_pass": bool(active_series.risk_pass), "downside_benefit_pass": benefit_bool, "carry_pass": carry_bool, "all_neighbor_risk_pass": neighbor_bool})
    accepted_bool = all(all(value_bool for key_str, value_bool in row_dict.items() if key_str != "scenario") for row_dict in verdict_list)
    result_df.to_csv(STUDY_PATH/"tables/active_alternative.csv", index=False)
    pd.DataFrame(return_dict).to_csv(STUDY_PATH/"data/active_alternative_returns.csv.gz")
    write_json(STUDY_PATH/"active_alternative_verdict.json", {"accepted_provisionally": accepted_bool, "scenario_checks": verdict_list, "cells": len(result_list), "post_result": True, "history_quality": "only2018-07-20 onward; no fabricatedpre-inceptionhistory; selection history alreadyseen", "configurations": configuration_dict})
    print(json.dumps({"accepted_provisionally": accepted_bool, "checks": verdict_list, "cells": len(result_list)}))
    print(result_df.loc[result_df.candidate.isin(["A3", "M0_matched"]), ["candidate", "scenario", "cagr", "max_drawdown", "es5_loss", "worst_5y"]].to_string(index=False))


if __name__ == "__main__":
    main()
