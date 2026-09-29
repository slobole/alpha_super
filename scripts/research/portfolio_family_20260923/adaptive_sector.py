"""Final bounded, post-result H10 sector-rebound substitution diagnostic.

Six fixed definitions, three accounting scenarios, annual outer rebalancing.
Seen-history exploration only; matched BIL controls distinguish dilution from
the incremental value of retaining a sector-rebound risk budget.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.portfolio_family_20260923 import analyze
from scripts.research.portfolio_family_20260923.freeze import STUDY_PATH, sha256_str


HPI_ID_STR = "strategy_mr_hpi_sp500_2_3_5_vote"
SECTOR_ID_STR = "strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"
LEVEL_LIST = ["075", "050", "025"]


def freeze_specification() -> dict:
    spec_path = STUDY_PATH / "adaptive_sector_spec.json"
    if spec_path.exists():
        spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
        verify_inputs(spec_dict)
        return spec_dict
    original_path = STUDY_PATH / "research_spec_frozen.json"
    original_dict = json.loads(original_path.read_text(encoding="utf-8"))
    original_candidate_dict = {candidate_dict["candidate_id"]: candidate_dict for candidate_dict in original_dict["portfolio"]["candidates"]}
    candidate_list, reference_list, test_list = [], [], []
    for level_str in LEVEL_LIST:
        original_id_str = "CORE_" + level_str
        reference_dict = original_candidate_dict[original_id_str]
        reference_list.append(reference_dict)
        for replacement_str, replacement_source_str in (("SECTOR", SECTOR_ID_STR), ("BIL", "benchmark_bil")):
            weight_dict = dict(reference_dict["weights"])
            replaced_float = weight_dict.pop(HPI_ID_STR)
            weight_dict[replacement_source_str] = replaced_float
            candidate_list.append({"candidate_id": f"H10_{replacement_str}_{level_str}",
                "original_id": original_id_str, "replacement": replacement_str,
                "replaced_hpi_budget": replaced_float, "weights": weight_dict})
        for comparator_str in (original_id_str, f"H10_BIL_{level_str}"):
            test_list.append({"candidate": f"H10_SECTOR_{level_str}", "comparator": comparator_str,
                              "test_id": f"H10_SECTOR_{level_str}_vs_{comparator_str}"})
    dependency_path_list = [original_path, STUDY_PATH / "source_manifest.json",
        STUDY_PATH / "source_audit_addendum.json", STUDY_PATH / "pre_result_amendment_01.json",
        STUDY_PATH / "benchmarks/benchmark_manifest.json", STUDY_PATH / "catalog_rules.json",
        STUDY_PATH / "tables/portfolio_metrics.csv", STUDY_PATH / "tables/all25_common_metrics.csv",
        STUDY_PATH / "verification/eligible_universe_decisions.json",
        STUDY_PATH / "verification/eligible_universe_decisions.md",
        Path(__file__), Path(analyze.__file__), Path(analyze.__file__).with_name("freeze.py")]
    spec_dict = {
        "schema_version": "adaptive-sector-substitution-v1", "hypothesis_id": "H10",
        "amendment_label": "H5", "amendment_label_note": "Original registry H5 remains momentum_split. H10 is the unique identifier for this new follow-up.",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_composed_returns_after_original_results_seen",
        "research_only": True, "seen_history": "Selected after inspecting the original portfolio and all25 results. All historical periods and strategy development are seen; no independent validation or causal alpha proof is created.",
        "mechanism": "Replace only the HPI stock-rebound budget with a fixed11-sector-ETF rebound strategy to reduce single-stock specificity and potentially competing stock orders, while preserving the rest of the macro/momentum/DV2 architecture. Reduced order competition is a mechanism hypothesis, not verified fill or execution evidence.",
        "matched_control": "Replace the identical HPI budget with BIL fund units. If risk reduction can be explained by BIL dilution without sufficient incremental sector reward, do not label the sector replacement added value.",
        "source_unchanged": True, "candidate_list": candidate_list, "reference_list": reference_list,
        "anchor": "2012-10-01", "end": "2026-07-31",
        "partition_list": original_dict["evaluation"]["partitions"],
        "scenario_list": ["native", "common_account", "conservative"],
        "rebalance": "annual_fixed", "outer_cost": .001,
        "timing": "Use existing source_components, strict_return_panel and simulate_book unchanged. Source Close_T to next-open contracts remain native. Synthetic annual resets use prior-close NAV before current returns;10bps initial fee and10bps per absolute outer-weight change. No physical inception or broker-transfer replay.",
        "calendar": "Exact primary anchor/end and identical internal sessions required; no filling or deletion of missing return rows. Final prepaid carry treatment follows the frozen analysis helper.",
        "gate_vs_original": {"minimum_cagr_difference": -.005, "maximum_es5_ratio": .95, "minimum_signed_max_drawdown_difference": 0.},
        "gate_vs_bil": {"minimum_cagr_difference": .005, "maximum_es5_ratio": 1.10, "minimum_signed_max_drawdown_difference": -.01},
        "gate_rule": "All three inequalities jointly pass on the full sample and in at least2of3 original partitions, separately in common_account and conservative. Both comparison gates must pass in both scenarios to retain a sector substitution as a forward research candidate. Native is contextual. No gate is relaxed after results. Floating comparison tolerance1e-12 only.",
        "bootstrap": {"tests": test_list, "seed": 20260923, "replicates": 2000,
            "block_sessions": 63, "scenario": "common_account", "method": "Existing synchronized circular moving-block paired mean bootstrap; one-sided centered p with plus-one correction;95% percentile interval for252*mean daily difference.",
            "family_correction": "Holm across only these6 exploratory comparisons. Separate from original19; neither family corrects prior strategy/portfolio search or post-result hypothesis selection.",
            "role": "Contextual paired-mean evidence, not a substitute for the joint economic/risk gates; no untouched validation."},
        "budget": {"prior_distinct_cells": 300, "new_definitions": 6, "new_cost_rebalance_cells": 18,
            "total_distinct_cells": 318, "original_cap": 320, "reference_recomputations": "Nine existing original candidate/scenario cells replayed only for parity and matched comparisons; no new definition or choice."},
        "stop_rule": "This is the last scientific follow-up. Exactly six fixed definitions times three scenarios, no alternative sector baskets, no other HPI/DV2 budgets, no fine weights, no additional adaptive branches after success or failure. Failed gates remain reported tradeoffs. Then stop research under318cells.",
        "inputs": [{"path_str": str(file_path.resolve()), "sha256_str": sha256_str(file_path)} for file_path in dependency_path_list],
        "outputs": ["tables/adaptive_sector_metrics.csv", "tables/adaptive_sector_subperiod_metrics.csv", "tables/adaptive_sector_primary_returns.csv.gz", "tables/adaptive_sector_gate_detail.csv", "tables/adaptive_sector_gates.csv", "tables/adaptive_sector_decisions.csv", "tables/adaptive_sector_paired_bootstrap.csv", "adaptive_sector_manifest.json", "verification/eligible_universe_decisions_h10_addendum.json", "verification/eligible_universe_decisions_h10_addendum.md"],
    }
    spec_path.write_text(json.dumps(spec_dict, indent=2) + "\n", encoding="utf-8")
    return spec_dict


def verify_inputs(spec_dict: dict) -> None:
    for input_dict in spec_dict["inputs"]:
        analyze.verify_file(input_dict)


def gate_components(candidate_dict: dict, comparator_dict: dict, threshold_dict: dict) -> dict:
    cagr_delta_float = candidate_dict["cagr"] - comparator_dict["cagr"]
    es_delta_float = candidate_dict["es5_loss"] - threshold_dict["maximum_es5_ratio"] * comparator_dict["es5_loss"]
    drawdown_delta_float = candidate_dict["max_drawdown"] - comparator_dict["max_drawdown"]
    return {"cagr_difference": cagr_delta_float,
        "es5_loss_ratio": candidate_dict["es5_loss"] / comparator_dict["es5_loss"] if comparator_dict["es5_loss"] != 0 else np.nan,
        "signed_drawdown_difference": drawdown_delta_float,
        "cagr_pass": bool(cagr_delta_float >= threshold_dict["minimum_cagr_difference"] - 1e-12),
        "es5_pass": bool(es_delta_float <= 1e-12),
        "drawdown_pass": bool(drawdown_delta_float >= threshold_dict["minimum_signed_max_drawdown_difference"] - 1e-12)}


def validate_gate_contract() -> None:
    """Synthetic boundary checks, independent of actual portfolio outcomes."""
    original_dict = {"cagr": .10, "es5_loss": .02, "max_drawdown": -.10}
    threshold_dict = {"minimum_cagr_difference": -.005, "maximum_es5_ratio": .95, "minimum_signed_max_drawdown_difference": 0.}
    boundary_dict = {"cagr": .095, "es5_loss": .019, "max_drawdown": -.10}
    passed_dict = gate_components(boundary_dict, original_dict, threshold_dict)
    assert all(passed_dict[field_str] for field_str in ("cagr_pass", "es5_pass", "drawdown_pass"))
    for field_str, delta_float, gate_str in (("cagr", -.000001, "cagr_pass"), ("es5_loss", .000001, "es5_pass"), ("max_drawdown", -.000001, "drawdown_pass")):
        failure_dict = dict(boundary_dict)
        failure_dict[field_str] += delta_float
        assert not gate_components(failure_dict, original_dict, threshold_dict)[gate_str]
    bil_threshold_dict = {"minimum_cagr_difference": .005, "maximum_es5_ratio": 1.1, "minimum_signed_max_drawdown_difference": -.01}
    bil_boundary_dict = {"cagr": .105, "es5_loss": .022, "max_drawdown": -.11}
    assert all(gate_components(bil_boundary_dict, original_dict, bil_threshold_dict)[field_str] for field_str in ("cagr_pass", "es5_pass", "drawdown_pass"))


def run_study(spec_dict: dict) -> None:
    manifest_path = STUDY_PATH / "adaptive_sector_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Preserve the completed adaptive diagnostic")
    validate_gate_contract()
    verify_inputs(spec_dict)
    original_dict = json.loads((STUDY_PATH / "research_spec_frozen.json").read_text(encoding="utf-8"))
    addendum_dict = json.loads((STUDY_PATH / "source_audit_addendum.json").read_text(encoding="utf-8"))
    analyze.preflight_source_ids(original_dict, addendum_dict)
    candidate_list = spec_dict["candidate_list"] + spec_dict["reference_list"]
    source_list = sorted(set().union(*(set(candidate_dict["weights"]) for candidate_dict in candidate_list)) | {"benchmark_spy", "benchmark_bil"})
    component_dict = {}
    for source_str in source_list:
        parent_str = "benchmarks" if source_str.startswith("benchmark_") else "data"
        component_dict[source_str], _ = analyze.source_components(STUDY_PATH / parent_str / source_str)
    original_metric_df = pd.read_csv(STUDY_PATH / "tables/portfolio_metrics.csv")
    original_metric_df = original_metric_df.loc[original_metric_df.rebalance.eq("annual_fixed")]
    new_id_set = {candidate_dict["candidate_id"] for candidate_dict in spec_dict["candidate_list"]}
    metric_list, period_list, parity_list = [], [], []
    primary_return_df = pd.DataFrame()
    metric_lookup_dict = {}
    for scenario_str in spec_dict["scenario_list"]:
        return_panel_df = analyze.strict_return_panel(component_dict, source_list, spec_dict["anchor"], spec_dict["end"], scenario_str)
        for candidate_dict in candidate_list:
            candidate_str = candidate_dict["candidate_id"]
            nav_series, _, _, _ = analyze.simulate_book(return_panel_df, candidate_dict["weights"],
                pd.Timestamp(spec_dict["anchor"]), spec_dict["rebalance"], spec_dict["outer_cost"])
            book_return_series = pd.Series(nav_series.to_numpy()[1:] / nav_series.to_numpy()[:-1] - 1, index=return_panel_df.index)
            identity_dict = {"hypothesis_id": "H10", "candidate_id": candidate_str, "scenario": scenario_str,
                "rebalance": spec_dict["rebalance"], "new_definition": candidate_str in new_id_set}
            full_dict = analyze.metrics_dict(book_return_series, return_panel_df.benchmark_spy, return_panel_df.benchmark_bil)
            metric_list.append({**identity_dict, **full_dict})
            metric_lookup_dict[(candidate_str, scenario_str, "full")] = full_dict
            for period_int, (start_str, end_str) in enumerate(spec_dict["partition_list"], start=1):
                period_series = book_return_series.loc[start_str:end_str]
                period_dict = analyze.metrics_dict(period_series, return_panel_df.loc[period_series.index, "benchmark_spy"], return_panel_df.loc[period_series.index, "benchmark_bil"])
                period_list.append({**identity_dict, "period": period_int, **period_dict})
                metric_lookup_dict[(candidate_str, scenario_str, f"partition_{period_int}")] = period_dict
            if candidate_str not in new_id_set:
                saved_dict = original_metric_df.loc[original_metric_df.candidate_id.eq(candidate_str) & original_metric_df.scenario.eq(scenario_str)].iloc[0].to_dict()
                for field_str in ("cagr", "volatility", "max_drawdown", "es5_loss", "total_return"):
                    if not np.isclose(full_dict[field_str], saved_dict[field_str], rtol=1e-11, atol=1e-12):
                        raise AssertionError(f"Original reference parity failed: {candidate_str}/{scenario_str}/{field_str}")
                parity_list.append({"candidate_id": candidate_str, "scenario": scenario_str, "passed": True})
            if scenario_str == "common_account":
                primary_return_df[candidate_str] = book_return_series
    detail_list, gate_list, decision_list = [], [], []
    for level_str in LEVEL_LIST:
        candidate_str = f"H10_SECTOR_{level_str}"
        for comparison_str, comparator_str in (("original", "CORE_" + level_str), ("bil", f"H10_BIL_{level_str}")):
            for scenario_str in ("common_account", "conservative"):
                period_pass_dict = {}
                for period_str in ("full", "partition_1", "partition_2", "partition_3"):
                    result_dict = gate_components(metric_lookup_dict[(candidate_str, scenario_str, period_str)],
                        metric_lookup_dict[(comparator_str, scenario_str, period_str)], spec_dict[f"gate_vs_{comparison_str}"])
                    joint_bool = all(result_dict[field_str] for field_str in ("cagr_pass", "es5_pass", "drawdown_pass"))
                    period_pass_dict[period_str] = joint_bool
                    detail_list.append({"candidate_id": candidate_str, "comparator": comparator_str,
                        "comparison": comparison_str, "scenario": scenario_str, "period": period_str,
                        **result_dict, "joint_pass": joint_bool})
                partition_count_int = sum(period_pass_dict[f"partition_{period_int}"] for period_int in (1, 2, 3))
                gate_list.append({"candidate_id": candidate_str, "comparator": comparator_str,
                    "comparison": comparison_str, "scenario": scenario_str, "full_pass": period_pass_dict["full"],
                    "partitions_passed": partition_count_int, "pass": period_pass_dict["full"] and partition_count_int >= 2})
        applicable_list = [gate_dict for gate_dict in gate_list if gate_dict["candidate_id"] == candidate_str]
        retain_bool = all(gate_dict["pass"] for gate_dict in applicable_list)
        decision_list.append({"candidate_id": candidate_str, "original_id": "CORE_" + level_str,
            "retain_forward_candidate": retain_bool, "passed_comparison_scenario_gates": sum(gate_dict["pass"] for gate_dict in applicable_list),
            "required_comparison_scenario_gates": 4,
            "disposition": "forward_research_candidate_seen_history_only" if retain_bool else "tradeoff_does_not_clear_all_frozen_gates",
            "stop": "No further variants; original definitions preserved; no client allocation authority."})
    bootstrap_dict = spec_dict["bootstrap"]
    bootstrap_df = analyze.paired_bootstrap(primary_return_df, bootstrap_dict["tests"], bootstrap_dict["seed"], bootstrap_dict["replicates"], bootstrap_dict["block_sessions"])
    bootstrap_df["family"] = "H10_exploratory_six_separate_from_original19"
    output_list = []

    def save_frame(name_str: str, frame_df: pd.DataFrame, index_bool: bool = False) -> None:
        file_path = STUDY_PATH / "tables" / f"adaptive_sector_{name_str}"
        compression_dict = {"method": "gzip", "mtime": 0} if name_str.endswith(".gz") else None
        frame_df.to_csv(file_path, index=index_bool, index_label="date" if index_bool else None,
            compression=compression_dict, float_format="%.17g")
        output_list.append({"path_str": str(file_path), "sha256_str": sha256_str(file_path), "rows": len(frame_df)})

    save_frame("metrics.csv", pd.DataFrame(metric_list))
    save_frame("subperiod_metrics.csv", pd.DataFrame(period_list))
    save_frame("primary_returns.csv.gz", primary_return_df, True)
    save_frame("gate_detail.csv", pd.DataFrame(detail_list))
    save_frame("gates.csv", pd.DataFrame(gate_list))
    save_frame("decisions.csv", pd.DataFrame(decision_list))
    save_frame("paired_bootstrap.csv", bootstrap_df)
    verify_inputs(spec_dict)
    completed_str = datetime.now(timezone.utc).isoformat()
    manifest_dict = {"hypothesis_id": "H10", "status": "complete_seen_history_exploratory_diagnostic",
        "completed_at": completed_str, "spec_path": str(STUDY_PATH / "adaptive_sector_spec.json"),
        "spec_sha256": sha256_str(STUDY_PATH / "adaptive_sector_spec.json"),
        "new_definitions": 6, "new_cells": 18, "total_distinct_cells": 318, "budget_cap": 320,
        "primary_return_sessions": len(primary_return_df), "original_reference_parity": parity_list,
        "synthetic_gate_boundary_checks_passed": True, "all_input_hashes_unchanged": True,
        "outputs": output_list, "decisions": decision_list,
        "inference_limit": bootstrap_dict["family_correction"],
        "stop_rule": spec_dict["stop_rule"]}
    manifest_path.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")
    role_addendum_dict = {"status": "additive_post_result_role_audit_update", "hypothesis_id": "H10",
        "created_at": completed_str, "original_role_audit_preserved": True,
        "original_json_sha256": sha256_str(STUDY_PATH / "verification/eligible_universe_decisions.json"),
        "original_markdown_sha256": sha256_str(STUDY_PATH / "verification/eligible_universe_decisions.md"),
        "adaptive_spec_sha256": manifest_dict["spec_sha256"], "adaptive_manifest_sha256": sha256_str(manifest_path),
        "affected_strategy_id": SECTOR_ID_STR, "compared_replaced_strategy_id": HPI_ID_STR,
        "role_update": "The previously untested primary replacement is now tested only at the three frozen HPI budgets against both original HPI and matched BIL; all other sector alternatives and budgets remain untested. The source strategy remains a sector-rebound alternative regardless of these specific gate outcomes.",
        "decisions": decision_list, "claim_limit": "Additional seen-history evidence; no independent validation, no source promotion or allocation change."}
    addendum_path = STUDY_PATH / "verification/eligible_universe_decisions_h10_addendum.json"
    addendum_markdown_path = addendum_path.with_suffix(".md")
    if addendum_path.exists() or addendum_markdown_path.exists():
        raise FileExistsError("Preserve any existing role audit amendment")
    addendum_path.write_text(json.dumps(role_addendum_dict, indent=2) + "\n", encoding="utf-8")
    line_list = ["# H10 additive eligible-universe role audit", "", "The original role audit is preserved. This amendment records the final post-result sector-rebound substitution test on already-seen history; it adds no allocation authority.", "", role_addendum_dict["role_update"], "", "| Substitution | All required gates passed | Disposition |", "|---|---|---|"]
    line_list.extend(f"| {decision_dict['candidate_id']} | {decision_dict['retain_forward_candidate']} | {decision_dict['disposition']} |" for decision_dict in decision_list)
    line_list.extend(["", "The matched BIL control is essential: reducing an equity-rebound budget can lower risk without demonstrating value from its replacement. Both the original-HPI and matched-BIL gates are required. Failed tests stay tradeoffs; no further basket, budget or weight search follows.", "", "Six Holm-adjusted paired-mean diagnostics form their own exploratory family. They neither amend the original19-test family nor correct prior research or the post-result choice of this hypothesis.", "", f"Original JSON SHA-256: `{role_addendum_dict['original_json_sha256']}`.", f"Adaptive specification SHA-256: `{manifest_dict['spec_sha256']}`."])
    addendum_markdown_path.write_text("\n".join(line_list) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_str(manifest_path),
        "completed_at": completed_str, "new_cells": 18, "total_cells": 318, "decisions": decision_list}))


def main() -> None:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--freeze", action="store_true", help="Write/verify specification without loading source return paths")
    argument_obj = parser_obj.parse_args()
    validate_gate_contract()
    spec_dict = freeze_specification()
    if argument_obj.freeze:
        print(json.dumps({"spec": str(STUDY_PATH / "adaptive_sector_spec.json"),
            "sha256": sha256_str(STUDY_PATH / "adaptive_sector_spec.json"), "frozen_at": spec_dict["frozen_at"],
            "hypothesis_id": "H10", "new_definitions": 6, "new_cells": 18}))
        return
    run_study(spec_dict)


if __name__ == "__main__":
    main()
