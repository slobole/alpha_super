"""Reconcile frozen portfolio-study lineage and execute saved-result notebook only.

No strategy, price, account, or research experiment is executed by this module.
The immutable original protocol remains primary; schema adaptation is additive.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_PATH = Path(__file__).resolve().parents[3]
STUDY_PATH = REPO_PATH / "results/research/portfolio_family_20260923"
SKILL_PATH = Path.home() / ".codex/skills/research-quant-signal-features/scripts"
ORIGINAL_SHA256_STR = "54a7e749658dc695ef316742d78c73c06f5bf582d8e9ef32a4623e435f285918"
SEEN_PERIOD_LIST = ["All source history and prior portfolio research already seen; primary 2012-10-02..2026-07-31; all-25 2019-04-05..2026-07-31; long history diagnostic only."]
BUDGET_DICT = {
    "target_active_minutes": 90, "hard_cap_active_minutes": 180,
    "max_adaptive_rounds": 2, "max_new_hypotheses_per_round": 5,
    "max_total_variants": 320, "max_parallel_lanes": 3,
}


def read_json(json_path: Path):
    return json.loads(json_path.read_text(encoding="utf-8"))


def file_hash(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def write_json(json_path: Path, value_obj) -> None:
    json_path.write_text(json.dumps(value_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def skill_module(module_name_str: str):
    module_spec = importlib.util.spec_from_file_location(module_name_str, SKILL_PATH / (module_name_str + ".py"))
    module_obj = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module_obj)
    return module_obj


def merge_catalog() -> list[dict]:
    catalog_list = read_json(STUDY_PATH / "catalog_rules.json")
    defensive_list = read_json(STUDY_PATH / "defensive_rules.json")
    defensive_by_import_dict = {row_dict["import"]: row_dict for row_dict in defensive_list}
    merged_list = []
    matched_set = set()
    for source_dict in catalog_list:
        merged_dict = copy.deepcopy(source_dict)
        import_str = source_dict["strategy_import"].split(":")[0]
        if import_str in defensive_by_import_dict:
            detail_dict = defensive_by_import_dict[import_str]
            matched_set.add(import_str)
            merged_dict.update({
                "analysis_alias": detail_dict["alias"],
                "friendly_name_he": detail_dict["friendly_name_he"],
                "audit_status": "source_rules_verified",
                "family": detail_dict["family"], "mechanism": detail_dict["mechanism"],
                "cadence": detail_dict["actual_order_cadence"],
                "signal_cadence": detail_dict["signal_cadence"],
                "strict_month_end_only": detail_dict["strict_month_end_only"],
                "universe": {"literal_definition": detail_dict["rules_formulas"][0]},
                "rules": {"literal_formulas": detail_dict["rules_formulas"]},
                "target_exposure": detail_dict["exposure"],
                "execution_timing": detail_dict["decision_and_fill"],
                "costs": detail_dict["costs"],
                "cash_policy": {"source_native": detail_dict["exposure"],
                                "common_tax_override": detail_dict["tax_override"]},
                "caveats": detail_dict["caveats"] + detail_dict["failure_scenarios"],
                "evidence": [{"path": evidence_str.rsplit(":", 1)[0],
                              "line": int(evidence_str.rsplit(":", 1)[1])}
                             for evidence_str in detail_dict["evidence_paths"]],
                "data_history": detail_dict["data_history"],
                "prior_study_provenance": detail_dict["prior_study_provenance"],
                "verification_status": detail_dict["verification_status"],
                "tier": detail_dict["maturity"],
            })
        merged_list.append(merged_dict)
    assert len(merged_list) == 25
    assert len({row_dict["strategy_import"] for row_dict in merged_list}) == 25
    assert matched_set == set(defensive_by_import_dict)
    assert all(row_dict["audit_status"] == "source_rules_verified" for row_dict in merged_list)
    assert sum(row_dict["audit_status"] == "source_rules_verified" for row_dict in catalog_list) == 19
    write_json(STUDY_PATH / "catalog_complete.json", merged_list)
    return merged_list


def effective_spec(original_dict: dict, now_str: str) -> dict:
    effective_dict = copy.deepcopy(original_dict)
    amendment_dict = read_json(STUDY_PATH / "pre_result_amendment_01.json")
    native_spec_dict = read_json(STUDY_PATH / "native_replay/spec.json")
    effective_dict.update({
        "schema_adapted_at": now_str, "scientific_change_bool": False,
        "original_protocol": {"path": "research_spec_frozen.json", "sha256": ORIGINAL_SHA256_STR,
                              "primary_authority_bool": True},
        "adaptation_note": "Additive schema normalization after results. Original freeze time is retained as protocol provenance, not this file's creation time. No rule, search space, gate, period, cost, or outcome is changed.",
        "amendment_links": [
            {"path": "pre_result_amendment_01.json", "sha256": file_hash(STUDY_PATH / "pre_result_amendment_01.json"),
             "amended_at": amendment_dict["amended_at"]},
            {"path": "native_replay/spec.json", "sha256": file_hash(STUDY_PATH / "native_replay/spec.json"),
             "frozen_at": native_spec_dict["frozen_at_utc_str"]}],
    })
    effective_dict["sources"] = [
        {"source_id": "source_" + str(index_int), "location": row_dict["path"],
         "role": row_dict["role"], "read_complete": row_dict["complete_read"]}
        for index_int, row_dict in enumerate(original_dict["sources"], 1)]
    data_dict = effective_dict["data"]
    data_dict.update({
        "vendor": "Native saved Norgate and source-specific macro inputs; heterogeneous source vintages remain disclosed.",
        "period": "Primary close anchor2012-10-01 through2026-07-31; all25 anchor2019-04-04; exact native long history diagnostics.",
        "as_of": "Source-specific saved vintages in source_manifest.json; archived benchmark panel through2026-09-11; study end2026-07-31.",
        "input_artifacts": [
            {"location": "source_manifest.json", "content_id": "sha256:" + data_dict["source_manifest_sha256"],
             "as_of": "Pre-result source selection; consult source-specific metadata"},
            {"location": data_dict["benchmark_input_path"], "content_id": "sha256:" + data_dict["benchmark_input_sha256"],
             "as_of": "Archived2026-09-15 qualification snapshot; observations through2026-09-11"}],
        "universes": ["25 PM_READY or WIRED sources frozen from alpha/strategy_registry.py; native stock PIT memberships and observed ETF coverage"],
        "benchmarks": ["BIL_net25", "SPY_net25", "$SPXTR gross total-return index companion"],
    })
    effective_dict["timing"]["decision"] = original_dict["timing"]["native_signals"] + "; " + original_dict["timing"]["allocation"]
    effective_dict["timing"]["terminal_value"] = original_dict["timing"]["exit"]
    effective_dict["signal"]["thresholds"] = ["No new source signal thresholds; source rules fixed. Candidate weights and decision gates are literal original protocol."]
    effective_dict["feature_roles"] = [
        {"feature": key_str, "roles": [value_str]}
        for key_str, value_str in original_dict["feature_roles"].items()]
    effective_dict["portfolio"].update({
        "engine": "Saved native standalone NAV components; synthetic sleeve-unit annual-fixed or drift construction; separate exact CORE5 stateful replay.",
        "maximum_positions": "Native source limits unchanged; portfolio candidate weights literal and sum to one.",
        "ranking": "No fitted ranking or full-sample optimizer; preselected families and economic roles.",
        "sizing": original_dict["portfolio"]["formula"],
        "cash": original_dict["portfolio"]["idle_cash"],
        "ensemble": "36 fixed portfolio definitions; two outer rebalance modes; three accounting scenarios.",
    })
    effective_dict["evaluation"].update({
        "periods": {
            "discovery": "All historical diagnostics are reused/seen; primary2012-10-02..2026-07-31, chronological partitions retained.",
            "validation": "No untouched historical validation; partitions and bootstrap are diagnostics only.",
            "confirmation": "No untouched historical confirmation; no forward or live evidence.",
            "full": "Primary2012-10-02..2026-07-31; all25 common2019-04-05..2026-07-31; native CORE5 first fill2012-10-01 with2012-09-28 cash anchor.",
        },
        "benchmark": original_dict["data"]["benchmark"],
    })
    costs_dict = effective_dict["costs"]
    costs_dict.update({
        "schema_cost_basis": "round_trip_bps is incremental slippage above heterogeneous source-native costs, not total trading cost. paper_like is the schema name for saved-native arm; no cost-free simulation was performed.",
        "paper_like": {"round_trip_bps": 0, "included_components": ["Saved source-native slippage/commission", "Native dividend withholding and borrow; heterogeneous and disclosed"], "source_scenario": "native"},
        "central_research": {"round_trip_bps": 0, "included_components": ["Saved native trading/borrow costs", "25% long dividend withholding", "Fixed5% collateral-adjusted funding scenario", "Declared10bps outer absolute-weight turnover cost"], "source_scenario": "common_account"},
        "conservative_survival": {"round_trip_bps": 20, "included_components": ["Central native trading costs and25% withholding", "Fixed8% funding scenario", "10bps incremental slippage each traded side", "Incremental borrow to5% target without double debit"], "source_scenario": "conservative"},
        "components": ["native slippage", "commission", "dividend withholding", "securities borrow", "separate financing", "outer sleeve-unit turnover"],
        "capacity_impact": {"separate_from_base": True, "calibrated": False,
                            "formula": "No calibrated ADV/auction/participation impact function; fixed-slippage stress is not capacity evidence."},
    })
    family_list = sorted({row_dict["family"] for row_dict in original_dict["portfolio"]["candidates"]})
    effective_dict["search_space"].update({
        "declared_families": family_list + ["standalone_diagnostics", "benchmark_controls", "core5_native_replay"],
        "axes": {"compositions": 36, "outer_rebalance_modes": 2, "accounting_scenarios": 3,
                 "standalone_sources": 25, "standalone_cost_scenarios": 3, "benchmarks": 2, "native_replay_and_parity_max": 7},
        "total_declared_variants": 320, "actual_evaluation_cells": 300,
        "count_semantics": "Budget320;216 composition cells+75 standalone cells+2 benchmarks+7 native/parity runs=300. Cost/rebalance cells are not independent discoveries; prior studies not summed as independent trials.",
    })
    effective_dict["promotion_rule"].update({
        "economic": original_dict["promotion_rule"]["satellite"] + " " + original_dict["promotion_rule"]["core_vs_bil"],
        "statistical": "Predeclared19 paired tests with Holm adjustment; no correction for all prior source search, no untouched holdout; at most forward hypothesis.",
        "risk": original_dict["promotion_rule"]["family"],
    })
    effective_dict["outputs"] = {
        "concise_report": "REPORT.md", "full_report": "REPORT_FULL.md",
        "notebook": "decision_notebook.ipynb", "knowledge_record": "knowledge_record.json",
        "manifest": "run_manifest.json", "tables": "tables", "charts": "charts",
        "html_report": "REPORT.html",
    }
    effective_dict["evidence_waivers"] = [
        {"layer": key_str, "reason": value_str, "promotion_effect": "Diagnostic/forward-hypothesis ceiling; no allocation or deployment approval."}
        for key_str, value_str in original_dict["evidence_waivers"].items()]
    effective_dict["adaptive_workflow"] = {
        "schema_version": "quant-research-workflow-v1", "profile": "standard",
        "state": "research_state.json", "hypothesis_registry": "hypothesis_registry.json",
        "experiment_ledger": "experiment_ledger.jsonl", "decision_log": "decision_log.jsonl",
        "source_rule_map": "SOURCE_RULE_MAP.md", "runtime_budget": BUDGET_DICT,
        "holdout_policy": original_dict["evaluation"]["holdout"],
        "post_result_policy": original_dict["adaptive_workflow"]["post_result_rule"],
        "original_budget_override_reason": original_dict["adaptive_workflow"]["budget_override_reason"],
    }
    write_json(STUDY_PATH / "research_spec_effective.json", effective_dict)
    return effective_dict


def source_rule_map(catalog_list: list[dict], now_str: str) -> None:
    text_list = [
        "# Portfolio-family source rule map", "",
        "This assembled map was written at " + now_str + ". The scientific protocol remains the immutable research_spec_frozen.json; the additive research_spec_effective.json only normalizes its schema.",
        "", "## Source and evidence boundary", "",
        "The authoritative merged catalog contains25 current PM_READY/WIRED sources:19 inventory mappings plus6 defensive mappings. Each entry below preserves literal current rules, timing, exposure, costs and evidence. Maturity is infrastructure compatibility, not client suitability.",
        "Saved historical source code at original artifact creation is unverified. Accordingly original-source baseline reproduction is not_reproducible; this is a provenance limitation, not a claim that the arithmetic failed. Separately, the seventh CORE5 native-zero run replicated current local engine NAV/cash/order economics/borrow/weights exactly against the same archived panel.",
        "", "## Timing and accounting", "",
        "```text\n[Native Close_T information] -> [Source-native decisions and quantities]\n        -> [Open_(T+1) fills; Flow uses its declared MOC schedule]\n        -> [Native fills/dividends/borrow/close NAV]\n        -> [Saved-unit composition; prior-close weights only]\n        -> [Fixed-holdings cost sensitivities; diagnostic gates]\nCORE5 supplement: native close accounting -> stateful funding -> future sizing\n*** CRITICAL *** no current/future session performance determines prior weights.\n```",
        "Synthetic annual portfolio rebalancing is a close-to-close sleeve-unit approximation, not physical portfolio order replay. Initial unit allocation includes10bps; client inception fills are separately represented by CORE5's2012-09-28 cash anchor. Native CORE5 funding uses rounded102% short collateral less cash, ACT/360 to next known session, with no terminal interval and native borrow charged once.",
        "", "## Search provenance", "",
        "Current budget320;36 fixed compositions x2 rebalance modes x3 cost scenarios=216 cells;25 standalone x3=75;2 benchmark runs;6 native CORE5 cost/capital runs plus1 plain-source parity run=7. Total300 evaluation cells. No optimizer or post-result adaptive round.",
        "Prior Foundry reported64 portfolio tests; prior CORE5 reported441 nominal source paths plus subsequent studies. Their overlap is unknown. These are not independent-trial counts and must not be summed into a statistical correction. Prior reports and source outcomes were already exposed; all chronological/long-history/bootstrap checks are diagnostic, with no untouched validation/confirmation.",
        "Exact prior-study locations and tax overrides are preserved in defensive_rules.json and DEFENSIVE_AUDIT.md; the original protocol also records pre-result exposure.",
        "", "## Current source rules", "",
    ]
    for row_dict in catalog_list:
        text_list.extend([
            "### " + row_dict["alias"], "",
            "- Import: `" + row_dict["strategy_import"] + "`",
            "- Mechanism: " + row_dict["mechanism"],
            "- Signal/order cadence: " + row_dict.get("signal_cadence", str(row_dict["cadence"])) + " / " + str(row_dict["cadence"]),
            "- Decision/fill: " + row_dict["execution_timing"],
            "- Rules: " + json.dumps(row_dict["rules"], ensure_ascii=False),
            "- Exposure: " + json.dumps(row_dict["target_exposure"], ensure_ascii=False),
            "- Costs: " + json.dumps(row_dict["costs"], ensure_ascii=False),
            "- Cash/accounting: " + json.dumps(row_dict["cash_policy"], ensure_ascii=False),
            "- Limits: " + " ".join(row_dict["caveats"]),
            "- Evidence: " + "; ".join(str(evidence_dict["path"]) + ":" + str(evidence_dict.get("line", 1)) for evidence_dict in row_dict["evidence"]),
            "",
        ])
    (STUDY_PATH / "SOURCE_RULE_MAP.md").write_text("\n".join(text_list) + "\n", encoding="utf-8")


def reconcile_state(original_dict: dict, now_str: str) -> tuple[dict, list[dict]]:
    state_dict = read_json(STUDY_PATH / "research_state.json")
    registry_dict = read_json(STUDY_PATH / "hypothesis_registry.json")
    original_h0_dict = copy.deepcopy(registry_dict["hypotheses"][0])
    original_h0_dict = registry_dict.get("original_intake_h0_record", original_h0_dict)
    original_freeze_str = original_dict["initial_frozen_at"]
    native_spec_dict = read_json(STUDY_PATH / "native_replay/spec.json")
    replay_dict = read_json(STUDY_PATH / "native_replay/replay_manifest.json")
    elapsed_int = math.ceil((datetime.fromisoformat(now_str) - datetime.fromisoformat(state_dict["created_at"])).total_seconds() / 60)
    state_dict.update({
        "updated_at": now_str, "phase": "closeout", "evidence_phase": "diagnosis",
        "source": {"location": "alpha/strategy_registry.py", "content_id": "sha256:" + file_hash(REPO_PATH / "alpha/strategy_registry.py"), "read_complete": True},
        "runtime_budget": {**BUDGET_DICT, "active_minutes_used": elapsed_int,
                           "active_minutes_basis": "Conservative elapsed wall-clock ceiling since intake, not measured CPU time or sum of agent minutes.",
                           "budget_overrun_approved": False, "budget_stop_reason": "All frozen evaluations finished; only audit/report closeout remains."},
        "locks": {"source_rule_map_frozen_at": now_str, "literal_baseline_frozen_at": original_freeze_str,
                  "validation_locked_at": None, "confirmation_locked_at": None},
        "lock_provenance": {"source_rule_map": "Assembled after results from verified current source audits; timestamp is actual assembly, not a claim of pre-result file creation.",
                            "literal_baseline": "Timestamp references the immutable original scientific protocol; no historical event is backdated.",
                            "native_replay": native_spec_dict["frozen_at_utc_str"]},
        "baseline": {
            "status": "completed", "replication_outcome": "not_reproducible",
            "executable_translation": "Current source rules audited; saved historical-code-at-creation identity unverified.",
            "primary_evidence": ["source_manifest.json", "source_audit_addendum.json", "catalog_complete.json"],
            "reason": "Saved source return artifacts do not establish the exact historical code used at creation. Current implementation hashes do not repair that historical provenance.",
            "local_native_zero_parity": {"replication_outcome": "replicated", "scope": "Seventh current CORE5 engine run against same frozen panel; no funding. Full results, economic orders and relative ID sequence, borrow, weights and targets equal exactly.",
                                         "passed": replay_dict["native_zero_funding_exact_parity_bool"],
                                         "evidence": "native_replay/replay_manifest.json"},
        },
        "holdouts": {"validation_period": "None untouched; all historical partitions reused and diagnostic.",
                     "validation_opened_at": None, "confirmation_period": "None untouched; no prospective confirmation.",
                     "confirmation_opened_at": None},
        "adaptive_search": {"rounds_completed": 0, "actual_new_hypotheses_by_round": [],
                            "declared_total_variants": 320, "actual_total_variants": 300,
                            "count_semantics": "Evaluation cells, not independent discoveries:216+75+2+7.",
                            "stop_reason": "Frozen diagnostic program complete; no additional search. Report review and bundle finalization pending."},
        "prior_search": {"foundry_reported_tests": 64, "core5_nominal_paths": 441,
                         "overlap": "Unknown, including later studies; do not sum as independent trials.",
                         "history_seen": SEEN_PERIOD_LIST},
        "final_decision": {"disposition": "diagnostic", "research_status": "diagnostic",
                           "provisional": True, "verdict": "Diagnostic evidence only; final economic interpretation and report audit remain with the primary agent. Maximum protocol status is forward_hypothesis.",
                           "next_gate": "Complete report/notebook/source audit, knowledge record and bundle manifest; prospective and physical implementation gates remain separate."},
        "protocol_provenance": {"primary": "research_spec_frozen.json", "sha256": ORIGINAL_SHA256_STR,
                                "initial_frozen_at": original_freeze_str, "effective_schema": "research_spec_effective.json"},
    })
    state_dict["artifacts"].update({"frozen_specification": "research_spec_effective.json",
                                   "original_frozen_specification": "research_spec_frozen.json",
                                   "notebook": "decision_notebook.ipynb", "catalog": "catalog_complete.json"})
    write_json(STUDY_PATH / "research_state.json", state_dict)
    family_count_dict = Counter(row_dict["family"] for row_dict in original_dict["portfolio"]["candidates"])
    hypothesis_list = [{
        **original_h0_dict, "title": "Fixed native source artifacts and historical provenance",
        "family": "baseline", "role": "diagnostic", "economic_mechanism": "Native strategies retain their literal source exposures; accounting scenarios reveal comparability limits.",
        "expected_direction": "No imposed performance direction; preserve source accounting and report frictions.",
        "falsifier": "Missing inputs or code-at-creation prevent faithful historical reproduction; arithmetic/source inconsistencies invalidate comparisons.",
        "frozen_at": original_freeze_str, "protocol_frozen_at": original_freeze_str,
        "data_periods_seen": SEEN_PERIOD_LIST, "declared_variant_count": 75,
        "experiment_ids": ["E_STANDALONE_75"], "status": "diagnosed",
        "evidence_summary": "25 saved native sources x3 cost scenarios; historical code-at-creation unverified.",
        "disposition": "diagnostic",
    }]
    for family_index_int, (family_str, count_int) in enumerate(sorted(family_count_dict.items()), 1):
        candidate_list = [row_dict for row_dict in original_dict["portfolio"]["candidates"] if row_dict["family"] == family_str]
        hypothesis_list.append({
            "hypothesis_id": "H" + str(family_index_int), "title": "Predeclared " + family_str,
            "provenance": "predeclared", "family": family_str, "role": "portfolio_construction",
            "classification": "portfolio_construction",
            "economic_mechanism": " ".join(dict.fromkeys(row_dict["mechanism"] for row_dict in candidate_list)),
            "expected_direction": "Test the declared role and risk/cost tradeoff without choosing an optimum after results.",
            "falsifier": "Failure of the original family risk, satellite, comparator or stability gate; all evidence remains historical diagnostic.",
            "created_at": now_str, "frozen_at": now_str, "protocol_frozen_at": original_freeze_str,
            "registration_note": "Structured registry assembled after execution from immutable pre-result definitions; timestamps are actual writing times.",
            "candidate_ids": [row_dict["candidate_id"] for row_dict in candidate_list],
            "data_periods_seen": SEEN_PERIOD_LIST, "declared_variant_count": count_int * 6,
            "experiment_ids": ["E_PORTFOLIO_216"], "status": "diagnosed", "disposition": "diagnostic",
        })
    hypothesis_list.extend([
        {"hypothesis_id": "H8", "title": "After-tax BIL and SPY native benchmark controls",
         "provenance": "predeclared", "family": "benchmark_controls", "role": "diagnostic", "classification": "diagnostic",
         "economic_mechanism": "Passive Treasury-bill/equity exposure with actual dividend tax and whole-share cash reinvestment.",
         "expected_direction": "Provide common-calendar opportunity-cost comparators; no required favorable strategy outcome.",
         "falsifier": "Future-open sizing, omitted first fill, wrong tax/adjustment, or cash identity failure.",
         "created_at": now_str, "frozen_at": now_str, "protocol_frozen_at": original_freeze_str,
         "data_periods_seen": SEEN_PERIOD_LIST, "declared_variant_count": 2, "experiment_ids": ["E_BENCHMARK_2"],
         "status": "diagnosed", "disposition": "diagnostic"},
        {"hypothesis_id": "H9", "title": "Stateful CORE5 capital and financing sensitivity",
         "provenance": "predeclared", "family": "core5_native_replay", "role": "diagnostic", "classification": "diagnostic",
         "economic_mechanism": "Whole shares, minimum fees and stateful financing can alter future quantities at different account capitals.",
         "expected_direction": "Identify implementation sensitivity; unchanged source signals and exact zero-funding parity are required.",
         "falsifier": "Changed native rules, double borrow, funding after terminal date, or zero-funding parity failure.",
         "created_at": now_str, "frozen_at": now_str, "protocol_frozen_at": native_spec_dict["frozen_at_utc_str"],
         "data_periods_seen": SEEN_PERIOD_LIST, "declared_variant_count": 7, "experiment_ids": ["E_CORE_REPLAY_7"],
         "status": "diagnosed", "disposition": "diagnostic"},
    ])
    assert len(hypothesis_list) == 10
    assert sum(row_dict["declared_variant_count"] for row_dict in hypothesis_list) == 300
    write_json(STUDY_PATH / "hypothesis_registry.json", {
        "schema_version": "quant-research-hypotheses-v1", "study_id": original_dict["study_id"], "updated_at": now_str,
        "original_intake_h0_record": original_h0_dict, "hypotheses": hypothesis_list})
    return state_dict, hypothesis_list


def append_events(state_dict: dict, hypothesis_list: list[dict], now_str: str) -> None:
    helper_obj = skill_module("record_adaptive_event")
    evaluation_list = [
        ("E_PORTFOLIO_216", 216, [row_dict["hypothesis_id"] for row_dict in hypothesis_list if row_dict["family"] not in {"baseline", "benchmark_controls", "core5_native_replay"}],
         "tables/portfolio_metrics.csv", "scripts/research/portfolio_family_20260923/analyze.py", "research_spec_frozen.json", "source_manifest.json"),
        ("E_STANDALONE_75", 75, ["H0"], "tables/all25_common_metrics.csv", "scripts/research/portfolio_family_20260923/analyze.py", "research_spec_frozen.json", "source_manifest.json"),
        ("E_BENCHMARK_2", 2, ["H8"], "benchmarks/benchmark_manifest.json", "scripts/research/portfolio_family_20260923/benchmarks.py", "research_spec_frozen.json", "source_manifest.json"),
        ("E_CORE_REPLAY_7", 7, ["H9"], "native_replay/replay_manifest.json", "scripts/research/portfolio_family_20260923/core_replay.py", "native_replay/spec.json", "native_replay/spec.json"),
    ]
    with tempfile.TemporaryDirectory(prefix="portfolio_lineage_") as temporary_str:
        temporary_path = Path(temporary_str)
        for experiment_id_str, count_int, hypothesis_id_list, result_str, code_str, spec_str, input_str in evaluation_list:
            record_dict = {
                "schema_version": "quant-research-experiment-v1", "experiment_id": experiment_id_str,
                "study_id": state_dict["study_id"], "recorded_at": now_str, "phase": "diagnosis",
                "hypothesis_ids": hypothesis_id_list, "data_periods_seen": SEEN_PERIOD_LIST,
                "declared_variant_count": count_int,
                "selection_role": "Predeclared diagnostic evaluation; aggregate cell count, not independent discoveries.",
                "status": "completed", "evidence_paths": [result_str, spec_str],
                "spec_content_id": "sha256:" + file_hash(STUDY_PATH / spec_str),
                "data_content_ids": ["sha256:" + file_hash(STUDY_PATH / input_str),
                                     "sha256:d355567b94b60f3349f3e6eb42341ac9c7a808de552c494abcc9584256ef4160"],
                "code_content_id": "sha256:" + file_hash(REPO_PATH / code_str),
                "result_content_id": "sha256:" + file_hash(STUDY_PATH / result_str),
                "registration_note": "Recorded after execution at actual timestamp. Scientific freeze dates remain in referenced immutable specs.",
            }
            record_path = temporary_path / (experiment_id_str + ".json")
            write_json(record_path, record_dict)
            print(helper_obj.append_record(STUDY_PATH, "experiment", record_path), experiment_id_str)
        decision_dict = {
            "schema_version": "quant-research-decision-event-v1", "event_id": "D0002_CLOSEOUT_RECONCILIATION",
            "study_id": state_dict["study_id"], "recorded_at": now_str, "phase": "closeout",
            "decision": "Reconcile original frozen protocol, schema, catalog and300 completed evaluation cells; remain in closeout.",
            "reason": "No scientific change or unseen validation. Historical source code provenance remains incomplete; local native-zero parity is separately replicated. Report audit and final bundle pending.",
            "evidence_paths": ["research_spec_frozen.json", "research_spec_effective.json", "catalog_complete.json",
                               "experiment_ledger.jsonl", "native_replay/replay_manifest.json", "decision_notebook.ipynb"],
            "holdout_consequence": "All history seen; no validation or confirmation opened. No new adaptive search.",
            "active_minutes_used": state_dict["runtime_budget"]["active_minutes_used"],
        }
        record_path = temporary_path / "decision.json"
        write_json(record_path, decision_dict)
        print(helper_obj.append_record(STUDY_PATH, "decision", record_path))


NOTEBOOK_CELL_JSON_STR = r'''[["markdown","# Portfolio family: executed diagnostic decision notebook\n\nThis notebook reads frozen inputs and completed research outputs only. It does not rerun a strategy or choose weights. All historical periods were already seen. The immutable original protocol is primary; the schema-adapted specification does not change scientific rules.\n\nThe saved-unit portfolio uses a 2012-10-01 close anchor and synthetic annual sleeve rebalancing. CORE5 native replay instead begins from actual cash on 2012-09-28 with first fills on 2012-10-01. These two starts serve different questions and their returns must not be equated."],["code","from pathlib import Path\nimport hashlib\nimport json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display\n\nstudy_path = Path.cwd()\ndef digest(file_path):\n    return hashlib.sha256(file_path.read_bytes()).hexdigest()\ndef read_object(relative_str):\n    return json.loads((study_path / relative_str).read_text(encoding=\"utf-8\"))\n\nprotocol_dict = read_object(\"research_spec_frozen.json\")\ncatalog_list = read_object(\"catalog_complete.json\")\nsource_manifest_dict = read_object(\"source_manifest.json\")\nreplay_manifest_dict = read_object(\"native_replay/replay_manifest.json\")\ninput_path_list = [\n    \"research_spec_frozen.json\", \"catalog_complete.json\", \"source_manifest.json\",\n    \"native_replay/spec.json\", \"native_replay/replay_manifest.json\",\n    \"tables/portfolio_metrics.csv\", \"tables/subperiod_metrics.csv\",\n    \"tables/frozen_gate_results.csv\", \"tables/benchmark_metrics.csv\",\n    \"tables/benchmark_returns.csv.gz\", \"tables/primary_portfolio_returns.csv.gz\",\n    \"tables/native_replay_metrics.csv\",\n]\nfor case_dict in replay_manifest_dict[\"case_metadata_list\"]:\n    for leaf_str in [\"nav.csv.gz\", \"financing.csv.gz\"]:\n        input_path_list.append(\"native_replay/\" + case_dict[\"case_id_str\"] + \"/\" + leaf_str)\ninput_hash_dict = {relative_str: digest(study_path / relative_str) for relative_str in input_path_list}\nassert input_hash_dict[\"research_spec_frozen.json\"] == \"54a7e749658dc695ef316742d78c73c06f5bf582d8e9ef32a4623e435f285918\"\nassert len(catalog_list) == 25\nassert all(row_dict[\"audit_status\"] == \"source_rules_verified\" for row_dict in catalog_list)\nassert replay_manifest_dict[\"native_zero_funding_exact_parity_bool\"]\ndisplay(pd.DataFrame([\n    {\"layer\": \"Portfolio compositions\", \"definitions\": 36, \"evaluation_cells\": 36 * 2 * 3},\n    {\"layer\": \"Standalone source scenarios\", \"definitions\": 25, \"evaluation_cells\": 25 * 3},\n    {\"layer\": \"Native BIL/SPY controls\", \"definitions\": 2, \"evaluation_cells\": 2},\n    {\"layer\": \"CORE5 scenarios and native parity\", \"definitions\": 7, \"evaluation_cells\": 7},\n]))\nprint(\"Total evaluation cells:\", 216 + 75 + 2 + 7, \"/ budget320. Not independent trials.\")\nprint(\"Catalog/source manifest:\", len(catalog_list), len(source_manifest_dict[\"source_record_list\"]))\nprint(\"Historical code-at-creation reproduction: not_reproducible; current local native-zero parity: replicated.\")\n"],["markdown","## 1. Predeclared risk steps\n\nThe order is frozen: CORE5 weights 100%, 75%, 50%, 25%, 0%. Each reduction must weakly increase volatility and daily expected shortfall on the full primary sample and at least two of three chronological partitions, under both cost scenarios. We calculate adjacent differences in this order; no sorting by results. Drawdown and return changes are displayed as tradeoffs."],["code","metric_df = pd.read_csv(study_path / \"tables/portfolio_metrics.csv\")\npartition_df = pd.read_csv(study_path / \"tables/subperiod_metrics.csv\")\ngate_df = pd.read_csv(study_path / \"tables/frozen_gate_results.csv\")\nrisk_order_list = [\"CORE_100\", \"CORE_075\", \"CORE_050\", \"CORE_025\", \"CORE_000\"]\nrisk_step_list = []\nfor scenario_str in [\"common_account\", \"conservative\"]:\n    scenario_df = metric_df[(metric_df[\"scenario\"] == scenario_str) & (metric_df[\"rebalance\"] == \"annual_fixed\")].set_index(\"candidate_id\")\n    for high_core_str, low_core_str in zip(risk_order_list[:-1], risk_order_list[1:]):\n        high_series = scenario_df.loc[high_core_str]\n        low_series = scenario_df.loc[low_core_str]\n        count_int = 0\n        for period_str, period_df in partition_df[(partition_df[\"scenario\"] == scenario_str) & (partition_df[\"rebalance\"] == \"annual_fixed\")].groupby(\"period\"):\n            pair_df = period_df.set_index(\"candidate_id\")\n            count_int += int((pair_df.loc[low_core_str, \"volatility\"] >= pair_df.loc[high_core_str, \"volatility\"]) and\n                             (pair_df.loc[low_core_str, \"es5_loss\"] >= pair_df.loc[high_core_str, \"es5_loss\"]))\n        risk_step_list.append({\n            \"scenario\": scenario_str, \"from\": high_core_str, \"to\": low_core_str,\n            \"vol_change_pp\": 100 * (low_series[\"volatility\"] - high_series[\"volatility\"]),\n            \"ES5_change_pp\": 100 * (low_series[\"es5_loss\"] - high_series[\"es5_loss\"]),\n            \"CAGR_change_pp\": 100 * (low_series[\"cagr\"] - high_series[\"cagr\"]),\n            \"MDD_change_pp\": 100 * (low_series[\"max_drawdown\"] - high_series[\"max_drawdown\"]),\n            \"partitions_with_both_risk_steps\": count_int,\n        })\nrisk_step_df = pd.DataFrame(risk_step_list)\ndisplay(risk_step_df)\ndisplay(gate_df.groupby([\"gate\", \"scenario\"], sort=False)[\"pass\"].agg([\"sum\", \"count\"]))\nprint(\"All subperiod and gate checks are diagnostics on seen history.\")\n"],["markdown","## 2. Market opportunity cost on identical dates\n\nThe table below calculates annualized paired arithmetic excess return and beta against the independently produced after-tax SPY/BIL controls. This is a diagnostic comparison of the five preselected models. SPY and BIL are implemented net of 25% long-dividend withholding; the separate $SPXTR companion is a gross index and is not substituted for the after-tax controls.\n\nFormula: annual excess mean = 252 × mean(r_portfolio − r_control); beta = covariance(r_portfolio, r_SPY) / variance(r_SPY). There is no future information entering strategy decisions because these are ex-post descriptive calculations."],["code","return_df = pd.read_csv(study_path / \"tables/primary_portfolio_returns.csv.gz\", index_col=\"date\", parse_dates=True)\nbenchmark_df = pd.read_csv(study_path / \"tables/benchmark_returns.csv.gz\", index_col=\"date\", parse_dates=True)\n# *** CRITICAL *** Ex-post diagnostics only: exact calendar equality, no date intersection or missing-return fill.\nassert return_df.index.equals(benchmark_df.index)\nassert not return_df[risk_order_list].isna().any().any()\nassert not benchmark_df.isna().any().any()\nmarket_list = []\nfor candidate_str in risk_order_list:\n    candidate_series = return_df[candidate_str]\n    market_series = benchmark_df[\"SPY_net25\"]\n    market_list.append({\n        \"candidate\": candidate_str, \"observations\": len(candidate_series),\n        \"annual_mean_excess_BIL_pp\": 25200 * (candidate_series - benchmark_df[\"BIL_net25\"]).mean(),\n        \"annual_mean_excess_SPY_pp\": 25200 * (candidate_series - market_series).mean(),\n        \"beta_SPY\": candidate_series.cov(market_series) / market_series.var(ddof=1),\n        \"daily_correlation_SPY\": candidate_series.corr(market_series),\n    })\nmarket_df = pd.DataFrame(market_list).set_index(\"candidate\")\nsource_comparison_df = metric_df[(metric_df[\"scenario\"] == \"common_account\") &\n                                 (metric_df[\"rebalance\"] == \"annual_fixed\")].set_index(\"candidate_id\")\nnp.testing.assert_allclose(market_df.loc[risk_order_list, \"beta_SPY\"], source_comparison_df.loc[risk_order_list, \"beta\"], rtol=1e-10, atol=1e-12)\ndisplay(market_df)\ndisplay(pd.read_csv(study_path / \"tables/benchmark_metrics.csv\")[[\"candidate_id\", \"cagr\", \"volatility\", \"max_drawdown\", \"es5_loss\"]])\nprint(\"Financing5% common and8% stress are fixed research scenarios, not actual broker rates.\")\n"],["markdown","## 3. Stateful CORE5 capital and cost replay\n\nThese six native scenarios preserve current source signals and sizing, with a true prior cash anchor. We recompute CAGR, volatility and drawdown from saved daily NAV; the first real fill is included. Funding is read from its ledger and normalized to initial capital. These diagnostics isolate capital/financing sensitivity; they do not prove whole-family physical rebalancing or live execution.\n\nFormula: r_t = NAV_t / NAV_(t−1) − 1; CAGR = (NAV_last / NAV_anchor)^(252 / N_returns) − 1; volatility = sample_std(r) × sqrt(252); drawdown = min(NAV / cumulative_max(NAV) − 1). Funding debit = max(0, rounded102% short collateral − cash) × scenario_rate × known_days_to_next_session / 360, with terminal interval zero."],["code","replay_metric_list = []\nfor case_dict in replay_manifest_dict[\"case_metadata_list\"]:\n    case_str = case_dict[\"case_id_str\"]\n    case_path = study_path / \"native_replay\" / case_str\n    nav_df = pd.read_csv(case_path / \"nav.csv.gz\", parse_dates=[\"date\"])\n    funding_df = pd.read_csv(case_path / \"financing.csv.gz\")\n    nav_vec = nav_df[\"total_value\"].to_numpy(dtype=float)\n    # *** CRITICAL *** Adjacent saved marks only; prior cash anchor includes first fills, no future sizing.\n    return_vec = nav_vec[1:] / nav_vec[:-1] - 1\n    capital_float = float(case_dict[\"capital_float\"])\n    assert nav_vec[0] == capital_float\n    assert str(nav_df[\"date\"].iat[0].date()) == \"2012-09-28\"\n    assert funding_df[\"calendar_days_int\"].iat[-1] == 0\n    assert funding_df[\"funding_fee_float\"].iat[-1] == 0\n    np.testing.assert_allclose(funding_df[\"cash_before_funding_float\"] - funding_df[\"funding_fee_float\"],\n                               funding_df[\"cash_after_funding_float\"], atol=1e-8)\n    replay_metric_list.append({\n        \"case\": case_str, \"capital\": capital_float, \"returns\": len(return_vec),\n        \"CAGR\": (nav_vec[-1] / nav_vec[0]) ** (252 / len(return_vec)) - 1,\n        \"volatility\": return_vec.std(ddof=1) * np.sqrt(252),\n        \"MDD\": (nav_vec / np.maximum.accumulate(nav_vec) - 1).min(),\n        \"ending_NAV_per_initial_dollar\": nav_vec[-1] / nav_vec[0],\n        \"cumulative_funding_per_initial_dollar\": funding_df[\"funding_fee_float\"].sum() / capital_float,\n        \"negative_cash_sessions\": int((nav_df[\"cash\"] < 0).sum()),\n    })\nreplay_metric_df = pd.DataFrame(replay_metric_list).set_index(\"case\")\nsaved_replay_df = pd.read_csv(study_path / \"tables/native_replay_metrics.csv\").set_index(\"case_id\")\nnp.testing.assert_allclose(replay_metric_df[\"CAGR\"], saved_replay_df.loc[replay_metric_df.index, \"cagr\"], rtol=1e-10, atol=1e-12)\nnp.testing.assert_allclose(replay_metric_df[\"MDD\"], saved_replay_df.loc[replay_metric_df.index, \"max_drawdown\"], rtol=1e-10, atol=1e-12)\ndisplay(replay_metric_df)\ncommon_case_list = [\"core5_common_1000000\", \"core5_common_750000\", \"core5_common_500000\", \"core5_common_250000\"]\ncapital_comparison_df = replay_metric_df.loc[common_case_list, [\"CAGR\", \"ending_NAV_per_initial_dollar\"]].copy()\ncapital_comparison_df[\"CAGR_difference_from_1m_bps\"] = 10000 * (capital_comparison_df[\"CAGR\"] - capital_comparison_df.loc[\"core5_common_1000000\", \"CAGR\"])\ndisplay(capital_comparison_df)\nprint(\"Seventh plain-native code parity:\", replay_manifest_dict[\"native_zero_funding_exact_parity_bool\"])\n"],["markdown","## Limits and reproducibility\n\n- Original source histories do not establish original code-at-creation identity. The local CORE5 parity result covers current code and this archived input only.\n- All history is reused. Prior Foundry64 and CORE5 nominal441 search counts overlap by an unknown amount. The300 current evaluation cells are not independent trials.\n- Synthetic annual sleeve-unit rebalancing and fixed-holdings cost overlays do not establish full physical account implementation. Native CORE5 replay covers only its own account scenarios.\n- No untouched validation, calibrated capacity, broker locate/recall, live-fill, ILS conversion, client suitability or capital-preservation guarantee was tested.\n- Statistical adjustment is limited to the predeclared19 paired comparisons; it does not erase prior research selection.\n\nThe final cell checks every input digest again. Successful execution means the notebook reproduced these diagnostic calculations from stable files, not that an investment claim has been validated."],["code","assert input_hash_dict == {relative_str: digest(study_path / relative_str) for relative_str in input_path_list}\nprint(\"All\", len(input_hash_dict), \"input hashes unchanged during notebook execution.\")\nprint(\"Protocol frozen at:\", protocol_dict[\"initial_frozen_at\"])\nprint(\"Notebook calculations completed; no strategy or optimizer rerun.\")\ndisplay(pd.DataFrame([{\"path\": key_str, \"sha256\": value_str} for key_str, value_str in input_hash_dict.items()]))\n"]]'''


def execute_notebook() -> None:
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook_obj = nbformat.v4.new_notebook()
    cell_list = json.loads(NOTEBOOK_CELL_JSON_STR)
    if (STUDY_PATH / "adaptive_sector_manifest.json").exists():
        adaptive_manifest_dict = read_json(STUDY_PATH / "adaptive_sector_manifest.json")
        adaptive_path_list = ["adaptive_sector_spec.json", "adaptive_sector_manifest.json"]
        adaptive_path_list.extend(Path(row_dict["path_str"]).relative_to(STUDY_PATH).as_posix() for row_dict in adaptive_manifest_dict["outputs"])
        cell_list[1][1] = cell_list[1][1].replace("input_hash_dict =", "input_path_list.extend(" + repr(adaptive_path_list) + ")\ninput_hash_dict =")
        cell_list[1][1] = cell_list[1][1].replace("Total evaluation cells:", "Original program evaluation cells:")
        cell_list[-2][1] = cell_list[-2][1].replace("The300 current evaluation cells", "The300 original plus18 post-result evaluation cells")
        cell_list = cell_list[:-2] + json.loads(ADAPTIVE_NOTEBOOK_CELL_JSON_STR) + cell_list[-2:]
    notebook_obj.cells = [
        nbformat.v4.new_markdown_cell(content_str) if kind_str == "markdown"
        else nbformat.v4.new_code_cell(content_str)
        for kind_str, content_str in cell_list
    ]
    notebook_obj.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python (study environment)", "language": "python"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "research_scope": "Saved artifact loading and diagnostic calculations only; no strategy rerun.",
        "original_spec_sha256": ORIGINAL_SHA256_STR,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    manager_obj = KernelManager(kernel_name="python3")
    manager_obj.kernel_spec.argv[0] = sys.executable
    client_obj = NotebookClient(notebook_obj, timeout=120, km=manager_obj,
                                resources={"metadata": {"path": str(STUDY_PATH)}})
    try:
        client_obj.execute()
    finally:
        if manager_obj.has_kernel:
            manager_obj.shutdown_kernel(now=True)
        manager_obj.cleanup_resources()
    code_cell_list = [cell_obj for cell_obj in notebook_obj.cells if cell_obj.cell_type == "code"]
    assert len(code_cell_list) >= 3
    assert all(cell_obj.execution_count is not None and cell_obj.outputs for cell_obj in code_cell_list)
    assert not any(output_obj.output_type == "error" for cell_obj in code_cell_list for output_obj in cell_obj.outputs)
    notebook_obj.metadata["executed_at"] = datetime.now(timezone.utc).isoformat()
    nbformat.write(notebook_obj, STUDY_PATH / "decision_notebook.ipynb")
    print("PASS: executed notebook", len(code_cell_list), "code cells")


def validate_outputs() -> dict:
    specification_module = skill_module("validate_research_spec")
    adaptive_module = skill_module("validate_adaptive_research")
    # Bundle validator imports its adjacent validators by module name.
    sys.path.insert(0, str(SKILL_PATH))
    bundle_module = skill_module("validate_research_bundle")
    failure_dict = {
        "effective_spec": specification_module.validate_research_spec(STUDY_PATH / "research_spec_effective.json"),
        "adaptive_state": adaptive_module.validate_adaptive_research(STUDY_PATH),
        "executed_notebook": bundle_module.validate_notebook(STUDY_PATH / "decision_notebook.ipynb", True),
    }
    assert file_hash(STUDY_PATH / "research_spec_frozen.json") == ORIGINAL_SHA256_STR
    for label_str, failure_list in failure_dict.items():
        print(("FAIL: " if failure_list else "PASS: ") + label_str, failure_list)
    return failure_dict


def main() -> None:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--register-sector", action="store_true")
    parser_obj.add_argument("--complete-sector", action="store_true")
    parser_obj.add_argument("--notebook-only", action="store_true")
    parser_obj.add_argument("--validate-only", action="store_true")
    argument_obj = parser_obj.parse_args()
    assert file_hash(STUDY_PATH / "research_spec_frozen.json") == ORIGINAL_SHA256_STR
    if argument_obj.register_sector:
        register_sector()
        return
    if argument_obj.complete_sector:
        complete_sector()
        return
    if argument_obj.validate_only:
        failure_dict = validate_outputs()
        raise SystemExit(int(any(failure_dict.values())))
    if argument_obj.notebook_only:
        execute_notebook()
        return
    # This is a one-time ledger reconciliation. Repeated notebook execution is explicit and cannot append new research trials.
    assert not (STUDY_PATH / "experiment_ledger.jsonl").read_text(encoding="utf-8").strip(), "Ledger already reconciled; use --notebook-only or --validate-only."
    now_str = datetime.now(timezone.utc).isoformat()
    original_dict = read_json(STUDY_PATH / "research_spec_frozen.json")
    assert read_json(STUDY_PATH / "analysis_run.json")["portfolio_cells"] == 216
    assert read_json(STUDY_PATH / "analysis_run.json")["standalone_cells"] == 75
    assert read_json(STUDY_PATH / "native_replay/replay_manifest.json")["total_run_count_int"] == 7
    catalog_list = merge_catalog()
    effective_spec(original_dict, now_str)
    source_rule_map(catalog_list, now_str)
    state_dict, hypothesis_list = reconcile_state(original_dict, now_str)
    execute_notebook()
    append_events(state_dict, hypothesis_list, now_str)
    failure_dict = validate_outputs()
    state_dict["validation_checks"] = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "failures": failure_dict,
        "full_bundle_status": "Pending primary-agent report audit, knowledge record and manifest.",
    }
    write_json(STUDY_PATH / "research_state.json", state_dict)
    raise SystemExit(int(any(failure_dict.values())))



ADAPTIVE_SPEC_SHA256_STR = "c28d1fb217495c627a8f2d2c53b40dcca67a99043bd0d3210c55c3880586e836"


def append_lineage_record(kind_str: str, record_dict: dict) -> None:
    helper_obj = skill_module("record_adaptive_event")
    with tempfile.TemporaryDirectory(prefix="portfolio_lineage_") as temporary_str:
        record_path = Path(temporary_str) / "record.json"
        write_json(record_path, record_dict)
        print(helper_obj.append_record(STUDY_PATH, kind_str, record_path))


def register_sector() -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    spec_path = STUDY_PATH / "adaptive_sector_spec.json"
    assert file_hash(spec_path) == ADAPTIVE_SPEC_SHA256_STR
    adaptive_spec_dict = read_json(spec_path)
    registry_dict = read_json(STUDY_PATH / "hypothesis_registry.json")
    assert not any(row_dict["hypothesis_id"] == "H10" for row_dict in registry_dict["hypotheses"])
    assert adaptive_spec_dict["budget"]["new_cost_rebalance_cells"] == 18
    effective_dict = read_json(STUDY_PATH / "research_spec_effective.json")
    effective_dict.update({
        "last_amended_at": now_str,
        "schema_adaptation_scientific_change_bool": False,
        "post_result_scientific_extension_bool": True,
        "post_result_amendments": [{
            "hypothesis_id": "H10", "amendment_label": "H5",
            "path": "adaptive_sector_spec.json", "sha256": ADAPTIVE_SPEC_SHA256_STR,
            "scientific_change_bool": True, "frozen_at": adaptive_spec_dict["frozen_at"],
            "registered_at": now_str, "history_status": adaptive_spec_dict["seen_history"],
            "literal_contract": adaptive_spec_dict,
        }],
        "adaptation_note": "Original schema adaptation changed no science and retains its actual schema_adapted_at. The separate H10 post-result amendment adds exactly6 definitions/18 cells. The immutable original protocol remains primary for its original program; H10 is not retrospectively called predeclared there.",
    })
    effective_dict["search_space"].update({
        "total_declared_variants": 318, "actual_evaluation_cells": 300,
        "planned_total_evaluation_cells": 318, "original_protocol_cap": 320,
        "original_completed_evaluation_cells": 300, "post_result_new_cells": 18,
        "count_semantics": "300 original evaluation cells plus18 explicitly post-result cells=318 planned, within original cap320. Six paired comparisons are diagnostics, not extra cell definitions. Nine original reference recomputations are parity checks.",
    })
    effective_dict["search_space"]["declared_families"].append("post_result_sector_replacement_H10")
    effective_dict["search_space"]["axes"].update({"post_result_sector_definitions": 6, "post_result_sector_cost_scenarios": 3})
    write_json(STUDY_PATH / "research_spec_effective.json", effective_dict)
    registry_dict["hypotheses"].append({
        "hypothesis_id": "H10", "amendment_label": "H5", "title": "Post-result sector-rebound replacement with matched BIL control",
        "provenance": "post_result_adaptive", "family": "post_result_sector_replacement",
        "role": "portfolio_construction", "classification": "portfolio_construction",
        "economic_mechanism": adaptive_spec_dict["mechanism"],
        "expected_direction": "Sector replacement should preserve reward while improving risk versus HPI and show sufficient reward beyond matched BIL dilution.",
        "falsifier": adaptive_spec_dict["gate_rule"],
        "created_at": now_str, "frozen_at": now_str, "protocol_frozen_at": adaptive_spec_dict["frozen_at"],
        "registration_note": "Actual structured-record writing timestamp after execution; independent immutable spec records the earlier scientific freeze. Review occurred before outcome publication, not before launch. H5 original momentum_split identity remains unchanged.",
        "data_periods_seen": SEEN_PERIOD_LIST + [adaptive_spec_dict["seen_history"]],
        "declared_variant_count": 18, "experiment_ids": [], "status": "planned",
        "disposition": "diagnostic", "spec_content_id": "sha256:" + ADAPTIVE_SPEC_SHA256_STR,
    })
    registry_dict["updated_at"] = now_str
    write_json(STUDY_PATH / "hypothesis_registry.json", registry_dict)
    state_dict = read_json(STUDY_PATH / "research_state.json")
    state_dict["updated_at"] = now_str
    state_dict["adaptive_search"].update({"declared_total_variants": 318, "pending_hypothesis": "H10",
                                         "stop_reason": "Exactly18 post-result cells authorized and frozen; final scientific follow-up. No further search."})
    state_dict["runtime_budget"]["active_minutes_used"] = math.ceil((datetime.fromisoformat(now_str) - datetime.fromisoformat(state_dict["created_at"])).total_seconds() / 60)
    state_dict["workflow_limitations"] = ["The original durable map/registry/ledgers were reconciled after results using independent immutable pre-result declarations. Schema pass does not imply that each durable file existed before the original run.",
                                         "H10 is explicitly a post-result hypothesis; no retrospective predeclaration or new untouched validation."]
    state_dict["protocol_provenance"]["post_result_amendment"] = {"path": "adaptive_sector_spec.json", "sha256": ADAPTIVE_SPEC_SHA256_STR, "frozen_at": adaptive_spec_dict["frozen_at"]}
    write_json(STUDY_PATH / "research_state.json", state_dict)
    map_path = STUDY_PATH / "SOURCE_RULE_MAP.md"
    with map_path.open("a", encoding="utf-8") as map_file:
        map_file.write("\n## H10 post-result amendment\n\nRegistered at " + now_str + "; independent protocol frozen at " + adaptive_spec_dict["frozen_at"] + ". The original300-cell program above remains unchanged. H10 adds6 definitions/18 cost cells: only HPI's budget in CORE75/50/25 is replaced by the unchanged11-sector rebound source or matched BIL. Current total planned318, original cap320. Six Holm-adjusted exploratory comparisons form a separate family from the original19, with no unseen holdout. Reduced stock-order competition remains a hypothesis, not fill evidence. Source rules, source universe and original gates are unchanged. No additional search after H10.\n")
    append_lineage_record("decision", {
        "schema_version": "quant-research-decision-event-v1", "event_id": "D0003_REGISTER_H10",
        "study_id": state_dict["study_id"], "recorded_at": now_str, "phase": "diagnosis",
        "decision": "Register final post-result H10 amendment:6 definitions,18 new cells,318 planned total under320.",
        "reason": adaptive_spec_dict["mechanism"] + " Matched BIL controls distinguish reduced risk through dilution from added sector value.",
        "evidence_paths": ["adaptive_sector_spec.json", "research_spec_effective.json", "hypothesis_registry.json"],
        "holdout_consequence": adaptive_spec_dict["seen_history"],
        "active_minutes_used": state_dict["runtime_budget"]["active_minutes_used"],
        "frozen_spec_at": adaptive_spec_dict["frozen_at"], "frozen_spec_sha256": ADAPTIVE_SPEC_SHA256_STR,
    })
    print("PASS: registered H10; original ledger untouched; planned318, completed300")


def complete_sector() -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    adaptive_spec_dict = read_json(STUDY_PATH / "adaptive_sector_spec.json")
    manifest_dict = read_json(STUDY_PATH / "adaptive_sector_manifest.json")
    assert file_hash(STUDY_PATH / "adaptive_sector_spec.json") == ADAPTIVE_SPEC_SHA256_STR == manifest_dict["spec_sha256"]
    assert manifest_dict["new_cells"] == 18 and manifest_dict["total_distinct_cells"] == 318
    assert manifest_dict["all_input_hashes_unchanged"] and manifest_dict["synthetic_gate_boundary_checks_passed"]
    for output_dict in manifest_dict["outputs"]:
        assert file_hash(Path(output_dict["path_str"])) == output_dict["sha256_str"]
    record_list = [json.loads(line_str) for line_str in (STUDY_PATH / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line_str.strip()]
    assert sum(row_dict["declared_variant_count"] for row_dict in record_list) == 300
    state_dict = read_json(STUDY_PATH / "research_state.json")
    append_lineage_record("experiment", {
        "schema_version": "quant-research-experiment-v1", "experiment_id": "E_H10_SECTOR_18",
        "study_id": state_dict["study_id"], "recorded_at": now_str, "phase": "adaptive_discovery",
        "hypothesis_ids": ["H10"], "data_periods_seen": SEEN_PERIOD_LIST + [adaptive_spec_dict["seen_history"]],
        "declared_variant_count": 18, "selection_role": "Explicit post-result diagnostic substitution;6 fixed definitions x3 costs. Six paired comparisons are contextual and separately Holm-adjusted;9 unchanged reference recomputations count as parity only.",
        "status": "completed", "evidence_paths": ["adaptive_sector_manifest.json", "adaptive_sector_spec.json", "tables/adaptive_sector_metrics.csv"],
        "spec_content_id": "sha256:" + ADAPTIVE_SPEC_SHA256_STR,
        "data_content_ids": ["sha256:" + row_dict["sha256_str"] for row_dict in adaptive_spec_dict["inputs"]],
        "code_content_id": "sha256:" + file_hash(REPO_PATH / "scripts/research/portfolio_family_20260923/adaptive_sector.py"),
        "result_content_id": "sha256:" + file_hash(STUDY_PATH / "adaptive_sector_manifest.json"),
        "execution_completed_at": manifest_dict["completed_at"],
    })
    registry_dict = read_json(STUDY_PATH / "hypothesis_registry.json")
    for hypothesis_dict in registry_dict["hypotheses"]:
        if hypothesis_dict["hypothesis_id"] == "H10":
            hypothesis_dict.update({"experiment_ids": ["E_H10_SECTOR_18"], "status": "diagnosed",
                                    "evidence_summary": "Completed fixed18-cell diagnostic; frozen gate decisions in adaptive_sector_manifest.json. No new research branch or untouched validation."})
    registry_dict["updated_at"] = now_str
    write_json(STUDY_PATH / "hypothesis_registry.json", registry_dict)
    state_dict["updated_at"] = now_str
    state_dict["adaptive_search"].update({
        "rounds_completed": 1, "actual_new_hypotheses_by_round": [1],
        "declared_total_variants": 318, "actual_total_variants": 318, "pending_hypothesis": None,
        "count_semantics": "Original300 plus18 post-result cells; distinct definitions/scenarios, not independent discoveries.",
        "stop_reason": adaptive_spec_dict["stop_rule"],
    })
    state_dict["runtime_budget"]["active_minutes_used"] = math.ceil((datetime.fromisoformat(now_str) - datetime.fromisoformat(state_dict["created_at"])).total_seconds() / 60)
    state_dict["runtime_budget"]["budget_stop_reason"] = "All318 declared cells finished; scientific search stopped. Report audit and bundle closeout only."
    state_dict["phase"] = "closeout"
    state_dict["evidence_phase"] = "diagnosis"
    write_json(STUDY_PATH / "research_state.json", state_dict)
    effective_dict = read_json(STUDY_PATH / "research_spec_effective.json")
    effective_dict["last_amended_at"] = now_str
    effective_dict["search_space"]["actual_evaluation_cells"] = 318
    effective_dict["post_result_amendments"][0]["completed_at"] = manifest_dict["completed_at"]
    effective_dict["post_result_amendments"][0]["result_manifest_sha256"] = file_hash(STUDY_PATH / "adaptive_sector_manifest.json")
    write_json(STUDY_PATH / "research_spec_effective.json", effective_dict)
    with (STUDY_PATH / "SOURCE_RULE_MAP.md").open("a", encoding="utf-8") as map_file:
        map_file.write("\nH10 execution completed at " + manifest_dict["completed_at"] + "; completion registered at " + now_str + ". Actual total318, one post-result round and one new hypothesis. The scientific stop rule is now reached; gate results remain diagnostic and no further search is authorized.\n")
    execute_notebook()
    append_lineage_record("decision", {
        "schema_version": "quant-research-decision-event-v1", "event_id": "D0004_COMPLETE_H10",
        "study_id": state_dict["study_id"], "recorded_at": now_str, "phase": "closeout",
        "decision": "Record H10 completion and stop scientific search at318 cells; retain diagnosis evidence stage.",
        "reason": "All18 frozen post-result cells completed; no added basket, budget, fine weights or branch. Original300 records and original protocol preserved.",
        "evidence_paths": ["adaptive_sector_manifest.json", "adaptive_sector_spec.json", "experiment_ledger.jsonl", "decision_notebook.ipynb"],
        "holdout_consequence": "No untouched validation or confirmation; separate six-test exploratory family does not correct post-result selection.",
        "active_minutes_used": state_dict["runtime_budget"]["active_minutes_used"],
    })
    failure_dict = validate_outputs()
    state_dict["validation_checks"] = {"checked_at": datetime.now(timezone.utc).isoformat(), "failures": failure_dict,
                                       "full_bundle_status": "Pending primary report audit, knowledge/state final verdict alignment and manifest."}
    write_json(STUDY_PATH / "research_state.json", state_dict)
    assert not any(failure_dict.values())


ADAPTIVE_NOTEBOOK_CELL_JSON_STR = r'''[["markdown","## 4. H10: explicitly post-result sector replacement\n\nH10 was frozen after the original results were inspected. It replaces only the HPI budget at CORE5 75%, 50% and 25% with the fixed 11-sector rebound strategy, alongside matched BIL controls. The original 300 evaluation cells remain intact; H10 adds six definitions × three cost scenarios = 18, for 318 total under the original cap of 320. The nine original reference recomputations are parity checks.\n\nWe recompute the frozen gate conjunction from its component differences. Each comparison requires all three conditions on the full sample and at least two of three partitions, under both common and conservative costs. Six paired mean tests use a separate Holm correction, which does not correct the post-result choice or prior search. No further scientific branch follows success or failure."],["code","adaptive_spec_dict = read_object(\"adaptive_sector_spec.json\")\nadaptive_manifest_dict = read_object(\"adaptive_sector_manifest.json\")\nadaptive_metric_df = pd.read_csv(study_path / \"tables/adaptive_sector_metrics.csv\")\nadaptive_detail_df = pd.read_csv(study_path / \"tables/adaptive_sector_gate_detail.csv\")\nadaptive_gate_df = pd.read_csv(study_path / \"tables/adaptive_sector_gates.csv\")\nadaptive_decision_df = pd.read_csv(study_path / \"tables/adaptive_sector_decisions.csv\")\nassert adaptive_metric_df[\"new_definition\"].sum() == 18\nassert adaptive_manifest_dict[\"new_cells\"] == 18 and 300 + 18 == 318\nrecomputed_gate_list = []\nfor gate_row in adaptive_gate_df.itertuples(index=False):\n    selected_df = adaptive_detail_df[(adaptive_detail_df[\"candidate_id\"] == gate_row.candidate_id) &\n                                     (adaptive_detail_df[\"comparison\"] == gate_row.comparison) &\n                                     (adaptive_detail_df[\"scenario\"] == gate_row.scenario)].copy()\n    threshold_dict = adaptive_spec_dict[\"gate_vs_\" + gate_row.comparison]\n    selected_df[\"recomputed_joint\"] = (\n        (selected_df[\"cagr_difference\"] >= threshold_dict[\"minimum_cagr_difference\"] - 1e-12) &\n        (selected_df[\"es5_loss_ratio\"] <= threshold_dict[\"maximum_es5_ratio\"] + 1e-12) &\n        (selected_df[\"signed_drawdown_difference\"] >= threshold_dict[\"minimum_signed_max_drawdown_difference\"] - 1e-12))\n    assert selected_df[\"recomputed_joint\"].equals(selected_df[\"joint_pass\"])\n    full_bool = bool(selected_df.loc[selected_df[\"period\"] == \"full\", \"recomputed_joint\"].item())\n    partition_int = int(selected_df.loc[selected_df[\"period\"] != \"full\", \"recomputed_joint\"].sum())\n    gate_bool = full_bool and partition_int >= 2\n    saved_bool = bool(adaptive_gate_df.loc[(adaptive_gate_df[\"candidate_id\"] == gate_row.candidate_id) &\n                                         (adaptive_gate_df[\"comparison\"] == gate_row.comparison) &\n                                         (adaptive_gate_df[\"scenario\"] == gate_row.scenario), \"pass\"].item())\n    assert gate_bool == saved_bool\n    recomputed_gate_list.append({\"candidate\": gate_row.candidate_id, \"comparison\": gate_row.comparison,\n                                 \"scenario\": gate_row.scenario, \"full_pass\": full_bool,\n                                 \"partitions_passed\": partition_int, \"gate_pass\": gate_bool})\ndisplay(pd.DataFrame(recomputed_gate_list))\ndisplay(adaptive_detail_df[adaptive_detail_df[\"period\"] == \"full\"][[\n    \"candidate_id\", \"comparison\", \"scenario\", \"cagr_difference\", \"es5_loss_ratio\", \"signed_drawdown_difference\"]])\ndisplay(adaptive_decision_df)\ndisplay(pd.read_csv(study_path / \"tables/adaptive_sector_paired_bootstrap.csv\"))\nprint(\"H10 protocol freeze:\", adaptive_spec_dict[\"frozen_at\"])\nprint(\"H10 execution completion:\", adaptive_manifest_dict[\"completed_at\"])\nprint(\"318 cells complete; evidence remains diagnostic; no further scientific search.\")\n"]]'''


if __name__ == "__main__":
    main()
