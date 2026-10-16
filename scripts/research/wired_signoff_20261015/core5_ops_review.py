"""Assemble read-only CORE5/operations signoff evidence; no market/broker IO."""
from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as element_tree

REPO_PATH = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_PATH / "results/review/wired_signoff_20261015"
CORE_PATH = REPO_PATH / "results/research/strategy/strategy_taa_adaptive_macro_core5"
BORROW_PATH = CORE_PATH / "borrow_cost_study/2026-08-25_core5_dbc_borrow_cost"


def read_json(path_obj: Path) -> dict:
    return json.loads(path_obj.read_text(encoding="utf-8"))


def manifest_dict(path_obj: Path) -> dict:
    try:
        display_str = path_obj.relative_to(REPO_PATH).as_posix()
    except ValueError:
        display_str = str(path_obj)
    return {"path": display_str, "bytes": path_obj.stat().st_size,
        "sha256": hashlib.sha256(path_obj.read_bytes()).hexdigest()}


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    baseline_receipt_list = [read_json(path_obj) for path_obj in sorted((OUTPUT_PATH / "baselines").glob("*/summary.json"))]
    final_baseline_list = [row_dict for row_dict in baseline_receipt_list if row_dict.get("status") == "pass"
        and row_dict.get("deterministic_equity") is True and row_dict.get("deterministic_economic_transactions") is True
        and row_dict.get("capital_honored") is True and row_dict.get("actual_capital_base") == 1_000_000
        and row_dict.get("actual_double_capital_base") == 2_000_000 and row_dict.get("seconds") is not None]
    statistic_index_path = OUTPUT_PATH / "audit_statistics/index.json"
    statistic_row_list = read_json(statistic_index_path) if statistic_index_path.exists() else []
    completion_dict = {"registered_strategy_count_int": len(read_json(OUTPUT_PATH / "frozen_manifest.json")["strategies"]),
        "baseline_receipt_count_int": len(baseline_receipt_list), "final_baseline_pass_count_int": len(final_baseline_list),
        "saved_statistics_pass_count_int": sum(row_dict.get("status") == "pass" for row_dict in statistic_row_list),
        "scope_str": "Completed canonical saved baselines and descriptive saved-ledger statistics; these counts are not deployed account acceptance or broker verification."}
    qualification_path = CORE_PATH / "wiring_qualification/2026-10-05_adapter_parity_final/qualification.json"
    qualification_dict = read_json(qualification_path)
    source_match_list = []
    for source_str, expected_str in qualification_dict["source_sha256_dict"].items():
        actual_str = hashlib.sha256((REPO_PATH / source_str).read_bytes()).hexdigest()
        source_match_list.append({"path_str": source_str, "saved_hash_str": expected_str,
            "current_hash_str": actual_str, "matches_bool": expected_str == actual_str})
    snapshot_path = CORE_PATH / "snapshot_data_qualification/2026-09-15_step2_final/snapshot_prices.parquet"
    snapshot_match_bool = hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == qualification_dict["prices_sha256_str"]
    summary_path = CORE_PATH / "vanilla_backtest/2026-09-30_103336/summary.json"
    metadata_path = summary_path.parent / "metadata.json"
    with (BORROW_PATH / "tables/borrow_rate_sweep.csv").open(encoding="utf-8", newline="") as handle_obj:
        borrow_row_list = list(csv.DictReader(handle_obj))
    reference_path = Path("C:/Users/User/Documents/workspace/pakal/pakal-research/reports/vardi_core5_robustness_qualification_study/REPORT.md")
    input_path_list = [summary_path, metadata_path, qualification_path, snapshot_path,
        BORROW_PATH / "tables/borrow_rate_sweep.csv", BORROW_PATH / "tables/promotion_gates.csv",
        BORROW_PATH / "research_spec_frozen.json", reference_path,
        REPO_PATH / "results/research/strategy_readiness_audit_20260928/core5/CORE5_FINDINGS.md",
        REPO_PATH / "results/research/strategy_readiness_audit_20260928/core5/a2_split_invariance.json",
        REPO_PATH / "results/research/strategy_readiness_audit_20260928/core5/c_capital_tradability.json",
        REPO_PATH / "docs/live/DAILY_EXECUTION_RULES_20261007.md"]
    core_dict = {
        "schema_version_str": "wired_signoff.review.v1", "family_str": "Adaptive macro / CORE5",
        "strategy_import_str": "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5",
        "generated_at_utc_str": datetime.now(UTC).isoformat(), "audit_model_budget_float": 1_000_000.0,
        "production_modified_bool": False, "broker_verified_bool": False, "new_strategy_variants_int": 0,
        "provisional_verdict_str": "Current-input USD1M baseline and conditional native-host replay are verified; research selection limits remain explicit, and physical account acceptance is separate unperformed evidence",
        "audit_completion_counts_dict": completion_dict,
        "old_published_summary_dict": read_json(summary_path),
        "old_published_accounting_dict": read_json(metadata_path)["accounting_policy"],
        "old_published_capital_float": read_json(metadata_path)["capital_base"],
        "old_qualification_dict": {"status_str": qualification_dict["status_str"],
            "decision_count_int": qualification_dict["decision_count_int"],
            "rebalance_count_int": qualification_dict["rebalance_count_int"],
            "dbc_two_leg_count_int": qualification_dict["dbc_two_leg_count_int"],
            "same_price_input_hash_bool": snapshot_match_bool, "source_match_list": source_match_list,
            "interpretation_str": "A passed historical run only certifies its saved sources. Changed hashes require current replay, not promotion of this old PASS."},
        "source_rules_dict": {
            "universe_list": ["SPY", "IEF", "GLD", "DBC", "UUP"], "parking_str": "BIL",
            "signal_adjustment_str": "TOTALRETURN", "execution_mark_adjustment_str": "CAPITALSPECIAL",
            "severity_formula_str": "d_T=1-P_T/max(P_s for s<=T)",
            "rank_formula_str": "R_T=(N_less+(N_equal+1)/2)/126 on inclusive trailing126 throughT",
            "alpha_formula_str": "a_T=R_T^2*(2/51)+(1-R_T^2)*(2/201)",
            "ama_formula_str": "AMA_T=a_T*P_T+(1-a_T)*AMA_(T-1); first finite alpha initializes toP_T",
            "long_formula_str": "L_i,T=1[SMA10_i,T>AMA_i,T]; risk long target=0.20*L_i,T",
            "dbc_short_formula_str": "when SMA10_DBC<AMA_DBC, w_DBC=-min(0.10,0.025/(std63(returns,ddof=1)*sqrt252))",
            "parking_formula_str": "w_BIL=1-0.20*sum(long_states); DBC short proceeds retained as cash, no reinvestment",
            "events_str": "initialization, any long-state change, actual XNYS month-end; short state/volatility alone are not events",
            "nav_formula_str": "NAV_T=broker_cash_T+sum(signed_actual_shares_i,T*Close_i,T)",
            "target_formula_str": "q_target_i=trunc(NAV_T*w_last_event_i/Close_i,T), signed truncation towardzero",
            "receipt_str": "If replayed event date and actual shares match applied receipt, hold existing shares despite drift/cash/history revision. Otherwise catchup from current observed account. Advance receipt only at after-close finalization when full frozen book achieved.",
            "timing_str": "Signal inputs throughCloseT; trusted same-session EOD due exchangeclose+10m; frozen next-session MOO; scheduled09:23:30; strict per-order09:28 deadline for09:30 open; calendar earlycloses/DST.",
            "research_costs_str": "2.5bps each executed leg; max($1,abs(shares)*$0.005) commission; long dividends75% net, short dividends100% debit; positive cash0%; negative cash financing unmodeled",
            "borrow_formula_str": "abs(q_DBC)*ceil(1.02*Close_DBC)*1%*calendar_days_to_next_session/360 after fills/mark; final session skips future accrual",
            "live_costs_str": "Observed broker account cash includes actual costs/distributions; adapter never debits synthetic research borrow again",
            "approved_failure_policy_str": "No margin/funding/borrow preview by owner2026-10-11. Independent broker order acceptance; no batch atomicity; uncertain send never resubmitted; one eligible MKT SELL completion per decision/asset; after-close exact owned cancellation then complete fresh actual-account snapshot; exceptions and next-day ordinary catchup."},
        "research_caveat_list": [
            {"id_str": "CORE5-SELECTION-01", "priority_str": "P1_signoff_disclosure", "type_str": "research_evidence_limit", "confidence_str": "high",
                "finding_str": "Original chain has441 nominal trials and no untouched holdout; original DBC-short portability gates rejected overlay, owner retained baseline. A later fixed-borrow-cost study passed narrower economic gates. Neither establishes independent alpha validation.",
                "source_list": [str(reference_path) + ":68-72", "results/research/leakage_hunt_20260927/def/DEF_FINDINGS.md:151-155"],
                "action_str": "Owner signs it as selected risk-control hypothesis with this limitation; preserve future observations as forward evidence. Do not silently remove owner-kept overlay."},
            {"id_str": "CORE5-BORROW-02", "priority_str": "P1_postdeploy_acceptance", "type_str": "approved_operational_limit", "confidence_str": "high",
                "finding_str": "The1% borrow rate is fixed research cost, not account-specific historical/current truth. Locate/rate/recall and independent partial/rejected legs remain real risks;10% is an event target, not continuous exposure cap.",
                "source_list": ["strategies/taa_beyond_6040/strategy_taa_adaptive_macro_core5.py:333-389,607-647", "alpha/live/runner.py:3407-3411", "docs/live/DAILY_EXECUTION_RULES_20261007.md:96-121"],
                "action_str": "After authorized deployment, verify routed margin account, actual DBC tradability/statement fees and rejection/exception workflow. No recommendation to reinstate removed previews without explicit owner policy change."},
            {"id_str": "CORE5-LEDGER-03", "priority_str": "P2_disclosure", "type_str": "backtest_approximation", "confidence_str": "high",
                "finding_str": "Conservative25% withholding/zero cash interest coexist with optimistic unfinanced negative cash, fixed borrow and adjusted-share commissions. Old baseline reports248 negative-cash days, minimum-0.667%NAV. Adjusted whole-share rounding makes fees/vintage metrics weakly dependent on future corporate-action adjustments.",
                "source_list": ["alpha/engine/strategy.py:121-122,480-500", "results/research/strategy_readiness_audit_20260928/core5/CORE5_FINDINGS.md:281-332"],
                "action_str": "Keep exact current $1M ledger metrics beside prior audit bounds; don't change historical semantics during signoff. Old audit bounds/current source fingerprints must be labeled, not promoted to fresh reruns."}],
        "borrow_stress_saved_rows_list": borrow_row_list,
        "quant_pitfalls_dict": {"lookahead_str": "Features/prefix truncated atCloseT; trailing and recursive formulas reviewed; synthetic future-tail tests included.",
            "survivorship_str": "Fixed chosen ETFs, no current-stock-membership substitution. Historical ETF selection remains selection conditioning.",
            "training_scaling_str": "No fitted normalization, learned model or randomized train/test split in this strategy.",
            "regime_str": "Original robustness report supports risk control more than stable excess return; no fresh holdout is created by rerunning seen data.",
            "corporate_actions_str": "Separate TR signals/CAPITALSPECIAL fills/native dividends; adjusted-share rounding/fees are a disclosed residual.",
            "sample_size_str": "About19 years but many selected trials; years/synthetic cases are not independent strategy trials.",
            "live_backtest_str": "Full frozen-share/event/leg parity is conditional on account state; physical fills, fee posting/borrow/dividend timing are unproven offline."},
        "artifact_manifest_list": [manifest_dict(path_obj) for path_obj in input_path_list if path_obj.exists()]}
    native_replay_path = OUTPUT_PATH / "core5_native_replay_current.json"
    if not native_replay_path.exists():
        native_replay_path = OUTPUT_PATH / "core5_native_replay_20260911_independent_shocks.json"
    if native_replay_path.exists():
        native_replay_dict = read_json(native_replay_path)
        core_dict["current_code_saved_input_native_replay_dict"] = {
            key_str: value_obj for key_str, value_obj in native_replay_dict.items() if key_str != "evidence_list"}
        core_dict["current_code_saved_input_native_replay_dict"]["artifact_dict"] = manifest_dict(native_replay_path)
    current_baseline_path = OUTPUT_PATH / "baselines/strategy_taa_adaptive_macro_core5/summary.json"
    current_statistics_path = OUTPUT_PATH / "audit_statistics/strategy_taa_adaptive_macro_core5/statistics.json"
    if current_baseline_path.exists() and current_statistics_path.exists():
        current_baseline_dict = read_json(current_baseline_path)
        current_statistics_dict = read_json(current_statistics_path)
        core_dict["current_usd1m_evidence_dict"] = {"baseline_artifact_dict": manifest_dict(current_baseline_path),
            "statistics_artifact_dict": manifest_dict(current_statistics_path),
            "completion_evidence_dict": current_statistics_dict["baseline_completion_evidence"],
            "standalone_metrics_dict": current_statistics_dict["standalone_metrics"],
            "negative_cash_days_int": current_baseline_dict.get("negative_cash_days"),
            "min_cash_fraction_float": current_baseline_dict.get("min_cash_fraction"),
            "cost_sensitivities_dict": current_statistics_dict["cost_sensitivities"],
            "liquidity_dict": current_statistics_dict["liquidity"],
            "cost_stress_limit_str": "Additional fixed-ledger dollar debits only, with original trades and no sizing/cash/turnover/dividend feedback or lost opportunity return; not a full strategy rerun."}
        order_diagnostic_path = current_statistics_path.parent / "order_diagnostics.csv.gz"
        import gzip
        with gzip.open(order_diagnostic_path, "rt", encoding="utf-8", newline="") as handle_obj:
            bil_order_list = [row_dict for row_dict in csv.DictReader(handle_obj)
                if row_dict["asset"] == "BIL" and row_dict["adv_status"] == "available"]
        worst_bil_dict = max(bil_order_list, key=lambda row_dict: float(row_dict["constant_1m_participation"]))
        bil_liquidity_dict = next(row_dict for row_dict in current_statistics_dict["liquidity"]["period_details"]["asset_list"]
            if row_dict["asset"] == "BIL")
        core_dict["bil_capacity_evidence_dict"] = {
            "worst_execution_date_str": worst_bil_dict["execution_date"],
            "worst_decision_date_str": worst_bil_dict["decision_date"],
            **{name_str: float(worst_bil_dict[name_str]) for name_str in
                ("constant_1m_notional", "native_trailing20_median_turnover", "constant_1m_participation",
                 "actual_ledger_notional", "actual_participation")},
            "recent_3y_max_constant_1m_participation": bil_liquidity_dict["last3y"]["constant_1m_available_subset"]["max"],
            "endpoint_dict": bil_liquidity_dict["current_endpoint"],
            "interpretation_str": "Historical executed-leg daily-volume diagnostic. ETF primary-market and underlying liquidity can exceed secondary tape liquidity; neither this ratio nor recent endpoint liquidity proves the old auction fill or its cost."}
        core_dict["research_caveat_list"].append({
            "id_str": "CORE5-CAPACITY-04", "priority_str": "P2_historical_cost_qualification",
            "type_str": "historical_execution_cost_evidence_limit", "confidence_str": "high_computed_saved_ledger",
            "affected_variant_list": ["core5"],
            "finding_str": "On 2008-01-08 the BIL leg is $399,998.22 at fixed USD1M NAV versus $2,785,566 prior20-session median native Turnover:14.3597% of daily ADV. Original compounded notional is $438,464.90 (15.7406% ADV). The uniform historical2.5bps execution cost is not empirically established by this ledger. Recent3y fixed1M maximum BIL participation is only0.0399194% ADV, which does not validate the old fill cost.",
            "source_list": ["results/review/wired_signoff_20261015/audit_statistics/strategy_taa_adaptive_macro_core5/statistics.json:liquidity.period_details.asset_list[BIL]",
                "results/review/wired_signoff_20261015/audit_statistics/strategy_taa_adaptive_macro_core5/order_diagnostics.csv.gz:execution_date=2008-01-08,asset=BIL"],
            "action_str": "Retain2.5bps as a disclosed historical execution assumption requiring cost qualification, rather than calling the historical net return empirically executable. Daily secondary-market tape is not an opening-auction impact model or a hard ETF capacity ceiling: underlying/creation-redemption liquidity may support execution. Do not describe this fill as impossible, or recent liquidity as proof of old cost."})
        core_dict["artifact_manifest_list"].extend(manifest_dict(path_obj) for path_obj in
            (current_statistics_path, order_diagnostic_path))
    (OUTPUT_PATH / "core5_review.json").write_text(json.dumps(core_dict, indent=2, sort_keys=True), encoding="utf-8")
    test_path = OUTPUT_PATH / "ops_offline_tests.xml"
    test_dict = {"status_str": "running_or_unavailable"}
    if test_path.exists():
        test_root_obj = element_tree.parse(test_path).getroot()
        suite_obj = test_root_obj.find("testsuite") if test_root_obj.tag != "testsuite" else test_root_obj
        test_dict = {"status_str": "completed", **suite_obj.attrib, "artifact_dict": manifest_dict(test_path)}
    additional_test_list = []
    for additional_path in (OUTPUT_PATH / "ops_transport_alert_tests.xml", OUTPUT_PATH / "audit_statistics_tests.xml",
            OUTPUT_PATH / "ops_doctor_quote_probe_tests.xml", OUTPUT_PATH / "core5_quant_tests.xml",
            OUTPUT_PATH / "ops_cached_harness_review_tests.xml"):
        if additional_path.exists():
            additional_root_obj = element_tree.parse(additional_path).getroot()
            additional_suite_obj = additional_root_obj.find("testsuite") if additional_root_obj.tag != "testsuite" else additional_root_obj
            additional_test_list.append({**additional_suite_obj.attrib, "artifact_dict": manifest_dict(additional_path)})
    ops_dict = {"schema_version_str": "wired_signoff.review.v1", "generated_at_utc_str": datetime.now(UTC).isoformat(),
        "production_modified_bool": False, "broker_verified_bool": False, "tests_dict": test_dict,
        "audit_completion_counts_dict": completion_dict,
        "additional_test_list": additional_test_list,
        "findings_list": [
            {"id_str": "OPS-DOCTOR-DAILY-01", "priority_str": "P2_before_using_doctor_for_daily_signoff", "confidence_str": "high_reproduced",
                "finding_str": "Normal submitted CORE5/capsule intraday cycle is labeled persisted_lifecycle_expired/BLOCK by Doctor, while the daily contract keeps it open until target close.",
                "source_list": ["alpha/live/doctor.py:380-406,647-655", "docs/live/DAILY_EXECUTION_RULES_20261007.md:117"],
                "reproduction_path_str": "results/review/wired_signoff_20261015/ops_doctor_probe.json",
                "recommendation_str": "Make diagnostic lifecycle classification aware of daily submitted cycles and after-close reconciliation deadline. Keep expired unsent MOO blocked; never turn uncertain send into resubmit. Add intraday submitted/partial, earlyclose, overdue and monthly regression cases."},
            {"id_str": "OPS-DOCTOR-CORE5QUOTE-03", "priority_str": "P2_before_using_doctor_for_core5_signoff", "confidence_str": "high_reproduced",
                "finding_str": "Doctor blocks CORE5 if livequotes fail, while the actual runner correctly constructs its VPlan from approved frozen Close_T prices without requiring a quote.",
                "source_list": ["alpha/live/doctor.py:1290-1325", "alpha/live/runner.py:3046-3055"],
                "reproduction_path_str": "results/review/wired_signoff_20261015/ops_doctor_quote_probe.json",
                "recommendation_str": "Align Doctor preview with the current execution contract: use frozen Close_T prices and label optional livequote diagnostics. Reproduce broker-free runner/Doctor parity and preserve ordinary strategies' quote guards."},
            {"id_str": "OPS-DOC-02", "priority_str": "P2_before_operator_handoff", "confidence_str": "high_source_verified",
                "finding_str": "Operator entry/gap/wiring docs mix removed funding attestation and old manual-recovery rules with current daily replay/receipt/catchup policy.",
                "source_list": ["LIVE_START_HERE.md:59-63", "ASSUMPTIONS_AND_GAPS.md:256", "docs/live/CORE5_WIRING_20261005.md:117-127", "docs/live/DAILY_EXECUTION_RULES_20261007.md:32-57,96-101"],
                "recommendation_str": "Replace stale instructions with authoritative daily policy and actual removed previews; preserve dated old test results as historical evidence."}],
        "existing_coverage_list": ["Doctor config/release/account/provenance/position/quote/VPlan guards",
            "Inspector source/consumer staleness and missed window", "V4 per-request signed fill and exactACK identity proof",
            "V4 daily partly sent/not sent/unconfirmed/completed-with-exceptions states", "Read-only data provenance and copy-only DO NEXT commands",
            "Daily exact cross-client ownership/cancellation, restart/idempotency, post-close transaction rollback and monthly scheduler isolation"],
        "enhancement_priority_list": [
            {"priority_str": "required_if_doctor_is_acceptance_gate", "change_str": "Daily lifecycle and CORE5 frozen-reference diagnostic distinctions from OPS-DOCTOR-DAILY-01 and OPS-DOCTOR-CORE5QUOTE-03"},
            {"priority_str": "required_operator_handoff", "change_str": "Repair contradictory instructions from OPS-DOC-02"},
            {"priority_str": "recommended_report_integration", "change_str": "Separate research signoff, offline execution parity and current deployed account acceptance; show revision/date/evidence and conditions. Operational green never substitutes for research approval."},
            {"priority_str": "optional", "change_str": "Family grouping and concise signoff cards linked to existing evidence; no new dashboard control or execution path needed."}],
        "not_proven_list": ["Actual deployed revision/config/VPS health", "Current account routing/margin/permissions/borrow", "Real auction execution/partial rejection rates/slippage", "Broker statement cash/dividend/borrow reconciliation"],
        "no_mutation_boundary_str": "Audit reads production code and writes only isolated audit scripts/results; no actual Doctor, Norgate client sync, broker, scheduler, notification, release or deploy command run."}
    (OUTPUT_PATH / "ops_review.json").write_text(json.dumps(ops_dict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"core5_old_source_hash_matches_int": sum(row_dict["matches_bool"] for row_dict in source_match_list),
        "core5_old_source_hash_total_int": len(source_match_list), "old_input_price_hash_matches_bool": snapshot_match_bool,
        "tests_dict": test_dict}, indent=2))


if __name__ == "__main__":
    main()
