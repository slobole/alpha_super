"""Prevent green audit labels from incomplete or misleading evidence receipts."""
from build_signoff_report import compact_liquidity_dict, execution_verified_bool, final_baseline_bool, hash_file, signoff_status_dict, verify_replay_sources, verify_saved_manifest
import pytest


def test_an_interim_backtest_pass_cannot_be_a_final_signature():
    assert not final_baseline_bool({"status": "pass", "capital": 1_000_000})


def test_same_capital_ratio_does_not_prove_requested_capital_was_used():
    receipt_dict = {"status": "pass", "deterministic_equity": True,
        "deterministic_economic_transactions": True, "capital_honored": True,
        "actual_capital_base": 100_000, "actual_double_capital_base": 200_000}
    assert not final_baseline_bool(receipt_dict)


def test_cash_only_months_cannot_satisfy_informative_month_requirement():
    receipt_dict = {"mechanics_pass_bool": True, "exact_int": 24, "attempted_int": 24, "informative_exact_int": 0}
    assert not execution_verified_bool("ndx", receipt_dict)


def test_daily_empty_pass_or_partial_replay_cannot_be_signed():
    assert not execution_verified_bool("mr", {"status_str": "pass"})
    assert not execution_verified_bool("capsules", {"status_str": "pass", "session_count_int": 24, "passed_session_count_int": 23})


def test_core5_receipt_uses_its_actual_status_schema():
    assert execution_verified_bool("core5", {"status_str": "passed", "native_host_session_count_int": 45,
        "independent_nav_quote_invariance_case_count_int": 180})


def test_daily_requires_real_orders_state_timestamps_and_date_control():
    row_dict = {"passed_bool": True, "strategy_state_equal_bool": True, "signal_close_timestamp_equal_bool": True,
        "execution_timestamp_equal_bool": True, "wrong_execution_date_positive_control_caught_bool": True}
    receipt_dict = {"status_str": "pass", "session_count_int": 24, "passed_session_count_int": 24,
        "coverage_dict": {"with_order_intent": 20}, "row_list": [row_dict.copy() for _ in range(24)]}
    assert execution_verified_bool("mr", receipt_dict)
    receipt_dict["row_list"][0]["strategy_state_equal_bool"] = None
    assert not execution_verified_bool("mr", receipt_dict)
    receipt_dict["row_list"][0]["strategy_state_equal_bool"] = True
    receipt_dict["coverage_dict"]["with_order_intent"] = 0
    assert not execution_verified_bool("capsules", receipt_dict)


def test_mechanical_parity_never_erases_confirmed_strategy_blockers():
    assert "חסום" in signoff_status_dict("dv2", "mr", True)["execution_status"]
    assert "חסום" in signoff_status_dict("taa_rank_tqqq", "taa", True)["execution_status"]
    assert "חסום" in signoff_status_dict("hpi_spmo", "capsules", True)["research_status"]
    assert "חסום" not in signoff_status_dict("dv2_ind", "mr", True)["execution_status"]


def test_changed_chart_or_statistics_code_cannot_publish(tmp_path):
    evidence_path = tmp_path / "chart.csv"
    evidence_path.write_text("value\n1\n", encoding="utf-8")
    receipt_dict = {"path": str(evidence_path), "sha256": hash_file(evidence_path)}
    verify_saved_manifest(receipt_dict)
    evidence_path.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_saved_manifest(receipt_dict)


def test_replay_requires_original_input_and_exact_archived_checker(tmp_path):
    input_path = tmp_path / "inputs"
    replay_path = tmp_path / "replay" / "replay.json"
    input_path.mkdir()
    replay_path.parent.mkdir()
    receipt_dict = {"source_unchanged_bool": True, "source_hash_dict": {}}
    for field_str, filename_str in (("input_sha256_str", "pricing.parquet"),
            ("auxiliary_sha256_str", "auxiliary.pkl.gz"), ("trace_sha256_str", "trace.pkl.gz")):
        (input_path / filename_str).write_bytes(b"original")
        receipt_dict[field_str] = hash_file(input_path / filename_str)
    archive_path = replay_path.parent / "executed_replay_checker.py"
    archive_path.write_bytes(b"exact executed checker")
    receipt_dict["source_hash_dict"]["scripts/research/absent/mr_native_replay.py"] = hash_file(archive_path)
    verify_replay_sources(receipt_dict, input_path, replay_path)
    (input_path / "pricing.parquet").write_bytes(b"changed input")
    with pytest.raises(ValueError):
        verify_replay_sources(receipt_dict, input_path, replay_path)
    (input_path / "pricing.parquet").write_bytes(b"original")
    archive_path.write_bytes(b"different checker")
    with pytest.raises(ValueError):
        verify_replay_sources(receipt_dict, input_path, replay_path)


def test_report_compaction_preserves_parking_legs_and_totals():
    source_dict = {"order_count": 1234, "missing_adv_order_count": 9, "period_details": {"asset_list": [
        {"asset": f"stock{index_int}", "last3y": {"constant_1m_available_subset": {"p99": index_int}},
            "full": {"constant_1m_available_subset": {"max": index_int}}} for index_int in range(30)] + [{"asset": "BIL"}, {"asset": "SPMO"}]}}
    compact_dict = compact_liquidity_dict(source_dict)
    assert compact_dict["order_count"] == 1234 and compact_dict["missing_adv_order_count"] == 9
    assert {"BIL", "SPMO", "stock29"}.issubset({row_dict["asset"] for row_dict in compact_dict["period_details"]["asset_list"]})
    assert len(source_dict["period_details"]["asset_list"]) == 32


def _owner_review_fixture_tuple():
    import json
    from build_signoff_report import ROOT_PATH
    report_dict = json.loads((ROOT_PATH / "owner_review_20261016/prior/audit_report.json").read_text(encoding="utf-8"))
    owner_review_dict = json.loads((ROOT_PATH / "owner_review_20261016/owner_decisions.json").read_text(encoding="utf-8"))
    return report_dict, owner_review_dict


def test_owner_acceptance_preserves_results_original_failures_and_deployment_status():
    from build_signoff_report import apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    reviewed_dict = apply_owner_review(report_dict, owner_review_dict)
    assert reviewed_dict["tests"] == report_dict["tests"]
    assert len(reviewed_dict["findings"]) == len(report_dict["findings"]) == 30
    for before_dict, after_dict in zip(report_dict["findings"], reviewed_dict["findings"]):
        assert after_dict["technical"] == before_dict["technical"]
        assert after_dict["priority"] == before_dict["priority"]
    for before_dict, after_dict in zip(report_dict["cards"], reviewed_dict["cards"]):
        for field_str in ("metrics", "benchmark", "subperiods", "liquidity", "costs", "accounting", "replay", "baseline_evidence", "chart", "deployment_status"):
            assert after_dict[field_str] == before_dict[field_str]
    assert "owner_review" not in report_dict


def test_owner_decisions_are_scoped_and_do_not_claim_technical_fixes():
    from build_signoff_report import apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    reviewed_dict = apply_owner_review(report_dict, owner_review_dict)
    finding_dict = {item_dict["id"]: item_dict for item_dict in reviewed_dict["findings"]}
    card_dict = {item_dict["variant"]: item_dict for item_dict in reviewed_dict["cards"]}
    assert finding_dict["CORE5-SELECTION-01"]["owner_disposition"]["status"] == "accepted"
    assert finding_dict["TAA-HELPER-01"]["owner_disposition"]["status"] == "accepted"
    assert finding_dict["MR-001"]["owner_disposition"]["status"] == "deferred"
    assert finding_dict["CORE5-BORROW-02"]["owner_disposition"]["status"] == "deferred"
    assert finding_dict["MR-010"]["owner_disposition"]["status"] == "historical_disclosure"
    assert "owner_disposition" not in finding_dict["NDX-VXN-02"]
    assert "עלויות עבר מסויגות" in card_dict["dv2_spmo"]["research_status"]
    assert "התקבלה" in card_dict["core5"]["research_status"]
    assert any(item_dict["id"] == "NEXT-CURRENT-CAPACITY" for item_dict in reviewed_dict["remaining_items"])


def test_owner_acceptance_cannot_replace_missing_replay_or_baseline_evidence():
    from build_signoff_report import apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    for card_dict in report_dict["cards"]:
        if card_dict["variant"] == "taa_rank_tqqq":
            card_dict["replay"] = {}
        if card_dict["variant"] == "core5":
            card_dict["baseline_evidence"]["deterministic_equity"] = False
    reviewed_dict = apply_owner_review(report_dict, owner_review_dict)
    card_dict = {item_dict["variant"]: item_dict for item_dict in reviewed_dict["cards"]}
    assert card_dict["taa_rank_tqqq"]["execution_status"] == "חסרות ראיות"
    assert card_dict["core5"]["research_status"] == "חסרות ראיות לשחזור"
    assert all("טרם נבדק" in item_dict["deployment_status"] for item_dict in reviewed_dict["cards"])


def test_owner_review_rejects_unknown_scope_and_is_idempotent():
    import copy
    from build_signoff_report import apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    reviewed_dict = apply_owner_review(report_dict, owner_review_dict)
    assert apply_owner_review(reviewed_dict, owner_review_dict) == reviewed_dict
    for field_str, value_str in (("id", "NOT-A-FINDING"), ("status", "fixed")):
        invalid_review_dict = copy.deepcopy(owner_review_dict)
        invalid_review_dict["decisions"][0][field_str] = value_str
        with pytest.raises(ValueError):
            apply_owner_review(report_dict, invalid_review_dict)



def test_daily_acceptance_keeps_legacy_parents_open_and_core5_separate():
    from build_signoff_report import apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    reviewed_dict = apply_owner_review(report_dict, owner_review_dict)
    finding_dict = {item_dict["id"]: item_dict for item_dict in reviewed_dict["findings"]}
    card_dict = {item_dict["variant"]: item_dict for item_dict in reviewed_dict["cards"]}
    decision_dict = finding_dict["MR-002"]["owner_disposition"]
    accepted_mr_set = {"dv2_ind", "dv2_cash", "dv2_bil", "dv2_spmo", "hpi_cash", "hpi_bil", "hpi_spmo"}
    legacy_set = {"dv2", "hpi235"}
    assert decision_dict["status"] == "partially_accepted"
    assert set(decision_dict["accepted_variant_list"]) == accepted_mr_set
    assert set(decision_dict["open_variant_list"]) == legacy_set
    assert set(finding_dict["MR-002"]["affected"]) == accepted_mr_set | legacy_set
    assert finding_dict["CORE5-BORROW-02"]["owner_disposition"]["status"] == "deferred"
    for variant_str in accepted_mr_set | {"core5"}:
        assert card_dict[variant_str]["same_auction_signoff_status"] == "accepted_residual_daily_risk"
        assert not any(item_str.startswith("סיכון החלפת משבצת שטרם הוכרע") for item_str in card_dict[variant_str]["conditions"])
    for variant_str in legacy_set:
        assert card_dict[variant_str]["same_auction_signoff_status"] == "open_legacy_risk"
        assert any(item_str.startswith("סיכון החלפת משבצת שטרם הוכרע") for item_str in card_dict[variant_str]["conditions"])
    remaining_dict = next(item_dict for item_dict in reviewed_dict["remaining_items"] if item_dict["id"] == "NEXT-SAME-AUCTION")
    assert set(remaining_dict["affected"]) == legacy_set
    assert all("same_auction_signoff_status" not in item_dict for item_dict in reviewed_dict["cards"] if item_dict["family"] in {"taa", "ndx"})
    assert all("טרם נבדק" in item_dict["deployment_status"] for item_dict in reviewed_dict["cards"])


def test_partial_acceptance_rejects_overlap_omission_and_unknown_variants():
    import copy
    from build_signoff_report import apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    for variant_list in (["dv2"], ["dv2_ind"], ["NOT-A-STRATEGY"]):
        invalid_dict = copy.deepcopy(owner_review_dict)
        decision_dict = next(item_dict for item_dict in invalid_dict["decisions"] if item_dict["id"] == "MR-002")
        decision_dict["accepted_variant_list"] = variant_list
        with pytest.raises(ValueError):
            apply_owner_review(report_dict, invalid_dict)
    invalid_dict = copy.deepcopy(owner_review_dict)
    invalid_dict["same_auction_followup"]["accepted_daily_variant_list"].append("dv2")
    with pytest.raises(ValueError):
        apply_owner_review(report_dict, invalid_dict)
    invalid_dict = copy.deepcopy(owner_review_dict)
    invalid_dict["remaining_items"][0]["affected"] = ["NOT-A-STRATEGY"]
    with pytest.raises(ValueError):
        apply_owner_review(report_dict, invalid_dict)


def test_acceptance_preserves_component_test_limits_and_earlier_proof():
    import json
    from build_signoff_report import ROOT_PATH, apply_owner_review
    report_dict, owner_review_dict = _owner_review_fixture_tuple()
    before_dict = json.loads((ROOT_PATH / "daily_risk_acceptance_20261016/before_audit_report.json").read_text(encoding="utf-8"))
    reviewed_dict = apply_owner_review(before_dict, owner_review_dict)
    assert reviewed_dict["same_auction_followup"]["evidence"] == before_dict["same_auction_followup"]["evidence"]
    assert reviewed_dict["same_auction_followup"]["evidence"]["live_broker_checked"] is False
    assert reviewed_dict["same_auction_followup"]["evidence"]["full_failed_exit_filled_buy_11_holdings_real_next_day_host_tested"] is False
    assert reviewed_dict["tests"] == before_dict["tests"]
    assert reviewed_dict["implementation_followup"] == before_dict["implementation_followup"]
    assert apply_owner_review(reviewed_dict, owner_review_dict) == reviewed_dict
