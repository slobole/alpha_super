from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, DecisionPlan, LivePriceSnapshot, LiveRelease
from alpha.live.state_store_v2 import LiveStateStore


def _capsule_inputs():
    signal_timestamp_ts = datetime(2026, 10, 1, 20, tzinfo=timezone.utc)
    execution_timestamp_ts = datetime(2026, 10, 2, 13, 30, tzinfo=timezone.utc)
    release_obj = LiveRelease(
        release_id_str="capsule.v1", user_id_str="owner", pod_id_str="capsule",
        account_route_str="DU123", strategy_import_str="strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil",
        mode_str="paper", session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str="capsule_test", params_dict={},
        risk_profile_str="standard", enabled_bool=False, source_path_str="test.yaml",
        pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
    )
    decision_plan_obj = DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=signal_timestamp_ts, submission_timestamp_ts=execution_timestamp_ts,
        target_execution_timestamp_ts=execution_timestamp_ts, execution_policy_str="next_open_moo",
        decision_base_position_map={"KEEP": 50.0, "BIL": 500.0}, strategy_state_dict={},
        snapshot_metadata_dict={
            "sizing_contract_str": "mr_capsule_close_targets_v1",
            "strategy_import_str": release_obj.strategy_import_str,
            "decision_nav_float": 100_000.0, "decision_cash_float": 5_000.0,
            "mr_capsule_parking_mode_str": "bil",
        },
        entry_target_weight_map_dict={"AAPL": 0.1}, entry_priority_list=["AAPL"],
        target_share_map_dict={"BIL": 400.0}, decision_plan_id_int=7,
    )
    broker_snapshot_obj = BrokerSnapshot(
        account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=execution_timestamp_ts,
        cash_float=5_000.0, total_value_float=120_000.0, net_liq_float=120_000.0,
        position_amount_map={"KEEP": 50.0, "BIL": 500.0},
    )
    live_price_snapshot_obj = LivePriceSnapshot(
        account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=execution_timestamp_ts,
        price_source_str="test_quote", asset_reference_price_map={"AAPL": 200.0, "BIL": 95.0},
    )
    return release_obj, decision_plan_obj, broker_snapshot_obj, live_price_snapshot_obj


def test_mixed_stock_value_and_parking_share_targets_preserve_close_nav_and_broker_deltas():
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    vplan_obj = build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)

    # 10% * $100K Close_T NAV / $200 quote = 50, rather than 60 from submit NAV.
    assert vplan_obj.target_share_map == {"AAPL": 50.0, "BIL": 400.0}
    assert vplan_obj.order_delta_map == {"AAPL": 50.0, "BIL": -100.0}
    assert "KEEP" not in vplan_obj.target_share_map
    assert vplan_obj.current_broker_position_map["KEEP"] == 50.0
    assert vplan_obj.pod_budget_float == 120_000.0  # Actual broker budget remains truthful.
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    assert {request_obj.asset_str: request_obj.amount_float for request_obj in request_list} == {"AAPL": 50.0, "BIL": -100.0}
    assert all(request_obj.broker_order_type_str == "MOO" for request_obj in request_list)
    assert all(request_obj.unit_str == "shares" and not request_obj.target_bool for request_obj in request_list)


@pytest.mark.parametrize("bil_target_int,expected_delta_int", [(600, 100), (400, -100), (500, 0), (0, -500)])
def test_absolute_target_resizes_in_both_directions_and_zero_delta_sends_no_order(bil_target_int, expected_delta_int):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    decision_plan_obj = replace(decision_plan_obj, target_share_map_dict={"BIL": bil_target_int})
    for bil_quote_float in (80.0, 110.0):
        quote_obj = replace(quote_obj, asset_reference_price_map={"AAPL": 200.0, "BIL": bil_quote_float})
        vplan_obj = build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)
        assert vplan_obj.target_share_map["BIL"] == bil_target_int
        assert vplan_obj.order_delta_map["BIL"] == expected_delta_int
        request_list = build_broker_order_request_list_from_vplan(vplan_obj)
        assert any(request_obj.asset_str == "BIL" for request_obj in request_list) is (expected_delta_int != 0)


@pytest.mark.parametrize("target_float", [-1, 1.5, float("nan"), float("inf"), -float("inf")])
def test_invalid_share_target_is_rejected_at_the_model_boundary(target_float):
    _, decision_plan_obj, _, _ = _capsule_inputs()
    with pytest.raises(ValueError, match="nonnegative whole shares"):
        replace(decision_plan_obj, target_share_map_dict={"BIL": target_float})


@pytest.mark.parametrize("intent_dict", [{"entry_target_weight_map_dict": {"BIL": 0.1}}, {"exit_asset_set": {"BIL"}}])
def test_ambiguous_targets_fail_instead_of_selecting_one_intent(intent_dict):
    _, decision_plan_obj, _, _ = _capsule_inputs()
    with pytest.raises(ValueError, match="overlap"):
        replace(decision_plan_obj, **intent_dict)


def test_full_target_book_cannot_contain_explicit_share_targets():
    _, decision_plan_obj, _, _ = _capsule_inputs()
    with pytest.raises(ValueError, match="require incremental"):
        replace(decision_plan_obj, decision_book_type_str="full_target_weight_book")


def test_cash_capsule_cannot_bypass_state_checks_through_full_target_family():
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    decision_plan_obj = replace(decision_plan_obj, target_share_map_dict={}, decision_book_type_str="full_target_weight_book")
    with pytest.raises(ValueError, match="requires an incremental"):
        build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)


@pytest.mark.parametrize("plan_change_dict", [
    {"execution_policy_str": "same_day_moc"},
    {"preserve_untouched_positions_bool": False},
    {"rebalance_omitted_assets_to_zero_bool": True},
])
def test_capsule_cannot_change_timing_or_omitted_position_semantics(plan_change_dict):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    with pytest.raises(ValueError, match="requires an incremental"):
        build_vplan(release_obj, replace(decision_plan_obj, **plan_change_dict), broker_snapshot_obj, quote_obj)


@pytest.mark.parametrize("fraction_float", [0.03, 0.5, float("nan"), 1.01])
def test_capsule_rejects_budget_fraction_that_could_mix_scaled_stocks_and_full_parking(fraction_float):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    with pytest.raises(ValueError, match="full-account"):
        build_vplan(replace(release_obj, pod_budget_fraction_float=fraction_float), decision_plan_obj, broker_snapshot_obj, quote_obj)


@pytest.mark.parametrize("position_dict", [
    {"KEEP": 50.0, "BIL": 499.0},  # Partial fill/manual trade.
    {"KEEP": 50.0, "BIL": 1_000.0},  # Split since the frozen decision.
    {"KEEP": 50.0, "BIL": 500.5},
    {"KEEP": 50.0, "BIL": -1.0},
    {"KEEP": 50.0, "BIL": float("nan")},
])
def test_broker_position_drift_or_invalid_units_block_stale_targets(position_dict):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    with pytest.raises(ValueError, match="positions"):
        build_vplan(release_obj, decision_plan_obj, replace(broker_snapshot_obj, position_amount_map=position_dict), quote_obj)


@pytest.mark.parametrize("cash_float", [5_001.0, 4_999.0, 0.0, -1_000.0, 50_000.0])
def test_finite_cash_changes_preserve_frozen_sizing(cash_float):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    baseline_vplan_obj = build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)
    changed_nav_float = broker_snapshot_obj.net_liq_float + cash_float - broker_snapshot_obj.cash_float
    changed_snapshot_obj = replace(
        broker_snapshot_obj, cash_float=cash_float,
        net_liq_float=changed_nav_float, total_value_float=changed_nav_float,
    )
    changed_vplan_obj = build_vplan(release_obj, decision_plan_obj, changed_snapshot_obj, quote_obj)
    assert changed_vplan_obj.target_share_map == baseline_vplan_obj.target_share_map
    assert changed_vplan_obj.order_delta_map == baseline_vplan_obj.order_delta_map
    assert changed_vplan_obj.net_liq_float == changed_nav_float
    assert decision_plan_obj.snapshot_metadata_dict["decision_cash_float"] == 5_000.0
    assert decision_plan_obj.snapshot_metadata_dict["decision_nav_float"] == 100_000.0
    changed_request_list = build_broker_order_request_list_from_vplan(changed_vplan_obj)
    assert all(request_obj.portfolio_value_float == changed_nav_float for request_obj in changed_request_list)
    # Broker NAV is audit metadata; every order field and request identity stays fixed.
    assert [replace(request_obj, portfolio_value_float=baseline_vplan_obj.net_liq_float) for request_obj in changed_request_list] == build_broker_order_request_list_from_vplan(baseline_vplan_obj)


@pytest.mark.parametrize("cash_float", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_broker_cash_still_blocks_frozen_decision(cash_float):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    with pytest.raises(ValueError, match="finite broker cash"):
        build_vplan(release_obj, decision_plan_obj, replace(broker_snapshot_obj, cash_float=cash_float), quote_obj)


@pytest.mark.parametrize("metadata_change_dict", [
    {"sizing_contract_str": ""}, {"strategy_import_str": "another_strategy"},
    {"mr_capsule_parking_mode_str": "spmo"}, {"decision_nav_float": 0},
    {"decision_nav_float": float("nan")}, {"decision_cash_float": float("inf")},
])
def test_capsule_requires_proven_identity_sizing_basis_and_parking_mode(metadata_change_dict):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    decision_plan_obj = replace(decision_plan_obj, snapshot_metadata_dict={**decision_plan_obj.snapshot_metadata_dict, **metadata_change_dict})
    with pytest.raises(ValueError, match="MR capsule"):
        build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)


@pytest.mark.parametrize("field_str", ["release_id_str", "user_id_str", "pod_id_str", "account_route_str"])
def test_decision_identity_must_match_release(field_str):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    with pytest.raises(ValueError, match="identity mismatch"):
        build_vplan(release_obj, replace(decision_plan_obj, **{field_str: "wrong"}), broker_snapshot_obj, quote_obj)


@pytest.mark.parametrize("snapshot_kind_str", ["broker", "quote"])
def test_snapshot_account_route_must_match_release(snapshot_kind_str):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    if snapshot_kind_str == "broker":
        broker_snapshot_obj = replace(broker_snapshot_obj, account_route_str="DU_OTHER")
    else:
        quote_obj = replace(quote_obj, account_route_str="DU_OTHER")
    with pytest.raises(ValueError, match="account routes"):
        build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)


@pytest.mark.parametrize("price_float", [0.0, -1.0, float("nan"), float("inf")])
def test_fixed_target_still_requires_finite_positive_quote(price_float):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    quote_obj = replace(quote_obj, asset_reference_price_map={"AAPL": 200.0, "BIL": price_float})
    with pytest.raises(ValueError, match="finite and positive"):
        build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)


def test_missing_quote_or_outstanding_order_blocks_new_capsule_batch():
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    with pytest.raises(ValueError, match="Missing live reference"):
        build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, replace(quote_obj, asset_reference_price_map={"AAPL": 200.0}))
    with pytest.raises(ValueError, match="outstanding broker orders"):
        build_vplan(release_obj, decision_plan_obj, replace(broker_snapshot_obj, open_order_id_list=["previous_order"]), quote_obj)


def test_only_mode_approved_etfs_accept_fixed_targets():
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    for target_map_dict in ({"SPMO": 5.0}, {"AAPL": 5.0}):
        invalid_plan_obj = replace(decision_plan_obj, entry_target_weight_map_dict={}, target_weight_map={}, target_share_map_dict=target_map_dict)
        with pytest.raises(ValueError, match="outside the parking mode"):
            build_vplan(release_obj, invalid_plan_obj, broker_snapshot_obj, quote_obj)
    release_obj = replace(release_obj, strategy_import_str=release_obj.strategy_import_str.removesuffix("bil") + "spmo")
    decision_plan_obj = replace(decision_plan_obj,
        snapshot_metadata_dict={**decision_plan_obj.snapshot_metadata_dict, "strategy_import_str": release_obj.strategy_import_str, "mr_capsule_parking_mode_str": "spmo"},
        target_share_map_dict={"SPMO": 5.0, "BIL": 400.0})
    quote_obj = replace(quote_obj, asset_reference_price_map={**quote_obj.asset_reference_price_map, "SPMO": 100.0})
    assert build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj).target_share_map["SPMO"] == 5.0


def test_cash_stage_uses_same_close_nav_contract_without_parking():
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    release_obj = replace(release_obj, strategy_import_str=release_obj.strategy_import_str.removesuffix("bil") + "cash")
    decision_plan_obj = replace(decision_plan_obj,
        snapshot_metadata_dict={**decision_plan_obj.snapshot_metadata_dict, "strategy_import_str": release_obj.strategy_import_str, "mr_capsule_parking_mode_str": "cash"},
        decision_base_position_map={"KEEP": 50.0}, target_share_map_dict={})
    broker_snapshot_obj = replace(broker_snapshot_obj, position_amount_map={"KEEP": 50.0})
    assert build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj).target_share_map == {"AAPL": 50.0}


def test_ordinary_incremental_strategy_keeps_fresh_broker_budget_sizing():
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    release_obj = replace(release_obj, strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy", pod_budget_fraction_float=0.5)
    with pytest.raises(ValueError, match="only for MR capsule"):
        build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj)
    decision_plan_obj = replace(decision_plan_obj, target_share_map_dict={})
    assert build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj).target_share_map == {"AAPL": 30.0}


def test_target_roundtrip_restart_preserves_request_keys_and_atomic_claim(tmp_path):
    release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj = _capsule_inputs()
    db_path_str = str(tmp_path / "capsule.sqlite3")
    state_store_obj = LiveStateStore(db_path_str)
    decision_plan_obj = state_store_obj.insert_decision_plan(decision_plan_obj)
    vplan_obj = state_store_obj.insert_vplan(build_vplan(release_obj, decision_plan_obj, broker_snapshot_obj, quote_obj))
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    restarted_store_obj = LiveStateStore(db_path_str)
    restored_plan_obj = restarted_store_obj.get_decision_plan_by_id(decision_plan_obj.decision_plan_id_int)
    restored_vplan_obj = restarted_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int)
    assert restored_plan_obj.target_share_map_dict == {"BIL": 400.0}
    assert build_broker_order_request_list_from_vplan(restored_vplan_obj) == request_list
    assert restarted_store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    assert not state_store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)


def test_old_decision_database_migrates_without_changing_existing_intents(tmp_path):
    _, decision_plan_obj, _, _ = _capsule_inputs()
    db_path_str = str(tmp_path / "old.sqlite3")
    state_store_obj = LiveStateStore(db_path_str)
    decision_plan_obj = state_store_obj.insert_decision_plan(replace(decision_plan_obj, target_share_map_dict={}))
    with sqlite3.connect(db_path_str) as connection_obj:
        connection_obj.execute("ALTER TABLE decision_plan DROP COLUMN target_share_json_str")
    migrated_store_obj = LiveStateStore(db_path_str)
    restored_plan_obj = migrated_store_obj.get_decision_plan_by_id(decision_plan_obj.decision_plan_id_int)
    assert restored_plan_obj.target_share_map_dict == {}
    assert restored_plan_obj.entry_target_weight_map_dict == {"AAPL": 0.1}
    assert restored_plan_obj.snapshot_metadata_dict == decision_plan_obj.snapshot_metadata_dict


def test_operator_decision_display_and_trace_show_explicit_parking_target(tmp_path, monkeypatch):
    from alpha.live import runner

    _, decision_plan_obj, _, _ = _capsule_inputs()
    decision_plan_obj = replace(decision_plan_obj, entry_target_weight_map_dict={}, target_weight_map={})
    state_store_obj = LiveStateStore(str(tmp_path / "display.sqlite3"))
    decision_plan_obj = state_store_obj.insert_decision_plan(decision_plan_obj)
    monkeypatch.setattr(runner, "_load_release_list_and_sync", lambda *argument_tuple, **keyword_dict: [])
    detail_dict = runner.show_decision_plan_summary(
        state_store_obj, decision_plan_obj.signal_timestamp_ts, str(tmp_path),
        decision_plan_id_int=decision_plan_obj.decision_plan_id_int,
    )
    assert detail_dict["decision_plan_dict_list"][0]["target_share_map_dict"] == {"BIL": 400.0}
    display_str = runner._render_decision_plan_detail_str(detail_dict)
    assert "BIL | shares=400" in display_str
    assert "- Targets: none" not in display_str
    assert runner._decision_plan_trace_payload_dict(decision_plan_obj)["target_share_map_dict"] == {"BIL": 400.0}
