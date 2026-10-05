"""Offline reviewed-resume service tests; broker order submission is forbidden."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from alpha.live import core5_adapter, core5_resume
from alpha.live.models import BrokerSnapshot
from alpha.live.state_store_v2 import LiveStateStore
import test_live_core5_adapter as fixture_module


class ResumeBroker:
    def __init__(self, release_obj, state_obj, as_of_ts):
        self.snapshot_obj = BrokerSnapshot(release_obj.account_route_str, as_of_ts,
            state_obj.cash_float, state_obj.total_value_float, state_obj.position_amount_map,
            net_liq_float=state_obj.total_value_float)
        self.account_read_count_int = 0
        self.absence_complete_bool = True
        self.after_read_fn = None

    def get_core5_account_snapshot(self, account_route_str):
        self.account_read_count_int += 1
        if self.after_read_fn is not None:
            self.after_read_fn()
        return self.snapshot_obj

    def get_capsule_order_state_snapshot(self, *args, **kwargs):
        return [], [], []

    def get_refreshed_order_evidence(self, account_route_str, since_timestamp_ts):
        return {"account_route_str": account_route_str, "source_str": "offline_test",
            "open_orders_complete_bool": True, "completed_orders_complete_bool": self.absence_complete_bool,
            "executions_complete_bool": True, "refresh_started_timestamp_str": self.snapshot_obj.snapshot_timestamp_ts.isoformat(),
            "refreshed_timestamp_str": self.snapshot_obj.snapshot_timestamp_ts.isoformat(),
            "coverage_since_timestamp_str": since_timestamp_ts.isoformat(), "order_row_list": [], "execution_row_list": []}

    def submit_order(self, *args, **kwargs):
        raise AssertionError("The resume service must never submit broker orders.")


@pytest.fixture
def context_obj(tmp_path, monkeypatch):
    release_obj = fixture_module.release_obj.__wrapped__()
    pricing_df = fixture_module.price_df.__wrapped__()
    state_obj = fixture_module._state(release_obj, "2026-09-11", {"BIL": 100}, {"last_signal_date_str": "2026-08-03"})
    as_of_ts = fixture_module._time("2026-09-14", 9)
    store_obj = LiveStateStore(str(tmp_path / "resume.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(state_obj)
    metadata_dict = {"norgate_snapshot_date_str": "2026-09-11", "norgate_data_profile_str": "norgate_eod_core5",
        "norgate_manifest_hash_str": "fixture_hash"}
    def builder_fn(release_obj, as_of_ts, state_obj):
        return core5_adapter.build_core5_resync_decision_from_prices(release_obj, as_of_ts,
            state_obj, pricing_df, metadata_dict)
    monkeypatch.setattr(core5_resume.core5_adapter, "build_core5_resync_decision_plan", builder_fn)
    clock_list = [as_of_ts]
    monkeypatch.setattr(core5_resume, "_utc_now_ts", lambda: clock_list[0])
    return SimpleNamespace(release_obj=release_obj, state_obj=state_obj, store_obj=store_obj,
        broker_obj=ResumeBroker(release_obj, state_obj, as_of_ts), as_of_ts=as_of_ts,
        clock_list=clock_list, metadata_dict=metadata_dict, pricing_df=pricing_df)


def _preview(context_obj):
    return core5_resume.preview_core5_resume(context_obj.store_obj, context_obj.broker_obj,
        context_obj.release_obj, context_obj.as_of_ts)


def _apply(context_obj, review_hash_str, **kwargs):
    return core5_resume.apply_core5_resume(context_obj.store_obj, context_obj.broker_obj,
        context_obj.release_obj, context_obj.as_of_ts, review_hash_str=review_hash_str,
        **{"reason_str": "Reviewed the missed session and terminal broker evidence.", "operator_str": "test_operator", **kwargs})


def _old_cycle(context_obj, date_str="2026-09-11", status_str="ready"):
    state_obj = fixture_module._state(context_obj.release_obj, date_str, {"BIL": 100})
    decision_obj = core5_adapter.build_core5_resync_decision_from_prices(context_obj.release_obj,
        fixture_module._time(date_str), state_obj, context_obj.pricing_df,
        {**context_obj.metadata_dict, "norgate_snapshot_date_str": date_str})
    decision_obj = context_obj.store_obj.insert_decision_plan(decision_obj)
    vplan_obj = context_obj.store_obj.insert_vplan(fixture_module._vplan(context_obj.release_obj, decision_obj))
    if status_str != "ready":
        with context_obj.store_obj._connect() as connection_obj:
            connection_obj.execute("UPDATE vplan SET status_str=? WHERE vplan_id_int=?", (status_str, vplan_obj.vplan_id_int))
            connection_obj.execute("UPDATE decision_plan SET status_str='submitted' WHERE decision_plan_id_int=?", (decision_obj.decision_plan_id_int,))
    return decision_obj, vplan_obj


def test_preview_then_apply_audits_intent_without_committing_memory(context_obj):
    preview_dict = _preview(context_obj)
    assert len(preview_dict["review_hash_str"]) == 64
    assert context_obj.store_obj.get_latest_decision_plan_for_pod(context_obj.release_obj.pod_id_str) is None
    assert context_obj.store_obj.get_pod_state(context_obj.release_obj.pod_id_str) == context_obj.state_obj
    applied_dict = _apply(context_obj, preview_dict["review_hash_str"])
    decision_obj = context_obj.store_obj.get_decision_plan_by_id(applied_dict["decision_plan_id_int"])
    assert decision_obj.status_str == "planned"
    assert decision_obj.snapshot_metadata_dict["base_strategy_state_dict"] == context_obj.state_obj.strategy_state_dict
    assert decision_obj.snapshot_metadata_dict["core5_resume_audit_dict"]["operator_str"] == "test_operator"
    assert decision_obj.strategy_state_dict["last_signal_date_str"] == "2026-09-11"
    assert context_obj.store_obj.get_pod_state(context_obj.release_obj.pod_id_str) == context_obj.state_obj
    assert context_obj.broker_obj.account_read_count_int == 2
    with pytest.raises(ValueError, match="review hash"):
        _apply(context_obj, preview_dict["review_hash_str"])


def test_same_session_ready_cycle_is_atomically_superseded_and_keeps_original_policy(context_obj):
    old_decision_obj, old_vplan_obj = _old_cycle(context_obj)
    preview_dict = _preview(context_obj)
    applied_dict = _apply(context_obj, preview_dict["review_hash_str"])
    assert applied_dict["intent_revision_int"] == 1
    old_decision_obj = context_obj.store_obj.get_decision_plan_by_id(old_decision_obj.decision_plan_id_int)
    assert old_decision_obj.status_str == "superseded"
    assert old_decision_obj.execution_policy_str == "next_open_moo"
    assert old_decision_obj.snapshot_metadata_dict["core5_superseded_audit_dict"]["review_hash_str"] == preview_dict["review_hash_str"]
    assert context_obj.store_obj.get_vplan_by_id(old_vplan_obj.vplan_id_int).status_str == "superseded"
    assert context_obj.store_obj.claim_vplan_for_submission(old_vplan_obj.vplan_id_int) is False


def test_submitting_crash_requires_complete_postclose_absence_evidence(context_obj):
    _, old_vplan_obj = _old_cycle(context_obj, "2026-09-10", "submitting")
    preview_dict = _preview(context_obj)
    assert preview_dict["prior_cycle_list"][0]["evidence_dict"]["terminal_bool"] is True
    applied_dict = _apply(context_obj, preview_dict["review_hash_str"])
    assert applied_dict["applied_bool"]
    assert context_obj.store_obj.get_vplan_by_id(old_vplan_obj.vplan_id_int).status_str == "superseded"
    assert context_obj.store_obj.get_pod_state(context_obj.release_obj.pod_id_str).strategy_state_dict == context_obj.state_obj.strategy_state_dict


def test_incomplete_absence_evidence_leaves_prior_cycle_unresolved(context_obj):
    _, old_vplan_obj = _old_cycle(context_obj, "2026-09-10", "submitting")
    context_obj.broker_obj.absence_complete_bool = False
    with pytest.raises(ValueError, match="incomplete"):
        _preview(context_obj)
    assert context_obj.store_obj.get_vplan_by_id(old_vplan_obj.vplan_id_int).status_str == "submitting"


@pytest.mark.parametrize("mutation_str", ["positions", "open_orders", "stale", "cutoff", "account"])
def test_account_refresh_must_be_fresh_flat_orders_and_match_eod(context_obj, mutation_str):
    preview_dict = _preview(context_obj)
    snapshot_obj = context_obj.broker_obj.snapshot_obj
    change_dict = {"positions": {"position_amount_map": {"BIL": 101}},
        "open_orders": {"open_order_id_list": ["unrelated_manual_order"]},
        "stale": {"snapshot_timestamp_ts": context_obj.as_of_ts - timedelta(seconds=1)},
        "cutoff": {"snapshot_timestamp_ts": context_obj.as_of_ts + timedelta(minutes=28)},
        "account": {"account_route_str": "DU_OTHER"}}[mutation_str]
    context_obj.broker_obj.snapshot_obj = replace(snapshot_obj, **change_dict)
    with pytest.raises(ValueError):
        _apply(context_obj, preview_dict["review_hash_str"])
    assert context_obj.store_obj.get_latest_decision_plan_for_pod(context_obj.release_obj.pod_id_str) is None


@pytest.mark.parametrize("mutation_str", ["snapshot_hash", "saved_memory", "release"])
def test_changed_review_inputs_require_new_hash(context_obj, mutation_str):
    preview_dict = _preview(context_obj)
    if mutation_str == "snapshot_hash":
        context_obj.metadata_dict["norgate_manifest_hash_str"] = "revised_data"
    elif mutation_str == "saved_memory":
        context_obj.store_obj.upsert_pod_state(replace(context_obj.state_obj, strategy_state_dict={"concurrent_completion": True}))
    else:
        context_obj.store_obj.upsert_release(replace(context_obj.release_obj, auto_submit_enabled_bool=True))
    with pytest.raises(ValueError):
        _apply(context_obj, preview_dict["review_hash_str"])


def test_selects_actual_eod_cash_and_current_memory_after_intermediate_state(context_obj):
    current_obj = replace(context_obj.state_obj, cash_float=1.0, snapshot_stage_str="pre_submit",
        updated_timestamp_ts=context_obj.as_of_ts, strategy_state_dict={"actual_committed_memory": True})
    context_obj.store_obj.upsert_pod_state(current_obj)
    preview_dict = _preview(context_obj)
    metadata_dict = preview_dict["decision_plan_dict"]["snapshot_metadata_dict"]
    assert metadata_dict["sizing_account_cash_float"] == context_obj.state_obj.cash_float
    assert metadata_dict["base_strategy_state_dict"] == current_obj.strategy_state_dict


def test_concurrent_change_during_final_broker_refresh_blocks_apply(context_obj):
    preview_dict = _preview(context_obj)
    context_obj.broker_obj.after_read_fn = lambda: context_obj.store_obj.upsert_pod_state(
        replace(context_obj.state_obj, strategy_state_dict={"concurrent_completion": True}))
    with pytest.raises(ValueError, match="changed during"):
        _apply(context_obj, preview_dict["review_hash_str"])
    assert context_obj.store_obj.get_latest_decision_plan_for_pod(context_obj.release_obj.pod_id_str) is None


def test_insert_failure_rolls_back_supersession(context_obj, monkeypatch):
    old_decision_obj, old_vplan_obj = _old_cycle(context_obj)
    preview_dict = _preview(context_obj)
    def fail_insert_fn(*args, **kwargs):
        raise RuntimeError("synthetic insert failure")
    monkeypatch.setattr(context_obj.store_obj, "insert_decision_plan", fail_insert_fn)
    with pytest.raises(RuntimeError, match="synthetic"):
        _apply(context_obj, preview_dict["review_hash_str"])
    assert context_obj.store_obj.get_decision_plan_by_id(old_decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    assert context_obj.store_obj.get_vplan_by_id(old_vplan_obj.vplan_id_int).status_str == "ready"


@pytest.mark.parametrize("missing_str", ["reason_str", "operator_str"])
def test_apply_requires_operator_and_reason(context_obj, missing_str):
    with pytest.raises(ValueError, match="operator and reason"):
        _apply(context_obj, "unreviewed", **{missing_str: " "})


def test_cutoff_crossing_during_refresh_blocks_persistence(context_obj):
    preview_dict = _preview(context_obj)
    context_obj.broker_obj.after_read_fn = lambda: context_obj.clock_list.__setitem__(0,
        context_obj.as_of_ts + timedelta(minutes=28))
    with pytest.raises(ValueError, match="cutoff"):
        _apply(context_obj, preview_dict["review_hash_str"])
    assert context_obj.store_obj.get_latest_decision_plan_for_pod(context_obj.release_obj.pod_id_str) is None


@pytest.mark.parametrize("proved_unclaimed_bool", [False, True])
def test_blocked_same_session_needs_durable_unclaimed_proof(context_obj, proved_unclaimed_bool):
    decision_obj, vplan_obj = _old_cycle(context_obj)
    if proved_unclaimed_bool:
        assert context_obj.store_obj.block_unsubmitted_core5_vplan(vplan_obj.vplan_id_int)
    else:
        with context_obj.store_obj._connect() as connection_obj:
            connection_obj.execute("UPDATE vplan SET status_str='blocked' WHERE vplan_id_int=?", (vplan_obj.vplan_id_int,))
            connection_obj.execute("UPDATE decision_plan SET status_str='blocked' WHERE decision_plan_id_int=?", (decision_obj.decision_plan_id_int,))
    if proved_unclaimed_bool:
        assert _apply(context_obj, _preview(context_obj)["review_hash_str"])["applied_bool"]
    else:
        with pytest.raises(ValueError, match="invalid or stale|uncertain or nonterminal"):
            _preview(context_obj)
        assert context_obj.store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "blocked"
        assert context_obj.store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "blocked"


def test_terminal_proof_changed_after_prepare_loses_transaction_race(context_obj, monkeypatch):
    _, vplan_obj = _old_cycle(context_obj, "2026-09-10", "submitting")
    preview_dict = _preview(context_obj)
    original_prepare_fn = core5_resume._prepare_core5_resume
    def raced_prepare_fn(*args, **kwargs):
        result_tuple = original_prepare_fn(*args, **kwargs)
        with context_obj.store_obj._connect() as connection_obj:
            connection_obj.execute("DELETE FROM vplan_execution_resolution WHERE vplan_id_int=?", (vplan_obj.vplan_id_int,))
        return result_tuple
    monkeypatch.setattr(core5_resume, "_prepare_core5_resume", raced_prepare_fn)
    with pytest.raises(ValueError, match="compare-and-set"):
        _apply(context_obj, preview_dict["review_hash_str"])
    assert context_obj.store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"


def test_completed_same_session_cannot_be_reopened(context_obj):
    decision_obj, _ = _old_cycle(context_obj)
    context_obj.store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "completed")
    with pytest.raises(ValueError, match="already completed"):
        _preview(context_obj)


def test_resume_does_not_add_a_live_cash_or_nav_change_guard(context_obj):
    preview_dict = _preview(context_obj)
    context_obj.broker_obj.snapshot_obj = replace(context_obj.broker_obj.snapshot_obj,
        cash_float=1.0, total_value_float=10_000.0, net_liq_float=10_000.0)
    applied_dict = _apply(context_obj, preview_dict["review_hash_str"])
    decision_obj = context_obj.store_obj.get_decision_plan_by_id(applied_dict["decision_plan_id_int"])
    assert decision_obj.snapshot_metadata_dict["sizing_account_cash_float"] == context_obj.state_obj.cash_float
