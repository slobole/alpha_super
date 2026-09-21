"""Activity completion rows require the existing scoped ACK/fill proof."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from alpha.live.dashboard_v4.activity_cycles import build_activity_cycles_dict
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.overview import build_overview_dict


@pytest.fixture(scope="module")
def saved_fixture_tuple():
    workspace_dict, _, provider_obj = build_demo_workspace_tuple()
    try:
        overview_dict = build_overview_dict(workspace_dict, None, provider_obj,
            as_of_ts=DEMO_NOW_TS, include_finance_bool=False)
        pod_id_str = overview_dict["pod_list"][0]["pod_id_str"]
        source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
        yield overview_dict, provider_obj, source_dict, pod_id_str
    finally:
        provider_obj.close()


def _projection_dict(saved_fixture_tuple, *, mutate_fn=None, from_ts=None, overview_fn=None):
    overview_dict, _, source_dict, pod_id_str = deepcopy(saved_fixture_tuple[0]), None, deepcopy(saved_fixture_tuple[2]), saved_fixture_tuple[3]
    overview_dict["pod_list"] = [pod_dict for pod_dict in overview_dict["pod_list"] if pod_dict["pod_id_str"] == pod_id_str]
    source_dict["cycle_list"] = [source_dict["selected_cycle_dict"]]
    if mutate_fn:
        mutate_fn(source_dict)
    if overview_fn:
        overview_fn(overview_dict)
    provider_obj = SimpleNamespace(get_pod_cycles_dict=lambda *args, **kwargs: deepcopy(source_dict))
    return build_activity_cycles_dict(provider_obj, overview_dict, as_of_ts=DEMO_NOW_TS,
        from_ts=from_ts or DEMO_NOW_TS - timedelta(days=7))


def test_complete_group_uses_saved_times_and_cycle_links_without_mutation(saved_fixture_tuple):
    source_dict = saved_fixture_tuple[2]
    before_dict = deepcopy(source_dict)
    result_dict = _projection_dict(saved_fixture_tuple)
    group_dict = next(row_dict for row_dict in result_dict["row_list"] if row_dict.get("fold_candidate_bool"))
    assert group_dict["title_str"] == "Open cycle completed."
    assert group_dict["timestamp_str"] == source_dict["reconciliation_dict"]["created_timestamp_str"]
    assert group_dict["timestamp_str"] != source_dict["pod_row_dict"]["as_of_timestamp_str"]
    assert group_dict["detail_str"] == "3 of 3 filled · Positions checked"
    assert group_dict["time_str"] == "09:36:12"
    assert "diff" not in group_dict["detail_str"]
    assert len(group_dict["child_list"]) == 6
    child_dict = {row_dict["stage_str"]: row_dict for row_dict in group_dict["child_list"]}
    assert child_dict["data"]["timestamp_str"] == "" and child_dict["data"]["time_str"] == "—"
    assert child_dict["decide"]["timestamp_str"] == source_dict["decision_dict"]["created_timestamp_str"]
    assert child_dict["plan"]["timestamp_str"] == source_dict["vplan_dict"]["created_timestamp_str"]
    assert child_dict["submit"]["timestamp_str"] == max(row_dict["response_timestamp_str"] for row_dict in source_dict["ack_list"])
    assert child_dict["fill"]["timestamp_str"] == source_dict["cycle_evidence_dict"]["actual_fill_timestamp_str"]
    assert child_dict["reconcile"]["detail_str"] == child_dict["submit"]["detail_str"] == ""
    assert child_dict["fill"]["detail_str"] == "3 of 3 filled"
    query_dict = parse_qs(urlsplit(group_dict["evidence_url_str"]).query)
    assert query_dict == {"cycle": [source_dict["selected_cycle_dict"]["cycle_key_str"]], "tab": ["plan"]}
    assert group_dict["release_id_str"] == source_dict["selected_release_dict"]["release_id_str"]
    assert result_dict["folded_key_list"] == []
    assert result_dict == _projection_dict(saved_fixture_tuple)
    assert source_dict == before_dict


def test_cross_day_child_keeps_saved_instant_and_separate_date(saved_fixture_tuple):
    result_dict = _projection_dict(saved_fixture_tuple,
        mutate_fn=lambda source_dict: source_dict["decision_dict"].update(created_timestamp_str="2026-09-07T22:00:00+00:00"))
    group_dict = next(row_dict for row_dict in result_dict["row_list"] if row_dict.get("fold_candidate_bool"))
    child_dict = {row_dict["stage_str"]: row_dict for row_dict in group_dict["child_list"]}
    assert child_dict["decide"]["timestamp_str"] == "2026-09-07T22:00:00+00:00"
    assert child_dict["decide"]["cross_day_str"] == "09-07"
    assert child_dict["decide"]["time_str"] == "18:00:00"
    assert child_dict["reconcile"]["cross_day_str"] == ""
    assert child_dict["data"]["cross_day_str"] == ""


@pytest.mark.parametrize("kind_str", ["missing_ack", "partial_fill", "missing_fill", "failed_reconcile", "unknown_data", "expired_plan", "stale_source"])
def test_unhealthy_or_unknown_core_never_produces_a_completion(saved_fixture_tuple, kind_str):
    def mutate(source_dict):
        if kind_str == "missing_ack":
            source_dict["ack_list"].pop()
        elif kind_str == "partial_fill":
            source_dict["cycle_evidence_dict"].update(state_str="partial", filled_order_count_int=2)
        elif kind_str == "missing_fill":
            source_dict["cycle_evidence_dict"] = {}
        elif kind_str == "failed_reconcile":
            source_dict["pod_row_dict"]["latest_reconciliation_status_str"] = "blocked"
        elif kind_str == "unknown_data":
            source_dict["pod_row_dict"]["norgate_snapshot_status_dict"] = {}
        elif kind_str == "expired_plan":
            source_dict["pod_row_dict"]["latest_vplan_status_str"] = "expired"
        elif kind_str == "stale_source":
            source_dict["pod_row_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(minutes=3)).isoformat()
    result_dict = _projection_dict(saved_fixture_tuple, mutate_fn=mutate)
    assert not any(row_dict.get("fold_candidate_bool") for row_dict in result_dict["row_list"])
    assert not any(row_dict["state_str"] in {"late", "fail"} for row_dict in result_dict["row_list"])


def test_stale_shell_does_not_erase_a_freshly_verified_saved_cycle(saved_fixture_tuple):
    result_dict = _projection_dict(saved_fixture_tuple,
        overview_fn=lambda overview_dict: overview_dict.update(source_fresh_bool=False))
    assert any(row_dict.get("fold_candidate_bool") for row_dict in result_dict["row_list"])


def test_explicit_stale_cycle_flag_is_not_overridden(saved_fixture_tuple):
    result_dict = _projection_dict(saved_fixture_tuple,
        mutate_fn=lambda source_dict: source_dict["pod_row_dict"].update(source_stale_bool=True))
    assert result_dict["row_list"] == [] and result_dict["warning_list"]


@pytest.mark.parametrize("field_tuple,value_obj", [
    (("decision_dict", "account_route_str"), "FOREIGN"),
    (("selected_release_dict", "mode_str"), "paper"),
    (("pod_row_dict", "pod_id_str"), "other-pod"),
    (("pod_row_dict", "latest_vplan_id_int"), 999),
    (("selected_cycle_dict", "release_id_str"), "wrong-release"),
])
def test_scope_mismatch_is_unavailable(saved_fixture_tuple, field_tuple, value_obj):
    def mutate(source_dict):
        # Keep the enumerated candidate fixed, so a selected identity change is visible.
        source_dict["cycle_list"] = deepcopy(source_dict["cycle_list"])
        source_dict[field_tuple[0]][field_tuple[1]] = value_obj
    result_dict = _projection_dict(saved_fixture_tuple, mutate_fn=mutate)
    assert result_dict["row_list"] == []
    assert result_dict["warning_list"]


@pytest.mark.parametrize("field_str", ["created_timestamp_str"])
@pytest.mark.parametrize("value_str", ["2026-09-08T13:35:00", "bad", "2099-01-01T00:00:00+00:00"])
def test_reconcile_observation_must_be_aware_real_and_not_future(saved_fixture_tuple, field_str, value_str):
    result_dict = _projection_dict(saved_fixture_tuple,
        mutate_fn=lambda source_dict: source_dict["reconciliation_dict"].update({field_str: value_str}))
    assert not any(row_dict.get("fold_candidate_bool") for row_dict in result_dict["row_list"])


def test_actual_reconcile_time_controls_intraday_boundary(saved_fixture_tuple):
    reconcile_ts = datetime.fromisoformat(saved_fixture_tuple[2]["reconciliation_dict"]["created_timestamp_str"])
    assert _projection_dict(saved_fixture_tuple, from_ts=reconcile_ts)["row_list"]
    assert _projection_dict(saved_fixture_tuple, from_ts=reconcile_ts + timedelta(microseconds=1))["row_list"] == []


def test_legacy_ack_boundary_is_not_reported_as_actual_submit_time(saved_fixture_tuple):
    def mutate(source_dict):
        for ack_dict in source_dict["ack_list"]:
            ack_dict["response_timestamp_str"] = source_dict["vplan_dict"]["submission_timestamp_str"]
    group_dict = _projection_dict(saved_fixture_tuple, mutate_fn=mutate)["row_list"][0]
    submit_dict = next(child_dict for child_dict in group_dict["child_list"] if child_dict["stage_str"] == "submit")
    assert submit_dict["timestamp_str"] == "" and submit_dict["time_str"] == "—"


def test_verified_no_order_cycle_can_complete_without_fabricated_fills(saved_fixture_tuple):
    def mutate(source_dict):
        source_dict["cycle_evidence_dict"].update(state_str="no_orders", order_count_int=0, filled_order_count_int=0, actual_fill_timestamp_str=None)
        source_dict["order_list"] = source_dict["ack_list"] = source_dict["fill_list"] = []
        source_dict["plan_row_list"] = []
        source_dict["pod_row_dict"].update(broker_order_count_int=0, broker_ack_count_int=0, fill_count_int=0)
    group_dict = _projection_dict(saved_fixture_tuple, mutate_fn=mutate)["row_list"][0]
    assert group_dict["detail_str"].startswith("No orders · Positions checked")
    assert all(child_dict["timestamp_str"] == "" for child_dict in group_dict["child_list"] if child_dict["stage_str"] in {"submit", "fill"})


def test_close_policy_names_close_cycle(saved_fixture_tuple):
    result_dict = _projection_dict(saved_fixture_tuple,
        mutate_fn=lambda source_dict: source_dict["selected_cycle_dict"].update(execution_policy_str="same_day_moc"))
    assert result_dict["row_list"][0]["title_str"] == "Close cycle completed."


def test_real_demo_history_keeps_eod_separate_and_problem_cycle_unfolded(saved_fixture_tuple):
    overview_dict, provider_obj, _, _ = saved_fixture_tuple
    result_dict = build_activity_cycles_dict(provider_obj, overview_dict, as_of_ts=DEMO_NOW_TS, from_ts=DEMO_NOW_TS - timedelta(days=10))
    assert any(row_dict["stage_str"] == "eod" and not row_dict["child_list"] for row_dict in result_dict["row_list"])
    problem_pod_str = overview_dict["pod_list"][1]["pod_id_str"]
    assert not any(row_dict["pod_id_str"] == problem_pod_str and row_dict["vplan_id_int"] == 2 and row_dict.get("fold_candidate_bool") for row_dict in result_dict["row_list"])
    assert result_dict["folded_key_list"] == []


def test_one_saved_eod_linked_to_two_cycles_is_emitted_once(saved_fixture_tuple):
    overview_dict, provider_obj, _, pod_id_str = saved_fixture_tuple
    overview_dict = deepcopy(overview_dict)
    overview_dict["pod_list"] = [pod_dict for pod_dict in overview_dict["pod_list"] if pod_dict["pod_id_str"] == pod_id_str]
    first_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS, vplan_id_int=1)
    second_dict = deepcopy(first_dict)
    second_dict["selected_cycle_dict"].update(cycle_key_str="vplan:9", decision_plan_id_int=9, vplan_id_int=9)
    second_dict["pod_row_dict"].update(latest_decision_plan_id_int=9, latest_vplan_id_int=9, latest_vplan_decision_plan_id_int=9)
    second_dict["decision_dict"]["decision_plan_id_int"] = 9
    for key_str in ("vplan_dict", "cycle_evidence_dict", "reconciliation_dict"):
        second_dict[key_str].update(vplan_id_int=9, decision_plan_id_int=9)
    cycle_list = [second_dict["selected_cycle_dict"], first_dict["selected_cycle_dict"]]
    first_dict["cycle_list"] = second_dict["cycle_list"] = cycle_list
    def get_cycles(*args, **selector_dict):
        return deepcopy(first_dict if selector_dict.get("vplan_id_int") == 1 else second_dict)
    result_dict = build_activity_cycles_dict(SimpleNamespace(get_pod_cycles_dict=get_cycles), overview_dict,
        as_of_ts=DEMO_NOW_TS, from_ts=DEMO_NOW_TS - timedelta(days=10))
    assert len([row_dict for row_dict in result_dict["row_list"] if row_dict["stage_str"] == "eod"]) == 1
    assert len([row_dict for row_dict in result_dict["row_list"] if row_dict.get("fold_candidate_bool")]) == 2


@pytest.mark.parametrize("pod_count_int,maximum_int", [(1, 12), (32, 60), (35, 60)])
def test_provider_reads_are_bounded_per_pod_and_overall(saved_fixture_tuple, pod_count_int, maximum_int):
    overview_dict = deepcopy(saved_fixture_tuple[0])
    template_dict = deepcopy(saved_fixture_tuple[2])
    pod_dict = overview_dict["pod_list"][0]
    overview_dict["pod_list"] = [dict(pod_dict, pod_id_str=f"pod-{index_int}") for index_int in range(pod_count_int)]
    call_list = []
    def get_cycles(pod_id_str, **selector_dict):
        call_list.append(pod_id_str)
        if selector_dict.get("decision_plan_id_int"):
            return {"status_str": "unknown"}
        source_dict = deepcopy(template_dict)
        source_dict["selected_cycle_dict"] = {}
        source_dict["cycle_list"] = [dict(template_dict["selected_cycle_dict"], cycle_key_str=f"vplan:{index_int}",
            decision_plan_id_int=index_int, vplan_id_int=index_int) for index_int in range(1, 61)]
        return source_dict
    result_dict = build_activity_cycles_dict(SimpleNamespace(get_pod_cycles_dict=get_cycles), overview_dict,
        as_of_ts=DEMO_NOW_TS, from_ts=DEMO_NOW_TS - timedelta(days=10))
    assert len(call_list) == maximum_int
    assert len(set(call_list)) <= 32
    assert max(call_list.count(pod_id_str) for pod_id_str in set(call_list)) <= 12
    assert any("read limit" in warning_str for warning_str in result_dict["warning_list"])


@pytest.mark.parametrize("change_dict", [{"mode_str": "paper"}, {"enabled_bool": False}])
def test_explicit_non_live_or_disabled_pod_is_not_read(saved_fixture_tuple, change_dict):
    overview_dict = deepcopy(saved_fixture_tuple[0])
    overview_dict["pod_list"] = [dict(overview_dict["pod_list"][0], **change_dict)]
    def forbidden(*args, **kwargs):
        pytest.fail("Excluded Pod must not be read")
    assert build_activity_cycles_dict(SimpleNamespace(get_pod_cycles_dict=forbidden), overview_dict,
        as_of_ts=DEMO_NOW_TS, from_ts=DEMO_NOW_TS - timedelta(days=7))["row_list"] == []


def test_bad_period_and_read_failures_do_not_escape_raw_errors(saved_fixture_tuple):
    def unavailable(*args, **kwargs):
        raise OSError("private path / token")
    provider_obj = SimpleNamespace(get_pod_cycles_dict=unavailable)
    overview_dict = saved_fixture_tuple[0]
    result_dict = build_activity_cycles_dict(provider_obj, overview_dict,
        as_of_ts=DEMO_NOW_TS, from_ts=DEMO_NOW_TS.replace(tzinfo=None))
    assert result_dict["row_list"] == [] and "invalid activity period" in result_dict["warning_list"][0]
    result_dict = build_activity_cycles_dict(provider_obj, overview_dict,
        as_of_ts=DEMO_NOW_TS, from_ts=DEMO_NOW_TS - timedelta(days=7))
    assert result_dict["warning_list"] and "private" not in str(result_dict)
