"""Status polling validates current LIVE ownership without financial acquisition."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import sqlite3
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.status import load_operations_workspace_dict
from test_live_dashboard import _write_release_manifest


AS_OF_TS = datetime(2026, 9, 21, 14, tzinfo=UTC)


@pytest.fixture(autouse=True)
def healthy_host_disk(monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v3.health.shutil.disk_usage",
        lambda path_str: SimpleNamespace(total=100, used=50, free=50))


@pytest.fixture
def source_tuple(tmp_path):
    release_root_obj = tmp_path / "releases"
    _write_release_manifest(release_root_obj, user_id_str="owner", pod_id_str="pod_live",
        mode_str="live", account_route_str="U100")
    summary_dict = {"as_of_timestamp_str": AS_OF_TS.isoformat(), "pod_row_dict_list": [{
        "mode_str": "live", "user_id_str": "owner", "release_id_str": "owner.pod_live.live.v1",
        "pod_id_str": "pod_live", "account_route_str": "U100", "db_status_str": "ok"}]}
    call_list = []

    def summary_fn():
        call_list.append("summary")
        return summary_dict

    provider_obj = SimpleNamespace(app_obj=lambda: SimpleNamespace(releases_root_path_str=str(release_root_obj)),
        get_summary_dict=summary_fn)
    return release_root_obj, summary_dict, provider_obj, call_list


def test_only_validated_enabled_live_identity_enters_status_without_history_reads(source_tuple, monkeypatch):
    root_obj, summary_dict, provider_obj, call_list = source_tuple
    for pod_str, mode_str, route_str, owner_str, enabled_bool in (
            ("pod_paper", "paper", "DU200", "owner", True),
            ("pod_sim", "incubation", "SIM_fixture", "owner", True),
            ("pod_disabled", "live", "U300", "owner", False),
            ("foreign_disabled", "live", "U400", "foreign", False)):
        _write_release_manifest(root_obj, user_id_str=owner_str, pod_id_str=pod_str,
            mode_str=mode_str, account_route_str=route_str, enabled_bool=enabled_bool)
        summary_dict["pod_row_dict_list"].append({"pod_id_str": pod_str, "mode_str": mode_str,
            "account_route_str": route_str, "user_id_str": owner_str})
    summary_dict["combined_book_dict"] = {"private_foreign_history": "must not escape"}
    original_dict = deepcopy(summary_dict)
    before_dict = {str(path_obj): (path_obj.read_bytes(), path_obj.stat().st_mtime_ns)
        for path_obj in root_obj.rglob("*.yaml")}

    def forbidden_fn(*args_tuple, **kwargs_dict):
        pytest.fail("Status attempted financial or historical acquisition")

    monkeypatch.setattr(sqlite3, "connect", forbidden_fn)
    monkeypatch.setattr("alpha.live.dashboard_v3.local_workspace.build_live_binding_obj_list", forbidden_fn)
    monkeypatch.setattr("alpha.live.dashboard_v3.local_workspace._saved_binding_list", forbidden_fn)
    monkeypatch.setattr("alpha.live.dashboard_v4.data.load_workspace_snapshot_tuple", forbidden_fn)
    monkeypatch.setattr("alpha.live.client_reporting.load_broker_reporting_snapshot", forbidden_fn)
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert call_list == ["summary"]
    assert result_dict["operations_error_str"] is None
    assert result_dict["client_dict"] == {"accounts": [], "display_name": "owner"}
    assert [row_dict["pod_id"] for row_dict in result_dict["operations_account_list"]] == ["pod_live"]
    assert result_dict["operations_account_list"][0]["display_name"] == "DVO2"
    assert result_dict["summary_dict"] == {"as_of_timestamp_str": AS_OF_TS.isoformat(),
        "pod_row_dict_list": [summary_dict["pod_row_dict_list"][0]]}
    assert summary_dict == original_dict
    assert before_dict == {str(path_obj): (path_obj.read_bytes(), path_obj.stat().st_mtime_ns)
        for path_obj in root_obj.rglob("*.yaml")}


@pytest.mark.parametrize("problem_str", ["foreign_owner", "duplicate_pod", "duplicate_account", "placeholder", "corrupt_yaml"])
def test_invalid_deployment_fails_before_summary_access(source_tuple, problem_str):
    root_obj, _, provider_obj, call_list = source_tuple
    if problem_str == "corrupt_yaml":
        (root_obj / "broken.yaml").write_text("identity: [broken", encoding="utf-8")
    else:
        _write_release_manifest(root_obj,
            user_id_str="foreign" if problem_str in {"foreign_owner", "duplicate_pod"} else "owner",
            pod_id_str="pod_live" if problem_str == "duplicate_pod" else "pod_other",
            mode_str="live", account_route_str="U100" if problem_str == "duplicate_account" else
                "YOUR_ACCOUNT" if problem_str == "placeholder" else "U200")
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert call_list == []
    assert result_dict["operations_account_list"] == []
    assert result_dict["summary_dict"] == {}
    assert result_dict["operations_error_str"] == "Local LIVE configuration could not be verified."


@pytest.mark.parametrize("field_str,value_str", [("mode_str", "paper"), ("user_id_str", "foreign"),
    ("release_id_str", "old-release"), ("account_route_str", "U999"), ("pod_id_str", "foreign_pod")])
def test_cached_foreign_or_previous_release_row_cannot_authorize_status(source_tuple, field_str, value_str):
    _, summary_dict, provider_obj, _ = source_tuple
    summary_dict["pod_row_dict_list"][0][field_str] = value_str
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert len(result_dict["operations_account_list"]) == 1
    assert result_dict["summary_dict"]["pod_row_dict_list"] == []
    overview_dict = build_overview_dict(result_dict, None, provider_obj, as_of_ts=AS_OF_TS, include_finance_bool=False)
    assert overview_dict["pod_list"][0]["state_str"] == "unk"
    assert overview_dict["system_dict"]["state_str"] == "unk"


@pytest.mark.parametrize("case_str", ["missing", "duplicate", "missing_owner", "missing_release"])
def test_absent_or_ambiguous_operational_evidence_remains_unknown(source_tuple, case_str):
    _, summary_dict, provider_obj, _ = source_tuple
    row_list = summary_dict["pod_row_dict_list"]
    if case_str == "missing":
        row_list.clear()
    elif case_str == "duplicate":
        row_list.append(deepcopy(row_list[0]))
    else:
        row_list[0].pop("user_id_str" if case_str == "missing_owner" else "release_id_str")
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["summary_dict"]["pod_row_dict_list"] == []
    overview_dict = build_overview_dict(result_dict, None, provider_obj, as_of_ts=AS_OF_TS, include_finance_bool=False)
    assert overview_dict["pod_list"][0]["state_str"] == "unk"


@pytest.mark.parametrize("source_obj", [None, [], {}, {"pod_row_dict_list": None}, {"pod_row_dict_list": ["bad"]}])
def test_malformed_summary_has_sanitized_failure(source_tuple, source_obj):
    _, _, provider_obj, _ = source_tuple
    provider_obj.get_summary_dict = lambda: source_obj
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["summary_dict"] == {}
    assert result_dict["operations_error_str"] == "Saved operations could not be read."


@pytest.mark.parametrize("exception_obj", [OSError("private path"), sqlite3.DatabaseError("private account"),
    ValueError("private source"), AttributeError("private provider")])
def test_summary_failure_preserves_known_pod_identity_and_hides_private_error(source_tuple, exception_obj):
    _, _, provider_obj, _ = source_tuple

    def failed_fn():
        raise exception_obj

    provider_obj.get_summary_dict = failed_fn
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["operations_account_list"][0]["pod_id"] == "pod_live"
    assert result_dict["summary_dict"] == {}
    assert result_dict["operations_error_str"] == "Saved operations could not be read."
    assert "private" not in str(result_dict)


@pytest.mark.parametrize("offset_int,expected_fresh_bool", [(0, True), (-119, True), (-121, False), (1, False)])
def test_source_timestamp_is_not_restamped_and_canonical_expiry_is_preserved(source_tuple, offset_int, expected_fresh_bool):
    _, summary_dict, provider_obj, _ = source_tuple
    timestamp_str = (AS_OF_TS + timedelta(seconds=offset_int)).isoformat()
    summary_dict["as_of_timestamp_str"] = timestamp_str
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["summary_dict"]["as_of_timestamp_str"] == timestamp_str
    overview_dict = build_overview_dict(result_dict, None, provider_obj, as_of_ts=AS_OF_TS, include_finance_bool=False)
    assert overview_dict["source_fresh_bool"] is expected_fresh_bool
    assert overview_dict["source_valid_ms_int"] == (max(0, 120 + offset_int) * 1000 if expected_fresh_bool else 0)


def test_missing_timestamp_remains_unknown(source_tuple):
    _, summary_dict, provider_obj, _ = source_tuple
    summary_dict.pop("as_of_timestamp_str")
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["summary_dict"]["as_of_timestamp_str"] is None
    overview_dict = build_overview_dict(result_dict, None, provider_obj, as_of_ts=AS_OF_TS, include_finance_bool=False)
    assert overview_dict["source_fresh_bool"] is False


def test_empty_release_directory_is_not_green_or_created(tmp_path):
    root_obj = tmp_path / "absent"
    provider_obj = SimpleNamespace(app_obj=lambda: SimpleNamespace(releases_root_path_str=str(root_obj)),
        get_summary_dict=lambda: {"as_of_timestamp_str": AS_OF_TS.isoformat(), "pod_row_dict_list": []})
    result_dict = load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS)
    overview_dict = build_overview_dict(result_dict, None, provider_obj, as_of_ts=AS_OF_TS, include_finance_bool=False)
    assert result_dict["operations_account_list"] == []
    assert overview_dict["system_dict"]["state_str"] == "unk"
    assert not root_obj.exists()


def test_naive_assessment_clock_is_rejected_before_provider(source_tuple):
    _, _, provider_obj, call_list = source_tuple
    with pytest.raises(ValueError, match="timezone-aware"):
        load_operations_workspace_dict(provider_obj, as_of_ts=AS_OF_TS.replace(tzinfo=None))
    assert call_list == []
