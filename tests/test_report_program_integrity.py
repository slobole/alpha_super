"""Saved report assignments are historical evidence, not editable display labels."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
import pytest

from alpha.live.dashboard_v4 import report_programs
from alpha.live.dashboard_v4.report_programs import (
    PROGRAM_BYTE_LIMIT_INT,
    PROGRAM_LIMIT_INT,
    ReportProgramConflictError,
    ReportProgramLimitError,
    ReportProgramReadError,
    ReportProgramStore,
    register_program_route,
)


def _store_obj(tmp_path, *, memory_bool=False):
    return ReportProgramStore(SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl")), memory_bool=memory_bool)


def _route_tuple(tmp_path, pod_set):
    store_obj = _store_obj(tmp_path)
    app_obj = Flask(__name__)
    register_program_route(app_obj, store_obj, lambda: pod_set)
    return app_obj.test_client(), store_obj


def _post_obj(client_obj, store_obj, program_dict):
    return client_obj.post("/monthly-report/programs", json={"confirmed_bool": True, "program_by_pod_dict": program_dict},
        headers={"Origin": "http://localhost", "X-Alpha-Action-Token": store_obj.action_token_str})


def test_missing_initial_file_and_legacy_valid_mapping(tmp_path):
    store_obj = _store_obj(tmp_path)
    assert store_obj.program_by_pod_dict() == {}
    path_obj = tmp_path / "report_programs.json"
    path_obj.write_text('{"program_by_pod_dict": {"pod_taa": " Growth ", "pod_ndx": "steady"}}', encoding="utf-8")
    assert store_obj.program_by_pod_dict() == {"pod_taa": "Growth", "pod_ndx": "Steady"}


@pytest.mark.parametrize("payload_str", [
    "not json", "[]", "{}", '{"program_by_pod_dict": []}',
    '{"program_by_pod_dict": {}, "foreign": 1}',
    '{"program_by_pod_dict": {"pod_taa": "Growth", "pod_bad": "<script>"}}',
    '{"program_by_pod_dict": {"pod_taa": ""}}',
    '{"program_by_pod_dict": {"pod_taa": "Portfolio"}}',
    '{"program_by_pod_dict": {"pod_taa": null}}',
    '{"program_by_pod_dict": {"pod_taa": 3}}',
    '{"program_by_pod_dict": {"": "Growth"}}',
    '{"program_by_pod_dict": {" pod_taa": "Growth"}}',
    '{"program_by_pod_dict": {"pod_taa": "Growth", "pod_taa": "Steady"}}',
    '{"program_by_pod_dict": {}, "program_by_pod_dict": {"pod_taa": "Growth"}}',
])
def test_invalid_existing_file_blocks_read_and_save_without_erasing_evidence(tmp_path, payload_str):
    path_obj = tmp_path / "report_programs.json"
    path_obj.write_text(payload_str, encoding="utf-8")
    store_obj = _store_obj(tmp_path)
    with pytest.raises(ReportProgramReadError):
        store_obj.program_by_pod_dict()
    with pytest.raises(ReportProgramReadError):
        store_obj.save({"pod_new": "Growth"})
    assert path_obj.read_text(encoding="utf-8") == payload_str
    assert not (tmp_path / "report_programs.json.tmp").exists()


def test_oversized_and_overlimit_existing_files_are_not_empty_mappings(tmp_path):
    path_obj = tmp_path / "report_programs.json"
    store_obj = _store_obj(tmp_path)
    for payload_str in (" " * (PROGRAM_BYTE_LIMIT_INT + 1), json.dumps({"program_by_pod_dict": {
            f"pod_{index_int}": "Growth" for index_int in range(PROGRAM_LIMIT_INT + 1)}})):
        path_obj.write_text(payload_str, encoding="utf-8")
        with pytest.raises(ReportProgramReadError):
            store_obj.program_by_pod_dict()
        with pytest.raises(ReportProgramReadError):
            store_obj.save({"pod_new": "Steady"})
        assert path_obj.read_text(encoding="utf-8") == payload_str


def test_read_permission_failure_never_becomes_an_empty_mapping(tmp_path, monkeypatch):
    store_obj = _store_obj(tmp_path)
    store_obj.save({"pod_old": "Steady"})
    path_obj = tmp_path / "report_programs.json"
    before_bytes = path_obj.read_bytes()
    original_read_fn = Path.read_text

    def unreadable_str(read_path_obj, *argument_tuple, **keyword_dict):
        if read_path_obj == path_obj:
            raise PermissionError("File is unavailable")
        return original_read_fn(read_path_obj, *argument_tuple, **keyword_dict)

    monkeypatch.setattr(Path, "read_text", unreadable_str)
    with pytest.raises(ReportProgramReadError):
        store_obj.program_by_pod_dict()
    with pytest.raises(ReportProgramReadError):
        store_obj.save({"pod_new": "Growth"})
    assert path_obj.read_bytes() == before_bytes


@pytest.mark.parametrize("memory_bool", [False, True])
def test_assignment_is_immutable_but_case_and_spacing_are_idempotent(tmp_path, memory_bool):
    store_obj = _store_obj(tmp_path, memory_bool=memory_bool)
    store_obj.save({"pod_taa": "gROWTH", "pod_ndx": "  Growth   Plus  ", "pod_empty": "Portfolio"})
    expected_dict = {"pod_taa": "Growth", "pod_ndx": "Growth Plus"}
    assert store_obj.program_by_pod_dict() == expected_dict
    store_obj.save({"pod_taa": " GROWTH ", "pod_ndx": "growth   plus"})
    assert store_obj.program_by_pod_dict() == expected_dict
    for replacement_str in ("", "Portfolio", "Steady"):
        with pytest.raises(ReportProgramConflictError):
            store_obj.save({"pod_taa": replacement_str})
        assert store_obj.program_by_pod_dict() == expected_dict


def test_batch_conflict_does_not_partly_save_new_assignments(tmp_path):
    store_obj = _store_obj(tmp_path)
    store_obj.save({"pod_old": "Steady"})
    path_obj = tmp_path / "report_programs.json"
    before_bytes = path_obj.read_bytes()
    with pytest.raises(ReportProgramConflictError):
        store_obj.save({"pod_new": "Growth", "pod_old": ""})
    assert path_obj.read_bytes() == before_bytes
    assert store_obj.program_by_pod_dict() == {"pod_old": "Steady"}


def test_limit_refuses_addition_without_dropping_the_oldest_assignment(tmp_path):
    store_obj = _store_obj(tmp_path)
    program_dict = {f"pod_{index_int}": "Growth" for index_int in range(PROGRAM_LIMIT_INT)}
    store_obj.save(program_dict)
    path_obj = tmp_path / "report_programs.json"
    before_bytes = path_obj.read_bytes()
    with pytest.raises(ReportProgramLimitError):
        store_obj.save({"pod_new": "Steady"})
    assert path_obj.read_bytes() == before_bytes
    assert store_obj.program_by_pod_dict() == program_dict


def test_serialized_byte_limit_also_preserves_saved_assignments(tmp_path):
    store_obj = _store_obj(tmp_path)
    store_obj.save({"pod_old": "Steady"})
    path_obj = tmp_path / "report_programs.json"
    before_bytes = path_obj.read_bytes()
    with pytest.raises(ReportProgramLimitError):
        store_obj.save({"pod_" + "x" * PROGRAM_BYTE_LIMIT_INT: "Growth"})
    assert path_obj.read_bytes() == before_bytes


def test_concurrent_requests_on_one_store_keep_every_assignment(tmp_path):
    store_obj = _store_obj(tmp_path)
    update_list = [{f"pod_{index_int}": "Growth"} for index_int in range(12)]
    with ThreadPoolExecutor(max_workers=4) as executor_obj:
        list(executor_obj.map(store_obj.save, update_list))
    assert store_obj.program_by_pod_dict() == {f"pod_{index_int}": "Growth" for index_int in range(12)}
    assert _store_obj(tmp_path).program_by_pod_dict() == store_obj.program_by_pod_dict()


def test_failed_replace_preserves_original_mapping_and_removes_temporary_file(tmp_path, monkeypatch):
    store_obj = _store_obj(tmp_path)
    store_obj.save({"pod_old": "Steady"})
    path_obj = tmp_path / "report_programs.json"
    before_bytes = path_obj.read_bytes()
    call_list = []

    def locked_replace(source_obj, destination_obj):
        call_list.append((source_obj, destination_obj))
        assert destination_obj.read_bytes() == before_bytes
        assert json.loads(source_obj.read_text(encoding="utf-8"))["program_by_pod_dict"] == {
            "pod_old": "Steady", "pod_new": "Growth"}
        raise PermissionError("File is in use")

    monkeypatch.setattr(report_programs.os, "replace", locked_replace)
    monkeypatch.setattr(report_programs.time, "sleep", lambda delay_float: None)
    with pytest.raises(PermissionError):
        store_obj.save({"pod_new": "Growth"})
    assert len(call_list) == report_programs.REPLACE_ATTEMPT_INT
    assert path_obj.read_bytes() == before_bytes
    assert not (tmp_path / "report_programs.json.tmp").exists()


def test_route_reports_locked_program_and_bad_saved_evidence(tmp_path):
    client_obj, store_obj = _route_tuple(tmp_path, {"pod_old", "pod_new"})
    assert _post_obj(client_obj, store_obj, {"pod_old": "Growth"}).status_code == 200
    response_obj = _post_obj(client_obj, store_obj, {"pod_old": ""})
    assert response_obj.status_code == 409 and response_obj.get_json()["error"] == "program_locked"
    assert "past reports" in response_obj.get_json()["message"]
    path_obj = tmp_path / "report_programs.json"
    path_obj.write_text("broken saved evidence", encoding="utf-8")
    response_obj = _post_obj(client_obj, store_obj, {"pod_new": "Steady"})
    assert response_obj.status_code == 503 and response_obj.get_json()["error"] == "programs_unreadable"
    assert path_obj.read_text(encoding="utf-8") == "broken saved evidence"


def test_route_limit_preserves_retired_pod_assignments(tmp_path):
    client_obj, store_obj = _route_tuple(tmp_path, {"pod_new"})
    program_dict = {f"retired_{index_int}": "Growth" for index_int in range(PROGRAM_LIMIT_INT)}
    store_obj.save(program_dict)
    response_obj = _post_obj(client_obj, store_obj, {"pod_new": "Steady"})
    assert response_obj.status_code == 409 and response_obj.get_json()["error"] == "program_limit"
    assert store_obj.program_by_pod_dict() == program_dict
