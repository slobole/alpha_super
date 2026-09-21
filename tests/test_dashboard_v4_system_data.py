from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import system_data
from alpha.live.release_manifest import parse_release_manifest
from test_live_dashboard import _write_release_manifest


BASE_TS = datetime(2026, 9, 21, 16, 0, tzinfo=timezone.utc)


def _target_obj(tmp_path, pod_str="pod_one", account_str="U123456", **override_dict):
    field_dict = {"pod_id_str": pod_str, "account_route_str": account_str, "user_id_str": "owner_one",
        "release_id_str": f"release_{pod_str}", "enabled_bool": True, "mode_str": "live",
        "execution_policy_str": "next_open_moo", "strategy_import_str": "strategies.dv2.strategy_mr_dv2:DVO2Strategy"}
    field_dict.update(override_dict)
    return SimpleNamespace(release_obj=SimpleNamespace(**field_dict), db_path_str=str(tmp_path / f"{pod_str}.sqlite"))


def _fixture_tuple(tmp_path, target_list=None):
    target_list = target_list if target_list is not None else [_target_obj(tmp_path)]
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "live_events.jsonl"), get_target_list=lambda: target_list)
    enabled_list = [target_obj for target_obj in target_list if target_obj.release_obj.mode_str == "live" and target_obj.release_obj.enabled_bool]
    workspace_dict = {"operations_account_list": [{"pod_id": target_obj.release_obj.pod_id_str,
        "account_route": target_obj.release_obj.account_route_str} for target_obj in enabled_list],
        "summary_dict": {"as_of_timestamp_str": BASE_TS.isoformat(), "pod_row_dict_list": [
            {**{field_str: getattr(target_obj.release_obj, field_str) for field_str in system_data.IDENTITY_FIELD_TUPLE}, "db_status_str": "ok"}
            for target_obj in enabled_list]}}
    return provider_obj, workspace_dict, target_list


def _load_dict(provider_obj, workspace_dict, **argument_dict):
    return system_data.load_system_source_dict(provider_obj, workspace_dict, as_of_ts=BASE_TS, **argument_dict)


def _save_report(tmp_path, target_list, **override_dict):
    report_dict = {"schema_version_str": "live_ops_inspector.v1", "mode_str": "all", "generated_at_utc_str": BASE_TS.isoformat(),
        "overall_severity_str": "green", "pod_report_dict_list": [{"pod_id_str": target_obj.release_obj.pod_id_str,
            "account_route_str": target_obj.release_obj.account_route_str, "mode_str": target_obj.release_obj.mode_str}
            for target_obj in target_list], "heartbeat_payload_dict": {"token": "secret-token"},
        "vps_id_str": "private-hostname", "reason_str": "C:/private/password.txt"}
    report_dict.update(override_dict)
    (tmp_path / "ops_report_latest.json").write_text(json.dumps(report_dict), encoding="utf-8")


def _save_sources(tmp_path, provider_obj, target_list):
    Path(provider_obj.event_log_path_str).write_bytes(b"saved log\n")
    os.utime(provider_obj.event_log_path_str, (BASE_TS.timestamp(), BASE_TS.timestamp()))
    for target_obj in target_list:
        Path(target_obj.db_path_str).write_bytes(b"saved state")
    _save_report(tmp_path, target_list)


def _flex_path_str(tmp_path, target_list, *, market_date_str="2026-09-18", imported_str="2026-09-19T10:15:00+00:00"):
    path_obj = tmp_path / "performance.sqlite"
    with sqlite3.connect(path_obj) as connection_obj:
        connection_obj.executescript("CREATE TABLE pod_binding(account_route_str,pod_id_str,enabled_bool_int);"
            "CREATE TABLE flex_import(import_id_int,imported_timestamp_str,raw_xml_str);"
            "CREATE TABLE daily_performance(account_route_str,pod_id_str,market_date_str,source_import_id_int);")
        connection_obj.execute("INSERT INTO flex_import VALUES(1,?,?)", (imported_str, "SECRET RAW ACCOUNT XML"))
        for target_obj in target_list:
            release_obj = target_obj.release_obj
            connection_obj.execute("INSERT INTO pod_binding VALUES(?,?,1)", (release_obj.account_route_str, release_obj.pod_id_str))
            connection_obj.execute("INSERT INTO daily_performance VALUES(?,?,?,1)",
                (release_obj.account_route_str, release_obj.pod_id_str, market_date_str))
    return str(path_obj)


def test_safe_evidence_has_no_paths_accounts_secrets_or_delivery_claim(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_sources(tmp_path, provider_obj, target_list)
    # Notification state is evaluation state, not a receipt; even fresh state
    # and intended heartbeat payload must not become proof of delivery.
    (tmp_path / "watchdog_notification_state.json").write_text(json.dumps({"last_updated_str": BASE_TS.isoformat(),
        "pod_severity_map_dict": {"pod_one": "green"}, "pending_red_previous_severity_map_dict": {}}))
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=_flex_path_str(tmp_path, target_list))
    assert result_dict["scope_verified_bool"] is True
    assert result_dict["watchdog_dict"]["now_str"] == "Report saved"
    assert result_dict["watchdog_dict"]["state_str"] == "ok"
    assert result_dict["database_dict"]["state_str"] == result_dict["event_log_dict"]["state_str"] == "ok"
    assert result_dict["alerts_dict"]["state_str"] == "ok"
    assert result_dict["alerts_dict"]["now_str"] == "No saved undelivered alerts"
    assert result_dict["deadman_dict"]["state_str"] == "skip"
    assert result_dict["deadman_dict"]["checked_bool"] is False
    assert result_dict["flex_dict"]["now_str"] == "Report close 2026-09-18"
    assert result_dict["flex_dict"]["state_str"] == "ok"
    assert result_dict["release_list"][0]["account_str"] == "•••456"
    serialized_str = json.dumps(result_dict)
    for private_str in (str(tmp_path), "U123456", "owner_one", "private-hostname", "secret-token", "password.txt", "SECRET RAW ACCOUNT XML"):
        assert private_str not in serialized_str


@pytest.mark.parametrize("case_str", ["mixed_owner", "duplicate_pod", "duplicate_account", "wrong_account", "wrong_pod", "operations_error"])
def test_invalid_scope_prevents_all_state_file_access(tmp_path, monkeypatch, case_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    if case_str == "mixed_owner":
        target_list.append(_target_obj(tmp_path, "pod_two", "U222", user_id_str="foreign_owner"))
    elif case_str == "duplicate_pod":
        target_list.append(_target_obj(tmp_path, account_str="U222"))
    elif case_str == "duplicate_account":
        target_list.append(_target_obj(tmp_path, "pod_two"))
    elif case_str == "wrong_account":
        workspace_dict["operations_account_list"][0]["account_route"] = "U999"
    elif case_str == "wrong_pod":
        workspace_dict["operations_account_list"][0]["pod_id"] = "foreign_pod"
    else:
        workspace_dict["operations_error_str"] = "PRIVATE raw failure"
    def forbidden_call(*argument_list, **argument_dict):
        pytest.fail("State I/O before valid scope")
    monkeypatch.setattr(Path, "open", forbidden_call)
    monkeypatch.setattr(Path, "stat", forbidden_call)
    monkeypatch.setattr(sqlite3, "connect", forbidden_call)
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str="forbidden.sqlite")
    assert result_dict["scope_verified_bool"] is False
    assert result_dict["release_list"] == []
    assert all(result_dict[key_str]["state_str"] == "unknown" for key_str in
        ("event_log_dict", "database_dict", "watchdog_dict", "flex_dict"))


def test_disabled_same_owner_metadata_is_shown_without_reading_its_db_or_paper(tmp_path, monkeypatch):
    target_list = [_target_obj(tmp_path), _target_obj(tmp_path, "off", "U222", enabled_bool=False),
        _target_obj(tmp_path, "paper", "DU222", mode_str="paper"),
        _target_obj(tmp_path, "foreign", "U333", enabled_bool=False, user_id_str="foreign_owner")]
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path, target_list)
    _save_sources(tmp_path, provider_obj, target_list[:1])
    forbidden_set = {target_obj.db_path_str for target_obj in target_list[1:]}
    original_stat_fn = Path.stat
    def checked_stat(path_obj, *argument_list, **argument_dict):
        assert str(path_obj) not in forbidden_set
        return original_stat_fn(path_obj, *argument_list, **argument_dict)
    monkeypatch.setattr(Path, "stat", checked_stat)
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert [(row_dict["pod_id_str"], row_dict["enabled_bool"]) for row_dict in result_dict["release_list"]] == [("pod_one", True), ("off", False)]
    assert result_dict["database_dict"]["state_str"] == "ok"


def test_missing_sources_do_not_create_files_or_claim_health(tmp_path):
    provider_obj, workspace_dict, _target_list = _fixture_tuple(tmp_path)
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=str(tmp_path / "missing.sqlite"))
    assert all(result_dict[key_str]["state_str"] == "unknown" for key_str in
        ("event_log_dict", "database_dict", "watchdog_dict", "flex_dict", "alerts_dict"))
    assert result_dict["deadman_dict"]["state_str"] == "skip"
    assert result_dict["deadman_dict"]["checked_bool"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("text_str", ["not JSON", "[]", '{"schema_version_str":"wrong"}', "[" * 2000, "x" * (system_data.JSON_BYTE_LIMIT_INT + 1)],
    ids=["malformed", "wrong_type", "wrong_schema", "nested", "oversized"])
def test_corrupt_or_oversized_watchdog_is_unknown(tmp_path, text_str):
    provider_obj, workspace_dict, _target_list = _fixture_tuple(tmp_path)
    (tmp_path / "ops_report_latest.json").write_text(text_str, encoding="utf-8")
    assert _load_dict(provider_obj, workspace_dict)["watchdog_dict"]["state_str"] == "unknown"


@pytest.mark.parametrize("seconds_int,state_str", [(0, "ok"), (-900, "ok"), (-901, "warning"), (1, "unknown")])
def test_watchdog_age_boundary_uses_report_time(tmp_path, seconds_int, state_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list, generated_at_utc_str=(BASE_TS + timedelta(seconds=seconds_int)).isoformat())
    assert _load_dict(provider_obj, workspace_dict)["watchdog_dict"]["state_str"] == state_str


@pytest.mark.parametrize("field_str,value_obj", [("account_route_str", "U999"), ("pod_id_str", "foreign"), ("mode_str", "paper")])
def test_watchdog_rejects_foreign_identity(tmp_path, field_str, value_obj):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    report_list = [{"pod_id_str": "pod_one", "account_route_str": "U123456", "mode_str": "live", field_str: value_obj}]
    _save_report(tmp_path, target_list, pod_report_dict_list=report_list)
    assert _load_dict(provider_obj, workspace_dict)["watchdog_dict"]["state_str"] == "unknown"


@pytest.mark.parametrize("seconds_int,status_str,state_str", [(-120, "ok", "ok"), (-121, "ok", "unknown"),
    (1, "ok", "unknown"), (0, "error", "error"), (0, "missing", "error"), (0, "private-error", "unknown")])
def test_database_readability_uses_fresh_exact_summary(tmp_path, seconds_int, status_str, state_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_sources(tmp_path, provider_obj, target_list)
    workspace_dict["summary_dict"]["as_of_timestamp_str"] = (BASE_TS + timedelta(seconds=seconds_int)).isoformat()
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["db_status_str"] = status_str
    assert _load_dict(provider_obj, workspace_dict)["database_dict"]["state_str"] == state_str


def test_summary_owner_or_release_mismatch_cannot_prove_db_readable(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_sources(tmp_path, provider_obj, target_list)
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["release_id_str"] = "old_release"
    assert _load_dict(provider_obj, workspace_dict)["database_dict"]["state_str"] == "unknown"


@pytest.mark.parametrize("timestamp_str,state_str", [((BASE_TS - timedelta(seconds=120)).isoformat(), "ok"),
    ((BASE_TS - timedelta(seconds=121)).isoformat(), "unknown"), ((BASE_TS + timedelta(seconds=1)).isoformat(), "unknown"),
    ("not a timestamp", "unknown"), ("2026-09-21T15:59:00", "unknown"), (None, "unknown")])
def test_each_database_row_timestamp_must_be_fresh_when_present(tmp_path, timestamp_str, state_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_sources(tmp_path, provider_obj, target_list)
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["as_of_timestamp_str"] = timestamp_str
    assert _load_dict(provider_obj, workspace_dict)["database_dict"]["state_str"] == state_str


def test_file_mtime_is_only_a_file_write_and_future_is_unknown(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_sources(tmp_path, provider_obj, target_list)
    for seconds_int, state_str in [(-86400, "warning"), (-3660, "ok"), (-3661, "warning"), (1, "unknown")]:
        modified_float = (BASE_TS + timedelta(seconds=seconds_int)).timestamp()
        os.utime(provider_obj.event_log_path_str, (modified_float, modified_float))
        result_dict = _load_dict(provider_obj, workspace_dict)["event_log_dict"]
        assert result_dict["state_str"] == state_str
        assert "scheduler health is checked separately" in result_dict["detail_str"]


@pytest.mark.parametrize("date_str,imported_str", [("2026-09-21", "2026-09-21T12:00:00+00:00"),
    ("2026-09-18", "2026-09-22T00:00:00+00:00"), ("2026-09-18", "2026-09-19T00:00:00"),
    ("2026-09-18", "2026-09-17T00:00:00+00:00"), ("2026-02-30", "2026-09-19T00:00:00+00:00")])
def test_flex_future_invalid_or_inconsistent_time_is_unavailable(tmp_path, date_str, imported_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=_flex_path_str(tmp_path, target_list,
        market_date_str=date_str, imported_str=imported_str))
    assert result_dict["flex_dict"]["now_str"] == "No verified saved report"


def test_flex_filters_foreign_accounts_without_reading_raw_xml(tmp_path, monkeypatch):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list + [_target_obj(tmp_path, "foreign", "U999")])
    original_connect_fn = sqlite3.connect
    statement_list = []
    def readonly_connect(*argument_list, **argument_dict):
        assert "mode=ro" in argument_list[0] and argument_dict.get("uri") is True
        connection_obj = original_connect_fn(*argument_list, **argument_dict)
        connection_obj.set_trace_callback(statement_list.append)
        return connection_obj
    monkeypatch.setattr(sqlite3, "connect", readonly_connect)
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)
    assert result_dict["flex_dict"]["now_str"] == "Report close 2026-09-18"
    assert all("raw_xml" not in statement_str and "detail_str" not in statement_str for statement_str in statement_list)
    assert any("query_only=ON" in statement_str for statement_str in statement_list)


@pytest.mark.parametrize("mutation_str", ["binding", "report", "duplicate"])
def test_flex_rejects_wrong_binding_or_report_pod(tmp_path, mutation_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list)
    with sqlite3.connect(path_str) as connection_obj:
        if mutation_str == "binding":
            connection_obj.execute("UPDATE pod_binding SET pod_id_str='foreign'")
        elif mutation_str == "report":
            connection_obj.execute("UPDATE daily_performance SET pod_id_str='foreign'")
        else:
            connection_obj.execute("INSERT INTO pod_binding SELECT * FROM pod_binding")
    assert _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["now_str"] == "No verified saved report"


def test_flex_writer_lock_fails_closed_and_recovers(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list)
    with sqlite3.connect(path_str) as connection_obj:
        connection_obj.execute("BEGIN EXCLUSIVE")
        assert _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["now_str"] == "No verified saved report"
        connection_obj.rollback()
    assert _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["now_str"] == "Report close 2026-09-18"


def test_naive_request_clock_rejected(tmp_path):
    provider_obj, workspace_dict, _target_list = _fixture_tuple(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        system_data.load_system_source_dict(provider_obj, workspace_dict, as_of_ts=BASE_TS.replace(tzinfo=None))


@pytest.mark.parametrize("foreign_bool", [False, True])
def test_disabled_only_manifest_scope_lists_config_without_runtime_reads(tmp_path, monkeypatch, foreign_bool):
    root_obj = tmp_path / "releases"
    for pod_str, owner_str in [("off_one", "owner"), ("off_two", "foreign" if foreign_bool else "owner")]:
        _write_release_manifest(root_obj, user_id_str=owner_str, pod_id_str=pod_str,
            mode_str="live", account_route_str="U111" if pod_str == "off_one" else "U222", enabled_bool=False)
    source_obj = SimpleNamespace(releases_root_path_str=str(root_obj), get_target_list=lambda: [])
    provider_obj = SimpleNamespace(app_obj=lambda: source_obj, event_log_path_str=str(tmp_path / "live_events.jsonl"))
    def forbidden_call(*argument_list, **argument_dict):
        pytest.fail("Disabled-only scope must not read runtime sources")
    for name_str in ("_database_dict", "_event_log_dict", "_watchdog_dict", "_flex_dict"):
        monkeypatch.setattr(system_data, name_str, forbidden_call)
    result_dict = _load_dict(provider_obj, {"operations_account_list": [], "summary_dict": {}})
    assert result_dict["scope_verified_bool"] is not foreign_bool
    assert len(result_dict["release_list"]) == (0 if foreign_bool else 2)
    assert all(row_dict["enabled_bool"] is False for row_dict in result_dict["release_list"])
    assert result_dict["event_log_dict"]["state_str"] == "unknown"


def test_manifest_metadata_disabled_rows_and_stale_target_revision(tmp_path):
    root_obj = tmp_path / "releases"
    _write_release_manifest(root_obj, user_id_str="owner", pod_id_str="active", mode_str="live", account_route_str="U111")
    _write_release_manifest(root_obj, user_id_str="owner", pod_id_str="disabled", mode_str="live", account_route_str="U222", enabled_bool=False)
    _write_release_manifest(root_obj, user_id_str="owner", pod_id_str="paper", mode_str="paper", account_route_str="DU333")
    release_list = [parse_release_manifest(str(path_obj)) for path_obj in root_obj.rglob("*.yaml")]
    active_obj = next(release_obj for release_obj in release_list if release_obj.pod_id_str == "active")
    target_obj = SimpleNamespace(release_obj=active_obj, db_path_str=str(tmp_path / "live.sqlite"))
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path, [target_obj])
    provider_obj.releases_root_path_str = str(root_obj)
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert {row_dict["pod_id_str"] for row_dict in result_dict["release_list"]} == {"active", "disabled"}
    target_list[0] = SimpleNamespace(release_obj=SimpleNamespace(**{**vars(active_obj), "release_id_str": "old_release"}), db_path_str=target_obj.db_path_str)
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert result_dict["release_list"] == []
    assert result_dict["scope_verified_bool"] is False


def test_manifest_count_is_bounded_before_parsing(tmp_path, monkeypatch):
    provider_obj, workspace_dict, _target_list = _fixture_tuple(tmp_path)
    root_obj = tmp_path / "releases"
    root_obj.mkdir()
    for index_int in range(3):
        (root_obj / f"{index_int}.yaml").write_text("not a manifest")
    provider_obj.releases_root_path_str = str(root_obj)
    monkeypatch.setattr(system_data, "MANIFEST_LIMIT_INT", 2)
    monkeypatch.setattr(system_data, "parse_release_manifest", lambda path_str: pytest.fail("Unbounded manifests parsed"))
    assert _load_dict(provider_obj, workspace_dict)["release_list"] == []


def test_file_read_sizes_are_bounded_and_log_is_never_scanned(tmp_path, monkeypatch):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_sources(tmp_path, provider_obj, target_list)
    original_open_fn = Path.open
    count_dict = {}
    class TrackedFile:
        def __init__(self, path_obj, *argument_list, **argument_dict):
            self.path_obj = path_obj
            self.file_obj = original_open_fn(path_obj, *argument_list, **argument_dict)
        def __enter__(self):
            return self
        def __exit__(self, *argument_list):
            self.file_obj.close()
        def read(self, amount_int=-1):
            count_dict[self.path_obj.name] = amount_int
            assert 0 <= amount_int <= system_data.JSON_BYTE_LIMIT_INT + 1
            return self.file_obj.read(amount_int)
    monkeypatch.setattr(Path, "open", lambda path_obj, *argument_list, **argument_dict: TrackedFile(path_obj, *argument_list, **argument_dict))
    _load_dict(provider_obj, workspace_dict)
    assert count_dict == {"live_events.jsonl": 1, "ops_report_latest.json": system_data.JSON_BYTE_LIMIT_INT + 1}


def test_stale_summary_identity_rejects_runtime_sources_before_io(tmp_path, monkeypatch):
    provider_obj, workspace_dict, _target_list = _fixture_tuple(tmp_path)
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["user_id_str"] = "prior_owner"
    monkeypatch.setattr(Path, "open", lambda *argument_list, **argument_dict: pytest.fail("Old owner read runtime source"))
    assert _load_dict(provider_obj, workspace_dict)["release_list"] == []


def test_watchdog_missing_expected_live_pod_or_duplicate_row_is_not_verified(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    for report_list in ([], [{"pod_id_str": "pod_one", "account_route_str": "U123456", "mode_str": "live"}] * 2):
        _save_report(tmp_path, target_list, pod_report_dict_list=report_list)
        assert _load_dict(provider_obj, workspace_dict)["watchdog_dict"]["now_str"] == "No verified saved report"


def test_all_pods_must_have_flex_coverage_and_output_uses_oldest_covered_date(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path, [_target_obj(tmp_path), _target_obj(tmp_path, "pod_two", "U999")])
    path_str = _flex_path_str(tmp_path, target_list)
    with sqlite3.connect(path_str) as connection_obj:
        connection_obj.execute("UPDATE daily_performance SET market_date_str='2026-09-17' WHERE account_route_str='U999'")
    assert "2026-09-17" in _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["now_str"]
    with sqlite3.connect(path_str) as connection_obj:
        connection_obj.execute("DELETE FROM daily_performance WHERE account_route_str='U999'")
    assert _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["now_str"] == "No verified saved report"


def _save_run(tmp_path, target_list, **override_dict):
    report_dict = json.loads((tmp_path / "ops_report_latest.json").read_text(encoding="utf-8"))
    receipt_dict = {"schema_version_str": "live_ops_watchdog_run.v1", "completed_at_utc_str": BASE_TS.isoformat(),
        "report_generated_at_utc_str": report_dict["generated_at_utc_str"], "mode_str": report_dict["mode_str"],
        "report_sha256_str": hashlib.sha256(json.dumps(report_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest(),
        "scope_list": [{key_str: getattr(target_obj.release_obj, key_str) for key_str in system_data.IDENTITY_FIELD_TUPLE}
            for target_obj in target_list], "heartbeat_status_str": "sent", "heartbeat_fail_signal_bool": False,
        "notification_configured_bool": True, "notification_pending_live_count_int": 0}
    receipt_dict.update(override_dict)
    (tmp_path / "ops_report_latest.run.json").write_text(json.dumps(receipt_dict), encoding="utf-8")


def _save_alerts(tmp_path, *, pending_dict=None, **override_dict):
    state_dict = {"last_updated_str": BASE_TS.isoformat(), "pod_severity_map_dict": {"pod_one": "red"},
        "pending_red_previous_severity_map_dict": pending_dict if pending_dict is not None else {}}
    state_dict.update(override_dict)
    (tmp_path / "watchdog_notification_state.json").write_text(json.dumps(state_dict), encoding="utf-8")


@pytest.mark.parametrize("status_str,fail_signal_bool,state_str,label_str", [
    ("sent", False, "ok", "Ping sent"), ("sent", True, "ok", "Fail signal sent"),
    ("failed", False, "error", "Ping failed"), ("disabled", False, "skip", "Not configured")])
def test_paired_watchdog_receipt_proves_completion_and_delivery_outcome(tmp_path, status_str, fail_signal_bool, state_str, label_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list, generated_at_utc_str=(BASE_TS - timedelta(seconds=5)).isoformat())
    _save_run(tmp_path, target_list, heartbeat_status_str=status_str, heartbeat_fail_signal_bool=fail_signal_bool)
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert result_dict["watchdog_dict"]["now_str"] == "Run completed"
    assert result_dict["watchdog_dict"]["last_timestamp_str"] == BASE_TS.isoformat()
    assert result_dict["deadman_dict"]["state_str"] == state_str
    assert result_dict["deadman_dict"]["now_str"] == label_str
    assert result_dict["deadman_dict"]["checked_bool"] is (status_str != "disabled")


@pytest.mark.parametrize("change_str", ["hash", "release", "owner", "account", "missing_pod", "duplicate_pod", "mode",
    "future", "naive", "before_report", "report_time", "pending_bool", "negative_pending", "unknown_status", "boolean_type"])
def test_malformed_or_mismatched_existing_receipt_never_turns_neutral_or_completed(tmp_path, change_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list)
    _save_run(tmp_path, target_list)
    path_obj = tmp_path / "ops_report_latest.run.json"
    receipt_dict = json.loads(path_obj.read_text(encoding="utf-8"))
    if change_str in {"release", "owner", "account"}:
        receipt_dict["scope_list"][0][{"release": "release_id_str", "owner": "user_id_str", "account": "account_route_str"}[change_str]] = "foreign"
    elif change_str == "missing_pod":
        receipt_dict["scope_list"] = []
    elif change_str == "duplicate_pod":
        receipt_dict["scope_list"] *= 2
    else:
        field_str, value_obj = {"hash": ("report_sha256_str", "0" * 64), "mode": ("mode_str", "paper"),
            "future": ("completed_at_utc_str", (BASE_TS + timedelta(seconds=1)).isoformat()),
            "naive": ("completed_at_utc_str", BASE_TS.replace(tzinfo=None).isoformat()),
            "before_report": ("completed_at_utc_str", (BASE_TS - timedelta(seconds=1)).isoformat()),
            "report_time": ("report_generated_at_utc_str", (BASE_TS - timedelta(seconds=1)).isoformat()),
            "pending_bool": ("notification_pending_live_count_int", True), "negative_pending": ("notification_pending_live_count_int", -1),
            "unknown_status": ("heartbeat_status_str", "success"), "boolean_type": ("notification_configured_bool", "true")}[change_str]
        receipt_dict[field_str] = value_obj
    path_obj.write_text(json.dumps(receipt_dict), encoding="utf-8")
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert result_dict["deadman_dict"]["state_str"] == "unknown"
    assert result_dict["deadman_dict"]["checked_bool"] is True
    assert result_dict["watchdog_dict"]["now_str"] == "Report saved"


def test_old_receipt_and_new_report_cannot_reuse_an_old_heartbeat(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    old_ts = BASE_TS - timedelta(seconds=901)
    _save_report(tmp_path, target_list, generated_at_utc_str=old_ts.isoformat())
    _save_run(tmp_path, target_list, completed_at_utc_str=old_ts.isoformat())
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert result_dict["watchdog_dict"]["state_str"] == result_dict["deadman_dict"]["state_str"] == "warning"
    _save_report(tmp_path, target_list)
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert result_dict["watchdog_dict"]["now_str"] == "Report saved"
    assert result_dict["deadman_dict"]["state_str"] == "unknown"


@pytest.mark.parametrize("report_age_int,watchdog_state_str,label_str", [(900, "ok", "Run completed"),
    (901, "warning", "Report over 15 min old")])
def test_recent_completion_cannot_renew_a_historical_watchdog_report(tmp_path, report_age_int, watchdog_state_str, label_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list, generated_at_utc_str=(BASE_TS - timedelta(seconds=report_age_int)).isoformat())
    _save_run(tmp_path, target_list)
    result_dict = _load_dict(provider_obj, workspace_dict)
    assert result_dict["watchdog_dict"]["state_str"] == watchdog_state_str
    assert result_dict["watchdog_dict"]["now_str"] == label_str
    assert result_dict["deadman_dict"]["state_str"] == "ok"
    assert result_dict["deadman_dict"]["now_str"] == "Ping sent"
    assert result_dict["deadman_dict"]["last_timestamp_str"] == BASE_TS.isoformat()


@pytest.mark.parametrize("configured_bool,pending_int,state_str", [(True, 0, "ok"), (True, 1, "warning"), (False, None, "skip")])
def test_receipt_alert_evidence_when_old_notification_file_is_absent(tmp_path, configured_bool, pending_int, state_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list)
    _save_run(tmp_path, target_list, notification_configured_bool=configured_bool, notification_pending_live_count_int=pending_int)
    result_dict = _load_dict(provider_obj, workspace_dict)["alerts_dict"]
    assert result_dict["state_str"] == state_str
    assert result_dict["checked_bool"] is configured_bool
    assert "delivered" not in result_dict["now_str"].lower() or result_dict["now_str"] == "No saved undelivered alerts"


def test_explicit_live_alert_backlog_is_scoped_and_does_not_claim_active_retry(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    paper_obj = _target_obj(tmp_path, "paper", "DU999", mode_str="paper")
    _save_report(tmp_path, target_list + [paper_obj])
    _save_alerts(tmp_path, pending_dict={"pod_one": "green", "paper": "green", "__inspector__": "green"},
        pod_severity_map_dict={"pod_one": "red", "paper": "red", "__inspector__": "red"})
    result_dict = _load_dict(provider_obj, workspace_dict)["alerts_dict"]
    assert result_dict["state_str"] == "warning"
    assert result_dict["now_str"] == "1 alert pending retry"
    assert "retrying" not in result_dict["now_str"]
    _save_alerts(tmp_path, pending_dict={"paper": "green"}, pod_severity_map_dict={"pod_one": "green", "paper": "red"})
    assert _load_dict(provider_obj, workspace_dict)["alerts_dict"]["now_str"] == "No saved undelivered alerts"


def test_filtered_live_report_does_not_attribute_all_mode_inspector_failure_to_live(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list, mode_str="live")
    _save_alerts(tmp_path, pending_dict={"__inspector__": "green", "paper": "green"},
        pod_severity_map_dict={"pod_one": "green", "__inspector__": "red", "paper": "red"})
    result_dict = _load_dict(provider_obj, workspace_dict)["alerts_dict"]
    assert result_dict["state_str"] == "ok"
    assert result_dict["now_str"] == "No saved undelivered alerts"


@pytest.mark.parametrize("change_str", ["legacy", "wrong_time", "future", "wrong_type", "missing_live", "contradiction"])
def test_unproven_alert_state_does_not_infer_empty_backlog(tmp_path, change_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    _save_report(tmp_path, target_list)
    _save_alerts(tmp_path)
    path_obj = tmp_path / "watchdog_notification_state.json"
    state_dict = json.loads(path_obj.read_text(encoding="utf-8"))
    if change_str == "legacy":
        state_dict.pop("pending_red_previous_severity_map_dict")
        _save_run(tmp_path, target_list)
    elif change_str in {"wrong_time", "future"}:
        state_dict["last_updated_str"] = (BASE_TS + timedelta(seconds=1 if change_str == "future" else -1)).isoformat()
    elif change_str == "wrong_type":
        state_dict["pending_red_previous_severity_map_dict"] = []
    elif change_str == "missing_live":
        state_dict["pod_severity_map_dict"] = {"foreign": "green"}
    else:
        state_dict["pod_severity_map_dict"] = {"pod_one": "green"}
        state_dict["pending_red_previous_severity_map_dict"] = {"pod_one": "green"}
    path_obj.write_text(json.dumps(state_dict), encoding="utf-8")
    assert _load_dict(provider_obj, workspace_dict)["alerts_dict"]["state_str"] == "unknown"


def _save_attempt(path_str, *, status_str="success", timestamp_str="2026-09-21T10:15:00+00:00", from_str="2026-09-01", to_str="2026-09-18"):
    with sqlite3.connect(path_str) as connection_obj:
        connection_obj.execute("CREATE TABLE IF NOT EXISTS sync_attempt(attempt_id_int INTEGER PRIMARY KEY,attempted_timestamp_str,status_str,request_from_date_str,request_to_date_str,detail_str)")
        connection_obj.execute("INSERT INTO sync_attempt(attempted_timestamp_str,status_str,request_from_date_str,request_to_date_str,detail_str) VALUES(?,?,?,?,?)",
            (timestamp_str, status_str, from_str, to_str, "PRIVATE exception token C:/private/file"))


def test_repeated_success_and_failed_flex_attempts_use_actual_receipt_time(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list)
    _save_attempt(path_str)
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]
    assert result_dict["state_str"] == "ok"
    assert result_dict["last_timestamp_str"] == "2026-09-21T10:15:00+00:00"
    _save_attempt(path_str, status_str="failed", timestamp_str="2026-09-21T11:00:00+00:00")
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]
    assert result_dict["state_str"] == "error" and result_dict["now_str"].startswith("Latest sync failed")
    assert "PRIVATE" not in json.dumps(result_dict)
    _save_attempt(path_str, timestamp_str="2026-09-21T11:15:00+00:00")
    assert _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["state_str"] == "ok"


def test_failed_first_sync_is_visible_even_without_account_history(tmp_path):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list)
    with sqlite3.connect(path_str) as connection_obj:
        connection_obj.execute("DELETE FROM daily_performance")
    _save_attempt(path_str, status_str="failed")
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]
    assert result_dict["state_str"] == "error"
    assert result_dict["now_str"] == "Latest sync failed · coverage unavailable"
    _save_attempt(path_str, timestamp_str="2026-09-21T11:00:00+00:00")
    assert _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]["state_str"] == "unknown"


@pytest.mark.parametrize("asof_str,coverage_str,state_str", [
    ("2026-09-21T21:00:00+00:00", "2026-09-18", "ok"),
    ("2026-09-22T11:59:00+00:00", "2026-09-18", "now"),
    ("2026-09-22T12:00:00+00:00", "2026-09-18", "warning"),
    ("2026-09-19T11:59:00+00:00", "2026-09-17", "now"),
    ("2026-09-19T12:00:00+00:00", "2026-09-17", "warning"),
    ("2026-09-08T12:00:00+00:00", "2026-09-04", "ok"),
])
def test_flex_uses_existing_next_day_eight_et_sla_not_immediate_market_close(tmp_path, asof_str, coverage_str, state_str):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list, market_date_str=coverage_str, imported_str=coverage_str + "T23:00:00+00:00")
    result_dict = system_data.load_system_source_dict(provider_obj, workspace_dict,
        as_of_ts=datetime.fromisoformat(asof_str), performance_db_path_str=path_str)["flex_dict"]
    assert result_dict["state_str"] == state_str
    assert result_dict["checked_bool"] is True


@pytest.mark.parametrize("override_dict", [{"timestamp_str": "2026-09-21T16:00:01+00:00"},
    {"timestamp_str": "2026-09-21T10:15:00"}, {"from_str": "2026-09-20"}, {"to_str": "invalid"}, {"status_str": "running"}])
def test_invalid_flex_attempt_receipt_is_not_ignored(tmp_path, override_dict):
    provider_obj, workspace_dict, target_list = _fixture_tuple(tmp_path)
    path_str = _flex_path_str(tmp_path, target_list)
    _save_attempt(path_str, **override_dict)
    result_dict = _load_dict(provider_obj, workspace_dict, performance_db_path_str=path_str)["flex_dict"]
    assert result_dict["state_str"] == "unknown"
    assert result_dict["now_str"] == "Saved sync receipt unavailable"
