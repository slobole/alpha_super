from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    MANIFEST_FILE_NAME_STR,
    PRICE_FILE_NAME_STR,
    write_snapshot_files,
)
from scripts.serve_norgate_snapshot_api import (
    NORGATE_API_TOKEN_HEADER_STR,
    NorgateSnapshotApiService,
    make_handler_class,
)
from scripts.sync_norgate_snapshots_api import (
    derive_required_profile_list,
    post_requirements_dict,
    sync_required_snapshots,
)


TOKEN_STR = "secret-token"
CLIENT_ID_STR = "client_caspersky"
PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
SNAPSHOT_DATE_STR = "2024-01-02"


def _price_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": SNAPSHOT_DATE_STR,
                "symbol_str": "SPY",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
            },
            {
                "date": SNAPSHOT_DATE_STR,
                "symbol_str": "$VIX",
                "adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "Open": 13.0,
                "High": 14.0,
                "Low": 12.5,
                "Close": 13.5,
            },
        ]
    )


def _write_snapshot(snapshot_root_path_obj: Path, profile_str: str = PROFILE_STR) -> Path:
    snapshot_dir_path_obj = snapshot_root_path_obj / profile_str / SNAPSHOT_DATE_STR
    if snapshot_dir_path_obj.exists():
        return snapshot_dir_path_obj
    return write_snapshot_files(
        snapshot_root_str=str(snapshot_root_path_obj),
        profile_str=profile_str,
        snapshot_date_str=SNAPSHOT_DATE_STR,
        price_df=_price_snapshot_df(),
        required_symbol_list=["SPY"],
        required_helper_symbol_list=["$VIX"],
        adjustment_mode_map_dict={
            "SPY": CAPITALSPECIAL_ADJUSTMENT_STR,
            "$VIX": CAPITALSPECIAL_ADJUSTMENT_STR,
        },
    )


def _fake_exporter_fn(**kwargs) -> Path:
    return _write_snapshot(
        Path(kwargs["snapshot_root_str"]),
        profile_str=str(kwargs["profile_str"]),
    )


def _corrupting_exporter_fn(**kwargs) -> Path:
    snapshot_dir_path_obj = _write_snapshot(
        Path(kwargs["snapshot_root_str"]),
        profile_str=str(kwargs["profile_str"]),
    )
    (snapshot_dir_path_obj / PRICE_FILE_NAME_STR).write_bytes(b"corrupted")
    return snapshot_dir_path_obj


def _raising_exporter_fn(**_kwargs) -> Path:
    raise RuntimeError("export boom")


@contextmanager
def _api_server(tmp_path: Path, exporter_fn=_fake_exporter_fn):
    service_obj = NorgateSnapshotApiService(
        service_root_path_obj=tmp_path / "norgate_service",
        token_str=TOKEN_STR,
        exporter_fn=exporter_fn,
    )
    handler_cls = make_handler_class(service_obj)
    httpd_obj = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread_obj = threading.Thread(target=httpd_obj.serve_forever, daemon=True)
    thread_obj.start()
    try:
        host_str, port_int = httpd_obj.server_address
        yield f"http://{host_str}:{port_int}", service_obj
    finally:
        httpd_obj.shutdown()
        httpd_obj.server_close()
        thread_obj.join(timeout=5)


def _write_release_manifest(
    releases_root_path_obj: Path,
    *,
    file_name_str: str,
    release_id_str: str,
    pod_id_str: str,
    profile_str: str,
    mode_str: str = "paper",
    enabled_bool: bool = True,
) -> None:
    account_route_str = "U123456" if mode_str == "live" else "DU123456"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / file_name_str).write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: {release_id_str}",
                "  user_id: test_user",
                f"  pod_id: {pod_id_str}",
                "broker:",
                f"  account_route: {account_route_str}",
                "strategy:",
                "  strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy",
                f"  data_profile_str: {profile_str}",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "risk:",
                "  risk_profile_str: standard",
                "deployment:",
                f"  mode: {mode_str}",
                f"  enabled_bool: {'true' if enabled_bool else 'false'}",
            ]
        ),
        encoding="utf-8",
    )


def test_api_rejects_missing_token(tmp_path):
    with _api_server(tmp_path) as (api_url_str, _service_obj):
        request_obj = Request(f"{api_url_str}/v1/clients/{CLIENT_ID_STR}/status")

        with pytest.raises(HTTPError) as exc_info:
            urlopen(request_obj, timeout=5)

    assert exc_info.value.code == 401


def test_api_rejects_unsupported_profile_and_writes_failure_status(tmp_path):
    with _api_server(tmp_path) as (api_url_str, _service_obj):
        with pytest.raises(RuntimeError, match="Unsupported EOD Norgate profile"):
            post_requirements_dict(
                api_url_str=api_url_str,
                token_str=TOKEN_STR,
                client_id_str=CLIENT_ID_STR,
                profile_list=["intraday_1m_plus_daily_pit"],
            )

    client_dir_path_obj = tmp_path / "norgate_service" / CLIENT_ID_STR
    status_dict = json.loads((client_dir_path_obj / "export_status.json").read_text(encoding="utf-8"))
    assert status_dict["status_str"] == "failed"
    assert "intraday_1m_plus_daily_pit" in (client_dir_path_obj / "required_profiles.txt").read_text(
        encoding="utf-8"
    )


def test_api_writes_audit_files_and_serves_only_snapshot_artifacts(tmp_path):
    with _api_server(tmp_path) as (api_url_str, _service_obj):
        response_dict = post_requirements_dict(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            profile_list=[PROFILE_STR],
        )

        manifest_url_path_str = response_dict["snapshot_file_list"][0]["manifest_url_path_str"]
        manifest_request_obj = Request(
            f"{api_url_str}{manifest_url_path_str}",
            headers={NORGATE_API_TOKEN_HEADER_STR: TOKEN_STR},
        )
        with urlopen(manifest_request_obj, timeout=5) as response_obj:
            manifest_dict = json.loads(response_obj.read().decode("utf-8"))

        bad_file_request_obj = Request(
            f"{api_url_str}/v1/clients/{CLIENT_ID_STR}/snapshots/{PROFILE_STR}/{SNAPSHOT_DATE_STR}/extra.txt",
            headers={NORGATE_API_TOKEN_HEADER_STR: TOKEN_STR},
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(bad_file_request_obj, timeout=5)

    client_dir_path_obj = tmp_path / "norgate_service" / CLIENT_ID_STR
    snapshot_dir_path_obj = client_dir_path_obj / "snapshots" / PROFILE_STR / SNAPSHOT_DATE_STR
    assert response_dict["status_str"] == "ready"
    assert manifest_dict["profile"] == PROFILE_STR
    assert (client_dir_path_obj / "required_profiles.txt").exists()
    assert (client_dir_path_obj / "accepted_profiles.txt").exists()
    assert (client_dir_path_obj / "export_status.json").exists()
    assert (snapshot_dir_path_obj / MANIFEST_FILE_NAME_STR).exists()
    assert (snapshot_dir_path_obj / PRICE_FILE_NAME_STR).exists()
    assert exc_info.value.code == 404


def test_api_retention_keeps_current_snapshot_and_removes_other_date_dirs(tmp_path):
    service_root_path_obj = tmp_path / "norgate_service"
    profile_root_path_obj = service_root_path_obj / CLIENT_ID_STR / "snapshots" / PROFILE_STR
    old_snapshot_dir_path_obj = profile_root_path_obj / "2024-01-01"
    non_date_dir_path_obj = profile_root_path_obj / "notes"
    other_profile_dir_path_obj = (
        service_root_path_obj / CLIENT_ID_STR / "snapshots" / "norgate_eod_sp500_pit" / "2024-01-01"
    )
    other_client_dir_path_obj = service_root_path_obj / "client_other" / "snapshots" / PROFILE_STR / "2024-01-01"
    for path_obj in [
        old_snapshot_dir_path_obj,
        non_date_dir_path_obj,
        other_profile_dir_path_obj,
        other_client_dir_path_obj,
    ]:
        path_obj.mkdir(parents=True)
        (path_obj / "marker.txt").write_text("keep?", encoding="utf-8")

    with _api_server(tmp_path) as (api_url_str, _service_obj):
        post_requirements_dict(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            profile_list=[PROFILE_STR],
        )

    current_snapshot_dir_path_obj = profile_root_path_obj / SNAPSHOT_DATE_STR
    export_log_str = (service_root_path_obj / CLIENT_ID_STR / "export.log").read_text(encoding="utf-8")
    assert current_snapshot_dir_path_obj.exists()
    assert not old_snapshot_dir_path_obj.exists()
    assert non_date_dir_path_obj.exists()
    assert other_profile_dir_path_obj.exists()
    assert other_client_dir_path_obj.exists()
    assert "retention removed profile=norgate_eod_etf_plus_vix_helper dates=2024-01-01" in export_log_str


def test_api_retention_does_not_delete_old_snapshot_when_export_fails(tmp_path):
    service_root_path_obj = tmp_path / "norgate_service"
    old_snapshot_dir_path_obj = service_root_path_obj / CLIENT_ID_STR / "snapshots" / PROFILE_STR / "2024-01-01"
    old_snapshot_dir_path_obj.mkdir(parents=True)
    (old_snapshot_dir_path_obj / "marker.txt").write_text("keep", encoding="utf-8")

    with _api_server(tmp_path, exporter_fn=_raising_exporter_fn) as (api_url_str, _service_obj):
        with pytest.raises(RuntimeError, match="export boom"):
            post_requirements_dict(
                api_url_str=api_url_str,
                token_str=TOKEN_STR,
                client_id_str=CLIENT_ID_STR,
                profile_list=[PROFILE_STR],
            )

    status_dict = json.loads(
        (service_root_path_obj / CLIENT_ID_STR / "export_status.json").read_text(encoding="utf-8")
    )
    assert old_snapshot_dir_path_obj.exists()
    assert status_dict["status_str"] == "failed"


def test_client_sync_derives_profiles_and_promotes_valid_snapshot(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        file_name_str="enabled.yaml",
        release_id_str="release.enabled",
        pod_id_str="pod_enabled",
        profile_str=PROFILE_STR,
    )
    _write_release_manifest(
        releases_root_path_obj,
        file_name_str="enabled_live.yaml",
        release_id_str="release.enabled_live",
        pod_id_str="pod_enabled_live",
        profile_str="norgate_eod_sp500_pit",
        mode_str="live",
    )
    _write_release_manifest(
        releases_root_path_obj,
        file_name_str="disabled.yaml",
        release_id_str="release.disabled",
        pod_id_str="pod_disabled",
        profile_str="norgate_eod_sp500_pit",
        enabled_bool=False,
    )

    assert derive_required_profile_list(str(releases_root_path_obj)) == [
        PROFILE_STR,
        "norgate_eod_sp500_pit",
    ]
    assert derive_required_profile_list(str(releases_root_path_obj), "paper") == [PROFILE_STR]

    local_root_path_obj = tmp_path / "local_snapshots"
    monkeypatch.setenv("NORGATE_API_TOKEN", TOKEN_STR)
    with _api_server(tmp_path) as (api_url_str, _service_obj):
        promoted_path_list = sync_required_snapshots(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            releases_root_path_str=str(releases_root_path_obj),
            local_root_path_str=str(local_root_path_obj),
        )

    assert promoted_path_list == [
        local_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR,
        local_root_path_obj / "norgate_eod_sp500_pit" / SNAPSHOT_DATE_STR,
    ]
    for promoted_path_obj in promoted_path_list:
        assert (promoted_path_obj / MANIFEST_FILE_NAME_STR).exists()
        assert (promoted_path_obj / PRICE_FILE_NAME_STR).exists()


def test_client_sync_refuses_hash_mismatch_and_does_not_promote(tmp_path):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(
        releases_root_path_obj,
        file_name_str="enabled.yaml",
        release_id_str="release.enabled",
        pod_id_str="pod_enabled",
        profile_str=PROFILE_STR,
    )
    local_root_path_obj = tmp_path / "local_snapshots"

    with _api_server(tmp_path, exporter_fn=_corrupting_exporter_fn) as (api_url_str, _service_obj):
        with pytest.raises(Exception, match="SHA256 mismatch"):
            sync_required_snapshots(
                api_url_str=api_url_str,
                token_str=TOKEN_STR,
                client_id_str=CLIENT_ID_STR,
                releases_root_path_str=str(releases_root_path_obj),
                local_root_path_str=str(local_root_path_obj),
            )

    assert not (local_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR).exists()
