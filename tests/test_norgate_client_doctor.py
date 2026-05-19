from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pandas as pd

from data.norgate_snapshot_store import (
    ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR,
    CAPITALSPECIAL_ADJUSTMENT_STR,
    MANIFEST_FILE_NAME_STR,
    PRICE_FILE_NAME_STR,
    write_snapshot_files,
)
from scripts import norgate_config_env
from scripts.doctor_norgate_client import main as doctor_main, run_norgate_client_doctor
from scripts.serve_norgate_snapshot_api import NorgateSnapshotApiService, make_handler_class


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


@contextmanager
def _api_server(tmp_path: Path, exporter_fn=_fake_exporter_fn) -> Iterator[str]:
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
        yield f"http://{host_str}:{port_int}"
    finally:
        httpd_obj.shutdown()
        httpd_obj.server_close()
        thread_obj.join(timeout=5)


def _write_release_manifest(
    releases_root_path_obj: Path,
    *,
    file_name_str: str = "enabled.yaml",
    profile_str: str = PROFILE_STR,
    enabled_bool: bool = True,
) -> None:
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / file_name_str).write_text(
        "\n".join(
            [
                "identity:",
                "  release_id: release.enabled",
                "  user_id: test_user",
                "  pod_id: pod_enabled",
                "broker:",
                "  account_route: DU123456",
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
                "  mode: paper",
                f"  enabled_bool: {'true' if enabled_bool else 'false'}",
            ]
        ),
        encoding="utf-8",
    )


def _silent_print(_line_str: str) -> None:
    return None


def _status_dict(report_obj) -> dict[str, str]:
    return {check_obj.name_str: check_obj.status_str for check_obj in report_obj.check_list}


def test_client_doctor_passes_with_fake_api_and_report_json(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    local_root_path_obj = tmp_path / "local_snapshots"
    report_path_obj = tmp_path / "reports" / "client_doctor.json"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    with _api_server(tmp_path) as api_url_str:
        report_obj = run_norgate_client_doctor(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            releases_root_path_str=str(releases_root_path_obj),
            local_root_path_str=str(local_root_path_obj),
            report_json_path_str=str(report_path_obj),
            printer_fn=_silent_print,
        )

    status_dict = _status_dict(report_obj)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    assert report_obj.result_str == "PASS"
    assert status_dict["enabled release profiles"] == "PASS"
    assert status_dict["api healthz"] == "PASS"
    assert status_dict["api token auth"] == "PASS"
    assert status_dict["sync snapshots"] == "PASS"
    assert status_dict["manifest hash validation"] == "PASS"
    assert status_dict["scheduler snapshot heartbeat"] == "PASS"
    assert report_dict["result_str"] == "PASS"
    assert (local_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR / MANIFEST_FILE_NAME_STR).exists()


def test_client_doctor_fails_when_token_missing(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    report_obj = run_norgate_client_doctor(
        api_url_str="http://127.0.0.1:8787",
        token_str=None,
        client_id_str=CLIENT_ID_STR,
        releases_root_path_str=str(releases_root_path_obj),
        local_root_path_str=str(tmp_path / "local_snapshots"),
        printer_fn=_silent_print,
    )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["api token env"] == "FAIL"


def test_client_doctor_fails_when_snapshot_mode_not_enabled(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "false")

    report_obj = run_norgate_client_doctor(
        api_url_str="http://127.0.0.1:8787",
        token_str=TOKEN_STR,
        client_id_str=CLIENT_ID_STR,
        releases_root_path_str=str(releases_root_path_obj),
        local_root_path_str=str(tmp_path / "local_snapshots"),
        printer_fn=_silent_print,
    )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["snapshot mode env"] == "FAIL"


def test_client_doctor_fails_when_no_enabled_releases(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj, enabled_bool=False)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    with _api_server(tmp_path) as api_url_str:
        report_obj = run_norgate_client_doctor(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            releases_root_path_str=str(releases_root_path_obj),
            local_root_path_str=str(tmp_path / "local_snapshots"),
            printer_fn=_silent_print,
        )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["enabled release profiles"] == "FAIL"


def test_client_doctor_fails_on_unsupported_server_profile(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj, profile_str="intraday_1m_plus_daily_pit")
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    with _api_server(tmp_path) as api_url_str:
        report_obj = run_norgate_client_doctor(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            releases_root_path_str=str(releases_root_path_obj),
            local_root_path_str=str(tmp_path / "local_snapshots"),
            printer_fn=_silent_print,
        )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["server-supported profiles"] == "FAIL"


def test_client_doctor_fails_when_api_unreachable(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    report_obj = run_norgate_client_doctor(
        api_url_str="http://127.0.0.1:9",
        token_str=TOKEN_STR,
        client_id_str=CLIENT_ID_STR,
        releases_root_path_str=str(releases_root_path_obj),
        local_root_path_str=str(tmp_path / "local_snapshots"),
        timeout_seconds_float=0.2,
        printer_fn=_silent_print,
    )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["api healthz"] == "FAIL"


def test_client_doctor_fails_when_token_invalid(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    with _api_server(tmp_path) as api_url_str:
        report_obj = run_norgate_client_doctor(
            api_url_str=api_url_str,
            token_str="wrong-token",
            client_id_str=CLIENT_ID_STR,
            releases_root_path_str=str(releases_root_path_obj),
            local_root_path_str=str(tmp_path / "local_snapshots"),
            printer_fn=_silent_print,
        )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["api token auth"] == "FAIL"


def test_client_doctor_refuses_hash_mismatch_and_does_not_promote(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    local_root_path_obj = tmp_path / "local_snapshots"
    _write_release_manifest(releases_root_path_obj)
    monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "true")

    with _api_server(tmp_path, exporter_fn=_corrupting_exporter_fn) as api_url_str:
        report_obj = run_norgate_client_doctor(
            api_url_str=api_url_str,
            token_str=TOKEN_STR,
            client_id_str=CLIENT_ID_STR,
            releases_root_path_str=str(releases_root_path_obj),
            local_root_path_str=str(local_root_path_obj),
            printer_fn=_silent_print,
        )

    assert report_obj.result_str == "FAIL"
    assert _status_dict(report_obj)["sync snapshots"] == "FAIL"
    assert not (local_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR).exists()


def test_client_doctor_main_uses_config_env_over_stale_shell_env(tmp_path, monkeypatch):
    releases_root_path_obj = tmp_path / "releases"
    local_root_path_obj = tmp_path / "local_snapshots"
    stale_root_path_obj = tmp_path / "stale_snapshots"
    _write_release_manifest(releases_root_path_obj)

    with _api_server(tmp_path) as api_url_str:
        config_path_obj = tmp_path / "config.env"
        config_path_obj.write_text(
            "\n".join(
                [
                    "ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true",
                    f"NORGATE_API_TOKEN={TOKEN_STR}",
                    f"NORGATE_API_URL={api_url_str}",
                    f"NORGATE_CLIENT_ID={CLIENT_ID_STR}",
                    f"NORGATE_RELEASES_ROOT={releases_root_path_obj}",
                    f"NORGATE_SNAPSHOT_ROOT={local_root_path_obj}",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "false")
        monkeypatch.setenv("NORGATE_API_TOKEN", "wrong-token")
        monkeypatch.setenv("NORGATE_API_URL", "http://127.0.0.1:9")
        monkeypatch.setenv("NORGATE_CLIENT_ID", "wrong-client")
        monkeypatch.setenv("NORGATE_RELEASES_ROOT", str(tmp_path / "missing_releases"))
        monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(stale_root_path_obj))
        monkeypatch.setattr(norgate_config_env, "default_config_env_path_obj", lambda: config_path_obj)
        monkeypatch.setattr(sys, "argv", ["doctor_norgate_client.py"])

        assert doctor_main() == 0

    assert (local_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR / MANIFEST_FILE_NAME_STR).exists()
    assert not (stale_root_path_obj / PROFILE_STR / SNAPSHOT_DATE_STR).exists()
