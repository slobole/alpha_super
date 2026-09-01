from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pytest

from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    write_snapshot_files,
)
from scripts.doctor_norgate_server import run_norgate_server_doctor
from scripts.serve_norgate_snapshot_api import (
    NORGATE_API_TOKEN_ENV_STR,
    NorgateSnapshotApiService,
    make_handler_class,
)


TOKEN_STR = "secret-token"
CLIENT_ID_STR = "doctor_server"
PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
SNAPSHOT_DATE_STR = "2026-05-18"


class FakeNorgateModule:
    def price_timeseries(self, symbol_str: str, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.DatetimeIndex(["2026-05-15", SNAPSHOT_DATE_STR]),
        )


def _silent_print(_line_str: str) -> None:
    return None


def _price_snapshot_df(
    *,
    spxtr_price_adjustment_sequence: Sequence[str] = (
        TOTALRETURN_ADJUSTMENT_STR,
    ),
    spxtr_price_date_str: str = SNAPSHOT_DATE_STR,
    spxtr_close_float: float = 10050.0,
) -> pd.DataFrame:
    price_record_list = [
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
    for spxtr_adjustment_str in spxtr_price_adjustment_sequence:
        price_record_list.append(
            {
                "date": spxtr_price_date_str,
                "symbol_str": "$SPXTR",
                "adjustment_str": spxtr_adjustment_str,
                "Open": 10000.0,
                "High": 10100.0,
                "Low": 9900.0,
                "Close": spxtr_close_float,
            }
        )
    return pd.DataFrame(price_record_list)


def _write_snapshot(
    snapshot_root_path_obj: Path,
    profile_str: str = PROFILE_STR,
    *,
    spxtr_required_bool: bool = True,
    spxtr_manifest_adjustment_obj: object | None = TOTALRETURN_ADJUSTMENT_STR,
    spxtr_price_adjustment_sequence: Sequence[str] = (
        TOTALRETURN_ADJUSTMENT_STR,
    ),
    spxtr_price_date_str: str = SNAPSHOT_DATE_STR,
    spxtr_close_float: float = 10050.0,
) -> Path:
    snapshot_dir_path_obj = snapshot_root_path_obj / profile_str / SNAPSHOT_DATE_STR
    if snapshot_dir_path_obj.exists():
        return snapshot_dir_path_obj
    return write_snapshot_files(
        snapshot_root_str=str(snapshot_root_path_obj),
        profile_str=profile_str,
        snapshot_date_str=SNAPSHOT_DATE_STR,
        price_df=_price_snapshot_df(
            spxtr_price_adjustment_sequence=spxtr_price_adjustment_sequence,
            spxtr_price_date_str=spxtr_price_date_str,
            spxtr_close_float=spxtr_close_float,
        ),
        required_symbol_list=["SPY", *(["$SPXTR"] if spxtr_required_bool else [])],
        required_helper_symbol_list=["$VIX"],
        adjustment_mode_map_dict={
            "SPY": CAPITALSPECIAL_ADJUSTMENT_STR,
            "$VIX": CAPITALSPECIAL_ADJUSTMENT_STR,
            **({"$SPXTR": spxtr_manifest_adjustment_obj} if spxtr_manifest_adjustment_obj is not None else {}),
        },
    )


def _fake_exporter_fn(**kwargs) -> Path:
    return _write_snapshot(
        Path(kwargs["snapshot_root_str"]),
        profile_str=str(kwargs["profile_str"]),
    )


@contextmanager
def _api_server(service_root_path_obj: Path) -> Iterator[str]:
    service_obj = NorgateSnapshotApiService(
        service_root_path_obj=service_root_path_obj,
        token_str=TOKEN_STR,
        exporter_fn=_fake_exporter_fn,
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


@contextmanager
def _raw_http_server(handler_cls: type[BaseHTTPRequestHandler]) -> Iterator[str]:
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


def _check_status_dict(report_obj) -> dict[str, str]:
    return {check_obj.name_str: check_obj.status_str for check_obj in report_obj.check_list}


def test_server_doctor_passes_with_fake_norgate_api_and_report_json(tmp_path, monkeypatch):
    service_root_path_obj = tmp_path / "norgate_service"
    report_path_obj = tmp_path / "doctor_reports" / "server_doctor_latest.json"
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    with _api_server(service_root_path_obj) as api_url_str:
        report_obj = run_norgate_server_doctor(
            service_root_path_str=str(service_root_path_obj),
            api_url_str=api_url_str,
            min_free_gb_float=0.0,
            timeout_seconds_float=5.0,
            report_json_path_str=str(report_path_obj),
            norgate_loader_fn=lambda: FakeNorgateModule(),
            printer_fn=_silent_print,
        )

    status_dict = _check_status_dict(report_obj)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    client_dir_path_obj = service_root_path_obj / CLIENT_ID_STR
    snapshot_dir_path_obj = client_dir_path_obj / "snapshots" / PROFILE_STR / SNAPSHOT_DATE_STR

    assert report_obj.result_str == "PASS"
    assert status_dict["norgatedata import"] == "PASS"
    assert status_dict["SPY query"] == "PASS"
    assert status_dict["$SPX query"] == "PASS"
    assert status_dict["api healthz"] == "PASS"
    assert status_dict["api token auth"] == "PASS"
    assert status_dict[f"export {PROFILE_STR}"] == "PASS"
    assert status_dict["manifest hash validation"] == "PASS"
    assert status_dict["$SPXTR TOTALRETURN snapshot contract"] == "PASS"
    assert report_dict["result_str"] == "PASS"
    assert report_dict["check_list"][0]["timestamp_utc_str"]
    assert (client_dir_path_obj / "required_profiles.txt").exists()
    assert (client_dir_path_obj / "accepted_profiles.txt").exists()
    assert (client_dir_path_obj / "export_status.json").exists()
    assert (client_dir_path_obj / "export.log").exists()
    assert snapshot_dir_path_obj.exists()


@pytest.mark.parametrize(
    (
        "spxtr_required_bool",
        "spxtr_manifest_adjustment_obj",
        "spxtr_price_adjustment_sequence",
        "expected_detail_str",
    ),
    [
        (
            False,
            TOTALRETURN_ADJUSTMENT_STR,
            (TOTALRETURN_ADJUSTMENT_STR,),
            "manifest required_symbols",
        ),
        (
            True,
            None,
            (TOTALRETURN_ADJUSTMENT_STR,),
            "manifest adjustment_modes",
        ),
        (
            True,
            TOTALRETURN_ADJUSTMENT_STR,
            (),
            "prices.parquet adjustments",
        ),
        (
            True,
            [TOTALRETURN_ADJUSTMENT_STR, CAPITALSPECIAL_ADJUSTMENT_STR],
            (TOTALRETURN_ADJUSTMENT_STR,),
            "manifest adjustment_modes",
        ),
        (
            True,
            TOTALRETURN_ADJUSTMENT_STR,
            (TOTALRETURN_ADJUSTMENT_STR, CAPITALSPECIAL_ADJUSTMENT_STR),
            "prices.parquet adjustments",
        ),
    ],
)
def test_server_doctor_fails_each_invalid_spxtr_total_return_contract_part(
    tmp_path,
    monkeypatch,
    spxtr_required_bool,
    spxtr_manifest_adjustment_obj,
    spxtr_price_adjustment_sequence,
    expected_detail_str,
):
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    def invalid_spxtr_exporter_fn(**kwargs) -> Path:
        return _write_snapshot(
            Path(kwargs["snapshot_root_str"]),
            profile_str=str(kwargs["profile_str"]),
            spxtr_required_bool=spxtr_required_bool,
            spxtr_manifest_adjustment_obj=spxtr_manifest_adjustment_obj,
            spxtr_price_adjustment_sequence=spxtr_price_adjustment_sequence,
        )

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(tmp_path / "norgate_service"),
        min_free_gb_float=0.0,
        norgate_loader_fn=lambda: FakeNorgateModule(),
        exporter_fn=invalid_spxtr_exporter_fn,
        printer_fn=_silent_print,
    )

    status_dict = _check_status_dict(report_obj)
    detail_dict = {
        check_obj.name_str: check_obj.detail_str
        for check_obj in report_obj.check_list
    }
    assert report_obj.result_str == "FAIL"
    assert status_dict["manifest hash validation"] == "PASS"
    assert status_dict["$SPXTR TOTALRETURN snapshot contract"] == "FAIL"
    assert expected_detail_str in detail_dict["$SPXTR TOTALRETURN snapshot contract"]


def test_server_doctor_accepts_single_item_total_return_adjustment_list(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    def list_adjustment_exporter_fn(**kwargs) -> Path:
        return _write_snapshot(
            Path(kwargs["snapshot_root_str"]),
            profile_str=str(kwargs["profile_str"]),
            spxtr_manifest_adjustment_obj=[TOTALRETURN_ADJUSTMENT_STR],
        )

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(tmp_path / "norgate_service"),
        min_free_gb_float=0.0,
        norgate_loader_fn=lambda: FakeNorgateModule(),
        exporter_fn=list_adjustment_exporter_fn,
        printer_fn=_silent_print,
    )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "PASS"
    assert status_dict["$SPXTR TOTALRETURN snapshot contract"] == "PASS"


@pytest.mark.parametrize(
    ("spxtr_price_date_str", "spxtr_close_float"),
    [
        ("2026-05-15", 10050.0),
        (SNAPSHOT_DATE_STR, float("inf")),
    ],
)
def test_server_doctor_fails_when_latest_spxtr_close_is_not_usable(
    tmp_path,
    monkeypatch,
    spxtr_price_date_str,
    spxtr_close_float,
):
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    def unusable_close_exporter_fn(**kwargs) -> Path:
        return _write_snapshot(
            Path(kwargs["snapshot_root_str"]),
            profile_str=str(kwargs["profile_str"]),
            spxtr_price_date_str=spxtr_price_date_str,
            spxtr_close_float=spxtr_close_float,
        )

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(tmp_path / "norgate_service"),
        min_free_gb_float=0.0,
        norgate_loader_fn=lambda: FakeNorgateModule(),
        exporter_fn=unusable_close_exporter_fn,
        printer_fn=_silent_print,
    )

    detail_dict = {
        check_obj.name_str: check_obj.detail_str
        for check_obj in report_obj.check_list
    }
    assert report_obj.result_str == "FAIL"
    assert (
        "prices.parquet latest usable session"
        in detail_dict["$SPXTR TOTALRETURN snapshot contract"]
    )


def test_server_doctor_skips_spxtr_contract_for_profile_that_does_not_require_it(
    tmp_path,
    monkeypatch,
):
    ndx_profile_str = "norgate_eod_ndx_pit_plus_vxn_helper"
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    def ndx_exporter_fn(**kwargs) -> Path:
        return _write_snapshot(
            Path(kwargs["snapshot_root_str"]),
            profile_str=str(kwargs["profile_str"]),
            spxtr_required_bool=False,
            spxtr_manifest_adjustment_obj=None,
            spxtr_price_adjustment_sequence=(),
        )

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(tmp_path / "norgate_service"),
        profile_str=ndx_profile_str,
        min_free_gb_float=0.0,
        norgate_loader_fn=lambda: FakeNorgateModule(),
        exporter_fn=ndx_exporter_fn,
        printer_fn=_silent_print,
    )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "PASS"
    assert "$SPXTR TOTALRETURN snapshot contract" not in status_dict


def test_server_doctor_fails_when_token_missing(tmp_path, monkeypatch):
    monkeypatch.delenv(NORGATE_API_TOKEN_ENV_STR, raising=False)

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(tmp_path / "norgate_service"),
        min_free_gb_float=0.0,
        norgate_loader_fn=lambda: FakeNorgateModule(),
        exporter_fn=_fake_exporter_fn,
        printer_fn=_silent_print,
    )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "FAIL"
    assert status_dict["api token env"] == "FAIL"


def test_server_doctor_fails_when_api_health_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(tmp_path / "norgate_service"),
        api_url_str="http://127.0.0.1:9",
        min_free_gb_float=0.0,
        timeout_seconds_float=0.2,
        norgate_loader_fn=lambda: FakeNorgateModule(),
        printer_fn=_silent_print,
    )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "FAIL"
    assert status_dict["api healthz"] == "FAIL"


def test_server_doctor_fails_when_api_accepts_missing_token(tmp_path, monkeypatch):
    class InsecureHandler(BaseHTTPRequestHandler):
        def log_message(self, _format_str, *_args) -> None:
            return None

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/healthz":
                self.wfile.write(b'{"status_str":"ok"}')
            else:
                self.wfile.write(b'{"status_str":"ready"}')

        def do_POST(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status_str":"failed","snapshot_file_list":[]}')

    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    with _raw_http_server(InsecureHandler) as api_url_str:
        report_obj = run_norgate_server_doctor(
            service_root_path_str=str(tmp_path / "norgate_service"),
            api_url_str=api_url_str,
            min_free_gb_float=0.0,
            timeout_seconds_float=5.0,
            norgate_loader_fn=lambda: FakeNorgateModule(),
            printer_fn=_silent_print,
        )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "FAIL"
    assert status_dict["api rejects missing token"] == "FAIL"


def test_server_doctor_fails_when_export_status_not_ready(tmp_path, monkeypatch):
    class NotReadyHandler(BaseHTTPRequestHandler):
        def log_message(self, _format_str, *_args) -> None:
            return None

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status_str":"ok"}')
                return
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status_str":"failed"}')

        def do_POST(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status_str":"failed","snapshot_file_list":[]}')

    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    with _raw_http_server(NotReadyHandler) as api_url_str:
        report_obj = run_norgate_server_doctor(
            service_root_path_str=str(tmp_path / "norgate_service"),
            api_url_str=api_url_str,
            min_free_gb_float=0.0,
            timeout_seconds_float=5.0,
            norgate_loader_fn=lambda: FakeNorgateModule(),
            printer_fn=_silent_print,
        )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "FAIL"
    assert status_dict[f"export {PROFILE_STR}"] == "FAIL"


def test_server_doctor_fails_when_manifest_validation_fails(tmp_path, monkeypatch):
    def failing_manifest_loader_fn(*_args, **_kwargs):
        raise RuntimeError("SHA256 mismatch")

    service_root_path_obj = tmp_path / "norgate_service"
    monkeypatch.setenv(NORGATE_API_TOKEN_ENV_STR, TOKEN_STR)

    with _api_server(service_root_path_obj) as api_url_str:
        report_obj = run_norgate_server_doctor(
            service_root_path_str=str(service_root_path_obj),
            api_url_str=api_url_str,
            min_free_gb_float=0.0,
            timeout_seconds_float=5.0,
            norgate_loader_fn=lambda: FakeNorgateModule(),
            manifest_loader_fn=failing_manifest_loader_fn,
            printer_fn=_silent_print,
        )

    status_dict = _check_status_dict(report_obj)
    assert report_obj.result_str == "FAIL"
    assert status_dict["manifest hash validation"] == "FAIL"
