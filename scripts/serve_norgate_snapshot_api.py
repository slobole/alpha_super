from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from data.norgate_snapshot_store import (
    MANIFEST_FILE_NAME_STR,
    PRICE_FILE_NAME_STR,
    UNIVERSE_FILE_NAME_STR,
)
from scripts.export_norgate_snapshot import (
    SUPPORTED_EOD_PROFILE_TUPLE,
    export_profile_snapshot,
)
from scripts.norgate_config_env import (
    NORGATE_API_HOST_ENV_STR,
    NORGATE_API_PORT_ENV_STR,
    NORGATE_SERVICE_ROOT_ENV_STR,
    env_int,
    env_str,
    load_config_env_file,
)


NORGATE_API_TOKEN_ENV_STR = "NORGATE_API_TOKEN"
NORGATE_API_TOKEN_HEADER_STR = "X-Alpha-Norgate-Token"
ALLOWED_SNAPSHOT_FILE_SET = {
    MANIFEST_FILE_NAME_STR,
    PRICE_FILE_NAME_STR,
    UNIVERSE_FILE_NAME_STR,
}

ExporterFn = Callable[..., Path]


class NorgateApiError(RuntimeError):
    def __init__(
        self,
        status_code_int: int,
        message_str: str,
        payload_dict: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message_str)
        self.status_code_int = int(status_code_int)
        self.message_str = str(message_str)
        self.payload_dict = payload_dict or {
            "status_str": "failed",
            "error_str": self.message_str,
        }


def _utc_now_str() -> str:
    return datetime.now(tz=UTC).isoformat()


def _validate_path_component_str(raw_value_str: str, field_name_str: str) -> str:
    value_str = str(raw_value_str).strip()
    if not value_str:
        raise NorgateApiError(400, f"{field_name_str} must not be empty.")
    allowed_char_set = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-$")
    if any(char_str not in allowed_char_set for char_str in value_str):
        raise NorgateApiError(400, f"{field_name_str} contains unsupported characters: {value_str!r}.")
    if value_str in {".", ".."}:
        raise NorgateApiError(400, f"{field_name_str} is invalid: {value_str!r}.")
    return value_str


def _write_text_atomic(path_obj: Path, text_str: str) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path_obj = path_obj.with_name(f".{path_obj.name}.tmp")
    tmp_path_obj.write_text(text_str, encoding="utf-8")
    tmp_path_obj.replace(path_obj)


def _write_json_atomic(path_obj: Path, payload_dict: dict[str, Any]) -> None:
    _write_text_atomic(
        path_obj,
        json.dumps(payload_dict, indent=2, sort_keys=True) + "\n",
    )


def _read_json_dict(path_obj: Path) -> dict[str, Any]:
    payload_obj = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise NorgateApiError(500, f"JSON file is not an object: {path_obj}")
    return payload_obj


class NorgateSnapshotApiService:
    def __init__(
        self,
        *,
        service_root_path_obj: Path,
        token_str: str,
        exporter_fn: ExporterFn = export_profile_snapshot,
        start_date_str: str = "1990-01-01",
    ) -> None:
        self.service_root_path_obj = service_root_path_obj.expanduser()
        self.client_root_path_obj = self.service_root_path_obj / "clients"
        self.snapshot_root_path_obj = self.service_root_path_obj / "snapshots"
        self.token_str = str(token_str)
        self.exporter_fn = exporter_fn
        self.start_date_str = str(start_date_str)

    def token_matches_bool(self, header_token_str: str | None) -> bool:
        if not self.token_str:
            return False
        return hmac.compare_digest(str(header_token_str or ""), self.token_str)

    def client_dir_path_obj(self, client_id_str: str) -> Path:
        safe_client_id_str = _validate_path_component_str(client_id_str, "client_id_str")
        return self.client_root_path_obj / safe_client_id_str

    def _status_path_obj(self, client_id_str: str) -> Path:
        return self.client_dir_path_obj(client_id_str) / "export_status.json"

    def _append_log(self, client_id_str: str, line_str: str) -> None:
        log_path_obj = self.client_dir_path_obj(client_id_str) / "export.log"
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with log_path_obj.open("a", encoding="utf-8") as log_file_obj:
            log_file_obj.write(f"{_utc_now_str()} {line_str}\n")

    def _write_profile_text_files(
        self,
        *,
        client_id_str: str,
        requested_profile_list: list[str],
        accepted_profile_list: list[str],
    ) -> None:
        client_dir_path_obj = self.client_dir_path_obj(client_id_str)
        _write_text_atomic(
            client_dir_path_obj / "required_profiles.txt",
            "\n".join(requested_profile_list) + "\n",
        )
        _write_text_atomic(
            client_dir_path_obj / "accepted_profiles.txt",
            "\n".join(accepted_profile_list) + "\n",
        )

    def _write_status(self, client_id_str: str, status_dict: dict[str, Any]) -> None:
        _write_json_atomic(self._status_path_obj(client_id_str), status_dict)

    def _build_snapshot_file_dict(self, client_id_str: str, profile_str: str, snapshot_dir_path_obj: Path) -> dict[str, Any]:
        manifest_path_obj = snapshot_dir_path_obj / MANIFEST_FILE_NAME_STR
        if not manifest_path_obj.exists():
            raise NorgateApiError(500, f"Export completed without manifest: {manifest_path_obj}")

        manifest_dict = _read_json_dict(manifest_path_obj)
        files_dict = manifest_dict.get("files", {})
        if not isinstance(files_dict, dict) or PRICE_FILE_NAME_STR not in files_dict:
            raise NorgateApiError(500, f"Manifest is missing {PRICE_FILE_NAME_STR}: {manifest_path_obj}")

        snapshot_date_str = snapshot_dir_path_obj.name
        file_url_path_dict = {
            MANIFEST_FILE_NAME_STR: (
                f"/v1/clients/{client_id_str}/snapshots/{profile_str}/{snapshot_date_str}/{MANIFEST_FILE_NAME_STR}"
            )
        }
        for file_name_str in sorted(files_dict):
            if file_name_str in ALLOWED_SNAPSHOT_FILE_SET:
                file_url_path_dict[file_name_str] = (
                    f"/v1/clients/{client_id_str}/snapshots/{profile_str}/{snapshot_date_str}/{file_name_str}"
                )

        return {
            "profile_str": profile_str,
            "snapshot_date_str": snapshot_date_str,
            "manifest_url_path_str": file_url_path_dict[MANIFEST_FILE_NAME_STR],
            "file_url_path_dict": file_url_path_dict,
        }

    def handle_requirements_dict(
        self,
        client_id_str: str,
        request_payload_dict: dict[str, Any],
    ) -> dict[str, Any]:
        safe_client_id_str = _validate_path_component_str(client_id_str, "client_id_str")
        raw_profile_list_obj = request_payload_dict.get("profile_list")
        if not isinstance(raw_profile_list_obj, list):
            raise NorgateApiError(400, "profile_list must be a list.")

        requested_profile_list = [
            str(profile_obj).strip()
            for profile_obj in raw_profile_list_obj
            if str(profile_obj).strip()
        ]
        requested_profile_list = list(dict.fromkeys(requested_profile_list))
        if len(requested_profile_list) == 0:
            raise NorgateApiError(400, "profile_list must contain at least one profile.")

        unsupported_profile_list = [
            profile_str
            for profile_str in requested_profile_list
            if profile_str not in SUPPORTED_EOD_PROFILE_TUPLE
        ]
        if len(unsupported_profile_list) > 0:
            error_str = f"Unsupported EOD Norgate profile(s): {unsupported_profile_list}"
            status_dict = {
                "client_id_str": safe_client_id_str,
                "status_str": "failed",
                "snapshot_date_str": None,
                "requested_profile_list": requested_profile_list,
                "accepted_profile_list": [],
                "snapshot_file_list": [],
                "generated_timestamp_utc_str": _utc_now_str(),
                "error_str": error_str,
            }
            self._write_profile_text_files(
                client_id_str=safe_client_id_str,
                requested_profile_list=requested_profile_list,
                accepted_profile_list=[],
            )
            self._write_status(safe_client_id_str, status_dict)
            self._append_log(safe_client_id_str, error_str)
            raise NorgateApiError(400, error_str, status_dict)

        self._write_profile_text_files(
            client_id_str=safe_client_id_str,
            requested_profile_list=requested_profile_list,
            accepted_profile_list=requested_profile_list,
        )

        snapshot_file_list: list[dict[str, Any]] = []
        try:
            for profile_str in requested_profile_list:
                snapshot_dir_path_obj = self.exporter_fn(
                    snapshot_root_str=str(self.snapshot_root_path_obj),
                    profile_str=profile_str,
                    snapshot_date_str=None,
                    start_date_str=self.start_date_str,
                    end_date_str=None,
                    overwrite_bool=False,
                )
                snapshot_file_list.append(
                    self._build_snapshot_file_dict(
                        safe_client_id_str,
                        profile_str,
                        Path(snapshot_dir_path_obj),
                    )
                )
        except Exception as exc:
            error_str = f"Norgate export failed: {exc}"
            status_dict = {
                "client_id_str": safe_client_id_str,
                "status_str": "failed",
                "snapshot_date_str": None,
                "requested_profile_list": requested_profile_list,
                "accepted_profile_list": requested_profile_list,
                "snapshot_file_list": [],
                "generated_timestamp_utc_str": _utc_now_str(),
                "error_str": error_str,
            }
            self._write_status(safe_client_id_str, status_dict)
            self._append_log(safe_client_id_str, error_str)
            raise NorgateApiError(500, error_str, status_dict) from exc

        snapshot_date_set = {
            str(snapshot_file_dict["snapshot_date_str"])
            for snapshot_file_dict in snapshot_file_list
        }
        snapshot_date_str = sorted(snapshot_date_set)[-1] if len(snapshot_date_set) == 1 else None
        status_dict = {
            "client_id_str": safe_client_id_str,
            "status_str": "ready",
            "snapshot_date_str": snapshot_date_str,
            "requested_profile_list": requested_profile_list,
            "accepted_profile_list": requested_profile_list,
            "snapshot_file_list": snapshot_file_list,
            "generated_timestamp_utc_str": _utc_now_str(),
            "error_str": None,
        }
        self._write_status(safe_client_id_str, status_dict)
        self._append_log(
            safe_client_id_str,
            f"ready profiles={','.join(requested_profile_list)} snapshot_date={snapshot_date_str}",
        )
        return status_dict

    def load_status_dict(self, client_id_str: str) -> dict[str, Any]:
        status_path_obj = self._status_path_obj(client_id_str)
        if not status_path_obj.exists():
            raise NorgateApiError(404, f"No export status exists for client {client_id_str}.")
        return _read_json_dict(status_path_obj)

    def resolve_snapshot_file_path_obj(
        self,
        *,
        client_id_str: str,
        profile_str: str,
        snapshot_date_str: str,
        file_name_str: str,
    ) -> Path:
        safe_client_id_str = _validate_path_component_str(client_id_str, "client_id_str")
        safe_profile_str = _validate_path_component_str(profile_str, "profile_str")
        safe_snapshot_date_str = _validate_path_component_str(snapshot_date_str, "snapshot_date_str")
        safe_file_name_str = _validate_path_component_str(file_name_str, "file_name_str")
        if safe_file_name_str not in ALLOWED_SNAPSHOT_FILE_SET:
            raise NorgateApiError(404, f"Unsupported snapshot file: {safe_file_name_str}")

        status_dict = self.load_status_dict(safe_client_id_str)
        accepted_profile_set = {str(profile_str) for profile_str in status_dict.get("accepted_profile_list", [])}
        if safe_profile_str not in accepted_profile_set:
            raise NorgateApiError(403, f"Client {safe_client_id_str} has not accepted profile {safe_profile_str}.")

        snapshot_file_path_obj = (
            self.snapshot_root_path_obj
            / safe_profile_str
            / safe_snapshot_date_str
            / safe_file_name_str
        )
        if not snapshot_file_path_obj.exists():
            raise NorgateApiError(404, f"Snapshot file does not exist: {safe_file_name_str}")
        return snapshot_file_path_obj


def _json_response_bytes(payload_dict: dict[str, Any]) -> bytes:
    return json.dumps(payload_dict, sort_keys=True).encode("utf-8")


def make_handler_class(service_obj: NorgateSnapshotApiService):
    class NorgateSnapshotApiHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code_int: int, payload_dict: dict[str, Any]) -> None:
            response_bytes = _json_response_bytes(payload_dict)
            self.send_response(status_code_int)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        def _send_error(self, error_obj: NorgateApiError) -> None:
            self._send_json(error_obj.status_code_int, error_obj.payload_dict)

        def _authenticated_bool(self) -> bool:
            return service_obj.token_matches_bool(self.headers.get(NORGATE_API_TOKEN_HEADER_STR))

        def _path_part_list(self) -> list[str]:
            parsed_url_obj = urlparse(self.path)
            return [
                unquote(part_str)
                for part_str in parsed_url_obj.path.split("/")
                if part_str
            ]

        def do_GET(self) -> None:
            try:
                path_part_list = self._path_part_list()
                if path_part_list == ["healthz"]:
                    self._send_json(200, {"status_str": "ok"})
                    return
                if not self._authenticated_bool():
                    raise NorgateApiError(401, "Missing or invalid Norgate API token.")

                if len(path_part_list) == 4 and path_part_list[:2] == ["v1", "clients"] and path_part_list[3] == "status":
                    self._send_json(200, service_obj.load_status_dict(path_part_list[2]))
                    return

                if (
                    len(path_part_list) == 7
                    and path_part_list[:2] == ["v1", "clients"]
                    and path_part_list[3] == "snapshots"
                ):
                    snapshot_file_path_obj = service_obj.resolve_snapshot_file_path_obj(
                        client_id_str=path_part_list[2],
                        profile_str=path_part_list[4],
                        snapshot_date_str=path_part_list[5],
                        file_name_str=path_part_list[6],
                    )
                    response_bytes = snapshot_file_path_obj.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(response_bytes)))
                    self.end_headers()
                    self.wfile.write(response_bytes)
                    return

                raise NorgateApiError(404, "Unknown endpoint.")
            except NorgateApiError as exc:
                self._send_error(exc)

        def do_POST(self) -> None:
            try:
                if not self._authenticated_bool():
                    raise NorgateApiError(401, "Missing or invalid Norgate API token.")

                path_part_list = self._path_part_list()
                if (
                    len(path_part_list) == 4
                    and path_part_list[:2] == ["v1", "clients"]
                    and path_part_list[3] == "requirements"
                ):
                    content_length_int = int(self.headers.get("Content-Length", "0"))
                    if content_length_int > 1024 * 1024:
                        raise NorgateApiError(413, "Request body is too large.")
                    request_bytes = self.rfile.read(content_length_int)
                    payload_obj = json.loads(request_bytes.decode("utf-8") or "{}")
                    if not isinstance(payload_obj, dict):
                        raise NorgateApiError(400, "Request body must be a JSON object.")
                    response_dict = service_obj.handle_requirements_dict(path_part_list[2], payload_obj)
                    self._send_json(200, response_dict)
                    return

                raise NorgateApiError(404, "Unknown endpoint.")
            except json.JSONDecodeError as exc:
                self._send_error(NorgateApiError(400, f"Invalid JSON request body: {exc}"))
            except NorgateApiError as exc:
                self._send_error(exc)

    return NorgateSnapshotApiHandler


def main() -> int:
    load_config_env_file()

    parser_obj = argparse.ArgumentParser(description="Serve private Norgate snapshot artifacts over HTTP.")
    parser_obj.add_argument(
        "--service-root",
        default=env_str(NORGATE_SERVICE_ROOT_ENV_STR),
        help="Norgate service root directory.",
    )
    parser_obj.add_argument(
        "--host",
        default=env_str(NORGATE_API_HOST_ENV_STR, "127.0.0.1"),
        help="Bind host. Use a Tailscale IP/host for remote clients.",
    )
    parser_obj.add_argument(
        "--port",
        type=int,
        default=env_int(NORGATE_API_PORT_ENV_STR, 8787),
        help="Bind port.",
    )
    parser_obj.add_argument(
        "--start-date",
        default=env_str("NORGATE_EXPORT_START_DATE", "1990-01-01"),
        help="First historical date to export.",
    )
    args_obj = parser_obj.parse_args()

    token_str = os.getenv(NORGATE_API_TOKEN_ENV_STR, "").strip()
    if not token_str:
        raise RuntimeError(f"{NORGATE_API_TOKEN_ENV_STR} must be set before starting the API.")
    if not args_obj.service_root:
        raise RuntimeError(f"--service-root or {NORGATE_SERVICE_ROOT_ENV_STR} must be set before starting the API.")

    service_obj = NorgateSnapshotApiService(
        service_root_path_obj=Path(args_obj.service_root),
        token_str=token_str,
        start_date_str=str(args_obj.start_date),
    )
    handler_cls = make_handler_class(service_obj)
    httpd_obj = ThreadingHTTPServer((str(args_obj.host), int(args_obj.port)), handler_cls)
    print(f"Norgate snapshot API listening on http://{args_obj.host}:{args_obj.port}")
    httpd_obj.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
