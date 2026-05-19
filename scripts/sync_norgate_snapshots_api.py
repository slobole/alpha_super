from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from alpha.live.release_manifest import load_release_list, select_enabled_release_list_for_mode
from data.norgate_snapshot_store import (
    MANIFEST_FILE_NAME_STR,
    NORGATE_SNAPSHOT_ROOT_ENV_STR,
    PRICE_FILE_NAME_STR,
    UNIVERSE_FILE_NAME_STR,
    load_valid_snapshot_manifest,
)
from scripts.serve_norgate_snapshot_api import (
    NORGATE_API_TOKEN_ENV_STR,
    NORGATE_API_TOKEN_HEADER_STR,
)
from scripts.norgate_config_env import (
    NORGATE_CLIENT_ID_ENV_STR,
    NORGATE_LOCAL_SNAPSHOT_ROOT_ENV_STR,
    NORGATE_RELEASES_ROOT_ENV_STR,
    NORGATE_SYNC_MODE_ENV_STR,
    env_str,
    load_config_env_file,
    norgate_api_url_from_env_str,
)


ALLOWED_DOWNLOAD_FILE_SET = {
    MANIFEST_FILE_NAME_STR,
    PRICE_FILE_NAME_STR,
    UNIVERSE_FILE_NAME_STR,
}


def _validate_path_component_str(raw_value_str: str, field_name_str: str) -> str:
    value_str = str(raw_value_str).strip()
    if not value_str:
        raise ValueError(f"{field_name_str} must not be empty.")
    allowed_char_set = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-$")
    if any(char_str not in allowed_char_set for char_str in value_str):
        raise ValueError(f"{field_name_str} contains unsupported characters: {value_str!r}.")
    if value_str in {".", ".."}:
        raise ValueError(f"{field_name_str} is invalid: {value_str!r}.")
    return value_str


@contextmanager
def _temporary_snapshot_root_env(snapshot_root_path_obj: Path) -> Iterator[None]:
    old_snapshot_root_str = os.environ.get(NORGATE_SNAPSHOT_ROOT_ENV_STR)
    os.environ[NORGATE_SNAPSHOT_ROOT_ENV_STR] = str(snapshot_root_path_obj)
    try:
        yield
    finally:
        if old_snapshot_root_str is None:
            os.environ.pop(NORGATE_SNAPSHOT_ROOT_ENV_STR, None)
        else:
            os.environ[NORGATE_SNAPSHOT_ROOT_ENV_STR] = old_snapshot_root_str


def derive_required_profile_list(releases_root_path_str: str, mode_str: str) -> list[str]:
    release_list = load_release_list(releases_root_path_str)
    selected_release_list = select_enabled_release_list_for_mode(
        release_list=release_list,
        env_mode_str=str(mode_str),
    )
    profile_list = sorted({release_obj.data_profile_str for release_obj in selected_release_list})
    if len(profile_list) == 0:
        raise RuntimeError(
            f"No enabled {mode_str} releases were found under {releases_root_path_str}."
        )
    return profile_list


def _build_url_str(api_url_str: str, url_path_str: str) -> str:
    return urljoin(api_url_str.rstrip("/") + "/", url_path_str.lstrip("/"))


def _read_error_body_str(error_obj: HTTPError) -> str:
    try:
        return error_obj.read().decode("utf-8")
    except Exception:
        return str(error_obj)


def _request_json_dict(
    *,
    api_url_str: str,
    token_str: str,
    url_path_str: str,
    method_str: str,
    payload_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_data_bytes = None
    if payload_dict is not None:
        request_data_bytes = json.dumps(payload_dict).encode("utf-8")
    request_obj = Request(
        _build_url_str(api_url_str, url_path_str),
        data=request_data_bytes,
        method=method_str,
        headers={
            NORGATE_API_TOKEN_HEADER_STR: token_str,
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request_obj, timeout=120) as response_obj:
            response_bytes = response_obj.read()
    except HTTPError as exc:
        raise RuntimeError(f"Norgate API returned HTTP {exc.code}: {_read_error_body_str(exc)}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Norgate API: {exc}") from exc

    payload_obj = json.loads(response_bytes.decode("utf-8"))
    if not isinstance(payload_obj, dict):
        raise RuntimeError("Norgate API response was not a JSON object.")
    return payload_obj


def post_requirements_dict(
    *,
    api_url_str: str,
    token_str: str,
    client_id_str: str,
    profile_list: list[str],
) -> dict[str, Any]:
    safe_client_id_str = _validate_path_component_str(client_id_str, "client_id_str")
    return _request_json_dict(
        api_url_str=api_url_str,
        token_str=token_str,
        url_path_str=f"/v1/clients/{safe_client_id_str}/requirements",
        method_str="POST",
        payload_dict={"profile_list": profile_list},
    )


def _download_file(
    *,
    api_url_str: str,
    token_str: str,
    url_path_str: str,
    output_path_obj: Path,
) -> None:
    request_obj = Request(
        _build_url_str(api_url_str, url_path_str),
        method="GET",
        headers={NORGATE_API_TOKEN_HEADER_STR: token_str},
    )
    try:
        with urlopen(request_obj, timeout=120) as response_obj:
            response_bytes = response_obj.read()
    except HTTPError as exc:
        raise RuntimeError(f"Norgate API returned HTTP {exc.code}: {_read_error_body_str(exc)}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Norgate API: {exc}") from exc

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_bytes(response_bytes)


def _validate_snapshot_at_root(
    *,
    snapshot_root_path_obj: Path,
    profile_str: str,
    snapshot_date_str: str,
) -> None:
    with _temporary_snapshot_root_env(snapshot_root_path_obj):
        load_valid_snapshot_manifest(profile_str, snapshot_date_str=snapshot_date_str)


def _existing_target_is_valid_bool(
    *,
    local_root_path_obj: Path,
    profile_str: str,
    snapshot_date_str: str,
) -> bool:
    try:
        _validate_snapshot_at_root(
            snapshot_root_path_obj=local_root_path_obj,
            profile_str=profile_str,
            snapshot_date_str=snapshot_date_str,
        )
    except Exception:
        return False
    return True


def _download_snapshot_entry(
    *,
    api_url_str: str,
    token_str: str,
    local_root_path_obj: Path,
    temp_root_path_obj: Path,
    snapshot_file_dict: dict[str, Any],
    overwrite_bool: bool,
) -> Path:
    profile_str = _validate_path_component_str(str(snapshot_file_dict["profile_str"]), "profile_str")
    snapshot_date_str = _validate_path_component_str(
        str(snapshot_file_dict["snapshot_date_str"]),
        "snapshot_date_str",
    )
    local_target_dir_path_obj = local_root_path_obj / profile_str / snapshot_date_str

    if local_target_dir_path_obj.exists() and not overwrite_bool:
        if _existing_target_is_valid_bool(
            local_root_path_obj=local_root_path_obj,
            profile_str=profile_str,
            snapshot_date_str=snapshot_date_str,
        ):
            return local_target_dir_path_obj
        raise RuntimeError(f"Existing local snapshot is invalid: {local_target_dir_path_obj}")

    file_url_path_dict = snapshot_file_dict.get("file_url_path_dict")
    if not isinstance(file_url_path_dict, dict):
        raise RuntimeError("snapshot_file_dict is missing file_url_path_dict.")

    temp_snapshot_dir_path_obj = temp_root_path_obj / profile_str / snapshot_date_str
    if temp_snapshot_dir_path_obj.exists():
        shutil.rmtree(temp_snapshot_dir_path_obj)
    temp_snapshot_dir_path_obj.mkdir(parents=True, exist_ok=True)

    for file_name_str, url_path_obj in sorted(file_url_path_dict.items()):
        safe_file_name_str = _validate_path_component_str(str(file_name_str), "file_name_str")
        if safe_file_name_str not in ALLOWED_DOWNLOAD_FILE_SET:
            raise RuntimeError(f"Refusing to download unsupported snapshot file: {safe_file_name_str}")
        _download_file(
            api_url_str=api_url_str,
            token_str=token_str,
            url_path_str=str(url_path_obj),
            output_path_obj=temp_snapshot_dir_path_obj / safe_file_name_str,
        )

    _validate_snapshot_at_root(
        snapshot_root_path_obj=temp_root_path_obj,
        profile_str=profile_str,
        snapshot_date_str=snapshot_date_str,
    )

    local_target_dir_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if local_target_dir_path_obj.exists():
        shutil.rmtree(local_target_dir_path_obj)
    temp_snapshot_dir_path_obj.rename(local_target_dir_path_obj)
    return local_target_dir_path_obj


def sync_required_snapshots(
    *,
    api_url_str: str,
    token_str: str,
    client_id_str: str,
    releases_root_path_str: str,
    mode_str: str,
    local_root_path_str: str,
    overwrite_bool: bool = False,
) -> list[Path]:
    profile_list = derive_required_profile_list(
        releases_root_path_str=releases_root_path_str,
        mode_str=mode_str,
    )
    response_dict = post_requirements_dict(
        api_url_str=api_url_str,
        token_str=token_str,
        client_id_str=client_id_str,
        profile_list=profile_list,
    )
    if response_dict.get("status_str") != "ready":
        raise RuntimeError(f"Norgate API did not return ready status: {response_dict}")

    snapshot_file_list_obj = response_dict.get("snapshot_file_list")
    if not isinstance(snapshot_file_list_obj, list) or len(snapshot_file_list_obj) == 0:
        raise RuntimeError("Norgate API returned no snapshot files.")

    local_root_path_obj = Path(local_root_path_str).expanduser()
    local_root_path_obj.mkdir(parents=True, exist_ok=True)
    temp_root_path_obj = local_root_path_obj / f".api-sync-{os.getpid()}"
    if temp_root_path_obj.exists():
        shutil.rmtree(temp_root_path_obj)
    temp_root_path_obj.mkdir(parents=True, exist_ok=True)

    promoted_path_list: list[Path] = []
    try:
        for snapshot_file_obj in snapshot_file_list_obj:
            if not isinstance(snapshot_file_obj, dict):
                raise RuntimeError("snapshot_file_list entries must be JSON objects.")
            promoted_path_list.append(
                _download_snapshot_entry(
                    api_url_str=api_url_str,
                    token_str=token_str,
                    local_root_path_obj=local_root_path_obj,
                    temp_root_path_obj=temp_root_path_obj,
                    snapshot_file_dict=snapshot_file_obj,
                    overwrite_bool=overwrite_bool,
                )
            )
    finally:
        if temp_root_path_obj.exists():
            shutil.rmtree(temp_root_path_obj)
    return promoted_path_list


def main() -> int:
    load_config_env_file()

    parser_obj = argparse.ArgumentParser(description="Sync Norgate artifact snapshots from the private API.")
    parser_obj.add_argument(
        "--api-url",
        default=norgate_api_url_from_env_str(),
        help="Base API URL, for example http://norgate-node:8787.",
    )
    parser_obj.add_argument(
        "--client-id",
        default=env_str(NORGATE_CLIENT_ID_ENV_STR),
        help="Client identifier used by the Norgate API.",
    )
    parser_obj.add_argument(
        "--releases-root",
        default=env_str(NORGATE_RELEASES_ROOT_ENV_STR),
        help="Local release manifest root.",
    )
    parser_obj.add_argument(
        "--mode",
        default=env_str(NORGATE_SYNC_MODE_ENV_STR),
        help="Release mode to sync, for example live or paper.",
    )
    parser_obj.add_argument(
        "--local-root",
        default=env_str(
            NORGATE_LOCAL_SNAPSHOT_ROOT_ENV_STR,
            env_str(NORGATE_SNAPSHOT_ROOT_ENV_STR),
        ),
        help="Local NORGATE_SNAPSHOT_ROOT path.",
    )
    parser_obj.add_argument("--overwrite", action="store_true", help="Replace existing local snapshot folders.")
    args_obj = parser_obj.parse_args()

    token_str = os.getenv(NORGATE_API_TOKEN_ENV_STR, "").strip()
    if not token_str:
        raise RuntimeError(f"{NORGATE_API_TOKEN_ENV_STR} must be set before syncing snapshots.")
    required_arg_name_tuple = ("api_url", "client_id", "releases_root", "mode", "local_root")
    for arg_name_str in required_arg_name_tuple:
        if not getattr(args_obj, arg_name_str):
            env_name_str = {
                "api_url": "NORGATE_API_URL or NORGATE_API_HOST/NORGATE_API_PORT",
                "client_id": NORGATE_CLIENT_ID_ENV_STR,
                "releases_root": NORGATE_RELEASES_ROOT_ENV_STR,
                "mode": NORGATE_SYNC_MODE_ENV_STR,
                "local_root": f"{NORGATE_LOCAL_SNAPSHOT_ROOT_ENV_STR} or {NORGATE_SNAPSHOT_ROOT_ENV_STR}",
            }[arg_name_str]
            raise RuntimeError(
                f"--{arg_name_str.replace('_', '-')} or {env_name_str} must be set before syncing snapshots."
            )

    promoted_path_list = sync_required_snapshots(
        api_url_str=str(args_obj.api_url),
        token_str=token_str,
        client_id_str=str(args_obj.client_id),
        releases_root_path_str=str(args_obj.releases_root),
        mode_str=str(args_obj.mode),
        local_root_path_str=str(args_obj.local_root),
        overwrite_bool=bool(args_obj.overwrite),
    )
    for promoted_path_obj in promoted_path_list:
        print(promoted_path_obj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
