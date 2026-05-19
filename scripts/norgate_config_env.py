from __future__ import annotations

import os
import sys
from pathlib import Path


repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)


CONFIG_ENV_FILE_NAME_STR = "config.env"
NORGATE_SERVICE_ROOT_ENV_STR = "NORGATE_SERVICE_ROOT"
NORGATE_API_HOST_ENV_STR = "NORGATE_API_HOST"
NORGATE_API_PORT_ENV_STR = "NORGATE_API_PORT"
NORGATE_API_URL_ENV_STR = "NORGATE_API_URL"
NORGATE_CLIENT_ID_ENV_STR = "NORGATE_CLIENT_ID"
NORGATE_RELEASES_ROOT_ENV_STR = "NORGATE_RELEASES_ROOT"
NORGATE_SYNC_MODE_ENV_STR = "NORGATE_SYNC_MODE"
NORGATE_LOCAL_SNAPSHOT_ROOT_ENV_STR = "NORGATE_LOCAL_SNAPSHOT_ROOT"
NORGATE_DOCTOR_REPORT_JSON_ENV_STR = "NORGATE_DOCTOR_REPORT_JSON"


def default_config_env_path_obj() -> Path:
    return repo_root_path / CONFIG_ENV_FILE_NAME_STR


def _normalize_env_value_str(raw_value_str: str) -> str:
    value_str = raw_value_str.strip()
    if len(value_str) >= 2 and value_str[0] == value_str[-1] and value_str[0] in {"'", '"'}:
        return value_str[1:-1]
    return value_str


def load_config_env_file(config_env_path_obj: Path | None = None) -> dict[str, str]:
    path_obj = (config_env_path_obj or default_config_env_path_obj()).expanduser()
    loaded_env_dict: dict[str, str] = {}
    if not path_obj.exists():
        return loaded_env_dict

    for line_number_int, raw_line_str in enumerate(path_obj.read_text(encoding="utf-8").splitlines(), start=1):
        line_str = raw_line_str.strip()
        if not line_str or line_str.startswith("#"):
            continue
        if line_str.startswith("export "):
            line_str = line_str[len("export ") :].strip()
        if "=" not in line_str:
            raise ValueError(f"Invalid config.env line {line_number_int}: expected KEY=value.")

        key_str, raw_value_str = line_str.split("=", 1)
        key_str = key_str.strip()
        if not key_str:
            raise ValueError(f"Invalid config.env line {line_number_int}: empty key.")
        value_str = _normalize_env_value_str(raw_value_str)
        loaded_env_dict[key_str] = value_str
        os.environ.setdefault(key_str, value_str)

    return loaded_env_dict


def env_str(name_str: str, default_str: str | None = None) -> str | None:
    value_str = os.getenv(name_str, "").strip()
    if value_str:
        return value_str
    return default_str


def env_float(name_str: str, default_float: float) -> float:
    value_str = os.getenv(name_str, "").strip()
    if not value_str:
        return default_float
    return float(value_str)


def env_int(name_str: str, default_int: int) -> int:
    value_str = os.getenv(name_str, "").strip()
    if not value_str:
        return default_int
    return int(value_str)


def norgate_api_url_from_env_str() -> str | None:
    api_url_str = env_str(NORGATE_API_URL_ENV_STR)
    if api_url_str:
        return api_url_str

    host_str = env_str(NORGATE_API_HOST_ENV_STR)
    if not host_str:
        return None
    port_int = env_int(NORGATE_API_PORT_ENV_STR, 8787)
    return f"http://{host_str}:{port_int}"
