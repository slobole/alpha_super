from __future__ import annotations

import os

import pytest

from scripts.norgate_config_env import (
    NORGATE_API_HOST_ENV_STR,
    NORGATE_API_PORT_ENV_STR,
    load_config_env_file,
    norgate_api_url_from_env_str,
)


def test_load_config_env_file_sets_missing_env_values(tmp_path, monkeypatch):
    config_path_obj = tmp_path / "config.env"
    config_path_obj.write_text(
        "\n".join(
            [
                "# comment",
                "NORGATE_API_TOKEN=file-token",
                "NORGATE_API_HOST=100.123.13.69",
                "NORGATE_API_PORT=8787",
                'QUOTED_VALUE="hello world"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NORGATE_API_TOKEN", raising=False)
    monkeypatch.delenv(NORGATE_API_HOST_ENV_STR, raising=False)
    monkeypatch.delenv(NORGATE_API_PORT_ENV_STR, raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)

    loaded_env_dict = load_config_env_file(config_path_obj)

    assert loaded_env_dict["NORGATE_API_TOKEN"] == "file-token"
    assert os.environ["NORGATE_API_TOKEN"] == "file-token"
    assert os.environ["QUOTED_VALUE"] == "hello world"
    assert norgate_api_url_from_env_str() == "http://100.123.13.69:8787"


def test_load_config_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    config_path_obj = tmp_path / "config.env"
    config_path_obj.write_text("NORGATE_API_TOKEN=file-token\n", encoding="utf-8")
    monkeypatch.setenv("NORGATE_API_TOKEN", "real-env-token")

    loaded_env_dict = load_config_env_file(config_path_obj)

    assert loaded_env_dict["NORGATE_API_TOKEN"] == "file-token"
    assert os.environ["NORGATE_API_TOKEN"] == "real-env-token"


def test_load_config_env_file_can_override_existing_env(tmp_path, monkeypatch):
    config_path_obj = tmp_path / "config.env"
    config_path_obj.write_text("NORGATE_API_TOKEN=file-token\n", encoding="utf-8")
    monkeypatch.setenv("NORGATE_API_TOKEN", "stale-env-token")

    loaded_env_dict = load_config_env_file(config_path_obj, override_existing_bool=True)

    assert loaded_env_dict["NORGATE_API_TOKEN"] == "file-token"
    assert os.environ["NORGATE_API_TOKEN"] == "file-token"


def test_load_config_env_file_rejects_malformed_line(tmp_path):
    config_path_obj = tmp_path / "config.env"
    config_path_obj.write_text("BAD_LINE_WITHOUT_EQUALS\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid config.env line"):
        load_config_env_file(config_path_obj)
