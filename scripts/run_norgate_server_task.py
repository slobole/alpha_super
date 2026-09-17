"""Config-first entry point for the on-demand Windows Norgate API task."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.norgate_config_env import load_config_env_file


def main() -> int:
    repo_root_path_obj = Path(__file__).resolve().parents[1]
    config_path_obj = repo_root_path_obj / "config.env"
    if not config_path_obj.is_file():
        raise FileNotFoundError(f"Norgate task requires {config_path_obj}")

    os.chdir(repo_root_path_obj)
    loaded_env_dict = load_config_env_file(config_path_obj, override_existing_bool=True)
    for key_str in (
        "NORGATE_API_TOKEN",
        "NORGATE_SERVICE_ROOT",
        "NORGATE_API_HOST",
        "NORGATE_API_PORT",
    ):
        if not loaded_env_dict.get(key_str, "").strip():
            raise ValueError(f"{key_str} must be set in the task's config.env")

    # Import only after loading the checkout's configuration, not the task's
    # possibly stale inherited environment. The API retains its CLI semantics.
    from scripts.export_norgate_snapshot import SUPPORTED_EOD_PROFILE_TUPLE
    from scripts.serve_norgate_snapshot_api import main as serve_api_main

    print(f"Norgate task pid={os.getpid()} checkout={repo_root_path_obj}", flush=True)
    print("LOADED_EOD_PROFILES", SUPPORTED_EOD_PROFILE_TUPLE, flush=True)
    return serve_api_main()


if __name__ == "__main__":
    raise SystemExit(main())
