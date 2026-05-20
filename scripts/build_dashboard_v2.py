from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
DASHBOARD_V2_PATH = REPO_ROOT_PATH / "alpha" / "live" / "dashboard_v2"
TOOL_ROOT_PATH = REPO_ROOT_PATH / ".cache" / "dashboard_v2_tools"
NPM_VERSION_STR = "10.9.2"
NPM_TGZ_URL_STR = f"https://registry.npmjs.org/npm/-/npm-{NPM_VERSION_STR}.tgz"


def _node_path_str() -> str:
    node_path_str = shutil.which("node")
    if node_path_str is not None:
        return node_path_str

    bundled_node_path_obj = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "bin"
        / "node.exe"
    )
    if bundled_node_path_obj.exists():
        return str(bundled_node_path_obj)
    raise RuntimeError("Node.js was not found. Install Node or run inside the Codex workspace runtime.")


def _npm_cli_path_obj() -> Path:
    return TOOL_ROOT_PATH / f"npm-{NPM_VERSION_STR}" / "package" / "bin" / "npm-cli.js"


def _ensure_npm_cli() -> Path:
    npm_cli_path_obj = _npm_cli_path_obj()
    if npm_cli_path_obj.exists():
        return npm_cli_path_obj

    TOOL_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    archive_path_obj = TOOL_ROOT_PATH / f"npm-{NPM_VERSION_STR}.tgz"
    extract_root_path_obj = TOOL_ROOT_PATH / f"npm-{NPM_VERSION_STR}"
    if not archive_path_obj.exists():
        urllib.request.urlretrieve(NPM_TGZ_URL_STR, archive_path_obj)
    extract_root_path_obj.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path_obj, "r:gz") as tar_obj:
        tar_obj.extractall(extract_root_path_obj, filter="data")
    if not npm_cli_path_obj.exists():
        raise RuntimeError(f"npm bootstrap failed; missing {npm_cli_path_obj}")
    return npm_cli_path_obj


def _run_npm(command_list: list[str]) -> None:
    npm_cli_path_obj = _ensure_npm_cli()
    env_dict = os.environ.copy()
    env_dict.setdefault("npm_config_fund", "false")
    env_dict.setdefault("npm_config_audit", "false")
    subprocess.run(
        [_node_path_str(), str(npm_cli_path_obj), *command_list],
        cwd=DASHBOARD_V2_PATH,
        check=True,
        env=env_dict,
    )


def main() -> int:
    parser_obj = argparse.ArgumentParser(description="Install and build the Dashboard V2 frontend.")
    parser_obj.add_argument("--install-only", action="store_true")
    parser_obj.add_argument("--skip-install", action="store_true")
    parsed_args_obj = parser_obj.parse_args()

    if not parsed_args_obj.skip_install:
        install_command_list = ["ci"] if (DASHBOARD_V2_PATH / "package-lock.json").exists() else ["install"]
        _run_npm(install_command_list)
    if not parsed_args_obj.install_only:
        _run_npm(["run", "typecheck"])
        _run_npm(["run", "build"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
