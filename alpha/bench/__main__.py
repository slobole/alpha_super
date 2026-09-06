"""CLI entry point: ``python -m alpha.bench``.

Binds to localhost by default — Bench is a single-operator research console,
not a service to expose. Like the live dashboard, it loads ``config.env`` at
startup so the subprocesses it launches (the analyzers) see the same Norgate
environment the operator uses from the terminal.
"""

from __future__ import annotations

import argparse

from alpha.bench.app import create_app
from alpha.bench.knowledge import build_knowledge_site


DEFAULT_HOST_STR = "127.0.0.1"
DEFAULT_PORT_INT = 8765


def main() -> int:
    arg_parser_obj = argparse.ArgumentParser(
        prog="python -m alpha.bench",
        description="Run the Bench research control panel.",
    )
    arg_parser_obj.add_argument("--host", default=DEFAULT_HOST_STR)
    arg_parser_obj.add_argument("--port", type=int, default=DEFAULT_PORT_INT)
    arg_parser_obj.add_argument(
        "--debug",
        action="store_true",
        help="Enable the Flask debug reloader (development only).",
    )
    arg_parser_obj.add_argument(
        "--skip-env-file",
        action="store_true",
        help="Do not auto-load config.env at startup.",
    )
    parsed_args_obj = arg_parser_obj.parse_args()

    if not parsed_args_obj.skip_env_file:
        # Best-effort: a missing config.env is fine on a workstation that already
        # has the Norgate Data Updater running locally.
        try:
            from scripts.norgate_config_env import load_config_env_file

            load_config_env_file(override_existing_bool=True)
        except Exception as exception_obj:  # noqa: BLE001 — never block startup on env loading
            print(f"[bench] config.env not loaded: {exception_obj}")

    knowledge_site_root_path_obj = None
    knowledge_build_error_str = None
    try:
        knowledge_site_root_path_obj = build_knowledge_site()
        print(f"[bench] knowledge base built at {knowledge_site_root_path_obj}")
    except Exception as exception_obj:  # noqa: BLE001 — BENCH remains available
        knowledge_build_error_str = str(exception_obj)
        print(f"[bench] knowledge base unavailable: {knowledge_build_error_str}")

    flask_app_obj = create_app(
        knowledge_site_root_path_obj=knowledge_site_root_path_obj,
        knowledge_build_error_str=knowledge_build_error_str,
    )
    print(f"[bench] http://{parsed_args_obj.host}:{parsed_args_obj.port}")
    print(
        f"[bench] knowledge http://{parsed_args_obj.host}:{parsed_args_obj.port}/knowledge/"
    )
    flask_app_obj.run(
        host=parsed_args_obj.host,
        port=parsed_args_obj.port,
        debug=parsed_args_obj.debug,
        use_reloader=parsed_args_obj.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
