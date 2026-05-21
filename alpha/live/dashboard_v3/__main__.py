"""CLI entry point: ``python -m alpha.live.dashboard_v3``.

Binds to localhost by default; Tailscale-side exposure is handled by
``tailscale serve <port>`` on the VPS, not by binding to 0.0.0.0.

The launcher loads ``config.env`` at startup so the same environment
variables the live runner uses — most importantly
``ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true`` on VPS hosts that have no
local Norgate Data Updater — are visible to the data builders the
dashboard calls into. Without this load, the dashboard would burn ~20 s
per refresh retrying NDU even when the operator opted into snapshots.
"""

from __future__ import annotations

import argparse

from alpha.live.dashboard_v3.app import create_app
from scripts.norgate_config_env import load_config_env_file


DEFAULT_HOST_STR = "127.0.0.1"
DEFAULT_PORT_INT = 8080


def main() -> int:
    arg_parser_obj = argparse.ArgumentParser(
        prog="python -m alpha.live.dashboard_v3",
        description="Run the Dashboard V3 operator console.",
    )
    arg_parser_obj.add_argument("--host", default=DEFAULT_HOST_STR)
    arg_parser_obj.add_argument("--port", type=int, default=DEFAULT_PORT_INT)
    arg_parser_obj.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug reloader (development only).",
    )
    arg_parser_obj.add_argument(
        "--skip-env-file",
        action="store_true",
        help="Do not auto-load config.env at startup (use only when the host already exports the required vars).",
    )
    parsed_args_obj = arg_parser_obj.parse_args()

    if not parsed_args_obj.skip_env_file:
        load_config_env_file(override_existing_bool=True)

    flask_app_obj = create_app()
    flask_app_obj.run(
        host=parsed_args_obj.host,
        port=parsed_args_obj.port,
        debug=parsed_args_obj.debug,
        use_reloader=parsed_args_obj.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
