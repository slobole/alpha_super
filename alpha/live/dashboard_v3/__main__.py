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
    arg_parser_obj.add_argument("--demo", action="store_true", help="Local-only synthetic client reporting preview; no config.env or real provider.")
    arg_parser_obj.add_argument("--client-registry", help="Local operator-maintained client reporting JSON path.")
    action_mode_obj = arg_parser_obj.add_mutually_exclusive_group()
    action_mode_obj.add_argument(
        "--read-only",
        action="store_true",
        default=True,
        help="Disable actions, generated trade sheets and notification writes/webhooks.",
    )
    action_mode_obj.add_argument(
        "--enable-actions", action="store_false", dest="read_only",
        help="Explicitly enable advanced operational actions; confirmations are still required.",
    )
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

    if parsed_args_obj.demo and parsed_args_obj.host not in {"127.0.0.1", "localhost", "::1"}:
        arg_parser_obj.error("Demonstration mode must bind to loopback only.")
    if parsed_args_obj.demo and not parsed_args_obj.read_only:
        arg_parser_obj.error("Demonstration mode cannot enable operational actions.")
    if not parsed_args_obj.skip_env_file and not parsed_args_obj.demo:
        load_config_env_file(override_existing_bool=True)

    if parsed_args_obj.demo:
        from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple

        registry_dict, snapshot_dict = build_demo_fixture_tuple()
        flask_app_obj = create_app(
            DemoOperationsProvider(), read_only_bool=True, demo_mode_bool=True,
            client_registry_dict=registry_dict,
            client_reporting_snapshot_fn=lambda client_id_str: snapshot_dict[client_id_str],
        )
    else:
        flask_app_obj = create_app(
            read_only_bool=parsed_args_obj.read_only,
            client_reporting_config_path_str=parsed_args_obj.client_registry,
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
