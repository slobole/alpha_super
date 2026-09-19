"""Run V4 beside V3. Default bind is local; this phase is always read-only."""

import argparse

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider


def main() -> int:
    parser_obj = argparse.ArgumentParser(description="Dashboard V4 · LIVE Overview · Read-only")
    parser_obj.add_argument("--host", default="127.0.0.1")
    parser_obj.add_argument("--port", type=int, default=8084)
    parser_obj.add_argument("--demo", action="store_true", help="Synthetic local preview; no live data or config.env.")
    parser_obj.add_argument("--read-only", action="store_true", default=True, help="Always enforced in this release.")
    parser_obj.add_argument("--skip-env-file", action="store_true")
    parser_obj.add_argument("--releases-root")
    parser_obj.add_argument("--config")
    parser_obj.add_argument("--performance-db")
    args_obj = parser_obj.parse_args()
    if args_obj.demo:
        from alpha.live.dashboard_v4.demo import create_demo_app
        flask_app_obj = create_demo_app()
    else:
        if not args_obj.skip_env_file:
            from scripts.norgate_config_env import load_config_env_file
            load_config_env_file()
        provider_kwargs_dict = {}
        if args_obj.releases_root:
            provider_kwargs_dict["releases_root_path_str"] = args_obj.releases_root
        if args_obj.config:
            provider_kwargs_dict["config_path_str"] = args_obj.config
        flask_app_obj = create_app(
            LiveDataProvider(**provider_kwargs_dict), performance_db_path_str=args_obj.performance_db)
    flask_app_obj.run(host=args_obj.host, port=args_obj.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
