"""Run V4 beside V3. Default bind is local. Only the Pod order panel can send orders (on by default; --disable-orders)."""

import os

# The import chain loads numpy/OpenBLAS, which reserves one thread and its
# buffers per core (about 1.1 GB commit). The dashboard does no heavy math.
for _thread_env_str in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env_str, "1")

import argparse
import logging
import re

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider


# Successful polls would print a line every few seconds per open tab.
QUIET_PATH_RE = re.compile(r"^(GET|HEAD) /(?:[^ ?]*/)?(?:refresh|status|tail|money|body)(?:[ ?]|$)")


class QuietPollFilter(logging.Filter):
    def filter(self, record_obj):
        arg_tuple = record_obj.args if isinstance(record_obj.args, tuple) else ()
        if len(arg_tuple) >= 2 and str(arg_tuple[1]) in {"200", "304"}:
            return QUIET_PATH_RE.match(str(arg_tuple[0])) is None
        return True


def main() -> int:
    parser_obj = argparse.ArgumentParser(description="Dashboard V4 · LIVE Overview · Read-only")
    parser_obj.add_argument("--host", default="127.0.0.1")
    parser_obj.add_argument("--port", type=int, default=8084)
    parser_obj.add_argument("--demo", action="store_true", help="Synthetic local preview; no live data or config.env.")
    parser_obj.add_argument("--demo-tools", action="store_true", help="Simulate Tools actions in memory; requires --demo.")
    parser_obj.add_argument("--demo-daily", action="store_true",
        help="Add five synthetic daily pods (CORE5, MR capsule) to the demo; requires --demo.")
    parser_obj.add_argument("--demo-backtest", action="store_true",
        help="Add a synthetic nightly backtest check report to the demo; requires --demo.")
    parser_obj.add_argument("--read-only", action="store_true", default=True, help="Everything except the order panel stays read-only.")
    parser_obj.add_argument("--disable-orders", action="store_true",
        help="Turn off the Pod order panel (Complete a missing trade, New order). It is on by default.")
    parser_obj.add_argument("--skip-env-file", action="store_true")
    parser_obj.add_argument("--releases-root")
    parser_obj.add_argument("--config")
    parser_obj.add_argument("--performance-db")
    args_obj = parser_obj.parse_args()
    if args_obj.demo_tools and not args_obj.demo:
        parser_obj.error("--demo-tools requires --demo; production execution is not connected.")
    if args_obj.demo_daily and not args_obj.demo:
        parser_obj.error("--demo-daily requires --demo.")
    if args_obj.demo_backtest and not args_obj.demo:
        parser_obj.error("--demo-backtest requires --demo.")
    if args_obj.demo:
        from alpha.live.dashboard_v4.demo import create_demo_app
        flask_app_obj = create_demo_app(demo_tools_bool=args_obj.demo_tools, console_appender_bool=True,
            demo_daily_bool=args_obj.demo_daily, demo_backtest_bool=args_obj.demo_backtest)
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
            LiveDataProvider(**provider_kwargs_dict), performance_db_path_str=args_obj.performance_db,
            orders_enabled_bool=not args_obj.disable_orders)
    logging.getLogger("werkzeug").addFilter(QuietPollFilter())
    flask_app_obj.run(host=args_obj.host, port=args_obj.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
