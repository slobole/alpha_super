from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.live.ops_report import build_heartbeat_payload_dict, post_heartbeat_bool


HEARTBEAT_URL_ENV_VAR_NAME_STR = "ALPHA_INSPECTOR_HEARTBEAT_URL"


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(description="Send Live OPS Inspector heartbeat.")
    parser_obj.add_argument("--url", dest="heartbeat_url_str", default=None)
    parser_obj.add_argument("--vps-id", dest="vps_id_str", default=None)
    parser_obj.add_argument("--timeout-seconds", dest="timeout_seconds_float", type=float, default=3.0)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parsed_args_obj = parser_obj.parse_args(argv_list)

    heartbeat_url_str = parsed_args_obj.heartbeat_url_str or os.getenv(
        HEARTBEAT_URL_ENV_VAR_NAME_STR,
        "",
    )
    payload_dict = build_heartbeat_payload_dict(vps_id_str=parsed_args_obj.vps_id_str)
    if not heartbeat_url_str:
        result_dict = {
            "status_str": "disabled",
            "reason_code_str": "heartbeat_url_missing",
            "env_var_name_str": HEARTBEAT_URL_ENV_VAR_NAME_STR,
            "payload_dict": payload_dict,
        }
        _print_result(result_dict, json_output_bool=parsed_args_obj.json_output_bool)
        return 0

    delivered_bool = post_heartbeat_bool(
        heartbeat_url_str,
        payload_dict,
        timeout_seconds_float=parsed_args_obj.timeout_seconds_float,
    )
    result_dict = {
        "status_str": "sent" if delivered_bool else "failed",
        "reason_code_str": "heartbeat_sent" if delivered_bool else "heartbeat_post_failed",
        "payload_dict": payload_dict,
    }
    _print_result(result_dict, json_output_bool=parsed_args_obj.json_output_bool)
    return 0 if delivered_bool else 1


def _print_result(result_dict: dict[str, object], *, json_output_bool: bool) -> None:
    if json_output_bool:
        print(json.dumps(result_dict, indent=2, sort_keys=True))
        return
    print(f"Heartbeat {result_dict['status_str']}: {result_dict['reason_code_str']}")


if __name__ == "__main__":
    raise SystemExit(main())
