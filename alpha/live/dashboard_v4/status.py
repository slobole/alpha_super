"""Current LIVE operations without financial history, bindings or Flex reads."""

from datetime import datetime
import sqlite3

import yaml

from alpha.live.dashboard_v3.operator_tools import strategy_display_name_str
from alpha.live.release_manifest import load_release_list, validate_enabled_deployment_for_mode


def load_operations_workspace_dict(provider_obj, *, as_of_ts: datetime):
    """Build only the input needed by ``build_overview_dict(include_finance_bool=False)``."""
    if as_of_ts.tzinfo is None or as_of_ts.utcoffset() is None:
        raise ValueError("An operations assessment requires a timezone-aware clock.")
    result_dict = {"client_dict": {"accounts": [], "display_name": "Live account"},
        "operations_account_list": [], "summary_dict": {}, "operations_error_str": None}
    try:
        release_list = load_release_list(provider_obj.app_obj().releases_root_path_str)
        validate_enabled_deployment_for_mode(release_list, "live")
        live_list = [release_obj for release_obj in release_list if release_obj.mode_str == "live"]
        enabled_list = [release_obj for release_obj in live_list if release_obj.enabled_bool]
        owner_set = {release_obj.user_id_str for release_obj in (enabled_list or live_list)}
        if len(owner_set) > 1:
            raise ValueError("No unique local client identity.")
        if owner_set:
            result_dict["client_dict"]["display_name"] = next(iter(owner_set))
        if len({release_obj.account_route_str for release_obj in enabled_list}) != len(enabled_list):
            raise ValueError("An enabled account belongs to multiple Pods.")
        result_dict["operations_account_list"] = [{"pod_id": release_obj.pod_id_str,
            "account_route": release_obj.account_route_str,
            "display_name": strategy_display_name_str({"pod_id_str": release_obj.pod_id_str,
                "strategy_import_str": release_obj.strategy_import_str})} for release_obj in enabled_list]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
        result_dict["operations_error_str"] = "Local LIVE configuration could not be verified."
        return result_dict
    try:
        source_dict = provider_obj.get_summary_dict()
        raw_list = source_dict["pod_row_dict_list"]
        if not isinstance(raw_list, list) or any(not isinstance(row_dict, dict) for row_dict in raw_list):
            raise ValueError("Invalid saved operations.")
        scoped_list = []
        for release_obj in enabled_list:
            match_list = [row_dict for row_dict in raw_list if row_dict.get("pod_id_str") == release_obj.pod_id_str]
            if len(match_list) != 1:
                continue
            row_dict = match_list[0]
            # A short-lived provider cache may belong to the previous release.
            # Never turn a reused Pod/account into evidence for a new owner.
            if all(row_dict.get(field_str) == getattr(release_obj, field_str) for field_str in
                   ("mode_str", "user_id_str", "release_id_str", "account_route_str")):
                scoped_list.append(row_dict)
        # Keep the actual observation time. Request completion and a successful
        # HTTP response cannot renew the saved evidence's 120-second lifetime.
        result_dict["summary_dict"] = {"as_of_timestamp_str": source_dict.get("as_of_timestamp_str"),
            "pod_row_dict_list": scoped_list}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, sqlite3.Error, yaml.YAMLError):
        result_dict["operations_error_str"] = "Saved operations could not be read."
    return result_dict
