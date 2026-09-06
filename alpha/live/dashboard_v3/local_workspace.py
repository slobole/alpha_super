"""Read-only single-VPS identity and available reporting history.

Releases own current operations. Existing performance bindings own historical
reporting windows; those dates are NOT funding, mandate inception or first fills.
No registry, database, binding or financial field contract is created here.
"""

from contextlib import closing
from datetime import date
import os
from pathlib import Path
import sqlite3

from alpha.live.ibkr_performance import PodPerformanceBinding
from alpha.live.client_reporting import ClientReportingError
from alpha.live.ibkr_performance_sync import build_live_binding_obj_list
from alpha.live.release_manifest import load_release_list, validate_enabled_deployment_for_mode
from alpha.live.dashboard_v3.operator_tools import strategy_display_name_str


class LocalReportingError(ClientReportingError):
    """Sanitized local mapping explanations safe to display to the operator."""


def _saved_binding_list(database_path_str):
    database_path_obj = Path(database_path_str)
    if not database_path_obj.is_file():
        return []
    with closing(sqlite3.connect(database_path_obj.resolve().as_uri() + "?mode=ro", uri=True)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        row_list = connection_obj.execute(
            "SELECT account_route_str, pod_id_str, return_start_date_str, "
            "return_end_date_str, enabled_bool_int, session_calendar_id_str FROM pod_binding"
        ).fetchall()
    return [PodPerformanceBinding(
        account_route_str=row_obj["account_route_str"], pod_id_str=row_obj["pod_id_str"],
        return_start_date_str=row_obj["return_start_date_str"], return_end_date_str=row_obj["return_end_date_str"],
        enabled_bool=bool(row_obj["enabled_bool_int"]), session_calendar_id_str=row_obj["session_calendar_id_str"],
    ) for row_obj in row_list]


def _merged_binding_list(saved_list, current_list):
    binding_dict, pod_route_dict = {}, {}
    for binding_obj in [*saved_list, *current_list]:
        if binding_obj.session_calendar_id_str != "XNYS":
            raise LocalReportingError(f"{binding_obj.pod_id_str}: saved performance calendar is not XNYS.")
        previous_obj = binding_dict.get(binding_obj.account_route_str)
        if (previous_obj and previous_obj.pod_id_str != binding_obj.pod_id_str) or (
            binding_obj.pod_id_str in pod_route_dict and pod_route_dict[binding_obj.pod_id_str] != binding_obj.account_route_str
        ):
            raise LocalReportingError(f"Account/Pod mapping changed for {binding_obj.account_route_str} / {binding_obj.pod_id_str}. Check saved IBKR bindings against the local releases before reporting.")
        if previous_obj:
            for field_str in ("return_start_date_str", "return_end_date_str"):
                old_str, new_str = getattr(previous_obj, field_str), getattr(binding_obj, field_str)
                if old_str and new_str and old_str != new_str:
                    label_str = "start" if field_str == "return_start_date_str" else "end"
                    raise LocalReportingError(f"{binding_obj.pod_id_str} / {binding_obj.account_route_str}: saved history {label_str} is {old_str}, but the local ledger says {new_str}. Check the ledger or reviewed reporting override.")
            # Preserve trusted history when a local ledger is absent/truncated.
            binding_obj = PodPerformanceBinding(
                account_route_str=binding_obj.account_route_str, pod_id_str=binding_obj.pod_id_str,
                return_start_date_str=previous_obj.return_start_date_str or binding_obj.return_start_date_str,
                return_end_date_str=previous_obj.return_end_date_str or binding_obj.return_end_date_str,
                enabled_bool=binding_obj.enabled_bool,
            )
        binding_dict[binding_obj.account_route_str] = binding_obj
        pod_route_dict[binding_obj.pod_id_str] = binding_obj.account_route_str
    return sorted(binding_dict.values(), key=lambda binding_obj: binding_obj.account_route_str)


def validate_local_bindings_unchanged(workspace_dict, database_path_str):
    """A concurrent sync must not pair a new import with stale ownership."""
    try:
        if _saved_binding_list(database_path_str) != workspace_dict.get("saved_binding_list", []):
            raise ValueError("Performance bindings changed during this request.")
    except (OSError, ValueError, sqlite3.Error) as exception_obj:
        raise LocalReportingError("IBKR account mapping changed during this read. Refresh the page before exporting.") from exception_obj


def local_financial_scope_complete_bool(workspace_dict, from_str, to_str):
    # Coverage boundaries are NOT client capital entry/exit. Never value an
    # absent account as zero before its first or after its last trusted EOD.
    return workspace_dict["financial_scope_complete_bool"] and all(
        account_dict["effective_from"] <= from_str and (account_dict.get("effective_to") or "9999-12-31") >= to_str
        for account_dict in workspace_dict["client_dict"]["accounts"]
    )


def build_local_workspace_dict(provider_obj, database_path_str, *, today_str):
    """Use the same release/config paths as the running dashboard provider."""
    client_dict = {
        "client_id": "local", "display_name": "Live account", "base_currency": "USD",
        "fee_basis": "IBKR-reported values; external fees are not assessed.",
        "query_name": os.getenv("IBKR_FLEX_QUERY_NAME_STR", "").strip() or "ALPHA_DAILY_TWR",
        "mandate_start_date": today_str, "accounts": [], "operations_source": "local",
        "reporting_scope": "local_available_history",
    }
    result_dict = {
        "client_dict": client_dict, "operations_account_list": [], "summary_dict": {},
        "operations_error_str": None, "financial_scope_complete_bool": False,
        "financial_error_str": None, "valuation_account_list": [],
    }
    try:
        app_obj = provider_obj.app_obj()
        release_list = load_release_list(app_obj.releases_root_path_str)
        validate_enabled_deployment_for_mode(release_list, "live")
        live_release_list = [release_obj for release_obj in release_list if release_obj.mode_str == "live"]
        enabled_list = [release_obj for release_obj in live_release_list if release_obj.enabled_bool]
        user_id_set = {release_obj.user_id_str for release_obj in enabled_list}
        if not user_id_set:
            user_id_set = {release_obj.user_id_str for release_obj in live_release_list}
        if len(user_id_set) > 1:
            raise ValueError("No unique local client identity.")
        if user_id_set:
            client_dict["display_name"] = next(iter(user_id_set))
        foreign_pair_set = {(release_obj.pod_id_str, release_obj.account_route_str)
            for release_obj in live_release_list if release_obj.user_id_str not in user_id_set}
        live_release_list = [release_obj for release_obj in live_release_list if release_obj.user_id_str in user_id_set]
        name_dict = {(release_obj.pod_id_str, release_obj.account_route_str): strategy_display_name_str({
            "pod_id_str": release_obj.pod_id_str, "strategy_import_str": release_obj.strategy_import_str,
        }) for release_obj in live_release_list}
        result_dict["operations_account_list"] = [{
            "pod_id": release_obj.pod_id_str, "account_route": release_obj.account_route_str,
            "display_name": name_dict[(release_obj.pod_id_str, release_obj.account_route_str)],
        } for release_obj in enabled_list]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        result_dict["operations_error_str"] = "Local LIVE configuration could not be verified. Open Advanced diagnostics."
        result_dict["financial_error_str"] = result_dict["operations_error_str"]
        return result_dict
    try:
        result_dict["summary_dict"] = provider_obj.get_summary_dict()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        result_dict["operations_error_str"] = "Saved operations could not be read. Open Advanced diagnostics."
    try:
        current_list = build_live_binding_obj_list(
            releases_root_path_str=app_obj.releases_root_path_str, config_path_str=app_obj.config_path_str,
            release_obj_list=live_release_list,
        ) if live_release_list else []
        saved_list = _saved_binding_list(database_path_str)
        result_dict["saved_binding_list"] = saved_list
        # Excluding another owner's disabled manifest must not erase evidence
        # that this account or Pod used to have a different identity.
        _merged_binding_list(saved_list, current_list)
        binding_list = _merged_binding_list([binding_obj for binding_obj in saved_list
            if (binding_obj.pod_id_str, binding_obj.account_route_str) not in foreign_pair_set], current_list)
        complete_bool = bool(binding_list)
        for binding_obj in binding_list:
            key_tuple = (binding_obj.pod_id_str, binding_obj.account_route_str)
            identity_dict = {
                "pod_id": binding_obj.pod_id_str, "account_route": binding_obj.account_route_str,
                "display_name": name_dict.get(key_tuple) or strategy_display_name_str({"pod_id_str": binding_obj.pod_id_str}),
            }
            result_dict["valuation_account_list"].append(identity_dict)
            start_str, end_str = binding_obj.return_start_date_str, binding_obj.return_end_date_str
            if not start_str or (not binding_obj.enabled_bool and not end_str):
                complete_bool = False
                continue
            if binding_obj.enabled_bool and end_str:
                complete_bool = False
            if date.fromisoformat(start_str).isoformat() != start_str or (end_str and (date.fromisoformat(end_str).isoformat() != end_str or end_str < start_str)):
                raise ValueError("Invalid reporting window.")
            client_dict["accounts"].append({
                **identity_dict,
                "effective_from": start_str, "effective_to": end_str,
            })
        result_dict["financial_scope_complete_bool"] = complete_bool
        if client_dict["accounts"]:
            client_dict["mandate_start_date"] = min(account_dict["effective_from"] for account_dict in client_dict["accounts"])
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exception_obj:
        # An unreadable/conflicting financial mapping must not hide operations.
        client_dict["accounts"] = []
        result_dict["valuation_account_list"] = []
        result_dict["financial_error_str"] = str(exception_obj) if isinstance(exception_obj, LocalReportingError) else "Saved IBKR account mapping could not be read or validated. Check the performance database and local ledger files."
    return result_dict
