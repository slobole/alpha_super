"""Reuse V3 readers while excluding non-LIVE targets before state acquisition."""

from dataclasses import replace
from datetime import datetime

from alpha.live.client_reporting import (
    BrokerReportingSnapshot,
    ClientReportingError,
    load_broker_reporting_snapshot,
)
from alpha.live.dashboard import DashboardApp
from alpha.live.dashboard_v3.data import DashboardDataProvider
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v3.local_workspace import (
    build_local_workspace_dict,
    validate_local_bindings_unchanged,
)
from alpha.live.dashboard_v4.evidence import load_cycle_evidence_dict
from alpha.live.dashboard_v4.pod_data import load_pod_cycles_dict
from alpha.live.dashboard_v4.scheduler_status import load_scheduler_status_dict
from alpha.live.dashboard_v4.tools import _arguments_list, powershell_command_str


class LiveReadOnlyApp(DashboardApp):
    def __post_init__(self) -> None:
        # Only get_target_list/load_config are needed. Do not create executors.
        pass

    def get_target_list(self):
        # V3 owns release/config validation and DB path resolution. Filtering
        # targets here prevents summary/cash readers opening PAPER/SIM state.
        return [target_obj for target_obj in super().get_target_list()
                if target_obj.release_obj.mode_str == "live"]


class LiveDataProvider(DashboardDataProvider):
    def get_scheduler_status_dict(self, pod_id_str, *, as_of_ts):
        try:
            target_obj = self.get_target_for_pod(pod_id_str)
        except (ValueError, OSError):
            target_obj = None
        if (target_obj is None or target_obj.release_obj.mode_str != "live"
                or not target_obj.release_obj.enabled_bool):
            return {"state_str": "unknown", "alive_bool": None}
        status_dict = load_scheduler_status_dict(self.event_log_path_str, pod_id_str, as_of_ts=as_of_ts)
        if status_dict["state_str"] in {"late", "stopped", "error"}:
            # Copy-only diagnostic, never invoked by the dashboard. next_due
            # synchronizes local release metadata when the operator runs it.
            argument_list = _arguments_list("next_due", "scheduler", "pod", target_obj, {
                "releases_root_str": str(self.releases_root_path_str or ""),
                "db_path_str": str(target_obj.db_path_str or ""),
                "log_path_str": str(self.event_log_path_str or "")})
            status_dict["check_command_str"] = powershell_command_str(argument_list)
        return status_dict

    def get_pod_cycles_dict(self, pod_id_str, *, as_of_ts, decision_plan_id_int=None, vplan_id_int=None):
        try:
            target_obj = self.get_target_for_pod(pod_id_str)
        except (ValueError, OSError):
            target_obj = None
        result_dict = load_pod_cycles_dict(target_obj, as_of_ts=as_of_ts,
            decision_plan_id_int=decision_plan_id_int, vplan_id_int=vplan_id_int)
        if result_dict.get("status_str") == "ok":
            # The history reader validates saved LIVE releases against the
            # current owner/Pod/account. Keep the same DB; only select the
            # validated release whose DecisionPlan and VPlan we are displaying.
            selected_release_dict = result_dict["selected_release_dict"]
            evidence_target_obj = replace(target_obj,
                release_obj=replace(target_obj.release_obj, **selected_release_dict))
            result_dict["cycle_evidence_dict"] = load_cycle_evidence_dict(
                evidence_target_obj, result_dict["pod_row_dict"], as_of_ts=as_of_ts)
        return result_dict

    def get_cycle_evidence_dict(self, pod_row_dict, *, as_of_ts):
        try:
            target_obj = self.get_target_for_pod(pod_row_dict["pod_id_str"])
        except (ValueError, OSError):
            return {"state_str": "unknown", "reason_str": "Fill source unavailable"}
        return load_cycle_evidence_dict(target_obj, pod_row_dict, as_of_ts=as_of_ts)

    def app_obj(self) -> LiveReadOnlyApp:
        if self._app_obj is None:
            self._app_obj = LiveReadOnlyApp(
                releases_root_path_str=self.releases_root_path_str,
                config_path_str=self.config_path_str,
                results_root_path_str=self.results_root_path_str,
                event_log_path_str=self.event_log_path_str,
            )
        return self._app_obj


def load_workspace_snapshot_tuple(provider_obj, database_path_str: str, *, as_of_ts: datetime):
    """One local owner, one immutable Flex snapshot, unchanged V3 identity checks."""
    workspace_dict = build_local_workspace_dict(
        provider_obj, database_path_str,
        today_str=as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat(),
    )
    if workspace_dict["financial_error_str"]:
        return workspace_dict, BrokerReportingSnapshot(
            unavailable_reason_str=workspace_dict["financial_error_str"])
    try:
        snapshot_obj = load_broker_reporting_snapshot(
            database_path_str,
            allowed_account_set={account_dict["account_route"]
                                 for account_dict in workspace_dict["valuation_account_list"]},
            query_name_str=workspace_dict["client_dict"]["query_name"],
        )
        validate_local_bindings_unchanged(workspace_dict, database_path_str)
    except (ClientReportingError, ValueError, OSError):
        snapshot_obj = BrokerReportingSnapshot(
            unavailable_reason_str="Saved account report could not be verified.")
    return workspace_dict, snapshot_obj
