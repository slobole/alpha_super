"""Explicit synthetic health evidence; never falls back to workstation services."""

from datetime import timedelta

from alpha.live.dashboard_v3.health import HealthCellDict, HealthRollup, _roll_up_freshness_cell


def build_demo_health_rollup(summary_dict):
    """Keep real saved-data rollup rules, with a synthetic disk warning."""
    row_list = [row_dict for row_dict in summary_dict.get("pod_row_dict_list", []) if row_dict.get("mode_str") == "live"]
    cell_list = [_roll_up_freshness_cell(row_list, label_str)
                 for label_str in ("Norgate", "Pod state", "EOD Snapshot")]
    cell_list.append(HealthCellDict("Disk", "78% used", "yellow", "41.0 GB free"))
    severity_str = min((cell_obj.severity_str for cell_obj in cell_list),
                       key={"red": 0, "yellow": 1, "gray": 2, "green": 3}.get)
    return HealthRollup(severity_str=severity_str, cell_dict_list=cell_list)


def attach_demo_system(provider_obj):
    for row_dict in provider_obj.row_list:
        row_dict["norgate_snapshot_status_dict"].update(severity_str="green", snapshot_date_str="2026-09-04",
            last_sync_utc_str="2026-09-04T21:42:10+00:00",
            required_snapshot_date_by_release_dict={row_dict["release_id_str"]: "2026-09-04"})
        row_dict["eod_snapshot_dict"]["severity_str"] = "green"
        row_dict["dtb3_latest_observation_date_str"] = "2026-09-04"
        row_dict["data_freshness_dict"]["item_dict_list"].append({"label_str": "DTB3/FRED",
            "severity_str": "green", "value_str": "2026-09-04"})

    def source_dict(workspace_dict, *, as_of_ts):
        name_dict = {item_dict["pod_id"]: item_dict.get("display_name", item_dict["pod_id"])
                     for item_dict in workspace_dict["operations_account_list"]}
        release_list = []
        for target_obj in provider_obj.get_target_list():
            release_obj = target_obj.release_obj
            if release_obj.mode_str != "live" or release_obj.pod_id_str not in name_dict:
                continue
            release_list.append({"pod_id_str": release_obj.pod_id_str,
                "name_str": name_dict[release_obj.pod_id_str], "mode_str": "live",
                "enabled_bool": release_obj.enabled_bool, "release_id_str": release_obj.release_id_str,
                "execution_policy_str": release_obj.execution_policy_str,
                "account_str": release_obj.account_route_str[:1] + "···" + release_obj.account_route_str[-3:]})
        return {"checked_timestamp_str": as_of_ts.isoformat(), "scope_verified_bool": True, "release_list": release_list,
            "watchdog_dict": {"state_str": "done", "now_str": "Run completed",
                "last_timestamp_str": (as_of_ts - timedelta(seconds=64)).isoformat(), "expected_str": "Report within 15 min"},
            "deadman_dict": {"state_str": "done", "now_str": "Fail signal sent",
                "last_timestamp_str": (as_of_ts - timedelta(seconds=64)).isoformat(), "expected_str": "After each watchdog run"},
            "alerts_dict": {"state_str": "done", "now_str": "No saved undelivered alerts",
                "last_timestamp_str": (as_of_ts - timedelta(seconds=65)).isoformat(), "expected_str": "When a Pod needs action"},
            "flex_dict": {"state_str": "done", "now_str": "Report close 2026-09-04",
                "last_timestamp_str": "2026-09-08T11:12:30+00:00", "expected_str": "Prior session by 08:00 ET"},
            "event_log_dict": {"state_str": "done", "now_str": "38 MB · recent file write",
                "last_timestamp_str": (as_of_ts - timedelta(seconds=12)).isoformat(), "expected_str": "A write at least every 60 min"},
            "database_dict": {"state_str": "done", "now_str": "4 of 4 readable · 212 MB",
                "last_timestamp_str": (as_of_ts - timedelta(minutes=6)).isoformat(), "expected_str": "Written when state changes"}}

    provider_obj.get_system_source_dict = source_dict
