"""Synthetic Activity events only; never fall back to workstation logs."""

from copy import deepcopy
from datetime import timedelta


def attach_demo_activity(provider_obj, *, as_of_ts):
    event_list = []
    for index_int, row_dict in enumerate(provider_obj.row_list[:2]):
        pod_id_str = row_dict["pod_id_str"]
        for code_str, minute_int, second_int in (("build_vplan_created", 20, 4), ("submit_vplan_completed", 23, 31)):
            event_list.append({"timestamp_str": as_of_ts.replace(hour=13, minute=minute_int, second=second_int).isoformat(),
                "event_type_str": code_str, "level_str": "info", "mode_str": "live", "pod_id_str": pod_id_str,
                "decision_plan_id_int": 2, "vplan_id_int": 2, "source_str": "Demo event log",
                "payload_dict": {"release_id_str": row_dict["release_id_str"], "order_count_int": 3}})
        if index_int == 1:
            event_list.append({"timestamp_str": as_of_ts.replace(hour=13, minute=24, second=40).isoformat(),
                "event_type_str": "submit_vplan_missing_broker_ack", "level_str": "critical", "mode_str": "live",
                "pod_id_str": pod_id_str, "decision_plan_id_int": 2, "vplan_id_int": 2, "source_str": "Demo event log",
                "payload_dict": {"release_id_str": row_dict["release_id_str"], "missing_ack_count_int": 1}})
            event_list.append({"timestamp_str": as_of_ts.replace(hour=13, minute=35, second=41).isoformat(),
                "event_type_str": "manual_order_submit_requested", "level_str": "warning", "mode_str": "live",
                "pod_id_str": pod_id_str, "source_str": "Demo event log",
                "payload_dict": {"release_id_str": row_dict["release_id_str"], "ticket_id_str": "demo-manual-1",
                    "asset_str": "GIS", "side_str": "SELL", "quantity_int": 2, "broker_order_type_str": "MKT"}})
    event_list.extend([
        {"timestamp_str": as_of_ts.replace(hour=13, minute=40, second=3).isoformat(), "event_type_str": "scheduler_error_retry",
            "level_str": "error", "mode_str": "live", "pod_id_str": "", "source_str": "Demo event log", "payload_dict": {}},
        {"timestamp_str": (as_of_ts - timedelta(days=1)).replace(hour=20, minute=31, second=8).isoformat(),
            "event_type_str": "operator_action_requested", "level_str": "info", "mode_str": "live",
            "pod_id_str": provider_obj.row_list[0]["pod_id_str"], "source_str": "Demo operator journal",
            "payload_dict": {"action_name_str": "export_trade_sheet", "status_str": "requested", "job_id_str": "demo-export-1"}},
        {"timestamp_str": (as_of_ts - timedelta(days=1)).replace(hour=12, minute=0, second=0).isoformat(),
            "event_type_str": "scheduler_started", "level_str": "info", "mode_str": "live", "pod_id_str": "",
            "source_str": "Demo event log", "payload_dict": {}},
    ])

    def get_activity_source_dict(*, as_of_ts, days_int):
        return {"event_list": deepcopy(event_list), "warning_list": [], "scope_key_str": "demo", "feed_available_bool": True}

    provider_obj.get_activity_source_dict = get_activity_source_dict
