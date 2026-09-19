"""Synthetic V4 preview. No configuration, state DB, or broker fallback."""

from copy import deepcopy
from datetime import UTC, datetime

from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple


DEMO_NOW_TS = datetime(2026, 9, 8, 13, 41, 7, tzinfo=UTC)


def build_demo_workspace_tuple():
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    client_dict = deepcopy(next(item_dict for item_dict in registry_dict["clients"]
                               if item_dict["client_id"] == "demo-client"))
    for account_dict, name_str in zip(client_dict["accounts"],
                                     ("DVO2", "QPI", "NDX Momentum", "TAA BTAL")):
        account_dict["display_name"] = name_str
    provider_obj = DemoOperationsProvider({"clients": [client_dict]})
    for index_int, row_dict in enumerate(provider_obj.row_list):
        daily_bool = index_int < 2
        row_dict.update(
            as_of_timestamp_str=DEMO_NOW_TS.isoformat(),
            strategy_name_str=client_dict["accounts"][index_int]["display_name"],
            db_exists_bool=True, health_str="green",
            next_action_str="wait", reason_code_str="cycle_completed" if daily_bool else "not_month_end_session",
            execution_policy_str="next_open_moo" if daily_bool else "next_month_first_open",
            signal_clock_str="daily_snapshot_ready" if daily_bool else "month_end_snapshot_ready",
            latest_decision_signal_timestamp_str="2026-09-04T20:00:00+00:00" if daily_bool else "2026-08-31T20:00:00+00:00",
            latest_vplan_decision_plan_id_int=2,
            latest_vplan_submission_timestamp_str="2026-09-08T13:23:30+00:00" if daily_bool else "2026-09-01T13:23:30+00:00",
            latest_vplan_target_execution_timestamp_str="2026-09-08T13:30:00+00:00" if daily_bool else "2026-09-01T13:30:00+00:00",
            latest_reconciliation_timestamp_str="2026-09-08T13:36:12+00:00" if daily_bool else "2026-09-01T13:36:12+00:00",
            broker_order_count_int=3, broker_ack_count_int=3, missing_ack_count_int=0,
            latest_submit_ack_status_str="complete", fill_count_int=3,
            required_action_dict={"severity_str": "green", "label_str": "No action", "detail_str": ""},
            debug_summary_dict={"severity_str": "green"},
            norgate_snapshot_status_dict={"status_str": "ready", "snapshot_fresh_for_cycle_bool": True,
                                         "snapshot_date_str": "2026-09-04"},
        )
        row_dict["eod_snapshot_dict"].update(
            status_str="waiting", expected_market_date_str="2026-09-08",
            expected_due_timestamp_str="2026-09-08T20:10:00+00:00",
            last_required_eod_present_bool=True, same_session_bool=False,
        )
        if index_int == 1:
            row_dict.update(
                health_str="red", latest_vplan_status_str="submitted",
                latest_decision_plan_status_str="submitted",
                broker_ack_count_int=2, missing_ack_count_int=1, fill_count_int=2,
                latest_submit_ack_status_str="missing_critical",
                latest_reconciliation_status_str="", latest_reconciliation_timestamp_str=None,
                latest_event_timestamp_str="2026-09-08T13:39:25+00:00",
                required_action_dict={"severity_str": "red", "label_str": "Review broker ACK",
                                      "reason_str": "1 of 3 orders has no saved broker ACK."},
                debug_summary_dict={"severity_str": "red"},
            )
    summary_dict = provider_obj.get_summary_dict()
    summary_dict["as_of_timestamp_str"] = DEMO_NOW_TS.isoformat()
    workspace_dict = {
        "client_dict": client_dict, "operations_account_list": client_dict["accounts"],
        "valuation_account_list": client_dict["accounts"], "summary_dict": summary_dict,
        "financial_scope_complete_bool": True, "financial_error_str": "", "operations_error_str": "",
    }
    return workspace_dict, snapshot_dict[client_dict["client_id"]], provider_obj


def create_demo_app():
    from alpha.live.dashboard_v4.app import create_app

    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    return create_app(provider_obj, demo_bool=True,
                      workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj),
                      now_fn=lambda: DEMO_NOW_TS)
