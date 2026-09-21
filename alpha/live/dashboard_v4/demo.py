"""Synthetic V4 preview with owned temporary state; no real-source fallback."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from time import monotonic

from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple
from alpha.live.dashboard_v4.pod_demo import DemoPodStore
from alpha.live.dashboard_v4.positions_data import load_positions_dict


DEMO_NOW_TS = datetime(2026, 9, 8, 13, 41, 7, tzinfo=UTC)


def build_demo_workspace_tuple(*, include_holdings_bool=False):
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    client_dict = deepcopy(next(item_dict for item_dict in registry_dict["clients"]
                               if item_dict["client_id"] == "demo-client"))
    for account_dict, name_str in zip(client_dict["accounts"],
                                     ("DVO2", "QPI", "NDX Momentum", "TAA BTAL")):
        account_dict["display_name"] = name_str
    provider_obj = DemoOperationsProvider({"clients": [client_dict]})
    for index_int, row_dict in enumerate(provider_obj.row_list):
        if include_holdings_bool:
            # Explicit visual-fixture cash, also seeded into the demo EOD DB.
            row_dict["eod_snapshot_dict"]["cash_float"] = round(row_dict["eod_snapshot_dict"]["equity_float"] * .066, 2)
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
    # One owned fixture store per provider, retained until provider cleanup.
    provider_obj.pod_store_obj = DemoPodStore(provider_obj.row_list, as_of_ts=DEMO_NOW_TS,
        include_portfolio_valuation_bool=include_holdings_bool)
    for method_str in ("get_pod_cycles_dict", "get_cycle_evidence_dict", "get_target_for_pod", "get_target_list", "close"):
        setattr(provider_obj, method_str, getattr(provider_obj.pod_store_obj, method_str))
    pod_store_obj = provider_obj.pod_store_obj

    def demo_positions_dict(pod_id_str, *, as_of_ts):
        # Only owned synthetic state is read. Saved execution prices are not
        # closing marks, so this quantity reader adds no price or P&L fixture.
        return load_positions_dict(pod_store_obj.get_target_for_pod(pod_id_str), as_of_ts=as_of_ts)

    provider_obj.get_positions_dict = demo_positions_dict

    def demo_scheduler_status_dict(pod_id_str, *, as_of_ts):
        # Synthetic preview evidence only; never fall back to workstation logs.
        if pod_id_str not in {row_dict["pod_id_str"] for row_dict in provider_obj.row_list}:
            return {"state_str": "unknown", "alive_bool": None}
        return {"state_str": "sleeping", "alive_bool": True,
            "last_seen_timestamp_str": as_of_ts.isoformat(),
            "promised_wake_timestamp_str": (as_of_ts + timedelta(seconds=30)).isoformat(),
            "checked_timestamp_str": as_of_ts.isoformat(),
            "next_phase_str": "post_execution_reconcile" if pod_id_str == "demo_1_1" else "eod_snapshot",
            "reason_code_str": "waiting_for_post_execution_reconcile" if pod_id_str == "demo_1_1" else "waiting_for_eod_snapshot"}
    provider_obj.get_scheduler_status_dict = demo_scheduler_status_dict
    from alpha.live.dashboard_v4.activity_demo import attach_demo_activity
    attach_demo_activity(provider_obj, as_of_ts=DEMO_NOW_TS)
    from alpha.live.dashboard_v4.system_demo import attach_demo_system
    attach_demo_system(provider_obj)
    summary_dict = provider_obj.get_summary_dict()
    summary_dict["as_of_timestamp_str"] = DEMO_NOW_TS.isoformat()
    workspace_dict = {
        "client_dict": client_dict, "operations_account_list": client_dict["accounts"],
        "valuation_account_list": client_dict["accounts"], "summary_dict": summary_dict,
        "financial_scope_complete_bool": True, "financial_error_str": "", "operations_error_str": "",
    }
    if include_holdings_bool:
        from alpha.live.dashboard_v4.demo_holdings import attach_demo_holdings
        attach_demo_holdings(provider_obj)
    return workspace_dict, snapshot_dict[client_dict["client_id"]], provider_obj


def create_demo_app():
    from alpha.live.dashboard_v4.app import create_app

    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    start_float = monotonic()

    def demo_now_ts():
        return DEMO_NOW_TS + timedelta(seconds=monotonic() - start_float)

    def operations_workspace_dict():
        current_ts = demo_now_ts()
        current_dict = deepcopy(workspace_dict)
        current_dict["summary_dict"]["as_of_timestamp_str"] = current_ts.isoformat()
        for row_dict in current_dict["summary_dict"]["pod_row_dict_list"]:
            row_dict["as_of_timestamp_str"] = current_ts.isoformat()
        return current_dict

    def workspace_snapshot_tuple():
        return operations_workspace_dict(), snapshot_obj

    return create_app(provider_obj, demo_bool=True,
                      workspace_snapshot_fn=workspace_snapshot_tuple,
                      operations_workspace_fn=operations_workspace_dict, now_fn=demo_now_ts)
