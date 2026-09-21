"""Owned temporary SQLite fixtures, read through the production V4 readers."""

from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory, gettempdir

from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_v4.evidence import load_cycle_evidence_dict
from alpha.live.dashboard_v4.pod_data import load_pod_cycles_dict
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import (
    BrokerOrderAck, BrokerOrderEvent, BrokerOrderFill, BrokerOrderRecord, BrokerSnapshot,
    DecisionPlan, LiveRelease, PodState, ReconciliationResult, VPlan, VPlanRow,
)
from alpha.live.state_store_v2 import LiveStateStore


DEMO_ASSET_TUPLE = (("AMD", "CRM", "DIS"), ("AAPL", "MSFT", "NVDA"),
                    ("AMZN", "AVGO", "NVDA"), ("IEF", "LQD", "TIP"))
DEMO_PRICE_TUPLE = ((158.42, 291.05, 112.80), (170.0, 220.0, 130.0),
                    (150.0, 120.0, 130.0), (80.0, 95.0, 100.0))


class _DemoStateStore(LiveStateStore):
    """Retain real writer behavior and close every setup connection on Windows."""

    def __init__(self, db_path_str):
        self._connection_list = []
        try:
            super().__init__(db_path_str)
        except Exception:
            self.close()
            raise

    def _connect(self):
        connection_obj = super()._connect()
        self._connection_list.append(connection_obj)
        # Disposable generated fixtures only: skip disk durability work while
        # seeding. Production StateStore settings and read-only readers stay unchanged.
        connection_obj.execute("PRAGMA synchronous=OFF")
        connection_obj.execute("PRAGMA journal_mode=MEMORY")
        return connection_obj

    def close(self):
        for connection_obj in self._connection_list:
            connection_obj.close()
        self._connection_list.clear()


class DemoPodStore:
    """Create once per provider; later access uses SQLite mode=ro.

    Only DEMO identities are accepted. Database names are generated indices
    inside this object's fresh temporary directory, never release config paths.
    TemporaryDirectory cleans up with the owning provider; close() supports
    deterministic application/test cleanup. No writer survives setup.
    """

    def __init__(self, row_list, *, as_of_ts):
        if len(row_list) != len(DEMO_ASSET_TUPLE) or any(
            row_dict.get("mode_str") != "live"
            or not str(row_dict.get("pod_id_str", "")).startswith("demo_")
            or not str(row_dict.get("account_route_str", "")).startswith("DEMO_")
            for row_dict in row_list
        ):
            raise ValueError("Synthetic DEMO identities are required")
        self._temp_obj = TemporaryDirectory(prefix="alpha-super-v4-demo-")
        self.directory_path_obj = Path(self._temp_obj.name).resolve()
        self.target_dict = {}
        try:
            for index_int, row_dict in enumerate(row_list):
                database_path_obj = (self.directory_path_obj / f"pod-{index_int}.sqlite3").resolve()
                if database_path_obj.parent != self.directory_path_obj:
                    raise ValueError("Invalid synthetic database path")
                release_obj = LiveRelease(
                    release_id_str=row_dict["release_id_str"], user_id_str="DEMO-owner",
                    pod_id_str=row_dict["pod_id_str"], account_route_str=row_dict["account_route_str"],
                    strategy_import_str="demo.synthetic:Pod", mode_str="live", session_calendar_id_str="XNYS",
                    signal_clock_str="eod_snapshot_ready" if index_int < 2 else "month_end_snapshot_ready",
                    execution_policy_str=row_dict["execution_policy_str"], data_profile_str="DEMO-profile",
                    params_dict={}, risk_profile_str="synthetic", enabled_bool=True, source_path_str="DEMO-only",
                    pod_budget_fraction_float=1.0, auto_submit_enabled_bool=False,
                )
                target_obj = DashboardPodTarget(release_obj, str(database_path_obj), False)
                self.target_dict[release_obj.pod_id_str] = target_obj
                self._seed_pod(target_obj, row_dict, index_int, as_of_ts)
        except Exception:
            self.close()
            raise

    def close(self):
        # Cleanup stays within the generated directory owned by this object.
        if (self.directory_path_obj.parent != Path(gettempdir()).resolve()
                or not self.directory_path_obj.name.startswith("alpha-super-v4-demo-")):
            raise ValueError("Refusing cleanup outside the owned demo directory")
        self._temp_obj.cleanup()

    def get_target_for_pod(self, pod_id_str):
        return self.target_dict.get(pod_id_str)

    def get_target_list(self):
        return list(self.target_dict.values())

    def get_pod_cycles_dict(self, pod_id_str, *, as_of_ts, decision_plan_id_int=None, vplan_id_int=None):
        target_obj = self.get_target_for_pod(pod_id_str)
        if target_obj is None:
            return {"status_str": "not_found", "reason_str": "Cycle not found"}
        result_dict = load_pod_cycles_dict(target_obj, as_of_ts=as_of_ts,
            decision_plan_id_int=decision_plan_id_int, vplan_id_int=vplan_id_int)
        if result_dict["status_str"] == "ok":
            result_dict["cycle_evidence_dict"] = load_cycle_evidence_dict(
                target_obj, result_dict["pod_row_dict"], as_of_ts=as_of_ts)
        return result_dict

    def get_cycle_evidence_dict(self, pod_row_dict, *, as_of_ts):
        return load_cycle_evidence_dict(self.get_target_for_pod(pod_row_dict.get("pod_id_str")),
            pod_row_dict, as_of_ts=as_of_ts)

    def _seed_pod(self, target_obj, row_dict, index_int, as_of_ts):
        with closing(_DemoStateStore(target_obj.db_path_str)) as store_obj:
            self._seed_pod_rows(store_obj, target_obj, row_dict, index_int, as_of_ts)

    def _seed_pod_rows(self, store_obj, target_obj, row_dict, index_int, as_of_ts):
        release_obj = target_obj.release_obj
        store_obj.upsert_release(release_obj)
        current_target_ts = datetime.fromisoformat(row_dict["latest_vplan_target_execution_timestamp_str"])
        prior_target_ts = current_target_ts - timedelta(days=4 if index_int < 2 else 29)
        current_signal_ts = datetime.fromisoformat(row_dict["latest_decision_signal_timestamp_str"])
        prior_signal_ts = (prior_target_ts - timedelta(days=3 if prior_target_ts.weekday() == 0 else 1)).replace(hour=20, minute=0, second=0)
        asset_tuple, price_tuple = DEMO_ASSET_TUPLE[index_int], DEMO_PRICE_TUPLE[index_int]
        price_map_dict = dict(zip(asset_tuple, price_tuple)) | {"SGOV": 100.0}
        cash_float = row_dict["eod_snapshot_dict"]["cash_float"]
        equity_float = row_dict["eod_snapshot_dict"]["equity_float"]
        eod_active_value_float = 44 * price_tuple[2] if index_int < 2 else 31 * price_tuple[0] + 17 * price_tuple[1]
        carry_float = (equity_float - cash_float - eod_active_value_float) / 100.0
        if carry_float <= 0:
            raise ValueError("Synthetic cash and holdings must reconcile")
        current_after_dict = {asset_tuple[0]: 31.0, asset_tuple[1]: 17.0, "SGOV": carry_float}
        current_before_dict = {asset_tuple[2]: 44.0, "SGOV": carry_float}
        for cycle_int, target_ts, signal_ts in ((1, prior_target_ts, prior_signal_ts), (2, current_target_ts, current_signal_ts)):
            before_dict, after_dict = ((current_after_dict, current_before_dict) if cycle_int == 1 else (current_before_dict, current_after_dict))
            self._seed_cycle(store_obj, release_obj, target_ts, signal_ts, before_dict, after_dict,
                price_map_dict, equity_float, issue_bool=index_int == 1 and cycle_int == 2)
            if target_ts.date() < as_of_ts.date():
                eod_ts = target_ts.replace(hour=20, minute=10, second=1)
                after_cash_float = equity_float - sum(share_float * price_map_dict[asset_str] for asset_str, share_float in after_dict.items())
                store_obj.upsert_pod_state(PodState(release_obj.pod_id_str, release_obj.user_id_str,
                    release_obj.account_route_str, dict(after_dict), after_cash_float, equity_float, {}, eod_ts,
                    snapshot_stage_str="eod", snapshot_source_str="broker"))
        # Finance closes on Sep 4: persist that same position/cash observation,
        # independently of the current execution cycle.
        eod_ts = datetime.fromisoformat(row_dict["eod_snapshot_dict"]["latest_timestamp_str"])
        eod_position_dict = current_before_dict if index_int < 2 else current_after_dict
        store_obj.upsert_pod_state(PodState(release_obj.pod_id_str, release_obj.user_id_str,
            release_obj.account_route_str, dict(eod_position_dict), cash_float, equity_float, {}, eod_ts,
            snapshot_stage_str="eod", snapshot_source_str="broker"))
        store_obj.upsert_broker_snapshot_cache(BrokerSnapshot(
            account_route_str=release_obj.account_route_str, snapshot_timestamp_ts=eod_ts,
            cash_float=cash_float, total_value_float=equity_float, net_liq_float=equity_float,
            position_amount_map=dict(eod_position_dict)))
        position_ts, position_dict = eod_ts, eod_position_dict
        if index_int == 0:
            position_ts, position_dict = current_target_ts + timedelta(minutes=6, seconds=12), current_after_dict
            current_cash_float = equity_float - sum(share_float * price_map_dict[asset_str] for asset_str, share_float in current_after_dict.items())
            store_obj.upsert_pod_state(PodState(release_obj.pod_id_str, release_obj.user_id_str,
                release_obj.account_route_str, dict(position_dict), current_cash_float, equity_float, {}, position_ts,
                snapshot_stage_str="post_execution", snapshot_source_str="broker"))
        # StateStore timestamps writes with wall time. Freeze only this newly
        # generated fixture's timestamps to its explicit demo observations.
        with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
            connection_obj.execute("PRAGMA synchronous=OFF")
            connection_obj.execute("PRAGMA journal_mode=MEMORY")
            connection_obj.execute("UPDATE live_release SET updated_timestamp_str=?", (as_of_ts.isoformat(),))
            connection_obj.execute("UPDATE pod_state_history SET recorded_timestamp_str=updated_timestamp_str")
            connection_obj.execute("UPDATE broker_snapshot_cache SET updated_timestamp_str=snapshot_timestamp_str")
        source_dict = self.get_pod_cycles_dict(release_obj.pod_id_str, as_of_ts=as_of_ts)
        if source_dict["status_str"] != "ok" or source_dict["cycle_evidence_dict"]["state_str"] not in {"complete", "partial"}:
            raise ValueError("Synthetic database failed the real evidence reader")
        for key_str, value_obj in source_dict["pod_row_dict"].items():
            if key_str not in {"eod_snapshot_dict", "required_action_dict"}:
                row_dict[key_str] = value_obj
        row_dict.update(cycle_evidence_dict=source_dict["cycle_evidence_dict"],
            user_id_str=release_obj.user_id_str, signal_clock_str=release_obj.signal_clock_str,
            latest_pod_state_timestamp_str=position_ts.isoformat(), latest_broker_snapshot_timestamp_str=eod_ts.isoformat(),
            latest_live_reference_snapshot_timestamp_str=(current_target_ts - timedelta(minutes=7)).isoformat(),
            latest_live_reference_source_str="DEMO saved reference",
            position_exposure_dict_list=[{"asset_str": asset_str, "share_float": share_float, "price_float": price_map_dict[asset_str]}
                for asset_str, share_float in sorted(position_dict.items())])

    @staticmethod
    def _seed_cycle(store_obj, release_obj, target_ts, signal_ts, before_dict, after_dict,
                    price_map_dict, equity_float, *, issue_bool):
        submission_ts = target_ts - timedelta(seconds=390)
        created_ts = submission_ts - timedelta(minutes=2)
        checked_ts = target_ts + timedelta(minutes=6, seconds=12)
        status_str = "submitted" if issue_bool else "completed"
        # Synthetic executions and valuations use the same fixed prices, with
        # zero fees: cash = NAV - sum(shares * price) on each side of the trade.
        before_cash_float = equity_float - sum(share_float * price_map_dict[asset_str] for asset_str, share_float in before_dict.items())
        after_cash_float = equity_float - sum(share_float * price_map_dict[asset_str] for asset_str, share_float in after_dict.items())
        identity_dict = {field_str: getattr(release_obj, field_str) for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str")}
        delta_map_dict = {asset_str: after_dict.get(asset_str, 0.0) - before_dict.get(asset_str, 0.0)
            for asset_str in sorted(set(before_dict) | set(after_dict)) if asset_str != "SGOV"}
        target_map_dict = {asset_str: after_dict[asset_str] * price_map_dict[asset_str] / equity_float for asset_str in delta_map_dict if after_dict.get(asset_str, 0) > 0}
        decision_obj = store_obj.insert_decision_plan(DecisionPlan(**identity_dict,
            signal_timestamp_ts=signal_ts, submission_timestamp_ts=submission_ts, target_execution_timestamp_ts=target_ts,
            execution_policy_str=release_obj.execution_policy_str, decision_base_position_map=dict(before_dict),
            snapshot_metadata_dict={"norgate_data_profile_str": release_obj.data_profile_str, "norgate_snapshot_date_str": signal_ts.date().isoformat()},
            strategy_state_dict={}, status_str=status_str, entry_target_weight_map_dict=target_map_dict,
            exit_asset_set={asset_str for asset_str, amount_float in delta_map_dict.items() if amount_float < 0}))
        plan_obj = store_obj.insert_vplan(VPlan(**identity_dict, decision_plan_id_int=decision_obj.decision_plan_id_int,
            signal_timestamp_ts=signal_ts, submission_timestamp_ts=submission_ts, target_execution_timestamp_ts=target_ts,
            execution_policy_str=release_obj.execution_policy_str, broker_snapshot_timestamp_ts=created_ts,
            live_reference_snapshot_timestamp_ts=created_ts, live_price_source_str="DEMO saved reference",
            net_liq_float=equity_float, available_funds_float=before_cash_float, excess_liquidity_float=before_cash_float,
            pod_budget_fraction_float=1.0, pod_budget_float=equity_float, current_broker_position_map=dict(before_dict),
            live_reference_price_map=dict(price_map_dict), target_share_map=dict(after_dict), order_delta_map=delta_map_dict,
            vplan_row_list=[VPlanRow(asset_str, before_dict.get(asset_str, 0), after_dict.get(asset_str, 0), amount_float,
                price_map_dict[asset_str], after_dict.get(asset_str, 0) * price_map_dict[asset_str], "MOO", "DEMO saved reference")
                for asset_str, amount_float in delta_map_dict.items()], status_str=status_str,
            submit_ack_status_str="missing_critical" if issue_bool else "complete", missing_ack_count_int=1 if issue_bool else 0,
            ack_coverage_ratio_float=2 / 3 if issue_bool else 1.0, submit_ack_checked_timestamp_ts=submission_ts + timedelta(seconds=1)))
        for index_int, request_obj in enumerate(build_broker_order_request_list_from_vplan(plan_obj)):
            filled_bool = not issue_bool or index_int != 1
            order_id_str = f"11842203{plan_obj.vplan_id_int}{index_int}"
            fill_ts, price_float = target_ts + timedelta(seconds=index_int), price_map_dict[request_obj.asset_str]
            order_fields_dict = {"broker_order_id_str": order_id_str, "decision_plan_id_int": decision_obj.decision_plan_id_int,
                "vplan_id_int": plan_obj.vplan_id_int, "account_route_str": release_obj.account_route_str,
                "asset_str": request_obj.asset_str, "order_request_key_str": request_obj.order_request_key_str}
            store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(**order_fields_dict,
                broker_order_type_str="MOO", unit_str="shares", amount_float=request_obj.amount_float,
                filled_amount_float=abs(request_obj.amount_float) if filled_bool else 0.0,
                remaining_amount_float=0.0 if filled_bool else abs(request_obj.amount_float),
                avg_fill_price_float=price_float if filled_bool else None,
                status_str="Filled" if filled_bool else "PendingSubmit", submitted_timestamp_ts=submission_ts,
                last_status_timestamp_ts=fill_ts if filled_bool else submission_ts, submission_key_str=request_obj.submission_key_str)])
            store_obj.upsert_vplan_broker_ack_list([BrokerOrderAck(**order_fields_dict,
                broker_order_type_str="MOO", local_submit_ack_bool=True, broker_response_ack_bool=filled_bool,
                ack_status_str="broker_acked" if filled_bool else "missing_critical", ack_source_str="open_order" if filled_bool else "missing",
                response_timestamp_ts=submission_ts + timedelta(seconds=1) if filled_bool else None)])
            store_obj.insert_vplan_broker_order_event_list([BrokerOrderEvent(**order_fields_dict,
                status_str="Submitted" if filled_bool else "PendingSubmit", filled_amount_float=0.0,
                remaining_amount_float=abs(request_obj.amount_float), avg_fill_price_float=None,
                event_timestamp_ts=submission_ts + timedelta(seconds=1),
                event_source_str="DEMO broker event", submission_key_str=request_obj.submission_key_str)])
            if filled_bool:
                store_obj.upsert_vplan_fill_list([BrokerOrderFill(broker_order_id_str=order_id_str,
                    decision_plan_id_int=decision_obj.decision_plan_id_int, vplan_id_int=plan_obj.vplan_id_int,
                    account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
                    fill_amount_float=request_obj.amount_float, fill_price_float=price_float, fill_timestamp_ts=fill_ts,
                    raw_payload_dict={"exec_id_str": f"DEMO-{release_obj.pod_id_str}-{order_id_str}"})])
                store_obj.insert_vplan_broker_order_event_list([BrokerOrderEvent(**order_fields_dict,
                    status_str="Filled", filled_amount_float=abs(request_obj.amount_float), remaining_amount_float=0.0,
                    avg_fill_price_float=price_float, event_timestamp_ts=fill_ts,
                    event_source_str="DEMO broker event", submission_key_str=request_obj.submission_key_str)])
        if not issue_bool:
            store_obj.insert_vplan_reconciliation_snapshot(release_obj.pod_id_str, decision_obj.decision_plan_id_int,
                plan_obj.vplan_id_int, "post_execution", ReconciliationResult(True, "passed", {}, dict(after_dict), dict(after_dict), after_cash_float, after_cash_float))
        with closing(sqlite3.connect(store_obj.db_path_str)) as connection_obj, connection_obj:
            connection_obj.execute("PRAGMA synchronous=OFF")
            connection_obj.execute("PRAGMA journal_mode=MEMORY")
            connection_obj.execute("UPDATE decision_plan SET created_timestamp_str=?,updated_timestamp_str=? WHERE decision_plan_id_int=?",
                ((signal_ts + timedelta(hours=2)).isoformat(), checked_ts.isoformat(), decision_obj.decision_plan_id_int))
            connection_obj.execute("UPDATE vplan SET created_timestamp_str=?,updated_timestamp_str=? WHERE vplan_id_int=?",
                (created_ts.isoformat(), checked_ts.isoformat(), plan_obj.vplan_id_int))
            connection_obj.execute("UPDATE vplan_reconciliation_snapshot SET created_timestamp_str=? WHERE vplan_id_int=?", (checked_ts.isoformat(), plan_obj.vplan_id_int))
