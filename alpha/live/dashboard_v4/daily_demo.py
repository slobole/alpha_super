"""Owned synthetic daily-pod (CORE5, MR capsule) state, seeded with the real writers.

Each demo pod is a LIVE-mode release with a DEMO account, a ``demo_daily_*`` id and
the real CORE5/capsule import string (as a string: no strategy is loaded). The
database is written once here, through the production state writers, and then
read through the production daily reader in SQLite mode=ro. Not wired into the
app yet; it never touches a real state DB, a broker or the network.
"""

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory, gettempdir

from alpha.live.core5_adapter import CORE5_CONTRACT_STR
from alpha.live.daily_notifications import enqueue_daily_cycle_overdue
from alpha.live.daily_reconcile import (
    DAILY_STRATEGY_IMPORT_TUPLE, claim_daily_completion_request, ensure_daily_reconcile_schema)
from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_v4.daily_data import load_daily_cycle_dict
from alpha.live.dashboard_v4.pod_demo import _DemoStateStore
from alpha.live.dispatch_state import record_dispatch_failure
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import (
    BrokerOrderAck, BrokerOrderFill, BrokerOrderRecord, BrokerOrderRequest, BrokerSnapshot,
    DecisionPlan, LiveRelease, VPlan, VPlanRow,
)


# Same instant as the V4 demo: Tue 2026-09-08 09:41:07 ET, 11 minutes after the
# open. Mon 09-07 is Labor Day, so the signal session is Fri 09-04.
DEMO_NOW_TS = datetime(2026, 9, 8, 13, 41, 7, tzinfo=UTC)
CORE5_IMPORT_STR = DAILY_STRATEGY_IMPORT_TUPLE[-1]
CAPSULE_IMPORT_STR = "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil"
DEMO_DIRECTORY_PREFIX_STR = "alpha-super-v4-daily-demo-"
# Two daily cycles: Thu 09-03 -> Fri 09-04 open, and Fri 09-04 -> Tue 09-08 open.
PREVIOUS_SIGNAL_TS = datetime(2026, 9, 3, 20, 0, tzinfo=UTC)
PREVIOUS_TARGET_TS = datetime(2026, 9, 4, 13, 30, tzinfo=UTC)
PREVIOUS_CLOSE_TS = datetime(2026, 9, 4, 20, 0, tzinfo=UTC)
CURRENT_SIGNAL_TS = datetime(2026, 9, 4, 20, 0, tzinfo=UTC)
CURRENT_TARGET_TS = datetime(2026, 9, 8, 13, 30, tzinfo=UTC)
# pod id, strategy, scenario, expected daily status key at DEMO_NOW_TS.
DEMO_DAILY_POD_TUPLE = (
    ("demo_daily_trading", CORE5_IMPORT_STR, "trading_with_completion", "trading"),
    ("demo_daily_exceptions", CAPSULE_IMPORT_STR, "last_close_exceptions", "trading"),
    ("demo_daily_not_sent", CORE5_IMPORT_STR, "not_sent", "not_sent"),
    ("demo_daily_unconfirmed", CAPSULE_IMPORT_STR, "dispatch_unconfirmed", "dispatch_unconfirmed"),
    ("demo_daily_overdue", CORE5_IMPORT_STR, "open_after_close", "open_after_close"),
)
PRICE_DICT = {"SPY": 650.0, "IEF": 96.0, "GLD": 330.0, "DBC": 23.0, "UUP": 27.0, "BIL": 91.5,
    "AAPL": 230.0, "MSFT": 500.0, "NVDA": 170.0}
BOOK_DICT = {
    CORE5_IMPORT_STR: ({"SPY": 40.0, "IEF": 120.0, "DBC": 300.0, "BIL": 200.0},
                       {"SPY": 45.0, "IEF": 110.0, "GLD": 20.0, "DBC": 0.0, "BIL": 180.0}),
    CAPSULE_IMPORT_STR: ({"AAPL": 30.0, "BIL": 400.0}, {"AAPL": 0.0, "MSFT": 12.0, "NVDA": 20.0, "BIL": 360.0}),
}


class DemoDailyStore:
    """Create once; later reads use SQLite mode=ro through the daily reader.

    Only DEMO identities are written, inside this object's fresh temporary
    directory. close() removes that directory and nothing else.
    """

    def __init__(self, *, as_of_ts=DEMO_NOW_TS):
        self.as_of_ts = as_of_ts
        self._temp_obj = TemporaryDirectory(prefix=DEMO_DIRECTORY_PREFIX_STR)
        self.directory_path_obj = Path(self._temp_obj.name).resolve()
        self.target_dict = {}
        self.expected_status_dict = {}
        try:
            for index_int, (pod_id_str, strategy_str, scenario_str, expected_str) in enumerate(DEMO_DAILY_POD_TUPLE):
                database_path_obj = (self.directory_path_obj / f"daily-{index_int}.sqlite3").resolve()
                if database_path_obj.parent != self.directory_path_obj:
                    raise ValueError("Invalid synthetic database path")
                release_obj = LiveRelease(f"{pod_id_str}.live.v1", "DEMO-owner", pod_id_str, f"DEMO_DU{index_int}",
                    strategy_str, "live", "XNYS", "eod_snapshot_ready", "next_open_moo", "DEMO-profile", {},
                    "synthetic", True, "DEMO-only", pod_budget_fraction_float=1.0, auto_submit_enabled_bool=True)
                target_obj = DashboardPodTarget(release_obj, str(database_path_obj), False)
                self.target_dict[pod_id_str] = target_obj
                self.expected_status_dict[pod_id_str] = expected_str
                with closing(_DemoStateStore(target_obj.db_path_str)) as store_obj:
                    self._seed_pod(store_obj, release_obj, scenario_str)
                self._freeze_timestamps(target_obj.db_path_str)
        except Exception:
            self.close()
            raise

    def close(self):
        # Cleanup stays within the generated directory owned by this object.
        if (self.directory_path_obj.parent != Path(gettempdir()).resolve()
                or not self.directory_path_obj.name.startswith(DEMO_DIRECTORY_PREFIX_STR)):
            raise ValueError("Refusing cleanup outside the owned daily demo directory")
        self._temp_obj.cleanup()

    def get_target_for_pod(self, pod_id_str):
        return self.target_dict.get(pod_id_str)

    def get_target_list(self):
        return list(self.target_dict.values())

    def get_daily_cycle_dict(self, pod_row_dict, *, as_of_ts):
        target_obj = self.get_target_for_pod(pod_row_dict.get("pod_id_str"))
        if target_obj is None or target_obj.release_obj.account_route_str != pod_row_dict.get("account_route_str"):
            return {"status_str": "unknown"}
        return load_daily_cycle_dict(target_obj, as_of_ts=as_of_ts)

    # Seeding: real writers only. Wall-clock write times are frozen afterwards.
    def _seed_pod(self, store_obj, release_obj, scenario_str):
        with store_obj._connect() as connection_obj:
            ensure_daily_reconcile_schema(connection_obj)
        store_obj.upsert_release(release_obj)
        before_dict, after_dict = BOOK_DICT[release_obj.strategy_import_str]
        if scenario_str == "open_after_close":
            # The Fri 09-04 cycle never closed: the serve stopped after the open.
            decision_obj, plan_obj = self._seed_cycle(store_obj, release_obj, PREVIOUS_SIGNAL_TS, PREVIOUS_TARGET_TS,
                before_dict, after_dict, submission_key_str="demo:overdue")
            self._mark_submitted(store_obj, decision_obj, plan_obj)
            self._seed_broker_records(store_obj, release_obj, decision_obj, plan_obj, PREVIOUS_TARGET_TS, filled_bool=False)
            exit_str = next(asset_str for asset_str, share_float in after_dict.items() if share_float == 0)
            enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, PREVIOUS_CLOSE_TS + timedelta(hours=1, minutes=5),
                error_str="DEMO broker session unavailable", owned_order_row_list=[{
                    "account_route_str": release_obj.account_route_str, "asset_str": exit_str, "side_str": "SELL",
                    "remaining_amount_float": before_dict[exit_str], "client_id_int": 7}])
            return
        # A prior cycle closed Fri 09-04 from holdings, cleanly or with exceptions.
        previous_obj, previous_plan_obj = self._seed_cycle(store_obj, release_obj, PREVIOUS_SIGNAL_TS, PREVIOUS_TARGET_TS,
            after_dict, before_dict, submission_key_str="demo:previous")
        self._mark_submitted(store_obj, previous_obj, previous_plan_obj)
        self._seed_broker_records(store_obj, release_obj, previous_obj, previous_plan_obj, PREVIOUS_TARGET_TS, filled_bool=True)
        actual_dict = dict(before_dict)
        exception_list = []
        if scenario_str == "last_close_exceptions":
            actual_dict.update(AAPL=26.0, BIL=410.0)
            exception_list = [
                {"asset_str": "AAPL", "side_str": "BUY", "quantity_float": 4.0, "reason_str": "actual_holding_differs_from_vplan",
                 "expected_share_float": 30.0, "actual_share_float": 26.0, "residual_amount_float": 4.0},
                {"asset_str": "BIL", "side_str": "SELL", "quantity_float": 10.0, "reason_str": "actual_holding_differs_from_vplan",
                 "expected_share_float": 400.0, "actual_share_float": 410.0, "residual_amount_float": -10.0}]
        close_as_of_ts = PREVIOUS_CLOSE_TS + timedelta(minutes=1, seconds=30)
        snapshot_obj = BrokerSnapshot(release_obj.account_route_str, close_as_of_ts, 2500.0, 100000.0,
            position_amount_map=actual_dict, net_liq_float=100000.0)
        store_obj.complete_daily_cycle(previous_obj.decision_plan_id_int, previous_plan_obj.vplan_id_int, snapshot_obj,
            "completed_with_exceptions" if exception_list else "completed", exception_list, close_as_of_ts)

        decision_obj, plan_obj = self._seed_cycle(store_obj, release_obj, CURRENT_SIGNAL_TS, CURRENT_TARGET_TS,
            before_dict, after_dict, submission_key_str="demo:current")
        if scenario_str in {"trading_with_completion", "last_close_exceptions"}:
            self._mark_submitted(store_obj, decision_obj, plan_obj)
            self._seed_broker_records(store_obj, release_obj, decision_obj, plan_obj, CURRENT_TARGET_TS, filled_bool=True)
            if scenario_str == "trading_with_completion":
                # The MOO exit left a remainder: one durable same-day MKT completion claim.
                exit_str = next(asset_str for asset_str, share_float in after_dict.items() if share_float == 0)
                claim_ts = CURRENT_TARGET_TS + timedelta(minutes=6)
                request_obj = BrokerOrderRequest(release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
                    account_route_str=release_obj.account_route_str, submission_key_str="demo:current",
                    order_request_key_str=f"demo:current:daily-completion:{exit_str}", asset_str=exit_str,
                    broker_order_type_str="MKT", order_class_str="MarketOrder", unit_str="shares", amount_float=-25.0,
                    target_bool=False, trade_id_int=None, sizing_reference_price_float=PRICE_DICT[exit_str],
                    portfolio_value_float=100000.0, decision_plan_id_int=decision_obj.decision_plan_id_int,
                    vplan_id_int=plan_obj.vplan_id_int, execution_deadline_timestamp_str=(CURRENT_TARGET_TS
                        .replace(hour=20, minute=0)).isoformat())
                if not claim_daily_completion_request(store_obj, release_obj, decision_obj, plan_obj, request_obj, claim_ts):
                    raise ValueError("Synthetic completion claim failed")
        elif scenario_str == "not_sent":
            store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "vplan_ready")
            result_str = record_dispatch_failure(store_obj, release_obj, plan_obj,
                build_broker_order_request_list_from_vplan(plan_obj), ValueError("DEMO order preflight rejected"),
                CURRENT_TARGET_TS - timedelta(seconds=90), before_send_bool=True)
            if result_str != "opening_dispatch_incomplete":
                raise ValueError("Synthetic dispatch failure was not recorded")
        elif scenario_str == "dispatch_unconfirmed":
            store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "vplan_ready")
            store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitting")
        else:
            raise ValueError("Unknown daily demo scenario")

    @staticmethod
    def _seed_cycle(store_obj, release_obj, signal_ts, target_ts, before_dict, after_dict, *, submission_key_str):
        identity_tuple = (release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str, release_obj.account_route_str)
        submission_ts = target_ts - timedelta(minutes=8)
        metadata_dict = {"norgate_data_profile_str": release_obj.data_profile_str,
            "norgate_snapshot_date_str": signal_ts.date().isoformat()}
        if release_obj.strategy_import_str == CORE5_IMPORT_STR:
            metadata_dict.update(sizing_contract_str=CORE5_CONTRACT_STR, fixed_target_share_map_dict=dict(after_dict))
        else:
            metadata_dict["sizing_contract_str"] = "mr_capsule_close_targets_v1"
        decision_obj = store_obj.insert_decision_plan(DecisionPlan(*identity_tuple, signal_ts, submission_ts, target_ts,
            "next_open_moo", dict(before_dict), metadata_dict, {"committed_after_close_bool": True},
            entry_target_weight_map_dict={asset_str: round(share_float * PRICE_DICT[asset_str] / 100000.0, 6)
                for asset_str, share_float in after_dict.items() if share_float > 0 and asset_str != "BIL"},
            exit_asset_set={asset_str for asset_str, share_float in after_dict.items() if share_float == 0},
            target_share_map_dict={"BIL": after_dict["BIL"]}))
        asset_list = sorted(set(before_dict) | set(after_dict))
        delta_dict = {asset_str: after_dict.get(asset_str, 0.0) - before_dict.get(asset_str, 0.0) for asset_str in asset_list}
        plan_obj = store_obj.insert_vplan(VPlan(*identity_tuple, decision_obj.decision_plan_id_int, signal_ts, submission_ts,
            target_ts, "next_open_moo", submission_ts, submission_ts, "DEMO saved reference", 100000.0, None, None, 1.0,
            100000.0, dict(before_dict), {asset_str: PRICE_DICT[asset_str] for asset_str in asset_list},
            {asset_str: after_dict.get(asset_str, 0.0) for asset_str in asset_list}, delta_dict,
            [VPlanRow(asset_str, before_dict.get(asset_str, 0.0), after_dict.get(asset_str, 0.0), delta_dict[asset_str],
                PRICE_DICT[asset_str], after_dict.get(asset_str, 0.0) * PRICE_DICT[asset_str], "MOO") for asset_str in asset_list],
            submission_key_str=submission_key_str))
        return decision_obj, plan_obj

    @staticmethod
    def _mark_submitted(store_obj, decision_obj, plan_obj):
        store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitted")
        store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "submitted")

    @staticmethod
    def _seed_broker_records(store_obj, release_obj, decision_obj, plan_obj, target_ts, *, filled_bool):
        submitted_ts = target_ts - timedelta(minutes=2, seconds=30)
        for index_int, request_obj in enumerate(build_broker_order_request_list_from_vplan(plan_obj)):
            order_id_str = f"DEMO-{plan_obj.vplan_id_int}-{index_int}"
            fields_dict = {"broker_order_id_str": order_id_str, "decision_plan_id_int": decision_obj.decision_plan_id_int,
                "vplan_id_int": plan_obj.vplan_id_int, "account_route_str": release_obj.account_route_str,
                "asset_str": request_obj.asset_str, "order_request_key_str": request_obj.order_request_key_str}
            fill_ts = target_ts + timedelta(seconds=index_int + 1)
            store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(**fields_dict, broker_order_type_str="MOO",
                unit_str="shares", amount_float=request_obj.amount_float,
                filled_amount_float=abs(request_obj.amount_float) if filled_bool else 0.0,
                remaining_amount_float=0.0 if filled_bool else abs(request_obj.amount_float),
                avg_fill_price_float=PRICE_DICT[request_obj.asset_str] if filled_bool else None,
                status_str="Filled" if filled_bool else "PreSubmitted", submitted_timestamp_ts=submitted_ts,
                last_status_timestamp_ts=fill_ts if filled_bool else submitted_ts, submission_key_str=request_obj.submission_key_str)])
            store_obj.upsert_vplan_broker_ack_list([BrokerOrderAck(**fields_dict, broker_order_type_str="MOO",
                local_submit_ack_bool=True, broker_response_ack_bool=True, ack_status_str="broker_acked",
                ack_source_str="open_order", response_timestamp_ts=submitted_ts + timedelta(seconds=1))])
            if filled_bool:
                store_obj.upsert_vplan_fill_list([BrokerOrderFill(broker_order_id_str=order_id_str,
                    decision_plan_id_int=decision_obj.decision_plan_id_int, vplan_id_int=plan_obj.vplan_id_int,
                    account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
                    fill_amount_float=request_obj.amount_float, fill_price_float=PRICE_DICT[request_obj.asset_str],
                    fill_timestamp_ts=fill_ts, raw_payload_dict={"exec_id_str": f"DEMO-exec-{order_id_str}"})])

    def _freeze_timestamps(self, db_path_str):
        """Writers stamp some rows with wall time; pin those to explicit demo times."""
        as_of_str = self.as_of_ts.isoformat()
        with closing(sqlite3.connect(db_path_str)) as connection_obj, connection_obj:
            connection_obj.execute("UPDATE live_release SET updated_timestamp_str=?", (as_of_str,))
            for table_str in ("decision_plan", "vplan"):
                for row_id_int, signal_str, target_str, updated_str in connection_obj.execute(
                        f"SELECT {table_str}_id_int, signal_timestamp_str, target_execution_timestamp_str, updated_timestamp_str FROM {table_str}"
                        ).fetchall():
                    signal_ts, target_ts = datetime.fromisoformat(signal_str), datetime.fromisoformat(target_str)
                    created_ts = signal_ts + timedelta(hours=2) if table_str == "decision_plan" else target_ts - timedelta(minutes=8)
                    updated_ts = datetime.fromisoformat(updated_str)
                    if updated_ts > self.as_of_ts:
                        updated_ts = target_ts - timedelta(seconds=90)
                    connection_obj.execute(f"UPDATE {table_str} SET created_timestamp_str=?, updated_timestamp_str=? WHERE {table_str}_id_int=?",
                        (created_ts.isoformat(), updated_ts.isoformat(), row_id_int))
            connection_obj.execute("UPDATE pod_state_history SET recorded_timestamp_str=updated_timestamp_str")
            connection_obj.execute("""UPDATE vplan_reconciliation_snapshot SET created_timestamp_str=?
                WHERE julianday(created_timestamp_str)>julianday(?)""",
                ((PREVIOUS_CLOSE_TS + timedelta(minutes=1, seconds=30)).isoformat(), as_of_str))
