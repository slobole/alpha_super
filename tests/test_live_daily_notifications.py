"""One durable DAILY exception per decision, including cycles without VPlans."""
from contextlib import closing
from datetime import UTC, datetime
import json
import sqlite3

import pytest

from alpha.live import mr_capsule_notifications as alert_module
from test_live_mr_capsule_notifications import _capsule_summary_dict


NOW_TS = datetime(2026, 10, 7, 20, 1, tzinfo=UTC)


def _enqueue(connection_obj, decision_int=1, vplan_int=None, exception_list=None):
    alert_module.enqueue_daily_exception_alert(connection_obj, decision_plan_id_int=decision_int,
        vplan_id_int=vplan_int, pod_id_str="capsule_one", account_route_str="DU_ONE", mode_str="paper",
        exception_list=exception_list if exception_list is not None else [{"asset_str": "SPY",
            "quantity_float": 4.0, "side_str": "SELL", "reason_str": "Exit remains at session close",
            "expected_share_float": 0.0, "actual_share_float": 4.0}], created_timestamp_ts=NOW_TS)


def _old_table(connection_obj):
    connection_obj.execute("""CREATE TABLE mr_capsule_execution_alert (
        vplan_id_int INTEGER NOT NULL, alert_kind_str TEXT NOT NULL, pod_id_str TEXT NOT NULL,
        account_route_str TEXT NOT NULL, mode_str TEXT NOT NULL, payload_json_str TEXT NOT NULL,
        created_timestamp_str TEXT NOT NULL, delivery_claim_str TEXT, delivery_claimed_timestamp_str TEXT,
        delivered_timestamp_str TEXT, attempt_count_int INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(vplan_id_int, alert_kind_str))""")
    connection_obj.execute("INSERT INTO mr_capsule_execution_alert VALUES "
        "(8,'accepted_residual','capsule_one','DU_ONE','paper','{}',?,'claim','claimed','delivered',3)",
        (NOW_TS.isoformat(),))


def test_old_alert_migration_preserves_delivery_and_nullable_decision_cycles(tmp_path):
    with closing(sqlite3.connect(tmp_path / "old.sqlite3")) as connection_obj:
        _old_table(connection_obj)
        connection_obj.commit()
        original_tuple = connection_obj.execute("SELECT * FROM mr_capsule_execution_alert").fetchone()
        alert_module.ensure_execution_alert_schema(connection_obj)
        migrated_tuple = connection_obj.execute("SELECT * FROM mr_capsule_execution_alert").fetchone()
        assert migrated_tuple[:-1] == original_tuple
        assert migrated_tuple[-1] is None
        _enqueue(connection_obj)
        _enqueue(connection_obj, decision_int=2)
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 3
        alert_module.ensure_execution_alert_schema(connection_obj)
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 3


def test_migration_joins_finalization_transaction_and_rolls_back(tmp_path):
    with closing(sqlite3.connect(tmp_path / "rollback.sqlite3")) as connection_obj:
        _old_table(connection_obj)
        connection_obj.commit()
        connection_obj.execute("BEGIN IMMEDIATE")
        alert_module.ensure_execution_alert_schema(connection_obj)
        _enqueue(connection_obj)
        connection_obj.rollback()
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 1
        assert "decision_plan_id_int" not in {row_obj[1] for row_obj in connection_obj.execute("PRAGMA table_info(mr_capsule_execution_alert)")}


def test_decision_only_and_vplan_exceptions_deduplicate_deliver_and_survive_restart(tmp_path):
    db_path_obj = tmp_path / "daily.sqlite3"
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        alert_module.ensure_execution_alert_schema(connection_obj)
        _enqueue(connection_obj)
        _enqueue(connection_obj, vplan_int=91)
        _enqueue(connection_obj, decision_int=2)
        _enqueue(connection_obj, decision_int=3, vplan_int=93)
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 3
    summary_dict = _capsule_summary_dict(db_path_obj)
    failed_list = alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="fake",
        webhook_poster_fn=lambda *_arg_tuple: False, now_ts=NOW_TS)
    assert len(failed_list) == 3 and not any(row_obj.delivered_bool for row_obj in failed_list)
    payload_list = []
    def send_fn(url_str, payload_dict):
        payload_list.append(payload_dict)
        return True
    delivered_list = alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="fake",
        webhook_poster_fn=send_fn, now_ts=NOW_TS)
    assert {row_obj.decision_plan_id_int for row_obj in delivered_list} == {1, 2, 3}
    assert [row_obj.vplan_id_int for row_obj in delivered_list] == [None, None, 93]
    assert len(payload_list) == 3
    assert all("SPY: side=SELL; quantity=4 shares; reason=Exit remains" in row_dict["content"] for row_dict in payload_list)
    assert "Decision 1 | VPlan not built" in payload_list[0]["content"]
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="fake", webhook_poster_fn=send_fn, now_ts=NOW_TS) == []
    with closing(sqlite3.connect(db_path_obj)) as connection_obj:
        assert connection_obj.execute("SELECT attempt_count_int FROM mr_capsule_execution_alert").fetchall() == [(2,), (2,), (2,)]


def test_unknown_decision_only_quantity_is_not_invented_and_empty_exceptions_do_not_alert(tmp_path):
    with closing(sqlite3.connect(tmp_path / "unknown.sqlite3")) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        alert_module.ensure_execution_alert_schema(connection_obj)
        _enqueue(connection_obj, exception_list=[])
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 0
        _enqueue(connection_obj, exception_list=[{"asset_str": "GLD", "quantity_float": None,
            "side_str": "BUY", "reason_str": "VPlan never built before cutoff", "target_weight_float": .2,
            "intent_str": "target weight"}])
        row_dict = dict(connection_obj.execute("SELECT * FROM mr_capsule_execution_alert").fetchone())
        payload_dict = alert_module._build_payload_dict(row_dict)
        assert "GLD: side=BUY; quantity=unknown shares" in payload_dict["content"]
        assert "target_weight=0.2; intent=target weight" in payload_dict["content"]
        assert payload_dict["allowed_mentions"] == {"parse": []}


@pytest.mark.parametrize("mutation_dict", [{"side_str": "UNKNOWN"}, {"quantity_float": -1},
    {"reason_str": ""}, {"quantity_float": float("nan")}])
def test_invalid_exception_cannot_be_enqueued(tmp_path, mutation_dict):
    with closing(sqlite3.connect(tmp_path / "invalid.sqlite3")) as connection_obj:
        alert_module.ensure_execution_alert_schema(connection_obj)
        with pytest.raises(ValueError):
            _enqueue(connection_obj, exception_list=[{"asset_str": "SPY", "quantity_float": 1,
                "side_str": "BUY", "reason_str": "missed", **mutation_dict}])
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 0
