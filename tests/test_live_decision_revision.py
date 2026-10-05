"""Decision replacement migration preserves immutable legacy identities."""
import sqlite3

import pytest

from alpha.live.decision_revision import migrate_decision_revisions


def test_legacy_decision_revision_migration_preserves_rows_constraints_and_sequence(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        connection_obj.executescript("""
            CREATE TABLE decision_plan (
                decision_plan_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                pod_id_str TEXT NOT NULL, signal_timestamp_str TEXT NOT NULL,
                execution_policy_str TEXT NOT NULL, payload_str TEXT NOT NULL,
                UNIQUE(pod_id_str, signal_timestamp_str, execution_policy_str));
            CREATE INDEX decision_payload_idx ON decision_plan(payload_str);
            CREATE TABLE vplan (decision_plan_id_int INTEGER, payload_str TEXT);
            INSERT INTO decision_plan VALUES(7,'ndx','2026-10-05','next_open_moo','original');
            INSERT INTO vplan VALUES(7,'linked');
            UPDATE sqlite_sequence SET seq=100 WHERE name='decision_plan';
        """)
        migrate_decision_revisions(connection_obj)
    with sqlite3.connect(database_path) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        migrate_decision_revisions(connection_obj)
        assert dict(connection_obj.execute("SELECT * FROM decision_plan").fetchone()) == {
            "decision_plan_id_int": 7, "pod_id_str": "ndx", "signal_timestamp_str": "2026-10-05",
            "execution_policy_str": "next_open_moo", "payload_str": "original", "intent_revision_int": 0}
        assert connection_obj.execute("SELECT payload_str FROM vplan WHERE decision_plan_id_int=7").fetchone()[0] == "linked"
        assert connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='decision_payload_idx'").fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            connection_obj.execute("INSERT INTO decision_plan(pod_id_str,signal_timestamp_str,execution_policy_str,payload_str) VALUES('ndx','2026-10-05','next_open_moo','duplicate')")
        cursor_obj = connection_obj.execute("INSERT INTO decision_plan(pod_id_str,signal_timestamp_str,execution_policy_str,payload_str,intent_revision_int) VALUES('ndx','2026-10-05','next_open_moo','reviewed',1)")
        assert cursor_obj.lastrowid == 101
        assert connection_obj.execute("SELECT payload_str FROM decision_plan WHERE decision_plan_id_int=7").fetchone()[0] == "original"


def test_unknown_decision_schema_refuses_migration_without_losing_history():
    with sqlite3.connect(":memory:") as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        connection_obj.executescript("CREATE TABLE decision_plan (payload_str TEXT); INSERT INTO decision_plan VALUES('preserve');")
        with pytest.raises(ValueError, match="Unrecognized"):
            migrate_decision_revisions(connection_obj)
        assert connection_obj.execute("SELECT payload_str FROM decision_plan").fetchone()[0] == "preserve"
