"""Preserve immutable decision history when a reviewed CORE5 intent replaces it."""
import re


def migrate_decision_revisions(connection_obj):
    if not connection_obj.in_transaction:
        connection_obj.execute("BEGIN IMMEDIATE")
    column_list = [row_obj["name"] for row_obj in connection_obj.execute("PRAGMA table_info(decision_plan)").fetchall()]
    if "intent_revision_int" in column_list:
        return
    schema_str = connection_obj.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='decision_plan'").fetchone()[0]
    # SQLite table-level UNIQUE constraints cannot be dropped independently.
    # Rebuild only this table, preserving primary keys, data, indexes and triggers.
    schema_str, count_int = re.subn(r"UNIQUE\s*\(\s*pod_id_str\s*,\s*signal_timestamp_str\s*,\s*execution_policy_str\s*\)",
        "intent_revision_int INTEGER NOT NULL DEFAULT 0, UNIQUE(pod_id_str, signal_timestamp_str, execution_policy_str, intent_revision_int)", schema_str)
    if count_int != 1 or connection_obj.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise ValueError("Unrecognized decision schema; cannot safely migrate reviewed intent revisions.")
    schema_str = re.sub(r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[\"`\[]?decision_plan[\"`\]]?",
        "CREATE TABLE decision_plan_revision_migration", schema_str, count=1, flags=re.IGNORECASE)
    index_sql_list = [row_obj[0] for row_obj in connection_obj.execute(
        "SELECT sql FROM sqlite_master WHERE tbl_name='decision_plan' AND type IN ('index','trigger') AND sql IS NOT NULL")]
    sequence_row_obj = connection_obj.execute("SELECT seq FROM sqlite_sequence WHERE name='decision_plan'").fetchone()
    connection_obj.execute(schema_str)
    column_str = ",".join('"' + name_str.replace('"', '""') + '"' for name_str in column_list)
    connection_obj.execute(f"INSERT INTO decision_plan_revision_migration ({column_str}) SELECT {column_str} FROM decision_plan")
    connection_obj.execute("DROP TABLE decision_plan")
    connection_obj.execute("ALTER TABLE decision_plan_revision_migration RENAME TO decision_plan")
    if sequence_row_obj is not None:
        connection_obj.execute("UPDATE sqlite_sequence SET seq=MAX(seq,?) WHERE name='decision_plan'", (sequence_row_obj[0],))
    for index_sql_str in index_sql_list:
        connection_obj.execute(index_sql_str)
