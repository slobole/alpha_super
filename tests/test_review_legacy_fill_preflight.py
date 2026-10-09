"""Offline duplicate preflight reads raw SQLite only and never changes evidence."""
import hashlib
from pathlib import Path
import sqlite3

import pytest

from scripts.review import preflight_legacy_fill_duplicates as preflight


def _database(tmp_path, duplicate_bool=False):
    database_path_obj = tmp_path / "offline copy #1.sqlite3"
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.executescript("""CREATE TABLE vplan_fill (
            fill_record_id_int INTEGER PRIMARY KEY, vplan_id_int INTEGER NOT NULL,
            broker_order_id_str TEXT NOT NULL, fill_timestamp_str TEXT NOT NULL,
            fill_amount_float REAL NOT NULL, fill_price_float REAL NOT NULL,
            broker_execution_id_str TEXT);
            CREATE UNIQUE INDEX execution_idx ON vplan_fill(broker_execution_id_str);
            INSERT INTO vplan_fill VALUES(1,3,'order1','2026-10-05T13:30:00+00:00',5,100,'exec1');""")
        if duplicate_bool:
            connection_obj.execute("INSERT INTO vplan_fill VALUES(2,3,'order1','2026-10-05T13:30:00+00:00',5,100,'exec2')")
    return database_path_obj


def _snapshot(directory_path_obj):
    return {path_obj.name: hashlib.sha256(path_obj.read_bytes()).hexdigest()
        for path_obj in directory_path_obj.iterdir() if path_obj.is_file()}


@pytest.mark.parametrize("duplicate_bool", [False, True])
def test_exact_tuple_detection_preserves_database_bytes_and_directory(tmp_path, duplicate_bool):
    database_path_obj = _database(tmp_path, duplicate_bool)
    before_dict = _snapshot(tmp_path)
    result_dict = preflight.inspect_duplicate_fills(database_path_obj)
    assert result_dict["status_str"] == ("duplicates_found" if duplicate_bool else "no_duplicate_tuples")
    assert result_dict["duplicate_group_count_int"] == int(duplicate_bool)
    assert result_dict["duplicate_row_count_int"] == (2 if duplicate_bool else 0)
    assert result_dict["broker_execution_id_column_bool"]
    assert "execution_idx" in str(result_dict["index_list"])
    if duplicate_bool:
        assert result_dict["duplicate_group_sample_list"][0]["first_fill_record_id_int"] == 1
        assert result_dict["duplicate_group_sample_list"][0]["last_fill_record_id_int"] == 2
    assert _snapshot(tmp_path) == before_dict


def test_preflight_preserves_baseline_timestamp_string_identity(tmp_path):
    database_path_obj = _database(tmp_path)
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.execute("INSERT INTO vplan_fill VALUES(2,3,'order1','2026-10-05T09:30:00-04:00',5,100,'exec2')")
    result_dict = preflight.inspect_duplicate_fills(database_path_obj)
    assert result_dict["status_str"] == "no_duplicate_tuples"
    assert result_dict["fill_row_count_int"] == 2


@pytest.mark.parametrize("suffix_str", ["-wal", "-shm", "-journal"])
def test_sidecars_abort_before_sqlite_open_and_are_never_removed(tmp_path, monkeypatch, suffix_str):
    database_path_obj = _database(tmp_path)
    Path(str(database_path_obj) + suffix_str).write_bytes(b"preserve this sidecar")
    before_dict = _snapshot(tmp_path)
    monkeypatch.setattr(preflight.sqlite3, "connect", lambda *args, **kwargs: pytest.fail("Must not open sidecar database"))
    result_dict = preflight.inspect_duplicate_fills(database_path_obj)
    assert result_dict["status_str"] == "error"
    assert "do not delete sidecars" in result_dict["error_str"]
    assert _snapshot(tmp_path) == before_dict


def test_missing_database_is_never_created(tmp_path):
    result_dict = preflight.inspect_duplicate_fills(tmp_path / "absent.sqlite3")
    assert result_dict["status_str"] == "error"
    assert list(tmp_path.iterdir()) == []


def test_unsupported_schema_is_not_a_clear_result(tmp_path):
    database_path_obj = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.execute("CREATE TABLE vplan_fill (wrong_column TEXT)")
    before_dict = _snapshot(tmp_path)
    result_dict = preflight.inspect_duplicate_fills(database_path_obj)
    assert result_dict["status_str"] == "error"
    assert "missing columns" in result_dict["error_str"]
    assert _snapshot(tmp_path) == before_dict


def test_changed_input_cannot_return_clear(tmp_path, monkeypatch):
    database_path_obj = _database(tmp_path)
    fingerprint_list = iter([(123,456), (123,457)])
    monkeypatch.setattr(preflight, "_file_fingerprint", lambda _path_obj: next(fingerprint_list))
    assert preflight.inspect_duplicate_fills(database_path_obj)["status_str"] == "error"


@pytest.mark.parametrize("duplicate_bool,expected_code_int", [(False,0), (True,1)])
def test_cli_exit_codes_and_output(tmp_path, capsys, duplicate_bool, expected_code_int):
    database_path_obj = _database(tmp_path, duplicate_bool)
    assert preflight.main(["--db", str(database_path_obj)]) == expected_code_int
    assert '"check_str": "duplicate_fill_tuples_only"' in capsys.readouterr().out
    assert preflight.main(["--db", str(tmp_path / "missing.sqlite3")]) == 2
