"""Read-only duplicate-key preflight for stable, offline SQLite database copies.

Run BEFORE opening a DB previously touched by 4624236 through LiveStateStore.
No migration, index creation, deduplication, checkpoint or runtime import occurs.
Exit 0: no duplicate baseline tuples; 1: duplicates block initialization;
2: inspection failed or input is not a stable, sidecar-free offline copy.
A pass checks duplicate tuples only; it is not deployment or rollback approval.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3


FILL_KEY_COLUMN_TUPLE = (
    "vplan_id_int", "broker_order_id_str", "fill_timestamp_str", "fill_amount_float", "fill_price_float",
)
SAMPLE_LIMIT_INT = 20


def _file_fingerprint(database_path_obj: Path) -> tuple[int, int]:
    stat_obj = database_path_obj.stat()
    return stat_obj.st_size, stat_obj.st_mtime_ns


def _require_no_sidecars(database_path_obj: Path) -> None:
    sidecar_list = [str(database_path_obj) + suffix_str for suffix_str in ("-wal", "-shm", "-journal")
                    if Path(str(database_path_obj) + suffix_str).exists()]
    if sidecar_list:
        raise ValueError("Use a verified stable offline copy without SQLite sidecars; do not delete sidecars: "
                         + ", ".join(sidecar_list))


def inspect_duplicate_fills(database_path_obj: Path) -> dict:
    """Inspect only the exact 54b417f uniqueness tuple; never normalize evidence."""
    result_dict = {"database_path_str": str(database_path_obj), "status_str": "error"}
    try:
        database_path_obj = database_path_obj.resolve(strict=True)
        result_dict["database_path_str"] = str(database_path_obj)
        if not database_path_obj.is_file():
            raise ValueError("Database path must be an existing regular file.")
        _require_no_sidecars(database_path_obj)
        before_tuple = _file_fingerprint(database_path_obj)
        # immutable avoids WAL/shared-memory side effects but requires the caller's
        # stable offline copy. Sidecars and observed changes are rejected, never repaired.
        uri_str = database_path_obj.as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri_str, uri=True)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("PRAGMA query_only=ON")
            connection_obj.execute("PRAGMA temp_store=MEMORY")
            schema_row_obj = connection_obj.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vplan_fill'").fetchone()
            if schema_row_obj is None:
                raise ValueError("No vplan_fill table; duplicate compatibility is unverified.")
            column_list = [dict(row_obj) for row_obj in connection_obj.execute("PRAGMA table_info(vplan_fill)")]
            column_name_set = {row_dict["name"] for row_dict in column_list}
            missing_set = {"fill_record_id_int", *FILL_KEY_COLUMN_TUPLE} - column_name_set
            if missing_set:
                raise ValueError("Unsupported vplan_fill schema; missing columns: " + ", ".join(sorted(missing_set)))
            result_dict.update(
                table_sql_str=schema_row_obj["sql"], column_list=column_list,
                index_list=[dict(row_obj) for row_obj in connection_obj.execute(
                    "SELECT name,sql FROM sqlite_master WHERE type='index' AND tbl_name='vplan_fill' ORDER BY name")],
                broker_execution_id_column_bool="broker_execution_id_str" in column_name_set,
            )
            key_sql_str = ", ".join(FILL_KEY_COLUMN_TUPLE)
            group_sql_str = (f"SELECT {key_sql_str}, COUNT(*) AS row_count_int, "
                "MIN(fill_record_id_int) AS first_fill_record_id_int, MAX(fill_record_id_int) AS last_fill_record_id_int "
                f"FROM vplan_fill GROUP BY {key_sql_str} HAVING COUNT(*) > 1")
            count_row_obj = connection_obj.execute(
                "SELECT COUNT(*) AS group_count_int, COALESCE(SUM(row_count_int),0) AS row_count_int "
                f"FROM ({group_sql_str})").fetchone()
            group_list = [dict(row_obj) for row_obj in connection_obj.execute(
                group_sql_str + f" ORDER BY {key_sql_str} LIMIT ?", (SAMPLE_LIMIT_INT,))]
            result_dict.update(
                fill_row_count_int=connection_obj.execute("SELECT COUNT(*) FROM vplan_fill").fetchone()[0],
                duplicate_group_count_int=count_row_obj["group_count_int"],
                duplicate_row_count_int=count_row_obj["row_count_int"],
                duplicate_group_sample_list=group_list, sample_limit_int=SAMPLE_LIMIT_INT,
            )
        _require_no_sidecars(database_path_obj)
        if _file_fingerprint(database_path_obj) != before_tuple:
            raise ValueError("Database changed during inspection; use a stable offline copy.")
        result_dict["status_str"] = "duplicates_found" if result_dict["duplicate_group_count_int"] else "no_duplicate_tuples"
    except (OSError, sqlite3.Error, ValueError) as error_obj:
        result_dict["status_str"] = "error"
        result_dict["error_str"] = str(error_obj)
    return result_dict


def main(argument_list=None) -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--db", required=True, nargs="+", type=Path,
        help="Explicit stable offline DB copies; adjacent WAL/SHM/journal files are rejected.")
    argument_obj = parser_obj.parse_args(argument_list)
    result_list = [inspect_duplicate_fills(path_obj) for path_obj in argument_obj.db]
    print(json.dumps({"baseline_str": "54b417f", "check_str": "duplicate_fill_tuples_only", "database_list": result_list}, indent=2))
    if any(result_dict["status_str"] == "error" for result_dict in result_list):
        return 2
    return 1 if any(result_dict["status_str"] == "duplicates_found" for result_dict in result_list) else 0


if __name__ == "__main__":
    raise SystemExit(main())
