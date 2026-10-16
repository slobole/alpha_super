"""Reproduce Doctor's daily-cycle diagnostic boundary without broker or data IO."""
from __future__ import annotations

from datetime import datetime
from contextlib import closing
import gc
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))

from alpha.live import doctor


def probe_dict() -> dict:
    row_list = []
    with tempfile.TemporaryDirectory(prefix="wired_signoff_doctor_") as temporary_str:
        db_path = Path(temporary_str) / "daily.sqlite3"
        with closing(sqlite3.connect(db_path)) as connection_obj:
            connection_obj.execute("CREATE TABLE decision_plan (decision_plan_id_int INTEGER,pod_id_str TEXT,status_str TEXT,signal_timestamp_str TEXT,submission_timestamp_str TEXT,target_execution_timestamp_str TEXT,execution_policy_str TEXT)")
            connection_obj.execute("CREATE TABLE vplan (vplan_id_int INTEGER,decision_plan_id_int INTEGER,pod_id_str TEXT,status_str TEXT,signal_timestamp_str TEXT,submission_timestamp_str TEXT,target_execution_timestamp_str TEXT,execution_policy_str TEXT)")
            connection_obj.execute("INSERT INTO decision_plan VALUES (1,'core5_probe','submitted','2026-10-02T16:00:00-04:00','2026-10-05T09:23:30-04:00','2026-10-05T09:30:00-04:00','next_open_moo')")
            connection_obj.execute("INSERT INTO vplan VALUES (1,1,'core5_probe','submitted','2026-10-02T16:00:00-04:00','2026-10-05T09:23:30-04:00','2026-10-05T09:30:00-04:00','next_open_moo')")
            connection_obj.commit()
        for time_str in ("2026-10-05T09:24:00-04:00", "2026-10-05T10:00:00-04:00", "2026-10-05T16:30:00-04:00"):
            lifecycle_dict = doctor.inspect_persisted_lifecycle_read_only_dict(db_path_str=str(db_path),
                pod_id_str="core5_probe", as_of_ts=datetime.fromisoformat(time_str))
            component_dict = doctor._persisted_lifecycle_result_dict(lifecycle_dict)
            row_list.append({"as_of_str": time_str, "status_str": component_dict["status_str"],
                "reason_code_str": component_dict["reason_code_str"], "detail_str": component_dict["detail_str"]})
        # SQLite context managers commit/rollback but do not close. Collect the
        # inspected reader's short-lived connection before Windows temp cleanup.
        gc.collect()
    assert row_list[0]["status_str"] == "WAIT"
    assert row_list[1]["status_str"] == "BLOCK"
    return {"status_str": "reproduced", "broker_io_bool": False, "rows": row_list,
        "finding_id_str": "OPS-DOCTOR-DAILY-01",
        "contract_str": "A submitted CORE5/capsule cycle normally remains open until its target exchange close.",
        "observed_str": "Doctor calls the ordinary intraday submitted cycle expired because it reuses the MOO submission-window expiry."}


if __name__ == "__main__":
    output_path = REPO_PATH / "results/review/wired_signoff_20261015/ops_doctor_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_dict = probe_dict()
    output_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    print(json.dumps(result_dict, indent=2))
