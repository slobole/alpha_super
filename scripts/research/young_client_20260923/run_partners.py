"""Part 2 runs: candidate partner sleeves at small pod sizes (see plan_v2_frozen.yaml).

Reuses run_small_account.run_job; every job gets a fresh process (max_tasks_per_child=1)
so one job's redirected log stream can never leak into the next.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_small_account as runner  # noqa: E402

JOB_LIST = (
    [(alias_str, capital_float) for alias_str in ("trinity", "infl_compass", "sector_vox_iyr", "disp_kie_ihi_sma", "vixm", "eom_flow", "ndx_vxn")
     for capital_float in (5329, 9326, 13322, 26645)]
    + [("taa_btal_lin_qqq", 5329), ("taa_btal_lin_qqq", 9326)]
)


def main() -> int:
    plan_path = Path(__file__).resolve().parent / "plan_v2_frozen.yaml"
    ledger_path = runner.STUDY_DIR_PATH / "experiment_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as ledger_obj:
        ledger_obj.write(json.dumps({"event_str": "plan_v2_frozen_before_runs", "recorded_at_utc_str": runner.ladder_runner.utc_now_str(),
                                     "plan_sha256_str": hashlib.sha256(plan_path.read_bytes()).hexdigest()}) + "\n")
    row_list = []
    with ProcessPoolExecutor(max_workers=5, max_tasks_per_child=1) as executor_obj:
        future_map = {executor_obj.submit(runner.run_job, a, float(c), "2026-09-22"): (a, c) for a, c in JOB_LIST}
        for future_obj in as_completed(future_map):
            alias_str, capital_float = future_map[future_obj]
            try:
                row_list.append(future_obj.result())
                print(f"done {alias_str} @ {capital_float:,}", flush=True)
            except Exception as exc:
                row_list.append({"tag_str": f"{alias_str}__{capital_float}", "error_str": repr(exc)})
                print(f"FAILED {alias_str} @ {capital_float:,}: {exc!r}", flush=True)
    pd.DataFrame(row_list).to_csv(runner.STUDY_DIR_PATH / "run_summary_partners.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
