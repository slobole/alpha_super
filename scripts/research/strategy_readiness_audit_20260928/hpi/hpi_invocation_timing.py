"""B2: what the live HPI host does when invoked on a weekend, on an exchange holiday, or one session late.

Real Norgate (direct mode), real host entry point, pod state = backtest state at the reference signal date.
Usage: uv run python hpi_invocation_timing.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

import hpi_common as hc
import hpi_live_replay as lr
from alpha.live import strategy_host

CASES = [
    # (label, reference signal session, invocation timestamp ET)
    ("weekend_saturday", "2025-06-13", datetime(2025, 6, 14, 10, 0, tzinfo=lr.ET)),
    ("exchange_holiday_july4", "2025-07-03", datetime(2025, 7, 4, 18, 0, tzinfo=lr.ET)),
    ("good_friday", "2024-03-28", datetime(2024, 3, 29, 18, 0, tzinfo=lr.ET)),
    ("one_session_late_same_state", "2025-06-12", datetime(2025, 6, 13, 18, 0, tzinfo=lr.ET)),
]


def main() -> None:
    idx = {r["signal_date"]: r for r in lr.load_records("vote")}
    rows = []
    for label, ref_str, ts in CASES:
        rec = idx[pd.Timestamp(ref_str)]
        plan = strategy_host.build_decision_plan_for_release(lr.release("vote"), ts, lr.pod_state("vote", rec))
        cmp = lr.compare(plan, rec)
        rows.append({"label": label, "reference_signal": ref_str, "invoked_at": ts.isoformat(),
                     "plan_signal_ts": str(plan.signal_timestamp_ts),
                     "plan_target_execution_ts": str(plan.target_execution_timestamp_ts),
                     "matches_reference_backtest_decision": cmp["match"],
                     "live_exits": cmp["live_exits"], "bt_exits": cmp["bt_exits"],
                     "live_entries": cmp["live_entries"], "bt_entries": cmp["bt_entries"]})
        print(rows[-1])
    hc.dump_json(rows, "live_replay/b2_invocation_timing.json")


if __name__ == "__main__":
    main()
