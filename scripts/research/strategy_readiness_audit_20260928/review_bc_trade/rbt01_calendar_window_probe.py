"""Review BC-trade RBT01: when does the rolling 20-year XNYS calendar window break each live TAA host?

For the three live-host TAA builders, load the execution price frame exactly as the host does (live config,
end = 2026-09-25) and record its first date. The host resolves month-ends by calling calendar.is_session() on EVERY
available date (scheduler_utils.py:291-296); exchange_calendars' default XNYS start is (process start - 20 years).
Break date = first execution date + 20 years. Also records the calendar's default END (process start + 1 year),
which bounds how long a single serve process can run before next-session lookups leave the calendar.
"""
from __future__ import annotations
import json, sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_bc_trade"
import exchange_calendars as xcals  # noqa: E402

VARIANTS = {
    "taa3x": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "btal_qqq": "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
}

def main():
    cal = xcals.get_calendar("XNYS")
    res = {"exchange_calendars_version": xcals.__version__, "calendar_first_session": str(cal.first_session.date()),
           "calendar_last_session": str(cal.last_session.date()), "pods": {}}
    try:
        cal.next_session(cal.last_session)
        res["next_session_after_last"] = "ok"
    except Exception as e:  # the in-process serve loop keeps this lru_cached calendar for its whole uptime
        res["next_session_after_last"] = f"{type(e).__name__}: {e}"
    base = import_module("strategies.taa_df.strategy_taa_df")
    for key, mod in VARIANTS.items():
        vm = import_module(mod)
        cfg = replace(vm.DEFAULT_CONFIG, end_date_str="2026-09-25")
        try:
            out = base.get_defense_first_data_with_snapshot(cfg)
            exec_df = out[0]
        except Exception as e:  # some variants may need their own loader
            res["pods"][key] = {"error": f"{type(e).__name__}: {e}"}
            continue
        first = pd.Timestamp(exec_df.index.min()).normalize()
        res["pods"][key] = {"config_start": cfg.start_date_str, "exec_first_date": str(first.date()),
                            "break_date_approx": str((first + pd.DateOffset(years=20)).date()),
                            "exec_columns": list(map(str, exec_df.columns))[:12]}
        print(key, res["pods"][key], flush=True)
    (OUT / "rbt01_calendar_window_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
