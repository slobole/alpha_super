"""Targeted B1 replay on trim-window dates (research-only).

Direct trim effect: entries of the UNTRIMMED run (live membership) into a name that at T is a member in the untrimmed
matrix but already dropped by the production 5-session trim.  For each such T (>= 2006-10-02) the live builder is fed
the PRODUCTION backtest's state at T and data ending at T; its plan is compared to the production backtest's orders.
Usage: uv run python aud_trim_replay.py <dv2|qpi>
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime

import pandas as pd

import aud_common as ac
import aud_data
import aud_live_replay as lr


def main(fam: str) -> None:
    data = aud_data.load()
    pricing, untrimmed, trimmed = data["pricing_df"], data["universe_untrimmed"], data["universe_trimmed"]
    del data
    logs = {}
    for arm in ("base", "untrimmed"):
        with (ac.OUT / "full_runs" / f"{fam}_{arm}" / "decision_log.pkl").open("rb") as h:
            logs[arm] = {r["decision_date"]: r for r in pickle.load(h)}
    ut = untrimmed.reindex(columns=trimmed.columns, fill_value=0)
    direct = []
    for T, r in logs["untrimmed"].items():
        for o in r["orders"]:
            if o["target"] or o["amount"] <= 0:
                continue
            a = o["asset"]
            if a in ut.columns and T in trimmed.index and ut.at[T, a] == 1 and trimmed.at[T, a] == 0:
                direct.append((T, a))
    rows = []
    release = lr.make_release(fam)
    for T, a in direct:
        if T < pd.Timestamp("2006-10-02") or T not in logs["base"]:
            rows.append({"decision_date": T.date().isoformat(), "asset": a, "replayed": False})
            continue
        rec = logs["base"][T]
        with lr.DataPatch(fam, pricing, untrimmed) as patch:
            patch.asof_data_ts = T
            plan = lr.BUILDER[fam](release, datetime(T.year, T.month, T.day, 20), lr.pod_state_from_record(fam, rec))
        cmp = lr.compare(plan, lr.backtest_intents(rec))
        rows.append({"decision_date": T.date().isoformat(), "asset": a, "replayed": True,
                     "asset_in_live_entries": a in plan.entry_priority_list, **cmp})
        print(rows[-1]["decision_date"], a, cmp["match"], cmp["live_entries"], cmp["bt_entries"], flush=True)
    out = {"fam": fam, "n_direct_trim_entries_untrimmed_run": len(direct), "rows": rows}
    (ac.OUT / "live_replay" / f"{fam}_trim_replay.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main(sys.argv[1])
