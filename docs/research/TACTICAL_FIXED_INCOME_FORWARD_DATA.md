# Tactical Fixed Income forward FRED data

This is a research shadow path. It does not change the frozen L14 backtest, create a live DecisionPlan, submit orders, or approve an allocation.

## Data flow

```text
FRED DGS10 / DGS3MO / DAAA / DBAA
    -> fresh download of all four, with no cache fallback
    -> new hash-checked snapshot directory and receipt times
    -> historical drift comparison against the frozen run
    -> month-end receipt gate (XNYS close through 17:15 ET)
    -> contiguous monthly forward targets from frozen median history
```

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\research\tactical_fi_forward_snapshot.py capture
.venv\Scripts\python.exe scripts\research\tactical_fi_forward_snapshot.py compare <snapshot-directory>
.venv\Scripts\python.exe scripts\research\tactical_fi_forward_snapshot.py forward
```

`capture` writes a new directory under `results/research/strategy/strategy_taa_tactical_fixed_income_ief_lqd/forward_fred_snapshots/`. It fails if any series cannot be freshly downloaded and never substitutes a cache. Each raw CSV has a SHA-256 hash and a receipt timestamp in `manifest.json`. This local archive is hash checked, not externally tamper proof. The timestamp proves when this process received a file, not when FRED originally published each observation.

`compare` first verifies those hashes. It compares the new current-vintage values against the frozen FRED files on the same historical dates, recomputes the frozen 289 monthly decisions on the same Norgate trading calendar, and reports target and NAV differences. This is a retrospective revision diagnostic, not a point-in-time historical replay. The old files and their governed hashes remain unchanged.

`forward` will only emit research shadow targets for snapshots whose four downloads finished between the XNYS month-end close and 17:15 ET. It uses the latest common observation no later than the prior session, rejects a common row more than two XNYS sessions old, and requires one eligible snapshot in every month after the frozen July 2026 decision. The median starts with the frozen prehistory plus all 289 frozen monthly spreads; each forward spread is appended exactly once. It records the target computation time separately, so a later reconstruction is not presented as a decision completed by 17:15 ET. No orders or financing model are produced.

The `forward` command exits with code `2` unless every reported shadow decision was computed by its 17:15 ET cutoff and the command runs in the latest decision's XNYS close-to-17:15 ET window. It still prints the JSON status on failure. On success, it writes a monthly record under `forward_fred_snapshots/decision_records/` with the source snapshot ID, manifest hash, target, and original computation time. Later runs pin that month's snapshot and verify the recomputed target against the record; they cannot silently replace it with a later capture. This local, hash-checked journal is not externally tamper proof, and the command is not a live scheduler.

## First comparison, 2026-09-24

Snapshot `20260924T073035366647Z` had a latest observation date of 2026-09-22 in all four series. Against the frozen period through 2026-08-19, the overlapping numeric observations had **zero changed values** in every series. The retrospective comparison found **0 of 289 changed signal rows, 0 changed target rows, and $0.00 terminal or maximum daily NAV difference** at the frozen $100,000 capital base.

This capture occurred on 2026-09-24, which was not the XNYS month-end session, so it cannot produce a forward target. No eligible August 2026 snapshot is present. A later September snapshot alone will fail the contiguous-month gate; filling August requires a separately reviewed historical vintage or an explicitly changed forward-history contract. The broader G-029 research and execution gaps remain open.
