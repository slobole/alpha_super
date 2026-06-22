# Live OPS Debugging Runbook

What to do when a Discord red alert fires (or the dashboard shows red). Follow
top to bottom and **stop at the first step that explains the problem** — you
will rarely reach the logs.

## Mental model: one source, zoom levels

The dashboard, the Discord alert, and `ops_report` all read the pod's SQLite DB
and release config at their own collection times. Treat differences as
timestamp/source differences first. The log files are not competing truths —
they are the detailed timeline behind that state.

```
  ONE source of truth = the pod's state (DB)

  Discord alert    ← 1 line: what & where
  ops_report.json  ← full structured reason + evidence per pod
  Dashboard        ← visual: WHERE in the lifecycle it broke
  doctor           ← deep: exactly WHICH component failed (incl. broker)
  logs             ← raw timeline, only if the above didn't explain it
```

## The funnel

```
  STEP 0  —  Read the Discord message                         (~5 sec)
     It already says: mode + pod_id + reason.
     e.g. "LIVE / pod_taa_01 turned RED — missed execution window"
     → you know WHICH pod and the CATEGORY.

  STEP 1  —  Open the dashboard for that mode                 (~30 sec)
     uv run python -m alpha.live.dashboard_v3      then open /live
     Expand the pod. The lifecycle timeline shows WHERE it broke:
        DB → Decision → VPlan → ACK → Fill → Reconcile → EOD
     The broken step is visually obvious.

  STEP 2  —  Run doctor on that ONE pod          (the main debug command)
     uv run python -m alpha.live.runner doctor --mode live --pod-id <pod_id> --json
     → PASS / WAIT / BLOCK per component, with the exact failing reason
       and live broker state (account visible? open orders? positions match?).
     90% of the time this IS the answer.

  STEP 3  —  Inspect the specific artifact         (only if you need numbers)
     about the plan?   → uv run python -m alpha.live.runner show_decision_plan --mode live --pod-id <pod_id>
     about the order?  → uv run python -m alpha.live.runner show_vplan        --mode live --pod-id <pod_id>
     about state?      → uv run python -m alpha.live.runner status            --mode live --pod-id <pod_id>

  STEP 4  —  Read logs — LAST resort, in this order:
     1. alpha/live/logs/live_critical_events.jsonl   ← small, high-signal first
     2. alpha/live/logs/pods/<pod_id>/...            ← that pod's full trace
     3. alpha/live/logs/live_events.jsonl            ← the firehose, rarely needed
     "Did I/someone do something?" → alpha/live/logs/operator_journal.jsonl
```

`doctor` is paper/live only and reaches the broker — do not run it in a loop. The
Inspector tells you *when* to look; `doctor` is what you run *on* the flagged pod.
See `docs/live/INSPECTOR_CONTRACT.md` for the read-only contract.

## Where each symptom points

```
  Discord reason / dashboard color     Look at                    Likely cause
  ──────────────────────────────────────────────────────────────────────────────
  "missed execution window" (RED)      timeline: Decision/VPlan   plan built, never
                                       step + doctor broker       submitted (TWS down,
                                                                  auto-submit off)
  pod RED, status "blocked"            doctor → which component   data gate / provenance
  pod GRAY (unknown state)             dashboard evidence + DB    missing snapshot / no
                                                                  state yet / setup gap
  Inspector stale (RED)                generated/source as-of     watchdog or dashboard
                                                                  not refreshing
  No Discord at all, but VPS silent    healthchecks.io alert      VPS dead / watchdog
                                                                  itself died (see below)
```

## Worked example — "missed execution window"

```
  Step 0  Discord → TAA live pod, missed window.
  Step 1  Dashboard /live → expand pod → timeline shows:
            Decision = planned ✓ , VPlan = (none) , target 09:30 already passed.
          → "It made a plan but never turned it into a submitted order."
  Step 2  doctor → WHY the submit never happened, e.g.:
            broker: account_not_visible   → TWS was down at the open
  Step 3  show_decision_plan → the target weights it WANTED to execute.
  Done.   You know what it intended and why it didn't happen.
```

## After you know the cause — what to DO

The Inspector diagnoses; it never fixes. The action is yours.

```
  Is the window still actionable?
     YES → manual rescue: check CURRENT prices, then submit manually
            (submit_vplan, or place the order yourself). A deliberate manual
            action OUTSIDE normal scheduled timing — never auto-chase.
     NO  → the opportunity is gone. Note it; do not chase a stale signal.

  Infra cause (TWS down, snapshot stale, env broken)?
     → fix the infra (restart TWS, re-sync Norgate), then RE-RUN doctor
       and confirm PASS before trusting the pod again.
```

## Special case: silence (no Discord, healthchecks alerted)

If healthchecks alerts that pings stopped but you got no Discord message, the VPS
or the watchdog itself is down — a dead machine cannot send Discord. Then:

```
  1. Can you reach the VPS at all? (RDP / ping)  → if not, it's down: restart it.
  2. VPS alive but watchdog silent?
        Get-ScheduledTaskInfo -TaskName AlphaLiveOpsWatchdog   (last run / result)
        uv run python scripts/live_ops_watchdog.py --json      (run it by hand)
     → fix whatever it prints (env, uv, config.env), then confirm pings resume.
```

## Hand off evidence to Codex

When the dashboard/doctor output is not enough, collect one redacted bundle and
send the zip for review:

```powershell
.\scripts\collect_vps_debug_bundle.ps1 -Mode live -PodId <pod_id>
```

If the live service uses an explicit DB path, pass the same path here:

```powershell
.\scripts\collect_vps_debug_bundle.ps1 -Mode live -PodId <pod_id> -DbPath <db_path>
```

The default bundle captures `ops_report`, high-signal log tails, redacted
`config.env`, release/config evidence, git state, and scheduler-task status. It
does not run `tick`, `submit_vplan`, `post_execution_reconcile`, or
`eod_snapshot`.

The zip is meant for trusted operator/Codex review. API tokens and webhooks are
redacted, but deeper opt-in commands can include account routes, positions,
cash, NetLiq, open orders, and current broker state.

Deeper checks are opt-in by design:

- `-IncludeRunnerDetails` runs `status`, `next_due`, and plan/report view
  commands. These are not order actions, but they can record diagnostic
  job/release metadata in the pod SQLite DB.
- `-IncludeDoctor -DoctorBrokerClientId <unused_id>` runs `doctor` with an
  explicit alternate IBKR client ID. Doctor can query broker state and update
  snapshot readiness metadata.
- `-IncludeNorgateDoctor` runs `doctor_norgate_client.py`, which can sync local
  snapshot files under `NORGATE_SNAPSHOT_ROOT`.
- `-ReleaseManifestPath <yaml> -IbkrProbeClientId <unused_id>` runs the
  standalone IBKR probe with an explicit unused client ID so it does not collide
  with the scheduler session.

## Command quick reference

```
  Dashboard            uv run python -m alpha.live.dashboard_v3
  Inspector (CLI)      uv run python -m alpha.live.runner ops_report --mode live --json
  Doctor (deep)        uv run python -m alpha.live.runner doctor --mode live --pod-id <pod_id> --json
  Debug bundle         .\scripts\collect_vps_debug_bundle.ps1 -Mode live -PodId <pod_id>
  Decision plan        uv run python -m alpha.live.runner show_decision_plan --mode live --pod-id <pod_id>
  VPlan                uv run python -m alpha.live.runner show_vplan --mode live --pod-id <pod_id>
  State                uv run python -m alpha.live.runner status --mode live --pod-id <pod_id>
  Manual submit        uv run python -m alpha.live.runner submit_vplan --mode live --pod-id <pod_id> --vplan-id <id>
  Reconcile            uv run python -m alpha.live.runner post_execution_reconcile --mode live --pod-id <pod_id>
  Watchdog (by hand)   uv run python scripts/live_ops_watchdog.py --json
```

Replace `live` with `paper` or `incubation` as needed (doctor: paper/live only).

**Golden rule:** go top to bottom, stop as soon as you understand it. The pile of
log files is a safety net you will mostly never touch.
