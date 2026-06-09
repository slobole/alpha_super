# Live OPS Inspector Contract

TL;DR: the Inspector is a read-only operating contract, not a trading system. It
proves that each enabled POD is fresh and understandable, or it raises a flag
with evidence and a next operator action. It never submits, cancels, resizes, or
rescues orders.

## Purpose

The Inspector answers one operator question:

```text
Can I trust the live book right now, or do I need to inspect something?
```

It is intentionally smaller than a new monitoring platform. Existing sources
remain authoritative:

- release YAMLs say which PODs are enabled;
- POD SQLite files hold DecisionPlan, VPlan, broker ACK, fill, reconcile, and
  EOD evidence;
- Dashboard V3 already builds the local per-POD truth surface;
- `doctor` remains the deeper preflight/readiness tool for a single POD.

The Inspector contract sits above those facts and applies the same rule every
time:

```text
unknown != green
stale != green
silent != green
```

## Status Rules

| Status | Meaning | Operator posture |
|---|---|---|
| `green` | Every required fact for the current view is proven and fresh. | No action. |
| `yellow` | The POD is waiting for an expected timing or manual-review step. | Watch or review when due. |
| `red` | A required action, missed window, unsafe state, stale report, or failed proof exists. | Inspect now. |
| `gray` | There is not enough evidence to prove state, or no enabled PODs exist. | Do not treat as healthy. |

If a consumer reads an old report, the consumer must downgrade it before
showing it. A stale green report is not green.

## TAA Missed-Window Invariant

This invariant comes from the early-month `TAA_DF` miss:

```text
If a monthly POD is enabled and its first-tradable-open execution window is
approaching or has passed, the operator surface must show a fresh valid
DecisionPlan/VPlan lifecycle or a red/gray reason before the opportunity can
silently disappear.
```

For `strategy_taa_df`, the intended live timing remains:

```text
month-end signal -> first tradable open of next month
```

Any later manual execution is a rescue workflow outside the normal scheduled
strategy timing. The Inspector may flag that a rescue review is needed; it must
not create or submit rescue orders.

## Report Contract

The local VPS report is the machine-readable Inspector artifact:

```text
uv run python -m alpha.live.runner ops_report --mode live --json
```

The report must include:

- `schema_version_str`;
- `generated_at_utc_str`;
- `source_as_of_utc_str`;
- `stale_after_seconds_int`;
- `source_stale_bool`;
- `overall_severity_str`;
- POD counts by severity;
- one `pod_report_dict_list` item per enabled POD in the selected mode;
- a `next_operator_action_dict` for every non-green POD.

The report is read-only. It may read release YAMLs, dashboard config, POD DBs,
logs, and existing dashboard summaries. It must not write SQLite job rows,
mutate POD state, submit/cancel orders, or change release config.

## Heartbeat Contract

The heartbeat is intentionally out of band from the rich report:

```text
VPS -> rich ops report -> dashboard/operator board
VPS -> tiny heartbeat -> external dead-man switch
```

The heartbeat proves only that the VPS or scheduled heartbeat job is alive. It
must not be the only path that carries report content. If the heartbeat is late,
the external watcher alerts even if the last rich report was green.

The scheduled watchdog (`scripts/live_ops_watchdog.py`) is the alert owner. It
runs report build, red-transition webhook, and heartbeat **last**, in one
process, so the dead-man switch monitors the inspector itself — a separate
always-on pinger would prove the wrong thing alive. On overall red it pings the
fail endpoint (`<url>/fail`) so the external watcher alerts even when the
webhook channel is down; on a fatal error it pings nothing.

## Operator Surfaces

Primary surface:

```text
Dashboard V3 /live
```

Secondary surfaces:

- red-transition webhook notifications;
- the `ops_report --json` command;
- optional persisted report files or external collectors in a later phase.

## Non-Goals

The Inspector does not:

- auto-submit orders;
- cancel orders;
- build rescue trades;
- rewrite POD databases;
- replace single-POD `doctor`;
- hide unknown, stale, or unreachable state behind green.
