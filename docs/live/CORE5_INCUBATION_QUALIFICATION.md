# CORE5 incubation integrity qualification

## Verdict

**2026-09-16 (Asia/Jerusalem): the two approved incubation defects are fixed and
passed controlled regression checks.** This is local integration evidence, not
a natural forward run, broker-fill qualification or LIVE approval.

CORE5's disabled local YAML, investment rules, short target cap, fixed Close-T
share quantities, commissions and quote retry/block policy are unchanged.

The before-fix evidence remains in
[the audit](../../results/research/strategy/strategy_taa_adaptive_macro_core5/incubation_qualification/2026-09-15_audit/AUDIT.md).
Its scripts intentionally reproduced the old defects; use the current tests
below to verify the corrected behavior.

## Behavior changes

### One complete settlement commit

Previously, order records, events, fills, cash entries and portfolio state
committed separately. A failure after fills could leave old cash/positions;
the old restart shortcut then treated any fill as a completed settlement.

Now opening references, order records/events, signed fills, cash entries,
portfolio state and state history use one SQLite transaction. A failure rolls
back the entire settlement. A write lock protects the checks of the saved plan,
financial baseline and prior completion before writing. A concurrent completed
attempt is a no-op. The adapter publishes cached results only after commit.

The existing fill JSON carries `incubation_settlement_version_int: 1` as evidence
that the full group was committed together. No database columns, tables or
pickle formats changed. Existing callers of the six write methods keep their
standalone transactions; the incubation transaction supplies one connection.

**Legacy pending fills:** unversioned fills from before this correction do not
prove that cash and portfolio state were applied. Such an in-flight settlement
now requires explicit accounting review. It is not automatically migrated,
replayed or declared complete. Already completed old cycles remain readable.
No existing operational state was repaired or reset during this change.

Cash math is unchanged:

```text
cash_after = cash_before - sum(signed_quantity * fill_price) - sum(commission)
position_after = position_before + signed_quantity
commission_per_leg = max(1 dollar, 0.005 dollars * abs(quantity))
```

### Opening prices belong to their target session

Previously, a current IBKR `ticker.open` could be labelled with an earlier
requested execution date.

Now an uncached incubation tick-open request must name the canonical exchange
open and occur during that same exchange date after open. The clock is checked
before and after the request, including date rollover during I/O.

Both cached and newly read records validate the account, requested asset,
target session, source, price, duplicates and timezone-aware capture time.
Capture must be after the target open, on that exchange date and no later than
the fresh post-read validation clock. The earlier adapter `as_of_ts` is not used
as the upper bound, so normal network latency does not cause false rejection.

A valid saved target-session open remains usable on later days. A missing
cached-price placeholder still permits a fresh same-session fetch. An uncached
earlier session does not fall back to today's open or a historical substitute.

```text
[target-session open available]
             |
   [validate session and source]
             |
 [calculate signed fills and cash]
             |
 [one transaction: all state or none]
             |
   [reconcile and commit strategy]
```

**Scope and source limits:** only incubation calls the changed
`get_tick_open_price_list` method. The separate general broker
`get_session_open_price_list` reader is unchanged by this fix. Capture time proves
when the field was read; IBKR/ib_async does not supply an independent timestamp
for the opening print through this field. The guard therefore prevents the
reproduced cross-session substitution, but does not certify vendor field
freshness or actual auction execution.

## Verification

- Broad relevant suite: **292 passed in 138.13 seconds**.
- Final incubation test file after two additional reviewer-requested assertions:
  **32 passed in 12.79 seconds**. Thirty overlap the broad run, giving **294
  distinct passing cases** across the final selected suite.
- Earlier focused pass: 42 passed; this is overlapping evidence, not additional
  independent cases.
- Three independent read-only reviewers covered parity, quant pitfalls, failure
  modes and coverage. No unresolved implementation findings remained.
- The review found a cached-`None` recovery regression during development; it
  was corrected and covered before acceptance.

The tests exercise actual IncubationBrokerAdapter and runner code with
controlled inputs and temporary databases, including:

- Initialization and both DBC direction changes with separate execution legs.
- Signed cash, per-leg commissions and preserved strategy state until reconcile.
- Exceptions after each of six writes, with every affected table rolled back.
- Abrupt child-process exit after the fill write, followed by rollback/recovery.
- Two simultaneous settlement attempts and changed cash/positions/strategy state.
- Restart after submit, after settlement and after completed reconciliation.
- Unversioned legacy pending fills and caller-owned transaction rollback.
- Missing/invalid prices, wrong sessions/sources/accounts, duplicate records,
  same-session future timestamps, I/O latency, exchange versus UTC date,
  cached missing-price recovery and valid old cache.
- Wrong-day/pre-open socket requests rejected before connecting; date rollover.
- Generic same-day MOC settlement still using its existing official-close source.

No strategy fitting, parameter search, performance improvement, new adjustment
type or corporate-action claim follows from these deterministic checks.
Dividend, borrow, financing and real-fill differences remain as previously
disclosed.

Reproduce the selected suite from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -B -m pytest --capture=sys -p no:cacheprovider tests/test_live_incubation.py tests/test_live_ibkr_socket_client.py tests/test_live_state_store.py tests/test_live_state_store_v2.py tests/test_live_core5_adapter.py tests/test_live_runner.py tests/test_live_scheduler_service.py tests/test_live_order_clerk.py tests/test_live_release_manifest.py tests/test_live_reconcile.py tests/test_live_dashboard.py -q
```

## Live-impact checklist

Tier 3; includes backward-compatible shared state writers.

| Surface | Result |
|---|---|
| Order timing | CORE5 remains Close-T decision and next-open MOO. Existing MOC settlement also passes. |
| Sizing | No quantity, target flag, weight, truncation or budget math changed. |
| Price sources | Existing IBKR tick-open and Norgate close sources retained; invalid session attribution is rejected. No silent substitution. |
| State/config | Existing schema and YAML preserved; additive fill JSON marker. Old unverified in-flight settlements require review as stated above. |
| Logs | Existing fields and ordinary reports retained; integrity errors give an explicit reason. |
| Windows/restart | Exception, abrupt-process-exit and two-store concurrency checks passed. Transaction connections close deterministically. |
| Released routes | No YAML, route, account, bootstrap balance or enabled/auto-submit flag changed in this phase. |

## Remaining activation work

Verify a working TWS connection and market-data access; provide fresh
`norgate_eod_core5` snapshots and process-scoped snapshot mode; then start only
the dedicated CORE5 incubation pod with its own persistent database in time
for a legitimate first EOD snapshot.

Observe natural EOD -> decision -> next-session SIM fill -> reconcile cycles.
The actual local forward run has not been started by this correction.
