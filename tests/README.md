# Test lanes and evidence

## Default: local fixtures

From the repository root, run `uv run pytest -q`. The default lane includes
synthetic strategy, engine, accounting, LIVE-routing and dashboard regressions.
`data_integration` and `integration` tests are skipped until explicitly enabled.
A skip is missing evidence, never a pass.

From pytest configuration onward (including collection and early fixtures), the
pytest process blocks external DNS, datagrams, and connections to ambient
services (including an already-running local IBKR gateway). A test may connect
to a loopback server it starts itself. This is a test guard, not an OS sandbox:
arbitrary subprocesses do not inherit the Python socket patch. Review subprocess
commands and use disposable paths before adding them to the default lane.

Norgate imports use a temporary empty root in the default lane. Tests that
exercise status, UI or portfolio construction must inject their own heartbeat,
price and saved-result fixtures. They must not make a real-data failure pass by
returning a generic ready/healthy verdict.

Bench route fixtures have an empty results directory by default. Only evidence
cases seed saved analyzer reports. Dashboard notification state, journals and
capture-lock tests use temporary paths. Test code must never use an operator's
webhook, state database, broker or Scout ledger as a write target. In a Git
worktree the default Scout ledger points at the main checkout; pass a temporary
ledger explicitly. Broad verification is safest in a disposable independent
checkout with the same candidate source and commit history.

## Local data integration: explicit prerequisites

Choose the relevant files, ensure the local data/saved runs are available, then
add `--run-data-integration -m data_integration`. For example:

```powershell
uv run pytest --run-data-integration -m data_integration tests/test_scout_gate.py
$env:ALPHA_RUN_NORGATE_SNAPSHOT_PARITY_BOOL = "true"
uv run pytest --run-data-integration tests/test_norgate_snapshot_parity.py
```

The snapshot suite also needs a matching `NORGATE_SNAPSHOT_ROOT`. The HPI replay
suite retains its separate `ALPHA_RUN_HPI_LIVE_PARITY_BOOL` opt-in and full-history
flag. Existing prerequisite checks remain; a skipped prerequisite does not
verify data parity. The `integration` marker is used for saved-artifact research
checks and can be selected explicitly with the same lane flag.

A test named "frozen" can still query Norgate to reproduce a pinned result.
The Tactical FI price/PIT reproduction and research control-book tests are
marked individually for this reason. Local FRED hash checks, synthetic PIT
causality tests, and the governed forward-history check remain in the default
lane. Do not replace those pinned integration assertions with synthetic results.

This lane does not authorize VPS access, broker actions, release enablement,
deployment or allocation. Those are separate operations.

## Keep tests that protect contracts

Keep timing, missing-data refusal, accounting continuity and negative tests.
Keep the four gates pinned to commit `54b417f`: legacy schema, daily shared
isolation, IBKR payload parity and daily watchdog isolation. Preserve Git history
when running them. A test is eligible for deletion only after its retired
behavior and all remaining consumers are demonstrated; age or failure alone is
not enough. Do not bulk-refresh golden hashes: explain and verify each rendering
change first.

JavaScript checks run separately: `node --test tests/*.test.cjs` (expand the file
list explicitly in shells that do not expand globs for native commands).

## Research proof scripts

The momentum proof wrapper now stops on nonzero child exits and retains the
error output. Nested snapshot-decision children must exit successfully and emit
a valid result; otherwise the parent saves the diagnostic and exits nonzero.
The NDX/TAA comparison fails for empty or different row sets.
Matching refusal rows are reported and mean equal refusal, not successful
decisions. Completion of the wrapper alone does not certify research parity:
review the `parity_rows` weight-gap evidence and the individual research reports.
