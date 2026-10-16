# Wired strategy sign-off audit, 2026-10-15

This is an isolated research and implementation audit of the 17 registry
identities at WIRED or LIVE. Each model starts with **USD 1,000,000 independently**.
The audit does not approve allocation, connect a broker, enable a scheduler,
change a release, deploy code, or send a notification.

## Evidence and conclusions

The owner-facing Hebrew report is
`results/review/wired_signoff_20261015/signoff_report.html`. It is standalone,
with embedded data and no network dependency. Its canvas companion uses the
installed `cursor/canvas` SDK. `audit_report.json` provides the compact data;
each card links to the complete statistics and source evidence.

Three signatures are deliberately separate: research reproducibility and
limitations; conditional execution-intent parity; and deployed-account
acceptance. A completed audit may contain blockers. The report never turns a
successful replay or Doctor result into investment approval.

## Method

- `audit_runner.py`: freeze production sources; prepare canonical native inputs;
  run canonical wrappers at USD1M, repeat USD1M, then USD2M. The engine's
  pre-iterate account state and post-iterate intent/state are saved.
- `run_cached_baselines.py`: run isolated workers on frozen input copies.
  Unprepared provider requests fail. FRED uses a saved local CSV.
- HPI directly loads native data outside the shared loader. Its saved adapter
  reuses the exact canonical post-loader price frame and PIT universe. All three
  capsule inputs also match their original price, metadata, universe and VIX
  preparations exactly. A repaired parent trace matches the original complete
  equity and cash paths; historical recorder-source drift remains disclosed.
- `audit_statistics.py`: daily-path metrics with first-day loss, common-close
  gross total-return benchmarks, subperiods, explicit missing data, causal
  trailing20-session native Turnover and fixed-USD1M order normalization.
  Original compounded order sizes are reported separately. Extra cost figures
  are fixed original-dollar ledger debits, not strategy reruns or impact models.
- Monthly actual-host replays compare target weights and exact dates. The
  `verify_monthly_bridge.py` comparison separately binds all265 replay decisions
  to the newly captured canonical engine's actual percentage orders.
- Daily actual-host replays compare entry order/ranking, exit intent, conditional
  sizing, extracted persisted state, parking share targets where applicable,
  exact session timestamps, and deliberately incorrect execution dates.
  CORE5 also tests independent NAV-only and quote-only changes.
- Adversarial, calendar, lifecycle and shared offline tests complement the
  replays. Snapshot identity/transport fixtures and oracle account seeds remain
  explicit limitations. These are not broker fills or closed-loop accounts.

## Reproduction and verification

Run from the repository root with its locked `uv` environment. All scripts write
only within this dated audit directory. Avoid overwriting frozen evidence when
starting a different audit; copy the workflow to a new dated artifact root.

```powershell
uv run python scripts/research/wired_signoff_20261015/verify_monthly_bridge.py
uv run python scripts/research/wired_signoff_20261015/build_signoff_report.py
uv run python -m pytest scripts/research/wired_signoff_20261015/test_audit_runner.py scripts/research/wired_signoff_20261015/test_report_gates.py scripts/research/wired_signoff_20261015/audit_statistics_test.py -q
```

The publisher refuses incomplete baselines, missing informative replay coverage,
failed selected tests, changed production sources, stale statistics/charts,
changed replay inputs or checkers, and mismatching monthly canonical intent.
An exact archived executed MR checker can qualify older evidence; an unrelated
current checker cannot silently replace it.

Production defects and research limitations are **recommendations only** in this
phase. The verification tier is Tier1 (research/backtest tooling); the production
source fingerprint is unchanged. Final verification and `{path, bytes, sha256}`
artifact records are saved alongside the report.
