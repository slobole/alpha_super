# Client daily TWR

Local reporting feature; not a trading, sync or deployment change.

## Contract

Legacy registries keep their current behavior: official account TWR for one
account covering the period, no invented combined TWR. Explicit `client_twr`
configuration selects one client method regardless of the number of strategies:

```json
"client_twr": {
  "method": "daily_nav_eod_v1",
  "reviewed_by": "Reviewer of this client's reporting contract",
  "evidence_ref": "Reviewed account scope, flow fields and timing evidence"
}
```

A reviewed non-overlapping `nav_bridge` is also required. The demo enables this
method on synthetic data only. No real-client registry is changed automatically.
Before real use the owner must accept the EOD approximation and the reviewer
must verify source-field coverage, fee basis and same-day internal transfers.
Configuration records that decision; it does not prove source authenticity.

## Mathematics and timing

For each reporting date D, use only the accounts owned on D, including cash:

```
B_D = sum(account starting NAV on D)
E_D = sum(account ending NAV on D)
F_D = sum(reviewed capital movements on D; inflow positive)
P_D = E_D - B_D - F_D
r_D = P_D / B_D
TWR = product(1 + r_D) - 1
```

`P_D` must reconcile to the independently reviewed economic components before
any return is available. Calculations use Decimal, without daily display rounding.
This is a daily EOD-flow convention: capital moves after the day's performance.
It is not exact intraday TWR and not an official consolidated IBKR return.
It can differ from official account TWR when timing conventions differ. Account
cards continue to display the official linked account returns without alteration.

Scope entries join at SOD with their opening NAV; departures leave after their
last owned EOD. Their capital is neither profit nor a reset of client history.
There is no intersection-only truncation or average of strategy-period returns.

Same-day internal transfers between included accounts net to zero. Daily net
zero is not proof of matched transfer events. Nonzero daily aggregate
`internalCashTransfers` blocks the entire client return: cash in transit or an
unknown counterparty must not shrink the return denominator. Transfers carried
in other source fields require the reviewed field/scope contract to classify
them correctly. No synthetic cash-in-transit ledger is inferred.

## Availability

Missing/unfinalized account days, failed NAV/flow bridges or continuity, nonzero
linking adjustments, unmatched internal transfers, nonpositive opening capital,
or daily returns at/below -100% withhold the full selected-period TWR and path.
There is no partial-series linking, forward fill, zero-capital rebasing or fallback
to averaged official returns. Dollar results and official account returns retain
their independent existing validity checks. Unsupported total-loss periods remain
unavailable; the existing broker parser rejects official daily returns <= -100%.

The configured method, its availability/reason and all daily return inputs enter
the report hash. A configured but unavailable client TWR keeps the investor
report DRAFT. Registries without the new contract retain their existing finality
policy. The dashboard and investor export consume one common result.

```
[Saved broker NAV + flows] -> [Scope + bridge checks] -> [Daily client returns]
                                                       |
                                                       v
                                               [Linked period TWR]
                                                       |
                                              [Dashboard and PDF]
```

## Source boundary

[IBKR PortfolioAnalyst whitepaper](https://www.ibkrguides.com/portfolioanalyst/pa-white-papers-llc.pdf)
describes aggregating daily NAV/flows and linking daily combined returns. Its EOD
timing note and numerical example are inconsistent (the example fits BOD flows).
Therefore this named EOD convention is explicit, not a claim of broker parity.
[Change in NAV fields](https://www.ibkrguides.com/reportingreference/reportguide/changeinnav_fq.htm)
also require a reviewed non-overlap contract, not blindly summing every field.
Real-source comparison is a separate pre-deployment gate, including a period with
flows; zero-flow agreement alone cannot establish timing parity.

## Local verification - 2026-09-06

- 888 focused regression tests passed in 39.55s, including 29 independent TWR
  cases and existing client, dashboard, IBKR performance, PDF, runner, scheduler,
  reconcile and release tests. No full research-suite claim is made.
- Offline browser matrix passed seven client views and seven advanced routes
  at 1440/768/390. Configured return headline/chart, account isolation and PDF
  download passed. Desktop overview and both final PDF pages were inspected.
- Tier 3: independent quant-pitfalls, parity, failure-modes and coverage reviews
  found no blocker. Failure reviewer also passed 13 pure in-memory cases.
- Regression correction: benchmark attachment must preserve the now-populated
  demo client return/path, not require the historical unavailable value.
  Synthetic PDF source wording is tested so it never claims actual IBKR inputs.

Live-impact checklist: next-open order timing, sizing/amount/target semantics,
reference price sources, state/pickle/SQLite formats, logging fields and released
YAML/routes are unchanged. Existing registry schema remains version 1 with an
optional validated reporting-only object. No new service, network path, Windows
file-lock behavior or process restart is introduced. Financial method version
is v3 and PDF renderer v6; issued report identities change intentionally.
No VPS, broker, sync, deployment, commit or push was performed for this feature.
Real-client activation and broker parity remain unverified and gated as above.
