# Client operator workspace

This is a private manager/operator console, not an investor portal. Normal use is
**one VPS = one client**; each LIVE Pod is one strategy and one linked IBKR account.
The local workspace opens directly, without a client selector or new registry. Investors
receive an intentionally exported PDF only; the application sends nothing to them.
These instructions describe local setup, not authorization to deploy or trade.

## Boundaries and flow

```text
[Existing local releases/config + saved performance bindings]
                   |
       +-----------+------------------+
       |                              |
       v                              v
[Enabled local LIVE Pods]     [Available reporting history]
[Saved operational rows]      [Saved raw IBKR Flex XML]
       |                              |
       v                              v
[Current status / refs]       [Validated NAV / flows / TWR]
       |                              |
       +----------->[Client screens]<-+
                                      |
                              [Preview hash check]
                                      |
                              [PDF + frozen JSON]

No arrow leads to a broker, scheduler or data-sync operation.
```

Activity is a separate saved-log projection of accounts owned **during the
selected interval**, including retired strategies. It does not follow today's
enabled-Pod list. The same selected-period event projection supplies Overview.

## Access and launch

No application login is required. The former
`ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR` setting is ignored and may be removed.
The real-data CLI still loads `config.env` for data-source settings;
`--demo` deliberately does not load that file. Access is controlled by the
host/tailnet: anyone who can reach the console can view financial data and,
if advanced actions are explicitly enabled, operate those controls. Restrict
access to operators; do not expose the service publicly or through Funnel.

Keep port 8080 on loopback and use Tailscale Serve HTTPS remotely. The application
does not trust browser-supplied proxy headers as proof of TLS. A direct remote
HTTP request is rejected. See `DASHBOARD_V3_RUNBOOK.md` for the proxy boundary.

A local synthetic preview is:

```powershell
uv run --locked python -m alpha.live.dashboard_v3 --demo --port 8080
```

Open `http://127.0.0.1:8080/clients`. Two visibly synthetic clients contain two and
four strategies. One has a deliberately modeled warning. No real provider,
config.env, database or broker is used; every operational action is disabled.
Synthetic financial history ends 2026-09-04. Later selected periods correctly
show incomplete coverage rather than fabricating new history.

For normal VPS use, **after a separate deployment review**, keep the existing command:

```powershell
uv run --locked python -m alpha.live.dashboard_v3
```

Open `http://127.0.0.1:8080/`. No new environment setting, account mapping or
client registry is needed. The normal CLI still loads the existing `config.env`
and defaults to read-only. Each client's VPS runs its own console; this does not
connect to other VPSs. `/vps` retains the advanced saved-evidence pages/tools.

The local adapter uses the running provider's release and dashboard-config paths,
including existing SQLite overrides, and the existing performance DB/query settings.
Enabled releases determine current operational scope. Saved performance bindings
retain retired history; disabled foreign-client templates do not hide current Pods.
Missing/corrupt financial sources never block operational pages or earlier local logs.

Financial dates include **available raw IBKR NAV history**, not just strategy returns;
they do not establish mandate inception, funding or first fill. Account values use
all verified local account identities, including retired or not-yet-started strategies.
For each date, `local NAV = sum(official account NAV)` only when every required
account has a finalized row. Missing values are not zero or carried forward.
Valid opening/closing values survive a missing middle date; the chart retains a gap.
Strategy history endpoints do not prove deposits, withdrawals or account closure.
Incomplete strategy history still withholds client P&L/TWR, but no longer suppresses
independently verified book NAV. Official strategy TWR and its date windows are unchanged.
Old NAV/TWR-only Flex imports do not prove cash movements or dollar profit. No MTM
field review or consolidated TWR method is invented to fill those gaps.

An existing explicitly reviewed reporting configuration is still supported as an
optional override (not a required setup step):

```powershell
uv run --locked python -m alpha.live.dashboard_v3 --read-only --client-registry C:\alpha\operator\client_registry.json
```

The equivalent environment variable is `ALPHA_CLIENT_REPORTING_CONFIG_PATH_STR`.
Keep this file outside Git and protect its filesystem permissions; it contains
account routes and server-local source paths. No URL/query can choose a DB path,
snapshot path or arbitrary command.

## Optional explicit registry contract

The following applies only to that optional override. It is a schema illustration, not real client data or a verified broker
field profile. Replace every example identity under an operator-reviewed setup.

```json
{
  "schema_version": 1,
  "clients": [{
    "client_id": "example-client",
    "display_name": "Example client",
    "base_currency": "USD",
    "mandate_start_date": "2026-06-01",
    "query_name": "EXAMPLE_DAILY_NAV",
    "fee_basis": "Describe exactly which broker charges and external fees are included.",
    "performance_db_path": "C:\\alpha\\operator\\saved_performance.sqlite3",
    "operations_source": "unconfigured",
    "accounts": [{
      "account_route": "EXAMPLE_ACCOUNT",
      "pod_id": "example_pod",
      "display_name": "Readable strategy name",
      "effective_from": "2026-06-01",
      "effective_to": null
    }]
  }]
}
```

Membership starts at beginning-of-day `effective_from` and ends at end-of-day
`effective_to`, inclusive. Keep retired account periods; scheduler enabled flags
do not define financial inception or erase history. Account ownership may not
overlap across clients. Each current Pod/account pair has one LIVE route. USD
is the only supported reporting currency; no implicit FX conversion is performed.

## Operational evidence

The default local workspace reads saved state for every enabled LIVE release,
including newly installed Pods without a first state or financial return. Missing
state remains unverified, while available DB/lifecycle evidence stays visible.
Operational date selection is independent of financial coverage. The source and
ownership options below describe the optional explicit-registry path.

- `operations_source: "unconfigured"` (default): explicit unavailable state,
  without asking the local provider for another client's data.
- `operations_source: "local"`: use the existing local dashboard's saved
  assessment. Match both Pod ID and broker account route with mode LIVE.
- `operations_source: "snapshot"`: read only `operations_snapshot_path` from
  server config. File is capped at 20 MB and must be an object containing
  `schema_version_int: 1`, the exact `client_id_str`, and `summary_dict` in the
  existing dashboard-summary shape (`as_of_timestamp_str`, `pod_row_dict_list`).
  No remote fetch/publisher is installed by this feature. A maintained exporter
  or read-only audit bundle must supply fresh evidence under its own authority.

Current scope uses today's New York date, regardless of the financial selector.
Duplicate/missing/incorrect identities, pre-mandate or future state, missing
freshness fields and a missing/future/older-than-120-second assessment cannot
produce green. Global Inspector/book totals are not carried into a client view.
Calendar missed-cycle findings affect the same headline as strategy status.
The status summary shows one compact line per affected Pod, prioritizing its
highest-severity cause. Deduplicated causes remain under Details; saved Pod flow
stays visible. Friendly stage labels do not turn a recorded fill into a completed
execution. Financial source errors name the missing/invalid evidence and preserve
current operations, including when a previously selectable NAV date loses its source.

The assessment time is **not** a process heartbeat. These views do not prove
that a Windows process is alive merely because rendering succeeded. The existing
watchdog/dead-man monitoring remains necessary. EOD evidence additionally checks
account route, trusted source and the actual session close plus runner buffer,
including weekends, holidays and early-close sessions.

Activity uses account periods overlapping the selected dates, including retired
strategies, independently of today's enabled/healthy Pods. Local logs are read
once per distinct Pod, capped at the latest 500 records per Pod before selection;
this is not a complete historical audit archive. Snapshot sources may optionally
include `summary_dict.event_dict_list` (at most 50,000 records within the existing
20 MB envelope). Missing/malformed exports are unavailable; there is no local
fallback. A timezone-aware `as_of_timestamp_str` bounds exported occurrences.

Events require exact Pod/account and explicit LIVE mode. Populated account or
mode aliases must agree. Canonical `event_timestamp_str` / `ts_utc` take precedence
over payload timestamps and must agree when both are present. Occurrences need a
timezone offset and must fall within both inclusive ownership endpoints and the
selected New York dates, never after the read/export time. Shared/ambiguous
payloads are omitted; only redacted scalar facts are displayed. The page caps at
1,000 attributable records and discloses truncation. Every source remains partial:
no matching events does not prove nothing happened. Read/export time is not a
continuous-coverage or process-health claim.

Overview shows up to three latest attributable plan/submission/reconciliation/EOD
or warning/error records, with the same date selection as Activity. Heartbeat
noise remains in Activity, not the overview excerpt. Investor reports do not read
or export logs. Invalid date selections retain the client and correction form,
return HTTP 400, and never silently clamp dates or display substituted results.

Exposure shows saved shares and saved VPlan reference prices with both timestamps.
`reference value = saved shares * saved reference price`. It is not a current
valuation or a same-date book allocation. No weight divides these references by
finalized Flex NAV. ETF dollar notional is not embedded leverage, look-through
exposure or beta; those remain unmeasured without an explicit instrument contract.

## Display contract

The operator pages use a compact, locally served light-only theme. Status and each Pod's
seven-stage flow remain visible; detailed causes and provenance are expandable.
Overview and Performance have numeric `%`/USD axes and a daily bar chart plus table,
with a Portfolio/strategy selector and `%`/`$` display toggle. Names are unchanged.

Overview's main chart also switches between account value (`$`, includes transfers)
and verified cumulative return (`%`). The percentage button is disabled when the
existing reporting contract provides no return; NAV change is never substituted.
Trading schedule is visible as one card per currently scoped strategy, preserving
each Pod's independently resolved Signal/Submit/Execute times and warnings. The
grouped calendar JSON remains available unchanged; missing times display an em dash.

Strategy daily `%` is the selected official IBKR account return. Daily dollars use
the existing validated NAV bridge; portfolio rows use the existing daily book and
configured consolidated-return series (or existing single-account fallback).
The same completeness and D+1 gates apply. No averaging, NAV differencing, date
filling or synthetic opening-day bar is introduced. Zero and unavailable differ;
line charts break at missing dates and retain visible isolated observations.
Strategy charts use their own labeled scales. JS toggles existing markup only.

Canonical `strategy_list[].daily_list` is additive evidence and enters the report
hash before export. Its addition changes hashes across this release, not financial
formulas or investor-export fields. The full method remains in downloadable JSON;
only a short source label appears on the main financial screens.

## Money and performance contract

### Saved LIVE/reference diagnostics

An account period can optionally pin `reference_summary_path` to an existing
reference-comparison `summary.json` in server configuration. Browser query strings
cannot choose paths. No latest-artifact discovery, regeneration, pickle loading
or broker call runs from the client panel. A pinned file is capped at 1 MB; the
exact bytes parsed are SHA-256 hashed. Change the pin deliberately when reviewing
a different archived comparison. This also works for retired account periods and
locally saved remote audit bundles without inferring today's release/account.

The Performance panel retains client and effective selected dates. It checks
saved Pod, LIVE mode and explicit account identity. The legacy exporter does not
include account identity: those artifacts remain metadata-only, not attributed
client money. Wrong identity withholds artifact facts; mismatched dates withhold
numeric comparison and never clip aggregate totals. Only exact attributable
intervals with an aware same-session `pod_state_history.eod` /
`eod_broker_netliq` actual record expose up to three finite recorded values
(reference starting budget, actual value, reference value), with their
source/basis/full original timestamp and an explicit warning that
marks, capital, code/data vintage and execution timing have not been aligned.

No artifact is certified replay by this legacy schema. Equal release IDs cannot
prove code parity; different current releases do not invalidate historical ones.
No P&L, return gap, statistical tracking error or slippage result is derived from
these values. Trade aggregates concern the target execution session and may turn
missing pricing components into zero. Dividend cash ledger policy is disclosed.
Comparison evidence never alters official accounting, investor output or current
operational health. Real account/release/decision-time evidence remains necessary
before claiming a true same-conditions LIVE replay.

Raw Flex imports are read in a read-only SQLite transaction, verified by checksum,
then applied in import order with exact requested-range replacement. A correction
that removes rows stays a tombstone; older facts do not reappear. Client ownership,
requested dates, source checksums and calculation version enter the report hash.

The dollar bridge is:

```text
Closing NAV = opening NAV + source capital movements
            + broker linking adjustments + mandate scope movements
            + investment P&L
```

Capital movements include explicit broker deposit/withdrawal, internal-cash,
asset-transfer and owner-payment fields. Linking adjustments and strategies
entering/leaving the mandate remain separate from profit. An independent,
operator-reviewed MTM field contract must reconcile each broker NAV row within
USD 0.01 before its residual is accepted as P&L; cumulative book continuity and
the final book bridge must also reconcile. Missing fields are unknown, never zero.

`nav_bridge` is optional. Without it, NAV and official account TWR can be shown,
but dollar P&L remains unknown. A valid profile specifies `profile_id`,
`evidence_ref`, `reviewed_by`, `mode: "MTM"`, `nonoverlap_confirmed: true`, and
non-overlapping `economic_fields`/optional `informational_fields`. Do not copy the
demo's synthetic `mtm` profile into production. Validate the actual expanded Flex
XML against the [IBKR Change in NAV reference](https://www.ibkrguides.com/reportingreference/reportguide/changeinnav_fq.htm).

The default single-VPS adapter supplies the built-in `ibkr_mtm_expanded_v1`
bridge and `daily_nav_eod_v1` return contract. No extra local registry or environment
configuration is required. See [expanded IBKR field contract](IBKR_NAV_FIELDS.md)
for the eight supported economic fields and the required-zero restrictions on
other components. Expanded saved rows can provide dollar P&L and portfolio daily
returns after the existing scope/coverage/bridge checks. Older seven-field rows
remain incomplete for dollars; profile activation does not repair missing history
or silently shorten requested dates. Page reads never import or configure data.

An optional reviewed `client_twr` enables the headline daily consolidated return
and return chart. Its single supported method is `daily_nav_eod_v1`; see
[Client TWR contract](CLIENT_TWR.md) for exact formulas, transfer restrictions,
configuration and pre-deployment evidence. It uses the same client method for
one or multiple accounts, while strategy returns remain official IBKR TWR.
Explicit registries without this object retain their old behavior; the local
adapter and synthetic demo enable it. Configured
but unavailable client TWR makes that report DRAFT. The legacy behavior below
applies to registries without `client_twr`.

Activity Flex uses D+1: current-day values and metrics remain pending. Production
onboarding must prove that each imported source is a qualifying finalized daily
statement, not an old intraday/manual snapshot that merely aged past midnight.
The date gate alone is not that proof. Retain observed non-session activity and
membership boundaries; no forward-filled missing NAV or return is invented.

For each account, chronological official daily TWR `r_t` is linked:

```text
I_0 = 1; I_t = I_(t-1) * (1 + r_t)
Selected return = I_n - 1
Peak_t = max(1, I_1, ..., I_t)
Drawdown_t = I_t / Peak_t - 1
Maximum drawdown = min(0, all selected Drawdown_t)
Monthly return = product(1 + r_t in that selected calendar month) - 1
```

Drawdown resets at the selection's opening baseline. Partial months and observed
exchange/non-session counts are explicit. No annualization or short-sample Sharpe
is used. Independent account returns require that account's dates; book NAV needs
the stricter common coverage of all active accounts. Missing flow classification
does not suppress an otherwise valid official account TWR.

Multiple accounts' TWRs are **not** averaged or relabeled Fund TWR. Exact client
TWR remains unavailable without a validated consolidated-return source/contract.
An account-value chart includes flows and is labeled as NAV, not return. First
reporting day, mandate inception and first broker fill remain different concepts.

## Market benchmark measurement

The optional per-client `benchmark` configuration selects one explicit saved
Norgate snapshot directory, never the network or an implicit latest snapshot:

```json
"benchmark": {
  "symbol": "SPY",
  "snapshot_directory": "C:/alpha/norgate_snapshots/norgate_eod_ndx_pit_plus_vxn_helper/2026-09-04"
}
```

This is a schema illustration, not a deployment command. Use an existing snapshot
whose `prices.parquet` contains `SPY/TOTALRETURN`; price-only `CAPITALSPECIAL`
cannot substitute. `$SPXTR/TOTALRETURN` is also supported, explicitly labeled
**S&P 500 total return**, not SPY. Benchmark changes must be chosen before reviewing
relative performance, not selected afterwards to flatter the strategy.

```text
[Official account TWR / account dates]  [Pinned manifest + hashed TR prices]
                   \                    /
                  [Exact interval / complete session coverage]
                                  |
                  [Account return / market return / difference in pp]
```

For each account period, `benchmark_return = end_TR_close / baseline_TR_close - 1`.
The baseline is the last exchange close strictly before the first account return
date. The endpoint is the last exchange close on or before the final account date.
Every required exchange session, including the baseline, needs an observed finite
positive price. No shorter intersection, missing-session fill, dividend add-back,
FX conversion, annualization or beta estimation is performed. Reported weekend
and holiday account returns (including fees) remain in account TWR; a closed-market
benchmark endpoint explicitly retains its last exchange close. This is a
retrospective valuation convention, never a signal input or decision-time replay.

`difference_pp = 100 * (official_account_return - benchmark_return)`. It is a
percentage-point difference, not dollar profit or risk-adjusted alpha. Account
costs follow the disclosed fee basis; no additional account trading/advisory
fees or investor-specific tax are deducted from the benchmark. SPY's own fund
expenses are already embedded in its market performance. Snapshot/price hashes,
dates and method are part of the same report object and exported evidence. The
PDF includes available account-level market comparisons with their method and
snapshot hashes; unsupported intervals remain explicitly unavailable.
An unavailable benchmark does not invalidate otherwise verified account finances.
The local demo benchmark is visibly synthetic, not historical SPY performance.

IBKR's own PortfolioAnalyst likewise distinguishes [cumulative/time-period
performance and benchmark comparison](https://www.ibkrguides.com/clientportal/performanceandstatements/pa_viewingaccountperformance.htm).
The saved-snapshot and exact-coverage rules above are this product's explicit
implementation contract, not an assertion of broker-certified benchmark data.

## Investor export

Overview, performance rows and report use the same accounting object. PDF and
PDF-plus-JSON exports require the current preview's report hash; a correction or
different selection returns HTTP 409 before rendering. The public snapshot omits
raw account/Pod IDs, paths, logs, credentials and query identifiers. FINAL requires
complete coverage, a reconciled dollar bridge and complete official per-account
returns. A verified multi-account report may be FINAL without consolidated TWR:
that metric reads "Not reported", never an estimate. Missing required account or
flow evidence remains DRAFT; synthetic data is DEMONSTRATION, never final.
FINAL means issued verified displayed facts, not external audit/certification.

Renderer v4 uses short, plain-language labels and one account-source sentence
instead of the long qualifications section. Demo output explicitly uses example
data, not actual IBKR results. Method, fee basis, limitations and source hashes
remain in the frozen public JSON; the PDF retains the full report ID and issue
time, draft/demo status and unavailable-return indicators. Accounting and report
eligibility are unchanged by this presentation-only revision.

The accounting `report_hash` is stable over unchanged facts. Issued-document
`document_hash` also includes the printed issue time and renderer version; it
identifies the issued JSON snapshot, not PDF bytes. Downloaded files are frozen.
There is no server-side issued-report archive or automatic investor distribution.
The current rendered document is English; Unicode/RTL names require separate
font/layout verification before an investor delivery.

## Advanced command boundary

Client routes are GET-only saved-evidence screens. Global `/vps` exposes the
existing advanced tools when not launched read-only. Those tools are not harmless
status commands: tick can trade; Doctor/sync may write; reconciliation changes
state; manual tickets send broker orders. The supported CLI defaults to read-only;
`--enable-actions` is an explicit opt-in and is rejected with `--demo`.

Confirmations expire after 120 seconds and are consumed atomically before dispatch.
They bind action, Pod, environment, account, full release, configured database and
saved execution state. Cancel/new preview for that Pod invalidates the old approval.
The worker rereads the target/state before any state-store creation or job writes.
Runner guards check loaded releases before metadata upserts and before adapter use;
direct submit/reconcile must match the captured plan objects, preserving the existing
ready/latest/auto-submit selection and SQL submission claim. Terminal historical plans
may belong to an older release on the same Pod/account; executable plans may not.

Tick explicitly authorizes one lifecycle cycle, including newly generated plans;
it is not confirmation of a fixed order list. The original tick lease remains.
These guards do not globally lock scheduler activity or promise exactly-once broker
execution. Run one dashboard process: a restart/other worker rejects unknown approvals.
After a timeout/dispatch exception, inspect job/broker evidence before a new attempt;
the consumed approval is never restored. Manual tickets freeze validated normalized
fields in a server-side preview; final POST cannot change the ticket.

The advanced catalog separately offers fixed, quoted PowerShell commands for copying
and action previews. Copy does not execute. Copied CLI commands use then-current state
and do not inherit the dashboard approval. Even `status` writes metadata; use the
saved status screen for strictly read-only inspection. No generic shell runs in the UI.

## Verification and remaining acceptance work

`scripts/review/check_local_workspace_ui.cjs` uses the actual DashboardDataProvider
with temporary production-format releases, config overrides, ledgers and saved Flex
imports. It exercises the non-demo launcher without a registry, all seven pages at
390/768/1440 pixels, visible Pod flow, historical Activity forms, daily scope/unit
selection, isolated-point opacity and the advanced routes. It asserts unchanged fixture bytes/mtimes and zero outgoing connection
attempts. Its synthetic warnings are not a production health assessment.

Run the same checker with `--expanded` for the complete-data path: two strategies,
a deposit day and a loss day, matching portfolio/strategy dollars, official
strategy returns, calculated portfolio returns, chart unit toggles and PDF export.
It also verifies that selecting mixed legacy/expanded history restores the
unavailable state without suppressing operational status.

Single-VPS wiring live-impact check: order/next-open timing, sizing amount/target
semantics, reference-price sources, released YAML intent, state/pickle formats,
SQLite schemas and consumed logging fields are unchanged. No migration or service
restart runs from a page. Existing path overrides are reused; financial SQLite is
opened read-only and concurrent binding changes fail closed. Dashboard restart
behavior and action confirmation gates are unchanged. Missing/corrupt sources and
unverified history remain explicit rather than causing a setup prerequisite.

`scripts/review/check_client_reporting_ui.cjs` runs a no-login, synthetic
loopback server, blocks external resources, visits every client tab at 390/768/1440
pixels, checks client isolation/date form/PDF download, Y-axis readability/gridline
alignment and daily scope/unit selection, saves screenshots and stops
its own child. `tests/test_client_operations.py`, `test_client_reporting.py`,
`test_client_views.py`, `test_investor_report.py` cover independent contracts.

`scripts/review/check_operator_confirmation_ui.cjs` independently exercises the
actual HTMX confirmation, replay rejection, cancellation, visible network-error
recovery and copying without execution, using a synthetic loopback provider.

This is not a production sign-off. Real-source onboarding/ownership/fee verification
and a separately authorized deployment remain necessary. Exact consolidated returns
and certified decision-time replays require additional source evidence; neither is
fabricated. Current investor output is English. Persistent issued-report archives,
RTL and automatic distribution remain optional future extensions.
