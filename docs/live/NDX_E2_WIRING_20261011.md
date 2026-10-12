# E2 independent monthly wiring - 2026-10-11

Both sector-capped momentum components are WIRED as separate account routes. Templates are disabled. This is code readiness, not deployment, broker-fill evidence, or an allocation decision. The existing uncapped NDX VXN route is unchanged.

## Owner-approved contract and scope

- Dollar ATR: `strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap:SectorCapVxnScaledAtrNormalizedNdxStrategy`.
- NATR20: `strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap:SectorCapNatr20VxnScaledNdxStrategy`.
- One independent account per component. Enabled E2 routes cannot share an account with another enabled pod. `pod_budget_fraction_float` is the fraction of that dedicated account, defaults to 1, and must be in (0, 1]. It does not establish ownership within a shared account.
- The owner clarified that execution should follow existing VXN practice: freeze month-end weights, calculate shares from broker NAV and reference quotes before the opening auction. No new shared sizing engine or daily execution policy is introduced.
- `portfolios/ndx_e2_sector_cap_5050.yaml` remains a research aggregation with monthly 50/50 rebalancing. Wiring these two accounts does not implement transfers or automatic 50/50 live allocation.

```text
[Month-end T: Norgate PIT prices, membership, VXN + frozen current GICS]
        | exact T snapshot, validated hashes, complete data
        v
[Existing ATR / NATR20 signals -> ranking -> 4-per-sector selection]
        | *** CRITICAL *** Close_T freezes names and weights
        v
[DecisionPlan + snapshot/map provenance in each account's local state]
        | first next-month session, pre-open broker NAV/reference quotes
        v
[Existing monthly VPlan: whole-share targets -> delta from actual holdings]
        | MOO submitted at normal open-minus-6m30s window
        v
[Next opening auction -> broker reconciliation -> persisted monthly completion]
```

## Unchanged quantitative rules

Let T be the actual last XNYS trading session of the month. C, H and L are CAPITALSPECIAL prices; U is Norgate Unadjusted Close. All rolling windows include only observations through T.

```text
ROC12_i,T = month_end_Close_i,T / month_end_Close_i,T-12months - 1
TR_i,d = max(H_i,d - L_i,d, abs(H_i,d - C_i,d-1), abs(L_i,d - C_i,d-1))
ATR20_nominal_i,T = mean(TR_i,T-19:T) * U_i,T / C_i,T
ATR_score_i,T = ROC12_i,T / ATR20_nominal_i,T
NATR_score_i,T = ROC12_i,T / (ATR20_nominal_i,T / U_i,T)
regime_T = (SPY_Close_T > SPY_SMA200_T)
eligible_i,T = PIT_Nasdaq100_i,T and (C_i,T > SMA100_i,T) and finite_score_i,T
scale_T = clip(22 / VXN_asof(T), 0.25, 1.0)
weight_i,T = scale_T / 10 for selected names; zero otherwise
cash_weight_T = 1 - sum(weight_i,T)
```

ATR20 is the simple trailing mean of true range, not Wilder smoothing. Rebasing the entire adjusted ATR by the decision-date raw/adjusted close factor preserves the existing corporate-action convention. SMA100 and SMA200 are trailing session means. No positive-ROC filter is added. VXN uses the latest positive finite observation on or before T; the existing as-of behavior does not impose a new maximum age.

When the SPY regime is off, targets are cash. Otherwise sort eligible names by descending score, then ascending symbol for ties. Walk that ordering, accepting at most four names in each GICS level-1 sector (including the shared UNKNOWN bucket), until ten names are selected or candidates run out. Unfilled slots and reduced VXN exposure stay in cash; weights are not renormalized. The existing research classes compute these rules; the live adapter does not reimplement their ranking or indicators.

No parameters were searched or optimized for this wiring. Frozen defaults above and existing costs are enforced by release validation; overrides require separate parity qualification. Research costs remain 2.5 bps per side, $0.005 per share with a $1 minimum. Real broker fees, partial fills and opening gaps remain actual execution outcomes, not backtest guarantees.

## Shares and timing

For pre-open reference quote P and actual broker account NAV:

```text
account_budget = broker_NAV * pod_budget_fraction
live_target_shares_i = floor(weight_i,T * account_budget / P_i,preopen)
order_delta_i = live_target_shares_i - actual_broker_holding_i
```

Names omitted from the full target book have zero targets. Existing monthly MOO submission, reconciliation, restart handling and duplicate-submit safeguards apply. The normal submission timestamp is 09:23:30 America/New_York for a 09:30 opening session; the exchange calendar determines the next session. A missed/expired monthly cycle is rejected, not turned into a late trade. Quantities use the available pre-submit reference, not the future auction print.

The research engine instead fixes quantities with Close_T NAV and prices. Therefore identical signals/weights do not promise identical shares, realized weights or P&L. The live adapter explicitly stores `ndx_e2_sizing_contract_str=legacy_monthly_preopen_v1` and `research_quantity_basis_str=Close_T`. Shared `execution_engine.py` sizing and broker price sourcing are unchanged.

## Frozen data contract

New profile: `norgate_eod_ndx_e2_pit_plus_vxn_helper`, schema v2.

- Norgate PIT Nasdaq 100 membership, retaining every symbol that was a member between 1999-01-01 and T, including delisted members. Exact membership tails; no current-constituent replacement or five-day tail trimming.
- CAPITALSPECIAL OHLC and Unadjusted Close for stock signals/nominal ATR and SPY; TOTALRETURN SPY and `$SPXTR` for reference benchmarks; CAPITALSPECIAL `$VXN` helper. ALLMARKETDAYS padding preserves the parents' convention. Native Dividend remains in schema v2.
- History starts in 1999, matching the two parents, and ends exactly at the snapshot date. No future rows enter live features.
- In addition to padded history, the producer requires one finite positive unpadded Close_T observation for every current PIT member, SPY and `$SPXTR`, in the matching adjustment mode. Its Close must equal the exported T value; a provider update during export is rejected. The exact endpoint map is recorded and validated by clients. Retired members are exempt, and VXN retains prior-as-of behavior.
- A complete GICS map is captured on the licensed Norgate producer with `classification_at_level(symbol, "GICS", "ClassificationId", 1)`. Native eight-digit IDs are preserved. A successful missing classification becomes UNKNOWN; provider errors abort export.
- The map, canonical SHA256, current-at-export basis, snapshot date and UTC capture timestamp travel in the manifest and DecisionPlan. The client never calls Norgate for classifications.
- Invalid map coverage/hash/provenance, missing raw-close anchors, invalid current-member prices, internal stock/benchmark session gaps, missing SPY history, unavailable monthly regime/features, or a snapshot other than exact T stop new decisions. Legitimate pre-IPO and post-delisting absence is retained. Valid regime-off remains an all-cash target.
- Manifest and file SHA256 validation is repeated on every E2 read. The full content check runs once per exact manifest per process, because the manifest pins both file hashes, the sector map and the Close_T endpoint attestations. A failed check is never reused. Identity is checked before and after decision/reference generation. No cached result may conceal a damaged new input.

GICS labels are current at export, not historical PIT labels. Replaying past months using one such map remains a historical classification approximation (G-025). Capturing a past-date price artifact today does not establish what sector labels were available back then.

## Reference comparison

E2 auto references run the real research wrappers under the E2 snapshot context. They use the frozen map projected onto the reference universe, while storing the full manifest map and hash. Strategy pickle, reference maps, comparison JSON/HTML and runner JSON carry the provenance and explicitly identify the result as a diagnostic current-sector-map reconstruction. Manual/old pickles without complete metadata report unavailable provenance.

The shared comparison contract remains `price_return_ledger_v1`, with the dividend ledger disabled as before. Research Close_T quantities, live pre-open quantities, current-sector reconstruction and accounting differences mean this reference is not an exact share or P&L parity gate. Stored individual decisions provide the contemporaneous input record.

## Operator handoff

Templates:

- [Dollar ATR template](release_templates/pod_ndx_atr_normalized_vxn_scaled_sector_cap_monthly_open.yaml.example)
- [NATR20 template](release_templates/pod_ndx_natr20_vxn_scaled_sector_cap_monthly_open.yaml.example)

Both are paper examples with `enabled_bool=false`, `auto_submit_enabled_bool=false`, independent placeholder accounts and separate client IDs. This work does not edit actual client release files or schedule a deployment. A separately authorized rollout must deploy compatible readers/exporter, register and validate the new profile on the producer, sync it to the intended clients, and assign each component its real dedicated account before enabling execution. Existing snapshot profiles remain intact.

## Verification and live impact

Tier 3, also touching Tier 2 snapshot transport and Tier 1 strategy context. Independent parity, failure-modes, coverage and quant-pitfalls review were used. Findings corrected during review: invalid historical data becoming apparent cash signals; missing `$SPXTR`; exporter history preceding SPY; actual provider GICS ID shape; reference validation incorrectly comparing a deployment subset with the full PIT map; and padded/stale closing prices incorrectly certifying a completed source observation.

Local native-data qualification on the final freshness-enforced code: a 2026-09-30 artifact exported successfully with 380 PIT symbols and 103 verified unpadded T endpoints. Manifest SHA256: `4757ffbfa107c7b51719d485f115f0c908940947f14f3966570bab5054866a9b`. Full validation took 5.28 seconds. Both actual host routes built ten-name decisions for 2026-10-01 09:30 ET, with sector caps and matching frozen exposure of 97.9519185%. Full-history host builds took 52.68 and 43.62 seconds locally while regression tests also ran. This is reconstructed data/mechanics evidence with synthetic account state, not a broker execution or historical-sector timing claim. Ignored local evidence, including source-code hashes: `results/review/ndx_e2_wiring_20261011/final/real_snapshot_smoke.json`. This supersedes the earlier artifact without endpoint attestations.

Live-impact checklist:

- Timing: unchanged monthly next-open/MOO path; exact month-end snapshot required.
- Sizing: unchanged full-target-weight formula; no new `amount`/`target` semantics.
- Reference price: unchanged broker pre-submit source; no close/open substitution.
- Persistence: no SQLite or existing config schema migration; optional E2 pickle metadata only.
- Logging: existing fields retained; E2 provenance is additive.
- Windows/restarts: isolated artifact staging and existing local store; tests cover reopen, retry and duplicate prevention. Full-history validation is measurable work, so decision preparation must complete before the submit window.
- Releases: both new examples parse disabled; existing routes/templates are regression-tested. Dedicated-account collisions involving enabled E2 fail validation.

Final verification:

- Consolidated regression: **2,338 passed, 12 skipped**, zero failures/errors, in 956.02 seconds. It covers all `test_live*.py` except the separately run new freshness file; Norgate/snapshot tests; MR snapshot tests; strategy and portfolio registries; E2 selection; NATR20; ATR variants; corporate-action invariance; and portfolio BENCH integration (91 files, 2,350 collected cases). Evidence: `results/review/ndx_e2_wiring_20261011/regression.xml`.
- Freshness regression: **26 additional passed** in `tests/test_live_ndx_e2_freshness.py`. Its combined run with the 25 snapshot tests passed all 51; those 25 are already counted in the consolidated run. Total distinct passing cases: **2,364**.
- Skips: six opt-in native-data HPI parity checks, two opt-in full-history HPI replays, and four opt-in matching-root direct-versus-snapshot parity checks. These skipped tests are not claimed as passing; the separate native E2 qualification above ran successfully.
- The 26 runtime warnings came from correlation calculations on deliberately flat or tiny synthetic reference samples. No E2 decision-validation warning was suppressed.
- Tier-3 triage, compilation of all 20 changed/new Python files, and `git diff --check` passed. Independent parity, quant-pitfalls, failure-modes and coverage reviews have no remaining material findings; final documentation corrections are included.

Residual risks are the documented current-GICS history gap, approved Close_T/pre-open quantity difference, and unqualified real broker fills/funding/partial fills. No production or broker operation is part of this change.
