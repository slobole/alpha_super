# Tactical FI: ALFRED point-in-time data and stale-input rule (handoff, 2026-09-28)

Research change for `strategies/taa_beyond_6040/strategy_taa_tactical_fixed_income_ief_lqd.py` (Tactical FI L14, 17–27% of the defensive books). It closes owner item #5 of `docs/research/LEAKAGE_HUNT_BOOKS_20260927.md` in research mode. It does **not** approve PAPER or LIVE use. No live code, release manifest, portfolio YAML, engine file, other strategy or existing loader behaviour was changed.

## 1. Summary

- **Problem.** The strategy reads Moody's DAAA/DBAA from frozen current-vintage FRED files. FRED stopped updating both series after the 2016-10-07 observation and backfilled the gap in March 2017. The frozen contract therefore decided 2016-10-31 … 2017-02-28 with values nobody could see at the time.
- **What now exists.**
  1. A hash-locked ALFRED snapshot: all four FRED inputs exactly as published on every decision date T (and on session T−1) from 2014-04-30 to 2026-07-31.
  2. A research mode, `fred_data_mode_str="alfred_point_in_time"`, that recomputes each of those decisions only from its own vintage.
  3. A fail-closed stale-input rule in both modes: a decision whose FRED row is too old produces no target.
- **Result.**
  - The headline is the vintage-T replay only, fixed in advance; the other variants are sensitivities.
  - It is point in time to the **day**. Within the day it relies on the module's release model (§8.2).
  - Point-in-time replay: **0 of 143 usable decisions change**, and the **5 outage decisions are blocked**.
  - Book CAGR 2.707% → **2.700%**; book Sharpe 1.064 → **1.075**.
  - The whole difference comes from January 2017. The frozen contract held 50/50 IEF/LQD that month on backfilled data; the fail-closed replay stayed in cash.
  - No restatement of the book numbers is needed.
- **Frozen contract unchanged.** The default mode reproduces the governed 289-row contract (hash `85f16e73…`). All 289 frozen decisions use observation T−1, so the stale rule never fires there.

## 2. What changed

| Path | Change |
|---|---|
| `alpha/data/alfred_snapshot.py` | **New** shared module (additive; `fred_loader.py` untouched). Keyless ALFRED fetch with exact-label validation, run-table encode/decode, hash-verified loader. |
| `data/research/tactical_yield_tbill_spread/alfred_pit_20260928/` | **New** snapshot: `alfred_{dgs10,dgs3mo,daaa,dbaa}_runs.csv` + `manifest.json` (≈2 MB). |
| `strategies/taa_beyond_6040/strategy_taa_tactical_fixed_income_ief_lqd.py` | New config fields and the point-in-time path, stale rule, provenance, and `run_variant` kwargs. The `$SPXTR` hash hunk is in a separate commit (see §9). |
| `scripts/research/build_tactical_fi_alfred_snapshot.py` | **New.** Rebuilds a snapshot into a new, empty directory; refuses to overwrite. |
| `scripts/research/run_tactical_fi_alfred_replay.py` | **New.** Frozen vs point-in-time vs study reproduction; writes `results/research/tactical_fi_alfred_pit_20260928/` (gitignored). |
| `tests/test_alfred_snapshot.py`, `tests/test_tactical_fi_alfred_point_in_time.py` | **New** tests (§7). |
| `.gitattributes` | One line: `/data/research/tactical_yield_tbill_spread/** -text`. It keeps the hash-locked CSV bytes from being rewritten. With `core.autocrlf=true`, a fresh Windows worktree checked the frozen FRED CSVs out with CRLF, and every frozen FRED hash failed. This happened in this worktree. |

## 3. Data: the ALFRED snapshot

- **Source.** The public ALFRED graph CSV endpoint `https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=S,S,…&vintage_date=d1,d2,…`, the same one the leakage hunt used.
  - No API key is used. `config.env` holds no FRED key; the owner chose the keyless path on 2026-09-28.
- **Vintages sampled.** Each XNYS month-end decision date T, taken from the strategy's own frozen Norgate IEF/LQD calendar, plus the session before T.
  - Range: 2014-04-30 … 2026-07-31.
  - Count: 148 decisions and 296 vintages per series.
  - Each vintage is the full history FRED had published by the end of that day.
- **Storage.**
  - One row per run: `observation_date, value, first_vintage_date, last_vintage_date`. A row means the value was the same in every sampled vintage from first to last.
  - Reconstruction for a sampled vintage is exact. Every vintage was checked against the raw download before writing.
  - An unsampled date raises, because the table cannot say what was published that day.
- **Provenance (`manifest.json`).**
  - Per series: each HTTP request's retrieval time (UTC), the SHA-256 of the raw response, and the byte count.
  - Per series: the file SHA-256, the vintage list, and the latest observation in every vintage.
  - The decision list, the builder script and its git HEAD.
  - A probe showing that ALFRED has no DAAA/DBAA vintage on 2014-04-01: it silently returns today's vintage under another label.
  - Retrieved 2026-09-28 11:07–11:24 UTC.
- **Pinned hashes (in the strategy).**
  - Manifest: `eb29d05e…436db`.
  - Point-in-time contract, vintage T: `4f74d479…1de040`.
  - Point-in-time contract, vintage T−1: `e98b104b…88867d`.
- **What the vintages show about the outage.** DAAA's latest published observation:

| Vintage | Latest DAAA observation |
|---|---|
| 2016-09-30 | 2016-09-29 |
| 2016-10-28 … 2017-02-28 (every sampled vintage) | **2016-10-07** |
| 2017-03-30 | 2017-03-29 (gap backfilled) |

## 4. The point-in-time rule

For decision T ≥ 2014-04-30:

1. The vintage date is `v = T` (`alfred_vintage_policy_str="decision_date"`, default) or `v =` session T−1 (`"previous_session"`, the conservative choice).
2. The panel is every series exactly as published on v. Its latest observation is ≤ v, which is checked.
3. The **unchanged** frozen rule is re-run from scratch on that panel, and only row T is kept. This has two consequences:
   - The spread at T uses only values published by v.
   - Every earlier monthly spread inside the expanding median is also rebuilt from vintage v. A backfill published after T cannot enter T's median.
4. Decisions before 2014-04-30 keep their frozen rows. They are labelled `frozen_current_vintage_before_alfred_archive`.
5. Cash accrual still uses the frozen DGS3MO file in both modes (see §8).

What happens to a blocked month later:

- The blocked decision itself never trades.
- A later decision rebuilds its median from its own vintage, so the blocked month re-enters that median with the backfilled value. This is legitimate, because the backfill was published by then.
- The forward-shadow script works differently: it appends only the spreads it recorded and never rebuilds from vintages. A live route must pick one rule (§10).

Guard: after the stale rule, a usable decision must not have a stale month inside its recomputed median. That would need an unapproved rule, either drop that month or keep it.

- It does not occur in 2014–2026: by 2017-03-31, the first decision after the outage, the backfill was complete.
- If a future snapshot produced it, the run fails loud.

## 5. The stale-input rule

**Definition.** For decision T, let o be the observation date of the common FRED row the rule selects: the latest date on which all four series have a value, published by the modeled cutoff. Then:

$$
\text{age}_T = \#\{\text{Norgate sessions } s : o < s \le T-1\}, \qquad \text{stale} \iff \text{age}_T > 2
$$

**Why 2.**

- Normal publication gives observation T−1, so age 0. A bond-market holiday or a one-day FRED delay gives 1.
- In the snapshot, the 284 usable decisions have age 0 (279 cases) or 1 (5 cases). The outage decisions have ages 15, 36, 57, 77 and 96.
- The limit of 2 leaves one session of margin and matches `MAX_COMMON_OBSERVATION_AGE_SESSIONS_INT = 2` in the forward-shadow script (`scripts/research/tactical_fi_forward_snapshot.py`).
- On the frozen files all 289 decisions have age 0.

**What happens** (`stale_input_policy_str`). In both policies a stale decision never produces a target: no decision is ever made on old data.

| Policy | Effect | Intended use |
|---|---|---|
| `raise` | Stops the run with `StaleMacroInputError`, naming the decision, the observation, its age and the count of stale decisions. | PAPER/LIVE |
| `block_and_hold` (research default) | No rebalance row and therefore no orders. The pod keeps its positions until the next non-stale decision. The block is recorded. | Historical replay |

- **Owner decision (approved 2026-09-28).** "Hold" is the natural result of fail-closed, because placing no order leaves the positions in place.
- **Blind spot.** The rule detects missing observation *dates*. It would not detect a feed that keeps printing an unchanged value on fresh dates.
  - The 2016-17 outage stopped the dates, so it is caught here.
  - A check on long runs of unchanged values is a possible addition. It is not implemented, because it needs its own threshold.
- **Evidence and tolerance.** Only one outage event exists, and any limit from 2 to 14 gives the same result on it. Under the previous-session vintage the normal age is already 1, so the effective tolerance there is one session.
- The alternative is to **go to cash** on a stale decision. That is an active trade made without information and is not implemented.
- In 2016-17 the difference is small: the pod was already in cash after 2016-09-30.

## 6. Before / after numbers

All variants use the same Norgate prices, 5 bps slippage, causal DGS3MO cash accrual and the start date 2002-08-01. Sharpe uses rf = 0. "Book" is 2012-10-02 … 2026-08-19.

| Variant | Book CAGR | Book Sharpe | Since 2014-05 CAGR / Sharpe | Full 2002-08 CAGR / Sharpe |
|---|---|---|---|---|
| Frozen current vintage (governed) | 2.707% | 1.064 | 2.958% / 1.418 | 4.278% / 0.970 |
| **ALFRED point in time, vintage T, block & hold** | **2.700%** | **1.075** | 2.950% / 1.448 | 4.274% / 0.972 |
| ALFRED point in time, vintage T−1, block & hold | 2.872% | 1.084 | 3.145% / 1.412 | 4.375% / 0.984 |
| Leakage-hunt reproduction (vintage T + current history > 45 days) | 2.736% | 1.079 | 2.991% / 1.443 | 4.295% / 0.975 |

How to read the table:

- Only the vintage-T row is the point-in-time headline; this was fixed before the comparison.
- The other rows are sensitivities. The spread across the four variants (Sharpe 1.064–1.084) comes from one outage episode and a one-session lag, not from signal.
- "0 flips" does not mean identical positions. The frozen run held 50/50 IEF/LQD for January 2017 on backfilled data; the point-in-time run held cash. That month is the whole 0.007 pp CAGR gap.
- Vintage T−1 is not merely a time-of-day-conservative copy of the rule. It always uses an observation one session older (age 1 on every decision), so it is a different lag rule that bounds the timing risk.

Decision-level comparison against the frozen contract (2014-04-30 onward):

- **Vintage T.** Of 148 decisions, 5 are blocked (2016-10-31, 2016-11-30, 2016-12-30, 2017-01-31, 2017-02-28) and 143 are usable. **0 usable decisions flip.**
- **Vintage T−1.** The same 5 are blocked. 5 decisions flip: 2015-04-30, 2015-09-30, 2015-11-30, 2016-04-29 and 2020-03-31.
  - These are boundary effects from using observation T−2 instead of T−1, not look-ahead.
  - The leakage hunt found the same flips.
- **Reproduction.** It matches the leakage hunt exactly: 2 flips, on 2016-12-30 (observation 2016-11-15) and 2017-01-31 (observation 2016-12-16), and the metrics equal `tfi_vintage_metrics.csv` to 1e-12.

**Correction to the leakage hunt's reading.**

- The study's "faithful" mode is *not* point in time during the outage.
- Its history older than 45 days came from the current vintage, so on 2016-12-30 it used the 2016-11-15 Moody's value, which FRED published only in March 2017.
- The strict replay above shows what was actually knowable: nothing newer than 2016-10-07.
- The study's conclusion still holds: the frozen contract is **not** optimistic in any material way. The strict replay's book CAGR is 0.007 pp lower and its Sharpe is 0.011 higher.

## 7. Tests

| File | What it proves |
|---|---|
| `tests/test_alfred_snapshot.py` (offline, 14 tests) | A substituted ALFRED label, an observation dated after its vintage, and repeated observation dates are rejected; an observation dated on the vintage date is accepted. Revisions A→B→A, disappearing observations and late backfills round-trip exactly. A corrupted table fails the round-trip check. Unsampled dates raise. Manifest, file, schema, row-count and missing-series errors raise. Fetch (mocked network): batching and response hashes are recorded; 4 failures raise `AlfredRequestError` (not a "missing vintage" error); one failure then success recovers; a substituted label raises. |
| `tests/test_tactical_fi_alfred_point_in_time.py` (offline, synthetic, 14 tests) | Age counting and the rule boundary: age 2 passes, age 3 blocks. A synthetic outage blocks the decision and emits no target under `block_and_hold`, and raises under `raise`. A value first published lower and revised later is used at its first-published value. **Replacing everything published after T with noise leaves every decision ≤ T unchanged**, while the noise does change a later decision, so the test can detect leakage. A revision of *older* history changes T's median threshold, proving the median is rebuilt from the vintage and not only row T. A never-backfilled gap fires the median-history guard. The panel rejects an observation after its vintage date. Every point-in-time row equals the spread rebuilt from its own vintage. The previous-session policy uses vintage T−1. Unknown modes are rejected. `run_variant` keeps `config_obj` modes unless a keyword overrides them. `run_info.json` records the mode, and point-in-time runs get their own name. |
| same file (real data, 7 tests) | The frozen mode is unchanged and stale-free (289 rows, age 0, contract hash). The real point-in-time run blocks exactly the 5 outage decisions and uses observation 2016-10-07 on 2016-12-30. All 148 point-in-time rows are rebuilt from their own vintage. `raise` stops at 2016-10-31. The previous-session contract hash, blocks and 5 flips are pinned. **Engine level:** the point-in-time run places no fill between 2016-11-01 and 2017-03-31, while the frozen run does. The study's 2 flips and metrics are reproduced to 1e-12. |

Run the replay:

```powershell
uv run python scripts/research/run_tactical_fi_alfred_replay.py
```

Run the strategy in either mode (Bench shows the same three fields as launch parameters):

```python
run_variant(fred_data_mode_str="alfred_point_in_time", alfred_vintage_policy_str="decision_date", stale_input_policy_str="block_and_hold")
```

Point-in-time runs are saved under `results/research/strategy/strategy_taa_tactical_fixed_income_ief_lqd__alfred_pit_<policy>/`. `run_info.json` parameters and `metadata.json` → `data_adjustment_policy` record:

- the mode, the vintage policy, the stale policy and the limit;
- the blocked decisions;
- the manifest and point-in-time contract hashes;
- `cash_rate_vintage_policy_str`.

### Verification record

- **Tier.** Tier 2, because of the new shared module `alpha/data/alfred_snapshot.py` and the data snapshot. The strategy and scripts are Tier 1. No Tier 3 surface is touched: nothing under `alpha/live/**` imports this strategy or the new module.
- **Reviewers (read-only).** Quant-pitfalls, parity and coverage.
  - Parity: no break on the default path. HEAD and this diff give identical frames, results, transactions, strategy name and results folder.
  - Quant: no look-ahead found.
- **Findings fixed.**
  - `run_variant` keyword defaults silently overrode a passed `config_obj`. They now default to `None`, and a test covers it.
  - Network failures are now `AlfredRequestError`, so the builder's pre-archive probe cannot record an outage as "no vintage".
  - Duplicate observation dates in an ALFRED response are now rejected.
  - Test gaps closed: the guard, older-history revision, panel boundary, fetch retries, loader negatives, the previous-session contract, and the engine-level hold.
  - Reporting findings (day-granular point in time, headline fixed in advance, 2020 revisions, stale-value blind spot, median-rule mismatch with the forward shadow) are written into §§4–10.
- **Full suite (before the last fixes).** 5,989 passed, 26 failed, 4 skipped.
  - All 26 failures are pre-existing: they fail identically on a clean detached HEAD worktree checked out the same way.
  - They fall into 5 `test_bench` catalog/count tests, 13 foundry and 7 ladder4 tests whose frozen spec-file byte hashes break under a CRLF checkout, and 1 PTA momentum date assertion.
  - The directly affected files were re-run after the last fixes (see the session report).

## 8. What remains unverifiable or approximate

1. **Before 2014-04-30.** 141 of 289 decisions (2002-07-31 … 2014-03-31) have no Moody's vintage, and before ~2005-06 no Treasury vintage either. They keep frozen current-vintage values, labelled as unverifiable.
   - This includes 2008, when the credit sleeve matters most.
   - Revisions in stress periods are real inside the covered window. Between the 2020-06-30 and 2020-07-31 vintages, DAAA values for late May – June 2020 were revised by up to 0.12 pp (7 observations) and DBAA by up to 0.03 pp (5 observations).
   - Pre-2014 revisions or outages therefore cannot be ruled out. Quote pre-2014 results as unverified.
2. **Time of day.** ALFRED vintages cover a whole calendar day.
   - In every usable decision, vintage T ends at observation T−1, so each decision uses a value FRED first showed on day T. That value may have appeared after the 17:15 ET cutoff.
   - Within the day, the replay therefore relies on the module's release model, not on ALFRED.
   - Vintage T−1 bounds the timing risk. It flips 5 boundary decisions and gives *higher* returns, so the timing effect is not systematically optimistic.
   - The module's release model (Moody's T−1 by 12:00 ET, Treasuries by 17:00 ET) cannot be verified from vintages.
3. **Cash accrual.** Daily DGS3MO accrual still uses the frozen current vintage. The leakage hunt found 0 recent-window revisions in ~15k observations and at most 0.10 pp on single historical days. This is negligible, but not point in time.
4. **Sampling.** The snapshot answers only the 296 sampled dates per series. A new decision calendar needs a new snapshot built by the builder script.
5. **Selection.** Unchanged: 38 frozen variants, familywise p = 0.77, no holdout (G-029).

## 9. Owner decisions

Approved by the owner on 2026-09-28. All three match the code defaults, so no code change was needed.

1. **Stale behaviour: approved.** "Block and hold" is the research replay semantics and "raise" is the rule for any PAPER/LIVE route. "Go to cash" is not adopted.
2. **Governed numbers: approved.** The frozen contract stays the governed book input. The point-in-time replay is the evidence that it is not materially optimistic (−0.007 pp CAGR, +0.011 Sharpe).
3. **Default vintage policy: approved.** Point-in-time runs use vintage T. Vintage T−1 stays a sensitivity.

Notes for applying the change:

- **`$SPXTR` hash hunk.** It is committed separately. It is the same 3-line benchmark-hash update that is uncommitted in the main checkout (owner approved building on it, 2026-09-28); the hunks are byte-identical, so apply it once.
- **Commit layout.** The strategy now imports `alpha/data/alfred_snapshot.py`, so both are in the same commit. Any edit to the strategy module changes its `module_sha256_str`. Saved ladder4/defensive-sleeve checkpoints will report "strategy code changed" and must be re-run; the `$SPXTR` edit alone already causes this.
- **`.gitattributes` rule.** It covers the whole `tactical_yield_tbill_spread` directory, including the existing frozen FRED CSVs, so a Windows checkout keeps their hashed bytes.

## 10. What a PAPER/LIVE route would still need

This change is research plumbing. It does not implement any of the following:

- A **fresh capture at each decision.** Download the four series between the close and 17:15 ET, apply the stale rule with `raise`, and alert the operator. The uncommitted forward-shadow script `scripts/research/tactical_fi_forward_snapshot.py` in the main checkout already has a matching two-session gate and a capture-time gate; it should adopt the same rule constants.
- **Publication timing.** FRED showed observation T−1 in vintage T for 100% of decisions from 2013 and 94% for Moody's; the misses are the outage and holiday-adjacent days. A live route should read the Fed H.15 page (16:15 ET) or fetch FRED after ~17:00 ET, and it must fail closed, not fall back to a cache.
  - The shared `load_daily_fred_series_snapshot` falls back to its cache on a download error and only *warns* on staleness. Do not use it as-is for this strategy's live input.
- **Median-history anchoring** for forward decisions. The options are frozen history plus each recorded decision (what the forward-shadow script does), or a point-in-time rebuild from vintages (what this replay does, see §4). They differ for blocked months. Live and research must use the same rule.
- **Release-time evidence.** Record FRED's last-updated timestamps at each capture so the modeled 17:00 ET (Treasury) and 12:00 ET (Moody's) availability can finally be checked.
- A **decision journal** and **alerting** for blocked months, the broader G-029 research gaps, and a PAPER/LIVE approval.
