# Scout P2 calibration study — frozen protocol

Date: 2026-09-30. Frozen and committed before any result was computed. Any change after the first result is an
amendment listed at the end of the report, never a silent edit. Design: `docs/plans/SCOUT_DESIGN.md` (D19, D20, D22,
section 13 P2 row).

## Question

Which of Scout's candidate tests separate **noise** and **dead strategies** from **real edges**, at what error
rates, and which set of gates should S5 (and S3's inference unit) use?

Two error rates matter equally (D22):
- **false pass:** a search over noise passes the gate set (money lost on luck);
- **false reject:** a real edge fails the gate set (a business lost to over-strictness).

## Part A — S5 gates on synthetic searches

### Data-generating processes (one asset, daily, T = 5,040 sessions = 20 years, zero drift)

| Case | Returns | Seeds |
|---|---|---|
| `noise_iid` | r_t = σ·ε_t, ε ~ N(0,1), σ = 1% | 100 |
| `noise_garch` | GARCH(1,1): σ²_t = ω + 0.08·r²_{t−1} + 0.90·σ²_{t−1}, Student-t(5) standardised shocks, 16% annual vol | 200 |
| `edge_030`, `edge_050`, `edge_080` | noise_garch plus planted momentum: r_t = a·m_{t−1} + σ_t·ε_t, m_{t−1} = mean(r_{t−20..t−1}) | 100 each |

`a` is calibrated once per case (bisection on a 400-year simulation) so that the population annual Sharpe of the
true configuration (L = 20, θ = 0) equals 0.3, 0.5 or 0.8.

### Search (the registered family, 21 configurations)

Signal z_t = (sum of the last L returns) / (σ̂_L · √L), with σ̂_L the trailing L-day standard deviation.
Position for day t+1 = sign(z_t) if |z_t| > θ, else 0 (long/short, next-day execution, no costs).
Grid: L ∈ {5, 10, 20, 40, 60, 120, 250} × θ ∈ {0, 0.25, 0.5}. Selection = plateau choice (`alpha.stats.selection`).

### Gates (as specified in the design before P2; thresholds not tuned here)

| Gate | Pass rule |
|---|---|
| **MCPT plain** | p ≤ 0.01; 200 date permutations; the full search + plateau selection re-run on each |
| **MCPT stratified** | same, shuffling only within terciles of trailing 63-day volatility |
| **DSR** | ≥ 0.95; V = variance of the 21 per-period Sharpes; N = correlation clusters at ρ = 0.5 |
| **Walk-forward** | registered design efficiency ≥ 0.5 and OOS Sharpe > 0, and ≥ 6 of the 8 designs with OOS Sharpe > 0 |
| **PBO** (diagnostic) | ≤ 0.20 (CSCV, S = 10, plateau selection) |
| **Naive** (strawman) | chosen configuration's Sharpe t-stat (Sharpe · √years) ≥ 2 |

### Decision rules (fixed now)

1. **MCPT null.** Keep the stratified null only if its false-pass rate on `noise_garch` is at most 5% and it loses
   no more than 10 percentage points of power against the plain null on `edge_050`. Otherwise keep plain.
2. **Gate set.** Candidate sets: each single gate; each pair of {MCPT, DSR, WFA}; all three; all three + PBO; and
   "any 2 of MCPT, DSR, WFA". Admissible sets have a false-pass rate ≤ 5% on `noise_garch` and a false-reject rate
   ≤ 30% on `edge_080`. Among admissible sets choose the highest power on `edge_050`; a tie within 3 percentage
   points goes to the set with fewer tests (lean). If no set is admissible, report that and choose the set with the
   lowest false-reject on `edge_080` among those with false-pass ≤ 5%.
3. **PBO** becomes a gate only if adding it to the chosen set lowers false-pass by ≥ 2 points at a power cost
   ≤ 3 points on `edge_050`.

## Part B — S3 inference unit on a synthetic event panel

200 stocks, 2,520 sessions: one market factor (GARCH as above, β ~ U(0.5, 1.5)) plus idiosyncratic noise (2% daily).
An event fires on a stock-day with probability 2% from a signal independent of future returns (`s3_null`), or with
a planted 5-day forward excess return of 0.2% on events (`s3_edge`). Hold = 5 sessions. 200 seeds each.

Tests at nominal 5% (two-sided for t, one-sided for permutation):
- **naive event t:** t-test on pooled event excess returns (each event independent);
- **date-clustered Newey-West t:** mean event excess return per date, NW lag 4;
- **within-date permutation:** 500 shuffles of event labels within each date.

Decision: S3 keeps the date-level unit if naive t over-rejects under the null and the date-level tests hold ≈ 5%.

## Part C — known-dead real cases (descriptive)

Saved grids of variant return series from studies whose edge failed out of sample (located separately). Each is
run through DSR, walk-forward and PBO on its in-sample period only; MCPT only where the search can be re-run.
The report states which gates would have stopped each one before its out-of-sample failure.

## Outputs

- `results/scout/p2_calibration/` (gitignored): per-seed results.
- `docs/research/SCOUT_P2_CALIBRATION_20260930.md`: tables of false-pass and power per gate and gate set, the
  decisions above, and amendments.
