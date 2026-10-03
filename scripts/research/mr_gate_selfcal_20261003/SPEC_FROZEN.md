# Self-calibrating DV2 stress gate (frozen 2026-10-03, before any run)

Owner (2026-10-03): instead of hand-picked numbers (VIX 20, 10 sessions), derive the gate from history so it learns
its own parameters. Goal: robustness, not more Sharpe.

## Candidates (every input uses data up to close t only)
- S1 (structural, primary): threshold_t = expanding mean of VIX closes since 1990-01-02 (min 500 sessions);
  memory_t = half-life of VIX shocks, round(ln 0.5 / ln phi_t), phi_t = expanding AR(1) coefficient of
  x = log VIX - expanding mean of log VIX (min 500 sessions), clipped to [5, 40] sessions.
  Gate: opens on the first close with VIX > threshold_t; memory = memory_t at the opening; closes on the first
  close with VIX <= threshold_t after >= memory sessions (C3 semantics).
- S2 (sensitivity of "normal"): as S1 with the expanding MEDIAN as threshold.
- S3 (learns from returns, walk-forward): every January from 1995, pick (threshold, memory) from
  {18, 20, 22, 25} x {5, 10, 15, 20} with the highest standalone DV2 Sharpe (T-bill cash) from 1991-01-02 to the
  previous December 31; use that pair's gate for the year.
References: C3 (20 / 10), and 20 / 15. Controls: ANY_OFF, ungated DV2, T-bills.
Setup identical to the earlier gate studies (DV2 wired replica, book {TAA .5, L .25, X .25}, blocks, +5 bps,
holdout 1995–1999).

## Decision rule (non-inferiority, because the aim is robustness)
S1 (or S2/S3) replaces C3 if: book Sharpe >= C3 - 0.02 in G-FULL and G-LONG; >= C3 - 0.03 in each of G-P1, G-P2,
G-P3; the same at +5 bps; and 1995–99 standalone >= C3 - 0.05. If several pass, prefer S1 (no chosen numbers), then
S2, then S3. Reported: the threshold and memory paths over time; S3's yearly picks.
