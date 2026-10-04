# HPI vote: previous-session turnover rank (frozen 2026-10-03, before any HPI run)

Hypothesis (from QPI, 2026-10-02; post hoc there, so tested here on a different strategy): ranking next-open
candidates by the PREVIOUS session's turnover instead of today's improves daily stock MR. Proposed mechanism: today's
turnover favours volume-spike (news) days. Caveat: the DV2 study found volume-spike dips revert as well as quiet ones.

QPI evidence so far: 2016–26 +0.025, 2004–26 +0.046, 1995–2003 holdout +0.054, bootstrap P 0.87.

## Test
- Strategy: `strategy_mr_hpi_sp500_2_3_5_vote` (HPI < 30 on >= 2 of the 2/3/5-day horizons, IBS < 0.10,
  Close > SMA200, PIT S&P 500; exit IBS > 0.90, RSI2 > 90 or leaving the index; 10 slots; engine costs).
- Arms: rank = Turnover_T (current) vs Turnover_{T-1} (previous session). Nothing else changes.
- Real engine, 2004-01-01 -> latest data, capital 100,000 (the strategy's default run).

## Pass
Sharpe(prev) > Sharpe(current) on 2004–2026 AND in both halves (2004–2014, 2015–2026), AND the paired block
bootstrap (20-day blocks, 2,000 draws) gives P(difference > 0) >= 0.90.
If it passes on HPI and was positive on QPI, the change is recommended (a ranking change with no extra trades);
otherwise it is filed as noise.

## Result (2026-10-03, real engine 2004-01-05 -> 2026-10-02): FAIL — previous-day rank is noise
| | Current rank | Previous-day rank |
|---|---|---|
| Sharpe 2004–26 | 1.049 | 1.015 |
| Sharpe 2004–14 | 0.979 | 0.942 |
| Sharpe 2015–26 | 1.111 | 1.079 |
| CAGR | 16.4% | 15.8% |
| Max DD | −17.7% | −19.0% |
Bootstrap P(prev better) 0.19. 90% of entries are common; daily correlation 0.98. The QPI near-miss does not replicate.
