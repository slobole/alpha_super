# Replay check: house trend-tilt strategy vs research T1 (FAIL)

Common window 2005-10-04 -> 2026-10-23, 5297 sessions; research parquet written 2026-11-05 09:56 UTC.
Correlation of daily returns 0.990 (pass >= 0.97); tracking error 0.94%/yr; exposure correlation 0.994, average gross 0.638 vs 0.644.
CAGR house 5.56% vs research 6.93% (gap -1.37%, pass within 1%); vol 6.48% vs 6.50%; max DD -9.32% vs -7.80%.
Tilt state agreement at month starts: all three assets 98.8% (VTI 98.8%, TLT 100.0%, GLD 100.0%); months with any disagreement (3): 2010-06, 2010-08, 2012-01.
Largest yearly gap 2010 (-3.47%); other years within 3.10%.
Gap decomposition (points/yr): idle cash at Fed funds in research, 0% in house -0.57%; house 25% dividend withholding (research is gross total return) -0.26%; commissions at the $100K default base -0.22% (near zero at fund scale); tilt-disagreement months -0.19%; residual (Open_(T+1) vs close fills, VTI vs SPY, price-vs-total-return vol, 63- vs 40-session overlay warm-up) -0.14%.
Biggest source: idle-cash yield convention. Stock leg VTI vs research stock series daily correlation 0.992; dividend ledger available_and_active.
Files: replay_check.csv (headline, by year, tilt agreement, diagnostics), tilt_agreement_by_month.csv, daily_returns_common.csv.
