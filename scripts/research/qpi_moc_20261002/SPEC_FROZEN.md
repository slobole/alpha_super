# QPI same-day close test (frozen 2026-10-02, before any result)

Question (owner, 2026-10-02): Quantitativo's "Murphy's Law" (Dec 2025) reports that entering QPI trades at the
same-day close beats the next open (+0.56% -> +0.64% per event, t 3.37). Our DV2 study found the opposite with a
real 15:45 decision. Does the article's improvement survive a decision taken at 15:45?

## Strategy (the repo's `strategy_mr_qpi_ibs_rsi_exit`, the article's final rule set)
- Universe: S&P 500 members on the decision day (Norgate PIT).
- Entry: QPI(3, 5y) < 30, Close > SMA200, 3-day return < 0, IBS < 0.10; all features finite.
- Rank: dollar turnover descending; 10 equal slots (NAV / 10).
- Exit: IBS > 0.90 or RSI2 > 90.
- Costs: engine (2.5 bps slippage per side, $0.005/share, $1 minimum); stress +5 bps per side.

## Decision states
- final: the day's final OHLC (what the article and the repo's `_moc_paper` file assume).
- 15:45: Alpaca SIP bars up to 15:45 mapped to Norgate by ratio to the official close (DV2 study, layer 2).
  QPI, SMA200, 3-day return, IBS and RSI2 are recomputed with today's value replaced by the 15:45 state.
  Turnover rank at 15:45 uses the previous session (today's volume is unknown).
  Names without Alpaca data keep the final state (optimistic for the close modes); coverage is reported.

## Modes (entry timing / exit timing)
- M0 next open / next open (engine default)
- M1 close, final state / next open (the article's "trading at the close"; the repo's `_moc_paper`)
- M2 close, 15:45 state / next open  <- the executable version of M1
- M3 close, final / close, final
- M4 close, 15:45 / close, 15:45
- M1r control: M1 with the previous-session turnover rank (isolates the rank change in M2)

## Window and gates
Primary window 2016-01-04 -> 2026-09-24 (Alpaca coverage). M0/M1/M3 also 2004 -> 2026 (article window).
Verdict "the close improvement is tradable" only if M2 Sharpe >= M0 Sharpe + 0.10 AND M2 >= M0 in both
2016-2020 and 2021-2026. Otherwise the article's gain is a final-close artefact.
Secondary: M4 vs M0; per-trade attribution of M2 vs M1 (common / 15:45-only / close-only entries).
