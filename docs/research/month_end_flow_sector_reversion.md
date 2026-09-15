# Month-End Flow + Sector Reversion

The requested research portfolio combines the existing Month-End Rebalancing Flow and US Sector ETF IBS Downshock VOX/IYR strategies. One allocation was fixed before reviewing the combined result: $100,000 total, $50,000 per sleeve, with no periodic transfers between sleeves. No weight search or strategy-rule change was performed.

Configuration: `portfolios/month_end_flow_sector_reversion.yaml`. The portfolio remains RESEARCH; both components satisfy the native manager's PM_READY engine contract. Flow's promotion followed the five completed analyzers and the capital/benchmark/determinism gate at `results/research/pm_readiness/2026-09-13_125726`.

```text
[$100,000 initial portfolio]
       |                       |
       v                       v
[$50,000 Month-End Flow]  [$50,000 Sector Reversion]
[scheduled MOC fills]    [Close T -> Open T+1 fills]
       |                       |
       +-- net daily NAV ------+
                   |
                   v
       [common-close anchor -> sum independently compounded sleeves]
```

The requested start is 2004-11-26 and the fixed endpoint is 2026-09-11. Sector Reversion retains its readiness rule: it needs a completed eligible decision close, so an execution start can be later than the requested date. The saved manager metadata records each effective start and the actual common window. Flow retains pressure history since July 2002 independently of the scoring start.

For the first common close t0, the existing Portfolio engine computes:

```text
E_portfolio(t) = 50000 * NAV_flow(t) / NAV_flow(t0)
               + 50000 * NAV_sector(t) / NAV_sector(t0)
w_i(t) = E_i(t) / E_portfolio(t)
```

The first common close is a capital anchor. Any earlier or anchor-day P&L is outside the reported return window. Verify the native child NAVs at this anchor before equating the portfolio curve with the raw sum of both $50,000 accounts. Subsequent weights drift; this is not a daily or annual reset to 50/50. No cross-sleeve cash netting or additional leverage is introduced.

Both strategies retain house commissions ($0.005/share, $1 minimum per order) and 2.5 bps per-side slippage. Flow additionally retains its 1% annual TLT borrow proxy, rounded 102% collateral and actual calendar days/360; Sector Reversion is long-only. Native cash dividends and applicable withholding remain in each account. Portfolio NAV compounds these net returns. Inherited trade-level statistics exclude the separate dividend/borrow cash flows; use NAV metrics for all-in performance.

The benchmark is the genuine Norgate $SPXTR series labeled $SPX. This request runs Vanilla only. BENCH can run the saved manager configuration again:

```powershell
uv run python strategies/run_portfolio_manager.py portfolios/month_end_flow_sector_reversion.yaml --max-workers 2
```

Known limitations are unchanged: financing of negative cash is absent, realized Flow gross weights can exceed target weights, and fixed borrow/auction assumptions are not observed execution evidence. This combination is an in-sample research observation of two previously studied strategies, not independent validation or LIVE deployment.

## Saved Vanilla result — 2026-09-13

[Open the native portfolio report](C:/Users/User/Documents/workspace/alpha_super/results/research/portfolio/month_end_flow_sector_reversion/vanilla_backtest/2026-09-13_130007/report.html)

The actual common window is **2004-11-29 through 2026-09-11**, 5,481 sessions. Both native sleeves equal exactly $50,000 at the first common close. No anchor-day P&L or costs were removed; the portfolio equals the sum of the two native accounts to within $0.000000002 throughout the window.

| Account over the same common window | CAGR | Sharpe, all sessions | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Combined portfolio | 10.19% | 1.262 | -10.06% |
| Month-End Flow sleeve | 11.93% | 1.155 | -13.83% |
| Sector Reversion sleeve | 7.58% | 0.890 | -20.05% |

Final portfolio NAV is **$824,905.68**. Daily return correlation between the sleeves is **0.213**. The initial 50/50 weights drift to **70.28% Flow / 29.72% Sector Reversion** at the endpoint. This run shows historical diversification with lower drawdown and higher Sharpe than either sleeve; it also has lower CAGR than Flow alone. No subsequent weight adjustment was selected from these results.

The saved accounting includes $9,984.14 Flow commissions, $5,604.66 Sector commissions, 2.5 bps slippage per side in both sleeves, and **$8,952.71 Flow borrow**. Native dividend cash flows also remain included. Separate Flow borrow, held-share, decision, monthly-signal and MOC-schedule CSVs were exported from the saved pod object. `verification.json` records the accounting equality, common dates, capital anchors, final weights, registry status and artifact hashes. Native PM saving does not snapshot every raw vendor price; these hashes identify saved account artifacts.

Verification: readiness passed capital, total-return benchmark and exact repeated terminal NAV checks; 327 regression tests passed, then the stale BENCH catalog count was updated for Flow and the failed test passed on a focused rerun (328 distinct checks). Triage is Tier 1; quant-pitfalls, parity and coverage reviews were completed. The portfolio remains RESEARCH.
