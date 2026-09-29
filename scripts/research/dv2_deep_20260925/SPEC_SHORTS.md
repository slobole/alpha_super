# Mean-reversion shorts study — frozen specification (2026-09-26)

Owner request: "run the next study on shorts". Research-only; no live path changes.

## Prior evidence (archived engine runs, 2004-2026, March-April 2026)

QPI short-only -0.1%/yr, Sharpe 0.09, max DD -77%; filtered QPI short +2.5%, 0.43, -47%; QPI IBS/RSI short -0.3%,
-79%; DV2 short with VIX filter +0.5%, 0.19, -52%; QPI long/short 17.6%, 0.80, -35% (worse than long-only QPI).
Win rates ~60% with fat left tails. Standalone MR shorts have failed before.

## Question

Is there a short leg that (1) earns a positive return net of borrow and (2) improves the long mean-reversion
capsule more than a plain S&P 500 beta hedge or cash does?

## Candidates (S&P 500 PIT, liquidity floor, 10 slots of 10%, decide after Close_T, fill Open_T+1, engine costs)

- SH1 mirror of DV2: DV2(126) > 90, Close < SMA200, R126 < -5%, rank NATR14 desc, exit Close < Low_{T-1}.
- SH2 = SH1 ranked by ADV63 desc.
- SH3 overbought with no trend filter: DV2(126) > 90 only, rank NATR14 desc, exit Close < Low_{T-1}.
- SH4 overbought in an uptrend (fade strength against the trend): DV2 > 90, Close > SMA200, R126 > 5%, exit
  Close < Low_{T-1}.
Shorts pay the full gross dividend (engine rule) and a borrow fee on short market value: base 0.5%/yr
(general collateral), stress 3%/yr. No hard-to-borrow or recall model: results are an upper bound.

## Book tests (pod model, daily returns summed by weight, annual reset where books are used)

Long capsule L = DV2 live 50 / HPI vote 50. Compare, over 2004-2026 and P1/P2/P3:
- L alone; L 80 + cash 20; L 80 + short pod 20; L hedged: L - beta_L x S&P 500 TR + beta_L x T-bill (full beta hedge).

## Gates (decided before results)

Short pod viable if: CAGR > 0 at base borrow, Sharpe > 0.3, and positive in at least 2 of P1/P2/P3.
Useful in the capsule if L 80 + short 20 beats L 80 + cash 20 on Sharpe AND max drawdown, in the full window
and in P3 (2021-2026). Otherwise the verdict is "no short leg", which is a valid result.

## Validation

Replica short mechanics must match a real-engine run of SH1 trade for trade (borrow 0).
