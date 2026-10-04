# VIX-trend gates for DV2: KAMA and moving average with a band (frozen 2026-10-03, before any run)

Owner (2026-10-03): gate open when VIX is above its KAMA (or above a 10-day average plus a band), closed when it falls
back below. Different kind of measure from C3: "VIX rising versus its recent level" instead of "VIX high".
Prior expectation (written before running): short-term relative measures open in calm markets on small VIX rises and
close mid-crisis while VIX falls from very high levels, when dip-buying pays best; so they should lose to C3.

## Definitions (VIX closes, known at close t; band gates: open when VIX > upper, close when VIX < lower)
- KAMA(n, 2, 30), Kaufman: ER = |V_t - V_{t-n}| / sum|dV| over n; SC = (ER (2/3 - 2/31) + 2/31)^2;
  KAMA_t = KAMA_{t-1} + SC (V_t - KAMA_{t-1}).
- K10_b0 / K10_b5 / K10_b10: upper = KAMA10 x (1 + b), lower = KAMA10. Neighbour: K20_b5.
- S10_b0 / S10_b5 / S10_b10: upper = SMA10 x (1 + b), lower = SMA10. Neighbour: S20_b5.
- K10_b5_AND_level: K10_b5 state AND VIX > expanding mean of VIX.
- K10_b5_OR_C3: K10_b5 state OR C3 state.
Reference C3 (VIX > 20, >= 10 sessions). Same setup, book, blocks, +5 bps and 1995–99 holdout as the gate studies.

## Decision rule (as the adaptive study)
Replaces C3 only if: book Sharpe >= C3 in G-FULL and G-LONG and >= C3 - 0.03 in each block; the same at +5 bps;
its neighbours (b grid; n = 20 neighbour) >= C3 - 0.03 in G-FULL; 1995–99 >= C3 - 0.05.

## Result (2026-10-03): all 10 fail; the prior expectation held. C3 stays.
| Gate | Book 2012–26 | 2008–26 | 2022–26 | +5 bps | 1995–99 | Switches/yr | Open days with VIX <= 20 |
|---|---|---|---|---|---|---|---|
| C3 (reference) | 1.558 | 1.383 | 1.475 | 1.511 | 2.16 | 8 | 17% |
| VIX > KAMA10 | 1.397 | 1.253 | 1.300 | 1.344 | 2.54 | 44 | 49% |
| VIX > KAMA10 + 5% band | 1.357 | 1.223 | 1.246 | 1.314 | 2.33 | 26 | 45% |
| VIX > SMA10 + 5% band | 1.408 | 1.260 | 1.386 | 1.359 | 2.18 | 32 | 52% |
| KAMA/SMA other cells | 1.35–1.38 | 1.22–1.27 | 1.25–1.36 | 1.31–1.33 | 1.64–2.35 | 17–54 | 39–56% |
| KAMA band AND VIX > learned mean | 1.424 | 1.268 | 1.354 | 1.397 | 2.07 | 17 | 3% |
| KAMA band OR C3 | 1.491 | 1.336 | 1.383 | 1.430 | 2.32 | 18 | 34% |
| Ungated DV2 (for reference) | 1.430 | 1.280 | 1.391 | 1.339 | 2.30 | | |
VIX-trend gates are open in calm markets (half their open days have VIX <= 20) and close while VIX is falling from
crisis highs, the best dip-buying weeks. Most are worse than no gate. They do best in the 1990s, when calm dips paid.
