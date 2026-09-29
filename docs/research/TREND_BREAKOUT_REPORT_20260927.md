# Trend / momentum / breakout search with movement-based stops (2026-09-27)

Research only; nothing live, released, scheduled or WIRED was changed, and nothing was committed.
- **Who did what.** The owner asked for a trend / momentum / breakout strategy and whether a movement-based stop-loss
  helps. As the owner asked, the study was designed and executed by a separate research session on a different model
  (Fable 5.1). The lead (Opus 5.5) reviewed and froze the design, re-derived the key numbers and ran post-hoc controls.
  An independent quant-pitfalls reviewer (Opus 5.5) checked everything read-only.
- **Frozen plan:** [TREND_BREAKOUT_PREREG_20260927.md](TREND_BREAKOUT_PREREG_20260927.md) (SHA-256 `24075321...`,
  frozen 14:21 before any cache, parity run or grid cell). Amendments, none of which changes a verdict:
  [TREND_BREAKOUT_PREREG_20260927_AMENDMENTS.md](TREND_BREAKOUT_PREREG_20260927_AMENDMENTS.md).
- **Code:** `scripts/research/trend_breakout_20260927/` and `tests/test_research_trend_breakout_20260927.py`.
- **Results:** `results/research/trend_breakout_20260927/` (`results.json`, `table_cells.csv`, `table_rule.csv`,
  `charts/`, `posthoc_*.json`).

## תקציר בעברית

**מה ביקשת.** אסטרטגיית טרנד, מומנטום או פריצה, ואולי סטופ לפי תזוזה.

**מה בדקנו.** שלוש משפחות, והכול הוקפא לפני שראינו תוצאות:
- **סטופ נגרר על פוד ה-NDX החי:** לפי ATR (פי 2 עד 5) או באחוזים (10%-25%), עם ובלי מילוי המקום שהתפנה. 16 גרסאות.
- **רגל פריצה יומית:** קונים מניה שסוגרת בשיא של 50, 100 או 250 יום, ויוצאים בסטופ נגרר לפי ATR.
  - הבדיקה העיקרית על S&P 500, עם ביקורת על NDX ועל Russell 1000.
  - 16 גרסאות.
- **מומנטום שיורי:** דירוג לפי החלק בתשואה שהשוק לא מסביר. 4 גרסאות.

**איך נשפט.**
- שינוי נחשב רק אם הוא משפר את תיק G3 בכל שלוש התקופות, בלי ירידה גרועה ביותר מ-2 נקודות.
- הוא חייב לעמוד גם בעלויות גבוהות יותר, גם במדדים האחרים ובנזילות מספיקה.
- הסימולטור תואם את המנוע האמיתי בדיוק מלא.
- כל מספר מרכזי חושב שוב אצלי ואצל בודק בלתי תלוי.

**תשובה 1: סטופ על הפוד החי – לא.**
- כל 16 הגרסאות מורידות את השארפ של הפוד כשהוא עומד לבד, ב-0.02 עד 0.19.
- הירידה המקסימלית בהיסטוריה המלאה לא משתפרת.
- סטופ הדוק נורה 25 עד 80 פעמים בשנה ומגדיל מאוד את המחזור.
- המכירה בפועל נמוכה בממוצע ב-1.3% עד 3.3% מרמת הסטופ. הסיבה: ההחלטה בסגירה והמכירה בפתיחה הבאה, אחרי הפער.
- סטופ רחב באחוזים (20%-25%) כמעט ניטרלי בתוך התיק, אבל לא משפר.

**תשובה 2: פריצה – גרסה אחת עברה את הכלל, בקושי ועם הסתייגויות.**
- **הגרסה:**
  - מניות S&P 500 שסוגרות בשיא של 100 יום.
  - כשיש יותר מועמדות ממקומות, בוחרים את השקטות ביותר.
  - יציאה בסטופ נגרר של פי 5 ATR, ועד 10 מניות.
  - היא רבע מהתיק, לצד חצי TAA ורבע NDX.
- **התוצאה בתיק, 2012-2026:**
  - שארפ 1.29 עולה ל-1.34.
  - ירידה מקסימלית 14.1% יורדת ל-11.3%.
  - תשואה שנתית 20.0% יורדת ל-18.7%.
  - זה עדיין מתחת לשער שלך, שארפ 1.35.
- **ההסתייגויות:**
  - אין מובהקות: אחרי תיקון לריבוי בדיקות p=0.85, ובהשוואה ישירה p=0.19.
  - רוב השיפור הוא פשוט תוספת של S&P 500. רבע תיק ב-SPY רגיל נותן כבר בערך 70% מהשיפור.
  - היתרון הייחודי של הפריצה יושב בעיקר ב-2008-2011, תקופה שבה ה-TAA הוא פרוקסי.
  - הרגל עצמה חלשה מאז 2022: שארפ 0.39 ו-4.1% בשנה.
  - בחמשת השבועות שאחרי סוף המדגם היא ירדה 5.6%, כשה-SPY עמד במקום. זה מעט מדי כדי לשפוט, אבל לא מעודד.
  - מה שעובד הוא הבחירה במניות השקטות, לא הפריצה עצמה. אותה פריצה עם דירוג לפי עוצמת מומנטום חלשה (שארפ 0.37 עד 0.49).
  - חלק מהמניות השקטות הן חברות בתהליך רכישה. 43 פוזיציות הוחזקו עד השלמת עסקה (ויית', ג'נזיים, סלג'ין, רד האט ועוד), ברווח טיפוסי של 1% עד 10%.

**תשובה 3: מומנטום שיורי – לא.**
- לבד הוא טוב יותר מהפוד החי: שארפ 0.81 מול 0.77.
- בתוך התיק הוא מפסיד ב-2022-2026, כמו כל הגרסאות חסרות היחידות שבדקנו אתמול.
- הוא גם לא עובר במדדים האחרים.

**שורה תחתונה.**
- **סטופ:** לא מוסיפים סטופ לפוד החי.
- **פריצה:** לא מכניסים את רגל הפריצה לחי.
  - אם רוצים להמשיך: קו צל קדימה בגרסה של 20 מניות, שהיא היציבה יותר.
  - הוא יושווה לבקרה פשוטה של S&P 500 עם מסנן 200 ימים, והחלטה תתקבל אחרי שנה.
- **השער 1.35:** טרנד, פריצה וסטופים לא מביאים את התיק לשם.
  - מה שעושה זאת עד היום הוא תוספת פודי ה-MR היומיים (שארפ 1.41) או ladder_4 (שארפ 1.43).
- **תקלה במנוע:** נמצאה תקלה קטנה, עמלות רפאים של דולר אחד במצב יחידות המניה החדש. ההשפעה זניחה, ומומלץ לתקן בנפרד.

## Bottom line (English)

- **Stops on the live NDX pod: no.**
  - All 16 stop overlays lower the pod's standalone Sharpe (-0.02 to -0.19), and none improves its full-history max
    drawdown.
  - Tight ATR stops fire 25-80 times a year, and fills average 1.3-3.3% below the stop level because the fill is the
    next open.
  - Loose percentage stops are roughly neutral inside the book.
- **Breakout: one mechanical pass, not a live change.**
  - The pass is S&P 500 quiet 100-day breakouts with a 5xATR20 trailing exit, 10 slots, as a quarter slot (0.5 TAA +
    0.25 L + 0.25 B). It clears R1-R5 and lifts G3 from 1.288 to 1.340 Sharpe, with drawdown -14.1% -> -11.3% and CAGR
    20.0% -> 18.7%.
  - It is not significant (Reality Check p = 0.85, paired p = 0.19), and the book stays below the 1.35 gate.
  - Most of the gain is S&P 500 diversification: a passive SPY quarter slot gets +0.036 of the +0.052.
  - The leg's own 2022-26 Sharpe is 0.39.
  - If pursued, run a forward shadow of the interior K = 20 cell against a gated S&P 500 control.
- **Residual momentum: no.** It beats L standalone, but loses 2022-26 inside the book and fails the other universes.
- **The owner's gate** is still reached only with the daily MR capsule (G3 + capsule 1.41) or ladder_4 (1.43). No
  trend-only lever found here gets there.

## 1. What was tested

| Family | Idea | Primary universe | Grid | Book role |
|---|---|---|---|---|
| A | Trailing stop on each position of the live NDX pod L (decided on the close, sold at the next open); freed slot stays in cash or is refilled | NDX | CH 2/3/4/5 x ATR20 and PT 10/15/20/25%, x CASH / REFILL = 16 | replaces L in G3 |
| B | Daily breakout: close above the previous N-day closing high, SPY > SMA200 and stock > SMA200, liquid (top 75% by Turnover), K slots, equal slot budget, chandelier k x ATR20 exit | S&P 500 (NDX, R1000 cross-checks) | B1: N 50/100/250 x k 3/5/8; B2: K 10/20/30 x rank R1 (ROC252/NATR20) / R2 (lowest NATR20) = 16 distinct | replacement and addition |
| C | Residual momentum in L's shell: 11-month mean / sd of market-model residuals | NDX (SP500, R1000 cross-checks) | RES12-1 W36 (anchor), RES12-0 W36, RES12-1 W24, TOT12-1 control = 4 | replacement and addition |

Books use the official pod model (annual reset, one run per window). L inside every book is the replica's L, which
equals the stored corrected sleeve. Blocks run:
- G-P1: 2008-03-04..2011-12-31 (synthetic TAA proxy);
- G-P2: 2012-10-02..2021-12-31;
- G-P3: 2022-01-01..2026-08-19.

## 2. Checks

- **Engine parity (gate before any grid).**
  - G1: the engine's own run of the live pod versus the replica: max daily difference 6.7e-16, CAGR 12.1316% both,
    identical positions on all 6,697 sessions. Versus the stored corrected sleeve: 5e-10.
  - G2: the real engine replayed four daily stop/breakout configurations on NDX: 6.7e-16 per day and 0 position
    mismatches each.
  - After the results the lead replayed the passing leg (B2 centre) through the engine on S&P 500: see section 11.
- **Download-date invariance.**
  - V1: random per-stock price constants leave every entry, stop, position and NAV identical, to 2.4e-15 once the
    engine's phantom $1 orders are cancelled in both runs (see section 10).
  - V2: features restated to each decision day from Norgate's unadjusted bars reproduce filters, breakout flags and stop
    decisions with 0 mismatches over 520 sessions per universe.
- **Tests:** 21 new unit tests pass (71 with the related suites).
- **Independent review:** no lookahead, no replica or engine bug, no misapplication of the rule. The reviewer:
  - recomputed all 621 entries and 578 stop exits of the passing leg from raw arrays;
  - reproduced the saved returns bit-for-bit with the current code;
  - found one material interpretation issue (section 9) and minor items (amendments R1-R8).

## 3. Baseline (G3 = 0.5 TAA + 0.5 L, official pod model)

| | G-P1 2008-11 | G-P2 2012-21 | G-P3 2022-26 | 2012-26 CAGR / Sharpe / MaxDD | 2008-26 Sharpe / MaxDD |
|---|---|---|---|---|---|
| G3 | 0.689 | 1.303 | 1.257 | 19.99% / 1.288 / -14.09% | 1.167 / -15.27% |

L alone 2000-26: 12.1% / 0.77 / -29.3%.

## 4. Family A - stops on the live pod

Differences versus L (standalone 2000-26) and versus G3 (book, 2012-26):

| Stop | Standalone Sharpe | Standalone CAGR (pp) | Book Sharpe | Book MaxDD (pp, + = better) | Stops per year | Mean fill vs stop level | Turnover (x/yr) |
|---|---|---|---|---|---|---|---|
| CH-2 cash | -0.185 | -5.7 | -0.075 | +0.9 | 58 | -1.3% | 13.5 |
| CH-3 cash | -0.146 | -4.2 | -0.042 | +0.5 | 37 | -1.4% | 10.7 |
| CH-4 cash | -0.104 | -2.9 | -0.041 | +0.1 | 22 | -1.4% | 9.1 |
| CH-5 cash | -0.083 | -2.1 | -0.063 | -0.0 | 13 | -1.4% | 8.5 |
| CH-2 refill | -0.146 | -3.3 | -0.112 | +0.9 | 83 | -1.3% | 22.0 |
| CH-3 refill | -0.080 | -2.0 | -0.055 | +0.6 | 43 | -1.4% | 14.1 |
| CH-4 refill | -0.079 | -1.8 | -0.054 | +0.4 | 23 | -1.4% | 10.8 |
| CH-5 refill | -0.100 | -2.0 | -0.067 | +0.1 | 13 | -1.3% | 9.3 |
| PT-10% cash | -0.056 | -2.9 | -0.031 | +0.9 | 25 | -2.0% | 9.4 |
| PT-15% cash | -0.027 | -1.7 | -0.004 | +0.8 | 12 | -2.1% | 8.5 |
| PT-20% cash | -0.050 | -1.5 | -0.002 | +0.4 | 6 | -2.5% | 8.2 |
| PT-25% cash | -0.019 | -0.6 | +0.006 | +0.1 | 3 | -3.0% | 8.0 |
| PT-10% refill | -0.089 | -2.5 | -0.026 | +1.2 | 31 | -2.2% | 11.5 |
| PT-15% refill | -0.053 | -1.6 | -0.010 | +0.9 | 13 | -2.2% | 9.3 |
| PT-20% refill | -0.049 | -1.3 | +0.003 | +0.5 | 7 | -2.6% | 8.4 |
| PT-25% refill | -0.029 | -0.7 | -0.002 | +0.1 | 3 | -3.3% | 8.2 |

L itself turns over 7.9x a year.
- **Standalone:** no stop beats L (Reality Check p = 0.99).
- **Full-history max drawdown** is not improved (L -29.3%; CH-5 -29.8%, PT-20 -32.2%). In 2022-26 stops cut the pod's
  drawdown by 2-3 pp at a cost of 0.05-0.10 Sharpe.
- **Gap fills:** 72-80% of stop exits gap through the level, and the 5th-percentile fill is 5-13% below it.
- **Inside the book** the loose percentage stops are roughly neutral: PT-20 cash beats the same-day G3 on 18 of 21
  rebalance offsets (median +0.016).
- **Frozen rule:** all four row candidates fail R1 and R4.

## 5. Family B - daily breakouts

Standalone 2000-26 and the addition book (0.5 TAA + 0.25 L + 0.25 B) on S&P 500:

| Cell | CAGR | Sharpe | MaxDD | Sharpe 2022-26 | Book 2012-26 Sharpe | Book MaxDD | Corr with L | Turnover |
|---|---|---|---|---|---|---|---|---|
| N50 k3 K20 R1 | 6.4% | 0.49 | -29% | 0.75 | 1.249 | -12.8% | 0.67 | 10.6 |
| N100 k5 K20 R1 (B0) | 6.0% | 0.43 | -39% | 0.70 | 1.237 | -13.6% | 0.68 | 4.6 |
| N250 k5 K20 R1 | 7.1% | 0.49 | -31% | 0.74 | 1.245 | -13.4% | 0.69 | 4.5 |
| N100 k8 K20 R1 | 7.1% | 0.46 | -51% | 0.45 | 1.254 | -14.1% | 0.65 | 1.9 |
| N100 k5 K10 R1 | 5.4% | 0.37 | -50% | 0.63 | 1.227 | -14.3% | 0.66 | 4.6 |
| N100 k5 K30 R1 | 6.4% | 0.46 | -38% | 0.73 | 1.252 | -13.3% | 0.70 | 4.5 |
| **N100 k5 K10 R2** | 7.9% | **0.69** | -26% | **0.39** | **1.340** | **-11.3%** | 0.48 | 4.7 |
| N100 k5 K20 R2 | 7.8% | 0.68 | -27% | 0.42 | 1.322 | -11.4% | 0.54 | 4.6 |
| N100 k5 K30 R2 | 7.1% | 0.64 | -25% | 0.52 | 1.308 | -11.7% | 0.59 | 4.5 |

- **Ranking by momentum (R1)** gives a weak leg on every universe (S&P 500 0.37-0.49).
- **Ranking by the quietest breakout (R2)** is what works on S&P 500 (0.64-0.69) and on Russell 1000 (0.66-0.71), but
  not on NDX (0.49-0.61, where K ordering reverses).
- **Sensitivities:** the VXN-scaled budget (0.47) and a regime exit (0.44) do not help.
- **Drawdowns** of R1 cells on NDX and R1000 reach -58..-61%.

## 6. Family C - residual momentum (NDX)

| Cell | Standalone CAGR / Sharpe / MaxDD | Sharpe 2022-26 | Addition book 2012-26 | Addition G-P3 | S&P 500 / R1000 standalone |
|---|---|---|---|---|---|
| RES12-1 W36 (anchor) | 13.4% / 0.81 / -22% | 0.62 | 1.279 | 1.177 | 0.56 / 0.59 |
| RES12-0 W36 | 11.8% / 0.73 / -27% | 0.52 | 1.256 | 1.120 | 0.56 / 0.51 |
| RES12-1 W24 | 13.9% / 0.85 / -26% | 0.62 | 1.301 | 1.186 | 0.61 / 0.55 |
| TOT12-1 (control) | 13.4% / 0.78 / -29% | 0.45 | 1.234 | 1.110 | 0.51 / 0.63 |

Residual momentum beats L standalone. But like every scale-free NDX score, it loses the 2022-26 block inside the book
(G3 1.257), and it trails the scale-free A0 shell on S&P 500 and Russell 1000.

## 7. The frozen rule, per candidate and role

Margins are neighbourhood medians minus G3. DD gaps are in pp, where + means better.

| Candidate | Role | G-P1 | G-P2 | G-P3 | DD 2012-26 / 2008-26 | R3 costs | R4 other universes | R5 capacity | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| A CH-5 cash | replacement | -0.007 | -0.080 | +0.005 | +0.05 / 0.0 | fail | fail / fail | pass | fail |
| A CH-4 refill | replacement | +0.009 | -0.057 | -0.048 | +0.38 / 0.0 | fail | fail / pass | pass | fail |
| A PT-20 cash | replacement | -0.015 | -0.017 | +0.041 | +0.37 / 0.0 | fail | fail / fail | pass | fail |
| A PT-25 refill | replacement | -0.025 | -0.013 | +0.028 | +0.30 / 0.0 | fail | fail / fail | pass | fail |
| B1 N100 k3 | replacement | -0.227 | -0.160 | -0.094 | +0.30 / -4.42 | fail | pass (NDX, R1000) | pass | fail |
| B1 N100 k3 | addition | -0.103 | -0.057 | -0.015 | +1.01 / -1.86 | fail | pass | pass | fail |
| B2 K10 R2 | replacement | +0.030 | +0.034 | -0.049 | +0.06 / -2.88 | fail | pass | pass | fail |
| **B2 K10 R2** | **addition** | **+0.026** | **+0.047** | **+0.039** | **+2.75 / -1.17** | **pass** | **NDX 0.53066 vs 0.53057; R1000 0.69 vs 0.47** | **pass ($50.7M)** | **pass** |
| C RES12-1 | replacement | +0.069 | +0.034 | -0.178 | -1.12 / -0.66 | fail | fail / fail | pass | fail |
| C RES12-1 | addition | +0.038 | +0.027 | -0.080 | -0.34 / -0.33 | fail | fail / fail | pass | fail |

The passing book (centre) makes 18.7% a year at Sharpe 1.340 with a -11.3% max drawdown on 2012-26, and Sharpe 1.21
with -16.2% on 2008-26. The owner's gate (1.35) is not met; it is met only under the daily-rebalanced sensitivity
(1.350). By the PREREG's wording this is "improves G3, book still below the gate".

## 8. Confidence labels

- **White's Reality Check** over the 56 book configurations: the best is this pass, +0.052, **p = 0.85**. 12.5% of
  configurations beat G3. Family A alone: p = 0.93. Standalone A versus L: p = 0.99.
- **Paired bootstrap**, B2 addition minus G3 (2012-26): +0.052, one-sided p = 0.19, 90% interval [-0.05, +0.15].
- **Deflated Sharpe** (N = 293): about 1.00 for every candidate and for G3. It tests Sharpe > 0 and is uninformative for
  the incremental claim.
- **Walk-forward** (candidate re-chosen on 2000-11 only): it picks the same cell. Out of sample the addition book is
  1.350 / 1.295 in 2012-21 / 2022-26 against G3's 1.303 / 1.257. The leg itself is 0.97 / 0.41, against L's 1.03 / 0.82.

## 9. Post-hoc controls (not pre-registered; they change no verdict)

**1. Mostly S&P 500 diversification.** This is the material review finding. The same quarter slot filled with:
- passive SPY (price only, no costs): -0.095 / +0.032 / +0.045 by block, 2012-26 +0.036;
- SPY gated by SMA200: -0.042 / -0.009 / +0.053, 2012-26 +0.010;
- the B2 leg: +0.036 / +0.059 / +0.040, 2012-26 +0.052.

The leg's value beyond plain SPY sits mainly in 2008-11, the proxy block. In 2022-26 it is slightly worse than SPY.

**2. The pass does not depend on the grid edge.** The interior neighbour (K 20) also passes R1-R5: +0.016 / +0.035 /
+0.040, DD +2.72 / -1.38, R4 NDX 0.570. The centre's NDX knife-edge comes from averaging the two edge cells.

**3. Takeover targets.** All 43 terminal liquidations of the leg are completed acquisitions (for example WYE 2009,
BNI 2010, GENZ 2011, CELG 2019, AET 2018, RHT 2019, TIF 2021, ALXN 2021, EA 2026).
- Each was held to the deal close for a typical +1% to +10%.
- Their terminal P&L share is 4.7-6.1% depending on the denominator, which straddles the PREREG's 5% flag.
- The quiet-breakout rank favours pending targets: the median NATR20 at entry is 0.7% for these trades against 1.5%
  for all trades. The momentum-ranked version holds about as many (55 acquisition-like of 65).
- The flat 25% haircut in the PREREG treats every delisting as a failure. It cuts the leg's standalone Sharpe from 0.69
  to 0.36, but none of these was a failure. A haircut applied only to distress-like cases changes nothing.

**4. After the study END** (2026-08-20..09-25, outside every block): the leg lost 5.6%, SPY gained 0.3% and L lost 0.8%.

## 10. Costs, capacity, and an engine issue

- **Costs.** +5 bps per side changes no verdict. The leg turns over 4.7x a year, holds 83% on average and makes about
  23 round trips a year, with a median holding of 71 sessions. Its stop exits fill 0.9% below the level on average (5th
  percentile -3.8%).
- **Capacity proxy (2021-26):**
  - Order at 5% of ADV20: the largest order reaches it at a $50.7M pod (L: $18.1M).
  - Order at 1% of ADV20: the 95th-percentile order reaches it at a $14.4M pod.
  - Market-on-open auctions are a small share of daily volume, so real opening-auction capacity is roughly ten times
    lower. That is still far above the owner's scale.
- **Engine issue ("phantom fills").** In the engine's historical-share mode (commit fb81e86), re-sizing a position to
  an unchanged whole-share count can leave a ~1e-5-share order, because U/Close drifts by about 1e-8 between days. The
  engine fills that order at the $1 minimum commission. It is backtest-only and tiny: about $41 over 26 years across 17
  cells, and one fill in the live pod's own ledger. It should still be fixed in `alpha/engine/strategy.py` (Tier 2, a
  separate task).

## 11. Engine replay of the passing leg on S&P 500

The PREREG made an S&P 500 engine replay optional, and the gate replayed family B only on NDX. After the results the
lead ran the real engine on the passing leg's own universe:
- The engine replayed the replica's intents for the B2 centre on S&P 500, 2000-01-04..2026-08-19.
- Daily-return correlation was 1.0, with a maximum daily difference of 6.7e-16.
- CAGR is 7.9138% in both, and the final NAV is identical to the cent ($739,876.24).
- Positions are identical on all 6,697 sessions.
- The engine booked 1,252 transactions against 1,209 intents; the 43 extra are exactly the 43 terminal liquidations.
- File: `parity_g2_SP500.json`.

## 12. Limits

- One history. A 14-year book Sharpe has a standard error of about 0.27. Every margin here is a few hundredths.
- G-P1 rests on the synthetic TAA proxy. The leg's distinct value over SPY sits mostly there.
- The capacity figures are a proxy.
- Terminal liquidations are valued at the last close, which is right for completed deals.
- Family C uses a market-only model; Fama-French factors were not available locally.
- Live fills were not inspected.

## Addendum (later on 2026-09-27): the T-bill slot control

A later check changes how the B2 pass should be read. It was done for the follow-up search
([NEW_STRATEGY_SEARCH_REPORT_20260927.md](NEW_STRATEGY_SEARCH_REPORT_20260927.md)) and is not part of this study's
frozen rule.

- **The control.** Put T-bills (BIL total return) in the same quarter slot: {TAA 0.5, L 0.25, BIL 0.25}, same official
  pod model. It scores 0.728 / 1.334 / 1.437 by block, and 1.365 over 2012-26 with a -11.7% drawdown.
- **The comparison.** The passing B2 addition book scored 1.340 over 2012-26, so it does not beat T-bills in the slot it
  was tested in.
- **What this means.** The B2 "improvement over G3" comes from shrinking the NDX pod, not from the breakout leg.
- **Consequence.** The optional forward shadow suggested in section 6 is withdrawn. Any future pod must beat the T-bill
  slot control.
